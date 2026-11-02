import argparse
import logging
import math
import os
import random
import sys
import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed
import cv2
import options as option
from models import create_model

#sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

import str_utils as str_util

import os.path as osp
import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
# from PIL import Imageoptim
from torch.utils.data import DataLoader
from itertools import cycle



def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", default="/root/autodl-tmp/StrDiffusion_Official/train/texture/config/inpainting/options/train/train_dunhuang_final.yml", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]
    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    mask_root = opt['degradation']['mask_root']
    train_set_mask = Datasetset_mask(mask_root)
    
    #### create train and val dataloader
    dataset_ratio = 1  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            train_loader_mask = DataLoader(train_set_mask, batch_size=dataset_opt["batch_size"], shuffle=True, num_workers=dataset_opt["n_workers"], drop_last=True)
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))


    assert train_loader is not None

    #### create model
    model = create_model(opt) 
    device = model.device
    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        
    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)
    
    # 实例化结构引导模型 S_sde
    S_sde = str_util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    
    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    # 使用 cycle 确保 mask 永远能跟上图像训练步数
    mask_iterator = cycle(train_loader_mask) 

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
            
        for gg, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            Y_GT = train_data["GT"]
            X_GT = train_data["GT_gray"]
            X_LQ = train_data["GT_edge"] 
            prompt_texts = train_data.get("prompt", None)
            
            # 获取当前 batch 的 mask
            mask = next(mask_iterator) 
            
            # SDE 状态生成
            timesteps, states = sde.generate_random_states(x0=Y_GT, mu=Y_GT*mask)
            
            # 喂入数据并优化
            model.feed_data(states, Y_GT*mask, Y_GT, mask, S_sde, X_GT, X_LQ, prompt=prompt_texts) 
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(current_step, warmup_iter=opt["train"]["warmup_iter"])

            # ✨ 核心修复：将保存逻辑移入此处，确保每 2000 步必存
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states at iter {:d}".format(current_step))
                    model.save(current_step) # 会自动生成 {current_step}_G.pth

            # 打印日志
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    if opt["use_tb_logger"] and rank <= 0:
                        tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

    if rank <= 0:
        model.save("latest")
        tb_logger.close()


from PIL import Image
from torchvision import transforms

class Datasetset_mask(Dataset):
    """自定义掩码加载器，包含大洞动态增强逻辑"""
    def __init__(self, THE_PATH):
        self.data = []
        for root, dirs, files in os.walk(THE_PATH, topdown=True):
            for name in files:
                if name.endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(os.path.join(root, name))
        print("Mask dataset length: {}".format(len(self.data)))
        
        # 移除 ToTensor()，保证这里输出的还是 2D 的 PIL Image
        self.transform = transforms.Compose([
            transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    # ✨ 补回不可或缺的 __len__ 方法
    def __len__(self):
        return len(self.data)

    def _draw_thick_strokes(self, mask_np):
        """生成真实的壁画剥落状大掩码"""
        # 此时 mask_np 的形状是二维的 (256, 256)，解包 h, w 完美契合
        h, w = mask_np.shape
        num_lines = random.randint(1, 4)
        for _ in range(num_lines):
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            thickness = random.randint(15, 45) # 极粗的线条模拟大块剥落
            cv2.line(mask_np, (x1, y1), (x2, y2), 0.0, thickness) # 0.0 为破损
        return mask_np

    def __getitem__(self, i):
        path = self.data[i]
        # 1. 拿到经过基础增强的 PIL Image (此时为 2D)
        mask_pil = self.transform(Image.open(path).convert('L')) 
        
        # 2. 转成 NumPy 数组并归一化到 0~1 (形状为 256x256)
        mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
        # 强制二值化，消除 resize 带来的灰度插值：1 代表完好，0 代表破损
        mask_np = (mask_np > 0.5).astype(np.float32) 
        
        damage_ratio = 1.0 - mask_np.mean()
        
        # 3. 动态画粗线条
        if damage_ratio < 0.15 and random.random() < 0.5:
            mask_np = self._draw_thick_strokes(mask_np)

        # 4. 手动转成 Tensor，并补充通道维度 -> (1, 256, 256)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        return mask_tensor

if __name__ == "__main__":
    import os

    cuda_home = os.getenv("CUDA_HOME")

    if cuda_home is None:
        print("CUDA_HOME environment variable is not set.")
    else:
        print("CUDA_HOME:", cuda_home)
    main()
