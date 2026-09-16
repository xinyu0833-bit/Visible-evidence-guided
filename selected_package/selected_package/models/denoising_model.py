import logging
from collections import OrderedDict
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from ema_pytorch import EMA

import torch.nn.functional as F
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion
from models.modules.loss import MatchingLoss

from transformers import CLIPTextModel, CLIPTokenizer

from .base_model import BaseModel

logger = logging.getLogger("base")


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type="nsgan", target_real_label=1.0, target_fake_label=0.0):
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        if type == "nsgan":
            self.criterion = nn.BCELoss()
        elif type == "lsgan":
            self.criterion = nn.MSELoss()
        elif type == "hinge":
            self.criterion = nn.ReLU()
        else:
            raise NotImplementedError(f"Unsupported adversarial loss type: {type}")

    def patchgan(self, outputs, is_real=None, is_disc=None):
        if self.type == "hinge":
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            return (-outputs).mean()

        labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
        return self.criterion(outputs, labels)

    def __call__(self, outputs, is_real=None, is_disc=None):
        return self.patchgan(outputs, is_real, is_disc)


class DenoisingModel(BaseModel):
    """
    Prompt-aware DenoisingModel for the multi-scale PromptAdapter + hole-aware routing version.

    Main changes:
    1. feed_data(..., prompt=None) is supported.
    2. Frozen CLIP text encoder is initialized lazily and works in train/test.
    3. prompt_context and mask are passed into sde.noise_fn.
    4. Only parameters containing "prompt_adapter" are optimized by default.
    5. Uses hole loss + boundary context loss, with optional prompt ranking loss.
    """

    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        if opt.get("dist", False):
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1

        train_opt = opt.get("train", None)
        if train_opt is None:
            train_opt = {}

        self.prompt_drop_prob = float(train_opt.get("prompt_drop_prob", 0.0))
        self.lambda_prompt_rank = float(train_opt.get("lambda_prompt_rank", 0.0))
        self.prompt_rank_margin = float(train_opt.get("prompt_rank_margin", 0.01))
        self.prompt_rank_every = int(train_opt.get("prompt_rank_every", 4))

        self.lambda_hole = float(train_opt.get("lambda_hole", 1.0))
        self.lambda_boundary = float(train_opt.get("lambda_boundary", 2.0))
        self.boundary_kernel_size = int(train_opt.get("boundary_kernel_size", 5))

        self.clip_model_path = train_opt.get(
            "clip_model_path",
            opt.get("clip_model_path", "openai/clip-vit-large-patch14"),
        )
        self.enable_prompt = bool(train_opt.get("enable_prompt", True))

        self.tokenizer = None
        self.text_encoder = None
        self.prompt_context = None
        self.prompt_context_mismatch = None

        self.model, self.dis = networks.define_G(opt)
        self.model = self.model.to(self.device)
        self.dis = self.dis.to(self.device)
        self.model = DataParallel(self.model)
        self.dis = DataParallel(self.dis)

        self.load()

        if self.is_train:
            self.model.train()
            self.dis.train()

            if self.enable_prompt:
                self._init_text_encoder()

            adapter_found = False
            for k, v in self.model.named_parameters():
                if "prompt_adapter" in k:
                    v.requires_grad = True
                    adapter_found = True
                else:
                    v.requires_grad = False

            if not adapter_found:
                names = [k for k, _ in self.model.named_parameters()][:30]
                print("ERROR: cannot find any parameter containing 'prompt_adapter'.")
                print("First parameter names:", names)
                raise ValueError("PromptAdapter is not mounted. Please check DenoisingUNet_arch.py.")

            is_weighted = train_opt.get("is_weighted", False)
            loss_type = train_opt.get("loss_type", "l1")
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.loss_tri = nn.TripletMarginLoss().to(self.device)
            self.adversarial_loss = AdversarialLoss(type="hinge").to(self.device)
            self.weight = train_opt.get("weight", 1.0)

            self.optimizer_d = torch.optim.Adam(
                self.dis.parameters(), lr=1e-4, betas=(0.5, 0.99)
            )

            wd_G = train_opt.get("weight_decay_G", 0)
            if wd_G is None:
                wd_G = 0

            optim_params = []
            trainable_names = []
            frozen_count = 0

            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                    trainable_names.append(k)
                else:
                    frozen_count += 1
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if len(optim_params) == 0:
                raise ValueError("No trainable parameters found.")

            if self.rank <= 0:
                logger.info("Trainable parameter groups: %d", len(optim_params))
                logger.info("Frozen parameter tensors: %d", frozen_count)
                logger.info("Trainable names preview: %s", trainable_names[:50])
                print("[INFO] Trainable parameter names preview:")
                for name in trainable_names[:50]:
                    print("  ", name)

            optimizer_type = train_opt.get("optimizer", "Adam")
            if optimizer_type == "Adam":
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif optimizer_type == "AdamW":
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif optimizer_type == "Lion":
                self.optimizer = Lion(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print("Not implemented optimizer, default using Adam!")
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            self.optimizers.append(self.optimizer)

            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt.get("restarts", []),
                            weights=train_opt.get("restart_weights", []),
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt.get("clear_state", False),
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"],
                        )
                    )
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def _init_text_encoder(self):
        """Lazy initialization for frozen CLIP text encoder."""
        if not self.enable_prompt:
            return
        if self.tokenizer is not None and self.text_encoder is not None:
            return

        print(f"[INFO] Initializing CLIP Text Encoder from [{self.clip_model_path}]")
        logger.info("Initializing CLIP Text Encoder from [%s]", self.clip_model_path)

        self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_path)
        self.text_encoder = CLIPTextModel.from_pretrained(self.clip_model_path).to(self.device)
        self.text_encoder.eval()

        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def _normalize_prompt_batch(self, prompt, batch_size):
        if isinstance(prompt, tuple):
            prompt = list(prompt)
        elif isinstance(prompt, str):
            prompt = [prompt] * batch_size
        elif prompt is None:
            return None
        elif not isinstance(prompt, list):
            prompt = list(prompt)

        prompt = ["" if p is None else str(p) for p in prompt]

        if len(prompt) == 1 and batch_size > 1:
            prompt = prompt * batch_size

        if len(prompt) != batch_size:
            if len(prompt) > batch_size:
                prompt = prompt[:batch_size]
            else:
                last = prompt[-1] if prompt else ""
                prompt = prompt + [last] * (batch_size - len(prompt))

        return prompt

    def _encode_prompts(self, prompts):
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            return self.text_encoder(tokens.input_ids)[0]

    def feed_data(self, state, LQ, GT, mask, S_sde, S_GT, S_LQ, prompt=None):
        self.state = state.to(self.device)
        self.condition = LQ.to(self.device)
        self.state_0 = GT.to(self.device)
        self.mask = mask.to(self.device)
        self.S_sde = S_sde
        self.S_GT = S_GT.to(self.device)
        self.S_LQ = S_LQ.to(self.device)

        self.prompt_context = None
        self.prompt_context_mismatch = None

        if prompt is None or not self.enable_prompt:
            return

        self._init_text_encoder()

        prompts = self._normalize_prompt_batch(prompt, self.condition.shape[0])
        if prompts is None:
            return

        if self.is_train and self.prompt_drop_prob > 0:
            if random.random() < self.prompt_drop_prob:
                prompts = [""] * len(prompts)

        self.prompt_context = self._encode_prompts(prompts)

        if self.is_train and self.lambda_prompt_rank > 0 and len(prompts) > 1:
            perm = torch.randperm(len(prompts)).tolist()
            if all(i == perm[i] for i in range(len(perm))):
                perm = perm[1:] + perm[:1]

            mismatch_prompts = [prompts[i] for i in perm]
            self.prompt_context_mismatch = self._encode_prompts(mismatch_prompts)

    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)

        yt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        timesteps = timesteps.to(self.device)

        S_timestep, S_optimum = self.S_sde.generate_random_states_texture(
            x0=self.S_GT,
            mu=self.S_LQ * self.mask,
            timesteps=timesteps,
        )
        S_optimum = self.S_sde.reverse_optimum_step(S_optimum, self.S_GT, timesteps)

        noise, _ = sde.noise_fn(
            self.state,
            timesteps.squeeze(),
            S_optimum,
            prompt_context=self.prompt_context,
            mask=self.mask,
        )

        score = sde.get_score_from_noise(noise, timesteps)
        yt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)

        self.optimizer.zero_grad()

        loss_l1 = F.l1_loss(yt_1_expection, yt_1_optimum, reduction="none")

        hole_region = 1.0 - self.mask
        loss_hole = (loss_l1 * hole_region).mean()

        k = int(self.boundary_kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1

        kernel = torch.ones(1, 1, k, k, device=self.device)
        dilated_hole = F.conv2d(hole_region, kernel, padding=k // 2).clamp(0, 1)
        boundary_region = (dilated_hole - hole_region).clamp(0, 1)
        loss_boundary = (loss_l1 * boundary_region).mean()

        loss_recon = self.lambda_hole * loss_hole + self.lambda_boundary * loss_boundary
        loss = loss_recon

        loss_prompt_rank = torch.tensor(0.0, device=self.device)
        use_rank_loss = (
            self.is_train
            and self.lambda_prompt_rank > 0
            and self.prompt_context is not None
            and self.prompt_context_mismatch is not None
            and self.prompt_rank_every > 0
            and step % self.prompt_rank_every == 0
        )

        if use_rank_loss:
            noise_mis, _ = sde.noise_fn(
                self.state,
                timesteps.squeeze(),
                S_optimum,
                prompt_context=self.prompt_context_mismatch,
                mask=self.mask,
            )

            score_mis = sde.get_score_from_noise(noise_mis, timesteps)
            yt_1_mis = sde.reverse_sde_step_mean(self.state, score_mis, timesteps)

            loss_mis_map = F.l1_loss(yt_1_mis, yt_1_optimum, reduction="none")
            loss_mis_hole = (loss_mis_map * hole_region).mean()
            loss_mis_boundary = (loss_mis_map * boundary_region).mean()
            loss_mis = self.lambda_hole * loss_mis_hole + self.lambda_boundary * loss_mis_boundary

            loss_prompt_rank = F.relu(
                loss_recon - loss_mis.detach() + float(self.prompt_rank_margin)
            )
            loss = loss + float(self.lambda_prompt_rank) * loss_prompt_rank

        loss.backward()
        self.optimizer.step()

        self.log_dict["loss"] = loss.item()
        self.log_dict["loss_hole"] = loss_hole.item()
        self.log_dict["loss_bound"] = loss_boundary.item()
        self.log_dict["loss_prompt_rank"] = loss_prompt_rank.item()

    def test(
        self,
        sde=None,
        save_states=False,
        save_dir="save_dir",
        GT=None,
        mask=None,
        S_sde=None,
        S_GT=None,
        S_LQ=None,
        structure_guide=None,
        **kwargs,
    ):
        sde.set_mu(self.condition)
        self.model.eval()

        with torch.no_grad():
            forward_kwargs = {
                "mask": self.mask,
            }

            if hasattr(self, "prompt_context") and self.prompt_context is not None:
                forward_kwargs["prompt_context"] = self.prompt_context

            if hasattr(self, "S_LQ") and self.S_LQ is not None:
                forward_kwargs["S"] = self.S_LQ

            self.output = sde.reverse_sde(
                self.state,
                save_states=save_states,
                save_dir=save_dir,
                **forward_kwargs,
            )

        self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__,
                self.model.module.__class__.__name__,
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)

        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            print("load-------------------------------")
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(
                load_path_G,
                self.model,
                self.opt["path"]["strict_load"],
            )

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
