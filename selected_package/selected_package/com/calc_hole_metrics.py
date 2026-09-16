import os
import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

def calculate_hole_metrics():
    # 1. 配置路径 
    results_dir = '/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_combined_ours&baseline/DUNHUANG_Test'
    mask_dir = '/root/autodl-tmp/data/mural/DUNHUANG/test/test_mask'

    # ✨ 修复 1：在生成目录里找的是原图 _r.png，千万别写成 _mask.jpg
    r_images = glob.glob(os.path.join(results_dir, '*_r.png'))
    
    if not r_images:
        print(f"❌ 在 {results_dir} 找不到原图！请检查路径。")
        return

    hole_psnrs, hole_ssims = [], []
    global_psnrs, global_ssims = [], []

    print(f"🚀 开始计算 {len(r_images)} 张图像的 Hole-Metrics...")

    for r_path in r_images:
        # 从 '501_r.png' 提取出 '501'
        base_name = os.path.basename(r_path).replace('_r.png', '')
        
        # 寻找对应的修复图 (SoftBlend.png)
        f_matches = glob.glob(os.path.join(results_dir, f"{base_name}_*f.png"))
        if not f_matches: 
            continue
        f_path = f_matches[0]

        # ✨ 修复 2：去 mask_dir 寻找对应的 JPG 掩码，拼接为 {base_name}_mask.jpg
        m_path = os.path.join(mask_dir, f"{base_name}_mask.jpg")
        if not os.path.exists(m_path):
            print(f"⚠️ 找不到 Mask: {m_path}，跳过")
            continue

        # 读取生成图和原图，并归一化到 0-1
        img_gt = cv2.imread(r_path).astype(np.float32) / 255.0
        img_pred = cv2.imread(f_path).astype(np.float32) / 255.0
        
        # ✨ 动态获取当前图片的宽高 (通常是 256, 256)
        h, w = img_gt.shape[:2]

        # ✨ 修复：读取 JPG 掩码，并将其 Resize 到与预测图完全一致的尺寸！
        mask_raw = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 进行严苛的二值化处理，消除 JPG 压缩噪点
        mask = (mask_raw > 127).astype(np.float32) # 大于 127 的强转为 1 (完好)，否则为 0 (破损)
        
        # 扩展为 3 通道以便矩阵运算
        mask_3c = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)

        # 定义破损区域: hole_mask = 1 - mask
        hole_mask_3c = 1.0 - mask_3c
        hole_mask_1c = 1.0 - mask
        
        # ---------------------------------------------------------
        # 🎯 计算 Hole-PSNR (仅在大洞区域的 MSE)
        # ---------------------------------------------------------
        mse_hole = np.sum(((img_gt - img_pred) ** 2) * hole_mask_3c) / (np.sum(hole_mask_3c) + 1e-8)
        hole_psnr = 10 * np.log10(1.0 / mse_hole) if mse_hole > 0 else 99.99
        hole_psnrs.append(hole_psnr)

        mse_global = np.mean((img_gt - img_pred) ** 2)
        global_psnrs.append(10 * np.log10(1.0 / mse_global))

        # ---------------------------------------------------------
        # 🎯 计算 Hole-SSIM (利用 full=True 导出 SSIM 矩阵映射图)
        # ---------------------------------------------------------
        _, ssim_map = ssim_metric(img_gt, img_pred, channel_axis=2, full=True, data_range=1.0)
        ssim_map_gray = np.mean(ssim_map, axis=2)
        
        hole_ssim = np.sum(ssim_map_gray * hole_mask_1c) / (np.sum(hole_mask_1c) + 1e-8)
        hole_ssims.append(hole_ssim)
        
        global_ssims.append(np.mean(ssim_map_gray))

    # --- 打印最终成绩单 ---
    print("\n" + "="*50)
    print("🏆 论文级精准评估报告 (Mask-Aware Metrics)")
    print("="*50)
    print(f"🌍 全局 PSNR (Global): {np.mean(global_psnrs):.4f}")
    print(f"🌍 全局 SSIM (Global): {np.mean(global_ssims):.4f}")
    print("-" * 50)
    print(f"🎯 破损区 PSNR (Hole-PSNR): {np.mean(hole_psnrs):.4f}")
    print(f"🎯 破损区 SSIM (Hole-SSIM): {np.mean(hole_ssims):.4f}")
    print("="*50)

if __name__ == "__main__":
    calculate_hole_metrics()