# Visible-evidence-guided

## 基于可见证据引导与掩膜感知多模态扩散的敦煌壁画虚拟补全

## 1. 数据与评价设置

### Dunhuang

Dunhuang Grottoes Painting Dataset and Benchmark 包含 500 幅开发图像和 100 幅独立测试图像。保持官方图像—掩膜配对。配置阶段以随机种子 42 从开发集抽取 50 幅作为验证集；配置确定后使用全部 500 幅开发图像训练。测试集不进行训练掩膜增强。

官方入口：<https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge>

### MuralDH

原始资源：<https://github.com/tearsheaven/MuralDH>

受控外部评价使用相关工作 760/201 划分评价子集中的 201 个真实损伤掩膜，并与随机种子 42 无放回抽取的 201 幅参考图像固定配对。参考 ID 与掩膜 ID 分别排序后对应。

参考式评价用于衡量真实损伤几何条件下被隐藏参考像素的重建一致性。原始真实受损图像另用于无参考质量评价。

相关划分来源：LABFNet，DOI：<https://doi.org/10.3390/jimaging12070332>

### muralv2

表 4 使用 PGRDiff 开源项目提供的 `Crack`、`FallenOff`、`Scratch`、`InsectInfestation` 四类掩膜。

下载入口：<https://github.com/CZY-Code/PGRDiff>

表 4 的数值均由本文评价流程重新计算。

## 2. 提示与控制实验

最终提示包含三个字段：

- `Content and style`
- `Degradation`
- `Restoration constraint`

`Degradation` 描述可观察退化状态及损伤邻近区域的可见边界或色彩过渡特征。精确损伤几何由二值掩膜提供。提示仅依据掩膜后可见证据生成，不访问缺失区域目标像素。

| 条件 | 设置 |
|---|---|
| Matched / sample-specific | 当前图像与其对应提示配对 |
| Fixed generic | 所有样本使用同一固定通用提示 |
| Shuffled | 图像与其他样本提示固定错配 |
| Zero-feature | 令 `C_p = 0`，保留 Adapter |
| Parameter-matched training | 保持网络结构与 Adapter 参数量一致，仅改变训练提示策略 |

## 3. 参考式评价脚本

### 3.1 脚本与依赖

评价脚本：

```text
eval_paired_predictions_paper_documented.py
```

依赖：

```bash
python -m pip install numpy pandas opencv-python scikit-image torch lpips
```

### 3.2 输入目录

```text
evaluation_inputs/
  predictions/
    img_1158crop_1_0.png
    img_1158crop_1_1.png
  references/
    img_1158crop_1_0.png
    img_1158crop_1_1.png
  masks/
    img_1158crop_1_0_mask.png
    img_1158crop_1_1_mask.png
```

预测图、参考图与掩膜按归一化文件名主干配对。RGB 图像使用 `INTER_AREA` 缩放，掩膜使用 `INTER_NEAREST`。默认评价分辨率为 256×256。

`--mask-convention known_white`：白色为已知区域，黑色为缺失区域。  
`--mask-convention hole_white`：白色为缺失区域。

### 3.3 运行命令

#### Dunhuang

```bash
python eval_paired_predictions_paper_documented.py \
  --pred-dir /path/to/dunhuang/predictions \
  --gt-dir /path/to/dunhuang/references \
  --mask-dir /path/to/dunhuang/masks \
  --outdir evaluation_results/dunhuang \
  --dataset-name Dunhuang \
  --expected 100 \
  --mask-convention known_white \
  --size 256 \
  --device cuda \
  --canny-low 100 \
  --canny-high 200 \
  --boundary-kernel 5 \
  --edge-tolerance 1.0
```

#### MuralDH controlled evaluation

```bash
python eval_paired_predictions_paper_documented.py \
  --pred-dir /path/to/muraldh_controlled/predictions \
  --gt-dir /path/to/muraldh_controlled/references \
  --mask-dir /path/to/muraldh_controlled/masks \
  --outdir evaluation_results/muraldh_controlled \
  --dataset-name MuralDH-controlled \
  --expected 201 \
  --mask-convention known_white \
  --size 256 \
  --device cuda \
  --canny-low 100 \
  --canny-high 200 \
  --boundary-kernel 5 \
  --edge-tolerance 1.0
```

### 3.4 输出

| 文件 | 内容 |
|---|---|
| `per_image_metrics.csv` | `id`、`hole_ratio`、逐图指标 |
| `summary.csv` | 数据集名称、记录数、平均缺失比例、指标均值 |
| `stratified_gpsnr.csv` | `[0,0.2)`、`[0.2,0.4)`、`[0.4,+∞)` 分层 G-PSNR |

## 4. 指标定义

### 4.1 G-PSNR / H-PSNR

RGB 像素范围为 `[0,1]`。G-PSNR 在全图计算；H-PSNR 仅在缺失区域计算。
```math
\mathrm{MSE}_H =
\frac{\sum_p H(p)\sum_{c=1}^{3}(x_c(p)-\hat{x}_c(p))^2}
{3\sum_p H(p)},
\qquad
\mathrm{H\!-\!PSNR}
=
10\log_{10}\frac{1}{\mathrm{MSE}_H}.
```
### 4.2 G-SSIM / H-SSIM

三个 RGB 通道分别调用：

```python
structural_similarity(
    gt[..., c],
    pr[..., c],
    data_range=1.0,
    win_size=7,
    gaussian_weights=False,
    use_sample_covariance=True,
    full=True,
)
```

三个局部 SSIM 图取平均：
```math
\bar{S}(p)
=
\frac{S_R(p)+S_G(p)+S_B(p)}{3}.
```
H-SSIM：
```math
\mathrm{H\!-\!SSIM}
=
\frac{\sum_p H(p)\bar{S}(p)}
{\sum_p H(p)}.
```
G-SSIM 为三个通道全图 SSIM 标量的平均。

实现文档：<https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity>

### 4.3 LPIPS

使用：

```python
lpips.LPIPS(net="alex")
```

完整 RGB 图像由 `[0,1]` 映射到 `[-1,1]` 后计算。

### 4.4 KB-F1

已知侧评价边界带：
```math
B_{\mathrm{eval}}
=
\operatorname{Dilate}_{5}(H)-H.
```
设置：

- Canny 阈值：100 / 200
- 边界膨胀核：5×5
- 距离：OpenCV `DIST_L2`
- 距离变换掩模尺寸：5
- 匹配容差：1 像素
- 双向 precision / recall 后计算 F1

## 5. 训练与评价配置

- 输入分辨率：256×256
- Batch size：8
- 主模型随机种子：1234、2024、3407
- 优化器：Adam
- 初始学习率：`1e-5`
- 学习率衰减：10,000 / 16,000 iterations × 0.5
- 训练长度：20,000 iterations
- Diffusion steps：100
- Noise schedule：cosine
- Maximum noise intensity：30
- `epsilon_SDE = 0.005`
- 硬件：单张 NVIDIA RTX 4090

LaMa、StrDiffusion 和 PGRDiff 使用统一 Dunhuang 训练数据、掩膜、分辨率、batch size、优化器、学习率计划和训练预算。PGRDiff 使用与本文方法相同的 sample-specific prompt 文本。

## 6. 数据与代码可用性

- Dunhuang：<https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge>
- MuralDH：<https://github.com/tearsheaven/MuralDH>
- muralv2 / PGRDiff：<https://github.com/CZY-Code/PGRDiff>

原始图像、掩膜及第三方权重的使用和再分发遵循各自许可。

## 7. 引用

使用本项目时，请引用论文正式发表版本及相关基础方法和数据资源。
