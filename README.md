# Visible-evidence-guided virtual completion of Dunhuang murals using mask-aware multimodal diffusion

**基于可见证据引导与掩膜感知多模态扩散的敦煌壁画虚拟补全**

本仓库对应上述论文的研究方法与评价流程。方法以**冻结的 StrDiffusion** 为结构引导骨干，通过 Qwen2-VL 生成仅基于可见内容的文本提示，使用冻结的 CLIP 编码文本，再由**可训练的多尺度 Prompt Adapter** 注入语义残差。**Mask-aware Routing** 根据二值掩膜限制残差的直接注入位置，不引入可训练参数。

> **研究范围**：本文研究的是数字虚拟补全（digital virtual completion）。带参考图像的合成遮挡评价衡量像素重建一致性；真实损伤掩膜评价衡量在真实损伤几何下的受控重建表现。两者均**不能证明生成内容就是历史原貌**。

## 1. 方法与关键定义

### 1.1 模型组成

| 模块 | 来源及作用 | 训练状态 |
|---|---|---|
| 结构与纹理恢复骨干 | StrDiffusion；继承 IR-SDE 随机过程和 SPADE 结构调制 | 冻结 |
| 可见证据提示生成 | Qwen2-VL-7B-Instruct，离线生成样本专属英文描述 | 不参与端到端训练 |
| 文本编码器 | CLIP ViT-L/14，最大 77 tokens，特征维度 768 | 冻结 |
| Multi-scale Prompt Adapter | 在 U-Net 的 `mid`、`dec1`、`dec2`、`dec3` 阶段融合 FiLM 和 cross-attention 残差 | **唯一可训练模块** |
| Mask-aware Routing | 依据二值掩膜构造 hole 和 known-side boundary 的残差注入门控 | 无可训练参数 |

StrDiffusion 的主网络采用六通道输入与三通道预测；其结构网络采用可见灰度图与边缘图作为条件。本文没有提出新的扩散随机过程，也不将 Qwen2-VL、CLIP、FiLM 或 cross-attention 视为独立新算法。

### 1.2 掩膜、结构条件与路由

**二值掩膜约定必须统一：`M=1` 为已知像素，`M=0` 为缺失像素。** 对参考图像 `x_0`，定义缺失区域和掩膜输入为：

$$
H=1-M,\qquad \mu=M\odot x_0.
$$

结构条件仅由掩膜后的输入构造，不从被遮挡区域的参考像素生成：

$$
G_{\mu}=M\odot\Phi_{\mathrm{gray}}(\mu),\qquad
E_{\mu}=M\odot\Phi_{\mathrm{edge}}(G_{\mu}),\qquad
S=\mathrm{Concat}(G_{\mu},E_{\mu}).
$$

论文中**路由、监督与评价**的边界核有不同作用，即使最终取值相同也不可混用：

| 参数 | 作用 | 论文最终设置 |
|---|---|---|
| `k_r` | Prompt residual 的 known-side routing boundary | `5` |
| `k_sup` | boundary-weighted training loss 的监督边界 | `5` |
| `k_eval` | KB-F1 的已知侧评价边界 | `5` |

路由侧边界通过一次方形核膨胀构造：

$$
B_{\mathrm{route}}^{(k_r)}
=\mathrm{Dilate}_{k_r}(H)-H.
$$

其中 `k_r` 为奇数大小的方形膨胀核；`k_r=1` 不额外产生 known-side boundary，因此只在缺失区内注入。将 hole 与边界图以**最近邻插值**缩放至第 `s` 个特征尺度：

$$
R_s^{(k_r)}(M)=
\mathrm{clip}\left(
\mathrm{Resize}_s(H)+
\mathrm{Resize}_s\left(B_{\mathrm{route}}^{(k_r)}\right),
0,1\right).
$$

残差注入为：

$$
h'_s=h_s+R_s^{(k_r)}(M)\odot r_p^{(s)}.
$$

路由图仅控制 Prompt Adapter 生成的语义残差**直接注入的位置**；它不单独预测语义，也不意味着整个扩散网络对已知区域完全没有间接影响。

### 1.3 边界加权训练目标

监督边界与路由边界分开定义：

$$
B_{\mathrm{sup}}=\mathrm{Dilate}_{k_{\mathrm{sup}}}(H)-H,
\qquad k_{\mathrm{sup}}=5.
$$

在冻结 StrDiffusion 的情况下，仅更新 Prompt Adapter，使用下列目标：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{hole}}+
2\mathcal{L}_{\mathrm{boundary}}.
$$

$$
\mathcal{L}_{\mathrm{hole}}
=\frac{\left\lVert H\odot
(\hat{x}_{t-1}-x^{*}_{t-1})\right\rVert_1}
{\lVert H\rVert_1+\epsilon},
\qquad
\mathcal{L}_{\mathrm{boundary}}
=\frac{\left\lVert B_{\mathrm{sup}}\odot
(\hat{x}_{t-1}-x^{*}_{t-1})\right\rVert_1}
{\lVert B_{\mathrm{sup}}\rVert_1+\epsilon}.
$$

这里 `x*_{t-1}` 是 StrDiffusion 逆过程的解析监督目标，`\epsilon>0` 是防止除零的数值项；**边界加权损失是本文方法的组成部分，不是从原始 StrDiffusion 直接继承的损失**。

## 2. 可见证据提示与对照实验

### 2.1 论文最终提示模板

Qwen2-VL-7B-Instruct **离线**读取掩膜后的壁画与二值掩膜，使用固定模板、greedy decoding、禁用采样。生成的文本与图像—掩膜记录绑定，训练/评价复用对应提示。不得将缺失区域的参考像素传给提示生成器。

**System instruction（与论文 Methods 一致）**

```text
Describe only visible evidence in the supplied masked Dunhuang mural image and binary
mask. Return one concise English description using the fields “Visible content”,
“Painted style and palette”, and “Mask morphology”. Do not infer or name content
hidden by the mask.
```

**User template**

```text
Masked mural image: <IMAGE>. Binary mask: <MASK>.
Produce the description in the required three-field format.
```

**三个字段（请勿使用旧版 `Content and style / Degradation / Restoration constraint`）**

```text
Visible content: [subjects, composition, and decorations that remain visible]
Painted style and palette: [visible colours, line work, and painting style]
Mask morphology: [coarse description of the observed mask geometry]
```

上述方括号是**字段用途说明，不是论文实验中的实际提示样例**。`Mask morphology` 只是粗粒度语言背景，不负责提供精确空间几何；精确 hole、边界和门控均来自二值掩膜。

冻结的 CLIP 将提示编码为：

$$
C_p=E_{\mathrm{text}}(P)\in\mathbb{R}^{B\times77\times768}.
$$

Adapter 使用 token-level cross-attention 和 FiLM 调制，经过尺度专属投影生成 `r_p`；训练只调整 Adapter 的参数。

### 2.2 提示与参数量控制

| 条件 | 含义 |
|---|---|
| Matched / sample-specific | 当前图像及掩膜使用对应的完整可见证据提示 |
| Partial omission | 固定权重下删除一个可见属性，其余文本不变 |
| Localized factual error | 固定权重下将同一可见属性改为错误值，其余文本不变 |
| Fixed generic | 全部样本使用同一通用壁画描述 |
| Shuffled | 使用预定置换，将其他测试样本的提示分配给当前输入 |
| Zero-feature | 保留 Adapter 架构，但将文本特征设为 `C_p=0` |
| Parameter-matched retraining | Adapter 架构和参数量保持一致，分别重新训练不同提示条件 |

论文 Table 8 为**同一 seed-1234 模型固定权重、仅改变推理文本**的控制；Table 9 为**保持 Adapter 架构及参数量、独立训练**的控制。这些提示对照主要是单种子实验，不应描述为三种子显著性检验。

## 3. 数据、掩膜与划分

### 3.1 Dunhuang

[Dunhuang Grottoes Painting Dataset and Benchmark](https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge) 共 600 幅图像，使用官方的 **500 幅 development / 100 幅 independent test** 划分，并保持官方一对一的图像—掩膜配对。

- 配置选择：以随机种子 `42` 从 500 幅 development 图像抽取 50 幅验证图像（其余 450 幅用于配置阶段训练）。
- 最终训练：配置锁定后，使用全部 500 幅 development 图像。
- 独立测试：使用官方 100 幅图像与其对应掩膜；**不进行随机训练掩膜增强**。
- 掩膜预处理：灰度化、最近邻缩放至 `256 × 256`、归一化、以 `0.5` 二值化；白色/1 为已知，黑色/0 为缺失。
- **仅训练期间**：当初始缺失比例小于 `0.15` 时，以 `0.5` 的概率额外绘制 1–4 条取零线段，线宽均匀采样于 15–45 pixels。水平翻转、垂直翻转和配置中的旋转对图像与掩膜同步应用。

线段增强旨在扩充训练缺失比例和尺度分布，**不是**对龟裂、剥落或磨蚀物理机理的真实模拟。

### 3.2 MuralDH：真实损伤掩膜的受控外部评价

- 数据资源：[MuralDH](https://github.com/tearsheaven/MuralDH)。
- 参考划分：[LABFNet](https://doi.org/10.3390/jimaging12070332) 所使用的 `760/201` 像素级损伤数据划分。
- 受控评价：选取其中评价子集的 **201 个真实损伤掩膜**；从 MuralDH 参考图像集合以 seed `42` **无放回**抽取 201 幅图像。参考 ID 与掩膜 ID 分别排序后按顺序固定配对。
- **关键限制**：参考图像与真实受损图像并非同一实体壁画的真实损伤前后配对；这些指标只衡量真实损伤几何下被隐藏参考像素的重建一致性。
- Ours 不使用 MuralDH 训练、微调或选择超参数。
- 原始真实受损的 MuralDH 图像另以 ARNIQA、BRISQUE、DBCNN 作**无参考补充评价**；因各方法有效记录覆盖不同，论文 Table 5 是描述性比较，不能视作严格配对排名。

论文还对 Dunhuang 与 MuralDH 参考图像进行了文件/像素 SHA-256 和 pHash/dHash 重复检查；检查未发现所述跨数据集重复候选，但不能据此推断两个数据资源在风格或主题上独立。

### 3.3 muralv2：损伤模式分析

论文 Table 4 使用 [PGRDiff](https://github.com/CZY-Code/PGRDiff) 项目提供的 `muralv2` 资源中的掩膜，经本文**二次提取、整理**形成如下四个分析子集：

- `Crack`
- `Fallen-off / irregular loss`
- `Decorative-pattern damage`
- `Interrupted contour`

这些是**本文重组后的分析标签，不应写作 PGRDiff 原始官方类别名**。Table 4 的结果由论文评价流程重新计算，且应结合各子集缺失面积比例解读。

## 4. 训练与公平比较设置

| 项目 | 论文设置 |
|---|---|
| 输入分辨率 | `256 × 256` |
| Batch size / workers | `8 / 8` |
| 主网络 / 结构网络 | ConditionalUNet，基础宽度 `64`，深度 `4` |
| 训练种子 | `1234`、`2024`、`3407`（最终模型及 Table 6 的 B–E） |
| 单因子控制 | 除另有说明，seed `1234` |
| 优化器 | Adam，`beta1=0.9`，`beta2=0.999` |
| 初始学习率 | `1e-5`，无 warm-up |
| 学习率计划 | 第 `10,000`、`16,000` 次迭代各乘 `0.5` |
| 训练时长 | `20,000` iterations |
| 随机过程 | `100` steps，cosine schedule，maximum noise intensity `30`，`epsilon_SDE=0.005` |
| 硬件 | 单张 NVIDIA RTX 4090，非分布式 |
| 最终路由核 | `k_r=5`（依据验证集选择） |

**基线协议**：LaMa、StrDiffusion（matched training）和 PGRDiff 使用相同的 Dunhuang 训练图像、掩膜、分辨率、batch size、Adam 优化器、学习率计划与训练预算；各方法保留自己的原生网络结构与损失函数。PGRDiff 在训练与测试时使用与 Ours 相同的样本专属提示。EdgeConnect、PowerPaint、RAD、LRDiff 和 MuralNet 使用作者发布权重；PowerPaint 因发布模型存在提示长度限制，采用其官方提示生成与推理流程。

请区分论文 Table 6 的 **A: official frozen StrDiffusion** 与主表的 **StrDiffusion (matched training)**，二者并不是同一评价配置。

## 5. 评价指标与实现细节

### 5.1 G-PSNR 和 H-PSNR

输入 RGB 数值范围为 `[0,1]`。G-PSNR 在完整图像上计算，H-PSNR 仅在 hole 位置计算：

$$
\mathrm{MSE}_{H}
=\frac{\sum_p H(p)\sum_{c=1}^{3}
(x_c(p)-\hat{x}_c(p))^2}{3\sum_p H(p)},
\qquad
\mathrm{H\!-\!PSNR}
=10\log_{10}\frac{1}{\mathrm{MSE}_{H}}.
$$

当误差为零时 PSNR 为正无穷；论文在汇总时只对有限结果求平均。

### 5.2 G-SSIM 和 H-SSIM

针对 RGB 三通道分别调用：

```python
from skimage.metrics import structural_similarity

score, local_map = structural_similarity(
    gt[..., c],
    prediction[..., c],
    data_range=1.0,
    win_size=7,
    gaussian_weights=False,
    use_sample_covariance=True,
    full=True,
)
```

使用默认的 `K1=0.01` 和 `K2=0.03`；三通道完整局部 SSIM 图先平均，再根据 hole map 选择**局部窗口中心位于缺失区**的像素：

$$
\bar{s}(p)=\frac{1}{3}\sum_{c=1}^{3}s_c(p),
\qquad
\mathrm{H\!-\!SSIM}
=\frac{\sum_p H(p)\bar{s}(p)}{\sum_p H(p)}.
$$

注意：H-SSIM 的局部窗口可以包含缺失区周围的已知像素；它既不是只在 hole 内计算局部统计量，也不是裁剪 hole 外接框后计算 SSIM。G-SSIM 取各通道的全图 scalar SSIM 平均（各 scalar 排除图像外侧 3 像素边界）；H-SSIM 直接从完整 local map 选取 hole 中心，不另行去掉边界。

### 5.3 LPIPS

使用 `lpips.LPIPS(net="alex")`，完整 RGB 图像先从 `[0,1]` 映射到 `[-1,1]`。**LPIPS 越低越好**；与 PSNR/SSIM 可能存在指标取舍。

### 5.4 KB-F1：已知侧边界边缘连续性

**评价边界与训练路由边界是不同概念：**

$$
B_{\mathrm{eval}}
=\mathrm{Dilate}_{k_{\mathrm{eval}}}(H)-H,
\qquad k_{\mathrm{eval}}=5.
$$

- 对完整 RGB 图先转 8-bit 灰度，再使用 Canny `100/200` 提取边缘。
- 仅保留位于已知侧评价边界带 `B_eval` 的边缘像素。
- 使用 OpenCV `DIST_L2` 距离变换，`maskSize=5`，匹配容差 `1` 像素。
- 双向匹配获得 precision `P` 和 recall `R`，再计算 `2PR/(P+R)`。
- 两个边缘集合都为空时定义 KB-F1 为 `1`；只有一方为空或 `P+R=0` 时定义为 `0`。

### 5.5 主要论文结果（非新的运行结果）

| 数据集 | 方法 | G-PSNR ↑ | H-PSNR ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| Dunhuang | LaMa (matched) | 35.3000 | 24.9412 | **0.0163** |
| Dunhuang | Ours（3 seeds，mean ± SD） | **36.6212 ± 0.0391** | **27.0445 ± 0.0670** | 0.0433 ± 0.0004 |
| MuralDH controlled | LaMa (matched) | 32.7210 | 16.5989 | 0.0432 |
| MuralDH controlled | Ours（3 seeds，mean ± SD） | **35.8066 ± 0.0595** | **18.6842 ± 0.0810** | **0.0375 ± 0.0005** |

Dunhuang 上 LaMa 的 LPIPS 优于 Ours，不能将论文结果笼统描述为“所有指标都最好”。三名相关领域专家对 30 个分层抽取的案例进行了盲评；论文报告 Ours 总评分 `4.14 ± 0.44`，95% CI 为 `[3.98, 4.30]`。该评价是针对已知数字参考图的领域意见，**不是对历史真实性的鉴定**；论文未量化评价者间一致性。

## 6. 参考式指标评价脚本

> 本节保留用户原 README 中声明的脚本接口：`eval_paired_predictions_paper_documented.py`。**只有实际取得该脚本、并确认其 CLI 参数与下方一致后，命令才能运行。README 不能替代尚未提交的代码文件。**

### 6.1 Python 依赖

```bash
python -m pip install numpy pandas opencv-python scikit-image torch lpips
```

### 6.2 预测、参考和掩膜目录

```text
evaluation_inputs/
├── predictions/
│   ├── sample_001.png
│   └── sample_002.png
├── references/
│   ├── sample_001.png
│   └── sample_002.png
└── masks/
    ├── sample_001_mask.png
    └── sample_002_mask.png
```

预测图、参考图与掩膜应按规范化文件名主干正确配对；RGB 图像在需要缩放时使用 `INTER_AREA`，掩膜使用 `INTER_NEAREST`。`--mask-convention known_white` 表示白色是已知区域，`hole_white` 表示白色是缺失区域；**论文主协议为 `known_white`**。

### 6.3 Dunhuang 独立测试

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

### 6.4 MuralDH 受控外部评价

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

原 README 描述的结果文件：

| 文件 | 内容 |
|---|---|
| `per_image_metrics.csv` | 样本 ID、缺失比例及逐图指标 |
| `summary.csv` | 数据集、有效记录数、平均缺失比例与指标汇总 |
| `stratified_gpsnr.csv` | 按缺失比例分层的 G-PSNR |

**复现检查**：按固定 ID 对齐预测、参考与掩膜；报告有效记录数与缺失比例；不要将不同 mask convention、不同测试图像或不同 baseline 训练协议混在一张无说明的表中。

## 7. 数据、代码与研究边界

- Dunhuang 官方入口：<https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge>
- MuralDH：<https://github.com/tearsheaven/MuralDH>
- MuralDH 相关 760/201 划分论文：<https://doi.org/10.3390/jimaging12070332>
- muralv2 / PGRDiff：<https://github.com/CZY-Code/PGRDiff>
- 论文代码仓库：<https://github.com/xinyu0833-bit/Visible-evidence-guided>

论文 Data Availability 说明：研究特定的固定记录 ID、确定性提示、派生掩膜子集、YAML 配置、输出 manifests 和 metric summaries 可向通讯作者合理请求。**若要主张完整公开复现，应将实际允许发布的脚本、配置、记录清单、提示与权重另行提交，并给出真实路径和许可证**；原始数据和第三方权重须遵循来源许可。

本方法仍受真实损伤区域无原貌真值、复杂装饰纹样、大面积缺损、掩膜准确性及冻结骨干能力所限。论文中的 reference-based 指标、无参考质量指标和专家评价分别回答不同问题，不应混为“历史真实性”的证据。

## 8. 引用

论文标题：*Visible-evidence-guided virtual completion of Dunhuang murals using mask-aware multimodal diffusion*。

正式发表后请使用出版方提供的 DOI 和 BibTeX；在此之前不填写未经确认的刊期或 DOI。
