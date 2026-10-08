# Visible-evidence-guided virtual completion of Dunhuang murals using mask-aware multimodal diffusion

**基于可见证据引导与掩膜感知多模态扩散的敦煌壁画虚拟补全**

本项目面向敦煌壁画的数字虚拟补全，在冻结的 StrDiffusion 结构引导扩散骨干上结合可见证据文本提示、多尺度 Prompt Adapter 和掩膜感知残差路由。通过受控重建评价、消融实验、提示鲁棒性分析及领域专家盲评，评估缺失区域的结构连续性、风格一致性与补全质量。

## 1. 方法框架

### 1.1 网络组成

| 组件 | 实现 | 参数状态 |
|---|---|---|
| 纹理与结构恢复骨干 | StrDiffusion；继承 IR-SDE 随机过程和 SPADE 结构调制 | 冻结 |
| 可见证据提示生成器 | Qwen2-VL-7B-Instruct；离线、确定性生成文本 | 冻结 |
| 文本编码器 | CLIP ViT-L/14；77 tokens，768 维特征 | 冻结 |
| Multi-scale Prompt Adapter | FiLM 与多头 cross-attention；注入层为 `mid`、`dec1`、`dec2`、`dec3` | 训练 |
| Mask-aware Routing | 基于二值掩膜的空间残差门控 | 无可训练参数 |

主纹理网络采用六通道输入和三通道逆状态预测；辅助结构网络采用双通道结构条件。SPADE 根据结构条件提供空间相关的特征调制。优化过程中仅更新 Prompt Adapter 参数。

### 1.2 掩膜与结构条件

二值掩膜 `M` 的取值约定为 **1 = 已知区域，0 = 缺失区域**。设 `x_0` 为完整参考图像，则缺失图 `H` 与输入条件图 `μ` 定义为：

```math
H = 1-M,\qquad \mu=M\odot x_0
```

结构条件由掩膜后可见信息构造：

```math
G_{\mu}=M\odot\Phi_{\mathrm{gray}}(\mu)
```

```math
E_{\mu}=M\odot\Phi_{\mathrm{edge}}(G_{\mu}),\qquad S=\mathrm{Concat}(G_{\mu},E_{\mu})
```

### 1.3 可见证据文本提示

Qwen2-VL-7B-Instruct 使用掩膜后壁画图像和二值掩膜，采用固定指令与 greedy decoding，在禁用采样的条件下为各图像—掩膜记录生成英文提示。提示生成不使用被遮挡区域的参考像素。

**System instruction**

```text
Describe only visible evidence in the supplied masked Dunhuang mural image and binary mask.
Return one concise English description using the fields “Visible content”, “Painted style
and palette”, and “Mask morphology”. Do not infer or name content hidden by the mask.
```

**User template**

```text
Masked mural image: <IMAGE>. Binary mask: <MASK>.
Produce the description in the required three-field format.
```

| 提示字段 | 信息范围 |
|---|---|
| `Visible content` | 可见人物、构图、装饰及其他图像内容 |
| `Painted style and palette` | 可见色彩、线条及绘画风格 |
| `Mask morphology` | 掩膜整体形态的粗粒度描述 |

`Mask morphology` 仅作为辅助语义上下文；像素级缺失几何、已知侧边界和残差路由均由二值掩膜确定。冻结 CLIP 编码样本提示：

```math
C_p=E_{\mathrm{text}}(P)\in\mathbb{R}^{B\times77\times768}
```

### 1.4 多尺度残差路由

令 `k_r` 为路由边界所用的奇数大小方形膨胀核。已知侧边界定义为：

```math
B_{\mathrm{route}}^{(k_r)}=\mathrm{Dilate}_{k_r}(H)-H
```

在第 `s` 个特征尺度上，缺失图及边界图通过最近邻插值缩放，构造残差门控：

```math
R_s^{(k_r)}(M)=\mathrm{clip}\bigl(\mathrm{Resize}_s(H)+\mathrm{Resize}_s(B_{\mathrm{route}}^{(k_r)}),0,1\bigr)
```

Prompt Adapter 生成的尺度专属残差通过门控注入 U-Net：

```math
h'_s=h_s+R_s^{(k_r)}(M)\odot r_p^{(s)}
```

`k_r = 1` 对应 hole-only routing，最终采用 `k_r = 5` 的 hole + known-side boundary routing。

### 1.5 边界加权优化目标

监督边界由独立的膨胀核构造：

```math
B_{\mathrm{sup}}=\mathrm{Dilate}_{k_{\mathrm{sup}}}(H)-H,\qquad k_{\mathrm{sup}}=5
```

Prompt Adapter 使用缺失区域损失和边界区域损失联合优化：

```math
\mathcal{L}=\mathcal{L}_{\mathrm{hole}}+2\mathcal{L}_{\mathrm{boundary}}
```

```math
\mathcal{L}_{\mathrm{hole}}=\frac{\|H\odot(\hat{x}_{t-1}-x^*_{t-1})\|_1}{\|H\|_1+\epsilon}
```

```math
\mathcal{L}_{\mathrm{boundary}}=\frac{\|B_{\mathrm{sup}}\odot(\hat{x}_{t-1}-x^*_{t-1})\|_1}{\|B_{\mathrm{sup}}\|_1+\epsilon}
```

其中 `x*_{t-1}` 为由 StrDiffusion 随机过程及离散化确定的解析监督目标，`ε` 为数值稳定项。

| 核参数 | 使用环节 | 最终取值 |
|---|---|---|
| `k_r` | 语义残差路由 | `5` |
| `k_sup` | 边界加权训练损失 | `5` |
| `k_eval` | KB-F1 已知侧边界评价 | `5` |

## 2. 数据集与实验划分

### 2.1 Dunhuang

[Dunhuang Grottoes Painting Dataset and Benchmark](https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge) 包含 600 幅壁画图像，其中 500 幅用于开发训练，100 幅用于独立测试。所有实验保留官方图像与损伤掩膜的逐样本配对关系。

| 划分 | 规模 | 设置 |
|---|---:|---|
| 配置阶段训练 | 450 | 从 500 幅开发图像中划出验证集 |
| 配置阶段验证 | 50 | 随机种子 `42` |
| 最终训练 | 500 | 配置固定后使用完整开发集 |
| 独立测试 | 100 | 官方图像—掩膜配对；不进行训练掩膜增强 |

训练时，当基础掩膜缺失比例小于 `0.15`，以 `0.5` 的概率附加 `1–4` 条宽度 `15–45` 像素的线段损伤。图像和掩膜同步执行翻转及旋转增强。附加损伤仅用于训练，不用于独立测试。

### 2.2 MuralDH

[MuralDH](https://github.com/tearsheaven/MuralDH) 的受控外部评价采用 [LABFNet](https://doi.org/10.3390/jimaging12070332) 相关研究 `760/201` 划分中的 201 个真实损伤掩膜，并从 MuralDH 参考图像集合中以随机种子 `42` 无放回抽取 201 幅图像。参考图像 ID 与掩膜 ID 分别排序后固定配对。

该协议以真实损伤掩膜构造受控遮挡，评价被隐藏参考像素的重建一致性；参考图像与实际受损图像不构成同一壁画的损伤前后配对。MuralDH 不参与模型训练、微调或超参数选择。原始受损图像另采用无参考图像质量指标评价。

### 2.3 muralv2 损伤模式

[PGRDiff](https://github.com/CZY-Code/PGRDiff) 提供的 muralv2 资源用于四类损伤模式分析。分析子集由该资源中的损伤掩膜重新提取与组织，标签如下：

| 分析类别 | 损伤形态 |
|---|---|
| `Crack` | 裂纹 |
| `Fallen-off / irregular loss` | 剥落与不规则缺损 |
| `Decorative-pattern damage` | 装饰纹样缺损 |
| `Interrupted contour` | 轮廓中断 |

上述四类标签为研究中的二次组织类别。

## 3. 训练与比较协议

### 3.1 训练设置

| 参数 | 设置 |
|---|---|
| 输入分辨率 | `256 × 256` |
| Batch size / DataLoader workers | `8 / 8` |
| 主网络 | ConditionalUNet；6 输入通道、3 输出通道、基础宽度 64、深度 4 |
| 结构网络 | ConditionalUNet；2 输入通道、基础宽度 64、深度 4 |
| 优化器 | Adam，`β1=0.9`，`β2=0.999` |
| 初始学习率 | `1e-5` |
| 学习率计划 | Iteration `10000`、`16000` 分别乘 `0.5` |
| 训练预算 | `20000` iterations |
| Diffusion steps | `100` |
| Noise schedule | Cosine |
| Maximum noise intensity | `30` |
| SDE 参数 | `epsilon_SDE=0.005` |
| 最终模型种子 | `1234`、`2024`、`3407` |
| 其他单因素实验种子 | `1234` |
| 计算设备 | NVIDIA RTX 4090（单卡） |

## 4. 评价指标

### 4.1 G-PSNR 与 H-PSNR

将 RGB 图像归一化至 `[0,1]`。G-PSNR 基于完整图像计算，H-PSNR 基于缺失区域计算：

```math
\mathrm{MSE}_{H}=\frac{\sum_p H(p)\sum_{c=1}^{3}(x_c(p)-\hat{x}_c(p))^2}{3\sum_p H(p)}
```

```math
\mathrm{H\!-\!PSNR}=10\log_{10}\frac{1}{\mathrm{MSE}_{H}}
```

### 4.2 G-SSIM 与 H-SSIM

每个 RGB 通道分别使用 `skimage.metrics.structural_similarity` 计算 SSIM：

```python
structural_similarity(
    reference[..., c],
    prediction[..., c],
    data_range=1.0,
    win_size=7,
    gaussian_weights=False,
    use_sample_covariance=True,
    full=True,
)
```

G-SSIM 为三个通道全图 SSIM 标量的平均；H-SSIM 使用完整局部 SSIM 图，并在缺失区域中心位置取均值：

```math
\bar{s}(p)=\frac{1}{3}\sum_{c=1}^{3}s_c(p)
```

```math
\mathrm{H\!-\!SSIM}=\frac{\sum_p H(p)\bar{s}(p)}{\sum_p H(p)}
```

局部窗口的统计量按完整邻域计算；H-SSIM 仅通过 `H` 选择计入平均的窗口中心。G-SSIM 的标量结果不计入图像外侧三个像素的边界，H-SSIM 使用完整局部图。

### 4.3 LPIPS

使用 AlexNet 特征的 `lpips.LPIPS(net="alex")`，在完整 RGB 图像上计算。输入从 `[0,1]` 映射到 `[-1,1]`。

### 4.4 KB-F1

已知侧评价边界：

```math
B_{\mathrm{eval}}=\mathrm{Dilate}_{k_{\mathrm{eval}}}(H)-H,\qquad k_{\mathrm{eval}}=5
```

| 设置 | 取值 |
|---|---|
| Canny 阈值 | `100 / 200` |
| 灰度输入 | 完整图像，8-bit |
| 评价边界核 | `5 × 5` 方形核 |
| 距离变换 | OpenCV `DIST_L2`，`maskSize=5` |
| 双向边缘匹配容差 | `1` 像素 |
| 两侧边缘均为空 | KB-F1 = `1` |
| 仅一侧为空 | KB-F1 = `0` |

双向边缘匹配计算 precision `P` 与 recall `R`，KB-F1 定义为：

```math
\mathrm{KB\!-\!F1}=\frac{2PR}{P+R}
```

当 `P+R=0` 时，KB-F1 定义为 `0`。

## 5. 参考式评价流程

### 5.1 评价依赖

```bash
python -m pip install numpy pandas opencv-python scikit-image torch lpips
```

### 5.2 文件组织

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

预测图、参考图及掩膜通过标准化文件名配对。RGB 图像使用 OpenCV `INTER_AREA` 缩放至 `256 × 256`，掩膜使用 `INTER_NEAREST`。默认白色表示已知区域，即 `--mask-convention known_white`。

### 5.3 Dunhuang 独立测试

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

### 5.4 MuralDH 受控外部评价

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

### 5.5 评价输出

| 文件 | 内容 |
|---|---|
| `per_image_metrics.csv` | 图像 ID、缺失比例与逐图指标 |
| `summary.csv` | 有效样本数、缺失比例与各指标均值 |
| `stratified_gpsnr.csv` | `[0,0.2)`、`[0.2,0.4)`、`[0.4,+∞)` 区间 G-PSNR |

评价脚本直接读取保存的预测结果，不对已知像素进行参考图像替换或额外颜色校正。指标先逐图计算，再对有限有效数值等权汇总。

## 6. 数据与代码可用性

| 资源 | 地址 |
|---|---|
| 项目代码 | [Visible-evidence-guided](https://github.com/xinyu0833-bit/Visible-evidence-guided) |
| Dunhuang | [Official benchmark](https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge) |
| MuralDH | [Dataset repository](https://github.com/tearsheaven/MuralDH) |
| MuralDH 相关划分 | [LABFNet](https://doi.org/10.3390/jimaging12070332) |
| muralv2 | [PGRDiff project](https://github.com/CZY-Code/PGRDiff) |

研究使用的固定记录标识符、确定性提示、派生掩膜子集、YAML 配置、输出清单及指标汇总可向通讯作者合理请求。原始图像、掩膜与第三方权重的使用和再分发遵循相应数据与模型提供方的许可条件。
