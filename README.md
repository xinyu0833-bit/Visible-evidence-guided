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

$$
H = 1-M,\qquad \mu=M\odot x_0
$$

结构条件由掩膜后可见信息构造：

$$
G_{\mu}=M\odot\Phi_{\mathrm{gray}}(\mu)
$$

$$
E_{\mu}=M\odot\Phi_{\mathrm{edge}}(G_{\mu}),\qquad S=\mathrm{Concat}(G_{\mu},E_{\mu})
$$

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

$$
C_p=E_{\mathrm{text}}(P)\in\mathbb{R}^{B\times77\times768}
$$


**公开的 Prompt 清单（按当前公开仓库文件名）：**

| 文件 | 记录范围 | 已核实字段 | 用途 |
|---|---:|---|---|
| [`DUNHUANG_train_prompts.json`](DUNHUANG_train_prompts.json) | `001–500`，500 条 | `image_id`, `masked_image`, `mask`, `final_prompt` | Dunhuang 训练 Prompt 记录 |
| [`DUNHUANG_test_prompts.json`](DUNHUANG_test_prompts.json) | `501–600`，100 条 | `image_id`, `masked_image`, `mask`, `final_prompt` | Dunhuang 独立测试 Prompt 记录 |
| [`MuralDH_test_prompts.json`](MuralDH_test_prompts.json) | `MuralDH_0001–MuralDH_0201`，201 条 | `image_id`, `masked_image`, `final_prompt` | MuralDH 受控评价的 Prompt 记录 |

三个 JSON 文件均使用以记录 ID 为键的对象结构。`final_prompt` 预期包含 `Visible content`、`Painted style and palette`、`Mask morphology` 三个字段，但**不能仅根据固定模板就断言每条生成文本均符合模板**：公开的 Dunhuang 测试记录 `599` 目前仍缺少后两个独立字段标题。建议根据实际实验日志进行全量审计；若需修订 Prompt，请保存旧版本和新版本，并明确论文所用版本，避免事后修改影响实验可追溯性。

JSON 中的 `/root/autodl-tmp/...` 为提示生成环境的历史本地路径，不是下载地址。复现时须使用 `image_id` 和固定输入清单将路径映射到本机数据；`MuralDH_test_prompts.json` 本身不含 `mask` 字段，**不能替代 201 组参考图像—损伤掩膜的原始 ID 配对清单**。

### 1.4 多尺度残差路由

令 `k_r` 为路由边界所用的奇数大小方形膨胀核。已知侧边界定义为：

$$
B_{\mathrm{route}}^{(k_r)}=\mathrm{Dilate}_{k_r}(H)-H
$$

在第 `s` 个特征尺度上，缺失图及边界图通过最近邻插值缩放，构造残差门控：

$$
R_s^{(k_r)}(M)=\mathrm{clip}\bigl(\mathrm{Resize}_s(H)+\mathrm{Resize}_s(B_{\mathrm{route}}^{(k_r)}),0,1\bigr)
$$

Prompt Adapter 生成的尺度专属残差通过门控注入 U-Net：

$$
h'_s=h_s+R_s^{(k_r)}(M)\odot r_p^{(s)}
$$

`k_r = 1` 对应 hole-only routing，最终采用 `k_r = 5` 的 hole + known-side boundary routing。

### 1.5 边界加权优化目标

监督边界由独立的膨胀核构造：

$$
B_{\mathrm{sup}}=\mathrm{Dilate}_{k_{\mathrm{sup}}}(H)-H,\qquad k_{\mathrm{sup}}=5
$$

Prompt Adapter 使用缺失区域损失和边界区域损失联合优化：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{hole}}+2\mathcal{L}_{\mathrm{boundary}}
$$

$$
\mathcal{L}_{\mathrm{hole}}=\frac{\|H\odot(\hat{x}_{t-1}-x^*_{t-1})\|_1}{\|H\|_1+\epsilon}
$$

$$
\mathcal{L}_{\mathrm{boundary}}=\frac{\|B_{\mathrm{sup}}\odot(\hat{x}_{t-1}-x^*_{t-1})\|_1}{\|B_{\mathrm{sup}}\|_1+\epsilon}
$$

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

[MuralDH](https://github.com/tearsheaven/MuralDH) 用于外部受控参考式评价。参考图像候选池来自数据包内的 `Mural_SR/Mural_DataSet`（归档候选池共 **512** 张）；真实损伤掩膜来自 `Mural_seg/test/labels`（**201** 张）。使用 `random.Random(42).sample()` 在自然排序的512张候选参考图中无放回抽取201张，再将参考图和掩膜原始ID分别自然排序并顺序配对。所有评价方法应使用同一组固定记录。

该协议的目的，是在来源于真实损伤标注的掩膜几何条件下，利用**独立的数字参考图像**评价被遮挡内容的重建一致性；两者不构成同一物理壁画的损坏前后配对，也不能证明未知的历史原貌。MuralDH 不参与模型训练、微调或超参数选择。原始受损图像另行采用无参考指标评价。

已整理的201组配对清单包含原始文件ID、路径、缺损比例与SHA-256，可作为复现资料。**这些重新整理的ID应先与论文原始推理记录核对，确认一致后再将其作为论文已报告结果的输入清单。** 本研究的201个mask与Dunhuang 100张测试图像的80/20缺损比例分组属于两个不同数据集，不应混用。

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


### 3.2 基线初始化与模型权重

**LaMa、StrDiffusion 和 PGRDiff 均使用各自官方发布的预训练权重初始化，并在匹配的Dunhuang训练记录上进行微调（fine-tuning）；均非从零训练（from scratch）。** 三者共享训练图像–掩膜记录、256×256分辨率及匹配的优化预算，但起始检查点、模型结构和原始预训练来源分别由各方法决定。

| 方法 | 初始化方式 | 上游官方项目 |
|---|---|---|
| LaMa | 官方预训练权重 → Fine-tuning | [LaMa](https://github.com/advimman/lama) |
| StrDiffusion | 官方预训练权重 → Fine-tuning | [StrDiffusion](https://github.com/htyjers/StrDiffusion) |
| PGRDiff | 官方预训练权重 → Fine-tuning | [PGRDiff](https://github.com/CZY-Code/PGRDiff) |

**作者提供的模型权重归档：**

- 文件：`pth.zip`
- [百度网盘下载](https://pan.baidu.com/s/113nThaxCpMGW2wWnu2UJcw)
- 提取码：`mgi2`

`pth.zip`为作者提供的权重下载入口。**官方起始权重与微调后的最终权重应分开标识**；具体压缩包成员文件与模型、训练阶段、实验运行ID之间的映射，以实际权重清单及训练日志为准，不能仅根据压缩文件名推定。

下载完成后可以在Windows PowerShell中校验ZIP文件：

```powershell
Get-FileHash "D:\desktop\data\pth.zip" -Algorithm SHA256
```

### 3.3 Prompt、图像—掩膜和实验版本追溯

| 材料 | 公开入口或核验依据 | 复现注意事项 |
|---|---|---|
| Dunhuang 训练 Prompt | [`DUNHUANG_train_prompts.json`](DUNHUANG_train_prompts.json) | 记录 001–500；须核对是否与报告结果所用快照一致 |
| Dunhuang 测试 Prompt | [`DUNHUANG_test_prompts.json`](DUNHUANG_test_prompts.json) | 记录 501–600；应对三字段格式逐条审计 |
| MuralDH 评价 Prompt | [`MuralDH_test_prompts.json`](MuralDH_test_prompts.json) | 记录 `MuralDH_0001–MuralDH_0201`；须与固定配对清单建立映射 |
| 开源代码及配置入口 | [`selected_package/selected_package`](selected_package/selected_package) | 以实际发布脚本及路径为准；请锁定仓库 commit |
| 201 组 MuralDH 真实损伤掩膜配对 | 固定 image–mask manifest / pairing CSV（以实验原始记录为准） | 应给出原始 `reference_id`、`mask_id`、合成记录 ID 和 SHA-256 |
| 初始及最终模型权重 | 各基线官方项目，以及下述作者权重归档 | 区分官方初始权重、目标域微调权重及最终模型 |
| 逐图指标与实验配置 | 对应训练、推理和评价的实际运行输出 | 按运行 ID 对应论文每一表格，未核验时勿声称可独立复算 |

建议对论文中每一个结果设置唯一 `run_id`，并记录 `git_commit`、`initial_checkpoint`、`final_checkpoint`、`prompt_file`、`prompt_sha256`、`image_mask_manifest`、`config`、`seed`、`evaluation_script`、`per_image_metrics.csv`。目前能公开访问三份 Prompt JSON 与项目源码入口，但**是否有逐实验一一对应的运行证据**，仍需依照真实目录和实验日志核对，不能以“代码已公开”替代核验。

**两套 MuralDH 流程须严格区分：**（1）在完整参考图上施加独立来源的 201 个真实损伤掩膜，用于受控参考式定量评价；（2）对原始受损壁画输出做无参考质量描述。两者的输入 ID、掩膜和可支持的结论不同，不能混用。

## 4. 评价指标

### 4.1 G-PSNR 与 H-PSNR

将 RGB 图像归一化至 `[0,1]`。G-PSNR 基于完整图像计算，H-PSNR 基于缺失区域计算：

$$
\mathrm{MSE}_{H}=\frac{\sum_p H(p)\sum_{c=1}^{3}(x_c(p)-\hat{x}_c(p))^2}{3\sum_p H(p)}
$$

$$
\mathrm{H\!-\!PSNR}=10\log_{10}\frac{1}{\mathrm{MSE}_{H}}
$$

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

$$
\bar{s}(p)=\frac{1}{3}\sum_{c=1}^{3}s_c(p)
$$

$$
\mathrm{H\!-\!SSIM}=\frac{\sum_p H(p)\bar{s}(p)}{\sum_p H(p)}
$$

局部窗口的统计量按完整邻域计算；H-SSIM 仅通过 `H` 选择计入平均的窗口中心。G-SSIM 的标量结果不计入图像外侧三个像素的边界，H-SSIM 使用完整局部图。

### 4.3 LPIPS

使用 AlexNet 特征的 `lpips.LPIPS(net="alex")`，在完整 RGB 图像上计算。输入从 `[0,1]` 映射到 `[-1,1]`。

### 4.4 KB-F1

已知侧评价边界：

$$
B_{\mathrm{eval}}=\mathrm{Dilate}_{k_{\mathrm{eval}}}(H)-H,\qquad k_{\mathrm{eval}}=5
$$

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

$$
\mathrm{KB\!-\!F1}=\frac{2PR}{P+R}
$$

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

下面的命令假设已经进入实际评价脚本所在目录；请先在公开源码中确认脚本路径和依赖版本，不要将此处的示例命令视为经过本次独立运行验证。

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

| 资源 | 公开地址 |
|---|---|
| 项目仓库 | [Visible-evidence-guided](https://github.com/xinyu0833-bit/Visible-evidence-guided) |
| 项目源码目录 | [`selected_package/selected_package`](selected_package/selected_package) |
| Dunhuang 训练 Prompt | [`DUNHUANG_train_prompts.json`](DUNHUANG_train_prompts.json) |
| Dunhuang 测试 Prompt | [`DUNHUANG_test_prompts.json`](DUNHUANG_test_prompts.json) |
| MuralDH 受控评价 Prompt | [`MuralDH_test_prompts.json`](MuralDH_test_prompts.json) |
| 作者发布的 `pth.zip` | [百度网盘](https://pan.baidu.com/s/113nThaxCpMGW2wWnu2UJcw)，提取码 `mgi2` |
| LaMa 官方仓库 | [advimman/lama](https://github.com/advimman/lama) |
| StrDiffusion 官方仓库 | [htyjers/StrDiffusion](https://github.com/htyjers/StrDiffusion) |
| PGRDiff 官方仓库 | [CZY-Code/PGRDiff](https://github.com/CZY-Code/PGRDiff) |
| Dunhuang 官方数据 | [Benchmark](https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge) |
| MuralDH 数据 | [Dataset repository](https://github.com/tearsheaven/MuralDH) |
| muralv2 | [PGRDiff project](https://github.com/CZY-Code/PGRDiff) |

仓库目前列出了训练/测试 Prompt JSON 与代码目录。代码公开与论文全部数值的独立复算是两项不同的要求：如需完全复现 Table 2–12，还应提供**对应原始运行的版本、精确模型权重映射、201 组配对清单、逐图评价结果和专家匿名评分/统计脚本**，并确保它们与论文中报告的实验一致。第三方图像、掩膜和模型权重应遵循原始数据许可。

## 7. 结果汇总边界与修订核对

- **Dunhuang 分组：** 100 张独立测试图像的缺损面积区间 `[0, 0.2)` 和 `[0.2, 0.4)` 分别包含 80、20 张，`≥0.4` 为 0 张；每幅图等权平均。由文中分组均值得到 PowerPaint 的 G-PSNR 为 `27.77554 dB`，RAD 为 `28.52526 dB`；这证明展示值的算术一致性，**不替代逐图指标核验**。
- **专家一致性：** 30 个案例、三名独立专家、三维度 Likert 评分；论文采用基于案例级 Overall 的双向绝对一致性 `ICC(A,1)` 和 `ICC(A,3)`。最终统计需要能够从匿名原始评分重算，并给出 95% 置信区间的方法与代码。
- **损伤类别边界：** 四个 `muralv2` 标签为本文二次整理的操作性掩膜分组，不构成经文保专家验证的完整病害分类。未系统测试盐析、褪色/变色、过往重绘、生物损害、水渍，以及复杂分支裂纹等类型；不应将结果外推到全部真实病害。
- **MuralDH 解释边界：** 参考图与实际损伤标注并非同一物理壁画的损伤前后配对。受控指标只表征施加外部损伤几何后的参考图像重建一致性，不证明历史原貌或现场保护效果。
- **版本锁定：** 在论文最终提交时记录仓库 commit、三个 Prompt 文件的 SHA-256，以及每张结果表关联的输入清单和模型权重。若 Prompt 发生更改，应说明数值来自哪个版本，而不是默认更新后 JSON 与此前实验自动一致。

