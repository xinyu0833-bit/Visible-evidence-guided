# Visible-evidence-guided virtual completion of Dunhuang murals using mask-aware multimodal diffusion

**基于可见证据引导与掩膜感知多模态扩散的敦煌壁画虚拟补全**

本仓库对应论文 *Visible-evidence-guided virtual completion of Dunhuang murals using mask-aware multimodal diffusion* 的修订稿。项目研究的是**数字虚拟补全（digital virtual completion）**：利用损伤后仍可观察的壁画证据，为缺失区域生成候选补全结果，而非恢复无法验证的历史原貌，也不替代文物保护人员的材料与图像学判断。

> **说明**：本 README 以当前修订稿的 Methods、Results、Tables 1–11 和 Code/Data Availability 为依据。论文中的实验结果与公开仓库里实际已上传的文件是两个不同问题；运行示例须结合实际可用的代码、权重、记录清单和数据路径使用。不要将 MuralDH 的受控参考式实验表述为同一壁画的真实损伤前后恢复。

## 1. 方法概览

本方法以官方 **StrDiffusion** 为冻结的结构引导扩散骨干，在不更新原有纹理网络、结构网络或 CLIP 文本编码器的前提下，学习一个 **Multi-scale Prompt Adapter**，并通过**参数无关的 mask-aware routing** 控制文本残差的直接注入区域。

核心流程：

1. 将参考图像与其一一对应的二值掩膜形成 masked mural image；推理阶段不读取缺失区域目标像素。
2. 使用本地 **Qwen2-VL-7B-Instruct** 对 masked image 与 binary mask 离线生成确定性的、样本级可见证据提示。
3. 通过冻结的 **CLIP ViT-L/14** 获取 token-level text features。
4. 在 U-Net 的 **middle block + 3 个 decoder stages** 注入多尺度 Prompt Adapter 残差；Adapter 结合 **FiLM** 与 **multi-head cross-attention**。
5. 利用二值掩膜构建 hole map 与已知侧边界带，仅在这些区域**直接**注入提示残差；后续卷积或注意力仍可能传播影响。
6. 仅优化 Prompt Adapter，采用缺失区域与已知侧边界的加权监督。

### 1.1 冻结部分与可训练部分

| 组成 | 配置与作用 | 是否训练 |
| --- | --- | --- |
| StrDiffusion texture / structure networks | 官方结构引导骨干，继承 IR-SDE mean-reverting stochastic process | 冻结 |
| SPADE-based structural pathway | 使用 masked input 导出的灰度与边缘条件进行空间自适应调制 | 冻结 |
| Qwen2-VL-7B-Instruct | 离线生成三字段 visible-evidence prompt | 不参与训练 |
| CLIP ViT-L/14 | 文本最大长度 77 tokens，特征维度 768 | 冻结 |
| Multi-scale Prompt Adapter | FiLM + token-level cross-attention，位于 middle block 与 3 个 decoder stages | **唯一可训练模块** |
| Mask-aware routing | 由二值掩膜确定 prompt residual 的直接注入区域 | 无可训练参数 |

主 ConditionalUNet 使用 **6 通道输入 / 3 通道输出**，base width = 64、depth = 4；结构网络使用 **2 通道输入**，base width = 64、depth = 4。纹理网络初始化自本研究使用的 StrDiffusion 5,000-iteration checkpoint，结构网络使用相应的预训练 checkpoint。

### 1.2 掩膜、路由与监督

二值掩膜 `M` 的约定为 **1 = 已知像素，0 = 缺失像素**，故：

```math
H = 1-M,\qquad \mu=M\odot x_0.
```

结构条件从 masked input 构造，而非从被遮挡的 ground truth 构造：

```math
G_\mu=M\odot\Phi_{\mathrm{gray}}(\mu),\qquad
E_\mu=M\odot\Phi_{\mathrm{edge}}(G_\mu),\qquad
S=\operatorname{Concat}(G_\mu,E_\mu).
```

分别定义三个用途不同的边界核，**即使最终取值相同也不可混用**：

| 参数 | 用途 | 修订稿最终设置 |
| --- | --- | --- |
| `k_r` | prompt residual 的 routing boundary | 5 |
| `k_sup` | boundary-weighted training loss | 5 |
| `k_eval` | KB-F1 已知侧评价边界 | 5 |

已知侧 routing boundary 与残差注入形式为：

```math
B_{\mathrm{route}}^{(k_r)}=\operatorname{Dilate}_{k_r}(H)-H,
```

```math
R_s^{(k_r)}(M)=
\operatorname{clip}\!\left(
\operatorname{Resize}_s(H)+
\operatorname{Resize}_s(B_{\mathrm{route}}^{(k_r)}),0,1
\right),
\qquad
h'_s=h_s+R_s^{(k_r)}(M)\odot r_p^{(s)}.
```

区域图缩放到不同特征尺度时使用 nearest-neighbour interpolation。`k_r=1` 时不额外引入已知侧边界带，对应 hole-only routing。

监督边界定义为 `B_sup = Dilate_5(H) - H`，Adapter 的优化目标为：

```math
\mathcal{L}=\mathcal{L}_{\mathrm{hole}}+2\mathcal{L}_{\mathrm{boundary}},
```

```math
\mathcal{L}_{\mathrm{hole}}=
\frac{\left\|H\odot(\hat{x}_{t-1}-x^{*}_{t-1})\right\|_1}
{\|H\|_1+\epsilon},
\qquad
\mathcal{L}_{\mathrm{boundary}}=
\frac{\left\|B_{\mathrm{sup}}\odot(\hat{x}_{t-1}-x^{*}_{t-1})\right\|_1}
{\|B_{\mathrm{sup}}\|_1+\epsilon}.
```

这里的 boundary-weighted loss 是本研究对 Prompt Adapter 使用的目标函数，不是冻结 StrDiffusion 骨干自带的训练损失。

## 2. 数据集与划分

### 2.1 Dunhuang（训练、验证与域内测试）

[Dunhuang Grottoes Painting Dataset and Benchmark](https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge) 包含 **500 幅开发图像**与 **100 幅独立测试图像**。

- 始终保留官方 **image–mask 一一配对关系**；测试图像不重新随机分配掩膜。
- 配置选择时，用随机种子 **42** 从开发集选出 **50 幅验证图像**，形成 **450/50** 配置阶段划分。
- 固定网络、提示模板、损失与 routing 参数后，最终使用 **全部 500 幅开发图像**训练。
- 官方 **100 幅测试图像及其对应掩膜**仅用于域内测试，不使用训练期随机掩膜增强。

### 2.2 MuralDH（受控外部评价与真实受损图像评价）

数据来源：[MuralDH](https://github.com/tearsheaven/MuralDH)；相关 **760/201** 划分参考 [LABFNet](https://doi.org/10.3390/jimaging12070332)。

本研究对 MuralDH 的使用有两种**互不等同**的设置：

**A. Reference-based controlled external evaluation（Table 1–3）**

1. 从上述相关工作划分的评价子集中提取 **201 个真实退化标注掩膜**；
2. 用种子 **42** 从 MuralDH 参考图像集合**无放回**地独立抽取 **201 幅参考图像**；
3. 将参考图像 ID 与掩膜 ID **各自排序**，按顺序一一配对，形成推理前固定的 201 个记录；
4. 掩膜应用于对应的独立参考图像后，评价被遮挡参考像素的重建一致性。

该设置使用**真实损伤几何**，但参考图像与损伤掩膜**不是同一物理壁画的“受损前／受损后”观测对**。因此，PSNR、SSIM 等指标不能证明补全结果符合壁画的真实历史原貌。

**B. Originally damaged images（Table 5）**

对原本就真实受损的 MuralDH 图像及其模型输出，单独计算 **ARNIQA、BRISQUE、DBCNN** 无参考质量指标。由于不同方法的有效输出记录覆盖范围不一致，Table 5 只能用于描述各自输出集合，**不可当作严格配对的性能排行榜**。

**外部评价约束**：本文方法**不在 MuralDH 上训练、微调或选择超参数**。

### 2.3 muralv2（损伤类型分析；Table 4）

数据入口：[PGRDiff / muralv2](https://github.com/CZY-Code/PGRDiff)。

论文依据该资源重新提取、组织掩膜，使用以下**本研究二次定义**的四种分析分组：

| 论文中的分析分组 | 含义 |
| --- | --- |
| `Crack` | 裂纹 |
| `Fallen-off / irregular loss` | 脱落／不规则材料缺失 |
| `Decorative-pattern damage` | 装饰纹样受损 |
| `Interrupted contour` | 轮廓中断 |

**注意**：这些是本文二次整理的分析标签，**不能直接宣称为 PGRDiff 数据集原始类别名称**。Table 4 数值由本研究在这些分组上重新评价得到，不属于 MuralDH 外部参考式评价。

### 2.4 跨数据集重复筛查

修订稿对 Dunhuang 的 500 幅开发图像、100 幅测试图像和 201 幅 MuralDH 参考图像，使用 file SHA-256、pixel-content SHA-256、pHash 与 dHash 检查精确及近似重复。所检查的 **100,500 组开发集–MuralDH 配对**和 **20,100 组测试集–MuralDH 配对**中，未发现文件级、像素级精确重复或 perceptual-hash 近重复候选。这不意味着两个资源在主题、风格或视觉分布上相互独立。

## 3. 训练掩膜与数据增强

- 每幅 Dunhuang 图像的官方配对掩膜为 base mask；不存在训练图像与外部随机掩膜池重新配对的步骤。
- 将掩膜转换为灰度，以 nearest-neighbour 缩放到 `256×256`，归一化到 `[0,1]` 后按 `0.5` 二值化，已知区域记为 1。
- **仅在训练中**：若初始缺失比例 `< 0.15`，以概率 `0.5` 增加 **1–4 条**端点随机的零值线段，线宽在 **15–45 像素**均匀采样。保留 base mask 已有的全部缺失区域。
- 水平翻转、垂直翻转与已配置的图像旋转对图像和掩膜**同步**实施，然后生成 masked input、结构条件与 prompt。
- 额外画线只是几何增强，**不是**对裂纹、剥落或磨蚀等物理劣化机制的真实模拟。
- **Dunhuang 测试集不做随机掩膜增强；MuralDH 外部评价直接使用选定的真实劣化标注掩膜。**

## 4. 确定性可见证据提示（Visible-evidence prompts）

Qwen2-VL-7B-Instruct 在本地**离线**处理 masked mural image 和 binary mask，使用确定性的 **greedy decoding**（关闭 sampling）；生成的提示与 image–mask record 一一绑定并在训练和评价中复用。

**最终稿严格使用以下三个英文条目**，不要再使用旧 README 中的 `Content and style / Degradation / Restoration constraint`：

1. `Visible content`：仅描述遮挡后仍可见的主体、构图和装饰信息。
2. `Painted style and palette`：仅描述可见色彩、线条与绘画风格。
3. `Mask morphology`：仅对整体掩膜形态作**粗略语言描述**；精确缺失几何、已知侧边界和残差路由始终由二值掩膜确定。其**独立增益尚未得到证实**，不将该字段单列为算法创新。

### 4.1 论文中的固定模板

System instruction：

```text
Describe only visible evidence in the supplied masked Dunhuang mural image and binary mask. Return one concise English description using the fields “Visible content”, “Painted style and palette”, and “Mask morphology”. Do not infer or name content hidden by the mask.
```

User template：

```text
Masked mural image: <IMAGE>. Binary mask: <MASK>. Produce the description in the required three-field format.
```

这里 `<IMAGE>` 和 `<MASK>` 是模板占位符；不得以缺失区域的真实参考像素生成提示。CLIP ViT-L/14 的 token-level 特征记为 `C_p`（最大 77 tokens、768 维），用于 Adapter 中的多尺度 FiLM 与 cross-attention。

### 4.2 提示控制实验（Tables 8–9）

| 条件 | Fixed-weight inference（Table 8） | Parameter-matched retraining（Table 9） |
| --- | --- | --- |
| `Matched` | 输入与自身 sample-specific prompt 匹配 | `Train-Matched` |
| `Partial omission` | 删除一个可见属性，其余保持不变 | `Train-Partial omission` |
| `Localized factual error` | 将同一可见属性改成错误值 | `Train-Localized factual error` |
| `Fixed generic` | 全体样本使用同一通用提示 | `Train-Fixed generic` |
| `Shuffled` | 按预先确定的规则错配样本与提示 | `Train-Shuffled` |
| `Zero-feature` | 直接令 `C_p = 0`，保留 Adapter 结构 | `Train-Zero-feature` |

- **Table 8**：固定 seed 1234 的最终模型权重，**仅改变推理提示**。
- **Table 9**：保留相同的 Adapter 结构与参数量，在各提示条件下分别从头训练；这些条件也使用 **seed 1234**。
- 这些单种子实验用于分析提示信息与模型容量的关系，**不应解释为多种子统计显著性检验**。

### 4.3 提示可靠性人工审核

三名独立评价者检查全部 **100 条 Dunhuang 测试提示**：`Acceptable = 84`、`Minor issue = 14`、`Unacceptable = 2`；模板合规率为 **100%**，有 **3%** 的提示包含无证据的遮挡内容推断，有 **1%** 存在重要遗漏。该审核结果不能表述为“98% 的提示完全准确”；论文未报告审核者间一致性系数。

## 5. 训练与推理配置（Table 11）

| 项目 | 设置 |
| --- | --- |
| 配置阶段划分 | Dunhuang 450 训练 / 50 验证；抽样种子 42 |
| 最终训练集 | 全部 500 幅 Dunhuang 开发图像 |
| 主模型种子 | `1234`、`2024`、`3407` |
| 核心组件消融种子 | B–E 使用同样三种子 |
| 其他单因素控制 | 默认 `1234` |
| 输入分辨率 | `256×256` |
| Batch size / workers | `8 / 8` |
| 优化器 | Adam，`β1=0.9`，`β2=0.999` |
| 初始学习率 | `1e-5`，无 warm-up |
| 学习率计划 | iteration 10,000 和 16,000 时分别乘以 `0.5` |
| 训练预算 | `20,000` iterations（约 63 batches/epoch、318 epochs） |
| Diffusion steps / schedule | `100` / `cosine` |
| Maximum noise intensity | `30` |
| `epsilon_SDE` | `0.005` |
| Routing / supervision / evaluation kernel | `k_r=5` / `k_sup=5` / `k_eval=5` |
| 硬件 | 单张 NVIDIA RTX 4090；非分布式 |

最终 `k_r = 5` 在**验证集**上确定；测试集的 `k_r` 对比只做敏感性分析，不用于事后选择最优参数。

## 6. 对比协议与主要结果

### 6.1 Baseline 公平性

| 方法 | 实验使用方式 |
| --- | --- |
| `LaMa`、`StrDiffusion`、`PGRDiff` | 使用同一 Dunhuang 开发图像、官方对应训练掩膜、`256×256` 分辨率、batch size、Adam 优化器、学习率计划和 `20,000` iterations；保留各自原生网络与损失 |
| `PGRDiff` 的文本条件 | 训练与测试均使用与 Ours 相同的 sample-specific prompts |
| `EdgeConnect`、`PowerPaint`、`RAD`、`LRDiff`、`MuralNet` | 使用作者公开的权重 |
| `PowerPaint` | 因已发布模型的提示长度限制，采用其官方 prompt-generation / inference procedure，不强行输入本研究完整三字段提示 |
| `LaMa`、`StrDiffusion` | 不人为增设文本分支 |

**务必区分**：Tables 1–3 的 `StrDiffusion (matched training)` 与 Table 6 配置 A 的 **official frozen StrDiffusion** 不是同一对照设置，不要混用其数值。GuidePaint、DiffuMural、MAPGR 在论文中作为相关工作讨论，**没有**同记录、同协议的直接数值结果。

### 6.2 主要重建指标（Tables 1–2）

论文对 Ours 的最终模型采用 **3 个独立随机种子的均值 ± 标准差（SD）**：

| 数据集 | G-PSNR ↑ (dB) | G-SSIM ↑ | LPIPS ↓ | H-PSNR ↑ (dB) | H-SSIM ↑ | KB-F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dunhuang | 36.6212 ± 0.0391 | 0.9640 ± 0.0003 | 0.0433 ± 0.0004 | 27.0445 ± 0.0670 | 0.8284 ± 0.0017 | 0.9253 ± 0.0017 |
| MuralDH controlled | 35.8066 ± 0.0595 | 0.9638 ± 0.0003 | 0.0375 ± 0.0005 | 18.6842 ± 0.0810 | 0.5538 ± 0.0023 | 0.8991 ± 0.0019 |

**解释边界**：Ours 的多项像素与结构指标更高，但在 Dunhuang 上 LaMa 的 **LPIPS 更低（0.0163 vs 0.0433）**；不能宣称全部指标均最优。MuralDH controlled 中的参考图像并非同一壁画受损前的照片。

### 6.3 不同损伤类型（Table 4）

以下 H-PSNR 单位均为 dB；分组属于本研究基于 muralv2 的二次整理。

| 损伤类型 | 平均缺失面积比例 | Ours | LaMa (matched) | StrDiffusion (matched) |
| --- | ---: | ---: | ---: | ---: |
| Crack | 12.85% | 25.8434 | 24.7371 | 20.2969 |
| Fallen-off / irregular loss | 7.56% | 19.4688 | 19.3872 | 14.3920 |
| Decorative-pattern damage | 14.33% | 24.3221 | 24.9091 | 19.3199 |
| Interrupted contour | 5.04% | 24.6241 | 23.0646 | 19.9521 |

Ours 在 `Decorative-pattern damage` 上的 H-PSNR **低于** LaMa，在 `Fallen-off / irregular loss` 上与 LaMa 很接近；这些是论文明确报告的限制。

### 6.4 核心组件消融（Table 6）

| 配置 | Prompt Adapter | Boundary loss | Routing | G-PSNR ↑ (dB) |
| --- | --- | --- | --- | ---: |
| A. Official frozen StrDiffusion | 否 | 否 | 无 | 31.2662 |
| B. Adapter only | 是 | 否 | Global | 35.4658 ± 0.0455 |
| C. + Boundary loss | 是 | 是 | Global | 35.8755 ± 0.0400 |
| D. Hole-only (`k_r=1`) | 是 | 是 | Hole | 36.3543 ± 0.0360 |
| E. Hole + boundary (`k_r=5`) | 是 | 是 | Hole + boundary | 36.6212 ± 0.0391 |

B–E 为三种子结果；A 为官方冻结参考。**A→B 的改进同时包含目标域 Adapter 训练与文本条件，不能把全部增益归因于语言。**

单种子（seed 1234）多尺度注入对比（Table 7）：`Single-scale = 35.8701 dB`，`Multi-scale = 36.6177 dB`。

Routing kernel 敏感性（seed 1234，固定 `k_sup=k_eval=5`）：

| `k_r` | 1 | 3 | 5 | 7 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| G-PSNR (dB) | 36.3550 | 36.5420 | 36.6177 | 36.2786 | 36.1327 |

### 6.5 Sample-specific prompt 控制结果（Tables 8–9）

以下均为 **Dunhuang / seed 1234 的单种子 G-PSNR**，与上文三种子平均值不是同一统计单位。

| 提示条件 | Table 8：固定权重推理 | Table 9：相同参数量重新训练 |
| --- | ---: | ---: |
| Matched | 36.6177 | 36.6177 |
| Partial omission | 36.3238 | 36.3526 |
| Localized factual error | 36.1774 | 36.2413 |
| Fixed generic | 35.9601 | 35.9957 |
| Shuffled | 35.9446 | 35.9834 |
| Zero-feature | 35.3279 | 35.1973 |

结果支持：在所报告的控制条件下，sample-specific 的可见证据提示提供了额外的语义条件信息；但**不代表 `Mask morphology` 字段自身的独立贡献已被证实**。

### 6.6 原始受损 MuralDH 的无参考结果（Table 5）

| 数据／方法 | ARNIQA ↑ | BRISQUE ↓ | DBCNN ↑ |
| --- | ---: | ---: | ---: |
| 原始受损 MuralDH 输入 | 0.5719 | 19.8628 | 0.5359 |
| StrDiffusion (matched training) | 0.6458 | 41.0225 | 0.6271 |
| LaMa (matched training) | 0.6394 | 40.5322 | 0.6255 |
| PGRDiff | 0.6404 | 37.1317 | 0.6286 |
| Ours | 0.6684 | 40.2336 | 0.6554 |

各行的有效输出记录覆盖不同，数据**不适合直接做严格的成对优劣检验**；更高的无参考质量分数也不等于历史内容更准确。

### 6.7 领域专家盲评（Table 10）

三位具有壁画保护或相关艺术分析经验的评价者，对按缺失面积分层抽样的 **30 个 Dunhuang 测试案例**进行盲评。每个案例就 `Structural fidelity`、`Artistic style consistency`、`Completion plausibility` 以 1–5 分评分；先平均专家评分，再对三个维度等权平均得到 Overall。

| 方法 | Overall（跨案例 mean ± SD） | 95% CI |
| --- | ---: | ---: |
| LaMa (matched training) | 3.32 ± 0.54 | [3.12, 3.52] |
| StrDiffusion (matched training) | 3.79 ± 0.34 | [3.66, 3.92] |
| MuralNet | 2.08 ± 0.92 | [1.74, 2.42] |
| Ours | **4.14 ± 0.44** | **[3.98, 4.30]** |

盲评采用已知的**数字参考图像**，不构成真实历史复原真实性验证；修订稿未报告专家间一致性系数。

## 7. 参考式评价脚本

原 README 指定的评价脚本名称为：

```text
eval_paired_predictions_paper_documented.py
```

> **运行前检查**：该名称沿用上传的原 README；本次提供的是文档更新，不包含或验证该 Python 脚本本体，也不保证脚本位于仓库根目录。请先确认真实代码位置，并把下方 `EVAL_SCRIPT` 指向实际脚本文件。模型训练、prompt generation 与无参考 ARNIQA/BRISQUE/DBCNN 评价**不等同于**这里的参考式指标脚本。

### 7.1 参考式评价依赖

原 README 中给出的评价脚本依赖为：

```bash
python -m pip install numpy pandas opencv-python scikit-image torch lpips
```

这**不是**完整训练环境的已验证依赖清单；训练与 Qwen2-VL/CLIP 推理还需要根据实际项目环境安装相应包和权重。

### 7.2 输入文件组织

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

上述仅为**配对命名示意**；实际 ID 应来自固定的 image–mask 记录清单，不能用文件名排序临时改变论文评测配对。原 README 约定预测、参考和掩膜通过归一化文件名主干匹配。

- RGB 图像用 OpenCV `INTER_AREA` 缩放到 `256×256`，转换为 RGB 并除以 255。
- mask 用 `INTER_NEAREST`；对于 `0/255` 灰度图，以 `>127` 判为白；对于 `0/1` 掩膜，以 `>0` 判为白。
- 默认 `known_white`：白色 `M=1` 表示已知区域，黑色表示缺失区域；仅当确实需要时才使用 `hole_white` 反转约定。
- **直接**从已保存的 predictions 与 references 计算指标：**不**用 GT 覆盖已知区域，也**不**附加颜色修正。
- 逐图求指标，对所有有效有限值**按图等权平均**；不按缺失面积对图像加权。

### 7.3 Dunhuang：100 幅官方独立测试图像

```bash
EVAL_SCRIPT=/path/to/eval_paired_predictions_paper_documented.py

python "$EVAL_SCRIPT" \
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

### 7.4 MuralDH：201 个固定的受控参考式记录

```bash
EVAL_SCRIPT=/path/to/eval_paired_predictions_paper_documented.py

python "$EVAL_SCRIPT" \
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

这些命令的参数名沿用原 README。执行前应使用**实际脚本的 `--help`** 核对参数可用性。

### 7.5 预期输出

按原 README 描述，参考式脚本生成：

| 文件 | 说明 |
| --- | --- |
| `per_image_metrics.csv` | `id`、`hole_ratio` 和逐图指标 |
| `summary.csv` | 数据集名称、有效记录数、平均缺失比例、指标均值 |
| `stratified_gpsnr.csv` | `[0,0.2)`、`[0.2,0.4)`、`[0.4,+∞)` 分层 G-PSNR |

区间为**左闭右开**；`[0.4,+∞)` 仅用于较大缺失比例记录的检查。最终应核对记录数量、文件名配对与非有限指标过滤。

## 8. 指标实现要点（对应论文 Methods）

### 8.1 G-PSNR 与 H-PSNR

RGB 值域 `[0,1]`。G-PSNR 计算全图全部 RGB 通道的 MSE，H-PSNR 仅按缺失区域中心像素计算：

```math
\mathrm{MSE}_H=
\frac{\sum_p H(p)\sum_{c=1}^{3}(x_c(p)-\hat{x}_c(p))^2}
{3\sum_p H(p)},
\qquad
\mathrm{H\!-\!PSNR}=10\log_{10}\frac{1}{\mathrm{MSE}_H}.
```

MSE 为 0 对应正无穷大；论文汇总时仅平均有限结果。

### 8.2 G-SSIM 与 H-SSIM

三个 RGB 通道分别使用 `skimage.metrics.structural_similarity`：

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

`K1=0.01`、`K2=0.03` 使用库默认值。若 `s_c(p)` 是各通道返回的 full local SSIM map：

```math
\bar{s}(p)=\frac{1}{3}\sum_{c=1}^{3}s_c(p),
\qquad
\mathrm{H\!-\!SSIM}=
\frac{\sum_p H(p)\bar{s}(p)}{\sum_p H(p)}.
```

**重点**：H-SSIM 按**局部 SSIM 窗口中心是否位于缺失区域**选点；靠近缺失边缘的 `7×7` 窗口**可以包含已知像素**，并非仅用 hole pixels 计算窗口统计，也不是裁剪到缺失外接框计算。G-SSIM 是三个通道全图 SSIM scalar 的平均（函数标量排除外缘 3 像素）；H-SSIM 使用 full map，按 hole 选点时**不额外剔除该外缘**。

### 8.3 LPIPS

采用 AlexNet：

```python
lpips.LPIPS(net="alex")
```

对完整 RGB 图像计算，输入从 `[0,1]` 映射到 `[-1,1]`；**LPIPS 越小越好**。

### 8.4 KB-F1：已知侧边界边缘一致性

```math
B_{\mathrm{eval}}=\operatorname{Dilate}_{k_{\mathrm{eval}}}(H)-H,
\qquad k_{\mathrm{eval}}=5.
```

- 使用一次 `5×5` 全 1 方形膨胀核，仅保留**已知侧边界带**内的边缘；
- 整图先转为 8-bit grayscale，再运行 Canny，阈值为 `100/200`；
- OpenCV `DIST_L2` distance transform，mask size = `5`；
- 以 **1 像素**容差计算双向 precision 与 recall，再求 F1；
- 两组边缘均为空时 `KB-F1=1`，只有一组为空或 `P+R=0` 时 `KB-F1=0`。

**评价边界 `k_eval=5` 和模型注入边界 `k_r` 不是同一个超参数**。

### 8.5 独立的无参考质量评价

`ARNIQA`、`BRISQUE`、`DBCNN` 仅用于**原始真实受损 MuralDH 图像及其输出**的另行评估，不属于上述 `eval_paired_predictions_paper_documented.py` 参考式指标列表；原 README 未提供该无参考评价的可执行脚本，因此这里不虚构对应命令。

## 9. 可重复性、文件可用性与注意事项

### 9.1 与论文对应的资源

- **代码仓库**：<https://github.com/xinyu0833-bit/Visible-evidence-guided>
- **Dunhuang benchmark**：<https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge>
- **MuralDH**：<https://github.com/tearsheaven/MuralDH>
- **MuralDH 相关划分（LABFNet）**：<https://doi.org/10.3390/jimaging12070332>
- **PGRDiff / muralv2**：<https://github.com/CZY-Code/PGRDiff>

根据修订稿 **Data Availability**，本研究生成的固定记录 ID、确定性 prompts、二次整理的 mask subsets、YAML 配置、output manifests 与 metric summaries 可向通讯作者**合理请求**；不要在文件尚未核实公开的情况下写成“全部已随仓库提供”。原始图像、掩膜和第三方模型权重的使用与再分发以各自提供方的许可为准。

建议复现时为每个评测运行保存：`image_id`、`reference_id`、`mask_id`、`prompt_id`、seed、模型 checkpoint、配置、预测文件、评价脚本版本及指标汇总；固定 sample–mask–prompt 的对应关系。

### 9.2 论文声明的主要限制

- **历史真实性**：虚拟补全结果不能据此认定为原壁画真正丢失的内容。
- **跨域范围**：Dunhuang 与 MuralDH 均属相关壁画视觉域；本文不宣称能够泛化至不相关文物类型。
- **复杂损伤**：大面积缺失、语义不明确、复杂构图、低质量掩膜和细粒度纹理仍有困难。
- **指标取舍**：局部几何、像素误差、perceptual distance 与领域专家判断可能不完全一致。
- **统计边界**：Table 7–9 与 routing-kernel 单因素控制为单种子；没有报告专家或提示审核的评审者间一致性；无参考评价的有效输出覆盖不等。
- **比较范围**：部分近期壁画专用方法（如 GuidePaint、DiffuMural、MAPGR）未能进行同记录的直接数值对照。

## 10. 引用

使用本项目时，请引用：

**Yuhai Yu, Xinyu Wei, Jiana Meng, Zongying Liu.**  
*Visible-evidence-guided virtual completion of Dunhuang murals using mask-aware multimodal diffusion.*

以论文**正式发表后提供的出版信息和 DOI** 为准；在未确认正式 DOI 前不填写虚构的 DOI 或发表状态。同时请遵循 StrDiffusion、CLIP、Qwen2-VL、PGRDiff 以及相关数据集的引用与许可要求。
