# Visible-evidence-guided virtual completion of Dunhuang murals using mask-aware multimodal diffusion

基于可见证据引导与掩膜感知多模态扩散的敦煌壁画虚拟补全

本项目使用冻结的 StrDiffusion 结构引导扩散骨干、Qwen2-VL-7B-Instruct 可见证据提示、冻结 CLIP ViT-L/14、多尺度 Prompt Adapter 与掩膜感知残差路由开展壁画缺失区域的数字虚拟补全。评价包括参考式重建、掩膜形态分组、消融、提示鲁棒性和专家盲评。

## 1. 方法框架

### 1.1 网络组成

| 组件 | 实现 | 本文训练时参数状态 |
|---|---|---|
| 纹理与结构恢复骨干 | StrDiffusion；IR-SDE 与 SPADE 结构调制 | 冻结 |
| 可见证据提示生成器 | Qwen2-VL-7B-Instruct | 冻结；离线生成 |
| 文本编码器 | CLIP ViT-L/14；77 tokens，768维 | 冻结 |
| Multi-scale Prompt Adapter | FiLM + multi-head cross-attention；`mid`、`dec1`、`dec2`、`dec3`注入 | 训练 |
| Mask-aware Routing | 二值掩膜空间门控 | 无可训练参数 |

主纹理网络使用六通道输入、三通道逆状态预测；结构网络接收双通道结构条件；优化时只更新Prompt Adapter。

### 1.2 掩膜与结构条件

`M=1`表示已知区域，`M=0`表示缺失区域；`H=1-M`。输入和结构条件为：

\[
\mu=M\odot x_0,\qquad
G_\mu=M\odot\Phi_{\mathrm{gray}}(\mu),\qquad
E_\mu=M\odot\Phi_{\mathrm{edge}}(G_\mu).
\]

### 1.3 可见证据提示

Qwen2-VL-7B-Instruct在禁用采样的greedy decoding条件下，从掩膜后图像及mask生成英文描述，不使用被遮挡的目标像素。论文所述生成指令使用`Visible content`、`Painted style and palette`、`Mask morphology`三字段。冻结CLIP编码提示：

\[
C_p=E_{\mathrm{text}}(P)\in\mathbb{R}^{B\times77\times768}.
\]

仓库已发布[`train_correct.json`](train_correct.json)，含逐样本`final_prompt`与`hole_ratio`。**该文件中示例文本的字段标题为`Content and style`、`Degradation`、`Restoration constraint`，与上述生成指令的字段标题不同。** 需要通过与实际训练运行对应的prompt文件版本及生成/后处理脚本明确两者关系；不要将两者在没有记录的情况下视为同一个模板版本。

### 1.4 多尺度残差路由

\[
B_{\mathrm{route}}^{(k_r)}=\mathrm{Dilate}_{k_r}(H)-H,\quad
R_s^{(k_r)}(M)=\mathrm{clip}\bigl(\mathrm{Resize}_s(H)+\mathrm{Resize}_s(B_{\mathrm{route}}^{(k_r)}),0,1\bigr).
\]

\[
h'_s=h_s+R_s^{(k_r)}(M)\odot r_p^{(s)}.
\]

最终`k_r=5`（缺失区+已知侧边界）；`k_r=1`用于hole-only routing对照。

### 1.5 边界加权优化

\[
B_{\mathrm{sup}}=\mathrm{Dilate}_{k_{\mathrm{sup}}}(H)-H,\quad k_{\mathrm{sup}}=5.
\]

\[
\mathcal{L}=\mathcal{L}_{\mathrm{hole}}+2\mathcal{L}_{\mathrm{boundary}},
\quad
\mathcal{L}_{\mathrm{hole}}=\frac{\|H\odot(\hat x_{t-1}-x^*_{t-1})\|_1}{\|H\|_1+\epsilon}.
\]

监督边界为真实缺损mask外的已知侧边界，`k_sup=5`。评价边界核`k_eval=5`。

## 2. 数据集与划分

### 2.1 Dunhuang

[Official Dunhuang Grottoes Painting Dataset and Benchmark](https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge)共有600幅图，500幅开发训练、100幅独立测试；配置选择阶段从500幅中用seed 42分出450训练/50验证，之后最终训练使用完整500。测试保持官方图像–掩膜一对一关系，不进行训练掩膜增强。

训练时若原mask缺损比例小于0.15，以0.5概率附加1–4条宽15–45像素的线段缺损；图像与mask同步翻转和旋转。增强只用于训练。

### 2.2 MuralDH controlled evaluation

原始数据：[MuralDH](https://github.com/tearsheaven/MuralDH)；相关工作：[LABFNet](https://doi.org/10.3390/jimaging12070332)。

本研究受控评价使用201个真实损伤分割mask，以及从`Mural_SR/Mural_DataSet`的512张候选壁画中按seed 42无放回抽取的201张参考图；两组原始ID分别自然排序，依序一一配对。mask来自`Mural_seg/test/labels`，所有方法使用同一组固定配对。受控评价不代表同一物理壁画的损坏前/损坏后历史真值配对。

配对记录：[pairs_manifest.csv](reproducibility/muraldh/pairs_manifest.csv)，
[split_protocol.json](reproducibility/muraldh/split_protocol.json)，
[候选池清单](reproducibility/muraldh/reference_pool_manifest.csv)。

**这些2026-10-25归档的配对记录应与实际生成论文Table 2–3结果时的历史输入日志核对后，才作为原实验的严格复现证据。**

### 2.3 muralv2

PGRDiff提供的muralv2资源用于作者重新组织的四种mask形态分析：`Crack`、`Fallen-off / irregular loss`、`Decorative-pattern damage`、`Interrupted contour`。这些是作者组织的掩膜形态子集，并非外部专家验证的全部病害分类。

## 3. 训练及比较协议

### 3.1 主要超参数

| 参数 | 取值 |
|---|---|
| 输入分辨率 | 256×256 |
| Batch size / workers | 8 / 8 |
| 网络 | ConditionalUNet：主网络6→3；结构网络2通道；base width 64，depth 4 |
| 优化器 | Adam，β1=0.9，β2=0.999 |
| 学习率 | 初始1e-5，10,000与16,000步各乘0.5 |
| 训练预算 | 20,000 iterations |
| Diffusion steps | 100 |
| Noise schedule / max intensity | cosine / 30 |
| ε_SDE | 0.005 |
| 主模型随机种子 | 1234、2024、3407 |
| 其他单因素实验种子 | 1234 |
| GPU | 单张NVIDIA RTX 4090 |

### 3.2 LaMa、StrDiffusion、PGRDiff的权重与初始化

**三个匹配训练基线均由各自官方发布的预训练权重初始化并在统一Dunhuang训练记录上微调，没有从头训练。**

| Baseline | Initialization | Official code |
|---|---|---|
| LaMa | Official pretrained → fine-tuning | [advimman/lama](https://github.com/advimman/lama) |
| StrDiffusion | Official pretrained → fine-tuning | [htyjers/StrDiffusion](https://github.com/htyjers/StrDiffusion) |
| PGRDiff | Official pretrained → fine-tuning | [CZY-Code/PGRDiff](https://github.com/CZY-Code/PGRDiff) |

训练图像、掩膜、256×256分辨率、优化预算匹配；各方法的原预训练数据、初始权重和网络结构并不相同。Ours中使用的冻结StrDiffusion骨干和单独微调的StrDiffusion基线是两个角色，需要在运行清单中区分。

**权重下载：**
- 作者提供：[`pth.zip`百度网盘链接](https://pan.baidu.com/s/113nThaxCpMGW2wWnu2UJcw)
- 提取码：`mgi2`
- **下载包成员文件与具体基线/运行ID的映射尚需由作者归档**，参见`reproducibility/baselines/weights_manifest_TEMPLATE.csv`。官方初始权重与最终微调检查点应分别标识。

建议下载后校验SHA-256：

```powershell
Get-FileHash "D:\desktop\data\pth.zip" -Algorithm SHA256
```

### 3.3 结果来源及prompt/配置对应关系

每个报告结果应关联 `method`、`run_id`、`initial_checkpoint`、`final_checkpoint`、`prompt_file`、`image_mask_manifest`、`config`、`evaluation_script`、`results_csv`及代码commit。

MuralDH 201条manifest已提供于本仓库`reproducibility/muraldh/`；Dunhuang 500/100记录和最终prompt/run/config版本需根据实验日志提供。

## 4. 评价指标

- **G-PSNR / H-PSNR**：全图/仅缺损区PSNR；RGB归一化到[0,1]后逐图计算并等权平均。
- **G-SSIM / H-SSIM**：按RGB通道用`skimage.metrics.structural_similarity`，`win_size=7`、`data_range=1`、`gaussian_weights=False`、`use_sample_covariance=True`、`full=True`。
- **LPIPS**：`lpips.LPIPS(net="alex")`，输入映射到[-1,1]。
- **KB-F1**：已知侧评价边界`Dilate_5(H)-H`；Canny阈值100/200，OpenCV DIST_L2，maskSize=5，容差1像素。两侧边缘均为空时KB-F1=1，仅一侧为空时为0。

## 5. 参考式评价流程

评价脚本：`eval_paired_predictions_paper_documented.py`（需对照实际仓库代码路径及版本）。

```bash
python -m pip install numpy pandas opencv-python scikit-image torch lpips
```

文件示意：

```text
evaluation_inputs/
  predictions/sample_001.png
  references/sample_001.png
  masks/sample_001_mask.png
```

默认`--mask-convention known_white`（白色=已知），如果输入的mask白色=缺失，应使用脚本支持的`hole_white`模式。RGB用INTER_AREA、mask用INTER_NEAREST缩放到256×256。

```bash
python eval_paired_predictions_paper_documented.py \
  --pred-dir /path/to/muraldh_controlled/predictions \
  --gt-dir /path/to/muraldh_controlled/references \
  --mask-dir /path/to/muraldh_controlled/known_masks \
  --outdir evaluation_results/muraldh_controlled \
  --dataset-name MuralDH-controlled \
  --expected 201 \
  --mask-convention known_white \
  --size 256 --device cuda \
  --canny-low 100 --canny-high 200 \
  --boundary-kernel 5 --edge-tolerance 1.0
```

输出应包括`per_image_metrics.csv`、`summary.csv`及`stratified_gpsnr.csv`；具体输出与评估版本以实际运行代码为准。

## 6. 数据与代码可用性

| 资源 | 地址 |
|---|---|
| 本项目 | [Visible-evidence-guided](https://github.com/xinyu0833-bit/Visible-evidence-guided) |
| LaMa官方项目 | [advimman/lama](https://github.com/advimman/lama) |
| StrDiffusion官方项目 | [htyjers/StrDiffusion](https://github.com/htyjers/StrDiffusion) |
| PGRDiff官方项目 | [CZY-Code/PGRDiff](https://github.com/CZY-Code/PGRDiff) |
| 本研究权重归档 `pth.zip` | [百度网盘](https://pan.baidu.com/s/113nThaxCpMGW2wWnu2UJcw)，提取码 `mgi2` |
| Dunhuang数据 | [官方benchmark](https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge) |
| MuralDH数据 | [MuralDH](https://github.com/tearsheaven/MuralDH) |
| MuralDH固定201配对 | [本仓库reproducibility/muraldh](reproducibility/muraldh) |

原始数据及第三方权重的使用和再分发应遵循相应许可。为精确复现全部论文表格，最终run/config/prompt版本和逐图结果仍须与原始训练日志对应。

## 7. 引用

使用本项目时，请引用论文正式发表版本及所用基础方法与数据资源。
