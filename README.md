# Visible-evidence-guided

## 基于可见证据引导与掩膜感知多模态扩散的敦煌壁画虚拟补全

项目地址：<https://github.com/xinyu0833-bit/Visible-evidence-guided>

```bash
git clone https://github.com/xinyu0833-bit/Visible-evidence-guided.git
cd Visible-evidence-guided
```

本项目研究冻结 StrDiffusion 主干上的可见证据条件与空间受限语义适配。方法用于数字虚拟补全、结果比较和专家辅助分析；生成结果不是经验证的历史原貌，也不能代替实体保护修复判断。

本 README 说明论文定义及随修订资料提供的参考式指标脚本。下述评价命令针对 `eval_paired_predictions_paper_documented.py`；将该脚本放入当前工作目录后运行。本文档不以未经核验的训练入口或权重路径替代实际发布配置。仓库中可获取的代码、检查点、提示和记录清单，以对应版本实际包含的文件为准。

## 1. 方法与参数更新范围

主干使用官方 StrDiffusion 的纹理和结构去噪网络，保留其源于 IR-SDE 的均值回归随机过程。主 ConditionalUNet 接收六通道 `Concat(x_t - μ, μ)`，输出三通道反向状态估计；基础宽度为 64，深度为 4。结构条件为掩膜输入的灰度和 Canny 边缘拼接。

\[
\mu=M\odot x_0,\qquad H=1-M,
\]
\[
\hat x_{t-1}
=F_{\theta_0,\phi}\bigl(\operatorname{Concat}(x_t-\mu,\mu),t,S,C_p,M\bigr).
\]

`M=1` 表示已知像素，`H=1` 表示缺失像素。`\theta_0` 是冻结参数，`\phi` **只包含 Prompt Adapter 参数**。

| 部分 | 状态与作用 |
|---|---|
| StrDiffusion 纹理网络、结构网络及结构调制 | 冻结，保留原恢复路径 |
| Qwen2-VL-7B-Instruct | 离线生成可见证据描述，不参与补全训练更新 |
| CLIP ViT-L/14 文本编码器 | 冻结，提供 token 级特征 |
| 多尺度 Prompt Adapter | 唯一可训练部分，位于中间块和三个解码阶段 |
| 掩膜感知路由 | 无可训练参数，由二值掩膜及其膨胀确定 |

Prompt Adapter 将 FiLM 与 token 级交叉注意力产生的分支融合为尺度特异残差。路由图为缺失区域与已知侧边界带之和，经最近邻插值缩放后门控残差。局部门控限定**直接注入位置**，不保证后续卷积和注意力作用后所有已知像素完全不变。

原有监督形式保持为：

\[
\mathcal L=\mathcal L_{\rm hole}+2\mathcal L_{\rm boundary}.
\]

监督对象是反向状态估计与解析反向目标的差异，不在本说明中改写为另一种通用 DDPM 噪声预测目标。路由核尺寸 `k_r`、监督核尺寸 `k_sup` 与评价核尺寸 `k_eval` 分开记录；核尺寸不等同于相同数值的边界带厚度。

## 2. 数据来源与不同评价设置

### Dunhuang

Dunhuang Grottoes Painting Dataset and Benchmark 提供 500 幅开发图像和 100 幅独立测试图像。保持官方图像—掩膜配对。配置阶段以种子 42 从开发集抽取 50 幅验证图像；配置确定后使用全部 500 幅开发图像训练。测试集不进行训练掩膜增强。

官方入口：<https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge>

### MuralDH

原始资源：<https://github.com/tearsheaven/MuralDH>

论文的受控外部评价使用相关工作 760/201 划分的评价子集中的 201 个真实损伤掩膜，与独立抽取的 201 幅参考图像形成固定记录。参考抽样使用种子 42，参考 ID 与掩膜 ID 分别排序后配对。

**这不是 MuralDH 原作者提供的同一壁画真实损伤前后配对恢复基准。** 参考式指标衡量真实损伤几何条件下被隐藏参考像素的重建一致性。对原始真实受损图像的无参考评价另行报告，不能与受控参考式记录混为一谈。不同方法的可用输出覆盖不完全相同，因此无参考集合均值不能作为严格的同图配对排名。

相关划分来源：LABFNet，DOI：<https://doi.org/10.3390/jimaging12070332>。

### muralv2：论文表 4 的掩膜来源

表 4 使用 **PGRDiff 开源项目提供的 muralv2 数据集**中的 `Crack`、`FallenOff`、`Scratch`、`InsectInfestation` 四类掩膜。下载入口见：

<https://github.com/CZY-Code/PGRDiff>

PGRDiff 项目 README 提供 muralv2 入口及 `muralv2/images`、`muralv2/masks` 的组织说明。表 4 的数值是本文利用这些掩膜进行评价所得，不是抄录 PGRDiff 论文的实验数值。muralv2 分类掩膜与 MuralDH 外部评价掩膜必须分别记录来源；不得互换名称。掩膜类别也不等于独立的“轮廓中断”或“装饰纹样受损”语义标签。

原始图像、掩膜及第三方权重的使用和再分发遵循各提供方的条件。

## 3. 提示与控制实验的含义

最终提示采用 `Content and style`、`Degradation`、`Restoration constraint` 三字段；精确损伤几何由二值掩膜提供。描述只依据掩膜后可见证据，不访问缺失区域目标像素。提示随固定对齐记录保存，避免图像、增强掩膜与文本相互错配。

| 条件 | 实际控制内容 | 不能由此推断 |
|---|---|---|
| Matched / sample-specific | 当前图像与其对应提示配对 | 不代表提示中的每项事实均正确 |
| Fixed generic | 所有样本共用固定通用文本 | 不等于在原提示中选择性删除一部分信息 |
| Shuffled | 图像与整条提示错配 | 不等于只改变一个颜色或对象属性 |
| Zero-feature | 令 `C_p=0`，保留 Adapter | 不等于删除 Adapter，也不等于部分描述遗漏 |
| 参数量匹配训练组 | 保持网络与参数量，改变训练提示策略 | 不构成局部事实错误的直接鲁棒性测试 |

现有实验考察提示条件敏感性和样本特异对齐。人工质量审核与下游错误影响是不同问题。**局部事实错误及选择性信息遗漏的直接干预结果尚未报告**，不要将上述对照称为对这两类错误的完整鲁棒性验证。

## 4. 参考式评价脚本

### 4.1 脚本范围与环境

随附脚本：

```text
eval_paired_predictions_paper_documented.py
```

它直接读取已经保存的预测、参考图和掩膜，不训练模型、不生成提示、不执行模型推理，也不计算无参考 IQA 或专家评分置信区间。

基础依赖包括 `numpy`、`pandas`、`opencv-python`、`scikit-image`、`torch` 和 `lpips`。以下命令仅安装评价所需包，不是论文原始环境的锁定文件：

```bash
python -m pip install numpy pandas opencv-python scikit-image torch lpips
```

GPU 运行需要与硬件、驱动相匹配的 PyTorch 环境；CPU 运行可使用 `--device cpu`。LPIPS 所需权重必须在运行环境可用，首次初始化可能需要下载。独立复现应记录实际 Python、PyTorch、OpenCV、scikit-image 和 LPIPS 版本，不应将未锁定依赖的安装命令视为原实验环境证明。

### 4.2 输入组织

以下仅为评价输入目录示例，不表示仓库已经包含原始数据：

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

脚本按完整的归一化文件名主干配对，保留裁剪标识的区分，不只抽取第一个数字。归一化主干重复会报错。

RGB 图像先以 `INTER_AREA` 缩放，再转为 RGB 浮点数并除以 255；掩膜使用 `INTER_NEAREST`。默认分辨率为 256×256。

`--mask-convention known_white` 表示白色为已知区域、黑色为缺失区域。若白色表示缺失，使用 `hole_white`。灰度掩膜最大值不超过 1 时，值大于 0 判为白色；其他情况以值大于 127 判白。空缺失区或全图缺失会报错。

### 4.3 运行示例

Dunhuang 独立测试的参考式指标：

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

MuralDH 受控参考式记录：

```bash
python eval_paired_predictions_paper_documented.py \
  --pred-dir /path/to/muraldh_controlled/predictions \
  --gt-dir /path/to/muraldh_controlled/references \
  --mask-dir /path/to/muraldh_controlled/masks \
  --outdir evaluation_results/muraldh_controlled \
  --dataset-name MuralDH-controlled \
  --expected 201 \
  --mask-convention known_white \
  --device cuda
```

路径必须指向实际数据。muralv2 各类别也可使用同一脚本分别评分，但类别记录、参考图配对与预期数量须来自实际清单，不应从上述示例猜测。

### 4.4 匹配与完整性注意事项

原脚本使用预测、参考和掩膜 ID 的**交集**，不是严格验证三者完整 ID 集合相等。`--expected` 只核验最终数量，不核验每一个 ID 都等于目标清单；默认值 0 表示关闭数量检查。

可选 `--manifest-dir` 实际接收一个**含图像文件的目录**，扫描该目录的 ID 后进一步取交集。它不解析 CSV 或 JSON 配对清单。运行后应检查逐图 CSV 中的 ID，并对照固定实验清单；不能仅凭数量一致断言配对完全正确。

### 4.5 指标定义

#### G-PSNR / H-PSNR

在 `[0,1]` RGB 上计算均方误差。G-PSNR 对所有像素和通道平均；H-PSNR 对缺失位置与三个通道平均：

\[
\mathrm{MSE}_H =
\frac{\sum_pH(p)\sum_{c=1}^{3}(x_c(p)-\hat x_c(p))^2}
{3\sum_pH(p)},\qquad
\mathrm{H\!-\!PSNR}=10\log_{10}(1/\mathrm{MSE}_H).
\]

零误差返回正无穷。

#### G-SSIM / H-SSIM

函数 `ssim_metrics(gt, pr, hole)` 对三个 RGB 通道分别调用：

```python
structural_similarity(
    gt[..., c], pr[..., c],
    data_range=1.0,
    win_size=7,
    gaussian_weights=False,
    use_sample_covariance=True,
    full=True,
)
```

即 7×7 均匀局部窗口，使用样本协方差。稳定常数由库默认提供，`K1=0.01`、`K2=0.03`。记三个完整局部 SSIM 图为 `S_R`、`S_G`、`S_B`：

\[
\bar S(p)=\frac{S_R(p)+S_G(p)+S_B(p)}{3},
\qquad
\mathrm{H\!-\!SSIM}=\frac{\sum_pH(p)\bar S(p)}{\sum_pH(p)}.
\]

代码最终执行：

```python
smap = np.mean(np.stack(maps, axis=0), axis=0)
h_ssim = float(smap[hole].mean())
```

**H-SSIM 是“窗口中心位于缺失区”的局部 SSIM 均值。** 计算局部统计时仍使用完整图像邻域，窗口可以包含已知像素。它不是仅用缺失像素计算窗口统计，也不是先裁剪缺失包围框或将已知区置零后再计算 SSIM。

G-SSIM 对库返回的三个通道标量取平均；该标量按库实现排除图像外侧三像素边界。H-SSIM 对完整局部图选点，不进行这一步裁边。局部滤波的图像外边界采用反射处理。复现时应锁定实际 scikit-image 版本。

实现文档：<https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity>

#### LPIPS

使用 `lpips.LPIPS(net="alex")`，完整 RGB 图像从 `[0,1]` 映射至 `[-1,1]` 后计算。不裁剪到缺失区域。

#### KB-F1

1. 用 5×5 全一方形核对缺失图膨胀一次，并去除原缺失区，得到已知侧边界带。
2. 在完整 8 位灰度参考图与预测图上以 100/200 阈值提取 Canny 边缘，再限制到边界带。
3. 使用 OpenCV `DIST_L2`、距离变换掩模尺寸 5，按一像素容差双向计算 precision/recall，取 F1。
4. 双方边缘均为空时取 1，仅一方为空或 precision+recall 为 0 时取 0。

这是逐图边界一致性指标，不是对损伤掩膜本身计算分割 F1。

### 4.6 输出与汇总规则

| 输出 | 内容 |
|---|---|
| `per_image_metrics.csv` | `id`、`hole_ratio`、六项逐图指标 |
| `summary.csv` | 数据集名称、匹配记录数、平均缺失比例、六项均值 |
| `stratified_gpsnr.csv` | `[0,0.2)`、`[0.2,0.4)`、`[0.4,+∞)` 三组的数量与 G-PSNR |

每项数据集指标先逐图计算，再由 `finite_mean` 对有限值等权平均。它排除 NaN 和正负无穷，因此零误差产生的正无穷 PSNR 不进入均值。`summary.csv` 的 `n` 是匹配记录数，不是每项指标的有限值数量。出现非有限值时，必须结合逐图文件解释均值。

不同面积分层的均值不能直接等权平均来代替全体均值。合并时应核验各组有效记录数，且不要忽略不低于 40% 的组。

脚本不做已知区粘贴、不做颜色后处理；但它本身不能证明上游保存图像时是否进行过这些操作，输出来源应由模型推理记录说明。

## 5. 文档与原计算的一致性

`eval_paired_predictions_paper_documented.py` 只增加原实现的定义和使用说明。除文档字符串外，可执行抽象语法树与本次提供的原始评价脚本一致；没有改变 H-SSIM、PSNR、KB-F1、LPIPS、图像预处理或结果汇总逻辑。

算法定义说明不等于重新运行了论文实验。主表、掩膜类别表、专家统计及提示条件实验需要各自的预测文件、配对清单和运行记录才能独立核验。

## 6. 端到端复现所需资源

完整实验复现还需要实际发布并彼此对应的：模型源码与训练/推理入口、原始冻结主干检查点、各随机种子的 Adapter 权重、固定图像—掩膜—提示记录、训练与评价配置、依赖版本，以及逐图统计。核验时应记录代码提交版本和文件校验值。

本 README 不提供未经核验的 checkpoint 下载地址，也不将单独的评价脚本视为完整模型发布证明。公开访问范围以仓库实际文件为准。

## 7. 致谢、引用与使用边界

感谢 StrDiffusion、IR-SDE、Qwen2-VL、CLIP、PGRDiff 及各数据资源的原作者。本项目未重新授权第三方代码、图像或预训练权重；请保留原始来源并遵守各自许可。论文的正式出版信息以最终发表版本为准。

数字补全结果应清楚标注为算法生成的候选内容，并与原始图像、损伤掩膜和生成条件共同保存。
