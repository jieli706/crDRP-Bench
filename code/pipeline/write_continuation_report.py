"""Write an independent Chinese progress report from recorded results; preserve earlier report."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];LOG=ROOT/'runs/continuation_20260926_125427'
def table(df):
    cols=list(df.columns)
    def fmt(v):
        if isinstance(v,float):return f'{v:.6g}'
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])
pc=ROOT/'runs/model_positive_control_20260925_160423';diag=ROOT/'runs/positive_control_diagnostic_20260926_125204';down=ROOT/'runs/paired_deconvolution_downstream_20260926_130844';cox=ROOT/'runs/clinical_cox_diagnostic_20260926_130209'
m=pd.read_csv(pc/'metrics.csv');m=m[['seed','method','perdrug_pearson_median','rmse']]
d=pd.read_csv(diag/'metrics.csv')[['seed','method','perdrug_pearson_median','rmse']]
p=pd.read_csv(down/'metrics.csv')[['seed','method','perdrug_pearson_median','rmse']]
b=pd.read_csv(down/'paired_donor_drug_bootstrap.csv');b=b[b.metric.eq('delta_median_perdrug_r')][['seed','method_a','method_b','effect_a_minus_b','ci_low','ci_high']]
c=pd.read_csv(cox/'model_cohort_summary.csv')
v=json.loads((LOG/'verification.json').read_text(encoding='utf-8'))
report=f'''# 中文续分析报告：数据准备、正对照及补充诊断

日期：2026-09-26。**这是续分析阶段报告，真实 GDSC 核心模型比较尚未完成。** 旧阶段报告和已完成结果全部保留。本报告及新增诊断只位于 `260925_Re`，不构成对稿件、原图、补充材料或回复信的修改，也没有投稿。

## 1. 核心数据与运行环境

已接收并处理 DepMap Public 26Q1 的 `Model.csv` 和 `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`。1775 条表达记录中选用 default profile，得到 1719 个 ModelID、19215 个基因；其中乳腺来源 71 个，按元数据排除非癌记录后乳腺癌 70 个。表达矩阵未发现重复 ModelID、重复基因符号、缺失、负值或无穷值，均能连接 Model 元数据。**70 是表达可用数，不是最终 GDSC 交集样本数。** 本步骤没有拟合 PCA、scaler 或响应相关特征。

数据位置：`data/processed/depmap_26q1_20260925_161400`。Model MD5 为 `a15d75dffcc5219111ca39598948df9a`，与历史代码记录一致；表达 SHA-256 为 `0377be80c525fde98cbd2c6e8b06bdf2a4014a9683eb70182c1f8649d711021a`。表达版本依据用户 26Q1 页面截图和文件结构，尚未取得发布方独立校验值，不能把本地哈希说成官方版本认证。

仍缺 Harmonized GDSC 25Q2 的三项文件：`GDSC2AUCMatrix.csv`、`GDSC2Log2IC50Matrix.csv`、`GDSC2ResponseCurves.csv`；以及 Sanger GDSC1 and GDSC2 的 `sanger-dose-response.csv`。2026-09-26 官方下载入口和下载 API 再次返回人机验证页；未绕过验证，未把 HTML 当成 CSV，未切换数据版本。最新用户截图中分类框、版本框和文件列表均为空，当前只能确认页面下载区域未呈现，不能认定数据已被删除。具体人工步骤见 `runs/continuation_20260926_125427/手动下载核心数据.md`。

GDSC 入库脚本已准备，包含三个历史 MD5 检查、样本/药物唯一性、名称连接歧义、缺失值和 IC50 实际数值尺度核对；**由于文件未到，这些检查尚未在真实 GDSC 上通过**。药物结构特征覆盖也仍待补齐，取得四个响应文件后仍须核查化学特征，不能用随机向量代替真实核心分析的药物输入。

发现本机既有 `D:/Anaconda3/envs/gat-env/python.exe` 可用：Python 3.10.16、PyTorch 2.0.0/CUDA 11.7，RTX 4060 实测可计算；后续神经网络诊断已用此环境。此前阶段报告中的 Python 3.11 CPU 环境仍保留，二者不可混用扩展包。机器物理内存约 16 GB，显存约 8 GB；本次 E 盘检查剩余约 30 GiB。

R 4.5.2 完整安装包经官方 MD5 `bc35c78e0a6c9790d7d540b44089f31f` 核验。常规安装器因卸载注册表项权限不足回滚；随后在分析目录内解压，修正启动进程的 Windows UTF-8 locale 后 R 和 survival 已正常运行。入口为 `code/run_local_R.ps1`，库、缓存、用户目录和临时目录均指定到本分析目录。BayesPrism 2.2.3 源版本锁定候选为 Git commit `19052052a6f30833b27e2148294459b0ba2f923e`，**尚未安装成功或运行**：依赖 scran 需要 bluster，而 Bioconductor 3.22 官方 Windows 二进制列表及软件页面没有该包，需进一步准备兼容的源码构建环境。探测过 3.21 镜像，但未采用其软件包，也未切换分析数据版本。

## 2. 原架构正对照：已知分室效应能否被学习

上一续分析运行 `runs/model_positive_control_20260925_160423` 使用归档的 crDRP-full/bulk、Transformer-full/bulk 定义，以及 GlobalXGB、药物均值和标签打乱对照。96 个合成细胞、16 个合成药物、8 个分室，seed 7/17/37、3 折细胞留出；同一外层划分、按细胞分组的内层验证、所有变换只在训练实体上拟合。没有预训练，神经网络均为一个配置、至多 120 epoch、patience 20、按内层 RMSE 选模型；XGB 为固定 300 树、深度 4。保存 36864 条 OOF 预测及检查点。

**全部数据和响应均为合成，不能作为真实药敏效果。** 固定调参配置数相同不意味着各模型耗时完全相同；训练日志和预算已保存。实际参数量 crDRP full/bulk 为 31882/29570，Transformer full/bulk 为 80578/79426；full 的细胞输入为 408 维，bulk 为 40 维，药物为 15 维，因此不宣称输入维度或参数量严格相等。因训练样本数限制，本次 PCA 维度与原默认配置不同。

{table(m)}

crDRP-full 在这个已知分室效应任务上获得稳定较高相关；Transformer-full 未通过该正对照，其结果接近打乱标签对照，而 Transformer-bulk 可以学习。保留全部三个种子，不筛选结果。配对细胞×药物 bootstrap（每 seed 1000 次）已保存于该运行的 `paired_two_way_bootstrap.csv`。

新增运行 `runs/positive_control_diagnostic_20260926_125204` 先复载 18 个原 full 检查点，与保存预测差异小于 1e-6，然后做明确标注的“标准化 Z 置零”诊断：保留真实生成用 pi、药物、标签、种子、划分、架构和训练预算，移除合成分室表达噪声。训练输入中每 token 的 Z 平方和均值为 50，而 pi 平方均值约 0.03；这提示输入尺度值得检查，但并非已证明的失败原因。

{table(d)}

置零后 Transformer 仍未表现出稳定恢复能力。因此不能把失败简单归结为表达噪声，也不能将这个更容易的诊断替代原失败试验。后续需要在独立开发数据和内层训练范围内检查模型优化、pi 的编码及早停行为，再做真实数据解释；本次没有以测试结果为依据反复调参寻找成功设置。

## 3. 真实图谱来源的配对混合物与下游预测

新增 `runs/paired_deconvolution_inputs_20260926_130450`，复用已完成的 Wu 图谱 QC 和供者分室均值。每 seed 将 26 位供者分成 12 位外部参考、5 位预测器训练、2 位内层验证、7 位测试，四组完全隔离。基因仅由参考供者选取；每供者/分室最多抽取 20 个已测量单细胞，形成 1682、1661、1692 个参考细胞的稀疏整数计数矩阵。

每 seed 生成 84 个混合物，每个 20000 UMI、2000 基因；train/validation/test 分别 30/12/42 个。所有方法必须使用已冻结的同一混合物、基因和划分，保留参考细胞 ID、供者 ID、原始计数、真实 pi 和真实表达 profiles。**混合物是模拟的；pi 表示所选基因空间中已归一化的 RNA 贡献比例，不能解释为实测肿瘤细胞比例。** 这与旧 420 个 NNLS 混合物试验是不同的四组供者隔离设计，旧结果未覆盖。

`runs/paired_deconvolution_downstream_20260926_130844` 已将 oracle 和 NNLS 输入接入实际 crDRP，并比较 bulk：oracle 使用生成数据时已知的分室比例和表达；NNLS 用同一参考细胞聚合后的表达估计比例，再按 pi×参考表达分配观察计数并归一化得到分室表达。后一步是明确规定的 NNLS 后处理，不是 BayesPrism 估计。

下游采用 16 个合成药物特征和预先规定的已知 pi 响应公式；没有真实药敏标签。所有模型共用预测器训练/验证/测试划分；PCA、scaler 和特征选择均仅训练混合物拟合，无预训练。full/oracle 与 full/NNLS 参数量均 30538、细胞输入 240 维；bulk 参数量 28866、细胞输入 29 维；药物输入 15 维。每模型/seed 最多 120 epoch，同一早停规则和一个配置，实际训练轮数及所选 epoch 全部保存。BayesPrism 未运行，因此是**部分完成的下游验证**。

{table(p)}

保存 6048 条测试预测。这里是每 seed 一次供者留出的测试预测，**不是覆盖全体供者的 OOF**；不同 seed 同时改变供者划分和模拟数据，不能当成纯初始化种子实验。每 seed 只有 7 个测试供者；bootstrap 按供者和药物双向抽样，对配对模型使用相同权重，同供者的 6 个混合物始终一起加权，避免将它们当作独立生物学重复。效应为逐药物 Pearson 中位数之差：

{table(b)}

置信区间仅针对固定模型、当前模拟设计和有限测试供者；不包含重新训练不确定性，不保证小样本下严格覆盖率，不合并三个 seed 为独立样本。oracle 在三个种子下均较 NNLS 高，NNLS 相对 bulk 的方向随种子变化。这支持“真实输入质量会影响这个构造任务的下游性能”，但不证明 GDSC 上的优劣，也不能证明 BayesPrism 可修复或不能修复问题。

## 4. 真实临床数据的 Cox 与比例风险诊断

复用已有患者清单，未重新下载：TCGA 生存可用 1076 人、150 个事件；METABRIC 1980 人、1144 个事件。新运行 `runs/clinical_cox_diagnostic_20260926_130209` 预设年龄每 10 年的模型及年龄+分期模型，用完整病例、不插补、不按 P 值挑变量；METABRIC 分期从一对一的 sample 表匹配。保存所有纳入/排除患者、模型 RDS、样本内线性预测、系数、Schoenfeld 检验与诊断图。

{table(c)}

TCGA 年龄+分期模型的分期 PH 检验 P=0.01467、全局 P=0.02354；METABRIC 年龄+分期模型中年龄和分期均显示明显时间依赖（P 约 2.17e-18、1.58e-5）。这些是探索性模型诊断，提示后续不能不检查就用单一恒定 HR 概括。不能由此断言原稿模型必然违例：本次不是原表达交集队列，也没有分室输入。没有重现 B/plasma 生存效应、治疗获益或原稿 log-rank 结果，亦未把样本内预测当作外部预测准确率。

## 5. 对应审稿要求的状态（按意见主题归并）

| 审稿主题 | 当前完成内容 | 仍未完成/不能支持 |
|---|---|---|
| 核心 full/bulk、GlobalXGB、Transformer 同划分比较 | 真实表达已入库；原架构合成正对照、划分与变换隔离完成 | 真实 GDSC 缺 4 文件，实际药物结构特征也待核查；不能给出真实药敏结论 |
| 预训练与预处理泄漏 | 已明确旧代码问题；新试验按训练实体拟合，无测试数据参与；新试验无预训练 | 严格非乳腺预训练及 drug-holdout 的真实运行待数据 |
| 容量匹配与调参预算 | 实际参数量、输入维度、训练预算全部记录 | 输入维度及参数量并非严格相等；尚无容量匹配的真实敏感性分析 |
| 已知分室效应正对照 | 已进入实际 crDRP 和 Transformer；包含打乱标签和 Z 置零诊断 | Transformer-full 未通过；不能继续把它的阴性结果作为分室无价值证据 |
| NNLS、BayesPrism、oracle 比较及误差传播 | 相同混合物已冻结；NNLS/oracle 已接入实际 crDRP 下游及 bulk 比较 | BayesPrism 环境未完成，三方法比较未完成；下游响应仍是合成 |
| 多种子、配对 bootstrap、效应与区间 | seed 7/17/37 均保留；合成 OOF 用细胞×药物；图谱混合物用供者×药物 | 真实 OOF、多初始化及重训练不确定性未完成 |
| 恶性/正常上皮区分、低纯度行为 | 旧供者留出 NNLS 结果保留；新混合物真实输入与参考严格隔离 | 缺独立真实纯度测量；不声称与原 Fig S5 数值等价 |
| 独立图谱/供者与跨癌种外推 | 新补充设计将参考、预测器训练、验证、测试供者四组隔离 | 仍来自 Wu 同一图谱；不是独立外部图谱，也未验证八分室适用于所有谱系 |
| 响应方差分层、pooled 与逐药物差异、drug-holdout 退化 | 指标定义分开，保留逐样本预测和无效相关计数；真实入库脚本准备 | 真实响应未取得，不能进行相应方差分层或药物留出验证 |
| 生存结论稳健性、年龄/分期调整和 PH 假设 | 已完成真实临床协变量 Cox/PH 诊断，列出缺失与事件数 | 还缺表达和分室比例的临床交集；未验证 B/plasma HR、分型交互或 FDR |
| 摘要/正文不同性能数字及 CV 定义对照 | 延续旧报告对 0.22/0.29、0.204/0.244/0.30、0.19/0.35 的来源差异核查 | 本次模拟结果不能与这些真实任务数字直接相减；不能选择最接近原稿的 seed |
| 数据资源可复核、明确 accession/版本与归档 | 数据来源、哈希、ID、划分、配置、检查点、预测、日志已保存 | 完整重建资源和可公开归档发布尚未完成；未创建 DOI 或对外发布 |
| 其他文字、图表、图例和回复要求 | 只记录后续应核对主题 | 依用户要求全部暂缓，不修改投稿材料 |

## 6. 可复核性、失败记录与停止点

9 个新增下游检查点已在 CPU 上结合保存的变换复载，最大预测差为 {v['max_cpu_replay_error']:.3g}。原项目 {v['prior_project_files_checked']} 个文件的尺寸/修改时间未变；先前 {v['prior_stage_hashed_files_checked']} 个归档文件 SHA-256 无变化。详见 `runs/continuation_20260926_125427/verification.json`、`downstream_checkpoint_replay.csv` 和 `new_analysis_artifacts_sha256.csv`。

未成功的下载、依赖安装、首轮临床表连接及首次下游浮点类型运行记录均保留，不删除、不冒充成功。成功结果以各运行 `COMPLETE.json` 的**具体 scope**为准；它不表示整个返修分析完成。R 初次解压启动的 locale 问题已修复，临床表连接已改用真实 sample 表，神经网络 pi 已统一 float32；相应复载核验已通过。

真实药敏小规模流程尚未通过，所以未开展完整 GDSC 训练；没有付费资源或账户需求。当前需要用户帮助的仍是官方正常浏览器取得上述 4 个指定数据文件。缺包编译、Transformer 优化诊断及其他待分析项保留为后续任务。获得核心文件后先核对版本/哈希、ID、重复、缺失与单位，再处理真实药物特征及端到端小规模测试。**无论结果是否与原稿一致，均保留全部预定种子；整体分析完成后先报告，未经用户明确确认不进入改稿阶段。**

软件/方法核查来源：[R 官方 Windows FAQ](https://cran.r-project.org/bin/windows/base/rw-FAQ.html)、[innoextract 官方说明](https://constexpr.org/innoextract/)、[BayesPrism 固定源码版本](https://github.com/Danko-Lab/BayesPrism/tree/19052052a6f30833b27e2148294459b0ba2f923e)、[Bioconductor 3.22 bluster 页面](https://bioconductor.org/packages/3.22/bioc/html/bluster.html)、[多向 bootstrap 方法论文](https://arxiv.org/abs/1106.2125)。bootstrap 的均值理论不能直接当作小样本相关中位数区间覆盖率的证明。
'''
path=ROOT/'reports/中文续分析报告_20260926.md';path.write_text(report,encoding='utf-8')
status={'status':'stage_report_ready_core_real_drug_analysis_blocked','report':str(path.relative_to(ROOT)),'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'missing_core_files':['GDSC2AUCMatrix.csv','GDSC2Log2IC50Matrix.csv','GDSC2ResponseCurves.csv','sanger-dose-response.csv'],'BayesPrism_completed':False,'Transformer_positive_control_passed':False,'manuscript_editing_authorized':False}
(LOG/'stage_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8');print(path)
