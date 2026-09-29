"""Generate a separate Chinese analysis report from saved results, never submission files."""
import argparse,json,csv,hashlib,shutil
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd
from leakage_safe import metrics
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--correction-index',required=True);p.add_argument('--supplement-index',required=True);p.add_argument('--verification',required=True);a=p.parse_args()
fix=json.loads(Path(a.correction_index).read_text());supp=json.loads(Path(a.supplement_index).read_text());V=Path(a.verification);verify=json.loads((V/'verification.json').read_text());stamp=datetime.now().strftime('%Y%m%d_%H%M%S');O=ROOT/'reports'/('真实数据分析_'+stamp);O.mkdir()
def link(path,label=None):
    path=Path(path);return '['+(label or path.name)+'](<'+path.resolve().as_posix()+'>)'
def table(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return 'NA' if not np.isfinite(x) else f'{x:.5g}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','|'+'---|'*len(df.columns)]+['| '+' | '.join(fmt(x) for x in r)+' |' for r in df.itertuples(index=False,name=None)])
def read(run,file):return pd.read_csv(Path(run)/file)
def predmetrics(run):
    run=Path(run);f=run/('paired_baseline_predictions.csv' if (run/'paired_baseline_predictions.csv').exists() else 'oof_predictions.csv');pred=pd.read_csv(f)
    return pd.DataFrame([{'seed':s,'method':m,**metrics(g)} for (s,m),g in pred.groupby(['seed','method'])])
def pivot(df,col):return df.pivot(index='method',columns='seed',values=col).rename(columns={7:'seed 7',17:'seed 17',37:'seed 37'}).reset_index()
def effects(run,file='paired_two_way_bootstrap.csv',methods=None):
    d=read(run,file);d=d[d.metric.eq('delta_median_perdrug_r')]
    if methods:d=d[d.method_a.isin(methods)]
    return d[['seed','method_a','method_b','effect_a_minus_b','ci_low','ci_high']]
core={k:predmetrics(fix[k]) for k in ['cell','pair','drug']}
count=ROOT/'runs/real_gdsc_cell_20260926_150657';countm=predmetrics(count)
mixture=ROOT/'runs/bayesprism_paired_downstream_20260926_150614';mixm=read(mixture,'metrics.csv');trans=read(supp['transfer'],'metrics.csv');prism=read(supp['prism'],'metrics.csv')
allm=[]
for name,df in [(k,core[k]) for k in core]+[('count_matched_cell',countm),('transfer_cell',trans),('PRISM_external_assay',prism),('synthetic_mixture_real_atlas',mixm)]:
    q=df.copy();q.insert(0,'analysis',name);allm.append(q)
pd.concat(allm).to_csv(O/'全部种子指标.csv',index=False,encoding='utf-8-sig')
alle=[]
for name,run,file in [(k,fix[k],'paired_two_way_bootstrap.csv') for k in core]+[('count_matched_cell',count,'paired_two_way_bootstrap.csv'),('transfer_cell',supp['transfer'],'paired_two_way_bootstrap.csv'),('PRISM_external_assay',supp['prism'],'paired_two_way_bootstrap.csv'),('synthetic_mixture_real_atlas',mixture,'paired_donor_drug_bootstrap.csv')]:
    q=read(run,file);q.insert(0,'analysis',name)
    if name=='synthetic_mixture_real_atlas':q['scope']='conditional_on_fixed_donor_holdout_models_NOT_OOF'
    alle.append(q)
pd.concat(alle).to_csv(O/'全部配对效应及置信区间.csv',index=False,encoding='utf-8-sig')
sections=[]
def add(s):sections.append(s.strip())
add('''# 真实数据补充验证独立分析报告

日期：2026-09-26。本报告只属于 `260925_Re` 分析目录，未修改稿件、原图、补充材料或回复信，未投稿。旧阶段报告及失败运行全部保留；其中“GDSC 文件缺失”“BayesPrism 未安装”等历史状态由本报告更新。**已完成可运行的核心真实数据比较及多项补充验证，但没有完成原稿全流程的精确复现，也没有完成所有审稿要求。**

结果不能支持将“本设置下未观察到稳定优势”扩展为“分室信息无价值”。真实 GDSC 的细胞留出表现弱；随机配对任务更容易，不能用它替换新细胞泛化。BayesPrism、NNLS 和 oracle 的配对模拟下游结果有差别，说明输入估计质量会影响构造任务，但不等于已证明真实药敏中的单一失败机制。Transformer-full 未通过已知分室效应正对照，其阴性结果不构成架构独立的机制证据。

## 1. 数据、版本与真实匹配

用户保存的官方目录清单已成功用于取得指定文件，没有绕过人机验证，也没有以其他发布版本替换目标版本。目录中的签名下载参数只留在本地源清单，不放进报告。

| 数据 | 本次版本和用途 | 核查 |
|---|---|---|
| DepMap Model、log2(TPM+1) 表达 | Public 26Q1 | 两个文件官方 MD5 匹配；1775 原始表达记录按 default profile 选得 1719 个模型、19215 基因；70 乳腺癌模型 |
| DepMap raw read count、OmicsProfiles | 同一 Public 26Q1 | 官方清单校验；45 个目标模型的 SequencingID 与 TPM 逐一相同；整数计数核查 |
| GDSC AUC、Log2IC50、ResponseCurves | Harmonized GDSC 25Q2 | 三个历史/官方 MD5 均匹配；另保存 Sanger sanger-dose-response.csv，用于身份及版本对照 |
| 药物结构 | PRISM 19Q4、ChEMBL 37 | Broad ID/唯一精确名称或同义词连接；不接受模糊首命中。ChEMBL37 压缩包官方 SHA-256 匹配 |
| PRISM 外部响应 | Repurposing Secondary Screen 19Q4 | 三个原始文件及哈希保存；只用于冻结模型的外部测定验证 |
| Wu 单细胞 | 原代码指向的 CELLxGENE 历史资产 0e3974a1-9488-4987-9767-21d4e389fac7 | 原始100064细胞、26供者，QC后100034；使用作者标签重建8类，未重跑原scVI/inferCNV |
| TCGA 表达 | Xena TCGA-BRCA STAR counts，元数据05-20-2024/GDC40 | 原脚本公开入口；下载快照/本地SHA与服务端时间保留；log2(count+1)逆变换，不误作TPM |
| TCGA/METABRIC 临床 | 已下载快照继续复用 | 患者/样本映射、事件和时间、协变量缺失均有清单；METABRIC本次未重建分室表达 |

GDSC 原始 AUC 矩阵为949细胞×286药物，230598个非缺失值，范围0.01214–1。与表达和乳腺癌元数据连接得到45细胞；按至少20个有效响应筛选，275药物、11146个有效细胞–药物对。模型使用有唯一可核对结构的133药物、5561对（54来自PRISM结构、79新增自ChEMBL37）；其余142个未进入当前结构特征集，逐药物连接、歧义与版本覆盖原因均保留在筛选表。没有按预测效果筛药。原稿117药物的最终ID清单未找到，不能声称133药物与原117药物完全一致。

注：原275中未纳入数量应直接以275−133=142为准；程序清单保留全部候选与每步排除原因。所有拟合只使用非缺失AUC，不填补响应；重复Model/药物键在分割前拒绝。PRISM技术重复按固定规则取中位数，并保留原行、重复数、屏幕来源和范围。

IC50 单位已用**同版**曲线解决，取代旧审计中跨版本对照的“unresolved”：62556个非缺失矩阵项均与曲线绝对存活率0.5交点的 log2(微摩尔) 一致，MAE 6.63×10⁻⁶、最大差4.73×10⁻⁵。25387个矩阵缺失但曲线可求有限交点，另142655个无有限交点；没有擅自补入缺失矩阵。EC50不等于绝对IC50。本次主终点仍为无量纲AUC，不能把AUC相关性写成IC50预测结果。

## 2. 评估与泄漏防护

三个预定种子7/17/37均保留；每种子三外层折。cell按ModelID留出、drug按药物ID留出，内层20%验证使用同一分组原则；pair任务允许同一细胞和药物在训练/测试出现，但细胞–药物对互斥。三任务回答不同问题。没有用测试集早停或挑参数。

缺失描述符的填补、基因/描述符变异筛选、scaler、PCA及后缩放均只用内层训练的唯一实体拟合，并保存fit IDs。药物原始3661维=2048位Morgan指纹+1613个Mordred二维描述符，固定结构计算可逐药物完成，统计变换不可见验证/测试药物。外部Wu参考的基因选择不使用GDSC响应。

主比较无预训练；神经网络统一最多120 epoch、patience20、batch256、AdamW lr0.001/weight_decay0.0003、Huber delta0.05、梯度裁剪5，均一个配置，按内层RMSE保存最佳模型，不在外层训练全集重拟合。GlobalXGB固定最多300树/depth4/lr0.05、同一验证早停20；Ridge alpha1；drug_mean只用训练药物均值，未知药物回退训练总体均值。一个配置不意味着算力、树模型容量或参数量严格相等。

预训练补充排除全部71个乳腺来源模型，得到654个非乳腺癌来源模型、81440对；按来源细胞划分10%验证。来源预训练最多20 epoch/patience5/batch512，目标微调恢复batch256及上述预算。来源预处理也使用目标训练折冻结的基底，验证和测试实体均不参与拟合。这是明确的20轮来源预算补充，不是原60/200轮训练的精确复现。仅运行cell迁移；严格drug-holdout预训练与Transformer预训练尚未完成。

统计主量为逐药物OOF Pearson的中位数；同时报告pooled r、RMSE、MAE、R²和逐药物预测方差。相关未定义时保留NA与数量，不能记为0。每seed1000次细胞×药物乘积多项式配对bootstrap，两模型使用相同权重与完全一致的键/折/真值；模拟混合物改按供者×药物抽样。报告中Δr是两个逐药物相关中位数之差；表中还另存逐药物差值的中位数，两者不可混写。ΔRMSE<0有利于前一模型。

区间条件于已训练模型，不包括重训练、参考图谱选择或所有预处理不确定性；未做预设等效性界值检验。每seed既改变折划分又改变初始化，不能解释为纯初始化重复，也不把三seed当作独立患者合并检验。多模型探索性区间未统一作多重比较校正。

## 3. 独立复核发现的数值问题及修正

在独立进程复载时发现TPM/NNLS逐分室PCA的近零特征值被后缩放放大：同一Transformer检查点在BLAS 2线程时最大预测差0.11935，16线程时恢复到1.75×10⁻⁷；CPU/CUDA差异不是根因。原运行内即时重载只能验证权重，未暴露跨进程预处理数值问题。

新增StableFoldTransform仅依据训练PCA特征值，将低于max(10⁻¹²,最大特征值×10⁻¹⁰)的数值零空间分量固定为0，保留输入尺寸。该阈值依据数值秩设定，没有观察测试效能调阈值。所有受影响full模型、全部种子及三任务重跑；bulk、Transformer-bulk、Ridge、XGB、drug_mean的非退化变换和原预测复用。来源预训练full也重新训练，未复用受影响权重。旧结果与诊断完整保留，但不作为最终有效full结果。

TPM分支仍采用每分室分别拟合的PCA，与原稿共享Z基底有方法差异；独立的原始计数NNLS/BayesPrism比较采用训练细胞×分室共享PCA，更接近原输入设计。这两个分支也改变参考供者与表达单位，不能把它们的差异归因于单一预处理步骤。
''')
for k,title in [('cell','细胞留出：新细胞泛化'),('pair','随机配对留出：已见实体组合'),('drug','药物留出：新药物泛化')]:
    add('## 4'+{'cell':'a','pair':'b','drug':'c'}[k]+'. '+title+'\n\n全部133药物、5561对；下表为逐药物OOF Pearson中位数。\n\n'+table(pivot(core[k],'perdrug_pearson_median'))+'\n\nRMSE：\n\n'+table(pivot(core[k],'rmse'))+'\n\n配对Δr及95%区间：\n\n'+table(effects(fix[k]))+'\n\n运行：'+link(fix[k]))
add('''cell结果不能从pair结果借用；对drug任务，训练药物与测试药物完全不交叉，drug_mean只能用总体均值，因此逐药物相关通常未定义。OOF合并的细胞任务drug_mean可以因不同折均值产生负相关，不代表它学习了细胞反应；另有折内相关诊断避免这种混淆。

TPM-cell分支参数量：crDRP-full 36618、bulk 34946（full多4.8%）；Transformer-full 85314、bulk 84802。full细胞输入192维，bulk23维，药物116维；这里每折仅24个训练细胞，bulk/PCA不能虚报40个有效方向。共享计数分支full39242、bulk34946（多12.3%），输入520/23维、药物116维。因此不能使用“参数量和有效输入维度严格匹配”的表述。各折实际维度、训练对数、早停轮次与耗时均在capacity_and_training_budget.csv；drug_mean表中记录的可用表达维度不代表该基线使用了表达，实际仅用药物ID。

## 5. 非乳腺预训练补充
'''+table(pivot(trans,'perdrug_pearson_median'))+'\n\n'+table(effects(supp['transfer']))+'\n\n运行：'+link(supp['transfer'])+'''

早期target batch512试验与随后batch256试验都保留；修正数值零空间后的最终运行使用新的来源预训练和batch256目标训练。只将同一最终预处理分支下的scratch/transfer作为配对比较，不能为接近原稿挑种子或引用早期更好结果。

## 6. 真实原始计数：NNLS与BayesPrism接入crDRP

使用同一45细胞整数read count、1896基因、1682个真实单细胞、12个外部参考供者。两个解卷积方法共用输入计数、基因、参考与外层划分，下游使用共享Z PCA。bulk也用同一所选基因空间log1p(CPM)，不会把TPM-bulk混入本比较。

BayesPrism 2.2.3固定提交19052052a6f30833b27e2148294459b0ba2f923e；运行initial Gibbs、chain1000/burn500/thin2、算法seed7。冻结外部参考、关闭依赖整个混合样本队列的基因过滤，不作联合跨样本reference update。故方法名是BayesPrism_fixed，不能伪装成原完整更新版本；仅一个算法链，后验CV不是已证明收敛或覆盖率的区间。三个下游种子复用同一解卷积输入。

NNLS分室表达为明确记录的条件计数分配；BayesPrism使用估计的Z；各分室归一化后log1p(CPM)输入。UMI参考与bulk read counts仍有平台/基因长度差异，细胞系不存在已知真实分室比例，不能把估计比例当作验证真值。

'''+table(pivot(countm,'perdrug_pearson_median'))+'\n\n'+table(effects(count,methods=['crdrp_full_bayesprism_fixed']))+'''

BayesPrism相对NNLS的Δr依seed为−0.04963、−0.03047、+0.05821，三个区间均跨0；相对bulk也没有稳定、区间排除0的逐药物优势。seed37的RMSE相对NNLS改善而seed7方向相反，不能只选seed37。该结果不支持“换用BayesPrism必然修复真实药敏预测”，也不足以证明所有BayesPrism版本均无帮助。

## 7. 同一测试混合物的oracle、NNLS、BayesPrism和bulk

真实Wu图谱计数用于构造混合物；**混合比例、混合物及下游响应是模拟的**。每seed将26供者分为参考12、预测器训练5、验证2、测试7；84混合物中30/12/42用于训练/验证/测试，每个20000 UMI。16个合成药物及已知分室效应响应公式完全冻结，oracle使用生成用真实pi和Z。pi是选定基因空间的RNA贡献，不等同组织学细胞纯度。

四种输入均接入同一实际crDRP和预算，测试混合物一致；既比较比例恢复，也比较下游预测。每seed672个测试对，合计四方法8064条预测；这些是供者留出测试，不是全体供者OOF。旧通用bootstrap文件scope字段写有OOF，属于通用模板残留，本报告汇总CSV纠正为fixed donor holdout，原文件保留。

'''+table(pivot(mixm,'perdrug_pearson_median'))+'\n\n'+table(effects(mixture,'paired_donor_drug_bootstrap.csv'))+'''

BayesPrism相对NNLS的Δr三个点估计均为正，但seed17区间跨0；oracle相对BayesPrism在seed7区间也跨0。结果显示估计误差能传播到这一已知信号任务的预测，不能将“模拟成功”说成真实GDSC成功。每seed只有7个测试供者，区间有限；测试供者所有混合物共享抽样权重，没有把42混合物当42独立患者。

补充按真实恶性RNA贡献<0.3、0.3–0.7、≥0.7分层的比例恢复和下游诊断。三seed低贡献组分别38/37/34个混合物，NNLS恶性MAE为0.1037/0.1389/0.0832，BayesPrism为0.0637/0.0765/0.0442；高贡献组只有0/0/1个，不能据此估计完整纯度响应曲线或声称高纯度退化规律。该描述不重拟合模型、不等同原Fig S5固定梯度试验。明细在runs/mixture_malignant_RNA_strata_20260926_152245。

## 8. 已知分室效应正对照与Transformer限制

实际归档crDRP-full、bulk、Transformer-full、bulk、GlobalXGB及标签打乱对照已运行96合成细胞×16药物、三seed/三折。crDRP-full逐药物r为0.99123/0.99155/0.98519；Transformer-full为−0.01558/−0.09056/−0.03561，未通过，而Transformer-bulk为0.88902/0.93737/0.92618。保留全部失败结果，没有用旧Ridge正对照替代。

另做Z置零诊断后Transformer仍不稳定（0.16456/−0.07165/−0.11242），不能简单归因于表达噪声。这里的合成正对照和真实TPM数值秩问题是不同诊断；本次没有根据测试集反复搜索超参数修饰正对照。Transformer优化能力、pi编码及更系统的内层开发仍待解决，因此不支持“两种充分验证架构共同排除分室效应”的强结论。

## 9. 独立测定、响应方差和临床诊断

PRISM 19Q4筛选STR通过、AUC有限、唯一结构/Broad-ID映射，未按R²或效能筛选。1577原行中273个AUC在[0,1]外，按原值保留，不为了与GDSC一致而截断；重复聚合后1034对，其中1021与GDSC评估对相交，涉及20细胞54药物。复用冻结的cell-heldout OOF预测，无PRISM训练、调参或校准；是外部测定平台验证，**不是独立生物学队列**。不同剂量范围和实验流程限制绝对RMSE直接比较；两测定pooled响应相关0.50678。

'''+table(pivot(prism,'perdrug_pearson_median'))+'''

响应方差分层仅使用训练折SD的固定区间<0.02、0.02–0.05、0.05–0.10、≥0.10，不用测试值分层挑模型。当前5561对的总响应平方和中56.63%为药物间均值差；因此pooled r约0.7不代表跨细胞排名准确。保存按药物/折的响应和预测SD、预测SD<0.01比例及折内Fisher汇总，后者只是补充诊断，不替换预设OOF主量。

TCGA计数表达可与1071个已有合格患者匹配，150事件；年龄/分期完整病例1034人、140事件。8个分室分别作年龄/分期调整Cox，统一BH校正；B/plasma另加PAM50敏感性。NNLS比例是新代理暴露，不是原BayesPrism比例。B/plasma每SD HR=0.95863，95%CI[0.79037,1.16271]，P=0.66787，8项BH q=0.76328；加PAM50后HR=0.94932，CI[0.77975,1.15575]，P=0.60439。8项均未过FDR0.05，myeloid名义P=0.01691但q=0.13532。

B/plasma约70.5%估计为0，三分位切点都为0，因此跳过三组KM，不人为拆分同值。B/plasma Cox暴露PH检验P=0.51561，但全局PH P=0.04065；加PAM50全局P=0.00642，不能无保留解释恒定HR。CAF均值约0.65、若干分室大量为0也提示平台/参考适配限制。没有独立METABRIC分室生存复现，没有证明治疗获益或因果效应。

既有纯临床协变量PH诊断继续保留：TCGA年龄+分期全局P=0.02354，METABRIC约2.68×10⁻²²；这些不是原稿B/plasma模型的直接否定。
''')
review=[
['R2-1 阴性结论与纯度/训练域','部分完成','cell/pair/drug分开；模型正对照、跨平台PRISM、计数分支','没有真肿瘤纯度测量；细胞系不能代替患者训练域'],
['R2-2 NNLS弱点与BayesPrism归因','部分完成','同混合物NNLS/BayesPrism/oracle进入实际下游；真实GDSC同计数比较','固定参考initial Gibbs不是原完整更新版本；未证明唯一机制'],
['R2-3 纯度梯度矛盾及外推','部分完成','供者隔离模拟、真实估计分布与不确定性保留','尚无原Fig S5固定纯度梯度/真纯度的精确重现'],
['R2-4 八分室跨癌种适用性','未验证广泛适用','非乳腺来源预训练明确排除目标','来源谱系可能缺失参考类型；不能称八分室适用全部癌种'],
['R2-5 肿瘤真实验证','部分完成','TCGA分室代理Cox、PAM50、PH、FDR','缺病理/独立分子分室真值及METABRIC表达复现'],
['R2-6 性能数字与CV对应','完成本次映射','每表注明任务、输入分支、预训练、seed与OOF范围','不能用本次133药物替原117药物数值'],
['R2-7 bulk可比性与容量','部分完成','同划分、预算；bulk和full参数/输入维度逐折保存','并非严格相等容量；未做完整维度/参数预算网格'],
['R2-8 统计、不确定性和响应方差','部分完成','三seed、配对细胞/药物bootstrap、训练SD分层、逐样本OOF','未做重复重训练bootstrap/独立初始种子网格/等效性检验'],
['R2-9 生存与组成混杂','部分完成','年龄/分期/PAM50、事件/缺失、8项FDR和PH；不拆分零值','新NNLS代理不是原暴露；不作治疗获益推断'],
['R2-10 数据可用性','部分完成','来源、版本、ID、筛选、模型、预测、日志和哈希','完整原资源、所有bulk矩阵和公开归档DOI未完成'],
['R3-major 正对照/oracle','完成指定构造验证','实际crDRP及Transformer正对照；oracle接入真实架构','Transformer正对照失败必须保留'],
['R3-major BayesPrism下游替代','完成一个明确变体','fixed BayesPrism实际GDSC与配对混合物下游','未完成canonical reference-update版本'],
['R3-major capacity matched定义','完成披露，敏感性未完','参数、输入维度、配置数和实际训练轮次','不能继续称严格容量匹配'],
['R3-major 独立scRNA/供者','部分完成','参考/训练/验证/测试供者四组隔离','仍同Wu图谱，独立图谱未引入；不能外推全癌种'],
['R2-minor1 BayesPrism gold standard措辞','分析中已区分','只称指定版本/变体，不把估计当真值','稿件措辞待用户确认后处理'],
['R2-minor2 GDSC/PRISM实验与整合来源','完成来源披露','原始实验测定与DepMap发布/整合版本分开记录','不改稿件'],
['R2-minor3 scatter与纯度趋势图例','部分完成','真实RNA贡献分层保存，明确高贡献组样本不足','原图不改；固定纯度完整曲线未重建'],
['R2-minor4 常数预测','完成诊断','逐药物预测SD、固定0.01阈值、折内相关和drug_mean例外','不把未定义相关填0'],
['R2-minor5 化学特征/PCA维度','完成本次披露','3661原始维度、折内116维上限及具体rank已保存','原稿34/11维说法不能直接沿用本次流程'],
['R2-minor6 PRISM drug-holdout异常','未完成该项精确验证','已完成PRISM外部测定cell-heldout验证','尚未重新训练PRISM drug-holdout，不能解释原0.25'],
['R2-minor7 Transformer机制/消融','部分完成','实际正对照及Z置零均披露失败','尚无token打乱/移除跨分室注意力的完整消融'],
['R2-minor8 pooled与逐药物指标','完成区分','OOF逐药物主量、pooled诊断、训练SD分层','不能用pooled替代逐药物结果'],
['R2-minor9 KM分组检验自由度','完成当前诊断','三组本应df2，当前切点重合故不强行KM','原图log-rank不改；未将原P作为本次结果'],
['R2-minor10 亚型标签/分层与leak-free标注','部分完成','元数据ID和癌谱系筛选、严格cell分割记录','未完整复建原表亚型/分型分层结果；原图例暂缓'],
['R3-minor1 数值总表','完成本次','全部种子、任务和模型汇总CSV与索引','原稿数字来源待改稿阶段统一措辞'],
['R3-minor2 通用组件与既有文献定位','部分完成','预训练与drug_mean作显式对照','文献定位与讨论段落不在本轮改动范围'],
['R3-minor3 accession/版本/归档','部分完成','官方来源、版本与文件哈希留档','未公开上传/发DOI；需后续明确发布授权']]
rd=pd.DataFrame(review,columns=['审稿主题','状态','本次证据','限制或后续']);rd.to_csv(O/'审稿要求逐项状态.csv',index=False,encoding='utf-8-sig')
add('## 10. 对应审稿意见的完成状态\n\n按两位审稿人截图的主题与主要意见序号归并；不是回复信。\n\n'+table(rd))
add('''## 11. 与原稿的差异及可支持结论

原文摘要0.22/0.29、Fig S3d的0.204/0.244/0.30、Fig S7b的0.19/0.35来自不同模型/任务/预训练与保留集，不能当成同一结果。本次逐药物相关必须与本报告具体任务、分支、种子一起引用；没有挑选接近这些数字的模型。细胞留出下本次真实表现明显弱于这些原文引述值，但因药物集、参考构建、特征基底、严格实体隔离、环境和预算不同，不能把数值差全部归因于泄漏修复。

原已缺失82项分析数据仍不等于全部重建完成：45个乳腺目标的核心药敏资源已重建，泛癌预训练只覆盖本次可匹配654来源模型；没有1719泛癌和全部TCGA/METABRIC的原八分室完整发布矩阵。作者标签参考替代原scVI/inferCNV，药物133而非原117；这些是透明的新验证设计，不冒充精确复现。

支持：本设置中观察不到full对bulk的稳定新细胞泛化优势；混合物已知信号可被crDRP检测，输入估计质量能影响该构造任务；pooled指标明显混入药物主效应；泛化任务必须分开。

不支持：分室信息普遍无价值；BayesPrism必然不能/能够修复所有真实数据；两种架构均已充分通过正对照；八分室适用于所有癌谱系；NNLS代理与原临床B/plasma效应等同；当前相关区间跨0即证明方法等效；模拟恢复性能即真实药敏性能。

## 12. 可复核性、环境、剩余需求与停止点

实际Python3.10.16/PyTorch2.0.0 CUDA11.7、RTX4060 8GB、内存约16GB；NumPy1.25.2、pandas2.0.3、scikit-learn1.3.0、XGBoost1.7.6、RDKit2023.3.2、mordredcommunity2.0.7。RDKit等与原声明环境不同，不能声称软件完全相同。R4.5.2及BayesPrism2.2.3已在分析目录内独立安装，Rtools45工具链与所有构建日志保留；没有系统升级或付费资源。

下载、来源与校验、筛选清单保存在data/raw、data/processed和runs/official_download_*；fold的split.csv、preprocessing.pkl、模型检查点、训练日志、预测、bootstrap抽样权重均保留。规范化原始数据、样本及药物ID均可从配置定位。未成功运行保留但不能因目录存在就认为完成，应检查COMPLETE文件和具体scope。
'''+f'\n\n独立审计检查原项目{verify["original_project_files_checked"]}个文件尺寸/时间，变化{len(verify["original_project_size_mtime_changes"])}项；旧分析{verify["prior_analysis_hashes_checked"]}个哈希，变化{len(verify["prior_analysis_hash_mismatches"])}项。复载{verify["replayed_neural_checkpoints"]}个神经网络检查点；当前有效结果最大CPU差{verify["max_CPU_replay_error_current_valid"]:.3g}。历史不稳定运行的失败另外保留，没有用修正后的成功掩盖失败。审计：'+link(V)+'''

目前核心下载不再需要用户操作，也没有账号或付费需求。精确重现仍需原117药物清单、原参考标签/处理矩阵、原划分与检查点；用户已说明没有备份，本次不再次要求寻找。独立图谱、METABRIC跨平台表达/临床验证、完整BayesPrism更新方案、Transformer开发和容量网格仍需后续独立设计与计算，不能标为已完成。

本次停在独立结果报告阶段，供用户决定后续验证优先级；**未经明确确认，不进入改稿、改原图/补充材料/回复信或投稿阶段。**
''')
transfer_verification=ROOT/'runs/transfer_verification_20260926_153034'
tv=json.loads((transfer_verification/'COMPLETE.json').read_text())
add('最终预训练补充另行独立复载'+str(tv['checkpoints_replayed'])+'个目标检查点，最大CPU差'+f'{tv["max_CPU_error"]:.3g}'+ '，全部乳腺来源排除与训练fit IDs检查通过。证据：'+link(transfer_verification))
index={'corrected_core':fix,'supplements':supp,'count_NNLS':'runs/real_gdsc_cell_20260926_145959','count_BayesPrism':str(count),'paired_downstream':str(mixture),'mixture_RNA_strata':'runs/mixture_malignant_RNA_strata_20260926_152245','verification':str(V),'transfer_verification':str(transfer_verification),'report_directory':str(O)}
(O/'运行索引.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'中文独立分析报告.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
# Figures are scientific diagnostics only, saved beside this independent report.
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
colors={'crdrp_full_nnls':'#C55B42','crdrp_bulk':'#386F9B','global_xgb':'#50845C'}
for ax,k in zip(axes,['cell','pair','drug']):
 for m,c in colors.items():
    g=core[k][core[k].method.eq(m)];ax.plot(g.seed.astype(str),g.perdrug_pearson_median,'o-',label=m,color=c)
 ax.axhline(0,color='#888888',lw=.8);ax.set_title(k+' holdout');ax.set_xlabel('Seed');ax.grid(alpha=.15)
axes[0].set_ylabel('Median per-drug OOF Pearson r');axes[2].legend(fontsize=8);fig.suptitle('Real GDSC: 45 cells, 133 drugs, 5,561 pairs; corrected PCA');fig.tight_layout();fig.savefig(O/'真实GDSC三种划分.png',dpi=180);plt.close(fig)
print(O)
