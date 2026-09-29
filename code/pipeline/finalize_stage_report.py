"""Verify saved results, produce diagnostics and an independent Chinese stage report."""
import sys,os,json,hashlib,pickle,importlib.util,importlib.metadata as md
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/packages'));os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/matplotlib_config')
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from leakage_safe import FoldTransform
OUT=ROOT/'reports';OUT.mkdir(exist_ok=True)
def latest(prefix):
    return sorted(p for p in (ROOT/'runs').glob(prefix+'*') if (p/'COMPLETE.json').exists())[-1]
smoke=latest('synthetic_smoke_');atlas=latest('atlas_nnls_pilot_')
qc=json.loads((atlas/'qc.json').read_text());counts=json.loads((smoke/'architecture_audit.json').read_text());pc=json.loads((smoke/'positive_control.json').read_text())
metrics=pd.read_csv(atlas/'nnls_metrics.csv');ci=pd.read_csv(atlas/'nnls_donor_bootstrap.csv');pred=pd.read_csv(atlas/'nnls_predictions.csv')
metrics.merge(ci,on=['seed','compartment']).to_csv(OUT/'NNLS_供者隔离结果.csv',index=False,encoding='utf-8-sig')

# Saved neural checkpoint -> saved fold transforms -> predictions must replay.
spec=importlib.util.spec_from_file_location('arch',ROOT/'source_snapshot/scripts/07_crdrp_model.py');arch=importlib.util.module_from_spec(spec);spec.loader.exec_module(arch)
fold=smoke/'seed_7/pair/fold_0'
with (fold/'preprocessing.pkl').open('rb') as f:trans=pickle.load(f)
with (fold/'full_finetuned.pt').open('rb') as f:ck=torch.load(f,map_location='cpu',weights_only=False)
data=np.load(smoke/'synthetic_data.npz');pairs=pd.read_csv(smoke/'synthetic_pairs.csv');saved=pd.read_csv(smoke/'predictions.csv')
chosen=saved[(saved.seed==7)&(saved.regime=='pair')&(saved.fold==0)&(saved.method=='full')]
p=pairs.set_index('pair_id').loc[chosen.pair_id];c=p.cell_row.to_numpy();d=p.drug_row.to_numpy()
zf=np.stack([trans['Z'][j].transform(data['Z'][:,j]) for j in range(8)],axis=1);df=trans['drug'].transform(data['drug'])
model=arch.CrDRP(z_dim=zf.shape[2],drug_dim=df.shape[1],hidden=16,emb_dim=16,dropout=.1);model.load_state_dict(ck['state_dict']);model.eval()
with torch.no_grad():yp=model(torch.from_numpy(data['pi'][c]),torch.from_numpy(zf[c]),torch.from_numpy(df[d]))[0].numpy()
delta=float(abs(yp-chosen.y_pred.to_numpy()).max());assert delta<1e-6
mix=np.load(atlas/'seed_7_heldout_mixtures.npz');signature=np.load(atlas/'seed_7_signature.npz')
donors=np.load(atlas/'donor_compartment_mean_profiles.npz');lookup={x:i for i,x in enumerate(donors['gene_ids'])};gidx=np.array([lookup[x] for x in mix['gene_ids']])
first=pred[pred.mixture_id==mix['mixture_ids'][0]];donor=first.donor_id.iloc[0];di=list(donors['donor_ids']).index(donor)
reconstructed=mix['truth'][0]@donors['means'][di][:,gidx]
assert np.allclose(reconstructed,mix['mixtures'][0],atol=1e-12)

# Read-only protection check: compare preexisting file size and mtime; code also checked by SHA256.
inventory=pd.read_csv(ROOT/'audit/file_inventory.csv');changed=[];project=ROOT.parents[1]
for r in inventory.itertuples():
    path=project/r.path
    if not path.exists():changed.append({'path':r.path,'reason':'missing'});continue
    s=path.stat()
    if s.st_size!=r.bytes or s.st_mtime_ns!=r.mtime_ns:changed.append({'path':r.path,'reason':'size_or_mtime_changed'})
verify={'saved_checkpoint_prediction_max_abs_difference':delta,'heldout_mixture_reconstruction_match':True,'preexisting_files_changed_by_size_or_mtime':changed,'preexisting_files_checked':len(inventory),'synthetic_oof_rows':len(saved),'atlas_mixtures':pred.mixture_id.nunique()}
(ROOT/'audit/final_verification.json').write_text(json.dumps(verify,indent=2,ensure_ascii=False),encoding='utf-8')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
order=['malignant_epithelial','normal_epithelial','CAF','PVL','endothelial','T_NK','B_plasma','myeloid']
for seed,color in zip([7,17,37],['#0072B2','#D55E00','#009E73']):
    m=metrics[metrics.seed==seed].set_index('compartment').loc[order]
    axes[0].plot(m.pearson,np.arange(8),marker='o',label=f'Seed {seed}',color=color)
    axes[1].plot(m.rmse,np.arange(8),marker='o',label=f'Seed {seed}',color=color)
for ax in axes:ax.set_yticks(np.arange(8),order);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend()
axes[0].set(xlabel='Pearson r',xlim=(-1,1),title='Held-out donor mixtures: NNLS correlation');axes[0].axvline(0,color='gray',lw=.8)
axes[1].set(xlabel='Fraction RMSE',title='NNLS error (lower is better)')
fig.suptitle('Pilot only: author-label reference, 7 held-out donors / seed, 140 simulated mixtures / seed',fontsize=11)
fig.savefig(OUT/'NNLS_donor_holdout_diagnostic.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
for col,seed in enumerate([7,17,37]):
 for row,comp in enumerate(order[:2]):
    ax=axes[row,col];s=pred[(pred.seed==seed)&(pred.compartment==comp)];ax.scatter(s.true_fraction,s.estimated_fraction,s=10,alpha=.6)
    ax.plot([0,1],[0,1],color='gray',ls='--');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Known simulated fraction',ylabel='NNLS estimate',title=f'{comp}\nseed {seed}')
fig.suptitle('Independent donors, simulated mixtures; not measured patient purity',fontsize=12)
fig.savefig(OUT/'NNLS_epithelial_scatter.png',dpi=180);plt.close(fig)

# Version comparison: same script names across historical copies, without declaring a source authoritative.
hashes=pd.read_csv(ROOT/'audit/code_hashes.csv');hashes['basename']=hashes.path.str.replace('\\','/').str.rsplit('/',n=1).str[-1]
version=hashes.groupby('basename').agg(copies=('path','size'),distinct_hashes=('sha256','nunique')).reset_index()
version.to_csv(ROOT/'audit/code_version_comparison.csv',index=False,encoding='utf-8-sig')
versions={}
for name in ['numpy','pandas','scipy','scikit-learn','torch','h5py','matplotlib','contourpy','cycler','fonttools','kiwisolver']:
    versions[name]=md.version(name)
(ROOT/'audit/analysis_runtime_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')

table='\n'.join(f"| {r.seed} | {r.compartment} | {r.pearson:.3f} | {r.rmse:.3f} | {r.bias:.3f} |" for r in metrics.itertuples() if r.compartment in order[:2])
report=f'''# 数据重建、环境核查与小规模分析报告（阶段报告）

日期：2026-09-25。所有新增内容均位于 `260925_Re`。**核心真实药敏 full/bulk 重跑尚未完成，不能把本报告当作完整返修分析结论。** 未修改任何稿件、原图、补充材料或审稿回复信。用户已确认没有原服务器/备份，后续必须按公开来源重建。

## 一、已完成与未完成

已核查父项目 1,673 个文件、217 份代码文件及 ZIP 中 132 个条目；保存原文件清单、代码 SHA-256、来源代码快照及同名脚本版本比较。历史所列 82 项待分发数据在本项目检索范围中全部缺失；代码可读不等于数据和结果可以复现。检索范围未扩展为整台机器扫描。

已下载并校验 5 个公开文件：Wu 单细胞历史 H5AD、TCGA 临床与生存表、METABRIC 患者及样本临床表。共约 845 MB；逐文件来源 URL、时间、服务器元数据和 SHA-256 见 `audit/download_manifest.json`。Wu 文件的校验和为 `04245d73cdcc0b473aa88ec655d1bcb051dbbd5f9a3d1aec4c2f37b1f0645df4`。没有把 HTTP 200 的 DepMap 人机验证 HTML 误存成 CSV。

已完成两类不同的小规模工作：①合成数据训练/评估管线测试；②真实公开单细胞表达重建及供者隔离的模拟混合物 NNLS 试验。**真实 GDSC 药敏的端到端测试仍未完成，因此未启动完整药敏分析。**

## 二、计算环境与版本

本机 Windows，16 个逻辑 CPU，物理内存约 15.85 GiB（标称 16 GB）；核查时可用约 2.9–3.8 GiB。GPU 为 NVIDIA RTX 4060，8,188 MiB 显存，驱动 560.94。E 盘初查约 35.9 GiB 空余。实际 Python 3.11.5、PyTorch 2.0.1+cpu，`torch.cuda.is_available()` 为 false；显卡存在不代表当前 Python 可以用 CUDA。

原仓库声明 Python 3.10/R 4.5.2/PyTorch 2.12.0 等，但没有原运行锁定文件或检查点可核对，不能认定本机与原环境一致。当前环境缺少 pyarrow、anndata、scanpy、XGBoost、Mordred 等；未在 PATH 和检查的常见安装位置发现可用 R/Rscript。为分块读数据和画诊断图，新增 h5py 3.11.0、matplotlib 3.8.4 等仅安装在本分析目录 `runtime/packages`，没有修改系统环境；完整实际版本见 `audit/analysis_runtime_versions.json`。

Anaconda 的 HTTPS SSL DLL 在本次启动环境中无法加载，已用本机 Python 3.9 的标准 HTTPS 下载器完成公开数据传输，数据处理继续使用 3.11。未关闭 TLS 验证。PyTorch 2.0.1 的中文路径直接保存失败，改用 Python 文件句柄保存检查点后通过重放验证。

## 三、代码核查和泄漏修复状态

| 问题 | 原代码证据（source_snapshot/scripts 内） | 当前处理与限制 |
|---|---|---|
| 主预训练包含乳腺目标样本 | 06e 构造全部 pan-cancer pairs；07a、07b 读取并复用该预训练 | 新 guard 排除全部目标乳腺细胞；已在故意混入目标样本的合成来源上验证，真实数据待接入 |
| 标为 leakfree 仍沿用全体特征基底 | 07d 的 leakfree_pairs 只过滤响应；06e 的 Z PCA、07b 的 bulk PCA 在全部 pan-cancer 上拟合 | 新 FoldTransform 将缺失值填补、基因选择、scaler、PCA 均限制于允许的训练实体，并归档 fit IDs；不复用旧 PCA |
| 药物留出的预训练标签污染 | 07b/08b 在各外层 fold 外建立或缓存同一个 pan-cancer 预训练模型 | 新过滤同时排除外层测试和内层验证药物；原 drug-holdout 不能直接解释为严格未见药物 |
| 药物特征的测试分布参与 | 06 baseline、06c 等全体药物 PCA；05 的全局描述符填补 | 新药物变换仅训练药物拟合；固定化学指纹本身可逐分子计算，但填补/缩放/PCA 需在划分内拟合 |
| 内层验证不遵循外层任务 | 07b 的 by_cells=False 为随机 pair 验证 | 新 cell/drug 任务内层同样按 cell/drug 分组；合成测试检查 train/val/test 不交叉 |
| 共用解卷积参考更新可能构成转导设置 | 06b 在同一 mixture 中运行全部细胞，BayesPrism 迭代估计参考 | 严格归纳评估需要冻结外部参考或逐划分训练/测试处理；目前未完成 BayesPrism 修复与运行 |
| L1 注意力正则是常数 | 07_crdrp_model 的 alpha=softmax，loss=Huber+lambda*sum(abs(alpha)) | 默认 alpha 非负且和为 1，实测 L1=1；新训练测试使用 Huber+AdamW，不宣称该项能稀疏化注意力 |
| 容量匹配缺乏可核对定义 | 默认 CrDRP / BulkDRP 参数配置 | 实测 full={counts['original_default_full']:,}、bulk={counts['original_default_bulk']:,}，相差 3,208（相对 bulk 约 8.9%）；输入维度也不同，不能只凭“matched”注释认定严格匹配 |

这些修复是在**新模块/试验管线**中实现，并未覆盖重写原代码，也未证明完整真实数据链路已无泄漏。真实运行前还需落实基因重复映射、药物化学结构连接、数据版本与去重等检查。后续主终点预先定义为逐药物 OOF Pearson 的中位数，pooled correlation 仅作诊断，不混用二者。

## 四、合成管线测试结果（不作为生物学证据）

目录：`runs/{smoke.name}`。36 个合成目标细胞、18 个额外来源细胞、12 个合成药物、8 个 compartment；seed 7/17/37，pair/cell/drug 三类任务，各 3 折，合计 27 个划分。full、bulk、Ridge 和训练药物均值均保存逐 pair OOF 预测（共 {len(saved):,} 行）、划分和配置；神经网络保存预训练和微调检查点。短训练使用小网络和固定预算，只用于流程检查，没有挑选测试集最优结果。

30 项记录包含分割排他性、预训练目标排除、测试特征污染不改变训练变换、重复 pair 拒绝及可识别 NNLS 单元测试。复载保存的 full 检查点和预处理后，与原逐样本预测最大差为 {delta:.2g}。已知 compartment 信号的独立合成 oracle Ridge 正对照 r={pc['oracle_r']:.4f}、RMSE={pc['oracle_rmse']:.4f}；无信息 bulk 均值 RMSE={pc['bulk_mean_rmse']:.4f}；训练标签打乱对照 r={pc['shuffled_train_label_r']:.3f}。这仅证明人为设定的信号可被 oracle 读出，**尚不能视为审稿人要求的 full 模型正对照已完成**。

## 五、真实单细胞数据重建与供者隔离 NNLS 试验

目录：`runs/{atlas.name}`。输入含 {qc['raw_cells']:,} 个细胞、{qc['raw_genes']:,} 个基因、{qc['donors']} 位供者。原始矩阵通过非负整数 UMI 检查。使用每细胞检测基因≥200、线粒体 UMI 比例≤20% 的预先规则，保留 {qc['qc_pass_cells']:,}、排除 {qc['qc_fail_cells']} 个细胞；细胞 ID、供者 ID、标签、QC 数值和全部基因 ID 均已保存。

此次使用数据作者的 `celltype_major` 标签，将 B-cells/Plasmablasts 合并为 B_plasma，其余映射到 8 类；**没有重跑原 scVI/inferCNV 聚类，因此是明确记录的新参考版本，不是原稿参考的精确复现。** 每细胞 UMI 总量归一化后求供者内 compartment 均值，再对训练供者等权构建参考。这里的混合系数指等文库归一化表达贡献；不能直接声称是肿瘤切片中的细胞纯度。

每个种子随机留出 7 位供者，其余 19 位构建参考；只用参考选择 2,000 个高 compartment 变异非线粒体基因。每位留出供者生成 20 个 Dirichlet 混合物（仅使用该供者≥20细胞的 compartment），每种子 140，共 420。NNLS 后归一化到和为 1。供者隔离，测试混合物不参与参考或基因选择。不同种子会共享供者，420 个模拟混合物**不是 420 个独立患者**。此设计不是原稿的癌细胞系加 TME、固定纯度梯度设计。

| seed | compartment | Pearson r | RMSE | 平均偏差（估计−真值） |
|---|---|---|---|---|
{table}

全部 8 类、各 seed 的指标与 500 次供者聚类 bootstrap 的 RMSE 95% 区间见 `NNLS_供者隔离结果.csv`。例如 seed 7 恶性上皮 RMSE 的区间约 0.145–0.243。每个划分仅 7 位测试供者，区间仅用于探索，不足以给出稳定总体推断。三次划分中的恶性/正常上皮恢复表现差异大，提示参考的供者适用性需要进一步检查；不能将问题归因于一个已被验证的单一机制，也不能推断 BayesPrism 的表现。

![供者隔离 NNLS 诊断](NNLS_donor_holdout_diagnostic.png)

![上皮比例诊断](NNLS_epithelial_scatter.png)

## 六、与原稿数字的关系及可支持的结论

| 原稿/附件引用 | 本次状态与解释 |
|---|---|
| full/bulk 约 0.22/0.29 | 真实药敏数据未取得，未重现、未验证，也没有替代数值 |
| leakfree 0.204/0.244 与 GlobalXGB 0.30 | 原 07d 代码指向 rawmal/bulk 的 pair-holdout；与全文 full 配置并不自动同义；完整预处理仍需修复，不能直接视为同一无泄漏主比较 |
| Transformer 0.19/0.35 | 未运行真实重分析；原共享预训练与特征处理也须按划分重建 |
| NNLS 全局 r≈0.78、BayesPrism r≈0.99 | 本次是不同供者、不同混合物定义及新参考，数值不可直接相减或用来宣称复现/推翻；BayesPrism 未运行 |
| 临床生存与免疫亚型结论 | 仅下载临床表，不构成复现；未估计新的 HR、P 值或 q 值 |

目前支持：本地无法靠现存文件精确重放；原代码中存在具体可证实的预训练/预处理信息使用问题；新拆分和存档管线通过小规模合成测试；基于真实单细胞的供者隔离 NNLS 试验在上皮区分上存在明显不稳定性。

目前不支持：判定 compartment-aware 在真实药敏任务中优于或劣于 bulk；将无显著差异等同于等效；宣称 BayesPrism 能解决问题或普遍失败；将模拟混合物结果推广到临床用药；确认原稿生存结论。没有按原稿目标调参、筛选种子或重构缺失响应。

## 七、缺口与执行边界

核心限制是 DepMap 入口需要人机验证，尚未获得指定版本的表达、ModelID 和 GDSC 响应文件。具体文件及优先级见同目录 `待获取数据与执行顺序.md`。当前没有付费或云账号需求；若用户完成正常浏览器访问并下载前六个核心文件，即可接续真实小规模测试。无需寻找已经确认不存在的原备份。

R/BayesPrism、CUDA Python、完整依赖和大数据预处理仍需准备，但不应为当前受阻的核心响应下载而盲目安装大体积环境或启动多进程。原 R 脚本默认 24/48 核并将 sparse 矩阵密集化，不适合直接照搬到本机当前可用内存。后续应在本目录建可锁定环境，优先限制线程、稀疏/分块读取，先实测峰值内存。

现有文件核查：对起始清单的 {len(inventory):,} 个文件检查尺寸与修改时间，发现变化 {len(changed)} 项，详见 `audit/final_verification.json`。所有本次写入、安装包缓存与诊断图均在分析目录内。**在结果报告阶段停止，不修改任何投稿材料。**

公开数据来源：[Wu CELLxGENE 集合](https://cellxgene.cziscience.com/collections/dea97145-f712-431c-a223-6b5f565f362a)、[GEO GSE176078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078)、[DepMap 官方 26Q1 发布说明](https://forum.depmap.org/t/announcing-the-26q1-release/4606)、[DepMap 下载入口](https://depmap.org/portal/download/all/)。原代码记载版本与本次实际取得文件的逐项证据另存 audit，不能用“最新”替代版本号。
'''
(OUT/'中文阶段分析报告.md').write_text(report,encoding='utf-8')
print(json.dumps(verify,ensure_ascii=False));print(OUT/'中文阶段分析报告.md')
