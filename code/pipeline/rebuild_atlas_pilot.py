"""Rebuild donor-by-compartment reference from real public counts, then donor-held-out NNLS pilot.
Author labels are used explicitly. This is not a replay of the original scVI/inferCNV pipeline.
Mixtures are simulated from held-out donor mean profiles, not observed tumor bulk or drug responses.
"""
import sys,json,hashlib,logging
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/packages'))
import h5py,numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.optimize import nnls
from leakage_safe import safe_corr
OUT=ROOT/'runs'/('atlas_nnls_pilot_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir(parents=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
COMP=['malignant_epithelial','normal_epithelial','CAF','PVL','endothelial','T_NK','B_plasma','myeloid']
MAP={'Cancer Epithelial':COMP[0],'Normal Epithelial':COMP[1],'CAFs':COMP[2],'PVL':COMP[3],'Endothelial':COMP[4],'T-cells':COMP[5],'B-cells':COMP[6],'Plasmablasts':COMP[6],'Myeloid':COMP[7]}
config={'input':'historical CELLxGENE asset from original code','annotation':'author celltype_major; B-cells + Plasmablasts merged; T-cells includes NK','not_original_reclustering':True,'minimum_detected_genes':200,'max_mito_fraction':.20,'normalization':'each cell divided by its total raw UMI, then donor/compartment mean','seeds':[7,17,37],'test_donors':7,'minimum_cells_per_test_compartment':20,'mixtures_per_donor':20,'signature_genes':2000,'training_donors_equally_weighted':True,'validation_scope':'donor-held-out simulated mixtures only; no measured drug response'}
(OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
def read_array(node):
    if isinstance(node,h5py.Group):
        categories=node['categories'].asstr()[:] if h5py.check_string_dtype(node['categories'].dtype) else node['categories'][:]
        codes=node['codes'][:];return np.array([categories[c] if c>=0 else None for c in codes])
    return node.asstr()[:] if h5py.check_string_dtype(node.dtype) else node[:]
with h5py.File(ROOT/'data/raw/scrna/wu2021_cellxgene.h5ad','r') as f:
    obs=pd.DataFrame({key:read_array(f['obs'][key]) for key in ['index','donor_id','celltype_major','celltype_minor','normal_cell_call']})
    obs=obs.rename(columns={'index':'cell_id'});obs['compartment']=obs.celltype_major.map(MAP)
    if obs.compartment.isna().any():raise ValueError('Unmapped label')
    gene_ids=read_array(f['raw/var/_index']);symbols=read_array(f['raw/var/feature_name'])
    mito=np.char.startswith(symbols.astype(str),'MT-')
    donors=sorted(obs.donor_id.unique());dmap={d:i for i,d in enumerate(donors)};cmap={c:i for i,c in enumerate(COMP)}
    drows=obs.donor_id.map(dmap).to_numpy();crows=obs.compartment.map(cmap).to_numpy()
    ng=len(gene_ids);sums=np.zeros((len(donors),8,ng),dtype=np.float64);numbers=np.zeros((len(donors),8),dtype=int)
    raw=f['raw/X'];indptr=raw['indptr'][:]
    kept=np.zeros(len(obs),bool);libs=np.zeros(len(obs));detected=np.zeros(len(obs),int);mtfrac=np.zeros(len(obs))
    integral=True;nonnegative=True
    for start in range(0,len(obs),512):
        end=min(start+512,len(obs));a,b=indptr[start],indptr[end];data=raw['data'][a:b]
        integral &= bool(np.all(data==np.rint(data)));nonnegative &= bool(np.all(data>=0))
        x=csr_matrix((data,raw['indices'][a:b],indptr[start:end+1]-a),shape=(end-start,ng))
        lib=np.asarray(x.sum(1)).ravel();nd=np.asarray((x>0).sum(1)).ravel();mt=np.asarray(x[:,mito].sum(1)).ravel()/np.maximum(lib,1)
        good=(lib>0)&(nd>=200)&(mt<=.2);kept[start:end]=good;libs[start:end]=lib;detected[start:end]=nd;mtfrac[start:end]=mt
        x=x.multiply((1/np.maximum(lib,1))[:,None]).tocsr()
        ix=np.arange(start,end)
        for group in np.unique(drows[ix]*8+crows[ix]):
            mask=good&((drows[ix]*8+crows[ix])==group)
            if mask.any():
                d,c=divmod(group,8);sums[d,c]+=np.asarray(x[mask].sum(0)).ravel();numbers[d,c]+=mask.sum()
        if start%10240==0:logging.info('Read %s/%s cells',end,len(obs))
assert integral and nonnegative
obs['raw_library_size']=libs;obs['raw_detected_genes']=detected;obs['raw_mito_fraction']=mtfrac;obs['qc_pass']=kept
obs.to_csv(OUT/'cell_metadata_and_qc.csv',index=False)
pd.DataFrame({'ensembl_id':gene_ids,'symbol':symbols,'mitochondrial':mito}).to_csv(OUT/'gene_ids.csv',index=False)
pd.DataFrame(numbers,index=donors,columns=COMP).rename_axis('donor_id').to_csv(OUT/'donor_compartment_counts.csv')
means=sums/np.maximum(numbers[:,:,None],1)
np.savez_compressed(OUT/'donor_compartment_mean_profiles.npz',means=means,n_cells=numbers,donor_ids=np.array(donors),compartments=np.array(COMP),gene_ids=gene_ids.astype(str),gene_symbols=symbols.astype(str))
qc={'raw_cells':len(obs),'raw_genes':ng,'donors':len(donors),'qc_pass_cells':int(kept.sum()),'qc_fail_cells':int((~kept).sum()),'integer_nonnegative_counts':integral and nonnegative,'authors_major_label_counts':obs.celltype_major.value_counts().to_dict(),'qc_compartment_counts':obs.loc[kept].compartment.value_counts().to_dict()}
(OUT/'qc.json').write_text(json.dumps(qc,indent=2),encoding='utf-8')
predictions=[];split_rows=[];summary=[]
for seed in config['seeds']:
    rng=np.random.default_rng(seed);perm=rng.permutation(len(donors));test=perm[:7];train=perm[7:]
    assert not set(train)&set(test)
    sig=np.stack([means[train[numbers[train,c]>0],c].mean(axis=0) for c in range(8)])
    assert np.isfinite(sig).all()
    # Gene selection depends only on reference donors; mitochondrial genes excluded.
    score=np.var(sig,axis=0);score[mito]=-np.inf;genes=np.argsort(-score,kind='stable')[:2000]
    ref=sig[:,genes].T*1e6
    np.savez_compressed(OUT/f'seed_{seed}_signature.npz',signature_cpm=ref,gene_ids=gene_ids[genes].astype(str),compartments=np.array(COMP),train_donors=np.array(donors)[train],test_donors=np.array(donors)[test])
    for d in range(len(donors)):split_rows.append({'seed':seed,'donor_id':donors[d],'role':'test' if d in test else 'reference'})
    mixtures=[];truth=[];mixture_ids=[]
    for d in test:
        present=numbers[d]>=20
        if present.sum()<2:continue
        for j in range(20):
            weights=np.zeros(8);weights[present]=rng.dirichlet(np.ones(present.sum()))
            mix=weights@means[d,:,genes].T if means[d,:,genes].shape[0]==len(genes) else weights@means[d,:,genes]
            mixture_id=f'WU_DONOR_MEAN_seed{seed}_{donors[d]}_{j:02}'
            pred=nnls(ref,mix*1e6,maxiter=100)[0]
            if pred.sum()<=0:raise ValueError('Empty NNLS estimate')
            pred=pred/pred.sum();mixtures.append(mix);truth.append(weights);mixture_ids.append(mixture_id)
            for c in range(8):predictions.append({'seed':seed,'donor_id':donors[d],'mixture_id':mixture_id,'compartment':COMP[c],'true_fraction':weights[c],'estimated_fraction':pred[c],'present_in_source_donor':bool(present[c])})
    np.savez_compressed(OUT/f'seed_{seed}_heldout_mixtures.npz',mixtures=np.stack(mixtures),truth=np.stack(truth),mixture_ids=np.array(mixture_ids),gene_ids=gene_ids[genes].astype(str))
    logging.info('NNLS donor holdout seed=%s mixtures=%s',seed,len(mixtures))
frame=pd.DataFrame(predictions);frame.to_csv(OUT/'nnls_predictions.csv',index=False);pd.DataFrame(split_rows).to_csv(OUT/'donor_splits.csv',index=False)
for (seed,comp),sub in frame.groupby(['seed','compartment']):
    err=sub.estimated_fraction-sub.true_fraction
    summary.append({'seed':seed,'compartment':comp,'n_mixtures':len(sub),'n_test_donors':sub.donor_id.nunique(),'pearson':safe_corr(sub.true_fraction,sub.estimated_fraction),'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(abs(err))),'bias':float(np.mean(err))})
pd.DataFrame(summary).to_csv(OUT/'nnls_metrics.csv',index=False)
# Cluster bootstrap by donor; repeated synthetic mixtures are not independent biological donors.
boot=[]
for (seed,comp),sub in frame.groupby(['seed','compartment']):
    rng=np.random.default_rng(seed);ds=sub.donor_id.unique();vals=[]
    for b in range(500):
        draw=rng.choice(ds,size=len(ds),replace=True);s=pd.concat([sub[sub.donor_id==d] for d in draw]);vals.append(np.sqrt(np.mean((s.estimated_fraction-s.true_fraction)**2)))
    boot.append({'seed':seed,'compartment':comp,'rmse_ci_low':float(np.percentile(vals,2.5)),'rmse_ci_high':float(np.percentile(vals,97.5)),'resampling_unit':'donor','bootstrap_replicates':500})
pd.DataFrame(boot).to_csv(OUT/'nnls_donor_bootstrap.csv',index=False)
(OUT/'COMPLETE.json').write_text(json.dumps({'status':'completed','scope':'real atlas QC plus donor-held-out simulated-mixture NNLS pilot','not_drug_response_benchmark':True,'total_mixtures':frame.mixture_id.nunique(),'run':str(OUT)},indent=2),encoding='utf-8')
print(json.dumps(qc));print(OUT)
