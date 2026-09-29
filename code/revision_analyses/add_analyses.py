import os,sys,json,pickle
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
W=Path(__file__).resolve().parent;ROOT=W.parent;R=ROOT/'260925_Re/runs';sys.path.insert(0,str(ROOT/'260925_Re/code'))
import numpy as np,pandas as pd
from scipy.optimize import nnls
from leakage_safe import safe_corr,metrics
from paired_two_way_bootstrap import paired_bootstrap

# Matched donor/drug intervals, including the direct comparison requested.
p=pd.read_csv(R/'bayesprism_paired_downstream_20260926_150614/heldout_predictions_all_methods.csv')
print('mixture methods',p.method.unique(),flush=True)
rows=[]
for s,g in p.groupby('seed'):
 for a in ['crdrp_full_oracle','crdrp_full_bayesprism']:
  # Infer only from unambiguous archived method labels.
  a=next(m for m in g.method.unique() if ('oracle' in m if 'oracle' in a else 'bayes' in m))
  b=next(m for m in g.method.unique() if 'bulk' in m)
  summary,draws,weights=paired_bootstrap(g,a,b,1000,int(s),cell_cluster_column='donor_id');summary['seed']=s;rows.append(summary)
  draws.to_csv(W/f'mixture_bootstrap_{s}_{a}_minus_bulk.csv',index=False)
  np.savez_compressed(W/f'mixture_weights_{s}_{a}.npz',**weights)
pd.concat(rows).to_csv(W/'mixture_effects.csv',index=False)

# Report the existing OOF estimand and its fold-aware companion side by side.
summ=[];drugrows=[]
for name,run in [('cell','real_gdsc_cell_20260926_151327'),('pair','real_gdsc_pair_20260926_151628'),('drug','real_gdsc_drug_20260926_152000')]:
 p=pd.read_csv(R/run/'paired_baseline_predictions.csv')
 for (s,m),g in p.groupby(['seed','method']):
  vals=[];oof=[]
  for d,h in g.groupby('drug_dpc'):
   rr=[];ww=[]
   for f,q in h.groupby('fold'):
    r=safe_corr(q.y_true,q.y_pred)
    if len(q)>3 and np.isfinite(r):rr.append(np.arctanh(np.clip(r,-.999999,.999999)));ww.append(len(q)-3)
   fr=np.tanh(np.average(rr,weights=ww)) if ww else np.nan
   ro=safe_corr(h.y_true,h.y_pred);vals.append(fr);oof.append(ro)
   drugrows.append(dict(analysis=name,seed=s,method=m,drug_dpc=d,within_fold_r=fr,oof_r=ro,valid_folds=len(ww)))
  summ.append(dict(analysis=name,seed=s,method=m,within_fold_median=np.nanmedian(vals) if np.isfinite(vals).any() else np.nan,oof_median=np.nanmedian(oof),n_fold_valid=int(np.isfinite(vals).sum())))
pd.DataFrame(summ).to_csv(W/'fold_metrics.csv',index=False);pd.DataFrame(drugrows).to_csv(W/'fold_drug_metrics.csv',index=False)

# Donor-separated NNLS diagnostics: algebraic reconstruction and pure-profile assignment.
A=R/'atlas_nnls_pilot_20260925_143102';d=np.load(A/'donor_compartment_mean_profiles.npz');means=d['means'];donors=d['donor_ids'];genes=d['gene_ids'];comp=d['compartments'];gi={str(v):i for i,v in enumerate(genes)}
pred=pd.read_csv(A/'nnls_predictions.csv');pure=[];checks=[];combined=[]
for seed in [7,17,37]:
 sig=np.load(A/f'seed_{seed}_signature.npz');S=sig['signature_cpm'];ix=[gi[str(v)] for v in sig['gene_ids']]
 mix=np.load(A/f'seed_{seed}_heldout_mixtures.npz');print('shapes',seed,S.shape,mix['mixtures'].shape,flush=True)
 # Exact recovery of reference-generated mixtures rules out solver/indexing scale error.
 rng=np.random.default_rng(seed);truth=rng.dirichlet(np.ones(8),100);recon=[]
 for w in truth:
  fit=nnls(S,S@w,maxiter=1000)[0];recon.append(fit/fit.sum())
 checks.append(dict(seed=seed,reference_generated_max_absolute_fraction_error=float(np.max(abs(np.array(recon)-truth))),signature_condition_number=float(np.linalg.cond(S))))
 for donor in sig['test_donors']:
  j=list(donors).index(donor)
  for k,c in enumerate(comp):
   if d['n_cells'][j,k]<20:continue
   y=means[j,k,ix]*1e6;fit=nnls(S,y,maxiter=1000)[0];fit=fit/fit.sum()
   for l,cc in enumerate(comp):pure.append(dict(seed=seed,donor_id=donor,true_compartment=c,assigned_compartment=cc,estimated_fraction=fit[l]))
 q=pred[pred.seed.eq(seed)];wide=q.pivot(index=['donor_id','mixture_id'],columns='compartment',values=['true_fraction','estimated_fraction'])
 for label,ks in [('malignant',['malignant_epithelial']),('normal',['normal_epithelial']),('combined_epithelial',['malignant_epithelial','normal_epithelial'])]:
  t=wide['true_fraction'][ks].sum(axis=1);e=wide['estimated_fraction'][ks].sum(axis=1)
  combined.append(dict(seed=seed,compartment=label,pearson=safe_corr(t,e),signed_bias=float(np.mean(e-t)),rmse=float(np.sqrt(np.mean((e-t)**2)))))
pd.DataFrame(checks).to_csv(W/'nnls_algebra_checks.csv',index=False);pd.DataFrame(pure).to_csv(W/'nnls_pure_donor_confusion.csv',index=False);pd.DataFrame(combined).to_csv(W/'nnls_epithelial_errors.csv',index=False)
(W/'diagnostics_COMPLETE.json').write_text(json.dumps({'status':'complete','bootstrap_replicates':1000,'seeds':[7,17,37]},indent=2))
print(pd.DataFrame(combined).to_string(index=False),flush=True)
