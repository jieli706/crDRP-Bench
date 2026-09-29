from pathlib import Path
import sys,json,hashlib
W=Path(__file__).resolve().parent;ROOT=W.parent;sys.path.insert(0,str(ROOT/'260925_Re/code'))
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import nnls
from paired_two_way_bootstrap import paired_bootstrap
O=W/'clinical';t=pd.read_csv(ROOT/'260925_Re/data/processed/tcga_nnls_survival_20260926_145356/patient_compartment_survival.csv');t['sample16']=t['sample'].str[:16]
a=pd.read_excel(O/'aran.xlsx',header=3);a=a[a['Cancer type'].eq('BRCA')].copy();a['sample16']=a['Sample ID'].str[:16];assert a.sample16.is_unique
q=t.merge(a[['sample16','CPE']],on='sample16',how='inner',validate='one_to_one');q.CPE=pd.to_numeric(q.CPE,errors='coerce');q=q.dropna(subset=['CPE','malignant_epithelial']);q.to_csv(O/'CPE_matched.csv',index=False)
rows=pd.read_csv(O/'purity_validation.csv').to_dict('records')
for name,x in [('malignant_epithelial',q.malignant_epithelial),('combined_epithelial',q.malignant_epithelial+q.normal_epithelial)]:
 rng=np.random.default_rng(20260929);b=[]
 for j in range(2000):
  ix=rng.integers(0,len(x),len(x));b.append(spearmanr(x.iloc[ix],q.CPE.iloc[ix]).statistic)
 rows.append(dict(exposure=name,comparator='CPE',n=len(q),spearman=spearmanr(x,q.CPE).statistic,pearson=pearsonr(x,q.CPE).statistic,ci_low=np.quantile(b,.025),ci_high=np.quantile(b,.975)))
pd.DataFrame(rows).to_csv(O/'purity_validation.csv',index=False)
sources=json.loads((O/'sources.json').read_text());sources=[s for s in sources if s['file']!='aran.xlsx'];sources.append(dict(file='aran.xlsx',url='https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fncomms9971/MediaObjects/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx',sha256=hashlib.sha256((O/'aran.xlsx').read_bytes()).hexdigest(),bytes=(O/'aran.xlsx').stat().st_size));(O/'sources.json').write_text(json.dumps(sources,indent=2))

# Reconstruct each saved donor mixture without NumPy advanced-index ambiguity.
A=ROOT/'260925_Re/runs/atlas_nnls_pilot_20260925_143102';d=np.load(A/'donor_compartment_mean_profiles.npz');geneidx={str(g):i for i,g in enumerate(d['gene_ids'])};pr=pd.read_csv(A/'nnls_predictions.csv');audit=[]
for seed in [7,17,37]:
 m=np.load(A/f'seed_{seed}_heldout_mixtures.npz');s=np.load(A/f'seed_{seed}_signature.npz');ix=[geneidx[str(g)] for g in m['gene_ids']];mixerr=[];fractionerr=[];same=[]
 for i,mid in enumerate(m['mixture_ids']):
  pp=pr[pr.mixture_id.eq(mid)];donor=pp.donor_id.iloc[0];j=list(d['donor_ids']).index(donor);D=d['means'][j][:,ix].T*1e6
  x=D@m['truth'][i];mixerr.append(np.max(abs(x-m['mixtures'][i]*1e6)))
  e=nnls(s['signature_cpm'],x,maxiter=1000)[0];e/=e.sum();expected=pp.set_index('compartment').loc[d['compartments'],'estimated_fraction'].to_numpy();fractionerr.append(np.max(abs(e-expected)))
  v=nnls(D,x,maxiter=1000)[0];v/=v.sum();same.append(np.max(abs(v-m['truth'][i])))
 audit.append(dict(seed=seed,mixture_cpm_max_abs_error=max(mixerr),stored_NNLS_fraction_max_abs_error=max(fractionerr),donor_matched_oracle_max_abs_error=max(same)))
pd.DataFrame(audit).to_csv(W/'nnls_saved_mixture_checks.csv',index=False)

# New crossed bootstrap, identical model budget and split definitions.
p=pd.read_csv(W/'xgb/predictions.csv');out=[]
for seed,g in p.groupby('seed'):
 su,dr,we=paired_bootstrap(g,'compartment_xgb','bulk_xgb_pca116',1000,int(seed));su['seed']=seed;out.append(su);dr.to_csv(W/f'xgb/bootstrap_{seed}.csv',index=False)
pd.concat(out).to_csv(W/'xgb/paired_effects.csv',index=False)

I=ROOT/'260925_Re/data/processed/transfer_inputs_20260926_142944';meta=pd.read_csv(I/'sample_ids.csv');pi=np.load(I/'pi.npy');meta['malignant_RNA']=pi[:,0];meta['CAF_RNA']=pi[:,2];meta.to_csv(W/'source_lineage_contributions.csv',index=False);meta.groupby(['role','OncotreeLineage']).malignant_RNA.agg(['size','median','min','max']).reset_index().to_csv(W/'source_lineage_summary.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(audit).to_string(index=False))
