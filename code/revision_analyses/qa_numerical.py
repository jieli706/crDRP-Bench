from pathlib import Path
import pandas as pd,numpy as np,json
W=Path(__file__).resolve().parent;ROOT=W.parent;R=ROOT/'260925_Re/runs'
checks={}
for task,summary in [('mixture',W/'mixture_effects.csv'),('updated_bayes',W/'bayesprism_updated/paired_effects.csv')]:
 q=pd.read_csv(summary)
 for _,r in q.iterrows():
  f=W/f'mixture_bootstrap_{r.seed}_{r.method_a}_minus_bulk.csv' if task=='mixture' else W/f'bayesprism_updated/bootstrap_{r.seed}_minus_{r.method_b}.csv'
  draws=pd.read_csv(f)[r.metric].dropna();lo,hi=np.percentile(draws,[2.5,97.5]);assert np.allclose([lo,hi],[r.ci_low,r.ci_high],rtol=0,atol=1e-10),(task,r.seed,r.metric)
 checks[task+'_saved_interval_rows_checked']=len(q)
for task,folder,file in [('updated_bayes',R/'real_gdsc_cell_20260929_170425','oof_predictions.csv'),('carcinoma',R/'carcinoma_transfer_20260929_165047','oof_predictions.csv'),('xgb',W/'xgb','predictions.csv')]:
 if not (folder/file).exists():
  options=list(folder.glob('*prediction*.csv'));assert len(options)==1,(task,options);file=options[0].name
 p=pd.read_csv(folder/file);m=pd.read_csv(folder/'metrics.csv')
 if task=='xgb':p[['y_true','y_pred']]=p[['y_true','y_pred']].astype('float32') # Restore the generating in-memory dtype from decimal CSV.
 for (seed,method),g in p.groupby(['seed','method']):
  r=[]
  for drug,h in g.groupby('drug_dpc'):
   if len(h)>2 and np.std(h.y_true)>1e-10 and np.std(h.y_pred)>1e-10:r.append(h.y_true.corr(h.y_pred))
  expected=m[(m.seed==seed)&m.method.eq(method)];assert len(expected)==1
  assert np.isclose(np.nanmedian(r),expected.iloc[0].perdrug_pearson_median,atol=1e-10),(task,seed,method)
 checks[task+'_metrics_recomputed']=len(m)
(W/'qa_numerical.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
