"""Prespecified malignant-RNA bins on frozen held-out mixtures, never refit models."""
import json
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd
from leakage_safe import metrics,safe_corr
ROOT=Path(__file__).resolve().parents[1];I=ROOT/'runs/paired_deconvolution_inputs_20260926_130450';B=ROOT/'runs/paired_bayesprism_fixed_20260926_142516';P=ROOT/'runs/bayesprism_paired_downstream_20260926_150614';O=ROOT/'runs'/('mixture_malignant_RNA_strata_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir();pred=pd.read_csv(P/'heldout_predictions_all_methods.csv');rows=[];recovery=[];merged=[]
for seed in [7,17,37]:
    s=I/f'seed_{seed}';a=np.load(s/'paired_inputs.npz');mid=pd.read_csv(s/'mixture_ids_and_roles.csv');comp=list(a['compartments']);mal=comp.index('malignant_epithelial');normal=comp.index('normal_epithelial');ids=a['mixture_ids'];bp=pd.read_csv(B/f'seed_{seed}/fractions.csv',index_col=0).loc[ids,comp].to_numpy();true=a['oracle_pi']
    truth=pd.DataFrame({'ModelID':ids,'true_malignant_RNA':true[:,mal],'true_normal_RNA':true[:,normal]});truth['bin']=pd.cut(truth.true_malignant_RNA,[-np.inf,.3,.7,np.inf],labels=['<0.3','0.3-0.7','>=0.7'],right=False).astype(str)
    q=pred[pred.seed.eq(seed)].merge(truth,on='ModelID',validate='many_to_one');merged.append(q)
    for (method,bin),g in q.groupby(['method','bin']):rows.append({'seed':seed,'method':method,'malignant_RNA_bin':bin,'n_mixtures':g.ModelID.nunique(),'n_donors':g.donor_id.nunique(),**metrics(g)})
    test=np.flatnonzero(mid.role.eq('test'))
    for name,est in [('NNLS',a['nnls_pi']),('BayesPrism_fixed',bp)]:
      for bin in ['<0.3','0.3-0.7','>=0.7']:
        ix=test[truth.iloc[test]['bin'].eq(bin).to_numpy()]
        recovery.append({'seed':seed,'method':name,'malignant_RNA_bin':bin,'n_test_mixtures':len(ix),'malignant_MAE':float(np.mean(abs(est[ix,mal]-true[ix,mal]))) if len(ix) else np.nan,'normal_MAE':float(np.mean(abs(est[ix,normal]-true[ix,normal]))) if len(ix) else np.nan,'malignant_bias':float(np.mean(est[ix,mal]-true[ix,mal])) if len(ix) else np.nan,'normal_bias':float(np.mean(est[ix,normal]-true[ix,normal])) if len(ix) else np.nan,'malignant_error_vs_normal_error_r':safe_corr(est[ix,mal]-true[ix,mal],est[ix,normal]-true[ix,normal])})
pd.DataFrame(rows).to_csv(O/'downstream_metrics_by_true_RNA_bin.csv',index=False);pd.DataFrame(recovery).to_csv(O/'fraction_recovery_by_true_RNA_bin.csv',index=False);pd.concat(merged).to_csv(O/'heldout_predictions_with_true_RNA_bins.csv',index=False)
(O/'config.json').write_text(json.dumps({'scope':'exploratory frozen simulated-mixture strata; NOT measured tumor purity and NOT original fixed-purity titration','cutpoints':[.3,.7],'selection':'test mixtures only, all seeds retained, no model refitting','interpretation':'small/empty strata reported; bins selected as descriptive cutpoints, no confirmatory hypothesis tests','predictions_source':str(P)},indent=2));(O/'COMPLETE.json').write_text('{"status":"descriptive_strata_complete"}');print(O)
