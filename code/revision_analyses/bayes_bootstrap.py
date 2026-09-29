from pathlib import Path
import sys,json
W=Path(__file__).resolve().parent;ROOT=W.parent;R=ROOT/'260925_Re/runs';B=W/'bayesprism_updated';sys.path.insert(0,str(ROOT/'260925_Re/code'))
import pandas as pd
from paired_two_way_bootstrap import paired_bootstrap
run=R/'real_gdsc_cell_20260929_170425';assert (run/'COMPLETE.json').exists()
(W/'updated_bayes_run.txt').write_text(str(run),encoding='utf-8')
baseline=pd.read_csv(R/'real_gdsc_cell_20260926_150657/paired_baseline_predictions.csv');new=pd.read_csv(run/'oof_predictions.csv');both=pd.concat([baseline,new]);both.to_csv(B/'paired_predictions.csv',index=False);out=[]
for seed,g in both.groupby('seed'):
 for b in ['crdrp_full_bayesprism_fixed','crdrp_full_nnls','crdrp_bulk']:
  su,dr,we=paired_bootstrap(g,'crdrp_full_bayesprism_updated',b,1000,int(seed));su['seed']=seed;out.append(su);dr.to_csv(B/f'bootstrap_{seed}_minus_{b}.csv',index=False)
pd.concat(out).to_csv(B/'paired_effects.csv',index=False);(B/'COMPLETE.txt').write_text('samplewise deconvolution and downstream validation complete')
