"""Repair numerical null-space defect; retain historical runs and unaffected baselines."""
import subprocess,sys,json,hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'runs'/('stable_pca_correction_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir();results={};F=ROOT/'data/processed/expanded_gdsc_features_20260926_142255'
def launch(extra,tag):
    before=set((ROOT/'runs').glob('real_gdsc_*'))
    cmd=[sys.executable,str(ROOT/'code/real_gdsc_benchmark.py'),'--features',str(F),'--stable-pca','--methods','crdrp_full_nnls','transformer_full_nnls']+extra
    with (O/(tag+'.log')).open('w',encoding='utf-8') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(tag+' failed; see log')
    new=[x for x in set((ROOT/'runs').glob('real_gdsc_*'))-before if (x/'COMPLETE.json').exists()];assert len(new)==1
    return new[0]
smoke=launch(['--smoke'],'smoke');results['smoke']=str(smoke)
old={'cell':'real_gdsc_cell_20260926_142546','pair':'real_gdsc_pair_20260926_144248','drug':'real_gdsc_drug_20260926_145210'}
for regime,name in old.items():
    base=O/('unchanged_baselines_'+regime);base.mkdir();P=ROOT/'runs'/name/'oof_predictions.csv';pred=pd.read_csv(P);pred=pred[~pred.method.isin(['crdrp_full_nnls','transformer_full_nnls'])];pred.to_csv(base/'oof_predictions.csv',index=False);(base/'provenance.json').write_text(json.dumps({'source':str(P),'sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'unchanged_bulk_drug_transform':True},indent=2))
    run=launch(['--smoke-run',str(smoke),'--regime',regime,'--baseline-run',str(base)],regime);results[regime]=str(run);(O/'results.json').write_text(json.dumps(results,indent=2));print(regime,run,flush=True)
(O/'COMPLETE.json').write_text(json.dumps({'reason':'PCA nullspace post-scaling amplified BLAS roundoff; fixed solely with training eigenvalues; no response-driven parameter choice','all_seeds_retained':True,'bulk_and_simple_baselines_reused':True,'results':results},indent=2));print(O,flush=True)
