from pathlib import Path
import time,sys,json,shutil,subprocess,hashlib
import numpy as np,pandas as pd
W=Path(__file__).resolve().parent;ROOT=W.parent;D=ROOT/'260925_Re';R=D/'runs';B=W/'bayesprism_updated';sys.path.insert(0,str(D/'code'))
from paired_two_way_bootstrap import paired_bootstrap
for attempt in range(900):
 if all((B/f'COMPLETE_part{i}.txt').exists() for i in range(4)):break
 time.sleep(2)
else:raise RuntimeError('BayesPrism partitions did not complete')
S=D/'data/processed/real_count_deconvolution_20260926_142741';oldF=D/'data/processed/expanded_gdsc_features_20260926_142255';F=D/'data/processed/count_features_bayesprism_updated_20260929';F.mkdir(exist_ok=True)
d=np.load(S/'real_count_inputs.npz');ids=d['cell_ids'];genes=d['gene_ids'];comp=d['compartments'];counts=d['counts']
theta=pd.concat([pd.read_csv(B/f'fractions_part{i}.csv',index_col=0) for i in range(4)]).loc[ids,comp];theta.to_csv(B/'fractions.csv');zs=[]
for k in comp:
 z=pd.concat([pd.read_csv(B/f'Z_{k}_part{i}.csv.gz',index_col=0) for i in range(4)]).loc[ids,genes];z.to_csv(B/f'Z_{k}.csv.gz');zs.append(z.to_numpy())
alloc=np.stack(zs,axis=1);error=float(abs(alloc.sum(1)-counts).max());assert error<1e-3
mass=alloc.sum(2,keepdims=True);norm=np.divide(alloc,mass,out=np.zeros_like(alloc),where=mass>0)
np.save(F/'deconvolution_pi.npy',theta.to_numpy(dtype='float32'));np.save(F/'deconvolution_Z_log1pCPM.npy',np.log1p(norm*1e6).astype('float32'));np.save(F/'bulk_log1pCPM.npy',np.log1p(counts/counts.sum(1,keepdims=True)*1e6).astype('float32'))
for name in ['sample_ids.csv','drug_ids.csv','drug_raw_features.npy','drug_feature_names.csv','response_pairs.csv']:shutil.copy2(oldF/name,F/name)
cfg={'input_variant':'BayesPrism samplewise reference update; final type fractions and initial posterior Z returned by get.exp','original_canonical_pipeline':True,'pi_array':'deconvolution_pi.npy','Z_array':'deconvolution_Z_log1pCPM.npy','bulk_array':'bulk_log1pCPM.npy','n_cells':45,'n_genes':1896,'count_reconstruction_max_error':error,'cross_sample_reference_update':False,'gibbs_chain':1000,'burn':500,'thin':2,'algorithm_seed':7}
(F/'config.json').write_text(json.dumps(cfg,indent=2));(F/'source_hashes.json').write_text(json.dumps([{'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [W/'canonical_bayes.R',B/'fractions.csv',S/'config.json']],indent=2));(F/'COMPLETE.json').write_text(json.dumps({'status':'ready'}))
args=[sys.executable,str(D/'code/real_gdsc_benchmark.py'),'--features',str(F),'--shared-z','--stable-pca','--methods','crdrp_full_bayesprism_updated']
subprocess.run(args+['--smoke'],check=True)
smoke=sorted(p for p in R.glob('real_gdsc_smoke_*') if (p/'COMPLETE.json').exists() and json.loads((p/'config.json').read_text()).get('input_directory')==str(F))[-1];subprocess.run(args+['--smoke-run',str(smoke)],check=True)
run=sorted(p for p in R.glob('real_gdsc_cell_*') if (p/'COMPLETE.json').exists() and json.loads((p/'config.json').read_text()).get('input_directory')==str(F))[-1];(W/'updated_bayes_run.txt').write_text(str(run),encoding='utf-8')
baseline=pd.read_csv(R/'real_gdsc_cell_20260926_150657/paired_baseline_predictions.csv');new=pd.read_csv(run/'oof_predictions.csv');both=pd.concat([baseline,new]);both.to_csv(B/'paired_predictions.csv',index=False);out=[]
for seed,g in both.groupby('seed'):
 for b in ['crdrp_full_bayesprism_fixed','crdrp_full_nnls','crdrp_bulk']:
  su,dr,we=paired_bootstrap(g,'crdrp_full_bayesprism_updated',b,1000,int(seed));su['seed']=seed;out.append(su);dr.to_csv(B/f'bootstrap_{seed}_minus_{b}.csv',index=False)
pd.concat(out).to_csv(B/'paired_effects.csv',index=False);(B/'COMPLETE.txt').write_text('samplewise deconvolution and downstream validation complete')
