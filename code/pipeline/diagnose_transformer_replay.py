"""Read-only checkpoint diagnostic for CPU/CUDA inference discrepancy."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import importlib.util,pickle,json
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch
import leakage_safe
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'runs/real_gdsc_cell_20260926_142546';F=ROOT/'data/processed/expanded_gdsc_features_20260926_142255';O=ROOT/'runs'/('transformer_replay_diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir()
s=importlib.util.spec_from_file_location('tf',ROOT/'source_snapshot/scripts/08_transformer_model.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);torch.set_num_threads(2)
P=B/'seed_7/fold_0';pred=pd.read_csv(P/'transformer_full_nnls_predictions.csv');pairs=pd.read_csv(B/'selected_response_pairs.csv');q=pred.merge(pairs[['pair_id','cell_row','drug_row']],on='pair_id',validate='one_to_one');ci=q.cell_row.to_numpy();di=q.drug_row.to_numpy()
with (P/'preprocessing.pkl').open('rb') as f:t=pickle.load(f)
with (P/'transformer_full_nnls.pt').open('rb') as f:ck=torch.load(f,map_location='cpu')
z=np.load(F/'nnls_Z_log1pCPM.npy');z=np.stack([t['Z'][k].transform(z[:,k]) for k in range(8)],axis=1)[ci];pi=np.load(F/'nnls_pi.npy')[ci];dr=t['drug'].transform(np.load(F/'drug_raw_features.npy'))[di];checks=[]
for device in ['cpu','cuda']:
 for mode in ['no_grad_fused','grad_enabled_unfused']:
  model=m.TransformerDRP(z_dim=ck['z_dim'],drug_dim=ck['drug_dim']).to(device);model.load_state_dict(ck['state_dict']);model.eval();args=[torch.tensor(x,device=device) for x in [pi,z,dr]]
  with torch.set_grad_enabled(mode=='grad_enabled_unfused'):yp=model(*args).detach().cpu().numpy()
  q[device+'_'+mode]=yp;checks.append({'device':device,'mode':mode,'max_error':float(np.max(abs(yp-q.y_pred))),'rmse_difference':float(np.sqrt(np.mean((yp-q.y_pred)**2)))})
from threadpoolctl import threadpool_limits,threadpool_info
for n in [1,2,4,8,16,32]:
 with threadpool_limits(limits=n,user_api='blas'):
  zz=np.load(F/'nnls_Z_log1pCPM.npy');zz=np.stack([t['Z'][k].transform(zz[:,k]) for k in range(8)],axis=1)[ci];dd=t['drug'].transform(np.load(F/'drug_raw_features.npy'))[di]
  with torch.no_grad():yp=model(torch.tensor(pi,device='cuda'),torch.tensor(zz,device='cuda'),torch.tensor(dd,device='cuda')).cpu().numpy()
  checks.append({'blas_threads':n,'z_max_diff':float(np.max(abs(zz-z))),'drug_max_diff':float(np.max(abs(dd-dr))),'max_error':float(np.max(abs(yp-q.y_pred)))})
q.to_csv(O/'predictions.csv',index=False);(O/'checks.json').write_text(json.dumps({'torch':torch.__version__,'threadpools':threadpool_info(),'checks':checks,'post_scale_min':[float(min(u.post_.scale_)) for u in t['Z']]},indent=2));print(json.dumps(checks));print(O)
