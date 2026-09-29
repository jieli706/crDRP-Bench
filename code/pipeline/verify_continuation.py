"""Replay saved downstream predictions and verify preservation without rewriting old results."""
import ast,csv,hashlib,importlib.util,json,pickle
from pathlib import Path
import numpy as np,pandas as pd,torch
from leakage_safe import FoldTransform
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/continuation_20260926_125427'
RUN=ROOT/'runs/paired_deconvolution_downstream_20260926_130844';INP=ROOT/'runs/paired_deconvolution_inputs_20260926_130450'
spec=importlib.util.spec_from_file_location('cr',ROOT/'source_snapshot/scripts/07_crdrp_model.py');cr=importlib.util.module_from_spec(spec);spec.loader.exec_module(cr)
torch.set_num_threads(2)
replay=[]
for seed in [7,17,37]:
    a=np.load(INP/f'seed_{seed}/paired_inputs.npz');d=np.load(RUN/f'seed_{seed}/synthetic_drugs_and_nnls_allocation.npz')
    mi={v:i for i,v in enumerate(a['mixture_ids'])};di={v:i for i,v in enumerate(d['drug_ids'])}
    for method in ['crdrp_oracle','crdrp_nnls','crdrp_bulk']:
        path=RUN/f'seed_{seed}'
        with (path/f'{method}_preprocessing.pkl').open('rb') as f:p=pickle.load(f)
        with (path/f'{method}.pt').open('rb') as f:ck=torch.load(f,map_location='cpu')
        frame=pd.read_csv(path/f'{method}_test_predictions.csv');ci=np.array([mi[v] for v in frame.ModelID]);dd=np.array([di[v] for v in frame.drug_dpc])
        drugs=torch.tensor(p['drug'].transform(d['drug_features'])[dd])
        if method=='crdrp_bulk':
            x=a['counts'];x=np.log1p(x/x.sum(1,keepdims=True)*1e6);bx=torch.tensor(p['bulk'].transform(x)[ci]);model=cr.BulkDRP(bulk_dim=ck['bulk_dim'],drug_dim=ck['drug_dim']);args=(bx,drugs)
        else:
            z=a['oracle_Z'] if method=='crdrp_oracle' else d['nnls_Z'];z=np.log1p(z*1e6)
            z=np.stack([p['Z'][k].transform(z[:,k]) for k in range(8)],axis=1)
            pi=a['oracle_pi'] if method=='crdrp_oracle' else a['nnls_pi'];model=cr.CrDRP(z_dim=ck['z_dim'],drug_dim=ck['drug_dim']);args=(torch.tensor(pi[ci],dtype=torch.float32),torch.tensor(z[ci]),drugs)
        model.load_state_dict(ck['state_dict']);model.eval()
        with torch.no_grad():yp=model(*args);yp=yp if method=='crdrp_bulk' else yp[0]
        err=float(np.max(abs(yp.numpy()-frame.y_pred.to_numpy())))
        if err>2e-5:raise ValueError(f'Replay exceeds tolerance: {seed}, {method}, {err}')
        replay.append({'seed':seed,'method':method,'CPU_replay_max_abs_difference':err,'tolerance':2e-5})
pd.DataFrame(replay).to_csv(OUT/'downstream_checkpoint_replay.csv',index=False)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
old=list(csv.DictReader((ROOT/'audit/file_inventory.csv').open(encoding='utf-8-sig')));changed=[]
for r in old:
    p=ROOT.parents[1]/r['path']
    if not p.exists() or p.stat().st_size!=int(r['bytes']) or p.stat().st_mtime_ns!=int(r['mtime_ns']):changed.append(r['path'])
prior=list(csv.DictReader((ROOT/'audit/analysis_artifact_sha256.csv').open(encoding='utf-8-sig')));mismatch=[]
for r in prior:
    p=ROOT/r['path']
    if not p.exists() or sha(p)!=r['sha256']:mismatch.append(r['path'])
for p in (ROOT/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
result={'prior_project_files_checked':len(old),'prior_project_files_changed_size_or_mtime':changed,'prior_stage_hashed_files_checked':len(prior),'prior_stage_hash_mismatches':mismatch,'downstream_checkpoints_replayed':len(replay),'max_cpu_replay_error':max(r['CPU_replay_max_abs_difference'] for r in replay),'python_syntax_all_pass':True}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
rows=[]
for folder in [RUN,INP,ROOT/'runs/positive_control_diagnostic_20260926_125204',ROOT/'runs/clinical_cox_diagnostic_20260926_130209',ROOT/'code']:
    for p in sorted(folder.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
with (OUT/'new_analysis_artifacts_sha256.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
