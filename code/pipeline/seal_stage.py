"""Record executable source hashes, legacy code evidence and reproducibility entrypoints."""
import ast,csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
rules={
 '07a_pretrain_pancancer.py':['pair_cell_idx','index.parquet'],
 '07b_finetune_breast.py':['genes = pe.var','StandardScaler().fit(Xp)','def pretrain_state','init = pretrain_state','by_cells=False'],
 '07d_leakfree_pretrain.py':['def leakfree_pairs','keep =','cell_bulk_shared'],
 '06e_shared_Z_featurize.py':['StandardScaler().fit(zp)','PCA(n_components=DZ','pair_cell'],
 '06_train_baseline.py':['def build_drug_pca','StandardScaler().fit(X)','GroupKFold'],
 '07_crdrp_model.py':['F.softmax','l1 =','return huber'],
 '06b_deconvolve_celllines.R':['as.matrix(sc)','new.prism','run.prism']}
evidence=[]
for name,needles in rules.items():
    p=ROOT/'source_snapshot/scripts'/name
    for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if any(x in line for x in needles):evidence.append({'file':'source_snapshot/scripts/'+name,'line':i,'code':line.strip(),'file_sha256':sha(p)})
with (ROOT/'audit/code_evidence.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['file','line','code','file_sha256']);w.writeheader();w.writerows(evidence)
git=subprocess.run(['git','-C',str(ROOT.parent/'github_new'),'rev-parse','HEAD'],capture_output=True,text=True)
(ROOT/'audit/source_git_status.json').write_text(json.dumps({'path':str(ROOT.parent/'github_new'),'commit':git.stdout.strip() if git.returncode==0 else None,'status':'commit_available' if git.returncode==0 else 'no_commit_resolved; content hashes used'},indent=2),encoding='utf-8')
for p in (ROOT/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
rows=[]
for folder in ['code','config','runs','reports','source_snapshot']:
    for p in sorted((ROOT/folder).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
with (ROOT/'audit/analysis_artifact_sha256.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
print(json.dumps({'hashed_artifacts':len(rows),'code_syntax':'all_pass','source_commit_available':git.returncode==0}))
