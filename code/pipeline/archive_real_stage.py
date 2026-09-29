"""Archive source bytes, model artifacts and report hashes without changing prior files."""
import argparse,csv,hashlib,json,shutil,sys,platform
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args();report=Path(a.report).resolve();index=json.loads((report/'运行索引.json').read_text(encoding='utf-8'));O=ROOT/'runs'/('real_stage_archive_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir();src=O/'code_snapshot';src.mkdir()
for f in (ROOT/'code').iterdir():
    if f.is_file():shutil.copy2(f,src/f.name)
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
# Recover the exact pre-stability CLI source only if the saved run SHA matches.
s=(ROOT/'code/real_gdsc_benchmark.py').read_text(encoding='utf-8')
s=s.replace("p.add_argument('--stable-pca',action='store_true');",'').replace('if a.stable_pca:\n    from stable_preprocessing import StableFoldTransform as FoldTransform\n','').replace("if not a.smoke:assert json.loads((Path(a.smoke_run)/'config.json').read_text()).get('stable_PCA_null_space',False)==a.stable_pca\n",'').replace('cfg.update(stable_PCA_null_space=a.stable_pca,','cfg.update(')
expected=json.loads((ROOT/'runs/real_gdsc_cell_20260926_150657/sources.json').read_text())[0]['sha256'];candidate=hashlib.sha256(s.encode()).hexdigest();recovery={'expected':expected,'recovered':candidate,'exact_match':expected==candidate}
if expected==candidate:(src/('real_gdsc_benchmark_'+expected[:12]+'.py')).write_bytes(s.encode('utf-8'))
(O/'source_recovery.json').write_text(json.dumps(recovery,indent=2))
folders=[ROOT/'data',ROOT/'code',ROOT/'runs',ROOT/'reports',ROOT/'source_snapshot'];files=[]
for folder in folders:
 for f in folder.rglob('*'):
    if f.is_file() and '__pycache__' not in f.parts and O not in f.parents and f.suffix not in ['.part','.tmp']:files.append(f)
for f in src.iterdir():files.append(f)
total=0
with (O/'artifact_sha256.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    w=csv.DictWriter(stream,fieldnames=['path','bytes','sha256']);w.writeheader()
    for f in sorted(set(files)):
        n=f.stat().st_size;w.writerow({'path':f.relative_to(ROOT).as_posix(),'bytes':n,'sha256':sha(f)});total+=n
res={'report':str(report),'files_hashed':len(set(files)),'bytes_hashed':total,'runtime_binaries_excluded':True,'downloads_do_not_imply_analysis_complete':True,'source_recovery':recovery,'original_project_not_modified':True,'python':sys.version,'platform':platform.platform(),'disk_free_E_bytes':shutil.disk_usage(ROOT).free}
(O/'COMPLETE.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(res,ensure_ascii=False));print(O)
