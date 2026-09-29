"""Read-only inventory outside the analysis root; all output stays inside it."""
import csv, hashlib, importlib.metadata as md, json, os, platform, shutil, subprocess, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
for d in ['audit','config','data/raw','data/processed','logs','reports','runs','source_snapshot']:
    (ROOT/d).mkdir(parents=True,exist_ok=True)
def write_json(name, value):
    (ROOT/'audit'/name).write_text(json.dumps(value, ensure_ascii=False, indent=2),encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def csvout(name, rows, fields):
    with (ROOT/'audit'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
rows=[]; code=[]; ziprows=[]
for base, dirs, files in os.walk(PROJECT):
    dirs[:]=[d for d in dirs if d not in ['.git','node_modules','.venv','260925_Re']]
    for name in files:
        p=Path(base)/name
        try:
            s=p.stat(); rel=p.relative_to(PROJECT).as_posix()
            row={'path':rel,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'extension':p.suffix.lower()}
            rows.append(row)
            if p.suffix.lower() in ['.py','.r','.sh']:
                code.append({**row,'sha256':digest(p)})
            if p.suffix.lower()=='.zip':
                with zipfile.ZipFile(p) as z:
                    ziprows += [{'archive':rel,'member':i.filename,'bytes':i.file_size} for i in z.infolist()]
        except Exception as e: rows.append({'path':str(p),'bytes':-1,'mtime_ns':0,'extension':str(type(e).__name__)})
csvout('file_inventory.csv',rows,['path','bytes','mtime_ns','extension'])
csvout('code_hashes.csv',code,['path','bytes','mtime_ns','extension','sha256'])
csvout('archive_inventory.csv',ziprows,['archive','member','bytes'])
src=ROOT.parent/'github_new'
for folder in ['scripts','src']:
    for p in (src/folder).glob('*'):
        if p.is_file():
            dst=ROOT/'source_snapshot'/folder/p.name; dst.parent.mkdir(exist_ok=True,parents=True)
            if not dst.exists(): shutil.copy2(p,dst)
for p in src.glob('*.md'):
    dst=ROOT/'source_snapshot'/p.name
    if not dst.exists(): shutil.copy2(p,dst)
env={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,
     'platform':platform.platform(),'logical_cpu':os.cpu_count(),'disks':{},'packages':{}}
for drive in ['C:/','D:/','E:/']:
    env['disks'][drive]=dict(zip(['total','used','free'],shutil.disk_usage(drive)))
for pkg in ['numpy','pandas','scipy','scikit-learn','torch','psutil','requests','pyarrow','h5py','anndata','scanpy','rdkit','xgboost','matplotlib','mordredcommunity']:
    try: env['packages'][pkg]=md.version(pkg)
    except md.PackageNotFoundError: env['packages'][pkg]=None
try:
    import psutil
    env['memory']=psutil.virtual_memory()._asdict()
except Exception as e: env['memory_error']=str(e)
try:
    import torch
    env['torch_cuda_available']=torch.cuda.is_available(); env['torch_cuda_build']=torch.version.cuda
except Exception as e: env['torch_error']=str(e)
try:
    env['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv'],text=True)
except Exception as e: env['gpu_error']=str(e)
write_json('environment.json',env)
with (ROOT/'audit'/'pip_freeze.txt').open('w',encoding='utf-8') as f:
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,stderr=subprocess.STDOUT)
manifests=list(PROJECT.glob('breast_all/**/DATA_UPLOAD_MANIFEST.csv'))
if manifests:
    m=next((p for p in manifests if 'repository_upload' in str(p)),manifests[0])
    with m.open(encoding='utf-8-sig') as f: requirements=list(csv.DictReader(f))
    for r in requirements:
        suffix=r['relative_path'].replace('\\','/')
        hits=[x['path'] for x in rows if x['path'].endswith(suffix)]
        r['local_matches']='|'.join(hits); r['status']='present' if hits else 'missing'
    csvout('required_data_status.csv',requirements,list(requirements[0]))
    write_json('inventory_summary.json',{'files':len(rows),'total_bytes':sum(max(0,r['bytes']) for r in rows),
       'code_files':len(code),'zip_members':len(ziprows),'required':len(requirements),
       'required_missing':sum(r['status']=='missing' for r in requirements),
       'source_manifest':str(m),'analysis_excluded':str(ROOT)})
print(json.dumps({'files':len(rows),'code':len(code),'memory':env.get('memory'),'packages':env['packages']},ensure_ascii=False))
