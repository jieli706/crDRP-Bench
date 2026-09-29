"""Stream documented public assets, retain provenance, never overwrite verified files."""
import hashlib,json,shutil,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
items=[
 ('scrna/wu2021_cellxgene.h5ad','https://datasets.cellxgene.cziscience.com/0e3974a1-9488-4987-9767-21d4e389fac7.h5ad','Wu CELLxGENE historical asset referenced by original code',843895986),
 ('bulk/tcga/TCGA-BRCA.clinical.tsv.gz','https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.clinical.tsv.gz','Xena GDC live object',None),
 ('bulk/tcga/TCGA-BRCA.survival.tsv.gz','https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.survival.tsv.gz','Xena GDC live object',None),
 ('bulk/metabric/data_clinical_patient.txt','https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_clinical_patient.txt','cBioPortal master snapshot; SHA256 identifies bytes',None),
 ('bulk/metabric/data_clinical_sample.txt','https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_clinical_sample.txt','cBioPortal master snapshot; SHA256 identifies bytes',None),
]
ledger=ROOT/'audit/download_manifest.json'
rows=json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else []
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
for rel,url,version,expected in items:
    p=ROOT/'data/raw'/rel; p.parent.mkdir(parents=True,exist_ok=True)
    prior=next((x for x in rows if x['path']==rel and x.get('status')=='verified_download'),None)
    if p.exists():
        if prior and sha(p)==prior['sha256']: print('verified existing',rel,flush=True); continue
        raise RuntimeError('Refusing overwrite of unverified existing file: '+str(p))
    row={'path':rel,'source_url':url,'version':version,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    try:
        if shutil.disk_usage(ROOT).free<5*1024**3+(expected or 100*1024**2):raise RuntimeError('Disk safety reserve too low')
        part=p.with_suffix(p.suffix+'.part'); total=0; h=hashlib.sha256()
        req=urllib.request.Request(url,headers={'User-Agent':'crDRP-reanalysis/1.0'})
        with urllib.request.urlopen(req,timeout=45) as r,part.open('wb') as f:
            length=int(r.headers.get('Content-Length','0')); row.update(http_content_length=length,etag=r.headers.get('ETag'),last_modified=r.headers.get('Last-Modified'))
            while True:
                b=r.read(1024*1024)
                if not b:break
                if total==0 and (b.lstrip().startswith(b'<!DOCTYPE html') or b.startswith(b'version https://git-lfs')):raise ValueError('HTML or LFS pointer is not data')
                f.write(b);h.update(b);total+=len(b)
                if total//(100*1024**2) != (total-len(b))//(100*1024**2): print(rel,total,'bytes',flush=True)
        if length and total!=length:raise ValueError('Content-Length mismatch')
        if expected and total!=expected:raise ValueError('Historical expected length mismatch')
        if rel.endswith('.h5ad'):
            with part.open('rb') as f:
                if f.read(8)!=b'\x89HDF\r\n\x1a\n':raise ValueError('Invalid HDF5 signature')
        part.rename(p)
        row.update(status='verified_download',bytes=total,sha256=h.hexdigest())
        print('download complete',rel,total,flush=True)
    except Exception as e:
        row.update(status='failed',error_type=type(e).__name__,error=str(e).split('?')[0][:250]);print('failed',rel,row['error'],flush=True)
    rows.append(row);ledger.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
