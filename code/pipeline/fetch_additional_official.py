"""Resumable official ChEMBL 37 and local Rtools downloads; no credentials."""
import concurrent.futures,hashlib,json,re,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def get(url):
    with urllib.request.urlopen(url,timeout=30) as r:return r.read()
C=ROOT/'data/raw/chembl37';C.mkdir(exist_ok=True)
base='https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/'
for n in ['checksums.txt','README','LICENSE','REQUIRED.ATTRIBUTION']:
    if not (C/n).exists():(C/n).write_bytes(get(base+n))
checks=(C/'checksums.txt').read_text();expected=[l.split()[0] for l in checks.splitlines() if 'chembl_37_chemreps.txt.gz' in l][0]
def download(args):
    url,dest,md5=args;part=dest.with_suffix(dest.suffix+'.part');result={'url':url,'path':str(dest.relative_to(ROOT)),'official_checksum':md5,'checksum_algorithm':'sha256' if md5 and len(md5)==64 else 'md5'};total=None;starttime=time.monotonic()
    if dest.exists():return {**result,'status':'already_present','bytes':dest.stat().st_size}
    try:
        while time.monotonic()-starttime<1200:
            offset=part.stat().st_size if part.exists() else 0
            if total is not None and offset==total:break
            req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{offset+4*1024*1024-1}'})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req,timeout=30) as r:
                        if r.status==206:
                            m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''));assert m and int(m[1])==offset
                            total=int(m[3]);b=r.read(int(m[2])-offset+1);assert len(b)==int(m[2])-offset+1
                            with part.open('ab') as f:f.write(b)
                        elif r.status==200 and offset==0:
                            with part.open('wb') as f:
                                while True:
                                    b=r.read(1024*1024)
                                    if not b:break
                                    f.write(b)
                            total=part.stat().st_size
                        else:raise ValueError('Range unsupported on resume')
                    break
                except Exception:
                    if attempt==2:raise
            print(dest.name,part.stat().st_size,'/',total,flush=True)
        if total is None or part.stat().st_size!=total:raise TimeoutError('Partial preserved')
        h=hashlib.md5();s=hashlib.sha256()
        with part.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b);s.update(b)
        if md5 and (s.hexdigest() if len(md5)==64 else h.hexdigest())!=md5:raise ValueError('Official checksum mismatch')
        part.rename(dest);result.update(status='completed',bytes=total,md5=h.hexdigest(),sha256=s.hexdigest())
    except Exception as e:result.update(status='failed_partial_preserved',error_type=type(e).__name__)
    dest.with_suffix(dest.suffix+'.provenance.json').write_text(json.dumps(result,indent=2));return result
needs=[(base+'chembl_37_chemreps.txt.gz',C/'chembl_37_chemreps.txt.gz',expected),('https://cran.r-project.org/bin/windows/Rtools/rtools45/files/rtools45-6768-6492.exe',ROOT/'runtime/installers/rtools45-6768-6492.exe',None)]
with concurrent.futures.ThreadPoolExecutor(2) as ex:
    for r in ex.map(download,needs):print(json.dumps(r),flush=True)
