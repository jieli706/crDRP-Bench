"""Fetch exact CRAN R version into local runtime; do not install or alter registry."""
import hashlib,json,urllib.request,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runtime/installers';out.mkdir(parents=True,exist_ok=True)
base='https://cran.r-project.org/bin/windows/base/old/4.5.2/'
download_url='https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/base/old/4.5.2/R-4.5.2-win.exe'
name='R-4.5.2-win.exe';p=out/name
check=urllib.request.urlopen(base+'md5sum.R-4.5.2.txt',timeout=30).read().decode()
(out/'md5sum.R-4.5.2.txt').write_text(check,encoding='utf-8');expected=check.split()[0].lower()
if p.exists() and hashlib.md5(p.read_bytes()).hexdigest()!=expected:
    p.rename(out/(name+'.invalid-'+hashlib.md5(p.read_bytes()).hexdigest()))
if not p.exists():
    started=time.monotonic()
    prior=p.with_suffix('.exe.part')
    if prior.exists():prior.rename(out/(name+'.previous-part-'+str(int(time.time()))))
    with urllib.request.urlopen(download_url,timeout=45) as r,p.with_suffix('.exe.part').open('wb') as f:
        print('CRAN mirror response',r.status,dict(r.headers),flush=True)
        total=0
        while True:
            b=r.read(65536)
            if not b:break
            f.write(b);total+=len(b)
            if time.monotonic()-started>300:raise TimeoutError('Download exceeded 5 minute cap')
        f.flush()
    assert hashlib.md5(p.with_suffix('.exe.part').read_bytes()).hexdigest()==expected,'Downloaded installer checksum mismatch; not executable'
    p.with_suffix('.exe.part').rename(p)
b=p.read_bytes();actual=hashlib.md5(b).hexdigest()
assert actual==expected,(actual,expected)
(out/'R_4.5.2_provenance.json').write_text(json.dumps({'source':base+name,'download_mirror':download_url,'mirror_verified_against':'https://cran.r-project.org/mirrors.html','version':'4.5.2','bytes':len(b),'md5_expected':expected,'md5_actual':actual,'sha256':hashlib.sha256(b).hexdigest()},indent=2),encoding='utf-8')
print('CRAN R 4.5.2 verified',len(b),'bytes')
