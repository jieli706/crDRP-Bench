"""Resume an exact-version public installer; only a verified file is usable."""
import hashlib,json,shutil,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'runtime/installers'
URL='https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/base/old/4.5.2/R-4.5.2-win.exe'
EXPECTED='bc35c78e0a6c9790d7d540b44089f31f';TOTAL=90233368
out=D/'R-4.5.2-win.exe';part=D/'R-4.5.2-win.exe.resume-part'
if out.exists():
    assert hashlib.md5(out.read_bytes()).hexdigest()==EXPECTED
    print('Verified installer already exists');raise SystemExit(0)
if not part.exists():
    candidates=list(D.glob('R-4.5.2-win.exe.previous-part-*'))+list(D.glob('R-4.5.2-win.exe.part'))
    best=max(candidates,key=lambda p:p.stat().st_size)
    shutil.copyfile(best,part)
log=D/('R_resume_'+time.strftime('%Y%m%d_%H%M%S')+'.jsonl')
deadline=time.monotonic()+600
attempt=0
while part.stat().st_size<TOTAL and time.monotonic()<deadline and attempt<25:
    attempt+=1;start=part.stat().st_size;end=min(start+1048576-1,TOTAL-1)
    try:
        req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req,timeout=20) as r:
            cr=r.headers.get('Content-Range','')
            if r.status!=206 or cr!=f'bytes {start}-{end}/{TOTAL}':
                raise ValueError(f'Server did not honor exact range: {r.status}, {cr}')
            b=r.read(end-start+1)
        if len(b)!=end-start+1:raise ValueError('Incomplete chunk')
        with part.open('ab') as f:f.write(b)
        row={'attempt':attempt,'start':start,'end':end,'success':True}
    except Exception as e:
        row={'attempt':attempt,'start':start,'end':end,'success':False,'error':str(e)}
    with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    print(row,flush=True)
if part.stat().st_size!=TOTAL:raise SystemExit('Incomplete download retained for later resume')
b=part.read_bytes();actual=hashlib.md5(b).hexdigest()
if actual!=EXPECTED:raise SystemExit('Checksum mismatch; installer will NOT be used')
part.rename(out)
(D/'R_4.5.2_provenance.json').write_text(json.dumps({'version':'4.5.2','download_url':URL,'checksum_source':'https://cran.r-project.org/bin/windows/base/old/4.5.2/md5sum.R-4.5.2.txt','bytes':len(b),'md5_expected':EXPECTED,'md5_actual':actual,'sha256':hashlib.sha256(b).hexdigest()},indent=2),encoding='utf-8')
print('Verified complete installer',out,flush=True)
