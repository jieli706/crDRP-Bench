"""Cache exact repository indices and dependency binaries for local R only."""
import concurrent.futures,gzip,hashlib,json,re,urllib.request,time,zipfile,io,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'runtime/installers/r-packages';D.mkdir(exist_ok=True)
def fetch(url,timeout=20):
    with urllib.request.urlopen(url,timeout=timeout) as r:return r.read()
def dcf(txt):
    result=[]
    for block in txt.strip().split('\n\n'):
        row={};last=None
        for line in block.splitlines():
            if line[:1].isspace() and last:row[last]+=' '+line.strip()
            elif ': ' in line:last,v=line.split(': ',1);row[last]=v
        if 'Package' in row:result.append(row)
    return result
repos={'CRAN':'https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/contrib/4.5',
       'BioCsoft':'https://bioconductor.statistik.tu-dortmund.de/packages/3.22/bioc/bin/windows/contrib/4.5'}
db={}
for key,url in repos.items():
    p=D/(key+'_PACKAGES.gz')
    if not p.exists():
        if key=='CRAN':b=(ROOT/'runtime/installers/source_probe_0.bin').read_bytes()
        else:b=fetch(url+'/PACKAGES.gz',30)
        gzip.decompress(b);p.write_bytes(b)
    for row in dcf(gzip.decompress(p.read_bytes()).decode()):row['repository']=url;db[row['Package']]=row
    print('Index cached',key,flush=True)
source_url='https://bioconductor.statistik.tu-dortmund.de/packages/3.22/bioc/src/contrib'
sp=D/'BioCsoft_source_PACKAGES.gz'
if not sp.exists():sp.write_bytes(fetch(source_url+'/PACKAGES.gz',30))
for row in dcf(gzip.decompress(sp.read_bytes()).decode()):
    if row['Package'] not in db:row['repository']=source_url;row['source']=True;db[row['Package']]=row
base={'R','base','compiler','datasets','graphics','grDevices','grid','methods','parallel','splines','stats','stats4','tcltk','tools','utils','boot','class','cluster','codetools','foreign','KernSmooth','lattice','MASS','Matrix','mgcv','nlme','nnet','rpart','spatial','survival'}
todo=['snowfall','NMF','gplots','scran','BiocParallel'];wanted={}
while todo:
    name=todo.pop()
    if name in base or name in wanted:continue
    if name not in db:raise RuntimeError('Dependency absent from exact binary indices: '+name)
    row=db[name];wanted[name]=row
    for field in ['Depends','Imports']+(['LinkingTo'] if row.get('source') else []):
        for dep in row.get(field,'').split(','):
            dep=re.sub(r'\s*\(.*?\)','',dep).strip()
            if dep:todo.append(dep)
(D/'dependency_plan.json').write_text(json.dumps(list(wanted.values()),indent=2),encoding='utf-8')
print('Dependency binary count',len(wanted),flush=True)
def download(row):
    name=row['Package']+'_'+row['Version']+('.tar.gz' if row.get('source') else '.zip');p=D/name;url=row['repository']+'/'+name
    expected=row.get('MD5sum')
    if p.exists() and (not expected or hashlib.md5(p.read_bytes()).hexdigest()==expected):
        return {'file':name,'url':url,'status':'already_verified','md5':hashlib.md5(p.read_bytes()).hexdigest()}
    err=''
    for attempt in range(2):
        try:
            b=fetch(url,30)
            if expected and hashlib.md5(b).hexdigest()!=expected:raise ValueError('checksum mismatch')
            if row.get('source'):
                with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as archive:assert archive.getmembers()
            elif not zipfile.is_zipfile(io.BytesIO(b)):raise ValueError('not a ZIP')
            p.write_bytes(b);print('Verified',name,len(b),flush=True)
            return {'file':name,'url':url,'status':'verified','bytes':len(b),'md5':hashlib.md5(b).hexdigest(),'sha256':hashlib.sha256(b).hexdigest()}
        except Exception as e:err=str(e)
    print('FAILED',name,err,flush=True)
    return {'file':name,'url':url,'status':'failed','error':err}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(download,wanted.values()))
(D/('download_manifest_'+time.strftime('%Y%m%d_%H%M%S')+'.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
print('Completed',sum(r['status']!='failed' for r in results),'/',len(results))
