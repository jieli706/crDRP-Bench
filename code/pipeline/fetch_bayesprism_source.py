import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'runtime/BayesPrism-source';D.mkdir(exist_ok=True)
sha='19052052a6f30833b27e2148294459b0ba2f923e';tree=json.loads((ROOT/'runtime/installers/BayesPrism_tree.json').read_text())
def fetch(row):
    p=D/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen('https://raw.githubusercontent.com/Danko-Lab/BayesPrism/'+sha+'/'+row['path'],timeout=30) as r:b=r.read()
        actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert actual==row['sha'];p.write_bytes(b)
    else:b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['sha']
    return {'path':row['path'],'git_blob_sha1':row['sha'],'sha256':hashlib.sha256(b).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(3) as ex:rows=list(ex.map(fetch,[r for r in tree['tree'] if r['path'].startswith('BayesPrism/') and r['type']=='blob']))
(D/'provenance.json').write_text(json.dumps({'commit':sha,'files':rows},indent=2));print('Verified source files',len(rows))
