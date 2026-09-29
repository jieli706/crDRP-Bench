"""Fetch only exact requested releases from the user's official download catalog.
Signed query strings are consumed locally and never copied into public logs.
"""
import argparse,csv,hashlib,json,re,time,urllib.request,urllib.parse,concurrent.futures
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];catalog=ROOT/'depmap_download_files.csv'
rows=list(csv.DictReader(catalog.open(encoding='utf-8-sig',newline='')))
parser=argparse.ArgumentParser();parser.add_argument('--prism',action='store_true');parser.add_argument('--counts',action='store_true');args=parser.parse_args()
needs=[('Harmonized GDSC 25Q2','GDSC2AUCMatrix.csv','gdsc2'),('Harmonized GDSC 25Q2','GDSC2Log2IC50Matrix.csv','gdsc2'),('Harmonized GDSC 25Q2','GDSC2ResponseCurves.csv','gdsc2'),('Sanger GDSC1 and GDSC2','sanger-dose-response.csv','gdsc2')]
if args.prism:needs=[('PRISM Repurposing 19Q4',n,'prism') for n in ['secondary-screen-replicate-treatment-info.csv','secondary-screen-cell-line-info.csv','secondary-screen-dose-response-curve-parameters.csv']]
if args.counts:needs=[('DepMap Public 26Q1',n,'depmap') for n in ['OmicsProfiles.csv','OmicsExpressionRawReadCountHumanProteinCodingGenes.csv']]
OUT=ROOT/'runs'/('official_download_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir()
def dig(p,algo):
    h=hashlib.new(algo)
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def download(item):
    release,name,folder=item;hits=[r for r in rows if r['release']==release and r['filename']==name]
    if len(hits)!=1:return {'filename':name,'status':'catalog_entry_not_unique','matches':len(hits)}
    row=hits[0];url=row['url'];u=urllib.parse.urlsplit(url)
    if u.scheme!='https' or u.hostname not in {'storage.googleapis.com','ndownloader.figshare.com','api.figshare.com','depmap.org'}:return {'filename':name,'status':'unexpected_url_host'}
    dest=ROOT/'data/raw'/folder/name;dest.parent.mkdir(exist_ok=True,parents=True);part=dest.with_suffix(dest.suffix+'.part')
    expected=row.get('md5_hash','');result={'release':release,'release_date':row['release_date'],'filename':name,'destination':str(dest.relative_to(ROOT)),'source_url_without_query':urllib.parse.urlunsplit((u.scheme,u.netloc,u.path,'','')),'catalog_sha256':dig(catalog,'sha256'),'expected_md5':expected,'started_utc':datetime.now(timezone.utc).isoformat()}
    if dest.exists():
        actual=dig(dest,'md5');result.update(status='existing_verified' if expected and actual==expected else 'existing_not_overwritten',actual_md5=actual,sha256=dig(dest,'sha256'),bytes=dest.stat().st_size);return result
    total=None;failures=0;deadline=time.monotonic()+600
    try:
        while time.monotonic()<deadline:
            start=part.stat().st_size if part.exists() else 0
            if total is not None and start==total:break
            end=min(start+1048576-1,total-1) if total else start+1048576-1
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=25) as r:
                    ctype=r.headers.get('Content-Type','')
                    if 'text/html' in ctype:raise ValueError('HTML verification/error page; no download')
                    if r.status==206:
                        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
                        if not match or int(match[1])!=start:raise ValueError('Range mismatch')
                        total=int(match[3]);size=int(match[2])-start+1;b=r.read(size)
                        if len(b)!=size:raise ValueError('Incomplete range')
                        with part.open('ab') as f:f.write(b)
                    elif r.status==200 and start==0:
                        b=r.read();total=len(b);part.write_bytes(b)
                    else:raise ValueError('Server ignored resume range')
                    result['content_type']=ctype;result['etag']=r.headers.get('ETag')
                failures=0
            except Exception as e:
                failures+=1
                print(name,'request failure',type(e).__name__,getattr(e,'code',''),flush=True)
                if failures>=3:raise
        if total is None or part.stat().st_size!=total:raise TimeoutError('Download time limit; partial preserved')
        prefix=part.open('rb').read(256).lstrip().lower()
        if prefix.startswith((b'<!doctype',b'<html',b'<?xml')):raise ValueError('Non-data document')
        actual=dig(part,'md5');result.update(actual_md5=actual,sha256=dig(part,'sha256'),bytes=part.stat().st_size)
        if expected and actual!=expected:raise ValueError('Official MD5 mismatch; partial retained')
        with part.open(encoding='utf-8-sig',newline='') as f:header=next(csv.reader(f))
        if len(header)<2:raise ValueError('Not a tabular CSV')
        part.rename(dest);result.update(status='downloaded_checksum_verified' if expected else 'downloaded_without_publisher_checksum',csv_columns=len(header),completed_utc=datetime.now(timezone.utc).isoformat())
        print(name,result['status'],result['bytes'],'bytes',flush=True)
    except Exception as e:
        result.update(status='failed',error_type=type(e).__name__,http_code=getattr(e,'code',None))
    (OUT/(name+'.provenance.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(download,needs))
(OUT/'download_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT);print(json.dumps([{k:r.get(k) for k in ['filename','status','bytes','error_type','http_code']} for r in results],indent=2))
if any(r['status']=='failed' for r in results):raise SystemExit(2)
