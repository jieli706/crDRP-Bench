"""Fetch original-script Xena objects, record provider version and ETag, no substitution."""
import hashlib,json,re,time,urllib.request
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/raw/bulk/tcga';base='https://gdc-hub.s3.us-east-1.amazonaws.com/download/';O=ROOT/'runs'/('tcga_expression_download_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir()
items=[('TCGA-BRCA.star_counts.tsv.json',base+'TCGA-BRCA.star_counts.tsv.json'),('TCGA-BRCA.star_counts.tsv.gz',base+'TCGA-BRCA.star_counts.tsv.gz'),('TCGASubtype.20170308.tsv.gz','https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGASubtype.20170308.tsv.gz')];result=[]
for name,url in items:
    p=D/name;part=p.with_suffix(p.suffix+'.part');row={'filename':name,'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    if p.exists():row.update(status='existing_not_overwritten');result.append(row);continue
    total=None;endtime=time.monotonic()+900
    try:
        while time.monotonic()<endtime:
            start=part.stat().st_size if part.exists() else 0
            if total is not None and start==total:break
            req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+4194304-1}'})
            with urllib.request.urlopen(req,timeout=30) as r:
                row.update(etag=r.headers.get('ETag'),last_modified=r.headers.get('Last-Modified'))
                if r.status==206:
                    m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''));assert m and int(m[1])==start;total=int(m[3]);b=r.read();assert len(b)==int(m[2])-start+1
                    with part.open('ab') as f:f.write(b)
                elif r.status==200 and start==0:b=r.read();part.write_bytes(b);total=len(b)
                else:raise ValueError('Range response mismatch')
            if total>1048576:print(name,part.stat().st_size,'/',total,flush=True)
        assert total is not None and part.stat().st_size==total
        with part.open('rb') as f:prefix=f.read(256)
        if name.endswith('.gz'):assert prefix.startswith(b'\x1f\x8b')
        else:json.loads(part.read_text())
        h=hashlib.sha256()
        with part.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        part.rename(p);row.update(status='complete',bytes=total,sha256=h.hexdigest(),checksum_scope='local SHA256; S3 multipart ETag is not treated as MD5')
    except Exception as e:row.update(status='failed_partial_preserved',error_type=type(e).__name__)
    result.append(row);(O/'manifest.json').write_text(json.dumps(result,indent=2))
print(O)
