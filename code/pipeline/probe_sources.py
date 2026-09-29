"""Public source availability checks; never save signed URLs or credentials."""
import csv, json, time, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URLS={
 'depmap_manifest':'https://depmap.org/portal/api/download/files',
 'wu_cellxgene':'https://datasets.cellxgene.cziscience.com/0e3974a1-9488-4987-9767-21d4e389fac7.h5ad',
 'wu_collection':'https://api.cellxgene.cziscience.com/curation/v1/collections/dea97145-f712-431c-a223-6b5f565f362a',
 'geo_supplement':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/suppl/',
 'tcga_clinical':'https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.clinical.tsv.gz',
 'metabric_clinical':'https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_clinical_patient.txt',
}
rows=[]
for key,url in URLS.items():
    row={'source':key,'url':url,'checked_utc':datetime.now(timezone.utc).isoformat()}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'crDRP-reanalysis-source-audit/1.0'})
        with urllib.request.urlopen(req,timeout=20) as r:
            data=r.read(65536)
            row.update(status=r.status,content_type=r.headers.get('Content-Type'),content_length=r.headers.get('Content-Length'),
                       etag=r.headers.get('ETag'),last_modified=r.headers.get('Last-Modified'))
        challenge=b'Verification' in data or b'reCAPTCHA' in data
        row['availability']='verification_required' if challenge else 'reachable_header_or_prefix_only'
        if key=='wu_collection' and not challenge:
            with urllib.request.urlopen(req,timeout=20) as r: content=r.read(4*1024*1024)
            obj=json.loads(content)
            # Official collection metadata contains public asset IDs; drop any signed URLs.
            def redact(v):
                if isinstance(v,dict): return {k:redact(x) for k,x in v.items()}
                if isinstance(v,list): return [redact(x) for x in v]
                if isinstance(v,str) and ('X-Amz-' in v or 'Signature=' in v): return v.split('?')[0]
                return v
            (ROOT/'audit'/'cellxgene_collection.json').write_text(json.dumps(redact(obj),indent=2),encoding='utf-8')
    except Exception as e:
        row.update(availability='request_failed',error_type=type(e).__name__,error=str(e).split('?')[0][:300])
    rows.append(row)
    (ROOT/'audit'/'source_probes.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print(key,row.get('status'),row['availability'],flush=True)
