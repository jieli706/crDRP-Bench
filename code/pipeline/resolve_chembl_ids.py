"""Resolve exact drug names/synonyms only. Structures will come from pinned ChEMBL 37."""
import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/raw/chembl37/name_resolution';D.mkdir(parents=True,exist_ok=True)
URL='https://api.platform.opentargets.org/api/v4/graphql'
def query(q):
    req=urllib.request.Request(URL,data=json.dumps({'query':q}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)
source=pd.read_csv(ROOT/'data/processed/real_gdsc_features_20260926_135410/drug_structure_resolution.csv');source=source[source.match_status.ne('unique_structure')]
def resolve(row):
    p=D/(row.drug_dpc+'.json')
    if p.exists():return json.loads(p.read_text())
    result={'drug_dpc':row.drug_dpc,'query_name':row.drug_name,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'source':URL,'policy':'exact case-insensitive whitespace-trimmed preferred name or synonym; never take first hit; structure later from fixed ChEMBL 37'}
    try:
        hits=query('{search(queryString:'+json.dumps(row.drug_name)+',entityNames:["drug"]){hits{id name}}}')
        result['search']=hits;candidates=[];accepted=[]
        for hit in hits['data']['search']['hits']:
            cid=hit['id']
            if not cid.startswith('CHEMBL'):continue
            detail=query('{drug(chemblId:'+json.dumps(cid)+'){id name synonyms{label source} tradeNames{label source}}}')
            drug=detail['data']['drug'];candidates.append(drug);names=[drug['name']]+[v['label'] for k in ['synonyms','tradeNames'] for v in drug[k]]
            if row.drug_name.upper().strip() in {v.upper().strip() for v in names}:accepted.append(cid)
        result.update(candidates=candidates,accepted_chembl_ids=sorted(set(accepted)),status='unique_exact_alias' if len(set(accepted))==1 else 'ambiguous_or_unmatched')
    except Exception as e:result.update(status='request_failed',error_type=type(e).__name__)
    p.write_text(json.dumps(result,indent=2),encoding='utf-8');print(row.drug_dpc,result['status'],flush=True);return result
with concurrent.futures.ThreadPoolExecutor(3) as ex:out=list(ex.map(resolve,source.itertuples(index=False)))
pd.DataFrame([{'drug_dpc':r['drug_dpc'],'drug_name':r['query_name'],'status':r['status'],'chembl_ids':';'.join(r.get('accepted_chembl_ids',[]))} for r in out]).to_csv(D/'resolution_summary.csv',index=False)
print(pd.Series([r['status'] for r in out]).value_counts().to_dict())
