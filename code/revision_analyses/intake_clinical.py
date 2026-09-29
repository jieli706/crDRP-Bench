from pathlib import Path
import requests,hashlib,json,pandas as pd
from bs4 import BeautifulSoup
W=Path(__file__).resolve().parent;O=W/'clinical';O.mkdir(exist_ok=True)
page=requests.get('https://gdc.cancer.gov/about-data/publications/pancanatlas',timeout=60);page.raise_for_status();soup=BeautifulSoup(page.text,'html.parser');urls={}
for a in soup.find_all('a',href=True):
 if 'TCGA-CDR-SupplementalTableS1.xlsx' in a.text:urls['cdr.xlsx']=a['href']
 if 'TCGA_mastercalls.abs_tables_JSedit.fixed.txt' in a.text:urls['absolute.tsv']=a['href']
urls['aran.xlsx']='https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fncomms9971/MediaObjects/41467_2015_BFncomms9971_MOESM2_ESM.xlsx'
rows=[]
for name,url in urls.items():
 try:
  r=requests.get(url,timeout=120);r.raise_for_status();(O/name).write_bytes(r.content)
  rows.append(dict(file=name,url=url,sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content)))
  print(name,len(r.content),flush=True)
 except Exception as e:print(name,type(e).__name__,str(e),flush=True)
(O/'sources.json').write_text(json.dumps(rows,indent=2))
for name in ['cdr.xlsx','aran.xlsx']:
 if (O/name).exists():
  f=pd.ExcelFile(O/name);print(name,f.sheet_names);print(pd.read_excel(f).head().to_string())
q=pd.read_csv(W.parent/'260925_Re/data/processed/transfer_inputs_20260926_142944/sample_ids.csv');print(q[q.role=='source'].OncotreePrimaryDisease.value_counts().to_string())
