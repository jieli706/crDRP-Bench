from pathlib import Path
import sys,pandas as pd
W=Path(__file__).resolve().parent;ROOT=W.parent
src=ROOT/'260925_Re/code/transfer_gdsc_benchmark.py';sys.path.insert(0,str(src.parent))
code=src.read_text(encoding='utf-8')
code=code.replace("'transfer_gdsc_'+", "'carcinoma_transfer_'+")
needle="basepairs=pd.read_csv(B/'selected_response_pairs.csv')"
insert="""
hist=meta[['OncotreePrimaryDisease','OncotreeSubtype']].fillna('').agg(' '.join,axis=1)
eligible=hist.str.contains('carcinoma',case=False,regex=False)
classification=meta[['ModelID','role','OncotreePrimaryDisease','OncotreeSubtype']].copy();classification['carcinoma_text_eligible']=eligible
classification.to_csv(O/'histology_selection.csv',index=False)
source=source[source.ModelID.isin(meta.loc[eligible,'ModelID'])].copy()
config['source']='Strict carcinoma sensitivity: explicit carcinoma or adenocarcinoma in DepMap primary-disease or subtype label; excludes breast and ambiguous histologies'
config['source_n_cells']=int(source.ModelID.nunique());config['source_n_pairs']=len(source)
(O/'config.json').write_text(json.dumps(config,indent=2))
"""
code=code.replace(needle,insert+'\n'+needle)
exec(compile(code,str(src),'exec'),{'__name__':'__main__','__file__':str(src)})
