# -*- coding: utf-8 -*-
"""Read exact-version public GDSC inputs, audit units/joins, preserve every decision."""
import hashlib,json,logging
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'data/raw/gdsc2'
required={'GDSC2AUCMatrix.csv':'dad54f0b49513722e1eb49fce9a91146','GDSC2Log2IC50Matrix.csv':'6894fa298404d40a6f262c280964684b','GDSC2ResponseCurves.csv':'b294ef10b2de993b68df0ed27d74280f','sanger-dose-response.csv':None}
missing=[n for n in required if not (RAW/n).is_file()]
if missing:raise SystemExit('Missing exact-version core inputs: '+', '.join(missing))
OUT=ROOT/'data/processed'/('gdsc_25q2_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir(parents=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
sources=[]
for name,expected in required.items():
    p=RAW/name;h=hashlib.sha256();m=hashlib.md5()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b);m.update(b)
    sources.append({'filename':name,'bytes':p.stat().st_size,'sha256':h.hexdigest(),'md5':m.hexdigest(),'archived_expected_md5':expected,'exact_checksum_match':m.hexdigest()==expected if expected else None})
    if expected and m.hexdigest()!=expected:raise ValueError('Version/content mismatch: '+name)
(OUT/'source_provenance.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
def matrix(name,value):
    a=pd.read_csv(RAW/name,index_col=0)
    if not a.index.is_unique or not a.columns.is_unique:raise ValueError('Duplicate matrix IDs: '+name)
    a.index=a.index.astype(str);a.columns=a.columns.astype(str)
    v=a.to_numpy(float)
    if np.isinf(v).any():raise ValueError('Infinite response')
    return a,a.rename_axis(index='ModelID',columns='drug_dpc').stack(dropna=False).rename(value).reset_index()
am,a=matrix('GDSC2AUCMatrix.csv','auc');im,i=matrix('GDSC2Log2IC50Matrix.csv','ic50_matrix_value')
rc=pd.read_csv(RAW/'GDSC2ResponseCurves.csv',low_memory=False)
rc.to_csv(OUT/'response_curves_copy.csv',index=False) if False else None
drug=rc[['CompoundID','CompoundName']].drop_duplicates().rename(columns={'CompoundID':'drug_dpc','CompoundName':'drug_name'})
drug['drug_dpc']=drug.drug_dpc.astype(str)
if drug.drug_dpc.duplicated().any():
    drug[drug.drug_dpc.duplicated(keep=False)].to_csv(OUT/'ambiguous_compound_names.csv',index=False);raise ValueError('CompoundID maps to multiple names')
response=a.merge(i,on=['ModelID','drug_dpc'],how='outer',validate='one_to_one').merge(drug,on='drug_dpc',how='left',validate='many_to_one')
response['drug_name_upper']=response.drug_name.str.upper().str.strip()
sanger=pd.read_csv(RAW/'sanger-dose-response.csv',low_memory=False)
sanger=sanger[sanger.DATASET.eq('GDSC2')].copy();sanger['drug_name_upper']=sanger.DRUG_NAME.str.upper().str.strip()
sanger['ModelID']=sanger.ARXSPAN_ID;sanger['broad_id_13']=sanger.BROAD_ID.astype('string').str[:13]
sanger[['ModelID','drug_name_upper']].value_counts().rename('n_source_rows').reset_index().to_csv(OUT/'sanger_replicate_counts.csv',index=False)
mapping=sanger[['drug_name_upper','broad_id_13']].dropna().drop_duplicates();counts=mapping.groupby('drug_name_upper').broad_id_13.nunique()
mapping['unambiguous_name']=mapping.drug_name_upper.map(counts).eq(1);mapping.to_csv(OUT/'drug_name_broad_id_candidates.csv',index=False)
safe=mapping[mapping.unambiguous_name].set_index('drug_name_upper').broad_id_13.to_dict();drug['drug_name_upper']=drug.drug_name.str.upper().str.strip();drug['broad_id_13']=drug.drug_name_upper.map(safe)
drug.to_csv(OUT/'drug_ids_and_names.csv',index=False)
sanger['ln_ic50_published']=np.log(pd.to_numeric(sanger.IC50_PUBLISHED,errors='coerce').where(lambda x:x>0))
sanger['log2_ic50_published']=sanger.ln_ic50_published/np.log(2)
ref=sanger.groupby(['ModelID','drug_name_upper'])[['ln_ic50_published','log2_ic50_published','log2.ic50']].median().reset_index()
unit=response.merge(ref,on=['ModelID','drug_name_upper'],how='inner').dropna(subset=['ic50_matrix_value','ln_ic50_published','log2_ic50_published'])
unit.to_csv(OUT/'IC50_unit_matched_pairs.csv',index=False)
errors={name:float(np.mean(abs(unit.ic50_matrix_value-unit[name]))) for name in ['ln_ic50_published','log2_ic50_published','log2.ic50']}
best=min(['ln_ic50_published','log2_ic50_published'],key=lambda x:errors[x]);other='log2_ic50_published' if best=='ln_ic50_published' else 'ln_ic50_published'
unit_clear=len(unit)>=100 and errors[best]<.01 and errors[other]>10*max(errors[best],1e-8)
unit_report={'matched_pairs':len(unit),'MAE':errors,'matrix_scale':best if unit_clear else 'unresolved','criterion':'n>=100, best MAE<0.01 and alternate MAE>10x best; never infer scale from filename alone'}
(OUT/'IC50_unit_audit.json').write_text(json.dumps(unit_report,indent=2),encoding='utf-8')
response['ic50_scale']=unit_report['matrix_scale'];response['pair_id']=response.ModelID+'::'+response.drug_dpc
assert response.pair_id.is_unique
response.to_csv(OUT/'pancancer_response.csv',index=False)
expr=sorted(p for p in (ROOT/'data/processed').glob('depmap_26q1_*') if (p/'COMPLETE.json').exists())[-1]
meta=pd.read_csv(expr/'sample_ids_and_metadata.csv');eligible=meta[meta.known_breast&meta.known_cancer]
available=set(response.loc[response.auc.notna(),'ModelID']);sample=eligible.copy();sample['has_GDSC_AUC']=sample.ModelID.isin(available);sample.to_csv(OUT/'breast_sample_filter.csv',index=False)
ids=set(sample.loc[sample.has_GDSC_AUC,'ModelID']);breast=response[response.ModelID.isin(ids)].copy()
dc=breast.groupby('drug_dpc').agg(n_auc=('auc','count'),n_ic50=('ic50_matrix_value','count')).reset_index();dc['retained_at_20_auc']=dc.n_auc.ge(20);dc.to_csv(OUT/'breast_drug_filter.csv',index=False)
keep=set(dc.loc[dc.retained_at_20_auc,'drug_dpc']);bp=breast[breast.drug_dpc.isin(keep)&breast.auc.notna()].copy();bp.to_csv(OUT/'breast_response_AUC.csv',index=False)
both=bp.dropna(subset=['ic50_matrix_value']);summary={'AUC_matrix_shape':list(am.shape),'IC50_matrix_shape':list(im.shape),'AUC_nonmissing':int(am.notna().sum().sum()),'AUC_min':float(np.nanmin(am)),'AUC_max':float(np.nanmax(am)),'AUC_outside_0_1':int(((am<0)|(am>1)).sum().sum()),'response_curves_columns':rc.columns.tolist(),'expression_intake':str(expr.relative_to(ROOT)),'breast_cancer_expression_samples':len(eligible),'breast_expression_GDSC_samples':len(ids),'breast_drugs_ge20_auc':len(keep),'breast_pairs_ge20_auc':len(bp),'unmapped_drug_name_pairs':int(response.drug_name.isna().sum()),'AUC_IC50_pearson':float(np.corrcoef(both.auc,both.ic50_matrix_value)[0,1]),'unit_audit':unit_report,'no_drug_feature_coverage_filter_yet':True,'preprocessing_fit':False}
(OUT/'cohort_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(OUT/'COMPLETE.json').write_text(json.dumps({'status':'completed_core_data_audit','units_resolved':unit_clear,'not_model_benchmark':True},indent=2),encoding='utf-8')
print(json.dumps(summary));print(OUT)
