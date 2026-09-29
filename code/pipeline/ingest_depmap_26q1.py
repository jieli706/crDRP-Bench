# -*- coding: utf-8 -*-
"""Version-anchored expression intake, no fitted transforms or outcome-based filtering."""
import csv,hashlib,json,logging,re
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'data/raw/depmap'
OUT=ROOT/'data/processed'/('depmap_26q1_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir(parents=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
def hashes(p):
    h=hashlib.sha256();m=hashlib.md5()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b);m.update(b)
    return {'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'md5':m.hexdigest()}
modelpath=RAW/'Model.csv';exprpath=RAW/'OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv'
sources=[hashes(modelpath),hashes(exprpath)]
assert sources[0]['md5']=='a15d75dffcc5219111ca39598948df9a','Model version mismatch with archived original download checksum'
model=pd.read_csv(modelpath);assert model.ModelID.is_unique
with exprpath.open(encoding='utf-8-sig',newline='') as f:header=next(csv.reader(f))
if len(header)!=len(set(header)):raise ValueError('Duplicate source column names')
metadata=['SequencingID','ModelConditionID','ModelID','IsDefaultEntryForMC','IsDefaultEntryForModel'];cols=[c for c in header if c.strip() and c not in metadata and not c.startswith('Unnamed')]
gene_rows=[]
for i,name in enumerate(cols):
    match=re.fullmatch(r'(.+) \((\d+)\)',name)
    if not match:raise ValueError('Unexpected gene column: '+name)
    gene_rows.append({'raw_column_index':i,'source_column':name,'symbol':match.group(1),'entrez_id':match.group(2)})
genes=pd.DataFrame(gene_rows);genes['duplicate_symbol']=genes.symbol.duplicated(keep=False)
# Retain original distinct gene IDs. No whole-cohort expression-based duplicate selection.
genes.to_csv(OUT/'gene_ids.csv',index=False)
arrays=[];meta=[];total=0;unique_defaults=[];missing=0;negative=0
for chunk in pd.read_csv(exprpath,chunksize=64,low_memory=False):
    total+=len(chunk);chunk=chunk[chunk.IsDefaultEntryForModel.eq('Yes')]
    if not len(chunk):continue
    values=chunk[cols].to_numpy(dtype=np.float32);missing+=int(np.isnan(values).sum());negative+=int((values<0).sum())
    if np.isinf(values).any():raise ValueError('Infinite expression')
    arrays.append(values);meta.append(chunk[metadata]);logging.info('rows read=%s retained=%s',total,sum(len(x) for x in arrays))
sample=pd.concat(meta,ignore_index=True);x=np.concatenate(arrays)
if sample.ModelID.duplicated().any():
    sample[sample.ModelID.duplicated(keep=False)].to_csv(OUT/'duplicate_default_entries.csv',index=False)
    raise ValueError('Multiple default entries per model; explicit adjudication required')
sample=sample.merge(model,on='ModelID',how='left',validate='one_to_one',indicator=True)
sample['known_breast']=sample.OncotreeLineage.eq('Breast');sample['known_cancer']=sample.OncotreePrimaryDisease.notna()&sample.OncotreePrimaryDisease.ne('Non-Cancerous');sample['row_index']=np.arange(len(sample))
sample.to_csv(OUT/'sample_ids_and_metadata.csv',index=False);np.save(OUT/'expression_log2TPM1.npy',x)
sample[sample.known_breast].to_csv(OUT/'all_breast_expression_samples.csv',index=False)
summary={'release':'DepMap Public 26Q1','manual_download_evidence':'user screenshot showed 26Q1; Model checksum equals archived expected value; expression publisher checksum not yet independently obtained','raw_expression_rows':total,'default_model_rows':len(sample),'raw_gene_columns':len(cols),'duplicate_gene_symbols':int(genes[genes.duplicate_symbol].symbol.nunique()),'missing_expression_values':missing,'negative_expression_values':negative,'expression_models_missing_metadata':int(sample._merge.ne('both').sum()),'breast_with_expression':int(sample.known_breast.sum()),'breast_cancer_with_expression':int((sample.known_breast&sample.known_cancer).sum()),'scale':'log2(TPM+1), per provider column specification; linear TPM = 2**x - 1','no_scaling_imputation_or_PCA_fit':True,'no_GDSC_intersection_yet':True}
(OUT/'provenance.json').write_text(json.dumps({'sources':sources,'summary':summary},indent=2),encoding='utf-8')
(OUT/'COMPLETE.json').write_text(json.dumps({'status':'completed_expression_intake','not_drug_response_benchmark':True},indent=2),encoding='utf-8')
print(json.dumps(summary));print(OUT)
