"""Expand only unambiguous structures using fixed ChEMBL 37; preserve first pilot."""
import csv,gzip,hashlib,json,shutil,sys
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime/packages310'))
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from mordred import Calculator,descriptors
RDLogger.DisableLog('rdApp.warning')
F=ROOT/'data/processed/real_gdsc_features_20260926_135410';C=ROOT/'data/raw/chembl37';O=ROOT/'data/processed'/('expanded_gdsc_features_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir()
ids=pd.read_csv(C/'name_resolution/resolution_summary.csv');valid=ids[ids.status.eq('unique_exact_alias')];wanted=set(valid.chembl_ids);found={}
with gzip.open(C/'chembl_37_chemreps.txt.gz','rt') as f:
    for row in csv.DictReader(f,delimiter='\t'):
        if row['chembl_id'] in wanted:
            assert row['chembl_id'] not in found;found[row['chembl_id']]=row
pd.DataFrame(found.values()).to_csv(O/'matched_ChEMBL37_records.csv',index=False)
old=pd.read_csv(F/'drug_ids.csv');lookup=pd.read_csv(F/'drug_structure_resolution.csv');lookup['structure_source']=np.where(lookup.match_status.eq('unique_structure'),'PRISM19Q4','unresolved');lookup['chembl_id']=lookup.drug_dpc.map(valid.set_index('drug_dpc').chembl_ids)
chooser=rdMolStandardize.LargestFragmentChooser();uncharger=rdMolStandardize.Uncharger();calc=Calculator(descriptors,ignore_3D=True)
for ix,row in lookup[lookup.chembl_id.notna()].iterrows():
    if row.chembl_id not in found:continue
    m=Chem.MolFromSmiles(found[row.chembl_id]['canonical_smiles'])
    if m is None:continue
    m=uncharger.uncharge(chooser.choose(m));Chem.Kekulize(m,clearAromaticFlags=False);lookup.loc[ix,'canonical_smiles']=Chem.MolToSmiles(m,canonical=True);lookup.loc[ix,'match_status']='unique_structure';lookup.loc[ix,'structure_source']='ChEMBL37_exact_OpenTargets_alias';lookup.loc[ix,'n_distinct_structures']=1
lookup.to_csv(O/'drug_structure_resolution.csv',index=False);selected=lookup[lookup.match_status.eq('unique_structure')].sort_values('drug_dpc').reset_index(drop=True);selected.to_csv(O/'drug_ids.csv',index=False)
old_raw=np.load(F/'drug_raw_features.npy');oldmap=dict(zip(old.drug_dpc,range(len(old))));allfeatures=[]
for i,row in selected.iterrows():
    if row.drug_dpc in oldmap:feat=old_raw[oldmap[row.drug_dpc]]
    else:
        m=Chem.MolFromSmiles(row.canonical_smiles);fp=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048);bits=np.zeros(2048,dtype='float32');DataStructs.ConvertToNumpyArray(fp,bits);v=[]
        for value in calc(m):
            try:value=float(value)
            except (ValueError,TypeError):value=np.nan
            v.append(value if np.isfinite(value) else np.nan)
        feat=np.r_[bits,np.array(v,dtype='float32')]
    allfeatures.append(feat)
    if i%25==0:print('Descriptors',i+1,'/',len(selected),flush=True)
np.save(O/'drug_raw_features.npy',np.stack(allfeatures))
for name in ['sample_ids.csv','bulk_log2TPM1.npy','nnls_pi.npy','nnls_Z_log1pCPM.npy','NNLS_reference_profiles.npy','NNLS_reference_genes.csv','NNLS_cellline_diagnostics.csv','drug_feature_names.csv']:
    shutil.copy2(F/name,O/name)
pairs=pd.read_csv(F/'response_structure_filter.csv');pairs['has_structure']=pairs.drug_dpc.isin(selected.drug_dpc);pairs['drug_row']=pairs.drug_dpc.map(dict(zip(selected.drug_dpc,selected.index)));cells=pd.read_csv(F/'sample_ids.csv');pairs['cell_row']=pairs.ModelID.map(dict(zip(cells.ModelID,cells.cell_row)));pairs.to_csv(O/'response_structure_filter.csv',index=False);use=pairs[pairs.has_structure].copy();use['drug_row']=use.drug_row.astype(int);use.to_csv(O/'response_pairs.csv',index=False)
cfg=json.loads((F/'config.json').read_text());cfg.update(n_drugs_with_unique_structure=len(selected),n_pairs=len(use),original_feature_directory=str(F.relative_to(ROOT)),drug_source='PRISM19Q4 plus official ChEMBL37 canonical smiles mapped via unique exact name/synonym in archived OpenTargets response; ambiguous IDs excluded; all original 54 drug feature rows reused',ChEMBL37_sha256=hashlib.sha256((C/'chembl_37_chemreps.txt.gz').read_bytes()).hexdigest());(O/'config.json').write_text(json.dumps(cfg,indent=2));(O/'source_hashes.json').write_text(json.dumps([{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [Path(__file__),F/'source_hashes.json',C/'name_resolution/resolution_summary.csv',O/'matched_ChEMBL37_records.csv']],indent=2));(O/'COMPLETE.json').write_text(json.dumps({'status':'real_expanded_features_ready','n_drugs':len(selected),'n_pairs':len(use)}));print(O)
