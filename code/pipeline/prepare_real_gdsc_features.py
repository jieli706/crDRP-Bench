"""Real GDSC AUC, real expression and chemical features for an explicit NNLS pilot.
No fitted cohort-wide drug preprocessing. This is not the archived BayesPrism pipeline.
"""
import sys,json,hashlib,logging,importlib.metadata
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime/packages310'))
import numpy as np,pandas as pd
from scipy.optimize import nnls
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from mordred import Calculator,descriptors
RDLogger.DisableLog('rdApp.warning')
OUT=ROOT/'data/processed'/('real_gdsc_features_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir()
logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
G=ROOT/'data/processed/gdsc_25q2_20260926_135058';E=ROOT/'data/processed/depmap_26q1_20260925_161400';A=ROOT/'runs/atlas_nnls_pilot_20260925_143102'
pairs=pd.read_csv(G/'breast_response_AUC.csv');dm=pd.read_csv(G/'drug_ids_and_names.csv');dm=dm[dm.drug_dpc.isin(pairs.drug_dpc)].copy()
src=ROOT/'data/raw/prism/secondary-screen-replicate-treatment-info.csv'
pr=pd.read_csv(src,usecols=['broad_id','name','smiles']).drop_duplicates().reset_index(drop=True)
pr['source_row']=np.arange(len(pr));pr['broad_id_13']=pr.broad_id.str[:13];pr['name_upper']=pr['name'].str.upper().str.strip()
chooser=rdMolStandardize.LargestFragmentChooser();uncharger=rdMolStandardize.Uncharger()
def canonical(s):
    try:
        m=Chem.MolFromSmiles(s)
        if m is None:return None
        m=uncharger.uncharge(chooser.choose(m));Chem.Kekulize(m,clearAromaticFlags=False)
        return Chem.MolToSmiles(m,canonical=True)
    except Exception:return None
cache={}
def structures(v):
    if not isinstance(v,str):return set()
    found=set()
    for s in set(v.split(',')):
        s=s.strip()
        if s not in cache:cache[s]=canonical(s)
        if cache[s]:found.add(cache[s])
    return found
pr['canonical_candidates']=pr.smiles.map(structures)
resolved=[]
for r in dm.itertuples(index=False):
    byid=pr[pr.broad_id_13.eq(r.broad_id_13)] if pd.notna(r.broad_id_13) else pr.iloc[:0]
    byname=pr[pr.name_upper.eq(r.drug_name_upper)]
    cand=pd.concat([byid,byname]).drop_duplicates('source_row');ss=set().union(*cand.canonical_candidates.tolist()) if len(cand) else set()
    resolved.append({'drug_dpc':r.drug_dpc,'drug_name':r.drug_name,'broad_id_13':r.broad_id_13,
      'canonical_smiles':next(iter(ss)) if len(ss)==1 else None,'n_distinct_structures':len(ss),'matched_PRISM_metadata_rows':';'.join(map(str,cand.source_row)),
      'match_status':'unique_structure' if len(ss)==1 else ('ambiguous_structure' if ss else 'no_structure_in_PRISM_19Q4'),
      'matched_by_broad_id':len(byid)>0,'matched_by_name':len(byname)>0})
lookup=pd.DataFrame(resolved);lookup.to_csv(OUT/'drug_structure_resolution.csv',index=False)
selected=lookup[lookup.match_status.eq('unique_structure')].sort_values('drug_dpc').reset_index(drop=True)
selected.to_csv(OUT/'drug_ids.csv',index=False)
logging.info('PRISM structure coverage %d/%d GDSC drugs',len(selected),len(dm))
mols=[Chem.MolFromSmiles(s) for s in selected.canonical_smiles];calc=Calculator(descriptors,ignore_3D=True)
features=[];qc=[]
for i,m in enumerate(mols):
    fp=AllChem.GetMorganFingerprintAsBitVect(m,radius=2,nBits=2048);bits=np.zeros(2048,dtype='float32');DataStructs.ConvertToNumpyArray(fp,bits)
    result=calc(m);vals=[]
    for v in result:
        try:v=float(v)
        except (TypeError,ValueError):v=np.nan
        vals.append(v if np.isfinite(v) else np.nan)
    features.append(np.r_[bits,np.array(vals,dtype='float32')]);qc.append({'drug_dpc':selected.drug_dpc.iloc[i],'missing_mordred':int(np.isnan(vals).sum())})
    if i%25==0:logging.info('Chemical features %d/%d',i+1,len(mols))
drug=np.stack(features);np.save(OUT/'drug_raw_features.npy',drug);pd.DataFrame(qc).to_csv(OUT/'descriptor_missingness.csv',index=False)
pd.DataFrame({'feature':['Morgan_'+str(i) for i in range(2048)]+[str(x) for x in calc.descriptors]}).to_csv(OUT/'drug_feature_names.csv',index=False)
meta=pd.read_csv(E/'sample_ids_and_metadata.csv');genes=pd.read_csv(E/'gene_ids.csv');cellids=np.sort(pairs.ModelID.unique());cm=meta.set_index('ModelID').loc[cellids].reset_index();cm['cell_row']=np.arange(len(cm));cm.to_csv(OUT/'sample_ids.csv',index=False)
expr=np.load(E/'expression_log2TPM1.npy',mmap_mode='r');bulk=np.asarray(expr[cm.row_index],dtype='float32');np.save(OUT/'bulk_log2TPM1.npy',bulk)
# External atlas signatures; drug responses and target expression values do not select genes.
atlas=np.load(A/'donor_compartment_mean_profiles.npz');means=atlas['means'];n=atlas['n_cells'];comp=atlas['compartments']
sig=np.stack([means[n[:,k]>0,k].mean(0) for k in range(8)])
symbol_to_col={s:i for i,s in enumerate(genes.symbol)};ref=np.zeros((8,len(genes)))
matched_counts=np.zeros(len(genes),int)
for j,s in enumerate(atlas['gene_symbols']):
    if s in symbol_to_col:ix=symbol_to_col[s];ref[:,ix]+=sig[:,j];matched_counts[ix]+=1
score=np.var(ref,axis=0);score[(matched_counts==0)|genes.symbol.str.startswith('MT-').to_numpy()]=-np.inf
selected_genes=np.argsort(-score,kind='stable')[:2000];ref=ref[:,selected_genes];ref=ref/ref.sum(1,keepdims=True)
gg=genes.iloc[selected_genes].copy();gg['atlas_genes_collapsed']=matched_counts[selected_genes];gg.to_csv(OUT/'NNLS_reference_genes.csv',index=False)
tpm=np.expm1(np.asarray(bulk[:,selected_genes],dtype='float64')*np.log(2));normalized=tpm/tpm.sum(1,keepdims=True)
pi=[];reconstruction=[]
for x in normalized:
    v=nnls(ref.T,x,maxiter=200)[0];v=v/v.sum();pi.append(v);reconstruction.append(np.linalg.norm(x-v@ref)/np.linalg.norm(x))
pi=np.array(pi);numer=pi[:,:,None]*ref[None];den=numer.sum(1,keepdims=True)
allocated=np.divide(numer,den,out=np.zeros_like(numer),where=den>0)*normalized[:,None,:]
mass=allocated.sum(2,keepdims=True);z=np.divide(allocated,mass,out=np.zeros_like(allocated),where=mass>0)
np.save(OUT/'nnls_pi.npy',pi.astype('float32'));np.save(OUT/'nnls_Z_log1pCPM.npy',np.log1p(z*1e6).astype('float32'));np.save(OUT/'NNLS_reference_profiles.npy',ref)
pq=pd.DataFrame(pi,columns=comp);pq.insert(0,'ModelID',cellids);pq['relative_reconstruction_error']=reconstruction;pq.to_csv(OUT/'NNLS_cellline_diagnostics.csv',index=False)
pairs['has_structure']=pairs.drug_dpc.isin(selected.drug_dpc);pairs.to_csv(OUT/'response_structure_filter.csv',index=False)
use=pairs[pairs.has_structure].copy();use['cell_row']=use.ModelID.map(dict(zip(cellids,np.arange(len(cellids)))));use['drug_row']=use.drug_dpc.map(dict(zip(selected.drug_dpc,selected.index)));use.to_csv(OUT/'response_pairs.csv',index=False)
cfg={'scope':'real_GDSC_AUC_NNLS_variant_without_pretraining','original_BayesPrism_pipeline':False,'reference':'external Wu author-label donor-balanced UMI signatures; no cell-line samples in reference','NNLS_scale':'selected-gene normalized inverse log2(TPM+1); UMI reference and bulk TPM have platform/length differences','NNLS_Z':'conditional reference-weighted count allocation, compartment-wise normalized; not BayesPrism Z','drug_source':'PRISM Repurposing 19Q4, unique canonical structures via Broad ID or exact uppercase name; no new ChEMBL version substituted','drug_features':'Morgan2048 radius2 + all raw Mordred 2D, missing values retained for fold-only processing','preprocessing_fitted':False,'n_cells':len(cellids),'n_drugs_before_structure':len(dm),'n_drugs_with_unique_PRISM_structure':len(selected),'n_pairs':len(use),'drug_feature_dimensions':drug.shape[1],'atlas_matched_symbols':int((matched_counts>0).sum()),'NNLS_genes':len(selected_genes),'seeds_for_followup':[7,17,37],'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','rdkit','mordredcommunity','scikit-learn','torch','xgboost']}}
(OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
paths=[src,G/'breast_response_AUC.csv',G/'drug_ids_and_names.csv',E/'expression_log2TPM1.npy',A/'donor_compartment_mean_profiles.npz',Path(__file__)]
(OUT/'source_hashes.json').write_text(json.dumps([{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],indent=2),encoding='utf-8')
(OUT/'COMPLETE.json').write_text(json.dumps({'status':'real_features_ready','n_cells':len(cellids),'n_drugs':len(selected),'n_pairs':len(use),'BayesPrism_features':False},indent=2),encoding='utf-8')
print(json.dumps(cfg,indent=2));print(OUT)
