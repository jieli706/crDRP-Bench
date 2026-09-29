"""Measured DepMap counts and measured external reference, matched inputs for NNLS/BP."""
import hashlib,json
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from scipy.io import mmread,mmwrite
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'data/processed/expanded_gdsc_features_20260926_142255';E=ROOT/'data/processed/depmap_26q1_20260925_161400';S=ROOT/'runs/paired_deconvolution_inputs_20260926_130450/seed_7';R=ROOT/'data/raw/depmap';O=ROOT/'data/processed'/('real_count_deconvolution_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir()
cells=pd.read_csv(F/'sample_ids.csv');eg=pd.read_csv(E/'gene_ids.csv');sg=pd.read_csv(S/'gene_ids.csv');sg['source_column']=sg.symbol.map(eg.set_index('symbol').source_column);sg['duplicated_atlas_symbol']=sg.symbol.duplicated(keep=False);sg['use']=sg.source_column.notna()&~sg.duplicated_atlas_symbol;sg.to_csv(O/'gene_selection.csv',index=False);g=sg[sg.use].copy();assert g.source_column.is_unique
cols=['SequencingID','ModelID','IsDefaultEntryForModel']+g.source_column.tolist();parts=[];audit=[]
for c in pd.read_csv(R/'OmicsExpressionRawReadCountHumanProteinCodingGenes.csv',usecols=cols,chunksize=64):
    hit=c[c.ModelID.isin(cells.ModelID)];audit.append(hit[cols[:3]]);parts.append(hit[hit.IsDefaultEntryForModel.eq('Yes')])
raw=pd.concat(parts);pd.concat(audit).to_csv(O/'default_count_sample_selection.csv',index=False);assert raw.ModelID.is_unique;raw=raw.set_index('ModelID').loc[cells.ModelID];assert np.array_equal(raw.SequencingID,cells.SequencingID)
x=raw[g.source_column].to_numpy(float);assert np.isfinite(x).all() and (x>=0).all() and np.equal(x,np.rint(x)).all()
pd.DataFrame(x,index=cells.ModelID,columns=g.gene_id).rename_axis('ModelID').to_csv(O/'mixture_counts.csv');cells.to_csv(O/'sample_ids.csv',index=False)
ref=mmread(str(S/'reference_counts_cells_by_genes.mtx')).tocsr()[:,np.flatnonzero(sg.use)];lab=pd.read_csv(S/'reference_cell_labels.csv');present=np.asarray(ref.sum(1)).ravel()>0;ref=ref[present];lab=lab.loc[present];lab.to_csv(O/'reference_cell_labels.csv',index=False);mmwrite(str(O/'reference_counts_cells_by_genes.mtx'),ref,field='integer');g[['gene_id','symbol']].to_csv(O/'gene_ids.csv',index=False)
comp=np.load(S/'paired_inputs.npz')['compartments'];sig=np.stack([np.asarray(ref[lab.compartment.eq(k).to_numpy()].sum(0)).ravel() for k in comp]);sig=sig/sig.sum(1,keepdims=True);pi=[]
for row in x:
    v=nnls(sig.T,row/row.sum(),maxiter=200)[0];pi.append(v/v.sum())
pi=np.array(pi);num=pi[:,:,None]*sig[None];den=num.sum(1,keepdims=True);alloc=np.divide(num,den,out=np.zeros_like(num),where=den>0)*x[:,None,:];mass=alloc.sum(2,keepdims=True);z=np.divide(alloc,mass,out=np.zeros_like(alloc),where=mass>0)
np.savez_compressed(O/'real_count_inputs.npz',counts=x,nnls_pi=pi,nnls_Z_log1pCPM=np.log1p(z*1e6),compartments=comp,cell_ids=cells.ModelID.to_numpy(dtype=str),gene_ids=g.gene_id.to_numpy(dtype=str),reference_profiles=sig)
pd.DataFrame(pi,index=cells.ModelID,columns=comp).to_csv(O/'nnls_fractions.csv')
info={'version':'DepMap Public 26Q1 raw read counts','n_cells':len(cells),'n_genes':len(g),'n_reference_cells':len(lab),'reference_donors':int(lab.donor_id.nunique()),'reference_source':str(S.relative_to(ROOT)),'sample_SequencingID_matches_original_TPM':True,'raw_integer_counts':True,'use':'NNLS and BayesPrism share raw count mixtures, reference cells and genes; no outcome-dependent selection; different input/reference branch from TPM 26-donor NNLS pilot','limitation':'bulk RNA-seq read counts vs single-cell UMI counts still differ by platform/gene length; no independent atlas and no true cell-line fractions','source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'OmicsExpressionRawReadCountHumanProteinCodingGenes.csv',S/'reference_counts_cells_by_genes.mtx',S/'gene_ids.csv',Path(__file__)]}}
(O/'config.json').write_text(json.dumps(info,indent=2));(O/'COMPLETE.json').write_text(json.dumps({'status':'real_count_matched_inputs_ready','n_cells':len(cells),'n_genes':len(g)}));print(json.dumps(info,indent=2));print(O)
