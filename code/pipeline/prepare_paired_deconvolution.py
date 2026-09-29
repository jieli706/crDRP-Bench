"""Freeze donor-isolated count mixtures for a paired NNLS/BayesPrism/oracle pilot.
Only expression is measured biology. Mixtures and future responses are simulated.
All three methods must consume exactly these same counts and gene IDs.
"""
import json,hashlib
from datetime import datetime
from pathlib import Path
import h5py,numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'runs/atlas_nnls_pilot_20260925_143102'
OUT=ROOT/'runs'/('paired_deconvolution_inputs_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir()
data=np.load(SRC/'donor_compartment_mean_profiles.npz');means=data['means'];numbers=data['n_cells'];donors=data['donor_ids'];comp=data['compartments'];gene_ids=data['gene_ids'];symbols=data['gene_symbols']
meta=pd.read_csv(SRC/'cell_metadata_and_qc.csv');rawpath=ROOT/'data/raw/scrna/wu2021_cellxgene.h5ad'
config={'seeds':[7,17,37],'reference_donors':12,'predictor_train_donors':5,'predictor_validation_donors':2,'predictor_test_donors':7,
 'genes':2000,'max_reference_cells_per_donor_compartment':20,'mixtures_per_predictor_donor':6,'counts_per_mixture':20000,
 'fractions':'Known RNA contribution fractions within selected gene space, after normalizing each donor-compartment profile to sum one. Not measured cell fractions.',
 'NNLS':'Same measured reference cells as BayesPrism; aggregate raw counts within compartment and normalize per compartment; NNLS coefficients normalized to sum one.',
 'oracle':'Known mixture fractions and true donor compartment expression profiles; privileged positive control only.',
 'split_policy':'Reference, predictor training, inner validation and test donor IDs are all disjoint. Gene selection only reference donors.',
 'future_downstream':'Use these frozen IDs for NNLS, BayesPrism and oracle under the same predictor splits; no drug-response results yet.',
 'scope':'Real atlas source, synthetic bulk mixtures; not observed tumors or GDSC responses'}
(OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
allchecks=[]
with h5py.File(rawpath,'r') as f:
    assert np.array_equal(f['obs/index'].asstr()[:],meta.cell_id.to_numpy())
    raw=f['raw/X'];indptr=raw['indptr'][:]
    for seed in config['seeds']:
        out=OUT/f'seed_{seed}';out.mkdir();rng=np.random.default_rng(seed);order=rng.permutation(len(donors))
        te=order[:7];reference=order[7:19];train=order[19:24];val=order[24:]
        roles={str(donors[i]):role for role,ix in [('test',te),('reference',reference),('train',train),('validation',val)] for i in ix}
        pd.DataFrame({'donor_id':donors,'role':[roles[str(d)] for d in donors]}).to_csv(out/'donor_split.csv',index=False)
        sig=np.stack([means[reference[numbers[reference,k]>0],k].mean(0) for k in range(8)])
        score=np.var(sig,axis=0);score[np.char.startswith(symbols.astype(str),'MT-')]=-np.inf
        genes=np.argsort(-score,kind='stable')[:2000];mapping=np.full(len(gene_ids),-1,dtype=int);mapping[genes]=np.arange(len(genes))
        pd.DataFrame({'gene_id':gene_ids[genes],'symbol':symbols[genes]}).to_csv(out/'gene_ids.csv',index=False)
        selected=[]
        eligible=meta[meta.qc_pass&meta.donor_id.isin(donors[reference])]
        for _,g in eligible.groupby(['donor_id','compartment'],sort=True):selected.extend(rng.choice(g.index,min(20,len(g)),replace=False).tolist())
        selected=np.sort(selected);vals=[];cols=[];ptr=[0]
        for row in selected:
            a,b=indptr[row],indptr[row+1];v=raw['data'][a:b];j=mapping[raw['indices'][a:b]];keep=j>=0
            vals.extend(v[keep]);cols.extend(j[keep]);ptr.append(len(vals))
        sc=csr_matrix((np.asarray(vals),np.asarray(cols),np.asarray(ptr)),shape=(len(selected),len(genes)))
        labels=meta.iloc[selected][['cell_id','donor_id','compartment']].copy();labels['cell_type']=labels.compartment.replace({'malignant_epithelial':'epithelial','normal_epithelial':'epithelial'})
        valid=np.asarray(sc.sum(1)).ravel()>0;sc=sc[valid];labels=labels.iloc[np.flatnonzero(valid)]
        assert set(labels.compartment)==set(comp)
        labels.to_csv(out/'reference_cell_labels.csv',index=False);mmwrite(str(out/'reference_counts_cells_by_genes.mtx'),sc,field='integer')
        ref=np.stack([np.asarray(sc[labels.compartment.to_numpy()==k].sum(0)).ravel() for k in comp]);ref=ref/ref.sum(1,keepdims=True)
        profiles=means[:,:,genes].copy();mass=profiles.sum(2,keepdims=True);profiles=np.divide(profiles,mass,out=np.zeros_like(profiles),where=mass>0)
        counts=[];truth=[];ids=[];rows=[];trueZ=[]
        for role,ds in [('train',train),('validation',val),('test',te)]:
            for di in ds:
                present=(numbers[di]>=20)&(mass[di,:,0]>0)
                if present.sum()<2:raise ValueError('Insufficient compartments for donor '+donors[di])
                for j in range(6):
                    pi=np.zeros(8);pi[present]=rng.dirichlet(np.ones(present.sum()))
                    p=pi@profiles[di];p=p/p.sum();x=rng.multinomial(20000,p)
                    mid=f'PAIRED_{seed}_{donors[di]}_{j:02}'
                    counts.append(x);truth.append(pi);trueZ.append(profiles[di]);ids.append(mid)
                    rows.append({'mixture_id':mid,'donor_id':donors[di],'role':role,'n_UMI':int(x.sum())})
        counts=np.asarray(counts);truth=np.asarray(truth);est=[]
        for x in counts:
            v=nnls(ref.T,x/x.sum(),maxiter=200)[0];est.append(v/v.sum())
        est=np.asarray(est)
        pd.DataFrame(counts,index=ids,columns=gene_ids[genes]).rename_axis('mixture_id').to_csv(out/'mixture_counts.csv')
        pd.DataFrame(rows).to_csv(out/'mixture_ids_and_roles.csv',index=False)
        pd.DataFrame(truth,index=ids,columns=comp).rename_axis('mixture_id').to_csv(out/'oracle_fractions.csv')
        pd.DataFrame(est,index=ids,columns=comp).rename_axis('mixture_id').to_csv(out/'nnls_fractions.csv')
        np.savez_compressed(out/'paired_inputs.npz',counts=counts,oracle_pi=truth,oracle_Z=np.asarray(trueZ,dtype='float32'),nnls_pi=est,reference_profiles=ref,mixture_ids=np.asarray(ids),compartments=comp,gene_ids=gene_ids[genes])
        allchecks.append({'seed':seed,'reference_cells':len(labels),'mixtures':len(ids),'reference_donors':len(reference),'predictor_train_donors':len(train),'validation_donors':len(val),'test_donors':len(te),'all_donor_roles_disjoint':len(roles)==len(donors),'integer_nonnegative_counts':bool(np.all(counts>=0)&np.all(counts==np.rint(counts))),'nnls_finite':bool(np.isfinite(est).all()),'max_fraction_sum_error':float(np.max(abs(est.sum(1)-1)))})
        print(allchecks[-1],flush=True)
(OUT/'checks.json').write_text(json.dumps(allchecks,indent=2),encoding='utf-8')
sources=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [SRC/'donor_compartment_mean_profiles.npz',SRC/'cell_metadata_and_qc.csv',Path(__file__)]]
(OUT/'provenance.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
(OUT/'COMPLETE.json').write_text(json.dumps({'status':'paired_inputs_prepared_and_NNLS_run','BayesPrism_completed':False,'downstream_completed':False,'not_real_drug_response':True},indent=2),encoding='utf-8')
print(OUT)
