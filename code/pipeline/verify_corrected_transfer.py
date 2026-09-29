"""Independently replay corrected transfer checkpoints and audit source exclusions."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import argparse,importlib.util,json,pickle
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch
import stable_preprocessing
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();R=Path(a.run).resolve();assert (R/'COMPLETE.json').exists();O=ROOT/'runs'/('transfer_verification_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir();I=ROOT/'data/processed/transfer_inputs_20260926_142944';F=ROOT/'data/processed/expanded_gdsc_features_20260926_142255'
meta=pd.read_csv(I/'sample_ids.csv');cellindex=dict(zip(meta.ModelID,meta.cell_row));drugs=pd.read_csv(F/'drug_ids.csv');drugindex=dict(zip(drugs.drug_dpc,range(len(drugs))));excluded=set(pd.read_csv(I/'all_breast_excluded_from_pretraining.csv').ModelID);target=pd.read_csv(I/'target_response.csv');use=np.sort(target.cell_row.unique());lookup=dict(zip(use,range(len(use))))
bulk=np.load(I/'bulk.npy',mmap_mode='r')[use];z=np.load(I/'Z.npy',mmap_mode='r')[use];pi=np.load(I/'pi.npy')[use];dr=np.load(F/'drug_raw_features.npy');sp=importlib.util.spec_from_file_location('cr',ROOT/'source_snapshot/scripts/07_crdrp_model.py');cr=importlib.util.module_from_spec(sp);sp.loader.exec_module(cr);torch.set_num_threads(2);rows=[]
for path in sorted(R.glob('seed_*/fold_*')):
    source=pd.read_csv(path/'source_split_and_responses.csv');split=pd.read_csv(path/'target_split.csv');tr=set(split.loc[split.role.eq('train'),'ModelID']);assert not set(source.ModelID)&excluded;assert not set(source.loc[source.role.eq('source_train'),'ModelID'])&set(source.loc[source.role.eq('source_validation'),'ModelID'])
    with (path/'preprocessing.pkl').open('rb') as f:t=pickle.load(f)
    assert set(t['bulk'].fit_ids_)==tr and all(set(x.fit_ids_)==tr for x in t['Z']);assert all(hasattr(x,'active_components_') for x in t['Z'])
    bf=t['bulk'].transform(bulk);df=t['drug'].transform(dr);zf=np.stack([t['Z'][k].transform(z[:,k]) for k in range(8)],axis=1)
    for checkpoint in path.glob('*_target.pt'):
        with checkpoint.open('rb') as f:ck=torch.load(f,map_location='cpu')
        method=ck['method'];pred=pd.read_csv(path/(method+'_predictions.csv'));ci=np.array([lookup[cellindex[x]] for x in pred.ModelID]);di=np.array([drugindex[x] for x in pred.drug_dpc]);full='full' in method
        model=cr.CrDRP(z_dim=ck['z_dim'],drug_dim=ck['drug_dim']) if full else cr.BulkDRP(bulk_dim=ck['bulk_dim'],drug_dim=ck['drug_dim']);model.load_state_dict(ck['state_dict']);model.eval()
        with torch.no_grad():yp=model(torch.tensor(pi[ci]),torch.tensor(zf[ci]),torch.tensor(df[di]))[0] if full else model(torch.tensor(bf[ci]),torch.tensor(df[di]))
        err=float(np.max(abs(yp.numpy()-pred.y_pred.to_numpy())));assert err<2e-5,(method,err);rows.append({'seed':ck['seed'],'fold':ck['fold'],'method':method,'CPU_max_difference':err,'no_breast_source':True,'train_only_stable_preprocessing':True})
pd.DataFrame(rows).to_csv(O/'replay_and_exclusion.csv',index=False);result={'run':str(R),'checkpoints_replayed':len(rows),'max_CPU_error':max(x['CPU_max_difference'] for x in rows),'all_source_exclusion_and_fit_ID_checks_passed':True};(O/'COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));print(O)
