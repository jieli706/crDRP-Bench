"""Independent CPU replay, split checks and preservation audit; creates a new run."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='2'
import ast,csv,hashlib,importlib.util,json,pickle,argparse
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from leakage_safe import assert_disjoint
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--bayes-run',required=True);p.add_argument('--correction-index');p.add_argument('--transfer-run');args=p.parse_args()
O=ROOT/'runs'/('real_stage_verification_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir()
torch.set_num_threads(2)
def module(name,file):
    sp=importlib.util.spec_from_file_location(name,ROOT/'source_snapshot/scripts'/file);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
cr=module('cr','07_crdrp_model.py');tf=module('tf','08_transformer_model.py')
runs=[ROOT/'runs'/x for x in ['real_gdsc_cell_20260926_142546','real_gdsc_pair_20260926_144248','real_gdsc_drug_20260926_145210','real_gdsc_cell_20260926_145959']]+[Path(args.bayes_run).resolve()]
replay=[];split_checks=[];coverage=[];stability=[]
if args.correction_index:
    fixes=json.loads(Path(args.correction_index).read_text())
    runs.extend(Path(fixes[k]) for k in ['cell','pair','drug'])
for run in runs:
    assert (run/'COMPLETE.json').exists(),str(run)
    cfg=json.loads((run/'config.json').read_text());F=Path(cfg['input_directory']);fc=json.loads((F/'config.json').read_text())
    pairs=pd.read_csv(run/'selected_response_pairs.csv');pred=pd.read_csv(run/'oof_predictions.csv');bulk=np.load(F/fc.get('bulk_array','bulk_log2TPM1.npy'));z=np.load(F/fc.get('Z_array','nnls_Z_log1pCPM.npy'));pi=np.load(F/fc.get('pi_array','nnls_pi.npy'));dr=np.load(F/'drug_raw_features.npy')
    for (seed,method),g in pred.groupby(['seed','method']):
        assert g.pair_id.is_unique and set(g.pair_id)==set(pairs.pair_id)
        coverage.append({'run':run.name,'seed':int(seed),'method':method,'unique_oof_pairs':len(g)})
    for seed in [7,17,37]:
      for fold in [0,1,2]:
        dest=run/f'seed_{seed}'/f'fold_{fold}';split=pd.read_csv(dest/'split.csv');assert np.array_equal(split.pair_id,pairs.pair_id)
        tr=np.flatnonzero(split.role.eq('train'));va=np.flatnonzero(split.role.eq('validation'));te=np.flatnonzero(split.role.eq('test'));assert_disjoint(pairs,tr,va,te,cfg['regime'])
        with (dest/'preprocessing.pkl').open('rb') as f:prep=pickle.load(f)
        tc=set(pairs.iloc[tr].ModelID);td=set(pairs.iloc[tr].drug_dpc)
        assert set(prep['bulk'].fit_ids_)==tc and set(prep['drug'].fit_ids_)==td
        expected={str(c)+'::comp_'+str(k) for c in tc for k in range(8)} if cfg.get('shared_Z_basis') else tc
        assert all(set(t.fit_ids_)==expected for t in prep['Z'])
        split_checks.append({'run':run.name,'seed':seed,'fold':fold,'regime':cfg['regime'],'exact_training_fit_ids':True,'no_heldout_entity_overlap':cfg['regime']!='pair'})
        # Replay every neural checkpoint of the primary cell and matched count branches;
        # pair/drug runs are independently split/coverage audited and already replayed in their run.
        if cfg['regime']!='cell':continue
        bf=prep['bulk'].transform(bulk);df=prep['drug'].transform(dr);zf=np.stack([prep['Z'][k].transform(z[:,k]) for k in range(8)],axis=1)
        if cfg.get('stable_PCA_null_space'):
            for nt in [1,16]:
                with threadpool_limits(limits=nt,user_api='blas'):
                    z2=np.stack([prep['Z'][k].transform(z[:,k]) for k in range(8)],axis=1)
                err=float(np.max(abs(z2-zf)));assert err<1e-5,('unstable corrected features',err)
                stability.append({'run':run.name,'seed':seed,'fold':fold,'blas_threads':nt,'max_feature_difference':err})
        for file in dest.glob('*.pt'):
            with file.open('rb') as f:ck=torch.load(f,map_location='cpu')
            method=ck['method'];q=pd.read_csv(dest/(method+'_predictions.csv'));joined=q.merge(pairs[['pair_id','cell_row','drug_row']],on='pair_id',validate='one_to_one',sort=False);ci=joined.cell_row.to_numpy();di=joined.drug_row.to_numpy();dx=torch.tensor(df[di])
            if method.startswith('crdrp_full_'):model=cr.CrDRP(z_dim=ck['z_dim'],drug_dim=ck['drug_dim'])
            elif method=='crdrp_bulk':model=cr.BulkDRP(bulk_dim=ck['bulk_dim'],drug_dim=ck['drug_dim'])
            elif method.startswith('transformer_full_'):model=tf.TransformerDRP(z_dim=ck['z_dim'],drug_dim=ck['drug_dim'])
            else:model=tf.TransformerBulkDRP(bulk_dim=ck['bulk_dim'],drug_dim=ck['drug_dim'])
            model.load_state_dict(ck['state_dict']);model.eval()
            with torch.no_grad():
                yp=model(torch.tensor(pi[ci]),torch.tensor(zf[ci]),dx) if 'full' in method else model(torch.tensor(bf[ci]),dx)
                if method.startswith('crdrp_full_'):yp=yp[0]
            err=float(np.max(abs(yp.numpy()-joined.y_pred.to_numpy())))
            historical_unstable=run.name=='real_gdsc_cell_20260926_142546' and 'full' in method
            if not historical_unstable:assert err<2e-5,(run,method,err)
            replay.append({'run':run.name,'seed':seed,'fold':fold,'method':method,'CPU_replay_max_abs_error':err,'historical_unstable_PCA':historical_unstable,'passed':err<2e-5})
transfer=Path(args.transfer_run) if args.transfer_run else ROOT/'runs/transfer_gdsc_20260926_150455';target=pd.read_csv(transfer/'oof_predictions.csv');all_breast=set(pd.read_csv(ROOT/'data/processed/depmap_26q1_20260925_161400/sample_ids_and_metadata.csv').query('known_breast == True').ModelID)
for f in transfer.glob('seed_*/fold_*/source_split_and_responses.csv'):
    s=pd.read_csv(f);assert not set(s.ModelID)&all_breast;assert not set(s.query('role == "source_train"').ModelID)&set(s.query('role == "source_validation"').ModelID)
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
old=list(csv.DictReader((ROOT/'audit/file_inventory.csv').open(encoding='utf-8-sig')));changed=[]
for r in old:
    f=ROOT.parents[1]/r['path']
    if not f.exists() or f.stat().st_size!=int(r['bytes']) or f.stat().st_mtime_ns!=int(r['mtime_ns']):changed.append(r['path'])
prior=list(csv.DictReader((ROOT/'audit/analysis_artifact_sha256.csv').open(encoding='utf-8-sig')));mismatch=[]
for r in prior:
    f=ROOT/r['path']
    if not f.exists() or sha(f)!=r['sha256']:mismatch.append(r['path'])
for f in (ROOT/'code').glob('*.py'):ast.parse(f.read_text(encoding='utf-8-sig'))
pd.DataFrame(replay).to_csv(O/'CPU_checkpoint_replay.csv',index=False);pd.DataFrame(split_checks).to_csv(O/'split_and_fit_checks.csv',index=False);pd.DataFrame(coverage).to_csv(O/'OOF_coverage.csv',index=False)
pd.DataFrame(stability).to_csv(O/'corrected_features_BLAS_thread_stability.csv',index=False)
res={'original_project_files_checked':len(old),'original_project_size_mtime_changes':changed,'prior_analysis_hashes_checked':len(prior),'prior_analysis_hash_mismatches':mismatch,'replayed_neural_checkpoints':len(replay),'max_CPU_replay_error_current_valid':max(q['CPU_replay_max_abs_error'] for q in replay if not q['historical_unstable_PCA']),'historical_unstable_replay_failures':[q for q in replay if not q['passed']],'split_folds_independently_checked':len(split_checks),'source_training_and_validation_disjoint':True,'all_breast_models_excluded_from_pretraining':True,'all_python_syntax_passed':True}
(O/'verification.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False));print(O)
