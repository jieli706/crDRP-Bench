"""Source-disjoint transfer supplement in the identical fold-fitted primary feature basis."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,importlib.util,json,pickle,hashlib,logging,time
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch
from sklearn.model_selection import GroupShuffleSplit
from leakage_safe import metrics,assert_disjoint
from paired_two_way_bootstrap import paired_bootstrap
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'data/processed/expanded_gdsc_features_20260926_142255';I=ROOT/'data/processed/transfer_inputs_20260926_142944'
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-run');p.add_argument('--reuse-source');p.add_argument('--base-run',required=True);a=p.parse_args();B=Path(a.base_run).resolve()
assert (B/'COMPLETE.json').exists()
if not a.smoke:assert json.loads((Path(a.smoke_run)/'COMPLETE.json').read_text())['all_checks_passed']
O=ROOT/'runs'/('transfer_gdsc_'+('smoke_' if a.smoke else '')+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(O/'run.log'),logging.StreamHandler()])
config={'source':'654 non-breast cancer models, 81440 real GDSC AUC source pairs before fold-specific drug restriction','target':str(B),'preprocessing':'reuse exact primary inner-training-only fitted transforms; no source validation, target validation or target test feature fitting; no source refitting','source_split':'10% by source cell, seed*100+fold; all breast lineage removed globally; target validation/test drugs also excluded for drug holdout','seeds':[7] if a.smoke else [7,17,37],'folds':[0] if a.smoke else [0,1,2],'pretrain_max_epochs':2 if a.smoke else 20,'pretrain_patience':5,'finetune_max_epochs':3 if a.smoke else 120,'finetune_patience':20,'batch':512,'learning_rate':.001,'weight_decay':.0003,'selection_metric':'inner RMSE in respective source or target domain','tuning_candidates':1,'freeze_weights':False,'methods':['crdrp_full_nnls_transfer','crdrp_bulk_transfer'],'scope':'fixed 20-epoch source pretraining budget supplement; not original 60/200-epoch pretraining replication; primary no-pretraining comparison remains separate'}
config.update(source_batch=512,target_batch=256,reused_source_run=a.reuse_source,target_budget_matched_to_primary=True)
(O/'config.json').write_text(json.dumps(config,indent=2));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dev='cuda' if torch.cuda.is_available() else 'cpu'
sp=importlib.util.spec_from_file_location('cr',ROOT/'source_snapshot/scripts/07_crdrp_model.py');cr=importlib.util.module_from_spec(sp);sp.loader.exec_module(cr)
source=pd.read_csv(I/'source_response.csv');target=pd.read_csv(I/'target_response.csv');meta=pd.read_csv(I/'sample_ids.csv');cells=meta.ModelID.to_numpy();assert not meta.loc[meta.role.eq('source'),'known_breast'].any();assert not set(source.ModelID)&set(target.ModelID)
basepairs=pd.read_csv(B/'selected_response_pairs.csv');assert np.array_equal(target.pair_id,basepairs.pair_id)
bulk=np.load(I/'bulk.npy',mmap_mode='r');z=np.load(I/'Z.npy',mmap_mode='r');pi=np.load(I/'pi.npy');drug=np.load(F/'drug_raw_features.npy');preds=[];budgets=[];checks=[]
for seed in config['seeds']:
 for fold in config['folds']:
    dest=O/f'seed_{seed}'/f'fold_{fold}';dest.mkdir(parents=True);base=B/f'seed_{seed}'/f'fold_{fold}';split=pd.read_csv(base/'split.csv');assert np.array_equal(split.pair_id,target.pair_id)
    tr=np.flatnonzero(split.role.eq('train'));va=np.flatnonzero(split.role.eq('validation'));te=np.flatnonzero(split.role.eq('test'));regime=json.loads((B/'config.json').read_text())['regime'];assert_disjoint(target,tr,va,te,regime)
    src=source.copy()
    if regime=='drug':src=src[~src.drug_dpc.isin(target.iloc[np.r_[va,te]].drug_dpc)]
    if a.smoke:src=src[src.ModelID.isin(sorted(src.ModelID.unique())[:60]) & src.drug_dpc.isin(sorted(src.drug_dpc.unique())[:20])]
    src=src.reset_index(drop=True);st,sv=next(GroupShuffleSplit(1,test_size=.1,random_state=seed*100+fold).split(src,groups=src.ModelID));src['role']='source_train';src.loc[sv,'role']='source_validation';src.to_csv(dest/'source_split_and_responses.csv',index=False);split.to_csv(dest/'target_split.csv',index=False)
    with (base/'preprocessing.pkl').open('rb') as f:prep=pickle.load(f)
    assert set(prep['bulk'].fit_ids_)==set(target.iloc[tr].ModelID)
    for t in prep['Z']:assert set(t.fit_ids_)==set(target.iloc[tr].ModelID)
    if regime=='drug':assert not set(prep['drug'].fit_ids_)&set(target.iloc[np.r_[va,te]].drug_dpc)
    bf=prep['bulk'].transform(bulk);df=prep['drug'].transform(drug);zf=np.stack([prep['Z'][k].transform(z[:,k]) for k in range(8)],axis=1);assert np.isfinite(zf).all()
    with (dest/'preprocessing.pkl').open('wb') as f:pickle.dump(prep,f)
    data={}
    for tag,frame in [('source',src),('target',target)]:data[tag]=tuple(torch.tensor(frame[col].to_numpy(),dtype=torch.float32 if col=='auc' else torch.long,device=dev) for col in ['cell_row','drug_row','auc'])
    tp=torch.tensor(pi,device=dev);tz=torch.tensor(zf,device=dev);tb=torch.tensor(bf,device=dev);td=torch.tensor(df,device=dev)
    for method in config['methods']:
        torch.manual_seed(seed*100+fold);np.random.seed(seed*100+fold);isfull='full' in method
        model=(cr.CrDRP(z_dim=zf.shape[2],drug_dim=df.shape[1]) if isfull else cr.BulkDRP(bulk_dim=bf.shape[1],drug_dim=df.shape[1])).to(dev)
        def predict(tag,ix):
            c,d,y=data[tag]
            return model(tp[c[ix]],tz[c[ix]],td[d[ix]])[0] if isfull else model(tb[c[ix]],td[d[ix]])
        def fit_phase(tag,train,valid,epochs,patience):
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);best=float('inf');bestep=-1;history=[];rng=np.random.default_rng(seed*100+fold);labels=data[tag][2]
            for ep in range(epochs):
                model.train();order=rng.permutation(train);losses=[]
                batch=512 if tag=='source' else 256
                for start in range(0,len(order),batch):
                    ix=order[start:start+batch];opt.zero_grad();loss=torch.nn.functional.huber_loss(predict(tag,ix),labels[ix],delta=.05);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss))
                model.eval();sse=0
                with torch.no_grad():
                    for start in range(0,len(valid),512):
                        ix=valid[start:start+512];sse+=float(torch.sum((predict(tag,ix)-labels[ix])**2))
                rmse=np.sqrt(sse/len(valid));history.append({'epoch':ep,'train_huber':float(np.mean(losses)),'validation_rmse':rmse})
                if rmse<best:best=rmse;bestep=ep;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if ep%5==0:logging.info('seed=%s fold=%s %s %s epoch=%s validation_rmse=%.5f',seed,fold,method,tag,ep,rmse)
                if ep-bestep>=patience:break
            model.load_state_dict(state);model.eval();pd.DataFrame(history).to_csv(dest/(method+'_'+tag+'_training.csv'),index=False)
            with (dest/(method+'_'+tag+'.pt')).open('wb') as f:torch.save({'state_dict':state,'seed':seed,'fold':fold,'phase':tag,'method':method,'best_epoch':bestep,'z_dim':zf.shape[2],'bulk_dim':bf.shape[1],'drug_dim':df.shape[1]},f)
            return len(history),bestep
        t0=time.time()
        if a.reuse_source:
            previous=Path(a.reuse_source)/f'seed_{seed}'/f'fold_{fold}'
            saved_split=pd.read_csv(previous/'source_split_and_responses.csv');assert np.array_equal(src.pair_id,saved_split.pair_id) and np.array_equal(src.role,saved_split.role)
            checkpoint=previous/(method+'_source.pt')
            with checkpoint.open('rb') as f:sc=torch.load(f,map_location=dev)
            model.load_state_dict(sc['state_dict']);sp_best=sc['best_epoch'];sp_ep=len(pd.read_csv(previous/(method+'_source_training.csv')))
            (dest/(method+'_source_reused.json')).write_text(json.dumps({'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},indent=2))
        else:sp_ep,sp_best=fit_phase('source',st,sv,config['pretrain_max_epochs'],config['pretrain_patience'])
        ft_ep,ft_best=fit_phase('target',tr,va,config['finetune_max_epochs'],config['finetune_patience'])
        with torch.no_grad():yp=np.concatenate([predict('target',te[j:j+512]).cpu().numpy() for j in range(0,len(te),512)])
        with (dest/(method+'_target.pt')).open('rb') as f:model.load_state_dict(torch.load(f,map_location=dev)['state_dict'])
        with torch.no_grad():replay=np.concatenate([predict('target',te[j:j+512]).cpu().numpy() for j in range(0,len(te),512)])
        assert np.isfinite(yp).all() and np.allclose(yp,replay,atol=1e-6)
        r=target.iloc[te][['ModelID','drug_dpc','pair_id']].copy();r['y_true']=target.auc.iloc[te];r['y_pred']=yp;r['method']=method;r['seed']=seed;r['fold']=fold;r['regime']=regime;preds.append(r);r.to_csv(dest/(method+'_predictions.csv'),index=False)
        budgets.append({'seed':seed,'fold':fold,'method':method,'source_training_pairs':len(st),'source_validation_pairs':len(sv),'source_epochs':sp_ep,'source_selected_epoch':sp_best,'target_epochs':ft_ep,'target_selected_epoch':ft_best,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.time()-t0})
        checks.append({'seed':seed,'fold':fold,'method':method,'no_breast_in_source':True,'train_only_feature_basis_reused':True,'replay_max_error':float(np.max(abs(yp-replay)))})
pred=pd.concat(preds,ignore_index=True);pred.to_csv(O/'oof_predictions.csv',index=False);pd.DataFrame(budgets).to_csv(O/'capacity_and_training_budget.csv',index=False);pd.DataFrame([{'seed':s,'method':m,**metrics(g)} for (s,m),g in pred.groupby(['seed','method'])]).to_csv(O/'metrics.csv',index=False)
if not a.smoke:
    for _,g in pred.groupby(['seed','method']):assert len(g)==len(target) and g.pair_id.is_unique
    primary=pd.read_csv(B/('paired_baseline_predictions.csv' if (B/'paired_baseline_predictions.csv').exists() else 'oof_predictions.csv'));combined=pd.concat([primary,pred],ignore_index=True);combined.to_csv(O/'paired_primary_and_transfer_predictions.csv',index=False);summaries=[]
    for seed in config['seeds']:
     for ma,mb in [('crdrp_full_nnls_transfer','crdrp_bulk_transfer'),('crdrp_full_nnls_transfer','crdrp_full_nnls'),('crdrp_bulk_transfer','crdrp_bulk')]:
        ss,draws,w=paired_bootstrap(combined[combined.seed==seed],ma,mb,1000,seed);ss['seed']=seed;summaries.append(ss);tag=f'{seed}_{ma}_minus_{mb}';draws.to_csv(O/('bootstrap_'+tag+'.csv'),index=False);np.savez_compressed(O/('bootstrap_weights_'+tag+'.npz'),**w)
    pd.concat(summaries).to_csv(O/'paired_two_way_bootstrap.csv',index=False)
(O/'checks.json').write_text(json.dumps(checks,indent=2));(O/'sources.json').write_text(json.dumps([{'path':str(q.relative_to(ROOT)),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in [Path(__file__),I/'config.json',B/'config.json']],indent=2));(O/'COMPLETE.json').write_text(json.dumps({'all_checks_passed':True,'scope':'transfer_smoke' if a.smoke else 'real_source_disjoint_transfer_supplement'}));print(O)
