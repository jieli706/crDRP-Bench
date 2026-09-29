"""Real AUC benchmark. NNLS-input variant, no pretraining; not canonical BayesPrism full."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,importlib.util,importlib.metadata,json,logging,pickle,sys,time
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch,xgboost as xgb
from sklearn.linear_model import Ridge
from leakage_safe import outer_splits,inner_split,assert_disjoint,FoldTransform,metrics,safe_corr
from paired_two_way_bootstrap import paired_bootstrap,verify_bootstrap
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-run');p.add_argument('--features');p.add_argument('--shared-z',action='store_true');p.add_argument('--stable-pca',action='store_true');p.add_argument('--methods',nargs='+');p.add_argument('--baseline-run');p.add_argument('--regime',choices=['cell','pair','drug'],default='cell');a=p.parse_args()
if a.stable_pca:
    from stable_preprocessing import StableFoldTransform as FoldTransform
if not a.smoke:
    if not a.smoke_run:raise ValueError('Full run requires passed real smoke run')
    gate=json.loads((Path(a.smoke_run)/'COMPLETE.json').read_text())
    assert gate['scope']=='real_GDSC_smoke' and gate['all_checks_passed']
F=ROOT/'data/processed/real_gdsc_features_20260926_135410'
if a.features:F=Path(a.features).resolve()
if not a.smoke:assert json.loads((Path(a.smoke_run)/'config.json').read_text())['input_directory']==str(F),'Smoke must validate the same feature release'
if not a.smoke:assert json.loads((Path(a.smoke_run)/'config.json').read_text()).get('shared_Z_basis',False)==a.shared_z
if not a.smoke:assert json.loads((Path(a.smoke_run)/'config.json').read_text()).get('stable_PCA_null_space',False)==a.stable_pca
OUT=ROOT/'runs'/('real_gdsc_'+('smoke' if a.smoke else a.regime)+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir()
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
DEV='cuda' if torch.cuda.is_available() else 'cpu'
def load(name,filename):
    s=importlib.util.spec_from_file_location(name,ROOT/'source_snapshot/scripts'/filename);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cr=load('cr','07_crdrp_model.py');tf=load('tf','08_transformer_model.py')
cfg={'data_kind':'real_GDSC_AUC','input_variant':'NNLS conditional allocation Z; NOT BayesPrism','pretraining':False,'seeds':[7] if a.smoke else [7,17,37],'regime':a.regime,'outer_folds':3,'neural_max_epochs':30 if a.smoke else 120,'patience':10 if a.smoke else 20,'batch_size':256,'learning_rate':.001,'weight_decay':.0003,'huber_delta':.05,'tuning_candidates_per_method':1,'inner_metric':'RMSE','refit_after_selection':False,'bootstrap_replicates':100 if a.smoke else 1000,'smoke_selection':'first 24 sorted ModelIDs and first 12 sorted structure-available drug IDs, no response-value selection','input_directory':str(F),'methods':['crdrp_full_nnls','crdrp_bulk','transformer_full_nnls','transformer_bulk','global_xgb','ridge','drug_mean'],'PCA':'each compartment separately, bulk and drug independently; fit only inner-training entities','XGB':'300 trees max, depth 4, lr .05, early stopping 20 on same inner validation','ridge_alpha':1.,'prediction_clipping':False,'inference':'conditional on fixed trained models and available 54-drug structure subset; no confirmatory equivalence claim'}
cfg.update(stable_PCA_null_space=a.stable_pca,shared_Z_basis=a.shared_z,baseline_run=a.baseline_run,feature_provenance=json.loads((F/'config.json').read_text()),inference='conditional on fixed models and explicitly selected chemical structure coverage; not a confirmatory equivalence test')
cfg['input_variant']=cfg['feature_provenance'].get('input_variant',cfg['input_variant'])
if a.methods:cfg['methods']=a.methods
if a.shared_z:cfg['PCA']='One shared Z PCA fitted only to training cell x compartment profiles; bulk and drug independently fitted only training entities'
(OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
(OUT/'environment.json').write_text(json.dumps({'python':sys.executable,'device':DEV,'packages':{q:importlib.metadata.version(q) for q in ['numpy','pandas','scipy','scikit-learn','torch','xgboost']}},indent=2))
pairs=pd.read_csv(F/'response_pairs.csv');cells=pd.read_csv(F/'sample_ids.csv').ModelID.to_numpy();drugs=pd.read_csv(F/'drug_ids.csv').drug_dpc.to_numpy()
if a.smoke:pairs=pairs[pairs.ModelID.isin(np.sort(cells)[:24]) & pairs.drug_dpc.isin(np.sort(drugs)[:12])]
pairs=pairs.reset_index(drop=True);pairs['pair_id']=pairs.ModelID+'::'+pairs.drug_dpc
assert pairs.pair_id.is_unique and pairs.auc.between(0,1).all() and pairs.auc.notna().all()
pairs.to_csv(OUT/'selected_response_pairs.csv',index=False)
featurecfg=cfg['feature_provenance'];bulk=np.load(F/featurecfg.get('bulk_array','bulk_log2TPM1.npy'));Z=np.load(F/featurecfg.get('Z_array','nnls_Z_log1pCPM.npy'));pi=np.load(F/featurecfg.get('pi_array','nnls_pi.npy'));dr=np.load(F/'drug_raw_features.npy')
c=pairs.cell_row.to_numpy();d=pairs.drug_row.to_numpy();y=pairs.auc.to_numpy(dtype='float32')
assert np.array_equal(cells[c],pairs.ModelID) and np.array_equal(drugs[d],pairs.drug_dpc)
assert np.isfinite(bulk).all() and np.isfinite(Z).all() and np.allclose(pi.sum(1),1,atol=1e-5) and not np.isinf(dr).any()
checks=[];predictions=[];capacities=[];variance=[]
for seed in cfg['seeds']:
 for fold,(outer_train,test) in enumerate(outer_splits(pairs,a.regime,seed,3)):
    tr,va=inner_split(pairs,outer_train,a.regime,seed+fold);assert_disjoint(pairs,tr,va,test,a.regime)
    path=OUT/f'seed_{seed}'/f'fold_{fold}';path.mkdir(parents=True)
    split=pairs[['pair_id','ModelID','drug_dpc']].copy();split['role']='test';split.loc[tr,'role']='train';split.loc[va,'role']='validation';split.to_csv(path/'split.csv',index=False)
    cf=np.sort(pairs.iloc[tr].cell_row.unique());dfit=np.sort(pairs.iloc[tr].drug_row.unique())
    bt=FoldTransform(40,seed=seed).fit(bulk[cf],cells[cf]);dt=FoldTransform(116,max_features=10000,seed=seed).fit(dr[dfit],drugs[dfit]);zt=[FoldTransform(64,seed=seed).fit(Z[cf,k],cells[cf]) for k in range(8)]
    if a.shared_z:
        ids=[str(cid)+'::comp_'+str(k) for cid in cells[cf] for k in range(8)]
        shared=FoldTransform(64,seed=seed).fit(Z[cf].reshape(-1,Z.shape[2]),ids);zt=[shared]*8
    bf=bt.transform(bulk);df=dt.transform(dr);zf=np.stack([zt[k].transform(Z[:,k]) for k in range(8)],axis=1)
    assert set(bt.fit_ids_)==set(pairs.iloc[tr].ModelID) and set(dt.fit_ids_)==set(pairs.iloc[tr].drug_dpc)
    assert all(np.isfinite(q).all() for q in [bf,df,zf])
    with (path/'preprocessing.pkl').open('wb') as f:pickle.dump({'bulk':bt,'drug':dt,'Z':zt},f)
    tx=np.c_[bf[c],df[d]];tp=torch.tensor(pi,device=DEV);tz=torch.tensor(zf,device=DEV);tb=torch.tensor(bf,device=DEV);td=torch.tensor(df,device=DEV);ct=torch.tensor(c,device=DEV);dd=torch.tensor(d,device=DEV);yt=torch.tensor(y,device=DEV)
    train_variance=pairs.iloc[tr].groupby('drug_dpc').auc.agg(['count','std']).reset_index();train_variance['seed']=seed;train_variance['fold']=fold;variance.append(train_variance)
    for method in cfg['methods']:
        torch.manual_seed(seed*100+fold);np.random.seed(seed*100+fold);start=time.time();bestep=0
        if method=='global_xgb':
            model=xgb.XGBRegressor(n_estimators=300,max_depth=4,learning_rate=.05,subsample=1,colsample_bytree=1,n_jobs=2,random_state=seed,objective='reg:squarederror',early_stopping_rounds=20)
            model.fit(tx[tr],y[tr],eval_set=[(tx[va],y[va])],verbose=False);yp=model.predict(tx[test]);(path/(method+'.json')).write_bytes(model.get_booster().save_raw(raw_format='json'));nparams=None;bestep=model.best_iteration
            replay=xgb.XGBRegressor();replay.load_model(bytearray((path/(method+'.json')).read_bytes()));yp2=replay.predict(tx[test])
        elif method=='ridge':
            model=Ridge(alpha=1.).fit(tx[tr],y[tr]);yp=model.predict(tx[test]);nparams=len(model.coef_)+1
            with (path/(method+'.pkl')).open('wb') as f:pickle.dump(model,f)
            with (path/(method+'.pkl')).open('rb') as f:yp2=pickle.load(f).predict(tx[test])
        elif method=='drug_mean':
            means=pairs.iloc[tr].groupby('drug_dpc').auc.mean();fallback=float(y[tr].mean());yp=pairs.iloc[test].drug_dpc.map(means).fillna(fallback).to_numpy();nparams=len(means)+1;yp2=yp.copy()
            (path/'drug_mean.json').write_text(json.dumps({'means':means.to_dict(),'fallback':fallback},indent=2))
        else:
            if method.startswith('crdrp_full_'):model=cr.CrDRP(z_dim=zf.shape[2],drug_dim=df.shape[1])
            elif method=='crdrp_bulk':model=cr.BulkDRP(bulk_dim=bf.shape[1],drug_dim=df.shape[1])
            elif method.startswith('transformer_full_'):model=tf.TransformerDRP(z_dim=zf.shape[2],drug_dim=df.shape[1])
            else:model=tf.TransformerBulkDRP(bulk_dim=bf.shape[1],drug_dim=df.shape[1])
            model=model.to(DEV);nparams=sum(p.numel() for p in model.parameters());opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
            def predict(ix):
                if 'full' in method:
                    out=model(tp[ct[ix]],tz[ct[ix]],td[dd[ix]]);return out[0] if method.startswith('crdrp') else out
                return model(tb[ct[ix]],td[dd[ix]])
            best=float('inf');bestep=-1;history=[];batchrng=np.random.default_rng(seed*100+fold)
            for epoch in range(cfg['neural_max_epochs']):
                model.train();order=batchrng.permutation(tr);losses=[]
                for j in range(0,len(order),256):
                    ix=order[j:j+256];opt.zero_grad();loss=torch.nn.functional.huber_loss(predict(ix),yt[ix],delta=.05);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
                model.eval()
                with torch.no_grad():vr=float(torch.sqrt(torch.mean((predict(va)-yt[va])**2)))
                history.append({'epoch':epoch,'train_huber':float(np.mean(losses)),'validation_rmse':vr})
                if vr<best:best=vr;bestep=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if epoch-bestep>=cfg['patience']:break
            model.load_state_dict(state);model.eval()
            with torch.no_grad():yp=predict(test).cpu().numpy()
            checkpoint=path/(method+'.pt')
            with checkpoint.open('wb') as f:torch.save({'state_dict':state,'method':method,'seed':seed,'fold':fold,'best_epoch':bestep,'z_dim':zf.shape[2],'bulk_dim':bf.shape[1],'drug_dim':df.shape[1],'pretraining':False},f)
            with checkpoint.open('rb') as f:model.load_state_dict(torch.load(f,map_location=DEV)['state_dict'])
            with torch.no_grad():yp2=predict(test).cpu().numpy()
            pd.DataFrame(history).to_csv(path/(method+'_training.csv'),index=False)
        assert np.isfinite(yp).all() and np.allclose(yp,yp2,atol=1e-6)
        checks.append({'seed':seed,'fold':fold,'method':method,'split_disjoint':True,'preprocessing_training_only':True,'finite_predictions':True,'checkpoint_max_error':float(np.max(abs(yp-yp2)))})
        r=pairs.iloc[test][['ModelID','drug_dpc','pair_id']].copy();r['y_true']=y[test];r['y_pred']=yp;r['method']=method;r['seed']=seed;r['fold']=fold;r['regime']=a.regime;predictions.append(r);r.to_csv(path/(method+'_predictions.csv'),index=False)
        capacities.append({'seed':seed,'fold':fold,'method':method,'trainable_parameters':nparams,'cell_input_dimensions':8*(zf.shape[2]+1) if 'full' in method else bf.shape[1],'drug_dimensions':df.shape[1],'training_pairs':len(tr),'validation_pairs':len(va),'test_pairs':len(test),'best_epoch_zero_indexed':bestep,'seconds':time.time()-start,'tuning_candidates':1})
        logging.info('seed=%s fold=%s method=%s %.1fs',seed,fold,method,time.time()-start)
allpred=pd.concat(predictions,ignore_index=True);allpred.to_csv(OUT/'oof_predictions.csv',index=False)
rows=[];perdrug=[]
for (seed,method),g in allpred.groupby(['seed','method']):
    assert len(g)==len(pairs) and g.pair_id.is_unique
    rows.append({'seed':seed,'method':method,**metrics(g)})
    for drug,h in g.groupby('drug_dpc'):perdrug.append({'seed':seed,'method':method,'drug_dpc':drug,'n':len(h),'response_std':h.y_true.std(),'prediction_std':h.y_pred.std(),'pearson':safe_corr(h.y_true,h.y_pred),'rmse':np.sqrt(np.mean((h.y_true-h.y_pred)**2))})
pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(perdrug).to_csv(OUT/'per_drug_metrics.csv',index=False);pd.DataFrame(capacities).to_csv(OUT/'capacity_and_training_budget.csv',index=False);pd.concat(variance).to_csv(OUT/'training_response_variance.csv',index=False)
(OUT/'checks.json').write_text(json.dumps({'model_checks':checks,'bootstrap':verify_bootstrap()},indent=2))
summaries=[]
comparison_predictions=allpred
if a.baseline_run:
    basepred=pd.read_csv(Path(a.baseline_run)/'oof_predictions.csv');assert not set(basepred.method)&set(allpred.method)
    comparison_predictions=pd.concat([basepred,allpred],ignore_index=True);comparison_predictions.to_csv(OUT/'paired_baseline_predictions.csv',index=False)
comparisons=[('crdrp_full_nnls','crdrp_bulk'),('transformer_full_nnls','transformer_bulk'),('crdrp_full_nnls','global_xgb'),('crdrp_full_bayesprism_fixed','crdrp_full_nnls'),('crdrp_full_bayesprism_fixed','crdrp_bulk')]
comparisons=[(x,y) for x,y in comparisons if x in set(comparison_predictions.method) and y in set(comparison_predictions.method)]
for seed in cfg['seeds']:
 for m,b in comparisons:
    ss,draws,weights=paired_bootstrap(comparison_predictions[comparison_predictions.seed==seed],m,b,cfg['bootstrap_replicates'],seed);ss['seed']=seed;summaries.append(ss);tag=f'{seed}_{m}_minus_{b}';draws.to_csv(OUT/('bootstrap_'+tag+'.csv'),index=False);np.savez_compressed(OUT/('bootstrap_weights_'+tag+'.npz'),**weights)
if summaries:pd.concat(summaries).to_csv(OUT/'paired_two_way_bootstrap.csv',index=False)
files=[Path(__file__),ROOT/'code/leakage_safe.py',ROOT/'code/paired_two_way_bootstrap.py',F/'config.json',F/'source_hashes.json']
(OUT/'sources.json').write_text(json.dumps([{'path':str(q.relative_to(ROOT)),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in files],indent=2))
(OUT/'COMPLETE.json').write_text(json.dumps({'scope':'real_GDSC_smoke' if a.smoke else 'real_GDSC_explicit_feature_variant_no_pretraining','all_checks_passed':True,'n_pairs':len(pairs),'n_cells':pairs.ModelID.nunique(),'n_drugs':pairs.drug_dpc.nunique(),'n_predictions':len(allpred),'transformer_positive_control_gate_passed':False},indent=2))
print(OUT)
