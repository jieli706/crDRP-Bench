# -*- coding: utf-8 -*-
"""Known compartment effect tested in actual archived neural architectures.
All cells, drug descriptors and responses here are synthetic. No GDSC result is claimed.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,importlib.util,importlib.metadata,json,logging,pickle,time,sys
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch,xgboost as xgb
from leakage_safe import outer_splits,inner_split,assert_disjoint,FoldTransform,metrics
from paired_two_way_bootstrap import paired_bootstrap,verify_bootstrap
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs'/('model_positive_control_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir()
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
DEV='cuda' if torch.cuda.is_available() else 'cpu'
def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'source_snapshot/scripts'/filename);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
cr=load('cr','07_crdrp_model.py');tf=load('tf','08_transformer_model.py')
config={'data_kind':'fully_synthetic_known_compartment_effect','data_seed':260925,'seeds':[7,17,37],'regime':'cell','outer_folds':3,'n_cells':96,'n_drugs':16,'n_compartments':8,'raw_gene_dimensions_per_compartment':64,'raw_drug_dimensions':116,'neural_max_epochs':120,'early_stopping_patience':20,'inner_selection_metric':'RMSE','batch_size':256,'learning_rate':.001,'weight_decay':.0003,'pretraining':False,'tuning_candidates':1,'positive_response_formula':'0.2 + 0.6*pi_0 + 0.08*tanh(drug_0) + 0.15*(pi_2-pi_5)*tanh(drug_1) + Gaussian(0,0.005)','negative_control':'shuffle train and inner validation response independently within each drug; test labels untouched','bootstrap':'1000 paired crossed cell/drug draws per seed; no pooling seeds as independent cells','device':DEV,'xgboost_n_estimators':300,'xgboost_max_depth':4}
(OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','xgboost']}
(OUT/'environment.json').write_text(json.dumps({'executable':sys.executable,'packages':versions,'device':DEV,'gpu':torch.cuda.get_device_name() if DEV=='cuda' else None},indent=2),encoding='utf-8')
rng=np.random.default_rng(config['data_seed']);nc=96;nd=16;pi=rng.dirichlet(np.full(8,.5),size=nc).astype('float32')
# Constant compartment signatures plus small cell variation: both full and bulk contain actual features.
signatures=rng.gamma(2,1,size=(8,64));Z=(signatures[None]+rng.normal(0,.05,(nc,8,64))).astype('float32');bulk=np.einsum('nk,nkg->ng',pi,Z)
dr=rng.normal(size=(nd,116)).astype('float32');cells=np.array([f'SYN_PC_CELL_{i:03}' for i in range(nc)]);drugs=np.array([f'SYN_PC_DRUG_{i:02}' for i in range(nd)])
c,d=np.indices((nc,nd));c=c.ravel();d=d.ravel();y=.2+.6*pi[c,0]+.08*np.tanh(dr[d,0])+.15*(pi[c,2]-pi[c,5])*np.tanh(dr[d,1])+rng.normal(0,.005,len(c))
pairs=pd.DataFrame({'ModelID':cells[c],'drug_dpc':drugs[d],'cell_row':c,'drug_row':d,'auc':y.astype('float32')});pairs['pair_id']=pairs.ModelID+'::'+pairs.drug_dpc
np.savez_compressed(OUT/'input_synthetic.npz',pi=pi,Z=Z,bulk=bulk,drug=dr,cell_ids=cells,drug_ids=drugs);pairs.to_csv(OUT/'response_synthetic.csv',index=False)
predictions=[];paramrows=[];splitchecks=[]
for seed in config['seeds']:
 for fold,(outer_train,test) in enumerate(outer_splits(pairs,'cell',seed,3)):
    tr,va=inner_split(pairs,outer_train,'cell',seed+fold);assert_disjoint(pairs,tr,va,test,'cell');splitchecks.append({'seed':seed,'fold':fold,'cell_disjoint':True})
    path=OUT/f'seed_{seed}'/f'fold_{fold}';path.mkdir(parents=True)
    split=pairs[['pair_id','ModelID','drug_dpc']].copy();split['role']='test';split.loc[tr,'role']='train';split.loc[va,'role']='validation';split.to_csv(path/'split.csv',index=False)
    cfit=np.sort(pairs.iloc[tr].cell_row.unique());dfit=np.sort(pairs.iloc[tr].drug_row.unique())
    bt=FoldTransform(40,seed=seed).fit(bulk[cfit],cells[cfit]);dt=FoldTransform(116,seed=seed).fit(dr[dfit],drugs[dfit]);zt=[FoldTransform(64,seed=seed).fit(Z[cfit,j],cells[cfit]) for j in range(8)]
    bf=bt.transform(bulk);df=dt.transform(dr);zf=np.stack([zt[j].transform(Z[:,j]) for j in range(8)],axis=1)
    with (path/'preprocessing.pkl').open('wb') as f:pickle.dump({'bulk':bt,'drug':dt,'Z':zt},f)
    tp=torch.tensor(pi,device=DEV);tz=torch.tensor(zf,device=DEV);tb=torch.tensor(bf,device=DEV);td=torch.tensor(df,device=DEV)
    ct=torch.tensor(c,device=DEV);dd=torch.tensor(d,device=DEV);yt=torch.tensor(pairs.auc.to_numpy(),device=DEV)
    perm_rng=np.random.default_rng(seed*100+fold);shuffled=pairs.auc.to_numpy().copy()
    for ix in [tr,va]:
        for drug_id in pairs.iloc[ix].drug_dpc.unique():
            ids=ix[pairs.iloc[ix].drug_dpc.to_numpy()==drug_id];shuffled[ids]=perm_rng.permutation(shuffled[ids])
    pd.DataFrame({'pair_id':pairs.pair_id,'original_response':pairs.auc,'negative_control_response':shuffled}).to_csv(path/'shuffled_labels.csv',index=False)
    for method in ['crdrp_full','crdrp_bulk','transformer_full','transformer_bulk','global_xgb','drug_mean','crdrp_full_shuffled','transformer_full_shuffled']:
        torch.manual_seed(seed*100+fold);np.random.seed(seed*100+fold);start=time.time()
        if method=='global_xgb':
            X=np.c_[bf[c],df[d]];model=xgb.XGBRegressor(n_estimators=300,max_depth=4,learning_rate=.05,subsample=1,colsample_bytree=1,n_jobs=2,random_state=seed,objective='reg:squarederror')
            model.fit(X[tr],pairs.auc.iloc[tr]);yp=model.predict(X[test]);model.save_model(str(path/'global_xgb.json'));nparams=None;bestep=300
        elif method=='drug_mean':
            means=pairs.iloc[tr].groupby('drug_dpc').auc.mean();yp=pairs.iloc[test].drug_dpc.map(means).to_numpy();nparams=len(means);bestep=0
            (path/'drug_mean.json').write_text(means.to_json(),encoding='utf-8')
        else:
            base=method.replace('_shuffled','');negative=method.endswith('_shuffled')
            if base=='crdrp_full':model=cr.CrDRP(z_dim=zf.shape[2],drug_dim=df.shape[1])
            elif base=='crdrp_bulk':model=cr.BulkDRP(bulk_dim=bf.shape[1],drug_dim=df.shape[1])
            elif base=='transformer_full':model=tf.TransformerDRP(z_dim=zf.shape[2],drug_dim=df.shape[1])
            else:model=tf.TransformerBulkDRP(bulk_dim=bf.shape[1],drug_dim=df.shape[1])
            model=model.to(DEV);nparams=sum(p.numel() for p in model.parameters());opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
            labels=torch.tensor(shuffled,device=DEV) if negative else yt
            def predict(ix):
                if base.endswith('full'):
                    out=model(tp[ct[ix]],tz[ct[ix]],td[dd[ix]])
                    return out[0] if base.startswith('crdrp') else out
                return model(tb[ct[ix]],td[dd[ix]])
            best=float('inf');bestep=-1;history=[];batchrng=np.random.default_rng(seed*100+fold)
            for epoch in range(120):
                model.train();order=batchrng.permutation(tr);losses=[]
                for j in range(0,len(order),256):
                    ix=order[j:j+256];opt.zero_grad();loss=torch.nn.functional.huber_loss(predict(ix),labels[ix],delta=.05);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
                model.eval()
                with torch.no_grad():vr=float(torch.sqrt(torch.mean((predict(va)-labels[va])**2)))
                history.append({'epoch':epoch,'train_huber':float(np.mean(losses)),'validation_rmse':vr})
                if vr<best:best=vr;bestep=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if epoch-bestep>=20:break
            model.load_state_dict(state);model.eval()
            with torch.no_grad():yp=predict(test).cpu().numpy()
            with (path/f'{method}.pt').open('wb') as f:torch.save({'state_dict':state,'seed':seed,'fold':fold,'best_epoch':bestep,'method':method,'data_kind':'synthetic','z_dim':zf.shape[2],'bulk_dim':bf.shape[1],'drug_dim':df.shape[1]},f)
            pd.DataFrame(history).to_csv(path/f'{method}_training.csv',index=False)
        result=pairs.iloc[test][['ModelID','drug_dpc','pair_id']].copy();result['y_true']=pairs.auc.iloc[test];result['y_pred']=yp;result['method']=method;result['seed']=seed;result['fold']=fold;result['regime']='cell';predictions.append(result)
        result.to_csv(path/f'{method}_predictions.csv',index=False)
        paramrows.append({'seed':seed,'fold':fold,'method':method,'trainable_parameters':nparams,'cell_input_dimensions':int(8*(zf.shape[2]+1)) if 'full' in method else bf.shape[1],'drug_dimensions':df.shape[1],'best_epoch':bestep,'seconds':time.time()-start,'tuning_candidates':1})
        logging.info('seed=%s fold=%s method=%s time=%.1fs best_epoch=%s',seed,fold,method,time.time()-start,bestep)
allpred=pd.concat(predictions,ignore_index=True);allpred.to_csv(OUT/'oof_predictions.csv',index=False)
rows=[]
for (seed,method),p in allpred.groupby(['seed','method']):
    assert len(p)==len(pairs) and p.pair_id.is_unique
    rows.append({'seed':seed,'method':method,**metrics(p)})
pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(paramrows).to_csv(OUT/'capacity_and_training_budget.csv',index=False)
checks=verify_bootstrap();(OUT/'checks.json').write_text(json.dumps({'splits':splitchecks,'bootstrap_checks':checks},indent=2),encoding='utf-8')
summaries=[]
for seed in config['seeds']:
    for a,b in [('crdrp_full','crdrp_bulk'),('transformer_full','transformer_bulk'),('crdrp_full','crdrp_full_shuffled'),('transformer_full','transformer_full_shuffled')]:
        ss,draws,weights=paired_bootstrap(allpred[allpred.seed==seed],a,b,1000,seed)
        tag=f'{seed}_{a}_minus_{b}';ss['seed']=seed;summaries.append(ss);draws.to_csv(OUT/f'bootstrap_{tag}.csv',index=False);np.savez_compressed(OUT/f'bootstrap_weights_{tag}.npz',**weights)
pd.concat(summaries).to_csv(OUT/'paired_two_way_bootstrap.csv',index=False)
(OUT/'COMPLETE.json').write_text(json.dumps({'status':'completed','scope':'actual_architecture_positive_control_with_synthetic_responses','not_real_GDSC':True,'n_oof_rows':len(allpred)},indent=2),encoding='utf-8')
print(OUT)
