import os,sys,json,pickle
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
W=Path(__file__).resolve().parent;ROOT=W.parent;sys.path.insert(0,str(ROOT/'260925_Re/code'))
import numpy as np,pandas as pd,xgboost as xgb
from stable_preprocessing import StableFoldTransform
from leakage_safe import metrics
F=ROOT/'260925_Re/data/processed/expanded_gdsc_features_20260926_142255';B=ROOT/'260925_Re/runs/real_gdsc_cell_20260926_151327'
O=W/'xgb';O.mkdir(exist_ok=True);p=pd.read_csv(F/'response_pairs.csv');p['pair_id']=p.ModelID+'::'+p.drug_dpc
cells=pd.read_csv(F/'sample_ids.csv').ModelID.to_numpy();drugs=pd.read_csv(F/'drug_ids.csv').drug_dpc.to_numpy();bulk=np.load(F/'bulk_log2TPM1.npy');z=np.load(F/'nnls_Z_log1pCPM.npy');pi=np.load(F/'nnls_pi.npy');dr=np.load(F/'drug_raw_features.npy');c=p.cell_row.to_numpy();d=p.drug_row.to_numpy();y=p.auc.to_numpy(dtype='float32');pred=[];bud=[]
for seed in [7,17,37]:
 for fold in range(3):
  folder=O/f'seed_{seed}_fold_{fold}';folder.mkdir(exist_ok=True)
  base=B/f'seed_{seed}'/f'fold_{fold}';split=pd.read_csv(base/'split.csv');assert np.array_equal(split.pair_id,p.pair_id)
  tr=np.flatnonzero(split.role.eq('train'));va=np.flatnonzero(split.role.eq('validation'));te=np.flatnonzero(split.role.eq('test'))
  with open(base/'preprocessing.pkl','rb') as f:prep=pickle.load(f)
  bf=prep['bulk'].transform(bulk);zf=np.stack([prep['Z'][k].transform(z[:,k]) for k in range(8)],axis=1);df=prep['drug'].transform(dr)
  for variant,dim in [('compartment_xgb',116),('bulk_xgb_pca32',32),('bulk_xgb_pca64',64),('bulk_xgb_pca116',116),('bulk_xgb_pca256',256)]:
   dt=prep['drug'] if dim==116 else StableFoldTransform(dim,max_features=10000,seed=seed).fit(dr[np.unique(d[tr])],drugs[np.unique(d[tr])]);df=dt.transform(dr)
   cf=np.c_[pi,zf.reshape(len(cells),-1)] if variant=='compartment_xgb' else bf
   xx=np.c_[cf[c],df[d]]
   model=xgb.XGBRegressor(n_estimators=300,max_depth=4,learning_rate=.05,subsample=1,colsample_bytree=1,n_jobs=2,random_state=seed,objective='reg:squarederror',early_stopping_rounds=20)
   model.fit(xx[tr],y[tr],eval_set=[(xx[va],y[va])],verbose=False);yp=model.predict(xx[te]);model.save_model(folder/(variant+'.json'))
   with open(folder/(variant+'_drug_transform.pkl'),'wb') as f:pickle.dump(dt,f)
   g=p.iloc[te][['ModelID','drug_dpc','pair_id']].copy();g['y_true']=y[te];g['y_pred']=yp;g['seed']=seed;g['fold']=fold;g['method']=variant;pred.append(g)
   bud.append(dict(seed=seed,fold=fold,method=variant,requested_drug_PCs=dim,actual_drug_PCs=df.shape[1],cell_dimensions=cf.shape[1],best_iteration=model.best_iteration))
   print(seed,fold,variant,df.shape[1],flush=True)
pd.concat(pred).to_csv(O/'predictions.csv',index=False);pd.DataFrame(bud).to_csv(O/'budgets.csv',index=False)
pd.DataFrame([dict(seed=s,method=m,**metrics(g)) for (s,m),g in pd.concat(pred).groupby(['seed','method'])]).to_csv(O/'metrics.csv',index=False)
(O/'COMPLETE.json').write_text(json.dumps({'status':'complete','same_splits_and_budgets':True,'PCA_256_rank_limited':True},indent=2))
