"""SYNTHETIC ONLY: exercise leakage guards, full/bulk training and complete artifacts."""
import hashlib,importlib.util,json,logging,os,pickle,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from scipy.optimize import nnls
from leakage_safe import outer_splits,inner_split,assert_disjoint,permitted_pretrain,FoldTransform,metrics,safe_corr
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs'/('synthetic_smoke_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir(parents=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',encoding='utf-8'),logging.StreamHandler()])
spec=importlib.util.spec_from_file_location('arch',ROOT/'source_snapshot/scripts/07_crdrp_model.py')
arch=importlib.util.module_from_spec(spec);spec.loader.exec_module(arch)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
SEEDS=[7,17,37]; all_metrics=[];checks=[];all_pred=[]
config={'data_kind':'SYNTHETIC_PIPELINE_TEST_NOT_BIOLOGICAL_RESULT','seeds':SEEDS,'outer_folds':3,'pretrain_epochs':4,'finetune_max_epochs':15,'inner_metric':'RMSE','full_bulk_hidden':16,'synthetic_cells_target':36,'synthetic_source_cells':18,'drugs':12}
(OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
g=np.random.default_rng(20260925); n=54;k=8;nd=12
pi=g.dirichlet(np.ones(k),size=n).astype('float32');z=g.normal(size=(n,k,16)).astype('float32')
bulk=np.einsum('nk,nkg->ng',pi,z);drug=g.normal(size=(nd,12)).astype('float32')
cells=np.array([f'SYN_CELL_{i:03}' for i in range(n)]);drugs=np.array([f'SYN_DRUG_{i:03}' for i in range(nd)])
ci,di=np.indices((n,nd));ci=ci.ravel();di=di.ravel()
y=(.5+.09*drug[di,0]+.12*pi[ci,0]+.06*z[ci,0,0]*drug[di,1]+g.normal(0,.01,len(ci))).astype('float32')
pairs=pd.DataFrame({'ModelID':cells[ci],'drug_dpc':drugs[di],'cell_row':ci,'drug_row':di,'auc':y})
pairs['pair_id']=pairs.ModelID+'::'+pairs.drug_dpc
target=pairs[pairs.cell_row<36].reset_index(drop=True)
# Source intentionally contains target cells and test drugs so guards are exercised.
np.savez_compressed(OUT/'synthetic_data.npz',pi=pi,Z=z,bulk=bulk,drug=drug,cell_ids=cells,drug_ids=drugs)
pairs.to_csv(OUT/'synthetic_pairs.csv',index=False)
def train_model(model,kind,zf,bf,df,rows,epochs,val=None):
    opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=3e-4)
    best=None;best_rmse=float('inf');trace=[]
    def pred(r):
        c=r.cell_row.to_numpy();d=r.drug_row.to_numpy()
        if kind=='full':return model(torch.from_numpy(pi[c]),torch.from_numpy(zf[c]),torch.from_numpy(df[d]))[0]
        return model(torch.from_numpy(bf[c]),torch.from_numpy(df[d]))
    for ep in range(epochs):
        model.train();opt.zero_grad();p=pred(rows)
        loss=torch.nn.functional.huber_loss(p,torch.from_numpy(rows.auc.to_numpy().astype('float32')),delta=.05)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        model.eval()
        with torch.no_grad():
            rmse=float(torch.sqrt(torch.mean((pred(val if val is not None else rows)-torch.from_numpy((val if val is not None else rows).auc.to_numpy().astype('float32')))**2)))
        trace.append({'epoch':ep,'loss':float(loss.detach()),'selection_rmse':rmse})
        if rmse<best_rmse:best_rmse=rmse;best={a:b.detach().clone() for a,b in model.state_dict().items()}
    # Fixed epochs for pretraining; only inner validation selects fine-tuning epoch.
    if val is not None:model.load_state_dict(best)
    return pred,trace

for seed in SEEDS:
 for regime in ['pair','cell','drug']:
  pred_rows=[]
  for fold,(outer_train,test) in enumerate(outer_splits(target,regime,seed,3)):
    train,val=inner_split(target,outer_train,regime,seed+fold);assert_disjoint(target,train,val,test,regime)
    source=permitted_pretrain(pairs,cells[:36],target,val,test,regime)
    assert not set(source.ModelID)&set(cells[:36])
    fitrows=pd.concat([source,target.iloc[train]])
    cfit=np.sort(fitrows.cell_row.unique());dfit=np.sort(fitrows.drug_row.unique())
    if regime=='drug':assert not set(dfit)&set(target.iloc[np.r_[val,test]].drug_row)
    folder=OUT/f'seed_{seed}'/regime/f'fold_{fold}';folder.mkdir(parents=True)
    ids=target[['pair_id','ModelID','drug_dpc']].copy();ids['role']='test';ids.loc[train,'role']='train';ids.loc[val,'role']='validation';ids.to_csv(folder/'split.csv',index=False)
    source[['pair_id','ModelID','drug_dpc']].to_csv(folder/'pretrain_pairs.csv',index=False)
    bt=FoldTransform(8,seed=seed).fit(bulk[cfit],cells[cfit]);dt=FoldTransform(8,seed=seed).fit(drug[dfit],drugs[dfit])
    zt=[FoldTransform(8,seed=seed).fit(z[cfit,j],cells[cfit]) for j in range(k)]
    bf=bt.transform(bulk);df=dt.transform(drug);zf=np.stack([zt[j].transform(z[:,j]) for j in range(k)],axis=1)
    with (folder/'preprocessing.pkl').open('wb') as f:pickle.dump({'bulk':bt,'drug':dt,'Z':zt},f)
    for name in ['full','bulk','ridge','drug_mean']:
        torch.manual_seed(seed*100+fold);np.random.seed(seed*100+fold)
        tr=target.iloc[train];va=target.iloc[val];te=target.iloc[test]
        if name in ['full','bulk']:
            model=arch.CrDRP(z_dim=zf.shape[2],drug_dim=df.shape[1],hidden=16,emb_dim=16,dropout=.1) if name=='full' else arch.BulkDRP(bulk_dim=bf.shape[1],drug_dim=df.shape[1],hidden=16,emb_dim=16,dropout=.1)
            _,pt=train_model(model,name,zf,bf,df,source,4)
            with (folder/f'{name}_pretrain.pt').open('wb') as f:
                torch.save({'state_dict':model.state_dict(),'data_kind':'synthetic','seed':seed},f)
            predict,trace=train_model(model,name,zf,bf,df,tr,15,va)
            model.eval()
            with torch.no_grad():yp=predict(te).numpy()
            with (folder/f'{name}_finetuned.pt').open('wb') as f:
                torch.save({'state_dict':model.state_dict(),'data_kind':'synthetic','seed':seed},f)
            pd.DataFrame(trace).to_csv(folder/f'{name}_training.csv',index=False)
            count=sum(p.numel() for p in model.parameters())
        elif name=='ridge':
            def features(r):return np.c_[bf[r.cell_row],df[r.drug_row]]
            model=Ridge(alpha=10).fit(features(tr),tr.auc);yp=model.predict(features(te));count=len(model.coef_)+1
            with (folder/'ridge.pkl').open('wb') as f:pickle.dump(model,f)
        else:
            means=tr.groupby('drug_dpc').auc.mean();yp=te.drug_dpc.map(means).fillna(tr.auc.mean()).to_numpy();count=len(means)+1
            (folder/'drug_mean.json').write_text(json.dumps({'means':means.to_dict(),'fallback':float(tr.auc.mean())}),encoding='utf-8')
        r=te[['pair_id','ModelID','drug_dpc']].copy();r['y_true']=te.auc;r['y_pred']=yp;r['method']=name;r['seed']=seed;r['regime']=regime;r['fold']=fold;r['parameters']=count
        assert np.isfinite(yp).all();pred_rows.append(r)
    checks.append({'seed':seed,'regime':regime,'fold':fold,'split_disjoint':True,'target_cells_excluded_pretrain':True,'n_pretrain':len(source),'n_fit_cells':len(cfit),'n_fit_drugs':len(dfit)})
  frame=pd.concat(pred_rows,ignore_index=True);all_pred.append(frame)
  for name,r in frame.groupby('method'):
    assert len(r)==len(target) and r.pair_id.nunique()==len(target)
    all_metrics.append({'seed':seed,'regime':regime,'method':name,**metrics(r)})
  logging.info('Completed synthetic seed=%s regime=%s',seed,regime)

# Adversarial tests: changing held-out features must not change fitted training transform.
x=g.normal(size=(20,10));fit=np.arange(10)
a=FoldTransform(5).fit(x[fit],list(map(str,fit)));poison=x.copy();poison[10:]=1e9
b=FoldTransform(5).fit(poison[fit],list(map(str,fit)))
assert np.array_equal(a.pca_.components_,b.pca_.components_)
checks.append({'test':'heldout_feature_poisoning','pass':True})
try: outer_splits(pd.concat([target,target.iloc[:1]]),'pair',7,3)
except ValueError:checks.append({'test':'duplicate_pair_rejected','pass':True})
else:raise AssertionError('Duplicates not rejected')

# Identifiable NNLS unit test; NOT independent reference biological validation.
sig=g.uniform(.1,1,(80,8));truth=g.dirichlet(np.ones(8),size=40);mix=truth@sig.T
estimated=np.stack([nnls(sig,r)[0] for r in mix]);err=float(abs(truth-estimated).max());assert err<1e-8
np.savez_compressed(OUT/'synthetic_nnls_inputs.npz',signature=sig,truth=truth,mixtures=mix)
pd.DataFrame(np.c_[truth,estimated],columns=[f'true_{i}' for i in range(8)]+[f'est_{i}' for i in range(8)]).to_csv(OUT/'synthetic_nnls_predictions.csv',index=False)
checks.append({'test':'identifiable_nnls','max_abs_error':err,'pass':True})

# A known compartment signal with deliberately uninformative bulk is a pipeline positive control.
# Exact fractions are an oracle. This does not establish crDRP sensitivity on biological samples.
frac=g.dirichlet(np.ones(8),size=200);response=2*frac[:,0]-frac[:,1]+g.normal(0,.01,200)
pc=Ridge(alpha=.01).fit(frac[:140],response[:140]);oracle_pred=pc.predict(frac[140:]);bulk_pred=np.repeat(response[:140].mean(),60)
null=Ridge(alpha=.01).fit(frac[:140],g.permutation(response[:140])).predict(frac[140:])
pcstats={'data_kind':'synthetic only','oracle_r':safe_corr(response[140:],oracle_pred),'oracle_rmse':float(np.sqrt(np.mean((response[140:]-oracle_pred)**2))),'bulk_mean_rmse':float(np.sqrt(np.mean((response[140:]-bulk_pred)**2))),'shuffled_train_label_r':safe_corr(response[140:],null)}
np.savez_compressed(OUT/'synthetic_positive_control_inputs.npz',fractions=frac,response=response,train_indices=np.arange(140),test_indices=np.arange(140,200),sample_ids=np.array([f'SYN_PC_{i}' for i in range(200)]))
assert pcstats['oracle_r']>.9
pd.DataFrame({'sample_id':[f'SYN_PC_{i}' for i in range(140,200)],'y_true':response[140:],'oracle_pred':oracle_pred,'bulk_pred':bulk_pred,'shuffled_label_pred':null}).to_csv(OUT/'synthetic_positive_control_predictions.csv',index=False)
with (OUT/'positive_control_ridge.pkl').open('wb') as f:pickle.dump(pc,f)
(OUT/'positive_control.json').write_text(json.dumps(pcstats,indent=2),encoding='utf-8')
pd.DataFrame(all_metrics).to_csv(OUT/'metrics.csv',index=False);pd.concat(all_pred).to_csv(OUT/'predictions.csv',index=False)
(OUT/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
counts={'original_default_full':arch.count_params(arch.CrDRP()),'original_default_bulk':arch.count_params(arch.BulkDRP()),'softmax_L1_constant':float(torch.softmax(torch.randn(10,8),dim=1).abs().sum(1).mean())}
(OUT/'architecture_audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
(OUT/'COMPLETE.json').write_text(json.dumps({'status':'synthetic_smoke_passed','not_real_benchmark':True,'n_split_checks':len(checks),'run':str(OUT),'utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')
print(OUT)
