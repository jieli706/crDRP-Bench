"""Append BayesPrism to already-frozen oracle/NNLS downstream comparisons; no reruns."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,importlib.util,json,pickle,hashlib,logging
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch
from leakage_safe import FoldTransform,metrics,safe_corr
from paired_two_way_bootstrap import paired_bootstrap
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--bayes-run',required=True);a=p.parse_args();B=Path(a.bayes_run).resolve();assert (B/'COMPLETE.txt').exists()
I=ROOT/'runs/paired_deconvolution_inputs_20260926_130450';P=ROOT/'runs/paired_deconvolution_downstream_20260926_130844';O=ROOT/'runs'/('bayesprism_paired_downstream_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir()
logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(O/'run.log'),logging.StreamHandler()]);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dev='cuda' if torch.cuda.is_available() else 'cpu';sp=importlib.util.spec_from_file_location('cr',ROOT/'source_snapshot/scripts/07_crdrp_model.py');cr=importlib.util.module_from_spec(sp);sp.loader.exec_module(cr)
preds=[];qc=[];budget=[];recover=[]
for seed in [7,17,37]:
    out=O/f'seed_{seed}';out.mkdir();src=I/f'seed_{seed}';old=P/f'seed_{seed}';a=np.load(src/'paired_inputs.npz');mid=pd.read_csv(src/'mixture_ids_and_roles.csv');comp=a['compartments'];genes=a['gene_ids'];ids=a['mixture_ids'];bp=B/f'seed_{seed}';pi=pd.read_csv(bp/'fractions.csv',index_col=0).loc[ids,comp].to_numpy(dtype='float32');zs=[]
    for k in comp:
        z=pd.read_csv(bp/f'Z_{k}.csv.gz',index_col=0);assert set(z.index)==set(ids);zs.append(z.reindex(index=ids,columns=genes,fill_value=0).to_numpy())
    alloc=np.stack(zs,axis=1);err=float(np.max(abs(alloc.sum(1)-a['counts'])));assert err<1e-6;mass=alloc.sum(2,keepdims=True);z=np.divide(alloc,mass,out=np.zeros_like(alloc),where=mass>0);np.savez_compressed(out/'bayesprism_predictor_inputs.npz',pi=pi,Z=z.astype('float32'),mixture_ids=ids,compartments=comp,gene_ids=genes)
    pairs=pd.read_csv(old/'synthetic_response_and_split.csv');drug=np.load(old/'synthetic_drugs_and_nnls_allocation.npz');dr=drug['drug_features'];drugs=drug['drug_ids'];c=pairs.ModelID.map(dict(zip(ids,range(len(ids))))).to_numpy();d=pairs.drug_dpc.map(dict(zip(drugs,range(len(drugs))))).to_numpy();tr=np.flatnonzero(pairs.role.eq('train'));va=np.flatnonzero(pairs.role.eq('validation'));te=np.flatnonzero(pairs.role.eq('test'));fit=np.flatnonzero(mid.role.eq('train'))
    with (old/'crdrp_nnls_preprocessing.pkl').open('rb') as f:dt=pickle.load(f)['drug']
    z=np.log1p(z*1e6);zt=[FoldTransform(64,seed=seed).fit(z[fit,k],ids[fit]) for k in range(8)];zf=np.stack([zt[k].transform(z[:,k]) for k in range(8)],axis=1);df=dt.transform(dr)
    torch.manual_seed(seed);np.random.seed(seed);model=cr.CrDRP(z_dim=zf.shape[2],drug_dim=df.shape[1]).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);tp=torch.tensor(pi,device=dev);tz=torch.tensor(zf,device=dev);td=torch.tensor(df,device=dev);ct=torch.tensor(c,device=dev);dd=torch.tensor(d,device=dev);yt=torch.tensor(pairs.y_true.to_numpy(dtype='float32'),device=dev)
    def predict(ix):return model(tp[ct[ix]],tz[ct[ix]],td[dd[ix]])[0]
    best=float('inf');bestep=-1;history=[];rng=np.random.default_rng(seed)
    for ep in range(120):
        model.train();order=rng.permutation(tr);losses=[]
        for pos in range(0,len(order),256):
            ix=order[pos:pos+256];opt.zero_grad();loss=torch.nn.functional.huber_loss(predict(ix),yt[ix],delta=.05);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
        model.eval()
        with torch.no_grad():vr=float(torch.sqrt(torch.mean((predict(va)-yt[va])**2)))
        history.append({'epoch':ep,'train_huber':float(np.mean(losses)),'validation_rmse':vr})
        if vr<best:best=vr;bestep=ep;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep-bestep>=20:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():yp=predict(te).cpu().numpy()
    with (out/'crdrp_bayesprism_fixed.pt').open('wb') as f:torch.save({'state_dict':state,'z_dim':zf.shape[2],'drug_dim':df.shape[1],'seed':seed,'best_epoch':bestep},f)
    with (out/'preprocessing.pkl').open('wb') as f:pickle.dump({'Z':zt,'drug':dt},f)
    with (out/'crdrp_bayesprism_fixed.pt').open('rb') as f:model.load_state_dict(torch.load(f,map_location=dev)['state_dict'])
    with torch.no_grad():repeat=predict(te).cpu().numpy()
    assert np.allclose(yp,repeat,atol=1e-6)
    pp=pairs.iloc[te].copy();pp['y_pred']=yp;pp['method']='crdrp_bayesprism_fixed';pp['seed']=seed;pp['fold']=0;preds.append(pp);pp.to_csv(out/'predictions.csv',index=False);pd.DataFrame(history).to_csv(out/'training.csv',index=False);qc.append({'seed':seed,'count_reconstruction_max_error':err,'replay_max_error':float(np.max(abs(yp-repeat)))})
    budget.append({'seed':seed,'method':'crdrp_bayesprism_fixed','parameters':sum(p.numel() for p in model.parameters()),'cell_dimensions':8*(zf.shape[2]+1),'drug_dimensions':df.shape[1],'epochs_run':len(history),'selected_epoch':bestep})
    test=np.flatnonzero(mid.role.eq('test'))
    for j,k in enumerate(comp):
        for name,estimated in [('NNLS',a['nnls_pi']),('BayesPrism_fixed',pi)]:recover.append({'seed':seed,'method':name,'compartment':k,'test_fraction_r':safe_corr(a['oracle_pi'][test,j],estimated[test,j]),'test_fraction_MAE':float(np.mean(abs(a['oracle_pi'][test,j]-estimated[test,j]))),'n_test_mixtures':len(test),'n_test_donors':7})
    logging.info('BayesPrism downstream seed=%s completed',seed)
oldpred=pd.read_csv(P/'heldout_predictions.csv');pred=pd.concat([oldpred,pd.concat(preds)],ignore_index=True);pred.to_csv(O/'heldout_predictions_all_methods.csv',index=False);pd.DataFrame([{'seed':s,'method':m,**metrics(g)} for (s,m),g in pred.groupby(['seed','method'])]).to_csv(O/'metrics.csv',index=False);summaries=[]
for seed in [7,17,37]:
 for ma,mb in [('crdrp_bayesprism_fixed','crdrp_nnls'),('crdrp_oracle','crdrp_bayesprism_fixed'),('crdrp_bayesprism_fixed','crdrp_bulk')]:
    ss,draws,w=paired_bootstrap(pred[pred.seed==seed],ma,mb,1000,seed,cell_cluster_column='donor_id');ss['seed']=seed;summaries.append(ss);tag=f'{seed}_{ma}_minus_{mb}';draws.to_csv(O/('bootstrap_'+tag+'.csv'),index=False);np.savez_compressed(O/('bootstrap_weights_'+tag+'.npz'),**w)
pd.concat(summaries).to_csv(O/'paired_donor_drug_bootstrap.csv',index=False);pd.DataFrame(recover).to_csv(O/'fraction_recovery_test_only.csv',index=False);pd.DataFrame(budget).to_csv(O/'training_budget.csv',index=False);(O/'checks.json').write_text(json.dumps(qc,indent=2));(O/'config.json').write_text(json.dumps({'scope':'real-atlas synthetic-mixture synthetic-response downstream comparison; NOT real drug response and NOT OOF','BayesPrism_variant':'2.2.3 initial Gibbs, fixed reference, no cross-sample updates','bayes_run':str(B),'original_downstream_run':str(P),'original_predictions_reused_without_retraining':True,'budget':'same 120 epochs, patience20, AdamW .001/.0003, batch256, Huber .05; same archived crDRP','bootstrap':'1000 donor x drug paired draws per seed; only 7 test donors; conditional fixed models','pretraining':False},indent=2));(O/'COMPLETE.json').write_text(json.dumps({'status':'paired_three_deconvolution_inputs_propagated_to_crdrp'}));print(O)
