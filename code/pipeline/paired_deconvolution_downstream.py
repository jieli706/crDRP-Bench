"""Real-atlas synthetic-response pilot: propagate NNLS/oracle inputs into crDRP.
BayesPrism is pending, so this is explicitly a partial paired-method validation.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import importlib.util,json,pickle,hashlib,sys,logging
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch
from leakage_safe import FoldTransform,metrics
from paired_two_way_bootstrap import paired_bootstrap,verify_bootstrap
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'runs/paired_deconvolution_inputs_20260926_130450'
OUT=ROOT/'runs'/('paired_deconvolution_downstream_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir()
logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
DEV='cuda' if torch.cuda.is_available() else 'cpu'
spec=importlib.util.spec_from_file_location('cr',ROOT/'source_snapshot/scripts/07_crdrp_model.py');cr=importlib.util.module_from_spec(spec);spec.loader.exec_module(cr)
drugsrc=ROOT/'runs/model_positive_control_20260925_160423/input_synthetic.npz';dr=np.load(drugsrc)['drug'];drugs=np.load(drugsrc)['drug_ids']
cfg={'input_run':str(SRC.relative_to(ROOT)),'seeds':[7,17,37],'methods':['crdrp_oracle','crdrp_nnls','crdrp_bulk'],
 'response_formula':'0.2+0.6*pi0+0.08*tanh(drug0)+0.15*(pi2-pi5)*tanh(drug1)+N(0,0.005)',
 'response_and_drug_features':'synthetic, not GDSC; stored drug features from initial positive control',
 'NNLS_Z':'Allocate observed counts proportionally to nnls_pi[k]*reference_profile[k,g]; normalize each allocated compartment profile across genes; not BayesPrism expression estimates.',
 'oracle_Z':'True selected-gene normalized donor-compartment profiles used to generate counts; privileged information.',
 'epochs':120,'patience':20,'lr':.001,'weight_decay':.0003,'batch':256,'tuning_candidates':1,'pretraining':False,
 'selection':'inner-validation RMSE; transforms fit only predictor training mixtures and training drugs',
 'resampling':'paired donor by drug bootstrap, 1000 draws; synthetic mixtures from same donor remain clustered',
 'scope':'partial paired downstream positive control, not real drug response and not complete BayesPrism comparison',
 'python':sys.executable,'torch':torch.__version__,'device':DEV}
(OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
verify_bootstrap();predictions=[];budgets=[];boot=[];checks=[]
for seed in cfg['seeds']:
    src=SRC/f'seed_{seed}';out=OUT/f'seed_{seed}';out.mkdir();a=np.load(src/'paired_inputs.npz');mid=pd.read_csv(src/'mixture_ids_and_roles.csv')
    assert np.array_equal(a['mixture_ids'],mid.mixture_id)
    pi=a['oracle_pi'];npi=a['nnls_pi'];ref=a['reference_profiles'];counts=a['counts'];bulk=np.log1p(counts/counts.sum(1,keepdims=True)*1e6)
    numer=npi[:,:,None]*ref[None,:,:];denom=numer.sum(1,keepdims=True)
    alloc=np.divide(numer,denom,out=np.zeros_like(numer),where=denom>0)*counts[:,None,:]
    mass=alloc.sum(2,keepdims=True);nz=np.divide(alloc,mass,out=np.zeros_like(alloc),where=mass>0)
    Z={'crdrp_oracle':np.log1p(a['oracle_Z']*1e6),'crdrp_nnls':np.log1p(nz*1e6)}
    fit=np.flatnonzero(mid.role.eq('train'));val=np.flatnonzero(mid.role.eq('validation'));test=np.flatnonzero(mid.role.eq('test'))
    for left,right in [(fit,val),(fit,test),(val,test)]:assert not set(mid.donor_id.iloc[left])&set(mid.donor_id.iloc[right])
    ci,di=np.indices((len(mid),len(drugs)));ci=ci.ravel();di=di.ravel();rng=np.random.default_rng(10000+seed)
    y=(.2+.6*pi[ci,0]+.08*np.tanh(dr[di,0])+.15*(pi[ci,2]-pi[ci,5])*np.tanh(dr[di,1])+rng.normal(0,.005,len(ci))).astype('float32')
    pairs=pd.DataFrame({'ModelID':mid.mixture_id.to_numpy()[ci],'donor_id':mid.donor_id.to_numpy()[ci],'drug_dpc':drugs[di],'y_true':y,'role':mid.role.to_numpy()[ci]})
    pairs['pair_id']=pairs.ModelID+'::'+pairs.drug_dpc;pairs.to_csv(out/'synthetic_response_and_split.csv',index=False)
    tr=np.flatnonzero(pairs.role.eq('train'));va=np.flatnonzero(pairs.role.eq('validation'));te=np.flatnonzero(pairs.role.eq('test'))
    dt=FoldTransform(116,seed=seed).fit(dr,drugs);df=dt.transform(dr)
    bt=FoldTransform(40,seed=seed).fit(bulk[fit],mid.mixture_id.iloc[fit]);bf=bt.transform(bulk)
    ci=torch.tensor(ci,device=DEV);di=torch.tensor(di,device=DEV);yt=torch.tensor(y,device=DEV);td=torch.tensor(df,device=DEV);tb=torch.tensor(bf,device=DEV)
    np.savez_compressed(out/'synthetic_drugs_and_nnls_allocation.npz',drug_features=dr,drug_ids=drugs,nnls_Z=nz.astype('float32'))
    for method in cfg['methods']:
        torch.manual_seed(seed);np.random.seed(seed)
        if method=='crdrp_bulk':model=cr.BulkDRP(bulk_dim=bf.shape[1],drug_dim=df.shape[1]);zdim=None;transforms={'bulk':bt,'drug':dt}
        else:
            zt=[FoldTransform(64,seed=seed).fit(Z[method][fit,k],mid.mixture_id.iloc[fit]) for k in range(8)]
            zf=np.stack([zt[k].transform(Z[method][:,k]) for k in range(8)],axis=1);zdim=zf.shape[2]
            tp=torch.tensor(pi if method=='crdrp_oracle' else npi,dtype=torch.float32,device=DEV);tz=torch.tensor(zf,device=DEV)
            model=cr.CrDRP(z_dim=zdim,drug_dim=df.shape[1]);transforms={'Z':zt,'drug':dt}
        model=model.to(DEV);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
        def predict(ix):
            return model(tb[ci[ix]],td[di[ix]]) if method=='crdrp_bulk' else model(tp[ci[ix]],tz[ci[ix]],td[di[ix]])[0]
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
        pp=pairs.iloc[te].copy();pp['y_pred']=yp;pp['method']=method;pp['seed']=seed;pp['fold']=0;predictions.append(pp)
        pp.to_csv(out/f'{method}_test_predictions.csv',index=False);pd.DataFrame(history).to_csv(out/f'{method}_training.csv',index=False)
        with (out/f'{method}_preprocessing.pkl').open('wb') as f:pickle.dump(transforms,f)
        with (out/f'{method}.pt').open('wb') as f:torch.save({'state_dict':state,'method':method,'z_dim':zdim,'drug_dim':df.shape[1],'bulk_dim':bf.shape[1],'seed':seed,'best_epoch':bestep},f)
        budgets.append({'seed':seed,'method':method,'parameters':sum(p.numel() for p in model.parameters()),'cell_dimensions':8*(zdim+1) if zdim else bf.shape[1],'drug_dimensions':df.shape[1],'epochs_run':len(history),'selected_epoch':bestep})
        logging.info('seed=%s method=%s best_epoch=%s',seed,method,bestep)
    checks.append({'seed':seed,'reference_train_val_test_donors_disjoint':True,'train_mixtures':len(fit),'validation_mixtures':len(val),'test_mixtures':len(test),'test_donors':mid.donor_id.iloc[test].nunique()})
pred=pd.concat(predictions,ignore_index=True);pred.to_csv(OUT/'heldout_predictions.csv',index=False)
rows=[{'seed':s,'method':m,**metrics(p)} for (s,m),p in pred.groupby(['seed','method'])];pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
for seed in cfg['seeds']:
    for ma,mb in [('crdrp_oracle','crdrp_nnls'),('crdrp_nnls','crdrp_bulk')]:
        ss,draws,weights=paired_bootstrap(pred[pred.seed==seed],ma,mb,1000,seed,cell_cluster_column='donor_id')
        ss['seed']=seed;ss['scope']='conditional_on_fixed_models_and_seven_test_donors';boot.append(ss)
        tag=f'{seed}_{ma}_minus_{mb}';draws.to_csv(OUT/f'bootstrap_{tag}.csv',index=False);np.savez_compressed(OUT/f'bootstrap_weights_{tag}.npz',**weights)
pd.concat(boot).to_csv(OUT/'paired_donor_drug_bootstrap.csv',index=False);pd.DataFrame(budgets).to_csv(OUT/'capacity_and_training.csv',index=False)
(OUT/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(OUT/'COMPLETE.json').write_text(json.dumps({'status':'partial_NNLS_oracle_bulk_downstream_completed','BayesPrism_completed':False,'real_drug_response':False,'n_heldout_predictions':len(pred)},indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False));print(OUT)
