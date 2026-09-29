"""Predeclared nuisance ablation; keeps the failed original positive control intact.
Reuses all original synthetic responses, seeds, splits and fitted transforms.
Zeroing standardized Z removes nuisance expression inputs, without tuning on tests.
This is an easier diagnostic task, not a replacement benchmark or real drug evidence.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import importlib.util,json,pickle,hashlib,logging,sys
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch
from leakage_safe import metrics,assert_disjoint
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'runs/model_positive_control_20260925_160423'
OUT=ROOT/'runs'/('positive_control_diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
OUT.mkdir()
logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(OUT/'run.log'),logging.StreamHandler()])
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
DEV='cuda' if torch.cuda.is_available() else 'cpu'
def load(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/'source_snapshot/scripts'/file)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cr=load('cr','07_crdrp_model.py');tf=load('tf','08_transformer_model.py')
cfg={'source_run':str(SRC.relative_to(ROOT)),'scope':'synthetic_nuisance_input_ablation','seeds':[7,17,37],
     'methods':['crdrp_full_zeroZ','transformer_full_zeroZ','transformer_full_zeroZ_shuffled'],
     'change':'Set all standardized compartment expression coordinates to zero; keep fractions, drug features and original responses',
     'selection':'Same 120 epochs maximum, patience 20, inner RMSE, 1 configuration; no test-based choice',
     'interpretation':'Easier diagnostic; cannot replace or erase the failed original positive control',
     'python':sys.executable,'torch':torch.__version__,'device':DEV}
(OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
raw=np.load(SRC/'input_synthetic.npz');pairs=pd.read_csv(SRC/'response_synthetic.csv')
original=pd.read_csv(SRC/'oof_predictions.csv');allpred=[];checks=[];budget=[];scales=[]
for seed in cfg['seeds']:
 for fold in range(3):
    src=SRC/f'seed_{seed}/fold_{fold}';out=OUT/f'seed_{seed}/fold_{fold}';out.mkdir(parents=True)
    split=pd.read_csv(src/'split.csv');assert split.pair_id.equals(pairs.pair_id)
    tr=np.flatnonzero(split.role.eq('train'));va=np.flatnonzero(split.role.eq('validation'));te=np.flatnonzero(split.role.eq('test'))
    assert_disjoint(pairs,tr,va,te,'cell');split.to_csv(out/'split.csv',index=False)
    with (src/'preprocessing.pkl').open('rb') as f:pre=pickle.load(f)
    z=np.stack([pre['Z'][j].transform(raw['Z'][:,j]) for j in range(8)],axis=1)
    d=pre['drug'].transform(raw['drug']);pi=raw['pi']
    fit=np.unique(pairs.cell_row.to_numpy()[tr])
    scales.append({'seed':seed,'fold':fold,'z_dimensions_per_compartment':z.shape[2],
       'training_mean_sum_squared_Z_per_token':float((z[fit]**2).sum(2).mean()),
       'training_mean_squared_fraction_per_token':float((pi[fit]**2).mean())})
    tp=torch.tensor(pi,device=DEV);tz=torch.tensor(z,device=DEV);zero=torch.zeros_like(tz);td=torch.tensor(d,device=DEV)
    c=torch.tensor(pairs.cell_row.to_numpy(),device=DEV);di=torch.tensor(pairs.drug_row.to_numpy(),device=DEV)
    yt=torch.tensor(pairs.auc.to_numpy(dtype='float32'),device=DEV)
    sh=pd.read_csv(src/'shuffled_labels.csv');assert sh.pair_id.equals(pairs.pair_id)
    ys=torch.tensor(sh.negative_control_response.to_numpy(dtype='float32'),device=DEV)
    # Replay saved original full checkpoints before running a changed condition.
    for name,klass in [('crdrp_full',cr.CrDRP),('transformer_full',tf.TransformerDRP)]:
        m=klass(z_dim=z.shape[2],drug_dim=d.shape[1]).to(DEV)
        with (src/f'{name}.pt').open('rb') as f:state=torch.load(f,map_location=DEV)
        m.load_state_dict(state['state_dict']);m.eval()
        with torch.no_grad():p=m(tp[c[te]],tz[c[te]],td[di[te]]);p=p[0] if name.startswith('crdrp') else p
        ref=original[(original.seed==seed)&(original.fold==fold)&(original.method==name)].set_index('pair_id').loc[pairs.iloc[te].pair_id]
        delta=float(np.max(abs(p.cpu().numpy()-ref.y_pred.to_numpy())))
        if delta>1e-6:raise ValueError(f'Checkpoint replay failure: {name} {delta}')
        checks.append({'seed':seed,'fold':fold,'method':name,'replay_max_abs_error':delta})
    for method in cfg['methods']:
        torch.manual_seed(seed*100+fold);np.random.seed(seed*100+fold)
        is_cr=method.startswith('crdrp');klass=cr.CrDRP if is_cr else tf.TransformerDRP
        model=klass(z_dim=z.shape[2],drug_dim=d.shape[1]).to(DEV)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003)
        labels=ys if method.endswith('shuffled') else yt
        def predict(ix):
            p=model(tp[c[ix]],zero[c[ix]],td[di[ix]])
            return p[0] if is_cr else p
        best=float('inf');bestep=-1;hist=[];rng=np.random.default_rng(seed*100+fold)
        for epoch in range(120):
            model.train();order=rng.permutation(tr);losses=[]
            for start in range(0,len(order),256):
                ix=order[start:start+256];opt.zero_grad();loss=torch.nn.functional.huber_loss(predict(ix),labels[ix],delta=.05)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
            model.eval()
            with torch.no_grad():vr=float(torch.sqrt(torch.mean((predict(va)-labels[va])**2)))
            hist.append({'epoch':epoch,'train_huber':float(np.mean(losses)),'validation_rmse':vr})
            if vr<best:best=vr;bestep=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch-bestep>=20:break
        model.load_state_dict(state);model.eval()
        with torch.no_grad():p=predict(te).cpu().numpy()
        pred=pairs.iloc[te][['ModelID','drug_dpc','pair_id']].copy();pred['y_true']=pairs.auc.iloc[te];pred['y_pred']=p
        pred['seed']=seed;pred['fold']=fold;pred['method']=method;allpred.append(pred)
        pred.to_csv(out/f'{method}_predictions.csv',index=False);pd.DataFrame(hist).to_csv(out/f'{method}_training.csv',index=False)
        with (out/f'{method}.pt').open('wb') as f:torch.save({'state_dict':state,'seed':seed,'fold':fold,'z_dim':z.shape[2],'drug_dim':d.shape[1],'best_epoch':bestep,'method':method,'input_condition':'standardized Z all zero'},f)
        budget.append({'seed':seed,'fold':fold,'method':method,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':bestep,'epochs_run':len(hist),'validation_rmse':best})
        logging.info('seed=%s fold=%s %s selected_epoch=%s',seed,fold,method,bestep)
pred=pd.concat(allpred,ignore_index=True);pred.to_csv(OUT/'oof_predictions.csv',index=False)
rows=[]
for (seed,method),g in pred.groupby(['seed','method']):
    assert len(g)==len(pairs) and g.pair_id.is_unique
    rows.append({'seed':seed,'method':method,**metrics(g)})
pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'checkpoint_replay.csv',index=False)
pd.DataFrame(budget).to_csv(OUT/'capacity_and_training.csv',index=False);pd.DataFrame(scales).to_csv(OUT/'input_scale_diagnostic.csv',index=False)
manifest=[]
for p in [SRC/'input_synthetic.npz',SRC/'response_synthetic.csv',SRC/'config.json',Path(__file__)]:
    manifest.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'input_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'COMPLETE.json').write_text(json.dumps({'status':'completed_diagnostic','n_oof':len(pred),'not_real_GDSC':True,'original_failure_retained':True},indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False));print(OUT)
