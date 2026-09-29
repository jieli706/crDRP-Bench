"""Training-variance strata and fold-aware correlation diagnostics, not selection."""
import argparse,json,pickle
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from leakage_safe import safe_corr
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--base-run',required=True);a=p.parse_args();B=Path(a.base_run).resolve();O=ROOT/'runs'/('real_response_diagnostics_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir()
pred=pd.read_csv(B/('paired_baseline_predictions.csv' if (B/'paired_baseline_predictions.csv').exists() else 'oof_predictions.csv'));trainv=pd.read_csv(B/'training_response_variance.csv');frame=pd.read_csv(B/'selected_response_pairs.csv');rows=[]
for (seed,method,fold,drug),g in pred.groupby(['seed','method','fold','drug_dpc']):
    rows.append({'seed':seed,'method':method,'fold':fold,'drug_dpc':drug,'n':len(g),'response_std':g.y_true.std(),'prediction_std':g.y_pred.std(),'pearson':safe_corr(g.y_true,g.y_pred),'rmse':np.sqrt(np.mean((g.y_true-g.y_pred)**2))})
d=pd.DataFrame(rows).merge(trainv.rename(columns={'count':'training_n','std':'training_response_std'}),on=['seed','fold','drug_dpc'],how='left',validate='many_to_one');d['variance_stratum']=pd.cut(d.training_response_std,[-np.inf,.02,.05,.1,np.inf],labels=['SD<0.02','0.02<=SD<0.05','0.05<=SD<0.10','SD>=0.10'],right=False).astype(str);d['prediction_SD_under_0.01']=d.prediction_std.lt(.01);d.to_csv(O/'fold_drug_metrics_and_training_variance.csv',index=False)
s=d.groupby(['seed','method','variance_stratum']).agg(fold_drug_groups=('n','size'),valid_correlations=('pearson','count'),median_correlation=('pearson','median'),median_prediction_SD=('prediction_std','median'),median_response_SD=('response_std','median'),median_group_RMSE=('rmse','median'),fraction_prediction_SD_under_0_01=('prediction_SD_under_0.01','mean')).reset_index();s.to_csv(O/'training_variance_strata_summary.csv',index=False)
fw=[]
for (seed,method,drug),g in d.groupby(['seed','method','drug_dpc']):
    v=g[np.isfinite(g.pearson)&g.n.gt(3)];weights=v.n.to_numpy()-3
    r=float(np.tanh(np.average(np.arctanh(np.clip(v.pearson,-.999999,.999999)),weights=weights))) if len(v) else np.nan
    fw.append({'seed':seed,'method':method,'drug_dpc':drug,'valid_folds':len(v),'fold_weighted_Fisher_r':r})
fw=pd.DataFrame(fw);fw.to_csv(O/'within_fold_fisher_correlations.csv',index=False);fw.groupby(['seed','method']).agg(n_valid_drugs=('fold_weighted_Fisher_r','count'),median_fold_weighted_r=('fold_weighted_Fisher_r','median')).reset_index().to_csv(O/'within_fold_summary.csv',index=False)
mean=frame.auc.mean();drugmeans=frame.groupby('drug_dpc').auc.agg(['mean','count','std']);share=float(np.sum(drugmeans['count']*(drugmeans['mean']-mean)**2)/np.sum((frame.auc-mean)**2));drugmeans.to_csv(O/'response_variance_by_drug.csv')
fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].hist(drugmeans['std'],bins=25,color='#3c6f9b');axes[0].set(xlabel='Observed within-drug AUC SD',ylabel='Number of drugs',title='Descriptive variance; no outcome filtering')
for method,color in [('crdrp_full_nnls','#C45345'),('crdrp_bulk','#316B9D'),('global_xgb','#39784E')]:
    v=d[(d.seed==7)&d.method.eq(method)];axes[1].scatter(v.response_std,v.prediction_std,s=10,alpha=.4,label=method,color=color)
axes[1].plot([0,.4],[0,.4],'k--',lw=.8);axes[1].set(xlabel='Test fold/drug observed AUC SD',ylabel='Prediction SD',title='Seed 7: all folds and drugs');axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(O/'variance_and_prediction_spread.png',dpi=180);plt.close(fig)
summary={'base_run':str(B.relative_to(ROOT)),'n_pairs':len(frame),'n_drugs':len(drugmeans),'descriptive_fraction_response_sum_squares_between_drugs':share,'strata_thresholds':[.02,.05,.1],'strata_assignment':'training-fold response SD only; no refitting or test-driven selection','within_fold_metric':'Fisher transform of within-fold per-drug r weighted by n-3, then median across drugs; complementary diagnostic, not silently replacing prespecified OOF metric','constant_prediction':'Per-drug constant predictors have undefined within-fold correlation. Their pooled OOF correlations may be nonzero due to different fold-specific training means; do not interpret that as biological ranking.','fixed_prediction_SD_threshold':.01,'clinical_or_causal_inference':False}
(O/'config_and_summary.json').write_text(json.dumps(summary,indent=2));(O/'COMPLETE.json').write_text(json.dumps({'status':'completed'}));print(json.dumps(summary,indent=2));print(O)
