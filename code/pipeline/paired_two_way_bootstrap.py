"""Paired crossed cell/drug bootstrap for fixed OOF predictions.
Product multinomial weights (pigeonhole bootstrap); no model refitting.
References: Owen & Eckles, https://arxiv.org/abs/1106.2125 .
Finite-sample coverage for a median of correlations is not guaranteed by mean theory.
"""
import numpy as np
import pandas as pd

def weighted_corr(y,p,w):
    keep=w>0;y=y[keep];p=p[keep];w=w[keep]
    if len(y)<3 or not w.sum():return np.nan
    ym=np.average(y,weights=w);pm=np.average(p,weights=w)
    yy=np.dot(w,(y-ym)**2);pp=np.dot(w,(p-pm)**2)
    return np.dot(w,(y-ym)*(p-pm))/np.sqrt(yy*pp) if min(yy,pp)>1e-16 else np.nan

def paired_bootstrap(frame,method_a,method_b,n_boot=1000,seed=7,cell_column='ModelID',drug_column='drug_dpc',cell_cluster_column=None):
    keys=[cell_column,drug_column]
    a=frame[frame.method==method_a].copy();b=frame[frame.method==method_b].copy()
    if a.duplicated(keys).any() or b.duplicated(keys).any():raise ValueError('Pass one seed and regime; predictions must be unique per cell-drug')
    matched=a.merge(b,on=keys,how='outer',suffixes=('_a','_b'),indicator=True,validate='one_to_one')
    if not matched._merge.eq('both').all():raise ValueError('Unmatched prediction pairs; do not silently intersect')
    if not np.allclose(matched.y_true_a,matched.y_true_b,atol=1e-10):raise ValueError('Outcome mismatch')
    if 'fold_a' in matched and not matched.fold_a.equals(matched.fold_b):raise ValueError('Models use different outer folds')
    y=matched.y_true_a.to_numpy();pa=matched.y_pred_a.to_numpy();pb=matched.y_pred_b.to_numpy()
    if not np.isfinite(np.c_[y,pa,pb]).all():raise ValueError('Nonfinite outcomes/predictions')
    cluster=matched[cell_column]
    if cell_cluster_column and cell_cluster_column!=cell_column:
        ca=matched[cell_cluster_column+'_a'];cb=matched[cell_cluster_column+'_b']
        if ca.isna().any() or not ca.equals(cb):raise ValueError('Cluster IDs missing or mismatched')
        if matched.assign(_cluster=ca).groupby(cell_column)._cluster.nunique().gt(1).any():raise ValueError('Cell maps to multiple clusters')
        cluster=ca
    cells,codes_c=np.unique(cluster,return_inverse=True);drugs,codes_d=np.unique(matched[drug_column],return_inverse=True)
    indices=[np.flatnonzero(codes_d==i) for i in range(len(drugs))]
    def evaluate(cw,dw):
        w=cw[codes_c]*dw[codes_d];ra=[];rb=[];paired=[];invalid=0
        for i,ix in enumerate(indices):
            if dw[i]==0:continue
            r1=weighted_corr(y[ix],pa[ix],cw[codes_c[ix]]);r2=weighted_corr(y[ix],pb[ix],cw[codes_c[ix]])
            if np.isfinite(r1) and np.isfinite(r2):
                ra.extend([r1]*int(dw[i]));rb.extend([r2]*int(dw[i]));paired.extend([r1-r2]*int(dw[i]))
            else:invalid+=int(dw[i])
        return {'delta_median_perdrug_r':float(np.median(ra)-np.median(rb)) if ra else np.nan,
                'median_paired_drug_delta_r':float(np.median(paired)) if paired else np.nan,
                'delta_rmse':float(np.sqrt(np.average((y-pa)**2,weights=w))-np.sqrt(np.average((y-pb)**2,weights=w))) if w.sum() else np.nan,
                'n_valid_drug_draws':len(ra),'n_invalid_drug_draws':invalid}
    point=evaluate(np.ones(len(cells),int),np.ones(len(drugs),int));rng=np.random.default_rng(seed);draws=[];cw_save=[];dw_save=[]
    for i in range(n_boot):
        cw=rng.multinomial(len(cells),np.full(len(cells),1/len(cells)))
        dw=rng.multinomial(len(drugs),np.full(len(drugs),1/len(drugs)))
        draws.append({'replicate':i,**evaluate(cw,dw)});cw_save.append(cw);dw_save.append(dw)
    draws=pd.DataFrame(draws);summary=[]
    for metric in ['delta_median_perdrug_r','median_paired_drug_delta_r','delta_rmse']:
        v=draws[metric].dropna()
        summary.append({'method_a':method_a,'method_b':method_b,'metric':metric,'effect_a_minus_b':point[metric],
                        'ci_low':float(v.quantile(.025)) if len(v) else np.nan,'ci_high':float(v.quantile(.975)) if len(v) else np.nan,
                        'valid_replicates':len(v),'requested_replicates':n_boot,'n_cells':len(cells),'n_drugs':len(drugs),
                        'n_point_valid_drugs':point['n_valid_drug_draws'],'n_point_invalid_drugs':point['n_invalid_drug_draws'],
                        'bootstrap_seed':seed,'scheme':'paired_product_multinomial_cells_times_drugs' if not cell_cluster_column else 'paired_product_multinomial_clusters_times_drugs','cell_resampling_unit':cell_cluster_column or cell_column,'scope':'conditional_on_fixed_OOF_models'})
    return pd.DataFrame(summary),draws,{'cell_ids':cells,'drug_ids':drugs,'cell_weights':np.array(cw_save),'drug_weights':np.array(dw_save)}

def verify_bootstrap():
    # Same models must give exactly zero in every paired replicate; multiplicity must equal explicit resampling.
    rows=[{'ModelID':f'C{c}','drug_dpc':f'D{d}','y_true':c+.1*d,'y_pred':c*.9+.12*d,'fold':c%2,'method':m} for c in range(10) for d in range(4) for m in ['a','b']]
    summary,draws,_=paired_bootstrap(pd.DataFrame(rows),'a','b',80)
    assert np.allclose(draws[['delta_median_perdrug_r','delta_rmse']].dropna(),0)
    y=np.array([1.,4.,2.,9.]);p=np.array([2.,2.,3.,8.]);w=np.array([2,1,3,1])
    assert np.allclose(weighted_corr(y,p,w),np.corrcoef(np.repeat(y,w),np.repeat(p,w))[0,1])
    try:paired_bootstrap(pd.DataFrame(rows).iloc[1:],'a','b',2)
    except ValueError:pass
    else:raise AssertionError('Unpaired inputs accepted')
    clustered=pd.DataFrame(rows);clustered['donor_id']=clustered.ModelID.str[1:].astype(int).floordiv(2).astype(str)
    _,cd,cw=paired_bootstrap(clustered,'a','b',20,cell_cluster_column='donor_id')
    assert len(cw['cell_ids'])==5 and np.allclose(cd.delta_rmse,0)
    return {'identical_model_zero_effect':True,'multiplicity_matches_explicit_resample':True,'unmatched_pair_rejected':True,'clustered_identical_model_zero_effect':True}
