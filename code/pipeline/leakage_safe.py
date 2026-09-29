"""Minimal reusable guards and fold-fitted transforms for the rebuilt benchmark."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, GroupShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler

def outer_splits(pairs,regime,seed,n_splits=5):
    if pairs[['ModelID','drug_dpc']].duplicated().any():
        raise ValueError('Resolve technical replicates before splitting cell-drug pairs')
    if regime=='pair': return list(KFold(n_splits,shuffle=True,random_state=seed).split(pairs))
    key={'cell':'ModelID','drug':'drug_dpc'}[regime]
    groups=np.sort(pairs[key].unique())
    return [(np.flatnonzero(pairs[key].isin(groups[a])),np.flatnonzero(pairs[key].isin(groups[b])))
            for a,b in KFold(n_splits,shuffle=True,random_state=seed).split(groups)]

def inner_split(pairs,train,regime,seed):
    if regime=='pair': a,b=next(ShuffleSplit(1,test_size=.2,random_state=seed).split(train))
    else:
        key={'cell':'ModelID','drug':'drug_dpc'}[regime]
        a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(train,groups=pairs.iloc[train][key]))
    return train[a],train[b]

def assert_disjoint(pairs,train,val,test,regime):
    for a,b in [(train,val),(train,test),(val,test)]:
        assert not set(a)&set(b),'Pair overlap'
        if regime!='pair':
            key={'cell':'ModelID','drug':'drug_dpc'}[regime]
            assert not set(pairs.iloc[a][key])&set(pairs.iloc[b][key]),'Entity overlap'

def permitted_pretrain(source,target_cells,pairs,val,test,regime):
    keep=~source.ModelID.isin(target_cells)
    if regime=='drug':keep &= ~source.drug_dpc.isin(pairs.iloc[np.r_[val,test]].drug_dpc)
    return source.loc[keep].copy()

class FoldTransform:
    def __init__(self,n_components=8,max_features=2000,seed=7):
        self.n_components=n_components;self.max_features=max_features;self.seed=seed
    def fit(self,x,ids):
        x=np.asarray(x,dtype=np.float64)
        if len(set(ids))!=len(ids) or len(x)!=len(ids):raise ValueError('Fit on unique entity IDs')
        if np.isinf(x).any():raise ValueError('Infinite feature')
        self.fit_ids_=list(ids)
        self.valid_=np.flatnonzero(np.any(np.isfinite(x),axis=0))
        if not len(self.valid_):raise ValueError('All features missing')
        self.imputer_=SimpleImputer(strategy='median').fit(x[:,self.valid_])
        a=self.imputer_.transform(x[:,self.valid_])
        self.selected_=np.argsort(-np.var(a,axis=0),kind='stable')[:self.max_features]
        a=a[:,self.selected_];self.scale_=StandardScaler().fit(a)
        a=self.scale_.transform(a)
        d=min(self.n_components,len(a)-1,a.shape[1])
        if d<1:raise ValueError('Insufficient training entities')
        self.pca_=PCA(n_components=d,svd_solver='full',random_state=self.seed).fit(a)
        self.post_=StandardScaler().fit(self.pca_.transform(a))
        return self
    def transform(self,x):
        a=self.imputer_.transform(np.asarray(x,dtype=np.float64)[:,self.valid_])[:,self.selected_]
        return self.post_.transform(self.pca_.transform(self.scale_.transform(a))).astype('float32')

def safe_corr(y,p):
    if len(y)<3 or np.std(y)<1e-10 or np.std(p)<1e-10:return float('nan')
    return float(np.corrcoef(y,p)[0,1])

def metrics(df):
    rs=[safe_corr(g.y_true,g.y_pred) for _,g in df.groupby('drug_dpc')]
    valid=np.asarray(rs)[np.isfinite(rs)]
    y=df.y_true.to_numpy();p=df.y_pred.to_numpy()
    return {'n_pairs':len(df),'n_drugs_total':len(rs),'n_drugs_valid':len(valid),
            'perdrug_pearson_median':float(np.median(valid)) if len(valid) else float('nan'),
            'pooled_pearson':safe_corr(y,p),'rmse':float(np.sqrt(np.mean((y-p)**2))),
            'mae':float(np.mean(abs(y-p))),'r2':float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))}
