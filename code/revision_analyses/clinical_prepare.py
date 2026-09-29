from pathlib import Path
import pandas as pd,numpy as np,json
from scipy.stats import spearmanr,pearsonr
W=Path(__file__).resolve().parent;O=W/'clinical';ROOT=W.parent
t=pd.read_csv(ROOT/'260925_Re/data/processed/tcga_nnls_survival_20260926_145356/patient_compartment_survival.csv')
a=pd.read_csv(O/'absolute.tsv',sep='\t');print('absolute columns',a.columns.tolist());print(a.head().to_string())
c=pd.read_excel(O/'cdr.xlsx',sheet_name='TCGA-CDR');c=c[c.type.eq('BRCA')];assert c.bcr_patient_barcode.is_unique
q=t.merge(c[['bcr_patient_barcode','PFI','PFI.time']],left_on='patient_id',right_on='bcr_patient_barcode',how='left',validate='one_to_one')
q['PFI_eligible']=q.PFI.isin([0,1]) & q['PFI.time'].gt(0);q.to_csv(O/'PFI_selection.csv',index=False);q[q.PFI_eligible].to_csv(O/'PFI_inputs.csv',index=False)
col=next(k for k in a.columns if k.lower()=='purity');idcol=next(k for k in a.columns if k.lower() in ['sample','sample_id','sampleid'])
a['sample16']=a[idcol].str[:16];a['patient_id']=a[idcol].str[:12];a=a[a.sample16.str[13:15].eq('01')];a=a.groupby(['patient_id','sample16'],as_index=False)[col].median()
print('TCGA columns',t.columns.tolist())
if 'sample' in t:
 t['sample16']=t['sample'].str[:16];joined=t.merge(a,on=['patient_id','sample16'],how='inner',validate='one_to_one')
else:
 a=a.sort_values('sample16').drop_duplicates('patient_id');joined=t.merge(a,on='patient_id',how='inner',validate='one_to_one')
joined=joined.dropna(subset=[col,'malignant_epithelial']);joined.to_csv(O/'ABSOLUTE_matched.csv',index=False)
rows=[]
for exposure in ['malignant_epithelial','combined_epithelial']:
 x=joined.malignant_epithelial if exposure=='malignant_epithelial' else joined.malignant_epithelial+joined.normal_epithelial;y=joined[col]
 rng=np.random.default_rng(20260929);bs=[]
 for i in range(2000):
  ix=rng.integers(0,len(x),len(x));bs.append(spearmanr(x.iloc[ix],y.iloc[ix]).statistic)
 rows.append(dict(exposure=exposure,comparator='ABSOLUTE',n=len(x),spearman=spearmanr(x,y).statistic,pearson=pearsonr(x,y).statistic,ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975)))
pd.DataFrame(rows).to_csv(O/'purity_validation.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
