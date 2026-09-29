# -*- coding: utf-8 -*-
"""Clinical-only survival diagnostics. Does not infer compartment effects."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime/packages'))
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/matplotlib_config')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'runs'/('clinical_survival_intake_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir()
RAW=ROOT/'data/raw/bulk'
c=pd.read_csv(RAW/'tcga/TCGA-BRCA.clinical.tsv.gz',sep='\t',low_memory=False)
s=pd.read_csv(RAW/'tcga/TCGA-BRCA.survival.tsv.gz',sep='\t')
m=pd.read_csv(RAW/'metabric/data_clinical_patient.txt',sep='\t',comment='#')
ms=pd.read_csv(RAW/'metabric/data_clinical_sample.txt',sep='\t',comment='#')
assert c['sample'].is_unique and s['sample'].is_unique and m.PATIENT_ID.is_unique
sources=[]
for p in RAW.rglob('*'):
 if p.is_file():sources.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
# Select primary tumor by BOTH barcode and explicit annotation; record all exclusions.
t=c.merge(s,on='sample',how='left',validate='one_to_one',indicator=True)
t['patient_id']=t['sample'].str.slice(0,12);t['barcode_primary']=t['sample'].str.split('-').str[3].str[:2].eq('01')
t['annotation_primary']=t['sample_type.samples'].eq('Primary Tumor');t['time_months']=pd.to_numeric(t['OS.time'],errors='coerce')/365.25*12;t['event']=pd.to_numeric(t.OS,errors='coerce')
t['age_years']=pd.to_numeric(t['age_at_earliest_diagnosis_in_years.diagnoses.xena_derived'],errors='coerce')
t['stage']=t['ajcc_pathologic_stage.diagnoses'];t['reason']='eligible'
t.loc[~(t.barcode_primary&t.annotation_primary),'reason']='not_consistent_primary_tumor'
t.loc[(t.reason=='eligible')&t._merge.ne('both'),'reason']='no_survival_match'
t.loc[(t.reason=='eligible')&(~t.event.isin([0,1])|t.time_months.isna()),'reason']='missing_or_invalid_survival'
t.loc[(t.reason=='eligible')&t.time_months.le(0),'reason']='nonpositive_followup'
candidate=t[t.reason=='eligible']
conflict=candidate.groupby('patient_id')[['time_months','event']].nunique().gt(1).any(axis=1)
conflicting_ids=conflict[conflict].index
t.loc[(t.reason=='eligible')&t.patient_id.isin(conflicting_ids),'reason']='conflicting_patient_survival'
eligible=t[t.reason=='eligible'].sort_values('sample');repeat=eligible[eligible.patient_id.duplicated()].index
t.loc[repeat,'reason']='duplicate_patient_lexicographic_sample_retained'
t.to_csv(OUT/'TCGA_sample_selection.csv',index=False)
td=t[t.reason=='eligible'][['patient_id','sample','time_months','event','age_years','stage']].copy();td.to_csv(OUT/'TCGA_patient_survival.csv',index=False)

mm=m.copy();mm['patient_id']=mm.PATIENT_ID;mm['time_months']=pd.to_numeric(mm.OS_MONTHS,errors='coerce');mm['event']=mm.OS_STATUS.map({'0:LIVING':0,'1:DECEASED':1});mm['age_years']=pd.to_numeric(mm.AGE_AT_DIAGNOSIS,errors='coerce')
mm['reason']='eligible';mm.loc[mm.time_months.isna()|mm.event.isna(),'reason']='missing_or_invalid_survival';mm.loc[(mm.reason=='eligible')&mm.time_months.le(0),'reason']='nonpositive_followup'
mm.to_csv(OUT/'METABRIC_patient_selection.csv',index=False)
md=mm[mm.reason=='eligible'].copy();md.to_csv(OUT/'METABRIC_patient_survival.csv',index=False)
ms.groupby('PATIENT_ID').size().rename('n_samples').to_csv(OUT/'METABRIC_samples_per_patient.csv')

def km(time,event):
    out=[{'time_months':0.,'at_risk':len(time),'events':0,'censored':0,'survival':1.}];surv=1.
    time=np.asarray(time);event=np.asarray(event)
    for tt in np.sort(np.unique(time)):
        risk=(time>=tt).sum();death=((time==tt)&(event==1)).sum();cens=((time==tt)&(event==0)).sum();surv*=1-death/risk
        out.append({'time_months':tt,'at_risk':int(risk),'events':int(death),'censored':int(cens),'survival':surv})
    return pd.DataFrame(out)
def median_km(k):
    below=k[k.survival<=.5];return float(below.time_months.iloc[0]) if len(below) else None
summaries=[];missing=[]
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,(cohort,frame) in zip(axes,[('TCGA primary tumors',td),('METABRIC clinical records',md)]):
    k=km(frame.time_months,frame.event);reverse=km(frame.time_months,1-frame.event)
    key=cohort.split()[0];k.to_csv(OUT/f'{key}_unadjusted_KM.csv',index=False);reverse.to_csv(OUT/f'{key}_reverse_KM_followup.csv',index=False)
    summaries.append({'cohort':cohort,'n_patients':len(frame),'deaths':int(frame.event.sum()),'censored':int((frame.event==0).sum()),'median_OS_months':median_km(k),'reverse_KM_median_followup_months':median_km(reverse),'min_months':float(frame.time_months.min()),'max_months':float(frame.time_months.max()),'age_missing':int(frame.age_years.isna().sum())})
    ax.step(k.time_months,k.survival,where='post');ax.set(xlabel='Months',ylabel='Unadjusted survival probability',ylim=(0,1.02),title=f'{cohort}\nn={len(frame)}, deaths={int(frame.event.sum())}')
fig.suptitle('Data intake only: no expression intersection or compartment adjustment',fontsize=10);fig.savefig(OUT/'unadjusted_survival_diagnostic.png',dpi=160);plt.close(fig)
summary=pd.DataFrame(summaries);summary.to_csv(OUT/'clinical_survival_summary.csv',index=False)
for cohort,frame,cols in [('TCGA',t,['age_years','stage']),('METABRIC',mm,['age_years','CLAUDIN_SUBTYPE','ER_IHC','VITAL_STATUS'])]:
 for col in cols:
    for value,g in frame.groupby(col,dropna=False):
        if col=='age_years':continue
        missing.append({'cohort':cohort,'variable':col,'category':str(value),'n_records':len(g),'survival_unusable':int((g.reason!='eligible').sum())})
pd.DataFrame(missing).to_csv(OUT/'survival_availability_by_covariates.csv',index=False)
audit={'TCGA_clinical_rows':len(c),'TCGA_survival_rows':len(s),'TCGA_sample_type_counts':c['sample_type.samples'].value_counts().to_dict(),'TCGA_primary_annotation_barcode_disagreement':int((t.barcode_primary!=t.annotation_primary).sum()),'TCGA_selection_reasons':t.reason.value_counts().to_dict(),'TCGA_conflicting_patient_ids':list(conflicting_ids),'METABRIC_patient_rows':len(m),'METABRIC_sample_rows':len(ms),'METABRIC_selection_reasons':mm.reason.value_counts().to_dict(),'METABRIC_unknown_OS_codes':m.OS_STATUS.dropna()[~m.OS_STATUS.dropna().isin(['0:LIVING','1:DECEASED'])].unique().tolist(),'scope':'clinical_only_not_original_expression_matched_cohort','Cox_PH_Schoenfeld_test':'blocked_no_compartment_proportions','B_plasma_survival_replication':'not_run'}
(OUT/'clinical_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
(OUT/'COMPLETE.json').write_text(json.dumps({'status':'completed','scope':'clinical_survival_data_intake_only','no_compartment_HR_estimated':True},indent=2),encoding='utf-8')
print(json.dumps(audit));print(summary.to_string(index=False));print(OUT)
