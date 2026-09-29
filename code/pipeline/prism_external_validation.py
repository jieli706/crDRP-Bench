"""Cross-assay validation of frozen GDSC cell-held-out OOF models, no PRISM fitting."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd
from leakage_safe import metrics
from paired_two_way_bootstrap import paired_bootstrap
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'runs'/('prism_external_validation_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir()
F=ROOT/'data/processed/real_gdsc_features_20260926_135410';B=ROOT/'runs/real_gdsc_cell_20260926_140256';R=ROOT/'data/raw/prism'
p=argparse.ArgumentParser();p.add_argument('--base-run');p.add_argument('--features');a=p.parse_args()
if a.base_run:B=Path(a.base_run).resolve()
if a.features:F=Path(a.features).resolve()
drugs=pd.read_csv(F/'drug_ids.csv');pr=pd.read_csv(R/'secondary-screen-replicate-treatment-info.csv',usecols=['broad_id','name','smiles']).drop_duplicates().reset_index(drop=True)
pr['broad_id_13']=pr.broad_id.str[:13];mapping=[]
for row in drugs.itertuples(index=False):
    if hasattr(row,'structure_source') and row.structure_source!='PRISM19Q4':continue
    if pd.isna(row.matched_PRISM_metadata_rows):continue
    indices=[int(x) for x in row.matched_PRISM_metadata_rows.split(';')];ids=pr.iloc[indices].broad_id_13.unique()
    for bid in ids:mapping.append({'drug_dpc':row.drug_dpc,'broad_id_13':bid,'drug_name':row.drug_name})
mapping=pd.DataFrame(mapping).drop_duplicates();amb=mapping.groupby('broad_id_13').drug_dpc.nunique();mapping['one_GDSC_drug']=mapping.broad_id_13.map(amb).eq(1);mapping.to_csv(O/'drug_mapping.csv',index=False);mp=mapping[mapping.one_GDSC_drug].set_index('broad_id_13').drug_dpc.to_dict()
cols=['broad_id','depmap_id','screen_id','auc','ec50','ic50','r2','passed_str_profiling','name'];p=pd.read_csv(R/'secondary-screen-dose-response-curve-parameters.csv',usecols=cols);p['source_row']=np.arange(len(p));p['broad_id_13']=p.broad_id.str[:13];p['drug_dpc']=p.broad_id_13.map(mp);cells=pd.read_csv(F/'sample_ids.csv').ModelID;p['eligible_cell']=p.depmap_id.isin(cells);p['eligible_drug']=p.drug_dpc.notna();p['finite_auc']=np.isfinite(p.auc);p['STR_pass']=p.passed_str_profiling.astype(str).str.lower().eq('true');p['selected']=p.eligible_cell&p.eligible_drug&p.finite_auc&p.STR_pass
p.loc[p.eligible_cell|p.eligible_drug].to_csv(O/'source_selection_audit.csv',index=False);q=p[p.selected].copy();q.to_csv(O/'selected_raw_response_rows.csv',index=False)
truth=q.groupby(['depmap_id','drug_dpc']).agg(prism_auc=('auc','median'),replicate_rows=('auc','size'),min_auc=('auc','min'),max_auc=('auc','max'),n_screens=('screen_id','nunique'),median_curve_r2=('r2','median')).reset_index().rename(columns={'depmap_id':'ModelID'});truth.to_csv(O/'prism_pair_truth.csv',index=False)
prediction_file=B/('paired_baseline_predictions.csv' if (B/'paired_baseline_predictions.csv').exists() else 'oof_predictions.csv')
pred=pd.read_csv(prediction_file);pred=pred.rename(columns={'y_true':'gdsc_auc'}).merge(truth,on=['ModelID','drug_dpc'],validate='many_to_one');pred['y_true']=pred.prism_auc
assert pred.groupby(['method','seed']).pair_id.nunique().nunique()==1
pred.to_csv(O/'external_predictions_frozen_GDSC_models.csv',index=False)
rows=[]
for (seed,method),g in pred.groupby(['seed','method']):rows.append({'seed':seed,'method':method,**metrics(g)})
pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);sums=[]
for seed in sorted(pred.seed.unique()):
 for a,b in [('crdrp_full_nnls','crdrp_bulk'),('transformer_full_nnls','transformer_bulk')]:
    s,draws,w=paired_bootstrap(pred[pred.seed==seed],a,b,1000,int(seed));s['seed']=seed;sums.append(s);tag=f'{seed}_{a}_minus_{b}';draws.to_csv(O/('bootstrap_'+tag+'.csv'),index=False);np.savez_compressed(O/('bootstrap_weights_'+tag+'.npz'),**w)
pd.concat(sums).to_csv(O/'paired_two_way_bootstrap.csv',index=False)
unique=pred[['ModelID','drug_dpc','gdsc_auc','prism_auc']].drop_duplicates();unique.to_csv(O/'cross_assay_pairs.csv',index=False)
report={'scope':'external assay on intersecting cell-drug pairs, frozen cell-held-out GDSC OOF models; NOT independent biological cohort','version':'PRISM Repurposing 19Q4','no_PRISM_training_tuning_or_preprocessing':True,'selection':'STR-pass, finite AUC, exact structure-verified Broad-ID mapping and existing 45-cell cohort; no R2 or AUC value threshold','technical_replicates':'median across retained source rows, counts/screens/range preserved; units dimensionless AUC, assay dose ranges differ from GDSC','do_not_clip_AUC':True,'source_rows':len(p),'selected_source_rows':len(q),'selected_auc_outside_0_1':int((~q.auc.between(0,1)).sum()),'unique_selected_pairs':len(truth),'evaluated_pairs':len(unique),'n_cells':unique.ModelID.nunique(),'n_drugs':unique.drug_dpc.nunique(),'pooled_assay_agreement':float(unique[['gdsc_auc','prism_auc']].corr().iloc[0,1]),'prior_model_directory':str(B.relative_to(ROOT)),'source_sha256':hashlib.sha256((R/'secondary-screen-dose-response-curve-parameters.csv').read_bytes()).hexdigest()}
(O/'config_and_audit.json').write_text(json.dumps(report,indent=2));(O/'COMPLETE.json').write_text(json.dumps({'status':'complete','scope':report['scope']}));print(json.dumps(report,indent=2));print(O)
