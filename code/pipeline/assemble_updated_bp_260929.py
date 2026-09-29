"""Model-ready matched raw-count branch; no predictor feature fitting here."""
import argparse,json,hashlib,shutil
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--bayes-run');a=p.parse_args();S=ROOT/'data/processed/real_count_deconvolution_20260926_142741';F=ROOT/'data/processed/expanded_gdsc_features_20260926_142255';variant='bayesprism_updated_samplewise' if a.bayes_run else 'nnls';O=ROOT/'data/processed'/('count_features_'+variant+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir()
d=np.load(S/'real_count_inputs.npz');counts=d['counts'];ids=d['cell_ids'];comp=d['compartments'];genes=d['gene_ids'];sources=[S/'config.json',F/'source_hashes.json',Path(__file__)]
if a.bayes_run:
    B=Path(a.bayes_run).resolve();assert (B/'COMPLETE.txt').exists();pi=pd.read_csv(B/'fractions.csv',index_col=0).loc[ids,comp].to_numpy();allocated=np.stack([pd.read_csv(B/f'Z_{k}.csv.gz',index_col=0).reindex(index=ids,columns=genes,fill_value=0).to_numpy() for k in comp],axis=1);err=float(np.max(abs(allocated.sum(1)-counts)));assert err<1e-3;mass=allocated.sum(2,keepdims=True);norm=np.divide(allocated,mass,out=np.zeros_like(allocated),where=mass>0);z=np.log1p(norm*1e6);sources.extend([B/'fractions.csv',B/'CONFIG.txt'])
else:pi=d['nnls_pi'];z=d['nnls_Z_log1pCPM'];err=None
assert np.isfinite(pi).all() and np.isfinite(z).all() and np.allclose(pi.sum(1),1,atol=1e-6)
np.save(O/'deconvolution_pi.npy',pi.astype('float32'));np.save(O/'deconvolution_Z_log1pCPM.npy',z.astype('float32'));np.save(O/'bulk_log1pCPM.npy',np.log1p(counts/counts.sum(1,keepdims=True)*1e6).astype('float32'))
for name in ['sample_ids.csv','drug_ids.csv','drug_raw_features.npy','drug_feature_names.csv','response_pairs.csv']:shutil.copy2(F/name,O/name)
assert np.array_equal(pd.read_csv(O/'sample_ids.csv').ModelID,ids)
cfg={'input_variant':variant+' on identical measured DepMap raw counts and 12-donor reference; matched count branch','original_canonical_pipeline':False,'pi_array':'deconvolution_pi.npy','Z_array':'deconvolution_Z_log1pCPM.npy','bulk_array':'bulk_log1pCPM.npy','n_cells':len(ids),'n_drugs':len(pd.read_csv(F/'drug_ids.csv')),'n_genes':len(genes),'n_pairs':len(pd.read_csv(F/'response_pairs.csv')),'real_count_input_directory':str(S),'chemical_feature_directory':str(F),'bayes_run':a.bayes_run,'count_reconstruction_max_error':err,'no_predictor_preprocessing_fit':True,'comparison_requirement':'same outer folds, training-only shared Z PCA; count-based bulk baseline must be retrained, cannot reuse TPM-based bulk','BayesPrism_definition':'Full BayesPrism updated separately per sample, final fractions and standard get.exp initial expression; no test-cohort pooling','remaining_platform_difference':'bulk read counts vs UMI counts; no true compartment fractions in cell lines'}
(O/'config.json').write_text(json.dumps(cfg,indent=2));(O/'source_hashes.json').write_text(json.dumps([{'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sources],indent=2));(O/'COMPLETE.json').write_text(json.dumps({'status':'count_features_ready','variant':variant}));print(O)

