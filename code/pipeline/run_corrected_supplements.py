"""Use completed corrected folds for supplements; new directories only."""
import argparse,subprocess,sys,json
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--correction-index',required=True);a=p.parse_args();fix=json.loads(Path(a.correction_index).read_text());base=fix['cell'];O=ROOT/'runs'/('corrected_supplements_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir();results={}
def launch(script,extra,tag,glob):
    before=set((ROOT/'runs').glob(glob))
    with (O/(tag+'.log')).open('w',encoding='utf-8') as log:r=subprocess.run([sys.executable,str(ROOT/'code'/script)]+extra,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(tag+' failed')
    new=[x for x in set((ROOT/'runs').glob(glob))-before if (x/'COMPLETE.json').exists()];assert len(new)==1;results[tag]=str(new[0]);(O/'results.json').write_text(json.dumps(results,indent=2));print(tag,new[0],flush=True);return new[0]
launch('prism_external_validation.py',['--base-run',base,'--features',str(ROOT/'data/processed/expanded_gdsc_features_20260926_142255')],'prism','prism_external_validation_*')
launch('real_response_diagnostics.py',['--base-run',base],'variance','real_response_diagnostics_*')
smoke=launch('transfer_gdsc_benchmark.py',['--base-run',base,'--smoke'],'transfer_smoke','transfer_gdsc_*')
launch('transfer_gdsc_benchmark.py',['--base-run',base,'--smoke-run',str(smoke)],'transfer','transfer_gdsc_*')
(O/'COMPLETE.json').write_text(json.dumps(results,indent=2));print(O,flush=True)
