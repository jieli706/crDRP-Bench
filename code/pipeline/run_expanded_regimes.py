"""Run remaining independent regimes sequentially to limit RAM; preserve every run."""
import subprocess,sys,json
from pathlib import Path
from datetime import datetime
root=Path(__file__).resolve().parents[1];out=root/'runs'/('expanded_regime_driver_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
rows=[]
for regime in ['pair','drug']:
    cmd=[sys.executable,str(root/'code/real_gdsc_benchmark.py'),'--features',str(root/'data/processed/expanded_gdsc_features_20260926_142255'),'--smoke-run',str(root/'runs/real_gdsc_smoke_20260926_142456'),'--regime',regime]
    print('Starting',regime,flush=True)
    with (out/(regime+'.log')).open('w',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    rows.append({'regime':regime,'returncode':r.returncode});(out/'status.json').write_text(json.dumps(rows,indent=2));print('Finished',regime,r.returncode,flush=True)
    if r.returncode:raise SystemExit(r.returncode)
print(out)
