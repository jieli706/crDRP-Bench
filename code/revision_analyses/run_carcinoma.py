from pathlib import Path
import subprocess,sys,json
W=Path(__file__).resolve().parent;ROOT=W.parent;R=ROOT/'260925_Re/runs';base=R/'real_gdsc_cell_20260926_151327'
args=[sys.executable,str(W/'carcinoma_transfer.py'),'--base-run',str(base)]
subprocess.run(args+['--smoke'],check=True)
smoke=sorted(R.glob('carcinoma_transfer_smoke_*'))[-1]
assert json.loads((smoke/'COMPLETE.json').read_text())['all_checks_passed']
subprocess.run(args+['--smoke-run',str(smoke)],check=True)
out=sorted(p for p in R.glob('carcinoma_transfer_*') if 'smoke' not in p.name)[-1]
(W/'carcinoma_run.txt').write_text(str(out))
