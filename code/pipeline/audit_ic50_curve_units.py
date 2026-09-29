"""Resolve scale against the same-release curve parameters, not an older release."""
import hashlib,json
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'data/raw/gdsc2';O=ROOT/'runs'/('ic50_curve_units_'+datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir()
m=pd.read_csv(R/'GDSC2Log2IC50Matrix.csv',index_col=0);v=m.rename_axis(index='ModelID',columns='CompoundID').stack(dropna=False).rename('matrix').reset_index();curves=pd.read_csv(R/'GDSC2ResponseCurves.csv');assert not curves[['ModelID','CompoundID']].duplicated().any();c=curves.merge(v,on=['ModelID','CompoundID'],validate='one_to_one')
with np.errstate(all='ignore'):
    c['derived_log2_IC50_uM']=np.log2(c.EC50)-np.log2((c.UpperAsymptote-c.LowerAsymptote)/(.5-c.LowerAsymptote)-1)/c.Slope
c['finite_crossing']=np.isfinite(c.derived_log2_IC50_uM);c['matrix_present']=c.matrix.notna();c['absolute_error']=abs(c.matrix-c.derived_log2_IC50_uM)
c.to_csv(O/'all_curve_unit_checks.csv',index=False);both=c[c.matrix_present & c.finite_crossing]
report={'definition':'absolute viability 0.5 crossing of Lower+(Upper-Lower)/(1+(dose/EC50)**(-Slope)); log2 of dose in uM','matrix_present':int(c.matrix_present.sum()),'curve_rows':len(c),'dose_units':curves.DoseUnit.value_counts().to_dict(),'crossing_and_present':len(both),'MAE_log2':float(both.absolute_error.mean()),'median_absolute_error':float(both.absolute_error.median()),'max_absolute_error':float(both.absolute_error.max()),'fraction_within_1e_3':float(both.absolute_error.lt(.001).mean()),'missing_matrix_finite_crossing':int((~c.matrix_present&c.finite_crossing).sum()),'missing_matrix_no_finite_crossing':int((~c.matrix_present&~c.finite_crossing).sum()),'source_files':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['GDSC2Log2IC50Matrix.csv','GDSC2ResponseCurves.csv']},'supersedes':'earlier cross-release Sanger comparison was unresolved; no values replaced or fitted','resolved':bool(both.absolute_error.mean()<1e-4 and both.absolute_error.lt(.001).mean()>.999)}
(O/'unit_resolution.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print(O)
