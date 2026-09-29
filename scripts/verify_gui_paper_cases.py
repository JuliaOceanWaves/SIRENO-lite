"""Compare five packaged annual GUI solves with retained paper results.

Run from repository root; requires the nested paper checkout. Typical runtime
is a few minutes. Numerical results and complete hourly traces go to
results/gui_paper_review/.
"""
from pathlib import Path
import json,time
import pandas as pd,numpy as np
from sirenolite import presets,model
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/gui_paper_review';out.mkdir(parents=True,exist_ok=True);ref=pd.read_csv(ROOT/'paper/paper_reproducibility/results/global_mission_cases/global_mission_case_summary.csv')
records=[]
for site,mission in [(presets.DEFAULT_SITE_KEY,m) for m in ['combined','scheduled_charging','pioneer_100w','potable_water']]+[('anchorage_alaska_46081_oneyear','scheduled_charging')]:
 c=presets.load_site_config(site,mission);t=time.monotonic();inputs=model.load_resource_inputs(c)
 m,v=model.build_model(c,inputs);model.solve_model(c,(m,v))
 r=ref[(ref.site_key==site)&(ref.mission_key==mission)].iloc[0]
 diffs={}
 for key,col in {'total_cost':'total_cost_usd','generator_scale':'generator_capacity_w','wind_scale':'wind_capacity_w','solar_scale':'solar_capacity_w','wave_scale':'wave_capacity_w','batt_scale':'battery_capacity_wh','potable_water_storage_scale':'potable_water_capacity_l'}.items():
  diffs[key]={'actual':float(v[key].value[-1]),'paper':float(r[col]),'absolute_difference':abs(v[key].value[-1]-r[col])}
 record={'site':site,'mission':mission,'seconds':time.monotonic()-t,'comparison':diffs,'summary':v['paper_summary'],'diagnostics':v['solution_diagnostics']}
 records.append(record);(out/'annual_comparison.json').write_text(json.dumps(records,indent=2)+'\n')
 model.build_timeseries_dataframe(c,v).to_csv(out/f'{site}__{mission}.csv.gz',index=False)
 print(json.dumps({'site':site,'mission':mission,'seconds':record['seconds'],'diffs':diffs}),flush=True)
 assert diffs['total_cost']['absolute_difference']<.005,record
 assert all(d['absolute_difference']<.1 for key,d in diffs.items() if key!='total_cost'),record
