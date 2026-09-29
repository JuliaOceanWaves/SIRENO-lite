"""Bundle the paper's 19 audited hourly resource profiles for offline GUI use."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper';REPRO=PAPER/'paper_reproducibility'
sys.path.insert(0,str(REPRO))
import run_global_mission_cases as annual
OUT=ROOT/'sirenolite/data'
metadata=pd.read_csv(annual.RESOURCE_SUMMARY).set_index('site_key')
logistics=annual.load_generator_logistics().set_index('site_key')
audit=pd.read_csv(annual.GLOBAL_AUDIT_SUMMARY).drop_duplicates('site_key').set_index('site_key')
records=[]
for site in annual.global_screen.baseline_sites():
 key=site['site_key'];row=metadata.loc[key];source=PAPER/row.output_csv
 charging,water=annual.annual_demand(annual.ANNUAL_MISSIONS[3],utc_offset_hours=annual.local_solar_utc_offset(float(row.longitude)))
 frame=pd.read_csv(source,usecols=['Wind','Solar','Wave'])
 for col in frame:frame[col]=annual.normalize_resource(frame[col],label=f'{key}:{col}')
 frame.insert(0,'Time',np.arange(8760))
 frame['Load']=(charging-100)/2500
 frame['Hydrogen']=0.
 frame['PotableWater']=water/151.416
 path=OUT/f'paper_{key}.csv.gz'
 frame.to_csv(path,index=False,float_format='%.15g',compression={'method':'gzip','mtime':0})
 records.append({'key':key,'label':site['site_label'],'latitude':float(row.latitude),'longitude':float(row.longitude),
                 'hours':8760,'start':'2024-01-01T00:00:00Z','file':path.name,
                 'resource_source':str(row.resource_source),'source_class':str(audit.loc[key,'source_class']),
                 'inclusion_status':str(audit.loc[key,'inclusion_status']),
                 'note':str(row.note),'solar_source':str(row.solar_source),'wind_source':str(row.wind_source),'wave_source':str(row.wave_source),
                 'source_urls':[str(row[k]) for k in ['nasa_power_url','ndbc_historical_url','open_meteo_marine_url'] if pd.notna(row[k])],
                 'source_path':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'delivery_trip_cost_usd':float(logistics.loc[key,'delivery_trip_cost_usd']),
                 'diesel_logistics_tier':str(logistics.loc[key,'diesel_logistics_tier']),
                 'local_solar_utc_offset_hours':annual.local_solar_utc_offset(float(row.longitude))})
(OUT/'paper_sites.json').write_text(json.dumps({'description':'Audited paper resource year; normalized per full-year source peak; daily service schedules shifted to local mean solar time; 8760 intervals, no initial inventory row.', 'wave_attribution':'Open-Meteo Marine historical wave data where indicated; see per-site source URLs and source descriptions. NOAA/NASA data where indicated.', 'sites':records},indent=2)+'\n')
print(f'Bundled {len(records)} sites, {sum((OUT/r["file"]).stat().st_size for r in records)/1e6:.2f} MB compressed')
