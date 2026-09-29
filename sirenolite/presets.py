"""Portable, offline starting cases from the annual mission study.

Each CSV contains 8760 physical intervals. Paper-mode loading adds the initial
inventory node, preserves annual resource normalization and uses local-solar
service schedules already stored with each site's resource chronology.
"""
from copy import deepcopy
import json
from pathlib import Path
from sirenolite import model

DATA=Path(__file__).resolve().parent/'data'
DEFAULT_SITE_KEY='hawaii_pacific_51000_oneyear'
DEFAULT_MISSION_KEY='combined'
MISSIONS={'potable_water':'Desalination','scheduled_charging':'Scheduled charging',
          'pioneer_100w':'Communications: constant 100 W','combined':'Combined service'}


def list_sites():
    return deepcopy(json.loads((DATA/'paper_sites.json').read_text(encoding='utf-8'))['sites'])


def load_site_config(site_key=DEFAULT_SITE_KEY, mission_key=DEFAULT_MISSION_KEY, horizon_hours=8760):
    if mission_key not in MISSIONS:
        raise ValueError(f'Unknown paper mission: {mission_key}')
    if isinstance(horizon_hours,bool) or int(horizon_hours)!=horizon_hours or not 1<=int(horizon_hours)<=8760:
        raise ValueError('Paper preset horizon must be 1–8760 whole hours.')
    matches=[s for s in list_sites() if s['key']==site_key]
    if not matches:raise ValueError(f'Unknown prepared location: {site_key}')
    site=matches[0];hours=int(horizon_hours)
    config=model.default_inputs()
    config['objective']='total_cost'
    config['solver'].update(backend='paper',remote=False)
    water=mission_key in ('potable_water','combined')
    charging=mission_key in ('scheduled_charging','combined')
    constant=mission_key in ('pioneer_100w','combined')
    config['simulation'].update(nhrs=hours+1,peak_load=2500. if charging else 0.,
        hotel_load=100. if constant else 0.,H2DailyDemand=0.,H2ODailyDemand=151.416 if water else 0.,
        data_file=str(DATA/site['file']),profile_start_offset_hours=0,doldrum_time=0.,doldrum_windows=[])
    config['costs'].update(diesel_energy=137381/3412*1000,
                           refill_service_cost=site['delivery_trip_cost_usd'])
    config['limits'].update(generator_max_capacity=20000.,
        batt_max_charge_rate=15000. if mission_key in ('combined','pioneer_100w') else 12000.,
        h2_min_storage_g=0.,h2_max_storage_g=0.,h2_max_charge_rate_gph=0.,
        potable_water_max_storage_l=100000. if water else 0.,
        potable_water_max_rate_gpm=(.3 if mission_key=='combined' else .2) if water else 0.)
    # Delivery count is for this modeled period; its operating cost repeats
    # with the same explicit lifetime/horizon factor as dispatch costs.
    config['paper'].update(delivery_repeat_factor=config['simulation']['lifespan']/hours)
    config['site']={k:site[k] for k in ['key','label','latitude','longitude','resource_source']}
    config['site'].update(mission_key=mission_key,horizon_hours=hours,
        schedule_basis=f'Local mean solar time (UTC{site["local_solar_utc_offset_hours"]:+d}); 2024 resource year')
    model.update_derived_config(config)
    return config
