# API Reference

## Module: `sirenolite.model`

### `default_inputs()`
Build a configuration dictionary with default values.

**Parameters**
- None

**Returns**

Configuration dict with sections:

- **costs**
- **limits**
- **efficiency**
- **mass**
- **simulation**
- **solver**
- **objective**

### `update_derived_config(config)`
Refresh derived fields after modifying inputs.

**Parameters**
- **config** (dict): configuration dictionary returned by `default_inputs()`.

**Updates**
- **costs.generator_vom_cost_perWh**: derived diesel variable cost.

### `load_resource_inputs(config)`
Load the resource CSV and build normalized input profiles.

**Parameters**
- **config** (dict): configuration dictionary.

**Returns**

Dictionary with keys:

- **Wind_unitfactor**: normalized wind resource (0-1).
- **Solar_unitfactor**: normalized solar resource (0-1).
- **Wave_unitfactor**: normalized wave resource (0-1).
- **Load**: scaled electric load (W).
- **Hydrogen_demand_g**: scaled hydrogen demand (g/hr).
- **PotableWater_demand_l**: scaled potable water demand (L/hr).

**Notes**
- `H2DailyDemand` scales the hydrogen demand profile; it is not an enforced daily total.
- `H2ODailyDemand` scales the potable water demand profile; it is not an enforced daily total.
- `Time` must be in 1-hour increments.
- If the CSV is shorter than `nhrs`, the data is looped to fill the horizon.
- The CSV uses a `Wave` column.

### `build_model(config, inputs)`
Construct the GEKKO model and variables.

**Parameters**
- **config** (dict): configuration dictionary.
- **inputs** (dict): outputs from `load_resource_inputs()`.

**Returns**

Tuple `(m, model_vars)` where:

- **m**: GEKKO model instance.
- **model_vars**: dictionary of variables and intermediates.

Model_vars keys:

- **load**: electrical load parameter (W).
- **hydrogen_demand_g**: hydrogen demand parameter (g/hr).
- **potable_water_demand_l**: potable water demand parameter (L/hr).
- **wind_unitfactor**: wind availability (0-1).
- **solar_unitfactor**: solar availability (0-1).
- **wave_unitfactor**: wave availability (0-1).
- **batt_power_inout**: battery power dispatch (W).
- **h2_power_in**: hydrogen production power (W).
- **potable_water_power_in**: potable water production power (W).
- **wind_cur**: wind curtailment control (0-1).
- **solar_cur**: solar curtailment control (0-1).
- **wave_cur**: wave curtailment control (0-1).
- **wind_scale**: wind capacity scale (W).
- **solar_scale**: solar capacity scale (W).
- **wave_scale**: wave capacity scale (W).
- **generator_scale**: generator capacity scale (W).
- **batt_scale**: battery capacity scale (Wh).
- **h2_storage_scale**: hydrogen storage capacity scale (g).
- **potable_water_storage_scale**: potable water storage capacity scale (L).
- **generator_unitfactor**: generator dispatch (0-1).
- **final_vector**: terminal-state selector.
- **batt_storagelevel**: battery energy state (Wh).
- **h2_storagelevel**: hydrogen storage state (g).
- **potable_water_storagelevel**: potable water storage state (L).
- **avg_generator_W**: average generator power over horizon (W).
- **avg_wind_W**: average wind power over horizon (W).
- **avg_sol_W**: average solar power over horizon (W).
- **avg_wave_W**: average wave power over horizon (W).
- **supported_mass**: total supported mass excluding the platform (kg).
- **floating_platform_mass**: derived platform mass (kg).
- **total_cost**: total system cost ($).
- **cost_per_watt**: cost objective ($/W).
- **total_mass**: mass objective (kg).
- **model**: reference to the GEKKO model instance.

### `solve_model(config, model)`
Apply solver options and solve the model.

**Parameters**
- **config** (dict): configuration dictionary.
- **model** (tuple): output from `build_model()`.

### `report_results(config, model_vars)`
Print summary results and generate standalone plots.

**Parameters**
- **config** (dict): configuration dictionary.
- **model_vars** (dict): `model_vars` from `build_model()`.

### `create_stacked_figure(config, model_vars, save_pdf=False, figsize=(10, 8))`
Create a stacked matplotlib figure (balance, production, storage).

**Parameters**
- **config** (dict): configuration dictionary.
- **model_vars** (dict): `model_vars` from `build_model()`.
- **save_pdf** (bool): save a PDF when `True`.
- **figsize** (tuple): matplotlib figure size.

**Returns**

- **matplotlib.figure.Figure**: the stacked figure.

**Notes**
- When `save_pdf` is `True`, writes `solution_{nhrs}_optimal_mix_plots_with_wave.pdf`.

### `build_timeseries_dataframe(config, model_vars)`
Create a pandas DataFrame of time-series outputs.

**Parameters**
- **config** (dict): configuration dictionary.
- **model_vars** (dict): `model_vars` from `build_model()`.

**Returns**

Pandas DataFrame with columns:

- **time_hr**: time in hours.
- **time_days**: time in days.
- **load_W**: electrical load (W).
- **hydrogen_demand_g**: hydrogen demand (g/hr).
- **potable_water_demand_l**: potable water demand (L/hr).
- **electric_demand_W**: total electric demand (W).
- **total_power_W**: total supplied power (W).
- **power_balance_W**: supply minus demand (W).
- **battery_power_W**: battery dispatch (W).
- **battery_storage_Wh**: battery energy (Wh).
- **h2_power_W**: hydrogen production power (W).
- **h2_storage_g**: hydrogen storage (g).
- **h2_storage_Wh_equiv**: hydrogen storage (Wh equivalent).
- **potable_power_W**: potable water production power (W).
- **potable_storage_l**: potable water storage (L).
- **potable_storage_Wh_equiv**: potable water storage (Wh equivalent).
- **generator_unitfactor**: generator dispatch (0-1).
- **generator_scale_W**: generator size (W).
- **generator_power_W**: generator output (W).
- **wind_unitfactor**: wind availability (0-1).
- **wind_cur**: wind curtailment (0-1).
- **wind_scale_W**: wind size (W).
- **wind_power_W**: wind output (W).
- **solar_unitfactor**: solar availability (0-1).
- **solar_cur**: solar curtailment (0-1).
- **solar_scale_W**: solar size (W).
- **solar_power_W**: solar output (W).
- **wave_unitfactor**: wave availability (0-1).
- **wave_cur**: wave curtailment (0-1).
- **wave_scale_W**: wave size (W).
- **wave_power_W**: wave output (W).

## Configuration Reference

### `costs`
- **generator_fix_cost**: generator capex ($/W).
- **wind_fix_cost**: wind capex ($/W).
- **solar_fix_cost**: solar capex ($/W).
- **wave_fix_cost**: wave capex ($/W).
- **batt_fix_cost**: battery capex ($/Wh).
- **h2_storage_fix_cost_per_g**: hydrogen storage capex ($/g).
- **potable_water_storage_fix_cost_per_l**: potable water storage capex ($/L).
- **floating_platform_cost_per_kg**: floating platform cost per mass ($/kg).
- **tank_size**: diesel tank size (gal).
- **diesel_cost**: diesel cost ($/gal).
- **diesel_energy**: diesel energy (Wh/gal).
- **refill_service_cost**: diesel refill service cost ($).
- **generator_vom_cost_perWh**: generator variable cost ($/Wh), derived.
- **batt_vom_cost_perWh**: battery variable cost ($/Wh).
- **wind_vom_cost_perWh**: wind variable cost ($/Wh).
- **solar_vom_cost_perWh**: solar variable cost ($/Wh).
- **wave_vom_cost_perWh**: wave variable cost ($/Wh).

### `limits`
- **generator_max_capacity**: generator max size (W).
- **wind_max_capacity**: wind max size (W).
- **solar_max_capacity**: solar max size (W).
- **wave_max_capacity**: wave max size (W).
- **batt_max_capacity**: battery max size (Wh).
- **generator_min_capacity**: generator min size (W).
- **wind_min_capacity**: wind min size (W).
- **solar_min_capacity**: solar min size (W).
- **wave_min_capacity**: wave min size (W).
- **batt_min_capacity**: battery min size (Wh).
- **batt_min_storage_level**: minimum battery state (Wh).
- **batt_max_ramp_up**: battery ramp limit (W per hour step).
- **h2_min_storage_g**: hydrogen storage min (g).
- **h2_max_storage_g**: hydrogen storage max (g).
- **potable_water_min_storage_l**: potable water storage min (L).
- **potable_water_max_storage_l**: potable water storage max (L).

### `efficiency`
- **gen_eff**: generator efficiency (fraction).
- **batt_round_trip_eff**: battery round-trip efficiency (fraction).
- **batt_final_charge**: final battery fraction (fraction of `batt_scale`).
- **h2_Wh_per_g**: energy per gram of hydrogen (Wh/g).
- **potable_water_Wh_per_l**: energy per liter of potable water (Wh/L).

### `mass`
- **generator_Kg_per_W**: generator mass factor (kg/W).
- **batt_Kg_per_Wh**: battery mass factor (kg/Wh).
- **wind_Kg_per_W**: wind mass factor (kg/W).
- **solar_Kg_per_W**: solar mass factor (kg/W).
- **wave_Kg_per_W**: wave mass factor (kg/W).
- **h2_storage_Kg_per_g**: hydrogen storage mass factor (kg/g).
- **potable_water_storage_Kg_per_l**: potable water storage mass factor (kg/L).
- **floating_platform_mass_per_supported_mass**: platform mass per supported mass (kg/kg).

### `simulation`
- **lifespan**: project lifespan (hr).
- **nhrs**: simulation horizon length (hr).
- **peak_load**: peak electrical load for scaling (W).
- **H2DailyDemand**: hydrogen demand scale (g/day).
- **H2ODailyDemand**: potable water demand scale (L/day).
- **data_file**: path to resource/demand CSV.
- **doldrum_time**: half-window for doldrum periods (hr).

### `simulation.doldrums`
- **wind**: wind doldrum center (fraction of horizon).
- **solar**: solar doldrum center (fraction of horizon).
- **wave**: wave doldrum center (fraction of horizon).

### `solver`
- **remote**: solve on GEKKO remote server when `True`.
- **imode**: GEKKO IMODE setting.
- **max_iter**: solver iteration limit.
- **solver**: GEKKO solver selector.
- **cv_type**: GEKKO CV type.

### `objective`
- **total_mass**: minimize total system mass.
- **cost_per_watt**: minimize levelized cost per watt.
