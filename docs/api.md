# API Reference

## Module: `sirenolite.model`

### `default_inputs()`
Build the historical GEKKO configuration. The desktop startup case instead comes from `sirenolite.presets.load_site_config()`.

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
- **paper**: discrete-diesel and dispatch-selector settings
- **site**: source/location/mission metadata

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
- `H2DailyDemand` is the peak hydrogen rate (g/h); it is not an enforced daily total.
- `H2ODailyDemand` is the peak potable-water rate (L/h); it is not an enforced daily total.
- `Time` must be in 1-hour increments.
- GEKKO mode repeats a short CSV with a warning; paper mode requires `nhrs - 1` intervals, normalizes the complete file before slicing, adds the initial state row, and rejects a horizon extending beyond the data.
- The CSV uses a `Wave` column.

### `build_model(config, inputs)`
Construct the selected GEKKO or local paper-mission model and variables.

**Parameters**
- **config** (dict): configuration dictionary.
- **inputs** (dict): outputs from `load_resource_inputs()`.

**Returns**

Tuple `(m, model_vars)` where:

- **m**: GEKKO instance or compatible paper-model object with time, status and cleanup.
- **model_vars**: dictionary of variables and intermediates.

Model_vars keys:

- **load**: electrical load parameter (W).
- **hydrogen_demand_g**: hydrogen demand parameter (g/hr).
- **potable_water_demand_l**: potable water demand parameter (L/hr).
- **wind_unitfactor**: wind availability (0-1).
- **solar_unitfactor**: solar availability (0-1).
- **wave_unitfactor**: wave availability (0-1).
- **batt_power_inout**: battery power at the electrical bus (W), positive for discharge and negative for charging.
- **batt_charge_power**, **batt_discharge_power**: nonnegative bus-side charging and discharging powers (W).
- **h2_power_in**: hydrogen production power (W).
- **potable_water_power_in**: potable water production power (W).
- **wind_cur**, **solar_cur**, **wave_cur**: legacy aliases for dispatch power in the model dictionary; use the exported CSV for availability fractions. There, 1 means all available power is used.
- **wind_scale**: wind capacity scale (W).
- **solar_scale**: solar capacity scale (W).
- **wave_scale**: wave capacity scale (W).
- **generator_scale**: generator capacity scale (W).
- **batt_scale**: battery capacity scale (Wh).
- **h2_storage_scale**: hydrogen storage capacity scale (g).
- **potable_water_storage_scale**: potable water storage capacity scale (L).
- **generator_unitfactor**: legacy alias for generator dispatch power (W) in the model dictionary. The exported CSV column of this name is the used fraction of installed generator capacity.
- **final_vector**: terminal-state selector.
- **batt_storagelevel**: battery energy state (Wh).
- **h2_storagelevel**: hydrogen storage state (g).
- **potable_water_storagelevel**: potable water storage state (L).
- **avg_generator_W**: average generator power over horizon (W).
- **avg_wind_W**: average wind power over horizon (W).
- **avg_sol_W**: average solar power over horizon (W).
- **avg_wave_W**: average wave power over horizon (W).
- **avg_load_W**: mean electrical demand, including hydrogen and water production loads, over solved intervals (W).
- **avg_batt_throughput_W**: mean charging plus discharging power over solved intervals (W).
- **supported_mass**: total supported mass excluding the platform (kg).
- **floating_platform_mass**: derived platform mass (kg).
- **total_cost**: total system cost ($).
- **cost_per_watt**: cost objective ($/W).
- **cost_per_mass**: cost objective ($/kg).
- **total_mass**: mass objective (kg).
- **model**: reference to the GEKKO model instance.

### `solve_model(config, model)`
Apply solver options, solve, and check physical feasibility. Raises on a rejected solver result or a failed physical residual check. The returned model dictionary receives `solution_diagnostics` containing named residuals, units, tolerances, and pass status. `solution_diagnostics(config, model_vars)` can also recompute these checks on a solved result.

This function updates the supplied tuple in place and returns `None`. Keep the
effective configuration unchanged between construction, solving, and reporting.
Call `model[0].cleanup()` in a `finally` block after extracting results to remove
temporary solver files; `run_to_directory` handles this automatically. See the
[Python workflow](example-usage.md#python-inspect-dispatch-and-export-a-time-series-csv).

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
- **is_initial_state**: true for row zero, which initializes storage and is not a solved dispatch interval. Rows 1 onward describe the preceding hourly intervals.
- **load_W**: electrical load (W).
- **hydrogen_demand_g**: hydrogen demand (g/hr).
- **potable_water_demand_l**: potable water demand (L/hr).
- **electric_demand_W**: total electric demand (W).
- **total_power_W**: total supplied power (W).
- **power_balance_W**: supply minus demand (W).
- **battery_power_W**: battery bus dispatch (W), positive for discharge and negative for charging.
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
- **wind_cur**: fraction of available wind power used (0-1).
- **wind_scale_W**: wind size (W).
- **wind_power_W**: wind output (W).
- **solar_unitfactor**: solar availability (0-1).
- **solar_cur**: fraction of available solar power used (0-1).
- **solar_scale_W**: solar size (W).
- **solar_power_W**: solar output (W).
- **wave_unitfactor**: wave availability (0-1).
- **wave_cur**: fraction of available wave power used (0-1).
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
- **batt_max_charge_rate**: battery max charge/discharge power (W).
- **h2_min_storage_g**: hydrogen storage min (g).
- **h2_max_storage_g**: hydrogen storage max (g).
- **h2_max_charge_rate_gph**: hydrogen production max rate (g/hr).
- **potable_water_min_storage_l**: potable water storage min (L).
- **potable_water_max_storage_l**: potable water storage max (L).
- **potable_water_max_rate_gpm**: potable-water production max rate (gal/min).

### `efficiency`
- **gen_eff**: generator efficiency (fraction).
- **batt_round_trip_eff**: battery round-trip efficiency (fraction).
- **batt_initial_charge**, **h2_initial_charge**, **potable_water_initial_charge**: exact initial fraction of installed capacity, from 0 to 1. Initial inventory is supplied at the mission boundary.
- **batt_final_charge**: exact final battery fraction of `batt_scale`, from 0 to 1. Hydrogen and water have no corresponding terminal refill constraint.
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
- **nhrs**: integer number of hourly samples, at least 2; the solved duration is `nhrs - 1` hours.
- **peak_load**: peak electrical load for scaling (W).
- **hotel_load**: fixed parasitic electrical load (W).
- **H2DailyDemand**: peak hydrogen demand rate (g/h; historical name).
- **H2ODailyDemand**: peak potable water demand rate (L/h; historical name).
- **data_file**: path to resource/demand CSV.
- **profile_start_offset_hours**: circular shift applied to the resource and demand profiles before truncation (hr).
- **doldrum_time**: half-window for doldrum periods (hr). The total zero-resource duration is `2 * doldrum_time`.
- **doldrum_windows**: list of explicit resource outages, e.g. `[{"start_hour": 8, "duration_hours": 3, "resources": ["wind", "wave"]}]`. Day-based start/duration/stop fields are also accepted. Overlapping windows combine. The start/stop bounds select sample indices, rounded outward for fractional values; start is inclusive and stop is exclusive. Row zero only initializes storage. For example, rows 8–10 have zero availability in this window and describe elapsed intervals 7–10 h. Plot shading follows those preceding solved intervals.

### `simulation.doldrums`
- **wind**: wind doldrum center (fraction of horizon).
- **solar**: solar doldrum center (fraction of horizon).
- **wave**: wave doldrum center (fraction of horizon).

### `solver`
- **remote**: default `False` for local solving. `True` sends the model and inputs to GEKKO's remote service.
- **imode**: GEKKO IMODE setting.
- **max_iter**: solver iteration limit.
- **solver**: GEKKO solver selector.
- **cv_type**: GEKKO CV type.
- **linear**: default `False`; must be false with active battery dispatch, which uses a nonlinear complementarity constraint.

### `objective`
- **total_mass**: minimize total system mass.
- **total_cost**: minimize total system cost.
- **cost_per_watt**: minimize cost divided by mean electrical demand, including optimized production loads ($/W, not levelized energy cost).
- **cost_per_mass**: minimize total system cost per total system mass.

## Shared configuration: `sirenolite.config`

`load_configuration(path)` returns `(config, optimize)` from a plain or wrapped JSON file. It merges defaults, validates settings, and resolves relative CSV paths against the JSON directory. Unknown settings raise `ValueError`.

`configuration_from_payload(payload, base_dir=None)` applies the same rules to a dictionary. `prepare_config(config, optimize=None)` returns an independent run snapshot, with unchecked capacities fixed by setting their maximum equal to their minimum. Neither changes the caller's configuration.

Canonical `gui.optimize` keys are `generator_scale`, `wind_scale`, `solar_scale`, `wave_scale`, `batt_scale`, `h2_storage_scale`, and `potable_water_storage_scale`.

The loader accepts these CSV aliases: `Time_h`; `Load_W_shape` or `Load_W`; `Solar_W_shape` or `Solar_profile`; `Wind_W_shape` or `Wind_profile`; `Wave_W_shape` or `Wave_profile`; `Hydrogen_g_per_h_shape` or `Hydrogen_g_per_h`; and `PotableWater_L_per_h_shape` or `PotableWater_L_per_h`. Their normalization follows the canonical columns regardless of the alias spelling.

## Headless export: `sirenolite.cli`

`run_to_directory(config, output_dir)` solves and writes a portable run configuration, retained input CSV, numerical summary with physical diagnostics and input hash, time-series CSV, and PDF; it returns the summary dictionary and cleans up temporary GEKKO files. Nonempty output directories are rejected. A staging directory is published only after all exports succeed. The command-line equivalent is `sirenolite --config case.json --output-dir results/case`.

## Module: `sirenolite.presets`

`list_sites()` returns 19 prepared sites with labels, coordinates, resource attribution, evidence class and source hashes. `load_site_config(site_key=DEFAULT_SITE_KEY, mission_key="combined", horizon_hours=8760)` returns a complete validated paper-mode configuration. Missions are `potable_water`, `scheduled_charging`, `pioneer_100w`, and `combined`. Prepared files are installed with the package; neither GUI nor CLI requires the manuscript checkout or Julia.
