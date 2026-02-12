# Example Usage

This page shows common workflows with the script and the GUI.

Script example: change objective and run
```python
from sirenolite.model import (
    default_inputs,
    update_derived_config,
    load_resource_inputs,
    build_model,
    solve_model,
    report_results,
)

config = default_inputs()
config["objective"] = "cost_per_watt"
update_derived_config(config)

inputs = load_resource_inputs(config)
model = build_model(config, inputs)
solve_model(config, model)
report_results(config, model[1])
```

Script example: export a time-series CSV
```python
from sirenolite.model import (
    default_inputs,
    load_resource_inputs,
    build_model,
    solve_model,
    build_timeseries_dataframe,
)

config = default_inputs()
inputs = load_resource_inputs(config)
model = build_model(config, inputs)
solve_model(config, model)
df = build_timeseries_dataframe(config, model[1])
df.to_csv("solution_timeseries.csv", index=False)
```

GUI example
1. Run `sirenolite-gui`.
2. Set inputs on the Inputs tab.
3. Choose objective and solver settings.
4. Click Run to solve and display plots.
5. Click Save PDF to export plots.
6. Click Save CSV to export time-series outputs.
