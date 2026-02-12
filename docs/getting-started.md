# Getting Started

This project is a Python-based optimizer built on GEKKO. You can run it as a script or via the GUI.

Prerequisites

- Python 3.9+
- Packages: `gekko`, `numpy`, `pandas`, `matplotlib`

Install
```bash
python3 -m pip install .
```

Editable install (development)
```bash
python3 -m pip install -e .
```

Run the GUI
```bash
sirenolite-gui
```

Run CLI with a custom configuration (no GUI)

Python script example
```python
from sirenolite import model

config = model.default_inputs()
config["simulation"]["data_file"] = "sirenolite/data/load_resource_data.csv"
config["simulation"]["nhrs"] = 720
config["solver"]["remote"] = False
config["objective"] = "cost_per_watt"
model.update_derived_config(config)

inputs = model.load_resource_inputs(config)
m, vars_ = model.build_model(config, inputs)
model.solve_model(config, (m, vars_))
model.report_results(config, vars_)
```

JSON config example

Generate a template:
```bash
sirenolite --dump-config config.json
```

Edit `config.json` and run:
```bash
sirenolite --config config.json
```

If you are working from the repo, a full template is also stored at `sirenolite/data/sirenolite_config.json`.

The JSON is merged onto defaults, so you can provide only the values you want to change.
The generated template will include the current default `simulation.data_file` path; update it if you want to point at a different CSV.

Derived fields
- The model recalculates derived values such as `generator_vom_cost_perWh` at runtime.

Resource data file
The model reads a CSV file defined by `simulation.data_file`. The default is:

`sirenolite/data/load_resource_data.csv`

When installed as a package, the default points to the packaged sample file; set `simulation.data_file` to your own CSV path when you want custom data.

Columns used by the model:
- `Time`, `Load`, `Solar`, `Wind`, `Wave`, `Hydrogen`, `PotableWater`

Notes:

- `Time` must be in 1-hour increments (0, 1, 2, ...). The model assumes hourly steps.
- If the CSV is shorter than the optimization horizon, the data is looped to fill the horizon (a warning is emitted).
- `Wave` is the wave resource input (generation potential including efficiency losses baked in).
- `PotableWater` is the potable water demand profile.

Remote vs local solve
- The GEKKO solver can run remotely (`solver.remote = True`) or locally.
- Remote solves use the GEKKO cloud service and require internet access.

Outputs
- PDF plots are saved in the project root when enabled in the GUI or `report_results`.
- CSV time-series output is available from `build_timeseries_dataframe` and the GUI.
