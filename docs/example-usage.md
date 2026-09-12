# Example Usage

Run these examples from the repository root after installation. They use the small
water mission rather than the longer default case. Use a new output directory for
each saved run.

## Python: load a case and save a complete run

```python
from sirenolite.config import load_configuration, prepare_config
from sirenolite.cli import run_to_directory

config, optimize = load_configuration(
    "docs/examples/potable-water-generation/sirenolite_config.json"
)
config["objective"] = "total_cost"
summary = run_to_directory(prepare_config(config, optimize), "results/python-water")
print(summary["solar_scale"], summary["potable_water_storage_scale"])
assert summary["diagnostics"]["passed"]
```

`prepare_config` validates the settings, refreshes derived costs, and applies the
same sizing toggles as the GUI and CLI. The original editable settings remain
unchanged. The result bundle includes the actual input CSV and its hash, resolved
settings, physical diagnostics, time series, and a PDF.

## Python: inspect dispatch and export a time-series CSV

```python
from pathlib import Path
from sirenolite import model
from sirenolite.config import load_configuration, prepare_config

config, optimize = load_configuration(
    "docs/examples/potable-water-generation/sirenolite_config.json"
)
config = prepare_config(config, optimize)
problem = model.build_model(config, model.load_resource_inputs(config))
try:
    model.solve_model(config, problem)
    frame = model.build_timeseries_dataframe(config, problem[1])
    solved = frame.loc[~frame["is_initial_state"]]
    # Each solved row represents one hour; the first row only initializes storage.
    water_delivered_l = solved["potable_water_demand_l"].sum()
    print(f"Water delivered: {water_delivered_l:.3f} L")
    print(problem[1]["solution_diagnostics"])
    Path("results").mkdir(exist_ok=True)
    frame.to_csv("results/python-water-timeseries.csv", index=False)
finally:
    problem[0].cleanup()
```

This lower-level API leaves temporary solver-file cleanup to the caller. Use
`model.report_results(config, problem[1])` before cleanup for interactive plots, or
`model.create_stacked_figure(config, problem[1])` to obtain an editable Matplotlib
figure. Positive `battery_power_W` means discharge; negative values mean charging.

## Desktop GUI
1. Run `sirenolite-gui`.
2. (Optional) At the top of the Inputs tab, choose a JSON config path (for example `sirenolite/data/sirenolite_config.json`) and click `Load`.
3. Set or adjust inputs, objective, and solver settings.
4. (Optional) Click `Save` to write the current GUI configuration to JSON.
5. Click Run to solve and display plots.
6. Click Save PDF to export plots.
7. Click Save CSV to export time-series outputs.


## CLI: save a complete run

```bash
sirenolite --config docs/examples/potable-water-generation/sirenolite_config.json --output-dir results/water
```

This writes `config.json`, `summary.json` with physical feasibility diagnostics and
source hash, the retained `resource.csv`, `timeseries.csv`, and `plots.pdf`. The JSON
sizing flags are honored. The complete output folder can be moved and its
`config.json` rerun into a new output directory.

For programmatic service totals, exclude `is_initial_state` and sum the hourly rates over the remaining one-hour intervals. See the [time and storage conventions](getting-started.md#time-and-storage-interpretation).

Use the [human review checklist](human-review.md) to compare GUI, CLI, and Python
results and exercise error recovery before adopting a new configuration.
