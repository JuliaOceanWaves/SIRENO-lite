# Getting Started

SIRENO-lite sizes generation and storage to meet electrical, hydrogen, and potable-water demand. Start with an example, check the input assumptions, then compare feasible designs. The normalized profiles and simplified component relations support qualitative architecture screening; device ratings and site suitability require further analysis.

## Install and launch

Use Python 3.9 or newer with Tk support. From the source directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
sirenolite-gui
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Developers can install `python -m pip install -e ".[test,docs]"`; JavaScript tests also require Node.js.

## First desktop run

1. In **Inputs**, choose a JSON configuration and select **Load**. The [examples](examples/index.md) expose smaller, specific missions.
2. Set the CSV, number of hourly samples, load scales, objective, and storage boundary conditions. Field descriptions show units.
3. For each technology, leave **Optimize sizing** checked to search between its minimum and maximum. Uncheck it to fix capacity at the minimum; a zero minimum disables that component's capacity. Dispatch is still optimized within the fixed size.
4. Select **Run**. Inputs are locked during the solve. The log reports progress and errors. If the window is closed while solving, it remains open until the current solve finishes so the solver is not left running without the app.
5. On **Run / Plot**, read **Design Summary** for installed capacities and peak stored quantities, then scroll to the plots to inspect supply, demand, and storage changes.
6. Save the configuration, CSV, and PDF. Changing inputs clears the previous result and disables export until a new successful run.

**Save** writes settings to the chosen JSON path; use a writable project folder. A failed load leaves the current settings intact. Loading a different case resets sizing toggles that the file does not specify.

The full default case includes all technologies and synthetic outages. In the local release review it took about eight minutes, while the three smaller documented examples took about 5–20 seconds each. Timing depends on the machine and inputs. Begin with an example when learning the interface; elapsed solve time remains visible in the status line.

## Command-line run and export

```bash
sirenolite --dump-config config.json
sirenolite --config config.json --output-dir results/my-case
```

The output directory receives `config.json`, `summary.json`, `resource.csv`, `timeseries.csv`, and `plots.pdf`. This command does not open a plotting window. Without `--output-dir`, the CLI uses the interactive reporting workflow.

Settings are merged onto defaults. Both interfaces accept a plain configuration or the older `{ "config": {...}, "gui": {...} }` envelope. The `gui.optimize` flags are honored by **both** interfaces. Unknown keys are errors, so misspelled settings cannot silently use defaults.

A relative `simulation.data_file` is resolved against the **JSON file's directory**. Keep a custom CSV beside its JSON and use its filename. Omit this setting to use the installed sample. Old files containing repository-relative paths need their paths updated. GUI-saved settings contain the resolved absolute path; update it when moving them. CLI result bundles retain `resource.csv` and reference it from their JSON, so the complete folder can be moved and replayed. `summary.json` records the source hash and software versions. Use a new output directory for each CLI run; nonempty directories are never replaced, and failed solves or exports leave no partial result bundle.

## Resource and demand CSV

Use these columns, with finite nonnegative values:

| Column | Interpretation |
| --- | --- |
| `Time` | Hourly sample times, such as 0, 1, 2, ... |
| `Load` | Electrical-demand shape, scaled to `peak_load` W; `hotel_load` W is added afterward |
| `Wind`, `Solar`, `Wave` | Electrical-generation availability shapes, each normalized to its selected-window maximum |
| `Hydrogen` | Demand shape scaled to `H2DailyDemand` g/h |
| `PotableWater` | Demand shape scaled to `H2ODailyDemand` L/h |

The historical names `H2DailyDemand` and `H2ODailyDemand` specify **peak hourly rates**, not enforced daily totals. A one-hour draw each day with scale 40 L/h demands 40 L per day; a constant profile with the same scale demands 960 L per day. Integrate the exported hourly demand to confirm the intended service.

The loader offsets all profiles together, repeats a short record with a warning, and selects `nhrs` samples. Normalization uses this selected window, so changing the window can change the effective resource scale. Synthetic outages are applied after normalization. Raw resource values are shapes, not absolute watts or site-qualified capacity factors.

Non-hourly times must be resampled before loading; the app does not resample them. Omitting `Time` explicitly warns that hourly spacing is assumed. Accepted legacy column aliases are listed in the [API reference](api.md).

## Time and storage interpretation

`nhrs` is the legacy name for the number of hourly **samples**. For example, 25 samples span 24 solved one-hour intervals. The first CSV row has `is_initial_state = True`: it sets storage at time zero and is not a solved dispatch interval. Evaluate demand delivery and power balance on the remaining rows. To model 336 full hours, supply 337 samples.

Battery initial and final fractions specify exact fractions of installed capacity. Hydrogen and water initial fractions also specify exact inventory; their final inventory is free within bounds. Initial supplies therefore can be consumed during a finite mission. The model does not impose periodic hydrogen or water replenishment. See the [theory guide](theory.md) before interpreting long-term supply or repeatable operation.

## Local and remote solving

Local solving defaults to GEKKO's bundled APOPT executable. The historical solver selection `3` is mapped to local solver `1` when `remote` is false; explicit solver `2` uses BPOPT. Enable `solver.remote` or pass `--remote` only when you intend to send the model and input data to the GEKKO remote service. Remote solves require network access and are not part of the offline release tests. See [GEKKO's model documentation](https://gekko.readthedocs.io/en/latest/model_methods.html).
