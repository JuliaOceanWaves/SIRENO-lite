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

1. The desktop opens with the paper's **Hawaii combined-service annual case**: desalination, scheduled charging, and a constant 100 W load. It includes the resource history and uses a local solver; no resource download is needed.
2. On **Inputs**, choose a location, one of the four missions, and a horizon of 168, 720, or 8760 hours, then select **Load selected case**. This replaces the costs, limits, resources, and local-solar service schedule. Alternatively load a JSON configuration or browse to your own location's hourly CSV.
3. For each technology, leave **Optimize sizing** checked to search between its minimum and maximum. Uncheck it to fix capacity at the minimum; a zero minimum disables that component's capacity. Dispatch is still optimized within the fixed size.
4. Select **Run current inputs** (or **Run** on the Results tab). Inputs are locked during the solve. If the window is closed while solving, it remains open until the solve finishes so the solver is not left running without the app.
5. On **Results / Run**, the source and storage pies appear beside supply, demand, and inventory histories for the solved horizon. The caption reports total system cost and mass. Select **Show details / log** for installed capacities, peak inventories, delivery count, and solver messages.
6. Save the configuration, CSV, and PDF. Changing inputs clears the previous result and disables export until a new successful run.

**Save** writes settings to the chosen JSON path; use a writable project folder. A failed load leaves the current settings intact. Loading a different case resets sizing toggles that the file does not specify.

The source pie reports dispatched energy over the horizon, excluding curtailed energy and battery discharge (which transfers already counted energy). The storage pie reports installed battery energy plus the electricity required to produce the stored water or hydrogen; its legend retains physical liters and grams. These product equivalents do not imply that water or hydrogen can discharge electricity.

The location picker contains the study's prepared sites, including resource proxies identified by their labels. A custom CSV enables other locations; choosing coordinates alone does not create a resource history. The seven-day and thirty-day options shorten the chronology while retaining the annual resource normalization. They are convenient exploratory cases, not substitutes for annual sizing. Runtime depends on the mission and inputs; elapsed solve time remains visible.

## Command-line run and export

```bash
sirenolite --dump-config config.json
sirenolite --config config.json --output-dir results/my-case
```

The output directory receives `config.json`, `summary.json`, `resource.csv` (or `resource.csv.gz`), `timeseries.csv`, and `plots.pdf`. This command does not open a plotting window. Without `--output-dir`, the CLI uses the interactive reporting workflow.

Settings are merged onto defaults. Both interfaces accept a plain configuration or the older `{ "config": {...}, "gui": {...} }` envelope. The `gui.optimize` flags are honored by **both** interfaces. Unknown keys are errors, so misspelled settings cannot silently use defaults.

A relative `simulation.data_file` is resolved against the **JSON file's directory**. Keep a custom CSV beside its JSON and use its filename. Omit this setting to use the installed sample. Old files containing repository-relative paths need their paths updated. GUI-saved settings contain the resolved absolute path; update it when moving them. CLI result bundles retain the CSV (compressed when supplied compressed) and reference it from their JSON, so the complete folder can be moved and replayed. `summary.json` records the source hash and software versions. Use a new output directory for each CLI run; nonempty directories are never replaced, and failed solves or exports leave no partial result bundle.

## Resource and demand CSV

Use these columns, with finite nonnegative values:

| Column | Interpretation |
| --- | --- |
| `Time` | Hourly sample times, such as 0, 1, 2, ... |
| `Load` | Electrical-demand shape, scaled to `peak_load` W; `hotel_load` W is added afterward |
| `Wind`, `Solar`, `Wave` | Electrical-generation availability shapes; normalization depends on the selected formulation, as described below |
| `Hydrogen` | Demand shape scaled to `H2DailyDemand` g/h |
| `PotableWater` | Demand shape scaled to `H2ODailyDemand` L/h |

The historical names `H2DailyDemand` and `H2ODailyDemand` specify **peak hourly rates**, not enforced daily totals. A one-hour draw each day with scale 40 L/h demands 40 L per day; a constant profile with the same scale demands 960 L per day. Integrate the exported hourly demand to confirm the intended service.

For the general GEKKO formulation, the loader offsets all profiles together, repeats a short record with a warning, and selects `nhrs` samples. Normalization uses this selected window, so changing the window can change the effective resource scale. Paper presets instead contain the normalized annual record, select `nhrs - 1` dispatch intervals, and prepend an initial inventory node. Synthetic outages are applied after normalization. Raw resource values are shapes, not absolute watts or site-qualified capacity factors.

Non-hourly times must be resampled before loading; the app does not resample them. Omitting `Time` explicitly warns that hourly spacing is assumed. Accepted legacy column aliases are listed in the [API reference](api.md).

## Time and storage interpretation

`nhrs` is the legacy name for the number of hourly **samples**. For example, 25 samples span 24 solved one-hour intervals. The first exported time-series row has `is_initial_state = True`: it sets storage at time zero and is not a solved dispatch interval. Paper-mode resource CSVs contain intervals only; the loader inserts that initial row. Evaluate demand delivery and power balance on the remaining rows. To model 336 full hours, use 337 state/output samples; paper mode requires 336 resource CSV rows, while legacy GEKKO requires 337.

**Paper missions** optimize the initial battery and water inventories and impose cyclic end = start conditions. The fraction fields are disabled because they do not apply. The local HiGHS formulation minimizes equipment, floating support, and lifetime operating costs, including discrete diesel deliveries, then selects minimum battery throughput within the primary cost tolerance. It supports the four electrical/desalination missions; hydrogen is disabled. Operating and delivery costs repeat with the project-lifetime / modeled-horizon ratio, updated when the GUI horizon changes.

**General GEKKO cases** use exact battery initial and final fractions. Hydrogen and water initial fractions also specify exact inventory; their final inventory is free within bounds. Initial supplies can therefore be consumed during a finite mission. See the [theory guide](theory.md) before interpreting long-term supply or repeatable operation. Loading an existing general configuration restores these controls and the general model's objectives.

## Local and remote solving

The desktop paper presets solve locally with SciPy/HiGHS. General configurations and the legacy CLI defaults use GEKKO's bundled APOPT executable. The historical solver selection `3` is mapped to local solver `1` when `remote` is false; explicit solver `2` uses BPOPT. Enable `solver.remote` or pass `--remote` only for GEKKO when you intend to send the model and input data to its remote service. Remote solves require network access and are not part of the offline release tests. See [GEKKO's model documentation](https://gekko.readthedocs.io/en/latest/model_methods.html).
