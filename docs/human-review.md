# Human Review Checklist

Run this checklist after installation or before relying on a new configuration.
Record the operating system, Python/app version, configuration, input CSV, and any
unexpected behavior. Keep new outputs under `results/human-review/` or another
scratch directory. The checks below complement the automated numerical and GUI
tests; they do not qualify a physical installation.

## Desktop GUI

1. Launch `sirenolite-gui`. Resize the window, scroll through **Inputs** and
   **Run / Plot**, and tab through controls. Check that labels, units, descriptions,
   buttons, and plot legends remain readable at your usual display scaling.
2. Load `docs/examples/potable-water-generation/sirenolite_config.json`. Confirm
   336 samples, a 151.416 L/h peak water draw, zero fixed electrical/H2 demand,
   local solving, and the `total_cost` objective. The configured CSV should resolve
   without changing the working directory.
3. Select **Run**. Inputs should lock while the status updates; the window should
   remain responsive. On success, expect approximately **115.53 W solar**,
   **250.71 L water capacity**, **$403.55 total modeled cost**, and **23.05 kg**.
   Other generation and battery capacities should be zero within numerical
   tolerance. These values describe this particular example and its bounds.
4. Read the plots alongside the summary. Solar-powered production fills the tank
   between draws; storage stays nonnegative. Capacity and peak inventory are
   separately identified. The first row initializes storage; 336 samples cover
   335 solved hours. Inspect plot axes and legends in the saved PDF too.
5. Save JSON, CSV, and PDF to your scratch folder using the native dialogs. Reopen
   each file; reload the JSON and confirm settings and sizing toggles survive.
   A GUI-saved JSON points to the original CSV, so retain that file or update its
   path when moving the case.
6. Set solar minimum capacity to **125 W** and uncheck its **Optimize sizing**
   control. Run again: solar capacity should remain at 125 W while dispatch and
   water storage can change. Reload the original example afterward.
7. Edit an input after a successful solve. Old results should clear and exports
   should disable. Try a malformed number or a missing CSV; the error should name
   the problem, leave the app usable, and provide no new result. Correct the input
   and confirm a successful rerun restores exports.
8. Try the [stable](examples/stable-electrical-power/index.md) and
   [intermittent](examples/short-duration-intermittent-electrical-power-draws/index.md)
   cases. Check their documented capacities and the initial/final battery reserve.
   While a run is active, request window closure; the app should defer closing
   until that solve finishes. Repeat launch/run/export in the standalone app if
   you intend to distribute that build.

## CLI and portable results

From the repository root, with the environment activated:

```bash
sirenolite --help
sirenolite --version
sirenolite --config docs/examples/potable-water-generation/sirenolite_config.json --output-dir results/human-review/water
sirenolite --config results/human-review/water/config.json --output-dir results/human-review/replay
```

Check that each completed run contains `config.json`, `resource.csv`,
`summary.json`, `timeseries.csv`, and `plots.pdf`. In `summary.json`, expect
`solver_status = 1`, `duration_hours = 335`, and `diagnostics.passed = true`.
Capacities and aggregate cost should agree with the GUI; alternate feasible
hourly schedules can occur, so identical dispatch is not required.

Copy the complete water result folder elsewhere and rerun its `config.json`
into another new folder. It should use its retained `resource.csv`, preserve the
input hash, and reproduce the design within solver tolerance. Rerunning into a
nonempty result folder should fail with an actionable message and preserve the
existing files. A misspelled setting or infeasible case should fail without
publishing a partial result bundle.

## Python API and code inspection

Run both [Python examples](example-usage.md). The complete-bundle path should agree
with the CLI, and the lower-level path should deliver **2119.824 L** over the
example's 14 modeled draws. Verify that the first inventory row is excluded from
service totals and that temporary solver files are cleaned up.

Review these interfaces when adapting the model:

| File or function | Review question |
| --- | --- |
| `sirenolite/config.py`: `load_configuration`, `prepare_config` | Are relative paths resolved correctly, unknown keys rejected, and fixed sizing flags applied to an independent run snapshot? |
| `sirenolite/model.py`: `load_resource_inputs` | Do hourly sampling, selected-window normalization, profile shifts, demand scaling, and synthetic outages represent the intended mission? |
| `build_model` | Are source capacity, battery power/energy, production-rate bounds, initial inventories, and terminal battery reserve appropriate? |
| `solve_model`, `solution_diagnostics` | Do accepted results satisfy electrical/storage balances, hard bounds, and cost/mass accounting in physical units? |
| `build_timeseries_dataframe`, `sirenolite/cli.py` | Do exported units, signs, source hashes, settings, and summary quantities describe the run actually solved? |

Read the [theory guide](theory.md) before changing equations. In particular,
positive battery power is discharge, water/H2 storage cannot supply electrical
power, water/H2 terminal refill is unconstrained, and normalized source sizes are
screening quantities. Confirm cost coefficients and objective denominators for
your intended comparison.

## Browser map

Run `mkdocs serve` and open **Hybrid Map Tool** in a current desktop browser, then
at a narrow window width. Select saved points using both mouse and keyboard;
add a map point, edit coordinates, and use **Use nearest profile**. Confirm that
the assigned resource profile is explicit and coordinate edits alone do not
silently change it.

Try the discrete WEC choices and fractional load/resource/cost settings. Read
the resulting estimates and **Profile source and limitations**. **View archived
dispatch** should remain labeled as the original 100 W, WEC 1x reference, even
after scenario settings change. These controls rescale archived designs and do
not execute a new optimization.

Save points, load them back, and open the downloaded report. Check that names,
coordinates, profile assignments, and settings survive. Attempt an invalid
import and confirm the existing points remain intact. Review keyboard focus,
scrolling, plot readability, and downloads on each browser you plan to support.
