# SIRENO-lite (System-Integrated Resilient ENergy Optimizer)

[![Tests](https://github.com/JuliaOceanWaves/SIRENO-Lite/actions/workflows/tests.yml/badge.svg)](https://github.com/JuliaOceanWaves/SIRENO-Lite/actions/workflows/tests.yml)
[![Coverage](https://raw.githubusercontent.com/JuliaOceanWaves/SIRENO-Lite/master/badges/coverage.svg)](https://github.com/JuliaOceanWaves/SIRENO-Lite/actions/workflows/tests.yml)
[![Docs](https://github.com/JuliaOceanWaves/SIRENO-Lite/actions/workflows/docs.yml/badge.svg)](https://github.com/JuliaOceanWaves/SIRENO-Lite/actions/workflows/docs.yml)
[![GUI Builds](https://github.com/JuliaOceanWaves/SIRENO-Lite/actions/workflows/build_gui.yml/badge.svg)](https://github.com/JuliaOceanWaves/SIRENO-Lite/actions/workflows/build_gui.yml)
[![Download](https://img.shields.io/github/v/release/JuliaOceanWaves/SIRENO-Lite?label=download)](https://github.com/JuliaOceanWaves/SIRENO-Lite/releases/latest)

<img src="docs/SIRENO-lite_Graphic.png" alt="SIRENO-lite graphic" width="240">

SIRENO-lite is a GEKKO-based dynamic optimization model for sizing and dispatching a hybrid system with wind, solar, wave, generator, battery, hydrogen production/storage, and potable water production/storage.

NOTE: Heavy AI assistance was used for developing the GUI, the docs, tests, reorganization, logo, and some additions to the base hybrid system optimization script.  Please use with caution and note the NOTICE and LICENSE file limitations on guarantees etc.

Documentation: [User guide](https://juliaoceanwaves.github.io/SIRENO-Lite/), [model equations](docs/theory.md), [human review checklist](docs/human-review.md), and [release review](RELEASE_REVIEW.md).

SIRENO-lite uses normalized resource profiles for qualitative architecture screening. Results guide further device and site analysis; they do not establish hardware ratings or experimental validation.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
sirenolite-gui
```

Python 3.9+ and Tk are required for the desktop GUI. On Windows, activate with `.venv\Scripts\Activate.ps1`. Local solving is the default.

In **Inputs**, load an example JSON, review the mission and capacity bounds, then select **Run**. Unchecking a sizing control fixes that capacity to its configured minimum. Save the successful run's CSV and plots from the GUI.

For a command-line run without plot windows:

```bash
sirenolite --dump-config config.json
sirenolite --config config.json --output-dir results/my-case
```

Both interfaces read the same JSON settings and sizing flags. Relative CSV paths are relative to the JSON directory. See [Getting Started](docs/getting-started.md) for a first run and [Examples](docs/examples/index.md) for mission configurations.

The historical `nhrs` setting counts hourly samples: 25 samples span 24 solved hours. Row zero initializes inventory. `H2DailyDemand` and `H2ODailyDemand` are peak hourly demand rates, not automatic daily totals; check integrated service demand before interpreting results.

## Development and verification

```bash
python -m pip install -e ".[test,docs]"
python -m pytest
mkdocs build --strict
```

Node.js is required for browser-map core tests. Real Tk tests need a desktop session; Linux CI uses `xvfb-run -a python -m pytest`. Set `SIRENOLITE_REQUIRE_GUI_TESTS=1` to fail when real GUI execution is unavailable. Set `SIRENOLITE_BROWSER_TESTS=1` to require the real Chrome interaction test; `SIRENOLITE_CHROME` can specify the executable. Coverage includes GUI code; analytic physical results, rejected infeasible cases, and actual interface workflows are the acceptance criteria. A coverage percentage alone is insufficient.

See [GUI_BUILD.md](GUI_BUILD.md) for standalone packaging and native solve checks. [Release downloads](https://github.com/JuliaOceanWaves/SIRENO-Lite/releases) are built per operating system; this review does not publish a new release.
