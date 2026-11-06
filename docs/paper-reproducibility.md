# Reproducing the annual paper designs

The package contains the normalized annual inputs for 19 locations, four mission
definitions, and the retained 76-case reference summary from the independent
Julia JuMP/HiGHS paper implementation. A fresh application checkout can reproduce
these annual capacity and cost comparisons offline, without Julia or access to
the separate manuscript checkout.

Install the application from the repository root, then choose a new output path:

```bash
python -m pip install .
python scripts/verify_gui_paper_cases.py --hourly --output-dir results/paper-check
```

The default compares Hawaii's four missions and Anchorage scheduled charging.
These include electrical and water service, battery and water storage, and
integer diesel deliveries. For the complete 19-location, four-mission campaign:

```bash
python scripts/verify_gui_paper_cases.py --all --output-dir results/paper-all
```

The full campaign takes longer than the unit suite. Each case solves 8760 hourly
intervals plus the initial inventory node. `--hourly` additionally writes each
complete dispatch to a compressed CSV. Existing output directories are refused
so a new run cannot replace retained evidence.

## What the comparison establishes

`annual_comparison.json` records the configuration, Python and solver versions,
reference provenance, elapsed time, diagnostics, and individual comparisons.
The command stops with an error if a case differs from the retained reference
or fails independent power, inventory, capacity, production, fuel-delivery, or
cost-accounting checks. Cost must agree within $0.005; capacities within 0.1 of
their reported units (W, Wh, or L). Integer installation and delivery decisions
must match exactly. These tolerances accommodate numerical solver differences;
they do not permit a different physical interpretation or an updated gold result.

Reference files are `sirenolite/data/paper_annual_reference.csv` and its JSON
provenance record. The original reference bytes, source revision, model/input
hashes, and resource-manifest hash are retained. Verification reads these files
without rewriting them. `paper_sites.json` records per-location source URLs,
data evidence classes, local-solar offsets, and resource hashes. Packaged annual
resource bytes are checked in the test suite.

The retained resource window is 365 days, 2024-01-01 through 2024-12-30 UTC
(8760 intervals); it does not include every day of that leap calendar year.
Resources retain UTC chronology. Scheduled charging and water withdrawal use
local mean solar time, rounded to the nearest hour. Resource normalization uses
the full annual input before any shorter horizon is selected. Battery and water
inventories are cyclic; operating and integer-delivery costs repeat over the
configured lifespan. See the [theory guide](theory.md) for the model boundary.
The replay establishes agreement with the retained numerical model; it is not
experimental validation or site/device qualification.

## Reusing a design in the GUI or CLI

The desktop's prepared location and mission selectors use these same bundled
inputs. Save its configuration and run it with:

```bash
sirenolite --config saved-case.json --output-dir results/saved-case
```

The resulting bundle contains the actual input bytes, configuration, source
hash, numerical summary, hourly dispatch, and plot. Move the whole output folder
to preserve relative input paths. Replaying its `config.json` into a new output
directory uses the retained input, rather than a resource file that may have
changed since the original solve.

## Separate manuscript studies

The author-owned `paper/` directory is a separate repository. This release
preserves it and does not publish its manuscript, presentation, or private
history. Its Julia environment, two-week REopt comparison and timing study,
diesel-cost sensitivity, and figure builders are separate workflows. Some
figure builders depend on locally archived inputs; the annual package replay
does not claim to recreate those figures or benchmark timings.

`scripts/build_gui_paper_presets.py` is the maintenance bridge from that paper
checkout to bundled inputs. Rebuilding scientific inputs requires the canonical
paper source and an explained numerical review. Ordinary installation,
GUI/CLI operation, tests, and the annual comparison do not require that bridge.
The browser map is an older archived constant-load study; its stored designs
and heuristic controls are distinct from the cyclic annual mission study.
See the [map documentation](hybrid-map-tool/index.md).
