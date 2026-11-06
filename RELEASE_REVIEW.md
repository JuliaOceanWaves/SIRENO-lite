# SIRENO-Lite public-release review

Started 2026-09-08. Scope: Python model, desktop GUI, configuration bridge and CLI,
browser map, packaging, examples, tests, and user documentation. The parent and
independent `paper/` checkouts were clean at the original review entry. The
2026-09-11 cleanup below starts from the retained, uncommitted review changes;
the user has now authorized committing the verified application work.
The paper's separate scientific campaign is not a release acceptance test for
this application's GEKKO solver.

## 2026-09-11 cleanup and commit audit

Scope: verify and finish the retained application changes, review real desktop
and browser behavior, public CLI/API and examples, build/installed-package paths,
and user documentation; then commit the reviewed work without pushing. Preserve
the independent `paper/` checkout and its new presentation files.

Initial Git status, complete diff and source hashes are retained under
`build/review-2026-09-11/`. Independent model, GUI/browser, and usage reviews were
integrated with packaging and full-suite verification. The detailed human
walkthrough is [docs/human-review.md](docs/human-review.md), linked from the README
and documentation navigation. Historical September 8 evidence is retained below.

### Corrections in this pass

- Outage plot shading now marks the actual selected preceding-hour dispatch
  intervals, including fractional bounds and clipping. No optimization equations,
  input data, or numerical gold values changed in this pass.
- Solution acceptance now independently reconstructs throughput, component and
  support mass, lifetime cost and cost ratios, for 48 physical/accounting checks.
  Deliberately corrupted finite results are rejected; all four objectives,
  initial inventories and unused installed storage were exercised.
- The desktop outage JSON editor and help now remain readable at 760×520.
  Browser scenario edits preserve imported coordinate precision. The map has a
  source-color legend and identifies marker sizes as qualitative.
- GUI fixture cleanup releases cyclic Tk/Matplotlib wrappers on their owning
  thread. Chrome sends its completed assertions directly to the local test
  server, independently of managed-browser shutdown. These test-harness changes
  resolved a combined-run failure and unraisable Tk cleanup warnings without
  suppressing warnings; pytest now treats such warnings as errors.
- Python examples use the shared configuration path and explicit solver cleanup.
  Documentation clarifies intervals, sign conventions, local solver choices,
  portable result bundles, fixed sizing and concrete expected review results.
- Package metadata retains the MIT license and includes both LICENSE and NOTICE
  using current SPDX metadata, with setuptools >=77 as the build requirement;
  see the [setuptools migration guide](https://setuptools.pypa.io/en/stable/userguide/license_migration.html).
  A writable PyInstaller cache override is documented after a sandbox cache
  permission failure; the maintained macOS script then built successfully.

### Fresh verification

| Gate | 2026-09-11 evidence |
| --- | --- |
| Initial retained source suite | 182 passed in 77.89 s, real Tk and Chrome required. |
| Final source suite | **193 passed in 27.09 s**, no skips or warnings, including actual Tk/Chrome; `final-tests.log`. |
| Extracted source archive | **193 passed in 40.49 s**, no skips or warnings, with real Tk/Chrome required, from a new temporary extraction; `sdist-tests.log`. |
| Installed wheel | Installed into a separate environment and launched outside the source tree; imported code path verified. Analytical 100 W generator, 3.9 kg system mass, four solved intervals, 48 diagnostics and exports pass; `installed-wheel/`. Dependencies came from that environment's base site packages (GEKKO 1.3.0, NumPy 2.4.6, pandas 3.0.3, Matplotlib 3.10.9). |
| Native macOS app | Maintained build script with PyInstaller 6.22.2; rebuilt `.app` completes an actual GUI worker/local GEKKO solve, 10 W wind sizing, zero bus residual and CSV/PDF/JSON exports; `frozen-smoke/smoke_report.json`. Runtime source remained unchanged after this build. |
| Full default mission | 400 samples / 399 hours solved in 503.75 s including loading and summary generation; 82.694417666 kg, $611423.57286, bus residual <9.1e-8 W. The process loaded before the new accounting checks: its 40 original diagnostics pass, and a separate independent CSV ledger verifies energy, material, mass and cost. Plots were regenerated with final outage shading and inspected. |
| Examples and API | All three 336-sample examples, five water candidates, fixed recommendation replay, both documented Python snippets and 125 W fixed-solar walkthrough pass. Existing documented capacities/costs match exactly. Independent ledgers use no model calculation helpers; `usage-audit.md`, `usage/independent-checks.json`. |
| Browser data and docs | Exporter reproduces retained `data.js` exactly. Strict MkDocs build passes; representative GUI/browser views and every example plot inspected. Default, installed-wheel and native PDFs pass qpdf checks. |
| Static compatibility | Python 3.9 grammar, workflow YAML, shell syntax and whitespace checks pass. These do not claim another Python/OS runtime was executed. |

Current line coverage includes all package modules: 1,623/1,834 lines (88.50%)
and 458/576 branches (79.51%); the combined pytest display rounds to 86%.
The native helper is exercised outside coverage instrumentation. Acceptance is
based on physical outcomes and interface workflows, not the coverage percentage.

The final source environment is macOS arm64, Python 3.11.5, GEKKO 1.3.2,
NumPy 2.4.2, pandas 3.0.1, Matplotlib 3.10.8 and Tk 8.6.15. Local APOPT and
BPOPT selections were exercised; requested local solver 3 uses the documented
APOPT fallback. Remote-server operation, Windows/Linux execution and the CI
Python 3.9/3.13 matrix were not run here. Maintainer signing/notarization and
publication remain separate. No push or external release was requested.

## Acceptance criteria

- Inputs have explicit units and bounds; invalid or ambiguous inputs fail before
  solving, and GUI and CLI interpret the same saved configuration identically.
- Actual local solves demonstrate power balance, storage energy/material
  conservation, hard rate/capacity limits, and analytically known results.
- GUI tests operate real Tk widgets through load, run, error, and export workflows.
  Mock tests are reserved for failures that cannot reasonably be induced live.
- Browser map labels distinguish stored solved dispatch from heuristic estimates;
  import/export, geographic selection, and numerical controls are exercised.
- Package builds contain required inputs and local solver executables. Installed
  entry points work outside the source directory. Available native GUI build is
  launched and solves; other platform evidence is clearly separated.
- Documentation states initialization/time conventions, normalization, objectives,
  fixed sizing, model limits, and reproducible commands. Documentation builds.
- Coverage reports include GUI code. A percentage is diagnostic, not a substitute
  for the behavior above. Gold outputs are changed only for explained corrections
  with independently checked expected results.

## Baseline evidence

- `.venv/bin/python -m pytest -q`: 75 passed in 19.97 s; 97.09% coverage.
  The prior coverage configuration omitted all of `sirenolite/gui.py`.
- GUI tests covered parsers and mock variables; CLI tests replaced the model with
  a mock. Neither established an actual GUI-to-solver or CLI-to-solver workflow.
- Python 3.11.5 on macOS; local GEKKO and Tk 8.6 available. Tk requires a local
  desktop connection outside the execution sandbox; an elevated probe succeeded.

## Findings and corrections

| Area | Reproduced defect | Correction and substantive evidence |
| --- | --- | --- |
| Time accounting | Constant 100 W generation reported 80 W on five samples | Average over four solved intervals; analytical 240-hour fuel/service cost = $536 including $200 capex |
| Production bounds | H2 limit silently raised to meet demand; zero water rate still allowed production | Hard caps; infeasible demand rejected, independent initial 4-unit inventory + 4-unit production serves 8 units |
| Battery | Required zero-transfer capacity vanished; simultaneous charge/discharge disposed of 40 Wh with zero load | Preserve installed mass/inventory; hard non-overlap constraint ; 100 Wh charge→81 Wh delivery at0.81 round-trip efficiency ; 99 W charge cap fails |
| Initial inventory | Fractions were upper bounds and restricted to0.5 | Exact initial fractions in [0,1], capacity/floor consistency, no invented initial supplies |
| Renewable outages | Removing selected maximum increased unmasked availability | Normalize before mask;[1,2,4,2] becomes[0.25,0.5,0,0.5] with peak outage |
| Input validation | NaN, malformed profiles, unsupported modes, fractional sample counts, ambiguous windows | Fail before solving with named fields; explicit hourly time and resource bounds |
| Solve acceptance | Solver return treated as sufficient evidence | Replay bus, material/energy, capacities, rates, averages, and boundary conditions in physical units; corrupted accepted-status results are rejected |
| GUI/CLI bridge | CLI ignored GUI sizing flags; unknown keys and cwd-relative paths silently changed cases | Shared parser/snapshot; seven fixed capacities verified in real GUI solve; CLI exports replay after moving input bundle |
| Desktop lifecycle | Worker accessed Tk and plotted; stale result remained exportable | UI-thread plotting/events, locked input snapshot, stale clearing, error recovery, deferred close until worker finishes |
| Export provenance | Re-reading source after solving could hash unsolved data; partial output files mixed runs | Retain and solve exact source bytes; source hash and versions; stage complete bundle; refuse nonempty destinations |
| Browser map | Wrong antimeridian distance, weak imports, ID collisions, transformed summary shown with baseline dispatch | Geographic/scale/import numerical tests; actual Chrome interactions; clearly identified stored 100 W/WEC1x reference plots |
| Map evidence | Frozen data pointed at moved source paths; heuristic scaling obscured storage assumptions | Preserve all 24 sites/120 designs; exact exporter check; every retained hourly bus/storage balance checked with serialization tolerances |
| Plots | t0 initialization obscured interval meaning; first solved interval omitted; right axes clipped | Explicit initial-row marker, all solved intervals plotted, physical labels, constrained layout and rendered axis-bound checks |
| Examples/docs | Capacity confused with peak inventory, stale economic results, wrong daily-rate and battery equations | Reexecuted all 3 documented examples plus 5-candidate water sweep; independent CSV ledgers; revised theory and workflow instructions |
| Packaging/CI | Local GEKKO binaries omitted; GUI excluded from coverage; release builds lacked actual solve checks | Bundle local solver; branch/line coverage includes GUI; required real Tk, Chrome, source tests and native smoke checks in CI |

## Verification completed during integration

- Initial integrated suite with actual desktop: 172 passed, one explicitly disabled
  Chrome test skipped (20.79 s). Branch-aware coverage 86% across all modules,
  including GUI and the separately executed native smoke helper. The former 97%
  number omitted the GUI and was line-only; these numbers are not comparable.
- Explicit real-Chrome gate: 1 passed, 22 deselected (41.08 s). The page's add/edit,
  import failure/atomicity, saved-point identity, discrete WEC selection, plotted
  baseline, fractional slider values, downloads, and keyboard behavior are tested.
- Actual source GUI workflows: load/save atomicity, invalid input before solve,
  physical solve/export, seven fixed sizes, edit invalidation, failure/recovery,
  scrolling and close/worker coordination. Dialog selection/error injection uses
  controlled callbacks; widgets, event loop, numerical solve, CSV and PDF are real.
- Native macOS arm64 `.app` built with PyInstaller 6.22.2 (directory mode, 158 MB).
  `dist/sirenolite_gui.app/Contents/MacOS/sirenolite_gui --smoke-test ...` passed:
  analytical 10 W wind capacity, zero W bus residual, 5 samples, Tk 8.6.15, and actual
  CSV/PDF/JSON exports. The final rebuild and packaged check also passed.
- Wheel and sdist built. Wheel installed into a separate environment; installed
  CLI launched from an unrelated directory and solved 100 W constant demand with
  generator 100 W, total mass 3.9 kg, 4 solved intervals and balanced exports.
- Three 336-sample examples (335 solved hours) reexecuted through CLI. Independent
  CSV checks reconstruct battery, H2 and water updates and electrical balance:

| Example | Generator W | Solar W | Battery Wh | Water L | Total cost $ | Total mass kg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stable electrical |1176.4595104|0|60000|0|373535.66818|669.8819209|
| Intermittent electrical |2500|0|10000|0|190463.20229|201.5|
| Potable water |0|115.53265178|0|250.70740966|403.55096245|23.048841795|

- Water sweep: all 5 candidate sets solved; selected solar-only recommendation
  replayed with every capacity fixed. Primary case rounds the draw to 151.416 L/h;
  sweep uses 151.41647136 L/h. Bounded water/generator capacities differ between
  candidates as documented. No claim of global optimality is made.
- MkDocs strict build passes. Exported PDF syntax checked using `qpdf --check`;
  rendered water and intermittent figures inspected and clipped demand axes fixed.
- Actual GUI views inspected at 1200×900 and 900×650. Screenshots are app-only;
  no unrelated desktop content was captured. Responsive help and summary order
  were refined following inspection.

Review artifacts are ignored under `build/release-review/`: logs, physical example
summaries, retained result bundles, plots, screenshots, site build and packages.
Final accepted checks and remaining platform limits are recorded below.

## Reproducible commands

```bash
SIRENOLITE_REQUIRE_GUI_TESTS=1 SIRENOLITE_BROWSER_TESTS=1 python -m pytest
# On Linux wrap the same command in xvfb-run -a.
python -m mkdocs build --strict
python -m build
python scripts/smoke_cli.py --output-dir build/installed-smoke
python docs/hybrid-map-tool/export_data.py --check
sirenolite --config docs/examples/stable-electrical-power/sirenolite_config.json --output-dir results/stable
sirenolite --config docs/examples/short-duration-intermittent-electrical-power-draws/sirenolite_config.json --output-dir results/intermittent
sirenolite --config docs/examples/potable-water-generation/sirenolite_config.json --output-dir results/water
python docs/examples/potable-water-generation/run_case.py
```

Native builds use the platform scripts in `scripts/`; their verification command
is documented in `GUI_BUILD.md`. GUI/Chrome execution here requires sandbox-approved
access to the local desktop. No auto-review action was rejected. Package downloads
first failed because the configured proxy could not resolve; direct PyPI access
succeeded in the project virtual environment. The older global PyInstaller had
incompatible hooks and was not used for the accepted native build.

## Evidence boundary

SIRENO-Lite supports qualitative architecture screening with normalized profiles
and simplified component relations. Software verification does not establish
hardware ratings, site qualification, system reliability, or experimental
validation. The September 8 review did not authorize publication, signing,
committing, or pushing. The September 11 request separately authorizes committing
the verified application changes; it does not request a push or publication.

## Final verification and remaining limits

Completed on 2026-09-08. Model, GUI, and independent accuracy reviews were
integrated; reported defects were reproduced, corrected, and rechecked.

| Gate | Final evidence |
| --- | --- |
| Full source suite | `MPLCONFIGDIR=/tmp/sirenolite-mpl SIRENOLITE_REQUIRE_GUI_TESTS=1 SIRENOLITE_BROWSER_TESTS=1 .venv/bin/python -m pytest -q -W error:constrained_layout`: **182 passed in 61.74 s**, no skips or warnings; `build/release-review/final-tests.log` |
| Distributed source | Added `MANIFEST.in` after detecting missing fixtures and browser assets. Extracted rebuilt sdist into a new `/tmp` directory and ran the same required Tk/Chrome suite with `--no-cov`: **182 passed in 64.67 s**, no skips or warnings; `final-sdist-tests.log`. CI now requires this check. |
| Installed wheel | Separate environment, installed entry point, unrelated working directory, packaged CSV: 100 W generator, 3.9 kg system mass, four solved intervals, all physical diagnostics passed; `final-installed-wheel/` |
| Native macOS app | Rebuilt from final runtime source using PyInstaller 6.22.2; actual Tk worker, local GEKKO solve, analytical 10 W wind capacity, zero bus residual, and CSV/PDF/JSON export passed; `final-frozen-smoke/smoke_report.json` |
| Default mission | All 400 samples / 399 solved hours completed in 480.10 s; mass 82.694417666 kg, cost $611423.57286; all 40 physical diagnostics passed, maximum bus residual below 9.1e-8 W; `default-diagnostics.json`, `plots/default.png` |
| Examples | Three complete 336-sample examples and all five water candidates solved. Exported energy/material ledgers independently checked; fixed-size water recommendation replayed successfully. |
| Browser evidence | Required Chrome test passed in the full suites. Frozen data regenerated exactly with `python docs/hybrid-map-tool/export_data.py --check`; all retained hourly bus/storage balances and all 120 design cost/mass ledgers checked. |
| Documentation and plots | `python -m mkdocs build --strict` passed. Actual GUI views inspected at 1200×900 and 900×650. Native and installed-wheel PDFs passed `qpdf --check`; representative example/default plots inspected. |
| Source consistency | Runtime wheel bytes match current source and `source-manifest.json`. Python 3.9 grammar and workflow YAML parse checks passed; `git diff --check` passed. These static checks do not establish another Python/OS runtime. |

Coverage includes every package module: 1,593 of 1,811 lines (87.96%) and 450 of
572 branches (78.67%); pytest's combined display rounds to 86%. The native smoke
helper is executed outside pytest instrumentation and remains shown as uncovered
there. No assertion was added merely to increase coverage. Acceptance rests on
known sizing/objective outcomes, hard infeasibility, conservation in physical
units, deliberately corrupted outputs, and real interface/solve/export workflows.
The old 97.09% result omitted the GUI and is not a comparable release metric.

The main test environment was Python 3.11.5, GEKKO 1.3.2, NumPy 2.4.2, pandas
3.0.1, Matplotlib 3.10.8, and Tk 8.6.15 on macOS arm64. The separate installed-wheel
check also passed with GEKKO 1.3.0, NumPy 2.4.6, pandas 3.0.3, and Matplotlib
3.10.9; this second check is a bounded packaging/solve check, not a second full
compatibility campaign.

Remaining publication gates:

- Execute the configured Linux/Windows source and native build/solve jobs, and
  Python 3.9/3.13 jobs. They were not run on this macOS workstation.
- The local macOS bundle has PyInstaller's ad hoc signature. Maintainer signing,
  notarization, and release publication remain separate distribution steps.
- The full default solve takes about eight minutes here; documented small
  examples are better first-run checks. Runtime depends on inputs and hardware.
- Automated widget tests and visual inspection do not establish all assistive
  technology behavior or every native file-dialog interaction.

At the end of the September 8 review, changes remained uncommitted and
unpublished. The independent paper checkout was not edited; an unrelated
untracked `paper/presentation/` appeared during that review and was preserved.
Scientific screening and validation limits above remain in force.
