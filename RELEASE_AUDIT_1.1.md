# SIRENO-Lite 1.1 release audit

Started 2026-11-06 from application commit `c86fa74` on `examples`.
Scope: governing models, public configuration and CLI, scientific tests, paper
reproduction, desktop and browser interfaces, documentation, distributions,
GitHub CI, pull request, merge to `master`, and tag `v1.1.0`.

The user authorizes corrections, publication, merge, and the minor release.
Potentially material changes to the paper's scientific results require a user
decision before the affected work proceeds. Kevin Moore's Sandia coding and
writing guidance governs this review. Three independent agents review physics,
paper reproduction, and interfaces; the coordinating agent reviews integration
and release evidence.

## Scientific boundary

The application screens normalized wind, solar, wave, and diesel generation
with electrical, hydrogen, and water service. The paper backend minimizes
equipment/support plus lifetime operations with hourly power and inventory
balances, cyclic battery/water inventories, and integer diesel deliveries.
The legacy GEKKO backend retains its finite-mission initial/terminal conditions
and objective options. These are distinct formulations, not interchangeable
claims of experimental validation. Defaults and retained results are preserved
unless a justified correction and independent replay establish otherwise.

## Initial state and acceptance

The application had no modified tracked files. Untracked `.codex/` skills and
`.venv-gui/` are local tooling. The nested `paper/` repository has author changes
and is preserved; its initial commit is
`318796d376aca6b6c41d62ab46c1c29ecb5f918b`. The application's original `origin`
points to Sandia GitLab. The verified public destination is
`https://github.com/JuliaOceanWaves/SIRENO-lite`, default branch `master`.
The audit branch includes 14 existing application commits ahead of GitHub master.

Acceptance requires meaningful physical/infeasibility tests; real GUI solve,
display, and export; browser interaction; strict documentation; wheel and source
archive checks outside the checkout; native packaged solves; preserved annual
paper results; and successful GitHub checks before merge and release tagging.

Local evidence lives in `build/release-audit-2026-11-06/`. The previous `.venv`
interpreter points to a removed installation. Verification uses the existing
`.venv-tk` Python 3.14 environment with freshly installed test/build dependencies.
Historical review files remain historical evidence, not claims of checks run in
this release.

## Findings and corrections

- Paper outage shading used the legacy initialization-row convention and could
  mark the wrong interval. It now follows the actual paper interval mask;
  explicit, fractional, clipped, and centered windows have solved regression
  checks. Dispatch equations and input bytes are unchanged.
- Shared diagnostics used GEKKO boundary conditions even for a paper result.
  They now call the paper diagnostics, which also verify source chronology,
  installation/delivery integrality, cost ratios, and the summary ledger.
  Deliberately corrupted reports are rejected.
- Direct model construction now refreshes derived fuel-cost metadata for either
  backend. Terminal model runs clean up solver files after success or failure.
- A repeat desktop run could finalize old Tk objects on the worker thread. Old
  widget cycles are collected on the Tk thread before starting the next worker.
  The Chrome harness separates asynchronous interaction from screenshot timing.
- The annual reference summary and source hashes are bundled. The verification
  script can replay five representative cases or all 76 without the private
  manuscript checkout. Preset regeneration stages new files or checks existing
  files; it cannot silently overwrite package inputs.
- Dense paper-model and preset code is formatted for review without changing
  optimization equations. Invalid fractional, Boolean, and non-finite preset
  horizons are rejected with a named input error.
- Wheel and native smoke checks exercise both the legacy solver and the paper
  default backend. The wheel check verifies its import location and version,
  checks cyclic storage, and replays the exported compressed-input bundle.
- Package version is sourced from `sirenolite.__version__`; 1.1.0 is a minor
  increment from 1.0.0. Explicit package discovery includes bundled data without
  setuptools' ambiguous-namespace warning. Local tooling and the independent
  paper checkout are excluded from application tracking.
- Release publishing depends on Linux/macOS/Windows native solves and reusable
  source/distribution/browser/annual verification. The tag must match package
  metadata. Only the release job has repository write permission. Published
  assets include GUI ZIPs, wheel, source archive, and SHA-256 checksums.
  Actions use supported releases pinned to immutable commits. Documentation
  deployment credentials are restricted to the deployment job.

## Verification

| Check | Evidence so far |
| --- | --- |
| Full source suite | `SIRENOLITE_REQUIRE_GUI_TESTS=1 SIRENOLITE_BROWSER_TESTS=1 ... python -m pytest -q`: **234 passed in 31.41 s**, no skips. Real Tk and isolated Chrome for Testing required. |
| Focused physics/API/gold | 148 passed; no gold values changed. |
| Final paper tests | 28 passed after formatting and diagnostic changes. |
| Canonical resource bridge | All 19 profiles, demand schedules, and source hashes agree; largest numerical difference 1.05e-15. |
| Five final annual cases | All pass; largest cost difference $3.65e-8 and capacity difference 4.55e-8 in reported units. Independent exported-CSV balances: power 4.55e-13 W, battery 1.82e-12 Wh, water 5.69e-14 L; lifetime-cost residual <1.5e-11 USD. |
| Full annual campaign | **76/76 pass**, 931.24 s cumulative solve time. Maximum cost difference $3.753e-7; capacity difference 4.903e-7 in reported W/Wh/L. All diesel installation/delivery decisions match. Assembly/solve/population ASTs are unchanged from the initial code; the final five-case replay separately exercises expanded diagnostics. |
| Examples | All three documented 336-sample examples solve and export with physical/accounting diagnostics. Water and intermittent figures match retained quantities. GEKKO 1.3.3 returns a nearby stable-service local solution: $0.90 (0.00024%) higher cost, 4.86 W higher generator capacity; documented without replacing retained reference. |
| Browser dataset | `python docs/hybrid-map-tool/export_data.py --check` reproduces retained `data.js` exactly. |
| Documentation/static | Strict MkDocs, workflow YAML parsing, Bash syntax, and whitespace checks pass. |
| Source GUI smoke | Both GEKKO and 168-hour paper workflows solve and export; paper power residual <2.3e-13 W, cyclic inventories pass. |
| Installed wheel | Fresh environment, import location verified outside checkout, version 1.1.0, both solvers and portable compressed-input replay pass. All packaged code/data bytes match current source. |
| Extracted source distribution | **234 passed in 34.23 s**, mandatory real Tk/Chrome, from the independently extracted archive. |
| Local native macOS application | Maintained PyInstaller build passes both actual GUI solves/exports. GEKKO: analytical 10 W wind and zero bus residual. Paper: 168 intervals, cyclic inventories, bus residual <2.3e-13 W. Exported PDFs pass qpdf checks. |
| GitHub CI | Initial hosted run passed representative annual replay and macOS native build/solve. It exposed Windows checkout newline conversion of hashed references, downloaded Chrome failure before browser tests, and missing dynamically imported `PIL._tkinter_finder` in the Linux bundle. `.gitattributes` preserves hashed data bytes, browser CI uses the runner's sandbox-configured installed Chrome, and native build scripts explicitly include the Pillow/Tk bridge. Final hosted gates remain required before merge. |

Coverage includes every application module: 2167/2435 lines (88.99%) and
564/718 branches (78.55%); the combined pytest display rounds to 87%.
The native smoke helper runs outside coverage instrumentation. Acceptance is
based on physical and interface behavior, not the coverage percentage.

## Evidence limits and retained manuscript state

No material numerical correction or change to the paper's scientific assumptions
has been made. The nested manuscript's original input/result bytes and author
changes are preserved. Its separate test suite has four pre-existing author-prose
assertion failures, plus five failures and three errors caused by missing local
Julia JuMP/HiGHS dependencies; 52 tests pass. This is not a claim that the nested
manuscript suite or REopt/timing workflow was rerun successfully. The public
annual replay compares against retained Julia results, not a newly executed
Julia campaign. All 35 retained sensitivity/resource-figure source hashes and
six representative-dispatch provenance hashes match their local retained assets;
some require the documented archive relocation.

The browser render and current exported paper figure were visually inspected.
Actual Tk workflows pass, but macOS rejected app-only window capture with
`could not create image from window`; current live tab screenshots could not be
visually inspected. Signing/notarization and exhaustive accessibility testing
are not established by these checks. No public release has been created yet.
