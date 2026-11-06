# Paper-case GUI workflow — 2026-09-15

User requests: show both GUI tabs in one new slide; demonstrate a solved paper case; add location selection and source/storage pies; make startup defaults a working paper case.

Scope: package GUI, portable local paper-mission solver, prepared geographic resources, tests/docs/packaging, actual GUI captures, and the final tracked PPTX/PDF. Preserve the user-edited deck and slide order. No commit/push requested.

Physical contract: paper mode minimizes equipment/support plus five-year operations; hourly dispatch satisfies hard load/production bounds and battery/water inventory balances with cyclic endpoints. Diesel has an optional >=500 W installation and integer 50-gallon deliveries. Full-year resource normalization precedes horizon slicing. Scheduled missions follow local mean solar time. Secondary dispatch selection fixes the primary design and deliveries and minimizes battery throughput within $1e-4 of primary cost. Legacy GEKKO configurations remain available and are not silently changed into paper studies.

Default: Hawaii combined-service annual case, selected to reproduce an existing paper result and show both battery and product-water storage. The picker offers 19 prepared study locations and custom resource CSV input for another location; arbitrary-coordinate resource downloads are not implemented.

Acceptance:
- [x] Presets/data/source manifest are portable and match paper inputs, schedules, costs and bounds.
- [x] Paper backend independent analytic/infeasible checks and full annual comparison pass.
- [x] Legacy and new GUI/CLI tests pass, including real Tk solve/export.
- [x] GUI source/storage pies use clear units and avoid double-counting battery discharge. Annual histories use labeled daily summaries with inventory ranges; hourly CSV remains available.
- [x] Default GUI solve succeeds and exports CSV/PDF/config.
- [x] Actual Inputs/Results screenshots recorded and visually inspected.
- [x] New slide 21 uses both screenshots; all 29 latest user slides preserved; complete 30-slide office render reviewed.
- [x] Final PPTX/PDF tracked/staged; code changes reviewed, no commit or push.

## Evidence to date

- All 19 packaged resource and local-solar demand profiles independently match canonical annual inputs within 5.55e-16.
- Five annual optimizations independently match paper capacities/cost: Hawaii all four missions plus Anchorage charging. Largest total-cost difference is $3.65e-8, largest capacity difference <5e-8 in reported units. Includes the nonzero diesel/discrete-delivery case.
- Nine analytical/backend checks pass, including cyclic inventory, rate/fuel infeasibility and diagnostic corruption checks.
- Final complete suite with `SIRENOLITE_REQUIRE_GUI_TESTS=1`: 208 passed, 1 optional browser test skipped; actual desktop tests included.
- Default actual GUI annual solve: Hawaii combined, $16,644.366482825655; exported CSV/PDF/config and passed all independent diagnostics. Both tabs were captured after unlock; the native macOS/Tk redraw issue required an ordinary resize, documented in GUI_BUILD.md.
- Wheel and sdist builds pass. An isolated extracted wheel includes all 19 sites and completes a 48-hour preset CLI solve/export/reload, including the compressed resource CSV.
- Strict MkDocs build passes. Native platform executable rebuilds are not part of this change; their scripts now include SciPy and all packaged profiles.

Completed: final 30-slide pair updated with both real tabs and automatic numbering; source-slide preservation, package/PDF checks and visual review passed. Final renderer: LibreOffice 26.2.6.3 (PowerPoint final export timed out). The local .venv now uses an editable install of this checkout, so `.venv/bin/sirenolite-gui` launches the updated GUI. Application code remains uncommitted; no push. Native platform executables were not rebuilt.

Human review: launch `.venv/bin/sirenolite-gui`; run the default annual Hawaii case; inspect pies/history and export CSV/PDF; switch mission/site and load the selected case; try a shorter horizon or custom resource CSV. On this macOS/Tk build, resizing the window restores any blank tab after a switch (known upstream redraw issue; solve/export unaffected).
