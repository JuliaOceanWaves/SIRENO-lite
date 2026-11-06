# Changelog

## 1.1.0

This minor release adds the local paper-mission solver and prepared desktop
workflow developed since 1.0.0, and audits their numerical and release behavior.

- Offline HiGHS paper solver with cyclic battery/water inventories, integer diesel
  deliveries, four missions, and 19 prepared annual resource profiles.
- Desktop location/mission selection, generation and storage summaries, shared
  GUI/CLI configuration, portable result bundles, and explicit failed-solve recovery.
- Self-contained replay of all 76 annual paper designs against retained Julia
  reference results, with source hashes and numerical acceptance tolerances.
- Independent physical/accounting diagnostics, analytical and infeasibility
  tests, real desktop/browser workflows, and preserved numerical gold results.
- Correct paper outage shading, backend-aware diagnostics, consistent derived
  metadata, solver cleanup, and Tk object cleanup before repeat solves.
- Installed and frozen distribution checks for both solvers; gated Linux/macOS/
  Windows GUI release assets, Python distributions, and SHA-256 checksums.

The separate manuscript repository is preserved. Annual replay does not replace
the independent REopt comparison, timing study, or archived figure workflows.
See [the release audit](RELEASE_AUDIT_1.1.md) for measured verification and limits.
