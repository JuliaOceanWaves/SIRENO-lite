# Stable Electrical Power

Goal: model a two-week stable-electric mission with no hydrogen or potable-water process demand.

Files:
- `docs/examples/stable-electrical-power/sirenolite_config.json`
- `sirenolite/data/load_resource_data.csv` (default packaged input profile used by this case)

Key adjustments from defaults:
- `objective = "total_cost"` to optimize the least-cost feasible steady-power system
- `simulation.nhrs = 336` hourly samples, spanning 335 solved hours
- `simulation.peak_load = 1200` W with `simulation.doldrum_time = 0`
- Uses the packaged default load/resource profile, which is already longer than 336 hours and does
  not rely on profile looping
- `simulation.H2DailyDemand = 0` and `simulation.H2ODailyDemand = 0`
- Hydrogen and potable-water subsystems are disabled by setting their storage and rate limits to
  zero
- Wind, solar, and wave are disabled in the reference case so the example focuses on stable service
  from dispatchable generation plus battery reserve
- Bounds are tightened to solve locally without oversized search space:
  - `generator_min_capacity = 300` W
  - `generator_max_capacity = 1500` W
  - `batt_min_capacity = 60000` Wh
  - `batt_max_capacity = 90000` Wh
  - `batt_min_storage_level = 15000` Wh
  - `batt_max_charge_rate = 15000` W
- Uses local APOPT (`solver.remote = false`, `solver.solver = 1`, `solver.max_iter = 100`)

Run:

```bash
python -m sirenolite --config docs/examples/stable-electrical-power/sirenolite_config.json
```

Retained local result (336 samples / 335 intervals):

| Quantity | Value |
| --- | ---: |
| Generator capacity | 1176.46 W |
| Battery capacity | 60000.00 Wh |
| Total modeled cost | $373,535.67 |
| Total modeled mass | 669.88 kg |

This nonlinear GEKKO case can converge to nearby feasible local designs across
solver versions. The 1.1 release check with GEKKO 1.3.3 returned 1181.32 W
generation, the same 60000 Wh battery, $373,536.57, and 670.07 kg, with all
physical/accounting diagnostics passing. The $0.90 cost difference is about
0.00024%; the table remains the retained reference rather than a required
global optimum or an exact cross-version regression target.

The battery is installed at its 60,000 Wh lower bound. Its specified initial fraction is 0.25 (15,000 Wh), matching the minimum storage level; its final fraction is 0.98 (58,800 Wh). The earlier default zero initial fraction was inconsistent with the positive storage floor. Capacity and final stored energy are different quantities.

This example follows the packaged electrical-load shape; it is not an exactly constant 1200 W load. Wind, solar, and wave are disabled, so the result demonstrates a generator-plus-battery reference case within those bounds, rather than a comparison across all architectures.

Reproduce and inspect the physical residuals:

```bash
sirenolite --config docs/examples/stable-electrical-power/sirenolite_config.json --output-dir results/stable
```
