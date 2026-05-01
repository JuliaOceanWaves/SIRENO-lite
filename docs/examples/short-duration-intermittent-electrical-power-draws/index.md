# Short Duration Intermittent Electrical Power Draws

Goal: model a two-week mission with mostly low electric demand interrupted by short high-power spikes.

Files:
- `docs/examples/short-duration-intermittent-electrical-power-draws/sirenolite_config.json`
- `docs/examples/short-duration-intermittent-electrical-power-draws/intermittent_load_profile.csv`

Key adjustments from defaults:
- Uses an explicit 336-hour CSV load profile with repeated short spikes across the full
  `simulation.nhrs = 336` hr optimization horizon
- `objective = "total_cost"` to find the cheapest feasible reserve configuration
- `simulation.peak_load = 2500` W to represent the short high-draw events
- `simulation.H2DailyDemand = 0` and `simulation.H2ODailyDemand = 0`
- Hydrogen and potable-water subsystems are fully disabled by setting their storage and rate
  limits to zero
- Wind, solar, and wave are disabled in the reference case so the example isolates the sizing
  and dispatch trade between dispatchable generation and battery buffering
- Battery bounds are tightened for the spiky load shape:
  - `batt_min_capacity = 10000` Wh
  - `batt_max_capacity = 20000` Wh
  - `batt_max_charge_rate = 12000` W
- Generator sizing is limited to `0-3000` W and solved locally with APOPT
  (`solver.remote = false`, `solver.solver = 1`)

Run:

```bash
python -m sirenolite --config docs/examples/short-duration-intermittent-electrical-power-draws/sirenolite_config.json
```

Verified local result from the current example inputs:
- Generator capacity: `2500.00` W
- Battery capacity: `9800.00` Wh
- Wind / solar / wave / H2 / potable-water storage: `0`
- Total cost: `$909,188.24`

Under this repeated spike profile, the cheapest feasible two-week design is a generator-plus-battery
reserve system. The battery stays at its minimum allowed size, which indicates the load shape is
primarily paying for dispatchable peak support rather than long-duration storage.
