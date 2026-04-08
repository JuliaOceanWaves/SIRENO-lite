# Potable Water Generation

Goal: solve a potable-water control co-design case where hybrid generation sizing and dispatch are optimized around intermittent potable-water draws.

Files:
- `docs/examples/potable-water-generation/sirenolite_config.json`
- `docs/examples/potable-water-generation/potable_water_profile.csv`

Key adjustments from defaults:
- Uses a custom CSV potable-water profile
- `simulation.peak_load = 0` to remove baseline electric load
- `simulation.H2DailyDemand = 0` and `limits.h2_min_storage_g = limits.h2_max_storage_g = 0`
- `objective = "total_cost"` to search for the lowest-cost feasible hybrid design
- `simulation.H2ODailyDemand = 151.416` L/day (40 US gallons/day)
- `simulation.nhrs = 336` hr for a two-week optimization horizon
- `simulation.profile_start_offset_hours = 1` so the horizon begins just after a potable-water draw instead of at the draw instant
- `simulation.doldrum_time = 0` because this example is focused on matching the demand profile rather than forcing a synthetic outage
- Tightened capacity and rate bounds so the local solve stays well-conditioned:
  - `generator_max_capacity = 0` W in the primary example config
  - `wind_max_capacity = 300` W
  - `solar_max_capacity = 300` W
  - `wave_max_capacity = 200` W
  - `potable_water_max_storage_l = 400` L
  - `potable_water_max_rate_gpm = 0.2`
  - `potable_water_Wh_per_l = 4.5`
- Uses local APOPT (`solver.remote = false`, `solver.solver = 1`)

Run:

```bash
sirenolite --config docs/examples/potable-water-generation/sirenolite_config.json
```

Iterative diagnostic runner:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplcache \
  .venv/bin/python docs/examples/potable-water-generation/run_case.py
```

This script:
- Forces local solving (`solver.remote = false`) for reproducible/offline runs
- Enforces the potable-case intent (no H2 demand/storage, 40 gal/day potable demand, 2-week horizon)
- Sweeps candidate technology sets: renewable hybrid, solar-only, wind-only, wave-only, and generator-only
- Writes:
  - `docs/examples/potable-water-generation/run_summary.csv`
  - `docs/examples/potable-water-generation/sirenolite_config.recommended.json`

Verified local result from the current example inputs:
- Solar capacity: `130.69` W
- Potable-water storage: `204.14` L
- Wind / wave / generator / battery: `0`
- Total cost: `$991.92`

The full sweep shows that the cheapest feasible design under the current demand profile is effectively solar-only. The renewable-hybrid search converges to the same solar-plus-water-storage solution as the explicit `solar_only` candidate, so the recommended JSON freezes that design for direct reproduction.
