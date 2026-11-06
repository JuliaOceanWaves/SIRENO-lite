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
- `simulation.H2ODailyDemand = 151.416` L/h peak demand. With this one-hour draw once per day, it represents approximately 40 US gallons per draw
- `simulation.nhrs = 336` hourly samples, spanning 335 solved hours
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
- Enforces the potable-case intent (no H2 demand/storage, a 40 US gallon daily draw, 336 samples spanning 335 solved hours)
- Sweeps candidate technology sets: wind/solar/wave hybrid, solar-only, wind-only, wave-only, and generator-only
- Writes:
  - `docs/examples/potable-water-generation/run_summary.csv`
  - `docs/examples/potable-water-generation/sirenolite_config.recommended.json`

Current local primary-case result (336 samples / 335 intervals):

| Quantity | Value |
| --- | ---: |
| Solar capacity | 115.53 W |
| Potable-water storage capacity | 250.71 L |
| Total modeled cost | $403.55 |
| Total modeled mass | 23.05 kg |

Wind, wave, generator, and battery sizes are zero. Solar production fills the tank between the prescribed draws. The tank reaches approximately 250.71 L; its capacity is a material inventory, not electrical storage. Initial water is zero and final water is free, so this is a finite mission with no imposed terminal refill.

The five-candidate sweep also found solar plus water storage as its lowest-cost successful candidate. The sweep uses the exact conversion of 40 US gallons (151.41647136 L), while the primary JSON rounds the peak draw to 151.416 L/h; this gives a small numerical difference. The wave-only candidate allows 1000 L of storage and the generator-only candidate uses a 500 W generation cap and at most 300 L storage. Treat the sweep as bounded local comparisons with those stated choices, not a proof of global optimality.

`run_summary.csv` records the regenerated sweep and `sirenolite_config.recommended.json` freezes its selected capacities. Both can be regenerated with `run_case.py`. Keep the CSV beside its JSON when moving the case.

For exploratory runs, keep the reference outputs intact by choosing separate
paths. The recommendation retains the input CSV's resolved path; move that CSV
too and update `simulation.data_file` before using the recommendation elsewhere.
The CLI result bundle below instead includes a portable copy of its input.

```bash
python docs/examples/potable-water-generation/run_case.py \
  --summary-csv results/water-sweep/summary.csv \
  --recommended-config results/water-sweep/recommended.json
```

`--quick` uses 168 samples (167 solved hours); `--nhrs N` uses `N` samples
(`N - 1` solved hours). Compare candidate outcomes only at the same duration.

```bash
sirenolite --config docs/examples/potable-water-generation/sirenolite_config.json --output-dir results/water
```
