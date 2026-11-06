# Short Duration Intermittent Electrical Power Draws

Goal: model a two-week mission with mostly low electric demand interrupted by short high-power spikes.

Files:
- `docs/examples/short-duration-intermittent-electrical-power-draws/sirenolite_config.json`
- `docs/examples/short-duration-intermittent-electrical-power-draws/intermittent_load_profile.csv`

Key adjustments from defaults:
- Uses an explicit 336-hour CSV load profile with repeated short spikes across the full
  `simulation.nhrs = 336` sample grid (335 solved hours)
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

Current local result (336 samples / 335 intervals):

| Quantity | Value |
| --- | ---: |
| Generator capacity | 2500.00 W |
| Battery capacity | 10000.00 Wh |
| Total modeled cost | $190,463.20 |
| Total modeled mass | 201.50 kg |

The 10,000 Wh battery is at its configured minimum capacity; 9800 Wh is its required final energy, not its installed size. In this solved dispatch, the generator follows the spikes and the battery mainly charges near the mission end to satisfy that final reserve. This run does not demonstrate a reduction of generator peak capacity through battery buffering. Changing reserve boundaries or allowed technologies poses a different design question.

```bash
sirenolite --config docs/examples/short-duration-intermittent-electrical-power-draws/sirenolite_config.json --output-dir results/intermittent
```
