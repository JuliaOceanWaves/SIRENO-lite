# Stable Electrical Power

Goal: prioritize stable electric service and remove hydrogen/water process demand.

Files:
- `docs/examples/stable-electrical-power/sirenolite_config.json`

Key adjustments from defaults:
- `simulation.H2DailyDemand = 0`
- `simulation.H2ODailyDemand = 0`
- `simulation.doldrum_time = 0` (no forced zero-resource window)
- Nonzero minimum generator/battery settings for reserve support
- `solver.remote = false` for local solve mode

Run:

```bash
sirenolite --config docs/examples/stable-electrical-power/sirenolite_config.json
```
