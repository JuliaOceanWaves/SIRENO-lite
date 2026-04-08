# Short Duration Intermittent Electrical Power Draws

Goal: model mostly low demand with short high-power spikes.

Files:
- `docs/examples/short-duration-intermittent-electrical-power-draws/sirenolite_config.json`
- `docs/examples/short-duration-intermittent-electrical-power-draws/intermittent_load_profile.csv`

Key adjustments from defaults:
- Uses a custom CSV load profile with short spikes
- `simulation.peak_load` increased to represent high draw events
- No hydrogen or potable-water process demand
- Higher battery max charge-rate and battery capacity limits

Run:

```bash
sirenolite --config docs/examples/short-duration-intermittent-electrical-power-draws/sirenolite_config.json
```
