#!/usr/bin/env python3
"""Verify the installed CLI and bundled inputs from an unrelated directory."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "objective": "total_cost",
        "simulation": {"nhrs": 5, "peak_load": 0, "hotel_load": 100,
                       "H2DailyDemand": 0, "H2ODailyDemand": 0, "doldrum_time": 0},
        "limits": {"wind_max_capacity": 0, "solar_max_capacity": 0,
                   "wave_max_capacity": 0, "batt_max_capacity": 0,
                   "h2_max_storage_g": 0, "h2_max_charge_rate_gph": 0,
                   "potable_water_max_storage_l": 0, "potable_water_max_rate_gpm": 0},
        "solver": {"remote": False, "solver": 1},
    }
    settings = output / "input.json"
    settings.write_text(json.dumps(config), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="sireno-installed-") as cwd:
        result = subprocess.run(
            ["sirenolite", "--config", str(settings), "--output-dir", str(output / "result")],
            cwd=cwd, capture_output=True, text=True, timeout=90,
        )
    (output / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    summary = json.loads((output / "result" / "summary.json").read_text())
    # Constant 100 W demand with only a generator has this analytic minimum.
    assert abs(summary["generator_scale"] - 100) < 1e-5, summary
    assert abs(summary["total_mass"] - 3.9) < 1e-5, summary
    assert summary["max_power_balance_error_W"] < 1e-6, summary
    assert summary["duration_hours"] == 4, summary
    print(f"Installed CLI passed constant-load solve and exports: {output}")


if __name__ == "__main__":
    main()
