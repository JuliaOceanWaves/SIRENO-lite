#!/usr/bin/env python3
"""Verify the installed wheel, both solvers, and replay outside the source tree."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run(command, cwd, output, name):
    """Retain subprocess evidence and report useful failures."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        command, cwd=cwd, env=environment, capture_output=True,
        text=True, timeout=180,
    )
    (output / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new or empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("sirenolite", path=str(Path(sys.executable).parent))
    if executable is None:
        raise RuntimeError("Install the wheel into the Python environment running this script.")
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
        identity = json.loads(run(
            [sys.executable, "-c", "import json, sirenolite; "
             "from importlib.metadata import version; "
             "print(json.dumps(dict(path=sirenolite.__file__, "
             "version=sirenolite.__version__, installed=version('sirenolite'))))"],
            cwd, output, "installed-identity",
        ))
        source_root = Path(__file__).resolve().parents[1]
        assert Path(identity["path"]).resolve().parent != source_root / "sirenolite", identity
        assert identity["version"] == identity["installed"], identity
        run([executable, "--config", str(settings), "--output-dir", str(output / "result")],
            cwd, output, "gekko")
        # Read packaged resources from the installed wheel, not this checkout.
        paper_json = run(
            [sys.executable, "-c", "import json; from sirenolite.presets import load_site_config; "
             "print(json.dumps(load_site_config(horizon_hours=48)))"],
            cwd, output, "paper-config",
        )
        paper_settings = output / "paper-input.json"
        paper_settings.write_text(paper_json, encoding="utf-8")
        run([executable, "--config", str(paper_settings), "--output-dir", str(output / "paper")],
            cwd, output, "paper")
        run([executable, "--config", str(output / "paper" / "config.json"),
             "--output-dir", str(output / "paper-replay")], cwd, output, "paper-replay")
    summary = json.loads((output / "result" / "summary.json").read_text())
    # Constant 100 W demand with only a generator has this analytic minimum.
    assert abs(summary["generator_scale"] - 100) < 1e-5, summary
    assert abs(summary["total_mass"] - 3.9) < 1e-5, summary
    assert summary["max_power_balance_error_W"] < 1e-6, summary
    assert summary["duration_hours"] == 4, summary
    assert summary["diagnostics"]["passed"], summary
    paper = json.loads((output / "paper" / "summary.json").read_text())
    replay = json.loads((output / "paper-replay" / "summary.json").read_text())
    assert paper["backend"] == "paper" and paper["duration_hours"] == 48, paper
    assert paper["diagnostics"]["passed"] and replay["diagnostics"]["passed"]
    assert paper["resource_sha256"] == replay["resource_sha256"]
    assert abs(paper["total_cost"] - replay["total_cost"]) < 0.005
    with (output / "paper" / "timeseries.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 49
    for storage in ("battery_storage_Wh", "potable_storage_l"):
        assert abs(float(rows[0][storage]) - float(rows[-1][storage])) < 1e-5
    print(f"Installed wheel passed both solvers, portable replay, and exports: {output}")


if __name__ == "__main__":
    main()
