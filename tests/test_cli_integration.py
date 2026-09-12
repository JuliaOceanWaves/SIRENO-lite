"""Real process / JSON / solver / exported-result tests, without mocking GEKKO."""

import json
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["MPLBACKEND"] = "Agg"
    return subprocess.run([sys.executable, "-m", "sirenolite", *args], cwd=cwd,
                          env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("optimize, expected_size", [(True, 10.0), (False, 20.0)])
def test_gui_json_to_cli_solves_and_exports_physical_result(tmp_path, optimize, expected_size):
    case = tmp_path / "case"
    case.mkdir()
    (case / "profile.csv").write_text(
        "Time,Load,Solar,Wind,Wave,Hydrogen,PotableWater\n"
        "0,1,0,1,0,0,0\n1,1,0,1,0,0,0\n2,1,0,1,0,0,0\n3,1,0,1,0,0,0\n"
    )
    payload = {
        "simulation": {"data_file": "profile.csv", "nhrs": 4, "peak_load": 10,
                       "H2DailyDemand": 0, "H2ODailyDemand": 0, "doldrum_time": 0},
        "limits": {"wind_min_capacity": 0 if optimize else 20,
                   "wind_max_capacity": 100, "solar_max_capacity": 0,
                   "wave_max_capacity": 0, "generator_max_capacity": 0,
                   "batt_max_capacity": 0, "h2_max_storage_g": 0,
                   "h2_max_charge_rate_gph": 0, "potable_water_max_storage_l": 0,
                   "potable_water_max_rate_gpm": 0},
        "solver": {"remote": False, "solver": 1, "linear": True},
        "gui": {"optimize": {"wind_scale": optimize}},
    }
    (case / "settings.json").write_text(json.dumps(payload))
    completed = run_cli(["--config", str(case / "settings.json"), "--output-dir", "results"], tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = tmp_path / "results"
    summary = json.loads((output / "summary.json").read_text())
    frame = pd.read_csv(output / "timeseries.csv")
    saved = json.loads((output / "config.json").read_text())
    assert summary["wind_scale"] == pytest.approx(expected_size, abs=1e-6)
    assert summary["total_mass"] == pytest.approx(expected_size * 0.03 * 1.3)
    assert summary["duration_hours"] == 3
    assert summary["solver_status"] == 1
    assert summary["max_power_balance_error_W"] < 1e-7
    np.testing.assert_allclose(frame["wind_power_W"].iloc[1:], 10, atol=1e-7)
    np.testing.assert_allclose(frame["power_balance_W"].iloc[1:], 0, atol=1e-7)
    np.testing.assert_allclose(frame["wind_cur"].iloc[1:], 10 / expected_size, atol=1e-7)
    # Three supplied intervals with no storage require exactly 30 Wh.
    assert frame["wind_power_W"].iloc[1:].sum() == pytest.approx(30)
    assert saved["simulation"]["data_file"] == "resource.csv"
    assert (output / "resource.csv").read_bytes() == (case / "profile.csv").read_bytes()
    assert summary["resource_sha256"] == hashlib.sha256((output / "resource.csv").read_bytes()).hexdigest()
    if not optimize:
        assert saved["limits"]["wind_max_capacity"] == 20
        moved = tmp_path / "moved-run"
        shutil.copytree(output, moved)
        (case / "profile.csv").unlink()
        replay = run_cli(["--config", str(moved / "config.json"), "--output-dir", "replay"], tmp_path)
        assert replay.returncode == 0, replay.stderr
        repeated = json.loads((tmp_path / "replay/summary.json").read_text())
        assert repeated["wind_scale"] == pytest.approx(20)
        assert repeated["resource_sha256"] == summary["resource_sha256"]
    assert (output / "plots.pdf").read_bytes().startswith(b"%PDF-")


def test_cli_will_not_replace_a_previous_result(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    (output / "summary.json").write_text('{"previous": true}')
    result = run_cli(["--output-dir", "results"], tmp_path)
    assert result.returncode == 2
    assert "not empty" in result.stderr
    assert (output / "summary.json").read_text() == '{"previous": true}'
    assert len(list(output.iterdir())) == 1


def test_export_failure_never_publishes_a_partial_result(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from sirenolite import cli, model
    config = model.default_inputs()
    config["simulation"].update(nhrs=3, peak_load=0, hotel_load=10, H2DailyDemand=0,
                                H2ODailyDemand=0, doldrum_time=0)
    config["limits"].update(wind_max_capacity=0, solar_max_capacity=0, wave_max_capacity=0,
                            batt_max_capacity=0, h2_max_storage_g=0, h2_max_charge_rate_gph=0,
                            potable_water_max_storage_l=0, potable_water_max_rate_gpm=0)
    def disk_failure(*args, **kwargs):
        raise OSError("Simulated PDF write failure")
    monkeypatch.setattr(Figure, "savefig", disk_failure)
    with pytest.raises(OSError, match="PDF write failure"):
        cli.run_to_directory(config, tmp_path / "result")
    assert list(tmp_path.iterdir()) == []


def test_resource_hash_describes_retained_solver_input_when_original_changes(tmp_path, monkeypatch):
    from sirenolite import cli, model
    config = model.default_inputs()
    config["simulation"].update(nhrs=3, peak_load=0, hotel_load=10, H2DailyDemand=0,
                                H2ODailyDemand=0, doldrum_time=0)
    config["limits"].update(wind_max_capacity=0, solar_max_capacity=0, wave_max_capacity=0,
                            batt_max_capacity=0, h2_max_storage_g=0, h2_max_charge_rate_gph=0,
                            potable_water_max_storage_l=0, potable_water_max_rate_gpm=0)
    original = tmp_path / "original.csv"
    data = Path(config["simulation"]["data_file"]).read_bytes()
    original.write_bytes(data)
    config["simulation"]["data_file"] = str(original)
    solve = model.solve_model
    def change_original_during_solve(*args):
        original.write_text("changed after loading")
        return solve(*args)
    monkeypatch.setattr(model, "solve_model", change_original_during_solve)
    summary = cli.run_to_directory(config, tmp_path / "result")
    assert (tmp_path / "result/resource.csv").read_bytes() == data
    assert summary["resource_sha256"] == hashlib.sha256(data).hexdigest()
    assert summary["generator_scale"] == pytest.approx(10)


def test_cli_invalid_file_reports_actionable_error_without_outputs(tmp_path):
    (tmp_path / "bad.json").write_text('{"simulation": {"peak_laod": 10}}')
    result = run_cli(["--config", "bad.json", "--output-dir", "results"], tmp_path)
    assert result.returncode == 2
    assert "simulation.peak_laod" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "results").exists()


def test_cli_infeasible_run_fails_without_exporting_a_solution(tmp_path):
    payload = {"simulation": {"nhrs": 3, "peak_load": 10, "H2DailyDemand": 0,
                               "H2ODailyDemand": 0, "doldrum_time": 0},
               "limits": {"wind_max_capacity": 0, "solar_max_capacity": 0,
                          "wave_max_capacity": 0, "generator_max_capacity": 0,
                          "batt_max_capacity": 0},
               "solver": {"remote": False}}
    (tmp_path / "impossible.json").write_text(json.dumps(payload))
    result = run_cli(["--config", "impossible.json", "--output-dir", "results"], tmp_path)
    assert result.returncode != 0
    assert "failed" in result.stderr.lower() or "infeasible" in result.stderr.lower()
    assert not (tmp_path / "results").exists()
