"""Public configuration contracts shared by GUI and CLI."""

from copy import deepcopy
import json

import pytest

from sirenolite.config import configuration_from_payload, load_configuration, prepare_config


def test_json_relative_profile_survives_another_working_directory(tmp_path, monkeypatch):
    folder = tmp_path / "case"
    folder.mkdir()
    profile = folder / "profiles.csv"
    profile.write_text("Time,Load,Solar,Wind,Wave,Hydrogen,PotableWater\n0,1,0,1,0,0,0\n1,1,0,1,0,0,0\n")
    path = folder / "config.json"
    path.write_text(json.dumps({"simulation": {"data_file": "profiles.csv", "nhrs": 2}}))
    monkeypatch.chdir(tmp_path)
    config, _ = load_configuration(path)
    from sirenolite import model
    profiles = model.load_resource_inputs(config)
    assert config["simulation"]["data_file"] == str(profile)
    assert profiles["Load"].tolist() == [1000.0, 1000.0]


def test_wrapped_config_preserves_fixed_sizing_and_input_snapshot():
    payload = {"config": {"limits": {"wind_min_capacity": 12, "wind_max_capacity": 80}},
               "gui": {"optimize": {"wind_scale": False}}}
    original = deepcopy(payload)
    config, optimize = configuration_from_payload(payload)
    run = prepare_config(config, optimize)
    assert run["limits"]["wind_min_capacity"] == run["limits"]["wind_max_capacity"] == 12
    assert config["limits"]["wind_max_capacity"] == 80
    run["simulation"]["doldrums"]["wind"] = 0.1
    assert config["simulation"]["doldrums"]["wind"] == 0.4
    assert payload == original


def test_legacy_charge_rate_takes_precedence_over_default():
    config, _ = configuration_from_payload({"limits": {"batt_max_ramp_up": 123.0}})
    assert config["limits"]["batt_max_charge_rate"] == 123.0
    assert "batt_max_ramp_up" not in config["limits"]


@pytest.mark.parametrize("payload, message", [
    (None, "JSON root"),
    ({"simulation": {"peak_laod": 123}}, "simulation.peak_laod"),
    ({"solver": []}, "solver must be a mapping"),
    ({"config": {}, "objective": "total_cost"}, "Unknown wrapped"),
    ({"gui": {"optimize": []}}, "gui.optimize"),
    ({"gui": {"optimize": {"wind": False}}}, "Unknown sizing"),
    ({"gui": {"optimize": {"wind_scale": 2}}}, "Cannot parse boolean"),
    ({"solver": {"remote": "sometimes"}}, "Cannot parse boolean"),
    ({"simulation": {"data_file": None}}, "nonempty path"),
])
def test_invalid_config_fails_before_model_build(payload, message):
    with pytest.raises(ValueError, match=message):
        configuration_from_payload(payload)


def test_false_string_never_enables_remote_solving():
    config, flags = configuration_from_payload({
        "limits": {"batt_max_capacity": 0},
        "solver": {"remote": "false", "linear": "true"},
        "gui": {"optimize": {"wind_scale": "off"}},
    })
    assert config["solver"]["remote"] is False
    assert config["solver"]["linear"] is True
    assert flags == {"wind_scale": False}


def test_linear_hint_uses_effective_disabled_battery_without_erasing_bounds():
    config, optimize = configuration_from_payload({
        "solver": {"linear": True}, "gui": {"optimize": {"batt_scale": False}},
    })
    assert config["limits"]["batt_max_capacity"] == 1440000
    assert config["solver"]["linear"] is True
    effective = prepare_config(config, optimize)
    assert effective["limits"]["batt_max_capacity"] == 0
    assert effective["solver"]["linear"] is True


@pytest.mark.parametrize("settings, message", [
    ({"limits": {"batt_min_capacity": 10}}, "complementarity"),
    ({"limits": {"batt_min_capacity": 10, "batt_max_capacity": 5}}, "greater than or equal"),
    ({"objective": "cost_per_watt"}, "ratio objective"),
])
def test_fixed_toggle_never_bypasses_input_or_solver_feasibility(settings, message):
    settings.update(solver={"linear": True}, gui={"optimize": {"batt_scale": False}})
    with pytest.raises(ValueError, match=message):
        configuration_from_payload(settings)
