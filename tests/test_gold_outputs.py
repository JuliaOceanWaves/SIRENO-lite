import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from sirenolite import model


GOLD_DIR = Path(__file__).resolve().parent / "gold"


def _project_config(config):
    keys_by_section = json.loads((GOLD_DIR / "default_config_core.json").read_text())
    projected = {}
    for section, keys in keys_by_section.items():
        if isinstance(keys, dict):
            projected[section] = {key: config[section][key] for key in keys}
        else:
            projected[section] = config[section]
    return projected


def _assert_nested_close(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), f"{path} keys differ"
        for key in expected:
            _assert_nested_close(actual[key], expected[key], f"{path}.{key}")
        return

    if isinstance(expected, Real) and not isinstance(expected, bool):
        assert math.isclose(
            float(actual),
            float(expected),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ), f"{path}: {actual!r} != {expected!r}"
        return

    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_default_config_matches_gold_core_values():
    config = model.default_inputs()
    model.update_derived_config(config)
    expected = json.loads((GOLD_DIR / "default_config_core.json").read_text())

    _assert_nested_close(_project_config(config), expected)


def test_timeseries_dataframe_matches_gold_core_values(model_vars_factory):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    actual = model.build_timeseries_dataframe(config, model_vars_factory(4))
    expected = pd.read_csv(GOLD_DIR / "timeseries_dataframe_core.csv")

    actual = actual.reindex(columns=expected.columns)
    pdt.assert_frame_equal(actual, expected, check_dtype=False, atol=1.0e-12, rtol=1.0e-12)


def test_load_resource_inputs_matches_gold_profile_transform():
    config = model.default_inputs()
    config["simulation"].update(
        {
            "data_file": str(GOLD_DIR / "resource_profile_input.csv"),
            "nhrs": 6,
            "peak_load": 100.0,
            "H2DailyDemand": 12.0,
            "H2ODailyDemand": 24.0,
            "profile_start_offset_hours": 1,
            "doldrum_time": 0,
            "doldrum_windows": [],
        }
    )

    with pytest.warns(UserWarning, match="looping data"):
        actual = pd.DataFrame(model.load_resource_inputs(config))
    actual.insert(0, "hour", np.arange(len(actual)))
    expected = pd.read_csv(GOLD_DIR / "resource_inputs_core.csv")

    pdt.assert_frame_equal(actual, expected, check_dtype=False, atol=1.0e-12, rtol=1.0e-12)


def test_wind_only_local_solve_matches_gold_dispatch():
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    config["simulation"]["H2DailyDemand"] = 0.0
    config["simulation"]["H2ODailyDemand"] = 0.0
    config["solver"]["remote"] = False
    config["solver"]["solver"] = 1
    config["limits"].update(
        {
            "generator_min_capacity": 0.0,
            "generator_max_capacity": 0.0,
            "wind_min_capacity": 10.0,
            "wind_max_capacity": 10.0,
            "solar_min_capacity": 0.0,
            "solar_max_capacity": 0.0,
            "wave_min_capacity": 0.0,
            "wave_max_capacity": 0.0,
            "batt_min_capacity": 0.0,
            "batt_max_capacity": 0.0,
            "batt_max_charge_rate": 0.0,
            "h2_min_storage_g": 0.0,
            "h2_max_storage_g": 0.0,
            "h2_max_charge_rate_gph": 0.0,
            "potable_water_min_storage_l": 0.0,
            "potable_water_max_storage_l": 0.0,
            "potable_water_max_rate_gpm": 0.0,
        }
    )
    model.update_derived_config(config)
    inputs = {
        "Wind_unitfactor": np.array([1.0, 0.5, 0.0, 1.0]),
        "Solar_unitfactor": np.zeros(4),
        "Wave_unitfactor": np.zeros(4),
        "Load": np.array([10.0, 5.0, 0.0, 10.0]),
        "Hydrogen_demand_g": np.zeros(4),
        "PotableWater_demand_l": np.zeros(4),
    }

    model_tuple = model.build_model(config, inputs)
    model.solve_model(config, model_tuple)
    actual = model.build_timeseries_dataframe(config, model_tuple[1])
    expected = pd.read_csv(GOLD_DIR / "wind_only_solved_timeseries.csv")

    actual = actual.reindex(columns=expected.columns)
    pdt.assert_frame_equal(actual, expected, check_dtype=False, atol=1.0e-8, rtol=1.0e-8)
