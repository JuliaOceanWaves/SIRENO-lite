import types
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sirenolite import model


def test_default_inputs_and_update_derived_config():
    config = model.default_inputs()
    assert "costs" in config
    assert config["costs"]["generator_vom_cost_perWh"] is not None
    assert config["costs"]["floating_platform_cost_per_kg"] > 0
    assert config["mass"]["floating_platform_mass_per_supported_mass"] > 0
    assert np.isclose(config["mass"]["batt_Kg_per_Wh"], 0.008)
    assert config["limits"]["batt_max_charge_rate"] > 0
    assert config["limits"]["potable_water_max_rate_gpm"] > 0
    assert config["simulation"]["hotel_load"] == 0.0
    assert config["efficiency"]["potable_water_Wh_per_l"] == 4.5
    for key in ("wind_vom_cost_perWh", "solar_vom_cost_perWh", "wave_vom_cost_perWh"):
        assert np.isclose(config["costs"][key], 1.0e-5)

    expected_generator_vom = (
        config["costs"]["diesel_cost"] / config["costs"]["diesel_energy"] / config["efficiency"]["gen_eff"]
        + config["costs"]["refill_service_cost"]
        / (
            config["costs"]["tank_size"]
            * config["costs"]["diesel_energy"]
            * config["efficiency"]["gen_eff"]
        )
    )
    assert np.isclose(config["costs"]["generator_vom_cost_perWh"], expected_generator_vom)

    config["costs"]["generator_vom_cost_perWh"] = None
    config["limits"]["h2_min_storage_g"] = 123.0
    config["limits"]["h2_max_storage_g"] = 456.0
    config["limits"]["potable_water_min_storage_l"] = 7.0
    config["limits"]["potable_water_max_storage_l"] = 89.0
    model.update_derived_config(config)
    assert config["costs"]["generator_vom_cost_perWh"] is not None
    assert config["limits"]["h2_min_storage_g"] == 123.0
    assert config["limits"]["h2_max_storage_g"] == 456.0
    assert config["limits"]["potable_water_min_storage_l"] == 7.0
    assert config["limits"]["potable_water_max_storage_l"] == 89.0


def test_migrate_legacy_limit_keys_converts_legacy_units():
    config = {
        "limits": {"batt_max_ramp_up": 321.0},
        "efficiency": {"potable_water_Wh_per_l": 0.0045},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.migrate_legacy_limit_keys(config, fallback_efficiency=model.default_inputs()["efficiency"])

    assert config["limits"]["batt_max_charge_rate"] == 321.0
    assert "batt_max_ramp_up" not in config["limits"]
    assert np.isclose(config["efficiency"]["potable_water_Wh_per_l"], 4.5)
    assert any("converting it to Wh/L" in str(item.message) for item in caught)


def test_build_doldrum_mask():
    mask = model.build_doldrum_mask(10, center_frac=0.5, half_window=2)
    assert mask.sum() == 6
    assert np.all(mask[3:7] == 0)


def test_build_doldrum_mask_clips_edges():
    mask = model.build_doldrum_mask(10, center_frac=0.1, half_window=3)
    assert np.all(mask[:4] == 0)
    assert np.all(mask[4:] == 1)


def test_build_doldrum_window_mask_supports_explicit_resource_windows():
    windows = [
        {"start_hour": 2, "duration_hours": 3, "resources": ["wind"]},
        {"start_day": 0.25, "duration_days": 0.125, "resources": ["solar", "wave"]},
    ]

    wind = model.build_doldrum_window_mask(10, windows, "wind")
    solar = model.build_doldrum_window_mask(10, windows, "solar")
    wave = model.build_doldrum_window_mask(10, windows, "wave")

    assert np.all(wind[:2] == 1)
    assert np.all(wind[2:5] == 0)
    assert np.all(wind[5:] == 1)
    assert np.all(solar[6:9] == 0)
    assert np.all(wave[6:9] == 0)
    assert np.all(solar[:6] == 1)
    assert np.all(wave[:6] == 1)


def test_doldrum_window_bounds_hours_supports_day_and_hour_inputs():
    assert model.doldrum_window_bounds_hours({"start_day": 1.0, "duration_days": 0.5}) == (
        24.0,
        36.0,
    )
    assert model.doldrum_window_bounds_hours({"start_hour": 3, "stop_hour": 7}) == (
        3.0,
        7.0,
    )


def _write_resource_csv(path, use_wave=True, nrows=4, time_step=1.0):
    data = {
        "Time": [i * time_step for i in range(nrows)],
        "Load": [10 + (i % 4) for i in range(nrows)],
        "Solar": [4 + (i % 3) for i in range(nrows)],
        "Wind": [3 + (i % 2) for i in range(nrows)],
        "Hydrogen": [1 + (i % 2) for i in range(nrows)],
        "PotableWater": [5 + (i % 2) for i in range(nrows)],
    }
    if use_wave:
        data["Wave"] = [2] * nrows
    else:
        data["Water"] = [2] * nrows
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def test_load_resource_inputs_wave_required(tmp_path):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    config["simulation"]["doldrums"] = {"wind": 0.5, "solar": 0.5, "wave": 0.5}

    wave_path = tmp_path / "wave.csv"
    _write_resource_csv(wave_path, use_wave=True)
    config["simulation"]["data_file"] = str(wave_path)
    inputs = model.load_resource_inputs(config)
    assert "Wave_unitfactor" in inputs
    assert np.all(np.isfinite(inputs["Wave_unitfactor"]))

    water_path = tmp_path / "water.csv"
    _write_resource_csv(water_path, use_wave=False)
    config["simulation"]["data_file"] = str(water_path)
    try:
        model.load_resource_inputs(config)
        assert False, "Expected missing Wave column to raise ValueError."
    except ValueError as exc:
        assert "Wave" in str(exc)
        assert "required columns" in str(exc)


def test_load_resource_inputs_applies_explicit_doldrum_windows(tmp_path):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 6
    config["simulation"]["doldrum_time"] = 0
    config["simulation"]["doldrums"] = {"wind": 0.5, "solar": 0.5, "wave": 0.5}
    config["simulation"]["doldrum_windows"] = [
        {"start_hour": 1, "stop_hour": 3, "resources": ["wind"]},
        {"start_hour": 4, "duration_hours": 1, "resources": ["all"]},
    ]

    csv_path = tmp_path / "resources.csv"
    _write_resource_csv(csv_path, nrows=6)
    config["simulation"]["data_file"] = str(csv_path)
    inputs = model.load_resource_inputs(config)

    assert np.all(inputs["Wind_unitfactor"][1:3] == 0.0)
    assert inputs["Solar_unitfactor"][1] > 0.0
    assert inputs["Wave_unitfactor"][1] > 0.0
    assert inputs["Wind_unitfactor"][4] == 0.0
    assert inputs["Solar_unitfactor"][4] == 0.0
    assert inputs["Wave_unitfactor"][4] == 0.0


def test_load_resource_inputs_loops_and_warns(tmp_path):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 6
    config["simulation"]["doldrums"] = {"wind": 0.5, "solar": 0.5, "wave": 0.5}

    csv_path = tmp_path / "short.csv"
    _write_resource_csv(csv_path, use_wave=True, nrows=4, time_step=0.5)
    config["simulation"]["data_file"] = str(csv_path)

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inputs = model.load_resource_inputs(config)

    assert len(inputs["Wave_unitfactor"]) == 6
    messages = [str(w.message) for w in caught]
    assert any("looping data" in msg for msg in messages)
    assert any("1-hour increments" in msg for msg in messages)


def test_load_resource_inputs_applies_profile_offset(tmp_path):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    config["simulation"]["profile_start_offset_hours"] = 1
    config["simulation"]["doldrum_time"] = 0
    config["simulation"]["peak_load"] = 8.0
    config["simulation"]["H2DailyDemand"] = 4.0
    config["simulation"]["H2ODailyDemand"] = 12.0

    csv_path = tmp_path / "offset.csv"
    pd.DataFrame(
        {
            "Time": [0, 1, 2, 3],
            "Load": [1, 2, 3, 4],
            "Solar": [0, 1, 0, 0],
            "Wind": [1, 0, 0, 0],
            "Wave": [0, 0, 2, 0],
            "Hydrogen": [1, 0, 0, 0],
            "PotableWater": [0, 0, 0, 3],
        }
    ).to_csv(csv_path, index=False)
    config["simulation"]["data_file"] = str(csv_path)

    inputs = model.load_resource_inputs(config)

    assert np.allclose(inputs["Load"], [4.0, 6.0, 8.0, 2.0])
    assert np.allclose(inputs["Hydrogen_demand_g"], [0.0, 0.0, 0.0, 4.0])
    assert np.allclose(inputs["PotableWater_demand_l"], [0.0, 0.0, 12.0, 0.0])


def test_load_resource_inputs_rejects_empty_csv(tmp_path):
    config = model.default_inputs()
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["Time", "Load", "Solar", "Wind", "Wave", "Hydrogen", "PotableWater"]).to_csv(
        csv_path,
        index=False,
    )
    config["simulation"]["data_file"] = str(csv_path)
    try:
        model.load_resource_inputs(config)
        assert False, "Expected empty CSV to raise ValueError."
    except ValueError as exc:
        assert "empty" in str(exc).lower()


def test_scale_for_plot_and_format_label():
    series = np.array([1.0, 2.0, 3.0])
    scaled, scale = model.scale_for_plot(series, 30.0, enable_scaling=True)
    assert np.isclose(scale, 10.0)
    assert np.allclose(scaled, series * 10.0)

    scaled, scale = model.scale_for_plot(series, 30.0, enable_scaling=True, order_of_magnitude=True)
    assert scale in (10.0, 100.0)

    scaled, scale = model.scale_for_plot(series, 30.0, enable_scaling=False)
    assert scale == 1.0

    scaled, scale = model.scale_for_plot(np.zeros(3), 0.0, enable_scaling=True)
    assert scale == 1.0

    assert model.format_label("Load", "W", 1.0) == "Load (W)"
    assert "1e" in model.format_label("Wave", "W", 1000.0)
    assert "x" in model.format_label("Wave", "W", 2.5)


def test_build_timeseries_dataframe(model_vars_factory):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 6
    vars_ = model_vars_factory(6)
    df = model.build_timeseries_dataframe(config, vars_)
    assert len(df) == 6
    assert "wave_power_W" in df.columns
    assert "potable_storage_Wh_equiv" in df.columns


def test_build_timeseries_dataframe_prefers_direct_dispatch_power(model_vars_factory):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    vars_ = model_vars_factory(4)
    fake_var = vars_["load"].__class__

    vars_["generator_scale"] = fake_var([0.0, 4.0, 4.0, 4.0])
    vars_["generator_unitfactor"] = fake_var([99.0, 99.0, 99.0, 99.0])
    vars_["generator_power"] = fake_var([0.0, 2.0, 4.0, 1.0])

    vars_["wind_unitfactor"] = fake_var([0.0, 0.5, 1.0, 0.25])
    vars_["wind_scale"] = fake_var([10.0, 10.0, 10.0, 10.0])
    vars_["wind_cur"] = fake_var([99.0, 99.0, 99.0, 99.0])
    vars_["wind_power"] = fake_var([0.0, 2.5, 5.0, 0.5])

    vars_["solar_unitfactor"] = fake_var([1.0, 0.5, 0.25, 0.0])
    vars_["solar_scale"] = fake_var([8.0, 8.0, 8.0, 8.0])
    vars_["solar_cur"] = fake_var([99.0, 99.0, 99.0, 99.0])
    vars_["solar_power"] = fake_var([4.0, 4.0, 1.0, 0.0])

    vars_["wave_unitfactor"] = fake_var([0.25, 0.5, 1.0, 1.0])
    vars_["wave_scale"] = fake_var([12.0, 12.0, 12.0, 12.0])
    vars_["wave_cur"] = fake_var([99.0, 99.0, 99.0, 99.0])
    vars_["wave_power"] = fake_var([3.0, 3.0, 6.0, 12.0])

    df = model.build_timeseries_dataframe(config, vars_)

    assert np.allclose(df["generator_unitfactor"], [0.0, 0.5, 1.0, 0.25])
    assert np.allclose(df["wind_cur"], [0.0, 0.5, 0.5, 0.2])
    assert np.allclose(df["solar_cur"], [0.5, 1.0, 0.5, 0.0])
    assert np.allclose(df["wave_cur"], [1.0, 0.5, 0.5, 1.0])
    expected_total = (
        df["battery_power_W"]
        + df["generator_power_W"]
        + df["wind_power_W"]
        + df["solar_power_W"]
        + df["wave_power_W"]
    )
    assert np.allclose(df["total_power_W"], expected_total)
    assert np.allclose(df["power_balance_W"], expected_total - df["electric_demand_W"])


def test_build_model_objectives():
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    config["solver"]["remote"] = False
    model.update_derived_config(config)
    inputs = {
        "Wind_unitfactor": np.ones(4),
        "Solar_unitfactor": np.ones(4),
        "Wave_unitfactor": np.ones(4),
        "Load": np.ones(4) * 10.0,
        "Hydrogen_demand_g": np.ones(4) * 2.0,
        "PotableWater_demand_l": np.ones(4) * 3.0,
    }
    m, vars_ = model.build_model(config, inputs)
    assert vars_["model"] is m
    assert "wave_scale" in vars_
    assert "supported_mass" in vars_
    assert "floating_platform_mass" in vars_

    config["objective"] = "cost_per_watt"
    m, vars_ = model.build_model(config, inputs)
    assert vars_["model"] is m
    assert "avg_batt_throughput_W" in vars_

    config["objective"] = "total_cost"
    m, vars_ = model.build_model(config, inputs)
    assert vars_["model"] is m

    config["objective"] = "cost_per_mass"
    m, vars_ = model.build_model(config, inputs)
    assert vars_["model"] is m


def test_build_model_uses_single_terminal_node_and_bounded_battery_initialization():
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 6
    config["solver"]["remote"] = False
    config["limits"]["batt_min_capacity"] = 10000.0
    config["limits"]["batt_max_capacity"] = 20000.0
    config["limits"]["batt_max_charge_rate"] = 12000.0
    model.update_derived_config(config)

    inputs = {
        "Wind_unitfactor": np.ones(6),
        "Solar_unitfactor": np.ones(6),
        "Wave_unitfactor": np.ones(6),
        "Load": np.ones(6) * 10.0,
        "Hydrogen_demand_g": np.zeros(6),
        "PotableWater_demand_l": np.zeros(6),
    }
    _, vars_ = model.build_model(config, inputs)

    initial_vector = np.array(list(vars_["initial_vector"].value), dtype=float)
    final_vector = np.array(list(vars_["final_vector"].value), dtype=float)
    assert np.count_nonzero(initial_vector) == 1
    assert initial_vector[0] == 1.0
    assert initial_vector[1:].sum() == 0.0
    assert np.count_nonzero(final_vector) == 1
    assert final_vector[-1] == 1.0
    assert final_vector[:-1].sum() == 0.0
    batt_scale_value = vars_["batt_scale"].value
    if hasattr(batt_scale_value, "value"):
        batt_scale_value = batt_scale_value.value
    assert np.isclose(batt_scale_value, 15000.0)


def test_potable_water_storage_mass_does_not_drive_platform_support():
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    config["simulation"]["H2ODailyDemand"] = 0.0
    config["solver"]["remote"] = False
    config["solver"]["solver"] = 1
    config["objective"] = "total_mass"
    config["limits"].update(
        {
            "generator_min_capacity": 0.0,
            "generator_max_capacity": 0.0,
            "wind_min_capacity": 0.0,
            "wind_max_capacity": 0.0,
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
            "potable_water_min_storage_l": 100.0,
            "potable_water_max_storage_l": 100.0,
            "potable_water_max_rate_gpm": 0.0,
        }
    )
    model.update_derived_config(config)
    inputs = {
        "Wind_unitfactor": np.zeros(4),
        "Solar_unitfactor": np.zeros(4),
        "Wave_unitfactor": np.zeros(4),
        "Load": np.zeros(4),
        "Hydrogen_demand_g": np.zeros(4),
        "PotableWater_demand_l": np.zeros(4),
    }

    model_tuple = model.build_model(config, inputs)
    model.solve_model(config, model_tuple)
    _, vars_ = model_tuple

    assert np.isclose(vars_["potable_water_storage_mass"].value[-1], 5.0)
    assert np.isclose(vars_["supported_mass"].value[-1], 0.0)
    assert np.isclose(vars_["floating_platform_mass"].value[-1], 0.0)
    assert np.isclose(vars_["total_mass"].value[-1], 5.0)


def test_battery_initial_charge_fraction_is_enforced():
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    config["solver"]["remote"] = False
    config["solver"]["solver"] = 1
    config["efficiency"]["batt_initial_charge"] = 0.45
    config["efficiency"]["batt_final_charge"] = 0.45
    config["limits"].update(
        {
            "generator_min_capacity": 0.0,
            "generator_max_capacity": 0.0,
            "wind_min_capacity": 0.0,
            "wind_max_capacity": 0.0,
            "solar_min_capacity": 0.0,
            "solar_max_capacity": 0.0,
            "wave_min_capacity": 0.0,
            "wave_max_capacity": 0.0,
            "batt_min_capacity": 100.0,
            "batt_max_capacity": 100.0,
            "batt_max_charge_rate": 10.0,
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
        "Wind_unitfactor": np.zeros(4),
        "Solar_unitfactor": np.zeros(4),
        "Wave_unitfactor": np.zeros(4),
        "Load": np.zeros(4),
        "Hydrogen_demand_g": np.zeros(4),
        "PotableWater_demand_l": np.zeros(4),
    }

    model_tuple = model.build_model(config, inputs)
    model.solve_model(config, model_tuple)
    timeseries = model.build_timeseries_dataframe(config, model_tuple[1])

    assert np.isclose(timeseries["battery_storage_Wh"].iloc[0], 45.0, atol=0.2)
    assert np.isclose(timeseries["battery_storage_Wh"].iloc[-1], 45.0, atol=0.2)


def test_potable_water_initial_charge_fraction_is_enforced():
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    config["solver"]["remote"] = False
    config["solver"]["solver"] = 1
    config["efficiency"]["potable_water_initial_charge"] = 0.4
    config["limits"].update(
        {
            "generator_min_capacity": 0.0,
            "generator_max_capacity": 0.0,
            "wind_min_capacity": 0.0,
            "wind_max_capacity": 0.0,
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
            "potable_water_min_storage_l": 100.0,
            "potable_water_max_storage_l": 100.0,
            "potable_water_max_rate_gpm": 0.0,
        }
    )
    model.update_derived_config(config)
    inputs = {
        "Wind_unitfactor": np.zeros(4),
        "Solar_unitfactor": np.zeros(4),
        "Wave_unitfactor": np.zeros(4),
        "Load": np.zeros(4),
        "Hydrogen_demand_g": np.zeros(4),
        "PotableWater_demand_l": np.array([0.0, 20.0, 20.0, 0.0]),
    }

    model_tuple = model.build_model(config, inputs)
    model.solve_model(config, model_tuple)
    timeseries = model.build_timeseries_dataframe(config, model_tuple[1])

    assert np.isclose(timeseries["potable_storage_l"].iloc[0], 40.0, atol=0.2)
    assert np.isclose(timeseries["potable_storage_l"].iloc[-1], 0.0, atol=0.2)


def test_build_model_disables_fixed_controls():
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    config["solver"]["remote"] = False
    config["limits"]["batt_min_capacity"] = 0.0
    config["limits"]["batt_max_capacity"] = 0.0
    config["limits"]["batt_max_charge_rate"] = 0.0
    config["limits"]["h2_min_storage_g"] = 0.0
    config["limits"]["h2_max_storage_g"] = 0.0
    config["limits"]["h2_max_charge_rate_gph"] = 0.0
    config["simulation"]["H2DailyDemand"] = 0.0
    config["limits"]["solar_min_capacity"] = 0.0
    config["limits"]["solar_max_capacity"] = 0.0
    config["limits"]["wave_min_capacity"] = 0.0
    config["limits"]["wave_max_capacity"] = 0.0
    config["limits"]["generator_min_capacity"] = 0.0
    config["limits"]["generator_max_capacity"] = 0.0
    model.update_derived_config(config)

    inputs = {
        "Wind_unitfactor": np.ones(4),
        "Solar_unitfactor": np.ones(4),
        "Wave_unitfactor": np.ones(4),
        "Load": np.ones(4) * 10.0,
        "Hydrogen_demand_g": np.zeros(4),
        "PotableWater_demand_l": np.ones(4) * 3.0,
    }
    _, vars_ = model.build_model(config, inputs)

    assert getattr(vars_["batt_power_inout"], "STATUS", 0) == 0
    assert getattr(vars_["h2_power_in"], "STATUS", 0) == 0
    assert getattr(vars_["batt_scale"], "STATUS", 0) == 0
    assert getattr(vars_["h2_storage_scale"], "STATUS", 0) == 0
    assert getattr(vars_["solar_scale"], "STATUS", 0) == 0
    assert getattr(vars_["wave_scale"], "STATUS", 0) == 0
    assert getattr(vars_["generator_scale"], "STATUS", 0) == 0
    assert getattr(vars_["solar_cur"], "STATUS", 0) == 0
    assert getattr(vars_["wave_cur"], "STATUS", 0) == 0
    assert getattr(vars_["generator_unitfactor"], "STATUS", 0) == 0
    assert vars_["wind_cur"].value is not None
    assert vars_["wind_scale"].STATUS == 1


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    (
        ("simulation", "nhrs", 1, "nhrs"),
        ("simulation", "profile_start_offset_hours", -1, "profile_start_offset_hours"),
        ("simulation", "peak_load", -1.0, "peak_load"),
        ("simulation", "hotel_load", -1.0, "hotel_load"),
        ("efficiency", "gen_eff", 0.0, "gen_eff"),
        ("efficiency", "batt_round_trip_eff", 1.5, "batt_round_trip_eff"),
        ("efficiency", "batt_initial_charge", 1.5, "batt_initial_charge"),
        ("efficiency", "batt_final_charge", 1.5, "batt_final_charge"),
        ("efficiency", "h2_initial_charge", 1.5, "h2_initial_charge"),
        ("efficiency", "h2_Wh_per_g", 0.0, "h2_Wh_per_g"),
        ("efficiency", "potable_water_initial_charge", 1.5, "potable_water_initial_charge"),
        ("efficiency", "potable_water_Wh_per_l", 0.0, "potable_water_Wh_per_l"),
        ("costs", "tank_size", 0.0, "tank_size"),
        ("costs", "diesel_energy", 0.0, "diesel_energy"),
        ("costs", "diesel_cost", -1.0, "diesel_cost"),
        ("costs", "refill_service_cost", -1.0, "refill_service_cost"),
        ("limits", "wind_min_capacity", -1.0, "wind_min_capacity"),
        ("limits", "batt_max_charge_rate", -1.0, "batt_max_charge_rate"),
    ),
)
def test_update_derived_config_validates_inputs(section, key, value, message):
    config = model.default_inputs()
    config[section][key] = value
    try:
        model.update_derived_config(config)
        assert False, f"Expected invalid {section}.{key} to raise ValueError."
    except ValueError as exc:
        assert message in str(exc)


def test_update_derived_config_rejects_inverted_limits_and_storage_floor():
    config = model.default_inputs()
    config["limits"]["solar_min_capacity"] = 2.0
    config["limits"]["solar_max_capacity"] = 1.0
    with pytest.raises(ValueError, match="solar_max_capacity"):
        model.update_derived_config(config)

    config = model.default_inputs()
    config["limits"]["batt_min_storage_level"] = config["limits"]["batt_max_capacity"] + 1.0
    with pytest.raises(ValueError, match="batt_min_storage_level"):
        model.update_derived_config(config)

    config = model.default_inputs()
    config["efficiency"]["batt_final_charge"] = 0.1
    config["limits"]["batt_max_capacity"] = 100.0
    config["limits"]["batt_min_storage_level"] = 20.0
    with pytest.raises(ValueError, match="batt_final_charge"):
        model.update_derived_config(config)


def test_update_derived_config_rejects_unknown_objective():
    config = model.default_inputs()
    config["objective"] = "least_unobtanium"
    with pytest.raises(ValueError, match="Unsupported objective"):
        model.update_derived_config(config)


def test_solve_model_sets_options():
    class DummyModel:
        def __init__(self):
            self.options = types.SimpleNamespace()
            self.solved = False

        def solve(self, **kwargs):
            self.solved = True

    dummy = DummyModel()
    config = model.default_inputs()
    config["solver"]["imode"] = 7
    config["solver"]["max_iter"] = 99
    config["solver"]["solver"] = 5
    config["solver"]["cv_type"] = 2
    model.solve_model(config, (dummy, None))
    assert dummy.options.IMODE == 7
    assert dummy.options.MAX_ITER == 99
    assert dummy.options.SOLVER == 5
    assert dummy.options.CV_TYPE == 2
    assert dummy.options.LINEAR == 0
    assert dummy.solved is True

    dummy = DummyModel()
    config = model.default_inputs()
    config["solver"]["remote"] = False
    config["solver"]["solver"] = 3
    config["solver"]["linear"] = True
    model.solve_model(config, (dummy, None))
    assert dummy.options.SOLVER == 1
    assert dummy.options.LINEAR == 1


def test_report_results_and_create_stacked_figure(monkeypatch, model_vars_factory):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 6
    vars_ = model_vars_factory(6)

    saved = []

    def fake_savefig(*args, **kwargs):
        saved.append(args[0] if args else None)

    monkeypatch.setattr(plt, "savefig", fake_savefig)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    model.report_results(config, vars_)
    assert len(saved) == 3

    from matplotlib.figure import Figure

    fig_saved = []

    def fake_fig_save(self, *args, **kwargs):
        fig_saved.append(args[0] if args else None)

    monkeypatch.setattr(Figure, "savefig", fake_fig_save)
    fig = model.create_stacked_figure(config, vars_, save_pdf=True, figsize=(6, 4))
    assert len(fig.axes) == 4
    assert fig_saved


def test_main_invokes_subroutines(monkeypatch):
    calls = []

    def fake_default_inputs():
        return {}

    def fake_load(config):
        calls.append("load")
        return {}

    def fake_build(config, inputs):
        calls.append("build")
        return ("model", {"vars": 1})

    def fake_solve(config, model_tuple):
        calls.append("solve")

    def fake_report(config, vars_):
        calls.append("report")

    monkeypatch.setattr(model, "default_inputs", fake_default_inputs)
    monkeypatch.setattr(model, "load_resource_inputs", fake_load)
    monkeypatch.setattr(model, "build_model", fake_build)
    monkeypatch.setattr(model, "solve_model", fake_solve)
    monkeypatch.setattr(model, "report_results", fake_report)

    model.main()
    assert calls == ["load", "build", "solve", "report"]
