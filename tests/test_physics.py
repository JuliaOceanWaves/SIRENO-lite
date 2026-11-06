"""Independent physical expectations exercised through local GEKKO optimization.

These tests use small, prescribed systems with analytic feasible dispatch or
known infeasibility. They deliberately do not refresh numerical gold files.
"""

import numpy as np
import pandas as pd
import pytest

from sirenolite import model


def isolated_config(n=5):
    config = model.default_inputs()
    config["simulation"].update(nhrs=n, H2DailyDemand=0.0, H2ODailyDemand=0.0)
    config["solver"].update(remote=False, solver=1)
    config["objective"] = "total_cost"
    for technology in ("generator", "wind", "solar", "wave", "batt"):
        config["limits"][f"{technology}_min_capacity"] = 0.0
        config["limits"][f"{technology}_max_capacity"] = 0.0
    config["limits"].update(
        batt_max_charge_rate=0.0,
        h2_min_storage_g=0.0,
        h2_max_storage_g=0.0,
        h2_max_charge_rate_gph=0.0,
        potable_water_min_storage_l=0.0,
        potable_water_max_storage_l=0.0,
        potable_water_max_rate_gpm=0.0,
    )
    return config


def resource_inputs(n=5, **overrides):
    inputs = {
        key: np.zeros(n)
        for key in (
            "Wind_unitfactor", "Solar_unitfactor", "Wave_unitfactor", "Load",
            "Hydrogen_demand_g", "PotableWater_demand_l",
        )
    }
    inputs.update({key: np.asarray(value, dtype=float) for key, value in overrides.items()})
    return inputs


def solve(config, inputs):
    model.update_derived_config(config)
    problem = model.build_model(config, inputs)
    model.solve_model(config, problem)
    return problem[1], model.build_timeseries_dataframe(config, problem[1])


def test_constant_generator_mean_and_lifetime_fuel_cost():
    config = isolated_config()
    config["limits"].update(generator_max_capacity=1000.0)
    config["simulation"]["lifespan"] = 240.0
    config["costs"].update(
        generator_fix_cost=2.0, diesel_cost=5.0, diesel_energy=1000.0,
        refill_service_cost=20.0, tank_size=10.0, floating_platform_cost_per_kg=0.0,
    )
    config["efficiency"]["gen_eff"] = 0.5
    values, frame = solve(config, resource_inputs(Load=np.full(5, 100.0)))
    np.testing.assert_allclose(frame.generator_power_W.iloc[1:], 100.0, atol=1e-7)
    assert values["avg_generator_W"].value[-1] == pytest.approx(100.0)
    # 24 kWh electric requires 48 gallons; fuel $240 + 4.8 refill services $96.
    assert values["total_cost"].value[-1] == pytest.approx(200.0 + 240.0 + 96.0)
    assert values["cost_per_watt"].value[-1] == pytest.approx(5.36)


@pytest.mark.parametrize("objective", model.SUPPORTED_OBJECTIVES)
def test_objective_selection_changes_the_optimal_technology_and_size(objective):
    config = isolated_config()
    config["objective"] = objective
    config["limits"].update(wind_max_capacity=200.0, solar_max_capacity=200.0)
    config["simulation"]["lifespan"] = 1.0
    config["costs"].update(wind_fix_cost=10.0, solar_fix_cost=1.0,
                           wind_vom_cost_perWh=1.0, solar_vom_cost_perWh=1.0)
    config["mass"].update(wind_Kg_per_W=0.1, solar_Kg_per_W=1.0,
                          floating_platform_mass_per_supported_mass=0.0)
    values, frame = solve(config, resource_inputs(Load=[100] * 5,
                                                Wind_unitfactor=[1] * 5, Solar_unitfactor=[1] * 5))
    # Wind is light but expensive; solar is heavy but inexpensive. Delivered
    # energy costs $100 regardless of technology. The mass ratio intentionally
    # illustrates its incentive to add unused cheap mass up to the size bound.
    expected = {"total_mass": (100.0, 0.0, 10.0),
                "total_cost": (0.0, 100.0, 200.0),
                "cost_per_watt": (0.0, 100.0, 2.0),
                "cost_per_mass": (0.0, 200.0, 1.5)}
    wind, solar, metric = expected[objective]
    assert values["wind_scale"].value[-1] == pytest.approx(wind, abs=1e-5)
    assert values["solar_scale"].value[-1] == pytest.approx(solar, abs=1e-5)
    assert values[objective].value[-1] == pytest.approx(metric)
    np.testing.assert_allclose(frame.total_power_W.iloc[1:], 100.0, atol=1e-5)


@pytest.mark.parametrize("resource", ["hydrogen", "water"])
def test_production_rate_is_a_hard_feasibility_limit(resource):
    config = isolated_config()
    config["limits"]["generator_max_capacity"] = 1000.0
    inputs = resource_inputs()
    if resource == "hydrogen":
        config["limits"]["h2_max_charge_rate_gph"] = 1.0
        inputs["Hydrogen_demand_g"][:] = 2.0
    else:
        # Explicitly disabling production cannot be overridden by demand.
        inputs["PotableWater_demand_l"][:] = 1.0
    with pytest.raises(Exception, match="(?i)(solution not found|infeasib)"):
        solve(config, inputs)


@pytest.mark.parametrize("resource", ["hydrogen", "water"])
def test_inventory_drawdown_and_production_at_capacity(resource):
    config = isolated_config()
    config["limits"]["generator_max_capacity"] = 1000.0
    inputs = resource_inputs()
    if resource == "hydrogen":
        config["limits"].update(h2_min_storage_g=10.0, h2_max_storage_g=10.0,
                                h2_max_charge_rate_gph=1.0)
        config["efficiency"]["h2_initial_charge"] = 0.4
        inputs["Hydrogen_demand_g"][:] = 2.0
        storage_column, power_column, energy = "h2_storage_g", "h2_power_W", 65.0
    else:
        config["limits"].update(potable_water_min_storage_l=10.0,
                                potable_water_max_storage_l=10.0,
                                potable_water_max_rate_gpm=1.0 / (60 * model.LITERS_PER_GALLON))
        config["efficiency"]["potable_water_initial_charge"] = 0.4
        inputs["PotableWater_demand_l"][:] = 2.0
        storage_column, power_column, energy = "potable_storage_l", "potable_power_W", 4.5
    _, frame = solve(config, inputs)
    # Four hours of demand consume initial inventory 4 plus production 4.
    np.testing.assert_allclose(frame[storage_column], [4, 3, 2, 1, 0], atol=1e-6)
    np.testing.assert_allclose(frame[power_column].iloc[1:], energy, atol=1e-6)
    np.testing.assert_allclose(frame.power_balance_W.iloc[1:], 0.0, atol=1e-6)


def test_disabled_battery_transfer_preserves_required_capacity_and_mass():
    config = isolated_config()
    config["limits"].update(batt_min_capacity=100.0, batt_max_capacity=100.0)
    config["efficiency"].update(batt_initial_charge=0.4, batt_final_charge=0.4)
    values, frame = solve(config, resource_inputs())
    np.testing.assert_allclose(frame.battery_storage_Wh, 40.0, atol=1e-6)
    assert values["batt_scale"].value[-1] == pytest.approx(100.0)
    # Battery 0.8 kg plus 30% supporting platform, even with no dispatch.
    assert values["total_mass"].value[-1] == pytest.approx(1.04)


def test_doldrum_does_not_increase_untouched_resource(tmp_path):
    config = isolated_config(4)
    config["simulation"].update(doldrum_time=0, doldrum_windows=[
        {"start_hour": 2, "duration_hours": 1, "resources": ["wind"]},
    ])
    data = pd.DataFrame({
        "Time": range(4), "Load": [1] * 4, "Wind": [1, 2, 4, 2],
        "Solar": [1] * 4, "Wave": [1] * 4, "Hydrogen": [0] * 4, "PotableWater": [0] * 4,
    })
    config["simulation"]["data_file"] = str(tmp_path / "resources.csv")
    data.to_csv(config["simulation"]["data_file"], index=False)
    inputs = model.load_resource_inputs(config)
    np.testing.assert_allclose(inputs["Wind_unitfactor"], [0.25, 0.5, 0.0, 0.5])


def test_battery_round_trip_cycle_matches_energy_and_power_limits():
    config = isolated_config(3)
    config["limits"].update(
        wind_min_capacity=100.0, wind_max_capacity=100.0,
        batt_max_capacity=200.0, batt_max_charge_rate=100.0,
    )
    config["efficiency"].update(batt_round_trip_eff=0.81,
                                batt_initial_charge=0.0, batt_final_charge=0.0)
    _, frame = solve(config, resource_inputs(3, Wind_unitfactor=[0, 1, 0], Load=[0, 0, 81]))
    # Charge one hour at 100 W, store 90 Wh, then deliver 81 Wh and end empty.
    np.testing.assert_allclose(frame.battery_power_W.iloc[1:], [-100.0, 81.0], atol=1e-5)
    np.testing.assert_allclose(frame.battery_storage_Wh, [0.0, 90.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(frame.power_balance_W.iloc[1:], 0.0, atol=1e-6)
    config["limits"]["batt_max_charge_rate"] = 99.0
    with pytest.raises(Exception, match="(?i)(solution not found|infeasib)"):
        solve(config, resource_inputs(3, Wind_unitfactor=[0, 1, 0], Load=[0, 0, 81]))


def test_battery_cannot_dump_inventory_by_charging_and_discharging_itself():
    config = isolated_config(3)
    config["limits"].update(batt_min_capacity=100.0, batt_max_capacity=100.0,
                            batt_max_charge_rate=1000.0)
    config["efficiency"].update(batt_initial_charge=0.4, batt_final_charge=0.0,
                                batt_round_trip_eff=0.81)
    # With no load or other sink, losing 40 Wh requires an unmodeled dump load.
    with pytest.raises(Exception, match="(?i)(solution not found|infeasib)"):
        solve(config, resource_inputs(3))


@pytest.mark.parametrize("resource", ["hydrogen", "water"])
def test_full_initial_inventory_is_preserved_without_demand(resource):
    config = isolated_config()
    config["limits"]["generator_max_capacity"] = 10.0
    if resource == "hydrogen":
        config["limits"].update(h2_min_storage_g=100.0, h2_max_storage_g=100.0)
        config["efficiency"]["h2_initial_charge"] = 1.0
        column = "h2_storage_g"
    else:
        config["limits"].update(potable_water_min_storage_l=100.0,
                                potable_water_max_storage_l=100.0)
        config["efficiency"]["potable_water_initial_charge"] = 1.0
        column = "potable_storage_l"
    _, frame = solve(config, resource_inputs(Load=np.ones(5)))
    np.testing.assert_allclose(frame[column], 100.0, atol=1e-7)


def test_direct_build_recomputes_fuel_cost_after_configuration_edit():
    config = isolated_config()
    config["limits"]["generator_max_capacity"] = 10.0
    config["costs"].update(diesel_energy=1000.0, diesel_cost=5.0, refill_service_cost=0.0)
    config["efficiency"]["gen_eff"] = 0.5
    problem = model.build_model(config, resource_inputs(Load=np.ones(5)))
    model.solve_model(config, problem)
    # Each electric Wh needs 0.002 gallon at $5/gallon.
    assert config["costs"]["generator_vom_cost_perWh"] == pytest.approx(0.01)


def test_direct_paper_build_records_fuel_cost_without_duplicate_delivery_cost():
    config = isolated_config()
    config["solver"]["backend"] = "paper"
    config["costs"].update(diesel_energy=1000, diesel_cost=5, refill_service_cost=200)
    config["efficiency"]["gen_eff"] = 0.5
    m, _ = model.build_model(config, resource_inputs())
    try:
        # Paper deliveries are charged discretely; its per-Wh input contains
        # fuel alone, including when constructed directly through this API.
        assert config["costs"]["generator_vom_cost_perWh"] == pytest.approx(0.01)
        assert m.config["costs"]["generator_vom_cost_perWh"] == pytest.approx(0.01)
    finally:
        m.cleanup()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, "bad"])
def test_csv_rejects_invalid_physical_resource_values(tmp_path, bad):
    config = isolated_config(3)
    data = pd.DataFrame({
        "Time": range(3), "Load": [1] * 3, "Wind": [1, bad, 2],
        "Solar": [1] * 3, "Wave": [1] * 3, "Hydrogen": [0] * 3, "PotableWater": [0] * 3,
    })
    config["simulation"]["data_file"] = str(tmp_path / "bad.csv")
    data.to_csv(config["simulation"]["data_file"], index=False)
    with pytest.raises(ValueError, match="Wind.*finite non-negative"):
        model.load_resource_inputs(config)


@pytest.mark.parametrize("key,bad,match", [
    ("Wind_unitfactor", [1, 1, 2, 1, 1], "interval"),
    ("Load", [1, 2], "shape"),
    ("Hydrogen_demand_g", [0, 0, np.nan, 0, 0], "finite"),
    ("PotableWater_demand_l", [0, 0, -1, 0, 0], "non-negative"),
])
def test_direct_model_api_rejects_invalid_profiles(key, bad, match):
    inputs = resource_inputs()
    inputs[key] = bad
    with pytest.raises(ValueError, match=match):
        model.build_model(isolated_config(), inputs)


@pytest.mark.parametrize("section,key,bad", [
    ("simulation", "peak_load", np.nan), ("mass", "wind_Kg_per_W", -0.1),
    ("costs", "wind_fix_cost", np.inf), ("simulation", "nhrs", 4.5),
    ("simulation", "nhrs", True), ("simulation", "lifespan", 0),
    ("simulation", "H2DailyDemand", -1), ("efficiency", "h2_Wh_per_g", "65"),
    ("solver", "remote", "false"), ("limits", "batt_max_capacity", np.inf),
    ("solver", "imode", 3), ("solver", "solver", 4), ("solver", "cv_type", 7),
    ("solver", "max_iter", 0),
])
def test_invalid_configuration_fails_before_solver(section, key, bad):
    config = isolated_config()
    config[section][key] = bad
    with pytest.raises(ValueError, match=key):
        model.build_model(config, resource_inputs())


def test_feasibility_replay_detects_balance_inventory_and_average_corruption():
    config = isolated_config()
    config["limits"].update(generator_max_capacity=1000.0, h2_min_storage_g=10.0,
                            h2_max_storage_g=10.0, h2_max_charge_rate_gph=1.0)
    config["efficiency"]["h2_initial_charge"] = 0.4
    values, _ = solve(config, resource_inputs(Hydrogen_demand_g=[2] * 5))
    assert values["solution_diagnostics"]["passed"]
    for key, check_name, unit in (
        ("generator_power", "electrical_balance", "W"),
        ("h2_storagelevel", "hydrogen_material_balance", "g"),
        ("avg_generator_W", "generator_average_power", "W"),
    ):
        original = list(values[key].value)
        changed = original.copy()
        changed[-1] += 1.0
        values[key].value = changed
        diagnostics = model.solution_diagnostics(config, values)
        assert not diagnostics["passed"]
        assert diagnostics["checks"][check_name]["max_residual"] == pytest.approx(1.0)
        assert diagnostics["checks"][check_name]["unit"] == unit
        values[key].value = original

    values["total_cost"].value = [np.nan] * 5
    with pytest.raises(RuntimeError, match="total_cost.*non-finite"):
        model.solution_diagnostics(config, values)


def test_success_status_does_not_bypass_physical_replay(monkeypatch):
    config = isolated_config()
    config["limits"]["generator_max_capacity"] = 1000.0
    values, _ = solve(config, resource_inputs(Load=[100] * 5))
    m = values["model"]
    # A successful status attached to a corrupted dispatch must fail acceptance.
    values["generator_power"].value = [0, 100, 101, 100, 100]
    monkeypatch.setattr(m, "solve", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="physical feasibility.*electrical_balance"):
        model.solve_model(config, (m, values))


@pytest.mark.parametrize("key,check_name", [
    ("total_cost", "total_cost_accounting"),
    ("total_mass", "total_mass_accounting"),
    ("cost_per_watt", "cost_per_watt_accounting"),
    ("cost_per_mass", "cost_per_mass_accounting"),
    ("avg_batt_throughput_W", "battery_average_throughput"),
])
def test_finite_but_incorrect_reported_economics_fail_acceptance(key, check_name, monkeypatch):
    config = isolated_config()
    config["limits"]["generator_max_capacity"] = 100.0
    values, _ = solve(config, resource_inputs(Load=[10] * 5))
    try:
        changed = list(values[key].value)
        changed[-1] += 10.0
        values[key].value = changed
        monkeypatch.setattr(values["model"], "solve", lambda **kwargs: None)
        with pytest.raises(RuntimeError, match=check_name):
            model.solve_model(config, (values["model"], values))
    finally:
        values["model"].cleanup()


@pytest.mark.parametrize("start,duration,removed_rows,expected_span", [
    (2, 1, [2], (1, 2)),
    (2.4, 0.7, [2, 3], (1, 3)),
    (0, 2, [1], (0, 1)),
    (4, 4, [4, 5], (3, 5)),
])
def test_outage_shading_matches_the_intervals_that_require_backup(
        tmp_path, start, duration, removed_rows, expected_span):
    config = isolated_config(6)
    config["limits"].update(wind_min_capacity=10.0, wind_max_capacity=10.0,
                            generator_max_capacity=10.0)
    config["simulation"].update(peak_load=10.0, doldrum_time=0, doldrum_windows=[
        {"start_hour": start, "duration_hours": duration, "resources": ["wind"]},
    ])
    resource = tmp_path / "resource.csv"
    pd.DataFrame({"Time": range(6), "Load": 1, "Wind": 1, "Solar": 0, "Wave": 0,
                  "Hydrogen": 0, "PotableWater": 0}).to_csv(resource, index=False)
    config["simulation"]["data_file"] = str(resource)
    values, frame = solve(config, model.load_resource_inputs(config))
    try:
        assert np.flatnonzero(frame.generator_power_W > 1e-6).tolist() == removed_rows
        # A dispatch row is the rate over the preceding hour; the shaded
        # rectangle must cover those same backup-energy intervals.
        figure = model.create_stacked_figure(config, values)
        patch = figure.axes[0].patches[0]
        np.testing.assert_allclose([24 * patch.get_x(), 24 * (patch.get_x() + patch.get_width())],
                                   expected_span, atol=1e-12)
        figure.clear()
    finally:
        values["model"].cleanup()


def test_centered_outage_shading_uses_rounded_sample_mask():
    simulation = isolated_config(6)["simulation"]
    simulation.update(doldrum_time=0.5, doldrums={"wind": 0.5})
    # Center sample 3 and a half-width of 0.5 remove row 2, not half of
    # each adjacent interval. Its solved interval is elapsed hour 1–2.
    spans = model._doldrum_day_spans(simulation)
    assert spans == [(1 / 24, 2 / 24, "Wind outage")]


@pytest.mark.parametrize("window,expected_hours", [
    ({"start_hour": 2, "duration_hours": 1}, (2, 3)),
    ({"start_hour": 2.4, "duration_hours": 0.7}, (2, 4)),
    ({"start_hour": 0, "duration_hours": 2}, (0, 2)),
    ({"start_hour": 4, "duration_hours": 4}, (4, 5)),
    (None, (1, 3)),
])
def test_paper_outage_shading_matches_actual_backup_intervals(tmp_path, window, expected_hours):
    config = isolated_config(6)
    config["solver"]["backend"] = "paper"
    config["paper"]["generator_min_installed_w"] = 0
    config["paper"]["primary_cost_tolerance_usd"] = 0
    config["limits"].update(wind_min_capacity=10, wind_max_capacity=10,
                            generator_max_capacity=10)
    config["simulation"].update(peak_load=10, doldrum_time=0,
                                doldrum_windows=[])
    if window is None:
        config["simulation"].update(doldrum_time=1, doldrums={"wind": 0.4, "solar": 0, "wave": 0})
    else:
        config["simulation"]["doldrum_windows"] = [dict(window, resources=["wind"])]
    resource = tmp_path / "resource.csv"
    pd.DataFrame({"Time": range(5), "Load": 1, "Wind": 1, "Solar": 0, "Wave": 0,
                  "Hydrogen": 0, "PotableWater": 0}).to_csv(resource, index=False)
    config["simulation"]["data_file"] = str(resource)
    values, frame = solve(config, model.load_resource_inputs(config))
    try:
        start, stop = expected_hours
        assert np.flatnonzero(frame.generator_power_W > 1e-5).tolist() == list(range(start + 1, stop + 1))
        figure = model.create_stacked_figure(config, values)
        try:
            actual_spans = [(24 * patch.get_x(), 24 * (patch.get_x() + patch.get_width()))
                            for patch in figure.axes[0].patches]
            assert any(np.allclose(span, expected_hours, atol=1e-12) for span in actual_spans)
        finally:
            figure.clear()
    finally:
        values["model"].cleanup()


def test_shared_diagnostics_replays_paper_cyclic_boundary_conditions():
    config = isolated_config(3)
    config["solver"]["backend"] = "paper"
    config["limits"].update(solar_max_capacity=1000, batt_max_capacity=1000,
                            batt_max_charge_rate=1000)
    config["efficiency"]["batt_round_trip_eff"] = 0.81
    values, _ = solve(config, resource_inputs(3, Load=[0, 90, 0], Solar_unitfactor=[0, 0, 1]))
    try:
        # Paper inventories are cyclic and freely initialized; GEKKO's default
        # initial/final charge fractions must not reject this valid solution.
        np.testing.assert_allclose(values["batt_storagelevel"].value, [100, 0, 100], atol=1e-6)
        assert model.solution_diagnostics(config, values)["passed"]
    finally:
        values["model"].cleanup()


def test_shortest_horizon_plot_contains_its_solved_interval(tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgb

    config = isolated_config(2)
    config["limits"]["generator_max_capacity"] = 100.0
    values, frame = solve(config, resource_inputs(2, Load=[10, 10]))
    assert frame.is_initial_state.tolist() == [True, False]
    fig = model.create_stacked_figure(config, values)
    balance = fig.axes[0]
    supplied = next(line for line in balance.lines if line.get_label() == "Power supplied (W)")
    np.testing.assert_allclose(supplied.get_xdata(), [1 / 24])
    np.testing.assert_allclose(supplied.get_ydata(), [10.0])
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())[:, :, :3]
    x, y = balance.transData.transform((1 / 24, 10.0))
    row, column = int(pixels.shape[0] - y), int(x)
    patch = pixels[row - 6:row + 7, column - 6:column + 7].astype(float)
    expected_color = np.asarray(to_rgb(supplied.get_color())) * 255
    # A single Line2D point without a marker silently rendered no dispatch.
    # Require actual supplied-power colored pixels at its physical coordinate.
    assert np.any(np.all(np.abs(patch - expected_color) < 15, axis=-1))
    fig.savefig(tmp_path / "single_interval.png")


def test_linear_solver_hint_rejects_nonlinear_battery_operation():
    config = isolated_config()
    config["solver"]["linear"] = True
    config["limits"].update(batt_max_capacity=100.0, batt_max_charge_rate=100.0)
    with pytest.raises(ValueError, match="linear.*complementarity"):
        model.build_model(config, resource_inputs())


def test_disabled_zero_demand_system_is_feasible_with_zero_cost_and_mass():
    config = isolated_config()
    values, frame = solve(config, resource_inputs())
    assert values["total_cost"].value[-1] == pytest.approx(0.0)
    assert values["total_mass"].value[-1] == pytest.approx(0.0)
    np.testing.assert_array_equal(frame.total_power_W, 0.0)
    assert values["solution_diagnostics"]["passed"]
    with pytest.raises(ValueError, match="infeasible.*no electrical source"):
        solve(config, resource_inputs(Load=[0, 1, 1, 1, 1]))


@pytest.mark.parametrize("window,match", [
    ({"start_days": 5, "duration_days": 1, "resources": ["wind"]}, "Unknown.*start_days"),
    ({"start_day": 5, "start_hour": 5, "duration_days": 1, "resources": ["wind"]}, "conflicting.*start"),
    ({"stop_day": 5, "duration_hours": 5, "resources": ["wind"]}, "conflicting.*stop"),
    ({"duration_days": 1, "resources": ["wind"], "resource": "solar"}, "conflicting.*resources"),
    ({"duration_days": 1, "resources": None}, "resources.*name"),
])
def test_outage_configuration_rejects_typos_and_ambiguous_timing(window, match):
    config = isolated_config()
    config["simulation"]["doldrum_windows"] = [window]
    with pytest.raises(ValueError, match=match):
        model.build_model(config, resource_inputs())
