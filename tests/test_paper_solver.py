"""Analytical checks of the portable paper model's physics and logistics."""

import numpy as np
import pytest

from sirenolite import model, paper_solver


def config_for_intervals(n):
    config = model.default_inputs()
    config["objective"] = "total_cost"
    config["solver"]["backend"] = "paper"
    config["simulation"].update(
        nhrs=n + 1, lifespan=float(n), hotel_load=0, H2DailyDemand=0, H2ODailyDemand=0
    )
    config["paper"] = dict(
        generator_min_installed_w=0,
        max_delivery_trips=20,
        delivery_repeat_factor=1,
        primary_cost_tolerance_usd=1e-4,
    )
    for tech in ("generator", "wind", "solar", "wave", "batt"):
        config["limits"][f"{tech}_min_capacity"] = 0
        config["limits"][f"{tech}_max_capacity"] = 0
    config["limits"].update(
        batt_max_charge_rate=0,
        h2_min_storage_g=0,
        h2_max_storage_g=0,
        h2_max_charge_rate_gph=0,
        potable_water_min_storage_l=0,
        potable_water_max_storage_l=0,
        potable_water_max_rate_gpm=0,
    )
    config["costs"]["floating_platform_cost_per_kg"] = 0
    return config


def inputs_for(load, **overrides):
    n = len(load)
    result = {
        key: np.zeros(n + 1)
        for key in (
            "Load",
            "Wind_unitfactor",
            "Solar_unitfactor",
            "Wave_unitfactor",
            "Hydrogen_demand_g",
            "PotableWater_demand_l",
        )
    }
    result["Load"][1:] = load
    for key, value in overrides.items():
        result[key][1:] = value
    return result


def solve(config, inputs):
    problem = paper_solver.build_model(config, inputs)
    paper_solver.solve_model(config, problem)
    assert problem[1]["solution_diagnostics"]["passed"]
    return problem[1]


def scalar(values, key):
    return values[key].value[-1]


def test_discrete_delivery_and_commodity_are_charged_once():
    config = config_for_intervals(4)
    config["limits"]["generator_max_capacity"] = 1000
    config["paper"]["generator_min_installed_w"] = 500
    config["costs"].update(
        generator_fix_cost=2,
        diesel_cost=5,
        diesel_energy=100,
        tank_size=5,
        refill_service_cost=20,
    )
    config["efficiency"]["gen_eff"] = 0.5
    values = solve(config, inputs_for([100] * 4))
    # 400Wh needs8gal: two5gal deliveries, not 1.6 services.
    assert scalar(values, "generator_scale") == pytest.approx(500)
    assert values["paper_summary"]["delivery_trips"] == 2
    assert scalar(values, "total_cost") == pytest.approx(1000 + 40 + 40)
    assert values["model"].options.APPSTATUS == 1
    assert model.build_timeseries_dataframe(config, values).shape[0] == 5


def test_cyclic_battery_can_begin_full_but_cannot_supply_free_energy():
    config = config_for_intervals(2)
    config["limits"].update(
        solar_max_capacity=1000, batt_max_capacity=1000, batt_max_charge_rate=1000
    )
    config["efficiency"]["batt_round_trip_eff"] = 0.81
    values = solve(config, inputs_for([90, 0], Solar_unitfactor=[0, 1]))
    # First interval90Wh AC requires100Wh stored; second must put100Wh back.
    assert scalar(values, "batt_scale") == pytest.approx(100, abs=1e-6)
    assert scalar(values, "solar_scale") == pytest.approx(100 / 0.9, abs=1e-6)
    assert values["batt_storagelevel"].value == pytest.approx([100, 0, 100], abs=1e-6)
    assert scalar(values, "generator_scale") == pytest.approx(0)


def test_cyclic_tank_production_and_energy_balance():
    config = config_for_intervals(2)
    config["limits"].update(
        solar_max_capacity=1000,
        potable_water_max_storage_l=100,
        potable_water_max_rate_gpm=1,
    )
    config["efficiency"]["potable_water_Wh_per_l"] = 4.5
    values = solve(
        config,
        inputs_for([0, 0], Solar_unitfactor=[0, 1], PotableWater_demand_l=[10, 0]),
    )
    assert scalar(values, "solar_scale") == pytest.approx(45, abs=1e-6)
    assert scalar(values, "potable_water_storage_scale") == pytest.approx(10, abs=1e-6)
    assert values["potable_water_storagelevel"].value == pytest.approx(
        [10, 0, 10], abs=1e-6
    )


def test_no_resources_with_cyclic_storage_is_infeasible():
    config = config_for_intervals(2)
    config["limits"].update(batt_max_capacity=1000, batt_max_charge_rate=1000)
    with pytest.raises(RuntimeError, match="primary optimization"):
        solve(config, inputs_for([100, 100]))


def test_delivery_limit_is_hard_feasibility_constraint():
    config = config_for_intervals(2)
    config["limits"]["generator_max_capacity"] = 1000
    config["costs"].update(tank_size=1, diesel_energy=100)
    config["efficiency"]["gen_eff"] = 1
    config["paper"]["max_delivery_trips"] = 1
    with pytest.raises(RuntimeError, match="primary optimization"):
        solve(config, inputs_for([100, 100]))


def test_hotel_load_and_zero_load():
    config = config_for_intervals(3)
    values = solve(config, inputs_for([0] * 3))
    assert scalar(values, "total_cost") == pytest.approx(0)
    config["simulation"]["hotel_load"] = 100
    config["limits"]["wind_max_capacity"] = 1000
    values = solve(config, inputs_for([0] * 3, Wind_unitfactor=[1] * 3))
    assert scalar(values, "wind_scale") == pytest.approx(100)


def test_rejects_unsupported_hydrogen_and_objectives():
    config = config_for_intervals(2)
    config["limits"]["h2_max_charge_rate_gph"] = 1
    with pytest.raises(ValueError, match="hydrogen"):
        paper_solver.build_model(config, inputs_for([0, 0]))
    config["limits"]["h2_max_charge_rate_gph"] = 0
    config["objective"] = "total_mass"
    with pytest.raises(ValueError, match="total_cost"):
        paper_solver.build_model(config, inputs_for([0, 0]))


def test_configuration_bridge_rejects_unknown_paper_controls():
    from sirenolite.config import configuration_from_payload

    config = config_for_intervals(2)
    config["paper"]["ignore_fuel_delivery_constraint"] = True
    with pytest.raises(
        ValueError,
        match="Unknown configuration key: paper.ignore_fuel_delivery_constraint",
    ):
        configuration_from_payload(config)


def test_diagnostics_detect_corrupted_reporting_and_array_length():
    config = config_for_intervals(2)
    config["limits"]["wind_max_capacity"] = 1000
    values = solve(config, inputs_for([100, 100], Wind_unitfactor=[1, 1]))
    values["total_mass"].value[-1] += 1
    assert not paper_solver.solution_diagnostics(config, values)["checks"][
        "total_mass_accounting"
    ]["passed"]
    values["wind_power"].value.pop()
    with pytest.raises(RuntimeError, match="Incomplete or non-finite"):
        paper_solver.solution_diagnostics(config, values)


@pytest.mark.parametrize(
    "field,check",
    [
        ("cost_per_watt", "cost_per_watt_accounting"),
        ("cost_per_mass", "cost_per_mass_accounting"),
        ("load", "electrical_demand_input"),
        ("potable_water_demand_l", "water_demand_input"),
        ("wind_unitfactor", "wind_resource_input"),
    ],
)
def test_diagnostics_reject_corrupted_source_and_reported_metrics(field, check):
    config = config_for_intervals(2)
    config["limits"]["wind_max_capacity"] = 1000
    values = solve(config, inputs_for([100, 100], Wind_unitfactor=[1, 1]))
    values[field].value[-1] += 1
    assert not paper_solver.solution_diagnostics(config, values)["checks"][check][
        "passed"
    ]


def test_diagnostics_reject_fractional_logistics_and_inconsistent_cost_summary():
    config = config_for_intervals(2)
    config["limits"]["generator_max_capacity"] = 1000
    config["costs"]["tank_size"] = 100
    values = solve(config, inputs_for([100, 100]))
    values["paper_summary"]["delivery_trips"] = 1.5
    checks = paper_solver.solution_diagnostics(config, values)["checks"]
    assert not checks["delivery_trip_integrality"]["passed"]
    values["paper_summary"]["generator_installed"] = 0.5
    checks = paper_solver.solution_diagnostics(config, values)["checks"]
    assert not checks["generator_installation_integrality"]["passed"]
