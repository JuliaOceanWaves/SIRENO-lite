"""Portable annual-paper sizing and dispatch model, solved locally by HiGHS.

Input row zero is an inventory initialization node; rows 1..N are one-hour
intervals. Inventories are cyclic with freely optimized initial levels. Diesel
uses an installed binary and integer full-tank delivery allowance. This model
supports electrical and desalination missions, with the paper's linear cost
objective and a second, fixed-design minimum-battery-throughput solve.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack

TECHNOLOGIES = ("generator", "wind", "solar", "wave")


class _Value:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float).tolist()


class _PaperModel:
    def __init__(self, config, inputs):
        self.config = deepcopy(config)
        self.inputs = {key: np.asarray(value, dtype=float).copy() for key, value in inputs.items()}
        self.time = np.arange(config["simulation"]["nhrs"], dtype=float)
        self.options = SimpleNamespace(APPSTATUS=0, SOLVESTATUS=0, OBJFCNVAL=np.nan)

    def cleanup(self):
        """No subprocess files are created by the in-process solver."""


def build_model(config, inputs):
    if config["objective"] != "total_cost":
        raise ValueError("The paper backend supports the total_cost objective only.")
    limits = config["limits"]
    if any(limits[key] > 0 for key in ("h2_min_storage_g", "h2_max_storage_g", "h2_max_charge_rate_gph")) or np.any(np.asarray(inputs["Hydrogen_demand_g"]) > 0):
        raise ValueError("The paper backend does not support hydrogen; disable its demand, capacity, and production rate.")
    nhrs = int(config["simulation"]["nhrs"])
    if nhrs < 2:
        raise ValueError("The paper backend requires at least one hourly interval.")
    for key in ("Wind_unitfactor", "Solar_unitfactor", "Wave_unitfactor", "Load", "Hydrogen_demand_g", "PotableWater_demand_l"):
        arr = np.asarray(inputs[key], dtype=float)
        if arr.shape != (nhrs,) or not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"Invalid paper input {key}: expected {nhrs} finite non-negative entries.")
        if key.endswith("unitfactor") and np.any(arr > 1):
            raise ValueError(f"Paper resource input {key} must lie between zero and one.")
    paper = config["paper"]
    for key in ("generator_min_installed_w", "max_delivery_trips", "delivery_repeat_factor", "primary_cost_tolerance_usd"):
        value = float(paper[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"paper.{key} must be finite and non-negative.")
    if float(paper["max_delivery_trips"]) != int(paper["max_delivery_trips"]):
        raise ValueError("paper.max_delivery_trips must be an integer.")
    m = _PaperModel(config, inputs)
    return m, {"model": m}


def _assemble(config, inputs):
    sim, limits, costs, eff, mass, paper = (config[key] for key in ("simulation", "limits", "costs", "efficiency", "mass", "paper"))
    n = int(sim["nhrs"]) - 1
    load = inputs["Load"][1:] + sim.get("hotel_load", 0.0)
    water_demand = inputs["PotableWater_demand_l"][1:]
    eta = np.sqrt(eff["batt_round_trip_eff"])
    energy_per_delivery = costs["tank_size"] * costs["diesel_energy"] * eff["gen_eff"]
    weight = sim["lifespan"] / n
    names = ("generator", "wind", "solar", "wave", "battery", "water", "installed", "trips")
    idx = {name: i for i, name in enumerate(names)}
    size = len(names)
    for name in ("generator_power", "wind_power", "solar_power", "wave_power", "charge", "discharge", "water_production", "battery_state", "water_state"):
        count = n + 1 if name.endswith("state") else n
        idx[name] = np.arange(size, size + count)
        size += count
    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    for name in TECHNOLOGIES:
        lower[idx[name]] = limits[f"{name}_min_capacity"]
        upper[idx[name]] = limits[f"{name}_max_capacity"]
    lower[idx["battery"]], upper[idx["battery"]] = limits["batt_min_capacity"], limits["batt_max_capacity"]
    lower[idx["water"]], upper[idx["water"]] = limits["potable_water_min_storage_l"], limits["potable_water_max_storage_l"]
    upper[idx["installed"]] = 1
    upper[idx["trips"]] = paper["max_delivery_trips"]
    upper[idx["charge"]] = upper[idx["discharge"]] = limits["batt_max_charge_rate"]
    upper[idx["water_production"]] = limits["potable_water_max_rate_gpm"] * 60 * 3.785411784
    lower[idx["battery_state"]] = limits["batt_min_storage_level"]
    integer = np.zeros(size, dtype=int)
    integer[[idx["installed"], idx["trips"]]] = 1
    objective = np.zeros(size)
    for name in TECHNOLOGIES:
        objective[idx[name]] = costs[f"{name}_fix_cost"] + costs["floating_platform_cost_per_kg"] * mass["floating_platform_mass_per_supported_mass"] * mass[f"{name}_Kg_per_W"]
        unit_cost = costs["diesel_cost"] / (costs["diesel_energy"] * eff["gen_eff"]) if name == "generator" else costs[f"{name}_vom_cost_perWh"]
        objective[idx[f"{name}_power"]] = weight * unit_cost
    objective[idx["battery"]] = costs["batt_fix_cost"] + costs["floating_platform_cost_per_kg"] * mass["floating_platform_mass_per_supported_mass"] * mass["batt_Kg_per_Wh"]
    objective[idx["water"]] = costs["potable_water_storage_fix_cost_per_l"]
    objective[idx["trips"]] = paper["delivery_repeat_factor"] * costs["refill_service_cost"]
    objective[idx["charge"]] = objective[idx["discharge"]] = weight * costs["batt_vom_cost_perWh"]
    rows, cols, data, rhs = [], [], [], []
    def add(terms, right=0.0, count=1):
        base = len(rhs)
        rr = np.arange(base, base + count)
        for columns, coefficients in terms:
            rows.extend(rr)
            cols.extend(np.broadcast_to(columns, (count,)))
            data.extend(np.broadcast_to(coefficients, (count,)))
        rhs.extend(np.broadcast_to(right, (count,)))
    # Equalities first, allowing both MILP and LP interfaces to use the same matrix.
    terms = [(idx[f"{name}_power"], 1.0) for name in TECHNOLOGIES]
    add(terms + [(idx["discharge"], 1.0), (idx["charge"], -1.0), (idx["water_production"], -eff["potable_water_Wh_per_l"])], load, n)
    add([(idx["battery_state"][1:], 1.0), (idx["battery_state"][:-1], -1.0), (idx["charge"], -eta), (idx["discharge"], 1 / eta)], count=n)
    add([(idx["water_state"][1:], 1.0), (idx["water_state"][:-1], -1.0), (idx["water_production"], -1.0)], -water_demand, n)
    for name in ("battery_state", "water_state"):
        add([(idx[name][-1], 1.0), (idx[name][0], -1.0)])
    neq = len(rhs)
    for name in TECHNOLOGIES:
        cf = 1.0 if name == "generator" else inputs[f"{name.title()}_unitfactor"][1:]
        add([(idx[f"{name}_power"], 1.0), (idx[name], -cf)], count=n)
    for name in ("battery", "water"):
        add([(idx[f"{name}_state"], 1.0), (idx[name], -1.0)], count=n + 1)
    add([(idx["generator"], 1.0), (idx["installed"], -limits["generator_max_capacity"])])
    add([(idx["generator"], -1.0), (idx["installed"], paper["generator_min_installed_w"])])
    add([(idx["trips"], 1.0), (idx["installed"], -paper["max_delivery_trips"])])
    add([(idx["installed"], 1.0), (idx["trips"], -1.0)])
    # Aggregate one-hour dispatch into the fuel allowance for this solved period.
    r = len(rhs)
    rows.extend([r] * (n + 1))
    cols.extend(idx["generator_power"])
    cols.append(idx["trips"])
    data.extend([1.0] * n)
    data.append(-energy_per_delivery)
    rhs.append(0.0)
    matrix = coo_matrix((data, (rows, cols)), shape=(len(rhs), size)).tocsc()
    return idx, lower, upper, integer, objective, matrix, np.asarray(rhs), neq


def solve_model(config, problem):
    m, values = problem
    # Use the snapshot assembled for this solve; later GUI edits cannot change it.
    config, inputs = m.config, m.inputs
    idx, lower, upper, integer, objective, matrix, rhs, neq = _assemble(config, inputs)
    row_lower = np.concatenate((rhs[:neq], np.full(len(rhs) - neq, -np.inf)))
    primary = milp(objective, integrality=integer, bounds=Bounds(lower, upper), constraints=LinearConstraint(matrix, row_lower, rhs), options={"mip_rel_gap": 1e-8})
    if not primary.success or primary.status != 0 or primary.x is None:
        raise RuntimeError(f"HiGHS primary optimization did not reach an optimum: {primary.message}")
    gap = getattr(primary, "mip_gap", None)
    if gap is not None and (not np.isfinite(gap) or gap > 1e-8 + 1e-12):
        raise RuntimeError(f"HiGHS primary relative gap exceeds 1e-8: {gap}")
    primary_x = primary.x.copy()
    # Fix the primary design and logistics before selecting a physical dispatch.
    for name in ("generator", "wind", "solar", "wave", "battery", "water", "installed", "trips"):
        value = float(primary_x[idx[name]])
        if name in ("installed", "trips"):
            value = round(value)
        lower[idx[name]] = upper[idx[name]] = max(0.0, value)
    tolerance = config["paper"]["primary_cost_tolerance_usd"]
    costrow = coo_matrix(objective.reshape(1, -1)).tocsc()
    aub = vstack((matrix[neq:], costrow, -costrow), format="csc")
    bub = np.concatenate((rhs[neq:], [primary.fun + tolerance, -primary.fun + tolerance]))
    selector = np.zeros(len(objective))
    selector[idx["charge"]] = selector[idx["discharge"]] = 1.0
    secondary = linprog(selector, A_ub=aub, b_ub=bub, A_eq=matrix[:neq], b_eq=rhs[:neq], bounds=np.column_stack((lower, upper)), method="highs", options={"primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8})
    if not secondary.success or secondary.status != 0 or secondary.x is None:
        raise RuntimeError(f"HiGHS battery-throughput selection did not reach an optimum: {secondary.message}")
    x = secondary.x
    _populate(config, inputs, values, idx, x, objective, primary, secondary)
    diagnostics = solution_diagnostics(config, values)
    values["solution_diagnostics"] = diagnostics
    if not diagnostics["passed"]:
        failed = [key for key, check in diagnostics["checks"].items() if not check["passed"]]
        raise RuntimeError("Paper solution failed physical checks: " + ", ".join(failed))
    m.options.APPSTATUS = m.options.SOLVESTATUS = 1
    m.options.OBJFCNVAL = float(objective @ x)


def _populate(config, inputs, values, idx, x, objective, primary, secondary):
    sim, costs, mass, eff = (config[key] for key in ("simulation", "costs", "mass", "efficiency"))
    n = int(sim["nhrs"]) - 1
    def setvalue(name, value):
        values[name] = _Value(np.broadcast_to(value, (n + 1,)))
    def interval(name, value):
        setvalue(name, np.r_[0.0, value])
    setvalue("load", inputs["Load"] + sim.get("hotel_load", 0.0))
    setvalue("potable_water_demand_l", inputs["PotableWater_demand_l"])
    for name in ("hydrogen_demand_g", "h2_power_in", "h2_storage_scale", "h2_storagelevel"):
        setvalue(name, 0.0)
    supported_mass = 0.0
    fixed_cost = 0.0
    for name in TECHNOLOGIES:
        capacity = x[idx[name]]
        dispatch = x[idx[f"{name}_power"]]
        setvalue(f"{name}_scale", capacity)
        interval(f"{name}_power", dispatch)
        cf = np.ones(n + 1) if name == "generator" else inputs[f"{name.title()}_unitfactor"]
        available = cf[1:] * capacity
        fraction = np.divide(dispatch, available, out=np.zeros(n), where=available > 0)
        if name == "generator":
            interval("generator_unitfactor", fraction)
        else:
            setvalue(f"{name}_unitfactor", cf)
            interval(f"{name}_cur", fraction)
        setvalue("avg_sol_W" if name == "solar" else f"avg_{name}_W", dispatch.mean())
        supported_mass += capacity * mass[f"{name}_Kg_per_W"]
        fixed_cost += capacity * costs[f"{name}_fix_cost"]
    battery, water = x[idx["battery"]], x[idx["water"]]
    supported_mass += battery * mass["batt_Kg_per_Wh"]
    tank_mass = water * mass["potable_water_storage_Kg_per_l"]
    platform_mass = supported_mass * mass["floating_platform_mass_per_supported_mass"]
    fixed_cost += battery * costs["batt_fix_cost"] + water * costs["potable_water_storage_fix_cost_per_l"]
    setvalue("batt_scale", battery)
    setvalue("potable_water_storage_scale", water)
    setvalue("batt_storagelevel", x[idx["battery_state"]])
    setvalue("potable_water_storagelevel", x[idx["water_state"]])
    charge, discharge = x[idx["charge"]], x[idx["discharge"]]
    interval("batt_charge_power", charge)
    interval("batt_discharge_power", discharge)
    interval("batt_power_inout", discharge - charge)
    water_power = x[idx["water_production"]] * eff["potable_water_Wh_per_l"]
    interval("potable_water_power_in", water_power)
    avg_load = np.mean(inputs["Load"][1:] + sim.get("hotel_load", 0) + water_power)
    setvalue("avg_load_W", avg_load)
    setvalue("avg_batt_throughput_W", np.mean(charge + discharge))
    setvalue("supported_mass", supported_mass)
    setvalue("floating_platform_mass", platform_mass)
    setvalue("potable_water_storage_mass", tank_mass)
    total_mass = supported_mass + platform_mass + tank_mass
    total_cost = float(objective @ x)
    setvalue("total_mass", total_mass)
    setvalue("total_cost", total_cost)
    setvalue("cost_per_watt", total_cost / (avg_load + 1e-6))
    setvalue("cost_per_mass", total_cost / (total_mass + 1e-6))
    setvalue("initial_vector", np.r_[1., np.zeros(n)])
    setvalue("final_vector", np.r_[np.zeros(n), 1.])
    values["paper_summary"] = {
        "backend": "SciPy/HiGHS", "primary_status": primary.message, "selector_status": secondary.message,
        "primary_cost_usd": float(primary.fun), "total_cost_usd": total_cost,
        "primary_cost_change_usd": total_cost - float(primary.fun),
        "primary_relative_gap": float(getattr(primary, "mip_gap", 0.0) or 0.0),
        "generator_installed": int(round(x[idx["installed"]])), "delivery_trips": int(round(x[idx["trips"]])),
        "fixed_capacity_cost_usd": float(fixed_cost), "supported_platform_cost_usd": float(platform_mass * costs["floating_platform_cost_per_kg"]),
        "weighted_operating_cost_usd": float(total_cost - fixed_cost - platform_mass * costs["floating_platform_cost_per_kg"]),
        "horizon_hours": n, "operating_weight": sim["lifespan"] / n,
    }


def solution_diagnostics(config, values):
    """Independent reconstruction in physical units, without matrix reuse."""
    limits, eff, costs, mass, sim, paper = (config[key] for key in ("limits", "efficiency", "costs", "mass", "simulation", "paper"))
    def arr(name):
        result = np.asarray(values[name].value, dtype=float)
        if result.shape != (int(sim["nhrs"]),) or not np.isfinite(result).all():
            raise RuntimeError(f"Incomplete or non-finite paper result {name}.")
        return result
    checks = {}
    def check(name, residual, unit, tolerance=1e-5):
        error = float(np.max(np.abs(residual)))
        checks[name] = {"max_residual": error, "unit": unit, "tolerance": tolerance, "passed": bool(error <= tolerance)}
    def bounds(name, value, low, high, unit):
        check(name, np.maximum(np.maximum(low - value, value - high), 0), unit)
    charge, discharge = arr("batt_charge_power")[1:], arr("batt_discharge_power")[1:]
    water_power = arr("potable_water_power_in")[1:]
    supply = discharge - charge
    expected_fixed, supported_mass, weighted_operating = 0.0, 0.0, 0.0
    weight = sim["lifespan"] / len(charge)
    for name in TECHNOLOGIES:
        capacity = arr(f"{name}_scale")[-1]
        dispatch = arr(f"{name}_power")[1:]
        available = capacity if name == "generator" else capacity * arr(f"{name}_unitfactor")[1:]
        bounds(f"{name}_availability", dispatch, 0, available, "W")
        bounds(f"{name}_capacity", capacity, limits[f"{name}_min_capacity"], limits[f"{name}_max_capacity"], "W")
        supply = supply + dispatch
        expected_fixed += capacity * costs[f"{name}_fix_cost"]
        supported_mass += capacity * mass[f"{name}_Kg_per_W"]
        unit_cost = costs["diesel_cost"] / costs["diesel_energy"] / eff["gen_eff"] if name == "generator" else costs[f"{name}_vom_cost_perWh"]
        weighted_operating += dispatch.sum() * weight * unit_cost
        average_key = "avg_sol_W" if name == "solar" else f"avg_{name}_W"
        check(f"{name}_average_power", arr(average_key)[-1] - dispatch.mean(), "W")
        check(f"{name}_constant_capacity", arr(f"{name}_scale") - capacity, "W")
    check("electrical_balance", supply - arr("load")[1:] - water_power, "W")
    battery, tank = arr("batt_storagelevel"), arr("potable_water_storagelevel")
    battery_cap, tank_cap = arr("batt_scale")[-1], arr("potable_water_storage_scale")[-1]
    eta = np.sqrt(eff["batt_round_trip_eff"])
    check("battery_state_update", np.diff(battery) - eta * charge + discharge / eta, "Wh")
    check("water_state_update", np.diff(tank) - water_power / eff["potable_water_Wh_per_l"] + arr("potable_water_demand_l")[1:], "L")
    check("battery_cyclic", battery[-1] - battery[0], "Wh")
    check("water_cyclic", tank[-1] - tank[0], "L")
    bounds("battery_inventory", battery, limits["batt_min_storage_level"], battery_cap, "Wh")
    bounds("water_inventory", tank, 0, tank_cap, "L")
    bounds("battery_capacity", battery_cap, limits["batt_min_capacity"], limits["batt_max_capacity"], "Wh")
    bounds("water_capacity", tank_cap, limits["potable_water_min_storage_l"], limits["potable_water_max_storage_l"], "L")
    bounds("battery_charge_rate", charge, 0, limits["batt_max_charge_rate"], "W")
    bounds("battery_discharge_rate", discharge, 0, limits["batt_max_charge_rate"], "W")
    bounds("water_production_rate", water_power / eff["potable_water_Wh_per_l"], 0, limits["potable_water_max_rate_gpm"] * 60 * 3.785411784, "L/h")
    check("simultaneous_battery_charge_discharge", np.minimum(charge, discharge), "W")
    summary = values["paper_summary"]
    installed, trips = summary["generator_installed"], summary["delivery_trips"]
    bounds("delivery_trip_count", trips, installed, paper["max_delivery_trips"] * installed, "trips")
    bounds("generator_installation", arr("generator_scale")[-1], paper["generator_min_installed_w"] * installed, limits["generator_max_capacity"] * installed, "W")
    bounds("fuel_delivery_budget", arr("generator_power")[1:].sum(), 0, trips * costs["tank_size"] * costs["diesel_energy"] * eff["gen_eff"], "Wh")
    expected_fixed += battery_cap * costs["batt_fix_cost"] + tank_cap * costs["potable_water_storage_fix_cost_per_l"]
    supported_mass += battery_cap * mass["batt_Kg_per_Wh"]
    platform_mass = supported_mass * mass["floating_platform_mass_per_supported_mass"]
    tank_mass = tank_cap * mass["potable_water_storage_Kg_per_l"]
    check("supported_mass_accounting", arr("supported_mass")[-1] - supported_mass, "kg")
    check("platform_mass_accounting", arr("floating_platform_mass")[-1] - platform_mass, "kg")
    check("tank_mass_accounting", arr("potable_water_storage_mass")[-1] - tank_mass, "kg")
    check("total_mass_accounting", arr("total_mass")[-1] - supported_mass - platform_mass - tank_mass, "kg")
    check("average_electrical_demand", arr("avg_load_W")[-1] - np.mean(arr("load")[1:] + water_power), "W")
    check("average_battery_throughput", arr("avg_batt_throughput_W")[-1] - np.mean(charge + discharge), "W")
    check("battery_constant_capacity", arr("batt_scale") - battery_cap, "Wh")
    check("water_constant_capacity", arr("potable_water_storage_scale") - tank_cap, "L")
    check("battery_bus_dispatch", arr("batt_power_inout")[1:] - discharge + charge, "W")
    support_cost = supported_mass * mass["floating_platform_mass_per_supported_mass"] * costs["floating_platform_cost_per_kg"]
    weighted_operating += trips * paper["delivery_repeat_factor"] * costs["refill_service_cost"] + weight * (charge + discharge).sum() * costs["batt_vom_cost_perWh"]
    check("total_cost_accounting", arr("total_cost")[-1] - expected_fixed - support_cost - weighted_operating, "USD")
    bounds("primary_cost_band", summary["total_cost_usd"], summary["primary_cost_usd"] - paper["primary_cost_tolerance_usd"], summary["primary_cost_usd"] + paper["primary_cost_tolerance_usd"], "USD")
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks,
            "duration_hours": len(charge), "solved_intervals": len(charge),
            "max_power_balance_W": checks["electrical_balance"]["max_residual"]}
