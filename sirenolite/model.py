import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from matplotlib.figure import Figure
from gekko import GEKKO  # registered package, i.e. conda install gekko


def default_inputs():
    """Centralize inputs to make tuning and reuse easier."""
    costs = {
        "generator_fix_cost": 0.4,  # $/W based on a 5000 watt generator costing $2k
        "wind_fix_cost": 4.0,  # $/W based on a 1kw nameplate airdolphin costing $4k
        "solar_fix_cost": 1.67,  # $/W based on a 3kw solar system costing $5k
        "wave_fix_cost": 2.0,  # $/W based on a CorPower 300kw system costing $600k (scaled down)
        "batt_fix_cost": 0.16,  # $/Wh based on a 100Ah 12v AGM battery costing $190
        "h2_storage_fix_cost_per_g": 1.0,  # $/g compressed H2 storage capacity
        "potable_water_storage_fix_cost_per_l": 0.5,  # $/L potable water storage capacity
        "floating_platform_cost_per_kg": 30.0,  # $/kg small buoy/platform structure cost
        "tank_size": 50.0,  # gallons
        "diesel_cost": 4.5,  # $/gal
        "diesel_energy": 44000,  # Wh/gal
        "refill_service_cost": 3000.0,  # $
        "generator_vom_cost_perWh": None,  # computed below
        "batt_vom_cost_perWh": 0.0,  # $/Wh
        "wind_vom_cost_perWh": 0.001,  # $/Wh
        "solar_vom_cost_perWh": 0.0004,  # $/Wh
        "wave_vom_cost_perWh": 0.001,  # $/Wh
    }

    limits = {
        "generator_max_capacity": 100000.0,  # W
        "wind_max_capacity": 100000.0,  # W
        "solar_max_capacity": 100000.0,  # W
        "wave_max_capacity": 100000.0,  # W
        "batt_max_capacity": 1440000,  # Wh
        "h2_min_storage_g": 0.0,  # g
        "h2_max_storage_g": 100000.0,  # g
        "potable_water_min_storage_l": 0.0,  # L
        "potable_water_max_storage_l": 100000.0,  # L
        "generator_min_capacity": 0.0,  # W
        "wind_min_capacity": 0.0,  # W
        "solar_min_capacity": 0.0,  # W
        "wave_min_capacity": 0.0,  # W
        "batt_min_capacity": 0.0,  # Wh
        "batt_min_storage_level": 0.0,  # Wh
        "batt_max_ramp_up": 500000,  # change in watts per hr (timestep)
    }

    efficiency = {
        "gen_eff": 0.2,  # 20% efficiency converting diesel energy to electricity
        "batt_round_trip_eff": 0.9,
        "batt_final_charge": 0.98,  # factor, i.e. 0.98 is 98%
        "h2_Wh_per_g": 65.0,  # Wh per gram of H2 produced
        "potable_water_Wh_per_l": 0.0045,  # Wh per liter of potable water produced
    }

    mass = {
        "generator_Kg_per_W": 0.03,
        "batt_Kg_per_Wh": 8.0,
        "wind_Kg_per_W": 0.03,
        "solar_Kg_per_W": 0.07,
        "wave_Kg_per_W": 0.08,
        "h2_storage_Kg_per_g": 0.01,
        "potable_water_storage_Kg_per_l": 0.05,
        "floating_platform_mass_per_supported_mass": 0.3,
    }

    simulation = {
        "lifespan": 5.0 * 365.0 * 24.0,  # hrs
        "nhrs": 400,
        "peak_load": 1000.0,  # W
        "H2DailyDemand": 400.0, # hydrogen grams
        "H2ODailyDemand": 265.0, # potable water, liters
        "data_file": str(Path(__file__).resolve().parent / "data" / "load_resource_data.csv"),
        "doldrum_time": 24,
        "doldrums": {"wind": 0.4, "solar": 0.5, "wave": 0.6},
    }

    solver = {
        "remote": True,  # set False to solve locally
        "imode": 6,
        "max_iter": 1000,
        "solver": 3,
        "cv_type": 1,
    }

    config = {
        "costs": costs,
        "limits": limits,
        "efficiency": efficiency,
        "mass": mass,
        "simulation": simulation,
        "solver": solver,
        "objective": "total_mass",
    }
    update_derived_config(config)
    return config


def update_derived_config(config):
    costs = config["costs"]
    eff = config["efficiency"]

    gen_cost_to_refill = costs["refill_service_cost"] / (
        costs["tank_size"] * costs["diesel_energy"]
    )
    costs["generator_vom_cost_perWh"] = (
        costs["diesel_cost"] / costs["diesel_energy"] / eff["gen_eff"]
        + gen_cost_to_refill
    )


def build_doldrum_mask(nhrs, center_frac, half_window):
    mask = np.ones(nhrs)
    center = int(nhrs * center_frac)
    mask[int(center - half_window) : int(center + half_window)] = 0
    return mask


def load_resource_inputs(config):
    sim = config["simulation"]
    nhrs = sim["nhrs"]
    peak_load = sim["peak_load"]
    H2DailyDemand = sim["H2DailyDemand"]
    H2ODailyDemand = sim["H2ODailyDemand"]
    half_window = sim["doldrum_time"]
    doldrums = sim["doldrums"]

    data = pd.read_csv(sim["data_file"])
    data_len = len(data)
    time_values = data["Time"].values if "Time" in data.columns else None
    if time_values is None:
        warnings.warn("CSV is missing a Time column; assuming 1-hour increments.")
    elif len(time_values) > 1:
        diffs = np.diff(time_values.astype(float))
        if not np.allclose(diffs, 1.0, rtol=1e-3, atol=1e-3):
            warnings.warn(
                "Time column must be in 1-hour increments; update the CSV Time column "
                "or resample data to 1-hour steps."
            )

    if data_len < nhrs:
        warnings.warn(
            f"Resource data length ({data_len}) is shorter than horizon ({nhrs}); "
            "looping data to fill the horizon."
        )
        repeats = int(np.ceil(nhrs / data_len))
        def extend(values):
            return np.tile(values, repeats)[:nhrs]
    else:
        def extend(values):
            return values[:nhrs]

    doldrums_wind = build_doldrum_mask(nhrs, doldrums["wind"], half_window)
    doldrums_solar = build_doldrum_mask(nhrs, doldrums["solar"], half_window)
    doldrums_wave = build_doldrum_mask(nhrs, doldrums["wave"], half_window)

    wind_watts = extend(data["Wind"].values) * doldrums_wind
    solar_watts = extend(data["Solar"].values) * doldrums_solar
    if "Wave" not in data.columns:
        raise ValueError("CSV is missing required Wave column.")
    wave_watts = extend(data["Wave"].values) * doldrums_wave
    load_watts = extend(data["Load"].values)
    hydrogen_g = extend(data["Hydrogen"].values)
    potableWater_l = extend(data["PotableWater"].values)  # liters then scaled to one

    def normalize_profile(values):
        max_val = float(np.max(values))
        if max_val <= 0:
            return np.zeros_like(values, dtype=float)
        return values / max_val

    return {
        "Wind_unitfactor": normalize_profile(wind_watts),
        "Solar_unitfactor": normalize_profile(solar_watts),
        "Wave_unitfactor": normalize_profile(wave_watts),
        "Load": normalize_profile(load_watts) * peak_load,
        "Hydrogen_demand_g": normalize_profile(hydrogen_g) * H2DailyDemand,
        "PotableWater_demand_l": normalize_profile(potableWater_l) * H2ODailyDemand,
    }


def to_array(values):
    return np.array(list(values), dtype=float)


def scale_for_plot(series, reference_max, enable_scaling=True, order_of_magnitude=False):
    if not enable_scaling:
        return series, 1.0
    max_series = float(np.max(np.abs(series)))
    if max_series <= 0 or reference_max <= 0:
        return series, 1.0
    scale = reference_max / max_series
    if order_of_magnitude:
        scale = 10 ** int(np.round(np.log10(scale)))
    return series * scale, scale


def format_label(name, unit, scale):
    if abs(scale - 1.0) < 0.05:
        return f"{name} ({unit})"
    exponent = int(np.floor(np.log10(abs(scale))))
    mantissa = scale / (10 ** exponent)
    if abs(mantissa - 1.0) < 0.05:
        scale_text = f"1e{exponent}"
    else:
        scale_text = f"{mantissa:.1f}e{exponent}"
    return f"{name} ({unit}, x{scale_text})"


def build_timeseries_dataframe(config, model_vars):
    m_vars = model_vars
    eff = config["efficiency"]
    time_hr = np.array(m_vars["model"].time)
    time_days = time_hr / 24.0

    load = to_array(m_vars["load"].value)
    hydrogen_demand_g = to_array(m_vars["hydrogen_demand_g"].value)
    potable_water_demand_l = to_array(m_vars["potable_water_demand_l"].value)
    batt_power_inout = to_array(m_vars["batt_power_inout"].value)
    batt_storage = to_array(m_vars["batt_storagelevel"].value)
    h2_power_in = to_array(m_vars["h2_power_in"].value)
    h2_storage_g = to_array(m_vars["h2_storagelevel"].value)
    potable_power_in = to_array(m_vars["potable_water_power_in"].value)
    potable_storage_l = to_array(m_vars["potable_water_storagelevel"].value)

    generator_unitfactor = to_array(m_vars["generator_unitfactor"].value)
    generator_scale = to_array(m_vars["generator_scale"].value)
    generator_power = generator_unitfactor * generator_scale

    wind_unitfactor = to_array(m_vars["wind_unitfactor"].value)
    wind_cur = to_array(m_vars["wind_cur"].value)
    wind_scale = to_array(m_vars["wind_scale"].value)
    wind_power = wind_unitfactor * wind_cur * wind_scale

    solar_unitfactor = to_array(m_vars["solar_unitfactor"].value)
    solar_cur = to_array(m_vars["solar_cur"].value)
    solar_scale = to_array(m_vars["solar_scale"].value)
    solar_power = solar_unitfactor * solar_cur * solar_scale

    wave_unitfactor = to_array(m_vars["wave_unitfactor"].value)
    wave_cur = to_array(m_vars["wave_cur"].value)
    wave_scale = to_array(m_vars["wave_scale"].value)
    wave_power = wave_unitfactor * wave_cur * wave_scale

    electric_demand = load + h2_power_in + potable_power_in
    total_power = batt_power_inout + generator_power + wind_power + solar_power + wave_power
    power_balance = total_power - electric_demand

    data = {
        "time_hr": time_hr,
        "time_days": time_days,
        "load_W": load,
        "hydrogen_demand_g": hydrogen_demand_g,
        "potable_water_demand_l": potable_water_demand_l,
        "electric_demand_W": electric_demand,
        "total_power_W": total_power,
        "power_balance_W": power_balance,
        "battery_power_W": batt_power_inout,
        "battery_storage_Wh": batt_storage,
        "h2_power_W": h2_power_in,
        "h2_storage_g": h2_storage_g,
        "h2_storage_Wh_equiv": h2_storage_g * eff["h2_Wh_per_g"],
        "potable_power_W": potable_power_in,
        "potable_storage_l": potable_storage_l,
        "potable_storage_Wh_equiv": potable_storage_l * eff["potable_water_Wh_per_l"],
        "generator_unitfactor": generator_unitfactor,
        "generator_scale_W": generator_scale,
        "generator_power_W": generator_power,
        "wind_unitfactor": wind_unitfactor,
        "wind_cur": wind_cur,
        "wind_scale_W": wind_scale,
        "wind_power_W": wind_power,
        "solar_unitfactor": solar_unitfactor,
        "solar_cur": solar_cur,
        "solar_scale_W": solar_scale,
        "solar_power_W": solar_power,
        "wave_unitfactor": wave_unitfactor,
        "wave_cur": wave_cur,
        "wave_scale_W": wave_scale,
        "wave_power_W": wave_power,
    }
    return pd.DataFrame(data)


def build_model(config, inputs):
    sim = config["simulation"]
    costs = config["costs"]
    limits = config["limits"]
    eff = config["efficiency"]
    mass = config["mass"]

    m = GEKKO(remote=config["solver"]["remote"])
    t = np.linspace(0, sim["nhrs"] - 1, sim["nhrs"])
    m.time = t

    load = m.Param(value=inputs["Load"])
    Hydrogen_demand_g = m.Param(value=inputs["Hydrogen_demand_g"])
    PotableWater_demand_l = m.Param(value=inputs["PotableWater_demand_l"])
    wind_unitfactor = m.Param(value=inputs["Wind_unitfactor"])
    solar_unitfactor = m.Param(value=inputs["Solar_unitfactor"])
    wave_unitfactor = m.Param(value=inputs["Wave_unitfactor"])

    h2_power_max = max(
        sim["peak_load"] * 10,
        float(np.max(inputs["Hydrogen_demand_g"])) * eff["h2_Wh_per_g"] * 2,
    )
    potable_water_power_max = max(
        sim["peak_load"] * 10,
        float(np.max(inputs["PotableWater_demand_l"])) * eff["potable_water_Wh_per_l"] * 2,
    )

    batt_power_inout = m.MV(value=0, lb=-sim["peak_load"] * 10, ub=sim["peak_load"] * 10)
    batt_power_inout.STATUS = 1
    h2_power_in = m.MV(value=0, lb=0, ub=h2_power_max)
    h2_power_in.STATUS = 1
    potable_water_power_in = m.MV(value=0, lb=0, ub=potable_water_power_max)
    potable_water_power_in.STATUS = 1

    wind_cur = m.MV(value=1.0, lb=0, ub=1, fixed_initial=False)
    wind_cur.STATUS = 1
    solar_cur = m.MV(value=1.0, lb=0, ub=1, fixed_initial=False)
    solar_cur.STATUS = 1
    wave_cur = m.MV(value=1.0, lb=0, ub=1, fixed_initial=False)
    wave_cur.STATUS = 1

    wind_scale = m.FV(
        value=(limits["wind_max_capacity"] + limits["wind_min_capacity"]) / 2,
        lb=limits["wind_min_capacity"],
        ub=limits["wind_max_capacity"],
    )
    wind_scale.STATUS = 1
    solar_scale = m.FV(
        value=(limits["solar_max_capacity"] + limits["solar_min_capacity"]) / 2,
        lb=limits["solar_min_capacity"],
        ub=limits["solar_max_capacity"],
    )
    solar_scale.STATUS = 1
    wave_scale = m.FV(
        value=(limits["wave_max_capacity"] + limits["wave_min_capacity"]) / 2,
        lb=limits["wave_min_capacity"],
        ub=limits["wave_max_capacity"],
    )
    wave_scale.STATUS = 1
    generator_scale = m.FV(
        value=(limits["generator_max_capacity"] + limits["generator_min_capacity"]) / 2,
        lb=limits["generator_min_capacity"],
        ub=limits["generator_max_capacity"],
    )
    generator_scale.STATUS = 1

    batt_scale = m.FV(value=1440, lb=limits["batt_min_capacity"], ub=limits["batt_max_capacity"])
    batt_scale.STATUS = 1

    h2_storage_scale = m.FV(
        value=limits["h2_max_storage_g"],
        lb=limits["h2_min_storage_g"],
        ub=limits["h2_max_storage_g"],
    )
    h2_storage_scale.STATUS = 1
    potable_water_storage_scale = m.FV(
        value=limits["potable_water_max_storage_l"],
        lb=limits["potable_water_min_storage_l"],
        ub=limits["potable_water_max_storage_l"],
    )
    potable_water_storage_scale.STATUS = 1

    generator_unitfactor = m.MV(value=0.5, lb=0, ub=1, fixed_initial=False)
    generator_unitfactor.STATUS = 1

    final = np.zeros(sim["nhrs"])
    final[-1] = 1
    final[-2] = 1
    final_vector = m.Param(final)

    batt_storagelevel = m.Var(
        value=limits["batt_min_storage_level"],
        lb=limits["batt_min_storage_level"],
        ub=limits["batt_max_capacity"],
    )
    h2_storagelevel = m.Var(
        value=limits["h2_min_storage_g"],
        lb=limits["h2_min_storage_g"],
        ub=limits["h2_max_storage_g"],
    )
    potable_water_storagelevel = m.Var(
        value=limits["potable_water_min_storage_l"],
        lb=limits["potable_water_min_storage_l"],
        ub=limits["potable_water_max_storage_l"],
    )

    batt_eff = m.Const(eff["batt_round_trip_eff"])

    m.Equation(batt_storagelevel.dt() == -batt_power_inout * batt_eff)
    m.Equation(batt_storagelevel * final_vector == batt_scale * eff["batt_final_charge"] * final_vector)
    m.Equation(batt_storagelevel <= batt_scale)
    m.Equation(h2_storagelevel.dt() == h2_power_in / eff["h2_Wh_per_g"] - Hydrogen_demand_g)
    m.Equation(h2_storagelevel <= h2_storage_scale)
    m.Equation(
        potable_water_storagelevel.dt()
        == potable_water_power_in / eff["potable_water_Wh_per_l"] - PotableWater_demand_l
    )
    m.Equation(potable_water_storagelevel <= potable_water_storage_scale)

    # m.Equation(batt_in.dt() <= limits["batt_max_ramp_up"])

    m.Equation(
        batt_power_inout
        + generator_unitfactor * generator_scale
        + wind_unitfactor * wind_cur * wind_scale
        + solar_unitfactor * solar_cur * solar_scale
        + wave_unitfactor * wave_cur * wave_scale
        == load + h2_power_in + potable_water_power_in
    )

    sum_load = m.Var(value=10000)
    electric_demand = m.Intermediate(load + h2_power_in + potable_water_power_in)
    m.Equation(sum_load == m.integral(electric_demand))

    avg_load_W = m.Intermediate(sum_load / sim["nhrs"])
    avg_sol_W = m.Intermediate(m.integral(solar_unitfactor * solar_cur * solar_scale) / sim["nhrs"])
    avg_wave_W = m.Intermediate(m.integral(wave_unitfactor * wave_cur * wave_scale) / sim["nhrs"])
    avg_wind_W = m.Intermediate(m.integral(wind_unitfactor * wind_cur * wind_scale) / sim["nhrs"])
    avg_generator_W = m.Intermediate(m.integral(generator_unitfactor * generator_scale) / sim["nhrs"])

    supported_mass = m.Intermediate(
        mass["generator_Kg_per_W"] * generator_scale
        + mass["batt_Kg_per_Wh"] * batt_scale
        + mass["wind_Kg_per_W"] * wind_scale
        + mass["solar_Kg_per_W"] * solar_scale
        + mass["wave_Kg_per_W"] * wave_scale
        + mass["h2_storage_Kg_per_g"] * h2_storage_scale
        + mass["potable_water_storage_Kg_per_l"] * potable_water_storage_scale
    )
    floating_platform_mass = m.Intermediate(
        supported_mass * mass["floating_platform_mass_per_supported_mass"]
    )

    total_cost = m.Intermediate(
        final_vector
        * (
            costs["generator_fix_cost"] * generator_scale
            + costs["batt_fix_cost"] * batt_scale
            + costs["wind_fix_cost"] * wind_scale
            + costs["solar_fix_cost"] * solar_scale
            + costs["wave_fix_cost"] * wave_scale
            + costs["h2_storage_fix_cost_per_g"] * h2_storage_scale
            + costs["potable_water_storage_fix_cost_per_l"] * potable_water_storage_scale
            + costs["floating_platform_cost_per_kg"] * floating_platform_mass
            + costs["generator_vom_cost_perWh"] * avg_generator_W * sim["lifespan"]
            + costs["batt_vom_cost_perWh"] * sim["lifespan"]
            + costs["wind_vom_cost_perWh"] * avg_wind_W * sim["lifespan"]
            + costs["solar_vom_cost_perWh"] * avg_sol_W * sim["lifespan"]
            + costs["wave_vom_cost_perWh"] * avg_wave_W * sim["lifespan"]
        )
    )
    cost_per_watt = m.Intermediate(total_cost / avg_load_W)

    total_mass = m.Intermediate(final_vector * (supported_mass + floating_platform_mass))

    objective = config.get("objective", "total_mass")
    if objective == "cost_per_watt":
        m.Obj(cost_per_watt)
    else:
        m.Obj(total_mass)

    return m, {
        "load": load,
        "hydrogen_demand_g": Hydrogen_demand_g,
        "potable_water_demand_l": PotableWater_demand_l,
        "wind_unitfactor": wind_unitfactor,
        "solar_unitfactor": solar_unitfactor,
        "wave_unitfactor": wave_unitfactor,
        "batt_power_inout": batt_power_inout,
        "h2_power_in": h2_power_in,
        "potable_water_power_in": potable_water_power_in,
        "wind_cur": wind_cur,
        "solar_cur": solar_cur,
        "wave_cur": wave_cur,
        "wind_scale": wind_scale,
        "solar_scale": solar_scale,
        "wave_scale": wave_scale,
        "generator_scale": generator_scale,
        "batt_scale": batt_scale,
        "h2_storage_scale": h2_storage_scale,
        "potable_water_storage_scale": potable_water_storage_scale,
        "generator_unitfactor": generator_unitfactor,
        "final_vector": final_vector,
        "batt_storagelevel": batt_storagelevel,
        "h2_storagelevel": h2_storagelevel,
        "potable_water_storagelevel": potable_water_storagelevel,
        "avg_generator_W": avg_generator_W,
        "avg_wind_W": avg_wind_W,
        "avg_sol_W": avg_sol_W,
        "avg_wave_W": avg_wave_W,
        "supported_mass": supported_mass,
        "floating_platform_mass": floating_platform_mass,
        "total_cost": total_cost,
        "cost_per_watt": cost_per_watt,
        "total_mass": total_mass,
        "model": m,
    }


def solve_model(config, model):
    m, _ = model
    solver = config["solver"]
    m.options.IMODE = solver["imode"]
    m.options.MAX_ITER = solver["max_iter"]
    m.options.SOLVER = solver["solver"]
    m.options.CV_TYPE = solver["cv_type"]
    m.solve()


def report_results(config, model_vars):
    m_vars = model_vars
    eff = config["efficiency"]
    print("System CostPerWatt ($/W):", m_vars["cost_per_watt"].value[-1])
    print("Total Cost ($):", m_vars["total_cost"].value[-1])
    print("Total Mass (kg):", m_vars["total_mass"].value[-1])
    print("Supported Mass (kg):", m_vars["supported_mass"].value[-1])
    print("Floating Platform Mass (kg):", m_vars["floating_platform_mass"].value[-1])
    print(
        "Floating Platform Cost ($):",
        m_vars["floating_platform_mass"].value[-1] * config["costs"]["floating_platform_cost_per_kg"],
    )
    print("Battery Size (Wh):", max(m_vars["batt_storagelevel"].value))
    print("Hydrogen Storage Max (g):", max(m_vars["h2_storagelevel"].value))
    print("Potable Water Storage Max (L):", max(m_vars["potable_water_storagelevel"].value))
    print("Wind Size (W):", m_vars["wind_scale"].value[-1])
    print("Solar Size (W):", m_vars["solar_scale"].value[-1])
    print("Wave Size (W):", m_vars["wave_scale"].value[-1])
    print("Generator Size (W):", m_vars["generator_scale"].value[-1])
    print("avg_generator_W (W):", m_vars["avg_generator_W"].value[-1])
    print("avg_wind_W (W):", m_vars["avg_wind_W"].value[-1])
    print("avg_sol_W (W):", m_vars["avg_sol_W"].value[-1])
    print("avg_wave_W (W):", m_vars["avg_wave_W"].value[-1])

    width = 10
    m = m_vars["model"]
    nhrs = config["simulation"]["nhrs"]
    load = to_array(m_vars["load"].value)
    hydrogen_demand_g = to_array(m_vars["hydrogen_demand_g"].value)
    potable_water_demand_l = to_array(m_vars["potable_water_demand_l"].value)

    gen_production = to_array(m_vars["generator_unitfactor"].value) * to_array(m_vars["generator_scale"].value)
    wind_production = (
        to_array(m_vars["wind_unitfactor"].value)
        * to_array(m_vars["wind_cur"].value)
        * to_array(m_vars["wind_scale"].value)
    )
    solar_production = (
        to_array(m_vars["solar_unitfactor"].value)
        * to_array(m_vars["solar_cur"].value)
        * to_array(m_vars["solar_scale"].value)
    )
    wave_production = (
        to_array(m_vars["wave_unitfactor"].value)
        * to_array(m_vars["wave_cur"].value)
        * to_array(m_vars["wave_scale"].value)
    )
    battery_production = to_array(m_vars["batt_power_inout"].value)
    h2_power_in = to_array(m_vars["h2_power_in"].value)
    potable_water_power_in = to_array(m_vars["potable_water_power_in"].value)
    electric_demand = load + h2_power_in + potable_water_power_in
    total_power = (
        battery_production + gen_production + wind_production + solar_production + wave_production
    )

    power_ref_max = float(np.max(np.abs(electric_demand)))
    demand_ref_max = max(
        float(np.max(np.abs(hydrogen_demand_g))), float(np.max(np.abs(potable_water_demand_l))), 1.0
    )

    plt.figure(figsize=(width, 1.9))
    plt.subplots_adjust(left=0.1, bottom=0.25, top=0.9, right=0.8)
    ax_balance = plt.gca()
    load_plot, load_scale = scale_for_plot(load, power_ref_max, enable_scaling=False)
    h2_plot, h2_scale = scale_for_plot(h2_power_in, power_ref_max, enable_scaling=False)
    water_plot, water_scale = scale_for_plot(potable_water_power_in, power_ref_max, enable_scaling=False)
    demand_plot, demand_scale = scale_for_plot(electric_demand, power_ref_max, enable_scaling=False)
    total_power_plot, total_power_scale = scale_for_plot(total_power, power_ref_max, enable_scaling=False)
    ax_balance.plot(
        m.time[2:] / 24.0,
        load_plot[2:],
        "-",
        label=format_label("Load", "W", load_scale),
    )
    ax_balance.plot(
        m.time[2:] / 24.0,
        h2_plot[2:],
        "-.",
        label=format_label("H2 Prod", "W", h2_scale),
    )
    ax_balance.plot(
        m.time[2:] / 24.0,
        water_plot[2:],
        ":",
        label=format_label("Water Desal", "W", water_scale),
    )
    ax_balance.plot(
        m.time[2:] / 24.0,
        demand_plot[2:],
        "-",
        label=format_label("Total Demand", "W", demand_scale),
    )
    ax_balance.plot(
        m.time[2:] / 24.0,
        total_power_plot[2:],
        "--",
        label=format_label("Total Power", "W", total_power_scale),
    )
    ax_balance_right = ax_balance.twinx()
    h2_demand_plot, h2_demand_scale = scale_for_plot(
        hydrogen_demand_g, demand_ref_max, enable_scaling=True
    )
    water_demand_plot, water_demand_scale = scale_for_plot(
        potable_water_demand_l, demand_ref_max, enable_scaling=True
    )
    ax_balance_right.plot(
        m.time[2:] / 24.0,
        h2_demand_plot[2:],
        color="purple",
        linestyle="-.",
        label=format_label("H2 Demand", "g", h2_demand_scale),
    )
    ax_balance_right.plot(
        m.time[2:] / 24.0,
        water_demand_plot[2:],
        color="teal",
        linestyle=":",
        label=format_label("Water Demand", "L", water_demand_scale),
    )
    ax_balance_right.set_ylabel("Demand (scaled)")
    plt.ylabel("Power Balance (W)")
    plt.xlim(0, nhrs / 24.0)
    handles_left, labels_left = ax_balance.get_legend_handles_labels()
    handles_right, labels_right = ax_balance_right.get_legend_handles_labels()
    ax_balance.legend(handles_left + handles_right, labels_left + labels_right)
    plt.xlabel("time (Days)")
    plt.savefig(f"solution_{nhrs}_optimal_mix_power_balance_with_wave.pdf")

    plt.figure(figsize=(width, 5))
    plt.subplots_adjust(left=0.1, bottom=0.25, top=0.9, right=0.88)
    wind_avail = to_array(m_vars["wind_unitfactor"].value) * to_array(m_vars["wind_scale"].value)
    solar_avail = to_array(m_vars["solar_unitfactor"].value) * to_array(m_vars["solar_scale"].value)
    wave_avail = to_array(m_vars["wave_unitfactor"].value) * to_array(m_vars["wave_scale"].value)

    wind_avail_plot, wind_avail_scale = scale_for_plot(wind_avail, power_ref_max, enable_scaling=False)
    solar_avail_plot, solar_avail_scale = scale_for_plot(solar_avail, power_ref_max, enable_scaling=False)
    wave_avail_plot, wave_avail_scale = scale_for_plot(wave_avail, power_ref_max, enable_scaling=False)
    wind_used_plot, wind_used_scale = scale_for_plot(wind_production, power_ref_max, enable_scaling=False)
    solar_used_plot, solar_used_scale = scale_for_plot(solar_production, power_ref_max, enable_scaling=False)
    wave_used_plot, wave_used_scale = scale_for_plot(wave_production, power_ref_max, enable_scaling=False)
    battery_plot, battery_scale = scale_for_plot(battery_production, power_ref_max, enable_scaling=False)
    gen_plot, gen_scale = scale_for_plot(gen_production, power_ref_max, enable_scaling=False)
    h2_power_plot, h2_power_scale = scale_for_plot(h2_power_in, power_ref_max, enable_scaling=False)
    potable_power_plot, potable_power_scale = scale_for_plot(
        potable_water_power_in, power_ref_max, enable_scaling=False
    )

    plt.plot(
        m.time / 24.0,
        wind_avail_plot,
        color="forestgreen",
        linestyle="--",
        label=format_label("Wind Avail", "W", wind_avail_scale),
    )
    plt.plot(
        m.time / 24.0,
        solar_avail_plot,
        color="goldenrod",
        linestyle="--",
        label=format_label("Solar Avail", "W", solar_avail_scale),
    )
    plt.plot(
        m.time / 24.0,
        wave_avail_plot,
        color="cornflowerblue",
        linestyle="--",
        label=format_label("Wave Avail", "W", wave_avail_scale),
    )
    plt.plot(
        m.time / 24.0,
        wind_used_plot,
        color="forestgreen",
        linestyle="-",
        label=format_label("Wind Used", "W", wind_used_scale),
    )
    plt.plot(
        m.time / 24.0,
        solar_used_plot,
        color="goldenrod",
        linestyle="-",
        label=format_label("Solar Used", "W", solar_used_scale),
    )
    plt.plot(
        m.time / 24.0,
        wave_used_plot,
        color="cornflowerblue",
        linestyle="-",
        label=format_label("Wave Used", "W", wave_used_scale),
    )
    plt.plot(
        m.time / 24.0,
        battery_plot,
        color="dimgrey",
        linestyle="-",
        label=format_label("Batt Used", "W", battery_scale),
    )
    plt.plot(
        m.time / 24.0,
        gen_plot,
        color="firebrick",
        label=format_label("Gen Used", "W", gen_scale),
    )
    plt.plot(
        m.time / 24.0,
        h2_power_plot,
        color="purple",
        label=format_label("H2 Prod", "W", h2_power_scale),
    )
    plt.plot(
        m.time / 24.0,
        potable_power_plot,
        color="teal",
        label=format_label("Water Desal", "W", potable_power_scale),
    )
    plt.ylabel("Power Production (W)")
    plt.xlim(0, nhrs / 24.0)
    plt.legend(loc=(0.5, 0.5))
    plt.xlabel("time (Days)")
    plt.savefig(f"solution_{nhrs}_optimal_mix_production_with_wave.pdf")

    plt.figure(figsize=(width, 2))
    plt.subplots_adjust(left=0.1, bottom=0.25, top=0.9, right=0.99)
    batt_storage = to_array(m_vars["batt_storagelevel"].value)
    h2_storage_Wh = to_array(m_vars["h2_storagelevel"].value) * eff["h2_Wh_per_g"]
    potable_storage_Wh = (
        to_array(m_vars["potable_water_storagelevel"].value) * eff["potable_water_Wh_per_l"]
    )
    storage_ref_max = float(np.max(np.abs(batt_storage)))
    batt_storage_plot, batt_storage_scale = scale_for_plot(
        batt_storage, storage_ref_max, enable_scaling=False
    )
    h2_storage_plot, h2_storage_scale = scale_for_plot(
        h2_storage_Wh, storage_ref_max, enable_scaling=True, order_of_magnitude=True
    )
    potable_storage_plot, potable_storage_scale = scale_for_plot(
        potable_storage_Wh, storage_ref_max, enable_scaling=True, order_of_magnitude=True
    )
    plt.plot(
        m.time / 24.0,
        batt_storage_plot,
        color="dimgrey",
        label=format_label("Battery Storage", "Wh", batt_storage_scale),
    )
    plt.plot(
        m.time / 24.0,
        h2_storage_plot,
        color="purple",
        label=format_label("H2 Storage", "Wh equiv", h2_storage_scale),
    )
    plt.plot(
        m.time / 24.0,
        potable_storage_plot,
        color="teal",
        label=format_label("Water Storage", "Wh equiv", potable_storage_scale),
    )
    plt.ylabel("Storage (Wh, note scaling)")
    plt.xlim(0, nhrs / 24.0)
    plt.legend()
    plt.xlabel("time (Days)")
    plt.savefig(f"solution_{nhrs}_optimal_mix_batt_storage_with_wave.pdf")
    plt.show()


def create_stacked_figure(config, model_vars, save_pdf=False, figsize=(10, 8)):
    m_vars = model_vars
    eff = config["efficiency"]
    m = m_vars["model"]
    nhrs = config["simulation"]["nhrs"]
    time_days = np.array(m.time) / 24.0
    load = to_array(m_vars["load"].value)
    hydrogen_demand_g = to_array(m_vars["hydrogen_demand_g"].value)
    potable_water_demand_l = to_array(m_vars["potable_water_demand_l"].value)

    gen_production = to_array(m_vars["generator_unitfactor"].value) * to_array(m_vars["generator_scale"].value)
    wind_production = (
        to_array(m_vars["wind_unitfactor"].value)
        * to_array(m_vars["wind_cur"].value)
        * to_array(m_vars["wind_scale"].value)
    )
    solar_production = (
        to_array(m_vars["solar_unitfactor"].value)
        * to_array(m_vars["solar_cur"].value)
        * to_array(m_vars["solar_scale"].value)
    )
    wave_production = (
        to_array(m_vars["wave_unitfactor"].value)
        * to_array(m_vars["wave_cur"].value)
        * to_array(m_vars["wave_scale"].value)
    )
    battery_production = to_array(m_vars["batt_power_inout"].value)
    h2_power_in = to_array(m_vars["h2_power_in"].value)
    potable_water_power_in = to_array(m_vars["potable_water_power_in"].value)
    electric_demand = load + h2_power_in + potable_water_power_in
    total_power = (
        battery_production + gen_production + wind_production + solar_production + wave_production
    )

    power_ref_max = float(np.max(np.abs(electric_demand)))
    demand_ref_max = max(
        float(np.max(np.abs(hydrogen_demand_g))), float(np.max(np.abs(potable_water_demand_l))), 1.0
    )

    fig = Figure(figsize=figsize)
    axes = fig.subplots(3, 1, sharex=True)
    fig.subplots_adjust(left=0.1, bottom=0.1, top=0.97, right=0.88, hspace=0.35)
    ax_balance, ax_prod, ax_storage = axes

    load_plot, load_scale = scale_for_plot(load, power_ref_max, enable_scaling=False)
    h2_plot, h2_scale = scale_for_plot(h2_power_in, power_ref_max, enable_scaling=False)
    water_plot, water_scale = scale_for_plot(potable_water_power_in, power_ref_max, enable_scaling=False)
    demand_plot, demand_scale = scale_for_plot(electric_demand, power_ref_max, enable_scaling=False)
    total_power_plot, total_power_scale = scale_for_plot(total_power, power_ref_max, enable_scaling=False)

    ax_balance.plot(
        time_days[2:],
        load_plot[2:],
        "-",
        label=format_label("Load", "W", load_scale),
    )
    ax_balance.plot(
        time_days[2:],
        h2_plot[2:],
        "-.",
        label=format_label("H2 Prod", "W", h2_scale),
    )
    ax_balance.plot(
        time_days[2:],
        water_plot[2:],
        ":",
        label=format_label("Water Desal", "W", water_scale),
    )
    ax_balance.plot(
        time_days[2:],
        demand_plot[2:],
        "--",
        label=format_label("Total Demand", "W", demand_scale),
    )
    ax_balance.plot(
        time_days[2:],
        total_power_plot[2:],
        "--",
        label=format_label("Total Power", "W", total_power_scale),
    )
    ax_balance_right = ax_balance.twinx()
    h2_demand_plot, h2_demand_scale = scale_for_plot(
        hydrogen_demand_g, demand_ref_max, enable_scaling=True
    )
    water_demand_plot, water_demand_scale = scale_for_plot(
        potable_water_demand_l, demand_ref_max, enable_scaling=True
    )
    ax_balance_right.plot(
        time_days[2:],
        h2_demand_plot[2:],
        color="purple",
        linestyle="-.",
        label=format_label("H2 Demand", "g", h2_demand_scale),
    )
    ax_balance_right.plot(
        time_days[2:],
        water_demand_plot[2:],
        color="teal",
        linestyle=":",
        label=format_label("Water Demand", "L", water_demand_scale),
    )
    ax_balance_right.set_ylabel("Demand (scaled)")
    ax_balance.set_ylabel("Power Balance (W)")
    ax_balance.set_xlim(0, nhrs / 24.0)
    handles_left, labels_left = ax_balance.get_legend_handles_labels()
    handles_right, labels_right = ax_balance_right.get_legend_handles_labels()
    ax_balance.legend(handles_left + handles_right, labels_left + labels_right)

    wind_avail = to_array(m_vars["wind_unitfactor"].value) * to_array(m_vars["wind_scale"].value)
    solar_avail = to_array(m_vars["solar_unitfactor"].value) * to_array(m_vars["solar_scale"].value)
    wave_avail = to_array(m_vars["wave_unitfactor"].value) * to_array(m_vars["wave_scale"].value)

    wind_avail_plot, wind_avail_scale = scale_for_plot(wind_avail, power_ref_max, enable_scaling=False)
    solar_avail_plot, solar_avail_scale = scale_for_plot(solar_avail, power_ref_max, enable_scaling=False)
    wave_avail_plot, wave_avail_scale = scale_for_plot(wave_avail, power_ref_max, enable_scaling=False)
    wind_used_plot, wind_used_scale = scale_for_plot(wind_production, power_ref_max, enable_scaling=False)
    solar_used_plot, solar_used_scale = scale_for_plot(solar_production, power_ref_max, enable_scaling=False)
    wave_used_plot, wave_used_scale = scale_for_plot(wave_production, power_ref_max, enable_scaling=False)
    battery_plot, battery_scale = scale_for_plot(battery_production, power_ref_max, enable_scaling=False)
    gen_plot, gen_scale = scale_for_plot(gen_production, power_ref_max, enable_scaling=False)
    h2_power_plot, h2_power_scale = scale_for_plot(h2_power_in, power_ref_max, enable_scaling=False)
    potable_power_plot, potable_power_scale = scale_for_plot(
        potable_water_power_in, power_ref_max, enable_scaling=False
    )

    ax_prod.plot(
        time_days,
        wind_avail_plot,
        color="forestgreen",
        linestyle="--",
        label=format_label("Wind Avail", "W", wind_avail_scale),
    )
    ax_prod.plot(
        time_days,
        solar_avail_plot,
        color="goldenrod",
        linestyle="--",
        label=format_label("Solar Avail", "W", solar_avail_scale),
    )
    ax_prod.plot(
        time_days,
        wave_avail_plot,
        color="cornflowerblue",
        linestyle="--",
        label=format_label("Wave Avail", "W", wave_avail_scale),
    )
    ax_prod.plot(
        time_days,
        wind_used_plot,
        color="forestgreen",
        linestyle="-",
        label=format_label("Wind Used", "W", wind_used_scale),
    )
    ax_prod.plot(
        time_days,
        solar_used_plot,
        color="goldenrod",
        linestyle="-",
        label=format_label("Solar Used", "W", solar_used_scale),
    )
    ax_prod.plot(
        time_days,
        wave_used_plot,
        color="cornflowerblue",
        linestyle="-",
        label=format_label("Wave Used", "W", wave_used_scale),
    )
    ax_prod.plot(
        time_days,
        battery_plot,
        color="dimgrey",
        linestyle="-",
        label=format_label("Batt Used", "W", battery_scale),
    )
    ax_prod.plot(
        time_days,
        gen_plot,
        color="firebrick",
        label=format_label("Gen Used", "W", gen_scale),
    )
    ax_prod.plot(
        time_days,
        h2_power_plot,
        color="purple",
        label=format_label("H2 Prod", "W", h2_power_scale),
    )
    ax_prod.plot(
        time_days,
        potable_power_plot,
        color="teal",
        label=format_label("Water Desal", "W", potable_power_scale),
    )
    ax_prod.set_ylabel("Power Production (W)")
    ax_prod.legend(loc=(0.5, 0.5))

    batt_storage = to_array(m_vars["batt_storagelevel"].value)
    h2_storage_Wh = to_array(m_vars["h2_storagelevel"].value) * eff["h2_Wh_per_g"]
    potable_storage_Wh = (
        to_array(m_vars["potable_water_storagelevel"].value) * eff["potable_water_Wh_per_l"]
    )
    storage_ref_max = float(np.max(np.abs(batt_storage)))
    batt_storage_plot, batt_storage_scale = scale_for_plot(
        batt_storage, storage_ref_max, enable_scaling=False
    )
    h2_storage_plot, h2_storage_scale = scale_for_plot(
        h2_storage_Wh, storage_ref_max, enable_scaling=True, order_of_magnitude=True
    )
    potable_storage_plot, potable_storage_scale = scale_for_plot(
        potable_storage_Wh, storage_ref_max, enable_scaling=True, order_of_magnitude=True
    )
    ax_storage.plot(
        time_days,
        batt_storage_plot,
        color="dimgrey",
        label=format_label("Battery Storage", "Wh", batt_storage_scale),
    )
    ax_storage.plot(
        time_days,
        h2_storage_plot,
        color="purple",
        label=format_label("H2 Storage", "Wh equiv", h2_storage_scale),
    )
    ax_storage.plot(
        time_days,
        potable_storage_plot,
        color="teal",
        label=format_label("Water Storage", "Wh equiv", potable_storage_scale),
    )
    ax_storage.set_ylabel("Storage (Wh, note scaling)")
    ax_storage.set_xlabel("time (Days)")
    ax_storage.legend()

    if save_pdf:
        fig.savefig(f"solution_{nhrs}_optimal_mix_plots_with_wave.pdf")
    return fig


def main(config=None):
    if config is None:
        config = default_inputs()
    else:
        update_derived_config(config)
    inputs = load_resource_inputs(config)
    model = build_model(config, inputs)
    _, vars_ = model
    solve_model(config, model)
    report_results(config, vars_)


if __name__ == "__main__":
    main()
