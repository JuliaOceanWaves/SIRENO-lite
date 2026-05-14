import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from gekko import GEKKO  # registered package, i.e. conda install gekko

LITERS_PER_GALLON = 3.785411784
SUPPORTED_OBJECTIVES = ("total_mass", "cost_per_watt", "total_cost", "cost_per_mass")
MIN_OBJECTIVE_DENOMINATOR = 1.0e-6
DEFAULT_BATT_MAX_CHARGE_RATE = 500000.0
DEFAULT_H2_MAX_CHARGE_RATE_GPH = 10000.0
DEFAULT_POTABLE_WATER_MAX_RATE_GPM = 10.0
COLUMN_ALIASES = {
    "Time": ("Time", "Time_h"),
    "Load": ("Load", "Load_W_shape", "Load_W"),
    "Solar": ("Solar", "Solar_W_shape", "Solar_profile"),
    "Wind": ("Wind", "Wind_W_shape", "Wind_profile"),
    "Wave": ("Wave", "Wave_W_shape", "Wave_profile"),
    "Hydrogen": ("Hydrogen", "Hydrogen_g_per_h_shape", "Hydrogen_g_per_h"),
    "PotableWater": ("PotableWater", "PotableWater_L_per_h_shape", "PotableWater_L_per_h"),
}
REQUIRED_RESOURCE_COLUMNS = ("Load", "Solar", "Wind", "Wave", "Hydrogen", "PotableWater")


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
        "batt_max_charge_rate": DEFAULT_BATT_MAX_CHARGE_RATE,  # W charge/discharge rate
        "h2_max_charge_rate_gph": DEFAULT_H2_MAX_CHARGE_RATE_GPH,  # g/hr
        "potable_water_max_rate_gpm": DEFAULT_POTABLE_WATER_MAX_RATE_GPM,  # gal/min
    }

    efficiency = {
        "gen_eff": 0.2,  # 20% efficiency converting diesel energy to electricity
        "batt_round_trip_eff": 0.9,
        "batt_initial_charge": 0.0,  # initial battery SOC fraction
        "batt_final_charge": 0.98,  # factor, i.e. 0.98 is 98%
        "h2_initial_charge": 0.0,  # initial hydrogen storage fraction
        "h2_Wh_per_g": 65.0,  # Wh per gram of H2 produced
        "potable_water_initial_charge": 0.0,  # initial potable-water storage fraction
        "potable_water_Wh_per_l": 4.5,  # Wh/L, roughly 4.5 kWh per cubic meter
    }

    mass = {
        "generator_Kg_per_W": 0.03,
        # Battery mass density. Expressed in kg/Wh to match the rest of the model.
        "batt_Kg_per_Wh": 0.008,
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
        "hotel_load": 0.0,  # W fixed parasitic/base load
        "H2DailyDemand": 400.0, # hydrogen grams
        "H2ODailyDemand": 265.0, # potable water, liters
        "data_file": str(Path(__file__).resolve().parent / "data" / "load_resource_data.csv"),
        "profile_start_offset_hours": 0,
        "doldrum_time": 24,
        "doldrums": {"wind": 0.4, "solar": 0.5, "wave": 0.6},
        "doldrum_windows": [],
    }

    solver = {
        "remote": True,  # set False to solve locally
        "imode": 6,
        "max_iter": 1000,
        "solver": 3,
        "cv_type": 1,
        "linear": False,
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


def migrate_legacy_limit_keys(config, fallback_efficiency=None):
    limits = config.setdefault("limits", {})
    eff = config.setdefault("efficiency", {})
    fallback_efficiency = fallback_efficiency or {}

    if "batt_max_charge_rate" not in limits and "batt_max_ramp_up" in limits:
        limits["batt_max_charge_rate"] = limits["batt_max_ramp_up"]
    limits.pop("batt_max_ramp_up", None)
    limits.setdefault("batt_max_charge_rate", DEFAULT_BATT_MAX_CHARGE_RATE)
    limits.setdefault("h2_max_charge_rate_gph", DEFAULT_H2_MAX_CHARGE_RATE_GPH)
    limits.setdefault("potable_water_max_rate_gpm", DEFAULT_POTABLE_WATER_MAX_RATE_GPM)
    eff.setdefault("batt_initial_charge", 0.0)
    eff.setdefault("h2_initial_charge", 0.0)
    eff.setdefault("potable_water_initial_charge", 0.0)

    if "potable_water_Wh_per_l" not in eff and "potable_water_Wh_per_l" in fallback_efficiency:
        eff["potable_water_Wh_per_l"] = fallback_efficiency["potable_water_Wh_per_l"]

    potable_energy = eff.get("potable_water_Wh_per_l")
    if potable_energy is None:
        return
    potable_energy = float(potable_energy)
    if 0.0 < potable_energy < 0.05:
        warnings.warn(
            "Interpreting potable_water_Wh_per_l as kWh/L from a legacy config and converting "
            "it to Wh/L. Update the config value to the converted Wh/L number to silence this warning."
        )
        eff["potable_water_Wh_per_l"] = potable_energy * 1000.0


def update_derived_config(config):
    migrate_legacy_limit_keys(config, fallback_efficiency=config.get("efficiency"))
    objective = config.get("objective", "total_mass")
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"Unsupported objective: {objective!r}. Expected one of {SUPPORTED_OBJECTIVES}."
        )
    validate_config(config)

    costs = config["costs"]
    eff = config["efficiency"]

    gen_cost_to_refill = costs["refill_service_cost"] / (
        costs["tank_size"] * costs["diesel_energy"] * eff["gen_eff"]
    )
    costs["generator_vom_cost_perWh"] = (
        costs["diesel_cost"] / costs["diesel_energy"] / eff["gen_eff"]
        + gen_cost_to_refill
    )


def validate_config(config):
    migrate_legacy_limit_keys(config, fallback_efficiency=config.get("efficiency"))
    sim = config["simulation"]
    costs = config["costs"]
    limits = config["limits"]
    eff = config["efficiency"]

    if int(sim["nhrs"]) < 2:
        raise ValueError("simulation.nhrs must be at least 2.")
    if int(sim.get("profile_start_offset_hours", 0)) < 0:
        raise ValueError("simulation.profile_start_offset_hours must be non-negative.")
    if float(sim.get("peak_load", 0.0)) < 0:
        raise ValueError("simulation.peak_load must be non-negative.")
    if float(sim.get("hotel_load", 0.0)) < 0:
        raise ValueError("simulation.hotel_load must be non-negative.")

    if not 0.0 < float(eff["gen_eff"]) <= 1.0:
        raise ValueError("efficiency.gen_eff must be in the interval (0, 1].")
    if not 0.0 < float(eff["batt_round_trip_eff"]) <= 1.0:
        raise ValueError("efficiency.batt_round_trip_eff must be in the interval (0, 1].")
    for key in ("batt_initial_charge", "h2_initial_charge", "potable_water_initial_charge"):
        if not 0.0 <= float(eff[key]) <= 0.5:
            raise ValueError(f"efficiency.{key} must be in the interval [0, 0.5].")
    if not 0.0 <= float(eff["batt_final_charge"]) <= 1.0:
        raise ValueError("efficiency.batt_final_charge must be in the interval [0, 1].")
    if float(eff["h2_Wh_per_g"]) <= 0.0:
        raise ValueError("efficiency.h2_Wh_per_g must be positive.")
    if float(eff["potable_water_Wh_per_l"]) <= 0.0:
        raise ValueError("efficiency.potable_water_Wh_per_l must be positive.")

    for key in ("tank_size", "diesel_energy"):
        if float(costs[key]) <= 0.0:
            raise ValueError(f"costs.{key} must be positive.")
    for key in ("diesel_cost", "refill_service_cost"):
        if float(costs[key]) < 0.0:
            raise ValueError(f"costs.{key} must be non-negative.")

    limit_pairs = (
        ("generator_min_capacity", "generator_max_capacity"),
        ("wind_min_capacity", "wind_max_capacity"),
        ("solar_min_capacity", "solar_max_capacity"),
        ("wave_min_capacity", "wave_max_capacity"),
        ("batt_min_capacity", "batt_max_capacity"),
        ("h2_min_storage_g", "h2_max_storage_g"),
        ("potable_water_min_storage_l", "potable_water_max_storage_l"),
    )
    for min_key, max_key in limit_pairs:
        min_value = float(limits[min_key])
        max_value = float(limits[max_key])
        if min_value < 0.0:
            raise ValueError(f"limits.{min_key} must be non-negative.")
        if max_value < min_value:
            raise ValueError(f"limits.{max_key} must be greater than or equal to limits.{min_key}.")

    rate_keys = (
        "batt_min_storage_level",
        "batt_max_charge_rate",
        "h2_max_charge_rate_gph",
        "potable_water_max_rate_gpm",
    )
    for key in rate_keys:
        if float(limits[key]) < 0.0:
            raise ValueError(f"limits.{key} must be non-negative.")
    if float(limits["batt_min_storage_level"]) > float(limits["batt_max_capacity"]):
        raise ValueError(
            "limits.batt_min_storage_level must be less than or equal to limits.batt_max_capacity."
        )
    if float(limits["batt_max_capacity"]) > 0.0:
        max_final_storage = float(eff["batt_final_charge"]) * float(limits["batt_max_capacity"])
        if float(limits["batt_min_storage_level"]) > max_final_storage + 1.0e-12:
            raise ValueError(
                "limits.batt_min_storage_level must be feasible with "
                "efficiency.batt_final_charge and limits.batt_max_capacity."
            )


def build_doldrum_mask(nhrs, center_frac, half_window):
    mask = np.ones(nhrs)
    center = int(nhrs * center_frac)
    start = max(int(center - half_window), 0)
    stop = min(int(center + half_window), nhrs)
    if start < stop:
        mask[start:stop] = 0
    return mask


def _window_resource_names(window):
    resources = window.get("resources", window.get("resource", ()))
    if isinstance(resources, str):
        resources = [resources]
    return {str(resource).lower() for resource in resources}


def _window_bounds_hours(window):
    if "start_day" in window:
        start = float(window["start_day"]) * 24.0
    else:
        start = float(window.get("start_hour", 0.0))

    if "stop_day" in window:
        stop = float(window["stop_day"]) * 24.0
    elif "stop_hour" in window:
        stop = float(window["stop_hour"])
    elif "duration_days" in window:
        stop = start + float(window["duration_days"]) * 24.0
    else:
        stop = start + float(window.get("duration_hours", 0.0))
    return start, stop


def build_doldrum_window_mask(nhrs, windows, resource):
    mask = np.ones(nhrs)
    resource = str(resource).lower()
    for window in windows or []:
        resources = _window_resource_names(window)
        if "all" not in resources and resource not in resources:
            continue
        start, stop = _window_bounds_hours(window)
        start_idx = max(int(np.floor(start)), 0)
        stop_idx = min(int(np.ceil(stop)), nhrs)
        if start_idx < stop_idx:
            mask[start_idx:stop_idx] = 0
    return mask


def load_resource_inputs(config):
    validate_config(config)
    sim = config["simulation"]
    nhrs = sim["nhrs"]
    peak_load = sim["peak_load"]
    H2DailyDemand = sim["H2DailyDemand"]
    H2ODailyDemand = sim["H2ODailyDemand"]
    half_window = sim["doldrum_time"]
    profile_start_offset_hours = int(sim.get("profile_start_offset_hours", 0))
    doldrums = sim["doldrums"]
    doldrum_windows = sim.get("doldrum_windows", [])

    data = pd.read_csv(sim["data_file"])
    data_len = len(data)
    if data_len == 0:
        raise ValueError("Resource CSV is empty.")
    def column_name(key):
        for candidate in COLUMN_ALIASES[key]:
            if candidate in data.columns:
                return candidate
        return None

    missing_columns = [column for column in REQUIRED_RESOURCE_COLUMNS if column_name(column) is None]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"CSV is missing required columns: {joined}.")
    time_col = column_name("Time")
    time_values = data[time_col].values if time_col is not None else None
    if time_values is None:
        warnings.warn("CSV is missing a Time column; assuming 1-hour increments.")
    elif len(time_values) > 1:
        diffs = np.diff(time_values.astype(float))
        if not np.allclose(diffs, 1.0, rtol=1e-3, atol=1e-3):
            warnings.warn(
                "Time column must be in 1-hour increments; update the CSV Time column "
                "or resample data to 1-hour steps."
            )

    if profile_start_offset_hours:
        offset = profile_start_offset_hours % data_len
    else:
        offset = 0

    def shift(values):
        if offset == 0:
            return values
        return np.roll(values, -offset)

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
    if doldrum_windows:
        doldrums_wind *= build_doldrum_window_mask(nhrs, doldrum_windows, "wind")
        doldrums_solar *= build_doldrum_window_mask(nhrs, doldrum_windows, "solar")
        doldrums_wave *= build_doldrum_window_mask(nhrs, doldrum_windows, "wave")

    wind_watts = extend(shift(data[column_name("Wind")].values)) * doldrums_wind
    solar_watts = extend(shift(data[column_name("Solar")].values)) * doldrums_solar
    wave_watts = extend(shift(data[column_name("Wave")].values)) * doldrums_wave
    load_watts = extend(shift(data[column_name("Load")].values))
    hydrogen_g = extend(shift(data[column_name("Hydrogen")].values))
    potableWater_l = extend(shift(data[column_name("PotableWater")].values))  # liters then scaled to one

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


def _has_signal(series, tol=1.0e-6):
    return float(np.max(np.abs(series))) > tol


def _plot_if_signal(ax, x, y, *args, always=False, **kwargs):
    if always or _has_signal(y):
        return ax.plot(x, y, *args, **kwargs)[0]
    return None


def _doldrum_day_spans(sim):
    half_window = float(sim.get("doldrum_time", 0.0))
    spans = []
    if half_window <= 0.0:
        center_spans = []
    else:
        nhrs = float(sim["nhrs"])
        center_spans = []
        for center_frac in sim.get("doldrums", {}).values():
            center = nhrs * float(center_frac)
            start = max(center - half_window, 0.0) / 24.0
            stop = min(center + half_window, nhrs) / 24.0
            if start < stop:
                center_spans.append((round(start, 6), round(stop, 6), "Worst-case renewable doldrum"))
    spans.extend(center_spans)
    for window in sim.get("doldrum_windows", []):
        start_hr, stop_hr = _window_bounds_hours(window)
        start = max(start_hr, 0.0) / 24.0
        stop = min(stop_hr, float(sim["nhrs"])) / 24.0
        if start < stop:
            resources = ", ".join(sorted(_window_resource_names(window)))
            label = window.get("label") or f"No {resources}"
            spans.append((round(start, 6), round(stop, 6), str(label)))
    return sorted(set(spans))


def _annotate_doldrums(axes, sim):
    spans = _doldrum_day_spans(sim)
    if not spans:
        return
    for ax in axes:
        for start, stop, _label in spans:
            ax.axvspan(start, stop, color="0.15", alpha=0.12, zorder=0)
    for start, stop, label in spans:
        axes[0].text(
            (start + stop) / 2.0,
            0.96,
            label,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="0.20",
        )


def _is_fixed(min_value, max_value, tol=1.0e-12):
    return abs(float(max_value) - float(min_value)) <= tol


def _fix_fv_if_bounded(fv, lower, upper):
    if _is_fixed(lower, upper):
        fv.STATUS = 0
        fv.value = float(lower)


def _zero_series_param(m, nhrs):
    return m.Param(value=np.zeros(nhrs))


def _inventory_initial_guess(demand, production_rate, lower, upper):
    demand = np.asarray(demand, dtype=float)
    guess = np.empty(len(demand), dtype=float)
    lower = float(lower)
    upper = float(upper)
    guess[0] = lower
    if len(demand) == 1:
        return guess

    rate = max(0.0, min(float(production_rate), float(np.mean(np.maximum(demand, 0.0)))))
    level = lower
    for i in range(1, len(demand)):
        level += rate - float(demand[i])
        level = max(lower, min(upper, level))
        guess[i] = level
    return guess


def _safe_ratio(numerator, denominator, tol=1.0e-12):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=np.abs(denominator) > tol,
    )


def _dispatch_from_vars(m_vars, tech):
    scale = to_array(m_vars[f"{tech}_scale"].value)
    if tech == "generator":
        available = scale
        legacy_fraction_key = "generator_unitfactor"
    else:
        available = to_array(m_vars[f"{tech}_unitfactor"].value) * scale
        legacy_fraction_key = f"{tech}_cur"

    power_key = f"{tech}_power"
    if power_key in m_vars:
        power = to_array(m_vars[power_key].value)
        fraction = _safe_ratio(power, available)
    else:
        fraction = to_array(m_vars[legacy_fraction_key].value)
        power = available * fraction
    return fraction, scale, power


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

    generator_unitfactor, generator_scale, generator_power = _dispatch_from_vars(
        m_vars, "generator"
    )
    wind_unitfactor = to_array(m_vars["wind_unitfactor"].value)
    wind_cur, wind_scale, wind_power = _dispatch_from_vars(m_vars, "wind")
    solar_unitfactor = to_array(m_vars["solar_unitfactor"].value)
    solar_cur, solar_scale, solar_power = _dispatch_from_vars(m_vars, "solar")
    wave_unitfactor = to_array(m_vars["wave_unitfactor"].value)
    wave_cur, wave_scale, wave_power = _dispatch_from_vars(m_vars, "wave")

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
    validate_config(config)
    sim = config["simulation"]
    costs = config["costs"]
    limits = config["limits"]
    eff = config["efficiency"]
    mass = config["mass"]

    m = GEKKO(remote=config["solver"]["remote"])
    t = np.linspace(0, sim["nhrs"] - 1, sim["nhrs"])
    m.time = t

    load = m.Param(value=inputs["Load"] + float(sim.get("hotel_load", 0.0)))
    Hydrogen_demand_g = m.Param(value=inputs["Hydrogen_demand_g"])
    PotableWater_demand_l = m.Param(value=inputs["PotableWater_demand_l"])
    wind_unitfactor = m.Param(value=inputs["Wind_unitfactor"])
    solar_unitfactor = m.Param(value=inputs["Solar_unitfactor"])
    wave_unitfactor = m.Param(value=inputs["Wave_unitfactor"])

    h2_power_max = max(
        limits["h2_max_charge_rate_gph"] * eff["h2_Wh_per_g"],
        float(np.max(inputs["Hydrogen_demand_g"])) * eff["h2_Wh_per_g"] * 2,
    )
    potable_water_rate_power = (
        limits["potable_water_max_rate_gpm"]
        * 60.0
        * LITERS_PER_GALLON
        * eff["potable_water_Wh_per_l"]
    )
    if float(limits["potable_water_max_rate_gpm"]) > 0.0:
        potable_water_power_max = potable_water_rate_power
    else:
        potable_water_power_max = (
            float(np.max(inputs["PotableWater_demand_l"])) * eff["potable_water_Wh_per_l"] * 2
        )

    battery_enabled = (
        float(limits["batt_max_capacity"]) > 0.0 and float(limits["batt_max_charge_rate"]) > 0.0
    )
    if battery_enabled:
        batt_initial_capacity = 0.5 * (limits["batt_min_capacity"] + limits["batt_max_capacity"])
        batt_initial_storage = max(
            limits["batt_min_storage_level"],
            batt_initial_capacity * eff["batt_initial_charge"],
        )
        batt_charge_power = m.Var(value=0, lb=0, ub=limits["batt_max_charge_rate"])
        batt_discharge_power = m.Var(value=0, lb=0, ub=limits["batt_max_charge_rate"])
        batt_power_inout = m.Intermediate(batt_discharge_power - batt_charge_power)
        batt_scale = m.FV(
            value=batt_initial_capacity,
            lb=limits["batt_min_capacity"],
            ub=limits["batt_max_capacity"],
        )
        batt_scale.STATUS = 1
        _fix_fv_if_bounded(batt_scale, limits["batt_min_capacity"], limits["batt_max_capacity"])
        batt_initial_level = m.FV(
            value=batt_initial_storage,
            lb=limits["batt_min_storage_level"],
            ub=limits["batt_max_capacity"],
        )
        batt_initial_level.STATUS = 1
        batt_storage_delta = m.Var(value=0.0)
        batt_storagelevel = m.Intermediate(batt_initial_level + batt_storage_delta)
        batt_one_way_eff = m.Const(np.sqrt(eff["batt_round_trip_eff"]))
    else:
        batt_power_inout = _zero_series_param(m, sim["nhrs"])
        batt_scale = _zero_series_param(m, sim["nhrs"])
        batt_storagelevel = _zero_series_param(m, sim["nhrs"])
        batt_charge_power = _zero_series_param(m, sim["nhrs"])
        batt_discharge_power = _zero_series_param(m, sim["nhrs"])

    h2_enabled = (
        float(limits["h2_max_storage_g"]) > 0.0
        or float(limits["h2_max_charge_rate_gph"]) > 0.0
        or float(np.max(inputs["Hydrogen_demand_g"])) > 0.0
    )
    if h2_enabled:
        h2_initial_storage = limits["h2_max_storage_g"] * eff["h2_initial_charge"]
        h2_power_in = m.Var(value=0, lb=0, ub=h2_power_max)
        h2_storage_scale = m.FV(
            value=limits["h2_max_storage_g"],
            lb=limits["h2_min_storage_g"],
            ub=limits["h2_max_storage_g"],
        )
        h2_storage_scale.STATUS = 1
        _fix_fv_if_bounded(h2_storage_scale, limits["h2_min_storage_g"], limits["h2_max_storage_g"])
        h2_initial_level = m.FV(
            value=h2_initial_storage,
            lb=0.0,
            ub=limits["h2_max_storage_g"],
        )
        h2_initial_level.STATUS = 1
        h2_storage_delta = m.Var(value=0.0)
        h2_storagelevel = m.Intermediate(h2_initial_level + h2_storage_delta)
    else:
        h2_power_in = _zero_series_param(m, sim["nhrs"])
        h2_storage_scale = _zero_series_param(m, sim["nhrs"])
        h2_storagelevel = _zero_series_param(m, sim["nhrs"])

    potable_water_enabled = (
        float(limits["potable_water_max_storage_l"]) > 0.0
        or float(limits["potable_water_min_storage_l"]) > 0.0
        or float(limits["potable_water_max_rate_gpm"]) > 0.0
        or float(np.max(inputs["PotableWater_demand_l"])) > 0.0
    )
    if potable_water_enabled:
        potable_production_rate_lph = 0.0
        if eff["potable_water_Wh_per_l"] > 0.0:
            potable_production_rate_lph = (
                potable_water_power_max / eff["potable_water_Wh_per_l"]
            )
        potable_initial_storage = (
            limits["potable_water_max_storage_l"] * eff["potable_water_initial_charge"]
        )
        potable_power_guess = np.full(
            sim["nhrs"],
            min(
                potable_water_power_max,
                float(np.mean(np.maximum(inputs["PotableWater_demand_l"], 0.0)))
                * eff["potable_water_Wh_per_l"],
            ),
        )
        potable_storage_guess = _inventory_initial_guess(
            inputs["PotableWater_demand_l"],
            potable_production_rate_lph,
            0.0,
            limits["potable_water_max_storage_l"],
        )
        potable_storage_guess[0] = potable_initial_storage
        potable_water_power_in = m.Var(value=potable_power_guess, lb=0, ub=potable_water_power_max)
        potable_water_storage_scale = m.FV(
            value=limits["potable_water_max_storage_l"],
            lb=limits["potable_water_min_storage_l"],
            ub=limits["potable_water_max_storage_l"],
        )
        potable_water_storage_scale.STATUS = 1
        _fix_fv_if_bounded(
            potable_water_storage_scale,
            limits["potable_water_min_storage_l"],
            limits["potable_water_max_storage_l"],
        )
        potable_water_initial_level = m.FV(
            value=potable_initial_storage,
            lb=0.0,
            ub=limits["potable_water_max_storage_l"],
        )
        potable_water_initial_level.STATUS = 1
        potable_water_storage_delta = m.Var(value=potable_storage_guess - potable_storage_guess[0])
        potable_water_storagelevel = m.Intermediate(
            potable_water_initial_level + potable_water_storage_delta
        )
    else:
        potable_water_power_in = _zero_series_param(m, sim["nhrs"])
        potable_water_storage_scale = _zero_series_param(m, sim["nhrs"])
        potable_water_storagelevel = _zero_series_param(m, sim["nhrs"])

    wind_enabled = float(limits["wind_max_capacity"]) > 0.0 or float(limits["wind_min_capacity"]) > 0.0
    if wind_enabled:
        wind_power = m.Var(value=0.0, lb=0.0, ub=limits["wind_max_capacity"])
        wind_cur = wind_power
        wind_scale = m.FV(
            value=(limits["wind_max_capacity"] + limits["wind_min_capacity"]) / 2,
            lb=limits["wind_min_capacity"],
            ub=limits["wind_max_capacity"],
        )
        wind_scale.STATUS = 1
        _fix_fv_if_bounded(wind_scale, limits["wind_min_capacity"], limits["wind_max_capacity"])
        m.Equation(wind_power <= wind_unitfactor * wind_scale)
    else:
        wind_cur = _zero_series_param(m, sim["nhrs"])
        wind_power = _zero_series_param(m, sim["nhrs"])
        wind_scale = _zero_series_param(m, sim["nhrs"])

    solar_enabled = (
        float(limits["solar_max_capacity"]) > 0.0 or float(limits["solar_min_capacity"]) > 0.0
    )
    if solar_enabled:
        solar_power = m.Var(value=0.0, lb=0.0, ub=limits["solar_max_capacity"])
        solar_cur = solar_power
        solar_scale = m.FV(
            value=(limits["solar_max_capacity"] + limits["solar_min_capacity"]) / 2,
            lb=limits["solar_min_capacity"],
            ub=limits["solar_max_capacity"],
        )
        solar_scale.STATUS = 1
        _fix_fv_if_bounded(solar_scale, limits["solar_min_capacity"], limits["solar_max_capacity"])
        m.Equation(solar_power <= solar_unitfactor * solar_scale)
    else:
        solar_cur = _zero_series_param(m, sim["nhrs"])
        solar_power = _zero_series_param(m, sim["nhrs"])
        solar_scale = _zero_series_param(m, sim["nhrs"])

    wave_enabled = float(limits["wave_max_capacity"]) > 0.0 or float(limits["wave_min_capacity"]) > 0.0
    if wave_enabled:
        wave_power = m.Var(value=0.0, lb=0.0, ub=limits["wave_max_capacity"])
        wave_cur = wave_power
        wave_scale = m.FV(
            value=(limits["wave_max_capacity"] + limits["wave_min_capacity"]) / 2,
            lb=limits["wave_min_capacity"],
            ub=limits["wave_max_capacity"],
        )
        wave_scale.STATUS = 1
        _fix_fv_if_bounded(wave_scale, limits["wave_min_capacity"], limits["wave_max_capacity"])
        m.Equation(wave_power <= wave_unitfactor * wave_scale)
    else:
        wave_cur = _zero_series_param(m, sim["nhrs"])
        wave_power = _zero_series_param(m, sim["nhrs"])
        wave_scale = _zero_series_param(m, sim["nhrs"])

    generator_enabled = (
        float(limits["generator_max_capacity"]) > 0.0
        or float(limits["generator_min_capacity"]) > 0.0
    )
    if generator_enabled:
        generator_power = m.Var(
            value=0.0,
            lb=0.0,
            ub=limits["generator_max_capacity"],
        )
        generator_scale = m.FV(
            value=(limits["generator_max_capacity"] + limits["generator_min_capacity"]) / 2,
            lb=limits["generator_min_capacity"],
            ub=limits["generator_max_capacity"],
        )
        generator_scale.STATUS = 1
        _fix_fv_if_bounded(
            generator_scale,
            limits["generator_min_capacity"],
            limits["generator_max_capacity"],
        )
        generator_unitfactor = generator_power
        m.Equation(generator_power <= generator_scale)
    else:
        generator_scale = _zero_series_param(m, sim["nhrs"])
        generator_unitfactor = _zero_series_param(m, sim["nhrs"])
        generator_power = _zero_series_param(m, sim["nhrs"])

    initial = np.zeros(sim["nhrs"])
    initial[0] = 1
    initial_vector = m.Param(initial)
    final = np.zeros(sim["nhrs"])
    final[-1] = 1
    final_vector = m.Param(final)
    if battery_enabled:
        m.Equation(
            batt_storage_delta.dt()
            == batt_charge_power * batt_one_way_eff - batt_discharge_power / batt_one_way_eff
        )
        m.Equation(batt_initial_level <= batt_scale * eff["batt_initial_charge"])
        m.Equation(batt_storagelevel * final_vector == batt_scale * eff["batt_final_charge"] * final_vector)
        m.Equation(batt_storagelevel >= limits["batt_min_storage_level"])
        m.Equation(batt_storagelevel <= batt_scale)
    if h2_enabled:
        m.Equation(h2_storage_delta.dt() == h2_power_in / eff["h2_Wh_per_g"] - Hydrogen_demand_g)
        m.Equation(h2_initial_level <= h2_storage_scale * eff["h2_initial_charge"])
        m.Equation(h2_storagelevel >= 0.0)
        m.Equation(h2_storagelevel <= h2_storage_scale)
    if potable_water_enabled:
        m.Equation(
            potable_water_storage_delta.dt()
            == potable_water_power_in / eff["potable_water_Wh_per_l"] - PotableWater_demand_l
        )
        m.Equation(
            potable_water_initial_level
            <= potable_water_storage_scale * eff["potable_water_initial_charge"]
        )
        m.Equation(potable_water_storagelevel >= 0.0)
        m.Equation(potable_water_storagelevel <= potable_water_storage_scale)

    # m.Equation(batt_in.dt() <= limits["batt_max_ramp_up"])

    m.Equation(
        batt_power_inout
        + generator_power
        + wind_power
        + solar_power
        + wave_power
        == load + h2_power_in + potable_water_power_in
    )

    sum_load = m.Var(value=10000)
    electric_demand = m.Intermediate(load + h2_power_in + potable_water_power_in)
    m.Equation(sum_load == m.integral(electric_demand))

    avg_load_W = m.Intermediate(sum_load / sim["nhrs"])
    avg_sol_W = m.Intermediate(m.integral(solar_power) / sim["nhrs"])
    avg_wave_W = m.Intermediate(m.integral(wave_power) / sim["nhrs"])
    avg_wind_W = m.Intermediate(m.integral(wind_power) / sim["nhrs"])
    avg_generator_W = m.Intermediate(m.integral(generator_power) / sim["nhrs"])
    avg_batt_throughput_W = m.Intermediate(m.integral(batt_charge_power + batt_discharge_power) / sim["nhrs"])

    potable_water_storage_mass = m.Intermediate(
        mass["potable_water_storage_Kg_per_l"] * potable_water_storage_scale
    )
    supported_mass = m.Intermediate(
        mass["generator_Kg_per_W"] * generator_scale
        + mass["batt_Kg_per_Wh"] * batt_scale
        + mass["wind_Kg_per_W"] * wind_scale
        + mass["solar_Kg_per_W"] * solar_scale
        + mass["wave_Kg_per_W"] * wave_scale
        + mass["h2_storage_Kg_per_g"] * h2_storage_scale
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
            + costs["batt_vom_cost_perWh"] * avg_batt_throughput_W * sim["lifespan"]
            + costs["wind_vom_cost_perWh"] * avg_wind_W * sim["lifespan"]
            + costs["solar_vom_cost_perWh"] * avg_sol_W * sim["lifespan"]
            + costs["wave_vom_cost_perWh"] * avg_wave_W * sim["lifespan"]
        )
    )
    cost_per_watt = m.Intermediate(total_cost / (avg_load_W + MIN_OBJECTIVE_DENOMINATOR))

    total_mass = m.Intermediate(
        final_vector * (supported_mass + floating_platform_mass + potable_water_storage_mass)
    )
    cost_per_mass = m.Intermediate(total_cost / (total_mass + MIN_OBJECTIVE_DENOMINATOR))

    objective = config.get("objective", "total_mass")
    if objective == "cost_per_watt":
        m.Obj(cost_per_watt)
    elif objective == "total_cost":
        m.Obj(total_cost)
    elif objective == "cost_per_mass":
        m.Obj(cost_per_mass)
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
        "wind_power": wind_power,
        "solar_power": solar_power,
        "wave_power": wave_power,
        "wind_scale": wind_scale,
        "solar_scale": solar_scale,
        "wave_scale": wave_scale,
        "generator_scale": generator_scale,
        "generator_power": generator_power,
        "batt_scale": batt_scale,
        "h2_storage_scale": h2_storage_scale,
        "potable_water_storage_scale": potable_water_storage_scale,
        "generator_unitfactor": generator_unitfactor,
        "initial_vector": initial_vector,
        "final_vector": final_vector,
        "batt_storagelevel": batt_storagelevel,
        "h2_storagelevel": h2_storagelevel,
        "potable_water_storagelevel": potable_water_storagelevel,
        "avg_generator_W": avg_generator_W,
        "avg_wind_W": avg_wind_W,
        "avg_sol_W": avg_sol_W,
        "avg_wave_W": avg_wave_W,
        "avg_batt_throughput_W": avg_batt_throughput_W,
        "supported_mass": supported_mass,
        "floating_platform_mass": floating_platform_mass,
        "potable_water_storage_mass": potable_water_storage_mass,
        "total_cost": total_cost,
        "cost_per_watt": cost_per_watt,
        "cost_per_mass": cost_per_mass,
        "total_mass": total_mass,
        "model": m,
    }


def solve_model(config, model):
    m, _ = model
    solver = config["solver"]
    m.options.IMODE = solver["imode"]
    m.options.MAX_ITER = solver["max_iter"]
    solver_id = solver["solver"]
    if not solver["remote"] and solver_id == 3:
        solver_id = 1
    m.options.SOLVER = solver_id
    m.options.CV_TYPE = solver["cv_type"]
    m.options.LINEAR = 1 if solver.get("linear", False) else 0
    m.solve(disp=False)


def report_results(config, model_vars):
    m_vars = model_vars
    eff = config["efficiency"]
    print("System CostPerWatt ($/W):", m_vars["cost_per_watt"].value[-1])
    print("Total Cost ($):", m_vars["total_cost"].value[-1])
    print("Total Mass (kg):", m_vars["total_mass"].value[-1])
    print("Supported Mass (kg):", m_vars["supported_mass"].value[-1])
    print("Floating Platform Mass (kg):", m_vars["floating_platform_mass"].value[-1])
    print("Unsupported Potable Water Tank Mass (kg):", m_vars["potable_water_storage_mass"].value[-1])
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

    _, _, gen_production = _dispatch_from_vars(m_vars, "generator")
    _, _, wind_production = _dispatch_from_vars(m_vars, "wind")
    _, _, solar_production = _dispatch_from_vars(m_vars, "solar")
    _, _, wave_production = _dispatch_from_vars(m_vars, "wave")
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
    if "agg" not in plt.get_backend().lower():
        plt.show()


def create_stacked_figure(config, model_vars, save_pdf=False, figsize=(10, 8)):
    timeseries_df = build_timeseries_dataframe(config, model_vars)
    return create_stacked_figure_from_dataframe(
        config,
        timeseries_df,
        save_pdf=save_pdf,
        figsize=figsize,
    )


def create_stacked_figure_from_dataframe(config, timeseries_df, save_pdf=False, figsize=(10, 8)):
    eff = config["efficiency"]
    sim = config["simulation"]
    nhrs = config["simulation"]["nhrs"]
    time_days = timeseries_df["time_days"].to_numpy(dtype=float)
    load = timeseries_df["load_W"].to_numpy(dtype=float)
    hydrogen_demand_g = timeseries_df["hydrogen_demand_g"].to_numpy(dtype=float)
    potable_water_demand_l = timeseries_df["potable_water_demand_l"].to_numpy(dtype=float)

    gen_production = timeseries_df["generator_power_W"].to_numpy(dtype=float)
    wind_production = timeseries_df["wind_power_W"].to_numpy(dtype=float)
    solar_production = timeseries_df["solar_power_W"].to_numpy(dtype=float)
    wave_production = timeseries_df["wave_power_W"].to_numpy(dtype=float)
    battery_production = timeseries_df["battery_power_W"].to_numpy(dtype=float)
    h2_power_in = timeseries_df["h2_power_W"].to_numpy(dtype=float)
    potable_water_power_in = timeseries_df["potable_power_W"].to_numpy(dtype=float)
    electric_demand = timeseries_df["electric_demand_W"].to_numpy(dtype=float)
    total_power = timeseries_df["total_power_W"].to_numpy(dtype=float)

    power_ref_max = float(np.max(np.abs(electric_demand)))
    demand_ref_max = max(
        float(np.max(np.abs(hydrogen_demand_g))), float(np.max(np.abs(potable_water_demand_l))), 1.0
    )

    fig = Figure(figsize=figsize)
    axes = fig.subplots(3, 1, sharex=True)
    fig.subplots_adjust(left=0.1, bottom=0.1, top=0.97, right=0.74, hspace=0.35)
    ax_balance, ax_prod, ax_storage = axes

    load_plot, load_scale = scale_for_plot(load, power_ref_max, enable_scaling=False)
    h2_plot, h2_scale = scale_for_plot(h2_power_in, power_ref_max, enable_scaling=False)
    water_plot, water_scale = scale_for_plot(potable_water_power_in, power_ref_max, enable_scaling=False)
    demand_plot, demand_scale = scale_for_plot(electric_demand, power_ref_max, enable_scaling=False)
    total_power_plot, total_power_scale = scale_for_plot(total_power, power_ref_max, enable_scaling=False)

    _plot_if_signal(
        ax_balance,
        time_days[2:],
        load_plot[2:],
        "-",
        label=format_label("Load", "W", load_scale),
    )
    _plot_if_signal(
        ax_balance,
        time_days[2:],
        h2_plot[2:],
        "-.",
        label=format_label("H2 Prod", "W", h2_scale),
    )
    _plot_if_signal(
        ax_balance,
        time_days[2:],
        water_plot[2:],
        ":",
        label=format_label("Water Desal", "W", water_scale),
    )
    _plot_if_signal(
        ax_balance,
        time_days[2:],
        demand_plot[2:],
        "--",
        label=format_label("Total Demand", "W", demand_scale),
        always=True,
    )
    _plot_if_signal(
        ax_balance,
        time_days[2:],
        total_power_plot[2:],
        "--",
        label=format_label("Total Power", "W", total_power_scale),
        always=True,
    )
    h2_demand_plot, h2_demand_scale = scale_for_plot(
        hydrogen_demand_g, demand_ref_max, enable_scaling=True
    )
    water_demand_plot, water_demand_scale = scale_for_plot(
        potable_water_demand_l, demand_ref_max, enable_scaling=True
    )
    ax_balance.set_ylabel("Power (W)")
    ax_balance.set_xlim(0, nhrs / 24.0)
    handles_left, labels_left = ax_balance.get_legend_handles_labels()
    handles_right, labels_right = [], []
    if _has_signal(hydrogen_demand_g) or _has_signal(potable_water_demand_l):
        ax_balance_right = ax_balance.twinx()
        _plot_if_signal(
            ax_balance_right,
            time_days[2:],
            h2_demand_plot[2:],
            color="purple",
            linestyle="-.",
            label=format_label("H2 Demand", "g", h2_demand_scale),
        )
        _plot_if_signal(
            ax_balance_right,
            time_days[2:],
            water_demand_plot[2:],
            color="teal",
            linestyle=":",
            label=format_label("Water Demand", "L", water_demand_scale),
        )
        ax_balance_right.set_ylabel("Demand (scaled)")
        handles_right, labels_right = ax_balance_right.get_legend_handles_labels()
    ax_balance.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize="small",
        frameon=False,
        borderaxespad=0.0,
    )

    wind_avail = (
        timeseries_df["wind_unitfactor"].to_numpy(dtype=float)
        * timeseries_df["wind_scale_W"].to_numpy(dtype=float)
    )
    solar_avail = (
        timeseries_df["solar_unitfactor"].to_numpy(dtype=float)
        * timeseries_df["solar_scale_W"].to_numpy(dtype=float)
    )
    wave_avail = (
        timeseries_df["wave_unitfactor"].to_numpy(dtype=float)
        * timeseries_df["wave_scale_W"].to_numpy(dtype=float)
    )

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

    production_series = (
        (
            wind_avail_plot,
            {"color": "forestgreen", "linestyle": "--", "alpha": 0.35, "lw": 1.0},
            format_label("Wind Avail", "W", wind_avail_scale),
        ),
        (
            solar_avail_plot,
            {"color": "goldenrod", "linestyle": "--", "alpha": 0.35, "lw": 1.0},
            format_label("Solar Avail", "W", solar_avail_scale),
        ),
        (
            wave_avail_plot,
            {"color": "cornflowerblue", "linestyle": "--", "alpha": 0.35, "lw": 1.0},
            format_label("Wave Avail", "W", wave_avail_scale),
        ),
        (
            wind_used_plot,
            {"color": "forestgreen", "linestyle": "-", "lw": 1.8, "zorder": 4},
            format_label("Wind Used", "W", wind_used_scale),
        ),
        (
            solar_used_plot,
            {"color": "goldenrod", "linestyle": "-", "lw": 1.8, "zorder": 4},
            format_label("Solar Used", "W", solar_used_scale),
        ),
        (
            wave_used_plot,
            {"color": "cornflowerblue", "linestyle": "-", "lw": 1.8, "zorder": 4},
            format_label("Wave Used", "W", wave_used_scale),
        ),
        (
            battery_plot,
            {"color": "0.25", "linestyle": "-", "lw": 1.9, "zorder": 6},
            format_label("Batt Used", "W", battery_scale),
        ),
        (
            gen_plot,
            {
                "color": "firebrick",
                "linestyle": "-.",
                "lw": 2.6,
                "marker": "o",
                "markevery": 12,
                "markersize": 2.5,
                "zorder": 10,
            },
            format_label("Gen Used", "W", gen_scale),
        ),
        (
            h2_power_plot,
            {"color": "purple", "linestyle": "-", "lw": 1.6, "zorder": 5},
            format_label("H2 Prod", "W", h2_power_scale),
        ),
        (
            potable_power_plot,
            {"color": "teal", "linestyle": "-", "lw": 1.6, "zorder": 5},
            format_label("Water Desal", "W", potable_power_scale),
        ),
    )
    for series, style, label in production_series:
        _plot_if_signal(ax_prod, time_days, series, label=label, **style)
    ax_prod.set_ylabel("Power Production (W)")
    if ax_prod.get_legend_handles_labels()[0]:
        ax_prod.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    batt_storage = timeseries_df["battery_storage_Wh"].to_numpy(dtype=float)
    h2_storage_Wh = timeseries_df["h2_storage_Wh_equiv"].to_numpy(dtype=float)
    potable_storage_Wh = timeseries_df["potable_storage_Wh_equiv"].to_numpy(dtype=float)
    storage_ref_max = max(
        float(np.max(np.abs(batt_storage))),
        float(np.max(np.abs(h2_storage_Wh))),
        float(np.max(np.abs(potable_storage_Wh))),
        1.0,
    )
    batt_storage_plot, batt_storage_scale = scale_for_plot(
        batt_storage, storage_ref_max, enable_scaling=False
    )
    h2_storage_plot, h2_storage_scale = scale_for_plot(
        h2_storage_Wh, storage_ref_max, enable_scaling=True, order_of_magnitude=True
    )
    potable_storage_plot, potable_storage_scale = scale_for_plot(
        potable_storage_Wh, storage_ref_max, enable_scaling=True, order_of_magnitude=True
    )
    storage_series = (
        (batt_storage_plot, "dimgrey", format_label("Battery Storage", "Wh", batt_storage_scale)),
        (h2_storage_plot, "purple", format_label("H2 Storage", "Wh equiv", h2_storage_scale)),
        (potable_storage_plot, "teal", format_label("Water Storage", "Wh equiv", potable_storage_scale)),
    )
    for series, color, label in storage_series:
        _plot_if_signal(ax_storage, time_days, series, color=color, label=label)
    ax_storage.set_ylabel("Storage (Wh equiv.)")
    ax_storage.set_xlabel("time (Days)")
    if ax_storage.get_legend_handles_labels()[0]:
        ax_storage.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    _annotate_doldrums(axes, sim)

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
