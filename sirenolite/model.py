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
        "wind_vom_cost_perWh": 1.0e-5,  # $/Wh, small renewable dispatch tie-breaker
        "solar_vom_cost_perWh": 1.0e-5,  # $/Wh, small renewable dispatch tie-breaker
        "wave_vom_cost_perWh": 1.0e-5,  # $/Wh, small renewable dispatch tie-breaker
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
        "backend": "gekko",  # paper presets opt into the portable HiGHS formulation
        "remote": False,  # local solving is the public default; remote solving is opt-in
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
        "paper": {"generator_min_installed_w": 500.0, "max_delivery_trips": 20,
                  "delivery_repeat_factor": 5.0, "primary_cost_tolerance_usd": 1.0e-4},
        "site": {"key": "", "label": "Custom resource CSV", "latitude": 0.0,
                 "longitude": 0.0, "mission_key": "", "horizon_hours": 0,
                 "resource_source": "", "schedule_basis": "CSV demand schedule"},
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
        + (0.0 if config["solver"].get("backend", "gekko") == "paper" else gen_cost_to_refill)
    )


def validate_config(config):
    migrate_legacy_limit_keys(config, fallback_efficiency=config.get("efficiency"))
    sim = config["simulation"]
    costs = config["costs"]
    limits = config["limits"]
    eff = config["efficiency"]

    objective = config.get("objective", "total_mass")
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported objective: {objective!r}. Expected one of {SUPPORTED_OBJECTIVES}.")

    backend = config["solver"].get("backend", "gekko")
    if backend not in ("gekko", "paper"):
        raise ValueError("solver.backend must be 'gekko' or 'paper'.")
    if backend == "paper":
        if objective != "total_cost":
            raise ValueError("Paper missions minimize total_cost; choose GEKKO for other objectives.")
        if config["solver"].get("remote", False):
            raise ValueError("Paper missions solve locally; remote solving is available only with GEKKO.")
        if any(float(limits[k]) > 0 for k in ("h2_min_storage_g", "h2_max_storage_g", "h2_max_charge_rate_gph")) or float(sim["H2DailyDemand"]) > 0:
            raise ValueError("Paper missions support electricity and desalination; disable hydrogen or choose GEKKO.")
        for key in ("generator_min_installed_w", "max_delivery_trips", "delivery_repeat_factor", "primary_cost_tolerance_usd"):
            _validate_number(config["paper"][key], "paper." + key, minimum=0.0,
                             integer=key == "max_delivery_trips")
    # Reject invalid physical inputs before they become obscure solver errors or
    # contaminate normalized profiles. Numeric strings are not a public API type.
    for section in ("costs", "limits", "efficiency", "mass"):
        for key, value in config[section].items():
            if section == "costs" and key == "generator_vom_cost_perWh":
                continue  # derived from fuel properties on every build
            _validate_number(value, f"{section}.{key}", minimum=0.0)
    for key in ("nhrs", "profile_start_offset_hours"):
        value = sim.get(key, 0)
        _validate_number(value, f"simulation.{key}", minimum=0.0, integer=True)
    for key in ("peak_load", "hotel_load", "H2DailyDemand", "H2ODailyDemand", "doldrum_time"):
        _validate_number(sim.get(key, 0.0), f"simulation.{key}", minimum=0.0)
    _validate_number(sim["lifespan"], "simulation.lifespan", minimum=0.0, strict=True)
    for key, value in sim["doldrums"].items():
        _validate_number(value, f"simulation.doldrums.{key}", minimum=0.0, maximum=1.0)
    windows = sim.get("doldrum_windows", [])
    if not isinstance(windows, list):
        raise ValueError("simulation.doldrum_windows must be a list of window objects.")
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f"simulation.doldrum_windows[{index}] must be an object.")
        unknown = set(window) - {"start_hour", "start_day", "stop_hour", "stop_day",
                                 "duration_hours", "duration_days", "resources", "resource", "label"}
        if unknown:
            raise ValueError(f"Unknown simulation.doldrum_windows[{index}] key(s): {', '.join(sorted(unknown))}")
        for alternatives in (("start_hour", "start_day"),
                             ("stop_hour", "stop_day", "duration_hours", "duration_days"),
                             ("resources", "resource")):
            if sum(key in window for key in alternatives) > 1:
                raise ValueError(f"simulation.doldrum_windows[{index}] has conflicting fields: {', '.join(alternatives)}.")
        resources = _window_resource_names(window)
        if not resources or resources - {"wind", "solar", "wave", "all"}:
            raise ValueError(f"simulation.doldrum_windows[{index}].resources must name wind, solar, wave, or all.")
        for key in ("start_hour", "stop_hour", "duration_hours", "start_day", "stop_day", "duration_days"):
            if key in window:
                _validate_number(window[key], f"simulation.doldrum_windows[{index}].{key}", minimum=0.0)
        start, stop = doldrum_window_bounds_hours(window)
        if stop <= start:
            raise ValueError(f"simulation.doldrum_windows[{index}] must have positive duration.")
    solver = config["solver"]
    for key in ("remote", "linear"):
        if key in solver and not isinstance(solver[key], (bool, np.bool_)):
            raise ValueError(f"solver.{key} must be a boolean.")
    for key in ("imode", "max_iter", "solver", "cv_type"):
        _validate_number(solver[key], f"solver.{key}", minimum=1, integer=True)
    if solver["imode"] != 6:
        raise ValueError("solver.imode must be 6 for simultaneous dynamic sizing and dispatch optimization.")
    if solver["solver"] not in (1, 2, 3):
        raise ValueError("solver.solver must be 1 (APOPT), 2 (BPOPT), or 3 (IPOPT; local fallback to APOPT).")
    if solver["cv_type"] not in (1, 2):
        raise ValueError("solver.cv_type must be 1 or 2.")
    if backend == "gekko" and solver.get("linear", False) and limits["batt_max_capacity"] > 0 and limits["batt_max_charge_rate"] > 0:
        raise ValueError("solver.linear must be false when battery dispatch is enabled (charge/discharge complementarity).")
    if solver.get("linear", False) and objective in ("cost_per_watt", "cost_per_mass"):
        raise ValueError("solver.linear must be false for a ratio objective.")

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
        if not 0.0 <= float(eff[key]) <= 1.0:
            raise ValueError(f"efficiency.{key} must be in the interval [0, 1].")
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
    if backend == "gekko" and float(limits["batt_max_capacity"]) > 0.0:
        max_final_storage = float(eff["batt_final_charge"]) * float(limits["batt_max_capacity"])
        if float(limits["batt_min_storage_level"]) > max_final_storage + 1.0e-12:
            raise ValueError(
                "limits.batt_min_storage_level must be feasible with "
                "efficiency.batt_final_charge and limits.batt_max_capacity."
            )
        max_initial_storage = float(eff["batt_initial_charge"]) * float(limits["batt_max_capacity"])
        if float(limits["batt_min_storage_level"]) > max_initial_storage + 1.0e-12:
            raise ValueError(
                "limits.batt_min_storage_level must be feasible with "
                "efficiency.batt_initial_charge and limits.batt_max_capacity."
            )


def _validate_number(value, name, *, minimum=None, maximum=None, integer=False, strict=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if integer and (not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        qualifier = "positive" if strict and minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")


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
    if not isinstance(resources, (list, tuple)):
        raise ValueError("Doldrum window resources must be a resource name or list of names.")
    return {str(resource).lower() for resource in resources}


def doldrum_window_bounds_hours(window):
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


def _window_bounds_hours(window):
    return doldrum_window_bounds_hours(window)


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


def _resource_column_name(data, key):
    for candidate in COLUMN_ALIASES[key]:
        if candidate in data.columns:
            return candidate
    return None


def _resource_columns(data):
    columns = {
        key: _resource_column_name(data, key)
        for key in (*REQUIRED_RESOURCE_COLUMNS, "Time")
    }
    missing_columns = [key for key in REQUIRED_RESOURCE_COLUMNS if columns[key] is None]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"CSV is missing required columns: {joined}.")
    return columns


def _warn_if_time_is_not_hourly(data, time_column):
    if time_column is None:
        warnings.warn("CSV is missing a Time column; assuming 1-hour increments.")
        return
    try:
        time_values = data[time_column].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV Time column must contain finite numeric hours.") from exc
    if not np.all(np.isfinite(time_values)):
        raise ValueError("CSV Time column must contain finite numeric hours.")
    if len(time_values) <= 1:
        return
    diffs = np.diff(time_values.astype(float))
    if not np.allclose(diffs, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(
            "Time column must be in 1-hour increments; resample the resource and demand "
            "data to 1-hour steps before solving. Changing timestamps alone changes the physics."
        )


def _offset_profile(values, offset):
    if offset == 0:
        return values
    return np.roll(values, -offset)


def _warn_if_profile_will_repeat(data_len, nhrs):
    if data_len < nhrs:
        warnings.warn(
            f"Resource data length ({data_len}) is shorter than horizon ({nhrs}); "
            "looping data to fill the horizon."
        )


def _repeat_or_trim_profile(values, nhrs):
    values = np.asarray(values)
    data_len = len(values)
    if data_len < nhrs:
        repeats = int(np.ceil(nhrs / data_len))
        return np.tile(values, repeats)[:nhrs]
    return values[:nhrs]


def _prepare_profile(data, column_name, *, offset, nhrs):
    values = _offset_profile(data[column_name].values, offset)
    return _repeat_or_trim_profile(values, nhrs)


def _normalize_profile(values):
    max_val = float(np.max(values))
    if max_val <= 0:
        return np.zeros_like(values, dtype=float)
    return values / max_val


def load_resource_inputs(config):
    validate_config(config)
    sim = config["simulation"]
    nhrs = sim["nhrs"]
    offset = int(sim.get("profile_start_offset_hours", 0))

    data = pd.read_csv(sim["data_file"])
    data_len = len(data)
    if data_len == 0:
        raise ValueError("Resource CSV is empty.")
    columns = _resource_columns(data)
    for key in REQUIRED_RESOURCE_COLUMNS:
        column = columns[key]
        try:
            values = data[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CSV column {column!r} must contain finite non-negative numbers.") from exc
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"CSV column {column!r} must contain finite non-negative numbers.")
        data[column] = values
    _warn_if_time_is_not_hourly(data, columns["Time"])
    if config["solver"].get("backend", "gekko") == "paper":
        # File rows are physical hourly intervals. Node zero is a separate
        # inventory boundary, not an omitted first hour of resource/demand.
        hours = int(nhrs) - 1
        if hours > data_len or offset + hours > data_len:
            raise ValueError("Paper horizon and start offset exceed available CSV hours; shorten the horizon or supply a longer profile.")
        def interval_values(key):
            full = _normalize_profile(data[columns[key]].to_numpy(dtype=float))
            values = full[offset:offset + hours]
            return np.r_[values[0], values]
        result = {"Wind_unitfactor": interval_values("Wind"),
                  "Solar_unitfactor": interval_values("Solar"),
                  "Wave_unitfactor": interval_values("Wave"),
                  "Load": interval_values("Load") * sim["peak_load"],
                  "Hydrogen_demand_g": interval_values("Hydrogen") * sim["H2DailyDemand"],
                  "PotableWater_demand_l": interval_values("PotableWater") * sim["H2ODailyDemand"]}
        for technology in ("wind", "solar", "wave"):
            mask = build_doldrum_mask(hours, sim["doldrums"][technology], sim["doldrum_time"])
            mask *= build_doldrum_window_mask(hours, sim.get("doldrum_windows", []), technology)
            result[technology.title() + "_unitfactor"][1:] *= mask
        return result
    _warn_if_profile_will_repeat(data_len, nhrs)

    offset = offset % data_len if offset else 0
    doldrums = sim["doldrums"]
    half_window = sim["doldrum_time"]

    doldrums_wind = build_doldrum_mask(nhrs, doldrums["wind"], half_window)
    doldrums_solar = build_doldrum_mask(nhrs, doldrums["solar"], half_window)
    doldrums_wave = build_doldrum_mask(nhrs, doldrums["wave"], half_window)
    doldrum_windows = sim.get("doldrum_windows", [])
    if doldrum_windows:
        doldrums_wind *= build_doldrum_window_mask(nhrs, doldrum_windows, "wind")
        doldrums_solar *= build_doldrum_window_mask(nhrs, doldrum_windows, "solar")
        doldrums_wave *= build_doldrum_window_mask(nhrs, doldrum_windows, "wave")

    wind_watts = _prepare_profile(data, columns["Wind"], offset=offset, nhrs=nhrs)
    solar_watts = _prepare_profile(data, columns["Solar"], offset=offset, nhrs=nhrs)
    wave_watts = _prepare_profile(data, columns["Wave"], offset=offset, nhrs=nhrs)
    load_watts = _prepare_profile(data, columns["Load"], offset=offset, nhrs=nhrs)
    hydrogen_g = _prepare_profile(data, columns["Hydrogen"], offset=offset, nhrs=nhrs)
    potable_water_l = _prepare_profile(data, columns["PotableWater"], offset=offset, nhrs=nhrs)

    return {
        "Wind_unitfactor": _normalize_profile(wind_watts) * doldrums_wind,
        "Solar_unitfactor": _normalize_profile(solar_watts) * doldrums_solar,
        "Wave_unitfactor": _normalize_profile(wave_watts) * doldrums_wave,
        "Load": _normalize_profile(load_watts) * sim["peak_load"],
        "Hydrogen_demand_g": _normalize_profile(hydrogen_g) * sim["H2DailyDemand"],
        "PotableWater_demand_l": _normalize_profile(potable_water_l) * sim["H2ODailyDemand"],
    }


def to_array(values):
    return np.array(list(values), dtype=float)


def scale_for_plot(series, reference_max, enable_scaling=True, order_of_magnitude=False):
    if not enable_scaling:
        return series, 1.0
    max_series = float(np.max(np.abs(series)))
    # Keep solver-scale numerical dust from becoming a visible inventory after
    # rescaling. This is the same signal threshold used by _plot_if_signal.
    if max_series <= 1.0e-6 or reference_max <= 0:
        return series, 1.0
    scale = reference_max / max_series
    if order_of_magnitude:
        scale = 10.0 ** int(np.round(np.log10(scale)))
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
        if len(x) == 1:
            kwargs.setdefault("marker", "o")
            kwargs.setdefault("markersize", 4)
        return ax.plot(x, y, *args, **kwargs)[0]
    return None


def _legend_above(ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(ncol, len(handles)),
            fontsize="small",
            frameon=False,
            borderaxespad=0.0,
        )


def _doldrum_day_spans(sim):
    """Shade the solved intervals selected by each outage's sample mask."""
    nhrs = int(sim["nhrs"])
    half_window = float(sim.get("doldrum_time", 0.0))
    spans = []

    def add_mask_spans(mask, label):
        # Row k describes (k-1, k] h; row zero is initialization. Use the
        # applied mask so rounding and horizon clipping match the solve.
        removed = np.asarray(mask[1:]) == 0
        edges = np.diff(np.r_[False, removed, False].astype(int))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            spans.append((start / 24.0, stop / 24.0, label))

    if half_window > 0.0:
        for resource, center_frac in sim.get("doldrums", {}).items():
            mask = build_doldrum_mask(nhrs, center_frac, half_window)
            add_mask_spans(mask, f"{resource.capitalize()} outage")
    for window in sim.get("doldrum_windows", []):
        resource_names = sorted(_window_resource_names(window))
        mask = build_doldrum_window_mask(nhrs, [window], resource_names[0])
        label = window.get("label") or f"No {', '.join(resource_names)}"
        add_mask_spans(mask, str(label))
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


def _add_generation_dispatch(m, nhrs, limits, tech, unitfactor=None):
    min_key = f"{tech}_min_capacity"
    max_key = f"{tech}_max_capacity"
    lower = limits[min_key]
    upper = limits[max_key]
    enabled = float(upper) > 0.0 or float(lower) > 0.0
    if not enabled:
        zero = _zero_series_param(m, nhrs)
        return zero, zero, zero

    power = m.Var(value=0.0, lb=0.0, ub=upper)
    scale = m.FV(value=(upper + lower) / 2, lb=lower, ub=upper)
    scale.STATUS = 1
    _fix_fv_if_bounded(scale, lower, upper)
    if unitfactor is None:
        m.Equation(power <= scale)
    else:
        m.Equation(power <= unitfactor * scale)
    return power, scale, power


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
        "is_initial_state": np.arange(len(time_hr)) == 0,
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


def _validated_model_inputs(config, inputs):
    checked = {}
    nhrs = config["simulation"]["nhrs"]
    for key in ("Wind_unitfactor", "Solar_unitfactor", "Wave_unitfactor", "Load",
                "Hydrogen_demand_g", "PotableWater_demand_l"):
        if key not in inputs:
            raise ValueError(f"Missing model input {key!r}.")
        try:
            values = np.asarray(inputs[key], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Model input {key!r} must contain finite non-negative numbers.") from exc
        if values.shape != (nhrs,):
            raise ValueError(f"Model input {key!r} must have shape ({nhrs},).")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Model input {key!r} must contain finite non-negative numbers.")
        if key.endswith("_unitfactor") and np.any(values > 1):
            raise ValueError(f"Model input {key!r} must be in the interval [0, 1].")
        checked[key] = values
    return checked


def build_model(config, inputs):
    if config["solver"].get("backend", "gekko") == "paper":
        validate_config(config)
        from sirenolite import paper_solver
        return paper_solver.build_model(config, _validated_model_inputs(config, inputs))
    update_derived_config(config)
    inputs = _validated_model_inputs(config, inputs)
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

    h2_power_max = limits["h2_max_charge_rate_gph"] * eff["h2_Wh_per_g"]
    potable_water_rate_power = (
        limits["potable_water_max_rate_gpm"]
        * 60.0
        * LITERS_PER_GALLON
        * eff["potable_water_Wh_per_l"]
    )
    potable_water_power_max = potable_water_rate_power

    battery_enabled = float(limits["batt_max_capacity"]) > 0.0
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

    wind_cur, wind_scale, wind_power = _add_generation_dispatch(
        m, sim["nhrs"], limits, "wind", wind_unitfactor
    )
    solar_cur, solar_scale, solar_power = _add_generation_dispatch(
        m, sim["nhrs"], limits, "solar", solar_unitfactor
    )
    wave_cur, wave_scale, wave_power = _add_generation_dispatch(
        m, sim["nhrs"], limits, "wave", wave_unitfactor
    )
    generator_unitfactor, generator_scale, generator_power = _add_generation_dispatch(
        m, sim["nhrs"], limits, "generator"
    )

    initial = np.zeros(sim["nhrs"])
    initial[0] = 1
    initial_vector = m.Param(initial)
    final = np.zeros(sim["nhrs"])
    final[-1] = 1
    final_vector = m.Param(final)
    if battery_enabled:
        # A single battery cannot charge and discharge at the same instant.
        # Enforce the physical mode constraint instead of pricing away cycling.
        # With both powers nonnegative, product <= 0 enforces exact
        # complementarity. Its slack keeps APOPT's Jacobian well formed at
        # the zero/zero starting point (a bare equality can crash local APOPT).
        m.Equation(batt_charge_power * batt_discharge_power <= 0)
        m.Equation(
            batt_storage_delta.dt()
            == batt_charge_power * batt_one_way_eff - batt_discharge_power / batt_one_way_eff
        )
        m.Equation(batt_initial_level == batt_scale * eff["batt_initial_charge"])
        m.Equation(batt_storagelevel * final_vector == batt_scale * eff["batt_final_charge"] * final_vector)
        m.Equation(batt_storagelevel >= limits["batt_min_storage_level"])
        m.Equation(batt_storagelevel <= batt_scale)
    if h2_enabled:
        m.Equation(h2_storage_delta.dt() == h2_power_in / eff["h2_Wh_per_g"] - Hydrogen_demand_g)
        m.Equation(h2_initial_level == h2_storage_scale * eff["h2_initial_charge"])
        m.Equation(h2_storagelevel >= 0.0)
        m.Equation(h2_storagelevel <= h2_storage_scale)
    if potable_water_enabled:
        m.Equation(
            potable_water_storage_delta.dt()
            == potable_water_power_in / eff["potable_water_Wh_per_l"] - PotableWater_demand_l
        )
        m.Equation(
            potable_water_initial_level
            == potable_water_storage_scale * eff["potable_water_initial_charge"]
        )
        m.Equation(potable_water_storagelevel >= 0.0)
        m.Equation(potable_water_storagelevel <= potable_water_storage_scale)

    # m.Equation(batt_in.dt() <= limits["batt_max_ramp_up"])

    electrical_controls = battery_enabled or h2_enabled or potable_water_enabled or any(
        limits[f"{technology}_max_capacity"] > 0 for technology in ("generator", "wind", "solar", "wave")
    )
    if electrical_controls:
        m.Equation(
            batt_power_inout + generator_power + wind_power + solar_power + wave_power
            == load + h2_power_in + potable_water_power_in
        )
    elif np.any(inputs["Load"][1:] + float(sim.get("hotel_load", 0.0)) > 0):
        raise ValueError("The system is infeasible: no electrical source is enabled to meet positive load.")
    # For a disabled zero-demand system, 0=0 carries no physical constraint and
    # would add a zero Jacobian row that local APOPT cannot factor.
    electric_demand = m.Intermediate(load + h2_power_in + potable_water_power_in)
    sum_load = m.integral(electric_demand)

    # The first node sets initial storage. With NODES=2, each later dispatch
    # applies to its preceding one-hour interval: nhrs nodes span nhrs-1 hours.
    duration_hours = float(t[-1] - t[0])
    avg_load_W = m.Intermediate(sum_load / duration_hours)
    avg_sol_W = m.Intermediate(m.integral(solar_power) / duration_hours)
    avg_wave_W = m.Intermediate(m.integral(wave_power) / duration_hours)
    avg_wind_W = m.Intermediate(m.integral(wind_power) / duration_hours)
    avg_generator_W = m.Intermediate(m.integral(generator_power) / duration_hours)
    avg_batt_throughput_W = m.Intermediate(m.integral(batt_charge_power + batt_discharge_power) / duration_hours)

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
        "batt_charge_power": batt_charge_power,
        "batt_discharge_power": batt_discharge_power,
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
        "avg_load_W": avg_load_W,
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
    validate_config(config)
    if config["solver"].get("backend", "gekko") == "paper":
        from sirenolite import paper_solver
        return paper_solver.solve_model(config, model)
    m, model_vars = model
    solver = config["solver"]
    m.options.IMODE = solver["imode"]
    m.options.MAX_ITER = solver["max_iter"]
    solver_id = solver["solver"]
    if not solver["remote"] and solver_id == 3:
        solver_id = 1
    m.options.SOLVER = solver_id
    m.options.CV_TYPE = solver["cv_type"]
    m.options.NODES = 2
    m.options.LINEAR = 1 if solver.get("linear", False) else 0
    m.solve(disp=False)
    if getattr(m.options, "APPSTATUS", 1) != 1 or getattr(m.options, "SOLVESTATUS", 1) != 1:
        raise RuntimeError("The solver did not report a successful feasible solution.")
    if model_vars is not None:
        diagnostics = solution_diagnostics(config, model_vars)
        model_vars["solution_diagnostics"] = diagnostics
        if not diagnostics["passed"]:
            failed = [name for name, check in diagnostics["checks"].items() if not check["passed"]]
            raise RuntimeError("Solved result violates physical feasibility checks: " + ", ".join(failed))


def solution_diagnostics(config, model_vars, *, atol=1e-5, rtol=1e-6):
    """Replay hourly balances and bounds in physical units after a solve.

    This verifies the discrete solution, not the fidelity of component models
    or accuracy under time refinement. Node zero supplies initial inventories;
    dispatch at subsequent nodes applies to the preceding interval.
    """
    limits, eff = config["limits"], config["efficiency"]
    time = np.asarray(model_vars["model"].time, dtype=float)
    dt = np.diff(time)
    checks = {}

    def values(name):
        result = to_array(model_vars[name].value)
        if result.shape != time.shape or not np.all(np.isfinite(result)):
            raise RuntimeError(f"Solved result {name!r} is incomplete or non-finite.")
        return result

    def check(name, residual, unit, *scales):
        magnitude = max([1.0] + [float(np.max(np.abs(scale))) for scale in scales])
        tolerance = float(atol + rtol * magnitude)
        error = float(np.max(np.abs(residual)))
        checks[name] = {"max_residual": error, "unit": unit,
                        "tolerance": tolerance, "passed": bool(error <= tolerance)}

    def bounds(name, value, lower, upper, unit):
        violation = np.maximum(np.maximum(lower - value, value - upper), 0.0)
        # Scale by actual operation/capacity, never a loose configured ceiling.
        check(name, violation, unit, value, lower)

    load = values("load")[1:]
    for name in ("total_cost", "total_mass", "cost_per_watt", "cost_per_mass"):
        values(name)  # Never publish non-finite objective/reporting quantities.
    charge, discharge = values("batt_charge_power")[1:], values("batt_discharge_power")[1:]
    hydrogen_power, water_power = values("h2_power_in")[1:], values("potable_water_power_in")[1:]
    demand = load + hydrogen_power + water_power
    supply = discharge - charge
    for technology in ("generator", "wind", "solar", "wave"):
        scale = values(f"{technology}_scale")
        power = values(f"{technology}_power")[1:]
        capacity = scale[1:]
        available = capacity if technology == "generator" else capacity * values(f"{technology}_unitfactor")[1:]
        bounds(f"{technology}_capacity", scale, limits[f"{technology}_min_capacity"],
               limits[f"{technology}_max_capacity"], "W")
        check(f"{technology}_constant_capacity", scale - scale[-1], "W", scale)
        bounds(f"{technology}_dispatch", power, 0.0, available, "W")
        average_key = "avg_sol_W" if technology == "solar" else f"avg_{technology}_W"
        check(f"{technology}_average_power", values(average_key)[-1] - np.dot(dt, power) / dt.sum(),
              "W", power)
        supply = supply + power
    check("electrical_balance", supply - demand, "W", supply, demand)
    check("average_electrical_demand", values("avg_load_W")[-1] - np.dot(dt, demand) / dt.sum(), "W", demand)
    check("battery_bus_dispatch", values("batt_power_inout")[1:] - (discharge - charge), "W", charge, discharge)

    battery = values("batt_storagelevel")
    battery_capacity = values("batt_scale")
    bounds("battery_capacity", battery_capacity, limits["batt_min_capacity"], limits["batt_max_capacity"], "Wh")
    check("battery_constant_capacity", battery_capacity - battery_capacity[-1], "Wh", battery_capacity)
    bounds("battery_inventory", battery, limits["batt_min_storage_level"], battery_capacity, "Wh")
    bounds("battery_charge_rate", charge, 0.0, limits["batt_max_charge_rate"], "W")
    bounds("battery_discharge_rate", discharge, 0.0, limits["batt_max_charge_rate"], "W")
    check("battery_simultaneous_charge_discharge", np.minimum(charge, discharge), "W", charge, discharge)
    eta = np.sqrt(eff["batt_round_trip_eff"])
    check("battery_energy_balance", np.diff(battery) - dt * (eta * charge - discharge / eta),
          "Wh", battery, dt * charge, dt * discharge)
    check("battery_initial_inventory", battery[0] - battery_capacity[0] * eff["batt_initial_charge"],
          "Wh", battery_capacity)
    check("battery_final_inventory", battery[-1] - battery_capacity[-1] * eff["batt_final_charge"],
          "Wh", battery_capacity)

    for name, power, storage_key, scale_key, demand_key, min_key, max_key, fraction_key, energy_key, rate, unit in (
        ("hydrogen", hydrogen_power, "h2_storagelevel", "h2_storage_scale", "hydrogen_demand_g",
         "h2_min_storage_g", "h2_max_storage_g", "h2_initial_charge", "h2_Wh_per_g",
         limits["h2_max_charge_rate_gph"], "g"),
        ("potable_water", water_power, "potable_water_storagelevel", "potable_water_storage_scale",
         "potable_water_demand_l", "potable_water_min_storage_l", "potable_water_max_storage_l",
         "potable_water_initial_charge", "potable_water_Wh_per_l",
         limits["potable_water_max_rate_gpm"] * 60 * LITERS_PER_GALLON, "L"),
    ):
        storage, capacity = values(storage_key), values(scale_key)
        production = power / eff[energy_key]
        consumption = values(demand_key)[1:]
        bounds(f"{name}_capacity", capacity, limits[min_key], limits[max_key], unit)
        check(f"{name}_constant_capacity", capacity - capacity[-1], unit, capacity)
        bounds(f"{name}_inventory", storage, 0.0, capacity, unit)
        bounds(f"{name}_production_rate", production, 0.0, rate, unit + "/h")
        check(f"{name}_material_balance", np.diff(storage) - dt * (production - consumption),
              unit, storage, dt * consumption, dt * production)
        check(f"{name}_initial_inventory", storage[0] - capacity[0] * eff[fraction_key], unit, capacity)

    # Reconstruct reported mass and lifetime cost from installed sizes and
    # interval dispatch, independently of GEKKO's reporting intermediates.
    mass, costs = config["mass"], config["costs"]
    component_sizes = {
        "generator": values("generator_scale")[-1],
        "wind": values("wind_scale")[-1],
        "solar": values("solar_scale")[-1],
        "wave": values("wave_scale")[-1],
        "batt": battery_capacity[-1],
        "h2_storage": values("h2_storage_scale")[-1],
        "potable_water_storage": values("potable_water_storage_scale")[-1],
    }
    supported_mass = sum(component_sizes[name] * mass[f"{name}_Kg_per_W"]
                         for name in ("generator", "wind", "solar", "wave"))
    supported_mass += component_sizes["batt"] * mass["batt_Kg_per_Wh"]
    supported_mass += component_sizes["h2_storage"] * mass["h2_storage_Kg_per_g"]
    platform_mass = supported_mass * mass["floating_platform_mass_per_supported_mass"]
    water_mass = component_sizes["potable_water_storage"] * mass["potable_water_storage_Kg_per_l"]
    expected_mass = supported_mass + platform_mass + water_mass
    for name, expected in (("supported_mass", supported_mass), ("floating_platform_mass", platform_mass),
                           ("potable_water_storage_mass", water_mass), ("total_mass", expected_mass)):
        check(f"{name}_accounting", values(name)[-1] - expected, "kg", expected)

    average_throughput = np.dot(dt, charge + discharge) / dt.sum()
    check("battery_average_throughput", values("avg_batt_throughput_W")[-1] - average_throughput,
          "W", charge + discharge)
    fixed_cost = sum(component_sizes[name] * costs[f"{name}_fix_cost"]
                     for name in ("generator", "wind", "solar", "wave", "batt"))
    fixed_cost += component_sizes["h2_storage"] * costs["h2_storage_fix_cost_per_g"]
    fixed_cost += component_sizes["potable_water_storage"] * costs["potable_water_storage_fix_cost_per_l"]
    fixed_cost += platform_mass * costs["floating_platform_cost_per_kg"]
    fuel_cost_per_wh = (costs["diesel_cost"] + costs["refill_service_cost"] / costs["tank_size"]) / (
        costs["diesel_energy"] * eff["gen_eff"])
    operating_cost_per_hour = average_throughput * costs["batt_vom_cost_perWh"]
    for name in ("generator", "wind", "solar", "wave"):
        unit_cost = fuel_cost_per_wh if name == "generator" else costs[f"{name}_vom_cost_perWh"]
        operating_cost_per_hour += np.dot(dt, values(f"{name}_power")[1:]) / dt.sum() * unit_cost
    expected_cost = fixed_cost + operating_cost_per_hour * config["simulation"]["lifespan"]
    check("total_cost_accounting", values("total_cost")[-1] - expected_cost, "USD", expected_cost)
    average_demand = np.dot(dt, demand) / dt.sum()
    for name, denominator, unit in (("cost_per_watt", average_demand, "USD/W"),
                                    ("cost_per_mass", expected_mass, "USD/kg")):
        expected = expected_cost / (denominator + MIN_OBJECTIVE_DENOMINATOR)
        check(f"{name}_accounting", values(name)[-1] - expected, unit, expected)

    return {"passed": all(item["passed"] for item in checks.values()),
            "duration_hours": float(time[-1] - time[0]), "solved_intervals": len(dt),
            "max_power_balance_W": checks["electrical_balance"]["max_residual"], "checks": checks}


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
    print("Battery Capacity (Wh):", m_vars["batt_scale"].value[-1])
    print("Battery Peak Inventory (Wh):", max(m_vars["batt_storagelevel"].value))
    print("Hydrogen Storage Capacity (g):", m_vars["h2_storage_scale"].value[-1])
    print("Hydrogen Peak Inventory (g):", max(m_vars["h2_storagelevel"].value))
    print("Potable Water Storage Capacity (L):", m_vars["potable_water_storage_scale"].value[-1])
    print("Potable Water Peak Inventory (L):", max(m_vars["potable_water_storagelevel"].value))
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
        m.time[1:] / 24.0,
        load_plot[1:],
        "-",
        label=format_label("Load", "W", load_scale),
    )
    ax_balance.plot(
        m.time[1:] / 24.0,
        h2_plot[1:],
        "-.",
        label=format_label("Hydrogen production", "W", h2_scale),
    )
    ax_balance.plot(
        m.time[1:] / 24.0,
        water_plot[1:],
        ":",
        label=format_label("Water production", "W", water_scale),
    )
    ax_balance.plot(
        m.time[1:] / 24.0,
        demand_plot[1:],
        "-",
        label=format_label("Power needed", "W", demand_scale),
    )
    ax_balance.plot(
        m.time[1:] / 24.0,
        total_power_plot[1:],
        "--",
        label=format_label("Power supplied", "W", total_power_scale),
    )
    ax_balance_right = ax_balance.twinx()
    h2_demand_plot, h2_demand_scale = scale_for_plot(
        hydrogen_demand_g, demand_ref_max, enable_scaling=True
    )
    water_demand_plot, water_demand_scale = scale_for_plot(
        potable_water_demand_l, demand_ref_max, enable_scaling=True
    )
    ax_balance_right.plot(
        m.time[1:] / 24.0,
        h2_demand_plot[1:],
        color="purple",
        linestyle="-.",
        label=format_label("Hydrogen needed", "g/h", h2_demand_scale),
    )
    ax_balance_right.plot(
        m.time[1:] / 24.0,
        water_demand_plot[1:],
        color="teal",
        linestyle=":",
        label=format_label("Water needed", "L/h", water_demand_scale),
    )
    ax_balance_right.set_ylabel("Demand [scaled units]")
    plt.ylabel("Power [W]")
    plt.xlim(0, (nhrs - 1) / 24.0)
    handles_left, labels_left = ax_balance.get_legend_handles_labels()
    handles_right, labels_right = ax_balance_right.get_legend_handles_labels()
    ax_balance.legend(handles_left + handles_right, labels_left + labels_right)
    plt.xlabel("Time [days]")
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
        label=format_label("Wind available", "W", wind_avail_scale),
    )
    plt.plot(
        m.time / 24.0,
        solar_avail_plot,
        color="goldenrod",
        linestyle="--",
        label=format_label("Solar available", "W", solar_avail_scale),
    )
    plt.plot(
        m.time / 24.0,
        wave_avail_plot,
        color="cornflowerblue",
        linestyle="--",
        label=format_label("Wave available", "W", wave_avail_scale),
    )
    plt.plot(
        m.time / 24.0,
        wind_used_plot,
        color="forestgreen",
        linestyle="-",
        label=format_label("Wind used", "W", wind_used_scale),
    )
    plt.plot(
        m.time / 24.0,
        solar_used_plot,
        color="goldenrod",
        linestyle="-",
        label=format_label("Solar used", "W", solar_used_scale),
    )
    plt.plot(
        m.time / 24.0,
        wave_used_plot,
        color="cornflowerblue",
        linestyle="-",
        label=format_label("Wave used", "W", wave_used_scale),
    )
    plt.plot(
        m.time / 24.0,
        battery_plot,
        color="dimgrey",
        linestyle="-",
        label=format_label("Battery power", "W", battery_scale),
    )
    plt.plot(
        m.time / 24.0,
        gen_plot,
        color="firebrick",
        label=format_label("Generator power", "W", gen_scale),
    )
    plt.plot(
        m.time / 24.0,
        h2_power_plot,
        color="purple",
        label=format_label("Hydrogen production", "W", h2_power_scale),
    )
    plt.plot(
        m.time / 24.0,
        potable_power_plot,
        color="teal",
        label=format_label("Water production", "W", potable_power_scale),
    )
    plt.ylabel("Power [W]")
    plt.xlim(0, (nhrs - 1) / 24.0)
    plt.legend(loc=(0.5, 0.5))
    plt.xlabel("Time [days]")
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
        label=format_label("Energy in battery", "Wh", batt_storage_scale),
    )
    plt.plot(
        m.time / 24.0,
        h2_storage_plot,
        color="purple",
        label=format_label("Stored hydrogen", "equivalent Wh", h2_storage_scale),
    )
    plt.plot(
        m.time / 24.0,
        potable_storage_plot,
        color="teal",
        label=format_label("Stored water", "equivalent Wh", potable_storage_scale),
    )
    plt.ylabel("Stored-energy scale [Wh]")
    plt.xlim(0, (nhrs - 1) / 24.0)
    plt.legend()
    plt.xlabel("Time [days]")
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

    # Constrained layout reserves space for the optional demand-rate twin axis
    # and legends at both export and embedded GUI sizes.
    fig = Figure(figsize=figsize, constrained_layout=True)
    axes = fig.subplots(3, 1, sharex=True)
    ax_balance, ax_prod, ax_storage = axes

    load_plot, load_scale = scale_for_plot(load, power_ref_max, enable_scaling=False)
    h2_plot, h2_scale = scale_for_plot(h2_power_in, power_ref_max, enable_scaling=False)
    water_plot, water_scale = scale_for_plot(potable_water_power_in, power_ref_max, enable_scaling=False)
    demand_plot, demand_scale = scale_for_plot(electric_demand, power_ref_max, enable_scaling=False)
    total_power_plot, total_power_scale = scale_for_plot(total_power, power_ref_max, enable_scaling=False)

    _plot_if_signal(
        ax_balance,
        time_days[1:],
        load_plot[1:],
        "-",
        label=format_label("Load", "W", load_scale),
    )
    _plot_if_signal(
        ax_balance,
        time_days[1:],
        h2_plot[1:],
        "-.",
        label=format_label("Hydrogen production", "W", h2_scale),
    )
    _plot_if_signal(
        ax_balance,
        time_days[1:],
        water_plot[1:],
        ":",
        label=format_label("Water production", "W", water_scale),
    )
    _plot_if_signal(
        ax_balance,
        time_days[1:],
        demand_plot[1:],
        "--",
        label=format_label("Power needed", "W", demand_scale),
        always=True,
    )
    _plot_if_signal(
        ax_balance,
        time_days[1:],
        total_power_plot[1:],
        "--",
        label=format_label("Power supplied", "W", total_power_scale),
        always=True,
    )
    h2_demand_plot, h2_demand_scale = scale_for_plot(
        hydrogen_demand_g, demand_ref_max, enable_scaling=True
    )
    water_demand_plot, water_demand_scale = scale_for_plot(
        potable_water_demand_l, demand_ref_max, enable_scaling=True
    )
    ax_balance.set_ylabel("Power (W)")
    ax_balance.set_xlim(0, (nhrs - 1) / 24.0)
    handles_left, labels_left = ax_balance.get_legend_handles_labels()
    handles_right, labels_right = [], []
    if _has_signal(hydrogen_demand_g) or _has_signal(potable_water_demand_l):
        ax_balance_right = ax_balance.twinx()
        _plot_if_signal(
            ax_balance_right,
            time_days[1:],
            h2_demand_plot[1:],
            color="purple",
            linestyle="-.",
            label=format_label("Hydrogen needed", "g/h", h2_demand_scale),
        )
        _plot_if_signal(
            ax_balance_right,
            time_days[1:],
            water_demand_plot[1:],
            color="teal",
            linestyle=":",
            label=format_label("Water needed", "L/h", water_demand_scale),
        )
        ax_balance_right.set_ylabel("Demand [scaled units]")
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
            format_label("Wind available", "W", wind_avail_scale),
        ),
        (
            solar_avail_plot,
            {"color": "goldenrod", "linestyle": "--", "alpha": 0.35, "lw": 1.0},
            format_label("Solar available", "W", solar_avail_scale),
        ),
        (
            wave_avail_plot,
            {"color": "cornflowerblue", "linestyle": "--", "alpha": 0.35, "lw": 1.0},
            format_label("Wave available", "W", wave_avail_scale),
        ),
        (
            wind_used_plot,
            {"color": "forestgreen", "linestyle": "-", "lw": 1.8, "zorder": 4},
            format_label("Wind used", "W", wind_used_scale),
        ),
        (
            solar_used_plot,
            {"color": "goldenrod", "linestyle": "-", "lw": 1.8, "zorder": 4},
            format_label("Solar used", "W", solar_used_scale),
        ),
        (
            wave_used_plot,
            {"color": "cornflowerblue", "linestyle": "-", "lw": 1.8, "zorder": 4},
            format_label("Wave used", "W", wave_used_scale),
        ),
        (
            battery_plot,
            {"color": "0.25", "linestyle": "-", "lw": 1.9, "zorder": 6},
            format_label("Battery power", "W", battery_scale),
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
            format_label("Generator power", "W", gen_scale),
        ),
        (
            h2_power_plot,
            {"color": "purple", "linestyle": "-", "lw": 1.6, "zorder": 5},
            format_label("Hydrogen production", "W", h2_power_scale),
        ),
        (
            potable_power_plot,
            {"color": "teal", "linestyle": "-", "lw": 1.6, "zorder": 5},
            format_label("Water production", "W", potable_power_scale),
        ),
    )
    for series, style, label in production_series:
        _plot_if_signal(ax_prod, time_days[1:], series[1:], label=label, **style)
    ax_prod.set_ylabel("Power [W]")
    _legend_above(ax_prod)

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
        (batt_storage_plot, "dimgrey", format_label("Energy in battery", "Wh", batt_storage_scale)),
        (h2_storage_plot, "purple", format_label("Stored hydrogen", "equivalent Wh", h2_storage_scale)),
        (potable_storage_plot, "teal", format_label("Stored water", "equivalent Wh", potable_storage_scale)),
    )
    for series, color, label in storage_series:
        _plot_if_signal(ax_storage, time_days, series, color=color, label=label)
    ax_storage.set_ylabel("Stored-energy scale [Wh]")
    ax_storage.set_xlabel("Time [days]")
    _legend_above(ax_storage)

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
