import gc
import json
from copy import deepcopy
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from types import SimpleNamespace
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def _ensure_matplotlib_cache_dir():
    if os.environ.get("MPLCONFIGDIR"):
        return
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_dir = base / "SIRENO-lite" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # If a prewarmed cache was bundled with the app, seed it on first run.
    if not any(cache_dir.glob("fontlist-v*.json")):
        candidate_dirs = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidate_dirs.append(Path(meipass) / "mplconfig_seed")
        candidate_dirs.append(Path(__file__).resolve().parent / "mplconfig_seed")
        for seed_dir in candidate_dirs:
            if not seed_dir.is_dir():
                continue
            for seed_file in seed_dir.iterdir():
                if seed_file.is_file():
                    shutil.copy2(seed_file, cache_dir / seed_file.name)
            break

    os.environ["MPLCONFIGDIR"] = str(cache_dir)


_ensure_matplotlib_cache_dir()

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

from sirenolite import model
from sirenolite.config import (
    configuration_from_payload,
    load_configuration,
    merge_overrides,
    parse_bool,
    prepare_config,
    split_json_payload,
)


class _PlotCanvas(FigureCanvasTkAgg):
    def resize(self, event):
        # A newly packed or withdrawn Tk canvas briefly reports 1x1 pixels.
        # Preserve the meaningful figure geometry until a usable layout exists;
        # feeding that transient size to constrained layout collapses its axes.
        if event.width < 200 or event.height < 200:
            return
        super().resize(event)


VALID_DOLDRUM_WINDOW_RESOURCES = {"all", "wind", "solar", "wave"}


def parse_doldrum_windows_text(text):
    """Parse GUI JSON for explicit resource-outage windows."""
    text = (text or "").strip()
    if not text:
        return []
    windows = json.loads(text)
    if not isinstance(windows, list):
        raise ValueError("doldrum_windows must be a JSON list.")

    validated = []
    for index, window in enumerate(windows, start=1):
        if not isinstance(window, dict):
            raise ValueError(f"doldrum_windows item {index} must be a JSON object.")

        resources = window.get("resources", window.get("resource"))
        if resources is None:
            raise ValueError(f"doldrum_windows item {index} must include resource(s).")
        if isinstance(resources, str):
            resource_names = [resources]
        elif isinstance(resources, list) and all(isinstance(item, str) for item in resources):
            resource_names = resources
        else:
            raise ValueError(
                f"doldrum_windows item {index} resource(s) must be a string or list of strings."
            )
        invalid = sorted(
            {resource.lower() for resource in resource_names}
            - VALID_DOLDRUM_WINDOW_RESOURCES
        )
        if invalid:
            joined = ", ".join(invalid)
            allowed = ", ".join(sorted(VALID_DOLDRUM_WINDOW_RESOURCES))
            raise ValueError(
                f"doldrum_windows item {index} has unsupported resource(s): {joined}. "
                f"Allowed values are: {allowed}."
            )

        try:
            start, stop = model.doldrum_window_bounds_hours(window)
        except Exception as exc:
            raise ValueError(
                f"doldrum_windows item {index} must define start plus stop or duration."
            ) from exc
        if stop <= start:
            raise ValueError(f"doldrum_windows item {index} stop time must be after start time.")
        validated.append(window)

    return validated


def format_doldrum_windows_text(windows):
    return json.dumps(windows or [], indent=2)


def parse_value(text, template):
    if isinstance(template, bool):
        return parse_bool(text)
    if isinstance(template, int) and not isinstance(template, bool):
        value = float(text)
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("Enter a whole number.")
        return int(value)
    if isinstance(template, float):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Enter a finite number.")
        return value
    return text


def build_json_payload(config, optimize_vars):
    payload = dict(config)
    payload["gui"] = {
        "optimize": {name: bool(var.get()) for name, var in optimize_vars.items()}
    }
    return payload


def result_composition(config, variables):
    """Dispatched source energy and installed storage on a common energy basis.

    The initial state is not a dispatch interval. Battery discharge is internal
    energy transfer, so is deliberately excluded from source production.
    Product storage is displayed as its electricity-to-produce equivalent;
    it does not imply that stored water or hydrogen can discharge electricity.
    """
    data = model.build_timeseries_dataframe(config, variables)
    dt = np.diff(data["time_days"].to_numpy(dtype=float)) * 24
    labels = ["Wind", "Solar", "Wave", "Diesel"]
    columns = ["wind_power_W", "solar_power_W", "wave_power_W", "generator_power_W"]
    generation = [float(np.dot(data[column].to_numpy(dtype=float)[1:], dt)) / 1000
                  for column in columns]
    capacity = lambda key: max(0.0, float(variables[key].value[-1]))
    battery = capacity("batt_scale") / 1000
    hydrogen = capacity("h2_storage_scale")
    water = capacity("potable_water_storage_scale")
    storage = [battery, water * config["efficiency"]["potable_water_Wh_per_l"] / 1000,
               hydrogen * config["efficiency"]["h2_Wh_per_g"] / 1000]
    return {"generation_labels": labels, "generation_kWh": np.maximum(generation, 0).tolist(),
            "storage_labels": [f"Battery: {battery:.3g} kWh", f"Water: {water:.3g} L",
                               f"Hydrogen: {hydrogen:.3g} g"], "storage_equivalent_kWh": storage}


def result_history(data, charge_W=None, discharge_W=None):
    """Aggregate display histories; retain extrema and never alter exported data.

    Daily bins contain dispatch intervals ending in that day. Means are duration
    weighted, while inventory ranges include both boundary nodes of each bin.
    """
    hours = data["time_hr"].to_numpy(dtype=float)
    dt = np.diff(hours)
    daily = hours[-1] - hours[0] >= 720
    groups = (np.floor((hours[1:] - hours[0] - 1e-9) / 24).astype(int)
              if daily else np.arange(len(dt)))
    indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    mean = lambda values: np.array([np.average(np.asarray(values)[1:][i], weights=dt[i]) for i in indices])
    columns = ["wind_power_W", "solar_power_W", "wave_power_W", "generator_power_W", "electric_demand_W"]
    power = {column: mean(data[column].to_numpy(dtype=float)) / 1000 for column in columns}
    net = data["battery_power_W"].to_numpy(dtype=float)
    power["charge_kW"] = mean(np.maximum(-net, 0) if charge_W is None else charge_W) / 1000
    power["discharge_kW"] = mean(np.maximum(net, 0) if discharge_W is None else discharge_W) / 1000
    storage = {}
    for column in ["battery_storage_Wh", "potable_storage_l", "h2_storage_g"]:
        values = data[column].to_numpy(dtype=float)
        storage[column] = {
            "min": np.array([values[i[0]:i[-1] + 2].min() for i in indices]),
            "max": np.array([values[i[0]:i[-1] + 2].max() for i in indices]),
            "end": np.array([values[i[-1] + 1] for i in indices]),
        }
    return {"daily": daily, "time_days": np.array([hours[i[-1] + 1] for i in indices]) / 24,
            "duration_hours": np.array([dt[i].sum() for i in indices]),
            "power": power, "storage": storage, "horizon_days": hours[-1] / 24}


def create_result_figure(config, variables):
    """Display source/storage composition and physically labeled dispatch trends."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(6, 4, width_ratios=[1.05, 1, 1, 1])
    axes = [fig.add_subplot(grid[i * 2:(i + 1) * 2, 1:]) for i in range(3)]
    data = model.build_timeseries_dataframe(config, variables)
    array = lambda key: model.to_array(variables[key].value) if key in variables else None
    history = result_history(data, array("batt_charge_power"), array("batt_discharge_power"))
    composition = result_composition(config, variables)
    source_colors = ["#14866d", "#c58f00", "#2563eb", "#4b5563"]
    storage_colors = ["#8b5cf6", "#06b6d4", "#bf6caf"]

    def pie(axis, values, labels, colors, title, units):
        total = sum(values)
        threshold = max(1e-8, total * 1e-7)
        active = [index for index, value in enumerate(values) if value > threshold]
        axis.set_title(title, fontsize=10, fontweight="bold", pad=8)
        if not active:
            axis.text(.5, .5, "None selected", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            return
        wedges, *_ = axis.pie([values[i] for i in active], colors=[colors[i] for i in active],
                             radius=.72, center=(0, .32),
                             startangle=90, autopct=lambda pct: f"{pct:.0f}%" if pct >= 5 else "",
                             textprops={"fontsize": 9}, wedgeprops={"edgecolor": "white"})
        axis.set_xlim(-1.15, 1.15)
        axis.set_ylim(-1.05, 1.25)
        axis.legend(wedges, [labels[i] for i in active], loc="upper center",
                    bbox_to_anchor=(.5, .29), fontsize=8, frameon=False,
                    ncol=2 if len(active) > 3 else 1)
        total_text = f"{total:,.0f}" if total >= 1000 else f"{total:.3g}"
        axis.text(.5, -.03, f"Total: {total_text} {units}", transform=axis.transAxes,
                  ha="center", fontsize=8)

    pie(fig.add_subplot(grid[:3, 0]), composition["generation_kWh"],
        composition["generation_labels"], source_colors, "Energy source", "kWh dispatched")
    pie(fig.add_subplot(grid[3:, 0]), composition["storage_equivalent_kWh"],
        composition["storage_labels"], storage_colors, "Storage capacity", "kWh equivalent")
    t, power = history["time_days"], history["power"]
    daily = history["daily"]
    average_label = "Daily mean" if daily else "Hourly"
    source_columns = ["wind_power_W", "solar_power_W", "wave_power_W", "generator_power_W"]
    active = [i for i, key in enumerate(source_columns) if np.max(np.abs(power[key])) > 1e-8]
    if active:
        axes[0].stackplot(t, *[power[source_columns[i]] for i in active],
                          colors=[source_colors[i] for i in active],
                          labels=[composition["generation_labels"][i] for i in active], alpha=.85)
    axes[0].plot(t, power["electric_demand_W"], color="#111827", lw=1.2,
                 label="Electrical consumption")
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_title(f"{average_label} dispatched sources and consumption", loc="left", fontsize=10)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left", fontsize=7, ncol=len(active) + 1,
                   bbox_to_anchor=(0, 1.02), framealpha=.95)
    charge, discharge = power["charge_kW"], power["discharge_kW"]
    if np.max(charge) > 1e-8 or np.max(discharge) > 1e-8:
        axes[1].fill_between(t, 0, discharge, color="#8b5cf6", alpha=.75, label="Discharge (+)")
        axes[1].fill_between(t, 0, -charge, color="#bba4ef", alpha=.85, label="Charge (−)")
        axes[1].legend(loc="upper right", fontsize=7, ncol=2)
    else:
        axes[1].text(.5, .5, "No battery cycling", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].axhline(0, color="0.5", lw=.5)
    axes[1].set_ylabel("Battery power (kW)")
    axes[1].set_title(f"{average_label} battery charge and discharge", loc="left", fontsize=10)
    for column, capacity_key, label, color in zip(
            ["battery_storage_Wh", "potable_storage_l", "h2_storage_g"],
            ["batt_scale", "potable_water_storage_scale", "h2_storage_scale"],
            ["Battery", "Water tank", "Hydrogen"], storage_colors):
        capacity = float(variables[capacity_key].value[-1])
        if capacity <= 1e-7:
            continue
        values = history["storage"][column]
        if daily:
            axes[2].fill_between(t, values["min"] / capacity * 100, values["max"] / capacity * 100,
                                 color=color, alpha=.18)
        axes[2].plot(t, values["end"] / capacity * 100, color=color, lw=1, label=label)
    axes[2].set_ylabel("Inventory (% capacity)")
    axes[2].set_ylim(-2, 102)
    axes[2].set_title("Daily inventory range (shade) and end-of-day level" if daily else
                      "Hourly inventory level", loc="left", fontsize=10)
    if axes[2].lines:
        axes[2].legend(loc="upper right", fontsize=7, ncol=3)
    else:
        axes[2].text(.5, .5, "No storage selected", ha="center", va="center", transform=axes[2].transAxes)
    for axis in axes:
        axis.set_xlim(0, history["horizon_days"])
        axis.tick_params(labelsize=8)
        axis.yaxis.label.set_size(9)
        axis.grid(axis="y", color="0.9", linewidth=.5)
        axis.set_axisbelow(True)
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[2].set_xlabel("Time (days)", fontsize=9)
    return fig


def create_app(root=None, show_error=None):
    """Build the desktop UI without starting its event loop.

    The returned controller exposes the same commands used by its real widgets,
    enabling desktop integration tests and embedding in a Tk application.
    """
    root = root if root is not None else tk.Tk()
    show_error = show_error or messagebox.showerror

    wrapping_labels = []

    def make_label(parent, **kwargs):
        label = ttk.Label(parent, **kwargs)
        if "wraplength" in kwargs:
            wrapping_labels.append(label)
            label.bind("<Configure>", lambda event: label.configure(wraplength=max(40, event.width - 12)))
        return label
    root.title("SIRENO-lite (System-Integrated Resilient ENergy Optimizer)")
    root.geometry("1200x900")
    root.minsize(760, 520)

    from sirenolite import presets
    defaults = presets.load_site_config(presets.DEFAULT_SITE_KEY)
    config_state = {"base": deepcopy(defaults)}

    field_vars = {}
    field_meta = {}
    field_widgets = {}
    optimize_vars = {}
    config_path_var = tk.StringVar(master=root, value="")
    objective_options = list(model.SUPPORTED_OBJECTIVES)
    doldrum_windows_text = {"widget": None}
    field_labels = {
        "data_file": "Resource CSV", "profile_start_offset_hours": "Profile start offset",
        "lifespan": "Project lifetime", "nhrs": "Hourly samples",
        "peak_load": "Peak electric load", "hotel_load": "Hotel / parasitic load",
        "H2DailyDemand": "Peak H2 demand", "H2ODailyDemand": "Peak water demand",
        "doldrum_time": "Outage half-window",
        "generator_min_installed_w": "Minimum installed diesel size",
        "max_delivery_trips": "Maximum delivery trips",
    }

    descriptions = {
        "costs": {
            "generator_fix_cost": "Generator capex ($/W)",
            "wind_fix_cost": "Wind capex ($/W)",
            "solar_fix_cost": "Solar capex ($/W)",
            "wave_fix_cost": "Wave capex ($/W)",
            "batt_fix_cost": "Battery capex ($/Wh)",
            "h2_storage_fix_cost_per_g": "H2 storage capex ($/g)",
            "potable_water_storage_fix_cost_per_l": "Potable water storage capex ($/L)",
            "floating_platform_cost_per_kg": (
                "Floating platform cost ($/kg). Set to 0 to remove platform cost."
            ),
            "tank_size": "Diesel tank size (gal)",
            "diesel_cost": "Diesel cost ($/gal)",
            "diesel_energy": "Diesel energy (Wh/gal)",
            "refill_service_cost": "Refill service cost ($)",
            "generator_vom_cost_perWh": "Derived generator VOM ($/Wh)",
            "batt_vom_cost_perWh": "Battery VOM ($/Wh)",
            "wind_vom_cost_perWh": "Wind VOM ($/Wh)",
            "solar_vom_cost_perWh": "Solar VOM ($/Wh)",
            "wave_vom_cost_perWh": "Wave VOM ($/Wh)",
        },
        "limits": {
            "generator_max_capacity": "Generator max capacity (W)",
            "wind_max_capacity": "Wind max capacity (W)",
            "solar_max_capacity": "Solar max capacity (W)",
            "wave_max_capacity": "Wave max capacity (W)",
            "batt_max_capacity": "Battery max capacity (Wh)",
            "generator_min_capacity": "Generator min capacity (W)",
            "wind_min_capacity": "Wind min capacity (W)",
            "solar_min_capacity": "Solar min capacity (W)",
            "wave_min_capacity": "Wave min capacity (W)",
            "batt_min_capacity": "Battery min capacity (Wh)",
            "batt_min_storage_level": "Battery min storage (Wh)",
            "batt_max_charge_rate": "Battery max charge/discharge rate (W)",
            "h2_min_storage_g": "H2 storage min (g)",
            "h2_max_storage_g": "H2 storage max (g)",
            "h2_max_charge_rate_gph": "H2 production max rate (g/hr)",
            "potable_water_min_storage_l": "Potable water storage min (L)",
            "potable_water_max_storage_l": "Potable water storage max (L)",
            "potable_water_max_rate_gpm": "Potable water production max rate (gal/min)",
        },
        "efficiency": {
            "gen_eff": "Generator efficiency (fraction)",
            "batt_round_trip_eff": "Battery round-trip efficiency",
            "batt_initial_charge": "Initial battery SOC (fraction, 0–1)",
            "batt_final_charge": "Final battery SOC (fraction, 0–1)",
            "h2_initial_charge": "Initial H2 inventory (capacity fraction, 0–1); may be consumed by horizon end",
            "h2_Wh_per_g": "H2 production energy (Wh/g)",
            "potable_water_initial_charge": "Initial water inventory (capacity fraction, 0–1); may be consumed by horizon end",
            "potable_water_Wh_per_l": "Desal energy (Wh/L)",
        },
        "mass": {
            "generator_Kg_per_W": "Generator mass (kg/W)",
            "batt_Kg_per_Wh": "Battery mass (kg/Wh)",
            "wind_Kg_per_W": "Wind mass (kg/W)",
            "solar_Kg_per_W": "Solar mass (kg/W)",
            "wave_Kg_per_W": "Wave mass (kg/W)",
            "h2_storage_Kg_per_g": "H2 storage mass (kg/g)",
            "potable_water_storage_Kg_per_l": "Potable water storage mass (kg/L)",
            "floating_platform_mass_per_supported_mass": (
                "Platform mass per supported mass (kg/kg). Set to 0 to remove platform mass."
            ),
        },
        "simulation": {
            "lifespan": "Project lifespan (hr)",
            "nhrs": "Hourly sample count (N samples span N−1 hr; minimum 2)",
            "peak_load": "Peak electric load (W)",
            "hotel_load": "Fixed hotel/parasitic load (W)",
            "H2DailyDemand": "H2 peak demand rate (g/hr); the CSV shape is scaled to this peak",
            "H2ODailyDemand": "Potable water peak demand rate (L/hr); the CSV shape is scaled to this peak",
            "data_file": "Resource/load CSV file",
            "profile_start_offset_hours": "Circular profile start offset (hr)",
            "doldrum_time": "Doldrum half-window (hr, resource set to zero around center)",
        },
        "doldrums": {
            "wind": "Wind doldrum center (fraction of horizon, 0-1; 0.5 is midpoint)",
            "solar": "Solar doldrum center (fraction of horizon, 0-1; 0.5 is midpoint)",
            "wave": "Wave doldrum center (fraction of horizon, 0-1; 0.5 is midpoint)",
        },
        "solver": {
            "remote": "Send the model and input profiles to the remote GEKKO server; uncheck to solve locally",
            "imode": "GEKKO IMODE",
            "max_iter": "Solver max iterations",
            "solver": "GEKKO solver id",
            "cv_type": "GEKKO CV type",
            "linear": "Use GEKKO linear solver mode",
        },
        "paper": {
            "generator_min_installed_w": "Minimum generator size if diesel is installed (W)",
            "max_delivery_trips": "Maximum diesel deliveries in the modeled horizon; each fills one tank",
        },
    }

    def apply_config_to_ui(config, gui_payload=None):
        # Validate the whole transaction before changing any visible field.
        config, optimize_payload = configuration_from_payload(
            {"config": config, "gui": gui_payload or {}}
        )
        config_state["base"] = deepcopy(config)
        objective = config.get("objective")
        if objective is not None:
            if objective not in objective_options:
                raise ValueError(
                    f"Unsupported objective: {objective!r}. Expected one of {objective_options}."
                )
            objective_var.set(objective)

        for (section, key), var in field_vars.items():
            if section == "doldrums":
                value = config["simulation"]["doldrums"][key]
            else:
                value = config[section][key]
            if isinstance(var, tk.BooleanVar):
                var.set(parse_bool(value))
            else:
                var.set("" if value is None else str(value))
        if doldrum_windows_text["widget"] is not None:
            widget = doldrum_windows_text["widget"]
            widget.delete("1.0", "end")
            widget.insert(
                "1.0",
                format_doldrum_windows_text(config["simulation"].get("doldrum_windows", [])),
            )

        for key, var in optimize_vars.items():
            var.set(optimize_payload.get(key, True))
        refresh_mode_controls()

    def add_entry(parent, section_key, key, value, row, widget_type="entry"):
        # Integer-valued physical defaults (Wh, Wh/gal, hours) still accept
        # fractional physical quantities. Only discrete settings require integers.
        integer_fields = {"nhrs", "profile_start_offset_hours", "imode", "max_iter", "solver", "cv_type", "max_delivery_trips"}
        if isinstance(value, int) and not isinstance(value, bool) and key not in integer_fields:
            value = float(value)
        make_label(parent, text=field_labels.get(key, key)).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        desc_text = descriptions.get(section_key, {}).get(key, "")
        if widget_type == "bool":
            var = tk.BooleanVar(master=root, value=bool(value))
            chk = ttk.Checkbutton(parent, variable=var)
            chk.grid(row=row, column=1, sticky="w", padx=6, pady=2)
            field_widgets[(section_key, key)] = chk
            make_label(parent, text=desc_text, wraplength=260).grid(
                row=row, column=3, sticky="w", padx=6, pady=2
            )
            field_vars[(section_key, key)] = var
            field_meta[(section_key, key)] = {"template": value}
            return
        if widget_type == "file":
            var = tk.StringVar(master=root, value=str(value))
            entry = ttk.Entry(parent, textvariable=var, width=18)
            entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
            field_widgets[(section_key, key)] = entry
            browse_btn = ttk.Button(
                parent,
                text="Browse",
                command=lambda v=var: browse_resource(v),
            )
            browse_btn.grid(row=row, column=2, sticky="w", padx=6, pady=2)
            make_label(parent, text=desc_text, wraplength=260).grid(
                row=row, column=3, sticky="w", padx=6, pady=2
            )
            field_vars[(section_key, key)] = var
            field_meta[(section_key, key)] = {"template": value}
            return

        var = tk.StringVar(master=root, value=str(value))
        entry = ttk.Entry(parent, textvariable=var, width=18)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        field_widgets[(section_key, key)] = entry
        make_label(parent, text=desc_text, wraplength=260).grid(
            row=row, column=3, sticky="w", padx=6, pady=2
        )
        field_vars[(section_key, key)] = var
        field_meta[(section_key, key)] = {"template": value}

    def browse_resource(var):
        selected = filedialog.askopenfilename(
            title="Select Resource Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            var.set(selected)

    def add_note(parent, label, description, row):
        make_label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        make_label(parent, text="(n/a)").grid(
            row=row, column=1, sticky="w", padx=6, pady=2
        )
        make_label(parent, text=description, wraplength=260).grid(
            row=row, column=3, sticky="w", padx=6, pady=2
        )

    def add_section_header(parent, text, row):
        make_label(parent, text=text).grid(row=row, column=0, sticky="w", padx=6, pady=(8, 2))

    def build_section(parent, title, section_key, values, special=None, order=None):
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.pack(fill="x", padx=10, pady=6)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        special = special or {}
        row = 0
        keys = order if order is not None else list(values.keys())
        for key in keys:
            value = values[key]
            if special.get(key) == "skip":
                continue
            widget_type = special.get(key)
            if widget_type is None and isinstance(value, bool):
                widget_type = "bool"
            add_entry(frame, section_key, key, value, row, widget_type=widget_type or "entry")
            row += 1
        return frame

    def build_component_section(parent, title, optimize_key, optimize_label, optimize_desc, categories):
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.pack(fill="x", padx=10, pady=6)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        row = 0
        if optimize_key is not None:
            var = tk.BooleanVar(master=root, value=True)
            optimize_vars[optimize_key] = var
            make_label(frame, text=optimize_label).grid(
                row=row, column=0, sticky="w", padx=6, pady=2
            )
            ttk.Checkbutton(frame, variable=var).grid(
                row=row, column=1, sticky="w", padx=6, pady=2
            )
            make_label(frame, text=optimize_desc, wraplength=260).grid(
                row=row, column=3, sticky="w", padx=6, pady=2
            )
            row += 1

        for category in categories:
            name = category["name"]
            items = category.get("items", [])
            notes = category.get("notes", [])
            note = category.get("note")
            if items or notes:
                add_section_header(frame, name, row)
                row += 1
                for section_key, key in items:
                    value = defaults[section_key][key]
                    add_entry(frame, section_key, key, value, row)
                    row += 1
                for label, description in notes:
                    add_note(frame, label, description, row)
                    row += 1
            else:
                add_note(frame, f"{name} (n/a)", note or "Not modeled for this component.", row)
                row += 1
        return frame

    def create_scrollable(parent):
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable_frame.bind("<Configure>", on_configure)
        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def on_canvas_configure(event):
            canvas.itemconfig(window_id, width=event.width)

        canvas.bind("<Configure>", on_canvas_configure)

        def on_mousewheel(event):
            # Toplevel bindings receive events from descendants. Scroll only
            # this panel, while letting text editors keep their own scrolling.
            if isinstance(event.widget, (tk.Text, tk.Listbox, ttk.Combobox)):
                return None
            widget = event.widget
            while widget is not root:
                if widget is canvas:
                    if getattr(event, "num", None) in (4, 5):
                        units = -1 if event.num == 4 else 1
                    else:
                        delta = event.delta
                        units = -int(delta) if sys.platform == "darwin" else -int(delta / 120)
                        if not units and delta:
                            units = -1 if delta > 0 else 1
                    canvas.yview_scroll(units, "units")
                    return "break"
                parent_name = widget.winfo_parent()
                if not parent_name:
                    break
                widget = widget.nametowidget(parent_name)
            return None

        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            root.bind(sequence, on_mousewheel, add="+")

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return scrollable_frame, canvas

    def build_config_from_ui():
        # Preserve backend, site provenance and physical settings which have no
        # dedicated widget; loading a preset/config is an atomic transaction.
        config = deepcopy(config_state["base"])
        config["objective"] = objective_var.get()
        for (section, key), meta in field_meta.items():
            var = field_vars[(section, key)]
            template = meta["template"]
            try:
                if isinstance(var, tk.BooleanVar):
                    value = bool(var.get())
                else:
                    text = var.get().strip()
                    if text == "":
                        raise ValueError("Blank value")
                    value = parse_value(text, template)
            except Exception as exc:
                show_error(
                    "Invalid input",
                    f"Invalid value for {section}.{key}: {exc}",
                )
                return None

            if section == "doldrums":
                config["simulation"]["doldrums"][key] = value
            else:
                config[section][key] = value

        if doldrum_windows_text["widget"] is not None:
            try:
                config["simulation"]["doldrum_windows"] = parse_doldrum_windows_text(
                    doldrum_windows_text["widget"].get("1.0", "end")
                )
            except Exception as exc:
                show_error(
                    "Invalid input",
                    f"Invalid value for simulation.doldrum_windows: {exc}",
                )
                return None

        original_source = Path(config_state["base"]["simulation"]["data_file"]).expanduser().resolve()
        current_source = Path(config["simulation"]["data_file"]).expanduser().resolve()
        if original_source != current_source:
            # A new CSV is a new location/resource history. Never label it as
            # the paper location from which the editable settings originated.
            config.pop("site", None)
        if config["solver"].get("backend") == "paper" and config["simulation"]["nhrs"] > 1:
            config["paper"]["delivery_repeat_factor"] = (
                config["simulation"]["lifespan"] / (config["simulation"]["nhrs"] - 1)
            )

        try:
            # A manually typed relative CSV path belongs to the launch cwd.
            # Preserve that resource identity when saving JSON elsewhere.
            config, _ = configuration_from_payload(build_json_payload(config, optimize_vars))
        except Exception as exc:
            show_error("Invalid input", str(exc))
            return None

        return config

    def resolve_config_path(for_save=False):
        path_text = config_path_var.get().strip()
        if not path_text:
            if for_save:
                selected = filedialog.asksaveasfilename(
                    title="Save JSON Config",
                    defaultextension=".json",
                    initialfile="sirenolite_config.json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                )
            else:
                selected = filedialog.askopenfilename(
                    title="Select JSON Config File",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                )
            if not selected:
                return None
            path_text = selected

        path = Path(path_text).expanduser()
        if path.suffix == "":
            path = path.with_suffix(".json")
        config_path_var.set(str(path))
        return path

    def on_browse_config():
        selected = filedialog.askopenfilename(
            title="Select JSON Config File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if selected:
            config_path_var.set(selected)

    def on_load_config():
        path = resolve_config_path(for_save=False)
        if path is None:
            return

        try:
            merged_config, optimize = load_configuration(path)
            apply_config_to_ui(merged_config, gui_payload={"optimize": optimize})
        except Exception as exc:
            show_error("Load failed", str(exc))
            return

        log_message(f"Loaded config: {path}")

    def on_save_config():
        path = resolve_config_path(for_save=True)
        if path is None:
            return

        config = build_config_from_ui()
        if config is None:
            log_message("Config save aborted due to invalid input.")
            return

        payload = build_json_payload(config, optimize_vars)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
        except Exception as exc:
            show_error("Save failed", str(exc))
            return

        config_path_var.set(str(path))
        log_message(f"Saved config: {path}")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    inputs_tab = ttk.Frame(notebook)
    run_tab = ttk.Frame(notebook)
    notebook.add(inputs_tab, text="Inputs")
    notebook.add(run_tab, text="Results / Run")

    scroll_frame, input_canvas = create_scrollable(inputs_tab)
    sites = presets.list_sites()
    site_by_label = {site["label"]: site for site in sites}
    default_site = next(site for site in sites if site["key"] == presets.DEFAULT_SITE_KEY)
    missions = {"Mission 1 · Desalination": "potable_water",
                "Mission 2 · Scheduled charging": "scheduled_charging",
                "Mission 3 · Constant 100 W": "pioneer_100w",
                "Mission 4 · Combined service": "combined"}
    site_var = tk.StringVar(master=root, value=default_site["label"])
    mission_var = tk.StringVar(master=root, value="Mission 4 · Combined service")
    horizon_var = tk.StringVar(master=root, value="8760")
    site_description_var = tk.StringVar(master=root)
    formulation_description_var = tk.StringVar(master=root)

    def refresh_mode_controls():
        paper_mode = config_state["base"]["solver"].get("backend") == "paper"
        objective_menu.configure(values=["total_cost"] if paper_mode else objective_options)
        for (section, key), widget in field_widgets.items():
            inactive = (section == "paper" and not paper_mode) or (paper_mode and (section == "solver" or key in (
                "batt_initial_charge", "batt_final_charge", "potable_water_initial_charge",
                "h2_initial_charge", "H2DailyDemand", "h2_Wh_per_g", "h2_min_storage_g",
                "h2_max_storage_g", "h2_max_charge_rate_gph", "h2_storage_fix_cost_per_g",
                "h2_storage_Kg_per_g")))
            widget.configure(state="disabled" if inactive else "normal")
        formulation_description_var.set(
            "Paper missions · local HiGHS · Cyclic inventories: end = start; initial levels optimized. "
            "Source sizing and dispatch minimize equipment + support + lifetime operating cost."
            if paper_mode else
            "General model · GEKKO · Initial and final inventory fractions use the editable settings below."
        )

    def describe_site(*_):
        site = site_by_label[site_var.get()]
        site_description_var.set(
            f"{site['latitude']:.2f}° latitude, {site['longitude']:.2f}° longitude · "
            "Bundled hourly resource history; no download needed."
        )

    def load_selected_site():
        if lifecycle["busy"]:
            return
        try:
            config = presets.load_site_config(site_by_label[site_var.get()]["key"],
                                             missions[mission_var.get()], int(horizon_var.get()))
            apply_config_to_ui(config)
            config_path_var.set("")
        except Exception as exc:
            show_error("Preset load failed", str(exc))
            return
        status_var.set("Paper case loaded. Select Results / Run, then Run.")
        log_message(f"Loaded {site_var.get()} / {mission_var.get()} / {horizon_var.get()} hours.")

    preset_frame = ttk.LabelFrame(scroll_frame, text="Start a simulation · location and mission", padding=8)
    preset_frame.pack(fill="x", padx=10, pady=6)
    preset_frame.columnconfigure(1, weight=1)
    preset_frame.columnconfigure(3, weight=1)
    make_label(preset_frame, text="Location").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    site_picker = ttk.Combobox(preset_frame, textvariable=site_var, values=list(site_by_label),
                               state="readonly", width=30)
    site_picker.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
    make_label(preset_frame, text="Mission").grid(row=0, column=2, sticky="w", padx=6, pady=4)
    mission_picker = ttk.Combobox(preset_frame, textvariable=mission_var, values=list(missions),
                                 state="readonly", width=29)
    mission_picker.grid(row=0, column=3, sticky="ew", padx=6, pady=4)
    make_label(preset_frame, text="Horizon (hours)").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    ttk.Combobox(preset_frame, textvariable=horizon_var, values=["168", "720", "8760"],
                 state="readonly", width=12).grid(row=1, column=1, sticky="w", padx=6, pady=4)
    load_site_button = ttk.Button(preset_frame, text="Load selected case", command=load_selected_site)
    load_site_button.grid(row=1, column=3, sticky="e", padx=6, pady=4)
    make_label(preset_frame, textvariable=site_description_var, wraplength=900).grid(
        row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
    make_label(preset_frame,
               text="Choose a paper location above, or use your own location's hourly CSV below. "
                    "Load a case to set its costs, limits, and load schedule; then edit any inputs and Run.",
               wraplength=900).grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
    ttk.Button(preset_frame, text="Run current inputs", command=lambda: on_run()).grid(
        row=4, column=3, sticky="e", padx=6, pady=4)
    make_label(preset_frame, textvariable=formulation_description_var, wraplength=900).grid(
        row=5, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
    site_var.trace_add("write", describe_site)
    describe_site()
    config_frame = ttk.LabelFrame(scroll_frame, text="Configuration File (JSON)", padding=6)
    config_frame.pack(fill="x", padx=10, pady=6)
    config_frame.columnconfigure(1, weight=1)
    make_label(config_frame, text="Configuration path").grid(row=0, column=0, sticky="w", padx=6, pady=2)
    ttk.Entry(config_frame, textvariable=config_path_var, width=60).grid(
        row=0, column=1, sticky="ew", padx=6, pady=2
    )
    ttk.Button(config_frame, text="Search...", command=on_browse_config).grid(
        row=0, column=2, sticky="w", padx=6, pady=2
    )
    ttk.Button(config_frame, text="Load", command=on_load_config).grid(
        row=0, column=3, sticky="w", padx=6, pady=2
    )
    ttk.Button(config_frame, text="Save", command=on_save_config).grid(
        row=0, column=4, sticky="w", padx=6, pady=2
    )
    make_label(
        config_frame,
        text=(
            "Load replaces all settings, including sizing choices. Relative CSV paths are resolved "
            "from the JSON folder. Save records the current inputs and sizing choices."
        ),
        wraplength=700,

    ).grid(row=1, column=1, columnspan=4, sticky="w", padx=6, pady=2)

    objective_frame = ttk.LabelFrame(scroll_frame, text="Objective", padding=6)
    objective_frame.pack(fill="x", padx=10, pady=6)
    objective_frame.columnconfigure(1, weight=1)
    objective_frame.columnconfigure(3, weight=1)
    make_label(objective_frame, text="Minimize").grid(row=0, column=0, sticky="w", padx=6, pady=2)
    objective_var = tk.StringVar(master=root, value=defaults["objective"])
    objective_menu = ttk.Combobox(
        objective_frame,
        textvariable=objective_var,
        values=objective_options,
        state="readonly",
        width=22,
    )
    objective_menu.grid(row=0, column=1, sticky="w", padx=6, pady=2)
    make_label(
        objective_frame,
        text="Optimization objective for the solver",
        wraplength=260,

    ).grid(row=0, column=3, sticky="w", padx=6, pady=2)
    simulation_order = [
        "data_file",
        "profile_start_offset_hours",
        "lifespan",
        "nhrs",
        "peak_load",
        "hotel_load",
        "H2DailyDemand",
        "H2ODailyDemand",
    ]
    build_section(
        scroll_frame,
        "Simulation",
        "simulation",
        defaults["simulation"],
        special={"data_file": "file"},
        order=simulation_order,
    )

    doldrums_frame = ttk.LabelFrame(scroll_frame, text="Doldrums", padding=6)
    doldrums_frame.pack(fill="x", padx=10, pady=6)
    doldrums_frame.columnconfigure(1, weight=1)
    doldrums_frame.columnconfigure(3, weight=1)
    row = 0
    add_entry(
        doldrums_frame,
        "simulation",
        "doldrum_time",
        defaults["simulation"]["doldrum_time"],
        row,
    )
    row += 1
    for key, value in defaults["simulation"]["doldrums"].items():
        add_entry(doldrums_frame, "doldrums", key, value, row)
        row += 1
    add_section_header(doldrums_frame, "Explicit outage windows", row)
    row += 1
    make_label(doldrums_frame, text="Outage windows (JSON)").grid(
        row=row, column=0, sticky="nw", padx=6, pady=2
    )
    # Give the multi-line editor and its help the panel's full width. Keeping
    # them beside the field label squeezed the help into a vertical strip at
    # the supported minimum window width.
    windows_frame = ttk.Frame(doldrums_frame)
    windows_frame.grid(row=row + 1, column=0, columnspan=4, sticky="ew", padx=6, pady=2)
    windows_frame.columnconfigure(0, weight=1)
    windows_text = tk.Text(windows_frame, height=7, width=52, wrap="word")
    windows_scroll = ttk.Scrollbar(windows_frame, orient="vertical", command=windows_text.yview)
    windows_text.configure(yscrollcommand=windows_scroll.set)
    windows_text.grid(row=0, column=0, sticky="ew")
    windows_scroll.grid(row=0, column=1, sticky="ns")
    windows_text.insert(
        "1.0",
        format_doldrum_windows_text(defaults["simulation"].get("doldrum_windows", [])),
    )
    doldrum_windows_text["widget"] = windows_text
    make_label(
        doldrums_frame,
        text=(
            'JSON list of explicit outage windows, e.g. [{"start_day": 5.5, '
            '"duration_days": 3, "resources": ["all"]}]. Set doldrum_time to 0 '
            "to use only explicit windows."
        ),
        wraplength=700,

    ).grid(row=row + 2, column=0, columnspan=4, sticky="ew", padx=6, pady=2)
    row += 1

    component_defs = [
        {
            "title": "Generator",
            "optimize_key": "generator_scale",
            "optimize_label": "Optimize generator sizing",
            "optimize_desc": "When unchecked, generator size is fixed at the minimum capacity.",
            "categories": [
                {
                    "name": "Cost",
                    "items": [
                        ("costs", "generator_fix_cost"),
                        ("costs", "tank_size"),
                        ("costs", "diesel_cost"),
                        ("costs", "diesel_energy"),
                        ("costs", "refill_service_cost"),
                    ],
                    "notes": [
                        (
                            "generator_vom_cost_perWh (derived)",
                            "Derived from diesel cost, energy, efficiency, and refill service cost.",
                        )
                    ],
                },
                {
                    "name": "Limits",
                    "items": [
                        ("limits", "generator_min_capacity"),
                        ("limits", "generator_max_capacity"),
                    ],
                },
                {"name": "Efficiency", "items": [("efficiency", "gen_eff")]},
                {"name": "Mass", "items": [("mass", "generator_Kg_per_W")]},
            ],
        },
        {
            "title": "Wind",
            "optimize_key": "wind_scale",
            "optimize_label": "Optimize wind sizing",
            "optimize_desc": "When unchecked, wind size is fixed at the minimum capacity.",
            "categories": [
                {
                    "name": "Cost",
                    "items": [
                        ("costs", "wind_fix_cost"),
                        ("costs", "wind_vom_cost_perWh"),
                    ],
                },
                {
                    "name": "Limits",
                    "items": [
                        ("limits", "wind_min_capacity"),
                        ("limits", "wind_max_capacity"),
                    ],
                },
                {
                    "name": "Efficiency",
                    "items": [],
                    "note": "Wind output is limited by the resource profile and curtailment.",
                },
                {"name": "Mass", "items": [("mass", "wind_Kg_per_W")]},
            ],
        },
        {
            "title": "Solar",
            "optimize_key": "solar_scale",
            "optimize_label": "Optimize solar sizing",
            "optimize_desc": "When unchecked, solar size is fixed at the minimum capacity.",
            "categories": [
                {
                    "name": "Cost",
                    "items": [
                        ("costs", "solar_fix_cost"),
                        ("costs", "solar_vom_cost_perWh"),
                    ],
                },
                {
                    "name": "Limits",
                    "items": [
                        ("limits", "solar_min_capacity"),
                        ("limits", "solar_max_capacity"),
                    ],
                },
                {
                    "name": "Efficiency",
                    "items": [],
                    "note": "Solar output is limited by the resource profile and curtailment.",
                },
                {"name": "Mass", "items": [("mass", "solar_Kg_per_W")]},
            ],
        },
        {
            "title": "Wave",
            "optimize_key": "wave_scale",
            "optimize_label": "Optimize wave sizing",
            "optimize_desc": "When unchecked, wave size is fixed at the minimum capacity.",
            "categories": [
                {
                    "name": "Cost",
                    "items": [
                        ("costs", "wave_fix_cost"),
                        ("costs", "wave_vom_cost_perWh"),
                    ],
                },
                {
                    "name": "Limits",
                    "items": [
                        ("limits", "wave_min_capacity"),
                        ("limits", "wave_max_capacity"),
                    ],
                },
                {
                    "name": "Efficiency",
                    "items": [],
                    "note": "Wave output is limited by the resource profile and curtailment.",
                },
                {"name": "Mass", "items": [("mass", "wave_Kg_per_W")]},
            ],
        },
        {
            "title": "Floating Platform",
            "optimize_key": None,
            "optimize_label": "",
            "optimize_desc": "",
            "categories": [
                {
                    "name": "Cost",
                    "items": [("costs", "floating_platform_cost_per_kg")],
                },
                {
                    "name": "Limits",
                    "items": [],
                    "note": "Derived from supported mass; no independent capacity limit.",
                },
                {
                    "name": "Efficiency",
                    "items": [],
                    "note": "No efficiency term; platform is a structural mass add-on.",
                },
                {
                    "name": "Mass",
                    "items": [("mass", "floating_platform_mass_per_supported_mass")],
                },
            ],
        },
        {
            "title": "Battery",
            "optimize_key": "batt_scale",
            "optimize_label": "Optimize battery sizing",
            "optimize_desc": "When unchecked, battery size is fixed at the minimum capacity.",
            "categories": [
                {
                    "name": "Cost",
                    "items": [
                        ("costs", "batt_fix_cost"),
                        ("costs", "batt_vom_cost_perWh"),
                    ],
                },
                {
                    "name": "Limits",
                    "items": [
                        ("limits", "batt_min_capacity"),
                        ("limits", "batt_max_capacity"),
                        ("limits", "batt_min_storage_level"),
                        ("limits", "batt_max_charge_rate"),
                    ],
                },
                {
                    "name": "Efficiency",
                    "items": [
                        ("efficiency", "batt_round_trip_eff"),
                        ("efficiency", "batt_initial_charge"),
                        ("efficiency", "batt_final_charge"),
                    ],
                },
                {"name": "Mass", "items": [("mass", "batt_Kg_per_Wh")]},
            ],
        },
        {
            "title": "Hydrogen Storage",
            "optimize_key": "h2_storage_scale",
            "optimize_label": "Optimize hydrogen storage sizing",
            "optimize_desc": "When unchecked, H2 storage is fixed at the minimum capacity.",
            "categories": [
                {"name": "Cost", "items": [("costs", "h2_storage_fix_cost_per_g")]},
                {
                    "name": "Limits",
                    "items": [
                        ("limits", "h2_min_storage_g"),
                        ("limits", "h2_max_storage_g"),
                        ("limits", "h2_max_charge_rate_gph"),
                    ],
                },
                {
                    "name": "Efficiency",
                    "items": [("efficiency", "h2_initial_charge"), ("efficiency", "h2_Wh_per_g")],
                },
                {"name": "Mass", "items": [("mass", "h2_storage_Kg_per_g")]},
            ],
        },
        {
            "title": "Potable Water Storage",
            "optimize_key": "potable_water_storage_scale",
            "optimize_label": "Optimize potable water storage sizing",
            "optimize_desc": "When unchecked, potable water storage is fixed at the minimum capacity.",
            "categories": [
                {
                    "name": "Cost",
                    "items": [("costs", "potable_water_storage_fix_cost_per_l")],
                },
                {
                    "name": "Limits",
                    "items": [
                        ("limits", "potable_water_min_storage_l"),
                        ("limits", "potable_water_max_storage_l"),
                        ("limits", "potable_water_max_rate_gpm"),
                    ],
                },
                {
                    "name": "Efficiency",
                    "items": [
                        ("efficiency", "potable_water_initial_charge"),
                        ("efficiency", "potable_water_Wh_per_l"),
                    ],
                },
                {
                    "name": "Mass",
                    "items": [("mass", "potable_water_storage_Kg_per_l")],
                },
            ],
        },
    ]

    for component in component_defs:
        build_component_section(
            scroll_frame,
            component["title"],
            component["optimize_key"],
            component["optimize_label"],
            component["optimize_desc"],
            component["categories"],
        )

    build_section(scroll_frame, "Paper missions · diesel installation and deliveries", "paper", defaults["paper"],
                  order=["generator_min_installed_w", "max_delivery_trips"])
    build_section(scroll_frame, "General model solver (GEKKO)", "solver", defaults["solver"], special={"backend": "skip"})

    run_scroll_frame = ttk.Frame(run_tab)
    run_scroll_frame.pack(fill="both", expand=True)

    run_controls = ttk.Frame(run_scroll_frame, padding=8)
    run_controls.pack(side="top", fill="x")
    run_controls.columnconfigure(2, weight=1)

    save_pdf_button = ttk.Button(
        run_controls, text="Save PDF...", command=lambda: on_save_pdf(), state="disabled"
    )
    save_pdf_button.grid(row=0, column=0, sticky="w", padx=6)
    save_csv_button = ttk.Button(
        run_controls, text="Save CSV...", command=lambda: on_save_csv(), state="disabled"
    )
    save_csv_button.grid(row=0, column=1, sticky="w", padx=6)
    status_var = tk.StringVar(master=root, value="Ready. Set inputs, then run the optimizer.")
    status_label = make_label(run_controls, textvariable=status_var, wraplength=600)
    status_label.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6, pady=(6, 0))

    log_frame = ttk.LabelFrame(run_scroll_frame, text="Run Log", padding=6)
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)
    log_text = tk.Text(log_frame, height=4, wrap="word")
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.grid(row=0, column=0, sticky="nsew")
    log_scroll.grid(row=0, column=1, sticky="ns")
    log_text.configure(state="disabled")

    results_caption_var = tk.StringVar(master=root, value="Select a location and mission on Inputs, then Run.")
    result_caption = make_label(run_scroll_frame, textvariable=results_caption_var,
                                wraplength=1000, padding=(16, 6))
    result_caption.pack(side="top", fill="x")

    plot_frame = ttk.Frame(run_scroll_frame)
    plot_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 6))

    canvas_container = ttk.Frame(plot_frame)
    canvas_container.pack(fill="both", expand=True)

    summary_frame = ttk.LabelFrame(run_scroll_frame, text="Design Summary", padding=6)
    summary_frame.columnconfigure(0, weight=1)
    summary_frame.rowconfigure(0, weight=1)
    summary_text = tk.Text(summary_frame, height=5, wrap="word")
    summary_scroll = ttk.Scrollbar(summary_frame, orient="vertical", command=summary_text.yview)
    summary_text.configure(yscrollcommand=summary_scroll.set)
    summary_text.grid(row=0, column=0, sticky="nsew")
    summary_scroll.grid(row=0, column=1, sticky="ns")
    summary_text.configure(state="disabled")

    details_var = tk.BooleanVar(master=root, value=False)

    def toggle_details():
        if details_var.get():
            summary_frame.pack(side="top", fill="x", padx=10, pady=3, before=plot_frame)
            log_frame.pack(side="top", fill="x", padx=10, pady=3, before=plot_frame)
        else:
            summary_frame.pack_forget()
            log_frame.pack_forget()

    ttk.Checkbutton(run_controls, text="Show details / log", variable=details_var,
                    command=toggle_details).grid(row=0, column=2, sticky="w", padx=6)

    current_canvas = {"canvas": None, "toolbar": None}

    def log_append(message, add_newline=True):
        log_text.configure(state="normal")
        log_text.insert("end", message)
        if add_newline and not message.endswith("\n"):
            log_text.insert("end", "\n")
        log_text.configure(state="disabled")
        log_text.see("end")

    def log_message(message):
        log_append(message, add_newline=True)

    events = Queue()
    lifecycle = {"busy": False, "closed": False, "close_requested": False, "started_at": None,
                 "poll_id": None, "worker": None}

    def process_events():
        # This is the only path from the worker to Tk and Matplotlib. Both
        # libraries stay on the thread that owns the window, including after().
        while True:
            try:
                event, payload = events.get_nowait()
            except Empty:
                break
            if event == "log":
                log_message(payload)
            elif lifecycle["close_requested"]:
                # Keep the event loop alive until GEKKO has returned. Closing
                # a daemon worker mid-solve could otherwise orphan its solver.
                lifecycle["busy"] = False
                on_close()
                return
            elif event == "success":
                try:
                    vars_, config = payload
                    fig = create_result_figure(config, vars_)
                    finish_success(fig, vars_, config)
                except Exception as exc:
                    finish_error(str(exc))
            elif event == "error":
                finish_error(payload)
        if not lifecycle["closed"]:
            if lifecycle["busy"]:
                elapsed = int(time.monotonic() - lifecycle["started_at"])
                if lifecycle["close_requested"]:
                    status_var.set(f"Closing after the current solve finishes ({elapsed} s elapsed).")
                else:
                    status_var.set(f"Running: {elapsed} s elapsed. Inputs are locked until completion.")
            lifecycle["poll_id"] = root.after(100, process_events)

    def update_summary(vars_, config):
        def fmt(value):
            return f"{float(value):.4g}"

        batt_max = float(model.to_array(vars_["batt_storagelevel"].value).max())
        h2_max = float(model.to_array(vars_["h2_storagelevel"].value).max())
        water_max = float(model.to_array(vars_["potable_water_storagelevel"].value).max())
        supported_mass = float(vars_["supported_mass"].value[-1])
        platform_mass = float(vars_["floating_platform_mass"].value[-1])
        platform_cost = platform_mass * float(config["costs"]["floating_platform_cost_per_kg"])

        lines = [
            f"Objective: {config['objective']}",
            f"Hourly samples: {config['simulation']['nhrs']} (span {config['simulation']['nhrs'] - 1} hr)",
            f"Total Cost ($): {fmt(vars_['total_cost'].value[-1])}",
            f"Total Mass (kg): {fmt(vars_['total_mass'].value[-1])}",
            f"Generator capacity (W): {fmt(vars_['generator_scale'].value[-1])}",
            f"Wind capacity (W): {fmt(vars_['wind_scale'].value[-1])}",
            f"Solar capacity (W): {fmt(vars_['solar_scale'].value[-1])}",
            f"Wave capacity (W): {fmt(vars_['wave_scale'].value[-1])}",
            f"Battery capacity (Wh): {fmt(vars_['batt_scale'].value[-1])}",
            f"H2 storage capacity (g): {fmt(vars_['h2_storage_scale'].value[-1])}",
            f"Water storage capacity (L): {fmt(vars_['potable_water_storage_scale'].value[-1])}",
            f"Cost per Watt ($/W): {fmt(vars_['cost_per_watt'].value[-1])}",
            f"Cost per Mass ($/kg): {fmt(vars_['cost_per_mass'].value[-1])}",
            f"Supported Mass (kg): {fmt(supported_mass)}",
            f"Floating Platform Mass (kg): {fmt(platform_mass)}",
            f"Floating Platform Cost ($): {fmt(platform_cost)}",
            f"Peak battery inventory (Wh): {fmt(batt_max)}",
            f"Peak H2 inventory (g): {fmt(h2_max)}",
            f"Peak water inventory (L): {fmt(water_max)}",
            f"Avg Gen (W): {fmt(vars_['avg_generator_W'].value[-1])}",
            f"Avg Wind (W): {fmt(vars_['avg_wind_W'].value[-1])}",
            f"Avg Solar (W): {fmt(vars_['avg_sol_W'].value[-1])}",
            f"Avg Wave (W): {fmt(vars_['avg_wave_W'].value[-1])}",
        ]
        summary_text.configure(state="normal")
        summary_text.delete("1.0", "end")
        if "paper_summary" in vars_:
            lines.append("Diesel delivery trips per modeled horizon: "
                         f"{vars_['paper_summary']['delivery_trips']}")
        summary_text.insert("end", "\n".join(lines))
        summary_text.configure(state="disabled")
        site = config.get("site", {})
        label = site.get("label", "Custom resource profile")
        mission_label = next((label for label, key in missions.items()
                              if key == site.get("mission_key")), "Custom mission")
        if site.get("key") and any(config[section] != config_state["base"][section]
                                    for section in ("simulation", "costs", "limits", "efficiency", "mass")):
            mission_label += " (edited inputs)"
        cost = float(vars_["total_cost"].value[-1])
        duration = config["simulation"]["nhrs"] - 1
        results_caption_var.set(
            f"{label} · {mission_label} · {duration:,} hours\n"
            f"Objective: {config['objective']} · "
            f"Total system cost: ${cost:,.0f} · Total mass: {float(vars_['total_mass'].value[-1]):,.1f} kg\n"
            "Source pies show dispatched energy. Product storage uses electricity-to-produce equivalents."
        )

    def render_figure(fig):
        if current_canvas["toolbar"] is not None:
            current_canvas["toolbar"].destroy()
        if current_canvas["canvas"] is not None:
            current_canvas["canvas"].get_tk_widget().destroy()

        canvas = _PlotCanvas(fig, master=canvas_container)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, canvas_container)
        toolbar.update()
        # Let Tk establish the embedded size before constrained layout runs.
        root.update_idletasks()
        canvas.draw()
        current_canvas["canvas"] = canvas
        current_canvas["toolbar"] = toolbar

    last_result = {"fig": None, "vars": None, "config": None}

    def clear_results(*_):
        had_results = last_result["vars"] is not None
        for key in last_result:
            last_result[key] = None
        if current_canvas["canvas"] is not None:
            # Clearing a Figure updates its toolbar; do this before destroying
            # the toolbar's Tk buttons.
            current_canvas["canvas"].figure.clear()
        if current_canvas["toolbar"] is not None:
            current_canvas["toolbar"].destroy()
            current_canvas["toolbar"] = None
        if current_canvas["canvas"] is not None:
            current_canvas["canvas"].get_tk_widget().destroy()
            current_canvas["canvas"] = None
        summary_text.configure(state="normal")
        summary_text.delete("1.0", "end")
        summary_text.configure(state="disabled")
        results_caption_var.set("Run the optimizer to see the source mix, storage, and dispatch together.")
        save_pdf_button.config(state="disabled")
        save_csv_button.config(state="disabled")
        if had_results:
            status_var.set("Inputs changed. Run again to generate current results.")

    def set_busy(busy):
        lifecycle["busy"] = busy
        lifecycle["started_at"] = time.monotonic() if busy else None
        run_button.config(state="disabled" if busy else "normal")
        notebook.tab(inputs_tab, state="disabled" if busy else "normal")

    def finish_success(fig, vars_, config):
        render_figure(fig)
        update_summary(vars_, config)
        last_result["fig"] = fig
        last_result["vars"] = vars_
        last_result["config"] = config
        save_pdf_button.config(state="normal")
        save_csv_button.config(state="normal")
        log_message("Solve complete. Outputs describe this run's input settings.")
        status_var.set("Solve complete. CSV and PDF exports are ready.")
        set_busy(False)

    def finish_error(error_message):
        clear_results()
        status_var.set("Run failed. Review the log and inputs, then try again.")
        set_busy(False)
        detail = error_message
        if "Solution Not Found" in error_message:
            detail = (
                "The solver did not find a solution. Check generation and storage limits, "
                "production rates, outage windows, and solver settings. A failed solve "
                "does not prove that no feasible design exists.\n\n" + error_message
            )
        show_error("Run failed", detail)
        log_message(f"Run failed: {error_message}")

    def require_results():
        if (
            last_result["fig"] is None
            or last_result["vars"] is None
            or last_result["config"] is None
        ):
            messagebox.showinfo("No results", "Run the optimizer before saving outputs.")
            return None
        return last_result

    def on_save_pdf(path=None):
        result = require_results()
        if result is None:
            return
        nhrs = int(result["config"]["simulation"]["nhrs"])
        path = path or filedialog.asksaveasfilename(
            title="Save Plots PDF",
            defaultextension=".pdf",
            initialfile=f"solution_{nhrs}_plots.pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            result["fig"].savefig(path)
        except Exception as exc:
            show_error("Save failed", str(exc))
            return
        log_message(f"Saved PDF: {path}")

    def on_save_csv(path=None):
        result = require_results()
        if result is None:
            return
        nhrs = int(result["config"]["simulation"]["nhrs"])
        path = path or filedialog.asksaveasfilename(
            title="Save Timeseries CSV",
            defaultextension=".csv",
            initialfile=f"solution_{nhrs}_timeseries.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            model.build_timeseries_dataframe(result["config"], result["vars"]).to_csv(
                path, index=False
            )
        except Exception as exc:
            show_error("Save failed", str(exc))
            return
        log_message(f"Saved CSV: {path}")

    def on_run():
        if lifecycle["busy"]:
            return
        config = build_config_from_ui()
        if config is None:
            log_message("Input error.")
            return
        # Snapshot every widget on the UI thread. Fixing bounds before model
        # construction also preserves correct disabled-component equations.
        try:
            config = prepare_config(config, {key: bool(var.get()) for key, var in optimize_vars.items()})
        except Exception as exc:
            show_error("Invalid input", str(exc))
            return
        clear_results()
        # Destroyed Matplotlib/Tk widgets can remain in reference cycles.
        # Collect them on the Tk thread before the solver worker allocates;
        # otherwise its garbage collection may finalize Tk objects off-thread.
        gc.collect()
        notebook.select(run_tab)
        set_busy(True)
        status_var.set("Running. Inputs are locked until the solve finishes.")
        backend = config["solver"].get("backend", "gekko")
        solver_name = "local paper model" if backend == "paper" else (
            f"{'remote' if config['solver']['remote'] else 'local'} GEKKO")
        log_message(f"Running {config['objective']} with {solver_name}...")

        def worker():
            model_tuple = None
            try:
                events.put(("log", "Loading data..."))
                inputs = model.load_resource_inputs(config)
                events.put(("log", "Building model..."))
                model_tuple = model.build_model(config, inputs)
                _, vars_ = model_tuple
                events.put(("log", "Running solver..."))
                model.solve_model(config, model_tuple)
                events.put(("success", (vars_, config)))
            except Exception as exc:
                events.put(("error", str(exc)))
            finally:
                if model_tuple is not None:
                    model_tuple[0].cleanup()

        lifecycle["worker"] = threading.Thread(target=worker, daemon=True)
        lifecycle["worker"].start()

    def on_close():
        if lifecycle["busy"]:
            lifecycle["close_requested"] = True
            status_var.set("Closing after the current solve finishes. Please wait.")
            log_message("Close requested; waiting for the solver to finish.")
            return
        lifecycle["closed"] = True
        if lifecycle["poll_id"] is not None:
            root.after_cancel(lifecycle["poll_id"])
        clear_results()
        root.destroy()

    def windows_changed(_event):
        if windows_text.edit_modified():
            clear_results()
            windows_text.edit_modified(False)

    run_button = ttk.Button(run_controls, text="Run", command=on_run)
    run_button.grid(row=0, column=2, sticky="e", padx=6)
    for var in [objective_var, *field_vars.values(), *optimize_vars.values()]:
        var.trace_add("write", clear_results)
    windows_text.edit_modified(False)
    windows_text.bind("<<Modified>>", windows_changed)
    for label in wrapping_labels:
        if label.winfo_manager() == "grid":
            label.grid_configure(sticky="ew")
    refresh_mode_controls()
    root.protocol("WM_DELETE_WINDOW", on_close)
    process_events()
    return SimpleNamespace(
        root=root, field_vars=field_vars, optimize_vars=optimize_vars,
        objective_var=objective_var, config_path_var=config_path_var,
        windows_text=windows_text, notebook=notebook, inputs_tab=inputs_tab,
        run_tab=run_tab, input_canvas=input_canvas, run_button=run_button, save_pdf_button=save_pdf_button,
        save_csv_button=save_csv_button, summary_text=summary_text,
        log_text=log_text, status_var=status_var, status_label=status_label, last_result=last_result,
        lifecycle=lifecycle, current_canvas=current_canvas,
        apply_config=apply_config_to_ui, build_config=build_config_from_ui,
        load_config=on_load_config, save_config=on_save_config,
        run=on_run, save_pdf=on_save_pdf, save_csv=on_save_csv, close=on_close,
        site_var=site_var, mission_var=mission_var, horizon_var=horizon_var,
        site_picker=site_picker, mission_picker=mission_picker,
        load_site=load_selected_site, load_site_button=load_site_button,
        results_caption_var=results_caption_var, details_var=details_var,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SIRENO-lite desktop optimizer")
    parser.add_argument("--smoke-test", metavar="OUTPUT_DIR", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.smoke_test:
        from sirenolite.gui_smoke import run_smoke_test
        run_smoke_test(args.smoke_test)
        return
    app = create_app()
    app.root.mainloop()


if __name__ == "__main__":
    main()
