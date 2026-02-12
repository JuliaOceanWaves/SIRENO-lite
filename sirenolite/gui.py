import threading
from queue import Queue, Empty
import tkinter as tk
from contextlib import redirect_stderr, redirect_stdout
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from sirenolite import model


def main():
    root = tk.Tk()
    root.title("SIRENO-lite (System-Integrated Resilient ENergy Optimizer)")
    root.geometry("1200x900")

    defaults = model.default_inputs()

    field_vars = {}
    field_meta = {}
    optimize_vars = {}

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
            "batt_max_ramp_up": "Battery ramp limit (W per hr)",
            "h2_min_storage_g": "H2 storage min (g)",
            "h2_max_storage_g": "H2 storage max (g)",
            "potable_water_min_storage_l": "Potable water storage min (L)",
            "potable_water_max_storage_l": "Potable water storage max (L)",
        },
        "efficiency": {
            "gen_eff": "Generator efficiency (fraction)",
            "batt_round_trip_eff": "Battery round-trip efficiency",
            "batt_final_charge": "Battery final SOC fraction",
            "h2_Wh_per_g": "H2 production energy (Wh/g)",
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
            "nhrs": "Optimization horizon (hr)",
            "peak_load": "Peak electric load (W)",
            "H2DailyDemand": "H2 daily demand (g/day)",
            "H2ODailyDemand": "Potable water daily demand (L/day)",
            "data_file": "Resource/load CSV file",
            "doldrum_time": "Doldrum half-window (hr, resource set to zero around center)",
        },
        "doldrums": {
            "wind": "Wind doldrum center (fraction of horizon, 0-1; 0.5 is midpoint)",
            "solar": "Solar doldrum center (fraction of horizon, 0-1; 0.5 is midpoint)",
            "wave": "Wave doldrum center (fraction of horizon, 0-1; 0.5 is midpoint)",
        },
        "solver": {
            "remote": "Solve on remote GEKKO server",
            "imode": "GEKKO IMODE",
            "max_iter": "Solver max iterations",
            "solver": "GEKKO solver id",
            "cv_type": "GEKKO CV type",
        },
    }

    def parse_value(text, template):
        if isinstance(template, bool):
            return bool(text)
        if isinstance(template, int) and not isinstance(template, bool):
            return int(float(text))
        if isinstance(template, float):
            return float(text)
        return text

    def add_entry(parent, section_key, key, value, row, widget_type="entry"):
        ttk.Label(parent, text=key).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        desc_text = descriptions.get(section_key, {}).get(key, "")
        if widget_type == "bool":
            var = tk.BooleanVar(value=bool(value))
            chk = ttk.Checkbutton(parent, variable=var)
            chk.grid(row=row, column=1, sticky="w", padx=6, pady=2)
            ttk.Label(parent, text=desc_text, wraplength=420, foreground="gray").grid(
                row=row, column=3, sticky="w", padx=6, pady=2
            )
            field_vars[(section_key, key)] = var
            field_meta[(section_key, key)] = {"template": value}
            return
        if widget_type == "file":
            var = tk.StringVar(value=str(value))
            entry = ttk.Entry(parent, textvariable=var, width=50)
            entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
            browse_btn = ttk.Button(
                parent,
                text="Browse",
                command=lambda v=var: v.set(
                    filedialog.askopenfilename(
                        title="Select Resource Data File",
                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                    )
                ),
            )
            browse_btn.grid(row=row, column=2, sticky="w", padx=6, pady=2)
            ttk.Label(parent, text=desc_text, wraplength=420, foreground="gray").grid(
                row=row, column=3, sticky="w", padx=6, pady=2
            )
            field_vars[(section_key, key)] = var
            field_meta[(section_key, key)] = {"template": value}
            return

        var = tk.StringVar(value=str(value))
        entry = ttk.Entry(parent, textvariable=var, width=18)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        ttk.Label(parent, text=desc_text, wraplength=420, foreground="gray").grid(
            row=row, column=3, sticky="w", padx=6, pady=2
        )
        field_vars[(section_key, key)] = var
        field_meta[(section_key, key)] = {"template": value}

    def add_note(parent, label, description, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(parent, text="(n/a)", foreground="gray").grid(
            row=row, column=1, sticky="w", padx=6, pady=2
        )
        ttk.Label(parent, text=description, wraplength=420, foreground="gray").grid(
            row=row, column=3, sticky="w", padx=6, pady=2
        )

    def add_section_header(parent, text, row):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=6, pady=(8, 2))

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
            var = tk.BooleanVar(value=True)
            optimize_vars[optimize_key] = var
            ttk.Label(frame, text=optimize_label).grid(
                row=row, column=0, sticky="w", padx=6, pady=2
            )
            ttk.Checkbutton(frame, variable=var).grid(
                row=row, column=1, sticky="w", padx=6, pady=2
            )
            ttk.Label(frame, text=optimize_desc, wraplength=420, foreground="gray").grid(
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

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return scrollable_frame, canvas

    def build_config_from_ui():
        config = model.default_inputs()
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
                messagebox.showerror(
                    "Invalid input",
                    f"Invalid value for {section}.{key}: {exc}",
                )
                return None

            if section == "doldrums":
                config["simulation"]["doldrums"][key] = value
            else:
                config[section][key] = value

        model.update_derived_config(config)

        return config

    optimize_targets = {
        "generator_scale": ("limits", "generator_min_capacity"),
        "wind_scale": ("limits", "wind_min_capacity"),
        "solar_scale": ("limits", "solar_min_capacity"),
        "wave_scale": ("limits", "wave_min_capacity"),
        "batt_scale": ("limits", "batt_min_capacity"),
        "h2_storage_scale": ("limits", "h2_min_storage_g"),
        "potable_water_storage_scale": ("limits", "potable_water_min_storage_l"),
    }

    def apply_optimize_flags(vars_, config):
        for var_name, min_key in optimize_targets.items():
            var = vars_.get(var_name)
            if var is None:
                continue
            enabled_var = optimize_vars.get(var_name)
            enabled = True if enabled_var is None else bool(enabled_var.get())
            if enabled:
                var.STATUS = 1
                continue
            var.STATUS = 0
            section, key = min_key
            fixed_value = float(config[section][key])
            var.value = fixed_value

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    inputs_tab = ttk.Frame(notebook)
    run_tab = ttk.Frame(notebook)
    notebook.add(inputs_tab, text="Inputs")
    notebook.add(run_tab, text="Run / Plot")

    scroll_frame, _ = create_scrollable(inputs_tab)
    objective_frame = ttk.LabelFrame(scroll_frame, text="Objective", padding=6)
    objective_frame.pack(fill="x", padx=10, pady=6)
    objective_frame.columnconfigure(1, weight=1)
    objective_frame.columnconfigure(3, weight=1)
    ttk.Label(objective_frame, text="objective").grid(row=0, column=0, sticky="w", padx=6, pady=2)
    objective_var = tk.StringVar(value="cost_per_watt")
    objective_menu = ttk.Combobox(
        objective_frame,
        textvariable=objective_var,
        values=["total_mass", "cost_per_watt"],
        state="readonly",
        width=22,
    )
    objective_menu.grid(row=0, column=1, sticky="w", padx=6, pady=2)
    ttk.Label(
        objective_frame,
        text="Optimization objective for the solver",
        wraplength=420,
        foreground="gray",
    ).grid(row=0, column=3, sticky="w", padx=6, pady=2)
    simulation_order = [
        "data_file",
        "lifespan",
        "nhrs",
        "peak_load",
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
                        ("limits", "batt_max_ramp_up"),
                    ],
                },
                {
                    "name": "Efficiency",
                    "items": [
                        ("efficiency", "batt_round_trip_eff"),
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
                    ],
                },
                {"name": "Efficiency", "items": [("efficiency", "h2_Wh_per_g")]},
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
                    ],
                },
                {
                    "name": "Efficiency",
                    "items": [("efficiency", "potable_water_Wh_per_l")],
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

    build_section(scroll_frame, "Solver", "solver", defaults["solver"])

    run_scroll_frame, _ = create_scrollable(run_tab)

    run_controls = ttk.Frame(run_scroll_frame, padding=8)
    run_controls.pack(side="top", fill="x")

    save_pdf_button = ttk.Button(
        run_controls, text="Save PDF...", command=lambda: on_save_pdf(), state="disabled"
    )
    save_pdf_button.pack(side="left", padx=6)
    save_csv_button = ttk.Button(
        run_controls, text="Save CSV...", command=lambda: on_save_csv(), state="disabled"
    )
    save_csv_button.pack(side="left", padx=6)

    log_frame = ttk.LabelFrame(run_scroll_frame, text="Run Log", padding=6)
    log_frame.pack(side="top", fill="x", padx=10, pady=(0, 6))
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)
    log_text = tk.Text(log_frame, height=10, wrap="word")
    log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=log_scroll.set)
    log_text.grid(row=0, column=0, sticky="nsew")
    log_scroll.grid(row=0, column=1, sticky="ns")
    log_text.configure(state="disabled")

    plot_frame = ttk.Frame(run_scroll_frame)
    plot_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 6))

    canvas_container = ttk.Frame(plot_frame)
    canvas_container.pack(fill="both", expand=True)

    summary_frame = ttk.LabelFrame(run_scroll_frame, text="Design Summary", padding=6)
    summary_frame.pack(side="top", fill="both", expand=False, padx=10, pady=6)
    summary_frame.columnconfigure(0, weight=1)
    summary_frame.rowconfigure(0, weight=1)
    summary_text = tk.Text(summary_frame, height=14, wrap="word")
    summary_scroll = ttk.Scrollbar(summary_frame, orient="vertical", command=summary_text.yview)
    summary_text.configure(yscrollcommand=summary_scroll.set)
    summary_text.grid(row=0, column=0, sticky="nsew")
    summary_scroll.grid(row=0, column=1, sticky="ns")
    summary_text.configure(state="disabled")

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

    log_queue = Queue()

    class QueueWriter:
        def __init__(self, queue):
            self.queue = queue
            self._buffer = ""

        def write(self, data):
            if not data:
                return
            self._buffer += data
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self.queue.put((line, True))

        def flush(self):
            if self._buffer:
                self.queue.put((self._buffer, True))
                self._buffer = ""

    def enqueue_log(message, add_newline=True):
        log_queue.put((message, add_newline))

    def process_log_queue():
        while True:
            try:
                message, add_newline = log_queue.get_nowait()
            except Empty:
                break
            log_append(message, add_newline=add_newline)
        root.after(100, process_log_queue)

    process_log_queue()

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
            f"Objective: {objective_var.get()}",
            f"Cost per Watt ($/W): {fmt(vars_['cost_per_watt'].value[-1])}",
            f"Total Cost ($): {fmt(vars_['total_cost'].value[-1])}",
            f"Total Mass (kg): {fmt(vars_['total_mass'].value[-1])}",
            f"Supported Mass (kg): {fmt(supported_mass)}",
            f"Floating Platform Mass (kg): {fmt(platform_mass)}",
            f"Floating Platform Cost ($): {fmt(platform_cost)}",
            f"Battery Scale (Wh): {fmt(vars_['batt_scale'].value[-1])}",
            f"Battery Max (Wh): {fmt(batt_max)}",
            f"H2 Storage Max (g): {fmt(h2_max)}",
            f"Potable Water Storage Max (L): {fmt(water_max)}",
            f"Wind Scale (W): {fmt(vars_['wind_scale'].value[-1])}",
            f"Solar Scale (W): {fmt(vars_['solar_scale'].value[-1])}",
            f"Wave Scale (W): {fmt(vars_['wave_scale'].value[-1])}",
            f"Generator Scale (W): {fmt(vars_['generator_scale'].value[-1])}",
            f"Avg Gen (W): {fmt(vars_['avg_generator_W'].value[-1])}",
            f"Avg Wind (W): {fmt(vars_['avg_wind_W'].value[-1])}",
            f"Avg Solar (W): {fmt(vars_['avg_sol_W'].value[-1])}",
            f"Avg Wave (W): {fmt(vars_['avg_wave_W'].value[-1])}",
        ]
        summary_text.configure(state="normal")
        summary_text.delete("1.0", "end")
        summary_text.insert("end", "\n".join(lines))
        summary_text.configure(state="disabled")

    def render_figure(fig):
        if current_canvas["toolbar"] is not None:
            current_canvas["toolbar"].destroy()
        if current_canvas["canvas"] is not None:
            current_canvas["canvas"].get_tk_widget().destroy()

        canvas = FigureCanvasTkAgg(fig, master=canvas_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, canvas_container)
        toolbar.update()
        current_canvas["canvas"] = canvas
        current_canvas["toolbar"] = toolbar

    last_result = {"fig": None, "vars": None, "config": None}

    def finish_success(fig, vars_, config):
        render_figure(fig)
        update_summary(vars_, config)
        last_result["fig"] = fig
        last_result["vars"] = vars_
        last_result["config"] = config
        save_pdf_button.config(state="normal")
        save_csv_button.config(state="normal")
        log_message("Done.")
        run_button.config(state="normal")

    def finish_error(error_message):
        messagebox.showerror("Run failed", error_message)
        log_message("Run failed.")
        run_button.config(state="normal")
        if last_result["fig"] is not None:
            save_pdf_button.config(state="normal")
            save_csv_button.config(state="normal")

    def require_results():
        if (
            last_result["fig"] is None
            or last_result["vars"] is None
            or last_result["config"] is None
        ):
            messagebox.showinfo("No results", "Run the optimizer before saving outputs.")
            return None
        return last_result

    def on_save_pdf():
        result = require_results()
        if result is None:
            return
        nhrs = int(result["config"]["simulation"]["nhrs"])
        path = filedialog.asksaveasfilename(
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
            messagebox.showerror("Save failed", str(exc))
            return
        log_message(f"Saved PDF: {path}")

    def on_save_csv():
        result = require_results()
        if result is None:
            return
        nhrs = int(result["config"]["simulation"]["nhrs"])
        path = filedialog.asksaveasfilename(
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
            messagebox.showerror("Save failed", str(exc))
            return
        log_message(f"Saved CSV: {path}")

    def on_run():
        log_message("Running...")
        run_button.config(state="disabled")
        save_pdf_button.config(state="disabled")
        save_csv_button.config(state="disabled")
        root.update_idletasks()

        config = build_config_from_ui()
        if config is None:
            log_message("Input error.")
            run_button.config(state="normal")
            if last_result["fig"] is not None:
                save_pdf_button.config(state="normal")
                save_csv_button.config(state="normal")
            return

        def worker():
            try:
                enqueue_log("Loading data...")
                inputs = model.load_resource_inputs(config)
                enqueue_log("Building model...")
                model_tuple = model.build_model(config, inputs)
                _, vars_ = model_tuple
                apply_optimize_flags(vars_, config)
                enqueue_log("Running solver...")
                writer = QueueWriter(log_queue)
                with redirect_stdout(writer), redirect_stderr(writer):
                    model.solve_model(config, model_tuple)
                writer.flush()
                fig = model.create_stacked_figure(config, vars_, save_pdf=False, figsize=(10, 16))
                root.after(0, finish_success, fig, vars_, config)
            except Exception as exc:
                root.after(0, finish_error, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    run_button = ttk.Button(run_controls, text="Run", command=on_run)
    run_button.pack(side="right", padx=6)

    root.mainloop()


if __name__ == "__main__":
    main()
