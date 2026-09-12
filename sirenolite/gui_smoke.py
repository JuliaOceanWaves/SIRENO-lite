"""Native bundle acceptance check through actual desktop widgets and solver."""

import json
from pathlib import Path
import time
import tkinter as tk

import numpy as np
import pandas as pd

from sirenolite import model
from sirenolite.config import CAPACITY_LIMITS
from sirenolite.gui import create_app


def run_smoke_test(output_dir):
    """Solve an analytically known wind-only case and export the displayed run.

    Invoked by the frozen executable's --smoke-test option. This deliberately
    includes Tk construction, worker/event-loop handoff, plotting, CSV and PDF.
    It raises on a failed run or a physically incorrect result.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "smoke_report.json"
    report = {"passed": False, "case": "constant 10 W load, wind only"}
    root = None
    app = None
    try:
        csv_path = output_dir / "resources.csv"
        pd.DataFrame({
            "Time": range(5), "Load": [1.0] * 5, "Solar": [1.0] * 5,
            "Wind": [1.0] * 5, "Wave": [1.0] * 5,
            "Hydrogen": [1.0] * 5, "PotableWater": [1.0] * 5,
        }).to_csv(csv_path, index=False)
        config = model.default_inputs()
        config["solver"]["remote"] = False
        config["objective"] = "total_mass"
        config["simulation"].update(
            nhrs=5, data_file=str(csv_path), peak_load=10.0,
            H2DailyDemand=0.0, H2ODailyDemand=0.0, doldrum_time=0.0,
        )
        for minimum, maximum in CAPACITY_LIMITS.values():
            config["limits"][minimum] = 0.0
            config["limits"][maximum] = 0.0
        config["limits"]["wind_max_capacity"] = 100.0
        errors = []
        root = tk.Tk()
        root.withdraw()
        app = create_app(root, show_error=lambda title, detail: errors.append(f"{title}: {detail}"))
        app.apply_config(config)
        root.update()
        app.run_button.invoke()
        deadline = time.monotonic() + 90.0
        while app.lifecycle["busy"] and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        root.update()
        if app.lifecycle["busy"]:
            raise RuntimeError("Desktop solve exceeded 90 seconds")
        if errors or app.last_result["vars"] is None:
            raise RuntimeError("; ".join(errors) or "No solved GUI result")
        result = app.last_result
        capacity = float(result["vars"]["wind_scale"].value[-1])
        if not np.isclose(capacity, 10.0, atol=1e-5, rtol=0):
            raise AssertionError(f"Wind capacity {capacity} W differs from analytical 10 W")
        app.save_csv(output_dir / "timeseries.csv")
        app.save_pdf(output_dir / "plots.pdf")
        app.config_path_var.set(str(output_dir / "config.json"))
        app.save_config()
        if errors:
            raise RuntimeError("; ".join(errors))
        data = pd.read_csv(output_dir / "timeseries.csv")
        residual = float(np.max(np.abs(data["power_balance_W"].iloc[1:])))
        if residual > 1e-5:
            raise AssertionError(f"Electrical balance residual is {residual} W")
        if not (output_dir / "plots.pdf").read_bytes().startswith(b"%PDF-"):
            raise AssertionError("GUI PDF export is invalid")
        report.update(passed=True, wind_capacity_W=capacity, max_power_balance_residual_W=residual,
                      samples=len(data), tk_version=root.tk.call("info", "patchlevel"))
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()
