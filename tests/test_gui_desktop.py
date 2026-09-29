"""Real Tk widgets, the event loop, and local GEKKO; no widget/solver doubles.

Run on a desktop (or Linux under xvfb-run). SIRENOLITE_REQUIRE_GUI_TESTS=1 turns a
missing display into a failure for the desktop release/CI gate.
"""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import tkinter as tk

import numpy as np
import pandas as pd
import pytest

from sirenolite import gui, model
from sirenolite.config import CAPACITY_LIMITS


pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def desktop_available():
    # macOS can abort the process rather than raise TclError when AppKit is
    # unavailable. Probe in a child so headless tests report the real limitation.
    probe = subprocess.run(
        [sys.executable, "-c", "import tkinter as t; r=t.Tk(); r.destroy()"],
        capture_output=True, text=True, timeout=20,
    )
    if probe.returncode:
        reason = "Actual Tk display unavailable: " + (probe.stderr.splitlines() or [str(probe.returncode)])[0]
        if os.environ.get("SIRENOLITE_REQUIRE_GUI_TESTS") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


@pytest.fixture(scope="module")
def desktop_app(desktop_available):
    # Tk on macOS owns process-global AppKit state; use one Tcl interpreter
    # for the desktop session, as the production application does.
    errors = []
    root = tk.Tk()
    root.withdraw()
    callback_errors = []
    root.report_callback_exception = lambda kind, value, _tb: callback_errors.append(f"{kind.__name__}: {value}")
    controller = gui.create_app(root, show_error=lambda title, text: errors.append((title, text)))
    controller.errors = errors
    controller.callback_errors = callback_errors
    root.update()
    try:
        yield controller
    finally:
        if not controller.lifecycle["closed"]:
            controller.close()
        # Pytest retains the yielded controller until after fixture teardown.
        # Release its references now, then collect cyclic Tk/Matplotlib wrappers
        # on the thread that owns Tcl, before subsequent browser/solver threads
        # can trigger collection of their Variable/Image finalizers.
        controller.__dict__.clear()
        del controller, root
        gc.collect()


@pytest.fixture
def app(desktop_app, monkeypatch):
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *_: None)
    desktop_app.errors.clear()
    desktop_app.callback_errors.clear()
    yield desktop_app
    worker = desktop_app.lifecycle["worker"]
    if worker is not None:
        worker.join(timeout=20)
        assert not worker.is_alive(), "GUI solve worker did not finish"
    gc.collect()
    assert desktop_app.callback_errors == [], "Unhandled Tk callback errors"


def local_config(tmp_path, **simulation):
    config = model.default_inputs()
    csv_path = tmp_path / "resources.csv"
    pd.DataFrame({
        "Time": range(4), "Load": [1] * 4, "Solar": [1] * 4,
        "Wind": [1] * 4, "Wave": [1] * 4,
        "Hydrogen": [1] * 4, "PotableWater": [1] * 4,
    }).to_csv(csv_path, index=False)
    config["solver"]["remote"] = False
    config["objective"] = "total_mass"
    config["simulation"].update(
        nhrs=4, data_file=str(csv_path), peak_load=10.0,
        H2DailyDemand=0.0, H2ODailyDemand=0.0, doldrum_time=0.0,
    )
    config["simulation"].update(simulation)
    for minimum, maximum in CAPACITY_LIMITS.values():
        config["limits"][minimum] = 0.0
        config["limits"][maximum] = 0.0
    config["limits"]["generator_max_capacity"] = 100.0
    model.update_derived_config(config)
    return config


def wait_for_run(app):
    deadline = time.monotonic() + 30
    while app.lifecycle["busy"] and time.monotonic() < deadline:
        app.root.update()
        time.sleep(0.01)
    app.root.update()
    assert not app.lifecycle["busy"], "Desktop solve did not finish within 30 seconds"


def test_widget_config_roundtrip_and_atomic_load(app, tmp_path):
    config = local_config(tmp_path)
    config["limits"]["batt_max_capacity"] = 12.5
    app.apply_config(config, {"optimize": {"generator_scale": False, "wind_scale": False}})
    app.windows_text.delete("1.0", "end")
    app.windows_text.insert("1.0", '[{"start_hour": 1, "stop_hour": 2, "resource": "wind"}]')
    app.config_path_var.set(str(tmp_path / "saved.json"))
    app.save_config()
    saved = json.loads((tmp_path / "saved.json").read_text())
    assert saved["limits"]["batt_max_capacity"] == 12.5
    assert saved["gui"]["optimize"]["generator_scale"] is False
    assert saved["simulation"]["doldrum_windows"][0]["stop_hour"] == 2

    # Load paths from the JSON location, even if launched from another cwd.
    saved["simulation"]["data_file"] = "resources.csv"
    (tmp_path / "saved.json").write_text(json.dumps(saved))
    app.field_vars[("limits", "batt_max_capacity")].set("999")
    app.load_config()
    assert app.build_config()["limits"]["batt_max_capacity"] == 12.5
    assert Path(app.build_config()["simulation"]["data_file"]) == tmp_path / "resources.csv"
    before = app.build_config()

    saved["simulation"]["peak_load"] = 999.0
    saved["gui"]["optimize"]["wind_scale"] = "perhaps"
    (tmp_path / "saved.json").write_text(json.dumps(saved))
    app.load_config()
    assert app.errors[-1][0] == "Load failed"
    assert app.build_config() == before
    assert app.optimize_vars["wind_scale"].get() is False

    # Missing optimize fields belong to the new file, not the previous session.
    (tmp_path / "plain.json").write_text(json.dumps({"simulation": {"nhrs": 5}}))
    app.config_path_var.set(str(tmp_path / "plain.json"))
    app.load_config()
    assert all(var.get() for var in app.optimize_vars.values())
    app.config_path_var.set(str(tmp_path / "absent.json"))
    app.load_config()
    assert not (tmp_path / "absent.json").exists()
    assert app.errors[-1][0] == "Load failed"


def test_fractional_node_count_rejected_before_worker(app, tmp_path):
    app.apply_config(local_config(tmp_path))
    app.field_vars[("simulation", "nhrs")].set("4.9")
    app.run_button.invoke()
    assert app.lifecycle["worker"] is None
    assert "simulation.nhrs" in app.errors[-1][1]
    assert app.run_button.instate(["!disabled"])


def test_location_mission_horizon_picker_loads_reproducible_settings(app, tmp_path):
    from sirenolite import presets
    site = next(item for item in presets.list_sites() if item["key"] == presets.DEFAULT_SITE_KEY)
    app.site_var.set(site["label"])
    app.mission_var.set("Mission 2 · Scheduled charging")
    app.horizon_var.set("168")
    app.load_site_button.invoke()
    config = app.build_config()
    assert not app.errors
    assert config["solver"]["backend"] == "paper"
    assert config["simulation"]["nhrs"] == 169
    assert config["site"]["mission_key"] == "scheduled_charging"
    assert config["site"]["key"] == presets.DEFAULT_SITE_KEY
    assert Path(config["simulation"]["data_file"]).is_file()
    app.config_path_var.set(str(tmp_path / "paper-case.json"))
    app.save_config()
    app.load_config()
    assert app.build_config() == config
    # Substituting an arbitrary location's CSV must remove the old site's name.
    source = tmp_path / "my-location.csv"
    source.write_bytes(Path(config["simulation"]["data_file"]).read_bytes())
    app.field_vars[("simulation", "data_file")].set(str(source))
    assert app.build_config()["site"]["key"] == ""
    app.field_vars[("simulation", "nhrs")].set("25")
    assert app.build_config()["paper"]["delivery_repeat_factor"] == pytest.approx(
        config["simulation"]["lifespan"] / 24)


def test_save_relative_resource_path_survives_moved_config(app, tmp_path, monkeypatch):
    config = local_config(tmp_path)
    app.apply_config(config)
    monkeypatch.chdir(tmp_path)
    app.field_vars[("simulation", "data_file")].set("resources.csv")
    destination = tmp_path / "configs" / "design.json"
    app.config_path_var.set(str(destination))
    app.save_config()
    assert Path(json.loads(destination.read_text())["simulation"]["data_file"]) == tmp_path / "resources.csv"
    monkeypatch.chdir(destination.parent)
    app.load_config()
    assert app.errors == []
    assert Path(app.build_config()["simulation"]["data_file"]) == tmp_path / "resources.csv"


def test_scroll_and_cancel_resource_browse_preserve_inputs(app, tmp_path, monkeypatch):
    app.apply_config(local_config(tmp_path))
    app.root.deiconify()
    app.notebook.select(app.inputs_tab)
    app.root.update()
    app.input_canvas.yview_moveto(0)
    start = app.input_canvas.yview()[0]
    app.input_canvas.event_generate("<MouseWheel>", delta=-120)
    app.root.update()
    assert app.input_canvas.yview()[0] > start
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **_: "")
    previous = app.field_vars[("simulation", "data_file")].get()

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    browse = next(widget for widget in descendants(app.inputs_tab)
                  if isinstance(widget, gui.ttk.Button) and widget.cget("text") == "Browse")
    browse.invoke()
    assert app.field_vars[("simulation", "data_file")].get() == previous
    app.root.withdraw()


def test_outage_editor_help_remains_readable_at_minimum_window_size(app):
    app.root.geometry("760x520")
    app.root.deiconify()
    app.notebook.select(app.inputs_tab)
    app.root.update()
    frame = app.windows_text.master.master
    help_label = next(child for child in frame.winfo_children()
                      if isinstance(child, gui.ttk.Label)
                      and str(child.cget("text")).startswith("JSON list of explicit outage windows"))
    # The editor and its instructions must remain usable on a small laptop:
    # the former side-by-side layout left a 60 px help column, taller than
    # the entire window, despite the panel itself fitting horizontally.
    assert app.windows_text.winfo_width() > 500
    assert help_label.winfo_width() > 500
    assert help_label.winfo_height() < 100
    app.root.withdraw()


def test_solve_export_invalidation_and_failure_recovery(app, tmp_path, monkeypatch):
    config = local_config(tmp_path)
    app.apply_config(config)
    app.root.geometry("900x650")
    app.root.deiconify()
    owner_thread = threading.get_ident()
    plot_threads = []
    make_figure = gui.create_result_figure

    def record_figure_thread(*args, **kwargs):
        plot_threads.append(threading.get_ident())
        return make_figure(*args, **kwargs)

    monkeypatch.setattr(gui, "create_result_figure", record_figure_thread)
    app.run_button.invoke()
    first_worker = app.lifecycle["worker"]
    app.run()  # duplicate commands cannot launch a second solve
    assert app.lifecycle["worker"] is first_worker
    assert app.notebook.tab(app.inputs_tab, "state") == "disabled"
    wait_for_run(app)
    assert app.errors == []
    assert plot_threads == [owner_thread]
    result = app.last_result
    assert result["vars"]["generator_scale"].value[-1] == pytest.approx(10.0, abs=1e-5)
    assert "Objective: total_mass" in app.summary_text.get("1.0", "end")
    assert app.save_csv_button.instate(["!disabled"])
    assert app.notebook.tab(app.inputs_tab, "state") == "normal"
    # At laptop width the status remains a readable one/two-line message,
    # rather than collapsing into a vertical strip or hiding the Run button.
    assert app.status_label.winfo_width() > 500
    assert app.status_label.winfo_height() < 60
    assert app.run_button.winfo_ismapped()

    csv_path, pdf_path = tmp_path / "dispatch.csv", tmp_path / "plots.pdf"
    paths = iter([str(csv_path), str(pdf_path)])
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **_: next(paths))
    app.save_csv_button.invoke()
    app.save_pdf_button.invoke()
    dispatch = pd.read_csv(csv_path)
    np.testing.assert_allclose(dispatch["generator_power_W"].iloc[1:], 10.0, atol=1e-5)
    np.testing.assert_allclose(dispatch["power_balance_W"].iloc[1:], 0.0, atol=1e-5)
    assert pdf_path.read_bytes().startswith(b"%PDF-")

    app.field_vars[("simulation", "peak_load")].set("20")
    assert app.last_result["vars"] is None
    assert app.save_csv_button.instate(["disabled"])
    assert not app.summary_text.get("1.0", "end").strip()
    app.run_button.invoke()
    wait_for_run(app)
    assert app.last_result["vars"]["generator_scale"].value[-1] == pytest.approx(20.0, abs=1e-5)

    app.field_vars[("limits", "generator_max_capacity")].set("1")
    app.run_button.invoke()
    wait_for_run(app)
    assert app.errors[-1][0] == "Run failed"
    assert app.last_result["vars"] is None
    assert app.save_pdf_button.instate(["disabled"])
    assert app.run_button.instate(["!disabled"])
    assert "Run failed:" in app.log_text.get("1.0", "end")
    app.field_vars[("limits", "generator_max_capacity")].set("100")
    app.run_button.invoke()
    wait_for_run(app)
    assert app.last_result["vars"]["generator_scale"].value[-1] == pytest.approx(20.0, abs=1e-5)


def test_all_fixed_sizing_controls_reach_actual_solver(app, tmp_path):
    # These historical keys specify peak rates, not daily totals. A constant
    # 1 g/h H2 profile requires 65 W, and 1 L/h water requires 4.5 W.
    config = local_config(tmp_path, H2DailyDemand=1.0, H2ODailyDemand=1.0)
    capacities = {
        "generator_scale": 500.0, "wind_scale": 20.0, "solar_scale": 30.0,
        "wave_scale": 40.0, "batt_scale": 100.0, "h2_storage_scale": 10.0,
        "potable_water_storage_scale": 10.0,
    }
    for name, capacity in capacities.items():
        minimum, maximum = CAPACITY_LIMITS[name]
        config["limits"][minimum] = capacity
        config["limits"][maximum] = capacity * 10
    config["efficiency"].update(batt_initial_charge=0.5, batt_final_charge=0.5,
                                h2_initial_charge=0.5, potable_water_initial_charge=0.5)
    app.apply_config(config, {"optimize": dict.fromkeys(capacities, False)})
    app.run_button.invoke()
    wait_for_run(app)
    assert app.errors == []
    for name, capacity in capacities.items():
        np.testing.assert_allclose(model.to_array(app.last_result["vars"][name].value), capacity, atol=1e-5)
        minimum, maximum = CAPACITY_LIMITS[name]
        assert app.last_result["config"]["limits"][maximum] == capacity
        # Preparing the run must not erase the editable optimization bounds.
        assert float(app.field_vars[("limits", maximum)].get()) == capacity * 10
    dispatch = model.build_timeseries_dataframe(app.last_result["config"], app.last_result["vars"])
    np.testing.assert_allclose(dispatch["power_balance_W"].iloc[1:], 0.0, atol=1e-4)
    np.testing.assert_allclose(dispatch["hydrogen_demand_g"], 1.0, atol=1e-8)
    np.testing.assert_allclose(dispatch["potable_water_demand_l"], 1.0, atol=1e-8)
    # Independently reconstruct inventory changes from delivered production
    # energy and the prescribed hourly withdrawals, including initial stock.
    np.testing.assert_allclose(np.diff(dispatch["h2_storage_g"]),
                               dispatch["h2_power_W"].iloc[1:] / 65.0 - 1.0, atol=1e-5)
    np.testing.assert_allclose(np.diff(dispatch["potable_storage_l"]),
                               dispatch["potable_power_W"].iloc[1:] / 4.5 - 1.0, atol=1e-5)
    np.testing.assert_allclose(dispatch[["battery_storage_Wh", "h2_storage_g", "potable_storage_l"]].iloc[0],
                               [50.0, 5.0, 5.0], atol=1e-5)
    battery_power = dispatch["battery_power_W"].to_numpy()[1:]
    battery_change = np.where(battery_power >= 0, -battery_power / np.sqrt(0.9),
                              -battery_power * np.sqrt(0.9))
    np.testing.assert_allclose(np.diff(dispatch["battery_storage_Wh"]), battery_change, atol=1e-5)


def test_linear_run_honors_disabled_battery_toggle(app, tmp_path):
    config = local_config(tmp_path)
    config["limits"]["batt_max_capacity"] = 1000.0
    config["solver"]["linear"] = True
    app.apply_config(config, {"optimize": {"batt_scale": False}})
    app.run_button.invoke()
    wait_for_run(app)
    assert app.errors == []
    assert app.last_result["vars"]["generator_scale"].value[-1] == pytest.approx(10)
    assert app.last_result["config"]["limits"]["batt_max_capacity"] == 0
    assert app.build_config()["limits"]["batt_max_capacity"] == 1000


def test_paper_combined_case_solves_through_widgets_and_displays_both_pies(app):
    from sirenolite import presets
    app.apply_config(presets.load_site_config(horizon_hours=168))
    app.run_button.invoke()
    wait_for_run(app)
    assert not app.errors
    result = app.last_result
    dispatch = model.build_timeseries_dataframe(result["config"], result["vars"])
    assert len(dispatch) == 169
    assert dispatch["electric_demand_W"].iloc[1:].max() > 2500
    np.testing.assert_allclose(dispatch["power_balance_W"].iloc[1:], 0, atol=1e-5)
    for column in ("battery_storage_Wh", "potable_storage_l"):
        assert dispatch[column].iloc[-1] == pytest.approx(dispatch[column].iloc[0], abs=1e-5)
    pies = [axis for axis in result["fig"].axes
            if axis.get_title() in ("Energy source", "Storage capacity")]
    assert len(pies) == 2
    assert all(axis.patches for axis in pies)
    assert "Combined service" in app.results_caption_var.get()
    assert "168 hours" in app.results_caption_var.get()
    assert result["vars"]["paper_summary"]["primary_status"]


def test_close_waits_for_actual_worker_completion(desktop_available, tmp_path):
    # Closing is exercised in its own process because it ends the Tk session.
    config = local_config(tmp_path)
    script = """
import json, sys, threading, time, tkinter as tk
from sirenolite import gui, model
started, release = threading.Event(), threading.Event()
original = model.load_resource_inputs
def paused_load(config):
    started.set()
    if not release.wait(timeout=10):
        raise TimeoutError("Worker never released")
    return original(config)
model.load_resource_inputs = paused_load
root = tk.Tk()
root.withdraw()
errors = []
app = gui.create_app(root, show_error=lambda *args: errors.append(args))
app.apply_config(json.loads(sys.argv[1]))
app.run_button.invoke()
assert started.wait(timeout=3)
app.close()
assert not app.lifecycle["closed"]
assert app.lifecycle["close_requested"]
assert app.root.winfo_exists()
release.set()
deadline = time.monotonic() + 20
while not app.lifecycle["closed"] and time.monotonic() < deadline:
    app.root.update()
    time.sleep(0.01)
app.lifecycle["worker"].join(timeout=20)
assert not app.lifecycle["worker"].is_alive()
assert app.lifecycle["closed"]
assert app.last_result["vars"] is None
assert not errors
print("worker_done")
"""
    result = subprocess.run([sys.executable, "-c", script, json.dumps(config)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "worker_done" in result.stdout
