import types

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sirenolite import model


def test_default_inputs_and_update_derived_config():
    config = model.default_inputs()
    assert "costs" in config
    assert config["costs"]["generator_vom_cost_perWh"] is not None
    assert config["costs"]["floating_platform_cost_per_kg"] > 0
    assert config["mass"]["floating_platform_mass_per_supported_mass"] > 0

    config["costs"]["generator_vom_cost_perWh"] = None
    config["limits"]["h2_min_storage_g"] = 123.0
    config["limits"]["h2_max_storage_g"] = 456.0
    config["limits"]["potable_water_min_storage_l"] = 7.0
    config["limits"]["potable_water_max_storage_l"] = 89.0
    model.update_derived_config(config)
    assert config["costs"]["generator_vom_cost_perWh"] is not None
    assert config["limits"]["h2_min_storage_g"] == 123.0
    assert config["limits"]["h2_max_storage_g"] == 456.0
    assert config["limits"]["potable_water_min_storage_l"] == 7.0
    assert config["limits"]["potable_water_max_storage_l"] == 89.0


def test_build_doldrum_mask():
    mask = model.build_doldrum_mask(10, center_frac=0.5, half_window=2)
    assert mask.sum() == 6
    assert np.all(mask[3:7] == 0)


def _write_resource_csv(path, use_wave=True, nrows=4, time_step=1.0):
    data = {
        "Time": [i * time_step for i in range(nrows)],
        "Load": [10 + (i % 4) for i in range(nrows)],
        "Solar": [4 + (i % 3) for i in range(nrows)],
        "Wind": [3 + (i % 2) for i in range(nrows)],
        "Hydrogen": [1 + (i % 2) for i in range(nrows)],
        "PotableWater": [5 + (i % 2) for i in range(nrows)],
    }
    if use_wave:
        data["Wave"] = [2, 2, 2, 2]
    else:
        data["Water"] = [2, 2, 2, 2]
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def test_load_resource_inputs_wave_required(tmp_path):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    config["simulation"]["doldrums"] = {"wind": 0.5, "solar": 0.5, "wave": 0.5}

    wave_path = tmp_path / "wave.csv"
    _write_resource_csv(wave_path, use_wave=True)
    config["simulation"]["data_file"] = str(wave_path)
    inputs = model.load_resource_inputs(config)
    assert "Wave_unitfactor" in inputs
    assert np.all(np.isfinite(inputs["Wave_unitfactor"]))

    water_path = tmp_path / "water.csv"
    _write_resource_csv(water_path, use_wave=False)
    config["simulation"]["data_file"] = str(water_path)
    try:
        model.load_resource_inputs(config)
        assert False, "Expected missing Wave column to raise ValueError."
    except ValueError as exc:
        assert "Wave" in str(exc)


def test_load_resource_inputs_loops_and_warns(tmp_path):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 6
    config["simulation"]["doldrums"] = {"wind": 0.5, "solar": 0.5, "wave": 0.5}

    csv_path = tmp_path / "short.csv"
    _write_resource_csv(csv_path, use_wave=True, nrows=4, time_step=0.5)
    config["simulation"]["data_file"] = str(csv_path)

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inputs = model.load_resource_inputs(config)

    assert len(inputs["Wave_unitfactor"]) == 6
    messages = [str(w.message) for w in caught]
    assert any("looping data" in msg for msg in messages)
    assert any("1-hour increments" in msg for msg in messages)


def test_scale_for_plot_and_format_label():
    series = np.array([1.0, 2.0, 3.0])
    scaled, scale = model.scale_for_plot(series, 30.0, enable_scaling=True)
    assert np.isclose(scale, 10.0)
    assert np.allclose(scaled, series * 10.0)

    scaled, scale = model.scale_for_plot(series, 30.0, enable_scaling=True, order_of_magnitude=True)
    assert scale in (10.0, 100.0)

    scaled, scale = model.scale_for_plot(series, 30.0, enable_scaling=False)
    assert scale == 1.0

    scaled, scale = model.scale_for_plot(np.zeros(3), 0.0, enable_scaling=True)
    assert scale == 1.0

    assert model.format_label("Load", "W", 1.0) == "Load (W)"
    assert "1e" in model.format_label("Wave", "W", 1000.0)
    assert "x" in model.format_label("Wave", "W", 2.5)


def test_build_timeseries_dataframe(model_vars_factory):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 6
    vars_ = model_vars_factory(6)
    df = model.build_timeseries_dataframe(config, vars_)
    assert len(df) == 6
    assert "wave_power_W" in df.columns
    assert "potable_storage_Wh_equiv" in df.columns


def test_build_model_objectives():
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 4
    config["solver"]["remote"] = False
    model.update_derived_config(config)
    inputs = {
        "Wind_unitfactor": np.ones(4),
        "Solar_unitfactor": np.ones(4),
        "Wave_unitfactor": np.ones(4),
        "Load": np.ones(4) * 10.0,
        "Hydrogen_demand_g": np.ones(4) * 2.0,
        "PotableWater_demand_l": np.ones(4) * 3.0,
    }
    m, vars_ = model.build_model(config, inputs)
    assert vars_["model"] is m
    assert "wave_scale" in vars_
    assert "supported_mass" in vars_
    assert "floating_platform_mass" in vars_

    config["objective"] = "cost_per_watt"
    m, vars_ = model.build_model(config, inputs)
    assert vars_["model"] is m


def test_solve_model_sets_options():
    class DummyModel:
        def __init__(self):
            self.options = types.SimpleNamespace()
            self.solved = False

        def solve(self):
            self.solved = True

    dummy = DummyModel()
    config = model.default_inputs()
    config["solver"]["imode"] = 7
    config["solver"]["max_iter"] = 99
    config["solver"]["solver"] = 5
    config["solver"]["cv_type"] = 2
    model.solve_model(config, (dummy, None))
    assert dummy.options.IMODE == 7
    assert dummy.options.MAX_ITER == 99
    assert dummy.options.SOLVER == 5
    assert dummy.options.CV_TYPE == 2
    assert dummy.solved is True


def test_report_results_and_create_stacked_figure(monkeypatch, model_vars_factory):
    config = model.default_inputs()
    config["simulation"]["nhrs"] = 6
    vars_ = model_vars_factory(6)

    saved = []

    def fake_savefig(*args, **kwargs):
        saved.append(args[0] if args else None)

    monkeypatch.setattr(plt, "savefig", fake_savefig)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    model.report_results(config, vars_)
    assert len(saved) == 3

    from matplotlib.figure import Figure

    fig_saved = []

    def fake_fig_save(self, *args, **kwargs):
        fig_saved.append(args[0] if args else None)

    monkeypatch.setattr(Figure, "savefig", fake_fig_save)
    fig = model.create_stacked_figure(config, vars_, save_pdf=True, figsize=(6, 4))
    assert len(fig.axes) == 4
    assert fig_saved


def test_main_invokes_subroutines(monkeypatch):
    calls = []

    def fake_default_inputs():
        return {}

    def fake_load(config):
        calls.append("load")
        return {}

    def fake_build(config, inputs):
        calls.append("build")
        return ("model", {"vars": 1})

    def fake_solve(config, model_tuple):
        calls.append("solve")

    def fake_report(config, vars_):
        calls.append("report")

    monkeypatch.setattr(model, "default_inputs", fake_default_inputs)
    monkeypatch.setattr(model, "load_resource_inputs", fake_load)
    monkeypatch.setattr(model, "build_model", fake_build)
    monkeypatch.setattr(model, "solve_model", fake_solve)
    monkeypatch.setattr(model, "report_results", fake_report)

    model.main()
    assert calls == ["load", "build", "solve", "report"]
