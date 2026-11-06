import pytest

from sirenolite.gui import (
    format_doldrum_windows_text,
    merge_overrides,
    parse_bool,
    parse_doldrum_windows_text,
    parse_value,
    split_json_payload,
)


def test_parse_doldrum_windows_text_accepts_empty_input():
    assert parse_doldrum_windows_text("") == []
    assert parse_doldrum_windows_text("  \n") == []


def test_parse_doldrum_windows_text_roundtrip():
    windows = [
        {"start_day": 5.5, "duration_days": 3, "resources": ["all"]},
        {"start_hour": 12, "stop_hour": 18, "resource": "wave"},
    ]

    text = format_doldrum_windows_text(windows)
    assert parse_doldrum_windows_text(text) == windows


@pytest.mark.parametrize(
    "payload, expected",
    [
        ("{}", "must be a JSON list"),
        ('[{"start_hour": 1, "duration_hours": 2}]', "must include resource"),
        (
            '[{"start_hour": 1, "duration_hours": 2, "resources": ["tide"]}]',
            "unsupported resource",
        ),
        (
            '[{"start_hour": 1, "duration_hours": 0, "resources": ["wind"]}]',
            "stop time must be after start time",
        ),
    ],
)
def test_parse_doldrum_windows_text_rejects_invalid_payloads(payload, expected):
    with pytest.raises(ValueError, match=expected):
        parse_doldrum_windows_text(payload)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("yes", True),
        ("Off", False),
    ],
)
def test_parse_bool_accepts_common_gui_values(value, expected):
    assert parse_bool(value) is expected


def test_parse_bool_rejects_ambiguous_text():
    with pytest.raises(ValueError, match="Cannot parse boolean"):
        parse_bool("maybe")


def test_parse_value_uses_template_type():
    assert parse_value("false", True) is False
    assert parse_value("4.0", 1) == 4
    assert parse_value("4.9", 1.0) == 4.9
    assert parse_value("unchanged", "template") == "unchanged"


@pytest.mark.parametrize("text, template", [("4.9", 1), ("nan", 1.0), ("inf", 1.0)])
def test_parse_value_rejects_fractional_counts_and_nonfinite_quantities(text, template):
    with pytest.raises(ValueError):
        parse_value(text, template)


def test_merge_overrides_recurses_without_losing_sibling_values():
    base = {"solver": {"remote": True, "max_iter": 100}, "objective": "total_mass"}
    overrides = {"solver": {"remote": False}, "objective": "total_cost"}

    result = merge_overrides(base, overrides)

    assert result is base
    assert base == {
        "solver": {"remote": False, "max_iter": 100},
        "objective": "total_cost",
    }


def test_split_json_payload_accepts_wrapped_and_legacy_shapes():
    wrapped = {"config": {"objective": "total_cost"}, "gui": {"optimize": {"wind": True}}}
    assert split_json_payload(wrapped) == (
        {"objective": "total_cost"},
        {"optimize": {"wind": True}},
    )

    legacy = {"objective": "total_mass", "gui": {"optimize": {"solar": False}}}
    assert split_json_payload(legacy) == (
        {"objective": "total_mass"},
        {"optimize": {"solar": False}},
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ([], "JSON root"),
        ({"config": []}, "config"),
        ({"gui": []}, "gui"),
    ],
)
def test_split_json_payload_rejects_invalid_shapes(payload, message):
    with pytest.raises(ValueError, match=message):
        split_json_payload(payload)


def test_result_composition_counts_source_energy_once_and_uses_storage_units(monkeypatch):
    """A nonuniform dispatch horizon exposes accidental sample-count averages.

    The fictitious large initial production and battery discharge must not enter
    source energy. Water and H2 capacities are converted from L and g before
    combining them with electrical battery energy.
    """
    import pandas as pd
    from types import SimpleNamespace
    from sirenolite import gui

    dispatch = pd.DataFrame({
        "time_days": [0, 1 / 24, 3 / 24],
        "wind_power_W": [999999, 1000, 0],
        "solar_power_W": [999999, 0, 1000],
        "wave_power_W": [999999, 500, 500],
        "generator_power_W": [999999, 0, 250],
        "battery_power_W": [999999, 10000, 10000],
    })
    monkeypatch.setattr(gui.model, "build_timeseries_dataframe", lambda *_: dispatch)
    variables = {key: SimpleNamespace(value=[value]) for key, value in {
        "batt_scale": 1000, "potable_water_storage_scale": 200, "h2_storage_scale": 10,
    }.items()}
    config = {"efficiency": {"potable_water_Wh_per_l": 4.5, "h2_Wh_per_g": 65}}
    result = gui.result_composition(config, variables)
    assert result["generation_kWh"] == pytest.approx([1, 2, 1.5, .5])
    assert result["storage_equivalent_kWh"] == pytest.approx([1, .9, .65])
    assert result["storage_labels"] == ["Battery: 1 kWh", "Water: 200 L", "Hydrogen: 10 g"]


def test_daily_display_preserves_energy_charge_directions_and_storage_extrema():
    import numpy as np
    import pandas as pd
    from sirenolite.gui import result_history
    columns = ["wind_power_W", "solar_power_W", "wave_power_W", "generator_power_W",
               "electric_demand_W", "battery_power_W", "battery_storage_Wh",
               "potable_storage_l", "h2_storage_g"]
    data = pd.DataFrame({column: np.zeros(721) for column in columns})
    data["time_hr"] = np.arange(721)
    data.loc[0, "wind_power_W"] = 999999  # initial state must not enter energy totals
    data.loc[1:, "wind_power_W"] = 1000
    data.loc[1:, "electric_demand_W"] = 1000
    data.loc[1:, "battery_power_W"] = np.tile([500, -500], 360)
    data.loc[24, "battery_storage_Wh"] = 100
    history = result_history(data)
    assert history["daily"]
    assert len(history["time_days"]) == 30
    assert np.dot(history["power"]["wind_power_W"], history["duration_hours"]) == pytest.approx(720)
    np.testing.assert_allclose(history["power"]["charge_kW"], .25)
    np.testing.assert_allclose(history["power"]["discharge_kW"], .25)
    # The midnight inventory belongs to both adjacent days' physical ranges.
    np.testing.assert_allclose(history["storage"]["battery_storage_Wh"]["max"][:2], [100, 100])
    np.testing.assert_allclose(history["storage"]["battery_storage_Wh"]["end"][:2], [100, 0])
    hourly = result_history(data.iloc[:25])
    assert not hourly["daily"]
    assert len(hourly["time_days"]) == 24
