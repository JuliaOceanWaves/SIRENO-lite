import pytest

from sirenolite.gui import (
    build_json_payload,
    format_doldrum_windows_text,
    merge_overrides,
    parse_bool,
    parse_doldrum_windows_text,
    parse_value,
    split_json_payload,
    _set_gekko_property_if_supported,
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
    assert parse_value("4.9", 1) == 4
    assert parse_value("4.9", 1.0) == 4.9
    assert parse_value("unchanged", "template") == "unchanged"


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


def test_build_json_payload_captures_optimizer_toggles():
    class Toggle:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    config = {"objective": "total_mass", "solver": {"remote": False}}
    payload = build_json_payload(config, {"wind": Toggle(1), "solar": Toggle(0)})

    assert payload["objective"] == "total_mass"
    assert payload["solver"] == {"remote": False}
    assert payload["gui"] == {"optimize": {"wind": True, "solar": False}}


def test_set_gekko_property_helper_reports_unsupported_properties():
    class AcceptsAny:
        pass

    class RejectsNamedProperty:
        def __setattr__(self, name, value):
            raise Exception(f"{name} is not a property of this variable")

    target = AcceptsAny()
    assert _set_gekko_property_if_supported(target, "STATUS", 1) is True
    assert target.STATUS == 1
    assert _set_gekko_property_if_supported(RejectsNamedProperty(), "STATUS", 1) is False
