import pytest

from sirenolite.gui import format_doldrum_windows_text, parse_doldrum_windows_text


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
