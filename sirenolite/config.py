"""Shared, side-effect-free configuration bridge for the CLI and desktop GUI."""

from copy import deepcopy
import json
from pathlib import Path

from sirenolite import model


CAPACITY_LIMITS = {
    "generator_scale": ("generator_min_capacity", "generator_max_capacity"),
    "wind_scale": ("wind_min_capacity", "wind_max_capacity"),
    "solar_scale": ("solar_min_capacity", "solar_max_capacity"),
    "wave_scale": ("wave_min_capacity", "wave_max_capacity"),
    "batt_scale": ("batt_min_capacity", "batt_max_capacity"),
    "h2_storage_scale": ("h2_min_storage_g", "h2_max_storage_g"),
    "potable_water_storage_scale": (
        "potable_water_min_storage_l", "potable_water_max_storage_l"
    ),
}


def parse_bool(value):
    """Accept explicit booleans and common user spellings, never truthy garbage."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot parse boolean from value: {value!r}")


def merge_overrides(base, overrides):
    """Merge partial settings, preserving unspecified siblings."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_overrides(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def split_json_payload(payload):
    """Read plain settings and the older {config, gui} envelope."""
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be a mapping/object.")
    gui_payload = payload.get("gui", {})
    if not isinstance(gui_payload, dict):
        raise ValueError("gui must be a mapping when provided.")
    if "config" in payload:
        extra = set(payload) - {"config", "gui"}
        if extra:
            raise ValueError(f"Unknown wrapped configuration key(s): {', '.join(sorted(extra))}")
        config_payload = payload["config"]
        if not isinstance(config_payload, dict):
            raise ValueError("config must be a mapping when provided.")
    else:
        config_payload = {key: value for key, value in payload.items() if key != "gui"}
    return config_payload, gui_payload


def _check_keys(overrides, defaults, prefix=""):
    for key, value in overrides.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in defaults:
            raise ValueError(f"Unknown configuration key: {name}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping/object.")
            _check_keys(value, defaults[key], name)


def _optimize_flags(gui_payload):
    if set(gui_payload) - {"optimize"}:
        raise ValueError("gui supports only the optimize section.")
    flags = gui_payload.get("optimize", {})
    if not isinstance(flags, dict):
        raise ValueError("gui.optimize must be a mapping/object.")
    unknown = set(flags) - set(CAPACITY_LIMITS)
    if unknown:
        raise ValueError(f"Unknown sizing control(s): {', '.join(sorted(unknown))}")
    return {key: parse_bool(value) for key, value in flags.items()}


def configuration_from_payload(payload, base_dir=None):
    """Return validated full settings and sizing flags, without changing input.

    Relative CSV paths are relative to the JSON file's directory. Omitting
    data_file uses the installed sample; the sample never depends on cwd.
    """
    overrides, gui_payload = split_json_payload(payload)
    overrides = deepcopy(overrides)
    legacy_limits = overrides.get("limits", {})
    if isinstance(legacy_limits, dict) and "batt_max_ramp_up" in legacy_limits:
        legacy = legacy_limits.pop("batt_max_ramp_up")
        legacy_limits.setdefault("batt_max_charge_rate", legacy)
    config = model.default_inputs()
    _check_keys(overrides, config)
    merge_overrides(config, overrides)
    for name in ("remote", "linear"):
        config["solver"][name] = parse_bool(config["solver"][name])
    optimize = _optimize_flags(gui_payload)
    data_file = config["simulation"]["data_file"]
    if not isinstance(data_file, str) or not data_file.strip():
        raise ValueError("simulation.data_file must be a nonempty path string.")
    path = Path(data_file).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    config["simulation"]["data_file"] = str(path.resolve())
    # Check original editable bounds even when a sizing toggle fixes capacity.
    # The linear solver hint, however, belongs to the effective run formulation.
    linear = config["solver"]["linear"]
    config["solver"]["linear"] = False
    try:
        model.update_derived_config(config)
    finally:
        config["solver"]["linear"] = linear
    effective = deepcopy(config)
    for name, enabled in optimize.items():
        if not enabled:
            minimum, maximum = CAPACITY_LIMITS[name]
            effective["limits"][maximum] = effective["limits"][minimum]
    model.validate_config(effective)
    return config, optimize


def load_configuration(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return configuration_from_payload(payload, base_dir=path.parent)


def prepare_config(config, optimize=None):
    """Snapshot a run; unchecked sizing fixes capacity to its configured minimum."""
    result, flags = configuration_from_payload({
        "config": config, "gui": {"optimize": optimize or {}}
    })
    for name, enabled in flags.items():
        if not enabled:
            minimum, maximum = CAPACITY_LIMITS[name]
            result["limits"][maximum] = result["limits"][minimum]
    model.update_derived_config(result)
    return result
