import argparse
import json
from pathlib import Path

from sirenolite import model, __version__


def load_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_json_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be a mapping/object.")
    gui_payload = payload.get("gui")
    if "config" in payload:
        config_payload = payload["config"] or {}
        if not isinstance(config_payload, dict):
            raise ValueError("config must be a mapping when provided.")
        return config_payload, gui_payload
    return {k: v for k, v in payload.items() if k != "gui"}, gui_payload


def apply_overrides(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            apply_overrides(base[key], value)
        else:
            base[key] = value
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run SIRENO-lite optimization.")
    parser.add_argument(
        "--version",
        action="version",
        version=f"sirenolite {__version__}",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config",
        help="Path to a JSON config file with overrides for the default inputs.",
    )
    group.add_argument(
        "--dump-config",
        help="Write a default JSON config template to PATH and exit.",
    )
    args = parser.parse_args(argv)

    if args.dump_config:
        config = model.default_inputs()
        output_path = Path(args.dump_config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        return

    if args.config:
        payload = load_config(args.config)
        overrides, _ = split_json_payload(payload)
        defaults = model.default_inputs()
        model.migrate_legacy_limit_keys(overrides, fallback_efficiency=defaults["efficiency"])
        config = defaults
        apply_overrides(config, overrides)
        model.main(config)
        return

    model.main()


if __name__ == "__main__":
    main()
