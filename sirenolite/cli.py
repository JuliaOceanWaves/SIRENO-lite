import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile

from sirenolite import model, __version__
from sirenolite.config import load_configuration, merge_overrides, prepare_config


def load_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_overrides(base, overrides):
    return merge_overrides(base, overrides)


def run_to_directory(config, output_dir):
    """Solve a retained input snapshot and publish a complete, portable bundle.

    Existing nonempty outputs are never replaced. A failed solve or export
    leaves no partial result directory.
    """
    config = prepare_config(config)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new run directory.")
    source = Path(config["simulation"]["data_file"])
    resource_bytes = source.read_bytes()
    resource_hash = hashlib.sha256(resource_bytes).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sirenolite-run-", dir=output.parent) as folder:
        staging = Path(folder)
        (staging / "resource.csv").write_bytes(resource_bytes)
        config["simulation"]["data_file"] = str(staging / "resource.csv")
        inputs = model.load_resource_inputs(config)
        m, variables = model.build_model(config, inputs)
        try:
            model.solve_model(config, (m, variables))
            frame = model.build_timeseries_dataframe(config, variables)
            summary = {
                "software": {"sirenolite": __version__, "python": platform.python_version(),
                             **{name: version(name) for name in ("gekko", "numpy", "pandas", "matplotlib")}},
                "source_path": str(source),
                "resource_sha256": resource_hash,
                "objective": config["objective"],
                "duration_hours": float(m.time[-1] - m.time[0]),
                "max_power_balance_error_W": float(frame["power_balance_W"].iloc[1:].abs().max()),
                "solver_status": int(m.options.APPSTATUS),
                "diagnostics": variables["solution_diagnostics"],
            }
            for name in ("total_cost", "total_mass", "cost_per_watt", "cost_per_mass",
                         "supported_mass", "floating_platform_mass", "generator_scale",
                         "wind_scale", "solar_scale", "wave_scale", "batt_scale",
                         "h2_storage_scale", "potable_water_storage_scale"):
                summary[name] = float(model.to_array(variables[name].value)[-1])
            frame.to_csv(staging / "timeseries.csv", index=False)
            config["simulation"]["data_file"] = "resource.csv"
            (staging / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            (staging / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            figure = model.create_stacked_figure(config, variables)
            try:
                figure.savefig(staging / "plots.pdf")
            finally:
                figure.clear()
            if output.exists():
                output.rmdir()  # Only an empty directory may be replaced.
            staging.rename(output)
            print(f"Saved results to {output}")
            return summary
        finally:
            m.cleanup()


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
    parser.add_argument("--output-dir", help="Save config, summary, CSV, and PDF without opening plots.")
    parser.add_argument("--remote", action="store_true", help="Send this solve to the GEKKO remote service.")
    args = parser.parse_args(argv)

    try:
        if args.dump_config:
            if args.output_dir or args.remote:
                parser.error("--dump-config cannot be combined with --output-dir or --remote")
            config = model.default_inputs()
            output_path = Path(args.dump_config).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return
        if args.config:
            config, optimize = load_configuration(args.config)
            config = prepare_config(config, optimize)
        else:
            config = model.default_inputs()
        if args.remote:
            config["solver"]["remote"] = True
        if args.output_dir:
            run_to_directory(config, args.output_dir)
        elif args.config or args.remote:
            model.main(config)
        else:
            model.main()
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    except Exception as exc:
        parser.exit(1, f"SIRENO-lite run failed: {exc}\n")


if __name__ == "__main__":
    main()
