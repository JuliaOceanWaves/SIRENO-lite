"""Replay packaged annual missions against retained canonical paper results.

Run from any directory with SIRENO-Lite installed. The default covers Hawaii's
four missions and Anchorage charging (including integer diesel deliveries).
Use --all for all 76 cases. No nested paper checkout or network is required.
Reference evidence is never rewritten. Outputs go to a new directory; hourly
traces are optional because the complete campaign can produce large files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import pandas as pd

from sirenolite import model, paper_solver, presets

REFERENCE = presets.DATA / "paper_annual_reference.csv"
PROVENANCE = presets.DATA / "paper_annual_reference.json"
COMPARISONS = {
    "total_cost": "total_cost_usd",
    "generator_scale": "generator_capacity_w",
    "wind_scale": "wind_capacity_w",
    "solar_scale": "solar_capacity_w",
    "wave_scale": "wave_capacity_w",
    "batt_scale": "battery_capacity_wh",
    "potable_water_storage_scale": "potable_water_capacity_l",
}


def reference_cases(all_cases=False):
    """Validate the retained evidence and select unique site/mission pairs."""
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    for path, key in (
        (REFERENCE, "reference_sha256"),
        (presets.DATA / "paper_sites.json", "resource_manifest_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance[key]:
            raise ValueError(f"Paper reference provenance mismatch: {path.name}")
    frame = pd.read_csv(REFERENCE)
    sites = presets.list_sites()
    for site in sites:
        path = presets.DATA / site["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != site["sha256"]:
            raise ValueError(f"Paper resource provenance mismatch: {path.name}")
    expected = {(s["key"], mission) for s in sites for mission in presets.MISSIONS}
    actual = list(zip(frame.site_key, frame.mission_key))
    if (
        len(actual) != len(expected)
        or set(actual) != expected
        or not frame.status.eq("OPTIMAL").all()
    ):
        raise ValueError(
            "Paper reference must contain one optimal result for every site and mission."
        )
    if not all_cases:
        frame = frame.loc[
            frame.site_key.eq(presets.DEFAULT_SITE_KEY)
            | (
                frame.site_key.eq("anchorage_alaska_46081_oneyear")
                & frame.mission_key.eq("scheduled_charging")
            )
        ]
    return frame, provenance


def compare_result(values, reference, provenance):
    differences = {}
    for key, column in COMPARISONS.items():
        actual, expected = float(values[key].value[-1]), float(reference[column])
        tolerance = provenance[
            (
                "cost_absolute_tolerance_usd"
                if key == "total_cost"
                else "capacity_absolute_tolerance"
            )
        ]
        difference = abs(actual - expected)
        differences[key] = {
            "actual": actual,
            "paper": expected,
            "absolute_difference": difference,
            "tolerance": tolerance,
            "passed": bool(difference <= tolerance),
        }
    for key, column in (
        ("generator_installed", "generator_installed"),
        ("delivery_trips", "generator_delivery_trips_per_year"),
    ):
        actual, expected = values["paper_summary"][key], int(reference[column])
        differences[key] = {
            "actual": actual,
            "paper": expected,
            "passed": actual == expected,
        }
    return differences


def run(output_dir, *, all_cases=False, hourly=False):
    references, provenance = reference_cases(all_cases)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("sirenolite", "numpy", "pandas", "scipy")
    }
    report = {
        "passed": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "source_sha256": {
            Path(module.__file__)
            .name: hashlib.sha256(Path(module.__file__).read_bytes())
            .hexdigest()
            for module in (model, paper_solver, presets)
        },
        "reference": provenance,
        "cases": [],
    }
    report_path = output_dir / "annual_comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for _, reference in references.iterrows():
        site, mission = reference.site_key, reference.mission_key
        config = presets.load_site_config(site, mission)
        start = time.monotonic()
        solver = None
        record = {
            "site": site,
            "mission": mission,
            "passed": False,
            "configuration": config,
        }
        report["cases"].append(record)
        try:
            problem = model.build_model(config, model.load_resource_inputs(config))
            solver, values = problem
            model.solve_model(config, problem)
            differences = compare_result(values, reference, provenance)
            record.update(
                {
                    "seconds": time.monotonic() - start,
                    "passed": all(d["passed"] for d in differences.values()),
                    "comparison": differences,
                    "summary": values["paper_summary"],
                    "diagnostics": model.solution_diagnostics(config, values),
                }
            )
            record["passed"] = record["passed"] and record["diagnostics"]["passed"]
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            if hourly:
                model.build_timeseries_dataframe(config, values).to_csv(
                    output_dir / f"{site}__{mission}.csv.gz", index=False
                )
            print(
                json.dumps(
                    {
                        key: record[key]
                        for key in ("site", "mission", "seconds", "passed")
                    }
                ),
                flush=True,
            )
            if not record["passed"]:
                raise RuntimeError(
                    f"Paper replay differs from retained evidence: {site}/{mission}; see {report_path}"
                )
        except Exception as exc:
            record.update(
                passed=False,
                seconds=time.monotonic() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            raise
        finally:
            if solver is not None:
                solver.cleanup()
    report["passed"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Replay all 19 locations and four missions."
    )
    parser.add_argument(
        "--hourly", action="store_true", help="Also retain complete hourly traces."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New destination; existing paths are refused.",
    )
    args = parser.parse_args(argv)
    run(args.output_dir, all_cases=args.all, hourly=args.hourly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
