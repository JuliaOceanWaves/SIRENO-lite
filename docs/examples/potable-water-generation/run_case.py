#!/usr/bin/env python3
"""Sweep the potable-water example across candidate technology sets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from sirenolite import model
from sirenolite.config import load_configuration, merge_overrides, prepare_config


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CASE_DIR / "sirenolite_config.json"
DEFAULT_HORIZON_HOURS = 336
DEFAULT_PROFILE_START_OFFSET_HOURS = 1
TARGET_POTABLE_DEMAND_L_PER_DAY = 40.0 * model.LITERS_PER_GALLON


@dataclass
class RunResult:
    name: str
    ok: bool
    elapsed_s: float
    iterations: int | None
    objective_value: float | None
    total_cost: float | None
    total_mass: float | None
    wind_scale_w: float | None
    solar_scale_w: float | None
    wave_scale_w: float | None
    generator_scale_w: float | None
    batt_scale_wh: float | None
    h2_storage_g: float | None
    potable_storage_l: float | None
    error: str = ""


@dataclass
class Candidate:
    name: str
    description: str
    mutate: Callable[[dict], None]


def deep_merge(base: dict, overrides: dict) -> dict:
    return merge_overrides(base, overrides)


def load_case_config(config_path: Path) -> tuple[dict, dict]:
    config, optimize = load_configuration(config_path)
    return config, {"optimize": optimize}


def parse_iterations(apopt_path: Path) -> int | None:
    if not apopt_path.exists():
        return None
    pattern = re.compile(r"^\s*(\d+)\s+[-+0-9.Ee]+\s+[-+0-9.Ee]+")
    iterations = []
    for line in apopt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if match:
            iterations.append(int(match.group(1)))
    if not iterations:
        return None
    return max(iterations)


def objective_value(config: dict, vars_: dict) -> float:
    model_obj = vars_.get("model")
    options = getattr(model_obj, "options", None)
    solved_obj = getattr(options, "objfcnval", None)
    if solved_obj is not None:
        try:
            solved_obj = float(solved_obj)
            if math.isfinite(solved_obj):
                return solved_obj
        except (TypeError, ValueError):
            pass

    objective = config.get("objective", "total_mass")
    if objective == "total_cost":
        return float(vars_["total_cost"].value[-1])
    if objective == "cost_per_watt":
        return float(vars_["cost_per_watt"].value[-1])
    if objective == "cost_per_mass":
        return float(vars_["cost_per_mass"].value[-1])
    return float(vars_["total_mass"].value[-1])


def run_single(name: str, config: dict) -> RunResult:
    t0 = time.time()
    model_tuple = None
    try:
        inputs = model.load_resource_inputs(config)
        model_tuple = model.build_model(config, inputs)
        _, vars_ = model_tuple
        model.solve_model(config, model_tuple)
        elapsed = time.time() - t0
        iterations = parse_iterations(Path(vars_["model"].path) / "APOPT.out")
        return RunResult(
            name=name,
            ok=True,
            elapsed_s=elapsed,
            iterations=iterations,
            objective_value=objective_value(config, vars_),
            total_cost=float(vars_["total_cost"].value[-1]),
            total_mass=float(vars_["total_mass"].value[-1]),
            wind_scale_w=float(vars_["wind_scale"].value[-1]),
            solar_scale_w=float(vars_["solar_scale"].value[-1]),
            wave_scale_w=float(vars_["wave_scale"].value[-1]),
            generator_scale_w=float(vars_["generator_scale"].value[-1]),
            batt_scale_wh=float(vars_["batt_scale"].value[-1]),
            h2_storage_g=float(vars_["h2_storage_scale"].value[-1]),
            potable_storage_l=float(vars_["potable_water_storage_scale"].value[-1]),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return RunResult(
            name=name,
            ok=False,
            elapsed_s=time.time() - t0,
            iterations=None,
            objective_value=None,
            total_cost=None,
            total_mass=None,
            wind_scale_w=None,
            solar_scale_w=None,
            wave_scale_w=None,
            generator_scale_w=None,
            batt_scale_wh=None,
            h2_storage_g=None,
            potable_storage_l=None,
            error=str(exc),
        )
    finally:
        if model_tuple is not None:
            model_tuple[0].cleanup()


def summarize(result: RunResult) -> str:
    if not result.ok:
        return f"{result.name}: FAIL in {result.elapsed_s:.2f}s | error={result.error}"
    return (
        f"{result.name}: OK in {result.elapsed_s:.2f}s | "
        f"iter={result.iterations} | objective={result.objective_value:.6g} | "
        f"cost={result.total_cost:.6g} | mass={result.total_mass:.6g} | "
        f"wind={result.wind_scale_w:.6g} W | "
        f"solar={result.solar_scale_w:.6g} W | "
        f"wave={result.wave_scale_w:.6g} W | "
        f"gen={result.generator_scale_w:.6g} W | "
        f"water_store={result.potable_storage_l:.6g} L"
    )


def apply_gui_optimize_flags(config: dict, gui_payload: dict) -> None:
    snapshot = prepare_config(config, gui_payload.get("optimize", {}))
    config.clear()
    config.update(snapshot)


def apply_potable_case_defaults(config: dict) -> None:
    deep_merge(
        config,
        {
            "objective": "total_cost",
            "simulation": {
                "peak_load": 0.0,
                "hotel_load": 0.0,
                "H2DailyDemand": 0.0,
                "H2ODailyDemand": TARGET_POTABLE_DEMAND_L_PER_DAY,
                "nhrs": DEFAULT_HORIZON_HOURS,
                "profile_start_offset_hours": DEFAULT_PROFILE_START_OFFSET_HOURS,
                "doldrum_time": 0,
            },
            "limits": {
                "batt_min_capacity": 0.0,
                "batt_max_capacity": 0.0,
                "batt_max_charge_rate": 0.0,
                "h2_min_storage_g": 0.0,
                "h2_max_storage_g": 0.0,
                "h2_max_charge_rate_gph": 0.0,
            },
            "efficiency": {"potable_water_Wh_per_l": 4.5},
            "solver": {"remote": False, "solver": 1},
        },
    )


def disable_technology(config: dict, name: str) -> None:
    mapping = {
        "generator": ("generator_min_capacity", "generator_max_capacity"),
        "wind": ("wind_min_capacity", "wind_max_capacity"),
        "solar": ("solar_min_capacity", "solar_max_capacity"),
        "wave": ("wave_min_capacity", "wave_max_capacity"),
    }
    min_key, max_key = mapping[name]
    config["limits"][min_key] = 0.0
    config["limits"][max_key] = 0.0


def build_candidates() -> list[Candidate]:
    return [
        Candidate(
            name="renewable_hybrid",
            description="Wind + solar + wave search with generator disabled.",
            mutate=lambda cfg: disable_technology(cfg, "generator"),
        ),
        Candidate(
            name="solar_only",
            description="Solar + potable storage only.",
            mutate=lambda cfg: (
                disable_technology(cfg, "wind"),
                disable_technology(cfg, "wave"),
                disable_technology(cfg, "generator"),
            ),
        ),
        Candidate(
            name="wind_only",
            description="Wind + potable storage only.",
            mutate=lambda cfg: (
                disable_technology(cfg, "solar"),
                disable_technology(cfg, "wave"),
                disable_technology(cfg, "generator"),
            ),
        ),
        Candidate(
            name="wave_only",
            description="Wave + potable storage only.",
            mutate=lambda cfg: (
                disable_technology(cfg, "wind"),
                disable_technology(cfg, "solar"),
                disable_technology(cfg, "generator"),
                cfg["limits"].update({"potable_water_max_storage_l": 1000.0}),
            ),
        ),
        Candidate(
            name="generator_only",
            description="Generator + potable storage only.",
            mutate=lambda cfg: (
                disable_technology(cfg, "wind"),
                disable_technology(cfg, "solar"),
                disable_technology(cfg, "wave"),
                cfg["limits"].update(
                    {
                        "generator_min_capacity": 0.0,
                        "generator_max_capacity": 500.0,
                        "potable_water_max_storage_l": min(
                            float(cfg["limits"]["potable_water_max_storage_l"]),
                            300.0,
                        ),
                    }
                ),
            ),
        ),
    ]


def choose_best(results: list[RunResult]) -> RunResult | None:
    successful = [r for r in results if r.ok and r.objective_value is not None]
    if not successful:
        return None
    return min(successful, key=lambda result: (float(result.objective_value), result.elapsed_s))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run potable-water technology sweep.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to potable-water JSON config.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 168 hourly samples (167 solved hours) for faster diagnostics.",
    )
    parser.add_argument(
        "--nhrs",
        type=int,
        default=None,
        help="Override the number of hourly samples; solved duration is nhrs - 1 hours.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=120,
        help="Solver max iterations for each candidate run.",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(CASE_DIR / "run_summary.csv"),
        help="Where to write run summary CSV.",
    )
    parser.add_argument(
        "--recommended-config",
        default=str(CASE_DIR / "sirenolite_config.recommended.json"),
        help="Where to write the best successful config JSON.",
    )
    return parser


def freeze_optimized_design(config: dict, result: RunResult) -> dict:
    payload = copy.deepcopy(config)
    fixed_limits = {
        "generator": ("generator_min_capacity", "generator_max_capacity", result.generator_scale_w),
        "wind": ("wind_min_capacity", "wind_max_capacity", result.wind_scale_w),
        "solar": ("solar_min_capacity", "solar_max_capacity", result.solar_scale_w),
        "wave": ("wave_min_capacity", "wave_max_capacity", result.wave_scale_w),
        "battery": ("batt_min_capacity", "batt_max_capacity", result.batt_scale_wh),
        "h2_storage": ("h2_min_storage_g", "h2_max_storage_g", result.h2_storage_g),
        "potable_water_storage": (
            "potable_water_min_storage_l",
            "potable_water_max_storage_l",
            result.potable_storage_l,
        ),
    }
    for min_key, max_key, value in fixed_limits.values():
        if value is None:
            continue
        payload["limits"][min_key] = float(value)
        payload["limits"][max_key] = float(value)

    payload["gui"] = {
        "optimize": {
            "generator_scale": False,
            "wind_scale": False,
            "solar_scale": False,
            "wave_scale": False,
            "batt_scale": False,
            "h2_storage_scale": False,
            "potable_water_storage_scale": False,
        }
    }
    payload["solver"]["remote"] = False
    payload["solver"]["solver"] = 1
    return payload


def main() -> int:
    args = create_parser().parse_args()
    config_path = Path(args.config).resolve()
    config, gui_payload = load_case_config(config_path)
    apply_gui_optimize_flags(config, gui_payload)
    apply_potable_case_defaults(config)

    config["solver"]["remote"] = False
    config["solver"]["max_iter"] = int(args.max_iter)
    if args.quick and args.nhrs is None:
        config["simulation"]["nhrs"] = 168
    if args.nhrs is not None:
        config["simulation"]["nhrs"] = int(args.nhrs)

    candidates = build_candidates()
    results: list[RunResult] = []
    configs_by_name: dict[str, dict] = {}
    meta_by_name = {candidate.name: candidate for candidate in candidates}
    for candidate in candidates:
        cfg = copy.deepcopy(config)
        candidate.mutate(cfg)
        model.update_derived_config(cfg)
        configs_by_name[candidate.name] = cfg
        result = run_single(candidate.name, cfg)
        results.append(result)
        print(summarize(result))

    rows = []
    for result in results:
        row = result.__dict__.copy()
        row["description"] = meta_by_name[result.name].description
        rows.append(row)
    df = pd.DataFrame(rows)
    summary_path = Path(args.summary_csv).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV written: {summary_path}")

    best = choose_best(results)
    if best is None:
        print("No successful run found.")
        return 1

    recommended_path = Path(args.recommended_config).resolve()
    recommended_path.parent.mkdir(parents=True, exist_ok=True)
    recommended_payload = freeze_optimized_design(configs_by_name[best.name], best)
    recommended_path.write_text(json.dumps(recommended_payload, indent=2, sort_keys=True) + "\n")
    print(f"Recommended config ({best.name}) written: {recommended_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
