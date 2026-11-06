#!/usr/bin/env python3
"""Rebuild the static map from retained, archived one-year LP results.

Run from any directory with the repository's Python environment. This exports
existing solver results; it does not solve a new case or acquire resource data.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PAPER_ROOT = REPO_ROOT / "paper"
COST_CASE_DIR = PAPER_ROOT / "archive" / "legacy_analysis" / "results" / "cost_per_watt_case"
RESOURCE_DIR = PAPER_ROOT / "paper_reproducibility" / "results" / "real_resource_profiles"
SITE_METADATA_PATH = SCRIPT_DIR / "site_metadata.json"
LAND_GEOJSON_PATH = RESOURCE_DIR / "ne_110m_land.geojson"
OUTPUT_PATH = REPO_ROOT / "docs" / "hybrid-map-tool" / "data.js"
BASELINE_VARIANT = "wec_1x"

SUMMARY_FIELDS = (
    "variant",
    "variant_label",
    "total_cost_usd",
    "total_mass_kg",
    "wind_scale_w",
    "solar_scale_w",
    "wave_scale_w",
    "batt_scale_wh",
    "renewable_capacity_w",
    "reported_cost_per_watt_usd_per_w",
    "max_power_balance_w",
    "battery_terminal_margin_wh",
    "wave_capture_fraction",
)

TIMESERIES_FIELDS = (
    "time_days",
    "load_W",
    "wind_power_W",
    "solar_power_W",
    "wave_power_W",
    "battery_storage_Wh",
    "battery_charge_W",
    "battery_discharge_W",
)


def rounded(values: pd.Series | np.ndarray, digits: int) -> list[float]:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def load_land(path: Path) -> list[list[list[float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    polygons: list[list[list[float]]] = []
    for feature in data["features"]:
        geometry = feature["geometry"]
        coordinate_groups = []
        if geometry["type"] == "Polygon":
            coordinate_groups = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            coordinate_groups = geometry["coordinates"]
        for polygon in coordinate_groups:
            if not polygon:
                continue
            ring = polygon[0]
            polygons.append([[round(float(lon), 3), round(float(lat), 3)] for lon, lat in ring])
    return polygons


def timeseries_for_site(case: str) -> dict[str, list[float]]:
    path = COST_CASE_DIR / f"{case}_{BASELINE_VARIANT}_timeseries.csv"
    frame = pd.read_csv(path)
    return {
        "timeDays": rounded(frame["time_days"], 4),
        "loadW": rounded(frame["load_W"], 3),
        "windW": rounded(frame["wind_power_W"], 3),
        "solarW": rounded(frame["solar_power_W"], 3),
        "waveW": rounded(frame["wave_power_W"], 3),
        "batteryKWh": rounded(frame["battery_storage_Wh"] / 1000.0, 4),
        "batteryChargeW": rounded(frame["battery_charge_W"], 3),
        "batteryDischargeW": rounded(frame["battery_discharge_W"], 3),
    }


def site_payload(summary_rows: pd.DataFrame, metadata: dict) -> dict:
    baseline = summary_rows[summary_rows["variant"] == BASELINE_VARIANT]
    if baseline.empty:
        raise ValueError(f"Missing {BASELINE_VARIANT} row for {metadata['siteKey']}.")
    baseline_row = baseline.iloc[0]
    variants = []
    for row in summary_rows.sort_values("wave_capture_fraction").itertuples(index=False):
        record = {
                field: (
                    round(float(getattr(row, field)), 6)
                    if isinstance(getattr(row, field), (float, int, np.floating, np.integer))
                    else getattr(row, field)
                )
                for field in SUMMARY_FIELDS
            }
        record["renewable_operating_cost_usd"] = round(sum(
            float(getattr(row, f"{resource}_operating_cost_usd"))
            for resource in ("wind", "solar", "wave")
        ), 6)
        variants.append(record)

    return {
        **metadata,
        "baseline": {
            "costUsd": round(float(baseline_row.total_cost_usd), 4),
            "costPerW": round(float(baseline_row.reported_cost_per_watt_usd_per_w), 4),
            "massKg": round(float(baseline_row.total_mass_kg), 4),
            "windW": round(float(baseline_row.wind_scale_w), 4),
            "solarW": round(float(baseline_row.solar_scale_w), 4),
            "waveW": round(float(baseline_row.wave_scale_w), 4),
            "batteryWh": round(float(baseline_row.batt_scale_wh), 4),
            "renewableW": round(float(baseline_row.renewable_capacity_w), 4),
        },
        "variants": variants,
        "timeseries": timeseries_for_site(str(baseline_row.case)),
    }


def build_payload() -> dict:
    # Preserve the profile descriptions belonging to this frozen study. The
    # current mission study has a different site selection and cannot replace it.
    profiles = json.loads(SITE_METADATA_PATH.read_text(encoding="utf-8"))
    sites = []
    for metadata in profiles:
        summary_path = COST_CASE_DIR / f"constant_100w_real_resource_{metadata['siteKey']}_summary.csv"
        site_rows = pd.read_csv(summary_path)
        if set(site_rows["site_key"]) != {metadata["siteKey"]}:
            raise ValueError(f"Unexpected site identity in {summary_path}.")
        if site_rows.empty:
            raise ValueError(f"Missing summary rows for {metadata['siteKey']}.")
        sites.append(site_payload(site_rows, metadata))
    return {
        "schemaVersion": 1,
        "baseLoadW": 100.0,
        "baselineVariant": BASELINE_VARIANT,
        "description": (
            "Archived SIRENO-Lite one-year LP cases with forced final battery recharge. "
            "These are distinct from the current cyclic mission study. Map clicks assign the nearest "
            "evaluated ocean profile; arbitrary lat/lon live resource lookup is not included. "
            "Some strategic additions use NASA POWER wind/solar with a documented wind-sea "
            "wave timing proxy where gridded marine data were not available in this run."
        ),
        "sources": {
            "summaryCsvPattern": str((COST_CASE_DIR / "constant_100w_real_resource_{siteKey}_summary.csv").relative_to(REPO_ROOT)),
            "siteMetadata": str(SITE_METADATA_PATH.relative_to(REPO_ROOT)),
            "landGeometry": "Natural Earth 1:110m land geometry cached by the paper workflow",
        },
        "scalingAssumptions": {
            "batteryRoundTripEfficiency": 0.9,
            "batteryMassKgPerWh": 0.008,
            "batteryCostUsdPerWh": 0.16,
            "platformMassPerSupportedMass": 0.3,
            "platformCostUsdPerKg": 30.0,
            "batteryEndpoint": "archived forced final recharge",
        },
        "land": load_land(LAND_GEOJSON_PATH),
        "sites": sites,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check exact reproducibility without writing data.js")
    args = parser.parse_args()
    payload = build_payload()
    rendered = "window.SIRENO_HYBRID_MAP_DATA=" + json.dumps(payload, separators=(",", ":")) + ";\n"
    if args.check:
        if OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            raise SystemExit("data.js differs from the retained source results; rebuild and review the difference.")
        print("data.js exactly reproduces the retained source results.")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
