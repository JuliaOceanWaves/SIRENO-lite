"""Rebuild or check offline resource presets against the independent paper checkout.

The application and annual replay do not require that checkout. This maintainer
utility does: --check compares its canonical profiles with all packaged sources;
--output-dir stages regenerated inputs for review without replacing package data.
It never refreshes the retained annual result reference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
REPRO = PAPER / "paper_reproducibility"
PACKAGED = ROOT / "sirenolite/data"


def build(output_dir):
    if not (REPRO / "run_global_mission_cases.py").is_file():
        raise FileNotFoundError(
            "Preset regeneration requires the independent paper checkout at paper/."
        )
    sys.path.insert(0, str(REPRO))
    annual = importlib.import_module("run_global_mission_cases")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = pd.read_csv(annual.RESOURCE_SUMMARY).set_index("site_key")
    logistics = annual.load_generator_logistics().set_index("site_key")
    audit = (
        pd.read_csv(annual.GLOBAL_AUDIT_SUMMARY)
        .drop_duplicates("site_key")
        .set_index("site_key")
    )
    records = []
    for site in annual.global_screen.baseline_sites():
        key = site["site_key"]
        row = metadata.loc[key]
        source = PAPER / row.output_csv
        offset = annual.local_solar_utc_offset(float(row.longitude))
        charging, water = annual.annual_demand(
            annual.ANNUAL_MISSIONS[3], utc_offset_hours=offset
        )
        frame = pd.read_csv(source, usecols=["Wind", "Solar", "Wave"])
        for column in frame:
            frame[column] = annual.normalize_resource(
                frame[column], label=f"{key}:{column}"
            )
        frame.insert(0, "Time", np.arange(8760))
        frame["Load"] = (charging - 100) / 2500
        frame["Hydrogen"] = 0.0
        frame["PotableWater"] = water / 151.416
        path = output_dir / f"paper_{key}.csv.gz"
        frame.to_csv(
            path,
            index=False,
            float_format="%.15g",
            compression={"method": "gzip", "mtime": 0},
        )
        records.append(
            {
                "key": key,
                "label": site["site_label"],
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "hours": 8760,
                "start": "2024-01-01T00:00:00Z",
                "file": path.name,
                "resource_source": str(row.resource_source),
                "source_class": str(audit.loc[key, "source_class"]),
                "inclusion_status": str(audit.loc[key, "inclusion_status"]),
                "note": str(row.note),
                "solar_source": str(row.solar_source),
                "wind_source": str(row.wind_source),
                "wave_source": str(row.wave_source),
                "source_urls": [
                    str(row[k])
                    for k in [
                        "nasa_power_url",
                        "ndbc_historical_url",
                        "open_meteo_marine_url",
                    ]
                    if pd.notna(row[k])
                ],
                "source_path": source.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "delivery_trip_cost_usd": float(
                    logistics.loc[key, "delivery_trip_cost_usd"]
                ),
                "diesel_logistics_tier": str(
                    logistics.loc[key, "diesel_logistics_tier"]
                ),
                "local_solar_utc_offset_hours": offset,
            }
        )
    manifest = {
        "description": "Audited paper resource year; normalized per full-year source peak; daily service schedules shifted to local mean solar time; 8760 intervals, no initial inventory row.",
        "wave_attribution": "Open-Meteo Marine historical wave data where indicated; see per-site source URLs and source descriptions. NOAA/NASA data where indicated.",
        "sites": records,
    }
    (output_dir / "paper_sites.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def check():
    with tempfile.TemporaryDirectory(prefix="sirenolite-paper-presets-") as temp:
        staged = Path(temp) / "profiles"
        rebuilt = build(staged)
        retained = json.loads(
            (PACKAGED / "paper_sites.json").read_text(encoding="utf-8")
        )
        if rebuilt.keys() != retained.keys() or len(rebuilt["sites"]) != len(
            retained["sites"]
        ):
            raise ValueError("Packaged paper site manifest structure has changed.")
        max_error = 0.0
        for fresh, old in zip(rebuilt["sites"], retained["sites"]):
            # Compression-library differences do not change physical inputs.
            # Check each stream hash against its own manifest and compare values.
            for record, directory in ((fresh, staged), (old, PACKAGED)):
                actual_hash = hashlib.sha256(
                    (directory / record["file"]).read_bytes()
                ).hexdigest()
                if actual_hash != record["sha256"]:
                    raise ValueError(f"Profile hash mismatch: {record['key']}")
            if {k: v for k, v in fresh.items() if k != "sha256"} != {
                k: v for k, v in old.items() if k != "sha256"
            }:
                raise ValueError(f"Profile source metadata differs: {old['key']}")
            a, b = pd.read_csv(staged / fresh["file"]), pd.read_csv(
                PACKAGED / old["file"]
            )
            if (
                list(a.columns) != list(b.columns)
                or a.shape != b.shape
                or not np.allclose(a, b, rtol=0, atol=1e-14)
            ):
                raise ValueError(f"Packaged resource chronology differs: {old['key']}")
            max_error = max(
                max_error, float(np.max(np.abs(a.to_numpy() - b.to_numpy())))
            )
        print(
            f"Verified {len(retained['sites'])} source profiles; maximum absolute difference {max_error:.3g}."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check",
        action="store_true",
        help="Compare canonical inputs without changing package data.",
    )
    action.add_argument(
        "--output-dir", type=Path, help="Stage regenerated profiles in a new directory."
    )
    args = parser.parse_args(argv)
    if args.check:
        check()
    else:
        manifest = build(args.output_dir)
        print(f"Staged {len(manifest['sites'])} profiles in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
