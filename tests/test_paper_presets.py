"""Contracts for portable paper inputs, interval accounting and provenance."""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sirenolite import model, presets
from sirenolite.config import configuration_from_payload


def test_all_sites_offline_sources_and_daily_demands():
    sites = presets.list_sites()
    assert len(sites) == 19 and len({s["key"] for s in sites}) == 19
    for site in sites:
        path = presets.DATA / site["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == site["sha256"]
        c = presets.load_site_config(site["key"], "combined")
        v = model.load_resource_inputs(c)
        # Independently integrate the repeated daily service, including hour 0.
        assert len(v["Load"]) == 8761
        assert np.sum(v["Load"][1:] + c["simulation"]["hotel_load"]) == pytest.approx(
            5684875
        )
        assert np.sum(v["PotableWater_demand_l"][1:]) == pytest.approx(365 * 151.416)
        local_hours = (np.arange(8760) + site["local_solar_utc_offset_hours"]) % 24
        assert set(local_hours[v["PotableWater_demand_l"][1:] > 0]) == {23}
        assert set(local_hours[v["Load"][1:] > 2000]) == {6, 7, 18, 19}
        for key in ("Wind", "Solar", "Wave"):
            assert (
                np.min(v[key + "_unitfactor"]) >= 0
                and np.max(v[key + "_unitfactor"]) <= 1
            )


def test_short_horizon_preserves_year_normalization_and_first_hour():
    annual = presets.load_site_config()
    full = model.load_resource_inputs(annual)
    short = presets.load_site_config(horizon_hours=168)
    sliced = model.load_resource_inputs(short)
    for key in full:
        assert np.allclose(sliced[key][1:], full[key][1:169], atol=1e-14)
    assert short["paper"]["delivery_repeat_factor"] == pytest.approx(43800 / 168)
    short["simulation"]["profile_start_offset_hours"] = 8760 - 167
    with pytest.raises(ValueError, match="exceed available"):
        model.load_resource_inputs(short)


def test_preset_roundtrip_and_rejected_backend_contracts():
    c = presets.load_site_config()
    roundtrip, _ = configuration_from_payload(c)
    assert roundtrip == c
    c["solver"]["remote"] = True
    with pytest.raises(ValueError, match="locally"):
        model.validate_config(c)
    c["solver"]["remote"] = False
    c["objective"] = "total_mass"
    with pytest.raises(ValueError, match="total_cost"):
        model.validate_config(c)
    for site, mission, hours in [
        ("unknown", "combined", 24),
        (presets.DEFAULT_SITE_KEY, "unknown", 24),
        (presets.DEFAULT_SITE_KEY, "combined", 8761),
    ]:
        with pytest.raises(ValueError):
            presets.load_site_config(site, mission, hours)


@pytest.mark.parametrize(
    "hours", [None, True, 0, -1, 1.5, float("inf"), float("nan"), "24"]
)
def test_invalid_preset_horizons_report_the_public_input_contract(hours):
    with pytest.raises(ValueError, match="1–8760 whole hours"):
        presets.load_site_config(horizon_hours=hours)


def test_portable_annual_reference_is_complete_and_selects_discrete_diesel_case():
    from scripts import verify_gui_paper_cases as replay

    all_cases, provenance = replay.reference_cases(all_cases=True)
    assert len(all_cases) == provenance["cases"] == 76
    selected, _ = replay.reference_cases()
    assert len(selected) == 5
    anchorage = selected.loc[
        selected.site_key.eq("anchorage_alaska_46081_oneyear")
    ].iloc[0]
    assert anchorage.generator_installed == 1
    assert anchorage.generator_delivery_trips_per_year == 1
    assert anchorage.generator_capacity_w > 500


def test_paper_replay_refuses_changed_reference_and_existing_outputs(
    tmp_path, monkeypatch
):
    from scripts import verify_gui_paper_cases as replay

    with pytest.raises(FileExistsError):
        replay.run(tmp_path)
    changed = tmp_path / "changed.csv"
    changed.write_bytes(replay.REFERENCE.read_bytes() + b"\n")
    monkeypatch.setattr(replay, "REFERENCE", changed)
    with pytest.raises(ValueError, match="provenance mismatch"):
        replay.reference_cases()
