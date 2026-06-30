import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "docs" / "hybrid-map-tool"
DATA_JS = TOOL_DIR / "data.js"
CORE_JS = TOOL_DIR / "app-core.js"


def load_data_payload():
    text = DATA_JS.read_text(encoding="utf-8")
    match = re.fullmatch(r"window\.SIRENO_HYBRID_MAP_DATA=(.*);\n?", text, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def test_hybrid_map_tool_data_schema():
    payload = load_data_payload()

    assert payload["schemaVersion"] == 1
    assert payload["baseLoadW"] == 100.0
    assert payload["baselineVariant"] == "wec_1x"
    assert len(payload["land"]) > 100
    assert len(payload["sites"]) >= 15

    for site in payload["sites"]:
        assert site["siteKey"].endswith("_oneyear")
        assert -60.0 <= site["latitude"] <= 86.0
        assert -180.0 <= site["longitude"] <= 180.0
        assert len(site["variants"]) == 5
        assert {variant["variant"] for variant in site["variants"]} == {
            "wec_0p25x",
            "wec_0p5x",
            "wec_1x",
            "wec_2x",
            "wec_4x",
        }
        timeseries = site["timeseries"]
        assert len(timeseries["timeDays"]) == 8760
        assert len(timeseries["loadW"]) == 8760
        assert len(timeseries["batteryKWh"]) == 8760
        assert min(timeseries["loadW"]) == 100.0
        assert max(timeseries["loadW"]) == 100.0


def test_hybrid_map_core_roundtrip_with_node():
    script = f"""
global.window = {{}};
const core = require({json.dumps(str(CORE_JS))});
require({json.dumps(str(DATA_JS))});
const data = global.window.SIRENO_HYBRID_MAP_DATA;
const nearest = core.nearestSite(data, 23.5, -153.8).site;
const controls = {{loadW: 150, resourceScale: 1.2, waveCapture: 2, outageDays: 1.5, costScale: 1.1}};
const solution = core.solutionForControls(nearest, controls);
const point = core.pointRecord({{id: "p1", lat: 23.5, lon: -153.8, site: nearest, controls}});
const dataset = core.datasetForSave([point]);
const loaded = core.validateDataset(dataset, data);
const report = core.reportMarkdown(loaded, data);
console.log(JSON.stringify({{
  siteKey: nearest.siteKey,
  variant: solution.variant,
  renewableW: solution.renewableW,
  batteryWh: solution.batteryWh,
  loadedCount: loaded.length,
  hasReport: report.includes("SIRENO-Lite Hybrid Map Report")
}}));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["siteKey"] == "hawaii_pacific_51000_oneyear"
    assert result["variant"] == "wec_2x"
    assert result["renewableW"] > 0.0
    assert result["batteryWh"] > 150.0 * 24.0
    assert result["loadedCount"] == 1
    assert result["hasReport"] is True


def test_hybrid_map_static_assets_reference_local_files():
    html = (TOOL_DIR / "tool.html").read_text(encoding="utf-8")

    assert "data.js" in html
    assert "app-core.js" in html
    assert "app.js" in html
    assert "https://" not in html
    assert "http://" not in html
