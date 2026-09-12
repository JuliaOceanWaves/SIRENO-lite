import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "docs" / "hybrid-map-tool"
DATA_JS = TOOL_DIR / "data.js"
CORE_JS = TOOL_DIR / "app-core.js"


def run_core(script):
    """Exercise the shipped JavaScript, with failures surfaced as assertions."""
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required for the public-release hybrid map tests.")
    completed = subprocess.run(
        [node, "-e", f"const core = require({json.dumps(str(CORE_JS))});\n" + script],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=30,
    )
    return json.loads(completed.stdout)


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


def test_archived_dispatch_obeys_bus_and_storage_energy_balances():
    """Check every retained hour with tolerances set by payload quantization.

    These verify archived solver output, not the physical resource measurements.
    Battery values are rounded to 0.0001 kWh, power to 0.001 W. State at
    sample t includes interval t for t >= 1 in this archived LP convention.
    """
    for site in load_data_payload()["sites"]:
        ts = {key: np.asarray(values) for key, values in site["timeseries"].items()}
        variant = next(item for item in site["variants"] if item["variant"] == "wec_1x")
        generation = ts["windW"] + ts["solarW"] + ts["waveW"]
        np.testing.assert_allclose(
            generation + ts["batteryDischargeW"], ts["loadW"] + ts["batteryChargeW"],
            rtol=0, atol=0.003, err_msg=site["siteKey"],
        )
        np.testing.assert_allclose(
            np.diff(ts["batteryKWh"] * 1000),
            np.sqrt(0.9) * ts["batteryChargeW"][1:] - ts["batteryDischargeW"][1:] / np.sqrt(0.9),
            rtol=0, atol=0.102, err_msg=site["siteKey"],
        )
        assert ts["batteryKWh"].min() >= 0
        assert ts["batteryKWh"].max() * 1000 <= variant["batt_scale_wh"] + 0.051
        assert ts["batteryKWh"][-1] * 1000 == pytest.approx(0.98 * variant["batt_scale_wh"], abs=0.051)
        assert variant["renewable_operating_cost_usd"] == pytest.approx(
            generation.mean() * 5 * 365 * 24 * 1e-5, abs=0.001,
        )


def test_archived_design_cost_and_mass_include_platform_and_operating_cost():
    for site in load_data_payload()["sites"]:
        for case in site["variants"]:
            wind, solar, wave, battery = (case[key] for key in (
                "wind_scale_w", "solar_scale_w", "wave_scale_w", "batt_scale_wh",
            ))
            supported_mass = wind * 0.03 + solar * 0.07 + wave * 0.08 + battery * 0.008
            platform_mass = supported_mass * 0.3
            assert case["total_mass_kg"] == pytest.approx(supported_mass + platform_mass, abs=2e-6)
            capital = wind * 4 + solar * 1.67 + wave * 2 + battery * 0.16 + platform_mass * 30
            assert case["total_cost_usd"] == pytest.approx(capital + case["renewable_operating_cost_usd"], abs=1e-5)


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
    payload = load_data_payload()
    site = next(site for site in payload["sites"] if site["siteKey"] == result["siteKey"])
    case = next(case for case in site["variants"] if case["variant"] == result["variant"])
    assert result["renewableW"] == pytest.approx(
        sum(case[key] for key in ("wind_scale_w", "solar_scale_w", "wave_scale_w")) * 1.25,
    )
    assert result["batteryWh"] == pytest.approx(case["batt_scale_wh"] * 1.5 + 5400 / np.sqrt(0.9))
    assert result["loadedCount"] == 1
    assert result["hasReport"] is True


def test_controls_scale_load_energy_and_resource_nameplate_separately():
    result = run_core("""
const site = {siteKey: 'toy', variants: [{variant: 'wec_1x', variant_label: 'WEC 1x',
  wave_capture_fraction: 1, wind_scale_w: 100, solar_scale_w: 200, wave_scale_w: 300,
  batt_scale_wh: 1000, total_cost_usd: 1000, total_mass_kg: 100,
  renewable_operating_cost_usd: 100}]};
console.log(JSON.stringify({
  base: core.solutionForControls(site, {}),
  doubleLoad: core.solutionForControls(site, {loadW: 200}),
  doubleResource: core.solutionForControls(site, {resourceScale: 2}),
  reserve: core.solutionForControls(site, {outageDays: 1}),
  doubleCost: core.solutionForControls(site, {costScale: 2})
}));
""")
    base = result["base"]
    for key in ("windW", "solarW", "waveW", "batteryWh", "costUsd", "massKg"):
        assert result["doubleLoad"][key] == pytest.approx(2 * base[key])
    assert result["doubleLoad"]["costPerW"] == pytest.approx(base["costPerW"])
    resource = result["doubleResource"]
    assert [resource[key] for key in ("windW", "solarW", "waveW")] == [50, 100, 150]
    assert resource["batteryWh"] == 1000  # Same net hourly energy demand.
    assert resource["costUsd"] == pytest.approx(666)  # Battery and operating costs persist.
    assert resource["massKg"] == pytest.approx(55.2)
    reserve = result["reserve"]
    assert reserve["reserveWh"] * np.sqrt(0.9) == pytest.approx(100 * 24)
    assert reserve["batteryWh"] - base["batteryWh"] == pytest.approx(reserve["reserveWh"])
    assert reserve["costUsd"] - base["costUsd"] == pytest.approx(reserve["reserveWh"] * 0.232)
    assert reserve["massKg"] - base["massKg"] == pytest.approx(reserve["reserveWh"] * 0.0104)
    assert result["doubleCost"]["costUsd"] == 2000
    for key in ("renewableW", "batteryWh", "massKg"):
        assert result["doubleCost"][key] == base[key]


def test_geographic_selection_wraps_the_antimeridian_and_uses_great_circle_distance():
    result = run_core("""
const acrossDateLine = {sites: [{siteKey: 'near', latitude: 0, longitude: -179},
  {siteKey: 'far', latitude: 0, longitude: 150}]};
const polar = {sites: [{siteKey: 'poleward', latitude: 85, longitude: 0},
  {siteKey: 'same-latitude', latitude: 80, longitude: 90}]};
const exact = core.nearestSite(acrossDateLine, 0, -179);
console.log(JSON.stringify({nearest: core.nearestSite(acrossDateLine, 0, 179).site.siteKey,
  distance: core.nearestSite(acrossDateLine, 0, 179).distance,
  polar: core.nearestSite(polar, 80, 0).site.siteKey,
  exactDistance: exact.distance,
  bounds: core.unproject(1000, 520),
  roundtrip: core.unproject(...Object.values(core.project(179, -45)))}));
""")
    assert result["nearest"] == "near"
    assert result["distance"] == pytest.approx(2)
    assert result["polar"] == "poleward"
    assert result["exactDistance"] == 0
    assert result["bounds"] == {"lon": 180, "lat": -60}
    assert result["roundtrip"] == pytest.approx({"lon": 179, "lat": -45})


@pytest.mark.parametrize("mutation, message", [
    ("dataset.points[0].lat = 91", "Latitude"),
    ("dataset.points[0].lat = null", "Latitude"),
    ("dataset.points[0].lon = NaN", "Longitude"),
    ("dataset.points[0].lon = '10'", "Longitude"),
    ("dataset.points[0].controls = []", "controls"),
    ("dataset.points[0].controls.loadW = 0", "loadW"),
    ("dataset.points[0].controls.resourceScale = Infinity", "resourceScale"),
    ("dataset.points[0].controls.outageDays = -1", "outageDays"),
    ("dataset.points[0].controls.costScale = null", "costScale"),
    ("dataset.points[0].controls.unknown = 1", "Unknown scenario control"),
    ("dataset.points[0].siteKey = 'missing'", "unknown site"),
    ("dataset.points.push({...dataset.points[0]})", "unique"),
    ("dataset.points[0].id = []", "text ID"),
    ("dataset.points[0] = null", "must be an object"),
])
def test_import_rejects_invalid_records_without_mutating_input(mutation, message):
    result = run_core(f"""
const data = {{sites: [{{siteKey: 'toy', variants: [{{wave_capture_fraction: 1}}]}}]}};
const dataset = {{schemaVersion: 1, points: [{{id: 'point-2', lat: 0, lon: 10, siteKey: 'toy', controls: {{}}}}]}};
{mutation};
const before = JSON.stringify(dataset);
let error = '';
try {{core.validateDataset(dataset, data);}} catch (exc) {{error = exc.message;}}
console.log(JSON.stringify({{error, unchanged: before === JSON.stringify(dataset)}}));
""")
    assert message in result["error"]
    assert result["unchanged"] is True


def test_import_normalizes_legacy_capture_and_allocates_unique_point_ids():
    result = run_core("""
const data = {sites: [{siteKey: 'toy', variants: [
  {wave_capture_fraction: 1}, {wave_capture_fraction: 2}, {wave_capture_fraction: 4}]}]};
const imported = core.validateDataset({schemaVersion: 1, points: [
  {id: 'point-2', lat: 0, lon: 10, siteKey: 'toy', controls: {waveCapture: 1.75}},
  {id: 'point-1', lat: 10, lon: 20, siteKey: 'toy', controls: {}}
]}, data);
const saved = core.datasetForSave(imported);
saved.points[0].controls.loadW = 150;
console.log(JSON.stringify({imported, nextId: core.nextPointId(imported),
  slice: core.sliceTimeseries({timeseries: {timeDays: Array.from({length: 48}, (_,i)=>i/24),
    loadW: Array.from({length:48}, (_,i)=>100+i)}}, 1.5, 14)}));
""")
    assert result["imported"][0]["controls"]["waveCapture"] == 2
    assert result["imported"][0]["controls"]["loadW"] == 100
    assert result["nextId"] == "point-3"
    assert result["slice"]["loadW"] == list(range(136, 148))
    assert result["slice"]["timeDays"] == pytest.approx([i / 24 for i in range(36, 48)])


def test_hybrid_map_static_assets_reference_local_files():
    html = (TOOL_DIR / "tool.html").read_text(encoding="utf-8")

    assert "data.js" in html
    assert "app-core.js" in html
    assert "app.js" in html
    assert "https://" not in html
    assert "http://" not in html


BROWSER_CHECKS = r"""
const frame = document.getElementById('tool');
frame.onload = async () => {
  const out = document.getElementById('browser-result');
  const w = frame.contentWindow, d = w.document;
  const failures = [];
  w.addEventListener('error', e => failures.push(e.message));
  const el = id => d.getElementById(id);
  const assert = (condition, message) => {if (!condition) throw new Error(message);};
  const count = () => d.querySelectorAll('.point-group').length;
  const click = id => el(id).click();
  const set = (id, value, event = 'input') => {
    el(id).value = value; el(id).dispatchEvent(new w.Event(event, {bubbles: true}));
  };
  const rows = () => Object.fromEntries([...el('solution-summary').querySelectorAll('dt')]
    .map(dt => [dt.textContent, dt.nextElementSibling.textContent]));
  let downloaded;
  w.URL.createObjectURL = blob => {downloaded = blob; return 'blob:test-download';};
  w.URL.revokeObjectURL = () => {};
  w.HTMLAnchorElement.prototype.click = function() {};
  const saved = async () => {click('save-dataset'); return JSON.parse(await downloaded.text());};
  const load = async text => {
    const originalRead = w.FileReader.prototype.readAsText;
    let completed;
    const done = new Promise(resolve => {completed = resolve;});
    w.FileReader.prototype.readAsText = function(...args) {
      this.addEventListener('loadend', completed, {once:true});
      return originalRead.apply(this, args);
    };
    const transfer = new w.DataTransfer();
    transfer.items.add(new w.File([text], 'points.json', {type:'application/json'}));
    el('load-dataset').files = transfer.files;
    el('load-dataset').dispatchEvent(new w.Event('change', {bubbles:true}));
    // Observe the real FileReader completion instead of racing its I/O with a
    // fixed timer (headless virtual time can advance before file I/O finishes).
    await done;
    w.FileReader.prototype.readAsText = originalRead;
  };
  try {
    assert(el('data-status').textContent.startsWith('24 archived'), 'all archived sites loaded');
    assert(el('detail-panel').hidden, 'initial detail is closed');
    assert(rows().Battery === '2.303 kWh', 'Hawaii reference storage displayed in kWh');
    click('add-point'); click('open-detail');
    assert(count() === 1, 'Add created one point');
    const image = el('dispatch-plot').toDataURL();
    set('load-w', '200');
    assert(rows().Battery === '4.606 kWh', 'doubling load doubles displayed battery');
    assert(rows().Capacity === '1.596 kW', 'doubling load doubles generation');
    set('resource-scale', '2');
    assert(rows().Battery === '4.606 kWh' && rows().Capacity === '0.798 kW',
      'resource multiplier preserves energy while reducing nameplate');
    set('wave-capture', '2');
    assert(rows()['WEC case'] === 'WEC 2x', 'discrete solved WEC case identified');
    assert(el('detail-title').textContent.includes('100 W / WEC 1x'), 'baseline plot labeled separately');
    assert(image === el('dispatch-plot').toDataURL(), 'baseline plot remains the archived series');
    set('start-day', '351'); set('window-days', '365');
    assert(el('window-readout').value === 'days 351-365', 'window label clips at available data');
    const beforeInvalid = await saved();
    set('latitude', ''); click('add-point');
    assert(count() === 1 && el('action-status').textContent.includes('both latitude'),
      'empty coordinates rejected with feedback');
    assert(JSON.stringify(await saved()) === JSON.stringify(beforeInvalid), 'invalid edit preserves point');
    set('latitude', '23.5');
    await load('{broken');
    assert(el('action-status').textContent.length > 0 && count() === 1, 'bad JSON preserves points and reports error');
    const duplicate = structuredClone(beforeInvalid);
    duplicate.points.push({...duplicate.points[0]});
    await load(JSON.stringify(duplicate));
    assert(el('action-status').textContent.includes('unique') && count() === 1, 'duplicate IDs rejected atomically');
    const imported = structuredClone(beforeInvalid);
    imported.points[0].id = 'point-2';
    imported.points[0].lat = 23.5123456789;
    imported.points[0].lon = -153.8123456789;
    imported.points[0].controls.loadW = 123.4;
    await load(JSON.stringify(imported));
    assert(el('action-status').textContent === '', 'valid import succeeds: '+el('action-status').textContent);
    assert(Number(el('load-w').value) === 123.4, 'imported value is not rounded to a slider step');
    set('cost-scale', '1.2');
    const afterScenarioEdit = (await saved()).points[0];
    assert(afterScenarioEdit.lat === imported.points[0].lat
      && afterScenarioEdit.lon === imported.points[0].lon,
      'scenario edits preserve imported coordinate precision');
    click('add-point');
    const afterAdd = await saved();
    assert(new Set(afterAdd.points.map(p=>p.id)).size === 2, 'add after import allocates unique ID');
    set('point-select', 'point-2', 'change');
    click('delete-point');
    assert(count() === 1 && (await saved()).points[0].id === 'point-1', 'delete affects selected point only');
    const svg = el('world-map');
    svg.style.height = '800px'; // Deliberately introduce SVG letterboxing.
    const screen = new w.DOMPoint(750, 130).matrixTransform(svg.getScreenCTM());
    svg.dispatchEvent(new w.MouseEvent('click', {clientX:screen.x, clientY:screen.y, bubbles:true}));
    const mapped = (await saved()).points.at(-1);
    // MouseEvent coordinates quantize to screen pixels in Chromium; this is
    // less than 0.2 degrees at the test viewport, far below the old mapping error.
    assert(Math.abs(mapped.lat - 49.5) < 0.2 && Math.abs(mapped.lon - 90) < 0.2,
      'letterboxed map coordinates use actual SVG transform');
    svg.style.height = '';
    assert(mapped.siteKey === w.SIRENOHybridCore.nearestSite(w.SIRENO_HYBRID_MAP_DATA,49.5,90).site.siteKey,
      'map point assigned nearest available profile');
    click('export-report');
    const report = await downloaded.text();
    assert(report.includes('Solved WEC capture case: 2.00x') && report.includes('Resource limitations:'),
      'report exports selected scenario and source limits');
    const single = w.SIRENO_HYBRID_MAP_DATA.sites.find(s => s.variants.some(v =>
      [v.wind_scale_w,v.solar_scale_w,v.wave_scale_w].filter(x=>x>0).length===1));
    const singleCase = single.variants.find(v=>[v.wind_scale_w,v.solar_scale_w,v.wave_scale_w].filter(x=>x>0).length===1);
    set('site-select', single.siteKey); set('wave-capture', singleCase.wave_capture_fraction);
    assert([...d.querySelectorAll('.point-group:last-child circle')].some(c=>c.hasAttribute('fill')),
      'single-resource marker renders a filled disk');
    assert(!failures.length, 'no uncaught browser errors: '+failures.join('; '));
    frame.style.width = '390px';
    await new Promise(resolve=>w.setTimeout(resolve,100));
    assert(d.documentElement.scrollWidth <= 391, 'mobile layout avoids horizontal overflow');
    frame.style.width = '100%';
    out.textContent = 'PASS: add, edit, physical scaling, baseline plot, window, invalid input, import, export, selection, SVG coordinates, single-resource marker, mobile layout';
  } catch (error) {out.textContent = 'FAIL: '+error.message+'; '+failures.join('; ');}
  finally {
    await fetch('/browser-result', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({result: out.textContent})});
  }
};
"""


@pytest.mark.browser
@pytest.mark.skipif(os.environ.get("SIRENOLITE_BROWSER_TESTS") != "1",
                    reason="Explicit browser gate: set SIRENOLITE_BROWSER_TESTS=1 and install Chrome/Chromium")
def test_real_browser_edit_import_export_and_plot_contracts(tmp_path):
    chrome = os.environ.get("SIRENOLITE_CHROME") or shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        candidate = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        chrome = str(candidate) if candidate.is_file() else None
    assert chrome, "The requested browser gate requires Chrome or Chromium (SIRENOLITE_CHROME)."
    harness = (
        '<!doctype html><html><body style="margin:0"><pre id="browser-result">RUNNING</pre>'
        '<iframe id="tool" style="width:100%;height:2000px;border:0"></iframe><script>'
        + BROWSER_CHECKS
        + "document.getElementById('tool').src='/tool.html';</script></body></html>"
    ).encode()
    completed = threading.Event()
    browser_result = {}

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/browser-check.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(harness)
            else:
                super().do_GET()

        def do_POST(self):
            if self.path != "/browser-result":
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            browser_result["text"] = payload["result"]
            self.send_response(204)
            self.end_headers()
            completed.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(TOOL_DIR)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    screenshot = os.environ.get("SIRENOLITE_BROWSER_SCREENSHOT", str(tmp_path / "map-browser.png"))
    command = [
        chrome, "--headless", "--no-first-run", "--no-default-browser-check",
        "--disable-background-networking", "--disable-component-update", "--disable-sync",
        "--disable-gpu", "--window-size=1280,2300", "--virtual-time-budget=5000",
        f"--user-data-dir={tmp_path / 'chrome-profile'}", f"--screenshot={screenshot}", "--dump-dom",
        f"http://127.0.0.1:{server.server_port}/browser-check.html",
    ]
    process = None
    stdout_path = tmp_path / "chrome-stdout.log"
    stderr_path = tmp_path / "chrome-stderr.log"

    def stop_browser(force=False):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=10)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass  # The isolated process exited between poll and cleanup.

    try:
        with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
            process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file,
                                       text=True, start_new_session=True)
            # The page reports its actual final assertion result to this local
            # harness. A managed Chrome process may remain alive indefinitely
            # after finishing, so process exit/DOM-dump timing is not the gate.
            # Keep a hard wall-clock bound for a stalled or failed renderer.
            deadline = time.monotonic() + 90
            while not completed.wait(timeout=0.1):
                if process.poll() is not None or time.monotonic() >= deadline:
                    break
            assert completed.is_set(), (
                f"Browser did not complete the checks; exit {process.poll()}: "
                + stderr_path.read_text()[-1500:]
            )
            assert browser_result["text"].startswith("PASS:"), browser_result["text"]
            # Give the optional visual artifact time to finish independently of
            # the functional result, without requiring background Chrome exit.
            screenshot_deadline = min(deadline, time.monotonic() + 10)
            while not Path(screenshot).is_file() and time.monotonic() < screenshot_deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.1)
    finally:
        if process is not None and process.poll() is None:
            stop_browser()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop_browser(force=True)
                process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
