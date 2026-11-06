# SIRENO-Lite Hybrid Map Tool

<iframe src="tool.html" title="SIRENO-Lite Hybrid Map Tool" style="width: 100%; height: 900px; border: 0;"></iframe>

The map compares 24 archived, one-year designs with a continuous 100 W electric
load. It runs locally in the browser using the checked-in results. A map click
assigns the nearest available ocean resource profile to a marker; it does not
retrieve wind, solar, or wave data at that coordinate. Coordinate edits retain
the selected profile; **Use nearest profile** changes that assignment. The
**Saved point** menu provides keyboard access to all markers.

Pie colors identify wind, solar, and wave shares of non-fuel generation sizing.
Marker sizes are qualitative; use **Design estimate** for numerical comparisons.

Select one of five solved wave-energy-converter (WEC) capture cases: 0.25, 0.5,
1, 2, or 4 times the reference capture. The other controls rescale that archived
design and do not run an optimizer. Generation capacity scales with load divided
by the uniform resource multiplier. Battery energy scales with load, since
the rescaled generation supplies the same hourly energy history. The resource
multiplier changes generation capital cost and supported mass; operating energy
cost and battery capacity follow the load. A common cost multiplier changes the
reported cost without changing the design.

Outage reserve adds stored energy equal to load times outage duration divided by
the one-way battery efficiency, `sqrt(0.9)`. It assumes that allowance is fully
charged when an outage begins. The added cost and mass include battery and
platform support: 0.232 USD/Wh and 0.0104 kg/Wh under the archived assumptions.
The reserve control does not model recharge, battery power limits, or outage
dispatch. All rescaled designs require a new optimization and feasibility check
before use in an engineering design.

**View archived dispatch** always shows the original 100 W, WEC 1x curves for
the selected profile, independent of scenario controls. These archived cases
force battery recharge at the end of the year and are distinct from the current
cyclic mission study. **Profile source and limitations** identifies measured,
gridded, and proxy resource inputs for each site. These comparisons support
qualitative architecture screening, not hardware validation.

**Save** downloads marker coordinates, assigned profiles, and controls as JSON.
**Load** replaces the current points after validating the complete file; save
your current points first if you want to retain them. Invalid files leave the
existing points intact. **Report** downloads the same estimates with scenario
settings and resource limitations. Files remain on your computer.

The data exporter is [export_data.py](export_data.py). From the repository root,
run `python docs/hybrid-map-tool/export_data.py` in the project environment to
rebuild the static payload from the retained paper archive. The independent
`paper/` checkout and its archived results must be present. Exporting does not
acquire new resource data or solve new cases. The payload records its input
paths and frozen scaling assumptions.

For release verification, run `python -m pytest tests/test_hybrid_map_tool.py`
with Node.js installed. The independent data checks reconstruct hourly bus and
storage balances plus component, platform, and operating costs. Run
`SIRENOLITE_BROWSER_TESTS=1 python -m pytest tests/test_hybrid_map_tool.py -m browser`
with Chrome or Chromium installed for real add/edit/import/export, map, plot, and
mobile-layout workflows. Set `SIRENOLITE_CHROME` if its executable is not found
automatically. The browser uses an isolated temporary profile. Use
`python docs/hybrid-map-tool/export_data.py --check` to verify that the payload
exactly reproduces the retained archive.
