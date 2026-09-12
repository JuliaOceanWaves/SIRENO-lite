(function () {
  "use strict";

  const data = window.SIRENO_HYBRID_MAP_DATA;
  const core = window.SIRENOHybridCore;
  const colors = { wind: "#14866d", solar: "#c58f00", wave: "#2563eb" };
  const svg = document.getElementById("world-map");
  const siteSelect = document.getElementById("site-select");
  const latInput = document.getElementById("latitude");
  const lonInput = document.getElementById("longitude");
  const summaryEl = document.getElementById("solution-summary");
  const mapReadout = document.getElementById("map-readout");
  const detailPanel = document.getElementById("detail-panel");
  const detailTitle = document.getElementById("detail-title");
  const canvas = document.getElementById("dispatch-plot");
  const ctx = canvas.getContext("2d");
  const startDayInput = document.getElementById("start-day");
  const windowDaysInput = document.getElementById("window-days");
  const windowReadout = document.getElementById("window-readout");
  const statusEl = document.getElementById("action-status");
  const pointSelect = document.getElementById("point-select");

  const controls = {
    loadW: document.getElementById("load-w"),
    resourceScale: document.getElementById("resource-scale"),
    waveCapture: document.getElementById("wave-capture"),
    outageDays: document.getElementById("outage-days"),
    costScale: document.getElementById("cost-scale"),
  };

  const outputs = {
    loadW: document.getElementById("load-w-out"),
    resourceScale: document.getElementById("resource-scale-out"),
    waveCapture: document.getElementById("wave-capture-out"),
    outageDays: document.getElementById("outage-days-out"),
    costScale: document.getElementById("cost-scale-out"),
  };

  const state = {
    points: [],
    selectedId: null,
  };

  function attempt(action) {
    try {
      action();
      statusEl.textContent = "";
    } catch (error) {
      statusEl.textContent = error.message || "The action could not be completed.";
    }
  }

  function inputCoordinates() {
    if (!latInput.value.trim() || !lonInput.value.trim()) {
      throw new Error("Enter both latitude and longitude before adding or updating a point.");
    }
    return core.validatedCoordinates(Number(latInput.value), Number(lonInput.value));
  }

  function siteByKey(siteKey) {
    return data.sites.find((site) => site.siteKey === siteKey) || data.sites[0];
  }

  function currentControls() {
    return {
      loadW: Number(controls.loadW.value),
      resourceScale: Number(controls.resourceScale.value),
      waveCapture: Number(controls.waveCapture.value),
      outageDays: Number(controls.outageDays.value),
      costScale: Number(controls.costScale.value),
    };
  }

  function setControls(values) {
    Object.keys(controls).forEach((key) => {
      if (values[key] !== undefined) {
        controls[key].value = values[key];
      }
    });
    updateControlOutputs();
  }

  function updateControlOutputs() {
    outputs.loadW.value = `${Number(controls.loadW.value).toFixed(1)} W`;
    outputs.resourceScale.value = `${Number(controls.resourceScale.value).toFixed(2)}x`;
    outputs.waveCapture.value = `${Number(controls.waveCapture.value).toFixed(2)}x`;
    outputs.outageDays.value = `${Number(controls.outageDays.value).toFixed(1)} d`;
    outputs.costScale.value = `${Number(controls.costScale.value).toFixed(2)}x`;
  }

  function selectedPoint() {
    return state.points.find((point) => point.id === state.selectedId) || null;
  }

  function selectedSite() {
    const point = selectedPoint();
    return point ? siteByKey(point.siteKey) : siteByKey(siteSelect.value);
  }

  function svgPathForRing(ring) {
    return ring.map(([lon, lat], index) => {
      const pt = core.project(lon, lat);
      return `${index === 0 ? "M" : "L"}${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`;
    }).join(" ") + " Z";
  }

  function markerRadius(solution) {
    const kw = Math.max(solution.renewableW / 1000, 0.05);
    return Math.min(28, Math.max(9, 7 + Math.sqrt(kw) * 6));
  }

  function piePath(cx, cy, radius, startAngle, endAngle) {
    const start = (startAngle - 90) * Math.PI / 180;
    const end = (endAngle - 90) * Math.PI / 180;
    const x1 = cx + radius * Math.cos(start);
    const y1 = cy + radius * Math.sin(start);
    const x2 = cx + radius * Math.cos(end);
    const y2 = cy + radius * Math.sin(end);
    const large = endAngle - startAngle > 180 ? 1 : 0;
    return `M${cx} ${cy} L${x1} ${y1} A${radius} ${radius} 0 ${large} 1 ${x2} ${y2} Z`;
  }

  function drawPie(group, point, solution) {
    const projected = core.project(point.lon, point.lat);
    const radius = markerRadius(solution);
    let angle = 0;
    core.pieFractions(solution).forEach((slice) => {
      if (slice.fraction <= 0) {
        return;
      }
      const end = angle + slice.fraction * 360;
      // A complete SVG arc with identical endpoints has no area.
      const fullCircle = slice.fraction >= 1 - 1e-12;
      const path = document.createElementNS("http://www.w3.org/2000/svg", fullCircle ? "circle" : "path");
      if (fullCircle) {
        path.setAttribute("cx", projected.x);
        path.setAttribute("cy", projected.y);
        path.setAttribute("r", radius);
      } else {
        path.setAttribute("d", piePath(projected.x, projected.y, radius, angle, end));
      }
      path.setAttribute("fill", colors[slice.key]);
      path.setAttribute("stroke", "#ffffff");
      path.setAttribute("stroke-width", "0.8");
      group.appendChild(path);
      angle = end;
    });
    const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    ring.setAttribute("cx", projected.x);
    ring.setAttribute("cy", projected.y);
    ring.setAttribute("r", radius);
    ring.setAttribute("class", "point-ring");
    group.appendChild(ring);
    if (point.id === state.selectedId) {
      const selected = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      selected.setAttribute("cx", projected.x);
      selected.setAttribute("cy", projected.y);
      selected.setAttribute("r", radius + 3);
      selected.setAttribute("class", "selected-ring");
      group.appendChild(selected);
    }
  }

  function drawMap() {
    svg.innerHTML = "";
    const clip = document.createElementNS("http://www.w3.org/2000/svg", "clipPath");
    clip.setAttribute("id", "map-extent");
    const extent = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    extent.setAttribute("width", 1000);
    extent.setAttribute("height", 520);
    clip.appendChild(extent);
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.appendChild(clip);
    svg.appendChild(defs);
    data.land.forEach((ring) => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", svgPathForRing(ring));
      path.setAttribute("class", "land-path");
      path.setAttribute("clip-path", "url(#map-extent)");
      svg.appendChild(path);
    });

    data.sites.forEach((site) => {
      const pt = core.project(site.longitude, site.latitude);
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("cx", pt.x);
      dot.setAttribute("cy", pt.y);
      dot.setAttribute("r", 3.1);
      dot.setAttribute("class", "site-dot");
      dot.addEventListener("click", (event) => {
        event.stopPropagation();
        latInput.value = site.latitude.toFixed(3);
        lonInput.value = site.longitude.toFixed(3);
        siteSelect.value = site.siteKey;
        attempt(addPointFromInputs);
      });
      svg.appendChild(dot);
    });

    state.points.forEach((point) => {
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("class", "point-group");
      group.setAttribute("role", "button");
      group.setAttribute("tabindex", "0");
      const site = siteByKey(point.siteKey);
      group.setAttribute("aria-label", `Select ${point.id}: ${site.label}`);
      const solution = core.solutionForControls(site, point.controls);
      drawPie(group, point, solution);
      group.addEventListener("click", (event) => {
        event.stopPropagation();
        selectPoint(point.id);
        openDetail();
      });
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectPoint(point.id);
          openDetail();
        }
      });
      svg.appendChild(group);
    });
  }

  function renderSummary() {
    const point = selectedPoint();
    const site = selectedSite();
    const values = point ? point.controls : currentControls();
    const solution = core.solutionForControls(site, values);
    const rows = [
      ["Profile", site.label],
      ["WEC case", solution.variantLabel],
      ["Capacity", `${core.formatNumber(solution.renewableW / 1000, 3)} kW`],
      ["Wind", `${core.formatNumber(solution.windW / 1000, 3)} kW`],
      ["Solar", `${core.formatNumber(solution.solarW / 1000, 3)} kW`],
      ["Wave", `${core.formatNumber(solution.waveW / 1000, 3)} kW`],
      ["Battery", `${core.formatNumber(solution.batteryWh / 1000, 3)} kWh`],
      ["Mass", `${core.formatNumber(solution.massKg, 1)} kg`],
      ["Cost", `$${core.formatNumber(solution.costUsd, 0)}`],
      ["Cost/W", `$${core.formatNumber(solution.costPerW, 2)}/W`],
    ];
    summaryEl.innerHTML = rows.map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join("");
    mapReadout.textContent = `${site.label}: ${solution.note}`;
    document.getElementById("profile-note").textContent = `${site.resourceSource}. ${site.note}`;
    document.getElementById("delete-point").disabled = !point;
    pointSelect.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = state.points.length ? "Select a saved point" : "No saved points";
    pointSelect.appendChild(placeholder);
    state.points.forEach((saved) => {
      const option = document.createElement("option");
      option.value = saved.id;
      option.textContent = `${saved.id}: ${siteByKey(saved.siteKey).label}`;
      pointSelect.appendChild(option);
    });
    pointSelect.value = state.selectedId || "";
    pointSelect.disabled = !state.points.length;
  }

  function selectPoint(id) {
    const point = state.points.find((item) => item.id === id);
    if (!point) {
      return;
    }
    state.selectedId = id;
    // Keep saved coordinates exact: a later scenario edit reads these fields
    // back into the point even when the user has not edited its location.
    latInput.value = String(point.lat);
    lonInput.value = String(point.lon);
    siteSelect.value = point.siteKey;
    setControls(point.controls);
    drawMap();
    renderSummary();
    drawPlot();
  }

  function addPointFromInputs() {
    const { lat, lon } = inputCoordinates();
    const nearest = core.nearestSite(data, lat, lon);
    const site = siteByKey(siteSelect.value || nearest.site.siteKey);
    const point = core.pointRecord({
      id: core.nextPointId(state.points),
      lat,
      lon,
      site,
      controls: currentControls(),
    });
    state.points.push(point);
    selectPoint(point.id);
  }

  function updateSelectedPoint() {
    const coordinates = inputCoordinates();
    const values = core.validatedControls(currentControls());
    const point = selectedPoint();
    if (!point) {
      renderSummary();
      drawPlot();
      return;
    }
    point.lat = coordinates.lat;
    point.lon = coordinates.lon;
    point.siteKey = siteSelect.value;
    point.controls = values;
    drawMap();
    renderSummary();
    drawPlot();
  }

  function deleteSelectedPoint() {
    const point = selectedPoint();
    if (!point) {
      return;
    }
    state.points = state.points.filter((item) => item.id !== point.id);
    state.selectedId = state.points.length ? state.points[state.points.length - 1].id : null;
    if (state.selectedId) {
      selectPoint(state.selectedId);
    } else {
      drawMap();
      renderSummary();
      drawPlot();
    }
  }

  function download(filename, mimeType, text) {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function saveDataset() {
    const payload = core.datasetForSave(state.points);
    download("sireno_hybrid_map_dataset.json", "application/json", JSON.stringify(payload, null, 2));
  }

  function exportReport() {
    download("sireno_hybrid_map_report.md", "text/markdown", core.reportMarkdown(state.points, data));
  }

  function loadDataset(file) {
    const reader = new FileReader();
    reader.onload = () => {
      attempt(() => {
        const parsed = JSON.parse(reader.result);
        // Validate the whole import before replacing the user's saved points.
        const points = core.validateDataset(parsed, data);
        state.points = points;
        state.selectedId = points.length ? points[0].id : null;
        if (state.selectedId) {
          selectPoint(state.selectedId);
        } else {
          drawMap();
          renderSummary();
          drawPlot();
        }
      });
    };
    reader.onerror = () => { statusEl.textContent = "Unable to read the selected dataset file."; };
    reader.readAsText(file);
  }

  function openDetail() {
    detailPanel.hidden = false;
    drawPlot();
  }

  function drawPlot() {
    const site = selectedSite();
    const startDay = Number(startDayInput.value);
    const windowDays = Number(windowDaysInput.value);
    const slice = core.sliceTimeseries(site, startDay, windowDays);
    if (!slice.timeDays.length) {
      return;
    }
    const width = canvas.width;
    const height = canvas.height;
    const pad = { left: 70, right: 22, top: 34, bottom: 42 };
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    const x0 = slice.timeDays[0];
    const x1 = slice.timeDays[slice.timeDays.length - 1] || x0 + 1;
    const seriesMax = Math.max(
      120,
      ...slice.windW,
      ...slice.solarW,
      ...slice.waveW,
      ...slice.loadW,
    );
    const battMax = Math.max(1, ...slice.batteryKWh);
    const mid = Math.round(height * 0.52);
    const topBottom = mid - 22;
    const lowerTop = mid + 28;
    const plotW = width - pad.left - pad.right;
    const xScale = (day) => pad.left + ((day - x0) / Math.max(x1 - x0, 1e-9)) * plotW;
    const yPower = (value) => pad.top + (1 - value / seriesMax) * (topBottom - pad.top);
    const yBatt = (value) => lowerTop + (1 - value / battMax) * (height - pad.bottom - lowerTop);

    function line(values, yFn, stroke, widthPx) {
      ctx.beginPath();
      values.forEach((value, index) => {
        const x = xScale(slice.timeDays[index]);
        const y = yFn(value);
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.strokeStyle = stroke;
      ctx.lineWidth = widthPx;
      ctx.stroke();
    }
    ctx.fillStyle = "#1d2528";
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i += 1) {
      const day = x0 + (x1 - x0) * i / 4;
      ctx.fillText(day.toFixed(1), xScale(day), height - pad.bottom + 17);
    }
    ctx.textAlign = "left";

    ctx.strokeStyle = "#dde3e2";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
      const yA = pad.top + i * (topBottom - pad.top) / 4;
      const yB = lowerTop + i * (height - pad.bottom - lowerTop) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, yA);
      ctx.lineTo(width - pad.right, yA);
      ctx.moveTo(pad.left, yB);
      ctx.lineTo(width - pad.right, yB);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText((seriesMax * (1 - i / 4)).toFixed(0), pad.left - 8, yA + 4);
      ctx.fillText((battMax * (1 - i / 4)).toFixed(1), pad.left - 8, yB + 4);
    }
    ctx.textAlign = "left";

    line(slice.windW, yPower, colors.wind, 1.4);
    line(slice.solarW, yPower, colors.solar, 1.4);
    line(slice.waveW, yPower, colors.wave, 1.4);
    line(slice.loadW, yPower, "#8f2e28", 1.8);
    line(slice.batteryKWh, yBatt, "#6f3fb5", 1.8);

    ctx.fillStyle = "#1d2528";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("Generation and demand [W]", pad.left, 19);
    ctx.fillText("Stored battery energy [kWh]", pad.left, lowerTop - 10);
    ctx.fillText("Day in archived year", width / 2 - 60, height - 7);
    ctx.fillStyle = colors.wind;
    ctx.fillText("Wind", width - 250, 19);
    ctx.fillStyle = colors.solar;
    ctx.fillText("Solar", width - 202, 19);
    ctx.fillStyle = colors.wave;
    ctx.fillText("Wave", width - 150, 19);
    ctx.fillStyle = "#8f2e28";
    ctx.fillText("Load", width - 96, 19);

    detailTitle.textContent = `${site.label}: archived 100 W / WEC 1x dispatch`;
    windowReadout.value = `days ${startDay}-${Math.min(startDay + windowDays, site.timeseries.timeDays.length / 24)}`;
    canvas.setAttribute("aria-label", `${detailTitle.textContent}. ${windowReadout.value}. Generation, load, and stored battery energy. Scenario controls do not change these reference curves.`);
  }

  function initialize() {
    document.getElementById("data-status").textContent = `${data.sites.length} archived one-year profiles; screening estimates, no live optimization`;
    data.sites.forEach((site) => {
      const option = document.createElement("option");
      option.value = site.siteKey;
      option.textContent = site.label;
      siteSelect.appendChild(option);
    });
    const first = data.sites.find((site) => site.siteKey.includes("hawaii")) || data.sites[0];
    siteSelect.value = first.siteKey;
    latInput.value = first.latitude.toFixed(3);
    lonInput.value = first.longitude.toFixed(3);
    updateControlOutputs();
    drawMap();
    renderSummary();
  }

  svg.addEventListener("click", (event) => {
    // SVG's viewBox can be letterboxed within its element. Transform actual
    // screen coordinates, rather than treating that whitespace as geography.
    const screenPoint = svg.createSVGPoint();
    screenPoint.x = event.clientX;
    screenPoint.y = event.clientY;
    const local = screenPoint.matrixTransform(svg.getScreenCTM().inverse());
    if (local.x < 0 || local.x > 1000 || local.y < 0 || local.y > 520) return;
    attempt(() => {
      const point = core.unproject(local.x, local.y);
      const nearest = core.nearestSite(data, point.lat, point.lon);
      latInput.value = point.lat.toFixed(3);
      lonInput.value = point.lon.toFixed(3);
      siteSelect.value = nearest.site.siteKey;
      addPointFromInputs();
    });
  });

  Object.values(controls).forEach((input) => {
    input.addEventListener("input", () => {
      updateControlOutputs();
      attempt(updateSelectedPoint);
    });
  });
  [latInput, lonInput, siteSelect].forEach((input) => input.addEventListener("input", () => attempt(updateSelectedPoint)));
  pointSelect.addEventListener("change", () => selectPoint(pointSelect.value));
  document.getElementById("add-point").addEventListener("click", () => attempt(addPointFromInputs));
  document.getElementById("nearest-profile").addEventListener("click", () => attempt(() => {
    const { lat, lon } = inputCoordinates();
    siteSelect.value = core.nearestSite(data, lat, lon).site.siteKey;
    updateSelectedPoint();
  }));
  document.getElementById("delete-point").addEventListener("click", deleteSelectedPoint);
  document.getElementById("save-dataset").addEventListener("click", saveDataset);
  document.getElementById("export-report").addEventListener("click", exportReport);
  document.getElementById("open-detail").addEventListener("click", openDetail);
  document.getElementById("close-detail").addEventListener("click", () => {
    detailPanel.hidden = true;
  });
  document.getElementById("load-dataset").addEventListener("change", (event) => {
    if (event.target.files.length) {
      loadDataset(event.target.files[0]);
      event.target.value = "";
    }
  });
  [startDayInput, windowDaysInput].forEach((input) => input.addEventListener("input", drawPlot));

  initialize();
})();
