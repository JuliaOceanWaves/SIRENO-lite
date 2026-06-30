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
    nextId: 1,
  };

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
    outputs.loadW.value = `${Number(controls.loadW.value).toFixed(0)} W`;
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
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", piePath(projected.x, projected.y, radius, angle, end));
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
    data.land.forEach((ring) => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", svgPathForRing(ring));
      path.setAttribute("class", "land-path");
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
        addPointFromInputs();
      });
      svg.appendChild(dot);
    });

    state.points.forEach((point) => {
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("class", "point-group");
      const site = siteByKey(point.siteKey);
      const solution = core.solutionForControls(site, point.controls);
      drawPie(group, point, solution);
      group.addEventListener("click", (event) => {
        event.stopPropagation();
        selectPoint(point.id);
        openDetail();
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
  }

  function selectPoint(id) {
    const point = state.points.find((item) => item.id === id);
    if (!point) {
      return;
    }
    state.selectedId = id;
    latInput.value = point.lat.toFixed(3);
    lonInput.value = point.lon.toFixed(3);
    siteSelect.value = point.siteKey;
    setControls(point.controls);
    drawMap();
    renderSummary();
    drawPlot();
  }

  function addPointFromInputs() {
    const lat = Number(latInput.value);
    const lon = Number(lonInput.value);
    const nearest = core.nearestSite(data, lat, lon);
    const site = siteByKey(siteSelect.value || nearest.site.siteKey);
    const point = core.pointRecord({
      id: `point-${state.nextId++}`,
      lat,
      lon,
      site,
      controls: currentControls(),
    });
    state.points.push(point);
    selectPoint(point.id);
  }

  function updateSelectedPoint() {
    const point = selectedPoint();
    if (!point) {
      renderSummary();
      return;
    }
    point.lat = Number(latInput.value);
    point.lon = Number(lonInput.value);
    point.siteKey = siteSelect.value;
    point.controls = currentControls();
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
      const parsed = JSON.parse(reader.result);
      state.points = core.validateDataset(parsed, data);
      state.nextId = state.points.length + 1;
      state.selectedId = state.points.length ? state.points[0].id : null;
      if (state.selectedId) {
        selectPoint(state.selectedId);
      } else {
        drawMap();
        renderSummary();
      }
    };
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
    const pad = { left: 56, right: 18, top: 24, bottom: 38 };
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
      ...slice.batteryChargeW,
      ...slice.batteryDischargeW,
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
    }

    line(slice.windW, yPower, colors.wind, 1.4);
    line(slice.solarW, yPower, colors.solar, 1.4);
    line(slice.waveW, yPower, colors.wave, 1.4);
    line(slice.loadW, yPower, "#8f2e28", 1.8);
    line(slice.batteryKWh, yBatt, "#6f3fb5", 1.8);

    ctx.fillStyle = "#1d2528";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("Dispatch [W]", 12, pad.top + 12);
    ctx.fillText("Battery [kWh]", 12, lowerTop + 12);
    ctx.fillText(`days ${x0.toFixed(0)}-${x1.toFixed(0)}`, pad.left, height - 12);
    ctx.fillStyle = colors.wind;
    ctx.fillText("Wind", width - 250, pad.top + 14);
    ctx.fillStyle = colors.solar;
    ctx.fillText("Solar", width - 202, pad.top + 14);
    ctx.fillStyle = colors.wave;
    ctx.fillText("Wave", width - 150, pad.top + 14);
    ctx.fillStyle = "#8f2e28";
    ctx.fillText("Load", width - 96, pad.top + 14);

    detailTitle.textContent = `${site.label} dispatch`;
    windowReadout.value = `days ${startDay}-${startDay + windowDays}`;
  }

  function initialize() {
    document.getElementById("data-status").textContent = `${data.sites.length} one-year LP profiles, baseline ${data.baselineVariant}`;
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
    const rect = svg.getBoundingClientRect();
    const point = core.unproject(
      (event.clientX - rect.left) * 1000 / rect.width,
      (event.clientY - rect.top) * 520 / rect.height,
    );
    const nearest = core.nearestSite(data, point.lat, point.lon);
    latInput.value = point.lat.toFixed(3);
    lonInput.value = point.lon.toFixed(3);
    siteSelect.value = nearest.site.siteKey;
    addPointFromInputs();
  });

  Object.values(controls).forEach((input) => {
    input.addEventListener("input", () => {
      updateControlOutputs();
      updateSelectedPoint();
    });
  });
  [latInput, lonInput, siteSelect].forEach((input) => input.addEventListener("input", updateSelectedPoint));
  document.getElementById("add-point").addEventListener("click", addPointFromInputs);
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
    }
  });
  [startDayInput, windowDaysInput].forEach((input) => input.addEventListener("input", drawPlot));

  initialize();
})();
