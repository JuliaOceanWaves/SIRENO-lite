(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.SIRENOHybridCore = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const MAP_BOUNDS = { lonMin: -180, lonMax: 180, latMin: -60, latMax: 86 };
  const BASE_LOAD_W = 100;
  const ONE_WAY_BATTERY_EFF = Math.sqrt(0.9);

  function clamp(value, minValue, maxValue) {
    return Math.min(Math.max(Number(value), minValue), maxValue);
  }

  function project(lon, lat, width = 1000, height = 520) {
    const x = ((Number(lon) - MAP_BOUNDS.lonMin) / (MAP_BOUNDS.lonMax - MAP_BOUNDS.lonMin)) * width;
    const y = ((MAP_BOUNDS.latMax - Number(lat)) / (MAP_BOUNDS.latMax - MAP_BOUNDS.latMin)) * height;
    return { x, y };
  }

  function unproject(x, y, width = 1000, height = 520) {
    const lon = MAP_BOUNDS.lonMin + (Number(x) / width) * (MAP_BOUNDS.lonMax - MAP_BOUNDS.lonMin);
    const lat = MAP_BOUNDS.latMax - (Number(y) / height) * (MAP_BOUNDS.latMax - MAP_BOUNDS.latMin);
    return {
      lon: clamp(lon, MAP_BOUNDS.lonMin, MAP_BOUNDS.lonMax),
      lat: clamp(lat, MAP_BOUNDS.latMin, MAP_BOUNDS.latMax),
    };
  }

  function distanceDeg(aLat, aLon, bLat, bLon) {
    const dLat = Number(aLat) - Number(bLat);
    const meanLat = ((Number(aLat) + Number(bLat)) * Math.PI) / 360;
    const dLon = (Number(aLon) - Number(bLon)) * Math.cos(meanLat);
    return Math.sqrt(dLat * dLat + dLon * dLon);
  }

  function nearestSite(data, lat, lon) {
    if (!data || !Array.isArray(data.sites) || data.sites.length === 0) {
      throw new Error("No map sites are available.");
    }
    return data.sites.reduce((best, site) => {
      const distance = distanceDeg(lat, lon, site.latitude, site.longitude);
      return distance < best.distance ? { site, distance } : best;
    }, { site: data.sites[0], distance: Infinity });
  }

  function variantForCapture(site, capture) {
    const target = Number(capture);
    return site.variants.reduce((best, variant) => {
      const distance = Math.abs(Number(variant.wave_capture_fraction) - target);
      return distance < best.distance ? { variant, distance } : best;
    }, { variant: site.variants[0], distance: Infinity }).variant;
  }

  function solutionForControls(site, controls) {
    const loadW = clamp(controls.loadW ?? BASE_LOAD_W, 1, 100000);
    const resourceScale = clamp(controls.resourceScale ?? 1, 0.01, 100);
    const costScale = clamp(controls.costScale ?? 1, 0, 1000);
    const outageDays = clamp(controls.outageDays ?? 0, 0, 365);
    const variant = variantForCapture(site, controls.waveCapture ?? 1);
    const linearScale = loadW / BASE_LOAD_W / resourceScale;
    const reserveWh = loadW * outageDays * 24 / ONE_WAY_BATTERY_EFF;
    const windW = Number(variant.wind_scale_w) * linearScale;
    const solarW = Number(variant.solar_scale_w) * linearScale;
    const waveW = Number(variant.wave_scale_w) * linearScale;
    const batteryWh = Number(variant.batt_scale_wh) * linearScale + reserveWh;
    const renewableW = windW + solarW + waveW;
    const baseCost = Number(variant.total_cost_usd) * linearScale;
    const reserveCost = reserveWh * 0.16;
    return {
      siteKey: site.siteKey,
      siteLabel: site.label,
      variant: variant.variant,
      variantLabel: variant.variant_label,
      loadW,
      resourceScale,
      costScale,
      outageDays,
      windW,
      solarW,
      waveW,
      batteryWh,
      renewableW,
      costUsd: (baseCost + reserveCost) * costScale,
      costPerW: ((baseCost + reserveCost) * costScale) / loadW,
      massKg: Number(variant.total_mass_kg) * linearScale + reserveWh * 0.008,
      reserveWh,
      note: outageDays > 0
        ? "Outage reserve is a static battery-energy adder in this hosted prototype; it is not a re-solved LP case."
        : "Precomputed one-year LP result scaled linearly for load and resource multipliers.",
    };
  }

  function pieFractions(solution) {
    const total = Math.max(solution.windW + solution.solarW + solution.waveW, 0);
    if (total <= 0) {
      return [
        { key: "wind", label: "Wind", value: 0, fraction: 0 },
        { key: "solar", label: "Solar", value: 0, fraction: 0 },
        { key: "wave", label: "Wave", value: 0, fraction: 0 },
      ];
    }
    return [
      { key: "wind", label: "Wind", value: solution.windW, fraction: solution.windW / total },
      { key: "solar", label: "Solar", value: solution.solarW, fraction: solution.solarW / total },
      { key: "wave", label: "Wave", value: solution.waveW, fraction: solution.waveW / total },
    ];
  }

  function sliceTimeseries(site, startDay, windowDays) {
    const ts = site.timeseries;
    const start = Math.max(0, Math.floor(Number(startDay) * 24));
    const stop = Math.min(ts.timeDays.length, start + Math.floor(Number(windowDays) * 24));
    const out = {};
    Object.keys(ts).forEach((key) => {
      out[key] = ts[key].slice(start, stop);
    });
    return out;
  }

  function pointRecord({ id, lat, lon, site, controls }) {
    return {
      id,
      lat: Number(lat),
      lon: Number(lon),
      siteKey: site.siteKey,
      controls: { ...controls },
    };
  }

  function datasetForSave(points) {
    return {
      schemaVersion: 1,
      createdBy: "SIRENO-Lite Hybrid Map Tool",
      points: points.map((point) => ({
        id: point.id,
        lat: point.lat,
        lon: point.lon,
        siteKey: point.siteKey,
        controls: { ...point.controls },
      })),
    };
  }

  function validateDataset(dataset, data) {
    if (!dataset || dataset.schemaVersion !== 1 || !Array.isArray(dataset.points)) {
      throw new Error("Saved dataset is not a SIRENO-Lite hybrid-map dataset.");
    }
    const siteKeys = new Set(data.sites.map((site) => site.siteKey));
    return dataset.points.map((point, index) => {
      if (!siteKeys.has(point.siteKey)) {
        throw new Error(`Saved point ${index + 1} references an unknown site: ${point.siteKey}`);
      }
      return {
        id: point.id || `loaded-${index + 1}`,
        lat: Number(point.lat),
        lon: Number(point.lon),
        siteKey: point.siteKey,
        controls: { ...point.controls },
      };
    });
  }

  function formatNumber(value, digits = 2) {
    return Number(value).toLocaleString("en-US", {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    });
  }

  function reportMarkdown(points, data) {
    const byKey = new Map(data.sites.map((site) => [site.siteKey, site]));
    const lines = [
      "# SIRENO-Lite Hybrid Map Report",
      "",
      `Saved points: ${points.length}`,
      "",
    ];
    points.forEach((point, index) => {
      const site = byKey.get(point.siteKey);
      const solution = solutionForControls(site, point.controls);
      lines.push(
        `## ${index + 1}. ${site.label}`,
        "",
        `- Coordinates: ${formatNumber(point.lat, 3)}, ${formatNumber(point.lon, 3)}`,
        `- Resource profile: ${site.resourceSource}`,
        `- Load: ${formatNumber(solution.loadW, 1)} W`,
        `- Selected renewable capacity: ${formatNumber(solution.renewableW / 1000, 3)} kW`,
        `- Wind / solar / wave: ${formatNumber(solution.windW / 1000, 3)} / ${formatNumber(solution.solarW / 1000, 3)} / ${formatNumber(solution.waveW / 1000, 3)} kW`,
        `- Battery: ${formatNumber(solution.batteryWh / 1000, 3)} kWh`,
        `- Cost: $${formatNumber(solution.costUsd, 2)} ($${formatNumber(solution.costPerW, 2)}/W)`,
        `- Note: ${solution.note}`,
        "",
      );
    });
    return lines.join("\n");
  }

  return {
    MAP_BOUNDS,
    BASE_LOAD_W,
    clamp,
    project,
    unproject,
    nearestSite,
    variantForCapture,
    solutionForControls,
    pieFractions,
    sliceTimeseries,
    pointRecord,
    datasetForSave,
    validateDataset,
    reportMarkdown,
    formatNumber,
  };
});
