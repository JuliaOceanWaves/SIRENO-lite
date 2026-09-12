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
  // Frozen LP assumptions: include the platform supported by added batteries.
  const BATTERY_MASS_KG_PER_WH = 0.008 * 1.3;
  const BATTERY_COST_USD_PER_WH = 0.16 + 0.008 * 0.3 * 30;
  const CONTROL_DEFAULTS = { loadW: 100, resourceScale: 1, waveCapture: 1, outageDays: 0, costScale: 1 };
  const CONTROL_LIMITS = { loadW: [25, 500], resourceScale: [0.5, 2], waveCapture: [0.25, 4], outageDays: [0, 14], costScale: [0.25, 4] };

  function boundedNumber(value, name, minValue, maxValue) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < minValue || value > maxValue) {
      throw new Error(`${name} must be a number from ${minValue} to ${maxValue}.`);
    }
    return value;
  }

  function validatedControls(values = {}) {
    if (!values || typeof values !== "object" || Array.isArray(values)) {
      throw new Error("Scenario controls must be an object.");
    }
    Object.keys(values).forEach((key) => {
      if (!Object.prototype.hasOwnProperty.call(CONTROL_DEFAULTS, key)) {
        throw new Error(`Unknown scenario control: ${key}`);
      }
    });
    return Object.fromEntries(Object.entries(CONTROL_DEFAULTS).map(([key, fallback]) => [
      key, boundedNumber(values[key] === undefined ? fallback : values[key], key, ...CONTROL_LIMITS[key]),
    ]));
  }

  function validatedCoordinates(lat, lon) {
    return {
      lat: boundedNumber(lat, "Latitude", MAP_BOUNDS.latMin, MAP_BOUNDS.latMax),
      lon: boundedNumber(lon, "Longitude", MAP_BOUNDS.lonMin, MAP_BOUNDS.lonMax),
    };
  }

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
    const rad = Math.PI / 180;
    const dLat = (bLat - aLat) * rad;
    const dLon = (bLon - aLon) * rad;
    const haversine = Math.sin(dLat / 2) ** 2
      + Math.cos(aLat * rad) * Math.cos(bLat * rad) * Math.sin(dLon / 2) ** 2;
    return 2 * Math.asin(Math.sqrt(clamp(haversine, 0, 1))) / rad;
  }

  function nearestSite(data, lat, lon) {
    validatedCoordinates(lat, lon);
    if (!data || !Array.isArray(data.sites) || data.sites.length === 0) {
      throw new Error("No map sites are available.");
    }
    return data.sites.reduce((best, site) => {
      const distance = distanceDeg(lat, lon, site.latitude, site.longitude);
      return distance < best.distance ? { site, distance } : best;
    }, { site: data.sites[0], distance: Infinity });
  }

  function variantForCapture(site, capture) {
    const target = boundedNumber(capture, "WEC capture", 0.25, 4);
    if (!site || !Array.isArray(site.variants) || !site.variants.length) {
      throw new Error("No solved WEC cases are available for this profile.");
    }
    return site.variants.reduce((best, variant) => {
      const distance = Math.abs(Number(variant.wave_capture_fraction) - target);
      return distance < best.distance ? { variant, distance } : best;
    }, { variant: site.variants[0], distance: Infinity }).variant;
  }

  function solutionForControls(site, controls) {
    const { loadW, resourceScale, costScale, outageDays, waveCapture } = validatedControls(controls);
    const variant = variantForCapture(site, waveCapture);
    const loadScale = loadW / BASE_LOAD_W;
    const linearScale = loadScale / resourceScale;
    const reserveWh = loadW * outageDays * 24 / ONE_WAY_BATTERY_EFF;
    const windW = Number(variant.wind_scale_w) * linearScale;
    const solarW = Number(variant.solar_scale_w) * linearScale;
    const waveW = Number(variant.wave_scale_w) * linearScale;
    // Inverse generation sizing preserves the delivered generation history for a
    // uniform resource multiplier. Storage follows that history and the load,
    // so reducing battery capacity by resourceScale would break its energy balance.
    const baseBatteryWh = Number(variant.batt_scale_wh);
    const batteryWh = baseBatteryWh * loadScale + reserveWh;
    const renewableW = windW + solarW + waveW;
    const operatingCost = Number(variant.renewable_operating_cost_usd);
    const fixedBatteryCost = baseBatteryWh * BATTERY_COST_USD_PER_WH;
    const renewableFixedCost = Number(variant.total_cost_usd) - fixedBatteryCost - operatingCost;
    const baseCost = renewableFixedCost * linearScale + (fixedBatteryCost + operatingCost) * loadScale;
    const reserveCost = reserveWh * BATTERY_COST_USD_PER_WH;
    return {
      siteKey: site.siteKey,
      siteLabel: site.label,
      variant: variant.variant,
      variantLabel: variant.variant_label,
      waveCapture: Number(variant.wave_capture_fraction),
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
      massKg: (Number(variant.total_mass_kg) - baseBatteryWh * BATTERY_MASS_KG_PER_WH) * linearScale
        + (baseBatteryWh * loadScale + reserveWh) * BATTERY_MASS_KG_PER_WH,
      reserveWh,
      note: outageDays > 0
        ? "Rescaled archived design with a fully charged outage-energy allowance. Recharge, outage dispatch, and capacity limits have not been re-solved."
        : "Archived LP design rescaled for load and uniform resource availability. The scenario has not been re-optimized or checked against capacity limits.",
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
    boundedNumber(startDay, "Start day", 0, (ts.timeDays.length - 1) / 24);
    boundedNumber(windowDays, "Window days", 1 / 24, 365);
    const start = Math.max(0, Math.floor(Number(startDay) * 24));
    const stop = Math.min(ts.timeDays.length, start + Math.floor(Number(windowDays) * 24));
    const out = {};
    Object.keys(ts).forEach((key) => {
      out[key] = ts[key].slice(start, stop);
    });
    return out;
  }

  function pointRecord({ id, lat, lon, site, controls }) {
    const coordinates = validatedCoordinates(lat, lon);
    const normalizedControls = validatedControls(controls);
    normalizedControls.waveCapture = Number(variantForCapture(site, normalizedControls.waveCapture).wave_capture_fraction);
    return {
      id,
      ...coordinates,
      siteKey: site.siteKey,
      controls: normalizedControls,
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
    const byKey = new Map(data.sites.map((site) => [site.siteKey, site]));
    const ids = new Set();
    return dataset.points.map((point, index) => {
      if (!point || typeof point !== "object" || Array.isArray(point)) {
        throw new Error(`Saved point ${index + 1} must be an object.`);
      }
      if (!byKey.has(point.siteKey)) {
        throw new Error(`Saved point ${index + 1} references an unknown site: ${point.siteKey}`);
      }
      const id = point.id === undefined ? `loaded-${index + 1}` : point.id;
      if (typeof id !== "string" || !id.trim() || ids.has(id)) {
        throw new Error(`Saved point ${index + 1} must have a unique, nonempty text ID.`);
      }
      ids.add(id);
      return pointRecord({ ...point, id, site: byKey.get(point.siteKey) });
    });
  }

  function nextPointId(points) {
    const ids = new Set(points.map((point) => point.id));
    let index = 1;
    while (ids.has(`point-${index}`)) index += 1;
    return `point-${index}`;
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
      "Frozen one-year, 100 W LP library; results are screening estimates. Resource profiles are assigned to markers, not obtained at their coordinates. These archived cases use forced final battery recharge and are not the current cyclic mission study.",
      "",
    ];
    points.forEach((point, index) => {
      const site = byKey.get(point.siteKey);
      const solution = solutionForControls(site, point.controls);
      lines.push(
        `## ${index + 1}. ${site.label}`,
        "",
        `- Coordinates: ${formatNumber(point.lat, 3)}, ${formatNumber(point.lon, 3)}`,
        `- Profile coordinates: ${formatNumber(site.latitude, 3)}, ${formatNumber(site.longitude, 3)}`,
        `- Resource profile: ${site.resourceSource}`,
        `- Resource limitations: ${site.note}`,
        `- Load: ${formatNumber(solution.loadW, 1)} W`,
        `- Solved WEC capture case: ${formatNumber(solution.waveCapture, 2)}x`,
        `- Resource / cost multiplier: ${formatNumber(solution.resourceScale, 2)} / ${formatNumber(solution.costScale, 2)}`,
        `- Outage allowance: ${formatNumber(solution.outageDays, 1)} days; ${formatNumber(solution.reserveWh / 1000, 3)} kWh stored energy`,
        `- Selected non-fuel generation sizing: ${formatNumber(solution.renewableW / 1000, 3)} kW`,
        `- Wind / solar / wave: ${formatNumber(solution.windW / 1000, 3)} / ${formatNumber(solution.solarW / 1000, 3)} / ${formatNumber(solution.waveW / 1000, 3)} kW`,
        `- Battery: ${formatNumber(solution.batteryWh / 1000, 3)} kWh`,
        `- Cost: $${formatNumber(solution.costUsd, 2)} ($${formatNumber(solution.costPerW, 2)}/W)`,
        `- Mass: ${formatNumber(solution.massKg, 2)} kg, including platform support`,
        `- Note: ${solution.note}`,
        "",
      );
    });
    return lines.join("\n");
  }

  return {
    MAP_BOUNDS,
    BASE_LOAD_W,
    CONTROL_DEFAULTS,
    validatedControls,
    validatedCoordinates,
    nextPointId,
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
