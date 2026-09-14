/* Sightline — the page.
 *
 * Everything shown here was produced by scripts/export_visuals.py from the
 * trained network: the feature maps are real activations on real Chicago zones,
 * not illustrations. The page only arranges them.
 *
 * No dependencies, no build step, no CDN. The map is hand-projected SVG and the
 * charts are plain canvas, which keeps the whole thing inspectable — and means
 * the page works offline from a folder.
 */
/* Every data file carries the same stamp as the code that reads it.
 *
 * Without it a returning visitor gets a fresh app.js against a cached
 * zones.json, and a shape mismatch surfaces as features quietly missing rather
 * than as an error — which is exactly how a slider built for sixty epochs came
 * back with seven steps. The version is read off this script's own URL so there
 * is one place to bump it, in the HTML, and no second copy to forget. */
const VERSION = new URL(document.currentScript.src, location.href)
  .searchParams.get("v") || "";
const url = (path) => (VERSION ? `${path}?v=${VERSION}` : path);

const COLD = [214, 227, 252], MID = [66, 133, 244], HOT = [217, 48, 37];

/* The one ramp, shared with the map, the charts and the exported feature maps,
   so that "bright" means the same thing in all three without a legend. */
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  const [a, b, k] = t < 0.5 ? [COLD, MID, t / 0.5] : [MID, HOT, (t - 0.5) / 0.5];
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * k)).join(",")})`;
}

const $ = (id) => document.getElementById(id);
const fmt = (n) => n >= 1000 ? Math.round(n).toLocaleString()
  : n >= 10 ? String(Math.round(n)) : n.toFixed(1);

let DATA, KEYS, RANK_T = {}, RANK_P = {}, selected = null;

/* Demand spans four orders of magnitude, so colour follows log10 — on a linear
   ramp every zone but the Loop would be the same shade. */
let LOG_MIN, LOG_MAX;
const norm = (v) => (Math.log10(Math.max(v, 1)) - LOG_MIN) / (LOG_MAX - LOG_MIN);
const shade = (v) => ramp(norm(v));

/* ------------------------------------------------------------------ boot */
(async function () {
  const [zj, geo, water] = await Promise.all([
    fetch(url("viz/zones.json")).then((r) => r.json()),
    fetch(url("viz/chicago.geojson")).then((r) => r.json()),
    fetch(url("viz/water.json")).then((r) => r.json()),
  ]);
  DATA = zj;
  KEYS = Object.keys(DATA.zones);

  // The colour scale is clamped to the 5th-95th percentile rather than the
  // extremes: one near-empty zone and one Loop would otherwise push every
  // ordinary neighbourhood into the middle of the ramp, which is where the
  // differences a viewer is looking for actually live. Values outside saturate.
  const sorted = KEYS.map((k) => DATA.zones[k].true).sort((a, b) => a - b);
  const pct = (p) => sorted[Math.min(sorted.length - 1,
    Math.round(p * (sorted.length - 1)))];
  LOG_MIN = Math.log10(Math.max(pct(0.05), 1));
  LOG_MAX = Math.log10(Math.max(pct(0.95), 10));

  KEYS.slice().sort((a, b) => DATA.zones[b].true - DATA.zones[a].true)
    .forEach((k, i) => { RANK_T[k] = i + 1; });
  KEYS.slice().sort((a, b) => DATA.zones[b].pred - DATA.zones[a].pred)
    .forEach((k, i) => { RANK_P[k] = i + 1; });

  $("p-count").textContent = KEYS.length + " zones";
  finding();
  $("lg-lo").textContent = fmt(10 ** LOG_MIN);
  $("lg-hi").textContent = fmt(10 ** LOG_MAX) + "+";
  showMetrics();
  showBaseline();
  showCeiling();

  initMap(geo, water);
  initEpochs();
  initSaliency();
  $("runlocal").addEventListener("click", runLocal);
  draw();

  // Open on the densest zone: it is where the story is clearest.
  select(KEYS.reduce((a, b) => DATA.zones[a].true > DATA.zones[b].true ? a : b));
})();

/* The most interesting thing the mosaics show is not in the pictures, it is in
   the sign of these four numbers — so they are read from the export rather than
   written into the prose, where they could quietly stop being true. */
function finding() {
  const r = DATA.layers.map((l) => l.response_r);
  if (r.some((v) => typeof v !== "number")) return;   // older export: say nothing
  const sig = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
  $("net-finding").innerHTML =
    `Then watch the response invert. Measure how hard each block's tiles fire against the
     demand the zone really has, and the first two blocks answer to <em>busy</em> —
     ${sig(r[0])} and ${sig(r[1])} — while the last runs the other way at ${sig(r[3])}.
     <b>The deep filters have learned to fire on emptiness</b>: vegetation, open ground,
     bare lots. Heavy demand is not what lights them up, it is what silences them, and
     the network reads the city off that silence. Hegewisch, the quietest zone in
     Chicago, ends up with the loudest final block on the page.`;
}

/* The objection the README used to carry without answering: that OSM gives you
   density directly. It is only worth putting on the page because it was scored
   the same way — same zones, same squares, same split — so the two rows can sit
   in one table without a footnote explaining why they cannot be compared. */
function showBaseline() {
  const b = DATA.baseline;
  if (!b) return;
  const n = DATA.metrics;
  const row = (label, m, win) => `<tr class="${win ? "win" : ""}">
    <td>${label}</td>
    <td class="${m.r2 < 0 ? "lose" : ""}">${m.r2 >= 0 ? "+" : "−"}${
    Math.abs(m.r2).toFixed(2)}</td>
    <td>${m.spearman.toFixed(2)}</td>
    <td>×${m.median_ratio_error.toFixed(2)}</td></tr>`;
  $("cmp-body").innerHTML =
    row(`OpenStreetMap road network — ${b.features.length} features`, b.chicago, false)
    + row("Satellite imagery — 49,152 pixels", n, true);
  $("cmp-note").innerHTML =
    `Ridge regression on metres of street, arterial share, junctions and segment count,
     the penalty chosen on New York alone. It is beaten on every column, and the ordering
     is where the gap is widest: <b>ρ 0.88 against ${b.chicago.spearman.toFixed(2)}</b>.
     Worth saying plainly, though: road density is weak here even in the city it was
     fitted on, reaching only ρ ${b.nyc_fit.spearman.toFixed(2)} there — and this is four
     road features, not all of OpenStreetMap. Building footprints and points of interest
     would likely do better, and were left out because fetching them cost thirty seconds
     a zone. What this shows is that <b>the most complete and universally available part
     of OSM does not carry the ordering, and the pixels do</b> — which is the part that
     matters for a city whose map is thin.`;
  $("baseline").hidden = false;
}

function showMetrics() {
  const m = DATA.metrics, n = DATA.nyc_metrics;
  $("m-nyc-r2").textContent = n.r2.toFixed(2);
  $("m-nyc-rho").textContent = n.spearman.toFixed(2);
  $("m-nyc-err").textContent = "×" + n.median_ratio_error.toFixed(2);
  $("m-chi-r2").textContent = m.r2.toFixed(2);
  $("m-chi-rho").textContent = m.spearman.toFixed(2);
  $("m-chi-err").textContent = "×" + m.median_ratio_error.toFixed(2);
}

/* --------------------------------------------------- the network strip */
const CHEVRON = '<div class="arrow"><svg width="20" height="20" viewBox="0 0 24 24"'
  + ' aria-hidden="true"><path d="M9 5l7 7-7 7" fill="none" stroke="currentColor"'
  + ' stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>';

function tile(src, label, sub, cls) {
  return `<figure class="stage ${cls || ""}">
    <div class="shot"><img src="${src}" alt="${label}"></div>
    <div class="st-l">${label}</div><div class="st-s">${sub}</div></figure>`;
}

function flow(key) {
  const z = DATA.zones[key];
  const parts = [tile(url(`viz/chips/${key}.png`), "INPUT", "128² · 1.28 km", "chip-in")];
  DATA.layers.forEach((l, i) => parts.push(
    tile(url(`viz/layers/${key}_L${i + 1}.png`), `BLOCK ${i + 1}`,
      `${l.channels} maps · ${l.shape[1]}²`)));
  $("flow").innerHTML = parts.join(CHEVRON) + CHEVRON
    + `<div class="outbox"><b>${fmt(z.pred)}</b><span>predicted trips/km²/day</span></div>`;
  $("net-sub").textContent =
    `${z.name} · 591k parameters · four convolutional blocks · trained on New York only`;
}

/* -------------------------------------------------------- the selection */
function select(key) {
  selected = key;
  const z = DATA.zones[key];
  const ratio = z.pred > z.true ? z.pred / z.true : z.true / z.pred;
  const dir = z.pred > z.true ? "over-called" : "under-called";

  $("d-name").textContent = z.name;
  $("d-meta").textContent = `${z.area_km2} km² · ${z.lat.toFixed(3)}, ${z.lon.toFixed(3)}`;
  $("d-true").textContent = fmt(z.true);
  $("d-pred").textContent = fmt(z.pred);
  $("d-rank-t").textContent = `#${RANK_T[key]} of ${KEYS.length}`;
  $("d-rank-p").innerHTML = `<span class="${
    Math.abs(RANK_T[key] - RANK_P[key]) <= 5 ? "good" : ""}">#${RANK_P[key]} of ${
    KEYS.length}</span>`;
  $("d-err").innerHTML = `<span class="off">×${ratio.toFixed(2)} ${dir}</span>`;

  flow(key);
  document.querySelectorAll("#map .zone").forEach((p) =>
    p.classList.toggle("sel", p.dataset.key === key));
  placePin();
  showSaliency();
  drawScatter();
}

/* ------------------------------------------------------------------ map */
const stage = document.querySelector(".mapstage");
const svg = $("map");
const view = { k: 1, x: 0, y: 0 };
let GEO, WATER, proj = null, vp = null, pin = null, W = 0, H = 0;

function initMap(geo, water) {
  GEO = geo.features.map((f) => ({
    key: f.properties.key,
    rings: (f.geometry.type === "Polygon" ? [f.geometry.coordinates]
      : f.geometry.coordinates).flat(),
  })).filter((f) => DATA.zones[f.key]);
  WATER = water;

  new ResizeObserver(schedule).observe(stage);
  layout();

  $("zin").onclick = () => zoom(1.6);
  $("zout").onclick = () => zoom(1 / 1.6);
  dragging();
}

/* Mercator y, converted back to degrees. The conversion is not cosmetic: the
   raw expression is in radians while longitude is in degrees, and sharing one
   scale between them collapses the city into a nine-pixel band. */
const merc = (lat) =>
  Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI / 180) / 2)) * (180 / Math.PI);

function layout() {
  const r = stage.getBoundingClientRect();
  if (!GEO || !r.width || !r.height) return;      // not loaded, or hidden
  W = Math.round(r.width); H = Math.round(r.height);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  for (const f of GEO) {
    for (const ring of f.rings) {
      for (const [lon, lat] of ring) {
        const y = merc(lat);
        if (lon < x0) x0 = lon; if (lon > x1) x1 = lon;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }

  // Leave the floating panel its own space, the way a map app does. The width
  // is read off the panel rather than repeated here, so the breakpoint that
  // drops it out of the overlay lives in one place — the stylesheet.
  const p = document.querySelector(".panel").getBoundingClientRect();
  const floats = getComputedStyle(document.querySelector(".panel")).position === "absolute";
  const inset = floats ? Math.min(p.width + 28, W * 0.45) : 0;
  const pad = 24;
  const bw = W - inset - pad * 2, bh = H - pad * 2;
  const s = Math.min(bw / (x1 - x0), bh / (y1 - y0));
  const ox = inset + pad + (bw - (x1 - x0) * s) / 2;
  const oy = pad + (bh - (y1 - y0) * s) / 2;
  proj = {
    x: (lon) => ox + (lon - x0) * s,
    y: (lat) => oy + (y1 - merc(lat)) * s,
  };

  const path = (rings) => rings.map((ring) =>
    "M" + ring.map(([lon, lat]) =>
      `${proj.x(lon).toFixed(1)},${proj.y(lat).toFixed(1)}`).join("L") + "Z").join("");

  const body = [];
  if (WATER.length) body.push(`<path class="lake" d="${path([WATER])}"></path>`);
  for (const f of GEO) {
    const z = DATA.zones[f.key];
    body.push(`<path class="zone${f.key === selected ? " sel" : ""}" d="${path(f.rings)}"`
      + ` fill="${shade(z.true)}" data-key="${f.key}"></path>`);
  }
  svg.innerHTML = `<g id="vp">${body.join("")}<g id="pin"></g></g>`;
  vp = $("vp"); pin = $("pin");

  svg.querySelectorAll(".zone").forEach((p) => {
    p.addEventListener("click", () => { if (!moved) select(p.dataset.key); });
    p.addEventListener("mouseenter", () => hover(p.dataset.key));
    p.addEventListener("mouseleave", () => { $("tip").hidden = true; });
  });
  svg.addEventListener("mousemove", moveTip);
  applyView();
  placePin();
}

function hover(key) {
  const z = DATA.zones[key];
  $("tip").innerHTML = `${z.name}<i>${fmt(z.true)} measured · ${fmt(z.pred)} predicted</i>`;
  $("tip").hidden = false;
}

function moveTip(e) {
  const r = stage.getBoundingClientRect();
  $("tip").style.left = (e.clientX - r.left) + "px";
  $("tip").style.top = (e.clientY - r.top) + "px";
}

/* The selected zone gets a map pin, counter-scaled so it keeps its size as the
   map zooms — the one thing on the map that is not geography. */
function placePin() {
  if (!pin || !selected || !proj) return;
  const z = DATA.zones[selected];
  const f = 1 / view.k;
  pin.innerHTML = `<g transform="translate(${proj.x(z.lon).toFixed(1)},`
    + `${proj.y(z.lat).toFixed(1)}) scale(${f.toFixed(3)}) translate(-12,-23)">`
    + '<path d="M12 2C8.1 2 5 5.1 5 9c0 5.2 7 14 7 14s7-8.8 7-14c0-3.9-3.1-7-7-7z"'
    + ' fill="#d93025" stroke="#fff" stroke-width="1.4"/>'
    + '<circle cx="12" cy="9" r="2.6" fill="#fff"/></g>';
}

function applyView() {
  if (!vp) return;
  vp.setAttribute("transform",
    `translate(${view.x.toFixed(1)} ${view.y.toFixed(1)}) scale(${view.k.toFixed(3)})`);
}

function zoom(f) {
  const k = Math.max(1, Math.min(8, view.k * f));
  const a = k / view.k;                     // about the centre of the stage
  view.x = W / 2 - (W / 2 - view.x) * a;
  view.y = H / 2 - (H / 2 - view.y) * a;
  view.k = k;
  clamp(); applyView(); placePin();
}

/* Keep the city from being dragged out of the frame entirely. */
function clamp() {
  const lo = 140;
  view.x = Math.max(lo - W * view.k, Math.min(W - lo, view.x));
  view.y = Math.max(lo - H * view.k, Math.min(H - lo, view.y));
}

let moved = false;
function dragging() {
  let on = false, sx = 0, sy = 0, ox = 0, oy = 0;
  svg.addEventListener("pointerdown", (e) => {
    on = true; moved = false; sx = e.clientX; sy = e.clientY; ox = view.x; oy = view.y;
    svg.setPointerCapture(e.pointerId); svg.classList.add("dragging");
  });
  svg.addEventListener("pointermove", (e) => {
    if (!on) return;
    const dx = e.clientX - sx, dy = e.clientY - sy;
    if (Math.abs(dx) + Math.abs(dy) > 4) moved = true;
    view.x = ox + dx; view.y = oy + dy;
    clamp(); applyView();
  });
  const end = (e) => {
    if (!on) return;
    on = false; svg.classList.remove("dragging");
    try { svg.releasePointerCapture(e.pointerId); } catch (_) { /* already gone */ }
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
}

/* --------------------------------------------------------------- charts */
function fit(cv) {
  const r = cv.getBoundingClientRect();
  if (!r.width || !r.height) return null;       // hidden: skip, redraw on resize
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, W: r.width, H: r.height };
}

function axisLabels(ctx, W, H, m, xlab, ylab) {
  ctx.fillStyle = "#858b92";
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(xlab, (m.l + W - m.r) / 2, H - 4);
  ctx.save();
  ctx.translate(10, (m.t + H - m.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylab, 0, 0); ctx.restore();
}

const SUP = ["⁰", "¹", "²", "³", "⁴"];

function drawScatter() {
  const cv = $("scatter"); if (!cv) return;
  const f = fit(cv); if (!f) return;
  const { ctx, W, H } = f, m = { l: 46, r: 14, t: 12, b: 34 };
  const lo = 0, hi = 4;                               // log10 trips/km²/day
  const X = (v) => m.l + (v - lo) / (hi - lo) * (W - m.l - m.r);
  const Y = (v) => H - m.b - (v - lo) / (hi - lo) * (H - m.t - m.b);

  ctx.strokeStyle = "#eceff2"; ctx.lineWidth = 1;
  ctx.fillStyle = "#858b92"; ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "right";
  for (let d = lo; d <= hi; d++) {
    ctx.beginPath(); ctx.moveTo(m.l, Y(d)); ctx.lineTo(W - m.r, Y(d)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(d), m.t); ctx.lineTo(X(d), H - m.b); ctx.stroke();
    ctx.fillText("10" + SUP[d], m.l - 6, Y(d) + 4);
  }
  ctx.strokeStyle = "#c3c9cf"; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(X(lo), Y(lo)); ctx.lineTo(X(hi), Y(hi)); ctx.stroke();
  ctx.setLineDash([]);

  for (const k of KEYS) {
    const z = DATA.zones[k], on = k === selected;
    ctx.beginPath();
    ctx.arc(X(Math.log10(Math.max(z.true, 1))), Y(Math.log10(Math.max(z.pred, 1))),
      on ? 7.5 : 4.5, 0, 6.284);
    ctx.fillStyle = shade(z.true);
    ctx.globalAlpha = on ? 1 : .82; ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = on ? 2.4 : 1; ctx.strokeStyle = on ? "#202124" : "#fff"; ctx.stroke();
  }
  axisLabels(ctx, W, H, m, "measured  trips/km²/day", "predicted");
}

/* A median hides its own tail, and the calibration curve flattening is only good
   news if it flattens somewhere good. Both facts are measured in calibrate.py
   and read from the export, so neither can go stale in the prose. */
function showCeiling() {
  const c = DATA.calibration_ceiling;
  if (!c || !$("ceiling")) return;
  const rows = DATA.calibration;
  const at = (k) => rows.find((r) => r.k === k);
  const pc = c.ratio_percentiles;
  $("ceiling").innerHTML =
    `Blunt about the tail, since a median hides it: uncalibrated, the typical zone is off
     by <b>×${pc["50"]}</b>, the 90th percentile by ×${pc["90"]}, the worst by
     ×${pc["100"]}, and only <b>${Math.round(c.within["x2"] * 100)}%</b> of zones land
     within a factor of two. As absolute numbers these are not usable.
     <br><br>Two things bound that. The error runs one way — the network over-calls
     Chicago almost everywhere, which is New York's scale showing through and exactly
     what one offset removes. And the distortion belongs to the transfer, not the model:
     regress predicted on measured <em>inside</em> New York and the slope is 0.97, no
     compression at all.
     <br><br>Fit the offset with all ${KEYS.length} answers in hand — cheating, not a
     result — and the best any correction reaches is <b>×${c.affine_ceiling.toFixed(2)}</b>.
     Three zones reach ×${at(3).median_ratio_error.toFixed(2)}: within four percent of
     knowing the whole city. Adding a slope is worse at every k, and the best slope with
     every answer visible is ${c.affine_ceiling_slope} — that is, none. One scalar is the
     right model here, not a shortcut.`;
  $("ceiling").hidden = false;
}

function drawCalibration() {
  const cv = $("calib"); if (!cv) return;
  const f = fit(cv); if (!f) return;
  const { ctx, W, H } = f, m = { l: 46, r: 14, t: 16, b: 34 };
  const rows = DATA.calibration;
  const maxE = Math.max(...rows.map((r) => r.median_ratio_error));
  const X = (i) => m.l + i / (rows.length - 1) * (W - m.l - m.r);
  const Y = (e) => H - m.b - (e - 1) / (maxE - 1) * (H - m.t - m.b);

  ctx.lineWidth = 1;
  for (let e = 1; e <= maxE + 1e-9; e += 0.5) {
    ctx.strokeStyle = "#eceff2";
    ctx.beginPath(); ctx.moveTo(m.l, Y(e)); ctx.lineTo(W - m.r, Y(e)); ctx.stroke();
    ctx.fillStyle = "#858b92"; ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right"; ctx.fillText("×" + e.toFixed(1), m.l - 6, Y(e) + 4);
  }

  ctx.beginPath();
  rows.forEach((r, i) => i ? ctx.lineTo(X(i), Y(r.median_ratio_error))
    : ctx.moveTo(X(i), Y(r.median_ratio_error)));
  ctx.strokeStyle = "#1a73e8"; ctx.lineWidth = 2.5; ctx.stroke();

  ctx.textAlign = "center";
  rows.forEach((r, i) => {
    const first = i === 1;                       // k = 1, the point that matters
    ctx.beginPath(); ctx.arc(X(i), Y(r.median_ratio_error), first ? 7 : 4.5, 0, 6.284);
    ctx.fillStyle = first ? "#fff" : "#1a73e8"; ctx.fill();
    ctx.lineWidth = first ? 3 : 1.5; ctx.strokeStyle = "#1a73e8"; ctx.stroke();
    ctx.fillStyle = "#858b92"; ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(String(r.k), X(i), H - m.b + 15);
  });

  ctx.fillStyle = "#202124"; ctx.font = "600 12px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("one zone → ×" + rows[1].median_ratio_error.toFixed(2),
    X(1) + 12, Y(rows[1].median_ratio_error) - 10);
  axisLabels(ctx, W, H, m, "zones where real demand is known", "median error factor");
}

/* Both of these can be reached by `pageshow` before the fetches have resolved. */
/* ------------------------------------- the network, computed right here */
const TILE = 4;                     // 4x4 = the same 16 channels the mosaics show

/* Paint one block's activation as the mosaic the exported PNGs use, from the
   channel order and the display anchors the export recorded — so the tile the
   browser computes is the same picture, not a similar one. */
function mosaic(cv, t, layer) {
  const n = TILE * TILE, side = TILE * (t.h + 1) - 1;
  cv.width = side; cv.height = side;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(side, side);
  img.data.fill(255);
  for (let i = 0; i < n; i++) {
    const ch = layer.order[i], top = layer.scale[i];
    const r = Math.floor(i / TILE), q = i % TILE;
    for (let y = 0; y < t.h; y++) {
      for (let x = 0; x < t.w; x++) {
        const v = t.data[ch * t.h * t.w + y * t.w + x];
        const u = top > 0 ? Math.pow(Math.min(Math.max(v / top, 0), 1), 0.7) : 0;
        const [cr, cg, cb] = rampRGB(u);
        const o = ((r * (t.h + 1) + y) * side + q * (t.w + 1) + x) * 4;
        img.data[o] = cr; img.data[o + 1] = cg; img.data[o + 2] = cb;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

function rampRGB(t) {
  t = Math.max(0, Math.min(1, t));
  const [a, b, k] = t < 0.5 ? [COLD, MID, t / 0.5] : [MID, HOT, (t - 0.5) / 0.5];
  return a.map((v, i) => Math.round(v + (b[i] - v) * k));
}

async function runLocal() {
  const btn = $("runlocal"), status = $("runstatus");
  if (!selected) return;
  btn.disabled = true;

  const cells = [`<figure class="stage chip-in"><div class="shot">
    <canvas id="live-in"></canvas></div><div class="st-l">INPUT</div>
    <div class="st-s">128² · 1.28 km</div></figure>`];
  DATA.layers.forEach((l, i) => cells.push(
    `<figure class="stage pending" id="live-s${i + 1}"><div class="shot">
      <canvas id="live-c${i + 1}"></canvas></div>
      <div class="st-l">BLOCK ${i + 1}</div>
      <div class="st-s">${l.channels} maps · ${l.shape[1]}²</div></figure>`));
  $("live").innerHTML = cells.join(CHEVRON) + CHEVRON
    + `<div class="outbox"><b id="live-out">…</b><span>computed here</span></div>`;
  $("live").hidden = false;

  status.textContent = "Fetching 2.4 MB of weights…";
  await Net.load();

  const px = Net.manifest().input.px;
  const { ctx, pixels } = await pixelsOf(url(`viz/chips/${selected}.png`), px);
  $("live-in").width = px; $("live-in").height = px;
  $("live-in").getContext("2d").drawImage(ctx.canvas, 0, 0);

  status.textContent = "Running the forward pass…";
  // Yield once so the strip paints before the main thread is taken for half a
  // second; otherwise the whole thing appears at the end and looks like a fake.
  await new Promise((r) => setTimeout(r, 30));

  const t0 = performance.now();
  const out = Net.run(pixels, px, (b, t) => {
    mosaic($(`live-c${b}`), t, DATA.layers[b - 1]);
    $(`live-s${b}`).classList.remove("pending");
  });
  const ms = Math.round(performance.now() - t0);

  $("live-out").textContent = fmt(out.density);
  const shipped = DATA.zones[selected].pred;
  const gap = Math.abs(out.density - shipped) / shipped;
  status.textContent = `Done in ${ms} ms on your machine.`;
  $("run-x").hidden = false;
  $("run-x").innerHTML =
    `Those feature maps were not fetched — your browser produced them, one block at a
     time, from 590,497 weights and the 128×128 pixels above. The estimate it reached is
     <b>${fmt(out.density)}</b> against the <b>${fmt(shipped)}</b> computed in PyTorch:
     ${gap < 0.005
      ? "the same number, to the precision printed here"
      : `a gap of ${(gap * 100).toFixed(1)}%`}. BatchNorm was folded into the convolutions
     at export, so the browser only ever does convolution, ReLU, an average and two
     matrix multiplies — which is all this model has ever been.`;
  btn.disabled = false;
}

/* Read an image into raw pixels for the network.

   Deliberately not `new Image()` + `await img.decode()`: that route goes through
   the rendering pipeline, which a hidden or throttled tab may never run, and the
   promise then never settles — the button sits on "running…" forever with no
   error anywhere. Reproduced exactly that way. createImageBitmap decodes off the
   rendering path and settles whatever the tab is doing. */
async function pixelsOf(url, px) {
  const blob = await fetch(url).then((r) => r.blob());
  const bmp = await createImageBitmap(blob);
  const cv = document.createElement("canvas");
  cv.width = cv.height = px;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0, px, px);
  if (bmp.close) bmp.close();
  return { ctx, pixels: ctx.getImageData(0, 0, px, px).data };
}

/* ------------------------------- occlusion: what the network actually used */
let SAL = null;

async function initSaliency() {
  try {
    SAL = await fetch(url("viz/saliency/index.json")).then((r) => r.json());
  } catch (_) {
    $("salcard").hidden = true;                 // export not run: say nothing
    return;
  }
  $("sal-map").addEventListener("click", probeAt);
  showSaliency();
}

function showSaliency() {
  if (!SAL || !selected) return;
  $("sal-chip").src = url(`viz/chips/${selected}.png`);
  $("sal-map").src = url(`viz/saliency/${selected}.png`);
  $("sal-sub").textContent =
    `${DATA.zones[selected].name} · ${SAL.probes_per_zone} occlusions · `
    + `${SAL.patch_px * SAL.metres_per_px} m patch · full scale ×`
    + `${(10 ** SAL.scale_log10).toFixed(2)}, shared across all ${KEYS.length} zones`;
  const cv = $("sal-probe");
  cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
  $("sal-probe-out").hidden = true;
}

/* Clicking runs the real thing rather than reading the stored map: one forward
   pass with a grey square where the pointer went. It is the same arithmetic the
   export did 196 times, which is the point — the reader can pick the square. */
async function probeAt(e) {
  const img = $("sal-map"), r = img.getBoundingClientRect();
  const out = $("sal-probe-out");
  out.hidden = false;
  out.innerHTML = "Loading the weights and running the network…";

  await Net.load();
  const P = SAL.patch_px, N = Net.manifest().input.px;
  const cx = Math.round(((e.clientX - r.left) / r.width) * N);
  const cy = Math.round(((e.clientY - r.top) / r.height) * N);
  const x0 = Math.max(0, Math.min(N - P, cx - P / 2));
  const y0 = Math.max(0, Math.min(N - P, cy - P / 2));

  const { ctx, pixels } = await pixelsOf(url(`viz/chips/${selected}.png`), N);
  const before = Net.run(pixels, N).density;

  ctx.fillStyle = "rgb(128,128,128)";
  ctx.fillRect(x0, y0, P, P);
  const after = Net.run(ctx.getImageData(0, 0, N, N).data, N).density;

  // Mark the square on the overlay so the number has a place attached to it.
  const ov = $("sal-probe");
  ov.width = ov.height = N;
  const oc = ov.getContext("2d");
  oc.clearRect(0, 0, N, N);
  oc.strokeStyle = "#202124"; oc.lineWidth = 2;
  oc.strokeRect(x0 + 1, y0 + 1, P - 2, P - 2);
  oc.strokeStyle = "#fff"; oc.lineWidth = 1;
  oc.strokeRect(x0 + 2, y0 + 2, P - 4, P - 4);

  const drop = (before - after) / before;
  out.innerHTML =
    `Covering that ${P * SAL.metres_per_px} m square took the estimate from
     <b>${fmt(before)}</b> to <b>${fmt(after)}</b> — ${
    Math.abs(drop) < 0.005 ? "no change at all"
      : drop > 0 ? `<b>${(drop * 100).toFixed(1)}% lower</b>, so the network was reading
           demand there` : `<b>${(-drop * 100).toFixed(1)}% higher</b>, so that patch was
           evidence against`}. Computed here, just now, by your browser.`;
}

/* ------------------------------------------------- watching it learn */
let EPOCHS = [], epochIdx = 0;

function initEpochs() {
  EPOCHS = Object.keys(DATA.snapshots || {}).map(Number).sort((a, b) => a - b);
  const slider = $("epoch");
  if (!EPOCHS.length || !slider) { return; }
  slider.max = String(EPOCHS.length - 1);
  slider.value = String(EPOCHS.length - 1);
  epochIdx = EPOCHS.length - 1;
  slider.addEventListener("input", () => {
    epochIdx = Number(slider.value);
    drawSnaps(); drawCurve();
  });
  drawSnaps(); drawCurve();
}

/* One square frame for both snapshot charts, on one shared scale: two panels
   that auto-scaled separately would hide the very thing being compared. */
function snapRange() {
  let lo = 1e9, hi = -1e9;
  for (const k of Object.keys(DATA.snapshots)) {
    for (const f of ["val_true", "val_pred", "chi_true", "chi_pred"]) {
      for (const v of DATA.snapshots[k][f]) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  }
  return [Math.floor(Math.min(lo, 0)), Math.ceil(hi)];
}

function drawSnap(id, tk, pk) {
  const cv = $(id); if (!cv) return;
  const box = fit(cv); if (!box) return;
  const { ctx, W, H } = box, m = { l: 46, r: 14, t: 12, b: 38 };
  const snap = DATA.snapshots[String(EPOCHS[epochIdx])];
  const [lo, hi] = snapRange();
  const side = Math.min(W - m.l - m.r, H - m.t - m.b);
  const ox = m.l, oy = m.t + (H - m.t - m.b - side) / 2;
  const X = (v) => ox + (v - lo) / (hi - lo) * side;
  const Y = (v) => oy + side - (v - lo) / (hi - lo) * side;

  ctx.strokeStyle = "#eceff2"; ctx.lineWidth = 1;
  ctx.fillStyle = "#858b92"; ctx.font = "11px system-ui, sans-serif";
  for (let d = Math.ceil(lo); d <= hi; d++) {
    ctx.beginPath(); ctx.moveTo(ox, Y(d)); ctx.lineTo(ox + side, Y(d)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(X(d), oy); ctx.lineTo(X(d), oy + side); ctx.stroke();
    if (d >= 0 && d <= 4) {
      ctx.textAlign = "right"; ctx.fillText("10" + SUP[d], ox - 6, Y(d) + 4);
      ctx.textAlign = "center"; ctx.fillText("10" + SUP[d], X(d), oy + side + 16);
    }
  }
  ctx.strokeStyle = "#c3c9cf"; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(X(lo), Y(lo)); ctx.lineTo(X(hi), Y(hi)); ctx.stroke();
  ctx.setLineDash([]);

  const t = snap[tk], p = snap[pk];
  for (let i = 0; i < t.length; i++) {
    // A prediction can start far outside the frame; clamping keeps it visible
    // and on the edge, rather than silently dropping the worst points.
    ctx.beginPath();
    ctx.arc(X(t[i]), Y(Math.min(Math.max(p[i], lo), hi)), 4.4, 0, 6.284);
    ctx.fillStyle = ramp((t[i] - lo) / (hi - lo));
    ctx.globalAlpha = .88; ctx.fill(); ctx.globalAlpha = 1;
    ctx.lineWidth = 1; ctx.strokeStyle = "#fff"; ctx.stroke();
  }
  ctx.fillStyle = "#858b92"; ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("measured  trips/km²/day", ox + side / 2, H - 4);
}

function drawSnaps() {
  if (!EPOCHS.length) return;
  const ep = EPOCHS[epochIdx];
  const row = (DATA.history || []).find((h) => h.epoch === ep);
  $("epoch-n").textContent = ep;
  $("epoch-note").textContent = row
    ? `Chicago at this epoch: rank ρ ${row.chi_rho.toFixed(2)}, R² ${
      row.chi_r2 >= 0 ? "+" : "−"}${Math.abs(row.chi_r2).toFixed(2)}, typical level error ×${
      row.chi_err < 100 ? row.chi_err.toFixed(2) : Math.round(row.chi_err)}`
    : "";
  drawSnap("snap-ny", "val_true", "val_pred");
  drawSnap("snap-chi", "chi_true", "chi_pred");
}

function drawCurve() {
  const cv = $("curve"); if (!cv) return;
  const box = fit(cv); if (!box) return;
  const { ctx, W, H } = box, m = { l: 46, r: 46, t: 16, b: 34 };
  const h = DATA.history || []; if (!h.length) return;
  const X = (e) => m.l + (e - 1) / (h.length - 1) * (W - m.l - m.r);
  // Left axis carries rho on its own narrow range; R2 is clamped to [-1,1] on
  // the right. Sharing one axis would flatten rho into a straight line at the
  // top and make the comparison say nothing.
  const YR = (v) => H - m.b - (Math.min(Math.max(v, 0.7), 0.95) - 0.7) / 0.25 * (H - m.t - m.b);
  const YQ = (v) => H - m.b - (Math.min(Math.max(v, -1), 1) + 1) / 2 * (H - m.t - m.b);

  ctx.strokeStyle = "#eceff2"; ctx.lineWidth = 1;
  ctx.font = "11px system-ui, sans-serif";
  for (const v of [0.7, 0.8, 0.9]) {
    ctx.beginPath(); ctx.moveTo(m.l, YR(v)); ctx.lineTo(W - m.r, YR(v)); ctx.stroke();
    ctx.fillStyle = "#1a73e8"; ctx.textAlign = "right";
    ctx.fillText("ρ " + v.toFixed(1), m.l - 6, YR(v) + 4);
  }
  for (const v of [-1, 0, 1]) {
    ctx.fillStyle = "#d93025"; ctx.textAlign = "left";
    ctx.fillText("R² " + (v > 0 ? "+" : "") + v, W - m.r + 6, YQ(v) + 4);
  }

  const line = (key, Y, col, width) => {
    ctx.beginPath();
    h.forEach((e, i) => i ? ctx.lineTo(X(e.epoch), Y(e[key])) : ctx.moveTo(X(e.epoch), Y(e[key])));
    ctx.strokeStyle = col; ctx.lineWidth = width; ctx.stroke();
  };
  line("chi_r2", YQ, "#d93025", 2);
  line("chi_rho", YR, "#1a73e8", 2.4);

  const ep = EPOCHS[epochIdx];
  if (ep) {
    ctx.strokeStyle = "#858b92"; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(X(ep), m.t); ctx.lineTo(X(ep), H - m.b); ctx.stroke();
  }
  const best = (DATA.training || {}).best_epoch;
  if (best) {
    ctx.strokeStyle = "#c3c9cf"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(best), m.t); ctx.lineTo(X(best), H - m.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#858b92"; ctx.textAlign = "center"; ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("shipped", X(best), m.t + 10);
  }
  ctx.fillStyle = "#858b92"; ctx.textAlign = "center";
  for (const e of [1, 10, 20, 30, 40, 50, h.length]) ctx.fillText(String(e), X(e), H - m.b + 16);
  ctx.fillText("epoch", (m.l + W - m.r) / 2, H - 4);
}

function draw() {
  if (!DATA) return;
  drawScatter(); drawCalibration();
  if (EPOCHS.length) { drawSnaps(); drawCurve(); }
}

/* The map and both charts are rasterised against the size of their box, so they
   have to be rebuilt whenever that size — or the pixel density behind it — is no
   longer what they were drawn for. Three separate triggers are needed, because
   none of them covers the others:
   - The box changes. A late webfont, an appearing scrollbar or the panel
     dropping out of the overlay all resize things without firing `resize`, so
     the boxes are observed directly. Rebuilding only ever sets backing-store
     sizes, never CSS ones, so this cannot feed itself.
   - The device pixel ratio changes. Browser zoom, or the window moving to a
     display of a different density: the box is identical, the observer stays
     silent, and the raster would be left stale and blurry.
   - The tab is shown for the first time. This is the one that actually bites: a
     page loaded in a background tab does not run the rendering loop, and
     ResizeObserver callbacks are delivered from inside that loop — so a map
     armed while the stage had no size would never be told it now has one, and
     would stay permanently empty. `visibilitychange` is the only signal that
     arrives, and it is why this is not simply an observer. */
function refresh() { layout(); draw(); }

let scheduled = 0;
function schedule() { clearTimeout(scheduled); scheduled = setTimeout(refresh, 100); }

new ResizeObserver(schedule).observe($("scatter"));
new ResizeObserver(schedule).observe($("calib"));
addEventListener("resize", schedule);
addEventListener("pageshow", refresh);
// On `document`, which is where visibilitychange is dispatched, and
// unconditional: rebuilding while still hidden costs one early return, whereas
// trusting `document.hidden` costs an empty map wherever it reads false late.
document.addEventListener("visibilitychange", refresh);
