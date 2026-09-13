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
    fetch("viz/zones.json").then((r) => r.json()),
    fetch("viz/chicago.geojson").then((r) => r.json()),
    fetch("viz/water.json").then((r) => r.json()),
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
  $("lg-lo").textContent = fmt(10 ** LOG_MIN);
  $("lg-hi").textContent = fmt(10 ** LOG_MAX) + "+";
  showMetrics();

  initMap(geo, water);
  draw();

  // Open on the densest zone: it is where the story is clearest.
  select(KEYS.reduce((a, b) => DATA.zones[a].true > DATA.zones[b].true ? a : b));
})();

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
  const parts = [tile(`viz/chips/${key}.jpg`, "INPUT", "128² · 1.28 km", "chip-in")];
  DATA.layers.forEach((l, i) => parts.push(
    tile(`viz/layers/${key}_L${i + 1}.png`, `BLOCK ${i + 1}`,
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

  new ResizeObserver(() => { layout(); }).observe(stage);
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
  if (!r.width || !r.height) return;              // hidden: nothing to lay out
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

function draw() { drawScatter(); drawCalibration(); }

/* A canvas has to be re-rasterised on two separate triggers, and neither one
   covers the other.
   - Its box changing: a late webfont, an appearing scrollbar or the panel
     reflowing all resize it without firing `resize`, so the boxes are observed
     directly. Redrawing sets the backing store, never the CSS size, so this
     cannot feed itself.
   - The device pixel ratio changing: browser zoom, or the window moving to a
     display of a different density. The box is unchanged, so the observer
     stays silent and the chart would keep a stale, blurry raster.
   ResizeObserver callbacks are delivered inside the rendering loop, which a
   hidden tab does not run — which is harmless, because the redraw arrives the
   moment the tab is shown again. */
let t = 0;
const redraw = () => { clearTimeout(t); t = setTimeout(draw, 100); };
new ResizeObserver(redraw).observe($("scatter"));
new ResizeObserver(redraw).observe($("calib"));
addEventListener("resize", redraw);
