"""One self-contained HTML page showing the three clustering views on the supply graph.

    from src.kg import viz; viz.write(kg, "clusters_map.html")

Layout is computed here (networkx spring layout, fixed seed) so the page needs no library and opens anywhere,
including Colab (IPython.display.HTML) and as a claude.ai artifact.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import networkx as nx

import math

from .cluster import ETO_TIER, LANES, TO_ETO_ROLE, Clusters
from .eto import ETO
from .load import KG


def data(kg: KG, threshold: float = 0.5, resolution: float = 1.0) -> dict:
    C = Clusters(kg)
    G = C.graph()
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    pos = nx.spring_layout(G.subgraph(comps[0]), weight="weight", seed=7, iterations=300, k=0.11)
    for j, comp in enumerate(comps[1:]):  # small components in a row under the giant one
        sub = nx.spring_layout(G.subgraph(comp), seed=7, scale=0.04)
        for n, (x, y) in sub.items():
            pos[n] = (-0.9 + 0.2 * j + x, 1.15 + y)
    prod, eco = C.product(threshold), C.ecosystem(resolution)
    roles, tiers = C.role_by_products(), C.tiers()
    eto = ETO().match(kg)
    deg = Counter(x for e in C.supply for x in e)
    eco_rank = {e: i for i, (e, _) in enumerate(Counter(eco.values()).most_common())}
    lanes = C.lanes()
    boxes = chain_boxes()
    chain = chain_positions(lanes, boxes, eco_rank, eco, deg)
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    W = max(b["x"] + b["w"] for b in boxes.values())
    H = max(b["y"] + b["h"] for b in boxes.values())
    norm = lambda x, y: (W * (x - min(xs)) / (max(xs) - min(xs)), H * (y - min(ys)) / (max(ys) - min(ys)))
    nodes = []
    for n in G.nodes:
        t = tiers.get(n, {})
        g = eto.get(n)
        graded = None
        if g:
            pred_role = TO_ETO_ROLE.get(roles.get(n)) if n in roles else None
            pred_tier = t.get("tier")
            graded = {"eto_role": g["eto_role"], "eto_stage": g["eto_stage"],
                      "role_ok": None if pred_role is None and n not in roles else pred_role == g["eto_role"],
                      "tier_ok": None if pred_tier is None else pred_tier == ETO_TIER[g["eto_role"]]}
        fx, fy = norm(*pos[n])
        cx, cy, cr = chain.get(n, (None, None, None))
        nodes.append({"id": n, "name": kg.name(n), "x": round(fx, 4), "y": round(fy, 4),
                      "cx": cx, "cy": cy, "cr": cr, "lane": lanes.get(n, {}).get("lane"),
                      "placed_by": lanes.get(n, {}).get("placed_by"),
                      "deg": deg[n], "eco": eco_rank.get(eco.get(n)), "prod": prod.get(n), "role": roles.get(n),
                      "tier": t.get("tier"), "tier_from": t.get("tier_from"), "makes": sorted(C.P.makes(n)),
                      "eto": graded})
    edges = [[a, b] for a, b in sorted(C.supply) if a in G and b in G]
    return {"nodes": nodes, "edges": edges, "n_eco": len(set(eco.values())), "boxes": boxes,
            "flows": C.lane_flows(lanes), "lane_names": LANES,
            "modularity": round(nx.community.modularity(G, [{c for c in eco if eco[c] == i} for i in set(eco.values())]), 2)}


def chain_boxes() -> dict:
    """Box per lane, arranged like the CSIS value-chain figure: the main chain on top, its inputs underneath."""
    W, H, GX, GY = 1.0, 1.0, 0.32, 0.42
    col = lambda i: i * (W + GX)
    top = {"chips": 0, "fabrication": 1, "assembly_test": 2, "systems_cloud": 3}
    boxes = {k: {"x": col(i), "y": 0.0, "w": W, "h": H} for k, i in top.items()}
    yb = H + GY
    boxes["design_sw_ip"] = {"x": col(0), "y": yb, "w": W, "h": 0.7}
    boxes["materials"] = {"x": col(1), "y": yb, "w": W / 2 - 0.03, "h": 0.7}
    boxes["equipment"] = {"x": col(1) + W / 2 + 0.03, "y": yb, "w": W / 2 - 0.03, "h": 0.7}
    boxes["unknown_inputs"] = {"x": col(2), "y": yb, "w": W, "h": 0.7}
    boxes["components_ems"] = {"x": col(3), "y": yb, "w": W, "h": 0.7}
    short = {"materials": "Materials", "equipment": "Equipment", "unknown_inputs": "Inputs, products unknown"}
    for k, b in boxes.items():
        b["label"] = short.get(k, LANES[k])
    return boxes


def chain_positions(lanes, boxes, eco_rank, eco, deg) -> dict:
    """Grid inside each box, members sorted by ecosystem then degree, so communities show as colour runs."""
    out = {}
    by = {}
    for c, v in lanes.items():
        by.setdefault(v["lane"], []).append(c)
    for lane, cs in by.items():
        b = boxes[lane]
        cs.sort(key=lambda c: (eco_rank.get(eco.get(c), 99), -deg[c]))
        pad, head = 0.05, 0.13
        w, h = b["w"] - 2 * pad, b["h"] - head - pad
        ncol = max(1, math.ceil(math.sqrt(len(cs) * w / h)))
        nrow = math.ceil(len(cs) / ncol)
        cw, ch = w / ncol, h / max(nrow, 1)
        for i, c in enumerate(cs):
            r_, k = divmod(i, ncol)
            r = min(min(cw, ch) * 0.42, 0.032) * min(1.0, 0.45 + 0.18 * math.sqrt(deg[c]))
            out[c] = (round(b["x"] + pad + (k + 0.5) * cw, 4), round(b["y"] + head + (r_ + 0.5) * ch, 4), round(r, 4))
    return out


def write(kg: KG, path, **kw) -> Path:
    html = TEMPLATE.replace("/*DATA*/null", json.dumps(data(kg, **kw), ensure_ascii=False))
    p = Path(path)
    p.write_text(html, encoding="utf-8")
    return p


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supply Graph Clusters</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  /* layout: graph canvas left, controls + detail column right; stacks under 900px */
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1d2433; --muted: #5e687a; --line: #d9dee6; --accent: #2457c5;
  --ok: #1f8a4c; --bad: #c43b3b; --edge: rgba(29,36,51,.13); --edge-hi: rgba(36,87,197,.75); --box: rgba(36,87,197,.05);
  --font: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12151c; --panel: #1a1f29; --ink: #e6e9ef; --muted: #9aa3b2; --line: #2c3442; --accent: #7aa2ff;
  --ok: #4cc283; --bad: #ff7a7a; --edge: rgba(230,233,239,.12); --edge-hi: rgba(122,162,255,.85); --box: rgba(122,162,255,.06); color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #12151c; --panel: #1a1f29; --ink: #e6e9ef; --muted: #9aa3b2; --line: #2c3442; --accent: #7aa2ff;
  --ok: #4cc283; --bad: #ff7a7a; --edge: rgba(230,233,239,.12); --edge-hi: rgba(122,162,255,.85); --box: rgba(122,162,255,.06); color-scheme: dark; }
html, body { height: 100%; margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 var(--font); }
.wrap { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 16px; padding: 16px; height: 100%; box-sizing: border-box; }
@media (max-width: 900px) { .wrap { grid-template-columns: 1fr; height: auto; } #graph { height: 70vh; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
#graph { position: relative; overflow: hidden; min-height: 420px; }
svg { width: 100%; height: 100%; display: block; cursor: grab; }
svg:active { cursor: grabbing; }
.side { display: flex; flex-direction: column; gap: 12px; overflow: auto; }
.side .card { padding: 14px; }
h1 { font-size: 17px; margin: 0 0 4px; font-weight: 600; text-wrap: balance; }
h2 { font-size: 12px; margin: 0 0 8px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
p { margin: 0; color: var(--muted); }
.seg { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; width: 100%; }
.seg button { font: 500 12px var(--font); padding: 7px 2px; border: 1px solid var(--line); background: transparent; color: var(--ink);
  border-radius: 6px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
input[type=search] { flex: 1; min-width: 120px; font: 13px var(--font); padding: 7px 9px; border: 1px solid var(--line); border-radius: 6px;
  background: var(--bg); color: var(--ink); }
label { font-size: 13px; color: var(--muted); display: flex; gap: 6px; align-items: center; }
.legend { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px; font-size: 12px; }
.legend span { display: flex; gap: 6px; align-items: center; min-width: 0; }
.legend i { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.legend i.hollow { background: transparent !important; border: 2px solid; width: 6px; height: 6px; }
.pill { display: inline-block; font: 500 11px var(--font); padding: 1px 7px; border-radius: 99px; border: 1px solid var(--line); margin: 2px 4px 0 0; }
.ok { color: var(--ok); border-color: var(--ok); } .bad { color: var(--bad); border-color: var(--bad); }
#detail .name { font-weight: 600; font-size: 15px; }
#detail ul { margin: 6px 0 0; padding-left: 18px; color: var(--muted); font-size: 12.5px; }
.pairs button { display: block; width: 100%; text-align: left; font: 12.5px var(--font); background: none; border: 0; border-bottom: 1px solid var(--line);
  padding: 5px 0; color: var(--ink); cursor: pointer; }
.pairs button:hover { color: var(--accent); }
#tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px;
  font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.12); display: none; max-width: 260px; }
.hint { position: absolute; left: 12px; bottom: 10px; font-size: 12px; color: var(--muted); }
.note { font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<div class="wrap">
  <div id="graph" class="card"><svg id="svg" role="img" aria-label="Supply graph"></svg><div id="tip"></div>
    <div class="hint">Drag to pan · scroll to zoom · click a company or a stage</div></div>
  <div class="side">
    <div class="card">
      <h1>Supply graph clusters</h1>
      <p id="sub"></p>
      <div class="row"><div class="seg" role="group" aria-label="View">
        <button id="v-chain" aria-pressed="true">Value chain</button><button id="v-eco" aria-pressed="false">Ecosystem</button>
        <button id="v-role" aria-pressed="false">Role</button><button id="v-tier" aria-pressed="false">Tier</button>
        <button id="v-eto" aria-pressed="false">vs ETO</button></div></div>
      <div class="row"><input id="q" type="search" placeholder="Find a company"><label><input id="leaf" type="checkbox"> Hide 1-link companies</label></div>
      <p class="note" id="viewnote"></p>
    </div>
    <div class="card"><h2 id="lg-title">Legend</h2><div id="legend" class="legend"></div></div>
    <div class="card" id="detail"><h2>Company</h2><p>Click a node to see its products, cluster labels and trading partners.</p></div>
    <div class="card"><h2>Same products, different ecosystem</h2><div id="pairs" class="pairs"></div></div>
  </div>
</div>
<script>
const D = /*DATA*/null;
const PAL = ["#2457c5","#d9822b","#2f9e77","#c43b7a","#7a5af5","#b59a00","#1e9fbf","#8c5a3c","#e0566b","#4f8f2f","#9b59b6","#3c6e71","#a3a3a3","#6b7a99","#c9a227"];
const ROLE = {materials:"#8c5a3c", equipment:"#d9822b", design_tools_ip:"#7a5af5", chips:"#2457c5", manufacturing:"#1e9fbf",
  components:"#b59a00", systems:"#2f9e77", cloud_software:"#c43b7a", other:"#a3a3a3"};
const TIER = {upstream:"#d9822b", chips:"#2457c5", downstream:"#2f9e77"};
const GREY = "#9aa3b2";
const NOTES = {
  chain: "Companies in the value-chain stage their products point to (CSIS figure, extended to AI systems). Ringed: placed by trading partners, products unknown. Colour: ecosystem. Click a stage for its supply links in and out.",
  eco: "Louvain communities of the supply graph: who trades with whom. These are the recommender's peer sets.",
  role: "Role from what the company makes, weighted by evidence. Grey: no product profile in the KG.",
  tier: "Upstream / chips / downstream. Ringed: placed one step above suppliers or below customers.",
  eto: "Graded against the CSET ETO Supply Chain Explorer. Only 38 companies have an ETO label."};
const byId = Object.fromEntries(D.nodes.map(n => [n.id, n]));
const nbr = {}; D.nodes.forEach(n => nbr[n.id] = {out: [], in: []});
D.edges.forEach(([a, b]) => { nbr[a].out.push(b); nbr[b].in.push(a); });
let view = "chain", sel = null, hideLeaf = false, q = "";
const svg = document.getElementById("svg"), NS = "http://www.w3.org/2000/svg";
const el = (t, a = {}, parent) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (parent) parent.appendChild(e); return e; };
const BX = Object.values(D.boxes), W = Math.max(...BX.map(b => b.x + b.w)), H = Math.max(...BX.map(b => b.y + b.h));
const pad = .12; let vb = [-pad, -pad - .45, W + 2 * pad, H + 2 * pad + .45];
const S = W / 900;
const defs = el("defs", {}, svg);
const mk = el("marker", {id: "arr", viewBox: "0 0 10 10", refX: 9, refY: 5, markerUnits: "userSpaceOnUse", markerWidth: S * 12, markerHeight: S * 12, orient: "auto"}, defs);
const arrowHead = el("path", {d: "M0,0 L10,5 L0,10 z"}, mk);
const root = el("g", {}, svg), boxG = el("g", {}, root), flowG = el("g", {}, root), eG = el("g", {}, root), nG = el("g", {}, root), lG = el("g", {}, root);
// stage boxes
const boxEls = Object.entries(D.boxes).map(([k, b]) => { const r = el("rect", {x: b.x, y: b.y, width: b.w, height: b.h, rx: .025}, boxG);
  r.style.cursor = "pointer"; r.addEventListener("click", ev => { ev.stopPropagation(); selectBox(selBox === k ? null : k); });
  const t = el("text", {x: b.x + .04, y: b.y + .08, "font-size": S * 13, "font-weight": 600, "font-family": "IBM Plex Sans, sans-serif"}, boxG);
  const cnt = D.nodes.filter(n => n.lane === k).length; t.textContent = `${b.label} · ${cnt}`; t.style.pointerEvents = "none"; return {r, t, k}; });
// stage-to-stage flows
const centre = k => ({x: D.boxes[k].x + D.boxes[k].w / 2, y: D.boxes[k].y + D.boxes[k].h / 2});
let selBox = null;
function edgePoint(b, tx, ty) { // where the line from the box centre towards (tx,ty) leaves the box
  const cx = b.x + b.w / 2, cy = b.y + b.h / 2, dx = tx - cx, dy = ty - cy;
  const k = Math.min(Math.abs((b.w / 2) / (dx || 1e-9)), Math.abs((b.h / 2) / (dy || 1e-9)));
  return [cx + dx * k, cy + dy * k]; }
const flowEls = D.flows.filter(f => f.from !== f.to).map(f => {
  const A = D.boxes[f.from], B = D.boxes[f.to], ac = centre(f.from), bc = centre(f.to);
  let x1, y1, x2, y2, cx, cy;
  if (Math.abs(A.y - B.y) < .01) {  // same row: leave from the top edge and arc over the boxes in between
    const span = Math.abs(bc.x - ac.x), dir = Math.sign(bc.x - ac.x);
    x1 = ac.x + dir * A.w * .15; x2 = bc.x - dir * B.w * .15 + (dir > 0 ? .06 : -.06);
    y1 = y2 = A.y; cx = (x1 + x2) / 2; cy = A.y - .12 - .1 * span * (dir > 0 ? 1 : 0.6);
  } else {
    [x1, y1] = edgePoint(A, bc.x, bc.y); [x2, y2] = edgePoint(B, ac.x, ac.y);
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, nx = -(y2 - y1), ny = x2 - x1, L = Math.hypot(nx, ny) || 1;
    cx = mx + nx / L * .08; cy = my + ny / L * .08;  // slight bend so A->B and B->A do not overlap
  }
  const path = el("path", {d: `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`, fill: "none", "stroke-width": S * (1.5 + 1.2 * Math.sqrt(f.links)),
    "marker-end": "url(#arr)", "stroke-linecap": "round"}, flowG);
  const lx = .25 * x1 + .5 * cx + .25 * x2, ly = .25 * y1 + .5 * cy + .25 * y2;
  const bg = el("rect", {x: lx - S * 15, y: ly - S * 10, width: S * 30, height: S * 18, rx: S * 4}, flowG);
  const t = el("text", {x: lx, y: ly + S * 4, "text-anchor": "middle", "font-size": S * 12, "font-family": "IBM Plex Mono, monospace"}, flowG);
  t.textContent = f.links;
  return {path, bg, t, f}; });
const lines = D.edges.map(([a, b]) => ({l: el("line", {}, eG), a, b}));
const circles = D.nodes.map(n => { const c = el("circle", {}, nG); n.px = n.x; n.py = n.y;
  c.addEventListener("click", ev => { ev.stopPropagation(); select(n.id); });
  c.addEventListener("mousemove", ev => tip(ev, n)); c.addEventListener("mouseleave", () => tipEl.style.display = "none"); return {c, n}; });
const labels = D.nodes.filter(n => n.deg >= 8).map(n => { const t = el("text", {"font-family": "IBM Plex Sans, sans-serif"}, lG); t.textContent = n.name; t.style.pointerEvents = "none"; return {t, n}; });
function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function target(n) { return view === "chain" ? (n.cx == null ? null : [n.cx, n.cy, n.cr]) : [n.x, n.y, S * (3 + 2.2 * Math.sqrt(n.deg))]; }
function colour(n) {
  if (view === "eco" || view === "chain") return n.eco == null ? GREY : PAL[n.eco % PAL.length];
  if (view === "role") return n.role ? ROLE[n.role] : GREY;
  if (view === "tier") return n.tier ? TIER[n.tier] : GREY;
  if (!n.eto) return GREY;
  const ok = n.eto.role_ok ?? n.eto.tier_ok; return ok == null ? GREY : (ok ? css("--ok") : css("--bad"));
}
function place(t) { // t in [0,1]: move nodes from previous to target position
  circles.forEach(({c, n}) => { const T = target(n); if (!T) { c.style.display = "none"; return; }
    const x = n.px + (T[0] - n.px) * t, y = n.py + (T[1] - n.py) * t; c.setAttribute("cx", x); c.setAttribute("cy", y); c.setAttribute("r", T[2]); n.vx = x; n.vy = y; });
  lines.forEach(({l, a, b}) => { const A = byId[a], B = byId[b]; l.setAttribute("x1", A.vx); l.setAttribute("y1", A.vy); l.setAttribute("x2", B.vx); l.setAttribute("y2", B.vy); });
  labels.forEach(({t, n}) => { const T = target(n); t.style.display = T && view !== "chain" ? "" : "none";
    if (T) { t.setAttribute("x", n.vx + S * 9); t.setAttribute("y", n.vy + S * 4); t.setAttribute("font-size", S * 11); } });
}
function animate() { const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches, t0 = performance.now(), dur = reduce ? 0 : 450;
  const step = now => { const t = dur ? Math.min(1, (now - t0) / dur) : 1, e = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; place(e);
    if (t < 1) requestAnimationFrame(step); else D.nodes.forEach(n => { n.px = n.vx; n.py = n.vy; }); }; requestAnimationFrame(step); }
function paint() {
  const chain = view === "chain";
  const hi = sel ? new Set([sel, ...nbr[sel].out, ...nbr[sel].in]) : null;
  const match = q ? new Set(D.nodes.filter(n => n.name.toLowerCase().includes(q)).map(n => n.id)) : null;
  boxEls.forEach(({r, t, k}) => { r.style.display = t.style.display = chain ? "" : "none"; r.setAttribute("fill", css("--box"));
    r.setAttribute("stroke", css("--accent")); r.setAttribute("stroke-width", S * (k === selBox ? 3 : 1.2)); t.setAttribute("fill", css("--accent")); });
  flowEls.forEach(({path, bg, t, f}) => { const on = chain && !sel && selBox && (f.from === selBox || f.to === selBox);
    [path, bg, t].forEach(x => x.style.display = on ? "" : "none");
    const out = f.from === selBox; path.setAttribute("stroke", out ? css("--accent") : css("--ok")); path.style.opacity = .7;
    bg.setAttribute("fill", css("--panel")); bg.setAttribute("stroke", out ? css("--accent") : css("--ok")); bg.setAttribute("stroke-width", S);
    t.setAttribute("fill", css("--ink")); });
  arrowHead.setAttribute("fill", css("--muted"));
  circles.forEach(({c, n}) => {
    const hidden = (hideLeaf && n.deg <= 1 && n.id !== sel) || !target(n); c.style.display = hidden ? "none" : "";
    const col = colour(n), hollow = ((view === "tier" || chain) && n.tier_from && n.tier_from !== "products" && !(chain && n.placed_by === "products")) || (view === "eto" && !n.eto);
    c.setAttribute("fill", hollow ? css("--panel") : col); c.setAttribute("stroke", hollow ? col : css("--panel"));
    c.setAttribute("stroke-width", S * (hollow ? 2 : 1));
    let op = 1; if (hi && !hi.has(n.id)) op = .15; if (match && !match.has(n.id)) op = .12; if (view === "eto" && !n.eto) op = Math.min(op, .35);
    c.style.opacity = op; });
  lines.forEach(({l, a, b}) => { const on = sel && (a === sel || b === sel);
    l.setAttribute("stroke", on ? css("--edge-hi") : css("--edge")); l.setAttribute("stroke-width", S * (on ? 2 : 1.1));
    const leaf = hideLeaf && (byId[a].deg <= 1 || byId[b].deg <= 1);
    l.style.display = (chain && !on) || (leaf && !on) || !target(byId[a]) || !target(byId[b]) ? "none" : ""; });
  labels.forEach(({t}) => t.setAttribute("fill", css("--ink")));
  document.getElementById("viewnote").textContent = NOTES[view];
  legend();
}
function legend() {
  const L = document.getElementById("legend"), T = document.getElementById("lg-title"); L.innerHTML = "";
  const item = (col, txt, hollow) => { const s = document.createElement("span"); const i = document.createElement("i");
    if (hollow) { i.className = "hollow"; i.style.borderColor = col; } else i.style.background = col; s.append(i, document.createTextNode(txt)); L.appendChild(s); };
  if (view === "eco" || view === "chain") { T.textContent = "Ecosystem (largest first, named by its biggest member)";
    const cnt = {}; D.nodes.forEach(n => cnt[n.eco] = (cnt[n.eco] || 0) + 1);
    Object.keys(cnt).sort((a, b) => a - b).forEach(k => { const top = D.nodes.filter(n => n.eco == k).sort((a, b) => b.deg - a.deg)[0];
      item(PAL[k % PAL.length], `${cnt[k]} · ${top.name}`); });
    if (view === "chain") item(GREY, "ring: placed by partners", true); }
  if (view === "role") { T.textContent = "Role from products (evidence-weighted)"; Object.entries(ROLE).forEach(([k, c]) => item(c, k.replace(/_/g, " "))); item(GREY, "no product profile"); }
  if (view === "tier") { T.textContent = "Tier (filled: from products · ring: from neighbours)"; Object.entries(TIER).forEach(([k, c]) => { item(c, k); item(c, k + " (propagated)", true); }); }
  if (view === "eto") { T.textContent = "Graded against CSET ETO"; item(css("--ok"), "matches ETO"); item(css("--bad"), "differs from ETO"); item(GREY, "ETO has no label", true); }
}
const tipEl = document.getElementById("tip");
function tip(ev, n) { const r = document.getElementById("graph").getBoundingClientRect();
  tipEl.innerHTML = `<b>${esc(n.name)}</b><br>${n.deg} supply link${n.deg === 1 ? "" : "s"}${n.lane ? " · " + esc(D.lane_names[n.lane]) : ""}`;
  tipEl.style.display = "block"; tipEl.style.left = Math.min(ev.clientX - r.left + 12, r.width - 270) + "px"; tipEl.style.top = (ev.clientY - r.top + 12) + "px"; }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])); }
const EMPTY = '<h2>Company</h2><p>Click a node to see its products, cluster labels and trading partners.</p>';
function select(id) { sel = id; selBox = null; const n = byId[id], box = document.getElementById("detail");
  const list = ids => ids.map(i => byId[i]).sort((a, b) => b.deg - a.deg).map(m => `<li>${esc(m.name)}${m.lane ? ` <span style="opacity:.7">· ${esc(D.lane_names[m.lane])}</span>` : ""}</li>`).join("") || "<li>none in the graph</li>";
  const eto = n.eto ? `<div style="margin-top:6px">ETO: <span class="pill">${n.eto.eto_role.replace(/_/g, " ")}</span><span class="pill">${n.eto.eto_stage || "no stage"}</span>` +
    (n.eto.role_ok != null ? `<span class="pill ${n.eto.role_ok ? "ok" : "bad"}">role ${n.eto.role_ok ? "matches" : "differs"}</span>` : "") +
    (n.eto.tier_ok != null ? `<span class="pill ${n.eto.tier_ok ? "ok" : "bad"}">tier ${n.eto.tier_ok ? "matches" : "differs"}</span>` : "") + `</div>` : "";
  box.innerHTML = `<h2>Company</h2><div class="name">${esc(n.name)}</div>
    <div style="margin-top:4px">${n.lane ? `<span class="pill">${esc(D.lane_names[n.lane])}${n.placed_by !== "products" ? " · " + n.placed_by : ""}</span>` : ""}<span class="pill">ecosystem ${n.eco}</span>${n.role ? `<span class="pill">${n.role.replace(/_/g, " ")}</span>` : ""}${n.prod ? `<span class="pill">product cluster ${n.prod}</span>` : ""}</div>${eto}
    <p style="margin-top:8px">Makes: ${n.makes.length ? n.makes.map(m => m.replace(/_/g, " ")).join(", ") : "no product profile"}</p>
    <h2 style="margin-top:10px">Supplies to (${nbr[id].out.length})</h2><ul>${list(nbr[id].out)}</ul>
    <h2 style="margin-top:10px">Buys from (${nbr[id].in.length})</h2><ul>${list(nbr[id].in)}</ul>`;
  paint(); }
svg.addEventListener("click", () => { sel = null; selBox = null; document.getElementById("detail").innerHTML = EMPTY; paint(); });
function selectBox(k) { selBox = k; sel = null; const box = document.getElementById("detail");
  if (!k) { box.innerHTML = EMPTY; paint(); return; }
  const rows = dir => D.flows.filter(f => (dir === "out" ? f.from : f.to) === k).map(f =>
    `<li>${esc(D.lane_names[dir === "out" ? f.to : f.from])} <span style="opacity:.7">· ${f.links}</span></li>`).join("") || "<li>none</li>";
  const self = D.flows.find(f => f.from === k && f.to === k);
  box.innerHTML = `<h2>Stage</h2><div class="name">${esc(D.lane_names[k])}</div>
    <p style="margin-top:6px">${D.nodes.filter(n => n.lane === k).length} companies${self ? ` · ${self.links} links inside the stage` : ""}. Blue arrows: it supplies. Green: it buys.</p>
    <h2 style="margin-top:10px">Supplies to</h2><ul>${rows("out")}</ul><h2 style="margin-top:10px">Buys from</h2><ul>${rows("in")}</ul>`;
  paint(); }
function setView(v) { view = v; document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-pressed", b.id === "v-" + v)); paint(); animate(); }
["chain", "eco", "role", "tier", "eto"].forEach(v => document.getElementById("v-" + v).addEventListener("click", () => setView(v)));
document.getElementById("q").addEventListener("input", e => { q = e.target.value.trim().toLowerCase(); paint(); });
document.getElementById("leaf").addEventListener("change", e => { hideLeaf = e.target.checked; paint(); });
function setVB() { svg.setAttribute("viewBox", vb.join(" ")); }
let drag = null;
svg.addEventListener("pointerdown", e => { drag = {x: e.clientX, y: e.clientY, vb: [...vb]}; });
window.addEventListener("pointerup", () => drag = null);
svg.addEventListener("pointermove", e => { if (!drag) return; const r = svg.getBoundingClientRect(), k = Math.max(vb[2] / r.width, vb[3] / r.height);
  vb[0] = drag.vb[0] - (e.clientX - drag.x) * k; vb[1] = drag.vb[1] - (e.clientY - drag.y) * k; setVB(); });
svg.addEventListener("wheel", e => { e.preventDefault(); const f = e.deltaY > 0 ? 1.15 : 1 / 1.15, r = svg.getBoundingClientRect();
  const k = Math.max(vb[2] / r.width, vb[3] / r.height), mx = vb[0] + (e.clientX - r.left) * k, my = vb[1] + (e.clientY - r.top) * k;
  vb = [mx - (mx - vb[0]) * f, my - (my - vb[1]) * f, vb[2] * f, vb[3] * f]; setVB(); }, {passive: false});
setVB();
document.getElementById("sub").textContent = `${D.nodes.length} companies, ${D.edges.length} supply links from the final Phase 1 KG · ${D.n_eco} ecosystems, modularity ${D.modularity}`;
const pairs = [];
D.nodes.filter(n => n.prod != null).forEach(a => D.nodes.filter(b => b.prod === a.prod && b.id > a.id && b.eco !== a.eco).forEach(b => pairs.push([a, b])));
pairs.sort((p, r) => (r[0].deg + r[1].deg) - (p[0].deg + p[1].deg));
const P = document.getElementById("pairs");
pairs.slice(0, 8).forEach(([a, b]) => { const btn = document.createElement("button");
  btn.textContent = `${a.name} · ${b.name}  (ecosystems ${a.eco} / ${b.eco})`;
  btn.addEventListener("click", ev => { ev.stopPropagation(); setView("eco"); select(a.id); });
  P.appendChild(btn); });
if (!pairs.length) P.innerHTML = "<p>None at this threshold.</p>";
D.nodes.forEach(n => { const T = target(n); if (T) { n.px = T[0]; n.py = T[1]; } });
place(1); paint();
const mq = window.matchMedia("(prefers-color-scheme: dark)"); mq.addEventListener && mq.addEventListener("change", paint);
new MutationObserver(paint).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
</script>
</body>
</html>
"""
