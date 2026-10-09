// Shared helpers of the in-browser players (docs/design.md). Inlined into each iframe before its own script.
const C = { ink: "#1B2430", muted: "#6B7685", rule: "#E3E6EA", quiet: "#CFE3F3", calm: "#0078C9",
  tight: "#EA7125", over: "#D6083B", line1: "#D6083B", good: "#009A49", step: "#702785", none: "#E9ECEF", empty: "#F7F8F9" };
const $ = (id) => document.getElementById(id);
const fmt = (x) => Math.round(x).toLocaleString("ru-RU").replace(/ /g, " ");
const sfmt = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const lerp = (a, b, f) => (a == null || b == null) ? (f < 0.5 ? a : b) : a + (b - a) * f;
const hex = (c) => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));
const mix = (a, b, f) => { const x = hex(a), y = hex(b); return "rgb(" + x.map((v, i) => Math.round(v + (y[i] - v) * f)).join(",") + ")"; };
// Load = demand / capacity. Four flat bands for maps, a continuous ramp for animated elements.
const OVER = 1.0;
function band(f) { if (f == null) return null; if (f < 0.5) return C.quiet; if (f <= 0.8) return C.calm; if (f <= OVER) return C.tight; return C.over; }
function smooth(f) {
  if (f == null) return C.none; if (f >= 1.03) return C.over;
  const st = [[0, "#E1EEF8"], [0.5, "#7DB7E3"], [0.8, C.calm], [0.85, C.tight], [1.0, C.tight], [1.03, C.over]];
  for (let i = 1; i < st.length; i++) if (f <= st[i][0]) return mix(st[i - 1][1], st[i][1], (f - st[i - 1][0]) / (st[i][0] - st[i - 1][0]));
  return C.over;
}
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };

// Weather / event banner (docs/design.md). `b` = payload.banner or null; host = empty #banner element.
function renderBanner(b, host) {
  if (!b) { host.style.display = "none"; return; }
  const chips = b.items.map(i => `<span class="bc ${i.kind}" title="${esc(i.note || "")}"><span>${i.kind === "weather" ? "Погода" : "Событие"}</span><b>${esc(i.text)}</b>` +
    (i.effect ? `<span>${esc(i.effect)}</span>` : "") + `<em>${esc(i.where)}${i.window ? ", " + esc(i.window) : ""}</em>` + (i.note ? `<em>${esc(i.note)}</em>` : "") + `</span>`).join("");
  const notes = (b.warnings || []).map(esc).join(" · ");
  host.className = "banner";
  host.innerHTML = `<span class="bt">Что влияет на загрузку</span>${chips}` + (b.mock ? `<span class="chip">mock</span>` : "") + (notes ? `<span class="bm">${notes}</span>` : "");
}
// Station tooltip text: name + reasons of one slot (+ anomaly mark).
const stationTip = (name, reasons, anom) => [name, ...(reasons || []), ...(anom ? ["аномалия"] : [])].join("\n");

