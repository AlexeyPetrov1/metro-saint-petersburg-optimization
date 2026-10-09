// Replay bridge: one clock for motion, queues, forecasts and optimizer decisions.
function latest(rows, t, fallback = null) {
  let lo = 0, hi = rows.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (rows[mid][0] <= t) lo = mid + 1; else hi = mid; }
  return lo ? rows[lo - 1][1] : fallback;
}
function drawAdapter(t) {
  const A = state.mode === "baseline" ? P.baseline_adapter : P.adapter;
  if (!A) return;
  const clockT = t;
  t = Math.min(t, A.end) - 0.000001;
  const sv = $("fleet"), w = sv.clientWidth, order = [...P.station_ids].reverse();
  sv.innerHTML = "";
  const L = 60, R = 30, dx = (w - L - R) / Math.max(1, order.length - 1), y = 96;
  const x = (sid) => L + order.indexOf(sid) * dx;
  el("line", {x1:L, x2:w-R, y1:y, y2:y, stroke:C.line1, "stroke-width":3}, sv);
  let waiting = 0, onboard = 0, onLine = 0, offLine = 0;
  const segmentLoads = {};
  order.forEach(sid => {
    const north = latest(A.queues[sid + ":1"] || [], t, 0);
    const south = latest(A.queues[sid + ":-1"] || [], t, 0), n = north + south;
    waiting += n;
    const g = el("g", {}, sv), name = P.stations[P.station_ids.indexOf(sid)];
    el("circle", {cx:x(sid), cy:y, r:Math.max(5, Math.min(20, Math.sqrt(n))), fill:n ? C.tight : "#fff", stroke:C.line1, "fill-opacity":.55}, g);
    el("title", {}, g).textContent = `${name}: на север ${north}, на юг ${south} чел.`;
    if (stripEls) {
      const s = P.station_ids.indexOf(sid), e = stripEls.st[s];
      e.q.setAttribute("r", n < 1 ? 0 : Math.max(5, Math.min(24, Math.sqrt(n) * 1.05)));
      e.qt.textContent = `${name}: на север ${north}, на юг ${south} чел.`;
    }
    el("text", {x:x(sid), y:140, transform:`rotate(-40 ${x(sid)} 140)`, "text-anchor":"end", "font-size":11, fill:C.ink}, sv).textContent = name;
  });
  const tracks = Object.fromEntries(A.tracks.map(r => [r.track_id, r]));
  A.trains.forEach(train => {
    const rows = A.motion[train.train_id] || [], loc = latest(rows, t, train.location), track = tracks[loc];
    const cars = Array.from({length:train.cars || 8}, (_,i) => latest(A.cars[train.train_id + ":C" + (i+1)] || [], t, 0));
    const total = cars.reduce((a,b)=>a+b,0); onboard += total;
    let origin = track ? track.from_station : loc, dest = track ? track.to_station : loc, fraction = 0;
    if (!order.includes(origin) || !order.includes(dest)) { offLine++; return; }
    onLine++;
    if (track) {
      let i = 0; while (i+1 < rows.length && rows[i+1][0] <= t) i++;
      const start = rows[i][0] < 0 ? ((train.initial_movement || {}).arrival_second || track.travel_seconds) - track.travel_seconds : rows[i][0];
      const arrival = rows.slice(i+1).find(r => !tracks[r[1]]);
      const end = arrival ? arrival[0] : start + track.travel_seconds;
      fraction = Math.max(0, Math.min(1, (t-start) / Math.max(1,end-start)));
    }
    const north = x(dest) > x(origin), px = x(origin) + (x(dest)-x(origin)) * fraction;
    const py = north ? 44 : 72, peak = Math.max(...cars), color = peak > 150 ? C.over : peak > 120 ? C.tight : C.calm;
    if (track) {
      const key = origin + ":" + dest;
      segmentLoads[key] = Math.max(segmentLoads[key] || 0, peak / 150);
    }
    const g = el("g", {}, sv);
    el("rect", {x:px-5, y:py-9, width:10, height:18, rx:3, fill:color, class:"train"}, g);
    el("title", {}, g).textContent = `${train.train_id}: ${total} пассажиров; вагоны: ${cars.join(", ")}`;
  });
  if (stripEls) for (let i = 0; i < P.station_ids.length - 1; i++) {
    const up = P.station_ids[i], down = P.station_ids[i+1];
    for (const [d,a,b] of [["north",down,up],["south",up,down]]) {
      const ratio = segmentLoads[a + ":" + b] ?? null, e = stripEls.segs[i][d];
      e.r.setAttribute("fill", smooth(ratio));
      e.t.textContent = ratio != null && ratio > .8 ? Math.round(ratio * 100) + "%" : "";
      e.ti.textContent = P.segs[i].tip + (ratio == null ? ": поезд в перегоне отсутствует" : `: самый плотный вагон ${Math.round(ratio * 150)} из 150 чел.`);
    }
  }
  $("fleetMeta").textContent = `На линии: ${onLine}; в депо, отстое или на подъездных путях: ${offLine}. В поездах ${fmt(onboard)} чел.; на платформах ${fmt(waiting)} чел.`;
  const cutoff = t;
  $("k_wait").textContent = fmt(waiting); $("k_denied").textContent = fmt(onboard);
  $("k_wpm").textContent = fmt(latest(A.entered || [], cutoff, 0));
  $("k_trains").textContent = onLine;
  ["d_wait","d_denied","d_wpm","d_trains"].forEach(id => { $(id).textContent = "По журналу симулятора"; $(id).className = "chip"; });
  const forecast = [...A.forecasts].reverse().find(f => f.second <= clockT);
  const fc = $("forecastRows"); fc.innerHTML = "";
  const isoMs = new Date(A.start).getTime() + clockT * 1000;
  if (!forecast) $("forecastMeta").textContent = "До этого момента нет опубликованного прогноза";
  else {
    const issued = forecast.issued_at.slice(11,16);
    const rows = forecast.rows.filter(r => new Date(r.interval_start).getTime() <= isoMs && isoMs < new Date(r.interval_end).getTime());
    $("forecastMeta").textContent = rows.length ? `Выпуск ${issued} МСК. Входы через турникеты, отдельно от очереди и загрузки состава.` : `Выпуск ${issued} МСК: текущий интервал не покрыт прогнозом`;
    [...order].forEach(sid => {
      const row = rows.find(r => r.station_id === sid), name = P.stations[P.station_ids.indexOf(sid)];
      const label = document.createElement("span"), value = document.createElement("b");
      label.textContent = name + (row && row.fallback ? " · историческая замена" : "");
      value.textContent = row ? fmt(row.q50) : "—";
      if (row) value.title = `q10 ${row.q10 ?? "нет"}; q50 ${row.q50}; q90 ${row.q90 ?? "нет"}`;
      fc.append(label,value);
    });
  }
  const decision = [...A.decisions].reverse().find(d => d.second <= clockT);
  $("decisionMeta").textContent = decision ? `${decision.candidate} · ${decision.status} · ${decision.release_authorized ? "проверки пройдены" : "разрешение на исполнение отсутствует"}` : "До этого момента нет решения";
  const box = $("decisionText"); box.innerHTML = "";
  const explanation = decision && decision.explanation;
  if (explanation && explanation.sections) {
    const source = document.createElement("p"); source.textContent = `Источник объяснения: ${(explanation.metadata || {}).source || "сохранённый ответ"}`; box.appendChild(source);
    ["action", "prediction", "rationale", "effects", "risks", "revisit"].forEach(section => {
      const rows = explanation.sections[section];
      if (!Array.isArray(rows)) return;
      rows.slice(0, 2).forEach(row => { const p = document.createElement("p"); p.textContent = (row.text || "").slice(0,700); box.appendChild(p); });
    });
  } else box.textContent = decision ? "Сохранённое объяснение отсутствует" : "";
}
function render() {
  renderBase();
  const t = Math.min(P.adapter ? P.adapter.end : Infinity, state.pos * 900);
  drawAdapter(t);
  // Our timestamps are Moscow local time; format without the browser's timezone.
  const local = new Date(new Date(P.adapter.start).getTime() + t * 1000 + 3 * 3600000);
  $("clock").textContent = local.toISOString().slice(11,19);
}
