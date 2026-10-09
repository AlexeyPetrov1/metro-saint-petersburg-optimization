"""Data for the in-browser simulator player: one JSON payload per compare run (no Streamlit)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from metro_control.events import station_reasons, team_banner
from metro_control.line import load_line
from metro_control.mock import _assumption
from metro_control.screen import action_row
from metro_control.timeline import KPI_KEYS, VARIANTS, Timeline, frame
from metro_control.timeutil import to_msk

DIRS = ("north", "south")
SCENARIO_RU = {
    "rail_surge": "наплыв с вокзалов",
    "snowfall": "снегопад",
    "quiet_weekend": "спокойный выходной",
}
MONTHS_RU = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _date_ru(iso: str) -> str:
    try:
        d = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{d.day} {MONTHS_RU[d.month - 1]}"


CRITICAL_FILL = 0.95  # display threshold: a full or almost full segment within the next hour
EASING_ACTIONS = ("add_reserve", "shift_peak")


def _critical(ac: dict[str, Any], seg: int | None, d: str | None, k: int, data: dict) -> bool:
    """An action answers a critical situation if it adds capacity where, without control,
    demand on the target segment reaches 95 % of its capacity or more within the next hour."""
    if seg is None or d is None or ac.get("action") not in EASING_ACTIONS:
        return False
    row = data["baseline"]["ratio"][d][seg]
    ahead = [x for x in row[max(0, k - 1) : k + 4] if x is not None]
    return bool(ahead) and max(ahead) >= CRITICAL_FILL


def window_ks(ac: dict[str, Any], frame_ts: list[Any]) -> list[int]:
    """Frames (snapshot at t summarises (t-15, t]) inside the action window: start < t <= end.
    Without start/end: start = as_of, end = start + policy_action_window_min. Garbage -> []."""
    try:
        start = _ts(ac["start"]) if ac.get("start") else _ts(ac["as_of"])
        end = (
            _ts(ac["end"])
            if ac.get("end")
            else start + timedelta(minutes=float(_assumption("policy_action_window_min")))  # type: ignore[arg-type]
        )
        return [
            k
            for k, t in enumerate(frame_ts)
            if start < (t if isinstance(t, datetime) else _ts(t)) <= end
        ]
    except (KeyError, TypeError, ValueError, AttributeError):
        return []


def _sum(vals: list[float | None]) -> float | None:
    xs = [x for x in vals if x is not None]
    return sum(xs) if xs else None


def before_after(data: dict, seg: int | None, d: str | None, ks: list[int]) -> dict | None:
    """Baseline vs policy over the frames `ks` on one segment/direction (data = payload["data"])."""
    if seg is None or d is None or not ks:
        return None

    def col(v: str, field: str) -> list[float | None]:
        row = data[v].get(field, {}).get(d)
        return [row[seg][k] for k in ks] if row else [None] * len(ks)

    peak = [max((x for x in col(v, "ratio") if x is not None), default=None) for v in VARIANTS]
    if all(p is None for p in peak):
        return None
    trains = [_sum(col(v, "trains")) for v in VARIANTS]
    left = [_sum(col(v, "left")) for v in VARIANTS]
    return {
        "k0": min(ks),
        "k1": max(ks),
        "peak": peak,
        "trains": None if None in trains else [int(x) for x in trains],  # type: ignore[arg-type]
        "left": [None if x is None else int(round(x)) for x in left],
    }


def sim_payload(
    tl: Timeline,
    names: dict[str, str],
    actions: list[dict[str, Any]],
    policy_label: str,
    events: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the player draws, indexed [variant][...][k] over all frames.

    Rows go from the north terminal (Девяткино) down to the south one, so `tl.stations`
    (south first) is reversed. Segment i joins rows i and i+1; `north` = i+1 -> i.
    """
    order = list(reversed(tl.stations))
    n = tl.n_frames
    segs = []
    for i in range(len(order) - 1):
        up, down = order[i], order[i + 1]
        segs.append(
            {
                "label": names.get(up, up),
                "tip": f"{names.get(up, up)} – {names.get(down, down)}",
                "north": f"{down}__{up}",
                "south": f"{up}__{down}",
            }
        )
    times = [f"{to_msk(_ts(frame(tl, 'policy', k)['t'])):%H:%M}" for k in range(n)]

    def seg_series(v: str, key: str, field: str, default: Any = ...) -> list[float | None]:
        out = []
        for k in range(n):
            s = frame(tl, v, k)["segments"].get(key)
            # runs written before "ratio" existed fall back to the train fill
            fb = s["fill"] if s is not None and default is ... else default
            out.append(None if s is None else s.get(field, fb))
        return out

    data: dict[str, Any] = {}
    for v in VARIANTS:
        fr = [frame(tl, v, k) for k in range(n)]
        data[v] = {
            "fill": {d: [seg_series(v, s[d], "fill") for s in segs] for d in DIRS},
            "left": {d: [seg_series(v, s[d], "left_behind") for s in segs] for d in DIRS},
            "ratio": {d: [seg_series(v, s[d], "ratio") for s in segs] for d in DIRS},
            "trains": {d: [seg_series(v, s[d], "trains", None) for s in segs] for d in DIRS},
            "queues": {
                d: [[f["queues"].get(st, {}).get(d, 0.0) for f in fr] for st in order] for d in DIRS
            },
            "kpi": {x: [f[x] for f in fr] for x in KPI_KEYS},
        }

    def moving(k: int) -> bool:
        return any(row[k] is not None for v in VARIANTS for d in DIRS for row in data[v]["fill"][d])

    service = [k for k in range(n) if moving(k)]
    first, last = (service[0], service[-1]) if service else (0, n - 1)
    where = {}
    for i, sg in enumerate(segs):
        for d in DIRS:
            where[sg[d]] = (i, d)
    frame_ts = [frame(tl, "policy", i)["t"] for i in range(n)]
    acts = []
    for ac in actions:
        try:
            t = _ts(ac["as_of"])
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        k = next((i for i in range(n) if _ts(frame(tl, "policy", i)["t"]) > t), n)
        if ac.get("status") == "source_error":
            what, target, status = "Ошибка источника", ac.get("source", "?"), ac.get("reason", "")
        else:
            what, target, status = action_row(ac, names)
        seg, d = where.get(str(ac.get("target", "")), (None, None))
        ks = window_ks(ac, frame_ts)
        err = ac.get("status") == "source_error"
        win = ""
        if ks and not err:
            lo, hi = _ts(frame_ts[ks[0]]) - timedelta(minutes=15), _ts(frame_ts[ks[-1]])
            win = f"{to_msk(lo):%H:%M}–{to_msk(hi):%H:%M}"
        acts.append(
            {
                "k": k,
                "time": f"{to_msk(t):%H:%M}",
                "what": what,
                "target": target,
                "status": status,
                "reason": str(ac.get("reason", "")),
                "seg": seg,
                "dir": d,
                "applied": ac.get("status") == "applied",
                "error": ac.get("status") == "source_error",
                "critical": _critical(ac, seg, d, k, data),
                "ba": None if err else before_after(data, seg, d, ks),
                "win": win,
            }
        )
    ev = events or {}
    anomaly, reasons = ev.get("anomaly") or {}, ev.get("reasons") or {}
    return {
        "banner": ev.get("banner"),
        "anomaly": [anomaly.get(s) or [False] * n for s in order],
        "reasons": [reasons.get(s) or [[] for _ in range(n)] for s in order],
        "scenario": SCENARIO_RU.get(tl.scenario, tl.scenario),
        "date": _date_ru(tl.date),
        "policy": policy_label,
        "times": times,
        "first": first,
        "last": last,
        "stations": [names.get(s, s) for s in order],
        "segs": [{"label": s["label"], "tip": s["tip"]} for s in segs],
        "data": data,
        "actions": acts,
    }


def load_payload(
    load_pkg: Any,
    names: dict[str, str],
    recs: list[dict[str, Any]] | None = None,
    ctx: dict | None = None,
    anomaly: dict[str, set[datetime]] | None = None,
) -> dict[str, Any]:
    """Forecast segment load for the in-browser strip: fill[dir][seg][t] over the package slots.

    `recs`: [{"title", "target" (shown text), "target_id" (segment or station id), "start", "end"
    (UTC datetimes), "window", "is_mock"}] — each is outlined on the strip during its window.
    `ctx`: person 2 team context (banner, reasons); `anomaly`: station -> flagged slot starts.
    """
    line = load_line()
    order = [s.id for s in sorted(line.stations, key=lambda s: s.order, reverse=True)]
    slots = sorted({r.ts for r in load_pkg.payload})
    rows = {(r.segment_id, r.ts): r for r in load_pkg.payload}
    segs, fill, demand = [], {d: [] for d in DIRS}, {d: [] for d in DIRS}
    for i in range(len(order) - 1):
        up, down = order[i], order[i + 1]
        keys = {"north": f"{down}__{up}", "south": f"{up}__{down}"}
        segs.append(
            {
                "label": names.get(up, up),
                "tip": f"{names.get(up, up)} – {names.get(down, down)}",
                **keys,
            }
        )
        for d in DIRS:
            fill[d].append([getattr(rows.get((keys[d], t)), "r", None) for t in slots])
            demand[d].append([getattr(rows.get((keys[d], t)), "demand", None) for t in slots])
    where = {sg[d]: {"seg": i, "dir": d} for i, sg in enumerate(segs) for d in DIRS}
    out_recs = []
    for n, rec in enumerate([r for r in recs or [] if r.get("action") != "none"], 1):
        tid = str(rec.get("target_id", ""))
        target = where.get(tid) or ({"station": order.index(tid)} if tid in order else None)
        out_recs.append(
            {
                "n": n,
                "title": rec["title"],
                "where": rec.get("target", ""),
                "window": rec.get("window", ""),
                "mock": bool(rec.get("is_mock")),
                "target": target,
                "slots": [i for i, t in enumerate(slots) if rec["start"] <= t < rec["end"]],
            }
        )
    by_station = station_reasons(ctx, slots)
    flagged = anomaly or {}
    return {
        "banner": team_banner(ctx, names),
        "reasons": [by_station.get(s) or [[] for _ in slots] for s in order],
        "anomaly": [[t in flagged.get(s, ()) for t in slots] for s in order],
        "times": [f"{to_msk(t):%H:%M}" for t in slots],
        "stations": [names.get(s, s) for s in order],
        "segs": [{"label": s["label"], "tip": s["tip"]} for s in segs],
        "fill": fill,
        "demand": demand,
        "recs": out_recs,
    }


ASSETS = Path(__file__).parent / "assets"


def _page(name: str, payload: dict[str, Any]) -> str:
    """A self-contained player page: shared ui.css / ui.js inlined, payload inlined
    (`</` escaped so data cannot close the script)."""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (ASSETS / name).read_text(encoding="utf-8")
    html = html.replace("/*__UI_CSS__*/", (ASSETS / "ui.css").read_text(encoding="utf-8"))
    html = html.replace("//__UI_JS__", (ASSETS / "ui.js").read_text(encoding="utf-8"))
    return html.replace("__PAYLOAD__", data)


def player_html(payload: dict[str, Any]) -> str:
    return _page("sim_player.html", payload)


def load_html(payload: dict[str, Any]) -> str:
    return _page("load_player.html", payload)

