"""Recorded 15-min timeline of a compare run and pure replay helpers (no Streamlit)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from metro_control.screen import band
from metro_control.sim import NORTH, SimParams, SimState, onboard, pending_wait_pax_min, waiting
from metro_control.timeutil import to_msk

FILE = "timeline.json"
STEP_MIN = 15
VARIANTS = ("baseline", "policy")
KPI_KEYS = ("waiting", "denied", "wait_pax_min", "trains_in_service")


def trains_on_line(state: SimState) -> int:
    """One-way trips that have left their origin terminal and not reached the far one;
    cancelled trips excluded."""
    departed = {x.train_id for x in state.log}
    return sum(1 for t in state.trains.values() if t.in_service and t.train_id in departed)


def snapshot(state: SimState, params: SimParams, log_from: int, t_utc: datetime) -> dict[str, Any]:
    """State at a 15-min boundary; segment loads come from stops logged since `log_from`."""
    queues: dict[str, dict[str, float]] = {}
    for (st, d), q in sorted(state.queues.items()):
        w = sum(sum(c.by_dest.values()) for c in q)
        if w > 0:
            queues.setdefault(st, {NORTH: 0.0, "south": 0.0})[d] = round(w, 1)
    segs: dict[str, dict[str, float]] = {}
    names = params.stations
    for x in state.log[log_from:]:
        i = names.index(x.station)
        j = i + 1 if x.direction == NORTH else i - 1
        if not 0 <= j < len(names):
            continue
        s = segs.setdefault(
            f"{x.station}__{names[j]}", {"fill": 0.0, "left_behind": 0.0, "load": 0.0, "trains": 0}
        )
        s["fill"] = max(s["fill"], x.load_after / params.capacity)
        s["left_behind"] += x.left_behind
        s["load"] += x.load_after
        s["trains"] += 1
    return {
        "t": t_utc.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entered": round(state.entered, 1),
        "alighted": round(state.alighted, 1),
        "denied": round(state.denied, 1),
        "waiting": round(waiting(state), 1),
        "onboard": round(onboard(state), 1),
        "wait_pax_min": round(state.wait_pax_min + pending_wait_pax_min(state), 1),
        "trains_in_service": trains_on_line(state),
        "queues": queues,
        "segments": {
            k: {
                "fill": round(v["fill"], 3),
                "left_behind": round(v["left_behind"], 1),
                "trains": v["trains"],
                # demand / capacity, as in the forecast: everyone who wanted to ride this segment
                # in the interval (carried + left on the platform) over all trains' places
                "ratio": round((v["load"] + v["left_behind"]) / (v["trains"] * params.capacity), 3),
            }
            for k, v in sorted(segs.items())
        },
    }


class Timeline(BaseModel):
    run_id: str
    scenario: str
    date: str
    step_min: int = STEP_MIN
    capacity: float
    stations: list[str]
    frames: dict[str, list[dict[str, Any]]]

    @field_validator("frames")
    @classmethod
    def _frames(cls, v: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
        if set(v) != set(VARIANTS):
            raise ValueError("frames: need baseline and policy")
        if len(v["baseline"]) != len(v["policy"]) or not v["baseline"]:
            raise ValueError("frames: baseline and policy must be non-empty and equal length")
        return v

    @model_validator(mode="after")
    def _keys(self) -> Timeline:
        for fr in self.frames.values():
            for f in fr:
                missing = {"t", "queues", "segments", *KPI_KEYS} - set(f)
                if missing:
                    raise ValueError(f"frame: missing {sorted(missing)}")
        return self

    @property
    def n_frames(self) -> int:
        return len(self.frames["baseline"])


def dump(tl: Timeline) -> str:
    body = json.dumps(tl.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return body + "\n"


def load_timeline(run_dir: Path | str) -> tuple[Timeline | None, str | None]:
    path = Path(run_dir) / FILE
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, f"{FILE}: file not found"
    except OSError as e:
        return None, f"{FILE}: cannot read ({e.__class__.__name__})"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, f"{FILE}: invalid JSON"
    try:
        return Timeline.model_validate(data), None
    except ValidationError as e:
        return None, f"{FILE}: schema error ({e.errors()[0]['msg']})"


def frame(tl: Timeline, variant: str, k: int) -> dict[str, Any]:
    return tl.frames[variant][max(0, min(k, tl.n_frames - 1))]


def compare_frame(tl: Timeline, k: int) -> dict[str, dict[str, float]]:
    b, p = frame(tl, "baseline", k), frame(tl, "policy", k)
    return {
        "baseline": {x: b[x] for x in KPI_KEYS},
        "policy": {x: p[x] for x in KPI_KEYS},
        "delta": {x: round(p[x] - b[x], 1) for x in KPI_KEYS},
    }


def segment_bands(fr: dict[str, Any], stations: list[str]) -> pl.DataFrame:
    rows = []
    for i in range(len(stations) - 1):
        for d, a, b in (
            ("north", stations[i], stations[i + 1]),
            ("south", stations[i + 1], stations[i]),
        ):
            s = fr["segments"].get(f"{a}__{b}")
            if s is None:
                rows.append((f"{a}__{b}", a, b, d, None, 0.0, "none"))
                continue
            bd = "high" if s["left_behind"] > 0 else band(s["fill"])
            rows.append((f"{a}__{b}", a, b, d, s["fill"], s["left_behind"], bd))
    return pl.DataFrame(
        rows,
        schema={
            "segment_id": pl.Utf8,
            "from_station": pl.Utf8,
            "to_station": pl.Utf8,
            "direction": pl.Utf8,
            "fill": pl.Float64,
            "left_behind": pl.Float64,
            "band": pl.Utf8,
        },
        orient="row",
    )


def series(tl: Timeline, upto_k: int) -> pl.DataFrame:
    rows = []
    for v in VARIANTS:
        for k in range(min(upto_k, tl.n_frames - 1) + 1):
            f = tl.frames[v][k]
            rows.append((k, f["t"], v, float(f["waiting"]), float(f["denied"])))
    return pl.DataFrame(
        rows,
        schema={
            "k": pl.Int64,
            "t": pl.Utf8,
            "variant": pl.Utf8,
            "waiting": pl.Float64,
            "denied": pl.Float64,
        },
        orient="row",
    )


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def actions_until(actions: list[dict[str, Any]], t_utc: datetime) -> list[dict[str, Any]]:
    """Actions decided strictly before t: frame k is recorded before the policy acts at k."""
    out = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        try:
            ts = _parse(a["as_of"])
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts < t_utc:
            out.append(a)
    return out


def find_sim_dir(env_value: str | None, runs_root: Path) -> Path | None:
    if env_value:
        return Path(env_value)
    if not runs_root.is_dir():
        return None
    for d in sorted(runs_root.glob("compare-*"), key=lambda x: x.name, reverse=True):
        if d.name.endswith("-forecast"):
            continue
        if (d / FILE).is_file():
            return d
    return None


def forecast_twin(sim_dir: Path | None) -> Path | None:
    """The `<sim_dir>-forecast` run (demand = forecast) if it holds a timeline file."""
    if sim_dir is None:
        return None
    twin = sim_dir.with_name(sim_dir.name + "-forecast")
    return twin if (twin / FILE).is_file() else None


@dataclass(frozen=True)
class Player:
    n_frames: int
    k: int = 0
    playing: bool = False


def step(p: Player) -> Player:
    k = min(p.k + 1, p.n_frames - 1)
    return replace(p, k=k, playing=p.playing and k < p.n_frames - 1)


def toggle(p: Player) -> Player:
    return replace(p, playing=not p.playing and p.k < p.n_frames - 1)


def reset(p: Player) -> Player:
    return replace(p, k=0, playing=False)


def _lerp(a: float | None, b: float | None, f: float) -> float | None:
    if a is None or b is None:
        return a if f < 0.5 else b
    return a + (b - a) * f


def blend_frames(a: dict[str, Any], b: dict[str, Any], f: float) -> dict[str, Any]:
    """Frame between a and b at fraction f in [0, 1) for smooth playback (display only)."""
    if f <= 0:
        return a
    out = {x: _lerp(a.get(x), b.get(x), f) for x in (*KPI_KEYS, "entered", "onboard")}
    out["t"] = a["t"]
    zero = {"north": 0.0, "south": 0.0}
    out["queues"] = {
        s: {d: _lerp(a["queues"].get(s, zero)[d], b["queues"].get(s, zero)[d], f) for d in zero}
        for s in set(a["queues"]) | set(b["queues"])
    }
    none = {"fill": None, "left_behind": 0.0}
    out["segments"] = {
        s: {
            "fill": _lerp(
                a["segments"].get(s, none)["fill"], b["segments"].get(s, none)["fill"], f
            ),
            "left_behind": _lerp(
                a["segments"].get(s, none)["left_behind"],
                b["segments"].get(s, none)["left_behind"],
                f,
            ),
        }
        for s in set(a["segments"]) | set(b["segments"])
    }
    return out


def tick(
    p: Player, now: float, last: float | None, period: float = 1.0
) -> tuple[Player, float | None]:
    """Advance one frame only when playing and a period has elapsed since `last`."""
    if not p.playing or not (last is None or now - last >= period * 0.9):
        return p, last
    return step(p), now


@dataclass(frozen=True)
class HeatGrid:
    """Segment x time matrix for one direction; `z[segment][time]`."""

    seg_labels: list[str]
    seg_tips: list[str]
    ks: list[int]
    times: list[str]
    z: list[list[float | None]]
    left_behind: list[list[float]]


def _seg_fill(fr: dict[str, Any], key: str) -> tuple[float | None, float]:
    s = fr["segments"].get(key)
    return (None, 0.0) if s is None else (s["fill"], s["left_behind"])


def heat_grid(
    tl: Timeline,
    variant: Literal["baseline", "policy", "diff"],
    direction: Literal["north", "south"],
    names: dict[str, str] | None = None,
) -> HeatGrid:
    """Heat-map matrix over service frames; diff = policy - baseline fill.

    Row label is the upper (later in `stations`) station of the pair.
    """
    nm = names or {}
    st = tl.stations
    pairs = []
    for i in range(len(st) - 1):
        a, b = (st[i], st[i + 1]) if direction == "north" else (st[i + 1], st[i])
        pairs.append((st[i + 1], a, b))
    ks = [
        k
        for k in range(tl.n_frames)
        if any(
            fr[k]["segments"].get(f"{a}__{b}") is not None
            for fr in tl.frames.values()
            for _, a, b in pairs
        )
    ]
    if ks:
        ks = list(range(ks[0], ks[-1] + 1))
    z: list[list[float | None]] = []
    lb: list[list[float]] = []
    for _, a, b in pairs:
        key = f"{a}__{b}"
        zr: list[float | None] = []
        lr: list[float] = []
        for k in ks:
            if variant == "diff":
                f0, _ = _seg_fill(tl.frames["baseline"][k], key)
                f1, l1 = _seg_fill(tl.frames["policy"][k], key)
                zr.append(None if f0 is None or f1 is None else f1 - f0)
                lr.append(l1)
            else:
                f, lv = _seg_fill(tl.frames[variant][k], key)
                zr.append(f)
                lr.append(lv)
        z.append(zr)
        lb.append(lr)
    return HeatGrid(
        seg_labels=[nm.get(u, u) for u, _, _ in pairs],
        seg_tips=[f"{nm.get(a, a)} → {nm.get(b, b)}" for _, a, b in pairs],
        ks=ks,
        times=[f"{to_msk(_parse(tl.frames['policy'][k]['t'])):%H:%M}" for k in ks],
        z=z,
        left_behind=lb,
    )

