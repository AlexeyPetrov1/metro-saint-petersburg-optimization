"""Plotly figures for the dispatcher screen (no Streamlit)."""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from metro_control.timeline import HeatGrid

PAPER = "#F4F5F7"
INK = "#1B2430"
MUTED = "#6B7685"
LINE1 = "#D6083B"
CALM_LO = "#E1EEF8"
CALM_HI = "#0078C9"
TIGHT = "#EA7125"
OVER = "#D6083B"
NONE_COLOR = "#C9CED6"
FONT = "Golos Text, sans-serif"

TIGHT_AT = 0.8
OVER_AT = 1.0
Z_MAX = 1.3


def _hex(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _mix(a: str, b: str, f: float) -> str:
    ca, cb = _hex(a), _hex(b)
    return "#" + "".join(f"{round(x + (y - x) * f):02X}" for x, y in zip(ca, cb, strict=True))


def load_color(fill: float | None) -> str:
    """Colour by the ТЗ bands: calm gradient up to 80%, tight to 100%, over above."""
    if fill is None:
        return NONE_COLOR
    if fill <= TIGHT_AT:
        return _mix(CALM_LO, CALM_HI, max(fill, 0.0) / TIGHT_AT)
    return TIGHT if fill <= OVER_AT else OVER


_T = TIGHT_AT / Z_MAX
_O = OVER_AT / Z_MAX
LOAD_SCALE = [
    [0.0, CALM_LO],
    [_T, CALM_HI],
    [_T, TIGHT],
    [_O, TIGHT],
    [_O, OVER],
    [1.0, OVER],
]
DIFF_SCALE = [[0.0, CALM_HI], [0.5, "#FFFFFF"], [1.0, OVER]]

# Heatmap: four flat bands (no gradient) so the day reads at a glance; the app draws the legend.
QUIET_AT = 0.5
QUIET = "#CFE3F3"
HEAT_BANDS = [
    ("до 50 %", QUIET),
    ("50–80 %", CALM_HI),
    ("80–100 %", TIGHT),
    ("больше 100 %", OVER),
]
_Q = QUIET_AT / Z_MAX
HEAT_SCALE = [
    [0.0, QUIET],
    [_Q, QUIET],
    [_Q, CALM_HI],
    [_T, CALM_HI],
    [_T, TIGHT],
    [_O, TIGHT],
    [_O, OVER],
    [1.0, OVER],
]
DIFF_BANDS = [
    ("политика разгрузила", CALM_HI),
    ("без изменений", "#FFFFFF"),
    ("добавила нагрузку", OVER),
]

_LAYOUT: dict[str, Any] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=INK, size=12),
)


def _hover(g: HeatGrid, diff: bool) -> list[list[str]]:
    out = []
    for i, row in enumerate(g.z):
        r = []
        for j, v in enumerate(row):
            head = f"{g.seg_tips[i]}, {g.times[j]}"
            if v is None:
                r.append(f"{head}: нет движения")
            elif diff:
                r.append(f"{head}: {v * 100:+.0f} п.п.")
            else:
                r.append(f"{head}: заполнение {v:.0%}, остались {g.left_behind[i][j]:.0f} чел.")
        out.append(r)
    return out


def heatmap_figure(
    north: HeatGrid, south: HeatGrid, playhead_time: str | float, diff: bool
) -> go.Figure:
    """playhead_time: a time label from the grid, or a fractional column index for smooth play."""
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=("На север", "На юг"),
    )
    if diff:
        kw: dict[str, Any] = dict(
            colorscale=DIFF_SCALE,
            zmin=-0.5,
            zmax=0.5,
            colorbar=dict(
                title="п.п.",
                tickvals=[-0.5, 0, 0.5],
                ticktext=["-50", "0", "+50"],
                thickness=10,
                len=0.9,
            ),
        )
    else:
        kw = dict(colorscale=HEAT_SCALE, zmin=0, zmax=Z_MAX)
    for col, g in ((1, north), (2, south)):
        k = dict(kw)
        k["showscale"] = False
        fig.add_trace(
            go.Heatmap(
                z=g.z,
                x=g.times,
                y=g.seg_labels,
                text=_hover(g, diff),
                hovertemplate="%{text}<extra></extra>",
                xgap=0,
                ygap=1,
                **k,
            ),
            row=1,
            col=col,
        )
        fig.add_vline(x=playhead_time, line=dict(color=INK, width=3), row=1, col=col)
    for col, g in ((1, north), (2, south)):
        ticks = [t for t in g.times if t.endswith(":00") and int(t[:2]) % 4 == 2]
        fig.update_xaxes(
            type="category",
            tickmode="array",
            tickvals=ticks,
            ticktext=ticks,
            tickangle=0,
            tickfont=dict(size=11),
            row=1,
            col=col,
        )
    # stations arrive south->north (first = Ветеранов), so the north terminal is drawn on top
    fig.update_yaxes(type="category", tickfont=dict(size=11))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=30), **_LAYOUT)
    return fig


def line_strip_figure(
    view: pl.DataFrame,
    queues: dict[str, dict[str, float]],
    names: dict[str, str],
) -> go.Figure:
    rows = {
        d: list(view.filter(pl.col("direction") == d).iter_rows(named=True))
        for d in ("north", "south")
    }
    ref = rows["north"] or rows["south"]
    ids = [r["from_station"] for r in ref] + ([ref[-1]["to_station"]] if ref else [])
    n = len(ids)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, max(n - 1, 0)],
            y=[0, 0],
            mode="lines",
            line=dict(color=LINE1, width=3),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for d, base in (("north", 0.55), ("south", -1.45)):
        rs = rows[d]
        fills = [r["fill"] for r in rs]
        pos = {sid: i for i, sid in enumerate(ids)}
        xs = [min(pos[r["from_station"]], pos[r["to_station"]]) + 0.5 for r in rs]
        tips = []
        for r in rs:
            a, b = (
                names.get(r["from_station"], r["from_station"]),
                names.get(r["to_station"], r["to_station"]),
            )
            f = r["fill"]
            tail = (
                "нет движения"
                if f is None
                else (f"заполнение {f:.0%}, остались {r['left_behind']:.0f} чел.")
            )
            tips.append(f"{a} → {b}: {tail}")
        fig.add_trace(
            go.Bar(
                x=xs,
                y=[0.9] * len(rs),
                base=[base] * len(rs),
                width=0.92,
                marker=dict(color=[load_color(f) for f in fills], line=dict(width=0)),
                text=[f"{f:.0%}" if f is not None and f > TIGHT_AT else "" for f in fills],
                textposition="inside",
                textfont=dict(color="#FFFFFF", size=10),
                textangle=0,
                hovertext=tips,
                hoverinfo="text",
                showlegend=False,
            )
        )
    qn = [queues.get(s, {}).get("north", 0.0) for s in ids]
    qs = [queues.get(s, {}).get("south", 0.0) for s in ids]
    tot = [a + b for a, b in zip(qn, qs, strict=True)]
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=[0] * n,
            mode="markers",
            marker=dict(
                size=[max(8.0, 2 * math.sqrt(t)) if t > 0 else 0 for t in tot],
                color=TIGHT,
                opacity=0.85,
                line=dict(width=1.5, color=INK),
            ),
            hovertext=[
                f"{names.get(s, s)}: ждут на север {a:.0f}, на юг {b:.0f}"
                for s, a, b in zip(ids, qn, qs, strict=True)
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(n)),
            y=[0] * n,
            mode="markers",
            marker=dict(size=8, color="#FFFFFF", line=dict(color=LINE1, width=2)),
            hovertext=[names.get(s, s) for s in ids],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=[names.get(s, s) for s in ids],
        tickangle=-45,
        tickfont=dict(size=11),
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        range=[max(n - 1, 0) + 0.6, -0.6],  # north terminal on the left
        side="bottom",
    )
    fig.update_yaxes(visible=False, range=[-1.7, 1.7])
    for txt, y in (("на север", 1.0), ("на юг", -1.0)):
        fig.add_annotation(
            x=0, xref="paper", y=y, text=txt, showarrow=False, xanchor="right", xshift=-4,
            font=dict(color=MUTED, size=12),
        )  # fmt: skip
    fig.update_layout(
        height=340, margin=dict(l=70, r=10, t=10, b=110), bargap=0, barmode="overlay", **_LAYOUT
    )
    return fig


def _clock(k: int) -> str:
    m = (180 + 15 * int(k)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def waiting_figure(series_df: pl.DataFrame, k_time: str | float) -> go.Figure:
    """k_time: a clock label, or a fractional frame index for smooth play."""
    fig = go.Figure()
    for v, nm, color, dash in (
        ("baseline", "без управления", MUTED, "dash"),
        ("policy", "с политикой (mock)", CALM_HI, "solid"),
    ):
        sv = series_df.filter(pl.col("variant") == v)
        fig.add_trace(
            go.Scatter(
                x=[_clock(k) for k in sv["k"]],
                y=sv["waiting"],
                name=nm,
                mode="lines",
                line=dict(color=color, width=2, dash=dash),
            )
        )
    fig.add_vline(x=k_time, line=dict(color=INK, width=2))
    ticks = [_clock(k) for k in range(0, 97, 8)]
    fig.update_xaxes(type="category", tickmode="array", tickvals=ticks, ticktext=ticks)
    fig.update_yaxes(title="ждут, чел.", gridcolor="#E3E6EA")
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=30),
        legend=dict(orientation="h", y=1.15),
        **_LAYOUT,
    )
    return fig

