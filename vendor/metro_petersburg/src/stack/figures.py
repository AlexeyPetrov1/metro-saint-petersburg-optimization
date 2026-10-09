"""Графики стекинга: reports/figures/spb_17_*.png (python -m src.stack --figures).

Палитра проекта (dataviz, проверена валидатором на фоне #fcfcfb): B1 — синий, как модель на прежних графиках;
B2 — оранжевый; B3 — бирюзовый; стекинг — фиолетовый; простое среднее, норма и факт — серые. Цвет закреплён за моделью
на всех трёх графиках. У бирюзового контраст с фоном ниже 3:1, поэтому линии подписаны прямо.
"""
import json

import numpy as np
import pandas as pd

from src import config, eda
from src.stack import base as B
from src.stack import data as D
from src.stack.__main__ import CROSS, OUT, SEPT

COLORS = {"b1": "#2a78d6", "b2": "#eb6834", "b3": "#1baf7a", "stack": "#4a3aa7", "mean": eda.MUTED}
FACT, NORM, SPAN = "#52514e", "#898781", "#e4e1f1"
SHORT = {"b1": "B1: часовая LightGBM", "b2": "B2: персистентность", "b3": "B3: GRU", "mean": "среднее B1–B3",
         "stack": "стекинг"}


def _save(fig, name: str) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / name)


def wape_horizon(path_name: str = "spb_17_wape_horizon.png") -> None:
    """WAPE по горизонтам: май + июль вне выборки (основной результат) и сентябрь."""
    import matplotlib.pyplot as plt

    panels = [(pd.read_csv(OUT / "cross_metrics.csv"), CROSS, "Май + июль, вне выборки (основной результат)")]
    if (OUT / "september_metrics.csv").exists():
        panels.append((pd.read_csv(OUT / "september_metrics.csv"), SEPT,
                       "Сентябрь, один прогон (часовые результаты уже известны)"))
    eda.style()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (m, period, title) in zip(axes, panels):
        m = m[(m.period == period) & (m.slice == "все слоты") & (m.horizon != "все")].copy()
        m["horizon"] = m.horizon.astype(int)
        ends = []
        for name in ("b2", "b3", "mean", "b1", "stack"):
            g = m[m.model == name].sort_values("horizon")
            ls = "--" if name == "mean" else "-"
            lw = 2.6 if name == "stack" else 2
            ax.plot(g.horizon, g.wape * 100, ls, color=COLORS[name], lw=lw, marker="o", ms=7 if name == "stack" else 6,
                    label=SHORT[name], zorder=3 if name == "stack" else 2)
            ends.append((g.wape.iloc[-1] * 100, name))
        ends.sort()
        last = -np.inf
        for v, name in ends:                                   # подписи у правого края без наложений
            y = max(v, last + 0.28)
            ax.annotate(SHORT[name], (120, v), xytext=(126, y), textcoords="data", va="center", fontsize=9.5,
                        color=eda.INK2, arrowprops=None)
            last = y
        ax.set_xticks(D.HORIZONS, [f"{h} мин" for h in D.HORIZONS])
        ax.set_xlim(24, 168)
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.set_xlabel("горизонт: конец слота − момент прогноза")
    axes[0].set_ylabel("WAPE получасового слота, %")
    axes[0].legend(loc="upper left", fontsize=9.5)
    fig.suptitle("Прогноз каждые 30 минут: точность по горизонтам", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    _save(fig, path_name)
    plt.close(fig)


def weights(path_name: str = "spb_17_stack_weights.png") -> None:
    """Веса стекинга (обучение на мае + июле) по горизонтам: доли B1, B2, B3 для q10, q50, q90."""
    import matplotlib.pyplot as plt

    w = pd.read_csv(OUT / "weights.csv")
    w = w[(w.fit == "май + июль") & (w.level == "горизонт")]
    eda.style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, q in zip(axes, ("q10", "q50", "q90")):
        g = w[w["quantile"] == q].sort_values("cell")
        x = np.arange(len(g))
        bottom = np.zeros(len(g))
        for name in B.BASES:
            vals = g[f"w_{name}"].to_numpy()
            ax.bar(x, vals, bottom=bottom, width=0.62, color=COLORS[name], edgecolor=eda.SURFACE, linewidth=2,
                   label=SHORT[name])
            for xi, b, v in zip(x, bottom, vals):
                if v >= 0.1:
                    ax.text(xi, b + v / 2, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=9.5,
                            color="white")
            bottom += vals
        ax.set_xticks(x, [f"{30 * int(c)}" for c in g.cell])
        ax.set_xlabel("горизонт, мин")
        ax.set_title(q, loc="left", fontsize=11.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("вес в пространстве z")
    axes[0].set_ylim(0, 1.0)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9.5)
    fig.suptitle("Кому стекинг верит: веса по горизонтам (обучение на мае + июле)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    _save(fig, path_name)
    plt.close(fig)


def demo(path_name: str | None = None) -> None:
    """Демо-вечер сентября: прогнозы каждые 30 минут — линия и станция."""
    import matplotlib.pyplot as plt

    run = json.loads((OUT / "september_run.json").read_text(encoding="utf-8"))
    day = pd.Timestamp(run["demo_day"])
    d = pd.read_csv(OUT / f"demo_{day:%d%m}.csv")
    names = pd.read_csv(config.SPB_STATIONS_CSV).set_index("station_id").name.to_dict()
    eda.style()
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.4), sharex=True)
    for ax, place in zip(axes, ("вся линия", run["demo_station"])):
        p = d[d.place == place]
        k1, k4 = p[p.k == 1].sort_values("slot_local"), p[p.k == 4].sort_values("slot_local")
        fact = pd.concat([k1, k4]).drop_duplicates("slot_local").sort_values("slot_local")
        x = lambda s: pd.to_datetime(f"{day.date()} " + s.slot_local)
        if "stack_q10" in k1 and k1.stack_q10.notna().any():
            ax.fill_between(x(k1), k1.stack_q10, k1.stack_q90, color=SPAN, lw=0, label="стекинг q10–q90, 30 мин")
        ax.plot(x(fact), fact.n, "--", color=NORM, lw=1.6, label="норма слота")
        ax.plot(x(k4), k4.b1, "-", color=COLORS["b1"], lw=1.6, alpha=0.55, marker="o", ms=5,
                label="B1, за 120 мин")
        ax.plot(x(k4), k4["stack"], "-", color=COLORS["stack"], lw=1.6, alpha=0.55, marker="o", ms=5,
                label="стекинг, за 120 мин")
        ax.plot(x(k1), k1.b1, "-", color=COLORS["b1"], lw=2, marker="o", ms=7, label="B1, за 30 мин")
        ax.plot(x(k1), k1["stack"], "-", color=COLORS["stack"], lw=2.6, marker="o", ms=7, label="стекинг, за 30 мин")
        ax.plot(x(fact), fact.y, "-", color=FACT, lw=2.2, marker="s", ms=6, label="факт", zorder=4)
        title = "Вся линия (сумма вестибюлей)" if place == "вся линия" else f"Станция «{names.get(place, place)}»"
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.set_ylabel("входов за 30 минут")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("начало получасового слота, местное время")
    fig.suptitle(f"Демо: вечер {day:%d.%m.%Y}, прогнозы каждые 30 минут", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    _save(fig, path_name or f"spb_17_demo_{day:%d%m}.png")
    plt.close(fig)


def make_all() -> None:
    wape_horizon()
    weights()
    if (OUT / "september_run.json").exists():
        demo()
