"""Графики для презентации (этап 5): лестница WAPE и демо-кейс 31.08. Важность признаков — model.gain_figure.

Запуск: python -m src.figures → reports/figures/spb_14_wape_ladder.png, spb_15_demo_0831.png.
Палитра — dataviz (проверена валидатором): синий — модель и бэктест, оранжевый — бейзлайн и сентябрь,
тёмно-серый — факт; цвет закреплён за сущностью.
"""
import json

import numpy as np
import pandas as pd

from src import backtest as BT
from src import config, eda
from src import eda_spb as E
from src import model as M

BLUE, ORANGE, FACT, NORM, SPAN = "#2a78d6", "#eb6834", "#52514e", "#898781", "#ecebe6"
LADDER = [("seasonal_naive", "Сезонный наивный (неделю назад)"), ("b4", "Норма b (4 недели)"),
          ("b4_level", "b × уровень 7 дней"), ("b4_x_r", "b × r(t)"), ("b4_level_x_r", "b × уровень × r(t)"),
          ("lgbm_base", "LightGBM: база"), ("lgbm_final", "LightGBM: финальная")]
DEMO_STATION, DEMO_DAY, DEMO_HOURS = "devyatkino", "2026-08-31", range(5, 15)
INCIDENT = ("2026-08-31 07:50", "2026-08-31 10:31")


def _pct(x: float) -> str:
    return f"{x * 100:.2f} %".replace(".", ",")


def _wape_all(res: pd.DataFrame) -> pd.Series:
    r = res[(res.fold == "все") & (res.slice == "все часы")]
    return r.groupby("model").wape.mean()


def ladder(path=config.FIGURES / "spb_14_wape_ladder.png") -> pd.DataFrame:
    """WAPE, все часы (среднее по t+1 и t+2): бэктест май–август и финальный тест на сентябре."""
    import matplotlib.pyplot as plt

    cfg = M.load_config()
    back = _wape_all(pd.read_csv(M.OUT / "model_results.csv")).rename({cfg["step_model"]: "lgbm_final"})
    sep_path = M.OUT / "final_results.csv"
    sep = _wape_all(pd.read_csv(sep_path)) if sep_path.exists() else pd.Series(dtype=float)
    df = pd.DataFrame([{"model": k, "label": lab, "back": back.get(k, np.nan), "sep": sep.get(k, np.nan)}
                       for k, lab in LADDER])
    eda.style()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(df))[::-1]
    hgt = 0.36
    for col, color, label, off in (("back", BLUE, "май–август, бэктест (4 фолда)", hgt / 2),
                                   ("sep", ORANGE, "1–29 сентября, финальный тест", -hgt / 2)):
        vals = df[col].to_numpy() * 100
        ax.barh(y + off, vals, height=hgt * 0.92, color=color, label=label)
        for yy, v in zip(y + off, vals):
            if not np.isnan(v):
                ax.text(v + 0.12, yy, f"{v:.2f} %".replace(".", ","), va="center", fontsize=10, color=eda.INK2)
    ax.set_yticks(y, df.label)
    for yy, (b_, s_) in zip(y, df[["back", "sep"]].to_numpy()):
        if np.isnan(s_) and not np.isnan(b_):
            ax.text(0.15, yy - hgt / 2, "в сентябре не запускалась (заморожена только финальная)", va="center",
                    fontsize=10, color=eda.MUTED)
    for t, m in zip(ax.get_yticklabels(), df.model):
        if m == "lgbm_final":
            t.set_fontweight("bold")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, np.nanmax(df[["back", "sep"]].to_numpy()) * 100 * 1.32)
    ax.set_xlabel("WAPE, все часы 05–00, среднее по горизонтам t+1 и t+2, %")
    ax.set_title("Лестница моделей: от сезонного наивного к LightGBM", loc="left")
    ax.legend(loc="lower right")
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return df


def _station(path, station: str) -> pd.DataFrame:
    recs = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    recs = recs[recs.station_id == station].copy()
    recs["ts"] = pd.to_datetime(recs.ts).dt.tz_localize(None)
    return recs


def demo_0831(path=config.FIGURES / "spb_15_demo_0831.png") -> pd.DataFrame:
    """Девяткино 31.08: факт, норма, бейзлайн и модель (q50 и q10–q90) по часам; t+1 и t+2 — две панели."""
    import matplotlib.pyplot as plt

    model, base = _station(M.AUG_OUT, DEMO_STATION), _station(BT.AUG_OUT, DEMO_STATION)
    d = E.load()
    g, ves = d.grid, d.vestibules
    cols = (ves.station_id == DEMO_STATION).to_numpy()
    fact = pd.Series(np.where(g.closed, np.nan, g.entries)[:, cols].sum(1), index=g.local)
    day = pd.Timestamp(DEMO_DAY)
    hours = [day + pd.Timedelta(hours=h) for h in DEMO_HOURS]

    eda.style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    rows = []
    for ax, h in zip(axes, (60, 120)):
        mm = model[(model.horizon_min == h) & model.ts.isin(hours)].set_index("ts").reindex(hours)
        bb = base[(base.horizon_min == h) & base.ts.isin(hours)].set_index("ts").reindex(hours)
        x = np.array([t.hour for t in hours])
        ax.axvspan(7 + 50 / 60, 10 + 31 / 60, color=SPAN, zorder=0)
        ax.text(7 + 50 / 60 + 0.08, 0.03, "инцидент\n07:50–10:31", transform=ax.get_xaxis_transform(), va="bottom",
                fontsize=10, color=eda.INK2)
        ax.fill_between(x, mm.q10, mm.q90, color=BLUE, alpha=0.12, linewidth=0, label="модель, q10–q90")
        ax.plot(x, mm.baseline, color=NORM, linewidth=1.2, label="норма b")
        ax.plot(x, bb.q50, color=ORANGE, marker="o", markersize=5, label="бейзлайн «b × уровень × r(t)»")
        ax.plot(x, mm.q50, color=BLUE, marker="o", markersize=5, label="модель LightGBM, q50")
        ax.plot(x, fact.reindex(hours).to_numpy(), color=FACT, marker="o", markersize=6, label="факт")
        ax.set_xticks(x, [f"{v:02d}" for v in x])
        ax.set_xlabel("час 31.08 (слот прогноза)")
        ax.set_title(f"горизонт {h} мин (t+{h // 60})", loc="left", fontsize=13)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
        rows.append(pd.DataFrame({"hour": x, "h": h // 60, "fact": fact.reindex(hours).to_numpy(),
                                  "norm": mm.baseline.to_numpy(), "baseline_q50": bb.q50.to_numpy(),
                                  "model_q10": mm.q10.to_numpy(), "model_q50": mm.q50.to_numpy(),
                                  "model_q90": mm.q90.to_numpy()}))
    axes[0].set_ylabel("входов в час, станция Девяткино")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(k) for k in ("факт", "модель LightGBM, q50", "модель, q10–q90", "бейзлайн «b × уровень × r(t)»",
                                       "норма b")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Демо-кейс 31.08: провал из-за инцидента и откат спроса, Девяткино\n"
                 "прогнозы model_aug.json и baseline_aug.json: обучение до 24.07, вне выборки", x=0.01, ha="left",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    print(ladder().round(4).to_string())
    print(demo_0831().round(0).to_string())


if __name__ == "__main__":
    main()
