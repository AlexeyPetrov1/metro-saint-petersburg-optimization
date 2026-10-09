"""Рисунки только для отчёта (docs/report/figures/): схема системы, фолды, пересмотренный отбор признаков, абляция,
калибровка интервала, правила флага. Всё — по готовым CSV/JSON в reports/backtest и data/; модель не обучается.

Запуск из корня: python docs/report/make_figures.py
Палитра — dataviz (первые три слота проверены валидатором на всех парах): синий, оранжевый, бирюзовый;
серый — нейтральные элементы (отброшено, «до поправки», зазор).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import backtest as BT  # noqa: E402
from src import eda  # noqa: E402

OUT = ROOT / "docs" / "report" / "figures"
BT_DIR = ROOT / "reports" / "backtest"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHT, INK2 = "#898781", "#d9d8d2", eda.INK2


def _plt():
    import matplotlib.pyplot as plt
    eda.style()
    return plt


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _comma(axis, nd: int = 1) -> None:
    """Подписи делений с десятичной запятой."""
    axis.set_major_formatter(lambda v, _: f"{v:.{nd}f}".replace(".", ","))


def _pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f} %".replace(".", ",")


# --- 1. Схема системы ---------------------------------------------------------------
def system() -> None:
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.2, 6.0)
    ax.axis("off")
    W, H = 2.75, 2.3
    xs = [0.1, 3.45, 6.8, 10.15]
    top = 3.55
    boxes = [
        (xs[0], top, "Данные", ["почасовые входы", "23 вестибюлей (АСКОП М),", "календарь, прогноз", "погоды, события"], False),
        (xs[1], top, "ML-прогноз", ["входы на 18 станций,", "t+1 и t+2: q10 / q50 / q90,", "норма, флаг аномалии,", "топ-3 причины"], True),
        (xs[2], top, "Загрузка перегонов", ["симулятор: поток", "по перегонам против", "провозной способности"], False),
        (xs[3], top, "Рекомендации", ["парность, вывод", "резерва составов,", "экономика"], False),
        (xs[1], 0.35, "LLM-слой", ["объяснение прогноза", "и рекомендации", "диспетчеру словами"], False),
        (xs[2], 0.35, "Дашборд", ["прогноз, интервалы,", "флаги, причины; режим", "повтора на любую дату"], False),
    ]
    for x, y, title, body, hl in boxes:
        w = W if title != "Дашборд" else xs[3] + W - xs[2]
        ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#e3eefb" if hl else "#f3f2ee", ec=BLUE if hl else GRAY, lw=2.2 if hl else 1))
        ax.text(x + w / 2, y + H - 0.28, title, ha="center", va="top", fontsize=12, fontweight="bold", color=eda.INK)
        ax.text(x + w / 2, y + H - 0.8, "\n".join(body), ha="center", va="top", fontsize=10, color=INK2,
                linespacing=1.45)
    ax.text(xs[1] + W / 2, top - 0.08, "этот отчёт", ha="center", va="top", fontsize=9.5, color=BLUE)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK2))
    mid = top + H / 2
    for a, b in zip(xs[:-1], xs[1:]):
        arrow((a + W + 0.03, mid), (b - 0.03, mid))
    ax.text((xs[1] + W + xs[2]) / 2, mid + 0.12, "контракт", ha="center", va="bottom", fontsize=9, color=INK2)
    arrow((xs[1] + W / 2, top - 0.35), (xs[1] + W / 2, 0.35 + H + 0.03))
    ax.text(xs[1] + W / 2 + 0.1, 3.05, "причины", ha="left", va="center", fontsize=9, color=INK2)
    arrow((xs[2] + W / 2, top - 0.03), (xs[2] + W / 2, 0.35 + H + 0.03))
    arrow((xs[3] + W / 2, top - 0.03), (xs[3] + W / 2, 0.35 + H + 0.03))
    arrow((xs[1] + W + 0.03, 0.35 + H / 2), (xs[2] - 0.03, 0.35 + H / 2))
    _save(fig, "fig_system")


# --- 2. Фолды -------------------------------------------------------------------------
def folds() -> None:
    plt = _plt()
    fs = BT.folds(final=True)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    start = BT.TRAIN_START
    for i, f in enumerate(fs[::-1]):
        y = i
        train_end = f.train_end
        ax.barh(y, (train_end - start).days, left=start, height=0.5, color=BLUE)
        ax.barh(y, (f.test_start - train_end).days, left=train_end, height=0.5, color=LIGHT)
        ax.barh(y, (f.test_end - f.test_start).days + 1, left=f.test_start, height=0.5,
                color=ORANGE if f.name != "final" else AQUA)
        lab = "финальный тест\n(один раз)" if f.name == "final" else f"фолд «{f.name}»"
        ax.text(start - pd.Timedelta(days=3), y, lab, ha="right", va="center", fontsize=11, color=eda.INK)
        ax.text(f.test_start + (f.test_end - f.test_start) / 2, y, f"{f.test_start:%d.%m}–{f.test_end:%d.%m}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.axvspan(BT.SEPTEMBER, BT.FINAL_END + pd.Timedelta(days=1), color="#f3f2ee", zorder=0)
    ax.text(BT.SEPTEMBER + pd.Timedelta(days=14), len(fs) - 0.35, "сентябрь закрыт\nдо заморозки",
            ha="center", va="bottom", fontsize=10, color=INK2)
    ax.set_yticks([])
    ax.set_xlim(start - pd.Timedelta(days=45), BT.FINAL_END + pd.Timedelta(days=4))
    ax.set_ylim(-0.6, len(fs) + 0.3)
    ax.xaxis.set_major_formatter(__import__("matplotlib.dates", fromlist=["DateFormatter"]).DateFormatter("%d.%m"))
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="обучение (с 09.02)"), Patch(color=LIGHT, label="зазор 7 суток"),
                       Patch(color=ORANGE, label="тест фолда"), Patch(color=AQUA, label="финальный тест")],
              loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=4)
    ax.set_title("Бэктест: расширяющееся окно, зазор 7 суток перед каждым тестом", loc="left")
    _save(fig, "fig_folds")


# --- 3. Пересмотренный отбор признаков ------------------------------------------------
def screening() -> None:
    plt = _plt()
    s = pd.read_csv(ROOT / "data" / "features_screening.csv")
    s = s[(s.h == 1) & s.dwape.notna()].sort_values("dwape")
    color = {"брать": BLUE, "проверить в абляции": ORANGE, "отбросить": GRAY}
    fig, ax = plt.subplots(figsize=(11.5, 0.27 * len(s) + 1.6))
    y = np.arange(len(s))
    for v, c in color.items():
        m = (s.verdict == v).to_numpy()
        ax.errorbar(s.dwape[m] * 100, y[m], xerr=[(s.dwape - s.dwape_lo)[m] * 100, (s.dwape_hi - s.dwape)[m] * 100],
                    fmt="o", color=c, ecolor=c, elinewidth=1.4, capsize=2, markersize=6, label=v)
    ax.axvline(0, color=eda.AXIS, lw=1)
    ax.set_yticks(y, [d if len(d) < 48 else d[:46] + "…" for d in s.description], fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ΔWAPE = WAPE(b) − WAPE(b · поправка), п. п., t+1; апрель–август; 95 % ДИ — блочный бутстреп по дням")
    _comma(ax.xaxis, 2)
    ax.set_title("Пересмотренный отбор: польза кандидата для прогноза и вердикт", loc="left")
    ax.legend(loc="lower right")
    _save(fig, "fig_screening")


# --- 4. Абляция -----------------------------------------------------------------------
def ablation() -> None:
    plt = _plt()
    a = pd.read_csv(BT_DIR / "model_ablation.csv")
    steps = a[~a.step.str.startswith("цель")].reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(steps))[::-1]
    kept = (steps.step == "база").to_numpy() | steps.kept.eq(True).to_numpy()
    labels = [st if k else f"{st} (отброшена)" for st, k in zip(steps.step, kept)]
    ax1.barh(y, steps.wape * 100, height=0.6, color=[BLUE if k else LIGHT for k in kept])
    for yy, v in zip(y, steps.wape):
        ax1.text(v * 100 + 0.05, yy, _pct(v, 2), va="center", fontsize=10, color=INK2)
    bl = pd.read_csv(BT_DIR / "baselines.csv")
    best = bl[(bl.model == "b4_level_x_r") & (bl.fold == "все") & (bl.slice == "все часы")].wape.mean()
    ax1.axvline(best * 100, color=ORANGE, lw=1.6)
    ax1.text(best * 100 + 0.04, -0.55, f"лучший\nбейзлайн\n{_pct(best, 2)}", color=INK2, fontsize=9.5, va="bottom")
    ax1.set_yticks(y, labels)
    ax1.set_xlim(4.5, 7.4)
    _comma(ax1.xaxis, 1)
    ax1.set_ylim(-0.6, len(steps) - 0.4)
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("WAPE, все часы, среднее по t+1 и t+2, %")
    ax1.set_title("WAPE после каждого шага", loc="left", fontsize=13)
    d = steps.iloc[1:]
    yd = np.arange(len(d))[::-1]
    kd = d.kept.astype(bool).to_numpy()
    for yy, (_, r), k in zip(yd, d.iterrows(), kd):
        ax2.errorbar(r.dprev_dwape * 100, yy, xerr=[[(r.dprev_dwape - r.dprev_lo) * 100], [(r.dprev_hi - r.dprev_dwape) * 100]],
                     fmt="o", color=BLUE if k else GRAY, capsize=3, markersize=7)
        ax2.text(r.dprev_hi * 100 + 0.03, yy, f"{_pct(r.dprev_dwape, 2).replace(' %', '')} п. п., "
                 f"{int(r.dprev_folds_pos)} из 4", va="center", fontsize=9.5, color=INK2)
    ax2.axvline(0, color=eda.AXIS, lw=1)
    ax2.set_yticks(yd, [st if k else f"{st} (отброшена)" for st, k in zip(d.step, kd)])
    ax2.set_xlim(-0.15, 1.35)
    _comma(ax2.xaxis, 1)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("ΔWAPE к предыдущему набору, п. п. (95 % ДИ)")
    ax2.set_title("Польза группы и фолды с Δ > 0", loc="left", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig_ablation")


# --- 5. Калибровка --------------------------------------------------------------------
def calibration() -> None:
    plt = _plt()
    c = pd.read_csv(BT_DIR / "model_calibration.csv")
    final = json.loads((BT_DIR / "model_final.json").read_text(encoding="utf-8"))
    c = c[c.model == final["step_model"]].groupby("fold", sort=False)[["test_cov_raw", "test_cov"]].mean()
    sep = pd.read_csv(BT_DIR / "final_calibration.csv")[["test_cov_raw", "test_cov"]].mean()
    c.loc["сентябрь\n(финальный тест)"] = sep
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.arange(len(c))
    for i, (name, r) in enumerate(c.iterrows()):
        ax.plot([i, i], [r.test_cov_raw * 100, r.test_cov * 100], color=LIGHT, lw=3, zorder=1)
        ax.scatter(i, r.test_cov_raw * 100, color=GRAY, s=60, zorder=2, label="до поправки" if i == 0 else None)
        ax.scatter(i, r.test_cov * 100, color=BLUE, s=70, zorder=3, label="после конформной поправки" if i == 0 else None)
        ax.text(i + 0.08, r.test_cov * 100, _pct(r.test_cov), va="center", fontsize=10, color=INK2)
        ax.text(i + 0.08, r.test_cov_raw * 100, _pct(r.test_cov_raw), va="center", fontsize=10, color=INK2)
    ax.axhline(80, color=ORANGE, lw=1.5)
    ax.text(len(c) - 0.6, 80.4, "цель 80 %", color=INK2, fontsize=10)
    ax.set_xticks(x, c.index)
    ax.set_xlim(-0.4, len(c) - 0.3)
    ax.set_ylabel("покрытие q10–q90 на тесте, %")
    _comma(ax.yaxis, 1)
    ax.set_title("Покрытие интервала q10–q90 до и после поправки (среднее по t+1 и t+2)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _save(fig, "fig_calibration")


# --- 6. Флаг аномалии -----------------------------------------------------------------
def flags() -> None:
    plt = _plt()
    f = pd.read_csv(BT_DIR / "model_flags.csv")
    f = f[(f.level == "вестибюли") & (f.h == "оба")]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    # смещения подписей (точки, ха): подобраны так, чтобы подписи не пересекались
    offs = {("модель", "порог p80"): (-8, 8, "right"), ("модель", "порог p90"): (8, 4, "left"),
            ("модель", "порог p95"): (8, -12, "left"), ("модель", "норма вне [q10; q90]"): (8, -20, "left"),
            ("бейзлайн", "порог p80"): (6, 8, "left"), ("бейзлайн", "порог p90"): (8, 4, "left"),
            ("бейзлайн", "порог p95"): (8, 4, "left"), ("бейзлайн", "норма вне [q10; q90]"): (-6, -18, "right")}
    for model, color, name in (("модель", BLUE, "модель LightGBM"), ("бейзлайн", ORANGE, "бейзлайн «b × уровень × r(t)»")):
        g = f[f.model == model]
        ax.scatter(g.recall, g.precision, color=color, s=70, label=name, zorder=3)
        for r in g.itertuples():
            dx, dy, ha = offs[(model, r.rule)]
            ax.annotate(r.rule + f" (F1 {r.f1:.2f})".replace(".", ","), (r.recall, r.precision), ha=ha,
                        textcoords="offset points", xytext=(dx, dy), fontsize=9.5, color=INK2)
    ax.axvline(0.6, color=GRAY, lw=1.2)
    ax.text(0.61, 0.03, "требование:\nrecall ≥ 0,6", fontsize=10, color=INK2, va="bottom")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    _comma(ax.xaxis, 1)
    _comma(ax.yaxis, 1)
    ax.set_xlabel("recall — доля аномальных часов, на которых флаг сработал")
    ax.set_ylabel("precision — доля срабатываний,\nгде час действительно аномален")
    ax.set_title("Правила флага аномалии на бэктесте (вестибюли, май–август)", loc="left")
    ax.legend(loc="upper right")
    _save(fig, "fig_flags")


# --- 7. Сравнение моделей -------------------------------------------------------------
def compare() -> None:
    """WAPE моделей на мае–августе (все и аномальные часы) и время обучения — по reports/compare/summary.csv."""
    path = ROOT / "reports" / "compare" / "summary.csv"
    if not path.exists():
        return
    plt = _plt()
    s = pd.read_csv(path)
    s = s.iloc[::-1].reset_index(drop=True)
    colors = [BLUE if m == "lgbm" else (ORANGE if v == "кандидат в следующую версию" else GRAY)
              for m, v in zip(s.model, s.verdict)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True, gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]})
    y = np.arange(len(s))
    for ax, col, title, nd in ((axes[0], "wape", "WAPE, все часы, %", 2), (axes[1], "wape_anomaly", "WAPE, аномальные часы, %", 1),
                               (axes[2], "seconds_per_fold", "обучение, с на фолд", 0)):
        vals = s[col].to_numpy() * (1 if col == "seconds_per_fold" else 100)
        ax.barh(y, vals, height=0.6, color=colors)
        for yy, v in zip(y, vals):
            ax.text(v * 1.01, yy, f"{v:.{nd}f}".replace(".", ","), va="center", fontsize=9.5, color=INK2)
        ax.set_xlim(0, vals.max() * 1.22)
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(axis="y", visible=False)
        _comma(ax.xaxis, 0)
    axes[0].set_yticks(y, s.label)
    fig.suptitle("Сравнение моделей на мае–августе (4 фолда): синим — замороженная LightGBM", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig_compare")


def main() -> None:
    for fn in (system, folds, screening, ablation, calibration, flags, compare):
        fn()
    print(f"рисунки отчёта: {', '.join(sorted(p.name for p in OUT.glob('fig_*.png')))}")


if __name__ == "__main__":
    main()
