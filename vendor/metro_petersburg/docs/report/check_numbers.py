"""Сверка ключевых чисел отчёта с источниками.

Каждое значение считается из reports/backtest/*.csv|json, data/features_screening.csv или logs/clean_spb.log,
форматируется так же, как в тексте («5,51 %», «+1,89 [+1,57; +2,24]»), и ищется в docs/report/report.md рядом
с ключевым словом контекста (в пределах ±400 символов). При любом расхождении — код возврата 1.

Запуск из корня: python docs/report/check_numbers.py
"""
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "reports" / "backtest"
REPORT = ROOT / "docs" / "report" / "report.md"
WINDOW = 400


def _r(x: float, nd: int) -> str:
    """Арифметическое округление (0,195 → 0,20) десятичной записи числа, а не его двоичного приближения."""
    q = Decimal(1).scaleb(-nd)
    d = x if isinstance(x, Decimal) else Decimal(repr(float(x)))
    return str(d.quantize(q, rounding=ROUND_HALF_UP)).replace(".", ",")


def pct(x: float, nd: int = 2) -> str:
    return _r(Decimal(repr(float(x))) * 100, nd) + " %"


def pp(x: float) -> str:
    v = Decimal(repr(float(x))) * 100
    return ("−" if x < 0 else "+") + _r(abs(v), 2)


def ci(d: float, lo: float, hi: float) -> str:
    return f"{pp(d)} [{pp(lo)}; {pp(hi)}]"


def f2(x: float) -> str:
    return _r(x, 2)


def num(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def wape(res: pd.DataFrame, model: str, sl: str = "все часы", h: int | None = None, col: str = "wape") -> float:
    r = res[(res.model == model) & (res.fold == "все") & (res.slice == sl)]
    return float(r[r.h == h][col].iloc[0]) if h else float(r[col].mean())


def build_checks() -> list[tuple[str, str, str]]:
    """(что проверяем, ожидаемая строка, слово контекста рядом)."""
    cfg = json.loads((B / "model_final.json").read_text(encoding="utf-8"))
    final = cfg["step_model"]
    best = cfg["baseline"]
    res = pd.read_csv(B / "model_results.csv")
    abl = pd.read_csv(B / "model_ablation.csv").set_index("step")
    loo = pd.read_csv(B / "model_loo.csv").set_index("group")
    fres = pd.read_csv(B / "final_results.csv")
    fcmp = pd.read_csv(B / "final_compare.csv").set_index("period")
    fcal = pd.read_csv(B / "final_calibration.csv").set_index("h")
    mcal = pd.read_csv(B / "model_calibration.csv")
    flags = pd.read_csv(B / "model_flags.csv")
    fflags = pd.read_csv(B / "final_flags.csv")
    scr = pd.read_csv(ROOT / "data" / "features_screening.csv")
    log = (ROOT / "logs" / "clean_spb.log").read_text(encoding="utf-8")

    c = []
    # --- бэктест: финальная модель и лучший бейзлайн
    c += [("бэктест: WAPE модели", pct(wape(res, final)), "финальн"),
          ("бэктест: WAPE модели t+1", pct(wape(res, final, h=1)), "t+1"),
          ("бэктест: WAPE модели t+2", pct(wape(res, final, h=2)), "t+2"),
          ("бэктест: пики модели", pct(wape(res, final, "пики 07–09, 17–19")), "пик"),
          ("бэктест: аномальные часы модели", pct(wape(res, final, "аномальные часы")), "аномальн"),
          ("бэктест: покрытие модели", pct(wape(res, final, col="coverage"), 1), "Покрыти"),
          ("бэктест: WAPE лучшего бейзлайна", pct(wape(res, best)), "бейзлайн"),
          ("бэктест: бейзлайн t+1 / t+2", f"{pct(wape(res, best, h=1))} / {pct(wape(res, best, h=2))}", "уровень × r(t)"),
          ("бэктест: аномальные часы бейзлайна", pct(wape(res, best, "аномальные часы")), "аномальн"),
          ("бэктест: покрытие бейзлайна", pct(wape(res, best, col="coverage"), 1), "уровень × r(t)")]
    n_test = res[(res.model == final) & (res.fold == "все") & (res.slice == "все часы")].n.sum()
    c.append(("бэктест: число тестовых строк", num(int(n_test)), "строк"))
    for m, label in (("seasonal_naive", "Сезонный наивный"), ("b4", "Норма b"), ("b8", "8 недел"),
                     ("b4_level", "уровень"), ("b4_x_r", "b × r(t)"), ("b4_x_ewm2", "сглаженное r")):
        c.append((f"лестница: {label}", pct(wape(res, m)), label))
    # --- абляция и цель
    dbl = abl.loc["+ погода"]
    c.append(("ΔWAPE модели к бейзлайну", ci(dbl.dbl_dwape, dbl.dbl_lo, dbl.dbl_hi), "бейзлайн"))
    for step in ("+ ядро динамики", "+ кросс", "+ скорость и ускорение", "+ вчера", "+ погода", "+ особые дни"):
        r = abl.loc[step]
        c.append((f"абляция {step}: ΔWAPE", ci(r.dprev_dwape, r.dprev_lo, r.dprev_hi), step.lstrip("+ ").split()[0]))
        c.append((f"абляция {step}: WAPE", pct(r.wape), step.lstrip("+ ").split()[0]))
    sd = abl.loc["+ особые дни"]
    c.append(("особые дни на своих днях", f"{pp(sd.dgroup_dwape)} п. п. [{f2(sd.dgroup_lo * 100)}; {pp(sd.dgroup_hi)}]",
              "8.05"))
    c.append(("покрытие до поправки, бэктест", pct(abl.loc["+ погода"].coverage_raw, 1), "поправк"))
    rl = cfg["ratio_vs_log1p"]
    c.append(("цель-отношение против log1p", ci(rl["dwape"], rl["lo"], rl["hi"]), "log1p"))
    c.append(("WAPE цели log1p", pct(abl.loc["цель log1p(y), полный набор"].wape), "log1p"))
    # --- «минус одна группа»
    for g in loo.index:
        r = loo.loc[g]
        c.append((f"минус «{g}»: потеря", ci(r.loss_dwape, r.loss_lo, r.loss_hi), g))
    # --- смесь
    t = cfg["blend"]["test"]
    c += [("смесь: WAPE", pct(t["wape_blend"]), "Смесь"),
          ("смесь: аномальные часы", pct(t["wape_anomaly_blend"]), "аномальн"),
          ("смесь: ΔWAPE все часы, ДИ", f"[{pp(t['all']['lo'])}; {pp(t['all']['hi'])}]", "0,064"),
          ("смесь: ΔWAPE аномальные, ДИ", ci(t["anomaly"]["dwape"], t["anomaly"]["lo"], t["anomaly"]["hi"]), "0,046")]
    # --- калибровка по фолдам
    cv = mcal[mcal.model == final].groupby("fold").test_cov.mean()
    c.append(("покрытие по фолдам, диапазон", f"{pct(cv.min(), 1)[:-2]}–{pct(cv.max(), 1)}", "фолдам"))
    # --- флаг
    fl = flags[(flags.model == "модель") & (flags.level == "вестибюли") & (flags.h == "оба") & (flags.rule == "порог p80")].iloc[0]
    c.append(("флаг p80, бэктест: P / R / F1", f"{f2(fl.precision)} / {f2(fl.recall)} / {f2(fl.f1)}", "p80"))
    ff = fflags[(fflags.model == "lgbm_final") & (fflags.level == "вестибюли") & (fflags.h.astype(str) == "оба")
                & (fflags.rule == "порог p80")].iloc[0]
    c.append(("флаг p80, сентябрь: P / R / F1", f"precision {f2(ff.precision)}, recall {f2(ff.recall)}, F1 {f2(ff.f1)}",
              "Модель"))
    # --- финальный тест
    fm = "lgbm_final"
    c += [("финал: WAPE модели", pct(wape(fres, fm)), "LightGBM"),
          ("финал: модель t+1 / t+2", f"{pct(wape(fres, fm, h=1))} / {pct(wape(fres, fm, h=2))}", "LightGBM"),
          ("финал: пики модели", pct(wape(fres, fm, "пики 07–09, 17–19")), "пик"),
          ("финал: аномальные часы модели", pct(wape(fres, fm, "аномальные часы")), "аномальн"),
          ("финал: покрытие модели", pct(wape(fres, fm, col="coverage"), 1), "покры"),
          ("финал: WAPE бейзлайна", pct(wape(fres, best)), "бейзлайн"),
          ("финал: сезонный наивный", pct(wape(fres, "seasonal_naive")), "Сезонный"),
          ("финал: норма b", pct(wape(fres, "b4")), "Норма b")]
    for period, ctx in (("весь тест", "сентябр"), ("01.09–14.09", "01–14.09"), ("15.09–29.09", "15–29.09"),
                        ("аномальные часы", "аномальн")):
        r = fcmp.loc[period]
        c.append((f"финал ΔWAPE: {period}", ci(r.dwape, r.lo, r.hi), ctx))
        c.append((f"финал WAPE модели: {period}", pct(r.wape_alt), ctx))
        c.append((f"финал WAPE бейзлайна: {period}", pct(r.wape_base), ctx))
    c += [("финал: покрытие t+1 / t+2", f"{pct(fcal.loc[1].test_cov, 1)} на t+1 и {pct(fcal.loc[2].test_cov, 1)} на t+2", "Калибровка"),
          ("финал: покрытие до поправки", f"{pct(fcal.loc[1].test_cov_raw, 1)} и {pct(fcal.loc[2].test_cov_raw, 1)}", "До поправки")]
    n_sep = fres[(fres.model == fm) & (fres.fold == "все") & (fres.slice == "все часы")].n.sum()
    c.append(("финал: число строк", num(int(n_sep)), "строк"))
    # --- отбор: share_dev
    for h in (1, 2):
        r = scr[(scr.candidate == "share_dev") & (scr.h == h)].iloc[0]
        c.append((f"share_dev, t+{h}", ci(r.dwape, r.dwape_lo, r.dwape_hi), "share_dev" if h == 1 else "t+2"))
    # --- сравнение моделей (reports/compare)
    cmp_dir = ROOT / "reports" / "compare"
    for fname, period in (("summary.csv", "май–август"), ("september.csv", "сентябрь")):
        sm = pd.read_csv(cmp_dir / fname).set_index("model")
        for m, label in (("linear", "Линейная КР"), ("catboost", "CatBoost"), ("xgboost", "XGBoost"),
                         ("lgbm_catboost", "LightGBM + CatBoost")):
            r = sm.loc[m]
            c.append((f"сравнение, {period}: {label}, WAPE и ΔWAPE",
                      f"{pct(r.wape)} | {pct(r.wape_h1)} / {pct(r.wape_h2)}", label))
            c.append((f"сравнение, {period}: {label}, ΔWAPE к LightGBM", ci(r.d_dwape, r.d_lo, r.d_hi), label))
        c.append((f"сравнение, {period}: LightGBM", pct(sm.loc["lgbm"].wape), "LightGBM (заморож.)"))
    sm = pd.read_csv(cmp_dir / "summary.csv").set_index("model")
    c.append(("сравнение: ансамбль в аномальных часах", ci(sm.loc["lgbm_catboost"].danom_dwape,
              sm.loc["lgbm_catboost"].danom_lo, sm.loc["lgbm_catboost"].danom_hi), "аномальных"))
    # --- 15-минутные данные (reports/intrahour, logs/clean_spb_15min.log)
    c += intrahour_checks()
    # --- прогноз каждые 30 минут: стекинг (reports/stack)
    c += stack_checks()
    # --- журнал очистки
    rows = int(re.search(r"строк: ([\d,]+);", log).group(1).replace(",", ""))
    total = int(re.search(r"сумма входов: ([\d,]+)", log).group(1).replace(",", ""))
    c += [("очистка: строк", num(rows), "строк"), ("очистка: сумма входов", num(total), "вход")]
    for label, ctx in (("без флага", "без флага"), ("метро закрыто (01–04)", "метро закрыто"),
                       ("вестибюль закрыт: режим", "по режиму"), ("инцидент", "инцидент")):
        n = int(re.search(rf"{re.escape(label)}\s+(\d+)", log).group(1))
        c.append((f"флаг «{label}»: строк", num(n), ctx))
    return c


def intrahour_checks() -> list[tuple[str, str, str]]:
    """Числа раздела «Данные по 15 минут и ранний сигнал»."""
    d = ROOT / "reports" / "intrahour"
    log = (ROOT / "logs" / "clean_spb_15min.log").read_text(encoding="utf-8")
    summ = pd.read_csv(d / "reconciliation_summary.csv")
    line = summ[summ.slice == "вся линия"].iloc[0]
    month = summ[summ.slice == "месяц"]
    ves = summ[summ.slice == "вестибюль"].set_index("key")
    fc = pd.read_csv(d / "signal_forecast.csv")
    dl = pd.read_csv(d / "signal_forecast_delta.csv")
    det = pd.read_csv(d / "detection_by_minute.csv")
    m0 = pd.read_csv(d / "detection_minute0.csv").set_index("period")
    cs = pd.read_csv(d / "detection_c.csv")
    cells = pd.read_csv(d / "profile_cells.csv")
    sel_p, sep_p = "фев + май + июль", "сентябрь"
    w = lambda df, col: (df[col] * df.total).sum() / df.total.sum()

    c = [("15 мин: строк", num(int(re.search(r"строк: ([\d,]+) =", log).group(1).replace(",", ""))), "строк"),
         ("15 мин: входов", num(int(re.search(r"сумма входов: ([\d,]+)", log).group(1).replace(",", ""))), "входов"),
         ("15 мин: точное совпадение", pct(line.exact_share, 1), "Точное совпадение"),
         ("15 мин: превышение источника", pct(line.excess), "выше часового"),
         ("15 мин: превышение по месяцам", f"+{pct(month.excess.min())[:-2]}…+{pct(month.excess.max())}", "по месяцам"),
         ("15 мин: превышение, мин. вестибюль", pct(ves.loc["vosstaniya_2"].excess), "Восстания-2"),
         ("15 мин: превышение, макс. вестибюль", pct(ves.loc["leninsky_prospekt_2"].excess), "Ленинский пр.-2"),
         ("15 мин: выбросов", f"{int(line.mismatch_hours) + 2} ч", "Выбросы")]
    sel = lambda p: fc[fc.period == p].set_index("label")
    for label, ctx in (("минута 0: норма b4", "норма b4"), ("минута 0: b × уровень × r(t)", "уровень × r(t)"),
                       ("15 мин: b4 · r_1", "r_1"), ("30 мин: b4 · r_2", "r_2"), ("45 мин: b4 · r_3", "r_3"),
                       ("60 мин: b4 · r_4", "весь час")):
        c.append((f"15 мин: WAPE «{label}»", f"{pct(sel(sel_p).loc[label].wape)} | {pct(sel(sep_p).loc[label].wape)}", ctx))
    c.append(("15 мин: WAPE модели t+1", pct(sel(sel_p).loc["минута 0: модель t+1 (q50)"].wape), "модель t+1"))
    dm = dl[dl.b == "f_model"].set_index("a")
    c += [("15 мин: ΔWAPE r_1 к модели", ci(dm.loc["r1"].delta_wape, dm.loc["r1"].lo, dm.loc["r1"].hi), "модел"),
          ("15 мин: ΔWAPE r_2 к модели", ci(dm.loc["r2"].delta_wape, dm.loc["r2"].lo, dm.loc["r2"].hi), "модел"),
          ("15 мин: WAPE r_2 и модели на тех же строках",
           f"{pct(dm.loc['r2'].wape_a)} против {pct(dm.loc['r2'].wape_b)}", "тех же строках"),
          ("15 мин: c_k", " / ".join(f2(x) for x in cs.c_f1), "c =")]
    for p_, ctx_r, ctx_p in ((sel_p, "Найдено аномальных", "Точность поднятых"), (sep_p, "Найдено, сентябрь",
                                                                                  "Точность, сентябрь")):
        dd = det[det.period == p_]
        c.append((f"15 мин: найдено, {p_}", " | ".join([pct(m0.loc[p_].recall, 0), *(pct(x, 0) for x in dd.recall)]), ctx_r))
        c.append((f"15 мин: точность, {p_}",
                  " | ".join([pct(m0.loc[p_].precision, 0), *(pct(x, 0) for x in dd.precision)]), ctx_p))
    wd = cells[cells.day3 == "рабочий"]
    work = wd[wd.hour.isin(range(7, 23))]
    hub = cells[cells.hour.isin([*range(6, 24), 0])]
    c += [("15 мин: U днём", pct(w(work, "U"), 1), "07–22"),
          ("15 мин: U вокзалов, рабочий", f"{pct(w(hub[hub.hub & (hub.day3 == 'рабочий')], 'U'), 1)} против "
                                          f"{pct(w(hub[~hub.hub & (hub.day3 == 'рабочий')], 'U'), 1)}", "Профиль чуть")]
    return c


def stack_checks() -> list[tuple[str, str, str]]:
    """Числа раздела «Прогноз каждые 30 минут: стекинг»."""
    d = ROOT / "reports" / "stack"
    cm, cd = pd.read_csv(d / "cross_metrics.csv"), pd.read_csv(d / "cross_delta.csv")
    sm, sd = pd.read_csv(d / "september_metrics.csv"), pd.read_csv(d / "september_delta.csv")
    w = pd.read_csv(d / "weights.csv")
    sel = pd.read_csv(d / "selection.csv")
    run = json.loads((d / "september_run.json").read_text(encoding="utf-8"))
    demo = pd.read_csv(d / f"demo_{pd.Timestamp(run['demo_day']):%d%m}.csv")
    choice = pd.read_csv(d / "demo_choice.csv")
    cross, may, jul, sep = ("май + июль (вне выборки)", "май (стекинг обучен на июле)",
                            "июль (стекинг обучен на мае)", "сентябрь")
    labels = {"b1": "B1: часовая LightGBM", "b2": "B2: персистентность", "b3": "B3: GRU", "mean": "Среднее B1–B3",
              "stack": "Стекинг"}

    def m(df, period, model, h="все", col="wape", sl="все слоты"):
        r = df[(df.period == period) & (df.model == model) & (df.horizon.astype(str) == h) & (df.slice == sl)]
        return float(r[col].iloc[0])

    def dl(df, period, h="все", sl="все слоты"):
        r = df[(df.period == period) & (df.vs == "B1") & (df.horizon.astype(str) == h) & (df.slice == sl)].iloc[0]
        return ci(r.dwape, r.lo, r.hi)

    c = []
    for model in ("b1", "b2", "b3", "mean"):
        row = " | ".join(pct(m(cm, cross, model, str(h))) for h in (30, 60, 90, 120, "все"))
        c.append((f"стекинг: WAPE {model}, май + июль", f"{row} | {pct(m(cm, cross, model, col='coverage'), 1)}",
                  labels[model]))
    for h in (30, 60, 90, 120, "все"):
        c.append((f"стекинг: WAPE стекинга, {h}", pct(m(cm, cross, "stack", str(h))), "Стекинг"))
    c.append(("стекинг: покрытие", pct(m(cm, cross, "stack", col="coverage"), 1), "Стекинг"))
    c.append(("стекинг: строк май + июль", num(int(m(cm, cross, "stack", col="n"))), "185"))
    for period, df in ((cross, cd), (may, cd), (jul, cd), (sep, sd)):
        c.append((f"стекинг: ΔWAPE к B1, все, {period}", dl(df, period), "Все слоты"))
        c.append((f"стекинг: ΔWAPE к B1, 30 мин, {period}", dl(df, period, "30"), "Горизонт 30"))
        c.append((f"стекинг: ΔWAPE к B1, 120 мин, {period}", dl(df, period, "120"), "Горизонт 120"))
        c.append((f"стекинг: ΔWAPE к B1, аномальные, {period}", dl(df, period, sl="аномальные часы"), "Аномальные"))
    c += [("стекинг: сентябрь, стекинг и B1",
           f"{pct(m(sm, sep, 'stack'))} против {pct(m(sm, sep, 'b1'))}", "Сентябрь"),
          ("стекинг: выбранный вариант весов", "по горизонту" if sel[sel.chosen].variant.iloc[0] == "V1" else "?",
           "простейший")]
    wq = w[(w.fit == "май + июль") & (w.level == "горизонт") & (w["quantile"] == "q50")].sort_values("cell")
    c.append(("стекинг: веса B1 в q50", " / ".join(f2(x) for x in wq.w_b1), "q50"))
    line = demo[(demo.place == "вся линия") & (demo.k == 1)]
    wp = lambda col: (line[col] - line.y).abs().sum() / line.y.sum()
    c += [("стекинг: демо, линия", f"{pct(wp('stack'))}, B1 — на {pct(wp('b1'))}, норма — на {pct(wp('n'))}",
           "По линии"),
          ("стекинг: демо, отклонение вечера", pct(choice.dev.abs().max(), 1), "07.09")]
    return c


def main() -> int:
    text = REPORT.read_text(encoding="utf-8").replace(" ", " ")
    text = re.sub(r"\s+", " ", text)              # переносы строк в исходнике — как пробелы
    fails = 0
    for what, value, ctx in build_checks():
        variants = {value, value.replace(" [", " п. п. [", 1)}
        pos = [m.start() for v in variants for m in re.finditer(re.escape(v), text)]
        near = any(ctx.lower() in text[max(0, p - WINDOW):p + WINDOW].lower() for p in pos)
        status = "OK" if near else ("НЕТ РЯДОМ С КОНТЕКСТОМ" if pos else "НЕТ В ОТЧЁТЕ")
        if status != "OK":
            fails += 1
        print(f"{'OK  ' if status == 'OK' else 'FAIL'} {what}: «{value}»" + ("" if status == "OK" else f" — {status} «{ctx}»"))
    total = len(build_checks())
    print(f"\nпроверок: {total}, расхождений: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
