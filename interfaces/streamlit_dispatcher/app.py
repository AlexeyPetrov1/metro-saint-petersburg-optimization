"""Streamlit dispatcher shell over the existing saved simulator artifacts."""

import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from interfaces.streamlit_dispatcher.adapter import Replay
from interfaces.streamlit_dispatcher.player import player_html

DEFAULT = ROOT / "data/processed/metro51_balanced_pilot_20261009_v2"
LABELS = {"reference": "Исходный график", "old": "Прежнее управление",
          "ensemble_old": "Ансамбль + прежний оптимизатор", "balanced": "Ансамбль + новое управление"}
st.set_page_config(page_title="Диспетчер: красная линия", layout="wide")
st.markdown('''<style>
[data-testid="stMainBlockContainer"]{padding-top:1.4rem}
[data-testid="stHeader"]{height:0;background:transparent}
[data-testid="stMetric"]{background:white;border:1px solid #E3E6EA;border-radius:10px;padding:12px}
[data-testid="stVerticalBlockBorderWrapper"]{background:white;border-radius:10px}
</style>''', unsafe_allow_html=True)
st.markdown('''<div style="font-size:26px;font-weight:700;color:#1B2430">Метро Петербурга</div>
<div style="display:flex;align-items:center;gap:12px;margin:8px 0 20px">
<span style="border-radius:50%;background:#D6083B;color:white;padding:4px 12px;font-weight:700">1</span>
<span style="font-size:18px;font-weight:600;color:#D6083B">Кировско-Выборгская линия</span></div>''',
            unsafe_allow_html=True)


@st.cache_resource(max_entries=2, show_spinner="Читаю журнал прогона…")
def load_replay(path, version):
    return Replay.from_arm(path)


def stamp(path):
    files = [path / "result.json", *(path / "controller").glob("*.json*"),
             *(path / "explanations").glob("explanation_*.json")]
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists())


with st.sidebar:
    st.caption("Интерфейс к сохранённым данным симулятора")
    campaign = Path(st.text_input("Папка эксперимента", os.environ.get("METRO_CAMPAIGN_DIR", str(DEFAULT)))).expanduser()
    if not campaign.is_absolute():
        campaign = ROOT / campaign
    arms = sorted(p.name for p in campaign.glob("*") if (p / "common_scenario.json.gz").exists())
    if not arms:
        st.error("В этой папке нет сохранённых вариантов эксперимента.")
        st.stop()
    arm = st.selectbox("Вариант", arms, index=arms.index("balanced") if "balanced" in arms else 0,
                       format_func=lambda a: LABELS.get(a, a), key="arm_choice")
    st.button("Обновить журналы", help="Перечитать новые файлы, если прогон работает отдельно.")
path = campaign / arm
if not (path / "result.json").exists() and not any((path / "controller").glob("resume_*.json.gz")):
    st.info("Прогон готовит данные. Пока нет записанного контрольного состояния; обновите журналы после его сохранения.")
    st.stop()
try:
    replay = load_replay(str(path), stamp(path))
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Не удалось прочитать сохранённый прогон: {exc}")
    st.stop()

with st.sidebar:
    if st.session_state.get("time_arm") != str(path):
        st.session_state["clock_second"] = min(7200, replay.end)
        st.session_state["time_arm"] = str(path)
    second = st.slider("Момент симуляции, секунды от начала", 0, replay.end,
                       step=1, key="clock_second", format="%d с")
    st.markdown(f"**{(replay.start + timedelta(seconds=second)):%d.%m.%Y %H:%M:%S} МСК**")
    st.caption(f"Записано до {(replay.start + timedelta(seconds=replay.end)):%H:%M:%S} МСК")
    if replay.partial:
        st.caption("Незавершённый прогон: доступна запись до последнего контрольного состояния.")
frame = replay.frame(second)
st.caption("Пассажиры и движение взяты из записи симулятора. Распределение поездок между станциями задано сценарием.")


def metrics():
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Входы через турникеты", "На платформе сейчас", "В вагонах сейчас", "Поездов на линии"],
                                 [frame["entered"], frame["waiting"], frame["onboard"], sum(t["on_line"] for t in frame["trains"])]):
        col.metric(label, f"{value:,}".replace(",", " "))


def forecast_page():
    metrics()
    forecast = frame["forecast"]
    if forecast is None:
        first = replay.forecasts[0]["issued_at"][11:16] if replay.forecasts else None
        st.info(f"К этому моменту прогноз ещё не опубликован. Первый выпуск: {first} МСК."
                if first else "В этом варианте эксперимента нет сохранённых выпусков прогноза.")
        return
    st.subheader("Прогноз входов на станции")
    st.caption(f"Модель: {forecast['model_version']} · выпуск {forecast['issued_at']} · обновление раз в 15 минут")
    if forecast["stale"]:
        st.warning("Срок последнего сохранённого прогноза закончился. Для текущего момента прогноз отсутствует.")
    sid = st.selectbox("Станция", replay.station_ids, format_func=replay.names.get, key="forecast_station")
    rows = [r for r in forecast["rows"] if r["station_id"] == sid]
    fig = go.Figure()
    tx = [datetime.fromisoformat(r["interval_start"]) for r in rows]
    for name, field, width, color, fill in [("q90", "q90", 0, "#0078C9", None),
                                          ("q10–q90", "q10", 0, "#0078C9", "tonexty"),
                                          ("Прогноз q50", "q50", 2.5, "#0078C9", None)]:
        fig.add_trace(go.Scatter(x=tx, y=[r[field] for r in rows], name=name,
                                mode="lines+markers" if field == "q50" else "lines",
                                line=dict(color=color, width=width), fill=fill,
                                fillcolor="rgba(0,120,201,.16)", showlegend=field != "q90"))
    history = replay.station_history(sid, second)
    fig.add_trace(go.Scatter(x=[datetime.fromisoformat(r["interval_start"]) for r in history],
                            y=[r["count"] for r in history], mode="lines+markers", name="Наблюдавшиеся входы",
                            line=dict(color="#6B7685", width=1.5)))
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=15,b=10), yaxis_title="Входы за 15 минут",
                      xaxis=dict(tickformat="%H:%M"), legend=dict(orientation="h"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Факт внутри текущих 15 минут ещё неполный. q10 и q90 взяты из модели; это не число людей на платформе.")
    if any(r["fallback"] for r in rows):
        st.caption("Для этой станции использована историческая замена, указанная в выпуске модели.")
    now = replay.start + timedelta(seconds=second)
    current = {r["station_id"]: r for r in forecast["rows"] if datetime.fromisoformat(r["interval_start"]) <= now < datetime.fromisoformat(r["interval_end"])}
    table = []
    for station in frame["stations"]:
        r = current.get(station["station_id"], {})
        table.append({"Станция": station["name"], "Прогноз входов / 15 мин": r.get("q50"),
                      "q10": r.get("q10"), "q90": r.get("q90"),
                      "Очередь на север": station["north"], "Очередь на юг": station["south"],
                      "Источник": "Историческая замена" if r.get("fallback") else "Ансамбль" if r else "Нет прогноза"})
    st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("Пассажиры в каждом вагоне"):
        st.dataframe([{"Состав": t["train_id"], "Место": t["location"], "На линии": t["on_line"],
                       **{f"Вагон {i+1}": count for i,count in enumerate(t["cars"])}}
                      for t in frame["trains"]], hide_index=True, width="stretch")


def simulation_page():
    st.caption("Пуск ускоряет просмотр записи в браузере. Прогноз, решение и пассажиры следуют за часами проигрывателя.")
    baseline = None
    ref = campaign / "reference"
    if arm != "reference" and (ref / "result.json").exists():
        baseline = load_replay(str(ref), stamp(ref))
    payload = replay.player_payload(baseline)
    payload["start_pos"] = min(second / 900, payload["last"])
    st.iframe(player_html(payload), height=2420)


def optimizer_page():
    metrics()
    st.subheader("Решение оптимизатора")
    decision = frame["decision"]
    if decision is None:
        st.info("К этому моменту оптимизатор ещё не опубликовал решение.")
        return
    selection = decision.get("selection", {})
    st.write(f"Момент: {decision.get('decision_time')} · действие: **{decision.get('applied_candidate_id', 'KEEP')}**")
    st.write(f"Статус проверки: `{selection.get('status', 'unknown')}`")
    if selection.get("release_authorized"):
        st.success("В сохранённом решении проверки разрешили исполнение.")
    else:
        st.info("Разрешение на исполнение отсутствует. KEEP может быть резервным действием.")
    candidates = decision.get("candidate_table", [])
    if candidates:
        if isinstance(candidates, dict):
            candidates = list(candidates.values())
        st.dataframe([{k: str(v) if isinstance(v, (dict,list)) else v for k,v in row.items()}
                      for row in candidates], hide_index=True, width="stretch")
    explanation = frame["explanation"]
    with st.container(border=True):
        st.subheader("Почему выбрано действие")
        if explanation:
            st.caption(f"Источник: {explanation.get('metadata', {}).get('source', 'сохранённый ответ')}")
            for section in ["action", "prediction", "rationale", "effects", "risks", "revisit"]:
                rows = explanation.get("sections", {}).get(section, [])
                if isinstance(rows, list):
                    for row in rows[:2]:
                        st.write(row.get("text", "")[:700])
            with st.expander("Полное объяснение и ссылки на факты"):
                st.json(explanation)
        else:
            st.info("Сохранённое объяснение отсутствует.")
    with st.expander("Полный пакет решения"):
        st.json(decision)


def data_page():
    st.subheader("Источники интерфейса")
    st.write(f"Эксперимент: `{campaign}`")
    st.write(f"Вариант: `{arm}` · {'контрольное состояние' if replay.partial else 'завершённая запись'}")
    st.dataframe([{"Данные": "Движение и пассажиры", "Источник": "movement_input, motion, passenger_checkpoint и common_scenario"},
                  {"Данные": "Прогнозы", "Источник": "controller/raw_forecast_*.json"},
                  {"Данные": "Оптимизатор", "Источник": "controller/decision_*.json"},
                  {"Данные": "Объяснения", "Источник": "explanations/explanation_*.json"}], hide_index=True)
    st.write("В этой оболочке доступен просмотр сохранённых результатов. Новый расчёт запускается существующими скриптами проекта отдельно.")
    st.download_button("Скачать состояние выбранного момента", data=json.dumps(frame,ensure_ascii=False,indent=2),
                       file_name=f"metro_frame_{second:05d}.json", mime="application/json")
    st.markdown("Основа интерфейса: [sirius-subway_simulation](https://github.com/KovalevNikolay710/sirius-subway_simulation), ветка dev, версия `2bbcdb97`.")


st.navigation([st.Page(forecast_page, title="Прогноз", icon=":material/insights:", default=True),
               st.Page(simulation_page, title="Симулятор", icon=":material/train:"),
               st.Page(optimizer_page, title="Оптимизатор", icon=":material/tune:"),
               st.Page(data_page, title="Данные", icon=":material/database:")]).run()
