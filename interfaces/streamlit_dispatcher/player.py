"""Reuse the upstream player assets unchanged; inject our read-only data panels."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "upstream/src/metro_control/assets"


def player_html(payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = (ASSETS / "sim_player.html").read_text(encoding="utf8")
    html = html.replace("/*__UI_CSS__*/", (ASSETS / "ui.css").read_text(encoding="utf8"))
    html = html.replace("//__UI_JS__", (ASSETS / "ui.js").read_text(encoding="utf8"))
    html = html.replace("__PAYLOAD__", data)
    html = html.replace('pos: P.first,', 'pos: P.start_pos ?? P.first,')
    html = html.replace("Загрузка линии за день", "Загрузка самого плотного вагона: снимки каждые 15 минут")
    html = html.replace("спрос/вместимость", "загрузка вагона")
    html = html.replace("спрос ", "загрузка вагона ")
    html = html.replace("вместимости, поезда заполнены до", "предела 150 чел., загрузка до")
    html = html.replace("С политикой", "Выбранный прогон").replace("Решения политики", "Решения оптимизатора")
    html = html.replace("Отказы в посадке с начала дня", "Пассажиры в поездах сейчас")
    html = html.replace("Ожидание с начала дня, пасс·мин", "Вошли через турникеты")
    panels = '''<section class="panel"><h3>Поезда и очереди в выбранный момент</h3>
<div class="sub">Наведение на состав: пассажиры в каждом из 8 вагонов. Круги: люди на платформе.</div>
<svg id="fleet" width="100%" height="280"></svg><div class="sub" id="fleetMeta"></div></section>
<section class="two"><div class="panel"><h3>Прогноз входов за 15 минут</h3>
<div class="sub" id="forecastMeta"></div><div id="forecastRows" class="forecast-grid"></div></div>
<div class="panel"><h3>Текущее решение и объяснение</h3><div class="sub" id="decisionMeta"></div>
<div id="decisionText"></div></div></section>'''
    html = html.replace('<section class="two">', panels + '\n<section class="two">', 1)
    style = '''<style>.forecast-grid{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 16px;font-size:12px}
#decisionText{font-size:13px;line-height:1.5;max-height:360px;overflow:auto;overflow-wrap:anywhere}#decisionText p{margin:6px 0}
#fleet .train:hover{stroke:#1B2430;stroke-width:2}#fleet .train{cursor:pointer}</style>'''
    html = html.replace("</head>", style + "</head>")
    html = html.replace("function render() {", "function renderBase() {")
    extension = (HERE / "player_adapter.js").read_text(encoding="utf8")
    html = html.replace("let lastTs = null;", extension + "\nlet lastTs = null;")
    return html
