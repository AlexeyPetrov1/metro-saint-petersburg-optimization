"""Compact offline viewer for the recorded 4.8 experiments."""

import argparse
import json
from pathlib import Path


def render_dashboard(comparison, benchmark):
    arms = [dict(row, to1_resource_equivalent_rub=row.get('to1_resource_equivalent_rub'),
                 saving_percent=row.get('saving_percent')) for row in comparison['arms']]
    payload = dict(reference=comparison.get('reference', {}), arms=arms,
        scores=[r for r in benchmark.get('scores', []) if r.get('scope') == 'all'],
        dates={k: v for k, v in benchmark.get('contract', {}).items() if k.endswith('_dates')})
    shared = comparison.get('forecast_comparison', {}).get('OLD_ML_SHARED_PROFILE')
    if shared:
        payload['scores'].append(dict(shared, model='OLD_ML_SHARED_PROFILE', stage='development'))
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    for character, escaped in (('&', '\\u0026'), ('<', '\\u003c'), ('>', '\\u003e'),
                               ('\u2028', '\\u2028'), ('\u2029', '\\u2029')):
        encoded = encoded.replace(character, escaped)
    return HTML.replace('__DATA__', encoded)


def write_dashboard(output, comparison, benchmark):
    path = Path(output) / 'dashboard.html'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        stream.write(render_dashboard(comparison, benchmark))
    return path


HTML = '''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Метро: результаты 4.8</title>
<style>
:root{font:16px/1.5 system-ui,Segoe UI,sans-serif;color:#192d43;background:#f3f6fa}
*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:30px 20px}
h1{font-size:30px;line-height:1.2;margin:10px 0}h2{font-size:21px;margin:0 0 14px}
p{margin:8px 0;color:#526477}.tag{color:#b62e42;font-weight:700;font-size:13px;letter-spacing:.06em}
section{background:white;border:1px solid #dbe4ee;border-radius:14px;padding:22px;margin-top:20px}
select{padding:9px 14px;font:inherit;border:1px solid #adbecf;border-radius:7px;margin:8px 0 16px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:#f4f7fb;padding:16px;border-radius:8px}
.value{font-weight:700;font-size:26px}.label{font-size:13px;color:#526477}.notice{border-left:4px solid #b62e42;padding:9px 14px;background:#fff1f2}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #dbe4ee}
.scroll{overflow:auto}svg{width:100%;height:auto}a{color:#126198}.fine{font-size:13px}.links{display:flex;gap:18px;flex-wrap:wrap}
@media(max-width:750px){.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:25px}section{padding:16px}}
</style><main>
<div class="tag">КРАСНАЯ ЛИНИЯ · ЗАПИСАННЫЕ РЕЗУЛЬТАТЫ 4.8</div>
<h1>Прогноз входов и расходы на движение</h1>
<p>30 сентября 2026 года. Управление проверено для общего эпизода 09:15–14:15; отправления продолжаются до 14:25, возврат составов проверен до 17:15.</p>
<p class="fine">Повторный ввод временно снятых составов проверен до 14:15. Возврат в депо при закрытии эпизода проверяется отдельно до 17:15.</p>
<p>Это конечный эпизод. Проценты ниже относятся к нему. Денежная экономия неизвестна: ₽ показывают расход ресурса ТО1, а не фактическую прибыль.</p>
<section><h2>Сравнение стратегий</h2><label for="arm-select">Прогноз для оптимизатора</label><br>
<select id="arm-select"></select><div id="cards" class="cards"></div>
<p id="arm-status" class="fine"></p><div id="cost-chart"></div>
<p id="reference-resource" class="fine"></p>
<p class="fine">Парк фиксирован: 68 составов, по 8 вагонов. Ставка ТО1: Балтиец 9,2 ₽/км, Юбилейный 7,4 ₽/км.</p></section>
<section><h2>Ошибка прогноза входов</h2>
<p>Архив GRU выдаёт слоты по 30 минут. Адаптер обновляет прогноз через 15 минут; перенос на :15 и :45 проверен отдельно и остаётся экспериментальным.</p>
<p class="fine">Общий профиль четвертей позволяет сопоставить GRU, медиану и прежнюю ML. Прежняя ML с исходным профилем также показана отдельной строкой.</p>
<div class="scroll"><table><thead><tr><th>Модель</th><th>Этап и даты</th><th>WAPE</th><th>Смещение</th></tr></thead><tbody id="forecast-table"></tbody></table></div>
<p class="fine">GRU обучена до 31 июля; старая LightGBM до 29 сентября и сравнивается только 30 сентября. Сентябрь уже исследовался в проекте, поэтому эти даты не являются нетронутой выборкой.</p></section>
<section><h2>Полный день: FAIL</h2><div class="notice">Остаются три нарушения лимита 1%: 07:45–08:00 — 1,0725%; 08:00–08:15 — 5,1280%; 08:15–08:30 — 1,2436%.</div>
<p>Физическая проверка пройдена: интервалы 113–480 секунд, максимум 53 состава, 758 936 пассажиров завершили поездки. Суточная экономическая приёмка остаётся открытой.</p>
<p class="fine">Входы через турникеты не измеряют загрузку вагонов. OD S1, распределение по вагонам и инфраструктура заданы в модели.</p>
<div class="links"><a href="report.md">Полный отчёт 4.8</a><a href="comparison.json">Данные сравнения</a><a href="controller_comparison.png">График управления</a><a href="../metro48_daily_review_20261009/report.md">Суточный отчёт</a><a href="../metro48_daily_review_20261009/daily_exceptional_windows.png">График полного дня</a><a href="../metro48_daily_fifo_independent_20261009_v7/acceptance.json">Независимая проверка</a></div></section>
</main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),select=document.getElementById('arm-select');
const names={GRU:'GRU с обновлением через 15 минут',ML:'Прежняя ML (общий профиль)',MEDIAN:'Медиана с общим профилем',MEDIAN_SHARED_PROFILE:'Медиана с общим профилем',OLD_ML:'Прежняя ML (исходный профиль)',OLD_ML_SHARED_PROFILE:'Прежняя ML (общий профиль)'};
const number=(x,suffix='')=>typeof x==='number'&&Number.isFinite(x)?x.toLocaleString('ru-RU',{maximumFractionDigits:2})+suffix:'неизвестно';
function cell(tag,text,parent){const node=document.createElement(tag);node.textContent=text;parent.appendChild(node);return node;}
data.arms.forEach((arm,i)=>{const option=cell('option',names[arm.arm]||arm.arm,select);option.value=i;});
function show(){const arm=data.arms[Number(select.value)];if(!arm)return;const cards=document.getElementById('cards');cards.replaceChildren();
 [['Пробег',arm.train_km,' км'],['На состав из 68',arm.mean_km_per_fixed_train,' км'],['Ресурс ТО1',arm.to1_resource_equivalent_rub,' ₽'],['Экономия ресурса',arm.saving_percent,'%']].forEach(([label,value,unit])=>{const card=cell('div','',cards);card.className='card';cell('div',label,card).className='label';cell('div',number(value,unit),card).className='value';});
 document.getElementById('arm-status').textContent='Независимая приёмка эпизода: '+(arm.independently_accepted===true?'PASS':arm.independently_accepted===false?'FAIL':'неизвестно')+'. Максимальное время решения: '+number(arm.maximum_pipeline_seconds,' с')+'. Доставка внешнего уведомления не измерена.';
 document.querySelectorAll('#cost-chart rect').forEach((bar,i)=>bar.setAttribute('fill',i===Number(select.value)?'#b62e42':'#3d789f'));
}
select.addEventListener('change',show);show();
document.getElementById('reference-resource').textContent='Исходный график: '+number(data.reference.to1_resource_equivalent_rub,' ₽ ресурса ТО1')+'; '+number(data.reference.train_km,' км')+'.';
const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 900 ${data.arms.length*52+15}`);svg.setAttribute('aria-label','Расход ресурса ТО1 по стратегиям');
const costs=data.arms.map(a=>a.to1_resource_equivalent_rub).filter(Number.isFinite),max=Math.max(1,...costs);
data.arms.forEach((arm,i)=>{const value=arm.to1_resource_equivalent_rub;const label=document.createElementNS(ns,'text');label.setAttribute('x',0);label.setAttribute('y',i*52+30);label.setAttribute('font-size',14);label.textContent=arm.arm;svg.appendChild(label);if(Number.isFinite(value)){const bar=document.createElementNS(ns,'rect');bar.setAttribute('x',90);bar.setAttribute('y',i*52+10);bar.setAttribute('width',Math.max(0,value)/max*570);bar.setAttribute('height',28);bar.setAttribute('fill',i===Number(select.value)?'#b62e42':'#3d789f');svg.appendChild(bar);}const total=document.createElementNS(ns,'text');total.setAttribute('x',685);total.setAttribute('y',i*52+30);total.setAttribute('font-size',14);total.textContent=number(value,' ₽');svg.appendChild(total);});document.getElementById('cost-chart').appendChild(svg);
const stages={calibration:'Калибровка',evaluation:'Проверка',development:'Разработка'},table=document.getElementById('forecast-table');
data.scores.forEach(row=>{const tr=cell('tr','',table),dates=data.dates[row.stage+'_dates']||[];[names[row.model]||row.model,(stages[row.stage]||row.stage)+' · '+dates.join(', '),number(row.wape_percent,'%'),number(row.signed_bias_percent,'%')].forEach(value=>cell('td',value,tr));});
</script></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--benchmark', required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
    print(write_dashboard(args.output, read(args.comparison), read(args.benchmark)))
