"""Write an offline replay viewer; no server or external assets are needed."""

from copy import deepcopy
import json
from pathlib import Path


def write_demo(output_dir, payload):
    data = deepcopy(payload)
    if data.get("schema_version", 1) not in (1, "1.0") or data.get("release_authorized", False) is not False:
        raise ValueError("Only schema 1, recorded, non-authorized demo output is supported")
    scenarios = data.setdefault("scenarios", [])
    if not isinstance(scenarios, list) or any(not isinstance(row, dict) for row in scenarios):
        raise ValueError("scenarios must be a list of objects")
    ids = [row.get("scenario_id") for row in scenarios if row.get("scenario_id") is not None]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate scenario_id")
    data.update(schema_version=1, release_authorized=False)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, escape in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                              ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(character, escape)
    target = Path(output_dir) / "demo.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_HTML.replace("__DEMO_PAYLOAD__", encoded), encoding="utf-8")
    return target


_HTML = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Метро: GOAL 4.3</title>
<style>
:root{color-scheme:light;--ink:#142437;--muted:#5b6b7b;--blue:#146ca4;--border:#dce4eb;--green:#146c43;--red:#ad2532}
*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1440px;margin:auto;padding:28px}h1{margin:12px 0 8px;font-size:29px}h2{font-size:18px;margin:0 0 14px}p{margin:8px 0}
.eyebrow{color:var(--blue);font-size:13px;font-weight:700;letter-spacing:.06em}.muted,.note{color:var(--muted)}.note{font-size:13px}
.badge{display:inline-block;padding:5px 10px;margin:4px 7px 4px 0;background:#e7eff5;border-radius:6px;font-size:13px}.good{color:var(--green)}.bad{color:var(--red)}
.panel{background:white;border:1px solid var(--border);border-radius:12px;padding:20px;margin-top:18px}.selectors{display:grid;grid-template-columns:1fr 1fr;gap:18px}
label{display:block;font-weight:600}select{display:block;width:100%;font:inherit;color:inherit;padding:10px;margin-top:7px;border:1px solid #b9c9d6;border-radius:7px;background:white}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.metric{padding:13px;background:#f4f7fa;border-radius:8px}.metric-name{font-size:12px;color:var(--muted)}.metric-value{font-size:23px;font-weight:650;margin-top:5px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chart{min-width:0}.chart svg{width:100%;height:auto;display:block}.empty{padding:35px;text-align:center;color:var(--muted);border:1px dashed var(--border);border-radius:7px}
.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid var(--border)}th{background:#f4f7fa;font-weight:650}th:first-child,td:first-child{text-align:left}.selected-row{background:#edf5fb}
.decision{padding:12px 0;border-bottom:1px solid var(--border)}.decision-title{font-weight:650}.decision p{margin:5px 0}.legend{font-size:12px;color:var(--muted);margin:7px 0;min-height:18px}details{margin-top:16px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f4f7fa;padding:12px}
@media(max-width:800px){main{padding:16px}.metrics{grid-template-columns:1fr 1fr}.charts,.selectors{grid-template-columns:1fr}h1{font-size:24px}.panel{padding:15px}}
</style>
</head>
<body>
<main>
<div class="eyebrow">GOAL 4.3 · ЛОКАЛЬНОЕ ДЕМО</div>
<h1 id="page-title">Регулирование движения и пробег составов</h1>
<p class="muted">Воспроизведение сохранённого расчёта. Время открытия страницы не является временем поиска.</p>
<span class="badge">Управление выпуском отключено</span><span class="badge" id="data-mode">Неизвестно</span>
<div class="panel selectors">
<label>Сценарий<select id="scenario-select" aria-label="Сценарий"></select></label>
<label>Стратегия<select id="strategy-select" aria-label="Стратегия"></select></label>
</div>
<div class="panel">
<h2>Результат выбранного расчёта</h2><div id="result-status" class="badge">Неизвестно</div>
<p id="incident-summary" class="muted"></p>
<div id="metric-grid" class="metrics"></div>
<p class="note">Срок выдачи проверенного результата: 480 секунд от обнаружения. Записанная длительность включает только этапы, измеренные в исходном расчёте.</p>
<p class="note">ТО1 показано как стоимость расхода ресурса. Потерянная выручка и денежные выплаты учитываются отдельно.</p>
<details><summary>Параметры инцидента</summary><pre id="incident-data"></pre></details>
</div>
<div class="panel"><h2>Рекомендации и время эффекта</h2><div id="decisions"></div></div>
<div class="panel"><h2>Сравнение стратегий в одном сценарии</h2><div class="table-scroll"><table id="comparison-table"></table></div></div>
<div class="panel"><h2>Варианты назначения пассажиров</h2><p class="note" id="od-metrics-scope">OD основной карточки неизвестен. S1/S2/S3 задают разные назначения; результаты условные.</p><div class="table-scroll"><table id="od-table"></table></div></div>
<div class="panel"><h2>Аудит смены и остатка очереди</h2><div id="service-audit"></div></div>
<div class="charts">
<div class="panel chart"><h2>Входы через турникеты и прогноз</h2><div id="entries-chart"></div></div>
<div class="panel chart"><h2>Загрузка вагона</h2><div id="load-chart"></div></div>
<div class="panel chart"><h2>Очередь на платформе</h2><div id="queue-chart"></div></div>
<div class="panel chart"><h2>Интервалы отправлений</h2><div id="headway-chart"></div></div>
<div class="panel chart"><h2>Количество составов на линии</h2><div id="fleet-chart"></div></div>
</div>
<div class="panel"><h2>Километры каждого состава относительно KEEP</h2>
<p class="note">Положительная разница означает сокращение пробега. Сопоставимость горизонта и завершение обязательств определяются сохранённым расчётом.</p>
<div class="table-scroll"><table id="per-train-table"></table></div></div>
<div class="panel"><h2>Происхождение и ограничения данных</h2><div id="provenance"></div></div>
</main>
<script id="demo-data" type="application/json">__DEMO_PAYLOAD__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('demo-data').textContent);
const scenarios=Array.isArray(data.scenarios)?data.scenarios:[];
const el=id=>document.getElementById(id);
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const number=(value,digits=2)=>finite(value)?new Intl.NumberFormat('ru-RU',{maximumFractionDigits:digits}).format(value):'Неизвестно';
const fmt=(value,unit='',digits=2)=>finite(value)?number(value,digits)+(unit?' '+unit:''):'Неизвестно';
const txt=value=>value===null||value===undefined||value===''?'Неизвестно':String(value);
const array=value=>Array.isArray(value)?value:[];
function node(tag,text,css){const item=document.createElement(tag);if(text!==undefined)item.textContent=String(text);if(css)item.className=css;return item;}
function options(select,rows,label){select.replaceChildren();rows.forEach((row,index)=>{const item=node('option',label(row,index));item.value=String(index);select.append(item);});select.disabled=!rows.length;}
function table(target,headers,rows,selected=-1){target.replaceChildren();const head=node('thead'),hr=node('tr');headers.forEach(label=>hr.append(node('th',label)));head.append(hr);target.append(head);const body=node('tbody');rows.forEach((values,index)=>{const tr=node('tr');if(index===selected)tr.className='selected-row';values.forEach(value=>tr.append(node('td',value)));body.append(tr);});if(!rows.length){const tr=node('tr'),td=node('td','Нет данных');td.colSpan=headers.length;tr.append(td);body.append(tr);}target.append(body);}
function metric(label,value,css){const box=node('div',undefined,'metric');box.append(node('div',label,'metric-name'));box.append(node('div',value,'metric-value'+(css?' '+css:'')));el('metric-grid').append(box);}
const strategyLabels={KEEP:'KEEP: продолжить план',REGULATE_ONLY:'Регулировать интервалы',WITHDRAW_ONLY:'Снять составы',COMBINED:'Интервалы и снятие',ADD:'Подать резерв',DEFER_WITHDRAWAL:'Отменить снятие',ROLLING_SELECTED:'Последовательные решения'};
const statusLabels={pass:'Допустимо',fail:'Есть нарушения',unknown:'Неизвестно',checked:'Проверенный расчёт',evaluated:'Проверенная альтернатива',feasible:'Допустимый проверенный план',conditional_resource_policy:'Выбор по пробегу при принятых условиях',search_incomplete:'Поиск не завершён',no_feasible_candidate:'Допустимый план не найден',evaluation_timeout:'Время проверки исчерпано'};
function modeLabel(mode){if(mode==='historical_demand_synthetic_incident'||typeof mode==='string'&&mode.includes('injected_incident'))return 'Исторические входы; искусственный инцидент';if(mode==='fixture')return 'Искусственный контрольный пример';if(typeof mode==='string'&&mode.includes('synthetic'))return 'Искусственный сценарий: '+mode;return txt(mode);}
// incident-duration:start
function incidentDurationLabel(incident){const minutes=incident.actual_duration_minutes??incident.duration_minutes??incident.n_minutes??incident.N;const parts=[];if(finite(minutes))parts.push('Остановка (факт): '+fmt(minutes,'мин'));if(Object.hasOwn(incident,'initial_estimated_duration_seconds'))parts.push('Первоначальная оценка: '+(finite(incident.initial_estimated_duration_seconds)?fmt(incident.initial_estimated_duration_seconds/60,'мин'):'неизвестна'));return parts.join(' · ');}
// incident-duration:end
// service-audit:start
function serviceAuditLines(comparison){const lines=['Область аудита: '+(comparison.audit_full_day===true?'полная смена':comparison.audit_full_day===false?'ограниченный горизонт':'неизвестна')+'. Допустимость решения на ближайшем горизонте проверяется отдельно.'];array(comparison.od_results).forEach(row=>{const scope=row.cohort_scope||{},windows=row.failed_15m_windows;lines.push(txt(row.od)+': очередь после смены и возвратов — '+txt(row.queued)+'; нарушенных 15-минутных окон — '+(Array.isArray(windows)?windows.length:'неизвестно')+'; наибольшая доля >150 в окне — '+(finite(row.maximum_window_share)?fmt(100*row.maximum_window_share,'%'):'неизвестно')+'. Ожидание относится к входам когорты '+txt(scope.start_second)+'–'+txt(scope.end_second)+' с.');});const age=comparison.queue_age_scope;if(age)lines.push('Возраст очереди измерен в сохранённом состоянии '+txt(age.checkpoint_second)+' с после завершения смены и возвратов; это не возраст на момент решения.');return lines;}
// service-audit:end
// decision-descriptor:start
function describeDecision(row,comparison){
const selection=row.selection||{};
const recommended=selection.selected_candidate_id||selection.mitigation_candidate_id||null;
const viewed=comparison.candidate_id==='ROLLING_SELECTED'?recommended:comparison.candidate_id||recommended;
const candidate=(Array.isArray(row.candidate_table)?row.candidate_table:[]).find(item=>item.candidate_id===viewed)||{};
return {recommended_candidate_id:recommended,viewed_candidate_id:viewed,is_mitigation:!selection.selected_candidate_id&&!!selection.mitigation_candidate_id,
status:selection.status||row.status||null,action:candidate.action||row.action||null,reason:row.reason||candidate.reason||null};
}
// decision-descriptor:end
// effect-lines:start
function effectLines(action,row){
const lines=[];if(!action||typeof action!=='object')return lines;
if(finite(action.timetable_change_second)){
lines.push('Первое изменение графика: '+fmt(action.timetable_change_second,'с')+' сценария');
Object.entries(action.timetable_change_by_station||{}).forEach(([station,second])=>lines.push('Станция '+station+': изменение графика на '+fmt(second,'с')));
lines.push('Влияние на пассажиров оценивается отдельно в сравнении.');
}
const entries=Array.isArray(action.actions)&&action.actions.length?action.actions:[action];
entries.forEach(entry=>{const effect=entry.actual_effect_second??entry.planned_effect_second;
if(finite(entry.depot_return_second)){lines.push('Состав '+txt(entry.train_id)+': возврат в депо на '+fmt(entry.depot_return_second,'с')+' сценария');return;}
if(entry.actual_effect_kind==='timetable change')return;
if(finite(effect)){const since=entry.decision_second??action.decision_second??row.decision_second;
lines.push((entry.train_id?'Состав '+txt(entry.train_id)+': ':'')+'расчётный эффект '+(finite(since)?'через '+fmt(effect-since,'с')+' после решения':'на '+fmt(effect,'с')+' сценария'));}
Object.entries(entry.actual_effect_by_station||{}).forEach(([station,second])=>lines.push('Станция '+station+': эффект на '+fmt(second,'с')+' сценария'));
});return lines;
}
// effect-lines:end
function renderDecisions(scenario,comparison){
const target=el('decisions');target.replaceChildren();const rows=array(comparison.decisions).length?comparison.decisions:array(scenario.decisions);
if(!rows.length){target.append(node('p','Нет сохранённой рекомендации','muted'));return;}
rows.forEach(row=>{if(typeof row!=='object'||row===null){target.append(node('p',txt(row)));return;}
const descriptor=describeDecision(row,comparison),item=node('div',undefined,'decision');
if(row.selection){item.append(node('p',(descriptor.is_mitigation?'Вариант снижения ущерба: ':'Сохранённая рекомендация: ')+txt(descriptor.recommended_candidate_id)));
item.append(node('p','Статус: '+(statusLabels[descriptor.status]||txt(descriptor.status))));}
const action=descriptor.action,kind=action&&typeof action==='object'?action.kind:action;
item.append(node('div','Просматриваемое действие: '+(strategyLabels[kind]||txt(kind||row.strategy||row.title||row.status)),'decision-title'));
const fields=[['Причина',descriptor.reason],['Время готовности',row.recommendation_ready_at],['Расчётный первый эффект',row.first_effect_at||row.effect_at],['Срок восстановления',row.recovery_at],['Ограничения',row.limitations]];
fields.forEach(([label,value])=>{if(value!==undefined&&value!==null)item.append(node('p',label+': '+(Array.isArray(value)?value.map(txt).join('; '):txt(value))));});
effectLines(action,row).forEach(line=>item.append(node('p',line)));target.append(item);
});}
function svgNode(tag,attrs,text){const item=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([key,value])=>item.setAttribute(key,String(value)));if(text!==undefined)item.textContent=String(text);return item;}
function chart(id,source,specs,limits=[],unit=''){const target=el(id);target.replaceChildren();const lines=specs.map(spec=>({...spec,points:array(source).filter(row=>row&&finite(row.second)&&finite(spec.value(row))).map(row=>[row.second,spec.value(row)]).sort((a,b)=>a[0]-b[0])}));const points=lines.flatMap(row=>row.points);if(!points.length){target.append(node('div','Нет данных','empty'));return;}const width=640,height=255,left=55,right=18,top=14,bottom=40;const xmin=Math.min(...points.map(p=>p[0])),xmax=Math.max(xmin+1,...points.map(p=>p[0]));const ymin=Math.min(0,...points.map(p=>p[1])),ymax=Math.max(1,...points.map(p=>p[1]),...limits.map(row=>row.value))*1.12;
const x=value=>left+(value-xmin)/(xmax-xmin)*(width-left-right),y=value=>height-bottom-(value-ymin)/(ymax-ymin)*(height-top-bottom);
const label=lines.map(line=>line.label).join(' · '),svg=svgNode('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':target.parentElement.querySelector('h2').textContent+': '+label});svg.append(svgNode('title',{},label));
for(let i=0;i<=4;i++){const value=ymin+(ymax-ymin)*i/4;svg.append(svgNode('line',{x1:left,y1:y(value),x2:width-right,y2:y(value),stroke:'#e2e8ee'}));svg.append(svgNode('text',{x:left-7,y:y(value)+4,'text-anchor':'end',fill:'#5b6b7b','font-size':11},number(value,0)));}
limits.forEach(limit=>{svg.append(svgNode('line',{x1:left,y1:y(limit.value),x2:width-right,y2:y(limit.value),stroke:limit.color,'stroke-dasharray':'5 4'}));svg.append(svgNode('text',{x:width-right-3,y:y(limit.value)-4,'text-anchor':'end',fill:limit.color,'font-size':11},limit.label));});
lines.forEach(line=>{if(!line.points.length)return;const path=line.points.map((point,index)=>(index?'L':'M')+x(point[0]).toFixed(2)+','+y(point[1]).toFixed(2)).join(' ');svg.append(svgNode('path',{d:path,fill:'none',stroke:line.color,'stroke-width':2}));if(line.points.length===1)svg.append(svgNode('circle',{cx:x(line.points[0][0]),cy:y(line.points[0][1]),r:3,fill:line.color}));});
svg.append(svgNode('text',{x:left,y:height-17,fill:'#5b6b7b','font-size':11},fmt(xmin,'с',0)));svg.append(svgNode('text',{x:width-right,y:height-17,'text-anchor':'end',fill:'#5b6b7b','font-size':11},fmt(xmax,'с',0)));target.append(svg);target.append(node('div',lines.filter(line=>line.points.length).map(line=>line.label).join(' · ')+(unit?' · '+unit:''),'legend'));}
function renderProvenance(scenario,comparison){const target=el('provenance');target.replaceChildren();target.append(node('p','Режим данных: '+modeLabel(scenario.evaluation_mode)));const rows=[['Подтверждённые параметры',scenario.confirmed_parameters||data.confirmed_parameters],['Допущения',scenario.assumptions||data.assumptions],['Неизвестные данные',scenario.unknowns||data.unknowns],['Ограничения вывода',comparison.limitations||scenario.limitations||data.limitations]];rows.forEach(([label,value])=>{const text=Array.isArray(value)?value.map(txt).join('; '):value&&typeof value==='object'?JSON.stringify(value,null,2):txt(value);target.append(node('p',label+': '+text));});}
function render(){const scenario=scenarios[Number(el('scenario-select').value)]||{};const comparisons=array(scenario.comparisons);const selected=Number(el('strategy-select').value)||0;const comparison=comparisons[selected]||{},m=comparison.metrics||{};const keep=comparisons.find(row=>row.strategy==='KEEP'),baseline=keep&&keep.metrics||{};el('data-mode').textContent=modeLabel(scenario.evaluation_mode);el('result-status').textContent=statusLabels[comparison.status]||txt(comparison.status);el('result-status').className='badge'+(['fail','no_feasible_candidate','evaluation_timeout'].includes(comparison.status)?' bad':'');const incident=scenario.incident||{};el('incident-summary').textContent=scenarios.length?[txt(scenario.title||scenario.scenario_id),incident.station_id?'Станция '+txt(incident.station_id):null,incidentDurationLabel(incident)||null].filter(Boolean).join(' · '):'Нет сохранённых сценариев';el('incident-data').textContent=JSON.stringify(incident,null,2);el('metric-grid').replaceChildren();
metric('Время сохранённого расчёта',fmt(comparison.elapsed_seconds,'с'),finite(comparison.elapsed_seconds)&&comparison.elapsed_seconds>480?'bad':'');metric('Среднее ожидание',fmt(m.mean_wait_seconds,'с'));metric('P95 ожидания',fmt(m.p95_wait_seconds,'с'));metric('Изменение ожидания к KEEP',finite(m.mean_wait_seconds)&&finite(baseline.mean_wait_seconds)?fmt(m.mean_wait_seconds-baseline.mean_wait_seconds,'с'):'Неизвестно');metric('Максимум в вагоне',fmt(m.max_car_load,'чел.',0),finite(m.max_car_load)&&m.max_car_load>160?'bad':'');metric('Исключительное время >150',finite(m.exceptional_time_share)?fmt(100*m.exceptional_time_share,'%'):'Неизвестно',finite(m.exceptional_time_share)&&m.exceptional_time_share>0.01?'bad':'');metric('Сопоставимый пробег',fmt(m.train_km,'поездо-км'));metric('Сокращение пробега к KEEP',fmt(m.saving_km,'поездо-км'),finite(m.saving_km)?m.saving_km>0?'good':m.saving_km<0?'bad':'':'');metric('Оценка ресурса ТО1',fmt(m.to1_resource_value_rub,'₽'));metric('Потерянная выручка',fmt(m.lost_revenue_rub,'₽'));
metric('Сокращение на состав (фиксированные 68)',fmt(m.mean_saving_km_per_train,'км/состав'));
metric('Возраст старейшего ожидающего',fmt(m.oldest_queue_age_seconds,'с'));
metric('Ожидание старейшего от турникета',fmt(m.oldest_turnstile_wait_age_seconds,'с'));
el('service-audit').replaceChildren();serviceAuditLines(comparison).forEach(line=>el('service-audit').append(node('p',line)));
el('od-metrics-scope').textContent='Метрики основной карточки: '+txt(comparison.card_od)+'. S1/S2/S3 задают разные назначения; результаты условные.';
table(el('od-table'),['OD','Seed','Среднее ожидание, с','P95, с','Макс./вагон','Искл. время, %','Интервалы соблюдены'],array(comparison.od_results).map(row=>[txt(row.od),txt(row.seed),number(row.mean_wait_seconds),number(row.p95_wait_seconds),number(row.max_car_load,0),finite(row.exceptional_time_share)?number(100*row.exceptional_time_share):'Неизвестно',row.headways_passed===true?'да':row.headways_passed===false?'нет':'Неизвестно']));
table(el('comparison-table'),['Стратегия','Статус','Расчёт, с','Среднее ожидание, с','P95, с','Макс./вагон','Искл. время, %','Пробег, км','Сокращено, км'],comparisons.map(row=>{const v=row.metrics||{};return [strategyLabels[row.strategy]||txt(row.strategy),statusLabels[row.status]||txt(row.status),number(row.elapsed_seconds),number(v.mean_wait_seconds),number(v.p95_wait_seconds),number(v.max_car_load,0),finite(v.exceptional_time_share)?number(v.exceptional_time_share*100):'Неизвестно',number(v.train_km),number(v.saving_km)];}),selected);
table(el('per-train-table'),['Состав','Тип','KEEP, км','Выбранный план, км','Сокращено, км'],array(comparison.per_train).map(row=>[txt(row.train_id),txt(row.train_type||row.type),number(row.baseline_km),number(row.candidate_km),number(row.saving_km)]));renderDecisions(scenario,comparison);const series=comparison.series||{};
chart('entries-chart',series.entries,[{label:'Входы: факт',color:'#146ca4',value:r=>r.actual},{label:'Почасовая медиана',color:'#dc850b',value:r=>r.forecast}],[],'человек за интервал');chart('load-chart',series.load,[{label:'Сохранённая загрузка',color:'#146ca4',value:r=>r.load}],[{value:120,label:'120: комфорт',color:'#14845a'},{value:150,label:'150: обычный предел',color:'#dc850b'},{value:160,label:'160: максимум',color:'#ad2532'}],'человек/вагон');chart('queue-chart',series.queue,[{label:'Очередь',color:'#146ca4',value:r=>r.queued}],[],'человек');chart('headway-chart',series.headway,[{label:series.headway_label||'Интервал',color:'#146ca4',value:r=>r.seconds}],[{value:113,label:'113: минимум в пик',color:'#14845a'},{value:150,label:'150: минимум вне пика',color:'#dc850b'},{value:480,label:'480: целевой максимум',color:'#ad2532'}],'секунд');chart('fleet-chart',series.fleet,[{label:'На линии, включая остановившиеся',color:'#146ca4',value:r=>r.count??r.active_trains??r.on_line}],[{value:53,label:'53: предел линии',color:'#ad2532'}],'составов');renderProvenance(scenario,comparison);}
function chooseScenario(){const scenario=scenarios[Number(el('scenario-select').value)]||{};options(el('strategy-select'),array(scenario.comparisons),row=>strategyLabels[row.strategy]||txt(row.strategy));render();}
el('page-title').textContent=txt(data.title||'Регулирование движения и пробег составов');document.title=el('page-title').textContent;options(el('scenario-select'),scenarios,row=>txt(row.title||row.scenario_id));el('scenario-select').addEventListener('change',chooseScenario);el('strategy-select').addEventListener('change',render);chooseScenario();
</script>
</body>
</html>
'''
