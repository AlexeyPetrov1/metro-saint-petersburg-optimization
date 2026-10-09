"""Render measured bridge and jitter comparisons with explicit scope limits."""

import json
from pathlib import Path

from metro44_bridge_independent import read, sha


def assemble(results, originals, jitter, scopes):
    journeys = {r['od']:r for r in originals}
    scope = {r['od']:r for r in scopes}
    scenarios = []
    for result in results:
        od = result['od']
        original = journeys[od]
        arms = []
        for name,metrics in [('KEEP',result['baseline_metrics']),('BRIDGE160',result['metrics'])]:
            whole = (scope[od]['baseline_whole_episode_admissible'] if name=='KEEP'
                     else result['complete_episode_exposure']['admissible'])
            journey = original['journeys'][name]
            arms.append(dict(strategy=name,train_km=metrics['train_km'],km_per_68=metrics['mean_km_per_fixed_train'],
                mean_wait_seconds=metrics['mean_wait_seconds'],p95_wait_seconds=metrics['p95_wait_seconds'],
                mean_journey_seconds=journey['mean_gate_to_exit_seconds'],p95_journey_seconds=journey['p95_gate_to_exit_seconds'],
                passengers_in_journey_cohort=journey['arrived'],journey_complete=journey['complete'],
                max_car_load_control=metrics['max_car_load'],control_admissible=metrics['actual_service_admissible'],
                whole_episode_admissible=whole))
        scenarios.append(dict(od=od,arms=arms,working_window_bounds=original['window_bounds'],
            jitter=[r for r in jitter if r['od']==od]))
    return dict(scenarios=scenarios,reference_scope='constant_53',fixed_fleet_denominator=68,
        controlled_window='09:15–14:15',journey_cohort='09:15–14:05',admissions='08:15–14:15',
        common_closure='17:15',change_time='10:00',bridge_seconds=480,
        learned_controller_verified=False,independent_holdout_verified=False,blue_line_transfer_completed=False,
        strict_wait_pareto_improvement=False,cash_saving_rub=None,lost_revenue_rub=None,net_profit_effect_rub=None)


PAGE = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>GOAL 4.4: BRIDGE и устойчивость</title>
<style>body{font:16px system-ui;max-width:1240px;margin:24px auto;padding:0 20px;color:#182532}h1{font-size:28px}.note{background:#eef4f8;border-left:5px solid #36607d;padding:15px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 6px;text-align:left;border-bottom:1px solid #ccd5de}select{font:inherit;padding:8px}pre{white-space:pre-wrap;background:#f2f5f8;padding:12px;font-size:13px}svg{width:100%;max-height:180px}a{color:#245978}</style>
<h1>Интервал после пика и полное время поездки</h1>
<p class="note">Разработческий опыт 30 сентября 2026 года, линия 1, учебная геометрия и предполагаемые OD.
Эталон KEEP начинается с 53 составами и сохраняет 53 на линии в рабочем периоде. Это отдельный опыт;
выигрыш над ним не доказывает выигрыш над сменным суточным графиком. Перенос на синюю линию не выполнен.</p>
<p>Входы: 08:15–14:15. Контроль: 09:15–14:15. Ожидание и поездка оценены на одинаковой группе входов 09:15–14:05;
общий хвост продолжается до 17:15. BRIDGE сохраняет события до 10:00 и исходные будущие цепочки составов,
использует переход 8 минут и затем интервал 160 секунд. Это стратегия по часам; ML-прогноз при её выборе не использовался.</p>
<label for="od">Сценарий назначения пассажиров </label><select id="od">OPTIONS</select>
<table><thead><tr><th>Стратегия</th><th>км / 68</th><th>Ожидание<br>mean / p95, с</th><th>Поездка от входа<br>mean / p95, с</th><th>Макс./вагон<br>контроль</th><th>Контроль<br>допустим</th><th>Весь эпизод<br>допустим</th></tr></thead><tbody id="rows"></tbody></table>
<p>Км/68 относятся ко всему общему закрытому эпизоду 08:15–17:15, включая возвраты. Ниже отдельно показана гарантированная граница сокращения внутри 09:15–14:15.</p>
<p id="bounds" class="note"></p><p id="scope"></p>
<label for="metric">Сравнить </label><select id="metric"><option value="mean_wait_seconds">Среднее ожидание</option><option value="p95_wait_seconds">p95 ожидания</option><option value="mean_journey_seconds">Средняя полная поездка</option><option value="p95_journey_seconds">p95 полной поездки</option><option value="km_per_68">Км/68 за закрытый эпизод</option></select>
<svg id="chart" viewBox="0 0 1060 160" aria-label="Сравнение двух стратегий"></svg>
<h2>Чувствительность ко времени входа внутри минуты</h2>
<p>Шесть пар: 2 seed × 3 OD. Сдвиг входа синтетический; наблюдаемые суммы по минутам и 15 минутам, направления назначения и число пассажиров сохранены. Это проверка чувствительности, не новые реальные наблюдения.</p>
<table><thead><tr><th>Seed</th><th>Δ mean ожидания, с</th><th>Δ p95 ожидания, с</th><th>Δ mean поездки, с</th><th>Δ p95 поездки, с</th><th>Оба контроля</th></tr></thead><tbody id="jitter"></tbody></table>
<p>Номинальное среднее ожидание уменьшилось, p95 ожидания вырос. При сдвигах внутри минуты выросли оба показателя ожидания; полное время поездки уменьшилось. Строгий выигрыш по ожиданию и пробегу одновременно не подтверждён.</p>
<p>В контролируемом периоде все три OD проходят. В S3 весь эпизод сохраняет две общие исторические перегрузки 08:30–09:00 до начала контроля; они не исправлены этой стратегией. На обеих конечных проверены ≤2 состава на общих терминальных ресурсах и ≤1 в standby, на линии ≤53; после общего хвоста все пассажиры завершили поездки и составы вернулись.</p>
<p>Рубли ТО1 показывают только расход ресурса обслуживания по пробегу. Денежная экономия, упущенная выручка и изменение прибыли неизвестны. 90/180 секунд были допущением исследовательского отбора, не требованием пользователя. Независимый holdout и работа ML-контроллера на сменном графике требуют отдельного опыта.</p>
<p>Аудиты журналов: <a href="../metro44_bridge_off160_independent_audit_20261009_v2/independent_audit.json">3 исходных OD</a> и <a href="../metro44_arrival_sensitivity_independent_audit_20261009_v2/independent_audit.json">12 журналов чувствительности</a>.</p>
<script>const report=PAYLOAD;
const fmt=x=>x===null||x===undefined?'неизвестно':Number(x).toLocaleString('ru-RU',{minimumFractionDigits:2,maximumFractionDigits:2});
const yes=x=>x===true?'да':x===false?'нет':'неизвестно';
function show(){const s=report.scenarios.find(x=>x.od===document.getElementById('od').value);window.activeScenario=s;
document.getElementById('rows').innerHTML=s.arms.map(r=>`<tr><td>${r.strategy}</td><td>${fmt(r.km_per_68)}</td><td>${fmt(r.mean_wait_seconds)} / ${fmt(r.p95_wait_seconds)}</td><td>${fmt(r.mean_journey_seconds)} / ${fmt(r.p95_journey_seconds)}</td><td>${r.max_car_load_control}</td><td>${yes(r.control_admissible)}</td><td>${yes(r.whole_episode_admissible)}</td></tr>`).join('');
const b=s.working_window_bounds;document.getElementById('bounds').textContent=`В общем рабочем окне сокращение пробега ≥ ${fmt(b.saving_train_km_lower_bound)} км; на постоянный парк 68: ≥ ${fmt(b.saving_km_per_fixed_train_lower_bound)} км/состав. Ресурс ТО1: ≥ ${fmt(b.saving_to1_resource_rub_lower_bound)} руб. эквивалента. Денежная экономия неизвестна.`;
document.getElementById('scope').textContent=`Полная поездка: ${s.arms[0].passengers_in_journey_cohort.toLocaleString('ru-RU')} одинаковых пассажиров завершили поездки в обеих стратегиях. S3 исторические нарушения относятся к общему прошлому, не к контролю.`;
document.getElementById('jitter').innerHTML=s.jitter.map(r=>`<tr><td>${r.seed}</td><td>${fmt(r.mean_wait_delta_seconds)}</td><td>${fmt(r.arms.BRIDGE160.p95_wait-r.arms.KEEP.p95_wait)}</td><td>${fmt(r.mean_journey_delta_seconds)}</td><td>${fmt(r.p95_journey_delta_seconds)}</td><td>${yes(r.arms.KEEP.controlled_admissible&&r.arms.BRIDGE160.controlled_admissible)}</td></tr>`).join('');draw();}
function draw(){const metric=document.getElementById('metric').value,s=window.activeScenario,max=Math.max(...s.arms.map(r=>r[metric]),1);
document.getElementById('chart').innerHTML=s.arms.map((r,i)=>{const width=760*r[metric]/max;return `<g><text x="8" y="${42+i*65}">${r.strategy}</text><rect x="150" y="${20+i*65}" width="${width}" height="30" fill="${i?'#477d63':'#557b9f'}"/><text x="${162+width}" y="${42+i*65}">${fmt(r[metric])}</text></g>`}).join('');}
document.getElementById('od').addEventListener('change',show);document.getElementById('metric').addEventListener('change',draw);show();</script></html>'''


def main():
    root=Path(__file__).resolve().parent
    bridge=root/'data/processed/metro44_bridge_off160_actual_6h_20261009_v2'
    sensitivity=root/'data/processed/metro44_arrival_sensitivity_20261009'
    audit=root/'data/processed/metro44_arrival_sensitivity_independent_audit_20261009_v2/independent_audit.json'
    original_audit=root/'data/processed/metro44_bridge_off160_independent_audit_20261009_v2/independent_audit.json'
    assert read(audit)['status']=='passed' and read(original_audit)['status']=='passed'
    inputs=[bridge/f'{od.lower()}_result.json' for od in ('S1','S2','S3')]+[
        sensitivity/'original_window_and_journey.json',audit,original_audit,bridge/'scope_audit.json']
    before={str(p):sha(p) for p in inputs}
    data=assemble([read(p) for p in inputs[:3]],read(inputs[3]),read(audit)['rows'],read(inputs[-1])['scenarios'])
    output=root/'data/processed/metro44_bridge_robust_report_20261009'
    output.mkdir(exist_ok=False)
    (output/'report.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    options=''.join(f'<option>{r["od"]}</option>' for r in data['scenarios'])
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    (output/'report.html').write_text(PAGE.replace('OPTIONS',options).replace('PAYLOAD',payload),'utf-8')
    assert {str(p):sha(p) for p in inputs}==before
    (output/'render_receipt.json').write_text(json.dumps(dict(status='passed',input_hashes=before,
        source_sha256=sha(Path(__file__)),result_sha256=sha(output/'report.json'),html_sha256=sha(output/'report.html'),
        source_results_preserved=True),ensure_ascii=False,indent=2),'utf-8')
    print(output/'report.html')


if __name__=='__main__':
    main()
