"""Verify and report the paired, complete daily metro experiments."""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from economics import write_json
from metro_ml_compare import _metrics, metric_delta
from metro_ml_verify import check_model_archive, check_preserved, file_hash, require, score_forecast
from simulation import digest


ROOT = Path(__file__).resolve().parent
STRATEGIES = ('BASELINE', 'MEDIAN', 'ML', 'ORACLE')


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def validate_pairing(arms):
    require(len(arms) == 4 and {a['strategy'] for a in arms} == set(STRATEGIES), 'four paired strategies required')
    require(len({a['groups_digest'] for a in arms}) == 1, 'different actual passengers')
    require(len({a['checkpoint_digest'] for a in arms}) == 1, 'different initial checkpoints')
    for arm in arms:
        expected = [] if arm['strategy'] == 'BASELINE' else list(range(3600, 64800, 900))
        require([d['decision_second'] for d in arm['decisions']] == expected, 'missing or reordered decision')


def experiment_conclusion(baseline, candidate):
    saving = baseline['train_km'] - candidate['train_km']
    if candidate['actual_service_admissible'] is not True:
        return 'Допустимая экономия не подтверждена: итоговый режим нарушает ограничения.'
    if baseline['actual_service_admissible'] is not True:
        tail = 'экономии пробега нет' if saving <= 0 else 'пробег ниже, но исходный режим был недопустим'
        return f'Нарушения в модели устранены; {tail}.'
    if saving > 0:
        return 'В этом сценарии найден допустимый режим с меньшим пробегом.'
    return 'Режим допустим, экономии пробега нет.'


def _same(left, right, label):
    if isinstance(left, dict):
        require(left.keys() == right.keys(), f'{label}: keys differ')
        for key in left:
            _same(left[key], right[key], f'{label}/{key}')
    elif isinstance(left, list):
        require(len(left) == len(right), f'{label}: row count differs')
        for i, (a, b) in enumerate(zip(left, right)):
            _same(a, b, f'{label}/{i}')
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        require(isinstance(right, (int, float)) and math.isclose(left, right, abs_tol=1e-8, rel_tol=1e-10), f'{label}: value differs')
    else:
        require(left == right, f'{label}: value differs')


def verify(output):
    output = Path(output).resolve()
    arms = [read(output / f'{s.lower()}_result.json') for s in STRATEGIES]
    if 'source_hashes' not in arms[0]:
        # Preserve the original baseline receipt; revalidate its exact saved state
        # with the frozen campaign code in a separate directory.
        refreshed = read(output / 'baseline_revalidation/baseline_result.json')
        _same(arms[0]['metrics'], refreshed['metrics'], 'preserved baseline/revalidation')
        refreshed['run']['folder'] = 'baseline_revalidation/' + refreshed['run']['folder']
        arms[0] = refreshed
    validate_pairing(arms)
    campaign = read(output / 'campaign_manifest.json')
    bootstrap = read(output / 'experiment_manifest.json')
    shared = read(output / 'baseline.json.gz')
    require(digest(bootstrap['contract']) == bootstrap['identity'] == campaign['baseline_identity'], 'baseline identity differs')
    require(digest(shared['groups']) == arms[0]['groups_digest'], 'baseline passenger groups changed')
    require(digest(shared['checkpoint']) == arms[0]['checkpoint_digest'], 'baseline warmup changed')
    require(campaign['controller_decisions_per_arm'] == 68 and campaign['actual_line'] == 1, 'campaign contract differs')
    for name, sha in campaign['source_hashes'].items():
        require(file_hash(ROOT / name) == sha, f'current source differs: {name}')
        require(file_hash(output / 'source_versions' / f'{sha}_{name}') == sha, f'archived source differs: {name}')
        require(all(a['source_hashes'][name] == sha for a in arms), f'strategies use different sources: {name}')
    frame = pd.read_parquet(output / 'observations.parquet')
    truth = pd.read_parquet(ROOT / 'data/processed/metro/entries_station_15m.parquet')
    keys = ['station_id', 'interval_start']
    bounds = (pd.Timestamp('2026-09-30T06:00:00+03:00'), pd.Timestamp('2026-10-01T00:00:00+03:00'))
    actual = frame[frame.interval_start.ge(bounds[0]) & frame.interval_start.lt(bounds[1])].copy()
    require(actual.is_complete.eq(True).all() and actual.source_id.eq('uploaded-spb-line1-15min').all(), 'day contains incomplete or fabricated observations')
    joined = actual.merge(truth[keys + ['entry_count']], on=keys, suffixes=('', '_original'), validate='one_to_one')
    require(len(joined) == 19 * 18 * 4 and joined.entry_count.eq(joined.entry_count_original).all(), 'actual targets differ from original dataset')
    forecasts = []
    first_hour = {}
    from forecasting import validate_bundle
    ids = [s['station_id'] for s in shared['data']['stations'] if s['kind'] == 'line']
    for name, sha in bootstrap['contract']['forecast_hashes'].items():
        path = output / 'forecasts' / name
        require(file_hash(path) == sha, f'forecast cache changed: {name}')
        bundle = read(path)
        validate_bundle(bundle, ids, bundle['issued_at'])
        strategy = name.split('_')[0]
        source = bundle['source_versions']
        if strategy == 'ORACLE':
            require(source.get('operationally_available') is False, 'oracle was not labelled as research')
        else:
            require(not source.get('future_truth_intentionally_used'), 'causal forecast uses future truth')
        score = score_forecast(bundle, frame, bundle['issued_at'])
        if strategy == 'ORACLE':
            require(score['absolute_error'] == 0, 'oracle is not exact')
        forecasts.append(dict(strategy=strategy, issued_at=bundle['issued_at'], **score))
        # Score each station/quarter once; the overlapping second hour is separate.
        end = pd.Timestamp(bundle['issued_at']) + pd.Timedelta(hours=1)
        first = dict(bundle, rows=[r for r in bundle['rows'] if pd.Timestamp(r['interval_start']) < end])
        first_hour.setdefault(strategy, []).append(score_forecast(first, frame, bundle['issued_at']))
    require(len(forecasts) == 51, 'missing forecast cache')
    unique_scores = {}
    for strategy, scores in first_hour.items():
        error = sum(s['absolute_error'] for s in scores)
        total = sum(s['actual_entries'] for s in scores)
        rows = sum(s['rows'] for s in scores)
        require(len(scores) == 17 and rows == 1292, 'unique forecast coverage differs')
        unique_scores[strategy] = dict(rows=rows, actual_entries=total, absolute_error=error,
            wape_percent=100 * error / total, scope='07:00–24:00; each station/quarter once')
    for arm in arms:
        strategy = arm['strategy']
        folder = output / arm['run']['folder']
        data, motion, profiles, groups, state = [read(folder / n) for n in
            ('movement_input.json', 'motion.json', 'profiles.json', 'groups.json.gz', 'passenger_checkpoint.json.gz')]
        require(digest(groups) == arm['groups_digest'], f'actual passengers changed: {strategy}')
        runner = SimpleNamespace(motion=motion, profiles=profiles, by_id={r['group_id']: r for r in groups}, duration=motion['duration_seconds'])
        recomputed = _metrics(data, runner, state, shared['roster'], data['headway_policy'], 3600, 64800)
        recomputed['peak_trains_on_line'] = motion['checks']['peak_trains_on_line']
        _same(arm['metrics'], recomputed, f'{strategy} reported metrics')
        previous = shared['data']
        for decision in arm['decisions']:
            second = decision['decision_second']
            require(read(output / 'controller' / strategy / f'decision_{second:05}.json') == decision, 'saved decision trace differs')
            current = read(output / 'controller' / strategy / f'plan_{second:05}.json.gz')
            old = {(r['train_id'], r['trip_id'], r['leg_index']): r for r in previous.get('leg_timing', []) if r['departure_second'] < second}
            new = {(r['train_id'], r['trip_id'], r['leg_index']): r for r in current.get('leg_timing', [])}
            require(all(new.get(key) == row for key, row in old.items()), f'executed timing changed: {strategy}/{second}')
            previous = current
        if arm['decisions']:
            require(previous == data, f'final plan differs from final journal: {strategy}')
    write_json(output / 'runs.json', [a['run'] for a in arms])
    from goal43_verify import verify as verify_ledgers
    ledgers = verify_ledgers(output)
    preserved = check_preserved(output / 'preserved_before.json')
    old_preserved = check_preserved(ROOT / 'data/processed/metro_ml_verification/preserved_before.json')
    models = check_model_archive(ROOT / 'data/raw/metro/lgbm_final.zip', ROOT / 'data/processed/metro_ml_inputs')
    receipt = dict(status='passed', completed_daily_runs=4, controller_decisions=204,
        original_day_rows_checked=len(joined), forecast_cache_files_checked=len(forecasts),
        unique_forecast_scores=unique_scores, forecast_horizons=forecasts,
        preserved_files=preserved, previous_preserved_files=old_preserved, unchanged_model_files=models,
        frozen_source_files=len(campaign['source_hashes']), executed_plan_prefixes_checked=204,
        maximum_decision_seconds={a['strategy']:a['maximum_decision_seconds'] for a in arms},
        eight_minute_calculation_budget_met=all(a['maximum_decision_seconds'] <= 480 for a in arms),
        calculation_latency_scope='precomputed forecast read, timetable generation and validation; inference and external delivery excluded',
        notification_delivery_measured=False,
        oracle_decision_inputs_override='future truth intentionally provided; shared kernel decision_inputs label does not describe oracle availability',
        operational_violation_count=sum(len(r['violations']) for r in ledgers['runs']),
        integrity_pass_does_not_mean_policy_admissibility=True, cash_saving_rub=None, lost_revenue_rub=None,
        release_authorized=False)
    write_json(output / 'acceptance.json', receipt)
    return arms, receipt


def _csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _series(output, arm):
    from goal43_replay import _series as saved_series
    folder = output / arm['run']['folder']
    data = read(folder / 'movement_input.json')
    motion = read(folder / 'motion.json')
    groups = read(folder / 'groups.json.gz')
    runner = SimpleNamespace(motion=motion, by_id={r['group_id']:r for r in groups})
    state = read(folder / 'passenger_checkpoint.json.gz')
    return saved_series(runner, state, 3600, 64800, dict(rows=[]), data)


def report(output):
    output = Path(output).resolve()
    arms, acceptance = verify(output)
    baseline = arms[0]['metrics']
    rows, details, per_train = [], [], []
    for arm in arms:
        m, strategy = arm['metrics'], arm['strategy']
        statuses = Counter(d['selection']['status'] for d in arm['decisions'])
        actions = Counter(d['selection'].get('selected_candidate_id') or d['selection'].get('mitigation_candidate_id') or 'KEEP' for d in arm['decisions'])
        delta = metric_delta(baseline, m)
        delta['scope'] = 'full daily generated plan including identical warmup and closed depot movements; no extrapolation'
        row = dict(strategy=strategy, train_km=m['train_km'], km_per_fixed_train=m['mean_km_per_fixed_train'],
            observed_km_reduction=delta['observed_saving_train_km'], observed_km_reduction_per_train=delta['observed_saving_km_per_fixed_train'],
            mean_wait_seconds=m['mean_wait_seconds'], p95_wait_seconds=m['p95_wait_seconds'],
            max_queue=m['max_queue'], max_platform_direction_queue=m['max_platform_direction_queue'],
            max_car_load=m['max_car_load'], peak_trains_on_line=m['peak_trains_on_line'],
            bad_quarter_windows=sum(w['status']=='fail' for w in m['policy_exposure']['windows']),
            actual_service_admissible=m['actual_service_admissible'], all_arrivals_boarded=m['all_service_arrivals_completed'],
            late_unboarded=m['cohort']['unboarded'], no_feasible_decisions=statuses['no_feasible_candidate'],
            decisions=len(arm['decisions']), max_calculation_seconds=arm['maximum_decision_seconds'],
            cash_saving_rub=None, lost_revenue_rub=None)
        rows.append(row)
        details.append(dict(**row, conclusion=experiment_conclusion(baseline,m), decisions_by_status=dict(statuses),
            actions=dict(actions), metrics=m, delta=delta, series=_series(output,arm)))
        for r in delta['per_train']:
            per_train.append(dict(strategy=strategy,**r))
    _csv(output/'rows.csv',rows)
    _csv(output/'train_km.csv',per_train)
    result = dict(date='2026-09-30', actual_line=1, policy_validation_days=1,
        independent_generalization_claim=False, no_retraining=True, oracle_is_operationally_available=False,
        bounded_candidate_search=True, rows=rows, comparisons=details, acceptance=acceptance,
        cash_saving_rub=None, lost_revenue_rub=None, release_authorized=False)
    write_json(output/'report.json',result)
    markdown = ['# Суточное сравнение управления', '',
        '30 сентября 2026 года. Входной поток из загруженного датасета. Исходные веса модели сохранены.', '',
        '| Вариант | Км составов | Км на состав (68) | Ожидание, с | Окон с нарушением | Допустим |',
        '|---|---:|---:|---:|---:|---|']
    for r in rows:
        wait = 'не определено' if r['mean_wait_seconds'] is None else f"{r['mean_wait_seconds']:.2f}"
        markdown.append(f"| {r['strategy']} | {r['train_km']:.3f} | {r['km_per_fixed_train']:.3f} | {wait} | {r['bad_quarter_windows']} | {'да' if r['actual_service_admissible'] else 'нет'} |")
    markdown += ['', *[f"{d['strategy']}: {d['conclusion']}" for d in details[1:]], '',
        'Все варианты получают одинаковый поток, парк из 68 составов и состояние после разогрева. Начальные 33 состава находятся на линии в 06:00. Управление выполняется в 07:00–23:45 с шагом 15 минут. Прогноз обновляется раз в час и охватывает два часа. После полуночи прекращается новый вход; уже вошедшие пассажиры остаются в расчёте.', '',
        'Пробег включает разогрев и возврат в депо. Средний пробег делится на неизменные 68 составов; подробности каждого состава сохранены в train_km.csv. Ожидание считается для общей когорты 07:00–23:50. Пассажиры последних десяти минут учитываются отдельно, включая не севших в поезд.', '',
        'ORACLE получает точный будущий входной поток только как исследовательский контроль. Его результат показывает возможности выбранного набора действий в этой модели. Глобальный оптимум не вычислялся.', '',
        'Ограничения выводов: данные и веса относятся к красной линии 1, перенос на синюю линию 2 не выполнен. Часовой график взят за основу сценария, но его дата и применимость к 2026 году не подтверждены. Геометрия, распределение поездок S1, готовность резервов и повторный разворот в депо заданы допущениями. Планировщик использует запас вместимости линии в четыре состава и нижнюю границу 129 секунд для построения расписаний; физические проверки применяют пользовательские 113/150 секунд. Поэтому поиск не охватывает все допустимые плотные расписания.', '',
        'Срок обучения модели заканчивается 29 сентября 2026 года. Единственный день после обучения использован для отладки этих экспериментов, поэтому результат не является независимой проверкой стратегии на новых днях.', '',
        'Упущенная выручка и денежная экономия остаются неизвестными. Отложенная посадка не доказывает потерю оплаченной поездки. Эквивалент ресурса ТО1 в файле сравнения не является денежной экономией.', '',
        f"Проверка целостности: {acceptance['completed_daily_runs']} суточных журнала, {acceptance['controller_decisions']} решений, {acceptance['unchanged_model_files']} неизменных файлов модели. Нарушения пассажирских ограничений сохраняются в verification_report.json."]
    (output/'report.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    _html(output,result)
    return result


def _html(output,result):
    table = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in (
        r['strategy'],f"{r['train_km']:,.1f}",f"{r['km_per_fixed_train']:.2f}",
        'не определено' if r['mean_wait_seconds'] is None else f"{r['mean_wait_seconds']:.2f}",
        r['bad_quarter_windows'],'да' if r['actual_service_admissible'] else 'нет'))+'</tr>' for r in result['rows'])
    payload = json.dumps([dict(strategy=d['strategy'],series=d['series']) for d in result['comparisons']],ensure_ascii=False).replace('<','\\u003c')
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Суточные эксперименты метро</title><style>body{font:16px system-ui;margin:24px auto;max-width:1100px;padding:0 18px;color:#152334;background:#f6f8fc}h1{font-size:28px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #dce3ed}select{padding:8px;font-size:16px}svg{width:100%;background:white;border:1px solid #dce3ed;margin-top:16px}p{line-height:1.55}.legend{display:flex;gap:20px;margin:12px 0;flex-wrap:wrap}.note{background:#fff3df;padding:15px;border-radius:8px}a{color:#245cc4}#tip{min-height:26px}</style>
<h1>Суточные эксперименты, 30.09.2026</h1><p>Красная линия 1 · 19 станций · одинаковый пассажиропоток · 68 составов · 204 решения</p>
<p class="note">Результаты относятся к заданному сценарию. Снижение пробега при нарушении ограничений не подтверждает допустимую экономию. ORACLE использует точные будущие входы как исследовательский контроль.</p>
<table><thead><tr><th>Вариант</th><th>Км составов</th><th>Км на состав</th><th>Ожидание, с</th><th>Окон с нарушением</th><th>Допустим</th></tr></thead><tbody>TABLE</tbody></table>
<p>Показатель: <select id="metric"><option value="fleet">Составы на линии</option><option value="queue">Очередь на платформах</option><option value="load">Максимум пассажиров в вагоне за минуту</option><option value="headway">Максимальный интервал за минуту, с</option></select></p><div id="legend" class="legend"></div><svg id="chart" viewBox="0 0 1050 370" role="img" aria-label="Сравнение показателей по времени"></svg><div id="tip" aria-live="polite">Наведите курсор на график для сравнения.</div>
<p><a href="report.md">Описание и ограничения</a> · <a href="rows.csv">Итоговая таблица CSV</a> · <a href="train_km.csv">Пробег каждого состава CSV</a> · <a href="acceptance.json">Проверки</a></p>
<p>Ожидание: общая когорта 07:00–23:50. Пробег: полный план с разогревом и возвратами. Дата исходного часового графика не подтверждена. Готовность резервов и распределение поездок заданы допущениями. Денежный эффект не определён.</p>
<script>const arms=PAYLOAD,colors=['#27364a','#bd7a16','#215ee8','#9b3cc8'],fields={fleet:'count',queue:'queued',load:'load',headway:'seconds'};const chart=document.getElementById('chart'),sel=document.getElementById('metric');document.getElementById('legend').innerHTML=arms.map((a,i)=>`<span style="color:${colors[i]}">● ${a.strategy}</span>`).join('');function points(a){return a.series[sel.value]}function val(p){return p[fields[sel.value]]}function clock(s){let m=Math.floor((21600+s)/60);return `${Math.floor(m/60)%24}`.padStart(2,'0')+':'+`${m%60}`.padStart(2,'0')}function draw(){let max=Math.max(1,...arms.flatMap(a=>points(a).map(val))),x=s=>60+(s-3600)/61200*960,y=v=>320-v/max*270;let grid='';for(let n=0;n<=5;n++){let v=max*n/5;grid+=`<line x1="60" x2="1020" y1="${y(v)}" y2="${y(v)}" stroke="#e1e7f0"/><text x="52" y="${y(v)+5}" text-anchor="end" font-size="12">${Math.round(v)}</text>`}for(let s=3600;s<=64800;s+=10800){grid+=`<text x="${x(s)}" y="350" text-anchor="middle" font-size="13">${clock(s)}</text>`}chart.innerHTML=grid+arms.map((a,i)=>`<polyline fill="none" stroke="${colors[i]}" stroke-width="2" points="${points(a).map(p=>`${x(p.second)},${y(val(p))}`).join(' ')}"/>`).join('');chart.onmousemove=e=>{let bounds=chart.getBoundingClientRect(),s=3600+Math.max(0,Math.min(1,((e.clientX-bounds.left)/bounds.width*1050-60)/960))*61200;document.getElementById('tip').textContent=clock(s)+' · '+arms.map(a=>{let ps=points(a),p=ps.reduce((best,p)=>Math.abs(p.second-s)<Math.abs(best.second-s)?p:best,ps[0]);return a.strategy+': '+(p?val(p):'нет данных')}).join(' · ')}}sel.onchange=draw;draw();</script></html>'''
    (output/'report.html').write_text(page.replace('TABLE',table).replace('PAYLOAD',payload),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    result=report(parser.parse_args().output)
    print(json.dumps(dict(status=result['acceptance']['status'],rows=result['rows']),ensure_ascii=False))
