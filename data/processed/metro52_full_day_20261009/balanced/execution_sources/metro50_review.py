"""Audit finished DAY50 artefacts and report admissible paired maintenance economics."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import csv
import gzip
import hashlib
import html
import json
import math
from pathlib import Path

from metro50_economics import economy_comparison, maintenance_rates
from metro50_peak_service import is_peak, peak_boarding_service


ARMS = ('reference', 'current5', 'dense10')
SIDES = ('without_sr_kr', 'with_sr_kr')
ENTRY_END, SERVICE_END, EXPECTED_ENTRIES = 64800, 65400, 758936
HASH_CACHE = {}


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def sha256(path):
    path = Path(path)
    stat = path.stat()
    key = str(path), stat.st_size, stat.st_mtime_ns
    if key not in HASH_CACHE:
        HASH_CACHE[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return HASH_CACHE[key]


def require_finished_results(root):
    missing = [arm for arm in ARMS if not (Path(root)/arm/'result.json').is_file()]
    if missing:
        raise FileNotFoundError('Unfinished daily results: '+', '.join(missing))


def audit_receipts(result):
    contract, rows = result['contract'], result['decisions']
    expected = [] if contract['arm'] == 'REFERENCE' else list(range(3600, ENTRY_END, 900))
    violations = []
    if contract.get('decision_times') != expected or contract.get('all_control_steps') is not True:
        violations.append('contract does not contain the complete control schedule')
    if [r.get('decision_second') for r in rows] != expected:
        violations.append('missing, repeated or unordered decision receipts')
    times = []
    for row in rows:
        seconds = row.get('pipeline_seconds')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 480:
            violations.append(f"invalid decision duration at {row.get('decision_second')}")
        else:
            times.append(seconds)
        if row.get('local_recommendation_within_480') is not True:
            violations.append(f"unconfirmed local receipt at {row.get('decision_second')}")
    return dict(passed=not violations, count=len(rows), expected_count=len(expected),
                maximum_seconds=max(times, default=None), violations=violations,
                external_notification_delivery_measured=False)


def nearest_rank(values, quantile):
    if not 0 < quantile <= 1:
        raise ValueError('quantile must be in (0,1]')
    ordered = sorted(values)
    return ordered[math.ceil(quantile*len(ordered))-1] if ordered else None


def peak_car_quantiles(runner, state, *, end_second=SERVICE_END):
    origin = datetime.fromisoformat(runner.motion['period_start'])
    loads, by_departure = [], defaultdict(set)
    for row in state['load_events']:
        event = runner.motion_by_id[row['motion_event_id']]
        if (row['event_type'] != 'departure' or not event.get('passenger_service')
                or not 0 <= row['second'] < end_second or not is_peak(origin+timedelta(seconds=row['second']))):
            continue
        loads.append(row['load'])
        by_departure[row['motion_event_id'], row['train_id']].add(row['car_id'])
    complete = (sum(len(cars) for cars in by_departure.values()) == len(loads)
                and all(cars == {c['car_id'] for c in runner.profiles[train]}
                        for (_, train), cars in by_departure.items()))
    return dict(observations=len(loads), departures=len(by_departure), p95=nearest_rank(loads, .95),
                p99=nearest_rank(loads, .99), maximum=max(loads, default=None),
                zero_observations=loads.count(0), complete_car_coverage=complete,
                scope='each car after boarding at weekday peak passenger departures; zero loads included; nearest rank',
                peak_windows='08:00-10:00 and 17:00-19:00 Europe/Moscow',
                occupied_car_time_weighted=False)


def report_saving(side, admissible, field):
    return side[field] if admissible else None


def audit_arm(folder):
    from metro44_catalog import physical_gate
    from metro44_focus import execution_prefix
    from metro_ml_compare import _metrics
    from passenger_groups import GroupedPassengerSimulation
    from simulation import digest, simulate

    folder = Path(folder)
    result, contract = read(folder/'result.json'), read(folder/'contract.json')
    common = read(folder/'common_scenario.json.gz')
    run = folder/result['run']['folder']
    data, motion, profiles = (read(run/name) for name in ('movement_input.json', 'motion.json', 'profiles.json'))
    groups, state = read(run/'groups.json.gz'), read(run/'passenger_checkpoint.json.gz')
    checks, errors = {}, []
    checks['result_contract_parity'] = result['contract'] == contract
    checks['full_day_contract'] = (contract.get('full_day') is True and contract.get('all_control_steps') is True
        and result.get('full_day') is True and result.get('complete_control') is True
        and contract.get('fixed_fleet_size') == 68 and contract.get('line_id') == 1
        and contract.get('entry_end_second') == ENTRY_END and contract.get('service_end_second') == SERVICE_END
        and contract.get('boarding_seed') == 7 and contract.get('boarding_capacity_per_car') == 150)
    source_failures = [p for p, value in contract['source_hashes'].items()
                       if not Path(p).is_file() or sha256(p) != value]
    checks['live_source_hashes_preserved'] = not source_failures and result.get('sources_preserved') is True
    checks['snapshot_manifest_matches'] = read(folder/'execution_sources'/'manifest.json') == contract['source_hashes']
    snapshot_failures = [p for p, value in contract['source_hashes'].items()
                        if Path(p).suffix == '.py' and Path(p).parent == Path(__file__).resolve().parent
                        and (not (folder/'execution_sources'/Path(p).name).is_file()
                             or sha256(folder/'execution_sources'/Path(p).name) != value)]
    checks['executed_code_snapshot_preserved'] = not snapshot_failures
    group_sort = lambda rows: sorted(rows, key=lambda g: g['group_id'])
    checks['common_groups_preserved'] = (digest(common['groups']) == contract['groups_digest']
                                        and group_sort(groups) == group_sort(common['groups']))
    checks['fleet_and_geometry_preserved'] = (all(data[k] == common['data'][k]
        for k in ('trains', 'stations', 'tracks', 'headway_policy')) and profiles == common['profiles'])
    roster = common['roster']
    checks['fixed_roster'] = (len(roster) == len({r['train_id'] for r in roster}) == 68
        and Counter(r['type'] for r in roster) == Counter(Yubileyny=31, Baltiets=37)
        and all(t['cars'] == 8 for t in data['trains']))
    checks['maintenance_objective_rates_current'] = contract.get('maintenance_rates_rub_per_km') == {
        kind: rate['without_sr_kr'] for kind, rate in maintenance_rates().items()}
    receipts = audit_receipts(result)
    checks['complete_control_receipts'] = receipts['passed']
    checks['saved_decision_journals_match'] = all(read(folder/'controller'/f"decision_{r['decision_second']:05}.json") == r
                                                for r in result['decisions'])
    compiled = simulate(data, data['plans'][0]['plan_id'])
    checks['saved_motion_parity'] = digest(compiled) == digest(motion)
    runner = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], groups, profiles,
                                        boarding_seed=7, boarding_capacity_per_car=150)
    checks['passenger_motion_parity'] = digest(runner.motion) == digest(motion)
    try:
        runner._validate_checkpoint(state, state['second'])
        checks['final_checkpoint_integrity'] = True
    except (ValueError, KeyError, TypeError) as error:
        checks['final_checkpoint_integrity'] = False
        errors.append(str(error))
    checks['executed_prefix_receipts_match'] = all(
        execution_prefix(motion, r['decision_second']) == r['executed_prefix_digest'] for r in result['decisions'])
    physical = physical_gate(data, 0, SERVICE_END, reference_motion=motion)
    checks['independent_physical_gate'] = physical['movement_passed'] is True
    metrics = _metrics(data, runner, state, roster, data['headway_policy'], 0, SERVICE_END)
    peak = peak_boarding_service(runner, state, end_second=ENTRY_END,
                                missed_share_limit=contract['settings']['missed_share_limit'], require_next_train=False)
    quantiles = peak_car_quantiles(runner, state)
    checks.update(peak_boarding_service=peak['passed'] is True,
        complete_peak_car_ledger=quantiles['complete_car_coverage'],
        independent_exposure=metrics['policy_exposure']['admissible'] is True,
        independent_headways=metrics['headways']['passed'] is True,
        all_entries_completed=state['entered'] == state['completed'] == sum(g['count'] for g in groups) == EXPECTED_ENTRIES,
        empty_final_passenger_state=state['queued'] == state['approaching'] == state['onboard_count'] == 0,
        mature_cohort_complete=metrics['mature_cohort']['complete'] is True,
        complete_service_cohort=metrics['cohort']['complete'] is True,
        closed_movement_obligations=motion['obligations_closed'] is True,
        complete_observation=state['second'] == motion['duration_seconds'] == 72000,
        reported_admissibility=result['metrics']['actual_service_admissible'] is True,
        saved_metric_scope_parity=all(result['metrics'][k] == metrics[k] for k in
            ('train_km', 'per_train', 'mean_wait_seconds', 'p95_wait_seconds', 'entered', 'completed', 'max_car_load')))
    metrics['peak_boarding_service'], metrics['peak_car_quantiles'] = peak, quantiles
    initial_crossings = len(motion['boundary_crossing_movements'])
    proof_files = ['result.json', 'contract.json', 'common_scenario.json.gz']
    artefacts = {name: sha256(folder/name) for name in proof_files}
    artefacts.update({f"{result['run']['folder']}/{name}": sha256(run/name)
                     for name in ('movement_input.json', 'motion.json', 'profiles.json', 'groups.json.gz', 'passenger_checkpoint.json.gz')})
    return dict(passed=all(checks.values()), checks=checks, errors=errors,
        failed_checks=[k for k, passed in checks.items() if not passed],
        sources_changed=source_failures, executed_snapshots_changed=snapshot_failures,
        controller=receipts, physical={k: v for k, v in physical.items() if not k.startswith('_')},
        metrics=metrics, artefact_sha256=artefacts, elapsed_run_seconds=result['elapsed_seconds'],
        initial_boundary_movements=initial_crossings, source_mileage_complete=motion['mileage_complete'],
        initial_fleet_digest=digest(data['trains']), groups_digest=digest(group_sort(groups)),
        profiles_digest=digest(profiles), common_scenario_digest=digest(common),
        source_hashes=contract['source_hashes'], contract=contract,
        full_passenger_replay_performed=False, checkpoint_verified_against_recompiled_runner=True)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _format(value, digits=2):
    if value is None:
        return 'не подтверждено'
    if isinstance(value, bool):
        return 'да' if value else 'нет'
    if isinstance(value, (int, float)):
        return f'{value:,.{digits}f}'.replace(',', '\u00a0')
    return html.escape(str(value))


def _table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+_format(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def html_report(comparison):
    audits, economics = comparison['audits'], comparison['economics']
    pieces = ["""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Метро: дневной эксперимент</title>
<style>body{font:16px/1.5 system-ui;margin:35px auto;max-width:1250px;padding:0 24px;color:#182535}h1,h2{line-height:1.2}table{width:100%;border-collapse:collapse;margin:16px 0 30px;font-size:14px}th,td{padding:9px;border:1px solid #cbd5e1;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf2f7}p{max-width:1000px}.note{background:#fff6da;padding:14px}.ok{background:#e5f5e9;padding:14px}a{color:#1264b0}@media print{body{margin:0;max-width:none}table{break-inside:avoid}}</style>
<h1>Красная линия: полный день 30 сентября 2026 года</h1>
<p>Один поток пассажиров, 68 составов (37 «Балтиец», 31 «Юбилейный»), 8 вагонов, одинаковый seed 7.
Вход пассажиров 06:00–00:00; проверка интервалов движения до00:10; закрытие движения и пассажирских обязательств до02:00.
Управление каждые15минут с07:00 до23:45 — 68 решений для каждого оптимизатора.</p>""".replace('до00', 'до 00').replace('до02', 'до 02').replace('каждые15', 'каждые 15').replace('с07', 'с 07').replace('до23', 'до 23')]
    all_passed = comparison['paired_inputs_passed'] and all(a['passed'] for a in audits.values())
    pieces.append('<p class="'+('ok' if all_passed else 'note')+'">'+(
        'Все три прогона прошли независимые проверки сохранённых результатов.' if all_passed else
        'Есть отклонённые или несопоставимые результаты. Их рассчитанное снижение стоимости не считается допустимой экономией.')+'</p>')
    service_rows = []
    for arm, audit in audits.items():
        m, p = audit['metrics'], audit['metrics']['peak_boarding_service']
        q, quarters = m['peak_car_quantiles'], p['intervals']
        service_rows.append([arm, audit['passed'], m['entered'], m['completed'], m['train_km'],
            m['mean_wait_seconds'], m['p95_wait_seconds'], q['p95'], q['p99'], q['maximum'],
            100*max((r['missed_first_share'] or 0 for r in quarters), default=0),
            p['aggregate']['second_train_missed_count'], m['max_platform_direction_queue'],
            audit['controller']['maximum_seconds']])
    pieces.append('<h2>Обслуживание пассажиров и проверка ограничений</h2>'+_table(
        ['Сценарий', 'Допустим', 'Вошли', 'Завершили', 'Км составов', 'Ожидание, среднее с', 'Ожидание p95, с',
         'Пик p95 чел/вагон', 'Пик p99', 'Пик максимум', 'Худший квартал: пропуск первого, %',
         'Пропустили второй, чел', 'Макс. очередь станция/направление', 'Макс. решение, с'], service_rows))
    pieces.append('<p>Нагрузки p95/p99: каждый вагон после посадки при пассажирском отправлении в будние08:00–10:00 и17:00–19:00; нулевые нагрузки включены, ближайший верхний ранг. '
        'Ожидание: все вошедшие за06:00–00:00, время на платформе до посадки. Пропуски: уникальные пассажиры в квартале прибытия на платформу; повторные попытки не подменяют число людей. '
        'current5 допускает до5%, dense10 до10% пропуска первого подходящего поезда в каждом пиковом квартале. Пропуск второго приведён отдельно. '
        'Принудительной случайной задержки пассажиров нет; окончательная очередь, подходящие к платформе и пассажиры в составах должны быть нулевыми.</p>'.replace('будние08', 'будние 08').replace('и17', 'и 17').replace('за06', 'за 06').replace('до5', 'до 5').replace('до10', 'до 10'))
    for period, title in [('daily', 'Стоимость и экономия за этот день'), ('annual', '250 повторений этого дня: годовой сценарий')]:
        rows = []
        for arm, economy in economics.items():
            e, valid = economy[period], economy['admissible']
            for field, label in [('reference_cost_rub', 'Исходная стоимость'), ('candidate_cost_rub', 'Стоимость сценария'),
                                 ('saving_rub', 'Допустимая экономия'), ('saving_percent', 'Экономия, %'),
                                 ('mean_reference_cost_per_fixed_train_rub', 'Исходная стоимость на каждый состав'),
                                 ('mean_candidate_cost_per_fixed_train_rub', 'Стоимость сценария на каждый состав'),
                                 ('mean_saving_per_fixed_train_rub', 'Экономия на один из68 составов')]:
                rows.append([arm+' — '+label.replace('из68', 'из 68')]+[
                    report_saving(e[s], valid, field) if 'saving' in field else e[s][field] for s in SIDES])
        pieces.append('<h2>'+title+'</h2>'+_table(['Показатель', 'Без СР и КР', 'С СР и КР'], rows))
    rows = []
    for arm, economy in economics.items():
        e = economy['daily']
        for field, label in [('reference_rub_per_km', 'Исходная стоимость, руб/км'),
                             ('candidate_rub_per_km', 'Стоимость сценария, руб/км'),
                             ('saving_rub_per_saved_km', 'Экономия, руб/снятый км'),
                             ('saving_rub_per_reference_km', 'Экономия, руб/исходный км')]:
            rows.append([arm+' — '+label]+[report_saving(e[s], economy['admissible'], field)
                         if 'saving' in field else e[s][field] for s in SIDES])
    pieces.append('<h2>Стоимость и экономия на километр</h2>'+_table(['Показатель', 'Без СР и КР', 'С СР и КР'], rows))
    for period, title in [('daily', 'Пределы расчёта на день'), ('annual', 'Пределы расчёта на250 повторений')]:
        rows = []
        for arm, economy in economics.items():
            for label, low, high in [('Исходная стоимость, руб', 'reference_cost_rub_min', 'reference_cost_rub_max'),
                                    ('Стоимость сценария, руб', 'candidate_cost_rub_min', 'candidate_cost_rub_max'),
                                    ('Допустимая экономия, руб', 'saving_rub_min', 'saving_rub_max'),
                                    ('Экономия, %', 'saving_percent_min', 'saving_percent_max')]:
                cells = []
                for side in SIDES:
                    bound = economy[period][side]['uncertainty']
                    cells.append((_format(bound[low])+' … '+_format(bound[high]))
                        if 'saving' not in low or economy['admissible'] else None)
                rows.append([arm+' — '+label]+cells)
        pieces.append('<h2>'+title.replace('на250', 'на 250')+'</h2>'+_table(['Диапазон', 'Без СР и КР', 'С СР и КР'], rows))
    pieces.append('<p>±5% применяется только к вычисленному пробегу каждого сравниваемого плана; ошибки двух планов могут иметь противоположные знаки. '
        '±20% применяется только к ценам СР и КР; одни цены используются для обоих планов. Интервалы обслуживания не меняются. '
        'Диапазоны являются границами чувствительности, а не доверительными интервалами.</p>')
    percent_rows = [[arm]+[report_saving(e['daily'][s], e['admissible'], 'saving_percent') for s in SIDES]
                    for arm, e in economics.items()]
    pieces.append('<h2>Проценты экономии</h2>'+_table(['Сценарий', 'Без СР и КР, %', 'С СР и КР, %'], percent_rows))
    quarter_rows = [[arm, r['interval_start'][11:16], r['passenger_count'],
        100*r['missed_first_share'] if r['missed_first_share'] is not None else None,
        r['second_train_missed_count'], r['status']] for arm, audit in audits.items()
        for r in audit['metrics']['peak_boarding_service']['intervals']]
    pieces.append('<h2>Пиковые кварталы</h2>'+_table(['Сценарий', 'Время', 'Пассажиров', 'Пропуск первого, %',
        'Пропуск второго, чел', 'Проверка первого'], quarter_rows))
    pieces.append('<h2>Границы применимости</h2><p>Исходный график и FIFO-посадка — расчётная конструкция, не подтверждённый рабочий график метро. '
        'Распределение поездок по направлениям и назначениям S1, случайная посадка и инфраструктура отстоя — инженерные допущения. '
        'Одно и то же зерно случайности помогает сравнивать сценарии, но не заменяет проверку на других днях и зернах. '
        'В начальном состоянии27 составов уже движутся; общий arrival-based ledger одинаково учитывает эти начальные перегоны. '
        'Данные этого дня переиспользованы при разработке, поэтому это дневная проверка, а не новый независимый статистический тест.</p>'.replace('состоянии27', 'состоянии 27'))
    pieces.append('<p>Стоимость — суммарное начисление ресурса ТО1, ТО2, ТР1, ТР2, ТР3 по километражу; СР и КР добавляются во второй колонке. '
        'Цены и интервалы взяты из условий пользователя, суммы считаются аддитивными. Экономия означает уменьшение расхода ремонтного ресурса, '
        'а не гарантированное уменьшение годовых платежей.250 повторений — отдельный сценарий; сезонность, выходные и аномалии не проверены. '
        'Энергия, зарплаты, кассовая экономия, отказы пассажиров и упущенный доход не оценены. Местное время расчёта рекомендации проверено; доставка уведомлений не измерена.</p>'.replace('платежей.250', 'платежей. 250'))
    pieces.append('<p><a href="comparison.json">Аудит и расчёты JSON</a> · <a href="economics.csv">Объединённые таблицы стоимости и экономии CSV</a> · '
        '<a href="per_train.csv">Каждый состав:68 строк на сценарий</a></p>'.replace('состав:68', 'состав: 68'))
    pieces.append('</html>')
    return '\n'.join(pieces)


def review_all(root):
    from simulation import digest
    root = Path(root).resolve()
    require_finished_results(root)
    audits = {arm: audit_arm(root/arm) for arm in ARMS}
    base = audits['reference']
    paired = all(all(a[key] == base[key] for key in
        ('initial_fleet_digest', 'groups_digest', 'profiles_digest', 'common_scenario_digest', 'source_hashes'))
        for a in audits.values())
    economics = {arm: economy_comparison(base['metrics'], audit['metrics'], annual_days=250,
        admissible=paired and base['passed'] and audit['passed']) for arm, audit in audits.items()}
    comparison = dict(version='day50-independent-review-v1', paired_inputs_passed=paired,
        audits=audits, economics=economics, all_results_admissible=paired and all(a['passed'] for a in audits.values()),
        annual_repetition_is_scenario=True, cash_saving_rub=None, lost_revenue_rub=None,
        review_module_sha256=sha256(__file__), review_digest=None)
    comparison['review_digest'] = digest({k: v for k, v in comparison.items() if k != 'review_digest'})
    rows, per_train = [], []
    for arm, economy in economics.items():
        for period in ('daily', 'annual'):
            e = economy[period]
            row = dict(scenario=arm, period=period, admissible=economy['admissible'],
                reference_km=e['reference_km'], candidate_km=e['candidate_km'], saved_km=e['saved_km'])
            for field in ('reference_cost_rub', 'candidate_cost_rub', 'saving_rub', 'saving_percent',
                          'mean_reference_cost_per_fixed_train_rub', 'mean_candidate_cost_per_fixed_train_rub',
                          'mean_saving_per_fixed_train_rub', 'saving_rub_per_saved_km', 'saving_rub_per_reference_km'):
                for side in SIDES:
                    row[field+'_'+side] = report_saving(e[side], economy['admissible'], field) if 'saving' in field else e[side][field]
            for side in SIDES:
                for field, value in e[side]['uncertainty'].items():
                    row[field+'_'+side] = value if 'saving' not in field or economy['admissible'] else None
            rows.append(row)
        for item in economy['per_train']:
            row = dict(scenario=arm, train_id=item['train_id'], train_type=item['train_type'], admissible=economy['admissible'])
            for period in ('daily', 'annual'):
                for field in ('reference_km', 'candidate_km', 'saved_km'):
                    row[period+'_'+field] = item[period][field]
                for side in SIDES:
                    for field in ('reference_cost_rub', 'candidate_cost_rub', 'saving_rub', 'saving_percent'):
                        row[period+'_'+field+'_'+side] = report_saving(item[period][side], economy['admissible'], field) if 'saving' in field else item[period][side][field]
            per_train.append(row)
    output = root/'review'
    if output.exists():
        raise FileExistsError('Preserve existing review; choose an isolated result root or archive review explicitly')
    output.mkdir()
    (output/'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_csv(output/'economics.csv', rows)
    _write_csv(output/'per_train.csv', per_train)
    (output/'report.html').write_text(html_report(comparison), encoding='utf-8')
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    result = review_all(args.root)
    print(json.dumps(dict(all_results_admissible=result['all_results_admissible'],
        paired_inputs_passed=result['paired_inputs_passed'],
        arms={arm: dict(passed=a['passed'], failed_checks=a['failed_checks']) for arm, a in result['audits'].items()}), ensure_ascii=False))
