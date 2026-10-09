"""Service-gated comparison, export, and independent journal verification."""

import argparse
import csv
import html
import json
import math
from pathlib import Path

from economics import write_json
from metro44_selection import economy_delta, resource_value


def service_valid(metrics):
    return (metrics.get('actual_service_admissible') is True
            and metrics.get('movement_obligations_closed') is True
            and metrics.get('cohort', {}).get('complete') is True
            and metrics.get('mature_cohort', {}).get('complete') is True
            and metrics.get('headways', {}).get('passed') is True
            and metrics.get('policy_exposure', {}).get('admissible') is True
            and metrics.get('terminal_resources_passed') is True
            and metrics.get('mean_wait_seconds') is not None
            and metrics.get('p95_wait_seconds') is not None)


def compare_results(results, reference_strategy='STANDARD', *, alternate_wait_budget=None):
    if alternate_wait_budget is not None and any(
            isinstance(alternate_wait_budget.get(key), bool)
            or not isinstance(alternate_wait_budget.get(key), (int, float))
            or not math.isfinite(alternate_wait_budget[key]) or alternate_wait_budget[key] < 0
            for key in ('mean_seconds', 'p95_seconds')):
        raise ValueError('alternate waiting budgets must be finite and nonnegative')
    rows = list(results)
    if not rows or len({r['strategy'] for r in rows}) != len(rows):
        raise ValueError('distinct results required')
    if len({(r['groups_digest'], r['checkpoint_digest']) for r in rows}) != 1:
        raise ValueError('comparison requires shared passengers and initial checkpoint')
    baseline = next((r for r in rows if r['strategy'] == reference_strategy), None)
    if baseline is None:
        raise ValueError('reference strategy is missing')
    base = baseline['metrics']
    base_valid = service_valid(base)
    comparisons, summaries = [], []
    for result in rows:
        metrics = result['metrics']
        resources = resource_value(metrics['per_train'])
        if abs(resources['train_km']-metrics['train_km']) > 1e-6:
            raise ValueError('per-train mileage differs from total')
        valid = service_valid(metrics)
        summaries.append(dict(strategy=result['strategy'], service_valid=valid,
            train_km=resources['train_km'], km_per_fixed_train=resources['mean_km_per_fixed_train'],
            to1_resource_used_rub=resources['to1_resource_equivalent_rub'],
            mean_wait_seconds=metrics.get('mean_wait_seconds'), p95_wait_seconds=metrics.get('p95_wait_seconds'),
            unboarded=metrics.get('cohort', {}).get('unboarded'),
            bad_fifteen_minute_windows=sum(w.get('status') == 'fail' for w in metrics.get('policy_exposure', {}).get('windows', [])),
            maximum_decision_seconds=result.get('maximum_decision_seconds'),
            research_oracle=result.get('research_oracle', result['strategy'] == 'MPC_ORACLE')))
        if result is baseline:
            continue
        reasons = []
        if not base_valid: reasons.append('reference_service_inadmissible')
        if not valid: reasons.append('candidate_service_inadmissible_or_pending')
        if baseline.get('economic_claim_requires_common_continuation') or result.get('economic_claim_requires_common_continuation'):
            left, right = baseline.get('common_continuation', {}), result.get('common_continuation', {})
            if not (left.get('verified') is True and right.get('verified') is True
                    and left.get('contract_digest') and left.get('contract_digest') == right.get('contract_digest')):
                reasons.append('common_continuation_not_verified')
        mean_delta = None if base.get('mean_wait_seconds') is None or metrics.get('mean_wait_seconds') is None else metrics['mean_wait_seconds']-base['mean_wait_seconds']
        p95_delta = None if base.get('p95_wait_seconds') is None or metrics.get('p95_wait_seconds') is None else metrics['p95_wait_seconds']-base['p95_wait_seconds']
        if mean_delta is None or mean_delta > 1e-9: reasons.append('mean_wait_budget_not_met')
        if p95_delta is None or p95_delta > 1e-9: reasons.append('p95_wait_budget_not_met')
        delta = economy_delta(base['per_train'], metrics['per_train'], service_admissible=base_valid and valid)
        if delta['observed_reduction_train_km'] <= 1e-9: reasons.append('no_mileage_reduction')
        if delta['to1_resource_equivalent_rub'] <= 1e-9: reasons.append('no_to1_resource_reduction')
        accepted = not reasons
        conditional_reasons = [reason for reason in reasons
                               if reason not in ('mean_wait_budget_not_met', 'p95_wait_budget_not_met')]
        bounds = result.get('window_savings_bounds', {})
        if alternate_wait_budget is None:
            conditional_reasons.append('alternate_wait_budget_not_declared')
        elif any(metrics.get(key) is None or metrics[key] > alternate_wait_budget[budget]
                 for key, budget in (('mean_wait_seconds', 'mean_seconds'), ('p95_wait_seconds', 'p95_seconds'))):
            conditional_reasons.append('alternate_wait_budget_not_met')
        if not (bounds.get('verified') is True and bounds.get('saving_train_km_lower_bound', 0) > 0
                and bounds.get('saving_to1_resource_rub_lower_bound', 0) > 0):
            conditional_reasons.append('positive_window_savings_not_verified')
        delta['accepted_saving_train_km'] = delta['observed_reduction_train_km'] if accepted else None
        comparisons.append(dict(delta, strategy=result['strategy'], reference=reference_strategy,
            accepted_improvement=accepted, reasons=reasons,
            mean_wait_delta_seconds=mean_delta, p95_wait_delta_seconds=p95_delta,
            conditional_development_improvement=not conditional_reasons,
            conditional_reasons=conditional_reasons, window_savings_bounds=bounds,
            research_oracle=result.get('research_oracle', result['strategy'] == 'MPC_ORACLE')))
    eligible = [r for r in comparisons if r['accepted_improvement'] and not r['research_oracle']]
    best = max(eligible, key=lambda r:r['to1_resource_equivalent_rub'], default=None)
    conditional = [r for r in comparisons if r['conditional_development_improvement'] and not r['research_oracle']]
    best_conditional = max(conditional, key=lambda r:r['window_savings_bounds']['saving_to1_resource_rub_lower_bound'], default=None)
    return dict(reference_strategy=reference_strategy, reference_service_valid=base_valid,
        rows=summaries, comparisons=comparisons, best_accepted_strategy=best['strategy'] if best else None,
        best_conditional_strategy=best_conditional['strategy'] if best_conditional else None,
        alternate_wait_budget=alternate_wait_budget,
        alternate_budget_status='development assumption; not a user requirement' if alternate_wait_budget else None,
        confirmed_cash_savings=False, development_period_used=True,
        independent_holdout_verified=False, actual_simulated_line_id=1,
        blue_line_transfer_completed=False, incomplete_acceptance=best is None,
        money_scope='TO1 wear-resource equivalent; avoided cash payments and lost paid trips are unknown',
        service_gate='complete passenger cohort, policy exposure, movement, mean and p95 no worse',
        fixed_fleet_denominator=68)


def report(output, *, reference_strategy='STANDARD', verify=True, alternate_wait_budget=None):
    output = Path(output)
    results = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(output.glob('*_result.json'))]
    compared = compare_results(results, reference_strategy, alternate_wait_budget=alternate_wait_budget)
    write_json(output/'comparison.json', compared)
    with (output/'train_km.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['strategy', 'train_id', 'train_type', 'km',
                                                   'to1_rub_per_km', 'to1_resource_equivalent_rub'])
        writer.writeheader()
        for result in results:
            for row in resource_value(result['metrics']['per_train'])['per_train']:
                writer.writerow(dict(strategy=result['strategy'], **row))
    records = [r['run'] for r in results]
    write_json(output/'runs.json', records)
    verification = None
    if verify:
        from goal43_verify import verify as independent_verify
        verification = independent_verify(output)
    acceptance = dict(compared, independent_journal_verification=verification,
        goal_status='accepted_development_improvement' if compared['best_accepted_strategy'] else
            'conditional_development_improvement' if compared['best_conditional_strategy'] else 'no_accepted_economic_improvement',
        sources_and_latency_require_separate_contract_verification=True)
    write_json(output/'acceptance.json', acceptance)
    lines = ['# Результаты GOAL 4.4', '', f'Эталон: {reference_strategy}.', '',
        '| Стратегия | км всего | км/68 | Ожидание, с | p95, с | Допустима |',
        '|---|---:|---:|---:|---:|---|']
    for row in compared['rows']:
        fmt = lambda value: 'неизвестно' if value is None else f'{value:.2f}'
        lines.append(f"| {row['strategy']} | {row['train_km']:.3f} | {row['km_per_fixed_train']:.3f} | {fmt(row['mean_wait_seconds'])} | {fmt(row['p95_wait_seconds'])} | {'да' if row['service_valid'] else 'нет'} |")
    lines.extend(['', f"Принятый результат: {compared['best_accepted_strategy'] or 'нет допустимого улучшения по всем условиям'}.", '',
        'Рубли в оценке ТО1 отражают ресурс обслуживания. Подтверждённые денежные сбережения, упущенная выручка и изменение прибыли неизвестны.', '',
        'Данные относятся к линии 1. День 30 сентября использован при разработке; независимая проверка и перенос на синюю линию не выполнены.', '',
        'В коротких опытах конец наблюдений не означает закрытие метро. Экономию от обрезанного хвоста нельзя принимать без общего продолжения пассажиропотока и движения.', '',
        'ORACLE получает будущие входы для исследовательского сравнения и недоступен диспетчеру.'])
    for row in compared['comparisons']:
        lines.extend(['', f"{row['strategy']}: изменение пробега {row['observed_reduction_train_km']:.3f} км; "
            f"эквивалент ресурса ТО1 {row['to1_resource_equivalent_rub']:.2f} руб. "
            f"{'Все условия сравнения выполнены.' if row['accepted_improvement'] else 'Причины непринятия: '+', '.join(row['reasons'])+'.'}"])
    if alternate_wait_budget:
        lines.extend(['', f"Исследовательский бюджет ожидания: среднее {alternate_wait_budget['mean_seconds']} с, p95 {alternate_wait_budget['p95_seconds']} с. Это допущение разработки, а не требование пользователя.",
            f"Допустимый компромисс с подтверждённым сокращением пробега внутри общего периода: {compared['best_conditional_strategy'] or 'не найден'}."])
    markdown = '\n'.join(lines)+'\n'
    (output/'report.md').write_text(markdown, encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for row in compared['rows']:
        if row['mean_wait_seconds'] is None: continue
        ax.scatter(row['mean_wait_seconds'], row['to1_resource_used_rub'],
            marker='o' if row['service_valid'] else 'x', s=65)
        ax.annotate(row['strategy'], (row['mean_wait_seconds'], row['to1_resource_used_rub']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set(xlabel='Среднее ожидание, с', ylabel='Расход ресурса ТО1, руб. эквивалента',
           title='Круг: допустимый режим. Крест: нарушение условий обслуживания.')
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output/'pareto.svg'); plt.close(fig)
    payload = json.dumps(compared, ensure_ascii=False).replace('</', '<\\/')
    options = ''.join(f'<option>{html.escape(r["strategy"])}</option>' for r in compared['rows'])
    table = ''.join(f'<tr><td>{html.escape(r["strategy"])}</td><td>{r["train_km"]:.3f}</td>'
        f'<td>{r["km_per_fixed_train"]:.3f}</td><td>{r["mean_wait_seconds"]}</td>'
        f'<td>{r["p95_wait_seconds"]}</td><td>{"да" if r["service_valid"] else "нет"}</td></tr>' for r in compared['rows'])
    markup = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>GOAL 4.4</title>
<style>body{font:16px system-ui;max-width:1120px;margin:32px auto;padding:0 20px;color:#182532}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd5de}select{font:inherit;padding:8px}pre{white-space:pre-wrap;background:#f2f5f8;padding:16px}img{max-width:100%}</style>
<h1>Пробег, обслуживание и ресурс ТО1</h1>
<p>День 30 сентября 2026 года, линия 1. Это разработческий опыт на учебной геометрии и предполагаемых OD.</p>
<table><thead><tr><th>Стратегия</th><th>км</th><th>км/68</th><th>Ожидание, с</th><th>p95, с</th><th>Допустима</th></tr></thead><tbody>TABLE</tbody></table>
<p>Эквивалент ресурса ТО1 рассчитан по 7,4 и 9,2 руб./км. Денежная экономия и упущенная выручка неизвестны.</p>
<p>Строгое улучшение: STRICT. Исследовательский компромисс: CONDITIONAL. Бюджет компромисса указан в проверках; он не является требованием пользователя.</p>
<img src="pareto.svg" alt="Сравнение ресурса ТО1 и ожидания">
<label for="strategy">Проверки стратегии </label><select id="strategy">OPTIONS</select><pre id="details"></pre>
<script>const report=PAYLOAD;function show(){const s=document.getElementById('strategy').value;const r=report.rows.find(x=>x.strategy===s);const c=report.comparisons.find(x=>x.strategy===s);document.getElementById('details').textContent=JSON.stringify({result:r,comparison:c||{reference:true}},null,2)}document.getElementById('strategy').addEventListener('change',show);show();</script></html>'''
    (output/'report.html').write_text(markup.replace('TABLE', table).replace('OPTIONS', options).replace('PAYLOAD', payload)
        .replace('STRICT', html.escape(compared['best_accepted_strategy'] or 'не найдено'))
        .replace('CONDITIONAL', html.escape(compared['best_conditional_strategy'] or 'не найден')), encoding='utf-8')
    return acceptance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'data/processed/metro44_20261009')
    parser.add_argument('--reference', default='STANDARD')
    parser.add_argument('--skip-journal-verification', action='store_true')
    args = parser.parse_args()
    print(json.dumps(report(args.output, reference_strategy=args.reference, verify=not args.skip_journal_verification),
                     ensure_ascii=False, default=str))
