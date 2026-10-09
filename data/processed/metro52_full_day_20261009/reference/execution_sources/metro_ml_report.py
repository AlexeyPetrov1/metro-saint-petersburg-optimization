"""Readable paired results from saved, independently verified artifacts."""
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
from pathlib import Path
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def case_conclusion(case):
    median, ml = [case['strategies'][key]['metrics'] for key in ('MEDIAN', 'ML')]
    left, right = [row['actual_service_admissible'] for row in (median, ml)]
    if not left and not right:
        return 'Оба варианта нарушают ограничения'
    if right and not left:
        return 'ML соблюдает ограничения; медиана нарушает'
    if left and not right:
        return 'Медиана соблюдает ограничения; ML нарушает'
    a = (median['train_km'], median['mean_wait_seconds'])
    b = (ml['train_km'], ml['mean_wait_seconds'])
    if None in (*a, *b):
        return 'Оценка ожидания неполна'
    if all(abs(x-y) < 1e-8 for x, y in zip(a, b)):
        return 'Результаты совпали'
    if all(x <= y for x, y in zip(b, a)):
        return 'ML снизил пробег или ожидание без ухудшения второго показателя'
    if all(x <= y for x, y in zip(a, b)):
        return 'Медиана снизила пробег или ожидание без ухудшения второго показателя'
    return 'Размен между пробегом и ожиданием'


def number(value):
    return 'неизвестно' if value is None else f'{value:,.2f}'.replace(',', ' ')


def build(output, evaluation):
    output, evaluation = Path(output), Path(evaluation)
    comparison = read(output/'report.json')
    acceptance = read(output/'acceptance.json')
    if acceptance['status'] != 'passed':
        raise ValueError('comparison integrity has not passed')
    verified = {row['case_id'] for row in acceptance['cases']}
    if verified != {row['case_id'] for row in comparison['cases']}:
        raise ValueError('comparison changed after independent verification')
    from goal43_replay import _series
    for case in comparison['cases']:
        keep = case['strategies']['KEEP']
        if keep.get('series') is None:
            folder = output/keep['run_folder']
            data, motion, summary = [read(folder/name) for name in ('movement_input.json', 'motion.json', 'summary.json')]
            with gzip.open(folder/'groups.json.gz', 'rt', encoding='utf-8') as stream:
                groups = json.load(stream)
            with gzip.open(folder/'passenger_checkpoint.json.gz', 'rt', encoding='utf-8') as stream:
                state = json.load(stream)
            runner = SimpleNamespace(motion=motion, by_id={row['group_id']: row for row in groups})
            first = keep['metrics']['mean_wait_scope']['start_second']
            last = summary['service_scope']['end_second']
            keep['series'] = _series(runner, state, first, last, {'rows': []}, data, service_end_second=last)
    errors = read(evaluation/'summary.json')
    profile = read(evaluation/'profile_control_summary.json')
    heldout = [r for r in errors['metrics']['hybrid19'] if r['temporal_holdout']]
    lines = ['# ML и медиана в существующем симуляторе', '',
        'Готовые веса подключены без повторного обучения. Их обучение заканчивается 29.09.2026. '
        'Модель и данные относятся к красной линии 1, Девяткино → Проспект Ветеранов. '
        'Синяя линия 2 этим набором не покрыта.', '',
        'Цель прогноза: входы через турникеты на станцию. Заполненность платформы и вагонов вычисляет '
        'существующий событийный симулятор. Для Технологического института используется явная медиана: '
        'ML покрывает 18 из 19 станций.', '',
        '## Проверка прогноза', '',
        '30 сентября находится после обучения. Проверены часовые обновления с 06:00 до 22:00, '
        'горизонты 1 и 2 часа. Это один день без семидневного отступа исходного бэктеста; '
        'погодные признаки этого дня отсутствуют и передаются как NaN. '
        'Каждый прогноз оценивается отдельно: повторяющиеся целевые интервалы разных горизонтов '
        'не считаются разными пассажирами.', '',
        '| Прогноз, 19 станций | WAPE, % | MAE, человек/15 мин | RMSE |',
        '|---|---:|---:|---:|']
    labels = {'simulator_observations': 'ML с наблюдениями симулятора и профилем прошлых четвертей',
              'exact_uploaded_hourly': 'ML с исходным часовым источником', 'median': 'Медиана, равные четверти'}
    for row in heldout:
        lines.append(f"| {labels.get(row['mode'], row['mode'])} | {number(100*row['wape'])} | {number(row['mae'])} | {number(row['rmse'])} |")
    control = [r for r in profile['metrics']['hybrid19_quarter'] if r['date'] == '2026-09-30']
    if control:
        row = control[0]
        lines.extend(['', f"Медиана с тем же профилем прошлых четвертей: WAPE {number(100*row['wape'])}%. "
            'Этот контроль отделяет эффект распределения внутри часа от прогноза часовой суммы.'])
    lines.extend(['', 'Даты 02.09, 16.09 и 29.09 пересекаются с обучением и калибровкой; '
        'их результаты сохранены как диагностика. Часовые входы и суммы исходных четвертей немного '
        'различаются, поэтому ошибки по обоим источникам сохранены отдельно.', '',
        '## Управление движением', '',
        'Варианты KEEP, MEDIAN и ML используют одинаковые исходные поезда, пассажиров, OD S1, seed 7 '
        'и ограничения. MEDIAN и ML используют один оптимизатор с общим набором кандидатов: '
        'сохранить график, регулировать интервал, снять до двух составов, совместить снятие и регулирование. '
        'В дополнительной задаче регулирования проверяется цель 130 с в пик и 180 с вне пика; '
        'это целевой интервал для CP-SAT, а фактические отправления проверяются отдельно. '
        'Поиск ограничен набором кандидатов и временем; результаты search_incomplete сохранены, '
        'глобальная оптимальность не заявляется.', '',
        'Каждый опыт длится четыре часа: час предварительного расчёта, два решения с шагом 15 минут '
        'и последующее исполнение выбранного плана с возвратом в депо. После второго решения новые команды '
        'не выдаются. Это проверка двух последовательных действий, а не полный суточный контроллер. '
        'Среднее ожидание относится к одинаковой когорте до окончания сервиса минус 600 с; '
        'поздние пассажиры и их незавершённые обязательства сохранены отдельно.', '',
        '| Дата, окно | Управление | Пробег, состав-км | Км на состав, парк 68 | Ожидание, с | Макс. очередь | Макс. на вагон | Ограничения |',
        '|---|---|---:|---:|---:|---:|---:|---|'])
    train_rows, case_notes = [], []
    for case in comparison['cases']:
        case_notes.append(f"- {case['date']}, {case['window']}: {case_conclusion(case)}.")
        for strategy, result in case['strategies'].items():
            metrics = result['metrics']
            ok = 'соблюдены' if metrics['actual_service_admissible'] else 'нарушены'
            lines.append(f"| {case['date']}, {case['window']} | {strategy} | {number(metrics['train_km'])} | "
                f"{number(metrics['mean_km_per_fixed_train'])} | {number(metrics['mean_wait_seconds'])} | "
                f"{metrics['max_queue']} | {metrics['max_car_load']} | {ok} |")
            for row in result['delta_vs_keep']['per_train']:
                train_rows.append(dict(date=case['date'], window=case['window'], strategy=strategy, **row))
    lines.extend(['', *case_notes, '',
        'Снижение пробега при нарушении ограничений не считается подтверждённым положительным эффектом. '
        'ТО1 переводится в ресурс по 15 000 км и указанным ценам; такой эквивалент не означает денежную экономию. '
        'Упущенная выручка и итоговая прибыль неизвестны без данных об отказах и затратах.', '',
        '## Проверка и файлы', '',
        f"Независимая проверка восстановила {acceptance['run_count']} журналов движения и пассажиров. "
        f"Сохранены {acceptance['preserved_files']} прежних файлов. Проверка целостности не отменяет "
        'записанных нарушений загрузки.', '',
        '[Числа сравнений](rows.csv), [машинный отчёт](report.json), [проверка](acceptance.json), '
        '[пробег каждого состава](train_km.csv).', '',
        f"Оценка модели: {evaluation.resolve()}.",
        'Воспроизведение команд и описание подключения: docs/metro_ml.md.', ''])
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (output/'train_km.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(train_rows[0]))
        writer.writeheader(); writer.writerows(train_rows)
    visual = dict(cases=[dict(case_id=case['case_id'], strategies={name: dict(
        metrics={key: result['metrics'][key] for key in ('train_km', 'mean_km_per_fixed_train',
            'mean_wait_seconds', 'max_queue', 'actual_service_admissible')}, series=result.get('series'))
        for name, result in case['strategies'].items()}) for case in comparison['cases']])
    data = json.dumps(visual, ensure_ascii=False).replace('<', '\\u003c')
    prose = html.escape('\n'.join(lines))
    page = """<!doctype html><html lang="ru"><meta charset="utf-8"><title>ML и медиана: проверка</title>
<style>body{font:16px system-ui;margin:36px auto;max-width:1180px;padding:0 20px;color:#172533;background:#f5f7fa}
h1{font-size:28px}select{padding:8px;margin:6px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;border-bottom:1px solid #d4dce4;text-align:left}
svg{background:white;width:100%;height:270px}pre{white-space:pre-wrap;font:14px system-ui;line-height:1.6} .note{background:#fff6d5;padding:15px}a{color:#145ac5}</style>
<h1>ML и медиана в симуляторе линии 1</h1>
<p class="note">Данные относятся к красной линии 1. Веса обучены по 29.09.2026.
Прогнозирует входы через турникеты. ML покрывает 18 станций, ещё одна использует медиану.</p>
<label>Опыт <select id="case"></select></label><label>График <select id="chart">
<option value="queue">Очередь</option><option value="load">Максимальная загрузка вагона</option>
<option value="headway">Максимальный интервал, с</option><option value="fleet">Составы на линии</option></select></label>
<p id="verdict"></p><table id="metrics"></table><svg id="plot" viewBox="0 0 1100 270"></svg>
<p>Серый: KEEP. Синий: MEDIAN. Зелёный: ML. Горизонтальная ось: минуты от начала периода.</p>
<p><a href="report.md">Отчёт</a> · <a href="rows.csv">Сводная таблица</a> · <a href="train_km.csv">Каждый состав</a> · <a href="acceptance.json">Проверка</a></p>
<details><summary>Методика и ограничения</summary><pre>PROSE</pre></details>
<script id="data" type="application/json">DATA</script><script>
const report=JSON.parse(document.getElementById('data').textContent), picker=document.getElementById('case');
report.cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.case_id;picker.append(o)});
function show(){const c=report.cases[Number(picker.value)], rows=Object.entries(c.strategies), chart=document.getElementById('chart').value;
document.getElementById('verdict').textContent='Прогноз лучше не означает, что действия соблюдают ограничения. Смотрите фактическую проверку ниже.';
const table=document.getElementById('metrics');table.replaceChildren();
const head=table.insertRow();['Управление','Пробег, км','Км/состав, парк 68','Ожидание, с','Очередь','Ограничения'].forEach(t=>{const cell=document.createElement('th');cell.textContent=t;head.append(cell)});
for(const [s,r] of rows){const m=r.metrics,tr=table.insertRow();[s,m.train_km.toFixed(2),m.mean_km_per_fixed_train.toFixed(2),m.mean_wait_seconds===null?'неизвестно':m.mean_wait_seconds.toFixed(2),m.max_queue,m.actual_service_admissible?'соблюдены':'нарушены'].forEach(t=>tr.insertCell().textContent=t)}
const sets=rows.map(([s,r])=>[s,(r.series&&r.series[chart]||[])]), value=r=>r.queued??r.load??r.seconds??r.count??0;
const all=sets.flatMap(s=>s[1]), xMax=Math.max(1,...all.map(r=>r.second)), yMax=Math.max(1,...all.map(value));
let svg='<path d="M45 15V235H1080" stroke="#6b7b8b" fill="none"/>';
for(const [s,points] of sets){const color=s==='ML'?'#087b5d':s==='MEDIAN'?'#225dc4':'#8a929b';if(points.length)svg+='<polyline fill="none" stroke="'+color+'" stroke-width="2" points="'+points.map(r=>(45+1035*r.second/xMax)+','+(235-210*value(r)/yMax)).join(' ')+'"/>'}
svg+='<text x="2" y="25">'+yMax.toFixed(0)+'</text><text x="1030" y="258">'+(xMax/60).toFixed(0)+' мин</text>';document.getElementById('plot').innerHTML=svg}
picker.onchange=show;document.getElementById('chart').onchange=show;show();
</script></html>"""
    (output/'report.html').write_text(page.replace('PROSE', prose).replace('DATA', data), encoding='utf-8')
    return dict(case_count=len(comparison['cases']), report=str(output/'report.html'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('evaluation', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.evaluation), ensure_ascii=False))
