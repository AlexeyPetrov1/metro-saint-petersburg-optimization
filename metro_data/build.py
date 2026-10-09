import importlib.metadata
import json
from pathlib import Path
import re

import pandas as pd

from .pipeline import (TIME_KEYS, aggregate_stations, load_map, read_hourly,
                       read_quarter, read_transport, require)
from .sources import file_hash, inventory


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def build_registry(refs, texts):
    draft = json.loads((refs / 'parameters_draft.json').read_text(encoding='utf-8-sig'))
    parameters = {}
    for key, p in draft['parameters'].items():
        source = p.get('source_id')
        parameters[key] = dict(value=p['value'], unit=p['unit'], status=p['status'],
                               source=source, note=p['note'], source_verification='reference_only')
    for key in ['headway_min', 'cars_per_train', 'regular_load_limit_per_car', 'max_trains_on_line']:
        parameters[key].update(source='current_user_task;parameters_draft.json#'+key,
                               source_verification='explicit_task_conditions')
    text = texts.get('D13', {}).get('text', '')
    ideal = re.search(r'идеале их\s+(\d+)\s+состава', text, re.I)
    release = re.search(r'выхода на Линию\s+(\d+)\s*[-–]\s*(\d+)\s*мин', text, re.I)
    require(ideal and release, 'D13: reserve text changed or missing')
    parameters['reference_roundtrip'].update(value=None, status='withdrawn', source='current_user_clarification',
        note='Пользователь отозвал 99 минут как ненастоящее значение. Оборот вычисляется из событий движения.',
        source_verification='explicit_task_conditions')
    parameters['reserve_ideal_total'].update(value=int(ideal[1]), source='D13',
                                             source_verification='source_text_read')

    def add(key, value, unit, status, source, note, verification='explicit_task_conditions'):
        parameters[key] = dict(value=value, unit=unit, status=status, source=source,
                               note=note, source_verification=verification)

    add('depots', ['Автово', 'Северное'], 'names', 'organizer_confirmed', 'current_user_task;D13',
        'Два депо; фактическое распределение готовых составов неизвестно.')
    add('reserve_file_release_range', [int(release[1])*60, int(release[2])*60], 's',
        'source_observed', 'D13', 'В файле указано время выхода на линию. Состав этапов не определён.', 'source_text_read')
    for key, value, note in [
        ('reserve_hot_release_reported', 420, '7 минут упомянуты в прежнем плане; исходная переписка отсутствует.'),
        ('reserve_regular_release_reported', 1800, '30 минут упомянуты в прежнем плане; исходная переписка отсутствует.'),
        ('reserve_travel_after_exit_reported', 420, 'Дополнительные 7 минут пути упомянуты в прежнем плане; маршрут неизвестен.'),
    ]:
        add(key, value, 's', 'source_observed', 'docs/reference/План_человека_4_экономика_оптимизация_рекомендации.docx',
            note, 'reference_only')
    parameters['release_delay'].update(value=None, status='conflict', source='D13;parameters_draft.json;prior_plan',
                                      note='7/30 минут и 15–20 минут описывают неоднозначные этапы. Единое время не установлено.')
    for key, unit, note in [
        ('reserve_preparation_time', 's', 'Время подготовки отдельно не установлено.'),
        ('reserve_line_entry_time', 's', 'Однозначное время подачи на линию не установлено.'),
        ('reserve_travel_to_target_time', 's', 'Путь до целевой станции и его длительность неизвестны.'),
        ('train_actual_readiness', 'train_state', 'Нет реестра фактической готовности поездов.'),
        ('repair_costs', 'RUB/repair', 'Нет стоимости отдельных ремонтных работ.'),
        ('working_timetable', 'train_events', 'Фото графика не задают полный набор событий.'),
        ('od_matrix', 'trips', 'Входы через турникеты не определяют направления поездок и пересадки.'),
        ('historical_publication_time', 'datetime', 'Историческое время публикации выгрузок неизвестно.'),
    ]:
        add(key, None, unit, 'unknown', None, note)
    for key, value, unit, note in [
        ('source_timezone', 'Europe/Moscow', 'timezone', 'Рабочая трактовка часового пояса по заданию.'),
        ('service_day_start', '03:00', 'local_time', 'Служебные сутки 03:00–03:00.'),
        ('after_midnight_next_calendar_day', True, 'boolean', '00:00–02:45 отнесены к следующей календарной дате; семантика источника не подтверждена.'),
        ('timestamp_is_interval_start', True, 'boolean', 'Метка считается началом интервала; семантика источника не подтверждена.'),
        ('hourly_timestamp_is_calendar_time', True, 'boolean', 'Часовая метка читается как календарное время; service_date вычисляется по границе 03:00.'),
    ]:
        add(key, value, unit, 'assumed', 'current_user_task', note)
    parameters['forecast_horizon'].update(status='assumed', note='7200 с сохранены как выбор прежнего плана; текущий этап не строит прогноз.')
    parameters['eo_nominal_runtime']['note'] += '; Original photo A02 unavailable; value copied from prepared draft, not confirmed.'
    rules = pd.read_csv(refs / 'maintenance_rules.csv', keep_default_na=False)
    require(len(rules) == 14 and not rules.duplicated(['series', 'maintenance_type']).any(),
            'Maintenance registry schema changed')
    require((rules.lower <= rules.nominal).all() and (rules.nominal <= rules.upper).all(),
            'Invalid maintenance tolerance')
    rules['train_type'] = rules.series.map({'81-722/723/724': 'Юбилейный', '81-725/726/727': 'Балтиец'})
    require(rules.train_type.notna().all(), 'Unknown maintenance series')
    rules['source'] = 'docs/reference/maintenance_rules.csv#' + rules.source_id
    rules['source_verification'] = 'reference_only'
    rules['upper_is_operating_target'] = False
    review_path = refs / 'maintenance_photo_review.json'
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding='utf-8'))
        for source_id, photo in review['sources'].items():
            for observed in photo['rows']:
                mask = rules.series.eq(photo['series']) & rules.maintenance_type.eq(observed['maintenance_type'])
                expected = [observed['nominal'] * 1000,
                            (observed['nominal'] - observed['lower_delta']) * 1000,
                            (observed['nominal'] + observed['upper_delta']) * 1000]
                require(mask.sum() == 1 and rules.loc[mask, ['nominal', 'lower', 'upper']].iloc[0].tolist() == expected
                        and rules.loc[mask, 'source_id'].iloc[0] == source_id
                        and rules.loc[mask, 'unit'].iloc[0] == 'km',
                        f'Photo review disagrees with maintenance CSV: {source_id}/{observed["maintenance_type"]}')
                rules.loc[mask, 'source'] = source_id + ';docs/reference/maintenance_photo_review.json'
                rules.loc[mask, 'source_verification'] = 'visual_review_chat_attachment'
                rules.loc[mask, 'source_locator'] = photo['source_locator']
                rules.loc[mask, 'source_unit'] = photo['source_unit']
                rules.loc[mask, 'source_condition'] = photo['source_condition']
                rules.loc[mask, 'note'] += '; ' + photo['note'] + ' ' + observed.get('note', '')
        photo = review['sources'].get('A02', {})
        if 'eo' in photo:
            eo = photo['eo']
            note = eo['note'] + ' Отсчёт: ' + eo['clock_start'] + '.'
            add('eo_inspection_validity', eo['inspection_validity_h'], 'h', 'source_observed', 'A02',
                note, 'visual_review_chat_attachment')
            validity = parameters['eo_inspection_validity']
            validity.update(lower=validity['value']-eo['tolerance_h'], upper=validity['value']+eo['tolerance_h'],
                            series=photo['series'], source_locator=photo['source_locator'],
                            upper_is_operating_target=False)
            parameters['eo_nominal_runtime'].update(validity, semantic_role='inspection_validity',
                note='Название ключа сохранено для совместимости. Это срок действия осмотра. ' + note)
            add('eo_frequency', eo['frequency_per_day'], 'per_day', 'source_observed', 'A02',
                'ЕО для серии ' + photo['series'] + ': один раз в сутки.', 'visual_review_chat_attachment')
            parameters['eo_frequency'].update(series=photo['series'], source_locator=photo['source_locator'])
    else:
        rules['note'] += '; original photo unavailable'
    rules['note'] += '; upper tolerance is not an operating target'
    return dict(schema_version='1.0', identifiers='local identifiers, not operator codes', parameters=parameters,
                maintenance_rules=rules.to_dict('records')), rules


def validate_tables(v, s, h, mapping):
    for name, table, identity, duration in [('15m', v, 'vestibule_id', 15),
                                          ('station', s, 'station_id', 15), ('1h', h, 'vestibule_id', 60)]:
        require(not table.empty, f'{name}: empty table')
        require(not table.duplicated(['service_date', identity, 'interval_start']).any(), f'{name}: duplicate keys')
        require(str(table.interval_start.dt.tz) == 'Europe/Moscow' and
                str(table.interval_end.dt.tz) == 'Europe/Moscow', f'{name}: missing or wrong timezone')
        require(table.interval_end.sub(table.interval_start).eq(pd.Timedelta(minutes=duration)).all(),
                f'{name}: wrong interval duration')
        require(table.interval_start.sub(pd.Timedelta(hours=3)).dt.date.eq(table.service_date).all(),
                f'{name}: service date does not match 03:00 boundary')
        require(table.entry_count.dropna().ge(0).all() and
                table.entry_count.dropna().mod(1).eq(0).all(), f'{name}: invalid counts')
    require(h.interval_start.dt.minute.eq(0).all() and h.interval_start.dt.second.eq(0).all(),
            '1h: expected whole-hour timestamps')
    expected_v = set(mapping.vestibule_id)
    require(set(v.vestibule_id) == expected_v, '15m: vestibule coverage differs from map')
    lookup = mapping.set_index('vestibule_id').station_id
    for name, table in [('15m', v), ('1h', h)]:
        require(table.vestibule_id.isin(lookup.index).all() and
                table.station_id.eq(table.vestibule_id.map(lookup)).all(), f'{name}: inconsistent mapping')
        require(table.quality_flag.eq('missing').eq(table.entry_count.isna()).all(), f'{name}: missing/zero flag mismatch')
    for name, table, identity in [('15m', v, 'vestibule_id'), ('station', s, 'station_id')]:
        sizes = table.groupby(['service_date', identity]).size()
        require(sizes.eq(96).all(), f'{name}: expected 96 intervals per day')
        require(table.interval_start.dt.second.eq(0).all() and table.interval_start.dt.minute.mod(15).eq(0).all(),
                f'{name}: interval grid changed')
        per_day = table.groupby('service_date')[identity].nunique()
        require(per_day.eq(24 if name == '15m' else 19).all(), f'{name}: incomplete expected grid')
    expected_s = aggregate_stations(v, mapping)
    cols = TIME_KEYS + ['station_id', 'entry_count', 'vestibules_expected', 'vestibules_present',
                        'vestibules_observed', 'is_complete', 'source_id', 'source_sheet']
    a = s.sort_values(['service_date', 'station_id', 'interval_start'])[cols].reset_index(drop=True)
    require(a.equals(expected_s[cols]), 'station: aggregation or completeness mismatch')
    require(v.entry_count.sum() == s.entry_count.sum(), 'station aggregation changed total')
    return ['unique keys', 'nonnegative integral counts', 'timezone and interval grid',
            'midnight and month boundary', '96 intervals and expected vestibules/stations',
            'explicit station mapping', 'missing values distinct from zero',
            'station sums and completeness recomputed']


def table_info(data):
    return dict(rows=len(data), service_days=data.service_date.nunique(),
                service_date_min=str(data.service_date.min()), service_date_max=str(data.service_date.max()),
                interval_start_min=data.interval_start.min().isoformat(),
                interval_start_max=data.interval_start.max().isoformat(),
                interval_end_max=data.interval_end.max().isoformat(),
                calendar_days=data.interval_start.dt.date.nunique(),
                vestibules=data.vestibule_id.nunique() if 'vestibule_id' in data else None,
                stations=data.station_id.nunique(), total_entries=int(data.entry_count.sum()),
                missing_values=int(data.entry_count.isna().sum()), zeros=int(data.entry_count.eq(0).sum()))


def compare_hourly(v, h):
    base = v.assign(hour=v.interval_start.dt.floor('h')).groupby(['vestibule_id', 'hour']).entry_count
    quarters = base.agg(['sum', 'count']).reset_index()
    quarters = quarters[quarters['count'] == 4]
    merged = quarters.merge(h[['vestibule_id', 'interval_start', 'entry_count']],
                            left_on=['vestibule_id', 'hour'], right_on=['vestibule_id', 'interval_start'])
    merged = merged.dropna(subset=['entry_count'])
    delta = merged['sum'] - merged.entry_count
    return dict(compared_hours=len(merged), equal_hours=int(delta.eq(0).sum()),
                differing_hours=int(delta.ne(0).sum()), mean_absolute_difference=float(delta.abs().mean()),
                note='Calendar alignment without shift, under the stated timestamp assumption; sources kept separate.')


def report_markdown(report):
    lines = ['Проверка данных первой линии метро', '',
             'Исходники прочитаны заново. Значения не исправлены по прежнему аудиту.', '',
             '| Таблица | Строк | Вестибюлей | Станций | Сумма входов |',
             '|---|---:|---:|---:|---:|']
    for name, info in report['tables'].items():
        lines.append(f'| {name} | {info["rows"]} | {info["vestibules"] or ""} | {info["stations"]} | {info["total_entries"]} |')
    lines += ['', '15-минутные служебные даты: февраль, май, июль и сентябрь 2026 года. '
              'Часовой источник хранится отдельно; календарное покрытие отражено в quality_report.json.', '',
              'Временные метки содержат Europe/Moscow. Перенос отметок 00:00–02:45 на следующий день '
              'и трактовка метки как начала интервала сохранены в parameters.json со статусом assumed.', '',
              'Проверки: ' + ', '.join(report['validation_passed']) + '.', '',
              f'Сверки с исходными итогами: {report["reconciliation"]["pass"]} совпадений, '
              f'{report["reconciliation"]["mismatch"]} расхождений, '
              f'{report["reconciliation"]["incomplete"]} неполных сверок.', '',
              '| Источник | Проверка | Опубликовано | Вычислено | Разница |',
              '|---|---|---:|---:|---:|']
    for c in report['discrepancies']:
        lines.append(f'| {c["source_id"]} | {c["check"]} | {c["expected"]} | {c["actual"]} | {c["delta"]} |')
    overlap = report['hourly_comparison']
    lines += ['', 'Разница вычисляется как сумма компонентов минус опубликованный итог. '
              'Составляющие нулевого пробега повторно к общему пробегу не прибавляются.', '',
              f'При принятом совмещении временных меток сравнимы {overlap["compared_hours"]} часов: '
              f'{overlap["equal_hours"]} совпадают, {overlap["differing_hours"]} различаются. '
              'Это сопоставление не подтверждает одинаковую методику учёта.', '',
              'Предупреждения:', '']
    lines += ['- ' + warning for warning in report['warnings']]
    lines += ['', 'Неизвестны цены энергии, нормы расхода, стоимость ремонтов, готовность поездов, '
              'фактический резерв, длительности отдельных этапов его выдачи, рабочее расписание, '
              'матрица поездок и историческое время публикации выгрузок.', '',
              'Входы через турникеты описывают входы пассажиров. Загрузка вагонов и рублёвая экономия '
              'на этом этапе не рассчитаны.', '',
              'Следующему этапу: entries_station_15m.parquet, entries_vestibule_15m.parquet, '
              'parameters.json, maintenance_rules.csv, transport_work.csv. '
              'Для проверки происхождения и качества: sources.csv, source_text.json, reconciliation.csv, '
              'quality_report.json, build_manifest.json.', '']
    return '\n'.join(lines)


def prepare(root, output):
    raw, refs = root / 'data/raw/metro', root / 'docs/reference'
    require(raw.exists(), 'Missing data/raw/metro; download the supplied Drive source folder')
    require(not output.resolve().is_relative_to(raw.resolve()) and not output.resolve().is_relative_to(refs.resolve()),
            'Output directory must not be inside source/reference directories')
    output.mkdir(parents=True, exist_ok=True)
    sources, texts = inventory(root, raw, refs)
    sources.to_csv(output / 'sources.csv', index=False, lineterminator='\n')
    require(not sources.read_status.eq('read_error').any(), 'Unreadable source; inspect sources.csv')
    mapping = load_map(refs / 'vestibule_map.csv')
    aliases = pd.read_csv(refs / 'hourly_vestibule_aliases.csv')
    quarter_tables, monthly, work_tables, checks, hourly = [], [], [], [], []

    def read_logged(reader, index, *args):
        try:
            return reader(*args)
        except Exception as e:
            sources.loc[index, ['read_status', 'read_result']] = ['schema_error', str(e)]
            sources.to_csv(output / 'sources.csv', index=False, lineterminator='\n')
            raise

    for i, src in sources[sources.source_kind == 'file'].iterrows():
        path = root / src.relative_path
        if src.actual_format == 'xlsx' and '15-мин' in path.name:
            data, audit, summary = read_logged(read_quarter, i, path, mapping, src.source_id)
            quarter_tables.append(data)
            checks.extend(audit)
            monthly.append(summary)
            sources.loc[i, ['read_status', 'read_result']] = ['extracted', f'{len(data)} observations; {summary["days"]} day sheets']
        elif src.actual_format == 'xlsx' and path.name == 'Пассажиропоток 2026 Линия 1.xlsx':
            data, hour_info = read_logged(read_hourly, i, path, mapping, aliases, src.source_id)
            hourly.append(data)
            checks.append(hour_info['footer_reconciliation'])
            sources.loc[i, ['read_status', 'read_result']] = ['extracted', f'{len(data)} observations; summary/footer excluded']
        elif src.actual_format == 'html' and path.stem.startswith('MO1'):
            data, audit = read_logged(read_transport, i, path, src.source_id, src.source_url)
            data['source_path'] = src.relative_path
            if any(c['status'] == 'mismatch' for c in audit): data['quality_flag'] = 'source_discrepancy'
            work_tables.append(data)
            checks.extend(audit)
            sources.loc[i, ['read_status', 'read_result']] = ['extracted_with_discrepancies' if any(
                c['status'] == 'mismatch' for c in audit) else 'extracted', f'{len(data)} line-1 metrics']
        elif src.read_status == 'pending_extraction':
            sources.loc[i, ['read_status', 'read_result']] = ['unhandled_layout', 'New table requires an explicit parser']
    sources.to_csv(output / 'sources.csv', index=False, lineterminator='\n')
    require(len(quarter_tables) >= 4 and len(hourly) == 1 and len(work_tables) >= 3, 'Required passenger/MO1 files missing')
    require(not sources.read_status.isin(['read_error', 'unhandled_layout', 'pending_extraction']).any(),
            'Unreadable or changed source; inspect sources.csv')
    v = pd.concat(quarter_tables, ignore_index=True).sort_values(
        ['service_date', 'vestibule_id', 'interval_start']).reset_index(drop=True)
    s, h = aggregate_stations(v, mapping), hourly[0]
    passed = validate_tables(v, s, h, mapping)
    work = pd.concat(work_tables, ignore_index=True).sort_values(['period_start', 'metric']).reset_index(drop=True)
    require(not work.duplicated(['line_id', 'period_start', 'period_end', 'metric']).any(), 'Duplicate transport metrics')
    registry, rules = build_registry(refs, texts)
    mapping.to_csv(output / 'vestibule_map.csv', index=False, lineterminator='\n')
    aliases.to_csv(output / 'hourly_vestibule_aliases.csv', index=False, lineterminator='\n')
    for name, table in [('entries_vestibule_15m', v), ('entries_station_15m', s), ('entries_vestibule_1h', h)]:
        table.to_parquet(output / (name + '.parquet'), index=False)
    work.to_csv(output / 'transport_work.csv', index=False, lineterminator='\n')
    rules.to_csv(output / 'maintenance_rules.csv', index=False, lineterminator='\n')
    pd.DataFrame(checks).to_csv(output / 'reconciliation.csv', index=False, lineterminator='\n')
    write_json(output / 'parameters.json', registry)
    write_json(output / 'source_text.json', texts)
    status_counts = pd.Series([c['status'] for c in checks]).value_counts()
    warnings = ['В CU3 дневных листов обнаружен дополнительный суточный итог линии. Он включён в сверки и исключён из наблюдений.',
                ('14 правил обслуживания сверены с фото таблиц 3 и 4 в чате. ЕО: один раз в сутки, срок действия осмотра 20 ±4 часа. '
                 'Исходные файлы фото локально недоступны; их размер и SHA-256 неизвестны. Дата и полный регламент неизвестны.'
                 if rules.source_verification.eq('visual_review_chat_attachment').all() else
                 'Фото регламентов не полностью проверены; строки с source_verification=reference_only перенесены из подготовленного реестра.'),
                'Сканированный PDF и встроенные изображения учтены по формату, хешу и метаданным. Их числа не признаны подтверждёнными параметрами.',
                'Часовой источник не содержит Технологический институт; 30.09.2026 присутствует только час 00.',
                'Между четырьмя месяцами основного ряда есть разрывы. Отсутствующие месяцы не заполнены нулями.']
    prior_rows = {'2026-02': 64512, '2026-05': 71424, '2026-07': 71424, '2026-09': 69120}
    for m in monthly:
        m['previous_rows'] = prior_rows.get(m['month'])
        m['matches_previous_rows'] = m['rows'] == m['previous_rows']
        if m['missing_days']: warnings.append(f'{m["source_id"]}: отсутствуют служебные даты {m["missing_days"]}.')
        if not m['matches_previous_rows']: warnings.append(f'{m["source_id"]}: число строк отличается от прежнего аудита; исходные значения сохранены.')
    changed = sources[sources.version_status.eq('changed')].source_id.tolist()
    if changed: warnings.append('Хеши отличаются от прежнего реестра: ' + ', '.join(changed) + '.')
    hour_day_sizes = h.groupby(h.interval_start.dt.date).size()
    hourly_info = dict(hour_info, calendar_daily_row_counts={str(k): int(n) for k, n in hour_day_sizes.value_counts().items()},
                       missing_vestibules=sorted(set(mapping.vestibule_id) - set(h.vestibule_id)))
    report = dict(tables={name: table_info(t) for name, t in [('entries_vestibule_15m', v),
                   ('entries_station_15m', s), ('entries_vestibule_1h', h)]},
                  monthly=sorted(monthly, key=lambda m: m['month']), hourly=hourly_info,
                  transport_rows=len(work), transport_periods=work[['period_start', 'period_end']].drop_duplicates().astype(str).to_dict('records'),
                  reconciliation={k: int(status_counts.get(k, 0)) for k in ['pass', 'mismatch', 'incomplete']},
                  discrepancies=[c for c in checks if c['status'] != 'pass'], warnings=warnings,
                  hourly_comparison=compare_hourly(v, h), station_incomplete_rows=int((~s.is_complete).sum()),
                  source_file_count=int(sources.source_kind.eq('file').sum()),
                  embedded_image_count=int(sources.source_kind.eq('embedded_image').sum()),
                  validation_passed=passed + ['source daily and line totals checked', 'empty and total rows excluded',
                                             'HTML XLS format and reported discrepancies preserved'])
    write_json(output / 'quality_report.json', report)
    (output / 'quality_report.md').write_text(report_markdown(report), encoding='utf-8')
    inputs = {r.relative_path: r.sha256 for r in sources.itertuples() if r.source_kind in ('file', 'reference')}
    for p in sorted((root / 'metro_data').glob('*.py')): inputs[p.relative_to(root).as_posix()] = file_hash(p)
    if (root / 'requirements.txt').exists(): inputs['requirements.txt'] = file_hash(root / 'requirements.txt')
    names = ['sources.csv', 'source_text.json', 'vestibule_map.csv', 'hourly_vestibule_aliases.csv',
             'entries_vestibule_15m.parquet', 'entries_station_15m.parquet', 'entries_vestibule_1h.parquet',
             'transport_work.csv', 'parameters.json', 'maintenance_rules.csv', 'reconciliation.csv',
             'quality_report.json', 'quality_report.md']
    manifest = dict(inputs=inputs, outputs={n: file_hash(output / n) for n in names},
                    dependencies={n: importlib.metadata.version(n) for n in ['pandas', 'pyarrow', 'openpyxl', 'pypdf', 'Pillow']})
    write_json(output / 'build_manifest.json', manifest)
    return report


def check(root, output):
    manifest = json.loads((output / 'build_manifest.json').read_text(encoding='utf-8'))
    observed = {p.relative_to(root).as_posix() for folder in ['data/raw/metro', 'docs/reference']
                for p in (root / folder).rglob('*') if p.is_file()}
    expected_sources = {p for p in manifest['inputs'] if p.startswith(('data/raw/metro/', 'docs/reference/'))}
    require(observed == expected_sources, 'Source file set changed; run prepare')
    for name, expected in manifest['inputs'].items():
        require((root / name).is_file() and file_hash(root / name) == expected, f'Input changed: {name}; run prepare')
    for name, expected in manifest['outputs'].items():
        require((output / name).is_file() and file_hash(output / name) == expected, f'Output changed: {name}; run prepare')
    mapping = load_map(root / 'docs/reference/vestibule_map.csv')
    v, s, h = [pd.read_parquet(output / (n + '.parquet')) for n in
               ['entries_vestibule_15m', 'entries_station_15m', 'entries_vestibule_1h']]
    passed = validate_tables(v, s, h, mapping)
    report = json.loads((output / 'quality_report.json').read_text(encoding='utf-8'))
    for name, data in [('entries_vestibule_15m', v), ('entries_station_15m', s), ('entries_vestibule_1h', h)]:
        require(table_info(data) == report['tables'][name], f'Report mismatch: {name}')
    audit = pd.read_csv(output / 'reconciliation.csv', dtype=str, keep_default_na=False)
    require(audit.status.value_counts().to_dict() == {k: n for k, n in report['reconciliation'].items() if n},
            'Reconciliation counts do not match report')
    work = pd.read_csv(output / 'transport_work.csv', dtype=str, keep_default_na=False)
    require(len(work) == report['transport_rows'] and work.line_id.eq('1').all() and
            not work.duplicated(['line_id', 'period_start', 'period_end', 'metric']).any(), 'Transport keys/line mismatch')
    saved = json.loads((output / 'parameters.json').read_text(encoding='utf-8'))
    for key, p in saved['parameters'].items():
        require(all(k in p for k in ['value', 'unit', 'status', 'source', 'note']), f'Parameter schema: {key}')
        require(p['status'] != 'unknown' or p['value'] is None, f'Unknown parameter has invented value: {key}')
    return dict(validation_passed=passed, input_and_output_hashes='pass',
                source_discrepancies_preserved=report['reconciliation']['mismatch'],
                tables=report['tables'])
