"""Keep a small, self-refreshing local progress page beside the run."""

import argparse
import html
import json
import time
from datetime import datetime
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}


def progress_snapshot(root):
    root = Path(root)
    status = read_json(root / 'status.json')
    contract = read_json(root / 'balanced/contract.json')
    decisions = sorted((root / 'balanced/controller').glob('decision_*.json'))
    latest = read_json(decisions[-1]) if decisions else {}
    total = len(range(contract.get('control_start_second', 900),
                      contract.get('cohort_end_second', 900), 900))
    llm = {'yandexgpt': 0, 'fallback': 0}
    for path in (root / 'balanced/explanations').glob('explanation_*.json'):
        metadata = read_json(path).get('metadata', {})
        if metadata:
            llm['yandexgpt' if metadata.get('source') == 'yandexgpt' else 'fallback'] += 1
    return dict(stage=status.get('stage', 'waiting_for_status'), error=status.get('error'),
                finished_steps=len(decisions), total_steps=total,
                last_decision=latest.get('decision_time'),
                full_day_finished=status.get('stage') == 'completed',
                skipped_arms=status.get('skipped_arms', []), llm=llm,
                updated_at=datetime.now().astimezone().isoformat(timespec='seconds'))


def write_progress(root, snapshot):
    root = Path(root)
    stages = {'running': 'Суточный расчёт продолжается',
              'reporting': 'Готовятся проверки, отчёты и PDF',
              'completed': 'Расчёт и отчёты готовы', 'failed': 'Обнаружена ошибка',
              'waiting_for_status': 'Ожидается запись состояния'}
    step = f"{snapshot['finished_steps']} / {snapshot['total_steps']}"
    rows = [('Состояние', stages.get(snapshot['stage'], snapshot['stage'])),
            ('Завершено решений оптимизатора', step),
            ('Время последнего решения в модели', snapshot['last_decision'] or 'Пока нет'),
            ('Файл обновлён', snapshot['updated_at']),
            ('Ответы YandexGPT', snapshot['llm'].get('yandexgpt', 0)),
            ('Объяснения из проверенного шаблона', snapshot['llm'].get('fallback', 0)),
            ('Исключены из очереди', ', '.join(snapshot['skipped_arms']) or 'Нет')]
    if snapshot.get('error'):
        rows.append(('Ошибка', snapshot['error']))
    text = 'Прогресс суточного прогона\n\n' + '\n'.join(f'{k}: {v}' for k, v in rows)
    text += '\n\nКоличество решений не включает последующие проверки и подготовку отчётов.\n'
    cells = ''.join(f'<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>' for k, v in rows)
    percent = 100 * snapshot['finished_steps'] / max(1, snapshot['total_steps'])
    page = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="10"><title>Прогресс прогона метро</title>
<style>body{{font:17px system-ui;max-width:900px;margin:40px auto;padding:20px;background:#f5f7fb;color:#182333}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:14px;text-align:left;border-bottom:1px solid #ddd}}
progress{{width:100%;height:24px}}a{{color:#165ec7}}</style></head><body>
<h1>Прогресс суточного прогона</h1><progress max="100" value="{percent:.1f}"></progress><table>{cells}</table>
<p>Страница перечитывается каждые 10 секунд. Оставьте её открытой в браузере.</p>
<p>Счётчик решений не включает последующие проверки и подготовку отчётов.</p>
<p><a href="http://localhost:8501" target="_blank">Открыть Streamlit</a> ·
<a href="review/report.pdf">PDF после завершения</a> · <a href="review/report.html">Полный отчёт после завершения</a></p>
</body></html>'''
    root.mkdir(parents=True, exist_ok=True)
    for name, content in [('progress.txt', text), ('progress.html', page)]:
        path = root / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content, encoding='utf8')
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--max-seconds', type=float, default=43200)
    args = parser.parse_args()
    if args.interval <= 0 or args.max_seconds <= 0:
        parser.error('interval and max-seconds must be positive')
    started = time.monotonic()
    while True:
        snapshot = progress_snapshot(args.root)
        write_progress(args.root, snapshot)
        if not args.watch or snapshot['full_day_finished'] or time.monotonic() - started >= args.max_seconds:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
