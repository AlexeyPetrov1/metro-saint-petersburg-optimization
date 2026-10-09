# Календарные признаки

Независимый детерминированный модуль: временные компоненты вычисляются из
datetime, статус дня берётся из явно переданного источника календаря.
Python 3.11+, Pydantic v2. LLM, API, интернет и ключи не используются.

```sh
python3 -m pip install -r external_data/calendar/requirements.txt
python3 -m unittest discover -s external_data/calendar/tests -v
python3 -m unittest discover -v
```

## Использование

```python
from datetime import date, datetime
from external_data.calendar import CalendarDay, FakeCalendarProvider, get_calendar_features

# Демонстрационная таблица, не официальный производственный календарь.
provider = FakeCalendarProvider({
    date(2026, 10, 10): CalendarDay(day_type="weekend", is_preholiday=None),
})
result = get_calendar_features(
    datetime.fromisoformat("2026-10-10T18:00:00+03:00"),
    provider=provider,
)
print(result.model_dump_json(indent=2))
```

```json
{
  "timestamp": "2026-10-10T18:00:00+03:00",
  "day_of_week": 5,
  "hour": 18,
  "minute": 0,
  "is_weekend": true,
  "is_workday": false,
  "is_holiday": false,
  "is_preholiday": null,
  "day_type": "weekend"
}
```

`provider` — обязательный keyword-аргумент. Без данных о праздниках и переносах
модуль не может достоверно классифицировать день и не подставляет правило
«суббота/воскресенье = выходной».

## Источник календаря и CLI

`CalendarProvider.get_day(day: date) -> CalendarDay` получает **московскую дату**.
Будущий адаптер производственного календаря реализует этот интерфейс.
Он должен возвращать `day_type` и, при наличии данных, `is_preholiday`.
Неизвестную дату нужно отклонять, а не классифицировать по дню недели.

`MappingCalendarProvider` читает явно заданную таблицу дат. Формат JSON:

```json
{
  "2026-10-10": {"day_type": "weekend", "is_preholiday": null}
}
```

Загрузка: `MappingCalendarProvider.from_file("calendar.json")`.
`FakeCalendarProvider` использует такой же словарь для тестов, ничего не
скачивает и не имитирует официальный источник. В `examples/demo_calendar.json`
есть одна демонстрационная дата; этот файл не является производственным
календарём России и не предназначен для использования в ML как официальный источник.

Smoke test из корня проекта:

```sh
python3 -m external_data.calendar.cli \
  --timestamp "2026-10-10T18:00:00+03:00" \
  --calendar-file external_data/calendar/examples/demo_calendar.json
```

CLI требует `--calendar-file`, чтобы не выдавать предположение за данные
производственного календаря. Успешный JSON идёт в stdout, ошибки — в stderr.
Код 0 — успех, 1 — ошибка файла или неизвестная дата, 2 — ошибка аргументов.

## Время и значения полей

- Вход функции — timezone-aware `datetime`. Naive datetime отклоняется.
- Сначала выполняется перевод в `Europe/Moscow`, затем поиск даты у провайдера.
  Например, `2026-10-04T22:15:00Z` становится `2026-10-05T01:15:00+03:00`.
- `day_of_week`: понедельник = 0, воскресенье = 6. `hour`: 0–23, `minute`: 0–59.
  Timestamp сохраняет секунды и микросекунды, округления нет.
- `day_type` — одна из категорий `workday`, `weekend`, `holiday`.
  Соответствующие три булевых флага взаимоисключающие.
- `is_weekend` обозначает обычный/перенесённый выходной **по источнику**,
  исключая праздничную категорию. Рабочая суббота: `is_workday=true`,
  `is_weekend=false`. Праздник в воскресенье: `is_holiday=true`,
  `is_weekend=false`. Если нужен именно признак субботы/воскресенья,
  он определяется по `day_of_week`, это другое понятие.
- `is_preholiday` точно копируется из источника: true/false/null.
  Ни соседство с праздником, ни сокращённый рабочий день не предполагаются.

Pydantic проверяет типы, диапазоны и согласованность компонентов с timestamp
и day_type. Полнота и официальная достоверность календаря остаются обязанностью
провайдера. Unit tests используют фиксированные классификации, включая
синтетические переносы, а не утверждают официальный статус всех тестовых дат.
