# Единый контракт внешних признаков

`external_data.features` объединяет готовые результаты `weather`, `calendar`,
`events` и `railway` в Pydantic v2 модель `ExternalFeatures` для единицы данных
**station × timestamp**, пригодную для передачи ML-разработчику.
Модуль сам не получает данные и не вызывает API. Используются зависимости уже
установленных модулей; дополнительные пакеты не нужны.

## Использование

```python
from external_data.features import build_external_features

# timestamp: datetime с часовым поясом
# weather: WeatherObservation | None
# calendar: CalendarFeatures для того же момента времени
# events: list[Event]
# railway: RailwayFeatures | None для той же станции и момента времени
result = build_external_features(
    timestamp=timestamp,
    weather=weather,
    calendar=calendar,
    events=events,
    station="Площадь Восстания",
    railway=railway,
)
print(result.model_dump_json(indent=2))
```

Входной `timestamp` обязан содержать часовой пояс: naive datetime отклоняется.
Время переводится в `Europe/Moscow` без изменения момента времени.
`calendar` обязателен, его timestamp должен совпадать с целевым моментом;
несовпадение вызывает `ValueError`. Календарные значения копируются без
пересчёта, включая рабочие субботы и переносы из исходного календаря.
`day_of_week`: понедельник — 0, воскресенье — 6.

Погодные значения и их единицы сохраняются из `WeatherObservation`;
timestamp и координаты наблюдения не входят во вложенный объект `weather`.
При `weather=None` все девять погодных полей равны `null`.
Подбор наблюдения или прогноза для нужного времени выполняет вызывающий код.

События агрегируются по всему переданному списку:

- `event_count` — длина списка, `has_event` — есть ли хотя бы одно событие.
- `event_types` — уникальные ненулевые типы в порядке первого появления.
- `total_expected_people` — сумма только известных значений. Если известных
  значений нет (включая пустой список), результат `null`. Известный ноль
  сохраняется как `0`. При частично неизвестной посещаемости сумма неполная.

Вызывающий код передаёт события, уже отобранные для нужной станции и времени,
и отвечает за удаление дубликатов. Locations — отдельный вспомогательный модуль:
его результат используется вызывающим кодом для выбора `events_for_station`.
Builder не вызывает Geocoder и не определяет принадлежность событий к станции.

Готовый `RailwayFeatures` копируется во вложенный `railway` без пересчёта окон
и счётчиков. Дублирующие `timestamp` и `metro_station` во вложенный объект не
включаются. Перед копированием builder требует точного совпадения
`railway.metro_station == station` и совпадения момента `railway.timestamp`
с целевым timestamp. Несовпадение вызывает `ValueError`, включая передачу
railway при `station=None`. Эквивалентные моменты с разными UTC offsets допустимы.
`minutes_to_next_arrival=None` сохраняется.

Для станции без учитываемого вокзала передавайте `railway=None`. Null также
может означать, что данные не переданы; это не эквивалент нулевому числу прибытий.
Старые вызовы builder сохраняются: `station` по-прежнему optional, новый аргумент
`railway=None` добавлен в конец сигнатуры. Старый JSON без railway можно прочитать;
новый JSON содержит дополнительный ключ `railway`, в том числе со значением null.
**Для реального ML рекомендуется `station != null`.**
Входные объекты не изменяются.

## Граница ответственности

```text
Passenger flow / lags (отдельный слой)
                 +
ExternalFeatures: weather + calendar + events + railway
                 ↓
          CatBoost / LightGBM (следующий слой)
```

Это схема будущего использования данных, а не запуск моделей внутри модуля.
ExternalFeatures НЕ содержит passenger flow, lags, ML prediction или количество
составов метро. Эти данные и вычисления добавляются следующими слоями системы.
Railway описывает прибытия железнодорожных поездов и электричек к вокзалам,
а не движение составов метро.

## Локальный пример

Из корня репозитория:

```bash
python3 -m external_data.features.cli
```

CLI использует только фиксированные демонстрационные данные, без ключей и
сетевых запросов. Все значения, включая railway Московского вокзала и уже
отобранные для Площади Восстания события, демонстрационные. Погода, календарь
и расписание не являются реальными наблюдениями или проверенным календарём.

```json
{
  "timestamp": "2026-10-10T18:00:00+03:00",
  "station": "Площадь Восстания",
  "weather": {
    "temperature": 6.0,
    "feels_like": null,
    "precipitation": 1.2,
    "precipitation_type": "RAIN",
    "wind_speed": 4.3,
    "wind_gust": null,
    "humidity": 85.0,
    "pressure": 750.0,
    "condition": "RAIN"
  },
  "calendar": {
    "day_of_week": 5,
    "hour": 18,
    "minute": 0,
    "is_weekend": true,
    "is_workday": false,
    "is_holiday": false,
    "is_preholiday": null,
    "day_type": "weekend"
  },
  "events": {
    "event_count": 2,
    "event_types": ["concert", "festival"],
    "total_expected_people": 1000,
    "has_event": true
  },
  "railway": {
    "railway_name": "Московский вокзал",
    "arrivals_next_15m": 0,
    "arrivals_next_30m": 1,
    "arrivals_next_60m": 3,
    "arrivals_next_120m": 7,
    "train_arrivals_next_30m": 1,
    "suburban_arrivals_next_30m": 0,
    "minutes_to_next_arrival": 18.0
  }
}
```

## Тесты без интернета

```bash
python3 -m unittest discover -s external_data/features/tests
python3 -m unittest discover
```
