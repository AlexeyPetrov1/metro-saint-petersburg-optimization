# Использование из ML

После установки зависимостей и загрузки ключей (ниже) достаточно:

```python
from external_data import (
    configure_external_data,
    get_external_features,
    external_features_to_ml_dict,
)

configure_external_data(
    weather_coordinates=(59.9343, 30.3351),
)

events = []  # Или уже подготовленный list[Event] для этой станции и времени.
features = get_external_features(
    station="Площадь Восстания",
    timestamp="2026-10-12T18:00:00+03:00",
    events=events,
)
row = external_features_to_ml_dict(features)
```

Для обычной ML-интеграции не нужно напрямую использовать внутренние модули
`weather`, `calendar`, `railway`, `features` и создавать Yandex providers вручную.
Дата в примере показывает формат: доступность погодного прогноза зависит от
времени запуска, историческую погоду facade не загружает.

## Назначение и граница ответственности

`external_data` готовит внешние признаки для прогноза пассажиропотока
1 линии метро Санкт-Петербурга. Единица данных — **station × timestamp**.
Facade собирает погоду, календарь, агрегаты переданных событий и железнодорожные
прибытия. Строгие Pydantic-модели проверяют результат.

В `external_data` **не входят** `passenger_flow`, lag features, baseline,
ML prediction и расчёт количества составов метро. Это ответственность
следующего слоя. Здесь нет обучения моделей, БД, scheduler или API-сервера.

| Модуль | Назначение | Источник |
| --- | --- | --- |
| [events](events/README.md) | Сырой текст → `list[Event]` через extraction | YandexGPT или OpenRouter |
| [weather](weather/README.md) | Координаты → текущая погода / почасовой прогноз | Yandex Weather, GraphQL |
| [calendar](calendar/README.md) | Московская дата → тип дня и временные признаки | Локальная производственная таблица |
| [locations](locations/README.md) | Место/адрес → координаты и расстояния до станций линии 1 | Yandex Geocoder, Haversine |
| [railway](railway/README.md) | Прибытия на вокзал → счётчики по временным окнам | Yandex Rasp |
| [features](features/README.md) | Готовые результаты → `ExternalFeatures` | Без собственных API-запросов |

## Публичный API

Импортировать из `external_data`:

| API | Контракт |
| --- | --- |
| `configure_external_data(*, weather_coordinates, calendar_file=None, calendar_provider=None)` | Настройка один раз на процесс; возвращает `ExternalDataClient`. Сеть и ключи при настройке не требуются. |
| `get_external_features(station, timestamp, events=None)` | Получает реальные внешние данные и возвращает вложенную Pydantic-модель `ExternalFeatures`. |
| `external_features_to_ml_dict(features)` | Преобразует готовую модель в новый плоский `dict` из 31 поля; без сети. |
| `get_external_feature_rows(requests)` | Последовательно обрабатывает словари с `station`, `timestamp`, необязательным `events`; возвращает `list[dict]`. |
| `ExternalDataClient` | Независимый клиент с методами `get_external_features` и `get_external_feature_rows`; обычно получается из `configure_external_data`. |

Также доступны `Event`, `ExternalFeatures`, `ExternalDataConfigurationError`.
При необходимости нескольких независимых конфигураций храните возвращённый
клиент и вызывайте его методы. Прямой конструктор клиента требует
`calendar_provider` и `weather_coordinates`; для обычной интеграции он не нужен.
Последний успешный `configure_external_data` задаёт клиент функций facade;
неуспешная перенастройка не заменяет предыдущий.

`station` — название одной из 19 станций линии 1. Регистр и пробелы по краям
нормализуются, неизвестная станция вызывает ошибку. `timestamp` — aware `datetime`
или ISO 8601 со смещением, например `+03:00` или `Z`. Он переводится в
`Europe/Moscow`; naive timestamp отклоняется. Batch работает последовательно,
не кэширует ответы и прекращается на первой ошибке, без частичного результата.

## Установка и окружение

Из корня проекта, в выбранном Python-окружении:

```bash
python3 -m pip install \
  -r external_data/events/requirements.txt \
  -r external_data/weather/requirements.txt \
  -r external_data/calendar/requirements.txt \
  -r external_data/locations/requirements.txt \
  -r external_data/railway/requirements.txt
```

Переменные из корневого `.env.example`:

| Переменные | Когда нужны |
| --- | --- |
| `YANDEX_WEATHER_API_KEY` | Погодные запросы facade |
| `YANDEX_RASP_API_KEY` | Запросы для станций с железнодорожным узлом |
| `YANDEX_API_KEY`, `YANDEX_FOLDER_ID`, `YANDEX_MODEL` | Отдельный extraction через YandexGPT |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | Альтернативный extraction через OpenRouter |
| `YANDEX_GEOCODER_API_KEY` | Отдельный этап геопривязки |

Это отдельные ключи разных сервисов; ключ YandexGPT не заменяет Weather,
Geocoder или Rasp. Имена моделей задаются окружением. Calendar не требует ключа.
Facade сам не вызывает LLM и Geocoder, поэтому их ключи для уже готовых событий
не нужны. `.env` автоматически Python-модулями не загружается. Для собственного
доверенного файла в текущем shell:

```bash
set -a
source .env
set +a
```

Не коммитить `.env` и секреты, не печатать ключи и HTTP-заголовки.
При отсутствующей переменной, 401/403, 429, сетевой ошибке или некорректном ответе
facade передаёт исключение провайдера: ошибка API не превращается в fake-данные.

## Что получает ML

`features` содержит `timestamp`, `station` и блоки `weather`, `calendar`,
`events`, `railway`. `row` — их плоская проекция. Ниже пример формата
с условными погодными данными и прибытиями, не реальный ответ API:

```json
{
  "timestamp": "2026-10-12T18:00:00+03:00",
  "station": "Площадь Восстания",
  "temperature": 6.0,
  "feels_like": null,
  "precipitation": 1.2,
  "precipitation_type": "RAIN",
  "wind_speed": 4.3,
  "wind_gust": null,
  "humidity": 85.0,
  "pressure": 750.0,
  "condition": "RAIN",
  "day_of_week": 0,
  "hour": 18,
  "minute": 0,
  "is_weekend": false,
  "is_workday": true,
  "is_holiday": false,
  "is_preholiday": false,
  "day_type": "workday",
  "event_count": 0,
  "event_types": [],
  "total_expected_people": null,
  "has_event": false,
  "railway_name": "Московский вокзал",
  "railway_arrivals_next_15m": 0,
  "railway_arrivals_next_30m": 1,
  "railway_arrivals_next_60m": 3,
  "railway_arrivals_next_120m": 7,
  "train_arrivals_next_30m": 1,
  "suburban_arrivals_next_30m": 0,
  "minutes_to_next_arrival": 18.0
}
```

В Python `null` соответствует `None`; пропуски не заменяются нулями.
`timestamp` в flat dict — строка ISO 8601, `event_types` остаётся списком.
`day_of_week`: понедельник 0, воскресенье 6. Температура — °C, ветер — м/с,
влажность — %, давление — мм рт. ст.; прогнозные осадки — мм за час.
Кодирование категорий и списков для ML выполняет следующий слой.

## Events и locations

Facade принимает уже подготовленный `list[Event]`. Автоматического поиска
событий в интернете нет. События нужно заранее отобрать для станции и времени
и удалить дубликаты: facade агрегирует весь переданный список.
`events=None` или `[]` дают `event_count=0`, но не доказывают отсутствие
событий в городе. Типы уникальны в порядке первого появления;
посещаемость — сумма только известных значений или `None`, если неизвестны все.
При частичных данных сумма не является полной ожидаемой посещаемостью.

Геопривязка `Event.location_name`/адреса через `locations` — отдельный
enrichment-этап. Он возвращает координаты и расстояния до нескольких станций.
Facade не вызывает геокодер и не выбирает радиус влияния мероприятия.

## Calendar: встроенный 2026 год

По умолчанию загружается
[`calendar/data/ru_production_calendar_2026.json`](calendar/data/ru_production_calendar_2026.json):
все 365 дат федерального производственного календаря для пятидневной недели.
[Источники и правила классификации](calendar/data/SOURCES.md) находятся рядом.
Это не демонстрационный `examples/demo_calendar.json`.

Приоритет: явный `calendar_provider` → явный `calendar_file` → встроенный 2026.
Если переданы оба override, используется provider, файл не читается.

```python
configure_external_data(
    calendar_file="custom_calendar.json",
    weather_coordinates=(59.9343, 30.3351),
)
# Для собственного источника вместо calendar_file: calendar_provider=my_provider.
```

Файл загружается и валидируется при настройке. Встроенный путь не зависит от
рабочей директории. Нет данных для московской даты →
`CalendarDataUnavailableError`, в частности за пределами 2026 года для default.
Неисправный custom-файл также вызывает ошибку, без перехода на default/demo.
Существующий отдельный calendar CLI по-прежнему требует свой `--calendar-file`.
Флаги типов дней взаимоисключающие: праздник в воскресенье — `is_holiday=true`,
`is_weekend=false`; перенесённый выходной — `weekend`. Сокращённый рабочий день
остаётся `workday`, с `is_preholiday=true`.

## Weather: координаты и ограничение времени

Общая пара `weather_coordinates=(lat, lon)` используется для всех станций.
Можно передать `{station: (lat, lon)}` для разных точек. Координаты не выводятся
из названия станции автоматически; отсутствующая точка в mapping — ошибка.

Почасовой forecast применяется без изменения значений на интервале
**[начало часа, начало следующего часа)**. Все 15-минутные точки внутри часа
используют одну и ту же прогнозную точку:

1. Запрашивается текущая погода. Её timestamp — корневой `serverTime` API,
   то есть время ответа сервера, не время измерения метеостанции.
2. Из current weather используется только время сервера для расчёта горизонта.
   Погодные значения current не подставляются даже при совпадении timestamp.
3. Если цель не раньше `serverTime` и не дальше 48 часов, запрашивается
   почасовой forecast на `max(1, ceil(разницы в часах))` часов.
4. Целевой timestamp переводится в `Europe/Moscow`; минуты, секунды и
   микросекунды обнуляются только для поиска `ForecastHour.time`.
   Выбирается исключительно точка начала соответствующего часа.
5. Нужной часовой точки нет, цель в прошлом или дальше 48 часов → все девять
   погодных полей `None`. API-ошибки при этом не скрываются.

Weather provider возвращает только точки в диапазоне от `serverTime` ответа
прогноза до `serverTime + hours`. Два HTTP-запроса имеют разные serverTime;
на границе часа нужная точка может уже оказаться раньше второго serverTime.
Исторические данные и точка начала уже текущего часа не гарантируются.

Если доступен прогноз на 18:00 и 19:00 и обе точки ещё внутри горизонта:

| ML timestamp | Результат |
| --- | --- |
| 18:00 | Точка прогноза 18:00 |
| 18:15 | Точка прогноза 18:00 |
| 18:30 | Точка прогноза 18:00 |
| 18:45 | Точка прогноза 18:00 |
| 19:00 | Точка прогноза 19:00 |

Если API содержит только 18:00, для 19:00 также будет `None`.
Интерполяции и смешивания соседних часов нет. Точка будущего часа не применяется
к более раннему timestamp; предыдущий час не заполняет пропуск следующего.
В `ExternalFeatures` и flat dict сохраняется исходный ML timestamp в
`Europe/Moscow`, а исходная `WeatherObservation` не изменяется. Осадки остаются
количеством за прогнозный час, не пересчитываются в количество за 15 минут.
Если API не вернул нужный час либо provider исключил его как прошедший,
погодные поля остаются `None`, без подстановки current weather или соседнего часа.

## Railway

Это прибытия **обычных железнодорожных поездов и электричек**, не поездов метро.
Узлы: Московский вокзал → Площадь Восстания, Финляндский → Площадь Ленина,
Балтийский → Балтийская, Девяткино → Девяткино. Для остальных станций API Rasp
не вызывается, `features.railway=None`, все восемь flat railway-полей — `None`.

Счётчики отражают прибытия в следующие 15/30/60/120 минут по расписанию,
а не число пассажиров. Railway не оценивает вместимость составов метро.

## Проверки

Все unit tests без настоящих API-ключей и интернета:

```bash
python3 -m unittest discover
```

Fake/mock providers используются для изолированных unit tests; production
facade по умолчанию использует реальные Weather/Rasp providers и локальный
производственный календарь. Он не переключается на fake при ошибках API.
Отдельные демонстрационные CLI не являются источником данных для facade.

Реальная проверка внешних интеграций из корня проекта:

```bash
./scripts/smoke_external_apis.sh
```

Скрипт запускает Events/YandexGPT, Weather, Locations/Geocoder и Railway/Rasp
с реальными запросами, а Calendar и Features — локально. Требуются ключи,
расходуется квота API. Отчёт и результаты сохраняются в `artifacts/`.
Это проверка отдельных интеграций; она не подтверждает заполнение погоды
на всех 15-минутных timestamp и не заменяет проверку facade на нужных датах.
