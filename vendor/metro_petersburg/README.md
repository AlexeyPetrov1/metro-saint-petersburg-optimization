# Прогноз пассажиропотока 1 линии метро СПб

Репозиторий команды Metro Petersburg. Две части:

- [ML-часть](#ml-часть-прогноз-входов) — прогноз входов на станции (`src/`, Макар);
- [внешние данные](#внешние-данные-external_data) — погода, календарь, события, ж/д расписания (`external_data/`, Денис).

## Как устроен репозиторий

| Папка / файл | Что | Отвечает |
|---|---|---|
| `src/` | ML: очистка, признаки, модель, бэктест, прогноз ([src/serve.py](src/serve.py)) | Макар |
| `tests/` | тесты ML и переходника (pytest) | Макар |
| `docs/` | ТЗ, EDA, бэктест, карточка модели, ход работ, отчёт | Макар |
| `data/reference/` | ручные справочники 1 линии: станции, вестибюли, события, инциденты | Макар |
| [data/reference/intrahour_profile.csv](data/reference/intrahour_profile.csv) | профиль потока внутри часа по 15-минутным данным — для симулятора ([описание](#5-профиль-внутри-часа-для-симулятора)) | Макар |
| `notebooks/`, `reports/` | EDA, таблицы и графики бэктеста | Макар |
| `external_data/` | внешние данные: погода, календарь, события, геокодер, ж/д расписания; тесты — в `*/tests/` (unittest) | Денис |
| `scripts/`, `.env.example` | smoke-проверка внешних API, шаблон ключей | Денис |
| [src/external_adapter.py](src/external_adapter.py), колонка `name_external` в [станциях](data/reference/spb_line1_stations.csv) | стык: external_data → прогноз | Макар, правки — вместе с Денисом |

`data/raw/`, `data/interim/`, `models/` в git не хранятся: данные и модели собираются командами из раздела
[«Как получить прогноз»](#как-получить-прогноз).

```text
external_data.weather   ── осадки (Яндекс Погода) ────────┐
external_data.railway   ── прибытия поездов (Расписания) ─┤
external_data.events    ── события из текстов ────────────┤
external_data.locations ── ближайшие станции к площадкам ─┘
                                   ↓
                      src/external_adapter.py
                        осадки → тот же признак модели; поезда и события → context
                                   ↓
                      src/serve.py: forecast(now)
                        records (контракт) · explanations (причины + context) · meta
```

Модель обучена на потоке, календаре и историческом прогнозе осадков Open-Meteo. Погода Яндекса подставляется в тот же
признак осадков. Поезда и события в модель не подаются, потому что она на них не обучена: они идут в
`explanations[].context` как материал для LLM-слоя. Замороженную модель переходник не меняет.

Установка обеих частей одной командой (Python 3.12, `external_data` требует 3.11+):

```bash
pip install -r requirements.txt
```

Тесты, без сети и ключей:

```bash
pytest -q                     # ML и переходник (tests/); тесты на data/interim и models/ пропускаются, если их нет
python -m unittest discover   # external_data
```

## Ключи API

Ключи передаются только через переменные окружения. Локальный `.env` (шаблон — [.env.example](.env.example)) в git не
попадает (`.gitignore`) и сам не загружается: перед запуском выполните `set -a; source .env; set +a`. ML-часть работает
без ключей: Open-Meteo и производственный календарь их не требуют. Ключи нужны только для живых внешних данных.
Тестам обеих частей ключи не нужны.

| Переменная | Сервис | Кто читает | Если не задана |
|---|---|---|---|
| `YANDEX_WEATHER_API_KEY` | Яндекс Погода | `serve.forecast` для «сейчас», `external_data.weather` | погода Open-Meteo, предупреждение в `meta.weather.warning` |
| `YANDEX_RASP_API_KEY` | Яндекс Расписания | `serve.forecast` для «сейчас» (`context.railway`), `external_data.railway` | `context.railway = null` |
| `YANDEX_GEOCODER_API_KEY` | Геокодер Яндекс Карт | `external_data.locations`, `external_adapter.locate_events` | события не привязываются к станциям |
| `YANDEX_API_KEY`, `YANDEX_FOLDER_ID`, `YANDEX_MODEL` | YandexGPT | `external_data.events` — события из текста | события не извлекаются |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | OpenRouter | `external_data.events --provider openrouter` | для Яндекса не нужны |

Ключи разных сервисов разные: ключ YandexGPT для Погоды не подходит. Подробнее о получении ключей — в README модулей
`external_data/*`.

## ML-часть: прогноз входов

ML-часть проекта: прогноз входов на 18 станций 1 линии на 1 и 2 часа вперёд с интервалом q10–q90, флагом аномалии
и тремя причинами прогноза для LLM-слоя.

- Модель — [docs/model_card.md](docs/model_card.md).
- Бэктест и финальный тест — [docs/backtest.md](docs/backtest.md).
- Ход работ — [docs/progress.md](docs/progress.md).
- 15-минутные данные, профиль внутри часа и ранний сигнал — [docs/intrahour_findings.md](docs/intrahour_findings.md).
- Контракт выхода — [src/contract.py](src/contract.py) и [data/predictions/contract.schema.json](data/predictions/contract.schema.json).

### Как получить прогноз

#### 1. Установка

Нужен Python 3.12. Файлы организаторов (`Пассажиропоток 2026 Линия 1.xlsx` и др.) кладутся в `data/raw/spb/`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.download --only weather_spb calendar   # погода Open-Meteo и производственный календарь
python -m src.clean_spb                              # → data/interim/
```

#### 2. Модели

Модели не хранятся в git, их нужно собрать (около 5 минут). Конфигурация — замороженная
[reports/backtest/model_final.json](reports/backtest/model_final.json); команды проверяют её хэш.

```bash
python -m src.model --save-folds        # models/lgbm_fold_2026-05 … 2026-08: обучение до начала каждого месяца
python -m src.serve --train --holdout   # models/lgbm_holdout: обучение по 24.08 (та же модель, что в финальном тесте)
python -m src.serve --train             # models/lgbm_final: все данные по 29.09
```

Сам финальный тест (`python -m src.model --final`) уже проведён и повторно не запускается.

#### 3. Прогноз

```bash
python -m src.serve --now "2026-08-31 08:59"
python -m src.serve --now "2026-08-31 08:59" --stations devyatkino,narvskaya --json forecast.json
```

Из Python (например, в Streamlit):

```python
from src import serve

fc = serve.forecast("2026-08-31 08:59")     # данные только до now; модель выбирается автоматически
fc.records        # прогноз по контракту: 18 станций × горизонты 60 и 120 мин
fc.explanations   # топ-3 причины на каждую запись
fc.meta           # какая модель, до какой даты обучена, вне выборки ли прогноз
serve.table(fc)   # всё одной таблицей
```

**Момент прогноза.** Час hh считается закрытым в hh:59. `now = 08:59` — это данные по 08 ч включительно и слоты 09:00
(горизонт 60 мин) и 10:00 (120 мин). Поток после этого часа не используется — это проверяет тест.

**Режим повтора.** Можно задать любой исторический `now` с 09.02.2026. По умолчанию (`--model auto`) берётся
самая свежая модель, у которой последние сутки обучения + 7 суток раньше `now`. Так прогноз на прошлую дату делает
модель, обученная до неё:

| `now` | Модель | Обучена по |
|---|---|---|
| 30.04 – 30.05 | `lgbm_fold_2026-05` | 23.04 |
| 31.05 – 29.06 | `lgbm_fold_2026-06` | 24.05 |
| 30.06 – 30.07 | `lgbm_fold_2026-07` | 23.06 |
| 31.07 – 30.08 | `lgbm_fold_2026-08` | 24.07 |
| 31.08 – 05.10 | `lgbm_holdout` | 24.08 |
| с 06.10 | `lgbm_final` | 29.09 |

Раньше 30.04 подходящей модели нет. Тогда используется `lgbm_final` с предупреждением «модель видела этот период»
в `meta.warning`: её веса обучены на более поздних данных. Модель можно задать и явно: `--model lgbm_final`.

**Погода.** `--weather` / `weather_source`: `auto` (по умолчанию), `openmeteo`, `yandex`. В `auto` для «сейчас» (`now`
в пределах часа от текущего времени) берётся прогноз Яндекс Погоды, для прошлых `now` — исторический прогноз
Open-Meteo, как при обучении. Без `YANDEX_WEATHER_API_KEY` или при ошибке API используется Open-Meteo с
предупреждением в `meta.weather.warning`.

Признаки модели — осадки за час до слота τ и за 3 часа до него: Open-Meteo метит час его концом, Яндекс — началом,
переходник сдвигает часы Яндекса на 1 ч. Для слотов 09 и 10 нужны часы 06–10, а к 08:59 они уже начались. Поэтому из
Яндекса берётся и прошедшая часть сегодняшнего прогноза (`past_hours`). Часы, которых Яндекс не дал, берутся
из Open-Meteo; источник каждого часа записан в `meta.weather.hours`.

Режим «сейчас» пока проверяется только тестами: поток в `data/interim` заканчивается 29.09, а живого потока нет.
`forecast()` на сегодняшнюю дату отвечает «вне периода данных».

```bash
python -m src.serve --now "2026-08-31 08:59" --weather yandex   # без ключа: Open-Meteo и предупреждение
```

**Контекст для LLM-слоя.** Поезда и события модель не видит, они передаются в `explanations[].context`:

```python
from external_data.locations import YandexGeocoderProvider
from external_data.railway import YandexRaspProvider
from src import external_adapter as A

located, skipped = A.locate_events(events, YandexGeocoderProvider())   # events — list[Event] из external_data.events
fc = serve.forecast(now, railway_provider=YandexRaspProvider(), events=located)
```

- Поезда — для 4 станций с вокзалом (Площадь Восстания, Площадь Ленина, Балтийская, Девяткино), на начало слота.
  `arrivals_next_60m` — прибытия за час слота. Для «сейчас» без провайдера Rasp подключается сам, если задан
  `YANDEX_RASP_API_KEY`.
- События — станция среди ближайших к площадке не дальше 1,5 км, слот — от начала события − 3 ч до его конца
  (или начала) + 3 ч. Это эвристика для контекста (`EVENT_RADIUS_M`, `EVENT_WINDOW`), а не признак модели.
- Ошибки этих источников прогноз не роняют: они пишутся в `meta.context.warning`.

#### 4. Формат выхода

`records` — список записей контракта ([src/contract.py](src/contract.py)). Пример — `--now "2026-08-31 08:59"`,
Девяткино, слот 09:00:

```json
{"station_id": "devyatkino", "ts": "2026-08-31T09:00:00+03:00", "horizon_min": 60,
 "q10": 6910, "q50": 7178, "q90": 7552, "baseline": 8295, "is_anomaly": true,
 "model_version": "lgbm_q_ratio_v1/lgbm_holdout"}
```

- `ts` — начало часового слота, местное время.
- `q10` / `q50` / `q90` — квантили входов на станцию за час.
- `baseline` — норма (медиана 4 последних таких же дней).
- `is_anomaly` — |q50 / baseline − 1| выше p80 группы станций и периода суток (порог выбран на бэктесте,
  [docs/backtest.md](docs/backtest.md), раздел «Флаг аномалии»).
- `model_version` — версия модели и папка в `models/`.

`explanations` — на каждую запись три признака с наибольшим вкладом в q50:

```json
{"station_id": "devyatkino", "ts": "2026-08-31T09:00:00+03:00", "horizon_min": 60,
 "reasons": [{"feature": "r_t", "text": "в последний час поток на 75 % ниже нормы", "effect_pct": -7.1},
             {"feature": "r_line", "text": "вся линия на 25 % ниже нормы", "effect_pct": -4.8},
             {"feature": "b4_tau", "text": "поправка на величину нормы (8 295 входов в 09 ч)", "effect_pct": -2.8}]}
```

`effect_pct` — на сколько процентов признак сдвинул прогноз q50 относительно среднего прогноза модели.

`context` — поезда и события рядом (пример: Площадь Восстания, слот 09:00, переданы провайдер поездов и событие):

```json
{"railway": {"timestamp": "2026-08-31T09:00:00+03:00", "railway_name": "Московский вокзал",
             "arrivals_next_15m": 0, "arrivals_next_30m": 1, "arrivals_next_60m": 3, "arrivals_next_120m": 3,
             "train_arrivals_next_30m": 1, "suburban_arrivals_next_30m": 0, "minutes_to_next_arrival": 18.0},
 "events": [{"event_name": "Barcelona Flamenco Ballet", "event_type": "concert",
             "start_time": "2026-08-31T11:30:00+03:00", "end_time": null, "location_name": "БКЗ Октябрьский",
             "expected_people": null, "source_url": null, "distance_m": 151}]}
```

Если источник не подключён или у станции нет вокзала, будет `{"railway": null, "events": []}`. Поля `railway` взяты
из контракта `RailwayFeatures` external_data.

`meta` — какая модель отвечала, откуда погода и какие внешние источники подключены:

```json
{"model": "lgbm_holdout", "train_start": "2026-02-09", "train_end": "2026-08-24", "out_of_sample": true,
 "warning": null, "now": "2026-08-31 08:59:00", "t0": "2026-08-31T08:00:00+03:00", "config_sha256": "2152d675…",
 "weather": {"requested": "auto", "source": "openmeteo", "warning": null,
             "hours": {"2026-08-31T06:00:00+03:00": "openmeteo", "2026-08-31T07:00:00+03:00": "openmeteo",
                       "2026-08-31T08:00:00+03:00": "openmeteo", "2026-08-31T09:00:00+03:00": "openmeteo"},
             "features": [{"ts": "2026-08-31T09:00:00+03:00", "fc_precip_tau": 0.0, "fc_precip_3h_tau": 0.6},
                          {"ts": "2026-08-31T10:00:00+03:00", "fc_precip_tau": 0.0, "fc_precip_3h_tau": 0.4}]},
 "context": {"railway": null, "events": 0, "warning": null}}
```

- `weather.source` — `openmeteo`, `yandex` или `yandex+openmeteo` (часть часов из каждого источника).
- `weather.hours` — начало часа осадков → источник: `yandex`, `openmeteo` или `нет данных`.
- `weather.features` — признаки погоды, которые видела модель, по слотам.
- `context.railway` — имя провайдера поездов или `null`; `context.events` — сколько событий передано.

Время ответа — около 2 секунд: нормы и признаки пересчитываются по данным до `now`.

#### 5. Профиль внутри часа (для симулятора)

[data/reference/intrahour_profile.csv](data/reference/intrahour_profile.csv) — доля входов часа, которая приходится
на каждую четверть часа. Построен по 15-минутным данным организаторов за февраль, май, июль и сентябрь 2026:
обычные сутки, без закрытий и выбросов. Пересобрать:

```bash
python -m src.clean_spb_15min   # 15-минутные файлы из data/raw/spb/15min/ → data/interim/spb_line1_15min.parquet
python -m src.intrahour         # → data/reference/intrahour_profile.csv и таблицы reports/intrahour/
```

| Колонка | Что |
|---|---|
| `station_id` | станция, как в [spb_line1_stations.csv](data/reference/spb_line1_stations.csv) |
| `vestibule_id` | вестибюль или `*` — станция целиком (сумма её вестибюлей) |
| `day_type` | `рабочий`, `суббота`, `воскресенье`; праздник берите как воскресенье |
| `hour` | час местного времени, 5…23 и 0; в 05 ч метро открывается около 05:30. Часы вне режима вестибюля не выводятся |
| `quarter` | 0 — :00–:14, 1 — :15–:29, 2 — :30–:44, 3 — :45–:59 |
| `share` | доля четверти; по каждой четвёрке (станция, вестибюль, тип дня, час) сумма = 1 |
| `n_days` | по скольким суткам посчитана доля |
| `source` | откуда доля: `вестибюль` / `станция` — своя ячейка; `группа` или `линия` — подстановка, если своих суток < 3 или входов < 200 |
| `note` | пометка; у Технологического института — профиль только по 4 месяцам 15-минутных данных (в часовых данных его нет) |

Поток за 15 минут = прогноз часа × `share`. Днём профиль почти ровный (около 5–6 % часа не на своём месте).
Резкие часы:
- открытие в 05 ч и полночь;
- утро рабочего дня: в 07 ч поток растёт к концу часа, в 08 ч убывает.

У вокзалов (Девяткино, пл. Ленина, пл. Восстания, Балтийская) доли сильнее гуляют день ото дня. Подробности —
[docs/intrahour_findings.md](docs/intrahour_findings.md).

## Внешние данные (external_data/)

Проект готовит внешние признаки для прогнозирования пассажиропотока **1 линии
метро Санкт-Петербурга**: погоду, календарь, городские события и прибытия
железнодорожных поездов к связанным станциям метро.

Единица итоговых данных — **station × timestamp**, например
«Площадь Восстания × 2026-10-08 18:00 Europe/Moscow».
Результат — Pydantic v2 модель `ExternalFeatures`, которую можно сериализовать
в JSON и передать ML-разработчику. Сейчас реализован слой внешних данных;
обучения и запуска ML-модели в `external_data` нет — они в `src/` (раздел выше).

### Архитектура

```text
Yandex Weather ──→ weather ───────────────────────────┐
YandexGPT ───────→ events ──→ события для станции ─────┤
                             ↑                      │
Yandex Geocoder → locations → координаты и расстояния│
Yandex Rasp ────→ railway ────────────────────────────┤
Локальный календарь → calendar ───────────────────────┘
                            ↓
               build_external_features(...)
                            ↓
                    ExternalFeatures
                            ↓
              ML model — будущий отдельный слой
```

Это схема передачи готовых результатов, а не автоматический сборщик всех API.
Модули вызываются независимо. `features` не делает сетевых запросов: вызывающий
код подбирает данные для станции и времени, затем передаёт их builder.

`locations` дополняет события географическими данными: явно указанный адрес
(предпочтительно) или название площадки → координаты → расстояния Haversine
до 19 станций линии 1. Он возвращает несколько ближайших станций, но не задаёт
радиус влияния и не определяет силу воздействия события. Выбор
`events_for_station` остаётся за вызывающим кодом.

**Passenger flow и lags не входят в ExternalFeatures.** ML-прогноз и расчёт
количества составов метро относятся к следующим слоям системы, которых здесь нет.

### Модули

Подробности контрактов, ограничений и Python API — в README каждого модуля.

| Модуль | Назначение | Вход | Выход | Внешний API |
| --- | --- | --- | --- | --- |
| [external_data/events](external_data/events/README.md) | Извлечение событий из уже полученного текста | Текст, published_at, метаданные источника, LLM provider | `EventExtractionResult` с `list[Event]` | YandexGPT или OpenRouter; есть fake |
| [external_data/weather](external_data/weather/README.md) | Текущая погода и почасовой прогноз | Координаты, горизонт прогноза | `WeatherObservation` или список наблюдений | Yandex Weather |
| [external_data/calendar](external_data/calendar/README.md) | Календарные признаки с учётом источника рабочих/выходных дней | Timestamp и calendar provider / локальная таблица | `CalendarFeatures` | Нет |
| [external_data/locations](external_data/locations/README.md) | Координаты места и ближайшие станции линии 1 | Название, optional address, geocoder, top_k | `LocationResolution` | Yandex Maps Geocoder; есть fake |
| [external_data/railway](external_data/railway/README.md) | Прибытия поездов и электричек в окнах 15/30/60/120 минут | Hub, timestamp, railway provider | `RailwayArrival` и `RailwayFeatures` | Yandex Rasp; есть fake |
| [external_data/features](external_data/features/README.md) | Единый объект для станции и времени | Готовые weather, calendar, events, railway | `ExternalFeatures` | Нет |

### API

Здесь перечислены API, которые используют существующие адаптеры:

- **YandexGPT / Yandex AI Studio** — превращает сырой текст в `Event` через
  OpenAI-compatible endpoint `https://ai.api.cloud.yandex.net/v1`.
  Возвращает structured JSON с Pydantic validation. Адрес разрешён только при
  наличии в исходном тексте; parser проверяет его подстрокой с нормализацией
  регистра и пробелов. `source_fragment` проверяется строго посимвольно.
- **Yandex Weather** — текущая погода и почасовые точки прогноза через
  `https://api.weather.yandex.ru/graphql/query`; авторизация заголовком
  `X-Yandex-Weather-Key`. Интерполяции до 15 минут и запроса исторической погоды нет.
- **Yandex Maps Geocoder** — адрес → координаты через
  `https://geocode-maps.yandex.ru/v1/`. Есть отдельные проверки результатов для
  адреса и названия площадки; сомнительный результат может быть отклонён.
- **Yandex Rasp** — прибытия поездов/электричек через
  `https://api.rasp.yandex-net.ru/v3.0/schedule/`, `event=arrival`.
  Используются train/suburban, pagination и обе даты при переходе окна через полночь.
- **Calendar** — локальный модуль, не API. Для CLI нужна явная таблица дат;
  одна demo-дата в репозитории не заменяет производственный календарь России.

Ключи этих четырёх сервисов **разные**. Подключение каждого сервиса описано в
соответствующем README выше. Для событий также реализован альтернативный
OpenRouter provider; он не обязателен для Yandex smoke tests.

### Установка

Нужен Python **3.11+**. Все команды ниже выполняются из корня репозитория.
Пример для bash/zsh:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install \
  -r external_data/events/requirements.txt \
  -r external_data/weather/requirements.txt \
  -r external_data/calendar/requirements.txt \
  -r external_data/locations/requirements.txt \
  -r external_data/railway/requirements.txt
```

Корневой [requirements.txt](requirements.txt) после объединения включает и эти зависимости;
команда выше ставит только `external_data`. `features` использует зависимости
существующих модулей и отдельного requirements не имеет. Основные библиотеки:
Pydantic v2, официальный OpenAI Python SDK и httpx.

Для запуска через pytest установите его отдельно — в requirements модулей
он не перечислен:

```bash
python3 -m pip install pytest
```

Без ключей и интернета после установки доступны demo `calendar`, demo `features`
и все unit tests.

### Environment

Создайте локальный `.env` на основе [.env.example](.env.example), если его ещё нет:

```bash
test -f .env || cp .env.example .env
```

Заполните ключи самостоятельно. Пример конфигурации Yandex без секретов:

```dotenv
YANDEX_API_KEY=
YANDEX_FOLDER_ID=
YANDEX_MODEL=yandexgpt-5-lite
YANDEX_WEATHER_API_KEY=
YANDEX_GEOCODER_API_KEY=
YANDEX_RASP_API_KEY=
```

В текущем файле `.env.example` значение модели — `yandexgpt/latest`; выше показан
вариант локальной настройки с `yandexgpt-5-lite`. Укажите доступное вашему
каталогу короткое имя модели. Provider не выбирает модель по умолчанию:
из `YANDEX_FOLDER_ID` и `YANDEX_MODEL` он собирает
`gpt://<folder_id>/<model>`. Folder ID должен относиться к каталогу с нужными правами.

Шаблон также содержит `OPENROUTER_API_KEY` и `OPENROUTER_MODEL`. Заполняйте их
только для `events --provider openrouter`; для Yandex они не нужны.

Отдельные CLI **не загружают `.env` автоматически**. Перед их запуском
экспортируйте настройки в текущем shell:

```bash
set -a
source .env
set +a
```

`source` выполняет shell-код: используйте только собственный доверенный `.env`.
Общий smoke-скрипт сам читает `.env` как literal assignments, без исполнения
команд и подстановок переменных.

### Запуск модулей

Даты ниже — фиксированные примеры. Для другой публикации передавайте её реальный
`published_at`, для актуального железнодорожного расписания — нужный timestamp.
Datetime должен содержать timezone; основная временная шкала — `Europe/Moscow`.
CLI печатают JSON в stdout, диагностику — в stderr.

#### Events: YandexGPT

```bash
python3 -m external_data.events.cli \
  --provider yandex \
  --text "12 октября в 19:30 в БКЗ Октябрьский по адресу Лиговский проспект, 6 состоится концерт Barcelona Flamenco Ballet" \
  --published-at "2026-10-07T12:00:00+03:00" \
  --source-type telegram
```

Также доступны `--source-url` и `--debug`. Одно сообщение может дать несколько
событий. Неизвестные значения остаются null; посещаемость не выводится из
вместимости площадки. Модуль не скачивает посты и не подключается к Telegram.
Для OpenRouter замените provider на `openrouter`; без флага это текущий default.

#### Weather: текущая погода и прогноз

```bash
python3 -m external_data.weather.cli --lat 59.9343 --lon 30.3351
python3 -m external_data.weather.cli --lat 59.9343 --lon 30.3351 --hours 2
```

`--hours` принимает 1–48. Базовый запрос не включает `feelsLike`; опция
`--with-feels-like` требует соответствующего доступа по тарифу и на Test может
вернуть GraphQL error. У текущей погоды `precipitation=null`; количество осадков
берётся из почасового прогноза. Пропуски не заполняются выдуманными значениями.

#### Calendar: локальная demo-таблица

```bash
python3 -m external_data.calendar.cli \
  --timestamp "2026-10-10T18:00:00+03:00" \
  --calendar-file external_data/calendar/examples/demo_calendar.json
```

Для других дат передайте собственную таблицу производственного календаря.
Неизвестная дата вызывает ошибку; рабочие субботы и переносы не угадываются.

#### Locations: адрес и ближайшие станции

```bash
python3 -m external_data.locations.cli \
  --location "БКЗ Октябрьский" \
  --address "Лиговский проспект, 6" \
  --top-k 3
```

Адрес имеет приоритет. Без `--address` используется название с проверкой
кандидатов и ограниченными fallback-запросами. `--debug` помогает проверить
фактически найденный адрес и координаты. Возвращаются расстояния в метрах,
не маршруты и не время пути. Кэш координат станций живёт только в памяти процесса.

#### Railway: прибытия к вокзалу

```bash
python3 -m external_data.railway.cli \
  --hub "Московский вокзал" \
  --timestamp "2026-10-08T18:00:00+03:00"
```

Доступны четыре hub: Московский вокзал → Площадь Восстания, Финляндский вокзал
→ Площадь Ленина, Балтийский вокзал → Балтийская, Девяткино → Девяткино.
Окна включают границы: `T <= arrival_time <= T + window`.
Это прибытия **железнодорожных поездов и электричек**, не составов метро.

#### Features: итоговый объект без API

```bash
python3 -m external_data.features.cli
```

CLI использует только локальные фиксированные demo-значения, включая расписание
и уже отобранные для станции события. Это демонстрация контракта, не сбор
актуальных данных всех сервисов.

### ExternalFeatures

Итог содержит `station`, `timestamp` и четыре блока: `weather`, `calendar`,
`events`, `railway`. `locations` служит отдельным enrichment для событий;
координаты и расстояния не входят в этот ML-контракт.

Builder принимает уже подготовленные объекты:

```python
from external_data.features import build_external_features

# timestamp: aware datetime
# weather: WeatherObservation | None
# calendar: CalendarFeatures для timestamp
# events_for_station: list[Event], отобранные вызывающим кодом
# railway: RailwayFeatures | None для station и timestamp
result = build_external_features(
    timestamp=timestamp,
    station="Площадь Восстания",
    weather=weather,
    calendar=calendar,
    events=events_for_station,
    railway=railway,
)
payload = result.model_dump(mode="json")
```

Важные правила:

- Для реального ML задавайте `station != null`. Optional station сохранён
  для обратной совместимости; с переданным railway станция обязательна.
- `calendar.timestamp` и `railway.timestamp` должны совпадать с моментом
  итогового объекта, а `railway.metro_station` — с `station`.
- Подбор погоды на нужное время и отбор событий по времени/станции выполняются
  до builder. Builder не вычисляет длительность или влияние события.
- `weather=None` даёт null во всех погодных полях; `railway=None` допустим,
  например для станции без учитываемого вокзала. Null не означает ноль.
- `event_count` — длина списка, `event_types` — уникальные ненулевые типы в
  порядке появления. `total_expected_people` суммирует только известные
  значения; если нет ни одного, результат null. Это не оценка пассажиропотока.
- Счётчики railway копируются без пересчёта. Его `minutes_to_next_arrival`
  может быть null или превышать 120, если ближайший загруженный рейс позже окон.

Полный JSON текущего demo CLI:

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

### Testing

Все unit tests запускаются из корня, без интернета и реальных API keys:

```bash
python3 -m unittest discover
```

Тесты написаны на `unittest` и совместимы с pytest: `python3 -m pytest external_data`.
Просто `pytest` из корня запускает тесты ML-части (`testpaths = tests` в [pytest.ini](pytest.ini)).

#### Общий smoke test

Существующий [scripts/smoke_external_apis.sh](scripts/smoke_external_apis.sh)
выполняет четыре **реальных API-проверки** и две локальные demo-проверки:

```bash
./scripts/smoke_external_apis.sh
```

Скрипт сам переходит в корень и читает `.env`, проверяет шесть Yandex-переменных
с выводом только `SET` / `MISSING`, запускает Events, Weather, Locations,
Railway, Calendar и Features. Для каждого сохраняет stdout, exit code и stderr
при ошибке или наличии диагностики. После FAIL остальные проверки продолжаются.

Результаты:

- `artifacts/external_api_smoke_<YYYY-MM-DD_HHMMSS>.md` — общий отчёт;
- одноимённая директория — очищенные от секретов `.stdout.json`, `.stderr.txt`
  и `.exitcode` для отдельных проверок.

`PASS` означает exit code 0, а не доказательство смысловой точности extraction
или геокодинга. Timeout каждой проверки — 300 секунд; общий exit code ненулевой
при сбое проверки или загрузки конфигурации. Smoke расходует API-квоты и
использует фиксированные даты из скрипта; отдельные CLI выше позволяют выбрать
другие даты без изменения скрипта. Календарь и features в smoke остаются demo.

### Что передаётся Макару / ML

Команда внешних данных передаёт `ExternalFeatures` для станции и момента времени.
ML-разработчик отдельно добавляет историю пассажиропотока, лаги и baseline:

```text
ExternalFeatures + passenger flow + lags + baseline
                          ↓
                  CatBoost / LightGBM
```

Это целевой следующий этап, а не существующий обучающий pipeline.
`external_data` не обучает модель и не возвращает ML prediction.
Преобразование JSON в ML-таблицу, кодирование категорий и обработка пропусков
определяются ML-слоем; `null` нельзя автоматически трактовать как нулевой спрос.

### Что пока НЕ делает external_data

- Не считает passenger flow и lag features.
- Не обучает и не запускает CatBoost/LightGBM, не рассчитывает baseline.
- Не рассчитывает количество составов метро.
- Не делает dispatcher recommendation.
- Не запускает scheduler или непрерывный сбор данных.
- Не является FastAPI-сервисом и не хранит данные в БД.
- Не скачивает афиши, не выполняет web scraping и не подключает Telegram API.
- Не определяет радиус/силу влияния события и не строит маршруты до метро.

### Типовые ошибки

| Симптом | Что проверить |
| --- | --- |
| Environment variable missing / MISSING | Нужная переменная заполнена и экспортирована. Файл `.env` сам по себе не загружается отдельными CLI. |
| `.env` заполнен, но CLI не видит ключ | Выполните `set -a; source .env; set +a` в том же shell, где запускаете CLI, либо используйте общий smoke-скрипт. |
| HTTP 401/403 | Ключ относится к нужному сервису, есть доступ и квота. Для YandexGPT проверьте folder ID, права и выбранную модель. |
| Новый ключ ещё не работает | Проверьте статус активации в кабинете соответствующего API; активация может быть не мгновенной. Затем повторите запрос. |
| HTTP 429 / timeout / ошибка сети | Проверьте квоту, соединение и доступность сервиса; повторите позже. Не печатайте URL с apikey для диагностики. |
| Weather GraphQL access error | Проверьте доступ к полю по тарифу; попробуйте без `--with-feels-like`. |
| Место не найдено с достаточной уверенностью | Передайте явно известный адрес через `--address`, проверьте кандидатов с `--debug`; не подставляйте первые попавшиеся координаты. |
| Календарная дата неизвестна | В локальной таблице должна быть запрошенная дата. Demo-файл покрывает только 2026-10-10. |
| Railway не совпадает со station/timestamp | Передавайте признаки именно нужной станции и того же момента времени. |

### Security

- `.env` и файлы `artifacts/` уже исключены через [.gitignore](.gitignore).
  В [.env.example](.env.example) должны оставаться только пустые ключи и примеры настроек.
- Не коммитьте секреты, локальное виртуальное окружение и реальные ключи.
- Не логируйте API keys, Authorization и полные URL с `apikey`.
- Используйте штатный `--debug` и smoke-отчёты с маскированием секретов.
  Перед передачей отчёта проверьте, нет ли в нём чувствительного текста источника.
- Не публикуйте `.env` и не вставляйте его содержимое в задачи или логи.
