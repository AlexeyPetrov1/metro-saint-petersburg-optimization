# Yandex Weather adapter

Независимый модуль текущей погоды и почасового прогноза для любых координат.
Python 3.11+, Pydantic v2, httpx. Не зависит от `external_data/events`.

Из корня репозитория:

```sh
python3 -m pip install -r external_data/weather/requirements.txt
python3 -m unittest discover -s external_data/weather/tests -v
python3 -m unittest discover -v
```

## Ключ и CLI

Получите отдельный ключ в [кабинете API Яндекс Погоды](https://yandex.ru/dev/weather/).
Порядок подключения и заголовок `X-Yandex-Weather-Key` описаны в
[официальном руководстве](https://yandex.ru/dev/weather/doc/ru/concepts/how-to).
Weather API key и ключ YandexGPT — разные ключи. Этот модуль читает только
`YANDEX_WEATHER_API_KEY`; переменная `YANDEX_API_KEY` не используется.

```sh
export YANDEX_WEATHER_API_KEY="ваш-отдельный-weather-key"
python3 -m external_data.weather.cli --lat 59.9343 --lon 30.3351
```

`--lat` и `--lon` обязательны. По умолчанию выводится текущая погода.
Для почасового прогноза добавьте `--hours 2` (допустимо 1–48).
Для запроса ощущаемой температуры добавьте `--with-feels-like`:
`feelsLike` требует тарифа выше Test; на «Тестовом» явный запрос этого поля
может вернуть GraphQL access error. Без этой опции `feels_like` остаётся
null, а поле не включается в запрос. Доступность полей проверяйте по
[спецификации и тарифам](https://yandex.ru/dev/weather/doc/ru/concepts/spectaql).

`.env.example` содержит пустое имя переменной; `.env` игнорируется Git и
автоматически не загружается. CLI выводит JSON в stdout, ошибки — в stderr.
Успех: exit code 0; ошибка запроса/данных/конфигурации: 1; аргументов argparse: 2.
Ключи, HTTP-заголовки и полные тела ошибок не выводятся. Сообщения `errors[].message`
GraphQL показываются в ошибке, включая ответы HTTP 400, после маскирования
ключа и секретов из окружения. Реальный вызов расходует квоту
выбранного тарифа. Unit tests используют `httpx.MockTransport`, без сети и ключей.

Для диагностики добавьте `--debug`:

```sh
python3 -m external_data.weather.cli --lat 59.9343 --lon 30.3351 --debug
```

В stderr выводятся точный GraphQL query, variables и сообщения GraphQL errors,
в stdout остаётся только JSON результата. Переводы строк внутри query
экранируются как `\n`. HTTP-заголовки и поле `extensions` не логируются.
Без `--debug` ошибки API также содержат конкретную причину, например
`Cannot query field ...` или `Access denied for field ...`.

Default-запрос ориентирован на «Тестовый» тариф:

```graphql
query CurrentWeather($point: PointInput!) {
  serverTime
  weatherByPoint(request: $point) {
    now {
      temperature(unit: CELSIUS)
      precType
      windSpeed(unit: METERS_PER_SECOND)
      windGust(unit: METERS_PER_SECOND)
      humidity
      pressure(unit: MM_HG)
      condition
    }
  }
}
```

Внутри `now` больше не запрашивается `time`: для него в спецификации нет
перечня тарифов. Вместо него используется корневое `serverTime`, для которого
явно указан «Тестовый». Это время ответа сервера, не момент измерения станции.
Для прогноза используются те же погодные поля плюс `ForecastHour.time` и
`accumulatedPrec` — они доступны на Test plan. `precProbability`/`precStrength`
не запрашиваются, поскольку текущая модель их не использует. Поля и тарифы:
[GraphQL-спецификация](https://yandex.ru/dev/weather/doc/ru/concepts/spectaql),
[параметры API](https://yandex.ru/dev/weather/doc/ru/concepts/parameters).

## Python API

```python
from contextlib import closing
from external_data.weather import YandexWeatherProvider

with closing(YandexWeatherProvider(timeout=15, include_feels_like=False)) as provider:
    now = provider.get_current_weather(59.9343, 30.3351)
    forecast = provider.get_hourly_forecast(59.9343, 30.3351, hours=2)
    print(now.model_dump_json(indent=2))
```

Оба метода используют POST на
`https://api.weather.yandex.ru/graphql/query`. GraphQL variables содержат
координаты; запрос выбирает только необходимые поля. Для прогноза запрашивается
ограниченное число дней с часами (`ceil(hours / 24) + 1`), чтобы покрыть переход
через полночь. У списка `ForecastDay.hours` нет аргумента ограничения по часам.
API и вложенность прогноза описаны в
[официальной документации](https://yandex.ru/dev/weather/doc/ru/concepts/forecast).

## Время и значения

| Поле | Смысл |
| --- | --- |
| timestamp | Корневое `serverTime` для текущей погоды; `ForecastHour.time` для прогноза; сохраняется UTC offset API |
| latitude, longitude | Координаты запроса в градусах |
| temperature, feels_like | °C |
| precipitation | `ForecastHour.accumulatedPrec`, суммарные осадки за прогнозный час, мм |
| precipitation_type | Исходное значение `precType`, например `RAIN` или `NO_TYPE` |
| wind_speed, wind_gust | м/с |
| humidity | %, 0–100 |
| pressure | мм рт. ст. (`MM_HG`) |
| condition | Исходный код API, например `RAIN`; без собственного перекодирования |

У текущей погоды `Now` нет количества осадков в мм: `precipitation=null`.
Тип осадков не преобразуется в количество и отсутствие осадков не подменяется
придуманным нулём. `Now` — текущая оценка сервиса, а не обязательно измерение
конкретной метеостанции. Определения полей приведены в
[GraphQL-спецификации](https://yandex.ru/dev/weather/doc/ru/concepts/spectaql).

`hours` означает горизонт от `serverTime` API. Возвращаются отсортированные
точки с timestamp в **[serverTime, serverTime + hours]**, включая границы.
Например, в 12:30 при `hours=2` это доступные точки 13:00 и 14:00; в 12:00
могут включаться 12:00, 13:00 и 14:00. Timestamp не округляется и не заменяется
часами компьютера. Пустой список означает отсутствие точек в этом интервале.
Дыры не заполняются, в 15-минутные интервалы данные не интерполируются.
Произвольный исторический момент и архив этим MVP не запрашиваются.
Необязательный `past_hours` (0–24, по умолчанию 0) расширяет окно назад:
**[serverTime − past_hours, serverTime + hours]**. Так берутся уже начавшиеся
часы сегодняшнего прогноза: их использует ML-прогноз (`src/external_adapter.py`).
Часы прошлых суток не запрашиваются.

Отсутствующие измерения и явные null сохраняются как None. Отсутствующий
timestamp, naive datetime, неверные координаты, отрицательные осадки/ветер,
влажность вне диапазона, NaN/Infinity и некорректные структуры — ошибки.
HTTP 401/403/429/5xx, timeout и GraphQL `errors` вызывают `YandexWeatherError`.
Частичный ответ с GraphQL errors не выдаётся как успешный набор features.
Автоматических повторов, fallback на другой API и переходов по redirects нет.
