# Оптимизация движения красной линии метро Петербурга

Симулятор пассажиров и составов, прогноз входов через турникеты раз в 15 минут,
полный ансамбль LightGBM + persistence + GRU + замороженный стекинг,
совместный оптимизатор интервалов и составов, проверка ограничений и интерфейс Streamlit.
YandexGPT объясняет проверенные решения; при сбое API используется сохранённое шаблонное объяснение.

Суточный прогон завершён. [PDF](data/processed/metro52_full_day_20261009/review/report.pdf), [HTML](data/processed/metro52_full_day_20261009/review/report.html), [отчёты и данные](data/processed/metro52_full_day_20261009). Результаты проверок безопасности и экономики приведены в отчётах. По просьбе пользователя
остаются только текущий ансамбль с новым оптимизатором и уже завершённый исходный график.
Варианты OLD и ENSEMBLE_OLD исключены до запуска. Исходный график не использует медиану:
это модельный график FIFO с календарными ограничениями, его совпадение с графиком оператора не подтверждено.
Исторические записи в комплекте служат тестовыми данными и примером интерфейса.
Они не подтверждают эффект новой версии. Денежная экономия в реальном метро не подтверждена.

Полная проверка первой опубликованной копии: **2080 тестов прошли**; отдельно проверены
upstream ML-код, Streamlit и загрузка весов. [Доказательства](verification_after_export/VERIFICATION_AFTER_EXPORT.md).
Последующие изменения обвязки кампании прошли 9 тестов, страницы прогресса 4 теста,
подготовки отчётов 34 теста. [Дополнения и SHA256](verification_after_scope/CHANGESET_SHA256.json).
Числа повторных и пересекающихся проверок не складываются.

Во время локального прогона файл `progress.html` в папке кампании обновляется каждые 10 секунд.
Он показывает сохранённые решения и отдельно отмечает подготовку отчётов.
Ограничение в 60 решений отменено пользователем: текущий расчёт продолжает работать до конца
без перезапуска. Полный период включает 71 решение управления и завершение поездок до 02:00.
Сравнение исходного графика и BALANCED, отчёты и публикация ожидают завершения обоих вариантов.
[Проверка продолжения и состояние](verification_full_day_resume_20261010/RESUME.md).
Проверки поддержки сокращённого опыта сохранены как история: 95 тестов обвязки и отчётов,
21 тест интерфейса прошли. [Доказательства](verification_after_stop60/VERIFICATION.md).
Данные прошлого эксперимента с медианой разобраны в [исторической записке](verification_after_scope/historical_median_review/provenance_review.md);
они не заменяют новое парное сравнение при одинаковых ограничениях.

## Установка

Нужен Python 3.12. Из корня проекта в PowerShell:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
Copy-Item .env.example .env
```

Заполните собственные параметры Yandex в локальном `.env`. Без API доступна шаблонная замена.
`requirements-runtime.txt` сохраняет версии после полной проверки, включая CPU Torch.
Файл указывает официальный дополнительный индекс PyTorch; версию `+cpu` сохраняйте.
Старые `requirements*.txt` оставлены как исходные документы, их установка может изменить проверенные версии.
Модель уже обучена; веса и подготовленные данные находятся в `data/processed/metro51_inputs_20261009`.
Начальное состояние симулятора находится в `data/processed/metro48_daily_fifo_20261009_v7/runs/DAILY_FIFO_S1_7`.
Сборка не требует исходного `data.zip` или повторного обучения.
Исходный ансамбль обучен для шагов по 30 минут. Адаптер обновляет выпуск каждые 15 минут
и разделяет его по четвертям часа. Срезы :15/:45 служат экспериментальной адаптацией;
переобучение ансамбля на шаг 15 минут не проводилось.

## Проверка и полный прогон

```powershell
& .\.venv\Scripts\python.exe -m pytest tests scripts/test_goal43_ui_probe.py -q --junitxml=core_tests.xml
& .\.venv\Scripts\python.exe -m pip install -r requirements-vendor-tests.txt
$projectRoot = (Get-Location).Path
$env:PYTHONPATH = "$projectRoot/data/processed/metro52_readiness_20261009/tests/vendor;$projectRoot/vendor/metro_petersburg"
Push-Location "$projectRoot/vendor/metro_petersburg"
& "$projectRoot/.venv/Scripts/python.exe" -m pytest -c pytest.ini -p readiness_vendor_inputs tests external_data -q --junitxml="$projectRoot/vendor_tests.xml"
Pop-Location
& .\.venv\Scripts\python.exe scripts/metro52_campaign.py --help
```

Проверка vendor использует специальный bootstrap из сохранённой папки проверки.
Он направляет чтение входов в опубликованный `metro51_inputs` и проверяет сохранённые контракты.
Этот набор проверяется отдельно от root tests; зависимости vendor requirements не подменяют проверенное окружение.

Новый прогон запускайте в новой папке через `scripts/metro52_campaign.py`, передав успешные
JUnit-файлы параметром `--junit` и путь `--output`, например:

```powershell
& .\.venv\Scripts\python.exe scripts/metro52_campaign.py --output data/processed/my_full_day --junit core_tests.xml vendor_tests.xml data/processed/metro52_readiness_20261009/tests/ui/tests_final.xml
```

Скрипт запускает четыре варианта,
объяснения YandexGPT, оценку прогноза, остановки составов и итоговые отчёты.
Период охватывает 06:00–02:00 с завершением пассажирских обязательств после полуночи.
Прогноз обновляется каждые 15 минут. Состояние сохраняется в `controller/resume_*.json.gz`.
Сравнение с исходным графиком относится к одному сценарию, а годовые суммы служат сценарной оценкой.

## Streamlit

```powershell
python -m venv interfaces/streamlit_dispatcher/.venv
& .\interfaces\streamlit_dispatcher\.venv\Scripts\python.exe -m pip install -r interfaces/streamlit_dispatcher/requirements-dev.txt
& .\interfaces\streamlit_dispatcher\start.ps1
```

Откройте http://localhost:8501. Введите путь новой кампании в поле «Папка эксперимента».
«Обновить журналы» показывает новый checkpoint без повторного моделирования.
До первой записи показывается ожидание, а завершённая запись доступна на весь сохранённый день.
Тесты интерфейса: `python -m pytest interfaces/streamlit_dispatcher/tests -q` в его отдельном окружении;
для браузерных проверок сервер должен работать. Используется Edge или установленный Playwright Chromium.

## Источники и состав публикации

- Обучение и подготовка признаков: [Deniskish/metro-petersburg](https://github.com/Deniskish/metro-petersburg),
  исходная версия `2f0fd0a3c11e5a9322bac1f65e68f5d891f816c9`, код в `vendor/metro_petersburg`.
- Оформление и проигрыватель: [KovalevNikolay710/sirius-subway_simulation](https://github.com/KovalevNikolay710/sirius-subway_simulation),
  версия `2bbcdb97dda3a2ba2f12490919bb3fdf57ce58ea`; исходные файлы и SHA находятся в интерфейсе.
- Исходные документы проекта: `docs/`, `config/`, `examples/`, прежний README сохранён как `docs/legacy_project_README.md`.
- Контрольные суммы первой публикации: `EXPORT_MANIFEST.json`; последующих изменений: `verification_after_scope/CHANGESET_SHA256.json` и `verification_after_stop60/CHANGESET_SHA256.json`.
- Проверки: `data/processed/metro52_readiness_20261009`; старые отчёты об ошибках сохранены как история проверки.
- Полнота локальных тестовых данных описана в `TEST_FIXTURE_AUDIT.json`.

`.env`, рабочий `.env.example`, окружения Python, Git metadata, кэши и остальные старые прогоны исключены.
Для файлов Git отключено преобразование окончаний строк: `.gitattributes` содержит `* -text`.
Используйте `git config core.autocrlf false`, чтобы сохранить байты исходников, связанные с контрольными суммами моделей.
На Windows используйте `git config core.longpaths true` и короткий путь клонирования, например `C:/metro`.
Два старых сырых документа организаторов имеют имена длиннее 255 байт UTF-8 и требуют Windows;
работа модели и симулятора использует подготовленные таблицы и не читает эти два документа.
Исторический `campaign.py` в тестовых данных привязан к корню экспортированной копии,
чтобы тесты не импортировали модули из старого рабочего каталога. Изменение отражено в манифесте.
Лицензии сторонних источников не заменены собственной лицензией проекта.

## Completed-day research (2026-10-10)

[Eight-page research PDF](data/processed/metro52_research_20261010/report.pdf) | [Analysis and tables](data/processed/metro52_research_20261010/report.md) | [Research archive](data/processed/metro52_research_20261010/research.zip).

The matched completed-day simulation saved 688.1 train-km. Central maintenance-resource savings are RUB 77,889 without SR/KR and RUB 113,416 with SR/KR (2.52% / 2.56%). This is not demonstrated cash savings or operational approval: both arms fail five local boarding cohorts; P99 waiting rises from 148 to 211 seconds; the final decision fell back to KEEP after an evaluation error; incident validation remains incomplete. The report includes uncertainty, annual scenario limits, per-train CSV, passenger distributions, forecast calibration, and next experiments.
