"""Create a reproducible public tree without workspace credentials or old campaigns."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", ".DS_Store"}
RUNTIME = ["data/processed/metro51_inputs_20261009",
           "data/processed/metro48_daily_fifo_20261009_v7/runs/DAILY_FIFO_S1_7"]
FIXTURES = ["data/processed/metro", "data/processed/economics", "data/processed/economics_audit",
            "data/processed/line1", "data/processed/simulation", "data/processed/metro48_gru_inputs_20261009",
            "data/processed/metro_ml_inputs/lgbm_final", "data/processed/metro_ml_inputs/data/reference",
            "data/processed/metro_daily_experiments_20261009/baseline.json.gz",
            "data/processed/metro_daily_experiments_20261009/forecasts/ML_09.json",
            "data/processed/metro_daily_experiments_20261009/forecasts/ML_10.json",
            "data/processed/metro44_catalog_pilot_20261009/observed_frame.parquet",
            "data/processed/metro44_catalog_cont6h_20261009/campaign.py",
            "data/processed/metro44_catalog_cp30_incumbent_20261009/directional_profile.json",
            "data/processed/metro44_wait_calendar_physical_20261009/wait_calendar_reference_plan.json.gz",
            "data/processed/metro44_continuation_6h_closed_20261009/runs/CONT6H_CLOSED_KEEP_S1_7/movement_input.json",
            "data/processed/metro44_bridge_off160_actual_6h_20261009_v2/runs/OFF160_BRIDGE_S1_7/movement_input.json",
            "data/processed/metro44_bridge_off160_actual_6h_20261009_v2/runs/OFF160_BRIDGE_S1_7/motion.json",
            "data/processed/metro47_withdrawal_normalized_actual_20261009/runs/NORMALIZED_REFERENCE_S1_7/movement_input.json"]
TOKENS = [
    ("yandex_api_key", re.compile(rb"AQV[Nn][A-Za-z0-9_-]{30,}")),
    ("github_token", re.compile(rb"(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("huggingface_token", re.compile(rb"hf_[A-Za-z0-9]{30,}")),
    ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]
ENV_TEMPLATE = "YANDEX_API_KEY=YOUR_YANDEX_API_KEY\nYANDEX_FOLDER_ID=YOUR_YANDEX_FOLDER_ID\nYANDEX_MODEL=yandexgpt-5-lite\n"


def long_path(path):
    path = Path(path).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" and not str(path).startswith("\\\\?\\") else path


def allowed(path):
    return not any(p in IGNORED or p.startswith(".env") for p in path.parts) and path.name != "secrets.toml"


def manifest(root=ROOT):
    root = Path(root).resolve()
    selected = {}

    def add(path, scope):
        path = Path(path)
        candidates = path.rglob("*") if path.is_dir() else [path]
        for p in candidates:
            if not p.is_file() or not allowed(p.relative_to(root)):
                continue
            if p.is_symlink() or not p.resolve().is_relative_to(root):
                raise ValueError("source symlink outside export contract")
            relative = p.relative_to(root).as_posix()
            if relative.startswith("interfaces/streamlit_dispatcher/artifacts/"):
                continue
            selected[relative] = dict(path=relative, scope=scope, bytes=p.stat().st_size,
                                     sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    for p in root.iterdir():
        if p.is_file() and p.suffix in {".py", ".md", ".txt", ".toml"}:
            add(p, "source")
    for name in ("tests", "scripts", "docs", "config", "examples", "metro_data", "interfaces", "vendor"):
        add(root / name, "source")
    for name in RUNTIME:
        add(root / name, "runtime")
    for name in FIXTURES:
        add(root / name, "legacy_fixture")
    for p in (root / "data/processed/metro_ml_inputs/data/interim").glob("*.parquet"):
        add(p, "legacy_fixture")
    for p in (root / "data/processed/metro_ml_inputs/data/raw").glob("*.*"):
        add(p, "legacy_fixture")
    # The existing UI suite uses one completed pilot and one interrupted pilot.
    pilot = root / "data/processed/metro51_balanced_pilot_20261009_v2"
    for arm in ("reference", "ensemble_old", "balanced"):
        for name in ("common_scenario.json.gz", "result.json", "controller", "explanations"):
            add(pilot / arm / name, "ui_fixture")
        result = pilot / arm / "result.json"
        if result.exists():
            run = json.loads(result.read_text(encoding="utf8"))["run"]["folder"]
            for name in ("movement_input.json", "motion.json", "passenger_checkpoint.json.gz"):
                add(pilot / arm / run / name, "ui_fixture")
    full_reference = root / "data/processed/metro50_full_day_20261009_v4/reference"
    for name in ("common_scenario.json.gz", "result.json"):
        add(full_reference / name, "ui_fixture")
    if (full_reference / "result.json").exists():
        run = json.loads((full_reference / "result.json").read_text(encoding="utf8"))["run"]["folder"]
        for name in ("movement_input.json", "motion.json", "passenger_checkpoint.json.gz"):
            add(full_reference / run / name, "ui_fixture")
    add(root / "data/processed/metro52_readiness_20261009", "verification")
    return sorted(selected.values(), key=lambda row: row["path"])


def workspace_credentials(root):
    found = set()
    for name in (".env", ".env.example"):
        path = Path(root) / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, _, value = line.partition("=")
            if re.search(r"(?:API_KEY|TOKEN|PASSWORD|SECRET)$", key.strip(), re.I) and len(value.strip()) >= 12:
                found.add(value.strip().strip("\"'"))
    return found


def secret_findings(root, rows, secrets=()):
    findings = []
    for row in rows:
        content = (Path(root) / row["path"]).read_bytes()
        hits = [("workspace_credential", content.find(value.encode())) for value in secrets]
        hits += [(kind, match.start()) for kind, pattern in TOKENS for match in pattern.finditer(content)]
        for kind, offset in hits:
            if offset >= 0:
                finding = dict(path=row["path"], line=content[:offset].count(b"\n") + 1, kind=kind)
                if finding not in findings:
                    findings.append(finding)
    return findings


def requirements():
    packages = ["numpy", "pandas", "pyarrow", "openpyxl", "pillow", "pypdf", "pytest", "lightgbm", "torch",
                "scipy", "matplotlib", "ortools", "openai", "pydantic", "PyYAML", "scikit-learn", "requests",
                "python-dotenv", "httpx", "python-dateutil", "tzdata"]
    rows = ["# Runtime versions used by the readiness environment; Python 3.12.",
            "# CPU Torch wheels are available from the official PyTorch index.",
            "--extra-index-url https://download.pytorch.org/whl/cpu"]
    for name in packages:
        try:
            rows.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return "\n".join(rows) + "\n"


def vendor_requirements():
    rows = ["-r requirements-runtime.txt", "# Optional dependencies for the separately tested upstream suite."]
    for name in ("beautifulsoup4", "statsmodels", "catboost", "xgboost", "scikit-learn", "optuna", "tqdm"):
        try:
            rows.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return "\n".join(rows) + "\n"


def readme(root=ROOT):
    status_path = Path(root) / "data/processed/metro52_full_day_20261009/status.json"
    stage = json.loads(status_path.read_text(encoding="utf8")).get("stage") if status_path.exists() else "not_started"
    status = ("Новый суточный прогон выполняется; итоговых результатов ещё нет." if stage in {"running", "reporting"}
              else "Новый суточный прогон будет запущен после полной проверки. Итоговых результатов ещё нет.")
    return """# Оптимизация движения красной линии метро Петербурга

Симулятор пассажиров и составов, прогноз входов через турникеты раз в 15 минут,
полный ансамбль LightGBM + persistence + GRU + замороженный стекинг,
совместный оптимизатор интервалов и составов, проверка ограничений и интерфейс Streamlit.
YandexGPT объясняет проверенные решения; при сбое API используется сохранённое шаблонное объяснение.

__CAMPAIGN_STATUS__
Исторические записи в комплекте служат тестовыми данными и примером интерфейса.
Они не подтверждают эффект новой версии. Денежная экономия в реальном метро не подтверждена.

## Установка

Нужен Python 3.12. Из корня проекта в PowerShell:

```powershell
python -m venv .venv
& .\\.venv\\Scripts\\python.exe -m pip install -r requirements-runtime.txt
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
& .\\.venv\\Scripts\\python.exe -m pytest tests scripts/test_goal43_ui_probe.py -q --junitxml=core_tests.xml
& .\\.venv\\Scripts\\python.exe -m pip install -r requirements-vendor-tests.txt
$projectRoot = (Get-Location).Path
$env:PYTHONPATH = "$projectRoot/data/processed/metro52_readiness_20261009/tests/vendor;$projectRoot/vendor/metro_petersburg"
Push-Location "$projectRoot/vendor/metro_petersburg"
& "$projectRoot/.venv/Scripts/python.exe" -m pytest -c pytest.ini -p readiness_vendor_inputs tests external_data -q --junitxml="$projectRoot/vendor_tests.xml"
Pop-Location
& .\\.venv\\Scripts\\python.exe scripts/metro52_campaign.py --help
```

Проверка vendor использует специальный bootstrap из сохранённой папки проверки.
Он направляет чтение входов в опубликованный `metro51_inputs` и проверяет сохранённые контракты.
Этот набор проверяется отдельно от root tests; зависимости vendor requirements не подменяют проверенное окружение.

Новый прогон запускайте в новой папке через `scripts/metro52_campaign.py`, передав успешные
JUnit-файлы параметром `--junit` и путь `--output`, например:

```powershell
& .\\.venv\\Scripts\\python.exe scripts/metro52_campaign.py --output data/processed/my_full_day --junit core_tests.xml vendor_tests.xml data/processed/metro52_readiness_20261009/tests/ui/tests_final.xml
```

Скрипт запускает четыре варианта,
объяснения YandexGPT, оценку прогноза, остановки составов и итоговые отчёты.
Период охватывает 06:00–02:00 с завершением пассажирских обязательств после полуночи.
Прогноз обновляется каждые 15 минут. Состояние сохраняется в `controller/resume_*.json.gz`.
Сравнение с исходным графиком относится к одному сценарию, а годовые суммы служат сценарной оценкой.

## Streamlit

```powershell
python -m venv interfaces/streamlit_dispatcher/.venv
& .\\interfaces\\streamlit_dispatcher\\.venv\\Scripts\\python.exe -m pip install -r interfaces/streamlit_dispatcher/requirements-dev.txt
& .\\interfaces\\streamlit_dispatcher\\start.ps1
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
- Контрольные суммы опубликованных файлов: `EXPORT_MANIFEST.json`.
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
""".replace("__CAMPAIGN_STATUS__", status)


def fixture_audit(root, rows):
    import ast
    selected = {r["path"] for r in rows}
    refs = []
    for path in (Path(root) / "tests").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value.replace("\\", "/")
            if value.startswith("data/processed/") and (Path(root) / value).is_file():
                refs.append(dict(test=path.name, line=node.lineno, path=value, included=value in selected))
    return dict(existing_literal_file_references=refs,
                literal_files_complete=all(r["included"] for r in refs),
                explicit_runtime_paths=RUNTIME, explicit_legacy_paths=FIXTURES,
                scope="Literal existing file paths plus explicit symbolic model/SOURCE/FRAME dependencies; raw archive regeneration excluded",
                all_legacy_execution_verified_in_export=False)


def build_export(root, destination, *, max_file_bytes=100_000_000, plan_only=False):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root.is_relative_to(destination):
        raise ValueError("export destination overlaps source")
    rows = manifest(root)
    findings = secret_findings(root, rows, workspace_credentials(root))
    if findings:
        raise ValueError("credential gate failed: " + json.dumps(findings))
    oversize = [r["path"] for r in rows if r["bytes"] >= max_file_bytes]
    if oversize:
        raise ValueError("GitHub file limit exceeded: " + json.dumps(oversize))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("export destination must be empty")
    destination = long_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    out = dict(files=rows, total_bytes=sum(r["bytes"] for r in rows), max_file_bytes=max((r["bytes"] for r in rows), default=0),
               secret_findings=findings, source_files_modified=False, plan_only=plan_only,
               transformations=[],
               omitted=["workspace .env and .env.example", "Python environments and caches", "Git metadata",
                        "original data.zip (>100 MB), training caches outside ensemble", "other historical campaign outputs"])
    if not plan_only:
        for row in rows:
            original, target = root / row["path"], destination / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("source changed while packaging: " + row["path"])
            if row["path"] == "data/processed/metro44_catalog_cont6h_20261009/campaign.py":
                content = target.read_bytes()
                portable = content.replace(b"ROOT = Path('C:/Users/lesha/HackathonSPB')",
                                           b"ROOT = Path(__file__).resolve().parents[3]")
                portable = portable.replace(b"SNAPSHOT = ROOT / 'data/processed/metro44_catalog_execution_v6'", b"SNAPSHOT = ROOT")
                if portable != content:
                    target.write_bytes(portable)
                    out["transformations"].append(dict(path=row["path"], reason="historical fixture uses exported runtime",
                                                       source_sha256=row["sha256"], sha256=hashlib.sha256(portable).hexdigest()))
        if (destination / "README.md").exists():
            (destination / "docs").mkdir(exist_ok=True)
            shutil.copyfile(destination / "README.md", destination / "docs/legacy_project_README.md")
        generated = {"README.md": readme(root), "requirements-runtime.txt": requirements(),
                     "requirements-vendor-tests.txt": vendor_requirements(), ".env.example": ENV_TEMPLATE,
                     ".gitattributes": "* -text\n", ".gitignore": ".env\n.env.*\n!.env.example\n.venv/\n__pycache__/\n.pytest_cache/\n*.pyc\ninterfaces/streamlit_dispatcher/artifacts/\n"}
        for name, content in generated.items():
            (destination / name).write_text(content, encoding="utf8", newline="\n")
        exported = [dict(path=p.relative_to(destination).as_posix(), bytes=p.stat().st_size,
                         sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in destination.rglob("*") if p.is_file()]
        out["source_files"] = rows
        out["files"] = sorted(exported, key=lambda row: row["path"])
        out["total_bytes"] = sum(r["bytes"] for r in exported)
    (destination / "TEST_FIXTURE_AUDIT.json").write_text(json.dumps(fixture_audit(root, rows), ensure_ascii=False, indent=2), encoding="utf8")
    (destination / "EXPORT_MANIFEST.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    result = build_export(args.source, args.destination, plan_only=args.plan)
    print(json.dumps({k: result[k] for k in ("total_bytes", "max_file_bytes", "secret_findings", "plan_only")}))
