"""Public export never inherits workspace credentials or oversized files."""

import json
from pathlib import Path

import pytest

from scripts.metro52_package import build_export, manifest, secret_findings


def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "simulation.py").write_text("VALUE = 1\n", encoding="utf8")
    (source / ".env").write_text("YANDEX_API_KEY=private-test-value-12345\n", encoding="utf8")
    (source / ".env.example").write_bytes((source / ".env").read_bytes())
    (source / "vendor/metro_petersburg/.git").mkdir(parents=True)
    (source / "vendor/metro_petersburg/.git/config").write_text("private metadata")
    (source / "vendor/metro_petersburg/src").mkdir()
    (source / "vendor/metro_petersburg/src/model.py").write_text("MODEL = 1\n")
    (source / "data/processed/unrelated_old_run").mkdir(parents=True)
    (source / "data/processed/unrelated_old_run/huge.json").write_text("old output")
    return source


def test_manifest_excludes_secrets_caches_and_old_runs(tmp_path):
    source = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    rows = manifest(source)
    names = {r["path"] for r in rows}
    assert "simulation.py" in names
    assert "vendor/metro_petersburg/src/model.py" in names
    assert not any(".env" in p or ".git" in p or "unrelated_old_run" in p for p in names)
    assert all(Path(p).read_bytes() == value for p, value in before.items())


def test_export_generates_placeholder_and_sha_manifest(tmp_path):
    source = fixture(tmp_path)
    export = tmp_path / "export"
    result = build_export(source, export)
    assert result["secret_findings"] == []
    assert not (export / ".env").exists()
    env = (export / ".env.example").read_text()
    assert "private-test-value-12345" not in env
    assert "YOUR_YANDEX_API_KEY" in env
    assert (export / "simulation.py").read_bytes() == (source / "simulation.py").read_bytes()
    saved = json.loads((export / "EXPORT_MANIFEST.json").read_text())
    assert any(r["path"] == "simulation.py" and r["sha256"] for r in saved["files"])


def test_secret_gate_reports_path_and_kind_without_secret(tmp_path):
    source = fixture(tmp_path)
    secret = "private-test-value-12345"
    (source / "leak.py").write_text(f'VALUE = "{secret}"')
    rows = manifest(source)
    findings = secret_findings(source, rows, [secret])
    assert findings == [{"path": "leak.py", "line": 1, "kind": "workspace_credential"}]
    with pytest.raises(ValueError, match="credential") as exc:
        build_export(source, tmp_path / "export")
    assert secret not in str(exc.value)
    assert not (tmp_path / "export/leak.py").exists()


def test_prefixed_tokens_and_file_limit_block_export(tmp_path):
    source = fixture(tmp_path)
    (source / "leak.py").write_text('TOKEN = "' + "ghp_" + "a" * 36 + '"')
    assert secret_findings(source, manifest(source))[0]["kind"] == "github_token"
    with pytest.raises(ValueError, match="credential"):
        build_export(source, tmp_path / "export")
    (source / "leak.py").unlink()
    with pytest.raises(ValueError, match="file limit"):
        build_export(source, tmp_path / "export", max_file_bytes=4)


def test_export_cannot_overwrite_source(tmp_path):
    source = fixture(tmp_path)
    with pytest.raises(ValueError, match="source"):
        build_export(source, source)


def test_historical_fixture_imports_only_exported_runtime(tmp_path):
    source = fixture(tmp_path)
    path = source / "data/processed/metro44_catalog_cont6h_20261009/campaign.py"
    path.parent.mkdir(parents=True)
    original = "from pathlib import Path\nROOT = Path('C:/Users/lesha/HackathonSPB')\nSNAPSHOT = ROOT / 'data/processed/metro44_catalog_execution_v6'\n"
    path.write_text(original)
    export = tmp_path / "export"
    result = build_export(source, export)
    saved = (export / path.relative_to(source)).read_text()
    assert "C:/Users/lesha/HackathonSPB" not in saved
    assert "SNAPSHOT = ROOT\n" in saved
    assert path.read_text() == original
    assert any(row["path"] == path.relative_to(source).as_posix() for row in result["transformations"])
