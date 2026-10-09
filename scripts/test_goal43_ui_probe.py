import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from goal43_demo import write_demo
from goal43_ui_probe import prepare_probe, read_receipt, probe, _compact_dom


def demo(tmp_path):
    write_demo(tmp_path, dict(schema_version=1, scenarios=[dict(scenario_id="fixture_offpeak",
        comparisons=[dict(strategy="ROLLING_SELECTED", per_train=[], series={})])]))
    return tmp_path


def test_probe_uses_a_copy_and_installs_error_capture_before_the_app(tmp_path):
    output = demo(tmp_path)
    original = (output / "demo.html").read_bytes()
    path = prepare_probe(output)
    source = path.read_text(encoding="utf-8")
    assert (output / "demo.html").read_bytes() == original
    assert source.index('id="goal43-ui-errors"') < source.index('id="demo-data"')
    assert "dispatchEvent(new Event('change'" in source
    assert "textContent=''" in source and "goal43-ui-receipt" in source
    assert "fixture_offpeak" in source and "ROLLING_SELECTED" in source


def test_receipt_parser_recovers_escaped_error_text_and_compacts_only_payload():
    dom = '<script id="demo-data" type="application/json">{"big":42}</script><pre id="goal43-ui-receipt">{"status":"failed","errors":["x &lt; y &amp; z"]}</pre>'
    assert read_receipt(dom)["errors"] == ["x < y & z"]
    compact = _compact_dom(dom)
    assert '"big"' not in compact
    assert '"errors"' in compact
    with pytest.raises(ValueError, match="receipt"):
        read_receipt("<html></html>")


def test_probe_persists_browser_receipt_and_fixed_size_screenshot(tmp_path, monkeypatch):
    output = demo(tmp_path)
    edge = tmp_path / "edge.exe"
    edge.write_bytes(b"fixture")
    receipt = dict(status="passed", scenario_count=1, view_count=1, errors=[],
                   views=[dict(passed=True)], demo_data_removed=True)
    def run(command, **kwargs):
        assert "--window-size=1440,1600" in command
        shot = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--screenshot=")))
        shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 1440, 1600))
        dom = '<script id="demo-data" type="application/json">{}</script><pre id="goal43-ui-receipt">' + json.dumps(receipt) + '</pre>'
        return SimpleNamespace(returncode=0, stdout=dom.encode(), stderr=b"")
    monkeypatch.setattr("goal43_ui_probe.subprocess.run", run)
    result = probe(output, edge_path=edge)
    assert result["status"] == "passed" and result["source_html_unchanged"]
    assert result["screenshot_dimensions"] == [1440, 1600]
    assert (output / "verification/ui_receipt.json").is_file()
    assert '>{}</script>' not in (output / "verification/ui_dom.html").read_text(encoding="utf-8")


def test_browser_failure_is_recorded_before_the_probe_raises(tmp_path, monkeypatch):
    output = demo(tmp_path)
    edge = tmp_path / "edge.exe"
    edge.write_bytes(b"fixture")
    monkeypatch.setattr("goal43_ui_probe.subprocess.run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout=b"", stderr=b"browser failed"))
    with pytest.raises(ValueError, match="UI probe"):
        probe(output, edge_path=edge)
    result = json.loads((output / "verification/ui_receipt.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
