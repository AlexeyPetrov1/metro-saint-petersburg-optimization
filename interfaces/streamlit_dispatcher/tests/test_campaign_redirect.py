import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from interfaces.streamlit_dispatcher.campaign import resolve_campaign
from interfaces.streamlit_dispatcher.tests.test_adapter import replay
from interfaces.streamlit_dispatcher.tests.test_campaign_progress import arm_files, write


def test_redirect_waits_for_completed_paired_artifacts(tmp_path):
    original, derived = tmp_path/"original", tmp_path/"derived"
    original.mkdir(); derived.mkdir()
    write(original/"campaign_scope.json", {"comparison_root": str(derived)})
    assert resolve_campaign(original, project_root=tmp_path) == original
    write(derived/"status.json", {"stage": "completed_partial"})
    assert resolve_campaign(original, project_root=tmp_path) == original
    for name in ("reference", "balanced"):
        write(derived/name/"result.json", {})
        write(derived/name/"common_scenario.json.gz", {})
    assert resolve_campaign(original, project_root=tmp_path) == derived
    (derived/"status.json").write_text("unfinished write", encoding="utf-8")
    assert resolve_campaign(original, project_root=tmp_path) == original


def test_running_app_redirects_on_refresh_only_after_ready(tmp_path, monkeypatch, replay):
    original, derived = tmp_path/"original", tmp_path/"derived"
    arm_files(original, replay, second=1000)
    monkeypatch.setenv("METRO_CAMPAIGN_DIR", str(original))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"), default_timeout=30).run()
    assert not app.exception and app.slider(key="clock_second").max == 1000
    write(original/"campaign_scope.json", {"comparison_root": str(derived)})
    arm = arm_files(derived, replay, second=54000, complete=True)
    import shutil
    shutil.copytree(arm, derived/"reference")
    app.button[0].click().run()
    assert not app.exception and app.slider(key="clock_second").max == 1000
    write(derived/"status.json", {"stage": "completed_partial"})
    app.button[0].click().run()
    assert not app.exception and app.slider(key="clock_second").max == 54000
