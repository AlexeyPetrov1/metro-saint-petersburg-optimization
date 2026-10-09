from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_forecast_and_arm_switch():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert any("Входы через турникеты" in m.label for m in app.metric)
    assert len(app.dataframe) >= 1
    app.selectbox(key="arm_choice").select("ensemble_old").run()
    assert not app.exception
    assert app.selectbox(key="arm_choice").value == "ensemble_old"


def test_missing_campaign_is_reported(monkeypatch):
    monkeypatch.setenv("METRO_CAMPAIGN_DIR", "nonexistent_ui_test_campaign")
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert len(app.error) == 1
