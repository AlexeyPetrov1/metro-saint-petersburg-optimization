"""Browser checks against the local Streamlit server; no simulation or model runs."""

import json
import os
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from interfaces.streamlit_dispatcher.adapter import Replay
from interfaces.streamlit_dispatcher.player import player_html

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "interfaces/streamlit_dispatcher/artifacts"


@pytest.fixture
def browser():
    executable = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, executable_path=str(executable) if executable.exists() else None)
        yield b
        b.close()


def test_player_clock_forecast_queue_and_train_load(browser):
    replay = Replay.from_arm(ROOT / "data/processed/metro51_balanced_pilot_20261009_v2/balanced")
    payload = replay.player_payload()
    payload["start_pos"] = 8
    page = browser.new_page(viewport={"width":1500, "height":1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda e: errors.append(e.text) if e.type == "error" else None)
    page.set_content(player_html(payload))
    page.wait_for_function("document.getElementById('clock').textContent === '08:00:00'")
    expected = replay.frame(7200)
    assert int(page.locator("#k_wait").inner_text().replace("\u202f", "").replace("\xa0", "")) == expected["waiting"]
    assert int(page.locator("#k_denied").inner_text().replace("\u202f", "").replace("\xa0", "")) == expected["onboard"]
    assert page.locator("#forecastRows b").count() == 19
    assert page.locator("#fleet .train").count() == sum(t["on_line"] for t in expected["trains"])
    assert "no_feasible_candidate" in page.locator("#decisionMeta").inner_text()
    for second in [0, 3599, 3600, 7199, 7200, replay.end]:
        page.evaluate("t => {seek(t/900); render();}", second)
        expected = replay.frame(second)
        waiting = page.locator("#k_wait").inner_text().replace("\u202f", "").replace("\xa0", "")
        assert int(waiting) == expected["waiting"]
        expected_forecast = expected["forecast"]
        if expected_forecast:
            issue = expected_forecast["issued_at"][11:16]
            assert issue in page.locator("#forecastMeta").inner_text()
        if second < 3600:
            assert "нет опубликованного" in page.locator("#forecastMeta").inner_text()
    assert not errors
    page.evaluate("() => {seek(8); render();}")
    ARTIFACTS.mkdir(exist_ok=True)
    assert page.locator("#decisionText").evaluate("e => e.clientHeight") <= 360
    page.screenshot(path=str(ARTIFACTS / "player.png"), full_page=True)
    page.screenshot(path=str(ARTIFACTS / "player_viewport.png"))
    page.close()


def test_streamlit_pages(browser):
    page = browser.new_page(viewport={"width":1600, "height":1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(os.environ.get("METRO_UI_TEST_URL", "http://127.0.0.1:8501"))
    page.get_by_text("Прогноз входов на станции", exact=True).wait_for(timeout=30000)
    page.locator('[data-testid="stDataFrame"]').first.wait_for(state="visible")
    page.wait_for_timeout(700)
    assert page.locator('[data-testid="stException"]').count() == 0
    page.screenshot(path=str(ARTIFACTS / "forecast.png"), full_page=True)
    page.get_by_role("link", name="Симулятор", exact=False).click()
    frame = page.frame_locator("iframe").first
    frame.locator("#clock").wait_for(timeout=30000)
    page.wait_for_timeout(700)
    assert frame.locator("#fleet .train").count() > 0
    assert frame.locator("#forecastRows b").count() == 19
    page.screenshot(path=str(ARTIFACTS / "simulator.png"), full_page=True)
    page.get_by_role("link", name="Оптимизатор", exact=False).click()
    page.get_by_text("Решение оптимизатора", exact=True).wait_for(timeout=30000)
    page.wait_for_timeout(700)
    assert page.locator('[data-testid="stException"]').count() == 0
    page.screenshot(path=str(ARTIFACTS / "optimizer.png"), full_page=True)
    page.get_by_role("link", name="Данные", exact=False).click()
    page.get_by_text("Источники интерфейса", exact=True).wait_for(timeout=30000)
    assert page.locator('[data-testid="stException"]').count() == 0
    assert not errors
    page.close()


def test_full_day_replay_at_evening_and_closure(browser):
    arm = ROOT / "data/processed/metro50_full_day_20261009_v4/reference"
    if not arm.exists():
        pytest.skip("local completed full-day campaign unavailable")
    replay = Replay.from_arm(arm)
    page = browser.new_page(viewport={"width":1500, "height":1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.set_content(player_html(replay.player_payload()))
    page.wait_for_function("document.getElementById('clock').textContent === '06:00:00'")
    for second, clock in [(39600, "17:00:00"), (46800, "19:00:00"), (64800, "00:00:00"),
                          (72000, "02:00:00")]:
        page.evaluate("t => {seek(t/900); render();}", second)
        expected = replay.frame(second)
        assert page.locator("#clock").inner_text() == clock
        for selector, key in [("#k_wait", "waiting"), ("#k_denied", "onboard"), ("#k_wpm", "entered")]:
            value = page.locator(selector).inner_text().replace("\u202f", "").replace("\xa0", "")
            assert int(value) == expected[key]
        assert "нет опубликованного прогноза" in page.locator("#forecastMeta").inner_text()
        assert "нет решения" in page.locator("#decisionMeta").inner_text()
    assert not errors
    page.close()
