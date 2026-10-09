import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("metro52_campaign_tested", ROOT / "scripts/metro52_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_readiness_rejects_failures_and_missing_suites(tmp_path):
    good = tmp_path / "good.xml"
    good.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>')
    assert campaign.check_suites([good])[0]["tests"] == 3
    bad = tmp_path / "bad.xml"
    bad.write_text('<testsuites><testsuite tests="3" failures="1" errors="0" skipped="0"/></testsuites>')
    with pytest.raises(ValueError):
        campaign.check_suites([good, bad])
    with pytest.raises(FileNotFoundError):
        campaign.check_suites([tmp_path / "missing.xml"])


def test_all_arms_use_full_day_and_quarter_control(tmp_path):
    commands = campaign.arm_commands("python", tmp_path)
    assert set(commands) == {"reference", "old", "ensemble_old", "balanced"}
    for name, command in commands.items():
        assert command[command.index("--stage")+1] == "full_day"
        assert command[command.index("--control-start-second")+1] == "900"
        assert command[command.index("--output")+1] == str(tmp_path / name)
    assert commands["balanced"][commands["balanced"].index("--arm")+1] == "BALANCED"


def test_memory_limit_allows_two_models_only_with_headroom():
    assert campaign.can_start_model(0, 2, 4_500_000_000)
    assert campaign.can_start_model(0, 2, 2_500_000_000)
    assert campaign.can_start_model(1, 2, 4_500_000_000)
    assert not campaign.can_start_model(0, 2, 1_000_000)
    assert not campaign.can_start_model(1, 2, 1_000_000_000)
    assert not campaign.can_start_model(2, 2, 9_000_000_000)


def test_incidents_use_reference_arm_not_nested_motion_folder(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'metro51_incident_probe', SimpleNamespace(
        run=lambda *args, **kwargs: calls.append((args, kwargs))))
    campaign.run_incidents(tmp_path)
    args, kwargs = calls[0]
    assert args == (tmp_path / 'incidents',)
    assert kwargs['source'] == tmp_path / 'reference'
    assert kwargs['minutes'] == (3, 8, 15)
