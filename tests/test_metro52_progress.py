import json
from pathlib import Path

from scripts.metro52_progress import progress_snapshot, write_progress


def test_progress_uses_contract_and_only_committed_decisions(tmp_path):
    arm = tmp_path / 'balanced'
    (arm / 'controller').mkdir(parents=True)
    (arm / 'contract.json').write_text(json.dumps(dict(
        control_start_second=900, cohort_end_second=64800)))
    (arm / 'controller/decision_00900.json').write_text(json.dumps(dict(
        decision_time='2026-09-30T06:15:00+03:00', applied_candidate_id='KEEP')))
    (arm / 'controller/decision_01800.json.tmp').write_text('{}')
    (tmp_path / 'status.json').write_text(json.dumps(dict(
        stage='running', pending_arms=[], skipped_arms=['old', 'ensemble_old'])))
    snapshot = progress_snapshot(tmp_path)
    assert (snapshot['finished_steps'], snapshot['total_steps']) == (1, 71)
    assert snapshot['last_decision'] == '2026-09-30T06:15:00+03:00'
    assert snapshot['full_day_finished'] is False
    assert snapshot['skipped_arms'] == ['old', 'ensemble_old']


def test_missing_or_incomplete_status_keeps_progress_readable(tmp_path):
    (tmp_path / 'status.json').write_text('{')
    snapshot = progress_snapshot(tmp_path)
    assert snapshot['stage'] == 'waiting_for_status'
    write_progress(tmp_path, snapshot)
    html = (tmp_path / 'progress.html').read_text(encoding='utf8')
    assert 'http-equiv="refresh"' in html
    assert 'charset="utf-8"' in html
    assert 'localhost:8501' in html
    assert '\u041f\u0440\u043e\u0433\u0440\u0435\u0441\u0441' in html


def test_completed_requires_campaign_completion_not_last_decision(tmp_path):
    (tmp_path / 'status.json').write_text(json.dumps(dict(stage='reporting')))
    assert progress_snapshot(tmp_path)['full_day_finished'] is False
    (tmp_path / 'status.json').write_text(json.dumps(dict(stage='completed')))
    assert progress_snapshot(tmp_path)['full_day_finished'] is True


def test_error_message_is_escaped_in_browser(tmp_path):
    snapshot = dict(stage='failed', error='<script>alert(1)</script>',
                    finished_steps=0, total_steps=0, last_decision=None,
                    full_day_finished=False, skipped_arms=[], llm={}, updated_at='now')
    write_progress(tmp_path, snapshot)
    html = (tmp_path / 'progress.html').read_text(encoding='utf8')
    assert '<script>' not in html
    assert '&lt;script&gt;' in html
