import json

import pandas as pd
import pytest

from metro_pipeline import load_monitoring_context, prepare_pipeline, replay_pipeline


def sources(tmp_path):
    q, h = [], []
    for day in ('2026-09-02', '2026-09-09', '2026-09-16'):
        start = pd.Timestamp(f'{day}T08:00:00+03:00')
        for n in range(4):
            at = start+pd.Timedelta(minutes=15*n)
            q.append(dict(station_id='L1_05', vestibule_id='V07', interval_start=at,
                interval_end=at+pd.Timedelta(minutes=15), entry_count=10+n,
                source_id='quarters'))
        h.append(dict(station_id='L1_05', vestibule_id='V07', interval_start=start,
            interval_end=start+pd.Timedelta(hours=1), entry_count=46, source_id='hourly'))
    quarter, hourly, events = (tmp_path/name for name in ('q.parquet', 'h.parquet', 'events.json'))
    pd.DataFrame(q).to_parquet(quarter, index=False)
    pd.DataFrame(h).to_parquet(hourly, index=False)
    events.write_text(json.dumps(dict(schema_version=1, events=[])), encoding='utf-8')
    return quarter, hourly, events


def test_prepare_preserves_counts_provenance_and_refuses_overwrite(tmp_path):
    q, h, events = sources(tmp_path)
    output = tmp_path/'run'
    result = prepare_pipeline(output, quarters_path=q, hourly_path=h, registry_path=events)
    observed = pd.read_parquet(output/'observations.parquet')
    assert observed.entry_count.sum() == 138
    assert observed.quality_status.eq('ok').all()
    assert result['line_ids'] == ['L1']
    assert result['movement_telemetry'] == 'unavailable'
    assert result['line2_flow'] == 'unavailable'
    assert result['preserved_inputs_verified'] is True
    assert 'metro_pipeline.py' in result['code_hashes']
    assert list((output/'source_versions').glob('*metro_pipeline.py'))
    assert result['calendar']['availability_assumption']
    with pytest.raises(ValueError, match='empty'):
        prepare_pipeline(output, quarters_path=q, hourly_path=h, registry_path=events)


def test_replay_is_causal_scores_raw_truth_and_preserves_outputs(tmp_path):
    q, h, events = sources(tmp_path)
    output = tmp_path/'run'
    prepare_pipeline(output, quarters_path=q, hourly_path=h, registry_path=events)
    summary = replay_pipeline(output, cases=[dict(as_of='2026-09-16T08:00:00+03:00',
        station_ids=['L1_05'], horizon_hours=1, purpose='fixture')])
    assert summary['models']['regime']['all']['wape'] == 0
    assert summary['models']['legacy']['all']['n'] == 4
    assert summary['models']['regime']['event_detection']['recall'] is None
    assert summary['preserved_inputs_verified'] is True
    assert (output/'replay/guard_decisions.json').exists()
    assert (output/'replay/monitor_alerts.json').exists()
    with pytest.raises(ValueError, match='empty'):
        replay_pipeline(output, cases=[])


def test_monitoring_context_enforces_snapshot_hashes_and_explicit_mode(tmp_path):
    q, h, events = sources(tmp_path)
    output = tmp_path/'run'
    prepare_pipeline(output, quarters_path=q, hourly_path=h, registry_path=events)
    context = load_monitoring_context(output)
    assert context['knowledge_mode'] == 'strict'
    assert context['movement'] is None
    assert len(context['observations']) == 12
    assert load_monitoring_context(tmp_path/'legacy') is None
    (output/'events.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        load_monitoring_context(output)


def two_station_sources(tmp_path):
    q, h, events = sources(tmp_path)
    for path in (q, h):
        first = pd.read_parquet(path)
        second = first.copy()
        second['station_id'], second['vestibule_id'] = 'L1_06', 'V08'
        pd.concat([first, second], ignore_index=True).to_parquet(path, index=False)
    return q, h, events


def test_replay_compares_common_eligible_rows_and_separates_purposes(tmp_path, monkeypatch):
    import metro_pipeline

    q, h, events = two_station_sources(tmp_path)
    output = tmp_path/'run'
    prepare_pipeline(output, quarters_path=q, hourly_path=h, registry_path=events)

    def partial_forecast(frame, station_ids, at, **kwargs):
        return metro_pipeline.hourly_median(frame, ['L1_05'], at, kwargs['horizon_hours'])

    monkeypatch.setattr(metro_pipeline, 'regime_hourly_median', partial_forecast)
    cases = [dict(as_of=f'2026-09-{day}T08:00:00+03:00', station_ids=['L1_05', 'L1_06'],
                  horizon_hours=1, purpose=purpose)
             for day, purpose in [('09', 'repair_diagnostic'), ('16', 'temporal_holdout')]]
    summary = replay_pipeline(output, cases=cases)
    assert summary['models']['legacy']['all']['n'] == 16
    assert summary['models']['regime']['all']['n'] == 8
    comparison = summary['comparison']
    assert comparison['common_eligible_rows'] == 8
    assert comparison['models']['legacy']['all']['n'] == comparison['models']['regime']['all']['n'] == 8
    assert comparison['models']['legacy']['by_purpose']['temporal_holdout']['all']['n'] == 4
    assert summary['models']['legacy']['by_purpose']['repair_diagnostic']['all']['n'] == 8
    assert summary['evaluation_scope']['requested_case_count'] == 2
    assert summary['evaluation_scope']['dates'] == ['2026-09-09', '2026-09-16']


def test_same_origin_cases_preserve_bundles_and_deduplicate_scoring_keys(tmp_path):
    q, h, events = two_station_sources(tmp_path)
    output = tmp_path/'run'
    prepare_pipeline(output, quarters_path=q, hourly_path=h, registry_path=events)
    cases = [dict(as_of='2026-09-16T08:00:00+03:00', station_ids=ids, horizon_hours=1, purpose='fixture')
             for ids in [['L1_05'], ['L1_05', 'L1_06']]]
    summary = replay_pipeline(output, cases=cases)
    bundles = [p for p in (output/'replay').glob('legacy_*.json') if p.name != 'legacy_scored_rows.json']
    assert len(bundles) == 2
    assert summary['models']['legacy']['all']['n'] == 8
    assert summary['comparison']['common_eligible_rows'] == 8
    assert summary['evaluation_scope']['repeated_forecast_keys']['legacy'] == 4
    assert summary['evaluation_scope']['warnings']


def test_failed_model_does_not_create_comparison_win(tmp_path, monkeypatch):
    import metro_pipeline

    q, h, events = sources(tmp_path)
    output = tmp_path/'run'
    prepare_pipeline(output, quarters_path=q, hourly_path=h, registry_path=events)

    def unavailable(*args, **kwargs):
        raise ValueError('event regime unavailable')

    monkeypatch.setattr(metro_pipeline, 'regime_hourly_median', unavailable)
    summary = replay_pipeline(output, cases=[dict(as_of='2026-09-16T08:00:00+03:00', station_ids=['L1_05'], purpose='fixture')])
    assert summary['comparison']['common_eligible_rows'] == 0
    assert summary['comparison']['status'] == 'unavailable'
    assert summary['unscorable_cases'][0]['model'] == 'regime'


def test_scope_reports_actual_targets_repeated_under_different_origins(tmp_path, monkeypatch):
    import metro_pipeline

    q, h, events = sources(tmp_path)
    output = tmp_path/'run'
    prepare_pipeline(output, quarters_path=q, hourly_path=h, registry_path=events)
    original = metro_pipeline.hourly_median

    def same_target(frame, ids, at, *args, **kwargs):
        bundle = original(frame, ids, '2026-09-16T08:00:00+03:00', 1)
        for field in ('as_of', 'issued_at', 'available_at'):
            bundle[field] = at
        return bundle

    monkeypatch.setattr(metro_pipeline, 'hourly_median', same_target)
    monkeypatch.setattr(metro_pipeline, 'regime_hourly_median', same_target)
    cases = [dict(as_of=f'2026-09-16T{hour}:00:00+03:00', station_ids=['L1_05'], purpose='fixture') for hour in ('07', '08')]
    summary = replay_pipeline(output, cases=cases)
    scope = summary['evaluation_scope']
    assert scope['observed_targets_with_multiple_origins'] == scope['unique_observed_targets'] == 4
    assert summary['comparison']['common_eligible_rows'] == 8
    assert any('correlated' in warning for warning in scope['warnings'])


def test_empty_replay_reports_unavailable_comparison(tmp_path):
    q, h, events = sources(tmp_path)
    output = tmp_path/'run'
    prepare_pipeline(output, quarters_path=q, hourly_path=h, registry_path=events)
    summary = replay_pipeline(output, cases=[])
    assert summary['comparison']['status'] == 'unavailable'
    assert summary['comparison']['models']['legacy']['all']['wape'] is None
    assert summary['evaluation_scope']['requested_case_count'] == 0
