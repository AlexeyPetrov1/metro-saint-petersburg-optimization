import copy
from datetime import datetime, timedelta

import pandas as pd
import pytest

from metro_daily import (
    reference_counts, desired_headway, oracle_bundle, proposal_specs,
    prepare_day, rebuild_timetable, exposure_safe,
)
from forecasting import validate_bundle
from goal43_policy import load_policy
from simulation import simulate
from rolling import operational_headways


def test_reference_counts_are_photo_rows_not_99_minute_arithmetic():
    counts = reference_counts()
    assert [counts[h] for h in (6, 7, 8, 9, 11, 12, 17, 23)] == [33, 50, 53, 43, 33, 30, 50, 18]


def test_requested_count_is_projected_to_physical_headway_across_route():
    policy = load_policy()
    assert desired_headway(datetime.fromisoformat('2026-09-30T07:00:00+03:00'), 6080, policy) >= 150
    assert desired_headway(datetime.fromisoformat('2026-09-30T09:30:00+03:00'), 6080, policy) >= 150
    assert 113 <= desired_headway(datetime.fromisoformat('2026-09-30T08:15:00+03:00'), 6080, policy) < 150
    assert desired_headway(datetime.fromisoformat('2026-09-30T12:00:00+03:00'), 6080, policy) == 203


def test_oracle_is_exact_station_time_join_and_truncates_at_service_end():
    start = pd.Timestamp('2026-09-30T23:00:00+03:00')
    rows = [dict(station_id=s, interval_start=start+pd.Timedelta(minutes=15*q),
        interval_end=start+pd.Timedelta(minutes=15*(q+1)), entry_count=100*q+i,
        is_complete=True, source_id='uploaded-spb-line1-15min')
        for q in range(4) for i, s in enumerate(['B', 'A'])]
    frame = pd.DataFrame(rows).sample(frac=1, random_state=7)
    bundle = oracle_bundle(frame, ['A', 'B'], start.isoformat(), (start+pd.Timedelta(hours=1)).isoformat())
    assert validate_bundle(bundle, ['A', 'B'], start.isoformat())['available_horizon_seconds'] == 3600
    assert bundle['evaluation_mode'] == 'research_oracle_future_entries'
    assert bundle['source_versions']['operationally_available'] is False
    assert {(r['station_id'], r['interval_start']): r['predicted_count'] for r in bundle['rows']} == {
        (r['station_id'], r['interval_start'].isoformat()): r['entry_count'] for r in rows}
    frame.loc[frame.index[0], 'is_complete'] = False
    with pytest.raises(ValueError, match='coverage'):
        oracle_bundle(frame, ['A', 'B'], start.isoformat(), (start+pd.Timedelta(hours=1)).isoformat())


def test_search_includes_one_two_four_trains_and_different_assignments():
    specs = proposal_specs()
    assert {s['count_delta'] for s in specs} >= {0, -1, -2, -4, 2}
    assert len({s['priority_rotation'] for s in specs if s['count_delta'] == -2}) >= 2
    assert len({s['candidate_id'] for s in specs}) == len(specs)


def test_a_global_pass_does_not_hide_a_bad_fifteen_minute_window():
    exposure = dict(admissible=False, global_status='pass', windows=[dict(status='fail')])
    assert not exposure_safe(exposure)


@pytest.fixture(scope='module')
def day():
    return prepare_day('2026-09-30', service_hours=6)


def test_baseline_contains_hourly_changes_depot_moves_and_closed_returns(day):
    data, roster, profiles, meta = day
    motion = simulate(data, data['plans'][0]['plan_id'])
    assert len(roster) == len(data['trains']) == 68
    assert motion['obligations_closed']
    assert motion['checks']['peak_trains_on_line'] <= 53
    assert operational_headways(data, motion, 0, meta['service_end_second'])['passed']
    assert any(t['route'][0].startswith('ENTRY_') for t in data['plans'][0]['trips'])
    assert any(t['route'][0].startswith('EXIT_') for t in data['plans'][0]['trips'])
    assert meta['reference_date_verified'] is False
    assert meta['initial_count'] == 33
    first = {(t['train_id'],t['trip_id']):t['departure_second'] for t in data['plans'][0]['trips']}
    assert all(r['departure_second'] >= first[(r['train_id'],r['trip_id'])]
               for r in data['leg_timing'] if r['leg_index']==0 and r['trip_id']!='@initial')


def test_whole_day_projection_preserves_all_headway_windows():
    data,_,_,meta = prepare_day('2026-09-30')
    motion=simulate(data,data['plans'][0]['plan_id'])
    assert operational_headways(data,motion,0,meta['service_end_second'])['passed']
    assert motion['obligations_closed']


def test_forecast_cache_rejects_future_issue_and_wrong_oracle_label(tmp_path):
    import json
    from metro_daily_run import read_forecast
    bundle=dict(schema_version='1.0',model_version='test',issued_at='2026-09-30T12:00:00+03:00',
        as_of='2026-09-30T12:00:00+03:00',available_at='2026-09-30T12:00:00+03:00',
        features_available_at='2026-09-30T11:00:00+03:00',target='turnstile_entries',unit='passengers_per_15m',
        coverage='complete',feature_version='test',source_versions=dict(test=True),included_event_ids=[],
        rows=[dict(station_id='A',interval_start='2026-09-30T12:00:00+03:00',
            interval_end='2026-09-30T12:15:00+03:00',predicted_count=100,coverage='complete')])
    folder=tmp_path/'forecasts';folder.mkdir()
    (folder/'ML_12.json').write_text(json.dumps(bundle))
    (folder/'ML_11.json').write_text(json.dumps(bundle))
    assert read_forecast(tmp_path,'ML',['A'],'2026-09-30T12:00:00+03:00')==bundle
    with pytest.raises(ValueError):read_forecast(tmp_path,'ML',['A'],'2026-09-30T11:45:00+03:00')
    (folder/'ORACLE_12.json').write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='oracle'):
        read_forecast(tmp_path,'ORACLE',['A'],'2026-09-30T12:00:00+03:00')


def test_known_service_closure_keeps_a_transport_clearance_horizon(tmp_path):
    import json
    from metro_daily_run import read_forecast
    start=pd.Timestamp('2026-09-30T23:00:00+03:00')
    frame=pd.DataFrame([dict(station_id='A',interval_start=start+pd.Timedelta(minutes=15*q),
        interval_end=start+pd.Timedelta(minutes=15*(q+1)),entry_count=10,is_complete=True,
        source_id='uploaded-spb-line1-15min') for q in range(4)])
    bundle=oracle_bundle(frame,['A'],start.isoformat(),'2026-10-01T00:00:00+03:00')
    folder=tmp_path/'forecasts';folder.mkdir()
    (folder/'ORACLE_23.json').write_text(json.dumps(bundle))
    forecast=read_forecast(tmp_path,'ORACLE',['A'],'2026-09-30T23:45:00+03:00')
    assert validate_bundle(forecast,['A'],'2026-09-30T23:45:00+03:00')['available_horizon_seconds']==4500
    assert [r['predicted_count'] for r in forecast['rows']]==[10]*4+[0]*4
    assert forecast['source_versions']['service_closure']['status']=='known experiment calendar, not fabricated observed data'


def test_replanning_preserves_executed_movement_and_couples_km_to_service(day):
    data, _, _, meta = day
    before = copy.deepcopy(data)
    changed, audit = rebuild_timetable(data, 3600, meta['service_end_second'], count_delta=-4,
        change_end_second=meta['service_end_second'], priority_rotation=1)
    assert data == before
    old = simulate(data, data['plans'][0]['plan_id'])
    new = simulate(changed, changed['plans'][0]['plan_id'])
    assert [e for e in old['events'] if e['second'] < 3600] == [e for e in new['events'] if e['second'] < 3600]
    assert new['obligations_closed']
    assert new['checks']['peak_trains_on_line'] <= 53
    assert audit['committed_until_second'] >= 3600+780
    assert audit['passenger_trips'] != sum(t['passenger'] for t in data['plans'][0]['trips'])
