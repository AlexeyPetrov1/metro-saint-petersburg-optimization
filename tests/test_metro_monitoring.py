import copy

import pandas as pd
import pytest

from metro_monitoring import (bounded_demand, control_guard, guard_candidates,
                              movement_state_at, validate_movement)


NOW = pd.Timestamp('2026-07-17T18:00:00+03:00')
STATIONS = ['L1_01', 'L1_02']


class Registry:
    def __init__(self, *, entrance='open', movement='normal', available_at=None):
        self.entrance, self.movement = entrance, movement
        self.available_at = available_at

    def state_at(self, station_id, interval_start, as_of, vestibule_id=None,
                 knowledge_mode='strict'):
        known = self.available_at is None or pd.Timestamp(as_of) >= self.available_at
        return dict(entrance_state=self.entrance if known else 'open',
                    movement_state=self.movement if known else 'normal',
                    demand_censored=known and self.entrance in {'closed', 'restricted'},
                    event_ids=['closure'] if known and self.entrance != 'open' else [],
                    conflicts=[], unknown_timing=[])


def entries(**changes):
    rows = [dict(station_id=sid, interval_start=NOW-pd.Timedelta(minutes=15),
                 interval_end=NOW, entry_count=992, is_complete=True,
                 quality_status='ok', quality_available_at=NOW) for sid in STATIONS]
    for row in rows:
        row.update(changes)
    return pd.DataFrame(rows)


def motion(**changes):
    rows = [dict(line_id='L1', station_id=sid, observed_at=NOW-pd.Timedelta(minutes=1),
                 available_at=NOW, movement_state='normal', measured_headway_s=180,
                 reference_headway_s=180) for sid in STATIONS]
    for row in rows:
        row.update(changes)
    return pd.DataFrame(rows)


def test_near_normal_entries_do_not_hide_movement_disruption():
    result = control_guard(entries(), Registry(), NOW, STATIONS,
                           movement=motion(movement_state='disrupted'))
    assert result['reduction_allowed'] is False
    assert 'movement_disrupted' in result['reasons']
    assert result['demand_status'] == 'observed'
    assert result['motion_coverage'] == 'complete'


def test_restricted_entry_drop_is_censored_without_invented_queue():
    result = control_guard(entries(entry_count=0), Registry(entrance='restricted'),
                           NOW, STATIONS, movement=motion())
    assert result['reduction_allowed'] is False
    assert 'demand_censored' in result['reasons']
    assert result['event_ids'] == ['closure']
    assert result['demand_status'] == 'censored'
    assert not any('queue' in key for key in result)
    assert bounded_demand(0, 'restricted') == dict(
        observed_admitted_count=0, true_demand_lower_bound=0,
        true_demand_upper_bound=None, censored=True)


def test_normal_available_signals_allow_reduction_and_inputs_are_preserved():
    frame, movement = entries(), motion()
    original_frame, original_motion = frame.copy(deep=True), movement.copy(deep=True)
    result = control_guard(frame, Registry(), NOW, STATIONS, movement=movement)
    assert result['reduction_allowed'] is True
    assert result['reasons'] == []
    assert pd.Timestamp(result['latest_movement_at']) == NOW-pd.Timedelta(minutes=1)
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_frame_equal(movement, original_motion)


def test_nullable_integer_observations_are_valid_counts():
    frame = entries()
    frame['entry_count'] = frame.entry_count.astype('Int64')
    result = control_guard(frame, Registry(), NOW, STATIONS, movement=motion())
    assert result['reduction_allowed'] is True
    assert result['demand_status'] == 'observed'


def test_known_headway_inflation_vetoes_normal_state_and_marks_conflict():
    result = control_guard(entries(), Registry(), NOW, STATIONS,
                           movement=motion(measured_headway_s=300))
    assert result['reduction_allowed'] is False
    assert 'movement_disrupted' in result['reasons']
    assert 'movement_state_headway_conflict' in result['reasons']


@pytest.mark.parametrize('changes', [
    {'observed_at': NOW-pd.Timedelta(minutes=16)},
    {'observed_at': NOW+pd.Timedelta(minutes=1), 'available_at': NOW+pd.Timedelta(minutes=1)},
    {'available_at': NOW+pd.Timedelta(minutes=1)},
])
def test_stale_future_and_late_telemetry_stays_unknown(changes):
    result = movement_state_at(motion(**changes), NOW, STATIONS)
    assert result['movement_state'] == 'unknown'
    assert result['motion_coverage'] == 'none'
    assert result['latest_movement_at'] is None
    assert result['unknowns']


def test_missing_optional_telemetry_discloses_unknown_and_strict_opt_in_blocks():
    optional = control_guard(entries(), Registry(), NOW, STATIONS)
    strict = control_guard(entries(), Registry(), NOW, STATIONS,
                           require_movement_coverage=True)
    assert optional['reduction_allowed'] is True
    assert optional['motion_coverage'] == 'none'
    assert 'movement_not_supplied' in optional['unknowns']
    assert strict['reduction_allowed'] is False
    assert 'movement_coverage_incomplete' in strict['reasons']


def test_publication_after_decision_does_not_leak_event_state():
    registry = Registry(entrance='closed', available_at=NOW+pd.Timedelta(minutes=1))
    result = control_guard(entries(), registry, NOW, STATIONS, movement=motion())
    assert result['reduction_allowed'] is True
    assert result['event_ids'] == []
    assert result['demand_status'] == 'observed'


def test_announced_future_closure_vetoes_reduction_over_decision_horizon():
    from metro_events import EventRegistry
    from test_metro_events import record
    event = record(scope={'level': 'line'}, published_at='2026-07-17T17:00:00+03:00',
                   ingested_at='2026-07-17T17:01:00+03:00', timing={'planned': {
                       'start': '2026-07-17T18:10:00+03:00', 'end': '2026-07-17T18:20:00+03:00'}})
    registry = EventRegistry.from_records([event])
    immediate = control_guard(entries(), registry, NOW, STATIONS, movement=motion())
    future = control_guard(entries(), registry, NOW, STATIONS, movement=motion(), horizon_seconds=1800)
    assert immediate['reduction_allowed'] is True
    assert future['reduction_allowed'] is False
    assert future['event_ids'] == ['closure']
    assert future['demand_status'] == 'censored'
    boundary = control_guard(entries(), registry, NOW, STATIONS, horizon_seconds=600)
    assert boundary['reduction_allowed'] is True


def test_future_horizon_cannot_use_late_publication_or_extend_ended_closure():
    from metro_events import EventRegistry
    from test_metro_events import record
    late = record(scope={'level': 'line'}, published_at='2026-07-17T18:01:00+03:00',
                  ingested_at='2026-07-17T18:01:00+03:00', timing={'planned': {
                      'start': '2026-07-17T18:10:00+03:00', 'end': '2026-07-17T18:20:00+03:00'}})
    ended = record(scope={'level': 'line'}, published_at='2026-07-17T17:00:00+03:00',
                   ingested_at='2026-07-17T17:01:00+03:00', timing={'planned': {
                       'start': '2026-07-17T17:30:00+03:00', 'end': '2026-07-17T17:40:00+03:00'}})
    for event in (late, ended):
        result = control_guard(entries(), EventRegistry.from_records([event]), NOW, STATIONS,
                               movement=motion(), horizon_seconds=1800)
        assert result['reduction_allowed'] is True
        assert result['event_ids'] == []


def test_received_at_blocks_counts_that_have_not_arrived_at_decision():
    result = control_guard(entries(received_at=NOW+pd.Timedelta(minutes=1)), Registry(), NOW,
                           STATIONS, movement=motion())
    assert result['reduction_allowed'] is False
    assert result['demand_status'] == 'unknown'
    assert 'entry_observation_missing' in result['reasons']


@pytest.mark.parametrize('duration', [-1, float('nan'), float('inf'), True])
def test_invalid_control_horizon_is_rejected(duration):
    with pytest.raises(ValueError, match='horizon'):
        control_guard(entries(), Registry(), NOW, STATIONS, horizon_seconds=duration)


def test_known_source_mismatch_blocks_but_late_diagnostic_does_not_leak():
    mismatch = entries(quality_status='source_mismatch')
    result = control_guard(mismatch, Registry(), NOW, STATIONS, movement=motion())
    assert result['reduction_allowed'] is False
    assert 'source_quality_unreliable' in result['reasons']
    late = mismatch.assign(quality_available_at=NOW+pd.Timedelta(hours=1))
    result = control_guard(late, Registry(), NOW, STATIONS, movement=motion())
    assert 'source_quality_unreliable' not in result['reasons']
    assert 'source_quality_unavailable' in result['unknowns']


def test_unknown_entry_values_are_not_zero_filled():
    result = control_guard(entries(entry_count=None), Registry(), NOW, STATIONS)
    assert result['demand_status'] == 'unknown'
    assert result['reduction_allowed'] is False
    assert 'entry_observation_missing' in result['reasons']


@pytest.mark.parametrize('changes', [
    {'observed_at': '2026-07-17T17:59:00'},
    {'available_at': '2026-07-17T18:00:00'},
    {'available_at': NOW-pd.Timedelta(minutes=2)},
    {'measured_headway_s': -1}, {'reference_headway_s': 0},
    {'measured_headway_s': float('inf')}, {'movement_state': 'maybe'},
    {'station_id': None, 'track_id': None},
])
def test_telemetry_contract_rejects_invalid_times_headways_states_and_location(changes):
    with pytest.raises(ValueError):
        validate_movement(motion(**changes))


def test_unknown_headway_values_stay_missing_and_are_never_imputed():
    result = validate_movement(motion(measured_headway_s=None, reference_headway_s=None))
    assert result.measured_headway_s.isna().all()
    assert result.reference_headway_s.isna().all()


def test_other_line_disruption_is_isolated_and_track_disruption_is_line_wide():
    other = motion(line_id='L2', station_id='L2_01', movement_state='disrupted')
    result = control_guard(entries(), Registry(), NOW, STATIONS, movement=other)
    assert result['reduction_allowed'] is True
    track = motion(station_id=None, track_id='L1_01L1_02_up', movement_state='disrupted')
    result = control_guard(entries(), Registry(), NOW, STATIONS, movement=track)
    assert result['reduction_allowed'] is False
    assert result['motion_coverage'] == 'partial'


def test_candidate_guard_rejects_only_reductions_and_keeps_originals():
    candidates = [
        dict(candidate_id='keep', action=dict(kind='KEEP'), status='movement_checked'),
        dict(candidate_id='release', action=dict(kind='ADD'), status='movement_checked'),
        dict(candidate_id='count', count_delta=-2),
        dict(candidate_id='withdraw', action=dict(kind='WITHDRAW_ONLY')),
        dict(candidate_id='nested', action=dict(kind='COMBINED', actions=[dict(kind='WITHDRAW')])),
        dict(candidate_id='positive', count_delta=2),
        dict(candidate_id='defer', action=dict(kind='DEFER_WITHDRAWAL')),
    ]
    original = copy.deepcopy(candidates)
    guard = control_guard(entries(), Registry(entrance='closed'), NOW, STATIONS)
    result = guard_candidates(candidates, guard)
    assert candidates == original
    for row in result:
        assert row['control_guard'] == guard
        if row['candidate_id'] in {'count', 'withdraw', 'nested'}:
            assert row['status'] == 'rejected'
            assert row['reason'].startswith('control_guard:')
        else:
            assert row.get('status') != 'rejected'


def test_newest_available_observation_supersedes_older_disruption():
    old = motion(observed_at=NOW-pd.Timedelta(minutes=4),
                 available_at=NOW-pd.Timedelta(minutes=3), movement_state='disrupted')
    result = movement_state_at(pd.concat([old, motion()]), NOW, STATIONS)
    assert result['movement_state'] == 'normal'
    assert result['motion_coverage'] == 'complete'


def test_conflicting_simultaneous_headways_cannot_hide_inflation():
    data = pd.concat([motion(measured_headway_s=300), motion()], ignore_index=True)
    result = control_guard(entries(), Registry(), NOW, STATIONS, movement=data)
    assert result['reduction_allowed'] is False
    assert 'movement_disrupted' in result['reasons']


def test_confirmed_disruption_with_uncertain_end_vetoes_reductions():
    from metro_events import EventRegistry
    from test_metro_events import record
    event = record(kind='train_disruption', scope={'level': 'line'}, entrance_state='open',
                   movement_state='disrupted', published_at='2026-07-17T17:42:00+03:00',
                   ingested_at='2026-07-17T17:43:00+03:00', timing={'observed': {
                       'start': '2026-07-17T17:42:00+03:00', 'end': None,
                       'date_end': '2026-07-17',
                       'restoration_report_at': '2026-07-17T18:41:00+03:00'}})
    result = control_guard(entries(), EventRegistry.from_records([event]), NOW, STATIONS)
    assert result['reduction_allowed'] is False
    assert 'movement_status_uncertain' in result['reasons']
    assert 'event_timing_unknown' in result['unknowns']


def test_brief_closed_entrance_inside_completed_quarter_remains_censored():
    from metro_events import EventRegistry
    from test_metro_events import record
    event = record(scope={'level': 'line'}, published_at='2026-07-17T17:40:00+03:00',
                   ingested_at='2026-07-17T17:41:00+03:00', timing={'planned': {
                       'start': '2026-07-17T17:50:00+03:00', 'end': '2026-07-17T17:52:00+03:00'}})
    result = control_guard(entries(), EventRegistry.from_records([event]), NOW, STATIONS)
    assert result['demand_status'] == 'censored'
    assert result['reduction_allowed'] is False
    assert 'demand_censored' in result['reasons']


@pytest.mark.parametrize('count', [-1, float('nan'), True, 1.5])
def test_bounded_demand_rejects_invalid_admitted_counts(count):
    with pytest.raises(ValueError):
        bounded_demand(count, 'closed')
