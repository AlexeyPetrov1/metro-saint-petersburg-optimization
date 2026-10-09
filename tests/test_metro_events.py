import pytest

from metro_events import EventRegistry


def record(**changes):
    value = {
        'event_id': 'closure', 'version_id': 'closure:v1', 'version': 1,
        'kind': 'planned_entrance_closure', 'line_id': 'L1',
        'scope': {'level': 'station', 'station_ids': ['L1_05']},
        'published_at': '2026-06-16T10:00:00+03:00',
        'ingested_at': '2026-06-16T10:05:00+03:00',
        'confidence': 'confirmed', 'sources': [{'url': 'https://metro.spb.ru/example'}],
        'entrance_state': 'closed', 'movement_state': 'normal',
        'timing': {'planned': {'date_start': '2026-06-18', 'date_end': '2026-08-28',
                              'windows': [{'weekdays': [0, 1, 2, 3, 4],
                                           'start_time': '08:10', 'end_time': '09:15'}]}}
    }
    value.update(changes)
    return value


def state(registry, at='2026-07-01T08:30:00+03:00', **kwargs):
    return registry.state_at('L1_05', at, kwargs.pop('as_of', at), **kwargs)


def test_historical_visibility_uses_both_publication_and_ingestion():
    registry = EventRegistry.from_records([record(ingested_at='2026-10-09T00:00:00+03:00')])
    assert state(registry)['entrance_state'] == 'open'
    assert state(registry, knowledge_mode='publication_replay')['entrance_state'] == 'closed'
    assert state(registry, as_of='2026-06-16T09:00:00+03:00', knowledge_mode='publication_replay')['event_ids'] == []
    assert state(registry, as_of='2026-10-09T01:00:00+03:00')['demand_censored']


def test_unknown_publication_is_never_available_for_online_replay():
    registry = EventRegistry.from_records([record(published_at=None)])
    assert state(registry, knowledge_mode='publication_replay')['event_ids'] == []
    assert registry.label_at('L1_05', '2026-07-01T08:30:00+03:00')['entrance_state'] == 'closed'


def test_versions_retain_knowledge_history():
    v2 = record(version=2, version_id='closure:v2', entrance_state='restricted',
                published_at='2026-07-02T08:00:00+03:00', ingested_at='2026-07-02T08:01:00+03:00')
    registry = EventRegistry.from_records([v2, record()])
    assert state(registry)['version_ids'] == ['closure:v1']
    assert state(registry, at='2026-07-02T08:30:00+03:00')['entrance_state'] == 'restricted'
    assert registry.label_at('L1_05', '2026-07-01T08:30:00+03:00')['version_ids'] == ['closure:v2']


def test_recurring_windows_have_inclusive_dates_and_half_open_times():
    registry = EventRegistry.from_records([record()])
    assert state(registry, '2026-08-28T09:14:59+03:00')['entrance_state'] == 'closed'
    for at in ('2026-08-28T09:15:00+03:00', '2026-08-29T08:30:00+03:00',
               '2026-08-31T08:30:00+03:00', '2026-07-04T08:30:00+03:00'):
        assert state(registry, at)['entrance_state'] == 'open'


def test_overnight_window_belongs_to_start_weekday_and_start_date():
    value = record(timing={'planned': {'date_start': '2026-06-19', 'date_end': '2026-06-19',
                   'windows': [{'weekdays': [4], 'start_time': '23:00', 'end_time': '02:00'}]}})
    registry = EventRegistry.from_records([value])
    assert state(registry, '2026-06-20T01:59:00+03:00')['entrance_state'] == 'closed'
    assert state(registry, '2026-06-20T02:00:00+03:00')['entrance_state'] == 'open'


def test_vestibule_closure_does_not_close_another_entrance():
    registry = EventRegistry.from_records([record(scope={'level': 'vestibule', 'station_ids': ['L1_10'],
                                                        'vestibule_ids': ['V12']})])
    at = '2026-07-01T08:30:00+03:00'
    assert registry.state_at('L1_10', at, at, vestibule_id='V12')['entrance_state'] == 'closed'
    assert registry.state_at('L1_10', at, at, vestibule_id='V13')['entrance_state'] == 'open'
    assert registry.state_at('L1_10', at, at)['entrance_state'] == 'restricted'


def test_line_scope_and_repair_without_closure_are_isolated():
    registry = EventRegistry.from_records([record(line_id='L2', scope={'level': 'line'},
                entrance_state='closed'), record(event_id='repair', version_id='repair:v1',
                kind='repair_without_station_closure', entrance_state='open')])
    assert state(registry)['event_ids'] == ['repair']
    assert not state(registry)['demand_censored']


def test_conflicting_plan_and_observation_are_preserved():
    value = record(conflicts=['flow_time_differs_from_plan'], timing={
        'planned': {'start': '2026-05-09T11:30:00+03:00', 'end': '2026-05-09T15:30:00+03:00'},
        'observed': {'start': '2026-05-09T06:30:00+03:00', 'end': '2026-05-09T13:30:00+03:00',
                     'knowledge_at': '2026-10-09T00:00:00+03:00'}})
    registry = EventRegistry.from_records([value])
    label = registry.label_at('L1_05', '2026-05-09T14:00:00+03:00')
    assert label['entrance_state'] == 'unknown'
    assert label['demand_censored']
    assert 'flow_time_differs_from_plan' in label['conflicts']


def test_replay_does_not_expose_later_audit_conflicts():
    value = record(ingested_at='2026-10-09T00:00:00+03:00',
        conflicts=['flow_time_differs_from_plan'], timing={
            'planned': {'start': '2026-07-01T08:00:00+03:00', 'end': '2026-07-01T09:00:00+03:00'},
            'observed': {'start': '2026-07-01T08:15:00+03:00', 'end': '2026-07-01T08:45:00+03:00',
                         'knowledge_at': '2026-10-09T00:00:00+03:00'}})
    registry = EventRegistry.from_records([value])
    assert state(registry, knowledge_mode='publication_replay')['conflicts'] == []
    assert registry.label_at('L1_05', '2026-07-01T08:30:00+03:00')['conflicts']
    assert state(registry, knowledge_mode='publication_replay',
                 as_of='2026-10-09T01:00:00+03:00')['conflicts']


def test_observations_only_override_their_supported_dates_and_time_windows():
    value = record(conflicts=['flow_time_differs_from_plan'], timing={
        'planned': {'date_start': '2026-06-18', 'date_end': '2026-08-28', 'windows': [
            {'weekdays': [0, 1, 2, 3, 4], 'start_time': '08:10', 'end_time': '09:15'},
            {'weekdays': [0, 1, 2, 3, 4], 'start_time': '16:40', 'end_time': '19:00', 'entrance_state': 'restricted'}]},
        'observed': {'date_start': '2026-07-01', 'date_end': '2026-07-31', 'windows': [
            {'weekdays': [0, 1, 2, 3, 4], 'start_time': '08:15', 'end_time': '09:00'}],
            'comparison_windows': [{'weekdays': [0, 1, 2, 3, 4], 'start_time': '08:10', 'end_time': '09:15'}]}})
    registry = EventRegistry.from_records([value])
    assert registry.label_at('L1_05', '2026-08-03T08:30:00+03:00')['entrance_state'] == 'closed'
    assert registry.label_at('L1_05', '2026-07-01T17:30:00+03:00')['entrance_state'] == 'restricted'
    assert registry.label_at('L1_05', '2026-07-01T09:10:00+03:00')['entrance_state'] == 'unknown'


def test_uncertain_continuation_keeps_timing_and_access_unknown():
    registry = EventRegistry.from_records([record(timing={'observed': {
        'date_start': '2026-02-28', 'date_end': '2026-03-18', 'uncertain': True,
        'windows': [{'weekdays': [0, 1, 2, 3, 4], 'start_time': '08:10', 'end_time': '09:20'}]}})])
    label = registry.label_at('L1_05', '2026-03-17T08:30:00+03:00')
    assert label['entrance_state'] == 'unknown'
    assert label['unknown_timing'] and label['demand_censored']


def test_unknown_end_and_restoration_report_do_not_invent_exact_duration():
    value = record(kind='train_disruption', entrance_state='open', movement_state='disrupted',
        timing={'observed': {'start': '2026-07-17T17:42:00+03:00', 'end': None,
                            'date_end': '2026-07-17',
                            'restoration_report_at': '2026-07-17T18:41:00+03:00'}})
    registry = EventRegistry.from_records([value])
    label = registry.label_at('L1_05', '2026-07-17T18:00:00+03:00')
    assert label['movement_state'] == 'unknown'
    assert label['unknown_timing']
    assert registry.label_at('L1_05', '2026-07-17T18:42:00+03:00')['movement_state'] == 'normal'
    assert registry.records[0]['timing']['observed']['end'] is None
    label = registry.label_during('L1_05', '2026-07-17T17:42:00+03:00', '2026-07-17T18:30:00+03:00')
    assert label['movement_state'] == 'unknown' and label['unknown_timing']


def test_date_only_incident_and_candidate_are_not_confirmed_disruptions():
    registry = EventRegistry.from_records([record(kind='passenger_incident', entrance_state='open',
        movement_state='unknown', timing={'observed': {'date_start': '2026-02-22',
                                                     'date_end': '2026-02-22', 'precision': 'date'}}),
        record(event_id='candidate', version_id='candidate:v1', confidence='candidate',
               kind='unconfirmed_flow_anomaly', entrance_state='unknown', movement_state='unknown')])
    label = registry.label_at('L1_05', '2026-02-22T15:00:00+03:00')
    assert label['movement_state'] == 'unknown'
    assert label['unknown_timing']
    label = registry.label_at('L1_05', '2026-07-01T08:30:00+03:00')
    assert label['movement_state'] == 'normal'
    assert label['candidate_event_ids'] == ['candidate']


@pytest.mark.parametrize('changes', [
    {'published_at': '2026-06-16'}, {'ingested_at': '2026-06-16T10:05:00'},
    {'scope': {'level': 'station', 'station_ids': ['L2_01']}},
    {'scope': {'level': 'vestibule', 'station_ids': ['L1_05'], 'vestibule_ids': ['V13']}},
    {'confidence': 'probably'}, {'version': 0},
    {'timing': {'planned': {'start': '2026-07-01T08:00:00', 'end': None}}},
    {'timing': {'planned': {'date_start': '2026-08-29', 'date_end': '2026-08-28'}}},
    {'timing': {'planned': {'date_start': '2026-08-28', 'windows': [
        {'weekdays': [7], 'start_time': '08:10', 'end_time': '09:15'}]}}},
])
def test_invalid_registry_records_are_rejected(changes):
    with pytest.raises(ValueError):
        EventRegistry.from_records([record(**changes)])


def test_bad_query_timestamp_and_duplicate_version_rejected():
    registry = EventRegistry.from_records([record()])
    with pytest.raises(ValueError):
        state(registry, '2026-07-01T08:30:00')
    with pytest.raises(ValueError):
        EventRegistry.from_records([record(), record()])


def test_interval_overlap_detects_brief_closure_and_respects_right_boundary():
    registry = EventRegistry.from_records([record(timing={'planned': {
        'start': '2026-07-01T08:07:00+03:00', 'end': '2026-07-01T08:10:00+03:00'}})])
    value = registry.state_during('L1_05', '2026-07-01T08:00:00+03:00',
        '2026-07-01T08:15:00+03:00', '2026-07-01T08:15:00+03:00')
    assert value['demand_censored'] and value['censored_seconds'] == 180
    assert value['interval_mixed_regimes']
    assert not registry.label_during('L1_05', '2026-07-01T08:00:00+03:00',
        '2026-07-01T08:07:00+03:00')['demand_censored']


def test_interval_overlap_handles_recurring_window_inside_quarter():
    registry = EventRegistry.from_records([record()])
    value = registry.label_during('L1_05', '2026-07-01T08:00:00+03:00', '2026-07-01T08:15:00+03:00')
    assert value['entrance_state'] == 'closed'
    assert value['censored_seconds'] == 300
    with pytest.raises(ValueError):
        registry.label_during('L1_05', '2026-07-01T08:15:00+03:00', '2026-07-01T08:15:00+03:00')


def test_curated_registry_preserves_source_contract():
    registry = EventRegistry.load()
    assert {r['event_id'] for r in registry.records} >= {'L1_0831', 'L1_NARV_REPAIR', 'L1_0509', 'L2_TECH_REPAIR'}
    at = '2026-07-01T08:30:00+03:00'
    assert registry.state_at('Нарвская', at, at)['event_ids'] == []
    assert registry.state_at('Нарвская', at, at, knowledge_mode='publication_replay')['entrance_state'] == 'closed'
    assert registry.label_at('L1_07', '2026-06-14T14:00:00+03:00')['entrance_state'] == 'open'
    assert registry.label_at('L1_10', '2026-05-09T14:00:00+03:00', vestibule_id='V12')['conflicts']
