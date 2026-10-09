from copy import deepcopy

import pytest

from simulation import simulate


def cycle(entry=540, depot='DEPOT_AVTOVO', release=720):
    return dict(schema_version='1.0', source='confirmed inclusive depot release test',
        period_start='2026-09-30T10:00:00+03:00', period_end='2026-09-30T11:00:00+03:00',
        stations=[dict(station_id='S',kind='line',stop_seconds=0,turnaround_seconds=180),
                  dict(station_id=depot,kind='depot',stop_seconds=0,turnaround_seconds=0)],
        tracks=[dict(track_id='EXIT_S',from_station='S',to_station=depot,length_km='1',travel_seconds=60),
                dict(track_id='ENTRY_S',from_station=depot,to_station='S',length_km='1',travel_seconds=60)],
        trains=[dict(train_id='T1',train_type='Baltiets',cars=8,location='S',state='standby',ready_after_seconds=0)],
        plans=[dict(plan_id='test',trips=[
            dict(trip_id='out',train_id='T1',departure_second=0,route=['EXIT_S'],passenger=False),
            dict(trip_id='back',train_id='T1',departure_second=entry,route=['ENTRY_S'],passenger=False)])],
        headway_policy={'parameters':{'depot_release_s':{'value':{'Avtovo':release,'Northern':780}}}})


@pytest.mark.parametrize('depot,release,entry', [('DEPOT_AVTOVO',720,540),('DEPOT_NORTHERN',780,600)])
def test_confirmed_depot_cycle_includes_travel_and_terminal_turn(depot,release,entry):
    data = cycle(entry,depot,release); original = deepcopy(data)
    result = simulate(data,'test')
    assert result['obligations_closed'] is True
    assert next(e['second'] for e in result['events']
                if e['trip_id'] == 'back' and e['event_type'] == 'turnaround_complete') == 60+release
    assert data == original


@pytest.mark.parametrize('depot,release,entry', [('DEPOT_AVTOVO',720,539),('DEPOT_NORTHERN',780,599)])
def test_one_second_short_of_depot_preparation_rejected(depot,release,entry):
    with pytest.raises(ValueError, match='depot preparation'):
        simulate(cycle(entry,depot,release),'test')


def test_unknown_depot_release_keeps_reversal_forbidden():
    data = cycle(); data.pop('headway_policy')
    with pytest.raises(ValueError, match='reversal needs'):
        simulate(data,'test')


def test_line_reversal_without_turnaround_stays_forbidden():
    data = cycle(); data['stations'][1]['kind'] = 'line'
    with pytest.raises(ValueError, match='reversal needs'):
        simulate(data,'test')
