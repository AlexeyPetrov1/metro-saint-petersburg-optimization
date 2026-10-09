from types import SimpleNamespace

from scripts.metro51_quick_review import selected_boarding


def inputs(count=10,second=1000,boards=None):
    groups=[dict(group_id='G',origin='A',platform_second=800,count=count,
        destinations=[dict(station_id='B',count=count)])]
    rows=boards if boards is not None else [dict(group_id='G',destination='B',second=950,count=count,motion_event_id='E')]
    return groups,dict(second=second,board_events=rows)


def test_late_boarding_is_included_for_the_selected_unique_cohort():
    groups,state=inputs()
    result=selected_boarding(groups,state,0,900)
    assert result['complete'] and result['passenger_count']==result['boarded']==10
    assert result['late_boarding_count']==10 and result['waiting']['mean_wait_seconds']==150


def test_incomplete_observation_is_not_reported_as_complete_or_zero_wait():
    groups,state=inputs(second=900,boards=[])
    result=selected_boarding(groups,state,0,900)
    assert not result['complete'] and result['unboarded']==10
    assert result['waiting']['mean_wait_seconds'] is None


def test_ledger_digest_detects_actual_time_and_count_changes_but_ignores_other_cohorts():
    groups,state=inputs();before=selected_boarding(groups,state,0,900)
    state['board_events'].append(dict(group_id='unselected',second=999,count=999))
    assert selected_boarding(groups,state,0,900)['board_ledger_sha256']==before['board_ledger_sha256']
    state['board_events'][0]['second']=951
    assert selected_boarding(groups,state,0,900)['board_ledger_sha256']!=before['board_ledger_sha256']
