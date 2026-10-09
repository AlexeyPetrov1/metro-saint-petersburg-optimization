import pandas as pd

from metro48_anomaly_replay import inject_entries, known_closure


def test_injected_scenario_keeps_original_and_unaffected_past():
    start=pd.date_range('2026-09-30T09:00:00+03:00',periods=8,freq='15min')
    frame=pd.DataFrame(dict(station_id=['L1_01']*8,interval_start=start,entry_count=[100]*8))
    changed=inject_entries(frame,['L1_01'],'2026-09-30T09:45:00+03:00','2026-09-30T10:15:00+03:00',1.5)
    assert frame.entry_count.tolist()==[100]*8
    assert changed.entry_count.tolist()==[100,100,100,150,150,100,100,100]


def test_unknown_future_closure_cannot_change_prediction():
    bundle={'source_versions':{},'included_event_ids':[], 'rows':[
        dict(station_id='L1_01',interval_start='2026-09-30T09:45:00+03:00',
             interval_end='2026-09-30T10:00:00+03:00',predicted_count=100)]}
    closure=dict(id='closure',station_id='L1_01',available_at='2026-09-30T09:37:13+03:00',
        start='2026-09-30T09:45:00+03:00',end='2026-09-30T10:15:00+03:00')
    assert known_closure(bundle,closure,'2026-09-30T09:30:00+03:00')['rows'][0]['predicted_count']==100
    assert known_closure(bundle,closure,'2026-09-30T09:45:00+03:00')['rows'][0]['predicted_count']==0
    assert bundle['rows'][0]['predicted_count']==100
