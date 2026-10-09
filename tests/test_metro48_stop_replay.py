from metro48_stop_replay import stop_event


def test_stop_is_selected_from_real_station_arrival_after_warmup():
    data={'plans':[{'trips':[{'trip_id':'trip','route':['01_down']}]}]}
    motion={'events':[
        dict(event_type='arrival',location='L1_12',second=3500,train_id='B001',trip_id='trip'),
        dict(event_type='arrival',location='L1_12',second=3682,train_id='B001',trip_id='trip')]}
    event=stop_event(data,motion)
    assert event['detected_second']==3683 and event['direction']=='down'
    assert 'actual_end_second' not in event
