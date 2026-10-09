from verify_goal4_2 import verify_work


def test_arrival_ledger_excludes_start_and_includes_end():
    data = dict(tracks=[dict(track_id='P', length_km='2')],
        trains=[dict(train_id='T', state='waiting', cars=2)],
        plans=[dict(trips=[dict(trip_id='H', passenger=False)])])
    motion = dict(events=[dict(train_id='T', trip_id='H', location='P',
        event_type=kind, second=second) for kind, second in
        [('departure', 1), ('arrival', 4), ('departure', 5), ('arrival', 8),
         ('departure', 9), ('arrival', 12)]])
    resources = dict(passenger_train_km=0, passenger_car_km=0,
        nonpassenger_train_km=2, nonpassenger_car_km=4, state_seconds=dict(waiting=4))
    verify_work(data, motion, resources, start=4, end=8)
