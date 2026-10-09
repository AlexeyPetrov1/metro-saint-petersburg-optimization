from metro44_bridge_view import assemble


def result(od, whole):
    metrics = dict(train_km=100, mean_km_per_fixed_train=100/68,
                   mean_wait_seconds=70, p95_wait_seconds=143, max_car_load=113,
                   actual_service_admissible=True)
    return dict(od=od, metrics=metrics, baseline_metrics=dict(metrics),
                complete_episode_exposure={'admissible': whole},
                terminal_resources={'passed': True})


def original(od):
    journey = dict(arrived=3,completed=3,complete=True,mean_gate_to_exit_seconds=500,
                   p95_gate_to_exit_seconds=700)
    return dict(od=od,journeys={'KEEP':journey,'BRIDGE160':journey},
                window_bounds={'saving_train_km_lower_bound':1,
                               'saving_km_per_fixed_train_lower_bound':1/68,
                               'saving_to1_resource_rub_lower_bound':8})


def test_controlled_pass_never_hides_the_historical_whole_episode_failure():
    data = assemble([result('S3',False)],[original('S3')],[],
                    [{'od':'S3','baseline_whole_episode_admissible':False}])
    rows = data['scenarios'][0]['arms']
    assert all(row['control_admissible'] for row in rows)
    assert not any(row['whole_episode_admissible'] for row in rows)
    assert data['cash_saving_rub'] is None
    assert data['reference_scope'] == 'constant_53'
    assert not data['learned_controller_verified']


def test_overload_inside_the_control_window_cannot_be_marked_safe():
    row = result('S1',True)
    row['metrics']['actual_service_admissible'] = False
    data = assemble([row],[original('S1')],[],
                    [{'od':'S1','baseline_whole_episode_admissible':True}])
    assert not data['scenarios'][0]['arms'][1]['control_admissible']
