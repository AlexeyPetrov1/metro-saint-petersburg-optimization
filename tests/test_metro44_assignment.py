import pytest

from metro44_assignment import solve_fixed_assignments


def train(name, kind, at='A', ready=0):
    return dict(train_id=name, train_type=kind, location=at, ready_second=ready)


def job(name, departure, ready, origin='A', destination='B', km=10):
    return dict(job_id=name, departure_second=departure, ready_second=ready,
                origin=origin, destination=destination, km=km)


def test_cheaper_ready_train_gets_the_same_fixed_service():
    result = solve_fixed_assignments([job('j', 100, 200)], [train('b','Baltiets'), train('y','Yubileyny')])
    assert result['assignments'] == {'j': 'y'}
    assert result['to1_resource_equivalent_rub'] == pytest.approx(74)
    assert result['train_km'] == 10
    assert result['release_authorized'] is False


def test_a_train_must_reach_the_correct_terminal_before_reuse():
    jobs = [job('up',100,200),job('down',250,350,'B','A')]
    result = solve_fixed_assignments(jobs,[train('y','Yubileyny')])
    assert result['assignments'] == {'up':'y','down':'y'}
    bad = solve_fixed_assignments([job('up',100,200),job('down',199,300,'B','A')],[train('y','Yubileyny')])
    assert bad['status'] == 'infeasible'
    assert bad['assignments'] == {}


def test_unready_cheap_train_cannot_replace_ready_train():
    result = solve_fixed_assignments([job('j',100,200)],[train('b','Baltiets'),train('y','Yubileyny',ready=101)])
    assert result['assignments'] == {'j':'b'}


def test_parallel_jobs_cannot_share_one_vehicle():
    result = solve_fixed_assignments([job('a',100,200),job('b',101,201)],[train('y','Yubileyny')])
    assert result['status'] == 'infeasible'


def test_duplicate_jobs_and_negative_distance_are_rejected():
    with pytest.raises(ValueError): solve_fixed_assignments([job('j',100,200),job('j',100,200)],[train('y','Yubileyny')])
    with pytest.raises(ValueError): solve_fixed_assignments([job('j',100,200,km=-1)],[train('y','Yubileyny')])


def test_reserve_can_replace_train_only_after_actual_depot_arrival():
    b, y = train('b','Baltiets',ready=50), train('y','Yubileyny')
    b.update(on_line=True, exit_seconds=20)
    y.update(on_line=False,entry_lead_seconds=30,exit_seconds=20,entry_km=1,exit_km=1)
    result = solve_fixed_assignments([job('j',100,200)],[b,y],line_train_limit=1)
    assert result['assignments'] == {'j':'b'}
    b['ready_second'] = 49
    result = solve_fixed_assignments([job('j',100,200)],[b,y],line_train_limit=1)
    assert result['assignments'] == {'j':'y'}
    assert result['train_km'] == 12
    assert result['to1_resource_equivalent_rub'] == pytest.approx(88.8)
    assert result['reserve_starts'] == {'y':70}
