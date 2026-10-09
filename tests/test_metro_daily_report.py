import copy

import pytest

from metro_daily_report import experiment_conclusion, validate_pairing


def metrics(safe, km, bad=0):
    return dict(actual_service_admissible=safe,train_km=km,mean_wait_seconds=70,
        policy_exposure=dict(windows=[dict(status='fail')]*bad))


def test_lower_mileage_with_violations_is_not_a_safe_win():
    answer=experiment_conclusion(metrics(False,1000,2),metrics(False,800,1))
    assert 'не подтверждена' in answer


def test_safe_candidate_can_be_identified_against_unsafe_baseline():
    answer=experiment_conclusion(metrics(False,1000,2),metrics(True,1100))
    assert 'устранены' in answer
    assert 'экономии пробега нет' in answer


def test_pairing_requires_the_same_truth_checkpoint_and_every_decision():
    common=dict(groups_digest='g',checkpoint_digest='c')
    baseline=dict(strategy='BASELINE',decisions=[],**common)
    decisions=[dict(decision_second=t) for t in range(3600,64800,900)]
    arms=[baseline]+[dict(strategy=s,decisions=copy.deepcopy(decisions),**common) for s in ('MEDIAN','ML','ORACLE')]
    validate_pairing(arms)
    arms[2]['decisions'].pop()
    with pytest.raises(ValueError,match='decision'):validate_pairing(arms)
    arms[2]['decisions']=copy.deepcopy(decisions)
    arms[2]['groups_digest']='different people'
    with pytest.raises(ValueError,match='passengers'):validate_pairing(arms)
