import copy

import pytest

from metro44_daily_resource_audit import mileage_ledger,compare_ledgers


def fixture(km=10):
    trains=[dict(train_id=f'B{i:03}',train_type='Baltiets')for i in range(37)]
    trains += [dict(train_id=f'Y{i:03}',train_type='Yubileyny')for i in range(31)]
    work=[dict(train_id=t['train_id'],train_type=t['train_type'],train_km=str(km),
               passenger_train_km=str(km),nonpassenger_train_km='0',car_km=str(8*km),cars=8)for t in trains]
    return trains,work


def test_typed_resource_costs_and_fixed68_are_calculated_independently():
    trains,work=fixture();before=copy.deepcopy((trains,work))
    ledger=mileage_ledger(trains,work)
    assert ledger['train_km']==680 and ledger['km_per_fixed68']==10
    assert ledger['to1_resource_rub']==5698
    assert ledger['per_train'][0]['to1_rub_per_km']==9.2
    assert ledger['per_train'][-1]['to1_rub_per_km']==7.4
    assert (trains,work)==before and ledger['cash_savings_rub']is None


def test_inactive_trains_remain_in_the_denominator():
    trains,work=fixture(0)
    work[0].update(train_km='10',passenger_train_km='10',car_km='80')
    ledger=mileage_ledger(trains,work)
    assert ledger['active_trains']==1 and ledger['inactive_trains']==67
    assert ledger['km_per_fixed68']==10/68 and ledger['to1_resource_rub']==92


def test_missing_identity_or_car_km_substitution_is_rejected():
    trains,work=fixture()
    with pytest.raises(ValueError,match='68'):mileage_ledger(trains,work[:-1])
    work[0]['train_km']='80'
    with pytest.raises(ValueError,match='passenger'):mileage_ledger(trains,work)


def test_unsafe_reference_or_missing_wait_budget_cannot_accept_resource_gain():
    trains,work=fixture();reference=mileage_ledger(trains,work)
    candidate=mileage_ledger(*fixture(8))
    result=compare_ledgers(reference,candidate,reference_safe=False,candidate_safe=True,
        reference_wait=(80,150),candidate_wait=(82,155),wait_budget=(90,180))
    assert result['observed_reduction_train_km']==136
    assert result['observed_reduction_km_per_fixed68']==2
    assert not result['resource_comparison_eligible'] and result['accepted_saving_train_km']is None
    assert result['cash_savings_rub']is None
    result=compare_ledgers(reference,candidate,reference_safe=True,candidate_safe=True,
        reference_wait=(80,150),candidate_wait=(82,155),wait_budget=None)
    assert not result['resource_comparison_eligible']


def test_declared_budget_is_checked_for_both_arms_and_strict_wait_is_separate():
    reference=mileage_ledger(*fixture());candidate=mileage_ledger(*fixture(8))
    result=compare_ledgers(reference,candidate,reference_safe=True,candidate_safe=True,
        reference_wait=(80,150),candidate_wait=(82,155),wait_budget=(90,180))
    assert result['resource_comparison_eligible'] and result['accepted_saving_train_km']==136
    assert not result['strict_no_wait_regression_passed']
    result=compare_ledgers(reference,candidate,reference_safe=True,candidate_safe=True,
        reference_wait=(91,150),candidate_wait=(82,155),wait_budget=(90,180))
    assert not result['reference_wait_budget_passed'] and not result['resource_comparison_eligible']


def test_same_fleet_and_types_are_required_for_a_paired_financial_ledger():
    reference=mileage_ledger(*fixture());candidate=mileage_ledger(*fixture(8))
    candidate['per_train'][0]['train_type']='yubileyny'
    with pytest.raises(ValueError,match='fleet'):
        compare_ledgers(reference,candidate,reference_safe=True,candidate_safe=True,
            reference_wait=(80,150),candidate_wait=(80,150),wait_budget=(90,180))


@pytest.mark.parametrize('budget',[(90,),(90,None)])
def test_both_wait_budget_components_must_be_declared(budget):
    result=compare_ledgers(mileage_ledger(*fixture()),mileage_ledger(*fixture(8)),
        reference_safe=True,candidate_safe=True,reference_wait=(80,150),candidate_wait=(80,150),wait_budget=budget)
    assert not result['resource_comparison_eligible'] and result['accepted_saving_train_km']is None
