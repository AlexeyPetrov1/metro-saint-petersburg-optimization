from copy import deepcopy

import pytest

from metro48_results import paired_summary


def result(km):
    rows = [dict(train_id=f'T{i}', train_type='Baltiets' if i < 37 else 'Yubileyny', km=km) for i in range(68)]
    return dict(full_day=False, maximum_pipeline_seconds=10,
        metrics=dict(per_train=rows, entered=100, completed=100, queued_at_closure=0,
            approaching_at_closure=0, onboard_at_closure=0, movement_obligations_closed=True,
            actual_service_admissible=True, cohort=dict(arrived=80, boarded=80, unboarded=0,
                complete=True, mean_wait_seconds=90, p95_wait_seconds=180)))


def receipt():
    return dict(accepted=True, full_day=False, exact_current_kernel_journal_parity=True,
        all_20_quarter_prefixes_current_movements_and_begun_services_preserved=True,
        local_recommendation_within_480=True, sources_and_inputs_preserved=True)


def test_resource_saving_keeps_fixed_denominator_and_unknown_cash():
    row = paired_summary(result(100), {'GRU': result(90)}, {'GRU': receipt()})['arms'][0]
    assert row['saving_percent'] == pytest.approx(10)
    assert row['saving_km_per_fixed_train'] == 10
    assert row['mean_km_per_fixed_train'] == 90
    assert row['cash_saving_rub'] is None


def test_failed_audit_or_incomplete_cohort_cannot_be_published_as_saving():
    bad = receipt(); bad['accepted'] = False
    with pytest.raises(ValueError):
        paired_summary(result(100), {'GRU': result(90)}, {'GRU': bad})
    bad = result(90); bad['metrics']['completed'] = 99
    with pytest.raises(ValueError):
        paired_summary(result(100), {'GRU': bad}, {'GRU': receipt()})


def test_day_scope_and_changed_train_type_are_rejected():
    bad = result(90); bad['full_day'] = True
    with pytest.raises(ValueError):
        paired_summary(result(100), {'GRU': bad}, {'GRU': receipt()})
    bad = deepcopy(result(90)); bad['metrics']['per_train'][0]['train_type'] = 'Yubileyny'
    with pytest.raises(ValueError):
        paired_summary(result(100), {'GRU': bad}, {'GRU': receipt()})
