import copy
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / 'data/processed/metro44_catalog_cont6h_20261009/campaign.py'
spec = importlib.util.spec_from_file_location('catalog_campaign', PATH)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def rows():
    exposure = dict(admissible=True, global_={})
    exposure = {'admissible': True, 'global': dict(max_load=75, exceptional_time_share=0,
        exceptional_car_seconds=0, passenger_car_seconds=1000, hard_limit_car_seconds=0), 'windows': []}
    def candidate(name, mean, p95, km):
        return dict(candidate_id=name, status='evaluated', passengers_checked=True, physical_passed=True,
            service_passed=True, metrics=dict(max_car_load=75, waiting_passenger_seconds=1000),
            committed_per_train=[dict(train_id=f'T{i}', train_type='Baltiets' if i < 37 else 'Yubileyny', km=km)
                for i in range(68)], trials=[dict(future_exposure=exposure, headways=dict(passed=True),
                mature_cohort=dict(complete=True, arrived=100, unboarded=0, mean_wait_seconds=mean,
                                  p95_wait_seconds=p95), initial_obligations=dict(complete=True, unboarded=0))])
    return [candidate('KEEP', 74.797, 139, 10), candidate('BRIDGE', 74.824, 143, 8)]


def test_raw_strict_counterfactual_preserves_keep_and_both_budgets():
    decision = dict(candidate_table=rows(), search_complete=True)
    before = copy.deepcopy(decision)
    selected = campaign.strict_counterfactual(decision)
    assert selected['selected_candidate_id'] == 'KEEP'
    assert selected['waiting_budgets'] == dict(mean_seconds=74.797, p95_seconds=139)
    assert decision == before


def test_strict_diagnostic_rejects_compact_schema_instead_of_silently_losing_budgets():
    decision = dict(candidate_table=rows(), search_complete=True)
    for row in decision['candidate_table']:
        row['trials'][0].pop('mature_cohort')
    with pytest.raises(ValueError, match='raw'):
        campaign.strict_counterfactual(decision)
