import copy
import json

import pytest

from metro44_run import public_decision


def decision():
    return dict(candidate_table=[dict(candidate_id='COST_SOLVER', service_passed=False, trials=[dict(
        scenario='S1', policy_exposure={'admissible':True,'global':{'max_load':150},'windows':[]},
        headways=dict(passed=True, violations=[],minimum_seconds=113,maximum_seconds=480),
        mature_cohort=dict(complete=True), initial_obligations=dict(complete=True,unboarded=0),
        verification_scope=dict(headway_end_second=65400, observation_end_second=72000),
        peak_boarding_service=dict(version='unique-peak-first-next-train-v1', status='failed', passed=False,
            missed_share_limit=.10, next_train_required=True, cohort_start_second=36000,
            cohort_end_second=42300, observation_end_second=43200,
            aggregate=dict(missed_first_count=110, first_opportunity_count=1000,
                missed_first_share=.11, missed_second_count=2, status='failed'),
            intervals=[dict(interval_start='2026-09-30T17:00:00+03:00', status='failed',
                missed_first_count=110, first_opportunity_count=1000, missed_second_count=2)],
            cohorts=[dict(station_id='L1_01', direction=1, status='failed')]))])])


def test_public_log_preserves_complete_service_rejection_evidence():
    source=decision(); before=copy.deepcopy(source)
    saved=json.loads(json.dumps(public_decision(source)))
    original=source['candidate_table'][0]['trials'][0]
    trial=saved['candidate_table'][0]['trials'][0]
    assert trial['peak_boarding_service']==original['peak_boarding_service']
    assert trial['verification_scope']==original['verification_scope']
    assert source==before


def test_retained_service_evidence_has_no_mutable_aliases():
    source=decision(); trial=public_decision(source)['candidate_table'][0]['trials'][0]
    trial['peak_boarding_service']['intervals'][0]['status']='passed'
    trial['peak_boarding_service']['aggregate']['missed_second_count']=0
    trial['verification_scope']['headway_end_second']=72000
    original=source['candidate_table'][0]['trials'][0]
    assert original['peak_boarding_service']['intervals'][0]['status']=='failed'
    assert original['peak_boarding_service']['aggregate']['missed_second_count']==2
    assert original['verification_scope']['headway_end_second']==65400


@pytest.mark.parametrize('missing',[('peak_boarding_service',),('verification_scope',),
    ('peak_boarding_service','verification_scope')])
def test_optional_evidence_preserves_legacy_absence(missing):
    source=decision(); original=source['candidate_table'][0]['trials'][0]
    for key in missing: original.pop(key)
    trial=public_decision(source)['candidate_table'][0]['trials'][0]
    for key in missing: assert key not in trial
    for key in {'peak_boarding_service','verification_scope'}-set(missing):
        assert trial[key]==original[key]
