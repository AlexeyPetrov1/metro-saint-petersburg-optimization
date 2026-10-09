import json
from pathlib import Path

import pytest

from metro51_run import atomic_json, control_times, filtered_groups, pilot_contract, resume_matches


def test_pilot_control_times_and_followup_are_explicit():
    contract = pilot_contract('BALANCED', 'pilot')
    assert control_times(contract['cohort_end_second'])[-1] == 15300
    assert contract['cohort_end_second'] == 16200
    assert contract['observation_end_second'] == 23400
    assert contract['full_day'] is False


def test_actual_groups_preserve_platform_tail_after_turnstile_boundary():
    groups = [dict(group_id='a',turnstile_second=16199,platform_second=16319),
              dict(group_id='b',turnstile_second=16200,platform_second=16320)]
    assert [g['group_id'] for g in filtered_groups(groups,16200)] == ['a']


def test_atomic_checkpoint_is_valid_json(tmp_path):
    path = tmp_path/'state.json'
    atomic_json(path, {'second':900, 'state':{'valid':True}})
    assert json.loads(path.read_text(encoding='utf-8'))['second'] == 900
    assert not list(tmp_path.glob('*.tmp'))


def test_resume_rejects_model_or_rule_changes():
    saved = {'arm':'BALANCED','source_hashes':{'model':'abc'},'stage':'pilot','seed':7}
    assert resume_matches(saved,dict(saved))
    assert not resume_matches(saved,{**saved,'source_hashes':{'model':'def'}})
    assert not resume_matches(saved,{**saved,'seed':8})
