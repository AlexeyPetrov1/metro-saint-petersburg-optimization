import pytest

from metro47_withdrawal_experiment import run


def test_normalization_rebinds_checkpoint_through_the_verified_branch():
    from metro47_withdrawal_experiment import normalize_shared
    calls = []
    class Runner:
        def __init__(self, value, *args): self.value = value
        def branch_checkpoint(self, state, source):
            calls.append((source.value, self.value, state))
            return dict(state, context_id='normalized')
    old = dict(plans=[dict(plan_id='P')], version='original')
    changed = dict(plans=[dict(plan_id='P')], version='normalized')
    shared = dict(data=old, groups=[], profiles={}, state=dict(second=3600, context_id='original'))
    result, audit = normalize_shared(shared, lambda *args: (changed, {'passed': True}), Runner)
    assert calls == [(old, changed, shared['state'])]
    assert result['state']['context_id'] == 'normalized'
    assert shared['state']['context_id'] == 'original'
    assert result['groups'] == shared['groups'] and audit['passed']


def test_previous_run_is_preserved(tmp_path):
    receipt = tmp_path / 'shared_contract.json'
    receipt.write_text('existing run', encoding='utf-8')
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert receipt.read_text(encoding='utf-8') == 'existing run'


def test_unavailable_oracle_cannot_be_run(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path / 'new', arm='ORACLE')
    assert not (tmp_path / 'new').exists()
