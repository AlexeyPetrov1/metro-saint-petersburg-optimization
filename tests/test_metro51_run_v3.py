import pytest

from metro51_run import control_times, pilot_contract, resume_matches


def test_early_causal_control_is_explicit_in_the_contract():
    contract = pilot_contract('BALANCED', 'pilot', control_start_second=900)
    assert contract['control_start_second'] == 900
    times = control_times(contract['cohort_end_second'], start_second=contract['control_start_second'])
    assert times[0] == 900
    assert times[-1] == 15300
    assert all(b-a == 900 for a, b in zip(times, times[1:]))


@pytest.mark.parametrize('start', [0, 901, True, 16200])
def test_invalid_control_starts_fail_before_creating_output(start):
    with pytest.raises(ValueError):
        pilot_contract('BALANCED', 'pilot', control_start_second=start)


def test_resume_cannot_change_the_start_of_dispatch_control():
    contract = dict(arm='BALANCED', stage='pilot', seed=7, source_hashes={}, control_start_second=900)
    assert not resume_matches(contract, dict(contract, control_start_second=3600))
