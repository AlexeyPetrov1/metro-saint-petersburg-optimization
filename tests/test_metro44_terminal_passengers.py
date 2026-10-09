import copy

import pytest

from metro44_terminal_passengers import require_eligible


def eligible():
    return dict(status='movement_checked', passenger_validation_eligible=True,
        source_ledger_preserved=True, audit=dict(movement_passed=True, peak_trains_on_line=50),
        terminal_total=dict(passed=True))


def test_only_independently_physical_candidate_can_receive_passengers():
    require_eligible(eligible())
    for field in ('source_ledger_preserved', 'passenger_validation_eligible'):
        rejected = copy.deepcopy(eligible())
        rejected[field] = False
        with pytest.raises(ValueError):
            require_eligible(rejected)


def test_line_capacity_guard_is_not_relaxed():
    rejected = eligible()
    rejected['audit']['peak_trains_on_line'] = 54
    with pytest.raises(ValueError):
        require_eligible(rejected)
