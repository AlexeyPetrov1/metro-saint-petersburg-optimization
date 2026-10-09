import pytest

from passenger_groups import allocate_preferred


def test_preference_conserves_passengers_and_keeps_physical_caps():
    assert allocate_preferred(8,[0,0],[10,10],[2,1]) == [5,3]
    assert allocate_preferred(10,[9,0],[10,10],[2,1]) == [1,9]


@pytest.mark.parametrize("weights", [[0,1],[True,1],[1],[-1,2]])
def test_invalid_preferences_are_rejected(weights):
    with pytest.raises(ValueError): allocate_preferred(8,[0,0],[10,10],weights)

