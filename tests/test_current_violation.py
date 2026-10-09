import pandas as pd

import rolling
from forecasting import hourly_median
from passenger_groups import GroupedPassengerSimulation
from tests.test_actions import inputs
from tests.test_forecasting import history


def test_committed_overload_rejects_candidates_without_future_seed_search(monkeypatch):
    data, registry, profiles, groups = inputs()
    for cars in profiles.values(): cars[0]['normal_threshold'] = 2
    source = GroupedPassengerSimulation(data, 'line1-baseline', groups, profiles)
    snapshot = source.run(end_second=10)['checkpoint']
    frame = history()
    for column in ('interval_start', 'interval_end'): frame[column] -= pd.Timedelta(hours=2)
    forecast = hourly_median(frame, ['A', 'B'], '2026-09-01T06:00:00+03:00')
    def unnecessary(*args, **kwargs): raise AssertionError('committed violation is already sufficient')
    monkeypatch.setattr(rolling, 'evaluate_candidates', unnecessary)
    decision = rolling.decide_snapshot(data, {'S1': source}, {'S1': snapshot}, profiles,
        registry, forecast, [('KEEP', None), ('ADD', 'R01')], 'mild', list(range(10)), early_rejection=True)
    assert decision['selection']['status'] == 'no_feasible_candidate'
    assert decision['trials'] == []
    assert decision['committed_violation']['load'] == 4
    assert all(row['tested_seeds'] == [] for row in decision['candidate_table'])
