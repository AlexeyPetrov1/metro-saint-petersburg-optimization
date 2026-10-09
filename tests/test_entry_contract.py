import copy

import pytest

from forecasting import hourly_median, validate_bundle
from passenger_groups import GroupedPassengerSimulation
from tests.test_actions import inputs
from tests.test_forecasting import bundle, history


@pytest.mark.parametrize('target', ['turnstile_exits', 'train_alightings', 'station_occupancy'])
def test_exit_or_stock_forecasts_cannot_create_entries(target):
    forecast = copy.deepcopy(bundle())
    forecast['target'] = target
    with pytest.raises(ValueError):
        validate_bundle(forecast, ['A', 'B'], '2026-09-01T08:00:00+03:00')


def test_exit_history_cannot_train_the_entry_median():
    with pytest.raises(ValueError, match='entries'):
        hourly_median(history().rename(columns={'entry_count': 'exit_count'}),
            ['A', 'B'], '2026-09-01T08:00:00+03:00')


def test_entries_approach_board_and_exit_at_their_od_destination():
    data, _, profiles, _ = inputs()
    groups = [dict(group_id='G', origin='A', turnstile_second=10, platform_second=130,
        count=5, destinations=[dict(station_id='B', count=5)])]
    runner = GroupedPassengerSimulation(data, 'line1-baseline', groups, profiles)
    early = runner.run(end_second=60)['metrics']['cumulative']
    assert early['entered'] == 5 and early['platform_arrived'] == early['completed'] == 0
    final = runner.run()['metrics']['cumulative']
    assert final['entered'] == final['boarded'] == final['completed'] == 5
