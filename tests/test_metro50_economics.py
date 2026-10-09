import copy

import pytest

from metro50_economics import economy_comparison, maintenance_rates


def fleet(km=100, **extra):
    return {'per_train': [dict(train_id=f'{prefix}{i:03}', train_type=kind, km=km, **extra)
            for prefix, kind, count in [('Y', 'Yubileyny', 31), ('B', 'Baltiets', 37)]
            for i in range(1, count + 1)]}


def test_rates_use_latest_intervals_and_train_prices_without_eightfold_multiplier():
    rates = maintenance_rates()
    y, b = rates['Yubileyny'], rates['Baltiets']
    assert y['components']['TO1']['interval_km'] == 12000
    assert y['components']['TO1']['rub_per_train_km'] == pytest.approx(111000 / 12000)
    assert y['without_sr_kr'] == pytest.approx(103.99769345238095)
    assert b['without_sr_kr'] == pytest.approx(113.19441964285715)
    assert y['with_sr_kr'] == pytest.approx(y['without_sr_kr'] + 44000000 / 1920000 + 88000000 / 5100000)
    assert b['with_sr_kr'] == pytest.approx(b['without_sr_kr'] + 44000000 / 1280000 + 88000000 / 5100000)
    assert y['components']['KR']['interval_km'] == 5100000


def test_daily_annual_and_individual_rows_reconcile_with_identical_fleet():
    base, candidate = fleet(100), fleet(80)
    original = copy.deepcopy((base, candidate))
    out = economy_comparison(base, candidate, admissible=True)
    assert (base, candidate) == original
    assert out['fixed_fleet_size'] == 68
    assert out['daily']['reference_km'] == 6800
    assert out['daily']['candidate_km'] == 5440
    assert out['daily']['saved_km'] == 1360
    for column in ('without_sr_kr', 'with_sr_kr'):
        day = out['daily'][column]
        assert day['saving_percent'] == pytest.approx(20)
        assert day['saving_rub'] == pytest.approx(sum(r['daily'][column]['saving_rub'] for r in out['per_train']))
        assert out['annual'][column]['saving_rub'] == pytest.approx(day['saving_rub'] * 250)
        assert day['mean_saving_per_fixed_train_rub'] == pytest.approx(day['saving_rub'] / 68)
        assert day['saving_rub_per_saved_km'] == pytest.approx(day['saving_rub'] / 1360)
        assert out['annual'][column]['reference_cost_rub'] == pytest.approx(day['reference_cost_rub'] * 250)
    assert out['annual']['scenario_only'] is True
    assert all(r['annual']['scenario_only'] for r in out['per_train'])
    assert out['cash_saving_rub'] is None
    assert out['net_profit_effect_rub'] is None
    assert out['accepted_resource_saving'] is True


def test_uncertainty_changes_computed_mileage_not_maintenance_intervals():
    out = economy_comparison(fleet(100), fleet(80))
    rates = maintenance_rates()
    weighted = 31 * rates['Yubileyny']['without_sr_kr'] + 37 * rates['Baltiets']['without_sr_kr']
    bounds = out['daily']['without_sr_kr']['uncertainty']
    assert bounds['saving_rub_min'] == pytest.approx(weighted * (100 * .95 - 80 * 1.05))
    assert bounds['saving_rub_max'] == pytest.approx(weighted * (100 * 1.05 - 80 * .95))
    assert bounds['saving_percent_min'] == pytest.approx(100 * (95 - 84) / 95)
    assert bounds['saving_percent_max'] == pytest.approx(100 * (105 - 76) / 105)
    assert out['uncertainty']['computed_mileage_relative_error'] == .05
    assert out['rates']['Yubileyny']['components']['TO1']['interval_km'] == 12000


def test_capital_price_error_is_shared_between_plans_and_only_capital_changes():
    out = economy_comparison(fleet(100), fleet(80))
    rates = maintenance_rates()
    routine = sum(n * rates[t]['without_sr_kr'] for t, n in [('Yubileyny', 31), ('Baltiets', 37)])
    capital = sum(n * (rates[t]['with_sr_kr'] - rates[t]['without_sr_kr']) for t, n in [('Yubileyny', 31), ('Baltiets', 37)])
    bounds = out['daily']['with_sr_kr']['uncertainty']
    assert bounds['saving_rub_min'] == pytest.approx((routine + .8 * capital) * 11)
    assert bounds['saving_rub_max'] == pytest.approx((routine + 1.2 * capital) * 29)
    assert bounds['reference_cost_rub_min'] == pytest.approx(95 * (routine + .8 * capital))
    equal = economy_comparison(fleet(100), fleet(100))
    assert equal['daily']['with_sr_kr']['saving_rub'] == 0
    assert equal['daily']['with_sr_kr']['uncertainty']['saving_rub_min'] < 0
    assert equal['accepted_resource_saving'] is False


def test_inadmissible_or_negative_saving_is_not_accepted():
    for candidate, admissible in [(fleet(80), False), (fleet(110), True)]:
        out = economy_comparison(fleet(100), candidate, admissible=admissible)
        assert out['accepted_resource_saving'] is False
    assert economy_comparison(fleet(100), fleet(110), admissible=True)['daily']['without_sr_kr']['saving_percent'] == pytest.approx(-10)


def test_shared_sr_kr_uncertainty_handles_opposite_type_mileage_changes():
    candidate = fleet()
    for row in candidate['per_train']:
        row['km'] = 0 if row['train_type'] == 'Yubileyny' else 160
    out = economy_comparison(fleet(100), candidate)
    rates = maintenance_rates()
    # Adverse mileage gives Y +2945 km, B -2701 km: SR negative, KR positive.
    routine = 2945 * rates['Yubileyny']['without_sr_kr'] - 2701 * rates['Baltiets']['without_sr_kr']
    sr = 44000000 * (2945 / 1920000 - 2701 / 1280000)
    kr = 88000000 * (244 / 5100000)
    assert sr < 0 < kr
    assert out['daily']['with_sr_kr']['uncertainty']['saving_rub_min'] == pytest.approx(routine + 1.2 * sr + .8 * kr)


def test_planned_and_executed_mileage_are_distinct_and_missing_plan_is_unknown():
    out = economy_comparison(fleet(100, planned_km=120), fleet(80, planned_km=110))
    assert out['daily']['saved_km'] == 1360
    assert out['planned']['daily']['saved_km'] == 680
    assert out['planned']['daily']['without_sr_kr']['saving_percent'] == pytest.approx(100 / 12)
    assert economy_comparison(fleet(), fleet())['planned'] is None
    wrapped = economy_comparison({'metrics': fleet(100)}, {'metrics': fleet(80)})
    assert wrapped['daily']['saved_km'] == 1360


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'type', 'nan', 'negative'])
def test_incomplete_mismatched_or_invalid_ledgers_are_rejected(mutation):
    candidate = fleet()
    if mutation == 'missing': candidate['per_train'].pop()
    if mutation == 'duplicate': candidate['per_train'][-1] = candidate['per_train'][0].copy()
    if mutation == 'type': candidate['per_train'][0]['train_type'] = 'Baltiets'
    if mutation == 'nan': candidate['per_train'][0]['km'] = float('nan')
    if mutation == 'negative': candidate['per_train'][0]['km'] = -1
    with pytest.raises(ValueError):
        economy_comparison(fleet(), candidate)


def test_small_saving_uncertainty_can_be_negative_and_zero_denominators_are_null():
    out = economy_comparison(fleet(100), fleet(99), admissible=True)
    assert out['daily']['without_sr_kr']['saving_percent'] == pytest.approx(1)
    assert out['daily']['without_sr_kr']['uncertainty']['saving_rub_min'] < 0
    zero = economy_comparison(fleet(0), fleet(0))
    assert zero['daily']['with_sr_kr']['saving_percent'] is None
    assert zero['daily']['with_sr_kr']['saving_rub_per_saved_km'] is None
