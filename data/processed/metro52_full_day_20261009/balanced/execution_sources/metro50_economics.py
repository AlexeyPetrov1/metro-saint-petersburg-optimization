"""Mileage-based maintenance accrual; annual repetition is an explicit scenario."""

from collections import Counter
from itertools import product
from math import isfinite


PRICES = {
    'Yubileyny': (111000, 313000, 714000, 3181000, 46651000, 44000000, 88000000),
    'Baltiets': (138000, 426000, 956000, 3409000, 47951000, 44000000, 88000000),
}
KINDS = ('TO1', 'TO2', 'TR1', 'TR2', 'TR3', 'SR', 'KR')
FLEET = Counter(Yubileyny=31, Baltiets=37)


def maintenance_rates():
    result = {}
    for kind, prices in PRICES.items():
        intervals = (12000, 42000, 160000, 320000, 640000,
                     1920000 if kind == 'Yubileyny' else 1280000, 5100000)
        components = {name: dict(price_rub_per_train=price, interval_km=km,
                                 rub_per_train_km=price / km)
                      for name, price, km in zip(KINDS, prices, intervals)}
        result[kind] = dict(components=components,
            without_sr_kr=sum(components[k]['rub_per_train_km'] for k in KINDS[:5]),
            with_sr_kr=sum(v['rub_per_train_km'] for v in components.values()))
    return result


def _distance(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('finite nonnegative computed train mileage required') from error
    if isinstance(value, bool) or not isfinite(number) or number < 0:
        raise ValueError('finite nonnegative computed train mileage required')
    return number


def _ledger(source):
    if 'metrics' in source:
        source = source['metrics']
    rows = source.get('per_train', [])
    ledger, planned = {}, {}
    for row in rows:
        tid, kind = row['train_id'], row['train_type']
        if tid in ledger or kind not in FLEET:
            raise ValueError('unique known train types required')
        ledger[tid] = (kind, _distance(row.get('executed_km', row.get('km'))))
        if row.get('planned_km') is not None:
            planned[tid] = (kind, _distance(row['planned_km']))
    if Counter(k for k, _ in ledger.values()) != FLEET:
        raise ValueError('complete fixed fleet: 31 Yubileyny and 37 Baltiets required')
    if source.get('planned_per_train') is not None:
        planned, _ = _ledger({'per_train': source['planned_per_train']})
    if planned and (set(planned) != set(ledger) or any(planned[t][0] != ledger[t][0] for t in ledger)):
        raise ValueError('planned and executed ledgers must cover the same fleet')
    return ledger, planned or None


def _cost(rows, rates, side, capital, sr=1, kr=1):
    return sum(r[side] * (rates[r[0]]['without_sr_kr'] + (
        sr * rates[r[0]]['components']['SR']['rub_per_train_km'] +
        kr * rates[r[0]]['components']['KR']['rub_per_train_km'] if capital else 0)) for r in rows)


def _period(rows, rates, days=1):
    rows = [(kind, base * days, candidate * days) for kind, base, candidate in rows]
    base_km, candidate_km = (sum(r[i] for r in rows) for i in (1, 2))
    saved_km = base_km - candidate_km
    result = dict(reference_km=base_km, candidate_km=candidate_km, saved_km=saved_km,
                  mean_saved_km_per_fixed_train=saved_km / len(rows))
    for column, capital in [('without_sr_kr', False), ('with_sr_kr', True)]:
        base, candidate = (_cost(rows, rates, side, capital) for side in (1, 2))
        saving = base - candidate
        price_bounds = (.8, 1.2) if capital else (1,)
        bounds = [(_cost(rows, rates, 1, capital, sr, kr) * bf,
                   _cost(rows, rates, 2, capital, sr, kr) * cf)
                  for bf, cf, sr, kr in product((.95, 1.05), (.95, 1.05), price_bounds, price_bounds)]
        percentages = [100 * (b - c) / b for b, c in bounds if b]
        result[column] = dict(reference_cost_rub=base, candidate_cost_rub=candidate,
            saving_rub=saving, saving_percent=100 * saving / base if base else None,
            mean_reference_cost_per_fixed_train_rub=base / len(rows),
            mean_candidate_cost_per_fixed_train_rub=candidate / len(rows),
            mean_saving_per_fixed_train_rub=saving / len(rows),
            reference_rub_per_km=base / base_km if base_km else None,
            candidate_rub_per_km=candidate / candidate_km if candidate_km else None,
            saving_rub_per_saved_km=saving / saved_km if saved_km else None,
            saving_rub_per_reference_km=saving / base_km if base_km else None,
            uncertainty=dict(reference_cost_rub_min=min(b for b, _ in bounds),
                reference_cost_rub_max=max(b for b, _ in bounds),
                candidate_cost_rub_min=min(c for _, c in bounds),
                candidate_cost_rub_max=max(c for _, c in bounds),
                saving_rub_min=min(b - c for b, c in bounds),
                saving_rub_max=max(b - c for b, c in bounds),
                saving_percent_min=min(percentages) if percentages else None,
                saving_percent_max=max(percentages) if percentages else None))
    result['maintenance_events'] = {kind: {name: dict(
        reference_events=sum(r[1] for r in rows if r[0] == kind) / item['interval_km'],
        candidate_events=sum(r[2] for r in rows if r[0] == kind) / item['interval_km'])
        for name, item in rate['components'].items()} for kind, rate in rates.items()}
    return result


def economy_comparison(reference, candidate, annual_days=250, admissible=False):
    """Compare complete per_train ledgers; optional planned_km stays separate from km.

    Inputs accept metrics or a result containing metrics. km/executed_km is actual
    movement mileage. Missing planned mileage remains unknown, never inferred.
    """
    if type(annual_days) is not int or annual_days <= 0 or type(admissible) is not bool:
        raise ValueError('positive scenario repetition and explicit admissibility required')
    base, base_plan = _ledger(reference)
    other, other_plan = _ledger(candidate)
    if set(base) != set(other) or any(base[t][0] != other[t][0] for t in base):
        raise ValueError('identical train IDs and types required')
    rates = maintenance_rates()
    rows = [(kind, km, other[t][1]) for t, (kind, km) in sorted(base.items())]
    daily, annual = _period(rows, rates), _period(rows, rates, annual_days)
    annual.update(days=annual_days, scenario_only=True,
                  scope='repetition of this simulated day; annual cash budget not established')
    planned = None
    if base_plan is not None and other_plan is not None:
        plans = [(kind, km, other_plan[t][1]) for t, (kind, km) in sorted(base_plan.items())]
        planned = dict(daily=_period(plans, rates), annual=_period(plans, rates, annual_days),
                       scope='planned full ledger, separately from executed mileage')
        planned['annual'].update(days=annual_days, scenario_only=True)
    per_train = [dict(train_id=t, train_type=base[t][0],
        daily=_period([row], rates), annual=_period([row], rates, annual_days))
        for t, row in zip(sorted(base), rows)]
    for row in per_train:
        row['annual'].update(days=annual_days, scenario_only=True)
    return dict(schema_version=1, fixed_fleet_size=68, rates=rates,
        daily=daily, annual=annual, per_train=per_train, planned=planned,
        admissible=admissible,
        accepted_resource_saving=admissible and all(daily[c]['saving_rub'] > 0 for c in ('without_sr_kr', 'with_sr_kr')),
        meets_one_percent_target=admissible and all((daily[c]['saving_percent'] or 0) >= 1 for c in ('without_sr_kr', 'with_sr_kr')),
        uncertainty=dict(computed_mileage_relative_error=.05, sr_kr_price_relative_error=.20,
            mileage_errors='independent reference and candidate adverse bounds',
            price_errors='SR and KR prices vary independently, each shared across both plans and both train types',
            maintenance_intervals_changed=False, bounds_are_confidence_intervals=False),
        accounting='additive mileage-based maintenance resource accrual; overlapping repair scope assumed additive',
        source='latest user-provided train prices and kilometre service intervals',
        cash_saving_rub=None, lost_revenue_rub=None, net_profit_effect_rub=None,
        traction_energy_and_crew_costs_known=False)
