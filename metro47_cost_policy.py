"""Select the least expensive checked schedule without a waiting objective."""

from collections import Counter
from math import isfinite

from metro44_selection import TYPE_ALIASES, _checked, _feasible, ranking_future_risk, resource_value


def normalized_maintenance_rates(rates):
    if rates is None:
        return None
    normalized = {}
    for kind, value in rates.items():
        name = TYPE_ALIASES.get(str(kind).casefold(), str(kind).casefold())
        try:
            rate = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError('finite positive maintenance rates required') from error
        if (name not in ('baltiets', 'yubileyny') or isinstance(value, bool)
                or not isfinite(rate) or rate <= 0 or name in normalized):
            raise ValueError('unique finite positive maintenance rates for both train types required')
        normalized[name] = rate
    if set(normalized) != {'baltiets', 'yubileyny'}:
        raise ValueError('maintenance rates required for both train types')
    return normalized


def choose_by_cost(rows, *, timed_out=False, search_incomplete=False,
                   exceptional_share_limit=0.01, hard_limit=160,
                   maintenance_rates=None, **unused_wait_budgets):
    rates = normalized_maintenance_rates(maintenance_rates)
    rows = list(rows)
    ids = [r['candidate_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate candidate identities')
    priced, fleet = [], None
    for row in rows:
        if not _checked(row):
            continue
        risk = ranking_future_risk(row, hard_limit=hard_limit)
        if not all(isfinite(n) for n in risk) or not _feasible(row, risk, exceptional_share_limit):
            continue
        if 'committed_per_train' not in row:
            continue
        resources = resource_value(row['committed_per_train'])
        if rates is not None:
            if Counter(r['train_type'] for r in resources['per_train']) != Counter(baltiets=37, yubileyny=31):
                raise ValueError('maintenance objective requires the fixed 37/31 fleet')
            for detail in resources['per_train']:
                detail.update(maintenance_rub_per_km=rates[detail['train_type']],
                    maintenance_resource_equivalent_rub=detail['km'] * rates[detail['train_type']])
            resources.update(maintenance_resource_equivalent_rub=sum(
                r['maintenance_resource_equivalent_rub'] for r in resources['per_train']),
                maintenance_rates_rub_per_km=rates,
                maintenance_scope='TO1+TO2+TR1+TR2+TR3; excludes SR and KR')
        identity = {r['train_id']: r['train_type'] for r in resources['per_train']}
        if fleet is not None and identity != fleet:
            raise ValueError('candidate fleet identities or types changed')
        fleet = identity
        priced.append((resources['maintenance_resource_equivalent_rub'] if rates is not None
            else resources['to1_resource_equivalent_rub'], row['candidate_id'] != 'KEEP',
            row['candidate_id'], resources))
    selected = min(priced, key=lambda r: r[:3]) if priced else None
    return dict(selected_candidate_id=selected[2] if selected else None,
        fallback_candidate_id='KEEP' if 'KEEP' in ids else None, mitigation_candidate_id=None,
        status='conditional_resource_policy' if selected else 'no_feasible_candidate',
        resources=selected[3] if selected else None, resource_accounting_complete=bool(selected),
        waiting_budgets=dict(mean_seconds=None, p95_seconds=None),
        objective=('minimum closed-schedule TO1+TO2+TR1+TR2+TR3 resource cost among checked candidates'
            if rates is not None else 'minimum closed-schedule TO1 resource cost among checked candidates'),
        economic_comparison_scope='same forecast; no waiting constraint or objective',
        search_complete=not timed_out and not search_incomplete,
        service_preservation_checked=bool(selected), release_authorized=False,
        economic_optimality=False, economic_improvement_claimed=False,
        cash_saving_rub=None, past_exposure_is_ranked=False)
