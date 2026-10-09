"""Independent typed mileage arithmetic; physical and service gates stay explicit."""

from decimal import Decimal

RATES={'baltiets':Decimal('138000')/15000,'yubileyny':Decimal('111000')/15000}


def number(value):
    result=Decimal(str(value))
    if not result.is_finite()or result<0:raise ValueError('finite nonnegative quantity required')
    return result


def mileage_ledger(trains,work):
    fleet={t['train_id']:t['train_type'].casefold()for t in trains}
    rows={r['train_id']:r for r in work}
    if len(trains)!=68 or len(fleet)!=68 or len(work)!=68 or len(rows)!=68 or fleet.keys()!=rows.keys():
        raise ValueError('identical 68 unique train identities required')
    details=[];km=wear=Decimal(0)
    for identity,kind in sorted(fleet.items()):
        row=rows[identity]
        if kind not in RATES or row['train_type'].casefold()!=kind:raise ValueError('unknown or changed train type')
        value=number(row['train_km']);passenger=number(row['passenger_train_km']);other=number(row['nonpassenger_train_km'])
        if value!=passenger+other:raise ValueError('train mileage differs from passenger plus nonpassenger mileage')
        if row['cars']!=8 or number(row['car_km'])!=8*value:raise ValueError('car mileage must be eight times train mileage')
        resource=value*RATES[kind];km+=value;wear+=resource
        details.append(dict(train_id=identity,train_type=kind,train_km=float(value),
            passenger_train_km=float(passenger),nonpassenger_train_km=float(other),
            to1_rub_per_km=float(RATES[kind]),to1_resource_rub=float(resource)))
    return dict(fixed_fleet_size=68,train_km=float(km),km_per_fixed68=float(km/68),to1_resource_rub=float(wear),
        active_trains=sum(r['train_km']>0 for r in details),inactive_trains=sum(r['train_km']==0 for r in details),
        per_train=details,cash_savings_rub=None,lost_revenue_rub=None,net_profit_rub=None,
        to1_resource_is_cash=False)


def compare_ledgers(reference,candidate,*,reference_safe,candidate_safe,reference_wait,candidate_wait,wait_budget):
    left={r['train_id']:r for r in reference['per_train']};right={r['train_id']:r for r in candidate['per_train']}
    if left.keys()!=right.keys()or any(left[t]['train_type']!=right[t]['train_type']for t in left):
        raise ValueError('paired fleet identities and types must remain fixed')
    def within(wait,budget):
        return (budget is not None and len(budget)==len(wait)==2
                and all(v is not None and limit is not None and number(v)<=number(limit)for v,limit in zip(wait,budget)))
    ref_wait,cand_wait=within(reference_wait,wait_budget),within(candidate_wait,wait_budget)
    eligible=reference_safe is True and candidate_safe is True and ref_wait and cand_wait
    details=[];km=wear=Decimal(0)
    for identity,old in left.items():
        delta=number(old['train_km'])-number(right[identity]['train_km']);resource=delta*RATES[old['train_type']]
        km+=delta;wear+=resource
        details.append(dict(train_id=identity,train_type=old['train_type'],reduction_train_km=float(delta),
                            to1_resource_reduction_rub=float(resource)))
    strict=all(a is not None and b is not None and number(b)<=number(a)for a,b in zip(reference_wait,candidate_wait))
    return dict(observed_reduction_train_km=float(km),observed_reduction_km_per_fixed68=float(km/68),
        observed_to1_resource_reduction_rub=float(wear),per_train=details,
        reference_wait_budget_passed=ref_wait,candidate_wait_budget_passed=cand_wait,
        resource_comparison_eligible=eligible,accepted_saving_train_km=float(km)if eligible and km>0 and wear>0 else None,
        strict_no_wait_regression_passed=strict,waiting_budget=wait_budget,
        cash_savings_rub=None,lost_revenue_rub=None,net_profit_rub=None,to1_resource_is_cash=False)
