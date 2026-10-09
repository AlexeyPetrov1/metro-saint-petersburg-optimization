"""GOAL 4.3 accounting adapter; decimal strings preserve exact JSON values."""

from __future__ import annotations

import copy
from datetime import timedelta
from decimal import Decimal

from economics import (
    Article, Evaluation, Quantity, compare, decimal, difference, exact_sum,
    instant, number, product, ratio, validate_articles, validate_plan,
)


VERSION = "goal4.3-economics-v1"
SOURCE = "user instructions, GOAL 4.3"
TYPES = {"yubileyny": "Yubileyny", "baltiets": "Baltiets"}
CONTEXT = ("demand_id", "initial_state_id", "incident_id", "random_world_id",
           "period_start", "period_end", "closure_end")


def _kind(value):
    value = str(value).casefold()
    value = {"юбилейный": "yubileyny", "балтиец": "baltiets"}.get(value, value)
    if value not in TYPES:
        raise ValueError(f"unsupported train type: {value}")
    return value


def _value(value, *, signed=False):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not an economic quantity")
    result = decimal(value)
    if not signed and result < 0:
        raise ValueError("quantity must be nonnegative")
    return result


def _policy(policy):
    from goal43_policy import load_policy
    return load_policy() if policy is None else policy


def _parameter(policy, name):
    from goal43_policy import parameter
    return parameter(policy, name)


def maintenance_registry(policy=None):
    """Prices are per eight-car train; annual counts are repeated service events."""
    policy = _policy(policy)
    prices = {_kind(kind): {work: number(_value(price)) for work, price in rows.items()}
              for kind, rows in _parameter(policy, "maintenance_prices_rub_by_type").items()}
    wagons = {_kind(kind): {work: number(_value(count)) for work, count in rows.items()}
              for kind, rows in _parameter(policy, "annual_car_service_counts").items()}
    interval = _value(_parameter(policy, "to1_interval_km"))
    if interval is None or interval <= 0:
        raise ValueError("TO1 interval must be positive")
    fare = _value(_parameter(policy, "fare_revenue_rub"))
    trips = _value(_parameter(policy, "annual_blue_line_trips2025"))
    if trips is None or trips != trips.to_integral_value():
        raise ValueError("annual trips must be an integer")
    return {
        "source": SOURCE, "prices_rub_per_train": prices, "cars_per_train": 8,
        "parameter_provenance": {name: copy.deepcopy(policy["parameters"][name]) for name in (
            "maintenance_prices_rub_by_type", "to1_interval_km", "annual_car_service_counts",
            "fare_revenue_rub", "annual_blue_line_trips2025")},
        "to1_interval_km": number(interval),
        "to1_resource_rates_rub_per_train_km": {
            kind: number(ratio(decimal(rows["TO1"]), interval)) for kind, rows in prices.items()},
        "repair_km_rates": None,
        "repair_label": {"TR1": "текущий ремонт", "TR2_TR3_legacy_mapping": None},
        "to1_rule_conflict": {"legacy_interval_km": "10000", "legacy_applicability": "unknown",
                              "new_profile_interval_km": number(interval), "precedence": "user instructions"},
        "annual_car_service_counts": wagons,
        "annual_service_equivalents": {
            kind: {work: number(ratio(decimal(count), Decimal(8))) for work, count in rows.items()}
            for kind, rows in wagons.items()},
        "annual_volume_unit": "equivalent whole-train service events; not unique trains",
        "annual_volume_period": None, "annual_volume_line": None,
        "fare_revenue_rub": number(fare),
        "blue_line_2025": {"line_id": 2, "year": 2025, "trips": int(trips),
                           "gross_revenue_rub": number(product(fare, trips)), "scale_line1": False},
        "earlier_699300000_trip_scope": "unknown; excluded from annual scaling",
    }


def _check_context(baseline, candidate):
    if any(key not in baseline or key not in candidate or baseline[key] != candidate[key] for key in CONTEXT):
        raise ValueError("comparison requires the same context, incident, random world and boundaries")
    if not baseline.get("plan_id") or not candidate.get("plan_id"):
        raise ValueError("plans need identifiers")
    if any(not baseline[key] for key in CONTEXT[:4]):
        raise ValueError("comparison needs explicit context identifiers")
    start, end, close = [instant(baseline[key]) for key in CONTEXT[-3:]]
    if not start < end <= close or any(t.utcoffset() != timedelta(hours=3) for t in (start, end, close)):
        raise ValueError("invalid common boundary or Moscow timezone")


def _evaluation(plan, amounts):
    return Evaluation(
        plan["plan_id"], plan["demand_id"], plan["initial_state_id"],
        plan["period_start"], plan["period_end"], plan["closure_end"],
        {key: Quantity(value, "train_km", plan.get("source", "supplied GOAL 4.3 simulation ledger"), "assumed")
         for key, value in amounts.items()},
        model_version=VERSION, evaluation_mode="evaluated", obligations_closed=True,
        constraint_status=plan.get("constraint_status", "unknown"),
        required_cash_articles=tuple(plan.get("required_cash_articles", ())),
    )


def _revenue(data, fare):
    data = {} if data is None else data
    trips = _value(data.get("additional_unpaid_trips"), signed=True)
    refund = _value(data.get("refunds_rub"), signed=True)
    margin = _value(data.get("contribution_margin_per_trip"))
    if trips is not None and trips != trips.to_integral_value():
        raise ValueError("additional unpaid trips must be an integer")
    if margin is not None and margin > fare:
        raise ValueError("contribution margin cannot exceed trip revenue")
    if trips and refund and data.get("loss_categories_disjoint") is not True:
        raise ValueError("unpaid trip losses and refunds need an explicit disjoint scope")
    unpaid_ids, refund_ids = data.get("unpaid_loss_ids", []), data.get("refund_loss_ids", [])
    if len(set(unpaid_ids)) != len(unpaid_ids) or len(set(refund_ids)) != len(refund_ids):
        raise ValueError("duplicate loss identifier")
    if set(unpaid_ids).intersection(refund_ids):
        raise ValueError("unpaid trip and refund loss identifiers overlap")
    trip_loss = None if trips is None else product(trips, fare)
    known = trips is not None and refund is not None
    revenue = exact_sum((trip_loss, refund)) if known else None
    profit = exact_sum((product(trips, margin), refund)) if known and margin is not None else None
    return {
        "fare_revenue_rub": number(fare), "additional_unpaid_trips": number(trips),
        "foregone_trip_revenue_rub": number(trip_loss), "refunds_rub": number(refund),
        "lost_revenue_rub": number(revenue), "status": "supplied_inputs" if known else "unknown",
        "paid_waiting_revenue_loss_rub": "0", "paid_denied_boarding_revenue_loss_rub": "0",
        "contribution_margin_per_trip": number(margin), "lost_profit_rub": number(profit),
        "profit_status": "configured_margin_only" if profit is not None else "unknown_variable_costs",
        "source": data.get("source", "not supplied"),
    }


def compare_train_economics(baseline, candidate, fleet, revenue=None, cash_articles=(), policy=None):
    """Compare equal worlds on a fixed train set, retaining incomplete obligations."""
    _check_context(baseline, candidate)
    kinds = {train: _kind(kind) for train, kind in fleet.items()}
    ids = sorted(kinds)
    if not ids or any(not isinstance(train, str) or not train for train in ids):
        raise ValueError("fixed train identifiers must be nonempty")
    if any(set(plan.get("trains", {})) != set(ids) for plan in (baseline, candidate)):
        raise ValueError("both plans must cover the fixed train identifier set")
    registry = maintenance_registry(policy)
    rates = {kind: decimal(rate) for kind, rate in registry["to1_resource_rates_rub_per_train_km"].items()}
    articles = [Article(f"TO1_{kind}", f"{kind}_train_km", rate, "RUB/train_km",
                        (f"maintenance:TO1:{kind}",), "resource_accrual", SOURCE,
                        "assumed", cycle_cost_complete=True) for kind, rate in rates.items()]
    extras = [item if isinstance(item, Article) else Article(**{
        **item, "covered_components": tuple(item["covered_components"])}) for item in cash_articles]
    validate_articles([*articles, *extras])
    complete = all(plan.get("coverage_complete") is True and plan.get("obligations_closed") is True
                   for plan in (baseline, candidate))
    rows, ledgers = {}, [{kind: [] for kind in TYPES} for _ in range(2)]
    for train in ids:
        kind = kinds[train]
        parts = []
        for plan in (baseline, candidate):
            row = plan["trains"][train]
            if "train_type" in row and _kind(row["train_type"]) != kind:
                raise ValueError("train type changed between fixed fleet and ledger")
            passenger, nonpassenger = (_value(row.get(key)) for key in ("passenger_km", "nonpassenger_km"))
            total = None if passenger is None or nonpassenger is None else exact_sum((passenger, nonpassenger))
            complete = complete and total is not None and not row.get("obligations")
            parts.append({"passenger_km": passenger, "nonpassenger_km": nonpassenger, "km": total,
                          "moving_seconds": _value(row.get("moving_seconds")),
                          "standby_seconds": _value(row.get("standby_seconds")),
                          "obligations": copy.deepcopy(row.get("obligations", []))})
        base, cand = parts
        delta = None if base["km"] is None or cand["km"] is None else difference(base["km"], cand["km"])
        rows[train] = {"train_id": train, "train_type": kind, "to1_rate_rub_per_train_km": number(rates[kind]),
                       "observed_km_delta": number(delta)}
        for label, part in zip(("baseline", "candidate"), parts):
            rows[train].update({f"{label}_{key}": number(value) if key != "obligations" else value
                               for key, value in part.items()})
        for key in ("passenger_km", "nonpassenger_km"):
            rows[train][f"saving_{key}"] = (None if base[key] is None or cand[key] is None
                                          else number(difference(base[key], cand[key])))
        for ledger, part in zip(ledgers, parts):
            ledger[kind].append(part["km"])
    for row in rows.values():
        row["saving_km"] = row["observed_km_delta"] if complete else None
        row["to1_resource_value_rub"] = (number(product(decimal(row["saving_km"]), rates[row["train_type"]]))
                                         if complete else None)
        row["to1_resource_fraction"] = (number(ratio(decimal(row["saving_km"]), decimal(registry["to1_interval_km"])))
                                         if complete else None)
    observed = None if any(row["observed_km_delta"] is None for row in rows.values()) else exact_sum(
        decimal(row["observed_km_delta"]) for row in rows.values())
    by_type = {}
    for kind in TYPES:
        type_rows = [row for row in rows.values() if row["train_type"] == kind]
        by_type[kind] = {"train_count": len(type_rows),
                         "saving_km": number(exact_sum(decimal(row["saving_km"]) for row in type_rows)) if complete else None,
                         "to1_resource_value_rub": number(exact_sum(decimal(row["to1_resource_value_rub"]) for row in type_rows)) if complete else None}
    detailed = None
    if complete:
        evaluations = [_evaluation(plan, {f"{kind}_train_km": exact_sum(ledger[kind]) for kind in TYPES})
                       for plan, ledger in zip((baseline, candidate), ledgers)]
        for evaluation in evaluations:
            validate_plan(evaluation)
        detailed = compare(*evaluations, [*articles, *extras], model_version=VERSION)
    cash_complete = complete and all(plan.get("cash_scope_complete") is True for plan in (baseline, candidate))
    if detailed is not None and not cash_complete:
        detailed["groups"]["cash"].update(baseline=None, candidate=None, savings=None,
                                          savings_pct=None, status="partial")
        detailed["status"] = "partial"
    revenue_result = _revenue(revenue, decimal(registry["fare_revenue_rub"]))
    return {
        "model_version": VERSION, "status": "complete" if complete else "partial",
        "fixed_train_ids": ids, "per_train": rows, "by_type": by_type,
        "totals": {"observed_km_delta": number(observed), "saving_km": number(observed) if complete else None,
                   "mean_saving_km_per_train": number(ratio(observed, Decimal(len(ids)))) if complete else None,
                   "to1_resource_value_rub": number(exact_sum(decimal(row["to1_resource_value_rub"]) for row in rows.values())) if complete else None},
        "boundary": {key: baseline[key] for key in CONTEXT[-3:]},
        "cost_comparison": detailed,
        "cash_savings_rub": detailed["groups"]["cash"]["savings"] if cash_complete else None,
        "cash_status": "configured_scope_only" if cash_complete else "unknown_operating_costs",
        "economic_status": "configured_scope_only" if cash_complete else "partial_unknown_operating_costs",
        "unknown_costs": ["energy", "labor", "other_operations"] if not cash_complete else [],
        "revenue": revenue_result, "net_profit_effect_rub": None,
        "maintenance_cash_savings_rub": None,
        "note": "TO1 resource value is not a cash payment saving; incomplete coverage cannot establish final mileage savings.",
        "release_authorized": False,
    }


def to1_deferral(data, policy=None):
    """Estimate next TO1 under a supplied constant future daily mileage plan."""
    interval = decimal(maintenance_registry(policy)["to1_interval_km"])
    values = [_value(data.get(key)) for key in ("km_since_to1", "baseline_evaluation_km",
                                               "candidate_evaluation_km", "future_daily_km")]
    odometer, base, cand, daily = values
    days = [None, None]
    due = False
    if all(value is not None for value in (odometer, base, cand)):
        remaining = [difference(interval, exact_sum((odometer, km))) for km in (base, cand)]
        days = [Decimal(0) if value <= 0 else ratio(value, daily) if daily is not None and daily > 0 else None
                for value in remaining]
        due = any(value <= 0 for value in remaining)
    at = instant(data["at"]) if data.get("at") else None
    dates = [(at + timedelta(seconds=float(product(day, Decimal(86400))))).isoformat()
             if day is not None and at is not None else None for day in days]
    return {"status": "due" if due else "assumed_daily_plan" if days[0] is not None else "unknown",
            "to1_interval_km": number(interval), "baseline_days_until_to1": number(days[0]),
            "candidate_days_until_to1": number(days[1]),
            "deferral_days": number(difference(days[1], days[0])) if all(day is not None for day in days) else None,
            "baseline_to1_at": dates[0], "candidate_to1_at": dates[1],
            "cash_savings_rub": None, "permanent_cancellation": False,
            "date_basis": "constant supplied daily mileage; no automatic maintenance reset"}


def revenue_sensitivity(data, policy=None):
    """Compare explicitly hypothetical unpaid-trip losses with one benefit basis."""
    registry = maintenance_registry(policy)
    fare = decimal(registry["fare_revenue_rub"])
    benefit = _value(data.get("benefit_rub"), signed=True)
    potential = _value(data.get("potential_unpaid_trips"))
    if benefit is None or potential is None or not data.get("benefit_kind") or fare <= 0:
        raise ValueError("sensitivity needs benefit, benefit_kind, potential trips and positive fare")
    scenarios = []
    for value in data.get("abandonment_shares", []):
        share = _value(value)
        if share is None or share > 1:
            raise ValueError("abandonment share must be in [0,1]")
        trips = product(potential, share)
        loss = product(trips, fare)
        scenarios.append({"abandonment_share": number(share), "additional_unpaid_trips": number(trips),
                          "lost_revenue_rub": number(loss), "comparison_rub": number(difference(benefit, loss)),
                          "status": "assumed"})
    return {"benefit_rub": number(benefit), "benefit_kind": data["benefit_kind"],
            "break_even_additional_unpaid_trips": number(ratio(max(Decimal(0), benefit), fare)),
            "scenarios": scenarios, "comparison_is_cash_profit": False,
            "note": "One selected benefit basis; resource accrual and the same maintenance invoice are never added."}
