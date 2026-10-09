"""Itemized economics. Monetary values are RUB; fixture prices are never observations."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory


MODEL_VERSION = "economics-v1.1"
KINDS = ("cash", "resource_accrual", "capital")
STATUSES = {"confirmed", "assumed", "partial", "unknown"}


def decimal(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid number: {value}") from exc
    if not number.is_finite():
        raise ValueError("number must be finite")
    return number


def number(value):
    if value is None:
        return None
    value = decimal(value)
    text = format(value, "f")
    return "0" if value == 0 else text.rstrip("0").rstrip(".") if "." in text else text


def exact_sum(values):
    values = list(values)
    if not values:
        return Decimal(0)
    with localcontext() as context:
        context.prec = max(28, max(v.adjusted() for v in values) - min(v.as_tuple().exponent for v in values)
                           + len(str(len(values))) + 2)
        return sum(values, Decimal(0))


def product(first, second):
    with localcontext() as context:
        context.prec = max(28, len(first.as_tuple().digits) + len(second.as_tuple().digits) + 2)
        return first * second


def difference(first, second):
    return exact_sum((first, second.copy_negate()))


def ratio(first, second):
    with localcontext() as context:
        context.prec = 28
        return first / second


def instant(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("time must include a timezone")
    return result


def status_of(statuses):
    statuses = list(statuses)
    if not statuses:
        return "confirmed"
    if all(s == "unknown" for s in statuses):
        return "unknown"
    if any(s in {"unknown", "partial"} for s in statuses):
        return "partial"
    return "assumed" if "assumed" in statuses else "confirmed"


@dataclass(frozen=True)
class Quantity:
    amount: Decimal | str | None
    unit: str
    source: str
    status: str = "confirmed"
    applicability: str = "applicable"

    def __post_init__(self):
        if self.status not in STATUSES or not self.unit or not self.source:
            raise ValueError("quantity needs unit, source and valid status")
        if self.applicability not in {"applicable", "not_applicable"}:
            raise ValueError("invalid quantity applicability")
        if self.applicability == "not_applicable":
            if self.amount is not None and decimal(self.amount) != 0:
                raise ValueError("not applicable quantity cannot have a nonzero amount")
            if self.status not in {"confirmed", "assumed"}:
                raise ValueError("not applicable quantity needs explicit evidence or assumption")
            object.__setattr__(self, "amount", Decimal(0))
        if self.amount is not None:
            amount = decimal(self.amount)
            if amount < 0 or self.status == "unknown":
                raise ValueError("known quantity must be nonnegative and not unknown")
            object.__setattr__(self, "amount", amount)
        else:
            object.__setattr__(self, "status", "unknown")

    def as_dict(self):
        return {"amount": number(self.amount), "unit": self.unit,
                "source": self.source, "status": self.status, "applicability": self.applicability}


@dataclass(frozen=True)
class Article:
    name: str
    quantity_key: str
    rate: Decimal | str | None
    unit: str
    covered_components: tuple[str, ...]
    accounting_kind: str
    source: str
    status: str = "unknown"
    cycle_cost_complete: bool = False
    vat_basis: str = "unspecified"

    def __post_init__(self):
        if self.accounting_kind not in KINDS or self.status not in STATUSES:
            raise ValueError("invalid article kind or status")
        if self.vat_basis not in {"included", "excluded", "not_applicable", "unspecified"}:
            raise ValueError("invalid VAT basis")
        if not self.name or not self.quantity_key or not self.source:
            raise ValueError("article needs name, quantity_key and source")
        if not self.unit.startswith("RUB/") or len(self.unit.split("/")) != 2:
            raise ValueError("rate unit must be RUB/<quantity unit>; convert currency explicitly")
        if not self.covered_components or len(set(self.covered_components)) != len(self.covered_components):
            raise ValueError("article needs unique covered_components")
        if self.rate is not None:
            value = decimal(self.rate)
            if value < 0 or self.status == "unknown":
                raise ValueError("known rate needs nonnegative value and evidence status")
            object.__setattr__(self, "rate", value)
            if self.accounting_kind == "resource_accrual" and self.cycle_cost_complete is not True:
                raise ValueError("resource pricing requires a complete cycle costing")
        else:
            object.__setattr__(self, "status", "unknown")


@dataclass(frozen=True)
class Event:
    event_id: str
    at: str
    quantities: dict[str, Quantity]
    source: str
    operation_id: str | None = None


@dataclass(frozen=True)
class Evaluation:
    plan_id: str
    demand_id: str
    initial_state_id: str
    period_start: str
    period_end: str
    closure_end: str
    quantities: dict[str, Quantity]
    events: tuple[Event, ...] = ()
    model_version: str = "fixture-v1"
    evaluation_mode: str = "fixture"
    obligations_closed: bool | None = None
    constraint_status: str = "unknown"
    required_cash_articles: tuple[str, ...] = ()


def validate_plan(plan):
    start, end, close = map(instant, (plan.period_start, plan.period_end, plan.closure_end))
    if not start < end <= close:
        raise ValueError("invalid period or closure boundary")
    if len({t.utcoffset() for t in (start, end, close)}) != 1:
        raise ValueError("boundary timezone offsets must match")
    if plan.obligations_closed is not True:
        raise ValueError("obligations must be explicitly closed, including return and passenger queues")
    if plan.evaluation_mode not in {"fixture", "evaluated"}:
        raise ValueError("evaluation_mode must be fixture or evaluated")
    if plan.constraint_status not in {"passed", "failed", "unknown"}:
        raise ValueError("invalid constraint status")
    if not all((plan.plan_id, plan.demand_id, plan.initial_state_id, plan.model_version)):
        raise ValueError("evaluation needs context identifiers")
    if isinstance(plan.required_cash_articles, str) or any(not isinstance(n, str) or not n for n in plan.required_cash_articles):
        raise ValueError("required_cash_articles must contain article names")
    if len(set(plan.required_cash_articles)) != len(plan.required_cash_articles):
        raise ValueError("required_cash_articles must be unique")
    ids, operations = set(), set()
    for event in plan.events:
        operation = event.operation_id or event.event_id
        if not event.event_id or event.event_id in ids or operation in operations or not event.source:
            raise ValueError("events need unique identifiers and sources")
        if instant(event.at).utcoffset() != start.utcoffset():
            raise ValueError("event timezone must match the boundary timezone")
        if not start <= instant(event.at) < close:
            raise ValueError("event outside closure boundary")
        ids.add(event.event_id)
        operations.add(operation)


def total_quantities(plan):
    """Core quantities exclude ALL event quantities, which are added exactly once."""
    parts = {}
    for ledger in [plan.quantities, *(event.quantities for event in plan.events)]:
        for key, quantity in ledger.items():
            parts.setdefault(key, []).append(quantity)
    totals = {}
    for key, quantities in parts.items():
        units = {q.unit for q in quantities}
        if len(units) != 1:
            raise ValueError(f"quantity unit mismatch: {key}")
        known = [q.amount for q in quantities if q.amount is not None]
        amount = exact_sum(known) if known else None
        status = status_of(q.status for q in quantities)
        applicability = "not_applicable" if all(q.applicability == "not_applicable" for q in quantities) else "applicable"
        totals[key] = Quantity(amount, quantities[0].unit,
                               "; ".join(dict.fromkeys(q.source for q in quantities)),
                               status, applicability)
    return totals


def validate_articles(articles):
    names, components = set(), set()
    for article in articles:
        if article.name in names:
            raise ValueError("duplicate article name")
        names.add(article.name)
        if components.intersection(article.covered_components):
            raise ValueError("covered_components overlap across accounting articles")
        components.update(article.covered_components)
    for kind in KINDS:
        bases = {a.vat_basis for a in articles if a.accounting_kind == kind}
        if len(bases) > 1:
            raise ValueError(f"incompatible VAT bases in {kind}")


def priced(quantity, article, period):
    if quantity.unit != article.unit.removeprefix("RUB/"):
        raise ValueError(f"quantity unit mismatch for {article.name}")
    known = None
    if quantity.applicability == "not_applicable":
        amount, status, reason = Decimal(0), quantity.status, "not_applicable"
    elif quantity.amount == 0 and quantity.status in {"confirmed", "assumed"}:
        amount, status, reason = Decimal(0), quantity.status, "known_zero"
    elif quantity.amount is None:
        amount, status, reason = None, "unknown", "unknown_quantity"
    elif article.rate is None:
        amount, status, reason = None, "partial", "unknown_rate"
    else:
        known = product(quantity.amount, article.rate)
        status = status_of((quantity.status, article.status))
        amount = None if status == "partial" else known
        reason = "partial_quantity_or_rate" if status == "partial" else None
        if article.vat_basis == "unspecified" and status == "confirmed":
            status, reason = "assumed", "vat_basis_unspecified"
    known = amount if amount is not None else known
    return {"quantity": quantity.as_dict(), "rate": number(article.rate),
            "rate_unit": article.unit, "rate_source": article.source,
            "rate_status": article.status, "amount": number(amount),
            "known_subtotal": number(known), "reason": reason, "vat_basis": article.vat_basis,
            "currency": "RUB", "period": period, "status": status}


def summed(values):
    return None if any(v is None for v in values) else exact_sum(decimal(v) for v in values)


def groups_for(rows):
    groups = {}
    for kind in KINDS:
        items = [r for r in rows if r["accounting_kind"] == kind]
        base = summed([r["baseline"]["amount"] for r in items])
        candidate = summed([r["candidate"]["amount"] for r in items])
        saving = summed([r["savings"] for r in items])
        groups[kind] = {
            "baseline": number(base), "candidate": number(candidate), "savings": number(saving),
            "savings_pct": number(ratio(product(Decimal(100), saving), base)) if saving is not None and base else None,
            "known_subtotal_baseline": number(exact_sum(decimal(r["baseline"]["known_subtotal"]) for r in items
                                                        if r["baseline"]["known_subtotal"] is not None)),
            "known_subtotal_candidate": number(exact_sum(decimal(r["candidate"]["known_subtotal"]) for r in items
                                                         if r["candidate"]["known_subtotal"] is not None)),
            "vat_basis": items[0]["baseline"]["vat_basis"] if items else None,
            "currency": "RUB", "status": status_of(r["status"] for r in items),
        }
    return groups


def accounting_scope(result):
    return (tuple(result.get("required_cash_articles", ())), tuple(sorted(
        (r["name"], r["quantity_key"], r["baseline"]["rate_unit"], r["accounting_kind"],
         tuple(sorted(r["covered_components"])), r["baseline"]["vat_basis"]) for r in result["articles"])))


def compare(baseline, candidate, articles, model_version=MODEL_VERSION):
    for plan in (baseline, candidate):
        validate_plan(plan)
    fields = ("demand_id", "initial_state_id", "period_start", "period_end", "closure_end",
              "model_version", "evaluation_mode", "required_cash_articles")
    if any(getattr(baseline, field) != getattr(candidate, field) for field in fields):
        raise ValueError("comparison requires identical demand, initial state, boundary and model version")
    articles = list(articles)
    validate_articles(articles)
    base, cand = total_quantities(baseline), total_quantities(candidate)
    period = {"start": baseline.period_start, "end": baseline.closure_end}
    rows = []
    for item in articles:
        unit = item.unit.removeprefix("RUB/")
        bq = base.get(item.quantity_key, Quantity(None, unit, "not supplied"))
        cq = cand.get(item.quantity_key, Quantity(None, unit, "not supplied"))
        bp, cp = priced(bq, item, period), priced(cq, item, period)
        complete = bq.status in {"confirmed", "assumed"} and cq.status in {"confirmed", "assumed"}
        if complete and bq.amount is not None and bq.amount == cq.amount:
            saving = Decimal(0)
        elif bp["amount"] is None or cp["amount"] is None:
            saving = None
        else:
            saving = difference(decimal(bp["amount"]), decimal(cp["amount"]))
        rows.append({"name": item.name, "quantity_key": item.quantity_key,
                     "covered_components": list(item.covered_components), "accounting_kind": item.accounting_kind,
                     "baseline": bp, "candidate": cp, "savings": number(saving),
                     "status": status_of((bp["status"], cp["status"]))})
    natural = {}
    for key in sorted(base.keys() | cand.keys()):
        bq, cq = base.get(key), cand.get(key)
        unit = (bq or cq).unit
        bq = bq or Quantity(None, unit, "not supplied")
        cq = cq or Quantity(None, unit, "not supplied")
        if bq.unit != cq.unit:
            raise ValueError(f"quantity unit mismatch: {key}")
        complete = bq.status in {"confirmed", "assumed"} and cq.status in {"confirmed", "assumed"}
        delta = difference(bq.amount, cq.amount) if complete else None
        natural[key] = {"baseline": bq.as_dict(), "candidate": cq.as_dict(),
                        "delta": number(delta), "unit": unit, "period": period}
    events = {label: [{"event_id": e.event_id, "operation_id": e.operation_id or e.event_id,
                       "operation_id_explicit": e.operation_id is not None, "at": e.at, "source": e.source,
                       "quantities": {key: q.as_dict() for key, q in sorted(e.quantities.items())}}
                      for e in plan.events] for label, plan in (("baseline", baseline), ("candidate", candidate))}
    missing = sorted(set(baseline.required_cash_articles) - {a.name for a in articles if a.accounting_kind == "cash"})
    groups = groups_for(rows)
    if missing:
        groups["cash"].update(baseline=None, candidate=None, savings=None, savings_pct=None, status="partial")
    return {"schema_version": "1.1", "model_version": model_version,
            "cost_scope": "configured_articles_only",
            "required_cash_articles": list(baseline.required_cash_articles), "missing_required_articles": missing,
            "evaluation_model_version": baseline.model_version, "evaluation_mode": baseline.evaluation_mode,
            "plans": {"baseline": baseline.plan_id, "candidate": candidate.plan_id},
            "demand_id": baseline.demand_id, "initial_state_id": baseline.initial_state_id,
            "boundary": {"period_start": baseline.period_start, "period_end": baseline.period_end,
                         "closure_end": baseline.closure_end, "obligations_closed": True},
            "articles": rows, "groups": groups, "natural_deltas": natural, "events": events,
            "status": "partial" if missing else status_of(r["status"] for r in rows),
            "operational_status": {"baseline": baseline.constraint_status, "candidate": candidate.constraint_status},
            "release_authorized": False}


def round_trip(one_way_km, cars, minutes, source):
    distance, duration = decimal(one_way_km), decimal(minutes)
    if not isinstance(cars, int) or isinstance(cars, bool) or cars <= 0 or distance < 0 or duration < 0:
        raise ValueError("invalid trip parameters")
    km = product(Decimal(2), distance)
    return {"passenger_train_km": Quantity(km, "train_km", source, "assumed"),
            "passenger_car_km": Quantity(product(km, Decimal(cars)), "car_km", source, "assumed"),
            "moving_hours": Quantity(ratio(duration, Decimal(60)), "train_h", source, "assumed")}


def break_even(switch_cost, running_per_hour, standby_per_hour, source, status="assumed"):
    values = [None if v is None else decimal(v) for v in (switch_cost, running_per_hour, standby_per_hour)]
    if any(v is not None and v < 0 for v in values) or status not in STATUSES or not source:
        raise ValueError("invalid threshold inputs")
    switching, running, standby = values
    hourly_delta = None if running is None or standby is None else difference(running, standby)
    threshold = ratio(switching, hourly_delta) if switching is not None and hourly_delta is not None and hourly_delta > 0 else None
    return {"threshold_hours": number(threshold), "switch_cost": number(switching),
            "switch_cost_unit": "RUB", "running_rate": number(running), "standby_rate": number(standby),
            "hourly_rate_unit": "RUB/train_h", "source": source,
            "status": "unknown" if any(v is None for v in values) else status,
            "reason": "unknown inputs" if any(v is None for v in values) else
                      "running rate <= standby rate" if hourly_delta <= 0 else "finite boundary; profitable strictly above it",
            "release_authorized": False}


def price_frontier(quantity_delta, other_savings, rate_unit, source):
    """Savings = quantity_delta * rate + other_savings, with other inputs frozen."""
    delta, offset = decimal(quantity_delta), decimal(other_savings)
    if not rate_unit.startswith("RUB/") or not source:
        raise ValueError("frontier needs a RUB rate unit and source")
    rate = ratio(offset.copy_negate(), delta) if delta else None
    profitable = "rate > boundary" if delta > 0 else "rate < boundary" if delta < 0 else "all rates" if offset > 0 else "no rates"
    if rate is not None and rate < 0:
        profitable = "all rates" if delta > 0 else "no rates"
    return {"rate": number(rate) if rate is not None and rate >= 0 else None,
            "unit": rate_unit, "quantity_delta": number(delta), "other_savings": number(offset),
            "other_savings_unit": "RUB", "source": source, "status": "assumed",
            "profitable_when": profitable,
            "domain": "rate >= 0; quantities and other rates held fixed"}


def compare_scenarios(scenarios, frontier_quantity_key=None):
    scenarios = list(scenarios)
    has_probabilities = any("probability" in case for case in scenarios)
    probabilities = []
    if has_probabilities:
        if any("probability" not in case for case in scenarios):
            raise ValueError("scenario probabilities must be supplied for every scenario")
        try:
            probabilities = [decimal(case["probability"]) for case in scenarios]
        except ValueError as exc:
            raise ValueError("scenario probabilities must be finite numbers") from exc
        if any(not 0 <= p <= 1 for p in probabilities) or exact_sum(probabilities) != 1:
            raise ValueError("scenario probabilities must be in [0,1] and sum to 1")
    rows = []
    names = set()
    context = None
    scope = None
    for case in scenarios:
        if case["name"] in names or decimal(case["delay_minutes"]) < 0:
            raise ValueError("scenario needs unique name and nonnegative explicit delay")
        names.add(case["name"])
        current = tuple(getattr(case["baseline"], key) for key in
                        ("demand_id", "initial_state_id", "period_start", "period_end", "closure_end",
                         "model_version", "evaluation_mode"))
        if context is not None and current != context:
            raise ValueError("sensitivity scenarios require the same context")
        context = current
        rows.append({"name": case["name"], "delay_minutes": number(case["delay_minutes"]),
                     "delay_source": case.get("delay_source", "explicit scenario; not simulated here"),
                     "result": compare(case["baseline"], case["candidate"], case["articles"])})
        current_scope = accounting_scope(rows[-1]["result"])
        if scope is not None and scope != current_scope:
            raise ValueError("sensitivity scenarios need identical accounting scope and VAT basis")
        scope = current_scope
    savings = [decimal(r["result"]["groups"]["cash"]["savings"]) for r in rows
               if r["result"]["groups"]["cash"]["savings"] is not None]
    sign_change = any(s < 0 for s in savings) and any(s > 0 for s in savings)
    frontiers = []
    if sign_change and frontier_quantity_key:
        for row in rows:
            result = row["result"]
            matched = [a for a in result["articles"] if a["accounting_kind"] == "cash"
                       and a["quantity_key"] == frontier_quantity_key]
            other = [a["savings"] for a in result["articles"] if a["accounting_kind"] == "cash" and a not in matched]
            if len(matched) == 1 and summed(other) is not None:
                delta = result["natural_deltas"][frontier_quantity_key]["delta"]
                if delta is not None:
                    frontiers.append({"scenario": row["name"], "quantity_key": frontier_quantity_key,
                                     "period": {"start": result["boundary"]["period_start"],
                                                "end": result["boundary"]["closure_end"]},
                                     **price_frontier(delta, summed(other),
                                     matched[0]["baseline"]["rate_unit"], "scenario quantities and other costs")})
    complete = bool(rows) and len(savings) == len(rows)
    result = {"scenarios": rows, "sign_change": sign_change, "frontiers": frontiers,
              "evaluation_mode": rows[0]["result"]["evaluation_mode"] if rows else "unknown",
              "status": status_of(row["result"]["status"] for row in rows) if rows else "unknown",
              "cash_savings_range": {"min": number(min(savings)) if complete else None,
                                     "max": number(max(savings)) if complete else None,
                                     "known_min": number(min(savings)) if savings else None,
                                     "known_max": number(max(savings)) if savings else None,
                                     "complete": complete, "currency": "RUB"},
              "alternatives": [r["name"] for r in rows], "release_authorized": False}
    if has_probabilities:
        terms = []
        for row, probability in zip(rows, probabilities):
            row["probability"] = number(probability)
            saving = row["result"]["groups"]["cash"]["savings"]
            terms.append(Decimal(0) if probability == 0 else None if saving is None else product(decimal(saving), probability))
        result["expected_savings"] = {"amount": number(None if any(t is None for t in terms) else exact_sum(terms)),
                                      "known_subtotal": number(exact_sum(t for t in terms if t is not None)),
                                      "currency": "RUB", "status": "partial" if any(t is None for t in terms) else "assumed",
                                      "source": "explicit scenario probabilities; no inferred probabilities"}
    return result


def aggregate_days(results):
    results = sorted(results, key=lambda r: instant(r["boundary"]["period_start"]))
    previous_end = None
    basis = None
    scope = None
    operations = {"baseline": set(), "candidate": set()}
    for result in results:
        if any(result[field] != results[0][field] for field in
               ("model_version", "evaluation_model_version", "evaluation_mode")):
            raise ValueError("whole days must share model version and evaluation mode")
        boundary = result["boundary"]
        start, end, close = map(instant, (boundary["period_start"], boundary["period_end"], boundary["closure_end"]))
        if end - start != timedelta(days=1) or close != end or boundary["obligations_closed"] is not True:
            raise ValueError("only whole days with obligations closed inside the day can be aggregated")
        if start.hour not in {0, 3} or start.minute or start.second or start.microsecond:
            raise ValueError("whole calendar days start at 00:00; service days at 03:00")
        current_basis = (start.hour, start.utcoffset())
        if len({t.utcoffset() for t in (start, end, close)}) != 1 or basis is not None and basis != current_basis:
            raise ValueError("whole day timezone and calendar/service basis must match")
        basis = current_basis
        current_scope = accounting_scope(result)
        if scope is not None and scope != current_scope:
            raise ValueError("whole days need identical accounting scope and VAT basis")
        scope = current_scope
        for side in operations:
            for event in result["events"][side]:
                if event["operation_id_explicit"]:
                    if event["operation_id"] in operations[side]:
                        raise ValueError("global operation identity repeated across whole days")
                    operations[side].add(event["operation_id"])
        if previous_end is not None and start < previous_end:
            raise ValueError("whole day boundaries overlap")
        previous_end = end
    rows = [row for result in results for row in result["articles"]]
    groups = groups_for(rows)
    for kind in KINDS:
        for field in ("baseline", "candidate", "savings"):
            groups[kind][field] = number(summed([r["groups"][kind][field] for r in results]))
        base, saving = groups[kind]["baseline"], groups[kind]["savings"]
        groups[kind]["savings_pct"] = number(ratio(product(Decimal(100), decimal(saving)), decimal(base))) if base not in {None, "0"} and saving is not None else None
        groups[kind]["status"] = status_of(r["groups"][kind]["status"] for r in results)
    return {"days": len(results), "boundaries": [r["boundary"] for r in results], "day_results": copy.deepcopy(results),
            "evaluation_mode": results[0]["evaluation_mode"] if results else "unknown",
            "model_version": MODEL_VERSION, "day_basis": "service" if basis and basis[0] == 3 else "calendar" if basis else None,
            "groups": groups, "status": status_of(r["status"] for r in results),
            "annualized": False, "release_authorized": False}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def maintenance_resources(rules_path, wagons, review_path, period=None):
    rules = read_csv(rules_path)
    review = json.loads(Path(review_path).read_text(encoding="utf-8-sig"))
    expected = {(source["series"], row["maintenance_type"], key): row
                for key, source in review["sources"].items() for row in source["rows"]}
    observed = [(rule["series"], rule["maintenance_type"], rule["source_id"]) for rule in rules]
    if len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise ValueError("maintenance rule/photo keys mismatch")
    for key, rule in zip(observed, rules):
        match = expected[key]
        nominal, lower, upper = map(decimal, (rule["nominal"], rule["lower"], rule["upper"]))
        copied = tuple(decimal(v) * 1000 for v in (
            match["nominal"], match["nominal"] - match["lower_delta"], match["nominal"] + match["upper_delta"]))
        if rule["unit"] != "km" or not 0 < lower <= nominal <= upper or (nominal, lower, upper) != copied:
            raise ValueError("maintenance rule/photo values or unit mismatch")
    rows, ids = [], set()
    for wagon in wagons:
        if not wagon["wagon_id"] or wagon["wagon_id"] in ids:
            raise ValueError("wagon ids must be unique")
        ids.add(wagon["wagon_id"])
        series_rules = [r for r in rules if r["series"] == wagon["series"]]
        if not series_rules:
            raise ValueError("unknown wagon series")
        planned = Quantity(wagon.get("planned_km"), "km", wagon["source"], wagon.get("status", "assumed"))
        for rule in series_rules:
            nominal, lower, upper = map(decimal, (rule["nominal"], rule["lower"], rule["upper"]))
            since = wagon.get("since_maintenance_km", {}).get(rule["maintenance_type"])
            since = None if since is None else Quantity(since, "km", wagon["source"], wagon.get("status", "assumed")).amount
            complete = planned.status in {"confirmed", "assumed"} and wagon.get("status", "assumed") in {"confirmed", "assumed"}
            after = exact_sum((since, planned.amount)) if complete and since is not None and planned.amount is not None else None
            rows.append({"wagon_id": wagon["wagon_id"], "series": rule["series"],
                         "maintenance_type": rule["maintenance_type"], "nominal_km": number(nominal),
                         "lower_km": number(lower), "upper_km": number(upper),
                         "since_maintenance_km": number(since), "planned_km": planned.as_dict(),
                         "planned_resource_fraction": number(ratio(planned.amount, nominal)) if complete and planned.amount is not None else None,
                         "resource_fraction_unit": "1", "period": period,
                         "remaining_nominal_km": number(difference(nominal, after)) if after is not None else None,
                         "history_status": "unknown" if since is None else wagon.get("status", "assumed"),
                         "source_id": rule["source_id"], "source_locator": rule["source_locator"],
                         "source_verification": rule["source_verification"], "photo_transfer_verification": "confirmed",
                         "regulation_applicability": "unknown", "upper_is_operating_target": False,
                         "monetary_amount": None, "note": rule["note"], "operational_status": "unknown"})
    eo = {source["series"]: {**source["eo"], "source_id": key,
                             "source_verification": "visual_review_chat_attachment",
                             "upper_is_operating_target": False,
                             "regulation_applicability": "unknown", "depot_time_accounting": "unknown"}
          for key, source in review["sources"].items() if "eo" in source}
    return {"rows": rows, "rules": [{**rule, "photo_transfer_verification": "confirmed",
                                    "regulation_applicability": "unknown"} for rule in rules],
            "model_version": MODEL_VERSION, "period": period, "eo": eo, "regulation_date": review["regulation_date"],
            "valid_from": review["valid_from"], "regulation_applicability": "unknown",
            "release_authorized": False, "price_status": "unknown"}


def natural_report(transport_path):
    """Reuse the prepared MO-I table; never parse the original HTML a second way."""
    data = read_csv(transport_path)
    periods = {}
    for row in data:
        if row["line_id"] != "1":
            raise ValueError("transport report must contain line 1 only")
        key = (row["period_start"], row["period_end"])
        metrics = periods.setdefault(key, {})
        if row["metric"] in metrics:
            raise ValueError("duplicate transport metric")
        metrics[row["metric"]] = row
    output = []
    aliases = {"car_km_passenger": "passenger_car_km", "car_km_zero": "nonpassenger_car_km",
               "car_km_total": "total_car_km", "car_km_depot": "depot_car_km_subset",
               "car_km_test": "test_car_km_subset", "train_km": "reported_train_km",
               "train_hours": "reported_train_hours", "moving_hours": "moving_hours", "idle_hours": "stopped_hours"}
    for (start, end), metrics in sorted(periods.items()):
        quantities = {}
        for metric, name in aliases.items():
            row = metrics[metric]
            expected_unit = "car_km" if metric.startswith("car_km") else "train_km" if metric == "train_km" else "train_h"
            if row["unit"] != expected_unit:
                raise ValueError("transport unit mismatch")
            quantities[name] = {**Quantity(row["value"], row["unit"], row["source_id"]).as_dict(),
                                "original_label": row["original_label"], "raw_value": row["raw_value"],
                                "source_locator": row["source_locator"], "source_path": row["source_path"],
                                "source_url": row["source_url"], "quality_flag": row["quality_flag"],
                                "verification_scope": "value transcribed from MO-I; source inconsistencies retained"}
        for name, unit in {"passenger_train_km": "train_km", "nonpassenger_train_km": "train_km",
                           "standby_hours": "train_h", "paid_person_hours": "person_h",
                           "purchased_energy": "kWh", "entry_events": "event", "withdrawal_events": "event"}.items():
            quantities[name] = Quantity(None, unit, "not separately supplied by MO-I").as_dict()
        value = lambda name: decimal(metrics[name]["value"])
        checks = {"car_km_total_delta": number(difference(exact_sum((value("car_km_passenger"), value("car_km_zero"))), value("car_km_total"))),
                  "train_hours_delta": number(difference(exact_sum((value("moving_hours"), value("idle_hours"))), value("train_hours"))),
                  "reported_train_km_x8_minus_passenger_car_km": number(difference(product(value("train_km"), Decimal(8)), value("car_km_passenger")))}
        output.append({"period_start": start, "period_end": end, "period_label": next(iter(metrics.values()))["period_label"],
                       "quantities": quantities, "series_breakdown": None, "checks": checks,
                       "cash_amount": None, "cash_status": "unknown", "rate_source": "not supplied",
                       "note": "Depot/test mileage are subsets of zero mileage. Stopped hours are not assumed to be depot standby."})
    return {"schema_version": "1.0", "model_version": MODEL_VERSION, "evaluation_mode": "observed_aggregate",
            "line_id": 1, "periods": output, "maintenance_history_status": "unknown", "release_authorized": False}


def load_costs(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    unsupported = set(config) - {"schema_version", "currency", "model_version", "note", "articles"}
    if unsupported:
        raise ValueError(f"unsupported cost configuration fields: {', '.join(sorted(unsupported))}")
    if config["currency"] != "RUB":
        raise ValueError("currency unit must be RUB; convert rates explicitly")
    if config.get("model_version", MODEL_VERSION) != MODEL_VERSION:
        raise ValueError("cost configuration model version mismatch")
    articles = [construct(Article, {**row, "covered_components": tuple(row["covered_components"])}) for row in config["articles"]]
    validate_articles(articles)
    return articles


def evaluation_from_dict(data):
    data = dict(data)
    data["quantities"] = {key: construct(Quantity, q) for key, q in data["quantities"].items()}
    data["events"] = tuple(construct(Event, {**e, "quantities": {key: construct(Quantity, q) for key, q in e["quantities"].items()}})
                           for e in data.get("events", []))
    if isinstance(data.get("required_cash_articles"), str):
        raise ValueError("required_cash_articles must be a list of article names")
    data["required_cash_articles"] = tuple(data.get("required_cash_articles", ()))
    return construct(Evaluation, data)


def construct(cls, data):
    unsupported = set(data) - {field.name for field in fields(cls)}
    if unsupported:
        raise ValueError(f"unsupported {cls.__name__} fields: {', '.join(sorted(unsupported))}")
    return cls(**data)


def write_json(path, data):
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                            prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summary_markdown(reports):
    lines = ["Экономические отчёты", "", "Учебные суммы ниже относятся только к указанным статьям. Реальные ставки неизвестны.", "",
             "| Сценарий | База, RUB | Кандидат, RUB | Разница, RUB | Статус | Режим |",
             "|---|---:|---:|---:|---|---|"]
    for row in reports["fixture_scenarios.json"]["scenarios"]:
        result, label = row["result"], row["name"]
        cash = result["groups"]["cash"]
        values = [cash[key] if cash[key] is not None else "неизвестно" for key in ("baseline", "candidate", "savings")]
        lines.append(f"| {label} | {' | '.join(values)} | {result['status']} | {result['evaluation_mode']} |")
    days = reports["fixture_days.json"]
    lines.extend(["", f"Целых суток: {days['days']}; учебная разница: {days['groups']['cash']['savings']} RUB. Годовой эффект не рассчитывается.",
                  "", "| Период | Пассажирские вагоно-км | Нулевые вагоно-км | Денежная оценка | Режим |",
                  "|---|---:|---:|---|---|"])
    for period in reports["natural_transport.json"]["periods"]:
        quantities = period["quantities"]
        lines.append(f"| {period['period_start']}–{period['period_end']} | {quantities['passenger_car_km']['amount']} | "
                     f"{quantities['nonpassenger_car_km']['amount']} | unknown | observed_aggregate |")
    lines.extend(["", "Все 14 интервалов перенесены верно по фото. Актуальность регламента, история обслуживания и допуск составов неизвестны.",
                  "ЕО Балтийца: один раз в сутки; срок действия осмотра 20 ±4 часа от выдачи на линию с площадки электродепо.",
                  "Учёт времени при заходах в депо не подтверждён. Верхний допуск не принят как цель эксплуатации.", ""])
    return "\n".join(lines)


def prepare_reports(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    protected = [root / name for name in ("data/raw", "docs/reference", "data/processed/metro", "data/processed/metro_repeat")]
    if output == root or any(output == path or path in output.parents for path in protected):
        raise ValueError("output must be separate from protected input directories")
    output.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix="economics-", dir=output.parent) as folder:
            staging = Path(folder).resolve()
            count = _prepare_reports(root, staging)
            for path in sorted(staging.iterdir(), key=lambda p: (p.name == "build_manifest.json", p.name)):
                path.replace(output / path.name)
        return count
    except (ValueError, TypeError, KeyError, OSError, ArithmeticError) as exc:
        write_json(output / "build_manifest.json", {"status": "error", "model_version": MODEL_VERSION, "message": str(exc)})
        write_text(output / "summary.md", "Сборка экономического отчёта завершилась ошибкой. Результаты этой папки не прошли текущую проверку.\n")
        raise


def _prepare_reports(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    transport = root / "data/processed/metro/transport_work.csv"
    rules = root / "data/processed/metro/maintenance_rules.csv"
    review = root / "docs/reference/maintenance_photo_review.json"
    costs = root / "config/costs/default.json"
    fixture_path = root / "examples/economics/withdrawal.json"
    snapshot = json.loads((transport.parent / "build_manifest.json").read_text(encoding="utf-8"))
    for path in (transport, rules, transport.parent / "quality_report.json"):
        if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot["outputs"][path.name]:
            raise ValueError(f"prepared input hash mismatch: {path.name}; run metro_data check")
    if hashlib.sha256(review.read_bytes()).hexdigest() != snapshot["inputs"][review.relative_to(root).as_posix()]:
        raise ValueError("photo review input hash mismatch; run metro_data check")
    load_costs(costs)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = []
    for case in fixture["scenarios"]:
        cases.append({**case, "baseline": evaluation_from_dict(fixture["baseline"]),
                      "candidate": evaluation_from_dict(case["candidate"]),
                      "articles": [Article(**{**a, "covered_components": tuple(a["covered_components"])}) for a in case["articles"]]})
    reports = {
        "natural_transport.json": natural_report(transport),
        "maintenance_resources.json": {"evaluation_mode": "observed_registry", **maintenance_resources(rules, [], review)},
        "fixture_scenarios.json": compare_scenarios(cases, "passenger_train_km"),
        "fixture_resources.json": {"evaluation_mode": "fixture", **maintenance_resources(
            rules, fixture["wagons"], review, {"start": fixture["baseline"]["period_start"],
                                              "end": fixture["baseline"]["closure_end"]})},
        "fixture_threshold.json": {"evaluation_mode": "fixture", **break_even(**fixture["threshold"])},
        "fixture_days.json": aggregate_days([
            compare(evaluation_from_dict(day["baseline"]), evaluation_from_dict(day["candidate"]),
                    [Article(**{**a, "covered_components": tuple(a["covered_components"])}) for a in day["articles"]])
            for day in fixture["days"]]),
    }
    reports["maintenance_resources.json"]["note"] = "Wagon history and planned mileage are unavailable; no per-wagon resource balance computed."
    reports["summary.md"] = summary_markdown(reports)
    for name, report in reports.items():
        write_text(output / name, report) if isinstance(report, str) else write_json(output / name, report)
    inputs = [transport, rules, review, costs, fixture_path, root / "economics.py", root / "requirements.txt"]
    inputs.extend(root / "data/processed/metro" / name for name in ("build_manifest.json", "quality_report.json"))
    manifest = {"model_version": MODEL_VERSION, "status": "success",
                "inputs": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                "outputs": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in sorted(reports)},
                "fixture_is_simulator_integration": False}
    write_json(output / "build_manifest.json", manifest)
    return len(reports) + 1


def check_reports(root, output):
    root, output = Path(root), Path(output)
    manifest = json.loads((output / "build_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise ValueError("last economic build failed or is incomplete")
    if manifest["model_version"] != MODEL_VERSION:
        raise ValueError("economic model version mismatch")
    for name, expected in manifest["inputs"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input hash mismatch: {name}")
    for name, expected in manifest["outputs"].items():
        path = output / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Output hash mismatch: {name}")
    return {"files": len(manifest["outputs"]) + 1, "hashes": "pass", "model_version": MODEL_VERSION,
            "fixture_is_simulator_integration": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "compare"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--costs", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "data/processed/economics"
    if args.command == "compare" and not all((args.baseline, args.candidate, args.costs, args.output)):
        parser.error("compare requires --baseline, --candidate, --costs and --output (JSON file)")
    if args.command == "compare" and output.resolve() in {p.resolve() for p in (args.baseline, args.candidate, args.costs)}:
        parser.error("output must differ from every input file")
    try:
        if args.command == "prepare":
            count = prepare_reports(args.root, output)
            print(f"Prepared {count} economic report files: {output}")
        elif args.command == "check":
            print(json.dumps(check_reports(args.root, output), indent=2))
        else:
            plans = [evaluation_from_dict(json.loads(p.read_text(encoding="utf-8-sig"))) for p in (args.baseline, args.candidate)]
            write_json(output, compare(*plans, load_costs(args.costs)))
    except (ValueError, TypeError, KeyError, OSError, ArithmeticError) as exc:
        if args.command == "compare":
            write_json(output, {"status": "error", "model_version": MODEL_VERSION, "message": str(exc)})
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
