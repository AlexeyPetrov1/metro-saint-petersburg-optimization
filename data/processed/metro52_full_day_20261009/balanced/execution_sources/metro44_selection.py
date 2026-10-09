"""Pure future-risk selection and fixed-fleet TO1 resource accounting.

Rows retain rolling.py's trial schema. Optional ``future_exposure`` overrides
``policy_exposure``; ``past_exposure`` is evidence only and is never ranked.
``committed_per_train`` must include the entire closed schedule and all 68
trains, including inactive trains with zero kilometers.
"""

from __future__ import annotations

from decimal import Decimal
from math import inf, isfinite


TO1_RUB_PER_KM = {"yubileyny": Decimal("111000") / 15000,
                  "baltiets": Decimal("138000") / 15000}
TYPE_ALIASES = {"юбилейный": "yubileyny", "балтиец": "baltiets"}


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def resource_value(per_train, *, fixed_fleet_size=68):
    """Kilometers and maintenance resource use; no avoided cash is inferred."""
    if isinstance(fixed_fleet_size, bool) or fixed_fleet_size != 68:
        raise ValueError("the fixed experiment fleet must contain 68 trains")
    rows = list(per_train)
    if len(rows) != fixed_fleet_size:
        raise ValueError("resource accounting requires all 68 fixed-fleet trains")
    ids = [row["train_id"] for row in rows]
    if any(not isinstance(t, str) or not t for t in ids) or len(set(ids)) != len(ids):
        raise ValueError("train identities must be unique and nonempty")
    details = []
    for row in rows:
        kind = str(row["train_type"]).casefold()
        kind = TYPE_ALIASES.get(kind, kind)
        if kind not in TO1_RUB_PER_KM:
            raise ValueError(f"unknown train type: {row['train_type']}")
        km = Decimal(str(_number(row["km"], "km")))
        rate = TO1_RUB_PER_KM[kind]
        details.append(dict(train_id=row["train_id"], train_type=kind, km=float(km),
                            to1_rub_per_km=float(rate), to1_resource_equivalent_rub=float(km * rate)))
    km = sum(Decimal(str(row["km"])) for row in details)
    wear = sum(Decimal(str(row["to1_resource_equivalent_rub"])) for row in details)
    return dict(fixed_fleet_size=fixed_fleet_size, train_km=float(km),
                mean_km_per_fixed_train=float(km / fixed_fleet_size), per_train=details,
                to1_resource_equivalent_rub=float(wear), fare_rub_per_paid_trip=50.9,
                cash_saving_rub=None, lost_revenue_rub=None, net_profit_effect_rub=None,
                resource_equivalent_is_cash_saving=False)


def economy_delta(baseline_per_train, changed_per_train, *, service_admissible):
    baseline, changed = resource_value(baseline_per_train), resource_value(changed_per_train)
    left = {row["train_id"]: row for row in baseline["per_train"]}
    right = {row["train_id"]: row for row in changed["per_train"]}
    if left.keys() != right.keys():
        raise ValueError("fixed fleet train identities changed")
    if any(left[t]["train_type"] != right[t]["train_type"] for t in left):
        raise ValueError("fixed fleet train type changed")
    reduction = baseline["train_km"] - changed["train_km"]
    details = [dict(train_id=t, train_type=left[t]["train_type"],
                    baseline_km=left[t]["km"], candidate_km=right[t]["km"],
                    reduction_km=left[t]["km"] - right[t]["km"],
                    to1_resource_equivalent_rub=left[t]["to1_resource_equivalent_rub"]
                    - right[t]["to1_resource_equivalent_rub"]) for t in sorted(left)]
    return dict(observed_reduction_train_km=reduction, observed_reduction_km_per_fixed_train=reduction / 68,
                accepted_saving_train_km=max(0, reduction) if service_admissible is True else None,
                to1_resource_equivalent_rub=baseline["to1_resource_equivalent_rub"]
                - changed["to1_resource_equivalent_rub"], per_train=details,
                service_admissible=service_admissible is True, resource_equivalent_is_cash_saving=False,
                fare_rub_per_paid_trip=50.9, cash_saving_rub=None, lost_revenue_rub=None,
                net_profit_effect_rub=None)


def _exposure(trial):
    return trial.get("future_exposure", trial.get("policy_exposure", {}))


def _pending(obligations):
    for key in ("pending_at_end", "pending", "unboarded"):
        if key in obligations:
            return _number(obligations[key], key)
    return 0 if obligations.get("complete") is True else inf


def _risk_number(value, name):
    return inf if value is None or value == inf else _number(value, name)


def ranking_future_risk(row, *, hard_limit=160):
    """Worst future scenario: hard risk, overload share/time, backlog, waiting.

    Max-load saturation at 160 cannot conceal a difference in exposure time.
    Unknown future evidence ranks as infinity and cannot become mitigation.
    """
    trials = row.get("trials", [])
    if not trials:
        return (inf,) * 7
    risks = []
    for trial in trials:
        exposure = _exposure(trial)
        total = exposure.get("global", {})
        summaries = [total, *exposure.get("windows", [])]
        shares = []
        for summary in summaries:
            share = summary.get("exceptional_time_share")
            if share is None:
                share = 0 if summary.get("passenger_car_seconds") == 0 else inf
            else:
                share = _number(share, "future exceptional share")
                if share > 1:
                    raise ValueError("future exceptional share must be at most one")
            shares.append(share)
        peak = _risk_number(total.get("max_load", row.get("metrics", {}).get("max_car_load")), "max_load")
        cohort = trial.get("mature_cohort", {})
        unserved = _risk_number(cohort.get("unboarded", 0 if cohort.get("complete") is True else None), "unboarded")
        obligations = _pending(trial.get("initial_obligations", {}))
        risks.append((max(0, peak - hard_limit),
                      _number(total.get("hard_limit_car_seconds", 0), "hard exposure"),
                      max(shares, default=inf),
                      _risk_number(total.get("exceptional_car_seconds"), "exceptional duration"),
                      unserved + obligations,
                      _number(trial.get("metrics", {}).get("waiting_passenger_seconds",
                                  row.get("metrics", {}).get("waiting_passenger_seconds", 0)), "waiting"), peak))
    return tuple(max(r[i] for r in risks) for i in range(7))


def _checked(row):
    return (row.get("status") == "evaluated" and row.get("passengers_checked") is True
            and row.get("physical_passed") is True and bool(row.get("trials"))
            and all(trial.get("headways", {}).get("passed") is True for trial in row["trials"]))


def _waits(row):
    cohorts = [trial.get("mature_cohort", {}) for trial in row.get("trials", [])]
    means, p95s = [], []
    for cohort in cohorts:
        if cohort.get("arrived") == 0 and cohort.get("complete") is True:
            means.append(0); p95s.append(0)
        elif cohort.get("mean_wait_seconds") is None or cohort.get("p95_wait_seconds") is None:
            return None
        else:
            means.append(_number(cohort["mean_wait_seconds"], "mean wait"))
            p95s.append(_number(cohort["p95_wait_seconds"], "p95 wait"))
    return (max(means), max(p95s)) if means else None


def _feasible(row, risk, share_limit):
    return (row.get("service_passed") is True and risk[0] == risk[1] == risk[4] == 0
            and risk[2] <= share_limit and all(
                (_exposure(t).get("admissible") is True or
                 _exposure(t).get("global", {}).get("passenger_car_seconds") == 0)
                and t.get("mature_cohort", {}).get("complete") is True
                and t.get("initial_obligations", {}).get("complete") is True
                for t in row["trials"]))


def choose_candidate(rows, *, mean_wait_budget_s=None, p95_wait_budget_s=None,
                     timed_out=False, search_incomplete=False, exceptional_share_limit=0.01,
                     hard_limit=160):
    """Select a service reference or budget-bounded resource plan; never approve.

    Without both explicit waiting budgets, feasible service can establish a
    reference, but cost/kilometer improvements are not claimed. Mitigation is
    returned separately and always remains inadmissible for release.
    """
    rows = list(rows)
    ids = [row["candidate_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate identities must be unique")
    for value in (mean_wait_budget_s, p95_wait_budget_s):
        if value is not None:
            _number(value, "wait budget")
    bounded = mean_wait_budget_s is not None and p95_wait_budget_s is not None
    checked = [(r, ranking_future_risk(r, hard_limit=hard_limit)) for r in rows if _checked(r)]
    checked = [(r, risk) for r, risk in checked if all(isfinite(v) for v in risk) and risk[0] == risk[1] == 0]
    feasible = [(r, risk, _waits(r)) for r, risk in checked if _feasible(r, risk, exceptional_share_limit)]
    feasible = [(r, risk, wait) for r, risk, wait in feasible if wait is not None
                and (mean_wait_budget_s is None or wait[0] <= mean_wait_budget_s)
                and (p95_wait_budget_s is None or wait[1] <= p95_wait_budget_s)]
    common = dict(selected_candidate_id=None, mitigation_candidate_id=None,
                  fallback_candidate_id="KEEP" if "KEEP" in ids else None,
                  release_authorized=False, economic_optimality=False, economic_improvement_claimed=False,
                  search_complete=not timed_out and not search_incomplete,
                  waiting_budgets=dict(mean_seconds=mean_wait_budget_s, p95_seconds=p95_wait_budget_s),
                  past_exposure_is_ranked=False)
    if not feasible:
        mitigation = min(checked, key=lambda pair: (pair[1], pair[0]["candidate_id"] != "KEEP",
                                                   pair[0]["candidate_id"]), default=None)
        return dict(common, status="evaluation_timeout" if timed_out else "no_feasible_candidate",
                    mitigation_candidate_id=mitigation[0]["candidate_id"] if mitigation else None,
                    ranking_future_risk=list(mitigation[1]) if mitigation else None)
    if not bounded:
        chosen, risk, wait = min(feasible, key=lambda item: (item[1], item[2], item[0]["candidate_id"] != "KEEP",
                                                            item[0]["candidate_id"]))
        return dict(common, status="feasible_service_reference", selected_candidate_id=chosen["candidate_id"],
                    ranking_future_risk=list(risk), future_wait_seconds=dict(mean=wait[0], p95=wait[1]),
                    service_preservation_checked=False)
    priced = []
    for row, risk, wait in feasible:
        if "committed_per_train" in row:
            resources = resource_value(row["committed_per_train"])
            priced.append((resources["to1_resource_equivalent_rub"], resources["train_km"], wait,
                           row["candidate_id"] != "KEEP", row["candidate_id"], row, risk, resources))
    if not priced:
        chosen, risk, wait = min(feasible, key=lambda item: (item[1], item[2], item[0]["candidate_id"] != "KEEP",
                                                            item[0]["candidate_id"]))
        return dict(common, status="feasible_service_reference", selected_candidate_id=chosen["candidate_id"],
                    ranking_future_risk=list(risk), service_preservation_checked=True,
                    resource_accounting_complete=False)
    _, _, wait, _, _, chosen, risk, resources = min(priced, key=lambda item: item[:5])
    return dict(common, status="search_incomplete" if timed_out or search_incomplete else "conditional_resource_policy",
                selected_candidate_id=chosen["candidate_id"], ranking_future_risk=list(risk),
                service_preservation_checked=True, resource_accounting_complete=True, resources=resources,
                future_wait_seconds=dict(mean=wait[0], p95=wait[1]),
                economic_comparison_scope="validated candidates under explicit same-horizon waiting budgets")
