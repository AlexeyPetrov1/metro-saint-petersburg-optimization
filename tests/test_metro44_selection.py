import copy

import pytest

from metro44_selection import choose_candidate, economy_delta, ranking_future_risk, resource_value


def fleet(km=10):
    return [{"train_id": f"T{i:02}", "train_type": "Baltiets" if i < 37 else "Yubileyny", "km": km}
            for i in range(68)]


def candidate(name="KEEP", *, share=0.005, duration=50, mean=80, p95=150,
              unserved=0, max_load=160, km=10, admissible=True):
    summary = {"exceptional_time_share": share, "exceptional_car_seconds": duration,
               "passenger_car_seconds": 10000, "hard_limit_car_seconds": 0, "max_load": max_load}
    exposure = {"admissible": admissible, "status": "pass" if admissible else "fail",
                "global": summary, "windows": [summary.copy()]}
    trial = {"policy_exposure": exposure, "headways": {"passed": True},
             "mature_cohort": {"complete": unserved == 0, "arrived": 1000, "unboarded": unserved,
                               "mean_wait_seconds": mean, "p95_wait_seconds": p95},
             "initial_obligations": {"complete": True, "pending_at_end": 0}}
    return {"candidate_id": name, "status": "evaluated", "passengers_checked": True,
            "physical_passed": True, "policy_passed": admissible, "service_passed": unserved == 0,
            "metrics": {"max_car_load": max_load, "waiting_passenger_seconds": mean * 1000},
            "trials": [trial], "committed_per_train": fleet(km)}


def choose(rows, **kwargs):
    return choose_candidate(rows, mean_wait_budget_s=100, p95_wait_budget_s=200, **kwargs)


def test_equal_160_peak_prefers_less_future_overload_even_if_more_mileage():
    bad = candidate("LOW_KM", share=0.08, duration=800, km=5, admissible=False)
    better = candidate("RELIEF", share=0.02, duration=200, km=15, admissible=False)
    result = choose([bad, better])
    assert result["mitigation_candidate_id"] == "RELIEF"
    assert result["selected_candidate_id"] is None
    assert result["release_authorized"] is False


def test_overload_risk_precedes_backlog_and_mitigation_is_never_approved():
    served = candidate("SERVED", share=0.03, duration=300, admissible=False)
    abandoned = candidate("ABANDONED", share=0.02, duration=200, unserved=20, admissible=False)
    assert ranking_future_risk(abandoned) < ranking_future_risk(served)
    result = choose([served, abandoned])
    assert result["mitigation_candidate_id"] == "ABANDONED"
    assert result["selected_candidate_id"] is None


def test_same_overload_risk_prefers_serving_existing_backlog():
    served = candidate("SERVED", share=0.03, duration=300, admissible=False)
    abandoned = candidate("ABANDONED", share=0.03, duration=300, unserved=20, admissible=False)
    assert ranking_future_risk(served) < ranking_future_risk(abandoned)


def test_common_past_failure_does_not_prevent_future_recovery():
    recovered = candidate("RECOVERY")
    recovered["past_exposure"] = {"admissible": False, "global": {"exceptional_time_share": 0.9}}
    before = copy.deepcopy(recovered)
    result = choose([recovered])
    assert result["selected_candidate_id"] == "RECOVERY"
    assert recovered == before
    assert result["economic_optimality"] is False


def test_worst_quarter_share_not_global_average_controls_risk():
    hidden = candidate("HIDDEN", share=0.005, duration=30, admissible=False)
    hidden["trials"][0]["policy_exposure"]["windows"][0]["exceptional_time_share"] = 0.08
    visible = candidate("VISIBLE", share=0.02, duration=200, admissible=False)
    assert choose([hidden, visible])["mitigation_candidate_id"] == "VISIBLE"


@pytest.mark.parametrize("field,value", [("passengers_checked", False), ("physical_passed", False),
                                         ("status", "evaluation_timeout")])
def test_unvalidated_candidates_never_become_mitigation(field, value):
    unsafe = candidate("UNVALIDATED", share=0, km=0)
    unsafe[field] = value
    keep = candidate("KEEP", share=0.03, admissible=False)
    result = choose([unsafe, keep])
    assert result["mitigation_candidate_id"] == "KEEP"
    assert result["selected_candidate_id"] is None


def test_missing_wait_budgets_can_establish_service_reference_but_not_economy():
    cheap = candidate("CHEAP", km=1)
    result = choose_candidate([candidate(), cheap])
    assert result["selected_candidate_id"] == "KEEP"
    assert result["status"] == "feasible_service_reference"
    assert result["economic_improvement_claimed"] is False


@pytest.mark.parametrize("field,value", [("mean_wait_seconds", 101), ("p95_wait_seconds", 201)])
def test_lower_mileage_cannot_trade_away_wait_budget(field, value):
    cheap = candidate("CHEAP", km=1)
    cheap["trials"][0]["mature_cohort"][field] = value
    assert choose([candidate(), cheap])["selected_candidate_id"] == "KEEP"


def test_type_specific_wear_can_prefer_more_total_kilometers():
    expensive = candidate("BALT_WORK", km=0)
    cheaper = candidate("YUB_WORK", km=0)
    expensive["committed_per_train"][0]["km"] = 100
    cheaper["committed_per_train"][37]["km"] = 110
    assert choose([expensive, cheaper])["selected_candidate_id"] == "YUB_WORK"


def test_initial_obligations_and_all_scenarios_must_pass():
    failed = candidate("CHEAP", km=1)
    failed["trials"].append(copy.deepcopy(failed["trials"][0]))
    failed["trials"][1]["initial_obligations"].update(complete=False, pending_at_end=30)
    assert choose([candidate(), failed])["selected_candidate_id"] == "KEEP"


def test_keep_fallback_does_not_approve_unknown_service():
    keep = candidate()
    keep["trials"] = []
    result = choose([keep])
    assert result["fallback_candidate_id"] == "KEEP"
    assert result["selected_candidate_id"] is None
    assert result["mitigation_candidate_id"] is None
    assert result["release_authorized"] is False


def test_missing_future_evidence_is_unknown_instead_of_invented_zero_risk():
    unknown = candidate("UNKNOWN", km=0)
    unknown["trials"][0].pop("mature_cohort")
    assert choose([unknown])["mitigation_candidate_id"] is None
    assert choose([unknown, candidate()])["selected_candidate_id"] == "KEEP"


def test_one_supplied_wait_budget_still_constrains_service_reference():
    slow = candidate("SLOW", mean=150, share=0.001, duration=10)
    result = choose_candidate([slow, candidate()], mean_wait_budget_s=100)
    assert result["selected_candidate_id"] == "KEEP"
    assert result["status"] == "feasible_service_reference"


def test_future_evidence_can_override_shared_past_failure_flag():
    recovered = candidate("RECOVERED")
    trial = recovered["trials"][0]
    trial["future_exposure"] = copy.deepcopy(trial["policy_exposure"])
    trial["policy_exposure"]["admissible"] = False
    trial["policy_exposure"]["global"]["exceptional_time_share"] = 0.9
    recovered["policy_passed"] = False
    assert choose([recovered])["selected_candidate_id"] == "RECOVERED"


def test_fixed_68_fleet_and_per_type_to1_resource_equivalent():
    value = resource_value(fleet())
    assert value["train_km"] == 680
    assert value["mean_km_per_fixed_train"] == 10
    assert value["to1_resource_equivalent_rub"] == pytest.approx(37 * 10 * 9.2 + 31 * 10 * 7.4)
    assert value["cash_saving_rub"] is None
    assert value["lost_revenue_rub"] is None
    assert value["fare_rub_per_paid_trip"] == 50.9


def test_missing_inactive_train_cannot_change_denominator():
    with pytest.raises(ValueError, match="68"):
        resource_value(fleet()[1:])
    duplicated = fleet()
    duplicated[-1]["train_id"] = duplicated[0]["train_id"]
    with pytest.raises(ValueError, match="unique"):
        resource_value(duplicated)


def test_negative_kilometers_and_unknown_type_are_rejected():
    bad = fleet()
    bad[0]["km"] = -1
    with pytest.raises(ValueError, match="km"):
        resource_value(bad)
    bad = fleet()
    bad[0]["train_type"] = "Other"
    with pytest.raises(ValueError, match="type"):
        resource_value(bad)


def test_reported_km_reduction_is_not_accepted_saving_under_overload():
    result = economy_delta(fleet(10), fleet(5), service_admissible=False)
    assert result["observed_reduction_train_km"] == 340
    assert result["observed_reduction_km_per_fixed_train"] == 5
    assert result["accepted_saving_train_km"] is None
    assert result["cash_saving_rub"] is None
    assert result["net_profit_effect_rub"] is None


def test_economy_preserves_train_identity_and_type():
    changed = fleet(5)
    changed[0]["train_type"] = "Yubileyny"
    with pytest.raises(ValueError, match="type"):
        economy_delta(fleet(10), changed, service_admissible=True)


def test_tied_resources_and_wait_prefer_keep_deterministically():
    rows = [candidate("OTHER"), candidate("KEEP")]
    assert choose(rows)["selected_candidate_id"] == "KEEP"
    assert choose(list(reversed(rows)))["selected_candidate_id"] == "KEEP"


def test_wait_budget_must_be_finite_and_nonnegative():
    with pytest.raises(ValueError, match="budget"):
        choose_candidate([], mean_wait_budget_s=float("nan"), p95_wait_budget_s=200)
