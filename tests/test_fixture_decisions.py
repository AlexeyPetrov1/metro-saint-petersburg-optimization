from metro_replay import fixture_examples


def test_simulated_hourly_forecasts_drive_keep_add_withdraw_and_refusal(tmp_path):
    results=fixture_examples(tmp_path)
    assert results["steady"]["selection"]["selected_candidate_id"] == "KEEP:plan"
    assert results["growth"]["selection"]["selected_candidate_id"] == "ADD:R01"
    assert results["low"]["selection"]["selected_candidate_id"] == "WITHDRAW:T01"
    assert results["saturated"]["selection"]["status"] == "no_feasible_candidate"
    assert all(r["evaluation_mode"]=="fixture" for r in results.values())

