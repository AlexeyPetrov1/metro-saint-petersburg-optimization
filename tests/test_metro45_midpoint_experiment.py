from metro45_midpoint_experiment import run_midpoint, assess


def metrics(mean=70,p95=140,complete=True):
    return dict(mean_wait_seconds=mean,p95_wait_seconds=p95,actual_service_admissible=True,
        all_service_arrivals_completed=True,movement_obligations_closed=True,
        cohort={'complete':complete},mature_cohort={'complete':complete},fixed_fleet_size=68)


def test_only_the_predeclared155_arm_is_run_without_modifying_the_original_driver():
    calls=[]
    def trial(output,**kwargs):
        calls.append((output,kwargs))
        return {'status':'evaluated'}
    assert run_midpoint('fresh-output',trial=trial)=={'status':'evaluated'}
    assert calls==[('fresh-output',dict(scenario='S1',mode='ML',policy='155',bridge_seconds=480))]


def test_p95_regression_prevents_strict_acceptance_but_keeps_explicit_development_tradeoff():
    result=assess(metrics(),metrics(p95=138),{'saving_train_km_lower_bound':40},physical=True)
    assert not result['strict_no_wait_regression']
    assert result['research_90_180_budget_passed']
    assert result['research_tradeoff_eligible']
    assert not result['release_authorized'] and result['cash_savings_rub'] is None


def test_incomplete_cohort_or_physical_failure_prevents_service_and_resource_promotion():
    bounds={'saving_train_km_lower_bound':1000}
    assert not assess(metrics(complete=False),metrics(),bounds,physical=True)['research_tradeoff_eligible']
    assert not assess(metrics(),metrics(),bounds,physical=False)['research_tradeoff_eligible']


def test_negative_window_bound_cannot_be_promoted_even_with_better_waiting():
    result=assess(metrics(mean=60,p95=120),metrics(),{'saving_train_km_lower_bound':-2},physical=True)
    assert result['strict_no_wait_regression']
    assert not result['strict_window_resource_improvement']
