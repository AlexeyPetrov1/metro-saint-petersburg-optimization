from metro_ml_report import case_conclusion


def result(admissible, km, wait):
    return {'metrics': dict(actual_service_admissible=admissible, train_km=km, mean_wait_seconds=wait)}


def test_resource_reduction_with_policy_failure_is_not_a_winner():
    case = {'strategies': {'MEDIAN': result(False, 80, 50), 'ML': result(True, 100, 60)}}
    assert case_conclusion(case) == 'ML соблюдает ограничения; медиана нарушает'


def test_no_winner_is_invented_for_a_resource_waiting_tradeoff():
    case = {'strategies': {'MEDIAN': result(True, 80, 70), 'ML': result(True, 100, 60)}}
    assert case_conclusion(case) == 'Размен между пробегом и ожиданием'


def test_equal_actions_and_metrics_are_reported_as_equal():
    case = {'strategies': {'MEDIAN': result(True, 100, 60), 'ML': result(True, 100, 60)}}
    assert case_conclusion(case) == 'Результаты совпали'
