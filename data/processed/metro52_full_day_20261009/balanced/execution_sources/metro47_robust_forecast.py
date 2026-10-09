"""Fixed engineering demand margin; original model and actual policy stay intact."""

from copy import deepcopy
from decimal import Decimal, ROUND_CEILING

from forecasting import validate_bundle
from simulation import digest


VERSION = 'comfort-capacity-engineering-margin-v1'


def apply_safety_margin(bundle, decision_time, *, normal_per_car=150, design_per_car=120):
    """Inflate both controllers' planning entries by 150/120, without calibration.

    Apply the returned bundle to both the demand solver and exact forecast trials.
    Future actual passengers, model weights, and the actual 150/160 policy are
    unchanged. This fixed engineering reserve has no statistical guarantee.
    """
    if (type(normal_per_car) is not int or type(design_per_car) is not int
            or not 0 < design_per_car <= normal_per_car):
        raise ValueError('positive integer design capacity cannot exceed normal capacity')
    versions = bundle.get('source_versions', {})
    if 'engineering_safety_margin' in versions:
        raise ValueError('forecast safety margin already applied')
    if (versions.get('future_truth_intentionally_used') or versions.get('future_actual_used')
            or versions.get('operationally_available') is False):
        raise ValueError('causal operational forecasts required')
    stations = sorted({r['station_id'] for r in bundle['rows']})
    validate_bundle(bundle, stations, decision_time)
    factor = Decimal(normal_per_car)/Decimal(design_per_car)
    result = deepcopy(bundle)
    for row in result['rows']:
        row['predicted_count'] = int((Decimal(str(row['predicted_count']))*factor)
                                    .to_integral_value(rounding=ROUND_CEILING))
    result['feature_version'] += '+'+VERSION
    result['rounding'] = 'ROUND_CEILING per quarter after fixed normal/design capacity multiplier'
    result['source_versions']['engineering_safety_margin'] = dict(
        version=VERSION, factor=float(factor), normal_per_car=normal_per_car,
        design_per_car=design_per_car, original_bundle_digest=digest(bundle),
        original_feature_version=bundle['feature_version'], decision_time=decision_time,
        original_timestamps={k: bundle[k] for k in
                             ('issued_at', 'as_of', 'available_at', 'features_available_at')},
        basis='dispatcher comfort 120 per car versus normal 150; fixed engineering assumption',
        scope='planning turnstile entries for solver and exact forecast trial; model weights unchanged',
        target='turnstile_entries', statistical_coverage_guarantee=False,
        historical_residual_calibration_used=False, future_actual_used=False,
        parameters_selected_from_current_run=False, passenger_policy_changed=False,
        waiting_objective_added=False, original_forecast_artifact_changed=False,
        actual_150_160_and_exceptional_exposure_checks_required=True,
        finite_zero_tail_preserved=True, zero_forecast_underprediction_covered=False)
    validate_bundle(result, stations, decision_time)
    return result
