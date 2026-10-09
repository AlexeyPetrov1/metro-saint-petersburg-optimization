"""Quarter-hour feedback with causal forecasts and the existing exact validators."""

from datetime import datetime, timedelta
from time import monotonic

from goal43_policy import minimum_headway
from metro44_catalog import calculate_catalog
from metro_daily_run import read_forecast
from metro44_planner_bridge import rebuild_bridge_timetable
from metro45_adaptive import adaptive_profile
from metro45_phase import rephase_bridge


def prepare_candidates(data, bundle, state, ids, second, end):
    at = (datetime.fromisoformat(data['period_start']) + timedelta(seconds=second)).isoformat()
    applicable = minimum_headway(datetime.fromisoformat(at), data['headway_policy']) >= 150
    metadata = dict(feedback_checkpoint_second=second, feedback_interval_seconds=900,
        forecast_issued_at=bundle['issued_at'], decision_at=at,
        offpeak_candidate_applicable=applicable, future_actual_demand_used=False)
    if not applicable:
        return [], metadata
    adaptive = adaptive_profile(bundle, ids, at, second, end, checkpoint=state)
    metadata['adaptive'] = adaptive
    return [dict(candidate_id='ADAPTIVE_150_160', builder='bridge',
                 headway=adaptive['profile'], bridge_seconds=480),
            dict(candidate_id='FIXED160_FUTURE_PHASE', builder='phase', headway=160,
                 bridge_seconds=480, shift_seconds=-40)], metadata


def calculate(data, runner, state, profiles, roster, source, strategy, ids, second, planning_end,
              candidate_limit, deadline, cancelled, *, observed_frame=None, past_bundles=(),
              use_residual_guard=False, quality_policy='strict', joint_builder=None):
    if monotonic() >= deadline or cancelled is not None and cancelled.is_set():
        raise TimeoutError('quarter-hour controller deadline')
    if use_residual_guard and observed_frame is None:
        raise ValueError('residual control requires available station observations')
    modes = {'MPC_ML': 'ML', 'MPC_BASELINE_FORECAST': 'MEDIAN'}
    if strategy not in modes:
        raise ValueError('causal ML or baseline control required')
    if use_residual_guard and modes[strategy] != 'ML':
        raise ValueError('ML residual cannot adjust a different baseline model')
    at = (datetime.fromisoformat(data['period_start']) + timedelta(seconds=second)).isoformat()
    raw = read_forecast(source, modes[strategy], ids, at)
    bundle = raw
    if use_residual_guard:
        from metro45_residual_nowcast import adjust_bundle
        from metro45_risk_envelope import build_risk_envelope
        adjusted = adjust_bundle(raw, observed_frame, at, past_bundles=past_bundles)
        bundle = build_risk_envelope(raw, adjusted, at)
    specs, feedback = prepare_candidates(data, bundle, state, ids, second, planning_end)
    builders = dict(bridge=rebuild_bridge_timetable, phase=rephase_bridge)
    if joint_builder is not None:
        builders['joint'] = joint_builder
        specs.append(dict(candidate_id='JOINT_DISPATCH', builder='joint'))
    result = calculate_catalog(data, runner, state, profiles, roster, source, strategy, ids,
        second, planning_end, candidate_limit, deadline, cancelled, quality_policy=quality_policy,
        builders=builders, profile_specs=specs, bundle_override=bundle)
    result['decision']['feedback45'] = dict(**feedback, residual_guard_used=use_residual_guard,
        original_point_estimate_retained=True, original_forecast_issued_at=raw['issued_at'],
        joint_candidate_available=joint_builder is not None)
    return result
