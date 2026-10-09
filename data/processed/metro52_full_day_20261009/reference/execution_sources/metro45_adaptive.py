"""Small causal dispatch policy; exact movement and passenger checks stay mandatory."""

from datetime import datetime, timedelta
from math import isfinite
from statistics import median


def adaptive_profile(bundle, station_ids, as_of, second, end, *, checkpoint=None,
                     scenario='S1', urgent_wait_seconds=120):
    from metro44_mpc import forecast_section_rates

    now = datetime.fromisoformat(as_of)
    if (type(second) is not int or type(end) is not int or second < 0 or
            second % 900 or end <= second):
        raise ValueError('ordered quarter-hour planning bounds required')
    if checkpoint is not None and checkpoint.get('second') != second:
        raise ValueError('checkpoint must match decision time')
    if not isfinite(urgent_wait_seconds) or urgent_wait_seconds < 0:
        raise ValueError('nonnegative queue age required')
    origin = now - timedelta(seconds=second)
    slots = forecast_section_rates(bundle, station_ids, as_of, period_start=origin.isoformat(),
                                   scenario=scenario, checkpoint=checkpoint)
    rates = {r['start_second']: max(r['up_rate'], r['down_rate']) for r in slots}
    positive = [rate for rate in rates.values() if rate > 0]
    threshold = median(positive) if positive else None
    queued = [] if checkpoint is None else [row for rows in checkpoint.get('queues', {}).values()
                                          for row in rows if row.get('count', 0) > 0]
    urgent = any(second - row['platform_second'] >= urgent_wait_seconds for row in queued)
    profile = []
    for at in range(second, end, 900):
        rate = rates.get(at)
        dense = threshold is not None and rate is not None and rate >= threshold
        h = 150 if dense or (at == second and urgent) else 160
        profile.append(dict(terminal='*', start_second=at, end_second=min(end, at + 900),
                            headway_seconds=h))
    return dict(profile=profile, observed_queue_urgent=urgent,
        forecast_covered_slot_count=sum(r['start_second'] in rates for r in profile),
        forecast_rate_threshold=threshold, continuation_headway_seconds=160,
        assumed_od=scenario, future_actual_demand_used=False,
        rule='150s at/above median positive forecast section rate or urgent current queue; otherwise160s',
        limitations='heuristic dispatch candidate; OD inferred; exact per-car and movement validation required')


def strict_improvement(baseline, candidate, bounds, *, physical_passed):
    if physical_passed is not True or any(candidate.get(k) is not True for k in (
            'actual_service_admissible', 'all_service_arrivals_completed',
            'movement_obligations_closed', 'terminal_resources_passed')):
        return False
    values = [baseline.get('mean_wait_seconds'), baseline.get('p95_wait_seconds'),
              candidate.get('mean_wait_seconds'), candidate.get('p95_wait_seconds'),
              bounds.get('saving_train_km_lower_bound'), bounds.get('saving_to1_resource_rub_lower_bound')]
    if any(type(v) not in (int, float) or not isfinite(v) for v in values):
        return False
    mean, p95, new_mean, new_p95, km, rub = values
    return new_mean <= mean and new_p95 <= p95 and km > 0 and rub > 0
