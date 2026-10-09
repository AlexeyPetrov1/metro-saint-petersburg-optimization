"""Independent peak boarding cohorts, reconstructed from movement and boarding ledgers."""

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import math

from forecasting import validate_bundle
from metro47_robust_forecast import apply_safety_margin


MOSCOW = timezone(timedelta(hours=3))
SCENARIOS = {
    'current5': dict(missed_share_limit=.05, peak_design_per_car=120, offpeak_design_per_car=120),
    'dense10': dict(missed_share_limit=.10, peak_design_per_car=150, offpeak_design_per_car=120),
}
COUNTS = ('passenger_count', 'first_opportunity_count', 'first_opportunity_censored_count',
          'missed_first_count', 'boarded_second_count', 'second_opportunity_count',
          'second_opportunity_censored_count', 'missed_second_count',
          'missed_opportunity_count', 'still_queued_count')


def is_peak(at):
    at = at.astimezone(MOSCOW)
    return at.weekday() < 5 and (8 <= at.hour < 10 or 17 <= at.hour < 19)


def apply_peak_safety_margin(bundle, decision_time, *, scenario='current5'):
    """Use a 120-car design off peak, and the named scenario's design during peaks."""
    if scenario not in SCENARIOS:
        raise ValueError('unknown peak scenario')
    config = SCENARIOS[scenario]
    result = apply_safety_margin(bundle, decision_time)
    for source, target in zip(bundle['rows'], result['rows']):
        design = config['peak_design_per_car'] if is_peak(datetime.fromisoformat(source['interval_start'])) else 120
        target['predicted_count'] = int((Decimal(str(source['predicted_count'])) * Decimal(150) / design)
                                        .to_integral_value(rounding=ROUND_CEILING))
    meta = result['source_versions']['engineering_safety_margin']
    meta.update(version='weekday-peak-design-margin-v1', scenario=scenario, factor=None, design_per_car=None,
                peak_factor=150/config['peak_design_per_car'], offpeak_factor=1.25,
                peak_design_per_car=config['peak_design_per_car'], offpeak_design_per_car=120,
                peak_windows='weekdays 08:00-10:00 and 17:00-19:00 Europe/Moscow',
                basis='fixed engineering scenario; design 120 off peak and named design during peak')
    result['feature_version'] = bundle['feature_version']+'+weekday-peak-design-margin-v1'
    validate_bundle(result, sorted({r['station_id'] for r in result['rows']}), decision_time)
    return result


def _summary(rows, limit, *, temporal_complete=True, require_next_train=False):
    result = {key: sum(r[key] for r in rows) for key in COUNTS}
    denominator = result['first_opportunity_count']
    result.update(missed_first_share=result['missed_first_count']/denominator if denominator else None,
                  second_train_missed_count=result['missed_second_count'],
                  max_boarded_wait_seconds=max((r['max_boarded_wait_seconds'] for r in rows), default=0),
                  max_current_wait_seconds=max((r['max_current_wait_seconds'] for r in rows), default=0),
                  temporal_coverage_complete=temporal_complete)
    violated = (denominator and Decimal(result['missed_first_count']) > Decimal(str(limit))*denominator
                or require_next_train and result['missed_second_count'])
    incomplete = (not temporal_complete or result['first_opportunity_censored_count']
                  or require_next_train and result['second_opportunity_censored_count'])
    result['status'] = ('failed' if violated else 'incomplete' if incomplete else
                        'passed' if result['passenger_count'] else 'not_applicable')
    return result


def peak_boarding_service(runner, state, *, start_second=0, end_second=None, missed_share_limit=.05,
                         require_next_train=False):
    """Check unique peak platform-arrival cohorts, with later followup to state.second.

    Misses count people once per first/second eligible departure, never the simulator's
    repeated denied-attempt counter. Legacy OD is recoverable from deterministic FIFO;
    new boarding ledgers may carry destination directly. Only the configured 5%/10%
    first-miss ceiling is binding by default; second misses remain descriptive.
    require_next_train explicitly enables an additional second-miss prohibition.
    Queue drain and mature-cohort completion need their separate existing checks;
    this metric neither enforces a random passenger skip nor certifies queue drain.
    Aggregate 15-minute intervals are binding; station/direction details are descriptive.
    """
    if (type(missed_share_limit) not in (int, float) or not math.isfinite(missed_share_limit)
            or not 0 <= missed_share_limit <= 1):
        raise ValueError('missed share limit must be finite and in [0,1]')
    if type(require_next_train) is not bool:
        raise ValueError('require_next_train must be boolean')
    observed = state['second']
    end = observed if end_second is None else end_second
    if (type(start_second) is not int or type(end) is not int
            or not 0 <= start_second <= end <= observed <= runner.duration):
        raise ValueError('invalid cohort or observation window')
    origin = datetime.fromisoformat(runner.motion['period_start'])
    if origin.tzinfo is None:
        raise ValueError('movement period requires a timezone')
    origin = origin.astimezone(MOSCOW)
    selected = {g['group_id'] for g in runner.groups if start_second <= g['platform_second'] < end
                and is_peak(origin+timedelta(seconds=g['platform_second']))}
    schedules, departures = defaultdict(list), {}
    for index, event in enumerate(runner.stream[:state['next_event_index']]):
        if event.get('event_type') != 'departure' or not event.get('passenger_service'):
            continue
        track = runner.tracks[event['location']]
        direction = 1 if runner.positions[track['to_station']] > runner.positions[track['from_station']] else -1
        destinations = {runner.tracks[t]['to_station'] for t in event['remaining_route']}
        departures[event['event_id']] = (index, event, destinations, direction)
        for destination in destinations:
            schedules[track['from_station'], direction, destination].append((event['second'], index, event['event_id']))

    original, remaining, boards = {}, {}, defaultdict(list)
    for group in runner.groups:
        if group['group_id'] not in selected:
            continue
        for destination in group['destinations']:
            key = group['group_id'], destination['station_id']
            original[key] = remaining[key] = destination['count']
    # Within each group/departure the legacy log has one positive row per eligible
    # destination in station order; its count cannot exceed that OD's residual.
    next_destination = defaultdict(int)
    for row in state['board_events']:
        if row['group_id'] not in selected:
            continue
        group = runner.by_id[row['group_id']]
        position, event, destinations, direction = departures[row['motion_event_id']]
        if row.get('destination') is not None:
            destination = row['destination']
        else:
            lower = next_destination[row['group_id'], row['motion_event_id']]
            candidates = sorted((d['station_id'] for d in group['destinations']
                                 if d['station_id'] in destinations and remaining.get((row['group_id'], d['station_id']), 0)),
                                key=runner.positions.get)
            destination = next((d for d in candidates if runner.positions[d] >= lower), None)
        key = row['group_id'], destination
        if (destination not in destinations or group['origin'] != runner.tracks[event['location']]['from_station']
                or type(row['count']) is not int or row['count'] <= 0
                or row['count'] > remaining.get(key, 0) or row['second'] != event['second']
                or row['second'] < group['platform_second']):
            raise ValueError('boarding ledger contradicts OD balance or movement')
        next_destination[row['group_id'], row['motion_event_id']] = runner.positions[destination]+1
        remaining[key] -= row['count']
        boards[key].append((event['second'], position, event['event_id'], row['count']))
    queued = Counter()
    for queue in state['queues'].values():
        for row in queue:
            if row['group_id'] in selected:
                queued[row['group_id'], row['destination']] += row['count']
    if any(queued[key] != count for key, count in remaining.items()) or any(key not in remaining for key in queued):
        raise ValueError('boarding/queue OD balance mismatch')

    details = defaultdict(list)
    for key, count in original.items():
        group = runner.by_id[key[0]]
        platform = group['platform_second']
        at = origin+timedelta(seconds=platform)
        if not start_second <= platform < end or not is_peak(at):
            continue
        destination = key[1]
        direction = 1 if runner.positions[destination] > runner.positions[group['origin']] else -1
        schedule = schedules[group['origin'], direction, destination]
        first = bisect_left(schedule, (platform, -1, ''))
        opportunities = len(schedule)-first
        by_rank = Counter()
        for second, position, event_id, boarded in boards[key]:
            rank = bisect_left(schedule, (second, position, event_id))-first
            if rank < 0:
                raise ValueError('boarding precedes first eligible departure')
            by_rank[rank] += boarded
        first_boarded = by_rank[0]
        missed_first = count-first_boarded if opportunities else 0
        row = dict(passenger_count=count, first_opportunity_count=count if opportunities else 0,
                   first_opportunity_censored_count=count if not opportunities else 0,
                   missed_first_count=missed_first, boarded_second_count=by_rank[1],
                   second_opportunity_count=missed_first if opportunities >= 2 else 0,
                   second_opportunity_censored_count=missed_first if opportunities == 1 else 0,
                   missed_second_count=count-first_boarded-by_rank[1] if opportunities >= 2 else 0,
                   missed_opportunity_count=sum(rank*n for rank, n in by_rank.items())+remaining[key]*opportunities,
                   still_queued_count=remaining[key],
                   max_boarded_wait_seconds=max((b[0]-platform for b in boards[key]), default=0),
                   max_current_wait_seconds=observed-platform if remaining[key] else 0)
        quarter = at.replace(minute=at.minute//15*15, second=0, microsecond=0)
        details[quarter, group['origin'], direction].append(row)

    cohorts, interval_rows = [], defaultdict(list)
    for (quarter, station, direction), rows in sorted(details.items()):
        complete = (origin+timedelta(seconds=start_second) <= quarter
                    and quarter+timedelta(minutes=15) <= origin+timedelta(seconds=end))
        row = dict(interval_start=quarter.isoformat(), interval_end=(quarter+timedelta(minutes=15)).isoformat(),
                   station_id=station, direction=direction,
                   **_summary(rows, missed_share_limit, temporal_complete=complete, require_next_train=require_next_train))
        cohorts.append(row)
        interval_rows[quarter].extend(rows)
    intervals = []
    for quarter, rows in sorted(interval_rows.items()):
        complete = (origin+timedelta(seconds=start_second) <= quarter
                    and quarter+timedelta(minutes=15) <= origin+timedelta(seconds=end))
        intervals.append(dict(interval_start=quarter.isoformat(), interval_end=(quarter+timedelta(minutes=15)).isoformat(),
                              **_summary(rows, missed_share_limit, temporal_complete=complete, require_next_train=require_next_train)))
    aggregate = _summary([r for rows in interval_rows.values() for r in rows], missed_share_limit,
                         temporal_complete=all(r['temporal_coverage_complete'] for r in intervals),
                         require_next_train=require_next_train)
    statuses = [aggregate['status']]+[r['status'] for r in intervals]
    status = ('failed' if 'failed' in statuses else 'incomplete' if 'incomplete' in statuses
              else aggregate['status'])
    return dict(version='unique-peak-first-next-train-v2', status=status,
                passed=True if status == 'passed' else False if status == 'failed' else None,
                missed_share_limit=missed_share_limit, next_train_required=require_next_train,
                cohort_start_second=start_second, cohort_end_second=end, observation_end_second=observed,
                approaching_count=state.get('approaching', 0), aggregate=aggregate,
                intervals=intervals, cohorts=cohorts,
                binding_scope='aggregate and each peak 15-minute platform-arrival cohort',
                station_scope='descriptive; no separate station percentage gate',
                capacity_denied_attempts_used=False)
