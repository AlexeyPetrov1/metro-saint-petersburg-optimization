"""Recover completed passenger journeys from destination FIFO and boarding history."""

from collections import Counter


def cohort_journey(runner, state, start, end):
    selected = {key: group for key, group in runner.by_id.items()
                if start <= group['turnstile_second'] < min(end, state['second'])}
    pending = {key: [dict(row) for row in group['destinations']] for key, group in selected.items()}
    arrivals = {(e['train_id'], e['trip_id'], e['location']): e['second']
                for e in runner.motion_by_id.values() if e['event_type'] == 'arrival'}
    histograms = {key: Counter() for key in ('ride', 'platform_to_exit', 'gate_to_exit')}
    boarded = completed = 0
    for row in state['board_events']:
        key = row['group_id']
        if key not in selected:
            continue
        event = runner.motion_by_id[row['motion_event_id']]
        eligible = {runner.tracks[track]['to_station'] for track in event['remaining_route']}
        destination = next((d for d in pending[key] if d['count'] and d['station_id'] in eligible), None)
        if destination is None or row['count'] > destination['count']:
            raise ValueError('boarding history contradicts destination FIFO')
        destination['count'] -= row['count']
        boarded += row['count']
        arrival = arrivals.get((event['train_id'], event['trip_id'], destination['station_id']))
        if arrival is None or arrival > state['second']:
            continue
        if arrival < row['second']:
            raise ValueError('alighting precedes boarding')
        completed += row['count']
        for metric, since in (('ride', row['second']),
                              ('platform_to_exit', selected[key]['platform_second']),
                              ('gate_to_exit', selected[key]['turnstile_second'])):
            histograms[metric][arrival-since] += row['count']
    arrived = sum(g['count'] for g in selected.values())
    complete = state['second'] >= end and arrived == completed
    result = dict(arrived=arrived, boarded=boarded, completed=completed, uncompleted=arrived-completed,
                  complete=complete, scope='same turnstile cohort; destination arrivals through saved end',
                  cash_value_of_passenger_time_rub=None)
    for metric, histogram in histograms.items():
        mean = p95 = None
        if complete and completed:
            mean = sum(seconds*count for seconds, count in histogram.items())/completed
            rank, cumulative = (95*completed+99)//100, 0
            for seconds, count in sorted(histogram.items()):
                cumulative += count
                if cumulative >= rank:
                    p95 = seconds
                    break
        result[f'mean_{metric}_seconds'] = mean
        result[f'p95_{metric}_seconds'] = p95
    return result
