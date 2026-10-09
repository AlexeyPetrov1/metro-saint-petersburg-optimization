"""Bound paired window mileage without assuming positions between track events."""

from collections import defaultdict
from decimal import Decimal


RATES = {'baltiets': Decimal('9.2'), 'yubileyny': Decimal('7.4')}


def window_ledger(data, motion, start, end):
    if not 0 <= start < end:
        raise ValueError('invalid window')
    trains = {row['train_id']: row for row in data['trains']}
    tracks = {row['track_id']: Decimal(str(row['length_km'])) for row in data['tracks']}
    events = defaultdict(list)
    for row in motion['events']:
        events[row['train_id']].append(row)
    ledgers, snapshots = {}, {}
    for train_id, initial in trains.items():
        rows = events[train_id]
        if any(a['second'] > b['second'] for a, b in zip(rows, rows[1:])):
            raise ValueError('movement events are not chronological')
        rate = RATES[initial['train_type'].casefold()]
        km = Decimal(0)
        departure = None
        for row in rows:
            if row['event_type'] == 'departure':
                departure = row
            elif row['event_type'] == 'arrival' and start < row['second'] <= end:
                track = departure['location'] if departure else initial['location']
                km += tracks[track]
        active = {}
        for boundary in (start, end):
            past = [row for row in rows if row['second'] <= boundary]
            last = past[-1] if past else initial
            state = last.get('next_state', last.get('state'))
            segment = None
            if state == 'moving':
                departed = next((row for row in reversed(past) if row['event_type'] == 'departure'), None)
                arrival = next((row for row in rows if row['event_type'] == 'arrival' and row['second'] > boundary), None)
                if arrival is None:
                    raise ValueError('unfinished movement has no recorded arrival')
                segment = dict(track_id=last['location'], departure_second=departed['second'] if departed else None,
                               arrival_second=arrival['second'], trip_id=arrival['trip_id'],
                               initial_movement=initial.get('initial_movement') if departed is None else None)
            active[boundary] = dict(state=state, location=last['location'], segment=segment)
        tail = active[end]['segment']
        upper = tracks[tail['track_id']] if tail and tail['departure_second'] != end else Decimal(0)
        ledgers[train_id] = dict(train_type=initial['train_type'], arrival_km=float(km),
            arrival_to1_resource_rub=float(km*rate), end_movement_km_upper_bound=float(upper),
            end_to1_resource_rub_upper_bound=float(upper*rate))
        snapshots[train_id] = active[start]
    total = lambda key: float(sum((Decimal(str(row[key])) for row in ledgers.values()), Decimal(0)))
    return dict(start_second=start, end_second=end, start_snapshot=snapshots, per_train=ledgers,
        arrival_train_km=total('arrival_km'), arrival_to1_resource_rub=total('arrival_to1_resource_rub'),
        end_movement_km_upper_bound=total('end_movement_km_upper_bound'),
        end_to1_resource_rub_upper_bound=total('end_to1_resource_rub_upper_bound'),
        boundary_convention='arrivals in (start,end]; states after events at each boundary')


def paired_window_bounds(reference, ref_motion, candidate, cand_motion, start, end):
    left, right = (window_ledger(data, motion, start, end)
                   for data, motion in ((reference, ref_motion), (candidate, cand_motion)))
    types = lambda item: {key: row['train_type'] for key, row in item['per_train'].items()}
    if not left['per_train'] or types(left) != types(right):
        raise ValueError('comparison requires identical fixed fleet and train types')
    if left['start_snapshot'] != right['start_snapshot']:
        raise ValueError('comparison requires common start states and movement obligations')
    delta = left['arrival_train_km'] - right['arrival_train_km']
    wear = left['arrival_to1_resource_rub'] - right['arrival_to1_resource_rub']
    lower, upper = delta-right['end_movement_km_upper_bound'], delta+left['end_movement_km_upper_bound']
    return dict(reference=left, candidate=right, fixed_fleet_size=len(left['per_train']),
        arrival_ledger_saving_train_km=delta, saving_train_km_lower_bound=lower,
        saving_train_km_upper_bound=upper, saving_km_per_fixed_train_lower_bound=lower/len(left['per_train']),
        saving_to1_resource_rub_lower_bound=wear-right['end_to1_resource_rub_upper_bound'],
        saving_to1_resource_rub_upper_bound=wear+left['end_to1_resource_rub_upper_bound'],
        common_start_verified=True, position_interpolation_assumed=False,
        cash_saving_rub=None, lost_revenue_rub=None, net_profit_effect_rub=None,
        scope='paired movement window; shared start cancels unknown boundary position; whole endpoint tracks bound remaining mileage')
