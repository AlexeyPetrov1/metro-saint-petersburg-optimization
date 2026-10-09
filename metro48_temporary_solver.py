"""CP-SAT selection of concrete temporary duties and their retained returns."""

from copy import deepcopy
from datetime import datetime, timedelta
from math import isfinite

from forecasting import validate_bundle
from metro44_mpc import forecast_section_rates
from metro44_run import station_offsets
from metro47_cost_policy import normalized_maintenance_rates
from metro47_withdrawal_solver import solve_withdrawal_options
from metro48_temporary import temporary_options
from simulation import digest


def solve_temporary_options(options, departures, *, option_values, rate_slots, as_of,
                             service_end_second, **kwargs):
    clocks = {(r['trip_id'], r['second']) for r in departures}
    for row in options:
        if (row['resume_trip_id'] in row['cancel_service_ids']
                or (row['resume_trip_id'], row['resume_second']) not in clocks
                or not as_of < row['resume_second'] <= row['return_by_second'] <= service_end_second):
            raise ValueError('temporary return must be a preserved real duty within its bound')
    result = solve_withdrawal_options(options, departures, option_values=option_values,
        rate_slots=rate_slots, as_of=as_of, service_end_second=service_end_second, **kwargs)
    selected = {r['train_id']: r for r in result['withdrawal_options']}
    result.update(action_scope='temporary withdrawal and preserved same-site re-entry',
        objective='cancelled_known_train_km_minus_new_EXIT_and_ENTRY_times_confirmed_TO1_rates',
        resume_trip_ids={tid: r['resume_trip_id'] for tid, r in selected.items()},
        resume_seconds={tid: r['resume_second'] for tid, r in selected.items()})
    return result


def solve_temporary_withdrawals(data, bundle, state, as_of, service_end_second, *, return_by_second,
                                deadline=None, no_good_cuts=(), max_withdrawals=None,
                                time_limit_seconds=5, car_concentration=1, scenario='S1',
                                maintenance_rates=None, peak_missed_share_limit=None):
    maintenance = normalized_maintenance_rates(maintenance_rates)
    if peak_missed_share_limit is not None and (type(peak_missed_share_limit) not in (int, float)
            or not isfinite(peak_missed_share_limit) or not 0 <= peak_missed_share_limit < 1):
        raise ValueError('peak missed share must be finite and in [0,1)')
    if state.get('second') != as_of:
        raise ValueError('current checkpoint clock mismatch')
    versions = bundle.get('source_versions', {})
    if versions.get('future_truth_intentionally_used') or versions.get('operationally_available') is False:
        raise ValueError('unavailable future passenger truth')
    at = (datetime.fromisoformat(data['period_start'])+timedelta(seconds=as_of)).isoformat()
    ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    validate_bundle(bundle, ids, at)
    rows = temporary_options(data, as_of, service_end_second,
                              return_by_second=return_by_second, deadline=deadline)
    tracks = {t['track_id']: t for t in data['tracks']}
    stations = {s['station_id']: s for s in data['stations']}
    trips = {t['trip_id']: t for t in data['plans'][0]['trips']}
    trains = {t['train_id']: t for t in data['trains']}
    offsets, clocks = station_offsets(data), {}

    def add(track_id, second, trip_id):
        track = tracks[track_id]
        if not all(stations[track[s]]['kind'] == 'line' for s in ('from_station', 'to_station')):
            return
        direction = 'up' if track_id.endswith('_up') else 'down'
        clocks[track_id, second] = dict(track_id=track_id, direction=direction, second=second,
            trip_id=trip_id, terminal_offset_seconds=offsets[direction][ids.index(track['from_station'])])

    for row in data.get('previous_departures', []):
        add(row['track_id'], row['departure_second'], '@history')
    for row in data['leg_timing']:
        track = (trains[row['train_id']]['location'] if row['trip_id'] == '@initial'
                 else trips[row['trip_id']]['route'][row['leg_index']])
        add(track, row['departure_second'], row['trip_id'])
    if type(car_concentration) not in (int, float) or not isfinite(car_concentration) or car_concentration < 1:
        raise ValueError('invalid car concentration')
    rates = forecast_section_rates(bundle, ids, at, period_start=data['period_start'],
        scenario=scenario, checkpoint=state, station_offsets_seconds=offsets)
    proposal_rates, discounted = rates, 0
    if peak_missed_share_limit is not None:
        from metro50_peak_service import is_peak
        proposal_rates = deepcopy(rates)
        origin = datetime.fromisoformat(data['period_start'])
        for slot in proposal_rates:
            if peak_missed_share_limit and is_peak(origin + timedelta(seconds=slot['start_second'])):
                discounted += 1
                for direction in ('up', 'down'):
                    slot[direction+'_rate'] *= 1 - peak_missed_share_limit
                    key = direction+'_section_rates'
                    if key in slot:
                        slot[key] = [v * (1 - peak_missed_share_limit) for v in slot[key]]
    to1_values = {r['train_id']: float(r['to1_resource_saving_rub']) for r in rows}
    if maintenance is not None:
        from metro44_selection import TYPE_ALIASES
        values = {r['train_id']: float(r['saved_km']) * maintenance[TYPE_ALIASES.get(
            r['train_type'].casefold(), r['train_type'].casefold())] for r in rows}
    else:
        values = to1_values
    result = solve_temporary_options(rows, list(clocks.values()), option_values=values,
        rate_slots=proposal_rates, as_of=as_of, service_end_second=service_end_second,
        capacity=int(1200/car_concentration), no_good_cuts=no_good_cuts,
        max_withdrawals=max_withdrawals, time_limit_seconds=time_limit_seconds, deadline=deadline)
    km = {r['train_id']: float(r['saved_km']) for r in rows}
    result.update(forecast_digest=digest(bundle), decision_second=as_of,
        future_actual_used=False, return_by_second=return_by_second,
        known_source_km_saving_by_train=km,
        selected_known_train_km_saving=sum(km[t] for t in result['selected_train_ids']),
        retained_clock_scope='all initial, executed, current and retained later SERVICE clocks remain exact',
        preparation_scope='same-site return; depot inclusive release or siding physical reversal',
        stabling_allocation_scope='candidate values use next quota slot; selected batch destinations and cost checked in replay',
        car_capacity_scope='150/car concentration proxy; actual160/1percent checked separately')
    if peak_missed_share_limit is not None:
        result['peak_waiting_proxy'] = dict(missed_share_limit=peak_missed_share_limit,
            discounted_slot_count=discounted, original_rates_digest=digest(rates),
            proposal_rates_digest=digest(proposal_rates), calibrated=False,
            scope='proposal-only crossing rate discount in weekday Moscow08-10/17-19 peak slots',
            model='rate*(1-first_miss_limit); queue carryover and individual first misses not modeled',
            exact_forecast_trials_unmodified=True, actual_capacity_changed=False,
            actual_passenger_admissibility_claimed=False, cash_saving_claimed=False)
    if maintenance is not None:
        result.update(estimated_maintenance_resource_saving_rub=sum(values[t] for t in result['selected_train_ids']),
            estimated_to1_resource_saving_rub=sum(to1_values[t] for t in result['selected_train_ids']),
            known_maintenance_saving_by_train=values, maintenance_rates_rub_per_km=maintenance,
            maintenance_scope='TO1+TO2+TR1+TR2+TR3; excludes SR and KR',
            objective='cancelled_known_train_km_minus_new_EXIT_and_ENTRY_times_TO1+TO2+TR1+TR2+TR3_rates')
    return result
