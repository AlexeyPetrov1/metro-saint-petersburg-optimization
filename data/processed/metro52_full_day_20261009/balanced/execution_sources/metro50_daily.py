"""Full-day paired cost control with stochastic boarding and peak service checks."""

import argparse
import copy
import gzip
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/processed/metro48_daily_fifo_20261009_v7/runs/DAILY_FIFO_S1_7'
FRAME = ROOT / 'data/processed/metro44_catalog_pilot_20261009/observed_frame.parquet'
ENTRY_END, SERVICE_END = 64800, 65400


def control_times(entry_end):
    return list(range(3600, entry_end, 900))


def return_boundary(second, entry_end):
    return min(second + 13800, entry_end)


def admission_forecast(bundle, entry_end):
    out = copy.deepcopy(bundle)
    end = datetime.fromisoformat(entry_end)
    for row in out['rows']:
        if datetime.fromisoformat(row['interval_start']) >= end:
            row['predicted_count'] = 0
    out['source_versions']['admission_end'] = entry_end
    return out


def closed_tail(bundle, horizon_end):
    out = copy.deepcopy(bundle)
    end = datetime.fromisoformat(horizon_end)
    latest = max(datetime.fromisoformat(r['interval_end']) for r in out['rows'])
    ids = sorted({r['station_id'] for r in out['rows']})
    while latest < end:
        stop = min(latest + timedelta(seconds=900), end)
        out['rows'].extend(dict(station_id=s, interval_start=latest.isoformat(),
            interval_end=stop.isoformat(), predicted_count=0, coverage='complete') for s in ids)
        latest = stop
    return out


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def run_day(output, *, arm='GRU', scenario='current5', seed=7, steps=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if arm not in ('REFERENCE', 'GRU', 'MEDIAN') or scenario not in ('current5', 'dense10'):
        raise ValueError('unsupported daily arm or scenario')
    if steps is not None and (type(steps) is not int or not 1 <= steps <= 68):
        raise ValueError('pilot steps must be 1..68')
    import pandas as pd
    from goal43_replay import save_run
    from metro44_catalog import physical_gate
    from metro44_focus import execution_prefix
    from metro44_run import bounded_decision, public_decision, selected_plan, write_json_once, write_gzip_once
    from metro_ml_compare import _metrics
    from metro47_catalog import calculate_cost_catalog
    from metro47_cost_policy import choose_by_cost
    from metro47_withdrawal_control import refine_withdrawals
    from metro48_forecasts import profiled_median, complete_tail
    from metro48_gru import FrozenQuarterGRU
    from metro48_temporary import build_temporary_candidate
    from metro48_temporary_solver import solve_temporary_withdrawals
    from metro50_peak_service import peak_boarding_service, apply_peak_safety_margin, SCENARIOS
    from metro50_economics import maintenance_rates
    from passenger_groups import GroupedPassengerSimulation
    from simulation import digest
    from withdrawal_stabling import attach_stabling

    began = monotonic()
    data, groups, profiles = (read(SOURCE / name) for name in ('movement_input.json', 'groups.json.gz', 'profiles.json'))
    data = attach_stabling(data)
    origin = datetime.fromisoformat(data['period_start'])
    if origin.hour != 6 or data['period_end'] != '2026-10-01T02:00:00+03:00':
        raise ValueError('expected preserved September30 full day')
    roster = [dict(train_id=t['train_id'], type=t['train_type']) for t in data['trains']]
    options = dict(boarding_seed=seed, boarding_capacity_per_car=150)
    runner = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], groups, profiles, **options)
    times = control_times(ENTRY_END) if arm != 'REFERENCE' else []
    if steps is not None:
        times = times[:steps]
    from metro48_gru import MODEL_INPUTS
    sources = [SOURCE / name for name in ('movement_input.json', 'groups.json.gz', 'profiles.json')]
    sources += [FRAME] + list(ROOT.glob('*.py'))
    sources += [p for p in MODEL_INPUTS.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output.mkdir(parents=True)
    settings = SCENARIOS[scenario]
    limit = settings['missed_share_limit']
    rates = {kind: price['without_sr_kr'] for kind, price in maintenance_rates().items()}
    contract = dict(arm=arm, scenario=scenario, boarding_seed=seed, boarding_capacity_per_car=150,
        period_start=data['period_start'], period_end=data['period_end'], entry_end_second=ENTRY_END,
        service_end_second=SERVICE_END, full_day=True, all_control_steps=steps is None,
        fixed_fleet_size=68, line_id=1, decision_times=times, source_hashes=hashes,
        groups_digest=digest(groups), settings=settings, forecasting='causal nativeGRU with quarter refresh',
        planning_horizon_seconds=14400, native_model_horizon_seconds=7200,
        historical_tail='causal past-only median beyond native model horizon',
        objective='TO1+TO2+TR1+TR2+TR3 mileage accrual; SR/KR reported separately',
        maintenance_rates_rub_per_km=rates,
        target='turnstile entries, not onboard load', real_operating_schedule=False,
        cash_saving_rub=None, annual_repeat_days=250, annual_repeat_is_scenario=True)
    write_json_once(output / 'contract.json', contract)
    snapshot = output / 'execution_sources'
    snapshot.mkdir()
    for p in ROOT.glob('*.py'):
        (snapshot / p.name).write_bytes(p.read_bytes())
    write_json_once(snapshot / 'manifest.json', hashes)
    write_gzip_once(output / 'common_scenario.json.gz', dict(data=data, groups=groups, profiles=profiles, roster=roster))
    decisions, previous, model = [], 0, None
    frame = pd.read_parquet(FRAME) if times else None
    ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    admission_end = (origin + timedelta(seconds=ENTRY_END)).isoformat()

    def service_check(trial, state, start, end):
        result = peak_boarding_service(trial, state, start_second=start, end_second=end // 900 * 900,
                                      missed_share_limit=limit)
        if result['status'] == 'not_applicable':
            result['passed'] = True
        return result

    for index, second in enumerate(times):
        state = runner.run(start_second=previous, end_second=second,
                           **({'checkpoint': state} if previous else {}))['checkpoint']
        at = (origin + timedelta(seconds=second)).isoformat()
        prefix = execution_prefix(runner.motion, second)

        def pipeline(deadline, cancelled):
            nonlocal model
            if model is None:
                model = FrozenQuarterGRU()
            raw = (model.forecast_bundle(frame, ids, at) if arm == 'GRU' else
                   profiled_median(frame, ids, at, profile_provider=model.station_profile))
            boundary = (origin + timedelta(seconds=min(second+14400, ENTRY_END))).isoformat()
            tail = complete_tail(raw, frame, ids, at, boundary)
            tail = closed_tail(tail, (origin + timedelta(seconds=min(second+14400, runner.duration))).isoformat())
            bounded = apply_peak_safety_margin(admission_forecast(tail, admission_end), at, scenario=scenario)
            back = return_boundary(second, ENTRY_END)
            def solve(**constraints):
                return solve_temporary_withdrawals(data, bounded, state, second, SERVICE_END,
                    return_by_second=back, deadline=deadline, time_limit_seconds=20,
                    maintenance_rates=rates, peak_missed_share_limit=limit, **constraints)
            def evaluate(selected):
                return calculate_cost_catalog(data, runner, state, profiles, roster, bounded,
                    second, SERVICE_END, deadline, cancelled,
                    proposal=dict(selected_train_ids=selected, return_by_second=back) if selected else None,
                    builder=build_temporary_candidate, passenger_service_check=service_check,
                    forecast_horizon_seconds=14400,
                    selection_policy=lambda rows, **kw: choose_by_cost(rows, maintenance_rates=rates, **kw))
            answer = refine_withdrawals(solve, evaluate, deadline=deadline, cancelled=cancelled)
            answer.update(raw_forecast=raw, bounded_forecast=bounded)
            return answer

        receipt = bounded_decision(pipeline, 480)
        if not receipt['completed']:
            write_json_once(output / f'timeout_{second:05}.json', dict(second=second, receipt=receipt))
            raise TimeoutError('daily controller exceeded480seconds; completed journals retained')
        answer = receipt['result']
        decision = answer['decision']
        changed, applied = selected_plan(decision['selection'], answer['candidates'], decision['candidate_table'])
        if changed is not data:
            new = GroupedPassengerSimulation(changed, changed['plans'][0]['plan_id'], groups, profiles, **options)
            if execution_prefix(new.motion, second) != prefix:
                raise ValueError('executed prefix changed')
            state = new.branch_checkpoint(state, runner)
            runner, data = new, changed
        decision.update(applied_candidate_id=applied, pipeline_seconds=receipt['elapsed_seconds'],
            executed_prefix_digest=prefix, local_recommendation_within_480=True, external_delivery_measured=False)
        saved = public_decision(decision)
        decisions.append(saved)
        folder = output / 'controller'
        write_json_once(folder / f'decision_{second:05}.json', saved)
        write_json_once(folder / f'forecast_{second:05}.json', answer['bounded_forecast'])
        write_json_once(folder / f'raw_forecast_{second:05}.json', answer['raw_forecast'])
        write_gzip_once(folder / f'resume_{second:05}.json.gz', dict(second=second, data=data, state=state))
        print(json.dumps(dict(stage='daily_control', arm=arm, scenario=scenario, index=index + 1,
            count=len(times), at=at, applied=applied, seconds=receipt['elapsed_seconds']), ensure_ascii=False), flush=True)
        previous = second

    state = runner.run(start_second=previous, **({'checkpoint': state} if previous else {}))['checkpoint']
    runner._validate_checkpoint(state, state['second'])
    physical = physical_gate(data, 0, SERVICE_END, reference_motion=runner.motion)
    service = peak_boarding_service(runner, state, end_second=ENTRY_END, missed_share_limit=limit)
    metrics = _metrics(data, runner, state, roster, data['headway_policy'], 0, SERVICE_END)
    metrics['peak_boarding_service'] = service
    metrics['physical_gate'] = {k: v for k, v in physical.items() if not k.startswith('_')}
    metrics['actual_service_admissible'] &= (physical['movement_passed'] and service['passed']
        and metrics['all_service_arrivals_completed'] and state['queued'] == state['approaching'] == state['onboard_count'] == 0)
    record = save_run(output, f'DAY50_{arm}_{scenario}_S{seed}', data, runner, state, roster,
        data['headway_policy'], 'DAY50', arm, 'S1', seed, full_day=True,
        cohort_start=0, cohort_end=ENTRY_END, service_end_second=SERVICE_END)
    result = dict(contract=contract, decisions=decisions, run=record, metrics=metrics,
        elapsed_seconds=monotonic() - began, sources_preserved=all(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()),
        full_day=True, complete_control=steps is None, cash_saving_rub=None, lost_revenue_rub=None)
    write_json_once(output / 'result.json', result)
    print(json.dumps(dict(stage='daily_complete', arm=arm, scenario=scenario,
        km=metrics['train_km'], passed=metrics['actual_service_admissible'],
        elapsed=result['elapsed_seconds']), ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['REFERENCE', 'GRU', 'MEDIAN'], default='GRU')
    parser.add_argument('--scenario', choices=['current5', 'dense10'], default='current5')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int)
    run_day(**vars(parser.parse_args()))
