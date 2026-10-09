"""Paired quarter-hour economic control on the preserved complete cohort."""

import argparse
import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'data/processed/metro44_catalog_cont6h_20261009/campaign.py'


def run(output, *, arm='MPC_ML', steps=20):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(steps) is not int or not 1 <= steps <= 20:
        raise ValueError('one to twenty consecutive control steps required')
    if arm not in ('MPC_ML', 'MPC_BASELINE_FORECAST'):
        raise ValueError('causal ML or baseline required')
    spec = importlib.util.spec_from_file_location('_preserved_economic47', OLD)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    from metro47_catalog import calculate_cost_catalog
    from metro47_demand_optimizer import optimize_demand_profile
    from metro44_planner_bridge import rebuild_bridge_timetable
    from metro44_planner_wait import rebuild_wait_timetable
    from dwell import timing_key
    paths = [Path(__file__).resolve(), OLD] + [ROOT / name for name in (
        'metro47_catalog.py', 'metro47_cost_policy.py', 'metro47_demand_optimizer.py')]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    output.mkdir(parents=True)
    shared, common = engine.prepare()
    common.update(quality_policy='cost_only', budgets_are_user_requirements=True,
        mean_budget_seconds=None, p95_budget_seconds=None, decision_times=common['decision_times'][:steps],
        controller47=True, full_twenty_slots=steps == 20, requested_line_id=1,
        objective='minimum TO1 resource cost subject to operational constraints',
        acceptance_target_saving_fraction=0.01, full_day=False)
    common['input_hashes'].update(before)
    (output / 'shared_contract.json').write_text(json.dumps(common, indent=2), encoding='utf-8')
    engine.write_gzip_once(output / 'shared_scenario.json.gz', shared)
    engine.OUTPUT = output
    engine.prepare = lambda: (copy.deepcopy(shared), copy.deepcopy(common))
    engine.strict_counterfactual = lambda decision: None

    def calculate(*args, **kwargs):
        data, runner, state, profiles, roster, _, _, _, second, end, _, deadline, cancelled = args[:13]
        bundle = kwargs['bundle_override']
        origin = datetime.fromisoformat(data['period_start'])
        forecast_end = max(int((datetime.fromisoformat(r['interval_end'])-origin).total_seconds())
            for r in bundle['rows'])
        solved = optimize_demand_profile(data, bundle, state, second,
            horizon_seconds=min(7200, common['cohort_end_second']-second, forecast_end-second),
            service_end_second=end, deadline=deadline)
        clocks = {timing_key(r['train_id'], r['trip_id'], r['leg_index']): r['departure_second']
                  for r in data['leg_timing']}
        depot_future = any(t['route'][0].startswith('ENTRY_') and
            clocks[timing_key(t['train_id'], t['trip_id'], 0)] >= second for t in data['plans'][0]['trips'])
        builder = rebuild_wait_timetable if depot_future else rebuild_bridge_timetable
        proposal = dict(headway=solved['profile']) if solved.get('profile') else None
        if proposal is not None:
            if depot_future:
                proposal['peak_calendar'] = False
            else:
                proposal['bridge_seconds'] = 0
        result = calculate_cost_catalog(data, runner, state, profiles, roster, bundle, second,
            end, deadline, cancelled, proposal=proposal, builder=builder)
        result['decision'].update(solver47=solved, catalogue47=dict(
            fixed_headway_grid_used=False, builder='wait' if depot_future else 'bridge',
            future_service_jobs_may_be_removed=True, waiting_budgets_removed=True))
        return result

    engine.calculate_catalog = calculate
    try:
        engine.run(arm)
    finally:
        after = hashes()
        engine.write_json_once(output / 'economic_extension.json', dict(
            before=before, after=after, sources_preserved=before == after,
            objective='TO1 resource only', waiting_constraints=None, fixed_grid_used=False,
            complete_control_slots=steps, full_day=False, cash_saving_rub=None))
    if before != after:
        raise ValueError('new controller sources changed during replay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['MPC_ML', 'MPC_BASELINE_FORECAST'], default='MPC_ML')
    parser.add_argument('--steps', type=int, default=20)
    run(**vars(parser.parse_args()))
