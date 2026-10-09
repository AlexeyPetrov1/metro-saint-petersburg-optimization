"""Joint train withdrawals with live forecasts and fifteen-minute feedback."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'data/processed/metro44_catalog_cont6h_20261009/campaign.py'


def normalize_shared(shared, normalizer, runner_class):
    original = shared['data']
    changed, audit = normalizer(original, 3600, 22200)
    before = runner_class(original, original['plans'][0]['plan_id'], shared['groups'], shared['profiles'])
    after = runner_class(changed, changed['plans'][0]['plan_id'], shared['groups'], shared['profiles'])
    result = copy.deepcopy(shared)
    result['data'] = changed
    result['state'] = after.branch_checkpoint(shared['state'], before)
    return result, audit


def run(output, *, arm='MPC_ML', steps=20):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(steps) is not int or not 1 <= steps <= 20:
        raise ValueError('one to twenty consecutive control steps required')
    if arm not in ('MPC_ML', 'MPC_BASELINE_FORECAST', 'REFERENCE'):
        raise ValueError('causal ML or baseline required')
    spec = importlib.util.spec_from_file_location('_preserved_withdrawal47', OLD)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    from metro47_catalog import calculate_cost_catalog
    from metro47_withdrawal import build_withdrawal_candidate, close_service_at
    from metro47_withdrawal_solver import solve_withdrawals
    from metro47_withdrawal_control import refine_withdrawals
    paths = [Path(__file__).resolve(), OLD] + [ROOT / name for name in (
        'metro47_catalog.py', 'metro47_cost_policy.py', 'metro47_withdrawal.py',
        'metro47_withdrawal_solver.py', 'metro47_withdrawal_control.py')]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    output.mkdir(parents=True)
    shared, common = engine.prepare()
    shared, normalization = normalize_shared(shared, close_service_at, engine.GroupedPassengerSimulation)
    common.update(original_checkpoint_digest=common['checkpoint_digest'],
        checkpoint_digest=engine.digest(shared['state']))
    common.update(quality_policy='cost_only', budgets_are_user_requirements=True,
        mean_budget_seconds=None, p95_budget_seconds=None, decision_times=common['decision_times'][:steps],
        controller47=True, full_twenty_slots=steps == 20, requested_line_id=1,
        objective='minimum TO1 resource cost subject to operational constraints',
        acceptance_target_saving_fraction=0.01, full_day=False, fixed_headway_grid_used=False,
        reference_only=arm == 'REFERENCE', baseline_normalization=normalization)
    common['input_hashes'].update(before)
    engine.write_json_once(output / 'shared_contract.json', common)
    engine.write_gzip_once(output / 'shared_scenario.json.gz', shared)
    engine.OUTPUT = output
    engine.prepare = lambda: (copy.deepcopy(shared), copy.deepcopy(common))
    engine.strict_counterfactual = lambda decision: None

    def calculate(*args, **kwargs):
        data, runner, state, profiles, roster, _, _, _, second, end, _, deadline, cancelled = args[:13]
        bundle = kwargs['bundle_override']
        if arm == 'REFERENCE':
            return calculate_cost_catalog(data, runner, state, profiles, roster, bundle,
                second, end, deadline, cancelled)

        def solve(**constraints):
            return solve_withdrawals(data, bundle, state, second, end,
                deadline=deadline, **constraints)

        def evaluate(ids):
            return calculate_cost_catalog(data, runner, state, profiles, roster, bundle, second,
                end, deadline, cancelled,
                proposal=dict(selected_train_ids=ids) if ids else None,
                builder=build_withdrawal_candidate)

        return refine_withdrawals(solve, evaluate, deadline=deadline, cancelled=cancelled)

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
        raise ValueError('withdrawal controller sources changed during replay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['MPC_ML', 'MPC_BASELINE_FORECAST', 'REFERENCE'], default='MPC_ML')
    parser.add_argument('--steps', type=int, default=20)
    run(**vars(parser.parse_args()))
