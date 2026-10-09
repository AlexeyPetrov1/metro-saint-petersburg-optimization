"""Replay the preserved twenty control slots with quarter-hour feedback."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'data/processed/metro44_catalog_cont6h_20261009/campaign.py'


def run(output, *, arm='MPC_ML', guard=False, quality='strict', steps=20):
    if guard and arm != 'MPC_ML':
        raise ValueError('residual guard requires ML')
    if type(steps) is not int or not 1 <= steps <= 20:
        raise ValueError('one to twenty consecutive control slots required')
    spec = importlib.util.spec_from_file_location('_preserved_catalog45', OLD)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    from metro45_controller import prepare_candidates
    from metro45_phase import rephase_bridge
    from metro44_planner_bridge import rebuild_bridge_timetable
    from metro45_residual_nowcast import adjust_bundle
    from metro45_risk_envelope import build_risk_envelope
    import pandas as pd
    engine.OUTPUT = Path(output).resolve()
    engine.OUTPUT.mkdir(parents=True, exist_ok=False)
    shared, common = engine.prepare()
    files = [Path(__file__).resolve(), OLD] + [ROOT / name for name in (
        'metro45_controller.py', 'metro45_adaptive.py', 'metro45_phase.py',
        'metro45_nowcast.py', 'metro45_residual_nowcast.py', 'metro45_risk_envelope.py')]
    extra_hashes = {str(path): engine.sha(path) for path in files}
    common.update(quality_policy=quality, budgets_are_user_requirements=False,
        mean_budget_seconds=90 if quality == 'pareto' else None,
        p95_budget_seconds=180 if quality == 'pareto' else None,
        decision_times=common['decision_times'][:steps], controller45=True,
        full_twenty_slots=steps == 20, residual_guard_used=guard)
    common['input_hashes'].update(extra_hashes)
    # Complete the new contract before running any decision or freezing receipts.
    (engine.OUTPUT / 'shared_contract.json').write_text(json.dumps(common, indent=2), 'utf-8')
    engine.prepare = lambda: (copy.deepcopy(shared), copy.deepcopy(common))
    original_calculate = engine.calculate_catalog
    frame = None
    history = []
    def calculate(*args, **kwargs):
        nonlocal frame
        data, runner, state, profiles, roster, source, strategy, ids, second, end = args[:10]
        raw = kwargs['bundle_override']
        from datetime import datetime, timedelta
        at = (datetime.fromisoformat(data['period_start']) + timedelta(seconds=second)).isoformat()
        bundle = raw
        if guard:
            if frame is None:
                frame = pd.read_parquet(engine.PILOT / 'observed_frame.parquet')
                hour = datetime.fromisoformat(at).hour
                history.extend(engine.read(engine.CACHE / 'forecasts' / f'ML_{h:02}.json')
                               for h in range(7, hour + 1))
            corrected = adjust_bundle(raw, frame, at, past_bundles=history)
            bundle = build_risk_envelope(raw, corrected, at)
            engine.write_json_once(engine.OUTPUT / 'controller' / arm / f'guard_{second:05}.json', bundle)
        specs, metadata = prepare_candidates(data, bundle, state, ids, second, end)
        result = original_calculate(*args, quality_policy=quality,
            builders=dict(bridge=rebuild_bridge_timetable, phase=rephase_bridge),
            profile_specs=specs, bundle_override=bundle)
        result['decision']['feedback45'] = dict(**metadata, residual_guard_used=guard,
            exact_motion_and_passenger_validation_used=True)
        if guard:
            history.append(copy.deepcopy(raw))
        return result
    engine.calculate_catalog = calculate
    engine.run(arm)
    after = {str(path): engine.sha(path) for path in files}
    engine.write_json_once(engine.OUTPUT / 'feedback_extension.json', dict(
        sources_before=extra_hashes, sources_after=after, sources_preserved=extra_hashes == after,
        residual_guard=guard, quality_policy=quality, completed_control_slots=steps,
        independent_holdout=False, online_pipeline_includes_actual_inference=True))
    if after != extra_hashes:
        raise ValueError('feedback extension changed during replay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['MPC_ML', 'MPC_BASELINE_FORECAST'], default='MPC_ML')
    parser.add_argument('--guard', action='store_true')
    parser.add_argument('--quality', choices=['strict', 'pareto'], default='strict')
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    run(args.output, arm=args.arm, guard=args.guard, quality=args.quality, steps=args.steps)
