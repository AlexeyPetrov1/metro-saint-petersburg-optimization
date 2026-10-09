"""Replay the validated dense physical reference against the common S1 cohort."""

import copy
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path
from time import monotonic


def require_eligible(result):
    if not (result['status'] == 'movement_checked' and result['passenger_validation_eligible']
            and result['source_ledger_preserved'] and result['audit']['movement_passed']
            and result['audit']['peak_trains_on_line'] <= 53 and result['terminal_total']['passed']):
        raise ValueError('passenger replay requires preserved and physically validated movement')


def load_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay():
    root = Path(__file__).resolve().parent
    output = root / 'data/processed/metro44_noidle_terminal_20261009'
    result = json.loads((output / 'result.json').read_text('utf-8'))
    require_eligible(result)
    if (output / 'passenger_result.json').exists() or (output / 'runs').exists():
        raise ValueError('refusing to overwrite previous passenger replay')
    ledger = json.loads((output / 'source_ledger.json').read_text('utf-8'))
    frozen = Path(ledger['frozen_snapshot'])
    if any(sha(frozen / name) != value for name,value in ledger['frozen_source_hashes'].items()):
        raise ValueError('frozen physical snapshot changed')
    sys.path.insert(0, str(frozen))
    from goal43_replay import save_run
    from goal43_verify import verify
    from metro_ml_compare import _metrics
    from passenger_groups import GroupedPassengerSimulation
    from simulation import digest
    baseline_path = root / 'data/processed/metro_daily_experiments_20261009/baseline.json.gz'
    shared = load_gzip(baseline_path)
    data = load_gzip(output / 'movement_input.json.gz')
    if digest(shared['data']) != result['input_data_digest']:
        raise ValueError('baseline input changed')
    if digest(shared['groups']) != shared['groups_digest']:
        raise ValueError('observed groups changed')
    receipt = dict(strategy='DENSE_TERMINAL_REFERENCE', forecast_used=False,
        origin_baseline_sha256=sha(baseline_path), original_checkpoint_digest=shared['checkpoint_digest'],
        groups_digest=shared['groups_digest'], movement_input_sha256=sha(output / 'movement_input.json.gz'),
        source_snapshot=str(frozen), source_hashes=ledger['frozen_source_hashes'], runner_sha256=sha(Path(__file__)),
        service_scope='07:00–24:00 declared admission window, with full continuation to 02:00',
        actual_simulated_line_id=1, requested_line_id=2, od_assumption='S1', release_authorized=False)
    (output / 'passenger_contract.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    shutil.copy2(Path(__file__), output / 'passenger_replay_source.py')
    started = monotonic()
    original = GroupedPassengerSimulation(shared['data'], shared['data']['plans'][0]['plan_id'], shared['groups'], shared['profiles'])
    runner = GroupedPassengerSimulation(data, data['plans'][0]['plan_id'], shared['groups'], shared['profiles'])
    state = runner.branch_checkpoint(copy.deepcopy(shared['checkpoint']), original)
    state = runner.run(start_second=3600, checkpoint=state)['checkpoint']
    end = shared['meta']['service_end_second']
    metrics = _metrics(data, runner, state, shared['roster'], data['headway_policy'], 3600, end)
    metrics['peak_trains_on_line'] = runner.motion['checks']['peak_trains_on_line']
    metrics['fully_service_admissible'] = metrics['actual_service_admissible'] and metrics['all_service_arrivals_completed'] and metrics['movement_obligations_closed']
    record = save_run(output, '2026-09-30_DENSE_TERMINAL_REFERENCE_S1_7', data, runner, state,
        shared['roster'], data['headway_policy'], 'physical_reference', 'DENSE_TERMINAL_REFERENCE', 'S1', 7,
        full_day=True, cohort_start=3600, cohort_end=end, service_end_second=end)
    (output / 'runs.json').write_text(json.dumps([record], ensure_ascii=False, indent=2) + '\n', 'utf-8')
    checked = verify(output)
    receipt['source_hashes_preserved'] = all(sha(frozen / name) == value for name,value in ledger['frozen_source_hashes'].items())
    if not receipt['source_hashes_preserved']:
        raise ValueError('physical snapshot changed during replay')
    final = dict(contract=receipt, metrics=metrics, run=record, verification_status=checked['status'],
                 elapsed_seconds=monotonic() - started, release_authorized=False)
    (output / 'passenger_result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps(dict(strategy=receipt['strategy'], elapsed_seconds=final['elapsed_seconds'],
        metrics={k:v for k,v in metrics.items() if k not in {'per_train','policy_exposure','headways','resources','control_window_resources'}}), ensure_ascii=False), flush=True)
    return final


if __name__ == '__main__':
    replay()
