"""Exact forecast validation of one solver proposal and its unchanged reference."""

import copy
from datetime import datetime, timedelta
from time import monotonic

from forecasting import validate_bundle
from metro44_catalog import physical_gate
from metro44_run import closed_per_train
from metro47_cost_policy import choose_by_cost
from rolling import decide_operational
from simulation import digest


def calculate_cost_catalog(data, runner, state, profiles, roster, bundle, second,
                           planning_end, deadline, cancelled, *, proposal=None, builder=None,
                           passenger_service_check=None, selection_policy=None, forecast_horizon_seconds=7200):
    began = monotonic()

    def check():
        if monotonic() >= deadline or cancelled is not None and cancelled.is_set():
            raise TimeoutError('cost controller deadline')

    check()
    versions = bundle.get('source_versions', {})
    if versions.get('future_truth_intentionally_used') or versions.get('operationally_available') is False:
        raise ValueError('causal forecasts required')
    at = (datetime.fromisoformat(data['period_start']) + timedelta(seconds=second)).isoformat()
    ids = [s['station_id'] for s in data['stations'] if s['kind'] == 'line']
    validate_bundle(bundle, ids, at)
    candidates, rejected, seen = [], [], set()

    def record(name, changed, metadata):
        check()
        audit = physical_gate(changed, second, planning_end, reference=data,
            reference_motion=getattr(runner, 'motion', None), deadline=deadline)
        if not audit['movement_passed']:
            rejected.append(dict(candidate_id=name, status='rejected',
                reason='independent physical gate failed', audit=audit))
            return
        identity = digest(changed)
        if identity in seen:
            return
        seen.add(identity)
        candidates.append(dict(candidate_id=name, strategy='KEEP' if name == 'KEEP' else 'REPLAN',
            status='movement_checked', _data=changed, audit={**metadata, **audit},
            action=dict(kind='KEEP' if name == 'KEEP' else 'REPLAN', decision_second=second)))

    record('KEEP', data, {})
    if proposal is not None and builder is not None:
        try:
            changed, audit = builder(copy.deepcopy(data), second, planning_end,
                deadline=deadline, phase_seconds=0, **proposal)
            record('COST_SOLVER', changed, audit)
        except ValueError as error:
            rejected.append(dict(candidate_id='COST_SOLVER', status='rejected', reason=str(error)))
    check()
    decision = (decide_operational(data, {'S1': runner}, {'S1': state}, profiles, {}, bundle,
        data['headway_policy'], candidates=candidates,
        budget_seconds=max(0, deadline-monotonic()-5), include_recovery=False,
        **({'passenger_service_check': passenger_service_check} if passenger_service_check is not None else {}),
        **({'forecast_horizon_seconds': forecast_horizon_seconds, 'service_end_second': planning_end}
           if forecast_horizon_seconds != 7200 else {}))
        if candidates else dict(candidate_table=[], search_complete=True))
    check()
    by_id = {r['candidate_id']: r for r in candidates}
    for row in decision['candidate_table']:
        if row.get('status') == 'evaluated' and row.get('passengers_checked') and row.get('physical_passed'):
            candidate = by_id[row['candidate_id']]
            row['committed_per_train'] = closed_per_train(candidate['_data'], roster,
                transport_work=candidate['audit']['_transport_work'])
    decision['candidate_table'].extend(rejected)
    decision['selection'] = (selection_policy or choose_by_cost)(decision['candidate_table'],
        search_incomplete=not decision.get('search_complete', False))
    decision.update(decision_second=second, decision_time=at, forecast_digest=digest(bundle),
        forecast_issued_at=bundle.get('issued_at'), waiting_constraints=None,
        cost_policy='minimum checked TO1 resource cost; waiting is measured only',
        future_actual_demand_used=False, cash_saving_rub=None,
        elapsed_seconds=monotonic()-began)
    if passenger_service_check is not None:
        decision.update(waiting_constraints='peak first/next eligible train service gate',
                        cost_policy='minimum checked maintenance resource cost with peak boarding constraints')
    check()
    return dict(decision=decision, candidates=candidates)
