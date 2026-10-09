"""Immediate local incident diagnosis for known and revised N-minute stops."""

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from dwell import compile_operational_timing
from goal43_notifications import analytics
from metro44_catalog import physical_gate
from metro48_incidents import process_incident
from simulation import digest, simulate


def stop_event(data,motion):
    event=next(e for e in motion['events'] if e['event_type']=='arrival' and e['location']=='L1_12' and 3600<e['second']<4500)
    trip=next(t for t in data['plans'][0]['trips'] if t['trip_id']==event['trip_id'])
    return dict(train_id=event['train_id'],station_id=event['location'],
        direction='up' if trip['route'][0].endswith('_up') else 'down',detected_second=event['second']+1)


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    source=root/'data/processed/metro48_reference_20261009_v2/runs/Q48_REFERENCE_temporary'
    data=json.loads((source/'movement_input.json').read_text(encoding='utf-8'))
    motion=json.loads((source/'motion.json').read_text(encoding='utf-8'))
    paths=[source/'movement_input.json',source/'motion.json',Path(__file__),root/'simulation.py',root/'metro48_incidents.py']
    hashes=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before=hashes();event=stop_event(data,motion);origin=datetime.fromisoformat(data['period_start'])
    at=lambda second:(origin+timedelta(seconds=second)).isoformat()
    results=[]
    for duration in (3,8,15):
        detected=event['detected_second'];identity=f'synthetic_stop_{duration}'
        incident=dict(event,incident_id=identity,severity='stop',allowed_actions=['regulate','reserve'],
            estimated_end_at=at(detected+duration*60),source='explicit injected fixture, not real telemetry')
        captured=[]
        def calculate(known,stamp,deadline,cancelled):
            estimate=int((datetime.fromisoformat(known['estimated_end_at'])-origin).total_seconds())
            changed,timing=compile_operational_timing(data,data['plans'][0]['plan_id'],'fixed',7,
                policy=data['headway_policy'],incidents=[dict(event,expected_end_second=estimate)],as_of=detected)
            new=simulate(changed,changed['plans'][0]['plan_id'])
            prefix=[e for e in new['events'] if e['second']<detected]==[e for e in motion['events'] if e['second']<detected]
            gate=physical_gate(changed,detected,22200,reference=changed,deadline=deadline)
            gate.pop('_transport_work',None)
            captured.append(dict(estimate_end_second=estimate,executed_prefix_preserved=prefix,
                movement_input_digest=digest(changed),physical=gate,
                incident_changes_future_movements=True,demand_forecast_changed_by_stop=False))
            passed=gate['movement_passed'] and prefix
            return dict(validated=passed,status='movement_checked' if passed else 'no_checked_schedule',
                action='retain trains; regulate departures around the blocked platform' if passed else None,
                protected_platform=event['station_id'],minimum_headway=gate['minimum_headway'],
                maximum_headway=gate['maximum_headway'],passengers_checked=False,
                scope='known-duration movement diagnosis; economic withdrawal suppressed until recovery',
                cash_saving_rub=None)
        ledger,receipt=process_incident([],incident,at(detected),calculate)
        results.append(dict(case=identity,duration_minutes=duration,receipt=receipt,
            ledger=ledger,analytics=analytics(ledger),diagnostics=captured,
            synthetic=True,passenger_recovery_compared=False,economic_gain_proven=False))
        print(json.dumps(dict(stage='stop_diagnosis',duration=duration,receipt=receipt)),flush=True)
    if before!=hashes():
        raise ValueError('incident diagnosis changed protected sources')
    result=dict(cases=results,source_hashes=before,sources_preserved=True,
        trigger='incident timestamp, never waits for next900s tick',
        revised_duration_supported_by_test=True,notification_delivery_to_dispatcher_unknown=True,
        actual_traffic_incidents_not_available=True,cash_saving_rub=None)
    (output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(**vars(parser.parse_args()))
