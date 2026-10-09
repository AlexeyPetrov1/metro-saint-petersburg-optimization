"""Labelled synthetic access/demand stresses; never a measured economic gain."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from metro48_forecasts import profiled_median, reprofile_hourly, score_rows
from metro48_gru import FrozenQuarterGRU
from metro_ml import FrozenMetroForecaster


def inject_entries(frame, stations, start, end, factor):
    changed=frame.copy()
    chosen=changed.station_id.isin(stations)&changed.interval_start.ge(pd.Timestamp(start))&changed.interval_start.lt(pd.Timestamp(end))
    changed.loc[chosen,'entry_count']=np.ceil(changed.loc[chosen,'entry_count']*factor).astype('int64')
    return changed


def known_closure(bundle, event, as_of):
    result=deepcopy(bundle)
    if pd.Timestamp(event['available_at'])>pd.Timestamp(as_of):
        return result
    for row in result['rows']:
        if (row['station_id']==event['station_id'] and pd.Timestamp(row['interval_start'])>=pd.Timestamp(event['start'])
                and pd.Timestamp(row['interval_end'])<=pd.Timestamp(event['end'])):
            row['predicted_count']=0
    result['included_event_ids'].append(event['id'])
    result['source_versions']['known_access']=dict(event=event,latent_demand_observed=False,future_actual_used=False)
    return result


def run(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent; path=root/'data/processed/metro44_catalog_pilot_20261009/observed_frame.parquet'
    protected=[path,Path(__file__),root/'metro48_gru.py',root/'metro48_forecasts.py']
    hashes=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    before=hashes(); frame=pd.read_parquet(path); model=FrozenQuarterGRU(); old=FrozenMetroForecaster(use_retrospective_history_flags=False)
    ids=list(model.mapping)
    closure=dict(id='synthetic-access',station_id='L1_01',available_at='2026-09-30T09:37:13+03:00',
        start='2026-09-30T09:45:00+03:00',end='2026-09-30T10:15:00+03:00')
    cases={'normal':frame,
        'surge_north_50pct':inject_entries(frame,['L1_15','L1_18','L1_19'],closure['start'],'2026-09-30T10:30:00+03:00',1.5),
        'station_access_closed_reopened':inject_entries(frame,['L1_01'],closure['start'],closure['end'],0)}
    times=pd.date_range('2026-09-30T09:30:00+03:00',periods=8,freq='15min')
    contract=dict(synthetic=True,dates_selected_before_scores=True,decision_times=[t.isoformat() for t in times],
        source_unchanged=True,truth_injection_is_offline_only=True,cash_saving_rub=None,
        claims='forecast adaptation only; movement disruption requires telemetry',input_hashes=before,closure=closure)
    (output/'contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    predictions=[]
    for name,truth in cases.items():
        started=monotonic()
        for at in times:
            bundle=model.forecast_bundle(truth,ids,at.isoformat())
            bundles=dict(GRU=bundle,ML=reprofile_hourly(old.forecast_bundle(truth,ids,at.isoformat()),model.station_profile),
                MEDIAN=profiled_median(truth,ids,at.isoformat(),profile_provider=model.station_profile))
            if name=='station_access_closed_reopened':
                bundles.update({method+'_ACCESS':known_closure(b,closure,at.isoformat()) for method,b in list(bundles.items())})
            for method,b in bundles.items():
                for row in b['rows']:
                    if pd.Timestamp(row['interval_start'])==at:
                        predictions.append(dict(row,issued_at=at.isoformat(),case=name,method=method))
        print(json.dumps(dict(stage='anomaly_forecasts',case=name,elapsed_seconds=monotonic()-started)),flush=True)
    table=pd.DataFrame(predictions); scores=[]
    for (name,method),part in table.groupby(['case','method'],sort=False):
        truth=cases[name]
        selected=part.drop(columns=['case','method']).to_dict('records')
        score=score_rows(selected,truth)
        affected=part[part.station_id.isin(['L1_15','L1_18','L1_19'] if name=='surge_north_50pct' else ['L1_01'])]
        scores.append(dict(case=name,method=method,**score,affected=score_rows(
            affected.drop(columns=['case','method']).to_dict('records'),truth)))
    table.to_csv(output/'predictions.csv',index=False,encoding='utf-8-sig')
    if before!=hashes():
        raise ValueError('anomaly replay changed protected assets')
    result=dict(contract=contract,scores=scores,sources_preserved=True,
        economic_gain_proven=False,movement_recovery_compared=False,external_detection_latency_unknown=True)
    (output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(**vars(parser.parse_args()))
