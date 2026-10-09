import copy

import pandas as pd
import pytest

from forecasting import validate_bundle
from metro45_risk_envelope import build_risk_envelope, model_scenarios
from simulation import digest


NOW='2026-09-30T10:15:00+03:00'


def forecasts():
    start=pd.Timestamp('2026-09-30T10:00:00+03:00')
    original=dict(schema_version='1.0',model_version='frozen-model',feature_version='hourly-v1',
        issued_at=start.isoformat(),as_of=start.isoformat(),available_at=start.isoformat(),
        features_available_at=start.isoformat(),target='turnstile_entries',unit='passengers_per_15m',
        coverage='complete',included_event_ids=[],source_versions={'model_sha256':{'q50':'original'}},
        rows=[dict(station_id=s,interval_start=(start+pd.Timedelta(minutes=15*q)).isoformat(),
            interval_end=(start+pd.Timedelta(minutes=15*(q+1))).isoformat(),predicted_count=100,
            coverage='complete') for s in ('A','B') for q in range(8)])
    adjusted=copy.deepcopy(original)
    adjusted.update(as_of=NOW,issued_at=NOW,available_at=NOW,features_available_at=NOW)
    adjusted['rows']=[r for r in adjusted['rows'] if pd.Timestamp(r['interval_start'])>=pd.Timestamp(NOW)]
    for station in ('A','B'):
        for row,count in zip([r for r in adjusted['rows'] if r['station_id']==station], [80,110,95,120,100,90,150]):
            row['predicted_count']=count
    adjusted['source_versions']['residual_nowcast']={'original_bundle_digest':digest(original)}
    return original,adjusted


def counts(bundle):
    return [r['predicted_count'] for r in bundle['rows'] if r['station_id']=='A']


def test_future_guard_is_exact_max_and_preserves_both_forecast_provenances():
    original,adjusted=forecasts()
    guard=build_risk_envelope(original,adjusted,NOW)
    assert counts(guard)==[100,110,100,120,100,100,150]
    assert guard['issued_at']==guard['available_at']==NOW
    assert all(pd.Timestamp(r['interval_start'])>=pd.Timestamp(NOW) for r in guard['rows'])
    meta=guard['source_versions']['risk_envelope']
    assert meta['original_bundle_digest']==digest(original)
    assert meta['adjusted_bundle_digest']==digest(adjusted)
    assert meta['rule']=='elementwise_max'
    assert not meta['is_point_estimate'] and not meta['train_load_guarantee']
    assert guard['source_versions']['model_sha256']==original['source_versions']['model_sha256']
    assert validate_bundle(guard,['A','B'],NOW)['status']=='passed'


def test_guard_never_increases_entry_underprediction_for_any_nonnegative_truth():
    original,adjusted=forecasts()
    guard=build_risk_envelope(original,adjusted,NOW)
    base={ (r['station_id'],r['interval_start']):r['predicted_count'] for r in original['rows'] }
    for truth in (0,1,50,95,100,115,160,1000):
        for row in guard['rows']:
            before=base[(row['station_id'],row['interval_start'])]
            assert max(0,truth-row['predicted_count'])<=max(0,truth-before)


def test_model_scenarios_are_independent_copies_and_inputs_remain_immutable():
    original,adjusted=forecasts()
    before_original,before_adjusted=copy.deepcopy(original),copy.deepcopy(adjusted)
    base,point,guard=model_scenarios(original,adjusted,NOW)
    assert counts(base)==[100]*7 and counts(point)==[80,110,95,120,100,90,150]
    base['source_versions']['model_sha256']['q50']='changed'
    point['rows'][0]['predicted_count']=999
    assert original==before_original and adjusted==before_adjusted
    assert guard['source_versions']['model_sha256']['q50']=='original'


@pytest.mark.parametrize('damage',['missing','extra','duplicate','shifted_end'])
def test_future_station_timestamp_coverage_must_match_exactly(damage):
    original,adjusted=forecasts()
    if damage=='missing':
        adjusted['rows']=[r for r in adjusted['rows'] if not r['interval_start'].endswith('11:45:00+03:00')]
    if damage=='extra':
        for station in ('A','B'):
            adjusted['rows'].append(dict(station_id=station,interval_start='2026-09-30T12:00:00+03:00',
                interval_end='2026-09-30T12:15:00+03:00',predicted_count=100,coverage='complete'))
    if damage=='duplicate':adjusted['rows'].append(copy.deepcopy(adjusted['rows'][0]))
    if damage=='shifted_end':adjusted['rows'][0]['interval_end']='2026-09-30T10:45:00+03:00'
    with pytest.raises(ValueError):build_risk_envelope(original,adjusted,NOW)


@pytest.mark.parametrize('damage',['future_issue','future_available','future_features','stale_original','naive'])
def test_future_or_stale_timestamps_cannot_be_promoted_as_current_guard(damage):
    original,adjusted=forecasts()
    if damage=='future_issue':adjusted['issued_at']='2026-09-30T10:16:00+03:00'
    if damage=='future_available':adjusted['available_at']='2026-09-30T10:16:00+03:00'
    if damage=='future_features':adjusted['features_available_at']='2026-09-30T10:16:00+03:00'
    if damage=='stale_original':original['issued_at']=original['available_at']='2026-09-30T09:00:00+03:00'
    with pytest.raises(ValueError):
        build_risk_envelope(original,adjusted,'2026-09-30T10:15:00' if damage=='naive' else NOW)


@pytest.mark.parametrize('damage',['weights','base_digest','unproven_change','double_guard'])
def test_weight_or_provenance_mismatch_and_double_guard_are_rejected(damage):
    original,adjusted=forecasts()
    if damage=='weights':adjusted['source_versions']['model_sha256']['q50']='another-model'
    if damage=='base_digest':adjusted['source_versions']['residual_nowcast']['original_bundle_digest']='wrong'
    if damage=='unproven_change':adjusted['source_versions'].pop('residual_nowcast')
    if damage=='double_guard':adjusted=build_risk_envelope(original,adjusted,NOW)
    with pytest.raises(ValueError):build_risk_envelope(original,adjusted,NOW)


def test_uncorrected_baseline_passthrough_needs_no_invented_weight_provenance():
    original,adjusted=forecasts()
    original['source_versions']={'source_ids':['median-source']}
    adjusted=copy.deepcopy(original)
    guard=build_risk_envelope(original,adjusted,NOW)
    assert counts(guard)==[100]*7
    assert 'model_sha256' not in guard['source_versions']


def test_guard_publication_is_current_computation_even_when_both_input_publications_are_earlier():
    original,_=forecasts()
    guard=build_risk_envelope(original,copy.deepcopy(original),NOW)
    assert guard['issued_at']==guard['available_at']==NOW
    assert guard['features_available_at']==original['features_available_at']


def test_guard_issued_time_is_no_earlier_than_both_input_publications():
    original,adjusted=forecasts()
    adjusted['available_at']='2026-09-30T10:15:01+03:00'
    guard=build_risk_envelope(original,adjusted,'2026-09-30T10:15:01+03:00')
    assert pd.Timestamp(guard['issued_at'])>=pd.Timestamp(adjusted['available_at'])
    assert len(guard['rows'])==12
