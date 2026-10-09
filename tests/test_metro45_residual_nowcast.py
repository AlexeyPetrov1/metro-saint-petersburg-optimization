import copy

import pandas as pd
import pytest

from forecasting import validate_bundle
from metro45_residual_nowcast import adjust_bundle
from simulation import digest


NOW = '2026-09-30T10:15:00+03:00'


def bundle(hour=10, count=100):
    start = pd.Timestamp(f'2026-09-30T{hour:02}:00:00+03:00')
    return dict(schema_version='1.0', model_version='frozen-model', feature_version='hourly-v1',
        issued_at=start.isoformat(), as_of=start.isoformat(), available_at=start.isoformat(),
        features_available_at=start.isoformat(), target='turnstile_entries', unit='passengers_per_15m',
        coverage='complete', included_event_ids=[], source_versions={'model_sha256': {'q50': 'original'}},
        refresh_seconds=3600, rows=[dict(station_id=s, interval_start=(start+pd.Timedelta(minutes=15*q)).isoformat(),
            interval_end=(start+pd.Timedelta(minutes=15*(q+1))).isoformat(), predicted_count=count,
            coverage='complete') for s in ('A', 'B') for q in range(8)])


def observed(counts=(200, 200)):
    return pd.DataFrame([dict(station_id=s, interval_start=start,
        interval_end=start+pd.Timedelta(minutes=15), entry_count=count, is_complete=True, source_id='observed')
        for s in ('A', 'B') for start, count in zip(
            pd.to_datetime(['2026-09-30T09:45:00+03:00', '2026-09-30T10:00:00+03:00']), counts)])


def predict(frame=None, past=None, **kwargs):
    return adjust_bundle(bundle(), observed() if frame is None else frame, NOW,
        past_bundles=[bundle(9)] if past is None else past, **kwargs)


def counts(out):
    return [r['predicted_count'] for r in out['rows'] if r['station_id']=='A']


def metadata(out):
    return out['source_versions']['residual_nowcast']


def test_prior_predictions_adjust_future_quarters_and_decay_to_original_within_one_hour():
    out=predict()
    assert counts(out)==[115,111,108,104,100,100,100]
    assert out['as_of']==out['issued_at']==out['available_at']==NOW
    assert out['source_versions']['model_sha256']==bundle()['source_versions']['model_sha256']
    assert out['model_version']==bundle()['model_version']
    assert metadata(out)['station_adjustments']['A']['observed_recent_count']==400
    assert metadata(out)['station_adjustments']['A']['predicted_recent_count']==200
    assert metadata(out)['station_adjustments']['A']['multiplier']==1.15
    assert validate_bundle(out,['A','B'],NOW)['status']=='passed'


def test_latest_forecast_available_before_each_quarter_is_used_independently_of_list_order():
    past=[bundle(9),bundle(10,50)]
    out=predict(observed((100,100)),past)
    assert counts(out)[0]==108
    assert predict(observed((100,100)),list(reversed(past)))==out
    chosen=[r for r in metadata(out)['used_predictions'] if r['station_id']=='A']
    assert [r['forecast_issued_at'] for r in chosen]==['2026-09-30T09:00:00+03:00','2026-09-30T10:00:00+03:00']
    assert all(pd.Timestamp(r['forecast_available_at'])<=pd.Timestamp(r['interval_start']) for r in chosen)


def test_forecast_published_after_interval_start_does_not_replace_the_earlier_reference():
    late=bundle(10,1)
    late['available_at']='2026-09-30T10:02:00+03:00'
    out=predict(observed((100,100)),[bundle(9),late])
    assert counts(out)==[100]*7
    assert all(r['forecast_issued_at'].endswith('09:00:00+03:00') for r in metadata(out)['used_predictions'])


def test_current_forecast_that_saw_completed_labels_cannot_predict_those_past_intervals():
    frame=observed((200,200))
    frame['interval_start']-=pd.Timedelta(minutes=15)
    frame['interval_end']-=pd.Timedelta(minutes=15)
    out=adjust_bundle(bundle(),frame,'2026-09-30T10:00:00+03:00',past_bundles=[bundle()])
    assert counts(out)==[100]*8
    assert metadata(out)['station_adjustments']['A']['status']=='no_prior_predictions'


def test_future_labels_and_future_issued_forecasts_cannot_change_prediction_or_provenance():
    frame=observed()
    future=frame.copy()
    future['interval_start']+=pd.Timedelta(hours=2)
    future['interval_end']+=pd.Timedelta(hours=2)
    future['entry_count']=-99999
    future_bundle=bundle(11,-99999)
    assert predict(pd.concat([frame,future]),[bundle(9),future_bundle])==predict(frame,[bundle(9)])


@pytest.mark.parametrize('column',['available_at','received_at'])
@pytest.mark.parametrize('value',['2026-09-30T10:16:00+03:00',None])
def test_complete_but_delayed_or_unknown_observations_do_not_enter_the_residual(column,value):
    frame=observed((100,200))
    frame[column]=frame.interval_end.copy()
    frame.loc[frame.interval_start.dt.hour.eq(10),column]=pd.Timestamp(value) if value else pd.NaT
    out=predict(frame)
    assert counts(out)==[100]*7
    assert metadata(out)['station_adjustments']['A']['supported_intervals']==1


def test_interval_completion_boundary_and_observation_receipt_are_both_enforced():
    frame=observed((100,200))
    frame['received_at']=frame.interval_end.copy()
    frame.loc[frame.interval_start.dt.hour.eq(10),'received_at']=pd.Timestamp('2026-09-30T10:16:00+03:00')
    assert counts(predict(frame))==[100]*7
    out=adjust_bundle(bundle(),frame,'2026-09-30T10:16:00+03:00',past_bundles=[bundle(9)])
    assert counts(out)[0]>100
    assert metadata(out)['station_adjustments']['A']['supported_intervals']==1
    unfinished=adjust_bundle(bundle(),observed(), '2026-09-30T10:07:00+03:00',past_bundles=[bundle(9)])
    assert metadata(unfinished)['station_adjustments']['A']['supported_intervals']==1
    assert all(pd.Timestamp(r['interval_start'])>=pd.Timestamp('2026-09-30T10:07:00+03:00') for r in unfinished['rows'])


def test_incomplete_missing_forecast_and_zero_prediction_have_explicit_unchanged_fallback():
    frame=observed()
    frame['is_complete']=False
    assert counts(predict(frame))==[100]*7
    assert metadata(predict(frame))['station_adjustments']['A']['status']=='no_recent_observations'
    assert counts(predict(past=[]))==[100]*7
    assert metadata(predict(past=[]))['station_adjustments']['A']['status']=='no_prior_predictions'
    assert counts(predict(past=[bundle(9,0)]))==[100]*7
    assert metadata(predict(past=[bundle(9,0)]))['station_adjustments']['A']['status']=='zero_predicted_volume'


def test_forecasts_from_another_model_are_not_residual_references():
    other=bundle(9)
    other['source_versions']['model_sha256']['q50']='another'
    assert counts(predict(past=[other]))==[100]*7


def test_input_forecast_observations_and_archived_predictions_remain_immutable():
    base,frame,past=bundle(),observed(),[bundle(9)]
    old_base,old_frame,old_past=copy.deepcopy(base),frame.copy(deep=True),copy.deepcopy(past)
    out=adjust_bundle(base,frame,NOW,past_bundles=past)
    assert base==old_base and past==old_past
    pd.testing.assert_frame_equal(frame,old_frame)
    assert metadata(out)['original_bundle_digest']==digest(base)


def test_already_adjusted_current_or_past_predictions_cannot_be_corrected_twice():
    out=predict()
    with pytest.raises(ValueError,match='already'):
        adjust_bundle(out,observed(),NOW,past_bundles=[bundle(9)])
    for tag in ('nowcast','residual_nowcast'):
        past=bundle(9)
        past['source_versions'][tag]={}
        with pytest.raises(ValueError,match='original'):
            predict(past=[past])


def test_conflicting_forecast_records_at_same_publication_time_are_rejected():
    with pytest.raises(ValueError,match='conflicting'):
        predict(past=[bundle(9,100),bundle(9,200)])


def test_rounding_is_half_up_and_final_changes_are_bounded_integer_counts():
    out=adjust_bundle(bundle(10,10),observed((120,120)),NOW,past_bundles=[bundle(9)])
    assert counts(out)[0]==11
    assert all(type(r['predicted_count']) is int for r in out['rows'])
    assert counts(predict(observed((0,0))))[0]==85
    out=adjust_bundle(bundle(10,0),observed(),NOW,past_bundles=[bundle(9)])
    assert counts(out)==[0]*7


@pytest.mark.parametrize('kwargs',[dict(gain=-1),dict(max_relative_change=float('nan')),
    dict(recent_minutes=20),dict(decay_minutes=0)])
def test_invalid_parameters_and_stale_original_bundle_are_not_promoted(kwargs):
    with pytest.raises(ValueError):
        predict(**kwargs)
    with pytest.raises(ValueError,match='stale'):
        adjust_bundle(bundle(),observed(),'2026-09-30T11:00:00+03:00',past_bundles=[bundle(9)])
