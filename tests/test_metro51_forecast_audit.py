import json

import pandas as pd
import pytest

from metro51_forecast_audit import evaluate_bundles, summarize, write_review


def _frame():
    times=pd.date_range('2026-09-30T08:00:00+03:00',periods=3,freq='15min')
    return pd.DataFrame(dict(station_id=['L1_17']*3,interval_start=times,
        interval_end=times+pd.Timedelta(minutes=15),entry_count=[100,200,0],is_complete=[True]*3))


def _bundle():
    frame=_frame()
    rows=[dict(station_id=r.station_id,interval_start=r.interval_start.isoformat(),
        interval_end=r.interval_end.isoformat(),predicted_count=p,coverage='complete')
        for r,p in zip(frame.itertuples(),[110,180,0])]
    quantiles=[dict(r,q50=r['predicted_count'],q10=lo,q90=hi)
        for r,lo,hi in zip(rows,[90,170,0],[120,190,0])]
    return dict(as_of='2026-09-30T08:00:00+03:00',issued_at='2026-09-30T08:00:00+03:00',
        model_version='test',rows=rows,source_versions=dict(quantile_rows=quantiles,
        shifted_quarter_inference=False,prediction_quantile='q50'))


def test_forecast_metrics_use_actual_future_only_for_evaluation():
    scored=evaluate_bundles([('BALANCED','a.json',_bundle())],_frame())
    summary=summarize(scored)[0]
    assert summary['wape']==pytest.approx(.1)
    assert summary['mae']==pytest.approx(10)
    assert summary['bias_fraction']==pytest.approx(-10/300)
    assert summary['q10_q90_coverage']==pytest.approx(2/3)
    assert summary['q90_upper_coverage']==pytest.approx(2/3)
    assert set(scored.target_peak)=={True}
    assert scored.lead_start_minutes.tolist()==[0,15,30]
    assert scored.lead_end_minutes.tolist()==[15,30,45]


def test_overlapping_forecasts_are_reported_as_repeated_targets():
    a=_bundle()
    b=json.loads(json.dumps(a))
    b['as_of']=b['issued_at']='2026-09-30T08:15:00+03:00'
    b['rows']=b['rows'][1:]
    b['source_versions']['quantile_rows']=b['source_versions']['quantile_rows'][1:]
    scored=evaluate_bundles([('BALANCED','a.json',a),('BALANCED','b.json',b)],_frame())
    summary=summarize(scored)[0]
    assert summary['n_predictions']==5
    assert summary['unique_station_targets']==3
    assert summary['n_origins']==2
    assert summary['overlapping_forecasts']


def test_incomplete_and_missing_truth_are_excluded_without_zero_imputation():
    frame=_frame().iloc[:2].copy()
    frame.loc[frame.index[1],'is_complete']=False
    scored=evaluate_bundles([('OLD','a.json',_bundle())],frame)
    assert scored.truth_status.tolist()==['complete','incomplete','missing']
    summary=summarize(scored)[0]
    assert summary['n_scored']==1
    assert summary['missing_truth']==1
    assert summary['incomplete_truth']==1


def test_uncalibrated_fallback_intervals_are_not_counted_as_coverage_proof():
    bundle=_bundle()
    bundle['source_versions']['uncalibrated_interval_station_ids']=['L1_17']
    scored=evaluate_bundles([('BALANCED','a.json',bundle)],_frame())
    summary=summarize(scored)[0]
    assert summary['n_interval_scored']==0
    assert summary['q10_q90_coverage'] is None


def test_old_model_without_quantiles_still_gets_point_error_metrics():
    bundle=_bundle()
    bundle['source_versions']={}
    scored=evaluate_bundles([('OLD','a.json',bundle)],_frame())
    summary=summarize(scored)[0]
    assert summary['wape']==pytest.approx(.1)
    assert summary['q90_upper_coverage'] is None


def test_rejects_duplicate_actual_and_forecast_keys():
    with pytest.raises(ValueError,match='duplicate actual'):
        evaluate_bundles([('OLD','a.json',_bundle())],pd.concat([_frame(),_frame().iloc[:1]]))
    bundle=_bundle()
    bundle['rows'].append(bundle['rows'][0].copy())
    with pytest.raises(ValueError,match='duplicate forecast'):
        evaluate_bundles([('OLD','a.json',bundle)],_frame())


def test_rejects_targets_before_issue_and_naive_timestamps():
    bundle=_bundle()
    bundle['issued_at']='2026-09-30T08:15:00+03:00'
    with pytest.raises(ValueError,match='before issue'):
        evaluate_bundles([('OLD','a.json',bundle)],_frame())
    bundle=_bundle()
    bundle['as_of']=bundle['issued_at']='2026-09-30T08:00:00'
    with pytest.raises(ValueError,match='timezone'):
        evaluate_bundles([('OLD','a.json',bundle)],_frame())


def test_separates_native_and_shifted_origin_and_weekday_peak():
    native=_bundle()
    shifted=_bundle()
    shifted['source_versions']['shifted_quarter_inference']=True
    scored=evaluate_bundles([('BALANCED','a.json',native),('BALANCED','b.json',shifted)],_frame())
    by_mode=summarize(scored,['arm','origin_mode'])
    assert {r['origin_mode'] for r in by_mode}=={'native30m','shifted15m'}


def test_review_reads_saved_files_and_writes_complete_report(tmp_path):
    controller=tmp_path/'balanced/controller'
    controller.mkdir(parents=True)
    (controller/'raw_forecast_03600.json').write_text(json.dumps(_bundle()),encoding='utf8')
    frame_path=tmp_path/'frame.parquet'
    _frame().to_parquet(frame_path)
    report=write_review(tmp_path,frame_path)
    assert report['forecast_files']==1
    assert (tmp_path/'forecast_review/report.json').exists()
    assert (tmp_path/'forecast_review/predictions.csv').exists()
    assert (tmp_path/'forecast_review/by_station.csv').exists()
    assert (tmp_path/'forecast_review/report.md').exists()
    assert report['evaluation_only']
    assert not report['independent_samples']
