"""Run supplied weights, not cached upstream predictions, against dated truth."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from forecasting import hourly_median
from metro_ml import (FrozenMetroForecaster, _profile_from_observed, file_hash,
                      observed_history, prepare_inputs, station_mapping)
from line1 import largest_remainders


def metrics(frame, actual='actual_count', prediction='predicted_count'):
    y,f = frame[actual].to_numpy(float),frame[prediction].to_numpy(float)
    valid = np.isfinite(y)&np.isfinite(f)
    y,f = y[valid],f[valid]
    if not len(y):
        return dict(n=0,actual_total=0,mae=None,rmse=None,wape=None)
    error = f-y
    return dict(n=len(y),actual_total=float(y.sum()),mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.square(error).mean())),wape=float(np.abs(error).sum()/y.sum()) if y.sum() else None)


def grouped_metrics(frame, keys):
    out = []
    for group,data in frame.groupby(keys,dropna=False,sort=True):
        group = group if isinstance(group,tuple) else (group,)
        out.append(dict(zip(keys,group))|metrics(data))
    return pd.DataFrame(out)


def add_profile_control(output):
    """Separate causal shape from the gain of frozen hourly boosters."""
    output = Path(output)
    files = ['median_profile_control_predictions.csv','profile_control_summary.json']
    if any((output/name).exists() for name in files):
        raise FileExistsError('profile control already exists; outputs are preserved')
    history = pd.read_parquet(output/'simulator_history.parquet')
    quarter = pd.read_csv(output/'quarter_predictions.csv')
    median = quarter[quarter['mode'].eq('median')].copy()
    median['target_hour'] = pd.to_datetime(median.target_start,utc=True).dt.floor('h')
    control_parts = []
    for origin,data in median.groupby('origin',sort=True):
        cutoff = pd.Timestamp(origin)
        observed = observed_history(history,cutoff)
        for (sid,target),part in data.groupby(['station_id','target_hour'],sort=True):
            part = part.sort_values('target_start').copy()
            total = int(part.predicted_count.sum())
            if sid!='L1_07':
                profile = _profile_from_observed(observed,sid,target.tz_convert('Europe/Moscow'))
                weights = np.maximum(0,np.rint(profile*1_000_000_000)).astype('int64').tolist()
                part['predicted_count'] = largest_remainders(total,weights)
            part['mode'] = 'MEDIAN_CAUSAL_PROFILE'
            part['source'] = 'median_same_past_profile' if sid!='L1_07' else 'median_fallback'
            control_parts.append(part.drop(columns='target_hour'))
    control = pd.concat(control_parts,ignore_index=True)
    control.to_csv(output/files[0],index=False,encoding='utf-8')
    combined = pd.concat([quarter,control],ignore_index=True)
    hourly = combined.copy()
    hourly['target_hour'] = pd.to_datetime(hourly.target_start,utc=True).dt.floor('h').astype(str)
    hourly = hourly.groupby(['date','origin','mode','station_id','target_hour','horizon_hours','temporal_holdout'],observed=True).agg(
        predicted_count=('predicted_count','sum'),actual_count=('actual_count','sum'),n=('actual_count','size')).reset_index()
    hourly = hourly[hourly.n.eq(4)]
    reports = {}
    for scope,ids in [('hybrid19',list(station_mapping())),('model18',[s for s in station_mapping() if s!='L1_07'])]:
        q = combined[combined.station_id.isin(ids)]
        h = hourly[hourly.station_id.isin(ids)]
        for name,frame in [('quarter',q),('hour',h)]:
            table = grouped_metrics(frame,['date','mode','temporal_holdout'])
            table.to_csv(output/f'metrics_profile_control_{scope}_{name}.csv',index=False,encoding='utf-8')
            reports[f'{scope}_{name}'] = table.to_dict(orient='records')
            grouped_metrics(frame,['date','mode','horizon_hours']).to_csv(
                output/f'metrics_profile_control_{scope}_{name}_horizon.csv',index=False,encoding='utf-8')
    ml_hourly = pd.read_csv(output/'hourly_quantiles.csv')
    raw = ml_hourly[ml_hourly.actual_original_hourly.notna()].rename(columns={'actual_original_hourly':'actual_count'})
    median_hourly = hourly[hourly['mode'].eq('median') & hourly.station_id.ne('L1_07')].copy()
    raw_truth = raw[['station_id','target_hour','actual_count']].drop_duplicates()
    raw_truth['target_hour'] = pd.to_datetime(raw_truth.target_hour,utc=True).dt.floor('h').astype(str)
    median_hourly = median_hourly.drop(columns='actual_count').merge(raw_truth,on=['station_id','target_hour'],how='inner',validate='many_to_one')
    original_domain = pd.concat([raw,median_hourly],ignore_index=True)
    grouped_metrics(original_domain,['date','mode','horizon_hours']).to_csv(
        output/'metrics_original_hourly_ml_vs_median.csv',index=False,encoding='utf-8')
    summary = dict(status='completed',control='MEDIAN_CAUSAL_PROFILE',
        explanation='same median hourly totals and same past-only quarter profile as ML; L1_07 keeps uniform median fallback',
        base_manifest_sha256=file_hash(output/'manifest.json'),metrics=reports)
    (output/files[1]).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    control_files = [p for p in output.glob('*') if 'profile_control' in p.name or p.name=='metrics_original_hourly_ml_vs_median.csv']
    (output/'profile_control_manifest.json').write_text(json.dumps({p.name:file_hash(p) for p in sorted(control_files)},indent=2),encoding='utf-8')
    return summary


def evaluate(dates, hours, output, holdout_full_day=True):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    prepare_inputs()
    station_ids = list(station_mapping())
    models = {mode:FrozenMetroForecaster(history_source=mode) for mode in
        ('simulator_observations','exact_uploaded_hourly')}
    history = models['simulator_observations'].load_simulator_frame()
    history.to_parquet(output/'simulator_history.parquet',index=False)
    truth = history[history.source_id.eq('uploaded-spb-line1-15min')].set_index(['station_id','interval_start']).entry_count
    raw_hourly = pd.read_parquet(models['simulator_observations'].input_dir/'data/interim/spb_line1_hourly.parquet')
    inverse = {v:k for k,v in station_mapping().items()}
    raw_hourly['station_id'] = raw_hourly.station_id.map(inverse)
    original_truth = raw_hourly.groupby(['station_id','ts_utc'],observed=True).entries.sum()
    quarter_rows, hourly_rows, coverage, feature_parts = [],[],[],[]
    for date in dates:
        selected = list(range(6,23)) if date=='2026-09-30' and holdout_full_day else hours
        for hour in selected:
            as_of = pd.Timestamp(f'{date}T{hour:02}:00:00+03:00')
            try:
                baseline = hourly_median(history,station_ids,as_of.isoformat())
            except ValueError as error:
                coverage.append(dict(date=date,origin=as_of.isoformat(),mode='median',status='failed',error=str(error)))
                continue
            bundles = {'median':baseline}
            for mode,model in models.items():
                try:
                    bundles[mode] = model.forecast_bundle(history,station_ids,as_of.isoformat())
                    result = model._compute(history,as_of)
                    features = result['features'].copy()
                    features['inference_mode'] = mode
                    features['issued_at'] = as_of
                    feature_parts.append(features[['issued_at','inference_mode','vestibule_id','station_id','t','tau',*model.meta['features']]])
                    for prediction in result['station'].itertuples(index=False):
                        sid = inverse[prediction.station_id]
                        try:
                            actual_hourly = float(original_truth.loc[(sid,prediction.tau)])
                        except KeyError:
                            actual_hourly = np.nan
                        quarter_keys = [(sid,prediction.tau.tz_convert('Europe/Moscow')+pd.Timedelta(minutes=15*q)) for q in range(4)]
                        actual_quarter_sum = sum(float(truth[k]) for k in quarter_keys) if all(k in truth.index for k in quarter_keys) else np.nan
                        hourly_rows.append(dict(date=date,origin=as_of.isoformat(),mode=mode,
                            station_id=sid,target_hour=prediction.tau.tz_convert('Europe/Moscow').isoformat(),horizon_hours=int(prediction.h),
                            predicted_count=float(prediction.q50),q10=float(prediction.q10),q90=float(prediction.q90),
                            actual_original_hourly=actual_hourly,actual_quarter_sum=actual_quarter_sum,
                            temporal_holdout=date>'2026-09-29',weather_missing=result['weather_missing']))
                    coverage.append(dict(date=date,origin=as_of.isoformat(),mode=mode,status='passed',rows=len(bundles[mode]['rows']),
                        temporal_holdout=bundles[mode]['source_versions']['temporal_holdout'],
                        weather_missing=bundles[mode]['source_versions']['weather_missing']))
                except ValueError as error:
                    coverage.append(dict(date=date,origin=as_of.isoformat(),mode=mode,status='failed',error=str(error)))
            for mode,bundle in bundles.items():
                for prediction in bundle['rows']:
                    start = pd.Timestamp(prediction['interval_start'])
                    key = (prediction['station_id'],start)
                    actual = float(truth[key]) if key in truth.index else np.nan
                    quarter_rows.append(dict(date=date,origin=as_of.isoformat(),mode=mode,
                        station_id=prediction['station_id'],target_start=start.isoformat(),
                        horizon_hours=int((start.floor('h')-as_of).total_seconds()//3600)+1,
                        predicted_count=prediction['predicted_count'],actual_count=actual,
                        source='median' if mode=='median' else ('median_fallback' if prediction['station_id']=='L1_07' else 'frozen_ml'),
                        temporal_holdout=date>'2026-09-29'))
            print(f'{date} {hour:02}:00: '+', '.join(bundles),flush=True)
    quarter = pd.DataFrame(quarter_rows)
    hourly = pd.DataFrame(hourly_rows)
    valid = quarter[quarter.actual_count.notna()].copy()
    valid.to_csv(output/'quarter_predictions.csv',index=False,encoding='utf-8')
    hourly.to_csv(output/'hourly_quantiles.csv',index=False,encoding='utf-8')
    pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False,encoding='utf-8')
    if feature_parts:
        pd.concat(feature_parts,ignore_index=True).to_parquet(output/'actual_model_features.parquet',index=False)
    reports = {}
    for scope,subset in [('hybrid19',valid),('model18',valid[valid.station_id.ne('L1_07')])]:
        for level,keys in [('date',['date','mode','temporal_holdout']),('date_horizon',['date','mode','horizon_hours']),
                           ('station',['date','mode','station_id']),('all',['mode','temporal_holdout'])]:
            table = grouped_metrics(subset,keys)
            table.to_csv(output/f'metrics_{scope}_{level}.csv',index=False,encoding='utf-8')
            if level=='date':
                reports[scope] = table.to_dict(orient='records')
    if not hourly.empty:
        for target in ['actual_original_hourly','actual_quarter_sum']:
            temp = hourly.rename(columns={target:'actual_count'})
            grouped_metrics(temp,['date','mode','horizon_hours']).to_csv(output/f'metrics_hourly_{target}.csv',index=False,encoding='utf-8')
    summary = dict(status='completed',train_end=models['simulator_observations'].meta['train_end'],
        artifact_hashes=models['simulator_observations'].hashes,
        upstream_commit=models['simulator_observations'].meta.get('config_sha256'),
        predictions_generated_from='supplied text boosters, no upstream prediction caches, no retraining',
        target='new turnstile entries per station; not platform occupancy or exits',
        dataset_line='Line1 Kirovsko-Vyborgskaya; blue Line2 is not represented',
        honest_limitations=['Dates through29September overlap fitting/calibration/early stopping.',
            '30September is one temporal holdout day, without the upstream7day embargo.',
            'Historical weather forecasts lack issuance timestamps; before30September availability is assumed.',
            '30September daytime weather is absent; both weather features remainNaN.',
            'Quarter totals differ from original hourly counts; both target domains are reported.',
            'TechnologyInstitute L1_07 has median fallback; model-only18 and hybrid19 evaluated separately.',
            'Missing historical quarter months use explicit uniform hourly warmup; excluded from quarter profile.',
            'All intrahour shares use completed past hours, never uploaded full-period profile.'],
        dates=dates,diagnostic_hours=hours,quarter_rows=len(valid),metrics=reports,
        coverage_failures=[r for r in coverage if r['status']=='failed'])
    from metro_ml import UPSTREAM_COMMIT
    summary['upstream_commit'] = UPSTREAM_COMMIT
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest = {p.name:file_hash(p) for p in sorted(output.glob('*')) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),quarter_rows=len(valid),metrics=reports,coverage_failures=summary['coverage_failures']),ensure_ascii=False),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dates',nargs='+',default=['2026-09-02','2026-09-16','2026-09-29','2026-09-30'])
    parser.add_argument('--hours',nargs='+',type=int,default=[6,8,12,17,20])
    parser.add_argument('--sample-holdout',action='store_true',help='use selected hours instead of06–22 on30Sep')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/metro_ml_evaluation'/datetime.now().strftime('run_%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    evaluate(args.dates,args.hours,args.output,not args.sample_holdout)
    add_profile_control(args.output)


if __name__=='__main__':
    main()
