"""Evaluate saved forecasts against future truth; never calls or fits a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metro_ml import file_hash


def _stamp(value):
    result=pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError('forecast timestamp needs timezone')
    return result


def evaluate_bundles(bundles,frame):
    if frame.duplicated(['station_id','interval_start']).any():
        raise ValueError('duplicate actual station-quarter intervals')
    truth=frame.set_index(['station_id','interval_start'])
    records=[]
    for arm,path,bundle in bundles:
        issued=_stamp(bundle['issued_at'])
        origin=_stamp(bundle.get('as_of',bundle['issued_at']))
        versions=bundle.get('source_versions',{})
        quantiles={}
        for row in versions.get('quantile_rows',[]):
            key=(row['station_id'],_stamp(row['interval_start']),_stamp(row['interval_end']))
            if key in quantiles:
                raise ValueError('duplicate forecast quantile rows')
            quantiles[key]=row
        uncalibrated=set(versions.get('uncalibrated_interval_station_ids',versions.get('fallback_station_ids',[])))
        seen=set()
        for row in bundle['rows']:
            start,end=_stamp(row['interval_start']),_stamp(row['interval_end'])
            if start<issued:
                raise ValueError('forecast target starts before issue')
            if end-start!=pd.Timedelta(minutes=15):
                raise ValueError('forecast target must span15minutes')
            key=(row['station_id'],start,end)
            if key in seen:
                raise ValueError('duplicate forecast station-quarter intervals')
            seen.add(key)
            q=quantiles.get(key,{})
            predicted=q.get('q50',row['predicted_count'])
            if isinstance(predicted,bool) or not isinstance(predicted,(int,np.integer)) or predicted<0:
                raise ValueError('invalid saved integer passenger prediction')
            actual_key=(row['station_id'],start)
            status='missing'
            actual=None
            if actual_key in truth.index:
                t=truth.loc[actual_key]
                status='complete' if bool(t.is_complete) and t.interval_end==end else 'incomplete'
                if status=='complete':
                    actual=int(t.entry_count)
                    if actual<0:
                        raise ValueError('invalid negative actual passenger count')
            local=start.tz_convert('Europe/Moscow')
            peak=local.dayofweek<5 and (8<=local.hour<10 or 17<=local.hour<19)
            lo,hi=q.get('q10'),q.get('q90')
            if lo is not None and hi is not None and not (0<=lo<=predicted<=hi):
                raise ValueError('unordered saved forecast quantiles')
            start_lead=int((start-origin)/pd.Timedelta(minutes=1))
            records.append(dict(arm=arm,forecast_file=str(path),origin=origin.isoformat(),
                model_version=bundle.get('model_version'),station_id=row['station_id'],
                interval_start=start.isoformat(),interval_end=end.isoformat(),
                target_peak=peak,origin_mode='shifted15m' if versions.get('shifted_quarter_inference') else 'native30m',
                lead_start_minutes=start_lead,lead_end_minutes=start_lead+15,native_slot=start_lead//30+1,
                point_quantile='q50' if q else versions.get('prediction_quantile','point_forecast'),
                decision_prediction_quantile=versions.get('prediction_quantile','point_forecast'),
                predicted=int(predicted),decision_predicted=int(row['predicted_count']),q10=lo,q90=hi,
                interval_calibrated=row['station_id'] not in uncalibrated and lo is not None and hi is not None,
                truth_status=status,truth=actual,
                absolute_error=None if actual is None else abs(int(predicted)-actual),
                signed_error=None if actual is None else int(predicted)-actual))
    return pd.DataFrame(records)


def _plain(value):
    return value.item() if isinstance(value,np.generic) else value


def summarize(scored,dimensions=('arm',)):
    if scored.empty:
        return []
    result=[]
    for keys,group in scored.groupby(list(dimensions),dropna=False,sort=True):
        if not isinstance(keys,tuple):
            keys=(keys,)
        complete=group[group.truth_status.eq('complete')]
        intervals=complete[complete.interval_calibrated]
        denominator=float(complete.truth.sum())
        n=len(complete)
        unique=len(group[['station_id','interval_start']].drop_duplicates())
        result.append(dict(**{key:_plain(value) for key,value in zip(dimensions,keys)},
            n_predictions=len(group),n_scored=n,n_origins=group.origin.nunique(),
            unique_station_targets=unique,overlapping_forecasts=len(group)>unique,
            missing_truth=int(group.truth_status.eq('missing').sum()),
            incomplete_truth=int(group.truth_status.eq('incomplete').sum()),actual_sum=int(denominator),
            absolute_error_sum=float(complete.absolute_error.sum()),
            wape=None if denominator==0 else float(complete.absolute_error.sum()/denominator),
            mae=None if n==0 else float(complete.absolute_error.mean()),
            bias_fraction=None if denominator==0 else float(complete.signed_error.sum()/denominator),
            mean_signed_error=None if n==0 else float(complete.signed_error.mean()),
            n_interval_scored=len(intervals),q10_q90_coverage=None if intervals.empty else
                float((intervals.truth.ge(intervals.q10)&intervals.truth.le(intervals.q90)).mean()),
            q90_upper_coverage=None if intervals.empty else float(intervals.truth.le(intervals.q90).mean()),
            q10_lower_coverage=None if intervals.empty else float(intervals.truth.ge(intervals.q10).mean())))
    return result


def write_review(run_root,frame_path,output=None):
    root=Path(run_root)
    output=Path(output or root/'forecast_review')
    output.mkdir(parents=True,exist_ok=True)
    paths=sorted(root.glob('*/controller/raw_forecast_*.json'))
    bundles=[(path.parent.parent.name,str(path.relative_to(root)),json.loads(path.read_text(encoding='utf8'))) for path in paths]
    frame=pd.read_parquet(frame_path)
    scored=evaluate_bundles(bundles,frame)
    scored.to_csv(output/'predictions.csv',index=False)
    breakdowns={
        'by_arm':summarize(scored),
        'by_station':summarize(scored,['arm','station_id']),
        'by_horizon':summarize(scored,['arm','lead_end_minutes']),
        'by_native_slot':summarize(scored,['arm','native_slot']),
        'by_origin_mode':summarize(scored,['arm','origin_mode']),
        'by_peak':summarize(scored,['arm','target_peak']),
        'by_station_horizon_peak':summarize(scored,['arm','station_id','lead_end_minutes','target_peak'])}
    for name,rows in breakdowns.items():
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    report=dict(evaluation_only=True,model_calls=0,model_training=False,independent_samples=False,
        forecast_files=len(paths),run_root=str(root.resolve()),source_frame=str(Path(frame_path).resolve()),
        source_hashes={str(Path(frame_path)):file_hash(frame_path),**{str(path.relative_to(root)):file_hash(path) for path in paths}},
        assumptions=[
            'Actual intervals after forecast issue are used only for retrospective scoring.',
            'Overlapping origins repeat the same station targets and are not independent samples.',
            'q10-q90 is a nominal80percent interval; q90 upper coverage is reported separately.',
            'Quarter intervals inherit split half-hour quantiles; quarter coverage is empirical and not guaranteed.',
            'Uncalibrated fallback station intervals are excluded from coverage metrics.',
            'Missing or incomplete truth is excluded without zero imputation.',
            'Sep30 is a reused development day; this report does not establish fresh holdout generalization.'],
        **breakdowns)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf8')
    lines=['# Проверка сохранённых прогнозов','',
        'Истина после выдачи прогноза используется только для оценки ошибки. Модель здесь не вызывается и не обучается.',
        'Перекрывающиеся прогнозы повторяют одни и те же целевые интервалы. Это не независимые наблюдения.',
        'Интервал q10–q90 имеет номинальное покрытие80%; доля факта≤q90 приведена отдельно.',
        'Неполная и отсутствующая истина исключены. Некалиброванные интервалы резервной медианы не доказывают покрытие.',
        '30сентября уже использовалось при разработке. Результат не подтверждает переносимость на новые дни.','',
        '| Вариант | Прогнозов | Оценено | WAPE | MAE, чел. | Смещение | Покрытие q10–q90 | Факт≤q90 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    pct=lambda value:'н/д' if value is None else f'{100*value:.2f}%'
    for row in report['by_arm']:
        mae='н/д' if row['mae'] is None else f'{row["mae"]:.2f}'
        lines.append(f'| {row["arm"]} | {row["n_predictions"]} | {row["n_scored"]} | {pct(row["wape"])} | {mae} | '
                     f'{pct(row["bias_fraction"])} | {pct(row["q10_q90_coverage"])} | {pct(row["q90_upper_coverage"])} |')
    lines.extend(['','Подробные таблицы: by_station.csv, by_horizon.csv, by_native_slot.csv, by_origin_mode.csv, by_peak.csv, by_station_horizon_peak.csv.'])
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--frame',type=Path,default=Path('data/processed/metro51_inputs_20261009/source_frame.parquet'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=write_review(args.root,args.frame,args.output)
    print(json.dumps(dict(forecast_files=report['forecast_files'],by_arm=report['by_arm']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
