"""Reproducible descriptive audit; detector candidates are not incident labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def causal_reference(values, eligible, window=4):
    history, median, scale, counts = [], [], [], []
    for y, use in zip(values, eligible):
        h = np.array(history[-window:])
        counts.append(len(h))
        b = np.median(h) if len(h) == window else np.nan
        median.append(b)
        scale.append(max(1.4826*np.median(abs(h-b)), np.sqrt(b), .05*b, 1) if np.isfinite(b) else np.nan)
        if use and np.isfinite(y):
            history.append(float(y))
    return np.array(median), np.array(scale), np.array(counts)


def corrected_hourly(frame):
    frame = frame.copy()
    shift = pd.to_timedelta((frame.interval_start.dt.hour < 3).astype(int), unit='D')
    frame['interval_start'] += shift
    frame['interval_end'] += shift
    return frame


def adjacent_windows(ids):
    ids = sorted(ids)
    return [tuple(ids[i:i+3]) for i in range(len(ids)-2)
            if int(ids[i+2][-2:])-int(ids[i][-2:]) == 2]


def add_reference(d, keys, eligible):
    d = d.sort_values('interval_start').reset_index(drop=True)
    d['reference_eligible'] = eligible(d)
    d['baseline'], d['scale'], d['n_reference'] = np.nan, np.nan, 0
    for _, part in d.groupby(keys, observed=True):
        b, s, n = causal_reference(part.entry_count, part.reference_eligible)
        d.loc[part.index, ['baseline', 'scale', 'n_reference']] = np.column_stack([b, s, n])
    d['ratio'] = d.entry_count/d.baseline.replace(0, np.nan)
    d['robust_z'] = (d.entry_count-d.baseline)/d.scale
    d['delta'] = d.entry_count-d.baseline
    return d


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    paths = [ROOT/'data/processed/goal4_3_source_refresh/entries_vestibule_1h.parquet',
             ROOT/'data/processed/metro/entries_vestibule_15m.parquet',
             ROOT/'data/processed/goal4_3_source_refresh/entries_station_15m.parquet',
             ROOT/'data/processed/metro/vestibule_map.csv',
             ROOT/'data/processed/metro_ml_inputs/data/raw/calendar_ru_2026_isdayoff.txt']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hraw, q = pd.read_parquet(paths[0]), pd.read_parquet(paths[1])
    h = corrected_hourly(hraw)
    names = pd.read_csv(paths[3]).drop_duplicates('station_id').set_index('station_id').station_name
    calendar = paths[4].read_text().strip()
    assert len(calendar) == 365 and set(calendar) <= set('012')
    codes = {d.strftime('%Y-%m-%d'): int(c) for d, c in zip(pd.date_range('2026-01-01', periods=365), calendar)}
    def ordinary(d):
        return (d.date.map(codes).eq(0) == d.dow.lt(5)) & d.date.gt('2026-01-11') & ~d.date.isin(['2026-05-09','2026-05-27','2026-05-30','2026-06-27','2026-06-28','2026-08-31'])
    daylight = h[h.interval_start.dt.hour.between(6, 23)]
    a = daylight.groupby(['station_id','interval_start'], as_index=False).entry_count.sum(min_count=1)
    a['date'] = a.interval_start.dt.strftime('%Y-%m-%d')
    a['dow'], a['hour'] = a.interval_start.dt.dayofweek, a.interval_start.dt.hour
    a = add_reference(a, ['station_id','dow','hour'], ordinary)
    a['station_name'] = a.station_id.map(names)
    a.to_parquet(output/'hourly_scores.parquet', index=False)
    candidates = a[(a.baseline >= 200) & (a.robust_z.abs() >= 4) & ((a.ratio < .5) | (a.ratio > 1.75))]
    candidates.to_csv(output/'hourly_candidates.csv', index=False, encoding='utf-8-sig')
    daily = a.groupby(['station_id','date','dow'], as_index=False).agg(entry_count=('entry_count','sum'), hours=('hour','size'))
    daily = daily[daily.hours.eq(18)].copy()
    daily['interval_start'] = pd.to_datetime(daily.date).dt.tz_localize('Europe/Moscow')
    daily = add_reference(daily, ['station_id','dow'], ordinary)
    daily['station_name'] = daily.station_id.map(names)
    daily.to_csv(output/'daily_scores.csv', index=False, encoding='utf-8-sig')
    network = []
    for ids in adjacent_windows(a.station_id.unique()):
        sub = a[a.station_id.isin(ids)].groupby('interval_start').agg(y=('entry_count','sum'), b=('baseline',lambda s:s.sum(min_count=3)), n=('station_id','nunique'))
        sub['ratio'] = sub.y/sub.b
        sub['delta'] = sub.y-sub.b
        for ts, row in sub[(sub.n==3)&(sub.b>=1000)&(sub.ratio<.75)].iterrows():
            future = sub.reindex([ts+pd.Timedelta(hours=k) for k in (1,2,3)])
            rebound = future.ratio.max()
            if pd.notna(rebound) and rebound > 1.05:
                network.append(dict(interval_start=ts, stations=','.join(ids), entry_count=row.y, baseline=row.b,
                    ratio=row.ratio, delta=row.delta, rebound_next_3h=rebound,
                    ordinary_day=bool(ordinary(pd.DataFrame({'date':[ts.strftime('%Y-%m-%d')],'dow':[ts.dayofweek]})).iloc[0])))
    net = pd.DataFrame(network).sort_values('delta')
    net.to_csv(output/'network_candidates.csv', index=False, encoding='utf-8-sig')
    # Independent-source check, restricted to complete daytime hours and shared vestibules.
    q['hour_start'] = q.interval_start.dt.floor('h')
    qs = q.groupby(['vestibule_id','hour_start'],as_index=False).agg(quarter_total=('entry_count',lambda s:s.sum(min_count=4)),n=('entry_count','count'))
    recon = qs[qs.n.eq(4)].merge(h[['vestibule_id','interval_start','entry_count']],left_on=['vestibule_id','hour_start'],right_on=['vestibule_id','interval_start'],validate='one_to_one')
    recon = recon[recon.hour_start.dt.hour.between(6,23)].copy()
    recon['delta'] = recon.quarter_total-recon.entry_count
    recon['mismatch'] = recon.delta.abs() > np.maximum(100,.2*recon.entry_count)
    recon.to_csv(output/'source_reconciliation.csv', index=False, encoding='utf-8-sig')
    # Fixed near-zero criterion, no retrospective closure flags used.
    q['date'] = q.interval_start.dt.strftime('%Y-%m-%d')
    q['slot'] = q.interval_start.dt.strftime('%H:%M')
    q['working'] = q.date.map(codes).eq(0)
    scheduled = []
    for vid, month, slots in [('V08',2,['08:15','08:30','08:45','09:00']),('V07',7,['08:15','08:30','08:45','09:00'])]:
        t = q[q.vestibule_id.eq(vid)&q.interval_start.dt.month.eq(month)&q.working&q.slot.isin(slots)]
        for slot,g in t.groupby('slot'):
            scheduled.append(dict(vestibule_id=vid,month=month,slot=slot,days=g.date.nunique(),near_zero_days=int((g.entry_count<=6).sum()),median=float(g.entry_count.median()),maximum=int(g.entry_count.max())))
    pd.DataFrame(scheduled).to_csv(output/'recurring_closures.csv',index=False,encoding='utf-8-sig')
    night = h[h.interval_start.dt.hour.isin([1,2,3,4])].groupby('interval_start').entry_count.sum()
    night[night>100].to_csv(output/'night_activity.csv',encoding='utf-8-sig')
    aug = a[a.date.eq('2026-08-31')]
    aug.to_csv(output/'incident_0831.csv',index=False,encoding='utf-8-sig')
    monthly = a.groupby([a.interval_start.dt.month,'station_id','date','dow']).entry_count.sum().reset_index(name='entries')
    monthly['workday'] = monthly.date.map(codes).eq(0)
    monthly[monthly.workday].groupby(['interval_start','station_id']).entries.median().rename('median_workday_entries_06_24').to_csv(output/'monthly_station.csv',encoding='utf-8-sig')
    summary = dict(hourly_rows=len(h),hourly_total=int(h.entry_count.sum()),hourly_stations=h.station_id.nunique(),
        quarter_rows=len(q),quarter_total=int(q.entry_count.sum()),quarter_days=q.service_date.nunique(),
        hourly_duplicates=int(h.duplicated(['vestibule_id','interval_start']).sum()),
        hourly_missing=int(h.entry_count.isna().sum()),quarter_missing=int(q.entry_count.isna().sum()),
        scored_station_hours=int(a.baseline.notna().sum()),hourly_candidates=len(candidates),
        network_candidates=len(net),reconciled_daytime_hours=len(recon),source_mismatches=int(recon.mismatch.sum()),
        source_signed_difference_percent=float(100*recon.delta.sum()/recon.entry_count.sum()),
        ordinary_network_top=net[net.ordinary_day].head(12).to_dict('records'),
        input_sha256=hashes,method='Four previous eligible same-weekday observations per station/hour; retrospective network rebound; descriptive thresholds not calibrated as incident classifier.')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    summary['inputs_unchanged'] = True
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['input_sha256','ordinary_network_top']},ensure_ascii=False,indent=2))
    print(net[net.ordinary_day].head(12).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
