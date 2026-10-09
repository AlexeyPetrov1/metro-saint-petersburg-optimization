"""Frozen native B1/B2/B3 stack, with explicit quarter refresh and causal inputs."""
from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import zipfile

import numpy as np
import pandas as pd

from forecasting import validate_bundle
from metro48_gru import FrozenQuarterGRU, build_window, decode_slots, split_slots
from metro_ml import (FrozenMetroForecaster, INPUTS, ROOT, VENDOR, _upstream,
                      file_hash, observed_history, safe_extract, station_mapping)
from simulation import digest

ENSEMBLE_INPUTS = ROOT / 'data/processed/metro51_inputs_20261009'


def _check_archive(archive, destination):
    root = Path(destination).resolve()
    seen = set()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename.replace('\\', '/')
            relative = PurePosixPath(name)
            target = (root / name).resolve()
            key = str(target).casefold()
            if (relative.is_absolute() or PureWindowsPath(name).drive or '..' in relative.parts or
                    ':' in name or not target.is_relative_to(root) or key in seen or
                    stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f'unsafe archive path: {member.filename}')
            seen.add(key)


def prepare_ensemble_inputs(models_archive, data_archive, destination=ENSEMBLE_INPUTS):
    destination = Path(destination)
    for archive in (models_archive, data_archive):
        _check_archive(archive, destination)
    for archive in (models_archive, data_archive):
        safe_extract(archive, destination)
    params_path = VENDOR / 'reports/stack/stack_params.json'
    output = destination / 'stack_params.json'
    if output.exists() and file_hash(output) != file_hash(params_path):
        raise ValueError('existing stack parameters differ from pinned vendor source')
    if not output.exists():
        output.write_bytes(params_path.read_bytes())
    manifest = dict(archives={name: dict(path=str(Path(path).resolve()), sha256=file_hash(path))
        for name, path in [('models', models_archive), ('data', data_archive)]},
        stack_params_source=str(params_path), stack_params_sha256=file_hash(output),
        files={str(p.relative_to(destination)).replace('\\','/'): file_hash(p)
            for p in sorted(destination.rglob('*')) if p.is_file() and p.name not in
            {'input_manifest.json', 'b2_quantiles_september.json', 'source_frame.parquet', 'contract.json'}})
    (destination / 'input_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    return manifest


def persistence_point(history, vestibules, as_of, slot_norms):
    now = pd.Timestamp(as_of)
    norms = np.asarray(slot_norms, float)
    if norms.shape != (len(vestibules),4) or not np.isfinite(norms).all() or (norms < 0).any():
        raise ValueError('invalid persistence normalization')
    h = history[(history.ts >= now-pd.Timedelta(minutes=30)) &
                (history.ts+pd.Timedelta(minutes=15) <= now)].copy()
    if h.duplicated(['ts','vestibule_id']).any():
        raise ValueError('duplicate persistence observations')
    good = np.isfinite(h.norm) & np.isfinite(h.entries) & (h.norm >= 0) & (h.entries >= 0)
    totals = h[good].groupby('vestibule_id')[['entries','norm']].sum().reindex(vestibules).fillna(0)
    ratio = np.divide(totals.entries.to_numpy(float), totals.norm.to_numpy(float),
                      out=np.ones(len(vestibules)), where=totals.norm.to_numpy()>0)
    ratio = np.clip(ratio, .3, 3.)
    return np.log((norms*ratio[:,None]+1)/(norms+1))


def slot_hour_parts(as_of, quarter_norms, vestibules):
    """Each native slot comprises two quarters, possibly in different hours."""
    now = pd.Timestamp(as_of)
    norms = np.asarray(quarter_norms, float)
    if norms.shape != (len(vestibules),8):
        raise ValueError('eight target quarter norms required')
    records = []
    t0 = now.floor('h')-pd.Timedelta(hours=1)
    for vi, vestibule in enumerate(vestibules):
        for j in range(8):
            target = now+pd.Timedelta(minutes=15*j)
            hour = target.floor('h')
            records.append(dict(vestibule_id=vestibule, vi=vi, k=j//2+1, quarter=j,
                hour_ts=hour.tz_convert('UTC'), quarter_in_hour=target.minute//15,
                h_hour=int((hour-t0)/pd.Timedelta(hours=1)), quarter_norm=float(norms[vi,j])))
    return pd.DataFrame(records)


def stack_predictions(rows, params):
    _upstream(ENSEMBLE_INPUTS if ENSEMBLE_INPUTS.exists() else INPUTS)
    meta = importlib.import_module('src.stack.meta')
    return meta.predict(rows, params)


class _NativeB1(FrozenMetroForecaster):
    def __init__(self, input_dir, model_dir):
        self.input_dir = Path(input_dir)
        self.history_source = 'exact_uploaded_hourly'
        self.allow_training_period = False
        self.event_registry = None
        self.knowledge_mode = 'strict'
        self.use_retrospective_history_flags = False
        self.E, self.BT, self.F, self.M = _upstream(self.input_dir)
        self.data = self.E.load()
        self.model, self.station_quantiles, self.meta = self.M.load_bundle(Path(model_dir))
        if self.meta['features'] != self.model.boosters[.5].feature_name():
            raise ValueError('LightGBM feature order differs from saved metadata')
        self.quarters = pd.read_parquet(self.input_dir/'data/interim/spb_line1_15min.parquet')
        self.mapping = station_mapping()
        self.inverse = {v:k for k,v in self.mapping.items()}
        self.model_ids = set(self.data.vestibules.station_id)
        self.hashes = {p.name:file_hash(p) for p in sorted(Path(model_dir).glob('*')) if p.is_file()}
        base = self.quarters.groupby(['station_id','ts_utc'],observed=True).entries.sum().reset_index()
        base['hour_start'] = base.ts_utc.dt.floor('h')
        self._source_hour_totals = base.groupby(['station_id','hour_start'],observed=True).entries.sum()
        self._cache = {}

    def _data_at(self, frame, cutoff):
        data, observed = super()._data_at(frame,cutoff)
        grid = data.grid
        end = max(grid.ts[-1],pd.Timestamp(cutoff).tz_convert('UTC')+pd.Timedelta(hours=3))
        times = pd.date_range(grid.ts[0],end,freq='h')
        values = np.full((len(times),len(data.vestibules)),np.nan)
        values[:len(grid.ts)] = grid.entries
        obs = observed.assign(hour_start=observed.interval_start.dt.floor('h').dt.tz_convert('UTC'))
        hours = obs.groupby(['station_id','hour_start'],observed=True).agg(total=('entry_count','sum'),n=('entry_count','size'))
        hours = hours[hours.n.eq(4)].reset_index()
        hours['station_id'] = hours.station_id.map(self.mapping)
        for station,ves in data.vestibules.groupby('station_id',sort=False):
            ix = pd.MultiIndex.from_arrays([np.repeat(station,len(times)),times])
            raw = self._source_hour_totals.reindex(ix).to_numpy(float)
            actual = hours.set_index(['station_id','hour_start']).total.reindex(ix).to_numpy(float)
            adjustment = np.divide(actual,raw,out=np.ones(len(times)),where=np.isfinite(raw)&(raw>0)&np.isfinite(actual))
            values[:,ves.pos.to_numpy()] *= adjustment[:,None]
        position = (times.tz_convert('Europe/Moscow').hour.to_numpy()-5)%24
        closed = np.ones_like(values,dtype=bool)
        for v in data.vestibules.itertuples(index=False):
            closed[:,v.pos] = (position < (v.first_hour-5)%24) | (position > (v.last_hour-5)%24)
        flagged = closed.copy()
        closed[:len(grid.ts)],flagged[:len(grid.ts)] = grid.closed,grid.flagged
        service_dates = (times.tz_convert('Europe/Moscow').tz_localize(None)-pd.Timedelta(hours=5)).normalize()
        regular = data.calendar.set_index('date').is_regular.reindex(service_dates).eq(True).to_numpy()
        extended = self.E.make_grid(times,values,closed,flagged,regular)
        return replace(data,grid=extended),observed

    def hourly_predictions(self, frame, now):
        cutoff = pd.Timestamp(now).floor('h')
        observed = observed_history(frame, cutoff)
        columns = ['station_id','interval_start','interval_end','entry_count']
        key = (cutoff.isoformat(), digest(pd.util.hash_pandas_object(observed[columns],index=False).tolist()))
        if key in self._cache:
            return self._cache[key]
        data, _ = self._data_at(frame, cutoff)
        panel = replace(self.BT.build_panel(data, final=True), thresholds=pd.DataFrame(self.meta['thresholds']))
        t0 = cutoff.tz_convert('UTC')-pd.Timedelta(hours=1)
        rows = self.BT.make_rows(panel, horizons=(1,2,3), for_export=True)
        rows = self.F.add_features(rows[rows.t.eq(t0)].copy(),panel)
        missing = rows.tau.map(data.forecast.set_index('ts_utc').precipitation).isna()
        rows.loc[missing,'fc_precip_tau'] = np.nan
        predicted = self.model.predict(rows)
        result = rows[['vestibule_id','tau','h','b4']].copy().rename(columns={'tau':'hour_ts','h':'h_hour'})
        for q in ('10','50','90'):
            result[f'm{q}'] = predicted[f'q{q}'].to_numpy(float)
        self._cache[key] = (result, rows)
        if len(self._cache)>8:
            del self._cache[next(iter(self._cache))]
        return result, rows


class FrozenQuarterEnsemble:
    def __init__(self, input_dir=ENSEMBLE_INPUTS, model_dir=None, params_path=None):
        self.input_dir = Path(input_dir)
        self.model_dir = Path(model_dir or self.input_dir/'models')
        self.params_path = Path(params_path or self.input_dir/'stack_params.json')
        self.E, self.BT, self.F, self.M = _upstream(self.input_dir)
        self.SB = importlib.import_module('src.stack.base')
        self.SD = importlib.import_module('src.stack.data')
        self.SM = importlib.import_module('src.stack.meta')
        self.SV = importlib.import_module('src.stack.evaluate')
        self.params = self.SM.load_params(self.params_path)
        if self.params['weights']['names'] != ['b1','b2','b3']:
            raise ValueError('saved meta stack is not the native B1/B2/B3 ensemble')
        self.b1 = _NativeB1(self.input_dir, self.model_dir/'lgbm_holdout')
        self.SB.check_oos(self.b1.meta, 9)
        self.gru = FrozenQuarterGRU(self.model_dir/'stack_gru_2026-09', self.input_dir)
        self.mapping, self.inverse = self.gru.mapping, self.gru.inverse
        self.vestibules, self.registry = self.gru.vestibules, self.gru.registry
        self._cache = {}
        self.rq = self._b2_calibration()
        self.hashes = dict(b1=self.b1.hashes, b3=self.gru.hashes,
                           stack_params=file_hash(self.params_path), b2=file_hash(self.input_dir/'b2_quantiles_september.json'))

    def _b2_calibration(self):
        output = self.input_dir/'b2_quantiles_september.json'
        source_paths = ['spb_line1_15min.parquet','spb_line1_15min_reconciliation.parquet','spb_line1_hourly.parquet']
        source_hashes = {name:file_hash(self.input_dir/'data/interim'/name) for name in source_paths}
        calibration_hashes = {name:file_hash(VENDOR/'src'/name) for name in ['stack/base.py','stack/data.py','intrahour.py','baseline.py']}
        if output.exists():
            payload = json.loads(output.read_text(encoding='utf8'))
            if payload['source_hashes'] != source_hashes or payload['code_hashes'] != calibration_hashes:
                raise ValueError('B2 calibration inputs changed; use a new extraction directory')
            return payload['quantiles']
        p = self.input_dir/'data/interim'
        h = self.gru.I.load_hours(p/'spb_line1_15min.parquet',p/'spb_line1_15min_reconciliation.parquet')
        # Native training rows only: all fitting observations end before September.
        h = h[h.month.isin([2,5,7])].copy()
        sd = self.SD.build(h)
        train = sd.rows[sd.rows.month.isin([2,5,7])]
        if train.empty or set(train.month.unique()) != {2,5,7}:
            raise ValueError('missing native February/May/July persistence calibration rows')
        rq = self.SB.fit_residual_quantiles(train, self.SB.b2_point(sd.q,train))
        payload = dict(months=[2,5,7], max_service_date=str(train.sday.max().date()),
                       n_rows=len(train), source_hashes=source_hashes, code_hashes=calibration_hashes,
                       method='native base.fit_residual_quantiles; no September labels', quantiles=rq)
        output.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf8')
        return rq

    def load_source_frame(self):
        return self.b1.load_simulator_frame()

    load_simulator_frame = load_source_frame

    def station_profile(self, station_id, target_hour):
        return self.gru.station_profile(station_id,target_hour)

    def _history(self, frame, now):
        observed = observed_history(frame, now)
        past = self.gru.quarters[(self.gru.quarters.ts >= now-pd.Timedelta(hours=4)) &
                                 (self.gru.quarters.ts+pd.Timedelta(minutes=15) <= now)].copy()
        past['station_id'] = past.station_slug.map(self.inverse)
        known = observed.set_index(['station_id','interval_start']).entry_count
        values = known.reindex(pd.MultiIndex.from_arrays([past.station_id,past.ts])).to_numpy(float)
        total = past.groupby(['station_id','ts']).entries.transform('sum').to_numpy(float)
        size = past.groupby(['station_id','ts']).entries.transform('size').to_numpy(float)
        share = np.divide(past.entries.to_numpy(float),total,out=1/size,where=total>0)
        past['entries'] = values*share
        return observed,past

    def _b1_slots(self, frame, now, quarter_norms):
        hourly, features = self.b1.hourly_predictions(frame,now)
        parts = slot_hour_parts(now,quarter_norms,self.vestibules)
        merged = parts.merge(hourly,on=['vestibule_id','hour_ts','h_hour'],how='left',validate='many_to_one')
        positive = merged.quarter_norm.gt(0)
        if merged.loc[positive,'m50'].isna().any():
            missing = merged.loc[positive & merged.m50.isna(),['vestibule_id','hour_ts','h_hour']]
            raise ValueError(f'native LightGBM lacks forecast for a positive target quarter norm: {missing.to_dict("records")}')
        counts = np.zeros((len(self.vestibules),8,3))
        for q, name in enumerate(['10','50','90']):
            # n_quarter = b4 * source_scale * p_quarter, including shifted cross-hour slots.
            scale = np.divide(merged.quarter_norm,merged.b4,
                              out=np.zeros(len(merged)),where=positive.to_numpy() & merged.b4.gt(0).to_numpy())
            counts[merged.vi,merged.quarter,q] = np.maximum(0,merged[f'm{name}'].fillna(0).to_numpy()*scale)
        slot_norms = quarter_norms.reshape(len(self.vestibules),4,2).sum(2)
        slot_counts = counts.reshape(len(self.vestibules),4,2,3).sum(2)
        return np.log((slot_counts+1)/(slot_norms[...,None]+1)),features

    def model_slots(self, frame, as_of):
        now = pd.Timestamp(as_of)
        if now.tzinfo is None or now.minute%15 or now.second or now.microsecond:
            raise ValueError('forecast needs a timezone-aware quarter boundary')
        service_date = (now.tz_convert('Europe/Moscow').tz_localize(None)-pd.Timedelta(hours=5)).normalize()
        latest_train = max(pd.Timestamp(self.b1.meta['train_end']),pd.Timestamp(self.gru.meta['info']['train_end']))
        if service_date <= latest_train+pd.Timedelta(days=7):
            raise ValueError('forecast overlaps model training or seven-day embargo')
        observed,past = self._history(frame,now)
        if observed.empty:
            raise ValueError('forecast has no completed observations')
        key = (now.isoformat(),digest(pd.util.hash_pandas_object(
            observed[['station_id','interval_start','interval_end','entry_count']],index=False).tolist()))
        if key in self._cache:
            return self._cache[key]
        x = build_window(past,self.vestibules,now)
        local = now.tz_convert('Europe/Moscow')
        day_type = self.gru.calendar.loc[service_date,'day_type']
        codes = np.array([[i,local.hour*4+local.minute//15,service_date.dayofweek,
                           self.gru.NN.DAY_TYPES.index(day_type)] for i in range(len(self.vestibules))],dtype=np.int64)
        import torch
        with torch.no_grad():
            xt,ct = torch.from_numpy(self.gru.scaler.transform(x)),torch.from_numpy(codes)
            b3 = np.mean([net(xt,ct).numpy().astype(float) for net in self.gru.nets],axis=0)
        qn = self.gru._target_norms(now)
        norms = qn.reshape(len(self.vestibules),4,2).sum(2)
        b1,features = self._b1_slots(frame,now,qn)
        b2_point = persistence_point(past,self.vestibules,now,norms)
        late_band = self.SD.BAND_OF[0]
        rows = pd.DataFrame([dict(vestibule_id=v,vi=i,k=k+1,n=norms[i,k],
                band4=self.SD.BAND_OF.get((now+pd.Timedelta(minutes=30*k)).tz_convert('Europe/Moscow').hour,late_band))
            for i,v in enumerate(self.vestibules) for k in range(4)])
        b2 = self.SB.apply_residual_quantiles(rows,b2_point.reshape(-1),self.rq).reshape(len(self.vestibules),4,3)
        for name,values in [('b1',b1),('b2',b2),('b3',b3)]:
            for qi,q in enumerate(['10','50','90']):
                rows[f'{name}_{q}'] = values[...,qi].reshape(-1)
        meta = self.SM.predict(rows,self.params)
        rows = rows.join(meta)
        z = rows[['stack_10','stack_50','stack_90']].to_numpy().reshape(len(self.vestibules),4,3)
        slots = decode_slots(z,norms)
        # Closed slots carry no passenger demand regardless of positive learned residual.
        slots[norms==0] = 0
        result = dict(rows=rows,slots=slots,quarter_norms=qn,window=x,codes=codes,b3_z=b3,
            features=features,observed=observed,native_update=now.minute%30==0)
        self._cache[key] = result
        if len(self._cache)>8:
            del self._cache[next(iter(self._cache))]
        return result

    def forecast_bundle(self, frame, station_ids, as_of, horizon_steps=8, horizon_hours=None,quantile='q50'):
        if horizon_hours is not None:
            horizon_steps = horizon_hours*4
        if (quantile not in ('q10','q50','q90') or horizon_steps not in (4,8) or not station_ids or len(set(station_ids))!=len(station_ids) or
                not set(station_ids)<=set(self.mapping)):
            raise ValueError('distinct red-line stations and horizon_steps 4 or 8 required')
        now = pd.Timestamp(as_of)
        prediction = self.model_slots(frame,now)
        qn = prediction['quarter_norms']
        covered = {self.inverse[self.registry.loc[v,'station_id']] for v in self.vestibules}
        fallback = sorted(set(station_ids)-covered)
        by_station = {sid: np.zeros((4,3)) for sid in station_ids}
        station_norm = {sid: np.zeros(8) for sid in station_ids}
        for i,v in enumerate(self.vestibules):
            sid = self.inverse[self.registry.loc[v,'station_id']]
            if sid in by_station:
                by_station[sid] += prediction['slots'][i]
                station_norm[sid] += qn[i]
        rows = []
        for sid in station_ids:
            if sid in fallback:
                fallback_bundle = self.gru.forecast_bundle(frame,[sid],now,horizon_hours=horizon_steps//4)
                for row in fallback_bundle['rows']:
                    rows.append(dict(row,q10=row['predicted_count'],q90=row['predicted_count'],
                                     model_component='explicit_uncovered_station_historical_median'))
                continue
            slot_rows = pd.DataFrame({'q50':by_station[sid][:,1],'k':range(1,5),
                'band4':[self.SD.BAND_OF.get((now+pd.Timedelta(minutes=30*k)).tz_convert('Europe/Moscow').hour,
                                           self.SD.BAND_OF[0]) for k in range(4)]})
            calibrated = self.SV.station_interval(slot_rows,self.params['station_quantiles'])
            counts = np.stack([split_slots(calibrated[f'q{q}'].to_numpy()[None,:],station_norm[sid][None,:])[0]
                               for q in ['10','50','90']],axis=-1)
            # Preserve ordered integer quantiles after independent apportionment.
            counts.sort(axis=-1)
            for j in range(horizon_steps):
                start = now+pd.Timedelta(minutes=15*j)
                rows.append(dict(station_id=sid,interval_start=start.isoformat(),
                    interval_end=(start+pd.Timedelta(minutes=15)).isoformat(),predicted_count=int(counts[j,1]),
                    q10=int(counts[j,0]),q90=int(counts[j,2]),coverage='complete',
                    model_component='native_B1_B2_B3_frozen_stack'))
        observations = prediction['observed']
        latest = pd.concat([observations.interval_end,*[observations[c] for c in
            ['available_at','received_at'] if c in observations]],axis=1).max().max()
        quantile_rows = [dict(row,q50=row['predicted_count']) for row in rows]
        canonical_rows = [dict(station_id=row['station_id'],interval_start=row['interval_start'],
            interval_end=row['interval_end'],predicted_count=row[quantile],coverage=row['coverage']) for row in quantile_rows]
        source_versions = dict(model_sha256=self.hashes,b1_train_end=self.b1.meta['train_end'],
            b3_train_end=self.gru.meta['info']['train_end'],meta_fit_months=self.params['fit_months'],
            b2_calibration_months=[2,5,7],model_family='native_frozen_B1_B2_B3_meta_stack',
            native_update_seconds=1800,shifted_quarter_inference=not prediction['native_update'],
            fallback_station_ids=fallback,future_truth_used=False,retrospective_history_flags=False,
            uncalibrated_interval_station_ids=fallback,
            latest_observation_available_at=latest.isoformat(),
            quarter_split='sum station slots; frozen station calibration; causal quarter norms; largest remainders',
            weather_availability='archive historical forecasts have no issue timestamps; availability before target assumed',
            b1_history='completed uploaded hourly counts; past observed quarter deviation relative to source adjusts hourly lags',
            prediction_quantile=quantile,quantile_rows=quantile_rows,
            component_logit_rows=prediction['rows'].to_dict('records'),
            frozen_meta_weights=self.params['weights'],
            preprocessing_input_digest=digest(dict(window=np.nan_to_num(prediction['window']).tolist(),
                codes=prediction['codes'].tolist(),norms=qn.tolist(),
                b1_features=pd.util.hash_pandas_object(prediction['features'],index=False).tolist())))
        bundle = dict(schema_version='1.0',model_version='native-b1-b2-b3-frozen-stack-quarter-refresh-v1',
            issued_at=now.isoformat(),as_of=now.isoformat(),available_at=now.isoformat(),
            features_available_at=latest.isoformat(),target='turnstile_entries',unit='passengers_per_15m',
            coverage='complete',feature_version='native26-hourly+16x4-GRU+twoquarter-persistence',
            source_versions=source_versions,included_event_ids=[],refresh_seconds=900,
            availability_assumption='Station observations enforced at interval end and available_at when supplied. '
                'Completed uploaded vestibule shares assumed available; historical weather publication times unknown.',
            rounding='calibrated station half-hour quantiles HALF_UP; largest remainders by causal quarter profile',
            rows=canonical_rows,evaluation_mode='native30m_meta_stack' if prediction['native_update'] else 'shifted15m_meta_stack_experimental',
            claimed_ml_WAPE=None)
        validate_bundle(bundle,station_ids,now.isoformat())
        return bundle

    def feature_evidence(self,frame,as_of,max_features=5):
        if not isinstance(max_features,int) or not 1<=max_features<=len(self.b1.meta['features']):
            raise ValueError('invalid number of contribution features')
        prediction = self.model_slots(frame,as_of)
        rows = prediction['features']
        contrib = self.M.contributions(self.b1.model,rows)
        raw = self.b1.model.raw(rows)[.5]
        error = float(np.max(np.abs(contrib.sum(axis=1).to_numpy()-raw)))
        records = []
        feature_names = self.b1.meta['features']
        for position,(_,row) in enumerate(rows.iterrows()):
            top = np.argsort(-np.abs(contrib.iloc[position][feature_names].to_numpy(float)),kind='stable')[:max_features]
            features = []
            for i in top:
                name = feature_names[i]
                value = row[name]
                value = None if pd.isna(value) else value.item() if isinstance(value,np.generic) else value
                features.append(dict(name=name,value=value,contribution_z=float(contrib.iloc[position][name])))
            records.append(dict(vestibule_id=row.vestibule_id,station_id=self.inverse[row.station_id],
                target_hour=row.tau.isoformat(),horizon_hours=int(row.h),bias_z=float(contrib.iloc[position].bias),
                raw_q50_z=float(raw[position]),features=features))
        return dict(method='LightGBM pred_contrib in raw q50 log-ratio space',rows=records,
                    max_sum_error=error,interpretation='model contribution, not causal effect',
                    component='B1 hourly; calibrated station stack is a subsequent transform')

    def contract(self):
        covered = {self.inverse[self.registry.loc[v,'station_id']] for v in self.vestibules}
        return dict(components=['b1_lightgbm','b2_persistence','b3_gru_seeds012','frozen_meta_stack'],
            input_dir=str(self.input_dir),model_dir=str(self.model_dir),weights_sha256=self.hashes,
            b1_train_end=self.b1.meta['train_end'],b3_train_end=self.gru.meta['info']['train_end'],
            meta_fit_months=self.params['fit_months'],b2_calibration_months=[2,5,7],
            native_update_seconds=1800,simulator_refresh_seconds=900,shifted_updates_are_experimental=True,
            feature_order=self.b1.meta['features'],vestibule_order=self.vestibules,
            uncovered_station_ids=sorted(set(self.mapping)-covered),station_quantiles='frozen meta params',
            b1_hourly_horizons=[1,2,3],hour3_extrapolation=True,
            temporal_holdout_requires='after max component train_end plus seven-day embargo',
            September_is_reused_development_period=True,future_truth_used=False,
            retrospective_history_flags=False,weather_publication_timestamps='unknown')
