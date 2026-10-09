"""Frozen upstream LightGBM, causal station arrivals, and simulator contract."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import sys
import zipfile

import numpy as np
import pandas as pd

from forecasting import hourly_median, validate_bundle
from line1 import largest_remainders

ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = '2f0fd0a3c11e5a9322bac1f65e68f5d891f816c9'
INPUTS = ROOT / 'data/processed/metro_ml_inputs'
VENDOR = ROOT / 'vendor/metro_petersburg'
MODEL_STATIONS = 18
LOCAL_VESTIBULE_IDS = {
    'prospekt_veteranov_1': 'V01', 'prospekt_veteranov_2': 'V02',
    'leninsky_prospekt_1': 'V03', 'leninsky_prospekt_2': 'V04',
    'avtovo_1': 'V05', 'kirovsky_zavod_1': 'V06', 'narvskaya_1': 'V07',
    'baltiyskaya_1': 'V08', 'tekhnologichesky_institut_1': 'V09',
    'pushkinskaya_1': 'V10', 'vladimirskaya_1': 'V11',
    'vosstaniya_1': 'V12', 'vosstaniya_2': 'V13', 'chernyshevskaya_1': 'V14',
    'ploshchad_lenina_1': 'V15', 'ploshchad_lenina_2': 'V16',
    'vyborgskaya_1': 'V17', 'lesnaya_1': 'V18', 'ploshchad_muzhestva_1': 'V19',
    'politekhnicheskaya_1': 'V20', 'akademicheskaya_1': 'V21',
    'grazhdansky_prospekt_1': 'V22', 'devyatkino_2': 'V23', 'devyatkino_1': 'V24',
}


def file_hash(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive, destination):
    """Validate all paths before writing; never replace different existing bytes."""
    root = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        destinations, seen = [], set()
        for item in z.infolist():
            name = item.filename.replace('\\', '/')
            relative = PurePosixPath(name)
            path = (root / name).resolve()
            unsafe = (relative.is_absolute() or PureWindowsPath(name).drive or
                '..' in relative.parts or ':' in name or not path.is_relative_to(root) or
                stat.S_ISLNK(item.external_attr >> 16))
            key = str(path).casefold()
            if unsafe or key in seen:
                raise ValueError(f'unsafe archive path: {item.filename}')
            seen.add(key)
            destinations.append((item, path))
        for item, path in destinations:
            if item.is_dir():
                continue
            if path.exists() and sha256(z.read(item)).hexdigest() != file_hash(path):
                raise ValueError(f'existing extracted file differs: {path}')
        for item, path in destinations:
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            elif not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as src, path.open('xb') as dst:
                    while block := src.read(1024*1024):
                        dst.write(block)


def prepare_inputs(raw_dir=None, input_dir=INPUTS):
    raw_dir = Path(raw_dir or ROOT / 'data/raw/metro')
    input_dir = Path(input_dir)
    for filename in ('data.zip', 'lgbm_final.zip'):
        safe_extract(raw_dir / filename, input_dir)
    manifest = dict(upstream_commit=UPSTREAM_COMMIT, archives={
        name: dict(path=str((raw_dir/name).resolve()), sha256=file_hash(raw_dir/name))
        for name in ('data.zip', 'lgbm_final.zip')}, models={
        p.name: file_hash(p) for p in sorted((input_dir/'lgbm_final').glob('*')) if p.is_file()})
    (input_dir/'input_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def station_mapping():
    south_to_north = ['prospekt_veteranov', 'leninsky_prospekt', 'avtovo', 'kirovsky_zavod',
        'narvskaya', 'baltiyskaya', 'tekhnologichesky_institut', 'pushkinskaya',
        'vladimirskaya', 'vosstaniya', 'chernyshevskaya', 'ploshchad_lenina',
        'vyborgskaya', 'lesnaya', 'ploshchad_muzhestva', 'politekhnicheskaya',
        'akademicheskaya', 'grazhdansky_prospekt', 'devyatkino']
    return {f'L1_{i:02}': name for i, name in enumerate(south_to_north, 1)}


def observed_history(frame, cutoff):
    from metro_quality import filter_available_quality
    needed = {'station_id', 'interval_start', 'interval_end', 'entry_count', 'is_complete', 'source_id'}
    if not needed.issubset(frame.columns):
        raise ValueError('history lacks complete station entry intervals')
    for key in ('interval_start', 'interval_end'):
        if not isinstance(frame[key].dtype, pd.DatetimeTZDtype):
            raise ValueError('history timestamps must include timezone')
    if frame.duplicated(['station_id', 'interval_start']).any():
        raise ValueError('duplicate historical intervals')
    data = frame[frame.interval_end.le(cutoff) & frame.is_complete.eq(True)].copy()
    for key in ('available_at', 'received_at'):
        if key in data:
            if not isinstance(data[key].dtype, pd.DatetimeTZDtype):
                raise ValueError(f'historical {key} needs timezone')
            data = data[data[key].notna() & data[key].le(cutoff)]
    duration = data.interval_end-data.interval_start
    if (not duration.eq(pd.Timedelta(minutes=15)).all() or
        not data.interval_start.dt.minute.mod(15).eq(0).all() or
        not data.interval_start.dt.second.eq(0).all() or
        not pd.api.types.is_integer_dtype(data.entry_count.dtype) or
        data.entry_count.isna().any() or data.entry_count.lt(0).any()):
        raise ValueError('invalid observed quarter-hour counts')
    return filter_available_quality(data, cutoff)


def causal_quarter_profile(frame, station_id, target_hour, cutoff):
    data = observed_history(frame, cutoff)
    return _profile_from_observed(data,station_id,target_hour)


def _profile_from_observed(data, station_id, target_hour):
    data = data[data.station_id.eq(station_id) & data.interval_start.dt.hour.eq(target_hour.hour)].copy()
    data = data[~data.source_id.eq('uploaded-hourly-uniform-warmup')]
    if data.empty:
        return np.full(4, .25)
    data['hour_start'] = data.interval_start.dt.floor('h')
    data['quarter'] = data.interval_start.dt.minute//15
    wide = data.pivot(index='hour_start', columns='quarter', values='entry_count').reindex(columns=range(4)).dropna()
    wide = wide[wide.sum(axis=1).gt(0)]
    same_day_type = (wide.index.dayofweek < 5) == (target_hour.dayofweek < 5)
    selected = wide[same_day_type]
    if selected.empty:
        selected = wide
    if selected.empty:
        return np.full(4, .25)
    shares = selected.tail(28).div(selected.tail(28).sum(axis=1), axis=0).median().to_numpy(float)
    return shares/shares.sum()


def _upstream(input_dir):
    """Load pinned package without editing its source or stored files."""
    if 'src' in sys.modules and not str(getattr(sys.modules['src'], '__file__', '')).startswith(str(VENDOR)):
        raise RuntimeError('upstream src module conflicts with another package')
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    config = importlib.import_module('src.config')
    data = Path(input_dir)/'data'
    config.DATA, config.RAW = data, data/'raw'
    config.INTERIM, config.REFERENCE = data/'interim', data/'reference'
    config.CALENDAR_OUT = config.INTERIM/'calendar_ru_2026.parquet'
    config.SPB_HOURLY = config.INTERIM/'spb_line1_hourly.parquet'
    for name, filename in [('SPB_STATIONS_CSV', 'spb_line1_stations.csv'),
        ('SPB_VESTIBULES_CSV', 'spb_line1_vestibules.csv'),
        ('SPB_VESTIBULES_15MIN_CSV', 'spb_line1_vestibules_15min.csv'),
        ('SPB_INCIDENTS_CSV', 'incidents.csv'), ('SPB_EVENTS_CSV', 'events_spb.csv'),
        ('LINE1_OPERATIONS_YAML', 'line1_operations.yaml')]:
        setattr(config, name, config.REFERENCE/filename)
    return tuple(importlib.import_module('src.'+name) for name in
        ('eda_spb', 'backtest', 'features', 'model'))


class FrozenMetroForecaster:
    """No fitting: 26 original features, saved calibration, hourly updates."""
    def __init__(self, input_dir=INPUTS, history_source='simulator_observations', allow_training_period=True,
                 event_registry=None, knowledge_mode='strict', use_retrospective_history_flags=True):
        if history_source not in ('simulator_observations', 'exact_uploaded_hourly'):
            raise ValueError('unknown history source')
        if knowledge_mode not in ('strict', 'publication_replay'):
            raise ValueError('knowledge_mode must be strict or publication_replay')
        if type(use_retrospective_history_flags) is not bool:
            raise ValueError('use_retrospective_history_flags must be an explicit boolean')
        self.event_registry = event_registry
        self.knowledge_mode = knowledge_mode
        self.use_retrospective_history_flags = use_retrospective_history_flags
        self.input_dir = Path(input_dir)
        self.history_source = history_source
        self.allow_training_period = allow_training_period
        self.E, self.BT, self.F, self.M = _upstream(self.input_dir)
        self.data = self.E.load()
        self.model, self.station_quantiles, self.meta = self.M.load_bundle(self.input_dir/'lgbm_final')
        if self.meta['features'] != list(self.model.boosters[.5].feature_name()):
            raise ValueError('weight feature order differs from metadata')
        self.quarters = pd.read_parquet(self.input_dir/'data/interim/spb_line1_15min.parquet')
        self.mapping = station_mapping()
        self.inverse = {v:k for k,v in self.mapping.items()}
        self.model_ids = set(self.data.vestibules.station_id)
        self.hashes = {p.name:file_hash(p) for p in sorted((self.input_dir/'lgbm_final').glob('*')) if p.is_file()}
        self._cache = {}

    def _restore_history_flags(self, ts, closed, flagged, cutoff):
        """Legacy archive labels have no historical publication provenance."""
        if not self.use_retrospective_history_flags:
            return
        original = self.data.grid
        past = ts[:len(original.ts)]+pd.Timedelta(hours=1) <= cutoff.tz_convert('UTC')
        closed[:len(original.ts)][past] = original.closed[past]
        flagged[:len(original.ts)][past] = original.flagged[past]

    def _apply_known_schedules(self, ts, closed, flagged, cutoff):
        """Apply opt-in evidence to future hours; retain retrospective flags."""
        if self.event_registry is None:
            return []
        ids = set()
        future = np.flatnonzero((ts >= cutoff.tz_convert('UTC')) &
                                (ts < cutoff.tz_convert('UTC')+pd.Timedelta(hours=2)))
        for vestibule in self.data.vestibules.itertuples(index=False):
            sid = self.inverse[vestibule.station_id]
            local_entrance = LOCAL_VESTIBULE_IDS.get(vestibule.vestibule_id)
            if local_entrance is None:
                raise ValueError(f'unknown local vestibule mapping: {vestibule.vestibule_id}')
            for i in future:
                state = self.event_registry.state_during(sid, ts[i], ts[i]+pd.Timedelta(hours=1),
                    cutoff, vestibule_id=local_entrance, knowledge_mode=self.knowledge_mode)
                ids.update(state['event_ids'])
                if 'planned_night_service' in state['event_kinds'] and not state['unknown_timing']:
                    closed[i, vestibule.pos] = False
                    flagged[i, vestibule.pos] = False
                if state['closed_seconds'] == state['interval_seconds']:
                    closed[i, vestibule.pos] = True
                if state['demand_censored'] or state['movement_state'] != 'normal':
                    flagged[i, vestibule.pos] = True
        return sorted(ids)

    def load_simulator_frame(self):
        q = self.quarters.copy()
        q['station_id'] = q.station_id.map(self.inverse)
        frame = q.groupby(['station_id', 'ts_utc'], observed=True).agg(
            entry_count=('entries', 'sum'), n=('entries', 'size'), missing=('entries', lambda x:x.isna().sum())).reset_index()
        frame['interval_start'] = frame.ts_utc.dt.tz_convert('Europe/Moscow')
        frame['interval_end'] = frame.interval_start+pd.Timedelta(minutes=15)
        frame['is_complete'] = frame.missing.eq(0)
        frame['entry_count'] = frame.entry_count.astype('int64')
        frame['source_id'] = 'uploaded-spb-line1-15min'
        keys = ['station_id', 'interval_start', 'interval_end', 'entry_count', 'is_complete', 'source_id']
        frame = frame[keys]
        hours = pd.read_parquet(self.input_dir/'data/interim/spb_line1_hourly.parquet')
        hours = hours.groupby(['station_id','ts_utc'], observed=True).entries.sum().reset_index()
        hours['station_id'] = hours.station_id.map(self.inverse)
        existing = set(zip(frame.station_id,frame.interval_start))
        warmup = []
        for row in hours.itertuples(index=False):
            hour = row.ts_utc.tz_convert('Europe/Moscow')
            if (row.station_id,hour) in existing:
                continue
            for q,count in enumerate(largest_remainders(int(row.entries),[1]*4)):
                start = hour+pd.Timedelta(minutes=15*q)
                warmup.append(dict(station_id=row.station_id,interval_start=start,
                    interval_end=start+pd.Timedelta(minutes=15),entry_count=count,is_complete=True,
                    source_id='uploaded-hourly-uniform-warmup'))
        frame = pd.concat([frame,pd.DataFrame(warmup)],ignore_index=True)
        return frame.sort_values(['station_id','interval_start'],ignore_index=True)

    def _data_at(self, frame, cutoff):
        observed = observed_history(frame, cutoff)
        observed['hour_start'] = observed.interval_start.dt.floor('h').dt.tz_convert('UTC')
        hours = observed.groupby(['station_id', 'hour_start'], observed=True).agg(
            total=('entry_count', 'sum'), n=('entry_count', 'size')).reset_index()
        hours = hours[hours.n.eq(4)]
        hours['station_id'] = hours.station_id.map(self.mapping)
        known = hours.set_index(['hour_start', 'station_id']).total
        original = self.data
        end = max(original.grid.ts[-1], cutoff.tz_convert('UTC')+pd.Timedelta(hours=1))
        ts = pd.date_range(original.grid.ts[0], end, freq='h')
        V = len(original.vestibules)
        entries = np.full((len(ts), V), np.nan)
        closed = np.ones((len(ts), V), dtype=bool)
        flagged = np.ones((len(ts), V), dtype=bool)
        # Future vestibule schedules are public; retrospective closures/incidents are hidden.
        local_hour = ts.tz_convert('Europe/Moscow').hour.to_numpy()
        service_pos = (local_hour-5)%24
        for vestibule in original.vestibules.itertuples(index=False):
            closed[:,vestibule.pos] = ((service_pos < (vestibule.first_hour-5)%24) |
                (service_pos > (vestibule.last_hour-5)%24))
        flagged[:] = closed
        old_n = len(original.grid.ts)
        self._restore_history_flags(ts, closed, flagged, cutoff)
        self._known_event_ids = self._apply_known_schedules(ts, closed, flagged, cutoff)
        # Only completed observed intervals can supply lagged entries.
        q = self.quarters[self.quarters.ts_utc+pd.Timedelta(minutes=15) <= cutoff.tz_convert('UTC')].copy()
        q['hour_start'] = q.ts_utc.dt.floor('h')
        qh = q.groupby(['hour_start','vestibule_id'], observed=True).agg(
            count=('entries','sum'), n=('entries','size')).reset_index()
        qh = qh[qh.n.eq(4)].set_index(['hour_start','vestibule_id'])['count']
        for station, ves in original.vestibules.groupby('station_id', sort=False):
            columns = ves.pos.to_numpy()
            observed_counts = known.reindex(pd.MultiIndex.from_arrays([ts, np.repeat(station, len(ts))])).to_numpy(float)
            for j, row in enumerate(ves.itertuples(index=False)):
                vals = qh.reindex(pd.MultiIndex.from_arrays([ts,np.repeat(row.vestibule_id,len(ts))])).to_numpy(float)
                raw = np.full(len(ts), np.nan)
                raw[:old_n] = original.grid.entries[:, row.pos]
                if self.history_source == 'exact_uploaded_hourly':
                    vals = np.where(np.isfinite(raw), raw, vals)
                else:
                    vals = np.where(np.isfinite(vals), vals, raw)
                entries[:,row.pos] = vals
            base = entries[:,columns].copy()
            if self.history_source == 'simulator_observations':
                denom = np.nansum(base,axis=1)
                proportions = np.divide(base,denom[:,None],out=np.full_like(base,1/len(columns)),where=denom[:,None]>0)
                entries[:,columns] = proportions*observed_counts[:,None]
            else:
                entries[np.ix_(np.isnan(observed_counts),columns)] = np.nan
        entries[ts+pd.Timedelta(hours=1)>cutoff.tz_convert('UTC')] = np.nan
        local = ts.tz_convert('Europe/Moscow').tz_localize(None)
        sday = (local-pd.Timedelta(hours=5)).normalize()
        regular = original.calendar.set_index('date').is_regular.reindex(sday).eq(True).to_numpy()
        grid = self.E.make_grid(ts,entries,closed,flagged,regular)
        return replace(original,grid=grid,df=original.df[original.df.ts_utc+pd.Timedelta(hours=1)<=cutoff.tz_convert('UTC')]), observed

    def _compute(self, frame, as_of):
        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is None:
            raise ValueError('decision time needs timezone')
        cutoff = stamp.tz_convert('Europe/Moscow').floor('h')
        observed = observed_history(frame,cutoff)
        obs_digest = sha256(pd.util.hash_pandas_object(observed.sort_values(['station_id','interval_start'])[
            ['station_id','interval_start','interval_end','entry_count','source_id']],index=False).to_numpy().tobytes()).hexdigest()
        availability_columns = [c for c in ('available_at', 'received_at', 'quality_available_at') if c in observed]
        availability_digest = None if not availability_columns else sha256(pd.util.hash_pandas_object(
            observed.sort_values(['station_id','interval_start'])[
                ['station_id','interval_start', *availability_columns]], index=False).to_numpy().tobytes()).hexdigest()
        event_digest = None if self.event_registry is None else sha256(json.dumps(
            self.event_registry.visible(cutoff, self.knowledge_mode), sort_keys=True, default=str).encode()).hexdigest()
        key = (cutoff.isoformat(),obs_digest,event_digest,availability_digest,self.use_retrospective_history_flags)
        if key in self._cache:
            return self._cache[key]
        data,_ = self._data_at(frame,cutoff)
        panel = self.BT.build_panel(data,final=True)
        panel = replace(panel,thresholds=pd.DataFrame(self.meta['thresholds']))
        t0 = cutoff.tz_convert('UTC')-pd.Timedelta(hours=1)
        rows = self.BT.make_rows(panel,for_export=True)
        rows = self.F.add_features(rows[rows.t.eq(t0)],panel)
        # Upstream turns NaN >= .5 into False; an absent forecast must stay unknown.
        forecast_precip = data.forecast.set_index('ts_utc').precipitation
        missing_weather = rows.tau.map(forecast_precip).isna()
        rows.loc[missing_weather,'fc_precip_tau'] = np.nan
        prediction = self.model.predict(rows)
        station = self.BT.station_frame(rows,prediction.q50,panel)
        station = station.join(self.station_quantiles.predict(station))
        latest_available = pd.concat([observed.interval_end, *[observed[c]
            for c in ('available_at', 'received_at') if c in observed]], axis=1).max().max()
        value = dict(cutoff=cutoff,features=rows,station=station,observation_digest=obs_digest,
            latest_observation=observed.interval_end.max(),weather_missing=bool(missing_weather.any()),
            latest_observation_available_at=latest_available, observation_availability_digest=availability_digest,
            known_event_ids=self._known_event_ids, event_registry_digest=event_digest)
        self._cache[key] = value
        if len(self._cache)>12:
            del self._cache[next(iter(self._cache))]
        return value

    def feature_rows(self, frame, as_of):
        return self._compute(frame,as_of)['features'].copy()

    def forecast_bundle(self, frame, station_ids, as_of, horizon_hours=2):
        if horizon_hours not in (1,2) or not station_ids or len(set(station_ids))!=len(station_ids):
            raise ValueError('model supports distinct stations and one/two hours')
        if not set(station_ids)<=set(self.mapping):
            raise ValueError('station coverage belongs to Line1, not Line2')
        result = self._compute(frame,as_of)
        cutoff = result['cutoff']
        temporal_holdout = (cutoff-pd.Timedelta(hours=5)).date()>pd.Timestamp(self.meta['train_end']).date()
        if not temporal_holdout and not self.allow_training_period:
            raise ValueError('weights saw this training period; enable diagnostic mode explicitly')
        fallback = [sid for sid in station_ids if self.mapping[sid] not in self.model_ids]
        fallback_bundle = hourly_median(frame,fallback,cutoff.isoformat(),horizon_hours) if fallback else None
        forecasts = result['station'].set_index(['station_id','tau'])
        rows = []
        profile_frame = observed_history(frame,cutoff)
        for h in range(horizon_hours):
            hour = cutoff+pd.Timedelta(hours=h)
            for sid in station_ids:
                if sid in fallback:
                    rows.extend([r.copy() for r in fallback_bundle['rows'] if r['interval_start'].startswith(hour.isoformat()[:13])])
                    continue
                index = (self.mapping[sid],hour.tz_convert('UTC'))
                if index not in forecasts.index:
                    raise ValueError(f'missing frozen forecast coverage for station {sid}/{hour}')
                forecast = forecasts.loc[index]
                if not np.isfinite(forecast.q50) or forecast.n!=forecast.n_open:
                    raise ValueError(f'incomplete frozen station coverage: {sid}/{hour}')
                total = int(np.floor(float(forecast.q50)+.5))
                profile = _profile_from_observed(profile_frame,sid,hour)
                weights = np.maximum(0,np.rint(profile*1_000_000_000)).astype('int64').tolist()
                for q,count in enumerate(largest_remainders(total,weights)):
                    start = hour+pd.Timedelta(minutes=15*q)
                    rows.append(dict(station_id=sid,interval_start=start.isoformat(),
                        interval_end=(start+pd.Timedelta(minutes=15)).isoformat(),predicted_count=int(count),coverage='complete'))
        bundle = dict(schema_version='1.0',model_version='frozen-lgbm-q-ratio-v1/'+UPSTREAM_COMMIT[:12],
            issued_at=cutoff.isoformat(),as_of=cutoff.isoformat(),available_at=cutoff.isoformat(),
            features_available_at=cutoff.isoformat(),target='turnstile_entries',
            unit='passengers_per_15m',coverage='complete',feature_version='upstream26-causal-hourly-plus-past-quarter-profile-v1',
            source_versions=dict(upstream_commit=UPSTREAM_COMMIT,model_sha256=self.hashes,
                train_start=self.meta['train_start'],train_end=self.meta['train_end'],
                temporal_holdout=temporal_holdout,upstream_embargo_holdout=(cutoff.tz_localize(None)>
                    pd.Timestamp(self.meta['train_end'])+pd.Timedelta(days=7)),
                history_source=self.history_source,observation_digest=result['observation_digest'],
                history_flag_policy='legacy_retrospective_flags_without_publication_provenance'
                    if self.use_retrospective_history_flags else 'public_schedule_and_available_quality_only',
                latest_observation_available_at=result['latest_observation_available_at'].isoformat(),
                observation_availability_digest=result['observation_availability_digest'],
                fallback_station_ids=fallback,weather_missing=result['weather_missing'],
                weather_availability='historical forecast supplied without issuance times; availability assumed before target',
                station_line='1 Kirovsko-Vyborgskaya; uploaded weights and data do not cover blue Line2'),
            availability_assumption='complete observed intervals assumed available at their end; provided available_at enforced',
            included_event_ids=result['known_event_ids'],refresh_seconds=3600,rounding='round hourly station q50; distribute by past-only quarter medians and largest remainders',
            rows=rows,evaluation_mode='temporal_holdout_no_embargo' if temporal_holdout else 'training_period_diagnostic',claimed_ml_WAPE=None)
        if self.event_registry is not None:
            bundle['source_versions'].update(event_knowledge_mode=self.knowledge_mode,
                event_registry_digest=result['event_registry_digest'],
                schedule_application='known future entrance hours only; fixed model night-hour coverage remains limited')
        validate_bundle(bundle,station_ids,as_of)
        return bundle
