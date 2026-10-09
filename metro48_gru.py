"""Frozen upstream B3: causal four-hour input, four native half-hour outputs."""
from __future__ import annotations

from hashlib import sha256
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting import validate_bundle
from line1 import largest_remainders
from metro_ml import INPUTS, VENDOR, _upstream, file_hash, observed_history, safe_extract, station_mapping
from simulation import digest

MODEL_INPUTS = Path(__file__).resolve().parent / 'data/processed/metro48_gru_inputs_20261009'


def extract_model(archive, destination=MODEL_INPUTS):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    safe_extract(archive, destination)
    return destination / 'stack_gru_2026-09'


def build_window(history, vestibules, as_of):
    now = pd.Timestamp(as_of)
    if now.tzinfo is None or now.minute % 15 or now.second or now.microsecond:
        raise ValueError('prediction requires a timezone-aware quarter boundary')
    past = history[history.ts + pd.Timedelta(minutes=15) <= now]
    if past.duplicated(['ts', 'vestibule_id']).any():
        raise ValueError('duplicate vestibule intervals')
    times = pd.date_range(now - pd.Timedelta(hours=4), periods=16, freq='15min')
    y = past.pivot(index='ts', columns='vestibule_id', values='entries').reindex(index=times, columns=vestibules).to_numpy(float)
    n = past.pivot(index='ts', columns='vestibule_id', values='norm').reindex(index=times, columns=vestibules).to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(n) & (y >= 0) & (n >= 0)
    z = np.full_like(y, np.nan)
    z[valid] = np.log((y[valid] + 1) / (n[valid] + 1))
    zl = np.where(valid.any(1), np.log((np.where(valid, y, 0).sum(1) + 1) /
                                     (np.where(valid, n, 0).sum(1) + 1)), np.nan)
    return np.stack([z.T, np.isnan(z.T), np.broadcast_to(zl, z.T.shape),
                     np.broadcast_to(np.isnan(zl), z.T.shape)], axis=-1).astype(np.float32)


def decode_slots(z, norms):
    z, norms = np.asarray(z, float), np.asarray(norms, float)
    if z.shape != (*norms.shape, 3) or not np.isfinite(z).all() or not np.isfinite(norms).all() or (norms < 0).any():
        raise ValueError('invalid native prediction or normalization')
    with np.errstate(over='ignore'):
        y = np.maximum(0, np.exp(z) * (norms[..., None] + 1) - 1)
    if not np.isfinite(y).all():
        raise ValueError('nonfinite decoded prediction')
    return y


def split_slots(counts, quarter_norms):
    counts, quarter_norms = np.asarray(counts, float), np.asarray(quarter_norms, float)
    if quarter_norms.shape != (len(counts), counts.shape[1] * 2):
        raise ValueError('two quarter norms required per native slot')
    result = np.zeros(quarter_norms.shape, dtype=np.int64)
    for i, slots in enumerate(counts):
        for k, value in enumerate(slots):
            shares = quarter_norms[i, 2*k:2*k+2]
            total = int(np.floor(value + .5))
            allocation = total * (shares / shares.sum() if shares.sum() > 0 else np.array([.5, .5]))
            whole = np.floor(allocation).astype(np.int64)
            order = np.argsort(-(allocation - whole), kind='stable')
            whole[order[:total-int(whole.sum())]] += 1
            result[i, 2*k:2*k+2] = whole
    return result


class FrozenQuarterGRU:
    def __init__(self, model_dir=MODEL_INPUTS / 'stack_gru_2026-09', input_dir=INPUTS,
                 use_retrospective_history_flags=False):
        self.input_dir, self.model_dir = Path(input_dir), Path(model_dir)
        self.retrospective_flags = use_retrospective_history_flags
        self.meta = json.loads((self.model_dir / 'meta.json').read_text(encoding='utf8'))
        self.E, _, _, _ = _upstream(self.input_dir)
        self.I = importlib.import_module('src.intrahour')
        self.B = importlib.import_module('src.baseline')
        self.NN = importlib.import_module('src.stack.nn')
        self.mapping = station_mapping()
        self.inverse = {v: k for k, v in self.mapping.items()}
        self._target_norm_cache = {}
        self._prepare()
        if len(self.vestibules) != self.meta['n_vestibules']:
            raise ValueError('vestibule vocabulary differs from weights')
        import torch
        torch.set_num_threads(self.meta['params']['threads'])
        torch.use_deterministic_algorithms(True)
        self.nets = []
        for seed in self.meta['params']['seeds']:
            net = self.NN._net(len(self.vestibules), self.meta['params'])
            net.load_state_dict(torch.load(self.model_dir / f'gru_seed{seed}.pt', map_location='cpu', weights_only=True))
            net.eval()
            self.nets.append(net)
        self.scaler = self.NN.Scaler(**self.meta['scaler'])
        self.hashes = {p.name: file_hash(p) for p in self.model_dir.iterdir() if p.is_file()}

    def _prepare(self):
        p = self.input_dir / 'data/interim'
        h = self.I.load_hours(p / 'spb_line1_15min.parquet', p / 'spb_line1_15min_reconciliation.parquet')
        h = h[h.in_hourly].copy().reset_index(drop=True)
        self.vestibules = sorted(h.vestibule_id.unique())
        self.registry = pd.read_csv(self.input_dir / 'data/reference/spb_line1_vestibules_15min.csv').set_index('vestibule_id')
        d = self.E.load()
        g = d.grid
        end = max(g.ts[-1], h.ts_utc.max() + pd.Timedelta(hours=3))
        times = pd.date_range(g.ts[0], end, freq='h')
        entries = np.full((len(times), len(d.vestibules)), np.nan)
        entries[:len(g.ts)] = g.entries
        closed = np.ones_like(entries, dtype=bool)
        for v in d.vestibules.itertuples():
            pos = (times.tz_convert('Europe/Moscow').hour - 5) % 24
            closed[:, v.pos] = (pos < (v.first_hour - 5) % 24) | (pos > (v.last_hour - 5) % 24)
        flagged = closed.copy()
        if self.retrospective_flags:
            flagged[:len(g.ts)] = g.flagged
        else:
            for flag in ['is_vestibule_closed', 'is_incident', 'is_source_mismatch', 'is_slot_closed']:
                h[flag] = False
            reg = self.registry.reindex(h.vestibule_id)
            pos = (h.hour.to_numpy() - 5) % 24
            h['ok'] = (pos >= (reg.first_hour.to_numpy() - 5) % 24) & (pos <= (reg.last_hour.to_numpy() - 5) % 24)
        local = times.tz_convert('Europe/Moscow').tz_localize(None)
        sday = (local - pd.Timedelta(hours=5)).normalize()
        regular = d.calendar.set_index('date').is_regular.reindex(sday).eq(True).to_numpy()
        grid = self.E.make_grid(times, entries, closed, flagged, regular)
        b = self.B.norm(grid, d.calendar)
        h = pd.concat([h, self.I.past_profile(h), self.I.source_scale(h)], axis=1)
        hi = times.get_indexer(h.ts_utc)
        vi = pd.Index(d.vestibules.vestibule_id).get_indexer(h.vestibule_id)
        h['b4'] = b[hi, vi]
        self.hours = h
        self.calendar = d.calendar.set_index('date')
        self._norm_grid, self._norm_times, self._norm_ves = b, times, list(d.vestibules.vestibule_id)
        parts = []
        good = h.ok & (h.b4 >= 20) & h[self.I.PCOLS].notna().all(1) & h.s_v.notna()
        for j in range(4):
            parts.append(pd.DataFrame(dict(ts=h.ts_utc + pd.Timedelta(minutes=15*j),
                vestibule_id=h.vestibule_id, station_slug=h.station_id,
                entries=h[f'y{j}'], norm=np.where(good, h.b4 * h.s_v * h[f'p{j}'], np.nan))))
        self.quarters = pd.concat(parts, ignore_index=True)
        self._hour_lookup = h.set_index(['ts_utc', 'vestibule_id'])

    def _target_norms(self, now):
        cache_key = pd.Timestamp(now).tz_convert('UTC')
        if cache_key in self._target_norm_cache:
            return self._target_norm_cache[cache_key].copy()
        starts = pd.date_range(now, periods=8, freq='15min').tz_convert('UTC')
        out = np.zeros((len(self.vestibules), 8))
        for j, start in enumerate(starts):
            hour = start.floor('h')
            quarter = start.tz_convert('Europe/Moscow').minute // 15
            sday = (start.tz_convert('Europe/Moscow').tz_localize(None) - pd.Timedelta(hours=5)).normalize()
            for i, v in enumerate(self.vestibules):
                reg = self.registry.loc[v]
                position = (start.tz_convert('Europe/Moscow').hour - 5) % 24
                if not (reg.first_hour - 5) % 24 <= position <= (reg.last_hour - 5) % 24:
                    continue
                key = (hour, v)
                if key in self._hour_lookup.index:
                    row = self._hour_lookup.loc[key]
                    value = row.b4 * row.s_v * row[f'p{quarter}']
                else:
                    # Month-end extension: causal same-weekday baseline and latest prior-day profile.
                    ti, vi = self._norm_times.get_indexer([hour])[0], self._norm_ves.index(v)
                    past = self.hours[(self.hours.vestibule_id == v) & (self.hours.sday < sday) &
                                      (self.hours.hour == start.tz_convert('Europe/Moscow').hour)]
                    if ti < 0 or past.empty:
                        raise ValueError(f'no causal target normalization for {v}/{start}')
                    row = past.iloc[-1]
                    value = self._norm_grid[ti, vi] * row.s_v * row[f'p{quarter}']
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f'no causal target normalization for {v}/{start}')
                out[i, j] = value
        self._target_norm_cache[cache_key] = out.copy()
        return out

    def model_slots(self, frame, as_of):
        now = pd.Timestamp(as_of)
        if now.tzinfo is None or now.minute % 15 or now.second or now.microsecond:
            raise ValueError('prediction requires a timezone-aware quarter boundary')
        recent = frame[frame.interval_end.le(now) & frame.interval_start.ge(now - pd.Timedelta(hours=4))]
        observed = observed_history(recent, now)
        if now.tz_convert('Europe/Moscow').tz_localize(None).normalize() <= pd.Timestamp(self.meta['info']['train_end']):
            raise ValueError('prediction date overlaps weight training')
        past = self.quarters[(self.quarters.ts >= now - pd.Timedelta(hours=4)) &
                             (self.quarters.ts + pd.Timedelta(minutes=15) <= now)].copy()
        past['station_id'] = past.station_slug.map(self.inverse)
        known = observed.set_index(['station_id', 'interval_start']).entry_count
        counts = known.reindex(pd.MultiIndex.from_arrays([past.station_id, past.ts])).to_numpy(float)
        total = past.groupby(['station_id', 'ts']).entries.transform('sum').to_numpy(float)
        size = past.groupby(['station_id', 'ts']).entries.transform('size').to_numpy(float)
        share = np.divide(past.entries.to_numpy(float), total, out=1/size, where=total > 0)
        past['entries'] = counts * share
        x = build_window(past, self.vestibules, now)
        sday = (now.tz_convert('Europe/Moscow').tz_localize(None) - pd.Timedelta(hours=5)).normalize()
        local = now.tz_convert('Europe/Moscow')
        day_type = self.calendar.loc[sday, 'day_type']
        codes = np.array([[i, local.hour*4 + local.minute//15, sday.dayofweek,
                           self.NN.DAY_TYPES.index(day_type)] for i in range(len(self.vestibules))], dtype=np.int64)
        import torch
        with torch.no_grad():
            xt = torch.from_numpy(self.scaler.transform(x))
            ct = torch.from_numpy(codes)
            z = np.mean([net(xt, ct).numpy().astype(float) for net in self.nets], axis=0)
        qn = self._target_norms(now)
        slots = decode_slots(z, qn.reshape(len(self.vestibules), 4, 2).sum(2))
        return dict(slots=slots, quarter_norms=qn, window=x, codes=codes, z=z,
                    observed=observed, native_update=now.minute % 30 == 0)

    def station_profile(self, station_id, target_hour):
        """Same causal profile used to divide GRU slots; no target entries."""
        if station_id not in self.mapping:
            raise ValueError('unknown red-line station')
        norms = self._target_norms(pd.Timestamp(target_hour).floor('h'))[:, :4]
        positions = [i for i, v in enumerate(self.vestibules)
                     if self.registry.loc[v, 'station_id'] == self.mapping[station_id]]
        weights = norms[positions].sum(0)
        return weights / weights.sum() if weights.sum() > 0 else np.full(4, .25)

    def forecast_bundle(self, frame, station_ids, as_of, horizon_hours=2):
        if horizon_hours not in (1, 2) or not station_ids or not set(station_ids) <= set(self.mapping) or len(set(station_ids)) != len(station_ids):
            raise ValueError('distinct red-line stations and horizon 1 or 2 required')
        now = pd.Timestamp(as_of)
        prediction = self.model_slots(frame, now)
        counts = split_slots(prediction['slots'][..., 1], prediction['quarter_norms'])
        by_station = {sid: np.zeros(8, dtype=np.int64) for sid in station_ids}
        for i, v in enumerate(self.vestibules):
            sid = self.inverse[self.registry.loc[v, 'station_id']]
            if sid in by_station:
                by_station[sid] += counts[i]
        covered = {self.inverse[self.registry.loc[v, 'station_id']] for v in self.vestibules}
        fallback = sorted(set(station_ids) - covered)
        observed = prediction['observed']
        if fallback:
            observed_fallback = observed_history(frame[frame.station_id.isin(fallback) & frame.interval_end.le(now)], now)
        for sid in fallback:
            data = observed_fallback[observed_fallback.station_id.eq(sid)].copy()
            data['hour'] = data.interval_start.dt.floor('h')
            hours = data.groupby('hour').agg(total=('entry_count', 'sum'), n=('entry_count', 'size'))
            hours = hours[hours.n.eq(4)]
            for j in range(8):
                target = now + pd.Timedelta(minutes=15*j)
                sample = hours[hours.index.hour == target.hour].total
                if sample.empty:
                    raise ValueError(f'missing fallback history for {sid}/{target.hour}')
                by_station[sid][j] = largest_remainders(int(np.floor(sample.median() + .5)), [1]*4)[target.minute//15]
        rows = [dict(station_id=sid, interval_start=(now + pd.Timedelta(minutes=15*j)).isoformat(),
                     interval_end=(now + pd.Timedelta(minutes=15*(j+1))).isoformat(),
                     predicted_count=int(by_station[sid][j]), coverage='complete')
                for j in range(horizon_hours * 4) for sid in station_ids]
        feature_end = observed.interval_end.max().isoformat()
        bundle = dict(schema_version='1.0', model_version='frozen-b3-gru-native30m-quarter-refresh-v1',
            issued_at=now.isoformat(), as_of=now.isoformat(), available_at=now.isoformat(),
            features_available_at=feature_end, target='turnstile_entries', unit='passengers_per_15m',
            coverage='complete', feature_version='original16x4-with-causal-quarter-slot-split-v1',
            source_versions=dict(weights=self.hashes, train_end=self.meta['info']['train_end'],
                model_family='upstream_standalone_B3_not_meta_stack', native_output_seconds=1800,
                native_training_update_seconds=1800, shifted_quarter_inference=not prediction['native_update'],
                fallback_station_ids=fallback, retrospective_history_flags=self.retrospective_flags,
                normalization='original baseline.norm+intrahour.past_profile+source_scale; strictly previous service days',
                quarter_split='native vestibule slot total allocated by causal target-quarter normalization',
                preprocessing_input_digest=digest(dict(window=np.nan_to_num(prediction['window']).tolist(),
                                                       codes=prediction['codes'].tolist(), norms=prediction['quarter_norms'].tolist())),
                future_truth_used=False, input_publication_times='unknown', target_anomaly_mask=False),
            included_event_ids=[], refresh_seconds=900,
            availability_assumption='Completed station intervals and original uploaded vestibule proportions are assumed available at interval end; calendar and entrance opening hours are treated as public schedules. Archive cleaning flags are disabled by default.',
            rounding='native half-hour totals rounded HALF_UP; largest remainders over causal quarter shares',
            rows=rows, evaluation_mode='frozen_GRU_native30m' if prediction['native_update'] else 'frozen_GRU_shifted15m_experimental',
            claimed_ml_WAPE=None)
        validate_bundle(bundle, station_ids, now.isoformat())
        return bundle

    def contract(self):
        return dict(meta=self.meta, vestibule_order=self.vestibules,
            weights_sha256=self.hashes, feature_shape=[23, 16, 4], output_shape=[23, 4, 3],
            input_channels=['log((quarter_entries+1)/(quarter_norm+1))', 'missing_ves',
                            'log((sum_valid_entries+1)/(sum_valid_norm+1))', 'missing_line'],
            native_target='four30minute_slot_log_deviations', native_training_update_seconds=1800,
            simulator_refresh_seconds=900, shifted_updates_are_experimental=True,
            standalone_prediction=True, full_stack_weights_in_archive=False,
            required_assets=['hourly.parquet', '15min.parquet', '15min_reconciliation.parquet',
                             'calendar2026', 'vestibule/reference schedules', 'threeweights+scaler'],
            uncovered_station_ids=sorted(set(self.mapping) - {self.inverse[self.registry.loc[v, 'station_id']] for v in self.vestibules}),
            historical_flags=self.retrospective_flags, original_source_files={
                name: file_hash(VENDOR / 'src' / name) for name in ['stack/nn.py', 'stack/data.py', 'intrahour.py', 'baseline.py']})
