"""Versioned station/hour medians, refreshed on the hour from past data only."""

from __future__ import annotations

import math
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from line1 import largest_remainders, minute_groups
from simulation import digest, fields_only, instant, integer


MODEL_VERSION = "station-hour-median-v1"


def hourly_median(frame, station_ids, as_of, horizon_hours=2):
    from metro_quality import filter_available_quality
    now = instant(as_of)
    if now.minute or now.second:
        raise ValueError("median refresh must occur on the hour")
    integer(horizon_hours, "horizon hours", 1)
    if not station_ids or len(set(station_ids)) != len(station_ids):
        raise ValueError("distinct station coverage is required")
    required = {"station_id", "interval_start", "interval_end", "entry_count", "is_complete", "source_id"}
    if not required <= set(frame.columns):
        raise ValueError("historical schema lacks station interval entries")
    data = filter_available_quality(frame, now)
    if data.duplicated(["station_id", "interval_start"]).any():
        raise ValueError("duplicate historical intervals")
    for key in ("interval_start", "interval_end"):
        if not isinstance(data[key].dtype, pd.DatetimeTZDtype):
            raise ValueError("history requires timezone-aware timestamps")
    if not pd.api.types.is_bool_dtype(data.is_complete.dtype):
        raise ValueError("history coverage must be boolean")
    data = data[(data.interval_end <= now) & data.station_id.isin(station_ids)]
    availability = "assumed: historical publication times are unavailable; complete intervals are assumed available at their end"
    provided = [key for key in ('available_at', 'received_at') if key in data]
    for key in provided:
        if not isinstance(data[key].dtype, pd.DatetimeTZDtype):
            raise ValueError(f"historical {key} needs a timezone")
        data = data[data[key].notna() & (data[key] <= now)]
    if provided:
        availability = "provided " + "/".join(provided) + " is enforced"
    data = data[data.is_complete.eq(True) & data.entry_count.notna()].copy()
    if (not pd.api.types.is_integer_dtype(data.entry_count.dtype)
            or pd.api.types.is_bool_dtype(data.entry_count.dtype) or (data.entry_count < 0).any()):
        raise ValueError("historical entries must be nonnegative integers")
    data["hour_start"] = data.interval_start.dt.floor("h")
    hours={(now+timedelta(hours=h)).hour for h in range(horizon_hours)}
    data=data[data.interval_start.dt.hour.isin(hours)]
    totals = []
    aligned=(data.interval_start.dt.minute.mod(15).eq(0) & data.interval_start.dt.second.eq(0)
        & data.interval_end.sub(data.interval_start).eq(pd.Timedelta(minutes=15)))
    data=data[aligned]
    feature_available = pd.concat([data[key].dt.tz_convert('UTC')
        for key in ('interval_end', *provided)], axis=1).max(axis=1).max()
    aggregate=data.groupby(["station_id","hour_start"],sort=True).agg(
        rows=("entry_count","size"),count=("entry_count","sum"),sources=("source_id","nunique"),
        latest_end=("interval_end","max"),source_id=("source_id","first")).reset_index()
    for row in aggregate[(aggregate.rows==4)&(aggregate.sources==1)].itertuples(index=False):
        totals.append(dict(station_id=row.station_id,hour=row.hour_start.hour,count=int(row.count),
            latest_end=row.latest_end,source_id=str(row.source_id)))
    statistics = pd.DataFrame(totals)
    rows = []
    for h in range(horizon_hours):
        hour = now + timedelta(hours=h)
        for station in station_ids:
            sample = statistics[(statistics.station_id == station) & (statistics.hour == hour.hour)] if totals else statistics
            if sample.empty:
                raise ValueError(f"missing complete historical coverage for {station}/{hour.hour:02}:00")
            median = Decimal(str(sample['count'].median()))
            total = int(median.to_integral_value(rounding=ROUND_HALF_UP))
            for q, count in enumerate(largest_remainders(total, [1]*4)):
                start = hour + timedelta(minutes=15*q)
                rows.append(dict(station_id=station, interval_start=start.isoformat(),
                    interval_end=(start+timedelta(minutes=15)).isoformat(), predicted_count=count,
                    coverage="complete", historical_hours=len(sample), hourly_median=str(median),
                    rounded_hour_count=total))
    latest = feature_available.isoformat()
    bundle = dict(schema_version="1.0", model_version=MODEL_VERSION, issued_at=as_of, as_of=as_of,
        available_at=as_of, features_available_at=latest, target="turnstile_entries", unit="passengers_per_15m",
        coverage="complete", feature_version="complete-station-hour-totals-v1",
        source_versions={"source_ids": sorted({t["source_id"] for t in totals}),
                         "latest_observation_available_at": latest,
                         "history_digest": digest(totals_for_digest(totals))},
        availability_assumption=availability, included_event_ids=[], refresh_seconds=3600,
        rounding="median of complete hourly totals; ROUND_HALF_UP, then largest remainders over four quarters",
        rows=rows, evaluation_mode="historical_median_forecast", claimed_ml_WAPE=None)
    validate_bundle(bundle, station_ids, as_of)
    return bundle


def totals_for_digest(totals):
    return [{**t, "latest_end": t["latest_end"].isoformat()} for t in totals]


def validate_bundle(bundle, station_ids, decision_time):
    fields_only(bundle, ("schema_version", "model_version", "issued_at", "as_of", "available_at",
        "features_available_at", "target", "unit", "coverage", "feature_version", "source_versions",
        "included_event_ids", "rows"), ("availability_assumption", "refresh_seconds", "rounding",
        "evaluation_mode", "claimed_ml_WAPE"))
    now = instant(decision_time)
    issued, cutoff, available, features = [instant(bundle[k]) for k in
        ("issued_at", "as_of", "available_at", "features_available_at")]
    if not features <= cutoff <= issued <= available <= now:
        raise ValueError("future features or forecast unavailable at decision time")
    if (now-issued).total_seconds() >= 3600:
        raise ValueError("stale hourly forecast")
    if (bundle["schema_version"] != "1.0" or bundle["target"] != "turnstile_entries"
            or bundle["unit"] != "passengers_per_15m" or bundle["coverage"] != "complete"):
        raise ValueError("forecast target, unit or coverage is incompatible with new arrivals")
    if not all(isinstance(bundle[k],str) and bundle[k].strip() for k in ("model_version", "feature_version")):
        raise ValueError("forecast needs model and feature versions")
    if not isinstance(bundle["source_versions"],dict) or not bundle["source_versions"]:
        raise ValueError("source versions are required")
    if not isinstance(bundle["included_event_ids"],list) or len(set(bundle["included_event_ids"])) != len(bundle["included_event_ids"]):
        raise ValueError("event identities must be unique")
    rows = bundle["rows"]
    if not isinstance(rows,list) or not rows:
        raise ValueError("empty forecast coverage")
    keys, intervals = set(), set()
    for row in rows:
        fields_only(row, ("station_id", "interval_start", "interval_end", "predicted_count", "coverage"),
                    ("historical_hours", "hourly_median", "rounded_hour_count"))
        start,end = instant(row["interval_start"]), instant(row["interval_end"])
        count = row["predicted_count"]
        if (type(count) not in (int,float) or not math.isfinite(count) or count < 0
                or row["coverage"] != "complete" or row["station_id"] not in station_ids
                or start < cutoff or end-start != timedelta(minutes=15) or start.minute % 15 or start.second):
            raise ValueError("invalid forecast interval, count or station coverage")
        key = (row["station_id"],start)
        if key in keys: raise ValueError("duplicate forecast interval")
        keys.add(key); intervals.add(start)
    first,last = min(intervals), max(intervals)
    starts = {first+timedelta(minutes=15*i) for i in range(int((last-first).total_seconds())//900+1)}
    if intervals != starts or keys != {(s,t) for s in station_ids for t in starts}:
        raise ValueError("incomplete forecast coverage")
    horizon = int((last+timedelta(minutes=15)-now).total_seconds())
    if horizon <= 0: raise ValueError("forecast has no future horizon")
    return dict(status="passed", available_horizon_seconds=horizon,
                publication_time_status=bundle.get("availability_assumption","provided"))


def forecast_groups(bundle, station_ids, period_start, delay=120, extra_event_ids=()):
    validate_bundle(bundle,station_ids,bundle["available_at"])
    if set(extra_event_ids) & set(bundle["included_event_ids"]):
        raise ValueError("event effect already included in the prediction")
    if extra_event_ids:
        raise ValueError("event demand transformation is unavailable; no arbitrary multiplier is applied")
    rows = []
    for row in bundle["rows"]:
        count = int(Decimal(str(row["predicted_count"])).to_integral_value(rounding=ROUND_HALF_UP))
        rows.append(dict(station_id=row["station_id"],interval_start=pd.Timestamp(row["interval_start"]),
            entry_count=count,source_id=bundle["model_version"]))
    frame = pd.DataFrame(rows)
    start = pd.Timestamp(period_start)
    end = max(frame.interval_start)+pd.Timedelta(minutes=15)
    return minute_groups(frame,station_ids,start,end,delay)

