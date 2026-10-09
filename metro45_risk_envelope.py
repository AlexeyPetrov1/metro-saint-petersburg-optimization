"""Entry-demand scenarios and a causal planning envelope, not a load guarantee."""
from copy import deepcopy

import pandas as pd

from forecasting import validate_bundle
from simulation import digest


VERSION='entry-risk-envelope-v1'


def _key(row):
    return row['station_id'],pd.Timestamp(row['interval_start']),pd.Timestamp(row['interval_end'])


def model_scenarios(original_bundle,adjusted_bundle,as_of):
    """Return independent (original future, corrected future, max-entry guard)."""
    now=pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ValueError('risk envelope cutoff requires timezone')
    stations=sorted({r['station_id'] for r in original_bundle['rows']})
    for bundle in (original_bundle,adjusted_bundle):
        validate_bundle(bundle,stations,now.isoformat())
        if 'risk_envelope' in bundle['source_versions']:
            raise ValueError('bundle already contains a risk envelope')
    if (original_bundle['model_version']!=adjusted_bundle['model_version'] or
            original_bundle['source_versions'].get('model_sha256')!=adjusted_bundle['source_versions'].get('model_sha256')):
        raise ValueError('risk envelope model weights must remain unchanged')
    if set(original_bundle['included_event_ids'])!=set(adjusted_bundle['included_event_ids']):
        raise ValueError('forecast event provenance differs')
    original,adjusted=deepcopy(original_bundle),deepcopy(adjusted_bundle)
    for bundle in (original,adjusted):
        bundle['rows']=[r for r in bundle['rows'] if pd.Timestamp(r['interval_start'])>=now]
    base={_key(r):r for r in original['rows']}
    corrected={_key(r):r for r in adjusted['rows']}
    if not base or base.keys()!=corrected.keys():
        raise ValueError('future station/timestamp keys must match exactly')
    corrections=[adjusted_bundle['source_versions'][tag] for tag in ('residual_nowcast','nowcast')
                 if tag in adjusted_bundle['source_versions']]
    if corrections:
        if any(not isinstance(meta,dict) or meta.get('original_bundle_digest')!=digest(original_bundle)
               for meta in corrections):
            raise ValueError('adjusted forecast original provenance does not match')
    elif any(base[key]['predicted_count']!=corrected[key]['predicted_count'] for key in base):
        raise ValueError('changed forecast lacks original correction provenance')
    guard=deepcopy(adjusted)
    guard['rows']=[{**row,'predicted_count':max(row['predicted_count'],corrected[_key(row)]['predicted_count'])}
                   for row in original['rows']]
    latest={key:max(pd.Timestamp(b[key]) for b in (original_bundle,adjusted_bundle))
            for key in ('as_of','features_available_at','issued_at','available_at')}
    issued=now
    guard.update(as_of=latest['as_of'].isoformat(),features_available_at=latest['features_available_at'].isoformat(),
        issued_at=issued.isoformat(),available_at=issued.isoformat(),
        feature_version=adjusted_bundle['feature_version']+'+'+VERSION,
        evaluation_mode='entry_demand_planning_envelope',rounding='exact maximum of published quarter counts')
    guard['source_versions']['risk_envelope']=dict(version=VERSION,rule='elementwise_max',as_of=now.isoformat(),
        original_bundle_digest=digest(original_bundle),adjusted_bundle_digest=digest(adjusted_bundle),
        original_future_bundle_digest=digest(original),adjusted_future_bundle_digest=digest(adjusted),
        original_source_versions=deepcopy(original_bundle['source_versions']),
        original_feature_version=original_bundle['feature_version'],adjusted_feature_version=adjusted_bundle['feature_version'],
        original_publication=original_bundle['available_at'],adjusted_publication=adjusted_bundle['available_at'],
        is_point_estimate=False,train_load_guarantee=False,physical_or_service_approval=False,
        bound='for identical true entry counts, positive underforecast never exceeds original ML underforecast',
        limitation='overforecast can grow; OD, queues, load and resulting actions still require separate validation')
    for bundle in (original,adjusted,guard):
        validate_bundle(bundle,stations,now.isoformat())
    return original,adjusted,guard


def build_risk_envelope(original_bundle,adjusted_bundle,as_of):
    """Return the future-only elementwise maximum with both input provenances."""
    return model_scenarios(original_bundle,adjusted_bundle,as_of)[2]
