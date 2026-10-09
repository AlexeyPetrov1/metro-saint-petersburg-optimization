"""Share existing terminal slots across depot entries and incoming line trains."""

from dwell import compile_operational_timing


ENTRY_DIRECTIONS = {'ENTRY_AVTOVO': 'ENTRY_AVTOVO_down',
                    'ENTRY_NORTHERN': 'ENTRY_NORTHERN_up'}


def rename_tracks(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, dict):
        return {rename_tracks(k, mapping): rename_tracks(v, mapping) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(rename_tracks(v, mapping) for v in value)
    return value


def _started_movements(data, as_of):
    tracks = {row['track_id']: row for row in data['tracks']}
    trips = {row['trip_id']: row for plan in data['plans'] for row in plan['trips']}
    trains = {row['train_id']: row for row in data['trains']}
    started = {}
    for row in data.get('leg_timing', []):
        if row['departure_second'] >= as_of:
            continue
        key = row['train_id'], row['trip_id'], row['leg_index']
        track = trains[row['train_id']]['location'] if row['trip_id'] == '@initial' else trips[row['trip_id']]['route'][row['leg_index']]
        arrival = row['departure_second'] + tracks[track]['travel_seconds'] + row.get('approach_holding_seconds', 0)
        started[key] = row['departure_second'], arrival
    return started


def compile_terminal_timing(data, plan_id, profile='fixed', seed=0, **kwargs):
    track_ids = {row['track_id'] for row in data['tracks']}
    mapping = {old: new for old,new in ENTRY_DIRECTIONS.items() if old in track_ids}
    if set(mapping.values()) & track_ids:
        raise ValueError('terminal entry alias collision')
    as_of = kwargs.get('as_of', 0)
    started = _started_movements(data, as_of)
    compiled, audit = compile_operational_timing(rename_tracks(data, mapping), plan_id,
        profile, seed, **rename_tracks(kwargs, mapping))
    inverse = {v:k for k,v in mapping.items()}
    compiled, audit = rename_tracks(compiled, inverse), rename_tracks(audit, inverse)
    if started != {k:v for k,v in _started_movements(compiled, as_of).items() if k in started}:
        raise ValueError('terminal grouping changes an already started movement')
    if compiled['tracks'] != data['tracks']:
        raise ValueError('terminal grouping changes physical tracks')
    audit['terminal_resource_grouping'] = 'existing entry and incoming line tracks share terminal slots; signalling unverified'
    return compiled, audit
