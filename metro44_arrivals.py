"""Separate arrival-time sensitivity; preserve the observed minute counts."""

from copy import deepcopy
from hashlib import sha256


def jitter_arrivals(groups, seed):
    result = deepcopy(groups)
    for group in result:
        gate, platform = group['turnstile_second'], group['platform_second']
        if any(not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in (gate, platform)) or platform < gate:
            raise ValueError('arrival times must be ordered nonnegative integer seconds')
        offset = int.from_bytes(sha256(f'{seed}:{group["group_id"]}'.encode()).digest()[:8], 'big') % 60
        shift = gate//60*60 + offset - gate
        group['turnstile_second'] += shift
        group['platform_second'] += shift
    return result
