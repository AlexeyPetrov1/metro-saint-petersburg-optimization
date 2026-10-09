"""Follow a published comparison only after both saved arms are ready."""

import json
from pathlib import Path


def resolve_campaign(root, *, project_root):
    root = Path(root)
    try:
        scope = json.loads((root/"campaign_scope.json").read_bytes())
        target = Path(scope["comparison_root"])
        if not target.is_absolute():
            target = Path(project_root)/target
        status = json.loads((target/"status.json").read_bytes())
        if status.get("stage") not in ("materialized_partial", "completed_partial", "completed"):
            return root
        if all((target/arm/name).is_file() for arm in ("reference", "balanced")
               for name in ("result.json", "common_scenario.json.gz")):
            return target
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return root
