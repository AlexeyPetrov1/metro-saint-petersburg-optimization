"""Internal candidate matching; lexical evidence only, no geographic inference."""

import re
from collections.abc import Callable

from .models import GeocodedLocation, GeoPoint


class GeocoderCandidate(GeoPoint):
    name: str | None = None
    formatted_address: str | None = None
    description: str | None = None
    text: str | None = None
    kind: str | None = None
    locality: str | None = None
    country_code: str | None = None


def _city_name(value: str) -> str:
    return re.sub(r"^(?:г\.?|город)\s+", "", value.strip().casefold()).strip()


def is_petersburg_address(address: str | None) -> bool:
    """Require a city component, not 'Санкт-Петербург' inside a street name."""
    return any(_city_name(part) in {"санкт-петербург", "спб"} for part in (address or "").split(","))


def resolve_address_candidates(
    query: str,
    fetch: Callable[[str], list[GeocoderCandidate]],
    debug: Callable[[str, object], None] = lambda label, value: None,
) -> GeocodedLocation | None:
    candidates = fetch(query)
    for index, candidate in enumerate(candidates, 1):
        # Structured locality takes precedence over an incidental city mention.
        in_city = (_city_name(candidate.locality) in {"санкт-петербург", "спб"}
                   if candidate.locality is not None else
                   is_petersburg_address(candidate.formatted_address) or is_petersburg_address(candidate.text))
        accepted = (in_city and candidate.country_code in (None, "RU")
                    and candidate.kind not in {"locality", "province", "country", "area", "district"})
        debug(f"candidate {index}", candidate.model_dump())
        debug("address candidate check", {"index": index, "in_city": in_city, "acceptable": accepted})
        if accepted:
            debug("selected candidate", candidate.model_dump())
            return GeocodedLocation(
                query=query, latitude=candidate.latitude, longitude=candidate.longitude,
                formatted_address=candidate.formatted_address,
            )
    debug("selected candidate", {"result": None, "reason": "address not found in Saint Petersburg"})
    return None


_CONTEXT = {"россия", "санкт", "петербург", "спб", "метро", "г", "город"}
_GENERIC = {
    "бкз", "большой", "концертный", "зал", "театр", "арена", "парк", "музей",
    "дворец", "центр", "улица", "ул", "проспект", "площадь", "на", "в", "имени",
}


def significant_tokens(value: str) -> frozenset[str]:
    tokens = set(re.findall(r"[а-яa-z0-9]+", value.casefold().replace("ё", "е")))
    # An explicit lexical abbreviation, not an inferred venue identity.
    if {"большой", "концертный", "зал"} <= tokens:
        tokens.difference_update({"большой", "концертный", "зал"})
        tokens.add("бкз")
    return frozenset(tokens - _CONTEXT)


def candidate_score(query: str, candidate: GeocoderCandidate) -> float:
    """Fraction of source-name tokens present; this is not a probability."""
    required = significant_tokens(query)
    available = frozenset().union(*(significant_tokens(value or "") for value in (
        candidate.name, candidate.formatted_address, candidate.description, candidate.text,
    )))
    return len(required & available) / len(required) if required else 0.0


def _acceptable(query: str, candidate: GeocoderCandidate, score: float) -> bool:
    required = significant_tokens(query)
    if score != 1.0 or not required - _GENERIC:
        return False
    if len(required) == 1:
        # A single name must stand alone, not merely occur in a longer address/name.
        return any(significant_tokens(value or "") == required for value in (
            candidate.name, candidate.formatted_address, candidate.text,
        ))
    return True


def query_variants(query: str) -> list[str]:
    """At most three requests; only expand an explicitly supplied venue category."""
    variants = [query]
    if "бкз" in significant_tokens(query) and not query.lstrip().casefold().startswith("метро "):
        name = re.sub(r",?\s*санкт-петербург\s*", "", query, flags=re.I).strip(" ,")
        variants += [
            f"{name} концертный зал, Санкт-Петербург",
            f"концертный зал {name}, Санкт-Петербург",
        ]
    return list(dict.fromkeys(variants))


def resolve_candidates(
    query: str,
    fetch: Callable[[str], list[GeocoderCandidate]],
    debug: Callable[[str, object], None] = lambda label, value: None,
) -> GeocodedLocation | None:
    for variant in query_variants(query):
        candidates = fetch(variant)
        scored = []
        for index, candidate in enumerate(candidates, 1):
            score = candidate_score(query, candidate)  # Always the original name.
            acceptable = _acceptable(query, candidate, score)
            debug(f"candidate {index}", candidate.model_dump())
            debug("candidate score", {"index": index, "score": score, "acceptable": acceptable})
            if acceptable:
                scored.append((score, candidate))
        if scored:
            # Stable API order for equal scores, identical logic in fake and real provider.
            selected = max(scored, key=lambda item: item[0])[1]
            debug("selected candidate", selected.model_dump())
            return GeocodedLocation(
                query=query, latitude=selected.latitude, longitude=selected.longitude,
                formatted_address=selected.formatted_address,
            )
    debug("selected candidate", {"result": None, "reason": "location could not be resolved confidently"})
    return None
