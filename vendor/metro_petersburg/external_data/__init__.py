"""Public ML interface; importing existing submodules performs no API setup."""

from importlib import import_module

__all__ = [
    "configure_external_data", "get_external_features", "external_features_to_ml_dict",
    "get_external_feature_rows", "ExternalDataClient", "ExternalDataConfigurationError",
    "Event", "ExternalFeatures",
]


def __getattr__(name: str):
    # Lazy exports preserve independent imports of calendar/weather/etc. and avoid
    # circular imports while the facade reuses their public functions and models.
    if name in __all__:
        value = getattr(import_module(".api", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
