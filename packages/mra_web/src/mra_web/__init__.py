"""Market Regime Analysis Web API."""

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("mra-web")
except PackageNotFoundError:  # pragma: no cover - not installed (e.g. bare source checkout)
    __version__ = "0.0.0+unknown"
