"""Market Regime Analysis CLI."""

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("mra-cli")
except PackageNotFoundError:  # pragma: no cover - not installed (e.g. bare source checkout)
    __version__ = "0.0.0+unknown"
