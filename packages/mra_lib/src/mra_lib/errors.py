"""
Exception hierarchy for ``mra_lib``.

Every error the library raises on purpose derives from :class:`MRAError`, so
callers can catch library failures with one ``except MRAError``. The concrete
classes also derive from the built-in they historically surfaced as
(``ValueError`` / ``ConnectionError``), so existing ``except ValueError``
handlers keep working.

Data-provider errors (:class:`ProviderError` and its subclasses
``InvalidSymbolError``, ``AuthError``, ``RateLimitError``) are part of the same
hierarchy; ``mra_lib.data_providers`` re-exports them. ``MarketRegimeAnalyzer`` lets them
propagate unchanged, so callers can map them directly (no cause-chain walking).

::

    MRAError
    ├── DataLoadError (ValueError)          data could not be loaded/validated
    ├── InsufficientDataError (ValueError)  not enough data for the operation
    ├── ModelNotFittedError (ValueError)    no fitted model for the request
    ├── InvalidParametersError (ValueError) bad user-supplied parameters (e.g. strategy)
    └── ProviderError                       raised by data providers
        ├── InvalidSymbolError (ValueError)
        ├── AuthError (ConnectionError)
        └── RateLimitError (ConnectionError)
"""


class MRAError(Exception):
    """Base class for errors raised deliberately by ``mra_lib``."""


class DataLoadError(MRAError, ValueError):
    """Market data could not be loaded or failed validation (empty, missing columns)."""


class InsufficientDataError(MRAError, ValueError):
    """There is not enough data (bars, rows, results) to perform the operation."""


class ModelNotFittedError(MRAError, ValueError):
    """No fitted model is available for the requested timeframe or operation."""


class InvalidParametersError(MRAError, ValueError):
    """User-supplied parameters (e.g. a strategy parameter file) are invalid."""


class ProviderError(MRAError):
    """Marker base for errors raised by data providers."""


class InvalidSymbolError(ProviderError, ValueError):
    """The symbol is unknown to the provider or has no data for the request."""


class AuthError(ProviderError, ConnectionError):
    """The provider rejected the credentials (missing, invalid, or not entitled)."""


class RateLimitError(ProviderError, ConnectionError):
    """The provider is throttling requests or the quota is exhausted."""


__all__ = [
    "AuthError",
    "DataLoadError",
    "InsufficientDataError",
    "InvalidParametersError",
    "InvalidSymbolError",
    "MRAError",
    "ModelNotFittedError",
    "ProviderError",
    "RateLimitError",
]
