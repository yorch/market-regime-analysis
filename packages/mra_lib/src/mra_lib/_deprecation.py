"""Helpers for the deprecated ``print_*`` report shims."""

import sys
import warnings


def write_deprecated_report(text: str, old: str, new: str) -> None:
    """Warn that ``old`` is deprecated in favour of ``new`` and write ``text`` to stdout.

    The library itself never prints; this exists only so the historical
    ``print_*`` methods keep working for external callers.
    """
    warnings.warn(
        f"{old}() is deprecated and will be removed in a future release; "
        f"use {new}() and print or log the returned string",
        DeprecationWarning,
        stacklevel=3,
    )
    sys.stdout.write(text + "\n")
