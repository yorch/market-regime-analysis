#!/usr/bin/env python3
"""
Dev runner — works without installing the package.

Usage:
    python run.py current-analysis --symbol SPY
    python run.py --help
"""

import sys

# Ensure workspace packages are importable from source
sys.path.insert(0, "packages/mra_lib/src")
sys.path.insert(0, "packages/mra_cli/src")
sys.path.insert(0, "packages/mra_web/src")

from mra_cli.main import cli

if __name__ == "__main__":
    cli()
