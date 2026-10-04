"""
Ticker symbol format shared by the storage layer, CLI and web API.
"""

import re

# Ticker symbols: letters, digits and . - ^ = (e.g. BRK.B, ^GSPC, ES=F), max 15 chars.
# The first character may not be '.', '-' or '=' (avoids CSV formula injection too).
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.\-^=]{0,14}$")
