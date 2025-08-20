"""Shared utilities used by example scripts to keep them DRY."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Banner:
    """Simple pretty printer for section banners in examples."""

    width: int = 60

    def title(self, text: str) -> None:
        print("\n" + "=" * self.width)
        print(text)
        print("=" * self.width)

    def step(self, text: str) -> None:
        print(f"\n{text}")
