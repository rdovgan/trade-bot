"""Models for market scanner."""

from dataclasses import dataclass, field
from typing import Optional

from ..core.enums import Regime


@dataclass
class CoinCandidate:
    """Represents a scored coin candidate from market scanning."""

    symbol: str
    score: float
    volume_24h: float
    price_change_pct: float
    spread_pct: float
    volatility: float
    regime: Optional[Regime] = None

    def __lt__(self, other: "CoinCandidate") -> bool:
        return self.score < other.score
