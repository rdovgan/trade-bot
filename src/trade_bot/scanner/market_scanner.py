"""Market scanner for discovering and ranking tradeable coins."""

import logging
import math
from typing import Dict, List, Optional, Set

from .models import CoinCandidate

logger = logging.getLogger(__name__)


class MarketScanner:
    """
    Scans all available pairs on an exchange, filters by criteria,
    scores them, and returns the top N candidates for trading.
    """

    def __init__(self, config: Optional[Dict] = None):
        defaults = self._default_config()
        if config:
            defaults.update(config)
        self.config = defaults
        self._last_candidates: List[CoinCandidate] = []

    @staticmethod
    def _default_config() -> Dict:
        return {
            "quote_currency": "USDT",
            "min_volume_24h": 1_000_000,
            "max_positions": 5,
            "blacklist": [],
            # scoring weights
            "weight_volume": 0.30,
            "weight_momentum": 0.25,
            "weight_spread": 0.25,
            "weight_volatility": 0.20,
            # volatility sweet-spot bounds (annualized approx)
            "vol_min": 0.10,
            "vol_max": 1.50,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan_market(self, exchange) -> List[CoinCandidate]:
        """
        Full market scan.

        *exchange* is expected to expose CCXT-compatible async methods:
        ``load_markets()``, ``fetch_tickers()``.
        """
        # load_markets with reload=False reuses cached data if available
        markets = await exchange.load_markets(reload=False)
        symbols_set = set(self._filter_pairs(markets))

        if not symbols_set:
            logger.warning("No symbols passed filtering")
            return []

        tickers = await exchange.fetch_tickers()

        candidates: List[CoinCandidate] = []
        for symbol, ticker in tickers.items():
            if symbol not in symbols_set:
                continue
            candidate = self._build_candidate(symbol, ticker)
            if candidate is not None:
                candidates.append(candidate)

        # Normalise & score
        candidates = self._normalise_and_score(candidates)

        # Sort descending by score
        candidates.sort(key=lambda c: c.score, reverse=True)
        self._last_candidates = candidates
        return candidates

    def get_top_candidates(self, n: Optional[int] = None) -> List[CoinCandidate]:
        """Return top *n* candidates from the last scan."""
        n = n or self.config["max_positions"]
        return self._last_candidates[:n]

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_pairs(self, markets: Dict) -> List[str]:
        """Filter active pairs with the configured quote currency."""
        quote = self.config["quote_currency"]
        blacklist: Set[str] = set(self.config.get("blacklist", []))
        result: List[str] = []

        for symbol, info in markets.items():
            if symbol in blacklist:
                continue
            if not info.get("active", True):
                continue
            if info.get("quote", "") != quote:
                continue
            if info.get("linear") is not True and info.get("type", "") not in ("swap", "future", "spot"):
                continue
            result.append(symbol)

        logger.info(f"Filtered {len(result)} {quote} pairs from {len(markets)} total")
        return result

    # ------------------------------------------------------------------
    # Building & scoring
    # ------------------------------------------------------------------

    def _build_candidate(self, symbol: str, ticker: Dict) -> Optional[CoinCandidate]:
        """Build a CoinCandidate from a ticker dict, or None if it fails filters."""
        volume_24h = ticker.get("quoteVolume") or 0.0
        if volume_24h < self.config["min_volume_24h"]:
            return None

        last = ticker.get("last") or 0.0
        bid = ticker.get("bid") or 0.0
        ask = ticker.get("ask") or 0.0
        change_pct = ticker.get("percentage") or 0.0  # 24h % change

        spread_pct = ((ask - bid) / last * 100) if last > 0 else 999.0

        # Very rough intra-day volatility proxy: high-low range / last
        high = ticker.get("high") or last
        low = ticker.get("low") or last
        volatility = ((high - low) / last) if last > 0 else 0.0

        return CoinCandidate(
            symbol=symbol,
            score=0.0,  # will be set after normalisation
            volume_24h=volume_24h,
            price_change_pct=change_pct,
            spread_pct=spread_pct,
            volatility=volatility,
        )

    def _normalise_and_score(self, candidates: List[CoinCandidate]) -> List[CoinCandidate]:
        """Normalise features to 0-1 and compute composite score."""
        if not candidates:
            return candidates

        # Collect raw values
        volumes = [c.volume_24h for c in candidates]
        momentums = [abs(c.price_change_pct) for c in candidates]
        spreads = [c.spread_pct for c in candidates]
        vols = [c.volatility for c in candidates]

        max_vol = max(volumes) or 1.0
        max_mom = max(momentums) or 1.0
        max_spread = max(spreads) or 1.0

        vol_min = self.config["vol_min"]
        vol_max = self.config["vol_max"]

        w_volume = self.config["weight_volume"]
        w_momentum = self.config["weight_momentum"]
        w_spread = self.config["weight_spread"]
        w_volatility = self.config["weight_volatility"]

        for c in candidates:
            norm_volume = c.volume_24h / max_vol
            norm_momentum = abs(c.price_change_pct) / max_mom if max_mom else 0.0
            # Lower spread is better -> invert
            norm_spread = 1.0 - (c.spread_pct / max_spread) if max_spread else 0.0
            # Volatility sweet-spot: peak at midpoint, drops at extremes
            mid = (vol_min + vol_max) / 2
            half_range = (vol_max - vol_min) / 2
            norm_vol = max(0.0, 1.0 - abs(c.volatility - mid) / half_range) if half_range else 0.0

            c.score = (
                w_volume * norm_volume
                + w_momentum * norm_momentum
                + w_spread * norm_spread
                + w_volatility * norm_vol
            )

        return candidates
