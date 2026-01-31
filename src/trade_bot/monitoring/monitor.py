"""Monitoring engine for exchange health, slippage tracking, and regime shifts."""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from ..core.enums import Regime
from ..core.models import MarketState, PositionState

logger = logging.getLogger(__name__)


@dataclass
class ExchangeHealthStatus:
    """Exchange health metrics."""
    is_healthy: bool = True
    last_response_ms: float = 0.0
    avg_response_ms: float = 0.0
    consecutive_failures: int = 0
    last_check: Optional[datetime] = None


@dataclass
class SlippageRecord:
    """Record of slippage for a single order."""
    symbol: str
    expected_price: float
    actual_price: float
    slippage_bps: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeShiftEvent:
    """Records a detected regime shift."""
    symbol: str
    old_regime: Regime
    new_regime: Regime
    timestamp: datetime = field(default_factory=datetime.now)


class MonitoringEngine:
    """Monitors exchange health, slippage, and regime shifts.

    Every monitoring cycle checks:
    - Exchange health (response times, failures)
    - Slippage vs expected
    - Regime shift since position entry
    """

    def __init__(
        self,
        max_response_ms: float = 5000.0,
        max_consecutive_failures: int = 3,
        max_slippage_bps: float = 10.0,
        response_history_size: int = 50,
    ):
        self.max_response_ms = max_response_ms
        self.max_consecutive_failures = max_consecutive_failures
        self.max_slippage_bps = max_slippage_bps

        self._health = ExchangeHealthStatus()
        self._response_times: deque = deque(maxlen=response_history_size)
        self._slippage_history: List[SlippageRecord] = []
        self._regime_at_entry: Dict[str, Regime] = {}
        self._regime_shift_events: List[RegimeShiftEvent] = []

        logger.info("Monitoring engine initialized")

    # ------------------------------------------------------------------
    # Exchange health
    # ------------------------------------------------------------------

    async def check_exchange_health(self, exchange_connector) -> ExchangeHealthStatus:
        """Ping the exchange and update health status."""
        start = time.monotonic()
        try:
            await exchange_connector.get_balance()
            elapsed_ms = (time.monotonic() - start) * 1000

            self._response_times.append(elapsed_ms)
            self._health.last_response_ms = elapsed_ms
            self._health.avg_response_ms = (
                sum(self._response_times) / len(self._response_times)
            )
            self._health.consecutive_failures = 0
            self._health.is_healthy = elapsed_ms <= self.max_response_ms
            self._health.last_check = datetime.now()

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            self._health.consecutive_failures += 1
            self._health.last_response_ms = elapsed_ms
            self._health.last_check = datetime.now()
            self._health.is_healthy = (
                self._health.consecutive_failures < self.max_consecutive_failures
            )
            logger.warning(
                f"Exchange health check failed ({self._health.consecutive_failures}): {e}"
            )

        return self._health

    @property
    def exchange_healthy(self) -> bool:
        return self._health.is_healthy

    # ------------------------------------------------------------------
    # Slippage tracking
    # ------------------------------------------------------------------

    def record_slippage(
        self,
        symbol: str,
        expected_price: float,
        actual_price: float,
    ) -> SlippageRecord:
        """Record and evaluate slippage for an executed order."""
        if expected_price == 0:
            slippage_bps = 0.0
        else:
            slippage_bps = abs(actual_price - expected_price) / expected_price * 10000

        record = SlippageRecord(
            symbol=symbol,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_bps=slippage_bps,
        )
        self._slippage_history.append(record)

        if slippage_bps > self.max_slippage_bps:
            logger.warning(
                f"Excessive slippage for {symbol}: {slippage_bps:.1f} bps "
                f"(expected {expected_price}, got {actual_price})"
            )

        return record

    def get_avg_slippage_bps(self, symbol: Optional[str] = None, last_n: int = 20) -> float:
        """Get average slippage in basis points."""
        records = self._slippage_history
        if symbol:
            records = [r for r in records if r.symbol == symbol]
        records = records[-last_n:]
        if not records:
            return 0.0
        return sum(r.slippage_bps for r in records) / len(records)

    def slippage_acceptable(self, symbol: Optional[str] = None) -> bool:
        """Check if recent slippage is within acceptable range."""
        return self.get_avg_slippage_bps(symbol) <= self.max_slippage_bps

    # ------------------------------------------------------------------
    # Regime shift detection
    # ------------------------------------------------------------------

    def record_entry_regime(self, symbol: str, regime: Regime):
        """Store the regime at position entry for later comparison."""
        self._regime_at_entry[symbol] = regime

    def clear_entry_regime(self, symbol: str):
        """Clear stored entry regime when position is closed."""
        self._regime_at_entry.pop(symbol, None)

    def check_regime_shift(
        self, symbol: str, current_regime: Regime
    ) -> Optional[RegimeShiftEvent]:
        """Check if regime has shifted since position entry."""
        entry_regime = self._regime_at_entry.get(symbol)
        if entry_regime is None:
            return None

        if current_regime != entry_regime:
            event = RegimeShiftEvent(
                symbol=symbol,
                old_regime=entry_regime,
                new_regime=current_regime,
            )
            self._regime_shift_events.append(event)
            logger.info(
                f"Regime shift detected for {symbol}: "
                f"{entry_regime.value} -> {current_regime.value}"
            )
            return event

        return None

    # ------------------------------------------------------------------
    # Full monitoring cycle
    # ------------------------------------------------------------------

    async def run_monitoring_cycle(
        self,
        exchange_connector,
        market_states: Dict[str, MarketState],
        position_states: Dict[str, PositionState],
    ) -> Dict:
        """Run a full monitoring cycle. Returns a summary dict."""
        # 1. Exchange health
        health = await self.check_exchange_health(exchange_connector)

        # 2. Regime shift checks for open positions
        regime_shifts = {}
        for symbol, market_state in market_states.items():
            event = self.check_regime_shift(symbol, market_state.regime_label)
            if event:
                regime_shifts[symbol] = event

        return {
            "exchange_healthy": health.is_healthy,
            "exchange_response_ms": health.last_response_ms,
            "regime_shifts": regime_shifts,
            "avg_slippage_bps": self.get_avg_slippage_bps(),
            "slippage_acceptable": self.slippage_acceptable(),
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_health(self) -> ExchangeHealthStatus:
        return self._health

    def get_slippage_history(self) -> List[SlippageRecord]:
        return list(self._slippage_history)

    def get_regime_shift_events(self) -> List[RegimeShiftEvent]:
        return list(self._regime_shift_events)
