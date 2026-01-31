"""Unit tests for MonitoringEngine."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from trade_bot.core.enums import Regime
from trade_bot.core.models import MarketState, PositionState
from trade_bot.monitoring.monitor import (
    MonitoringEngine,
    ExchangeHealthStatus,
    SlippageRecord,
    RegimeShiftEvent,
)


class MockExchange:
    """Simple mock exchange for health checks."""

    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    async def get_balance(self):
        if self.should_fail:
            raise ConnectionError("Exchange down")
        return {"USDT": 10000.0}


class TestMonitoringEngine:
    def setup_method(self):
        self.engine = MonitoringEngine(
            max_response_ms=5000,
            max_consecutive_failures=3,
            max_slippage_bps=10.0,
        )

    # --- Exchange health ---

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        exchange = MockExchange()
        health = await self.engine.check_exchange_health(exchange)
        assert health.is_healthy is True
        assert health.consecutive_failures == 0
        assert health.last_check is not None

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        exchange = MockExchange(should_fail=True)
        health = await self.engine.check_exchange_health(exchange)
        assert health.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_health_degraded_after_failures(self):
        exchange = MockExchange(should_fail=True)
        for _ in range(3):
            await self.engine.check_exchange_health(exchange)
        assert self.engine.exchange_healthy is False

    @pytest.mark.asyncio
    async def test_health_recovers(self):
        exchange_bad = MockExchange(should_fail=True)
        for _ in range(3):
            await self.engine.check_exchange_health(exchange_bad)
        exchange_good = MockExchange(should_fail=False)
        await self.engine.check_exchange_health(exchange_good)
        assert self.engine.exchange_healthy is True

    # --- Slippage ---

    def test_record_slippage(self):
        record = self.engine.record_slippage("BTC/USDT", 50000.0, 50005.0)
        assert record.slippage_bps == pytest.approx(1.0, abs=0.1)

    def test_record_slippage_zero_expected(self):
        record = self.engine.record_slippage("BTC/USDT", 0.0, 100.0)
        assert record.slippage_bps == 0.0

    def test_excessive_slippage_warning(self):
        self.engine.record_slippage("BTC/USDT", 100.0, 100.2)
        # 20 bps > 10 bps threshold — logged as warning

    def test_get_avg_slippage(self):
        self.engine.record_slippage("BTC/USDT", 100.0, 100.01)
        self.engine.record_slippage("BTC/USDT", 100.0, 100.02)
        avg = self.engine.get_avg_slippage_bps()
        assert avg > 0

    def test_get_avg_slippage_filtered(self):
        self.engine.record_slippage("BTC/USDT", 100.0, 100.01)
        self.engine.record_slippage("ETH/USDT", 100.0, 100.05)
        avg_btc = self.engine.get_avg_slippage_bps(symbol="BTC/USDT")
        avg_eth = self.engine.get_avg_slippage_bps(symbol="ETH/USDT")
        assert avg_btc < avg_eth

    def test_slippage_acceptable(self):
        self.engine.record_slippage("BTC/USDT", 100.0, 100.005)
        assert self.engine.slippage_acceptable() is True

    def test_slippage_not_acceptable(self):
        for _ in range(5):
            self.engine.record_slippage("BTC/USDT", 100.0, 100.20)
        assert self.engine.slippage_acceptable() is False

    # --- Regime shift ---

    def test_record_entry_regime(self):
        self.engine.record_entry_regime("BTC/USDT", Regime.TRENDING)
        assert self.engine._regime_at_entry["BTC/USDT"] == Regime.TRENDING

    def test_clear_entry_regime(self):
        self.engine.record_entry_regime("BTC/USDT", Regime.TRENDING)
        self.engine.clear_entry_regime("BTC/USDT")
        assert "BTC/USDT" not in self.engine._regime_at_entry

    def test_no_regime_shift(self):
        self.engine.record_entry_regime("BTC/USDT", Regime.TRENDING)
        event = self.engine.check_regime_shift("BTC/USDT", Regime.TRENDING)
        assert event is None

    def test_regime_shift_detected(self):
        self.engine.record_entry_regime("BTC/USDT", Regime.TRENDING)
        event = self.engine.check_regime_shift("BTC/USDT", Regime.HIGH_VOL)
        assert event is not None
        assert event.old_regime == Regime.TRENDING
        assert event.new_regime == Regime.HIGH_VOL

    def test_regime_shift_no_entry(self):
        event = self.engine.check_regime_shift("BTC/USDT", Regime.TRENDING)
        assert event is None

    # --- Full cycle ---

    @pytest.mark.asyncio
    async def test_run_monitoring_cycle(self):
        exchange = MockExchange()
        ms = MagicMock(spec=MarketState)
        ms.regime_label = Regime.TRENDING
        ps = MagicMock(spec=PositionState)

        result = await self.engine.run_monitoring_cycle(
            exchange, {"BTC/USDT": ms}, {"BTC/USDT": ps}
        )
        assert result["exchange_healthy"] is True
        assert "regime_shifts" in result
        assert "avg_slippage_bps" in result

    # --- Accessors ---

    def test_get_health(self):
        health = self.engine.get_health()
        assert isinstance(health, ExchangeHealthStatus)

    def test_get_slippage_history(self):
        assert self.engine.get_slippage_history() == []
        self.engine.record_slippage("BTC/USDT", 100, 100.01)
        assert len(self.engine.get_slippage_history()) == 1

    def test_get_regime_shift_events(self):
        assert self.engine.get_regime_shift_events() == []
