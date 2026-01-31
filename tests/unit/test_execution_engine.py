"""Unit tests for ExecutionEngine with mocked exchange."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from trade_bot.core.models import (
    TradingAction, Order, Trade, PositionState,
    OrderType, OrderStatus, Side,
)
from trade_bot.core.enums import Action, Regime
from trade_bot.execution.engine import (
    ExchangeConnector,
    CCXTExchangeConnector,
    PositionMonitor,
    ExecutionEngine,
)


class MockExchangeConnector(ExchangeConnector):
    """Mock exchange connector for testing."""

    def __init__(self):
        self.orders = {}
        self._order_counter = 0
        self._balance = {"USDT": 10000.0}
        self._positions = []

    async def create_order(self, symbol, order_type, side, amount, price=None, stop_price=None):
        self._order_counter += 1
        order_id = f"mock-{self._order_counter}"
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=amount,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.FILLED,
            filled_quantity=amount,
            average_fill_price=price or 100.0,
        )
        self.orders[order_id] = order
        return order

    async def cancel_order(self, order_id, symbol):
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False

    async def get_order(self, order_id, symbol):
        return self.orders.get(order_id)

    async def get_balance(self):
        return self._balance

    async def get_positions(self):
        return self._positions

    async def close(self):
        pass


@pytest.fixture
def mock_exchange():
    return MockExchangeConnector()


@pytest.fixture
def position_monitor(mock_exchange):
    return PositionMonitor(mock_exchange)


@pytest.fixture
def execution_engine(mock_exchange, position_monitor):
    return ExecutionEngine(mock_exchange, position_monitor)


class TestExecutionEngine:
    @pytest.mark.asyncio
    async def test_execute_hold_returns_none(self, execution_engine):
        action = TradingAction(
            action=Action.HOLD, size=0, expected_return=0,
            expected_risk=0, confidence=1.0,
        )
        result = await execution_engine.execute_action(action, "BTC/USDT", 50000.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_buy_creates_order(self, execution_engine):
        action = TradingAction(
            action=Action.BUY, size=0.1, stop_loss=49000,
            take_profit=52000, expected_return=2000,
            expected_risk=1000, confidence=0.8,
        )
        order = await execution_engine.execute_action(action, "BTC/USDT", 50000.0)
        assert order is not None
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_execute_sell_creates_order(self, execution_engine):
        action = TradingAction(
            action=Action.SELL, size=0.1, stop_loss=51000,
            take_profit=48000, expected_return=2000,
            expected_risk=1000, confidence=0.8,
        )
        order = await execution_engine.execute_action(action, "BTC/USDT", 50000.0)
        assert order is not None

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, execution_engine, mock_exchange):
        # Create some orders
        action = TradingAction(
            action=Action.BUY, size=0.1, stop_loss=49000,
            take_profit=52000, expected_return=2000,
            expected_risk=1000, confidence=0.8,
        )
        await execution_engine.execute_action(action, "BTC/USDT", 50000.0)
        await execution_engine.cancel_all_orders()

    def test_get_pending_orders(self, execution_engine):
        orders = execution_engine.get_pending_orders()
        assert isinstance(orders, list)

    def test_get_completed_orders(self, execution_engine):
        orders = execution_engine.get_completed_orders()
        assert isinstance(orders, list)

    def test_get_trades(self, execution_engine):
        trades = execution_engine.get_trades()
        assert isinstance(trades, list)


class TestPositionMonitor:
    def test_get_position_none(self, position_monitor):
        assert position_monitor.get_position("BTC/USDT") is None

    def test_get_all_positions(self, position_monitor):
        positions = position_monitor.get_all_positions()
        assert isinstance(positions, dict)

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, position_monitor):
        await position_monitor.start_monitoring(["BTC/USDT"])
        assert position_monitor._running is True
        await position_monitor.stop_monitoring()
        assert position_monitor._running is False
