"""Unit tests for ExecutionEngine with mocked exchange."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

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
        self._order_status_override = {}  # For testing order status changes

    async def create_order(self, symbol, order_type, side, amount, price=None, stop_price=None):
        self._order_counter += 1
        order_id = f"mock-{self._order_counter}"

        # Stop orders start as pending
        initial_status = OrderStatus.PENDING if order_type == OrderType.STOP else OrderStatus.FILLED

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=amount,
            price=price,
            stop_price=stop_price,
            status=initial_status,
            filled_quantity=amount if initial_status == OrderStatus.FILLED else 0,
            average_fill_price=price or 50000.0 if initial_status == OrderStatus.FILLED else None,
            commission=amount * 0.001 if initial_status == OrderStatus.FILLED else 0,  # 0.1% commission
        )
        self.orders[order_id] = order
        return order

    async def cancel_order(self, order_id, symbol):
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_order(self, order_id, symbol):
        order = self.orders.get(order_id)
        if not order:
            return None

        # Apply status override if set (for testing)
        if order_id in self._order_status_override:
            order.status = self._order_status_override[order_id]
            if order.status == OrderStatus.FILLED and order.filled_quantity == 0:
                order.filled_quantity = order.quantity
                order.average_fill_price = order.stop_price or order.price or 50000.0
                order.commission = order.quantity * 0.001

        return order

    async def get_balance(self):
        return self._balance

    async def get_positions(self):
        return self._positions

    async def close(self):
        pass

    def simulate_stop_fill(self, order_id):
        """Simulate a stop loss order getting filled."""
        self._order_status_override[order_id] = OrderStatus.FILLED


class MockJournal:
    """Mock journal for testing."""

    def __init__(self):
        self.recorded_trades = []

    def record_trade(self, trade, market_state):
        """Record trade with new API signature (Trade, MarketState objects)."""
        self.recorded_trades.append({
            'trade': trade,
            'market_state': market_state,
        })


@pytest.fixture
def mock_exchange():
    return MockExchangeConnector()


@pytest.fixture
def mock_journal():
    return MockJournal()


@pytest.fixture
def position_monitor(mock_exchange):
    return PositionMonitor(mock_exchange)


@pytest.fixture
def execution_engine(mock_exchange, position_monitor, mock_journal):
    return ExecutionEngine(mock_exchange, position_monitor, journal=mock_journal)


class TestStopLossMonitoring:
    """Tests for stop-loss monitoring and order lifecycle (Phase 1)."""

    @pytest.mark.asyncio
    async def test_stop_order_created_and_linked(self, execution_engine, mock_exchange):
        """Test that stop-loss orders are created and linked to trades."""
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            stop_loss=49000,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )

        # Execute action with stop loss
        order = await execution_engine.execute_action(action, "BTC/USDT", 50000.0)

        # Should have created 2 orders: entry + stop
        assert len(mock_exchange.orders) == 2

        # Find stop order
        stop_orders = [o for o in mock_exchange.orders.values() if o.order_type == OrderType.STOP]
        assert len(stop_orders) == 1

        stop_order = stop_orders[0]
        assert stop_order.stop_price == 49000

        # Should be linked in tracking dictionaries
        assert len(execution_engine._stop_to_trade) > 0

    @pytest.mark.asyncio
    async def test_stop_loss_fill_completes_trade(self, execution_engine, mock_exchange, mock_journal):
        """Test that stop-loss fills are detected and complete the trade."""
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            stop_loss=49000,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )

        # Execute action
        order = await execution_engine.execute_action(action, "BTC/USDT", 50000.0)

        # Get the stop order
        stop_orders = [o for o in mock_exchange.orders.values() if o.order_type == OrderType.STOP]
        stop_order = stop_orders[0]

        # Start monitoring
        await execution_engine.start_monitoring()

        # Simulate stop loss getting filled
        mock_exchange.simulate_stop_fill(stop_order.id)

        # Wait for monitoring to detect the fill
        await asyncio.sleep(2.5)

        # Stop monitoring
        await execution_engine.stop_monitoring()

        # Trade should be completed and persisted to journal
        assert len(mock_journal.recorded_trades) > 0
        recorded = mock_journal.recorded_trades[0]
        trade = recorded['trade']
        assert trade.exit_price == stop_order.stop_price
        assert trade.pnl < 0  # Should be a loss

    @pytest.mark.asyncio
    async def test_order_monitoring_task_starts_and_stops(self, execution_engine):
        """Test that order monitoring task can be started and stopped."""
        await execution_engine.start_monitoring()
        assert execution_engine._running is True
        assert execution_engine._monitor_task is not None

        await execution_engine.stop_monitoring()
        assert execution_engine._running is False


class TestTradePnLTracking:
    """Tests for trade PnL tracking and completion (Phase 2)."""

    @pytest.mark.asyncio
    async def test_trade_creation_with_costs(self, execution_engine, mock_exchange):
        """Test that trades are created with commission tracking."""
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            stop_loss=49000,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )

        await execution_engine.execute_action(action, "BTC/USDT", 50000.0)

        trades = execution_engine.get_trades()
        assert len(trades) == 1

        trade = trades[0]
        assert trade.status == "open"
        assert trade.commission_entry > 0
        assert trade.commission_exit == 0
        assert trade.funding_cost == 0

    @pytest.mark.asyncio
    async def test_manual_close_completes_trade(self, execution_engine, mock_exchange, mock_journal):
        """Test that manual closes complete trades with PnL."""
        # Create a position
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            stop_loss=49000,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )
        await execution_engine.execute_action(action, "BTC/USDT", 50000.0)

        # Create exit order
        exit_order = await mock_exchange.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side=Side.SHORT,
            amount=0.1,
            price=51000.0
        )

        # Complete trade manually
        await execution_engine.complete_trade_on_manual_close("BTC/USDT", exit_order)

        # Check trade was completed
        trades = execution_engine.get_trades()
        assert trades[0].status == "closed"
        assert trades[0].exit_price == 51000.0
        assert trades[0].pnl > 0  # Should be a profit

        # Check persisted to journal
        assert len(mock_journal.recorded_trades) == 1

    @pytest.mark.asyncio
    async def test_pnl_calculation_includes_all_costs(self, execution_engine, mock_exchange):
        """Test that PnL includes commissions and funding costs."""
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            stop_loss=49000,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )
        await execution_engine.execute_action(action, "BTC/USDT", 50000.0)

        # Get trade and add funding cost
        trade = execution_engine.get_trades()[0]
        trade.funding_cost = 10.0

        # Create exit order
        exit_order = await mock_exchange.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side=Side.SHORT,
            amount=0.1,
            price=51000.0
        )

        # Complete trade
        await execution_engine.complete_trade_on_manual_close("BTC/USDT", exit_order)

        # PnL should account for all costs
        completed_trade = execution_engine.get_trades()[0]
        gross_pnl = (51000.0 - 50000.0) * 0.1  # 100.0
        expected_net = gross_pnl - completed_trade.commission_entry - completed_trade.commission_exit - 10.0

        assert abs(completed_trade.pnl - expected_net) < 0.01

    @pytest.mark.asyncio
    async def test_r_multiple_calculation(self, execution_engine, mock_exchange):
        """Test R-multiple calculation."""
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            stop_loss=49000,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )
        await execution_engine.execute_action(action, "BTC/USDT", 50000.0)

        # Manually set stop_loss on trade (since mock doesn't preserve it perfectly)
        trade = execution_engine.get_trades()[0]
        trade.stop_loss = 49000

        exit_order = await mock_exchange.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side=Side.SHORT,
            amount=0.1,
            price=51000.0
        )

        await execution_engine.complete_trade_on_manual_close("BTC/USDT", exit_order)

        trade = execution_engine.get_trades()[0]
        # R-multiple should be (net_pnl / risk)
        # Risk = (50000 - 49000) * 0.1 = 100
        assert trade.r_multiple > 0


class TestPositionReconciliation:
    """Tests for position reconciliation (Phase 6)."""

    @pytest.mark.asyncio
    async def test_position_reconciliation_no_discrepancies(self, execution_engine, mock_exchange, position_monitor):
        """Test reconciliation when positions match."""
        # Set up matching positions
        mock_exchange._positions = [{
            'symbol': 'BTC/USDT',
            'size': 0.1,
            'entry_price': 50000.0,
            'unrealized_pnl': 100.0
        }]

        # Update internal positions
        await position_monitor._update_positions()

        # Reconcile
        discrepancies = await execution_engine.reconcile_positions()

        assert len(discrepancies) == 0

    @pytest.mark.asyncio
    async def test_position_reconciliation_detects_mismatch(self, execution_engine, mock_exchange, position_monitor):
        """Test reconciliation detects size mismatches."""
        # Set exchange position
        mock_exchange._positions = [{
            'symbol': 'BTC/USDT',
            'size': 0.2,
            'entry_price': 50000.0,
            'unrealized_pnl': 100.0
        }]

        # Set different internal position
        position_monitor._positions['BTC/USDT'] = PositionState()
        position_monitor._positions['BTC/USDT'].position_size = 0.1
        position_monitor._positions['BTC/USDT'].current_side = Side.LONG

        # Reconcile
        discrepancies = await execution_engine.reconcile_positions()

        assert 'BTC/USDT' in discrepancies
        assert 'Size mismatch' in discrepancies['BTC/USDT']

        # Should auto-correct to match exchange
        assert position_monitor._positions['BTC/USDT'].position_size == 0.2

    @pytest.mark.asyncio
    async def test_position_reconciliation_detects_ghost_position(self, execution_engine, mock_exchange, position_monitor):
        """Test reconciliation detects ghost positions."""
        # No position on exchange
        mock_exchange._positions = []

        # But internal thinks there's a position
        position_monitor._positions['BTC/USDT'] = PositionState()
        position_monitor._positions['BTC/USDT'].position_size = 0.1
        position_monitor._positions['BTC/USDT'].current_side = Side.LONG

        # Reconcile
        discrepancies = await execution_engine.reconcile_positions()

        assert 'BTC/USDT' in discrepancies
        assert 'Ghost position' in discrepancies['BTC/USDT']

        # Should clear the ghost position
        assert position_monitor._positions['BTC/USDT'].position_size == 0
        assert position_monitor._positions['BTC/USDT'].current_side == Side.FLAT


class TestSlippageTracking:
    """Tests for slippage calculation and tracking (Phase 4)."""

    @pytest.mark.asyncio
    async def test_slippage_calculation(self, execution_engine, mock_exchange):
        """Test that slippage is calculated correctly."""
        # Mock will fill at 50000 when we expect 49000
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )

        await execution_engine.execute_action(action, "BTC/USDT", 49000.0)

        # Check trade has slippage
        trades = execution_engine.get_trades()
        assert len(trades) == 1

        # Slippage should be: |50000 - 49000| / 49000 * 10000 bps
        expected_slippage = abs((50000.0 - 49000.0) / 49000.0) * 10000
        assert abs(trades[0].slippage - expected_slippage) < 1


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
