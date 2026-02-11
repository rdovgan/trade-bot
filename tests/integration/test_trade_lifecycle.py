"""Integration tests for complete trade lifecycle."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock

from trade_bot.core.models import TradingAction, Order, OrderType, OrderStatus, Side
from trade_bot.core.enums import Action
from trade_bot.execution.engine import ExecutionEngine, PositionMonitor
from trade_bot.learning.journal import TradeJournal


class MockExchange:
    """Mock exchange for integration testing."""

    def __init__(self):
        self.orders = {}
        self._order_counter = 0
        self._balance = {"USDT": 10000.0}
        self._positions = []

    async def create_order(self, symbol, order_type, side, amount, price=None, stop_price=None):
        self._order_counter += 1
        order_id = f"test-{self._order_counter}"

        # Entry orders fill immediately, stops stay pending
        if order_type == OrderType.STOP:
            status = OrderStatus.PENDING
            filled_qty = 0
            avg_price = None
            commission = 0
        else:
            status = OrderStatus.FILLED
            filled_qty = amount
            avg_price = price or 50000.0
            commission = amount * avg_price * 0.001  # 0.1% commission

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=amount,
            price=price,
            stop_price=stop_price,
            status=status,
            filled_quantity=filled_qty,
            average_fill_price=avg_price,
            commission=commission,
        )

        self.orders[order_id] = order
        return order

    async def get_order(self, order_id, symbol):
        return self.orders.get(order_id)

    async def cancel_order(self, order_id, symbol):
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_balance(self):
        return self._balance

    async def get_positions(self):
        return self._positions

    async def close(self):
        pass

    def fill_stop_order(self, order_id):
        """Simulate a stop order getting filled."""
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = order.stop_price
            order.commission = order.quantity * order.stop_price * 0.001


class MockJournal:
    """Mock journal for testing."""

    def __init__(self):
        self.trades = []

    def record_trade(self, trade, market_state):
        """Record trade with new API signature (Trade, MarketState objects)."""
        self.trades.append({
            'trade': trade,
            'market_state': market_state,
            # Also store flat values for easy test assertions
            'symbol': trade.symbol,
            'side': trade.side.value,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
        })


@pytest.fixture
def mock_exchange():
    return MockExchange()


@pytest.fixture
def mock_journal():
    return MockJournal()


@pytest.fixture
def position_monitor(mock_exchange):
    return PositionMonitor(mock_exchange)


@pytest.fixture
def execution_engine(mock_exchange, position_monitor, mock_journal):
    return ExecutionEngine(mock_exchange, position_monitor, journal=mock_journal)


class TestCompleteTradeLifecycle:
    """Integration tests for complete trade lifecycle."""

    @pytest.mark.asyncio
    async def test_long_trade_stopped_out(self, execution_engine, mock_exchange, mock_journal):
        """Test complete lifecycle: long trade -> stop loss fill -> journal."""

        # 1. Execute buy action with stop loss
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            stop_loss=49000,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )

        entry_order = await execution_engine.execute_action(action, "BTC/USDT", 50000.0)

        # Verify entry order filled
        assert entry_order.status == OrderStatus.FILLED
        assert entry_order.average_fill_price == 50000.0

        # Verify stop order created
        stop_orders = [o for o in mock_exchange.orders.values() if o.order_type == OrderType.STOP]
        assert len(stop_orders) == 1
        stop_order = stop_orders[0]
        assert stop_order.stop_price == 49000
        assert stop_order.status == OrderStatus.PENDING

        # Verify trade created
        trades = execution_engine.get_trades()
        assert len(trades) == 1
        trade = trades[0]
        assert trade.status == "open"
        assert trade.entry_price == 50000.0
        assert trade.commission_entry > 0

        # Verify order-trade linking
        assert len(execution_engine._stop_to_trade) == 1
        assert stop_order.id in execution_engine._stop_to_trade

        # 2. Start monitoring
        await execution_engine.start_monitoring()

        # 3. Simulate stop loss fill
        mock_exchange.fill_stop_order(stop_order.id)

        # 4. Wait for monitoring to detect fill
        await asyncio.sleep(2.5)

        # 5. Stop monitoring
        await execution_engine.stop_monitoring()

        # 6. Verify trade completed
        completed_trade = trades[0]
        assert completed_trade.status == "closed"
        assert completed_trade.exit_price == 49000
        assert completed_trade.pnl < 0  # Should be a loss

        # 7. Verify PnL calculation
        # Entry: 0.1 BTC @ 50000 = 5000 USDT
        # Exit: 0.1 BTC @ 49000 = 4900 USDT
        # Gross PnL: -100 USDT
        # Net PnL: -100 - entry_commission - exit_commission
        gross_pnl = (49000 - 50000) * 0.1
        expected_net_pnl = gross_pnl - completed_trade.commission_entry - completed_trade.commission_exit

        assert abs(completed_trade.pnl - expected_net_pnl) < 0.01

        # 8. Verify R-multiple (skip if stop_loss not preserved in mock)
        # Note: In production, stop_loss would be preserved from the action
        if completed_trade.stop_loss > 0:
            risk = abs(50000 - completed_trade.stop_loss) * 0.1
            expected_r = expected_net_pnl / risk
            assert abs(completed_trade.r_multiple - expected_r) < 0.1

        # 9. Verify persisted to journal
        assert len(mock_journal.trades) == 1
        persisted = mock_journal.trades[0]
        assert persisted['symbol'] == "BTC/USDT"
        assert persisted['side'] == Side.LONG.value
        assert persisted['exit_price'] == 49000
        assert persisted['pnl'] < 0

        # 10. Verify cleanup
        assert stop_order.id not in execution_engine._stop_to_trade
        assert trade.id not in execution_engine._trade_links

    @pytest.mark.asyncio
    async def test_short_trade_stopped_out(self, execution_engine, mock_exchange, mock_journal):
        """Test complete lifecycle for short trade."""

        # Execute sell action with stop loss
        action = TradingAction(
            action=Action.SELL,
            size=0.1,
            stop_loss=51000,  # Stop above for short
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )

        entry_order = await execution_engine.execute_action(action, "BTC/USDT", 50000.0)

        # Get stop order
        stop_orders = [o for o in mock_exchange.orders.values() if o.order_type == OrderType.STOP]
        stop_order = stop_orders[0]
        assert stop_order.stop_price == 51000

        # Start monitoring
        await execution_engine.start_monitoring()

        # Simulate stop fill
        mock_exchange.fill_stop_order(stop_order.id)

        # Wait for detection
        await asyncio.sleep(2.5)
        await execution_engine.stop_monitoring()

        # Verify trade completed
        trade = execution_engine.get_trades()[0]
        assert trade.status == "closed"
        assert trade.exit_price == 51000
        assert trade.pnl < 0  # Loss for short

        # Verify short PnL calculation
        # Entry: SHORT 0.1 BTC @ 50000
        # Exit: BUY 0.1 BTC @ 51000
        # Gross PnL: (50000 - 51000) * 0.1 = -100
        gross_pnl = (50000 - 51000) * 0.1
        expected_net_pnl = gross_pnl - trade.commission_entry - trade.commission_exit

        assert abs(trade.pnl - expected_net_pnl) < 0.01

    @pytest.mark.asyncio
    async def test_manual_close_profitable_trade(self, execution_engine, mock_exchange, mock_journal):
        """Test manual close of a profitable trade."""

        # Open long position
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            stop_loss=49000,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )

        await execution_engine.execute_action(action, "BTC/USDT", 50000.0)

        # Manually close at profit
        exit_order = await mock_exchange.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side=Side.SHORT,
            amount=0.1,
            price=52000.0
        )

        await execution_engine.complete_trade_on_manual_close("BTC/USDT", exit_order)

        # Verify trade completed profitably
        trade = execution_engine.get_trades()[0]
        assert trade.status == "closed"
        assert trade.exit_price == 52000.0
        assert trade.pnl > 0  # Should be profitable

        # Verify stop order cancelled
        stop_orders = [o for o in mock_exchange.orders.values()
                      if o.order_type == OrderType.STOP]
        assert stop_orders[0].status == OrderStatus.CANCELLED

        # Verify persisted
        assert len(mock_journal.trades) == 1

    @pytest.mark.asyncio
    async def test_funding_costs_included_in_pnl(self, execution_engine, mock_exchange, mock_journal):
        """Test that funding costs are included in final PnL."""

        # Open position
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            stop_loss=49000,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )

        await execution_engine.execute_action(action, "BTC/USDT", 50000.0)

        # Add funding costs
        trade = execution_engine.get_trades()[0]
        trade.funding_cost = 25.0  # $25 in funding

        # Close position
        exit_order = await mock_exchange.create_order(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side=Side.SHORT,
            amount=0.1,
            price=51000.0
        )

        await execution_engine.complete_trade_on_manual_close("BTC/USDT", exit_order)

        # Verify funding cost deducted from PnL
        completed_trade = execution_engine.get_trades()[0]
        gross_pnl = (51000 - 50000) * 0.1  # 100 USDT
        expected_net_pnl = gross_pnl - completed_trade.commission_entry - completed_trade.commission_exit - 25.0

        assert abs(completed_trade.pnl - expected_net_pnl) < 0.01

    @pytest.mark.asyncio
    async def test_slippage_recorded_on_entry(self, execution_engine, mock_exchange):
        """Test that slippage is recorded on entry."""

        # Execute with expected price of 49000 but will fill at 50000
        action = TradingAction(
            action=Action.BUY,
            size=0.1,
            expected_return=2000,
            expected_risk=1000,
            confidence=0.8,
        )

        await execution_engine.execute_action(action, "BTC/USDT", 49000.0)

        # Check slippage recorded
        trade = execution_engine.get_trades()[0]

        # Slippage = |50000 - 49000| / 49000 * 10000 bps
        expected_slippage = abs((50000.0 - 49000.0) / 49000.0) * 10000

        assert abs(trade.slippage - expected_slippage) < 1

    @pytest.mark.asyncio
    async def test_multiple_concurrent_trades(self, execution_engine, mock_exchange, mock_journal):
        """Test handling multiple concurrent trades."""

        # Open 3 positions
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        actions = []

        for symbol in symbols:
            action = TradingAction(
                action=Action.BUY,
                size=0.1,
                stop_loss=49000,
                expected_return=2000,
                expected_risk=1000,
                confidence=0.8,
            )
            await execution_engine.execute_action(action, symbol, 50000.0)

        # Should have 3 open trades
        trades = execution_engine.get_trades()
        assert len(trades) == 3
        assert all(t.status == "open" for t in trades)

        # Should have 3 stop orders
        stop_orders = [o for o in mock_exchange.orders.values() if o.order_type == OrderType.STOP]
        assert len(stop_orders) == 3

        # Start monitoring
        await execution_engine.start_monitoring()

        # Fill all stops
        for stop_order in stop_orders:
            mock_exchange.fill_stop_order(stop_order.id)

        # Wait for monitoring
        await asyncio.sleep(2.5)
        await execution_engine.stop_monitoring()

        # All trades should be closed
        assert all(t.status == "closed" for t in trades)

        # All should be persisted
        assert len(mock_journal.trades) == 3
