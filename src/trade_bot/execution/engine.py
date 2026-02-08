"""Execution engine for order management and position monitoring."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator
import asyncio
import logging
from datetime import datetime, timedelta
import uuid

import ccxt.async_support as ccxt

import pandas as pd

from ..core.models import (
    MarketState, PositionState, AccountState, Order, Trade,
    TradingAction, OrderType, OrderStatus, Side
)
from ..core.enums import Action, Regime
from ..core.state_lock import state_manager

logger = logging.getLogger(__name__)


class PositionFetchError(Exception):
    """Raised when positions cannot be fetched from exchange."""
    pass


class ExchangeConnector(ABC):
    """Abstract base class for exchange connectors."""
    
    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: Side,
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Create an order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order status."""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Get open positions."""
        pass


class CCXTExchangeConnector(ExchangeConnector):
    """CCXT-based exchange connector."""
    
    def __init__(self, exchange_name: str, config: Optional[Dict] = None):
        """Initialize CCXT exchange connector."""
        self.exchange_name = exchange_name
        self.config = config or {}
        
        # Initialize exchange
        is_sandbox = self.config.get('sandbox', False)
        # Binance uses 'future', Bybit/OKX use 'linear'
        type_map = {'binance': 'future', 'binanceusdm': 'future'}
        default_type = type_map.get(exchange_name, 'linear')
        ccxt_keys = {k: v for k, v in self.config.items() if k != 'sandbox'}
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {'defaultType': default_type},
            **ccxt_keys
        })
        if is_sandbox:
            from ..data.connector import _apply_sandbox_urls
            _apply_sandbox_urls(self.exchange, exchange_name)

        logger.info(f"Initialized CCXT exchange connector for {exchange_name} (sandbox={is_sandbox}, type={default_type})")
    
    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: Side,
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """Create an order."""
        try:
            # Map order types
            ccxt_type = self._map_order_type(order_type)
            ccxt_side = self._map_side(side)

            # Clamp amount to exchange market limits and precision
            amount = await self._clamp_amount(symbol, amount)
            if amount <= 0:
                raise ValueError(f"Order amount for {symbol} is zero after applying exchange limits")

            # Create order
            if order_type == OrderType.MARKET:
                result = await self.exchange.create_market_order(symbol, ccxt_side, amount)
            elif order_type == OrderType.LIMIT:
                result = await self.exchange.create_limit_order(symbol, ccxt_side, amount, price)
            elif order_type == OrderType.STOP:
                # Create a Stop Market order (stops are always reverse side to close position)
                params = {}
                # Bybit requires triggerDirection
                if self.exchange_name == 'bybit':
                    params['triggerDirection'] = 'descending' if ccxt_side == 'sell' else 'ascending'
                
                # Use 'stop_market' type for safety - ensures immediate fill when trigger hits
                result = await self.exchange.create_order(
                    symbol=symbol, 
                    type='stop_market', 
                    side=ccxt_side, 
                    amount=amount, 
                    price=None,
                    params={
                        'stopPrice': stop_price,
                        **params
                    }
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Convert to our Order model
            order = Order(
                id=result.get('id', ''),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=amount,
                price=price,
                stop_price=stop_price,
                status=self._map_order_status(result.get('status', 'open')),
                created_at=datetime.fromtimestamp(result['timestamp'] / 1000) if result.get('timestamp') else datetime.now(),
                filled_quantity=result.get('filled') or 0,
                average_fill_price=result.get('average'),
                commission=float(result.get('fee', {}).get('cost', 0) or 0) if result.get('fee') else 0,
            )

            logger.info(f"Created order: {order.id} {order.side.value} {order.quantity} {symbol}")
            return order

        except Exception as e:
            error_str = str(e)
            # Handle insufficient margin as a normal operational condition
            if '-2019' in error_str or 'Margin is insufficient' in error_str or 'insufficient' in error_str.lower():
                logger.warning(f"Insufficient margin to create order for {symbol}: {e}")
                return None
            # Handle other common operational errors
            elif any(code in error_str for code in ['-1121', '-2010', '-4131']):  # Invalid symbol, new order rejected, balance insufficient
                logger.warning(f"Order rejected for {symbol}: {e}")
                return None
            else:
                logger.error(f"Error creating order: {e}")
                raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order status."""
        try:
            result = await self.exchange.fetch_order(order_id, symbol, params={'acknowledged': True})
            
            raw_side = result.get('side') or 'buy'
            side_map = {'buy': Side.LONG, 'sell': Side.SHORT}
            mapped_side = side_map.get(raw_side.lower(), Side.LONG)

            raw_type = result.get('type') or 'market'
            try:
                mapped_type = OrderType(raw_type.lower())
            except ValueError:
                mapped_type = OrderType.MARKET

            order = Order(
                id=result.get('id', ''),
                symbol=symbol,
                side=mapped_side,
                order_type=mapped_type,
                quantity=result.get('amount') or 0,
                price=result.get('price'),
                stop_price=result.get('stopPrice'),
                status=self._map_order_status(result.get('status')),
                created_at=datetime.fromtimestamp(result['timestamp'] / 1000) if result.get('timestamp') else datetime.now(),
                filled_quantity=result.get('filled') or 0,
                average_fill_price=result.get('average'),
                commission=float(result.get('fee', {}).get('cost', 0) or 0) if result.get('fee') else 0,
            )
            
            return order

        except Exception as e:
            # Common case: order doesn't exist (filled/cancelled/expired)
            # Log at debug level to reduce noise
            error_str = str(e)
            if any(phrase in error_str for phrase in ['does not exist', 'not found', '-2013']):
                logger.debug(f"Order {order_id} not found on exchange: {e}")
            else:
                logger.error(f"Error fetching order {order_id}: {e}")
            return None
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance (total = free + used margin)."""
        try:
            balance = await self.exchange.fetch_balance()
            # Use 'total' not 'free' — on futures, margin-in-use is excluded from 'free'
            total = balance.get('total', {})
            result = {
                asset: float(amount)
                for asset, amount in total.items()
                if amount and float(amount) > 0
            }
            if not result:
                # Fallback to 'free' if 'total' is empty
                free = balance.get('free', {})
                result = {
                    asset: float(amount)
                    for asset, amount in free.items()
                    if amount and float(amount) > 0
                }
            logger.info(f"Balance fetched: {result}")
            return result
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}
    
    async def get_positions(self) -> List[Dict]:
        """Get open positions from exchange with error handling.

        Raises:
            PositionFetchError: If positions cannot be fetched due to network/exchange errors.
        """
        try:
            if hasattr(self.exchange, 'fetch_positions'):
                positions = await self.exchange.fetch_positions()
                if positions is None:
                    raise PositionFetchError("Exchange returned None for positions")
                return [
                    {
                        'symbol': pos.get('symbol') or '',
                        'side': pos.get('side') or '',
                        'size': float(pos.get('contracts') or 0),
                        'entry_price': float(pos.get('entryPrice') or 0),
                        'unrealized_pnl': float(pos.get('unrealizedPnl') or 0),
                    }
                    for pos in positions
                    if float(pos.get('contracts') or 0) != 0
                ]
            else:
                logger.warning("Exchange does not support fetch_positions")
                return []
        except ccxt.NetworkError as e:
            raise PositionFetchError(f"Network error fetching positions: {e}")
        except ccxt.ExchangeError as e:
            raise PositionFetchError(f"Exchange error fetching positions: {e}")
        except PositionFetchError:
            raise
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise PositionFetchError(f"Unexpected error fetching positions: {e}")
    
    async def _clamp_amount(self, symbol: str, amount: float) -> float:
        """Clamp order amount to exchange min/max and precision."""
        try:
            if not self.exchange.markets:
                await self.exchange.load_markets()
            market = self.exchange.markets.get(symbol)
            if not market:
                return amount

            limits = market.get('limits', {}).get('amount', {})
            max_qty = limits.get('max')
            min_qty = limits.get('min')

            if max_qty and amount > max_qty:
                logger.warning(f"{symbol}: clamping amount {amount} -> {max_qty} (exchange max)")
                amount = max_qty
            if min_qty and amount < min_qty:
                logger.warning(f"{symbol}: amount {amount} below exchange min {min_qty}")
                return 0

            # Apply precision
            amount = self.exchange.amount_to_precision(symbol, amount)
            return float(amount)
        except Exception as e:
            logger.warning(f"Could not clamp amount for {symbol}: {e}")
            return amount

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map order type to CCXT format."""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'stop',
            OrderType.STOP_LIMIT: 'stop_limit',
        }
        return mapping.get(order_type, 'market')
    
    def _map_side(self, side: Side) -> str:
        """Map side to CCXT format."""
        if side == Side.LONG:
            return 'buy'
        elif side == Side.SHORT:
            return 'sell'
        return side.value.lower()
    
    def _map_order_status(self, ccxt_status: str) -> OrderStatus:
        """Map CCXT order status to our format."""
        mapping = {
            'open': OrderStatus.PENDING,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'cancelled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
        }
        if not ccxt_status:
            return OrderStatus.PENDING
        return mapping.get(ccxt_status.lower(), OrderStatus.PENDING)
    
    async def close(self):
        """Close exchange connection."""
        await self.exchange.close()
        logger.info(f"Closed connection to {self.exchange_name}")


class PositionMonitor:
    """Monitors open positions and manages risk."""

    def __init__(self, exchange_connector: ExchangeConnector):
        """Initialize position monitor."""
        self.exchange = exchange_connector
        self._running = False
        self._positions: Dict[str, PositionState] = {}
        self._positions_stale = False  # Flag to indicate stale position data
        logger.info("Position monitor initialized")

    @property
    def positions_stale(self) -> bool:
        """Check if position data is stale due to fetch errors."""
        return self._positions_stale
    
    async def start_monitoring(self, symbols: List[str]):
        """Start monitoring positions for given symbols."""
        self._running = True
        self.symbols = symbols
        
        asyncio.create_task(self._monitor_loop())
        logger.info(f"Started monitoring positions for {len(symbols)} symbols")
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self._running = False
        logger.info("Stopped position monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Update positions
                await self._update_positions()
                
                # Check for risk events
                await self._check_risk_events()
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in position monitoring loop: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _update_positions(self):
        """Update position information with locking.

        On success, clears the stale flag. On error, sets stale flag but keeps existing data.
        """
        async with state_manager.lock_state("positions"):
            try:
                positions = await self.exchange.get_positions()

                # Track which symbols were seen from exchange
                seen_symbols = set()

                for pos_data in positions:
                    symbol = pos_data['symbol']
                    seen_symbols.add(symbol)

                    # Update position state
                    if symbol not in self._positions:
                        self._positions[symbol] = PositionState()

                    position = self._positions[symbol]
                    position.position_size = pos_data['size']
                    position.entry_price = pos_data['entry_price']
                    position.unrealized_pnl = pos_data['unrealized_pnl']

                    # Update side based on size
                    if pos_data['size'] > 0:
                        position.current_side = Side.LONG
                    elif pos_data['size'] < 0:
                        position.current_side = Side.SHORT
                    else:
                        position.current_side = Side.FLAT

                # Mark positions not on exchange as flat (position closed externally)
                for symbol in list(self._positions.keys()):
                    if symbol not in seen_symbols:
                        if self._positions[symbol].current_side != Side.FLAT:
                            logger.info(f"Position {symbol} no longer on exchange, marking as FLAT")
                            self._positions[symbol].current_side = Side.FLAT
                            self._positions[symbol].position_size = 0

                # Successfully updated - clear stale flag
                self._positions_stale = False

            except PositionFetchError as e:
                logger.error(f"Failed to fetch positions: {e}")
                # DON'T clear positions on error - keep stale data
                # Set flag to prevent trading with stale data
                self._positions_stale = True
            except Exception as e:
                logger.error(f"Error updating positions: {e}")
                self._positions_stale = True
    
    async def _check_risk_events(self):
        """Check for risk events that require action."""
        for symbol, position in self._positions.items():
            if position.current_side == Side.FLAT:
                continue
            
            # This would integrate with risk management
            # For now, just log significant P&L moves
            if abs(position.unrealized_pnl) > 100:  # $100 threshold
                logger.warning(
                    f"Significant P&L move for {symbol}: "
                    f"{position.unrealized_pnl:.2f}"
                )
    
    def get_position(self, symbol: str) -> Optional[PositionState]:
        """Get position state for a symbol (sync, no locking - use get_position_safe for concurrent access)."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, PositionState]:
        """Get all positions (sync, no locking - use get_all_positions_safe for concurrent access)."""
        return self._positions.copy()

    async def get_position_safe(self, symbol: str) -> Optional[PositionState]:
        """Get position with locking for concurrent access."""
        async with state_manager.lock_state("positions"):
            return self._positions.get(symbol)

    async def get_all_positions_safe(self) -> Dict[str, PositionState]:
        """Get all positions with locking for concurrent access."""
        async with state_manager.lock_state("positions"):
            return dict(self._positions)


class ExecutionEngine:
    """Main execution engine for order management."""
    
    def __init__(
        self,
        exchange_connector: ExchangeConnector,
        position_monitor: PositionMonitor,
        journal=None
    ):
        """Initialize execution engine."""
        self.exchange = exchange_connector
        self.position_monitor = position_monitor
        self.journal = journal

        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._completed_orders: List[Order] = []

        # Trade tracking
        self._trades: List[Trade] = []

        # Order-trade linking for stop loss monitoring
        self._trade_links: Dict[str, str] = {}  # trade_id -> stop_order_id
        self._stop_to_trade: Dict[str, str] = {}  # stop_order_id -> trade_id

        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info("Execution engine initialized")

    async def _add_trade(self, trade: Trade):
        """Add trade with locking."""
        async with state_manager.lock_state("trades"):
            self._trades.append(trade)

    async def get_trades_safe(self) -> List[Trade]:
        """Get trades with locking."""
        async with state_manager.lock_state("trades"):
            return list(self._trades)

    async def get_open_trade(self, symbol: str) -> Optional[Trade]:
        """Get open trade for symbol with locking."""
        async with state_manager.lock_state("trades"):
            return next((t for t in self._trades if t.symbol == symbol and t.status == "open"), None)

    async def start_monitoring(self):
        """Start background order monitoring task."""
        if not self._running:
            self._running = True
            self._monitor_task = asyncio.create_task(self._monitor_pending_orders())
            logger.info("Started order monitoring task")

    async def stop_monitoring(self):
        """Stop background order monitoring task."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped order monitoring task")

    async def execute_action(
        self,
        action: TradingAction,
        symbol: str,
        current_price: float
    ) -> Optional[Order]:
        """Execute a trading action."""
        try:
            if action.action == Action.HOLD:
                logger.info("Hold action - no execution required")
                return None
            
            # Determine order parameters
            side = Side.LONG if action.action == Action.BUY else Side.SHORT
            
            # Create stop loss order if specified
            stop_order_id = None
            if action.stop_loss:
                stop_order = await self._create_stop_loss_order(
                    symbol, side, action.size, action.stop_loss
                )
                if stop_order:
                    async with state_manager.lock_state("orders"):
                        self._pending_orders[stop_order.id] = stop_order
                    stop_order_id = stop_order.id

            # Create main order
            order_type = OrderType.MARKET  # Default to market orders
            price = None

            # For limit orders, we would use action.take_profit
            # For now, we'll use market orders for immediate execution

            order = await self.exchange.create_order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                amount=action.size,
                price=price
            )

            # Handle failed order creation (e.g., insufficient margin)
            if order is None:
                logger.warning(f"Order creation failed for {symbol}, action not executed")
                # Cancel stop order if we created one
                if stop_order_id:
                    try:
                        await self.exchange.cancel_order(stop_order_id, symbol)
                        async with state_manager.lock_state("orders"):
                            self._pending_orders.pop(stop_order_id, None)
                    except Exception:
                        pass
                return None

            # Track order
            async with state_manager.lock_state("orders"):
                self._pending_orders[order.id] = order

            logger.info(
                f"Executed {action.action.value} order: "
                f"{action.size} {symbol} @ {current_price}"
            )

            # Wait for order to fill (for market orders)
            if order_type == OrderType.MARKET:
                await self._wait_for_fill(order, stop_order_id=stop_order_id, expected_price=current_price)

            return order

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            raise
    
    async def _create_stop_loss_order(
        self,
        symbol: str,
        side: Side,
        amount: float,
        stop_price: float
    ) -> Optional[Order]:
        """Create a stop loss order."""
        try:
            # For stop loss, we need to reverse the side
            stop_side = Side.SHORT if side == Side.LONG else Side.LONG
            
            stop_order = await self.exchange.create_order(
                symbol=symbol,
                order_type=OrderType.STOP,
                side=stop_side,
                amount=amount,
                stop_price=stop_price
            )
            
            logger.info(f"Created stop loss order: {stop_order.id} @ {stop_price}")
            return stop_order

        except Exception as e:
            error_str = str(e)
            # Handle insufficient margin and other operational errors
            if any(phrase in error_str for phrase in ['-2019', 'insufficient', '-1121', '-2010', '-4131']):
                logger.warning(f"Could not create stop loss order for {symbol}: {e}")
            else:
                logger.error(f"Error creating stop loss order: {e}")
            return None
    
    async def _wait_for_fill(self, order: Order, timeout: int = 30, stop_order_id: Optional[str] = None, expected_price: float = 0):
        """Wait for order to fill."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            try:
                updated_order = await self.exchange.get_order(order.id, order.symbol)

                if updated_order is None:
                    logger.warning(f"Could not fetch order {order.id} status")
                    await asyncio.sleep(0.5)
                    continue  # Keep waiting, don't assume filled

                if updated_order.status == OrderStatus.FILLED:
                    # Move to completed orders
                    async with state_manager.lock_state("orders"):
                        self._pending_orders.pop(order.id, None)
                        self._completed_orders.append(updated_order)

                    # Calculate slippage
                    if updated_order.average_fill_price and expected_price > 0:
                        slippage_bps = abs((updated_order.average_fill_price - expected_price) / expected_price) * 10000
                    else:
                        slippage_bps = 0

                    # Create trade record with slippage
                    trade_id = await self._create_trade_record(updated_order, slippage=slippage_bps)

                    # Link stop order to trade if provided
                    if trade_id and stop_order_id:
                        self._trade_links[trade_id] = stop_order_id
                        self._stop_to_trade[stop_order_id] = trade_id
                        logger.info(f"Linked stop order {stop_order_id} to trade {trade_id}")

                    logger.info(f"Order filled: {order.id}, slippage: {slippage_bps:.2f} bps")
                    return True

                elif updated_order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    async with state_manager.lock_state("orders"):
                        self._pending_orders.pop(order.id, None)
                    self._completed_orders.append(updated_order)
                    logger.warning(f"Order {order.id} {updated_order.status.value}")
                    return False
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error checking order status: {e}")
                await asyncio.sleep(1)
        
        # Timeout
        logger.warning(f"Order {order.id} timed out")
        return False
    
    async def _create_trade_record(self, order: Order, slippage: float = 0) -> Optional[str]:
        """Create a trade record from filled order. Returns trade_id."""
        try:
            trade = Trade(
                id=str(uuid.uuid4()),
                symbol=order.symbol,
                side=order.side,
                quantity=order.filled_quantity,
                entry_price=order.average_fill_price or 0,
                exit_price=0,  # Will be set when position is closed
                entry_time=order.created_at,
                exit_time=datetime.now(),  # Will be updated when closed
                duration=timedelta(0),  # Will be updated when closed
                pnl=0,  # Will be calculated when closed
                pnl_pct=0,  # Will be calculated when closed
                r_multiple=0,  # Will be calculated when closed
                mae=0,  # Will be calculated from position monitoring
                mfe=0,  # Will be calculated from position monitoring
                stop_loss=order.stop_price or 0,
                take_profit=None,  # Will be set if applicable
                regime=Regime.LOW_VOL,  # Will be set from market state
                volatility_percentile=50,  # Will be set from market state
                liquidity_score=0.5,  # Will be set from market state
                slippage=slippage,
                confidence=0.5,  # Will be set from decision
                status="open",
                commission_entry=order.commission,
                commission_exit=0.0,
                funding_cost=0.0,
            )

            await self._add_trade(trade)
            logger.info(f"Created trade record: {trade.id}")
            return trade.id

        except Exception as e:
            logger.error(f"Error creating trade record: {e}")
            return None
    
    async def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all pending orders."""
        async with state_manager.lock_state("orders"):
            orders_to_cancel = list(self._pending_orders.values())

            if symbol:
                orders_to_cancel = [o for o in orders_to_cancel if o.symbol == symbol]

        for order in orders_to_cancel:
            try:
                success = await self.exchange.cancel_order(order.id, order.symbol)
                if success:
                    async with state_manager.lock_state("orders"):
                        self._pending_orders.pop(order.id, None)
                    logger.info(f"Cancelled order: {order.id}")
            except Exception as e:
                logger.error(f"Error cancelling order {order.id}: {e}")

    async def _monitor_pending_orders(self):
        """Background task to monitor all pending orders (especially stops)."""
        while self._running:
            try:
                async with state_manager.lock_state("orders"):
                    orders_to_check = list(self._pending_orders.items())

                for order_id, order in orders_to_check:
                    try:
                        updated_order = await self.exchange.get_order(order_id, order.symbol)

                        if updated_order is None:
                            # Order doesn't exist on exchange (get_order caught exception and returned None)
                            logger.warning(f"Order {order_id} not found on exchange, removing from tracking")
                            async with state_manager.lock_state("orders"):
                                self._pending_orders.pop(order_id, None)
                            # Clean up any trade links
                            if order_id in self._stop_to_trade:
                                trade_id = self._stop_to_trade.pop(order_id)
                                self._trade_links.pop(trade_id, None)
                                logger.info(f"Cleaned up trade links for missing order {order_id}")
                            continue

                        if updated_order.status == OrderStatus.FILLED:
                            await self._handle_order_fill(updated_order)
                        elif updated_order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                            await self._handle_order_cancel(updated_order)
                    except Exception as e:
                        logger.error(f"Unexpected error checking order {order_id}: {e}")

                await asyncio.sleep(2)  # Check every 2 seconds

            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(5)

    async def _handle_order_fill(self, order: Order):
        """Handle filled order - update trade if it's a stop loss."""
        # Remove from pending
        async with state_manager.lock_state("orders"):
            self._pending_orders.pop(order.id, None)
            self._completed_orders.append(order)

        # Check if this is a stop-loss order
        if order.id in self._stop_to_trade:
            trade_id = self._stop_to_trade[order.id]
            await self._complete_trade_on_stop(trade_id, order)
            logger.info(f"Stop-loss order {order.id} filled for trade {trade_id}")
        else:
            # Entry order - trade record already created in _wait_for_fill
            logger.info(f"Entry order {order.id} filled")

    async def _handle_order_cancel(self, order: Order):
        """Handle cancelled or rejected order."""
        async with state_manager.lock_state("orders"):
            self._pending_orders.pop(order.id, None)

        logger.warning(f"Order {order.id} was cancelled/rejected: status={order.status}")

        # Clean up trade linking if it was a stop order
        if order.id in self._stop_to_trade:
            trade_id = self._stop_to_trade.pop(order.id)
            self._trade_links.pop(trade_id, None)
            logger.warning(f"Stop-loss order {order.id} cancelled for trade {trade_id}")

    async def _complete_trade_on_stop(self, trade_id: str, exit_order: Order):
        """Complete a trade when stop-loss fills."""
        # Find the original trade
        trade = next((t for t in self._trades if t.id == trade_id), None)
        if not trade:
            logger.error(f"Trade {trade_id} not found for stop fill")
            return

        # Validate exit price
        exit_price = exit_order.average_fill_price or exit_order.price
        if not exit_price or exit_price <= 0:
            logger.error(f"Invalid exit price for trade {trade_id}: {exit_price}")
            raise ValueError(f"Cannot complete trade with invalid exit price: {exit_price}")

        # Validate entry price
        entry_price = trade.entry_price
        if not entry_price or entry_price <= 0:
            logger.error(f"Invalid entry price for trade {trade_id}: {entry_price}")
            raise ValueError(f"Cannot complete trade with invalid entry price: {entry_price}")

        quantity = trade.quantity

        # Long: (exit - entry) * quantity
        # Short: (entry - exit) * quantity
        if trade.side == Side.LONG:
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        # Calculate net PnL (subtract all costs)
        net_pnl = gross_pnl - trade.commission_entry - exit_order.commission - trade.funding_cost

        # Update trade record
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.duration = trade.exit_time - trade.entry_time
        trade.pnl = net_pnl
        trade.pnl_pct = (net_pnl / (entry_price * quantity)) * 100 if entry_price * quantity > 0 else 0

        # Calculate R-multiple
        if trade.stop_loss and trade.stop_loss != entry_price:
            risk_per_unit = abs(entry_price - trade.stop_loss)
            risk_total = risk_per_unit * quantity
            trade.r_multiple = net_pnl / risk_total if risk_total > 0 else 0
        else:
            trade.r_multiple = 0

        trade.commission_exit = exit_order.commission
        trade.status = "closed"

        # Persist to journal
        await self._persist_trade_to_journal(trade)

        # Clean up linking
        self._trade_links.pop(trade_id, None)
        self._stop_to_trade.pop(exit_order.id, None)

        logger.info(
            f"Completed trade {trade_id}: PnL=${net_pnl:.2f} ({trade.pnl_pct:.2f}%), "
            f"R={trade.r_multiple:.2f}x"
        )

    async def _persist_trade_to_journal(self, trade: Trade, market_state: Optional[MarketState] = None):
        """Persist completed trade to journal database."""
        if not self.journal:
            logger.warning("No journal configured - trade not persisted")
            return

        try:
            # Create minimal market state if not provided
            if market_state is None:
                market_state = MarketState(
                    ohlcv=pd.DataFrame(),  # Empty DataFrame for completed trades
                    current_price=trade.exit_price or trade.entry_price,
                    atr=0.0,
                    realized_volatility=0.0,
                    spread=0.0,
                    order_book_imbalance=0.0,
                    volume_delta=0.0,
                    liquidity_score=trade.liquidity_score,
                    regime_label=trade.regime,
                    volatility_percentile=trade.volatility_percentile,
                )

            self.journal.record_trade(trade, market_state)
            logger.info(f"Persisted trade {trade.id} to journal")
        except Exception as e:
            logger.error(f"Failed to persist trade {trade.id}: {e}")

    async def complete_trade_on_manual_close(self, symbol: str, exit_order: Order):
        """Complete trade after manual close - validates order is filled."""
        # Validate order status
        if exit_order.status != OrderStatus.FILLED:
            raise ValueError(f"Cannot complete trade with unfilled order: {exit_order.status}")

        # Validate exit price exists
        exit_price = exit_order.average_fill_price or exit_order.price
        if not exit_price or exit_price <= 0:
            raise ValueError(f"Exit order has no valid fill price: {exit_price}")

        # Find open trade for this symbol
        open_trade = await self.get_open_trade(symbol)
        if not open_trade:
            logger.warning(f"No open trade found for {symbol}")
            return

        # Validate entry price
        entry_price = open_trade.entry_price
        if not entry_price or entry_price <= 0:
            raise ValueError(f"Cannot complete trade with invalid entry price: {entry_price}")

        quantity = open_trade.quantity

        # Calculate PnL
        if open_trade.side == Side.LONG:
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        # Calculate net PnL
        net_pnl = gross_pnl - open_trade.commission_entry - exit_order.commission - open_trade.funding_cost

        # Update trade record
        open_trade.exit_price = exit_price
        open_trade.exit_time = datetime.now()
        open_trade.duration = open_trade.exit_time - open_trade.entry_time
        open_trade.pnl = net_pnl
        open_trade.pnl_pct = (net_pnl / (entry_price * quantity)) * 100 if entry_price * quantity > 0 else 0

        # Calculate R-multiple
        if open_trade.stop_loss and open_trade.stop_loss != entry_price:
            risk_per_unit = abs(entry_price - open_trade.stop_loss)
            risk_total = risk_per_unit * quantity
            open_trade.r_multiple = net_pnl / risk_total if risk_total > 0 else 0
        else:
            open_trade.r_multiple = 0

        open_trade.commission_exit = exit_order.commission
        open_trade.status = "closed"

        # Persist to journal
        await self._persist_trade_to_journal(open_trade)

        # Cancel associated stop-loss order if exists
        if open_trade.id in self._trade_links:
            stop_order_id = self._trade_links.pop(open_trade.id)
            self._stop_to_trade.pop(stop_order_id, None)
            try:
                await self.exchange.cancel_order(stop_order_id, symbol)
                logger.info(f"Cancelled stop-loss order {stop_order_id} for manual close")
            except Exception as e:
                logger.warning(f"Could not cancel stop order {stop_order_id}: {e}")

        logger.info(
            f"Completed manual close for {symbol}: PnL=${net_pnl:.2f} ({open_trade.pnl_pct:.2f}%), "
            f"R={open_trade.r_multiple:.2f}x"
        )

    async def reconcile_positions(self) -> Dict[str, str]:
        """Reconcile internal position tracking with exchange."""
        discrepancies = {}

        try:
            # Get exchange positions
            exchange_positions = await self.exchange.get_positions()
            exchange_by_symbol = {p['symbol']: p for p in exchange_positions}

            # Get internal positions
            internal_positions = self.position_monitor.get_all_positions()

            # Compare with internal tracking
            for symbol, internal_pos in internal_positions.items():
                exchange_pos = exchange_by_symbol.get(symbol)

                if exchange_pos:
                    # Check size mismatch
                    if abs(internal_pos.position_size - exchange_pos['size']) > 0.001:
                        discrepancy = f"Size mismatch: internal={internal_pos.position_size}, exchange={exchange_pos['size']}"
                        discrepancies[symbol] = discrepancy
                        logger.error(f"{symbol}: {discrepancy}")

                        # Fix: trust exchange
                        internal_pos.position_size = exchange_pos['size']
                        if exchange_pos['size'] > 0:
                            internal_pos.current_side = Side.LONG
                        elif exchange_pos['size'] < 0:
                            internal_pos.current_side = Side.SHORT
                        else:
                            internal_pos.current_side = Side.FLAT
                else:
                    # Internal thinks we have position, exchange doesn't
                    if abs(internal_pos.position_size) > 0.001:
                        discrepancies[symbol] = "Ghost position - not on exchange"
                        logger.error(f"{symbol}: Ghost position detected")
                        internal_pos.position_size = 0
                        internal_pos.current_side = Side.FLAT

        except Exception as e:
            logger.error(f"Error reconciling positions: {e}")

        return discrepancies

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return list(self._pending_orders.values())

    def get_completed_orders(self) -> List[Order]:
        """Get all completed orders."""
        return self._completed_orders.copy()

    def get_trades(self) -> List[Trade]:
        """Get all trades."""
        return self._trades.copy()
    
    async def close(self):
        """Close execution engine."""
        # Cancel all pending orders
        await self.cancel_all_orders()
        
        # Close exchange connection
        await self.exchange.close()
        
        logger.info("Execution engine closed")
