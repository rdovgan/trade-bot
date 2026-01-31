"""Execution engine for order management and position monitoring."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator
import asyncio
import logging
from datetime import datetime, timedelta
import uuid

import ccxt.async_support as ccxt

from ..core.models import (
    MarketState, PositionState, AccountState, Order, Trade,
    TradingAction, OrderType, OrderStatus, Side
)
from ..core.enums import Action, Regime

logger = logging.getLogger(__name__)


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
        default_type = 'linear'
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
                # Bybit requires triggerDirection for stop/trigger orders
                trigger_dir = 'descending' if ccxt_side == 'sell' else 'ascending'
                result = await self.exchange.create_order(
                    symbol, 'market', ccxt_side, amount, None,
                    {
                        'stopLossPrice': stop_price,
                        'triggerPrice': stop_price,
                        'triggerDirection': trigger_dir,
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
            logger.error(f"Error fetching order {order_id}: {e}")
            return None
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        try:
            balance = await self.exchange.fetch_balance()
            free = balance.get('free', {})
            return {
                asset: float(amount)
                for asset, amount in free.items()
                if amount and float(amount) > 0
            }
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}
    
    async def get_positions(self) -> List[Dict]:
        """Get open positions."""
        try:
            if hasattr(self.exchange, 'fetch_positions'):
                positions = await self.exchange.fetch_positions()
                return [
                    {
                        'symbol': pos.get('symbol', ''),
                        'side': pos.get('side', ''),
                        'size': pos.get('contracts', 0),
                        'entry_price': pos.get('entryPrice', 0),
                        'unrealized_pnl': pos.get('unrealizedPnl', 0),
                    }
                    for pos in positions
                    if float(pos.get('contracts', 0)) != 0
                ]
            else:
                return []
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
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
        logger.info("Position monitor initialized")
    
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
        """Update position information."""
        try:
            positions = await self.exchange.get_positions()
            
            for pos_data in positions:
                symbol = pos_data['symbol']
                
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
                
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
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
        """Get position state for a symbol."""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionState]:
        """Get all positions."""
        return self._positions.copy()


class ExecutionEngine:
    """Main execution engine for order management."""
    
    def __init__(
        self,
        exchange_connector: ExchangeConnector,
        position_monitor: PositionMonitor
    ):
        """Initialize execution engine."""
        self.exchange = exchange_connector
        self.position_monitor = position_monitor
        
        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._completed_orders: List[Order] = []
        
        # Trade tracking
        self._trades: List[Trade] = []
        
        logger.info("Execution engine initialized")
    
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
            if action.stop_loss:
                stop_order = await self._create_stop_loss_order(
                    symbol, side, action.size, action.stop_loss
                )
                if stop_order:
                    self._pending_orders[stop_order.id] = stop_order
            
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
            
            # Track order
            self._pending_orders[order.id] = order
            
            logger.info(
                f"Executed {action.action.value} order: "
                f"{action.size} {symbol} @ {current_price}"
            )
            
            # Wait for order to fill (for market orders)
            if order_type == OrderType.MARKET:
                await self._wait_for_fill(order)
            
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
            logger.error(f"Error creating stop loss order: {e}")
            return None
    
    async def _wait_for_fill(self, order: Order, timeout: int = 30):
        """Wait for order to fill."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                updated_order = await self.exchange.get_order(order.id, order.symbol)

                if updated_order is None:
                    logger.warning(f"Could not fetch order {order.id}, assuming filled for market orders")
                    self._pending_orders.pop(order.id, None)
                    return True

                if updated_order.status == OrderStatus.FILLED:
                    # Move to completed orders
                    self._pending_orders.pop(order.id, None)
                    self._completed_orders.append(updated_order)
                    
                    # Create trade record
                    await self._create_trade_record(updated_order)
                    
                    logger.info(f"Order filled: {order.id}")
                    return True
                
                elif updated_order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
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
    
    async def _create_trade_record(self, order: Order):
        """Create a trade record from filled order."""
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
                slippage=0,  # Will be calculated
                confidence=0.5,  # Will be set from decision
            )
            
            self._trades.append(trade)
            logger.info(f"Created trade record: {trade.id}")
            
        except Exception as e:
            logger.error(f"Error creating trade record: {e}")
    
    async def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all pending orders."""
        orders_to_cancel = list(self._pending_orders.values())
        
        if symbol:
            orders_to_cancel = [o for o in orders_to_cancel if o.symbol == symbol]
        
        for order in orders_to_cancel:
            try:
                success = await self.exchange.cancel_order(order.id, order.symbol)
                if success:
                    self._pending_orders.pop(order.id, None)
                    logger.info(f"Cancelled order: {order.id}")
            except Exception as e:
                logger.error(f"Error cancelling order {order.id}: {e}")
    
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
