"""Emergency close procedures with proper validation."""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..core.models import MarketState, PositionState, TradingAction, Action, Side
from ..core.enums import OrderType

logger = logging.getLogger(__name__)


class EmergencyManager:
    """Manages emergency close procedures with proper validation."""
    
    def __init__(self, max_emergency_attempts: int = 3):
        """Initialize emergency manager."""
        self.max_emergency_attempts = max_emergency_attempts
        self._emergency_active = False
        logger.info("Emergency manager initialized")
    
    async def emergency_close_all(
        self,
        positions: Dict[str, PositionState],
        execution_engine,
        data_manager,
        symbols: List[str]
    ) -> Dict[str, bool]:
        """
        Emergency close all positions with proper validation.
        
        Args:
            positions: Dictionary of positions to close
            execution_engine: Execution engine instance
            data_manager: Data manager for market data
            symbols: List of symbols to check
            
        Returns:
            Dictionary of symbol -> close success status
        """
        if self._emergency_active:
            logger.warning("Emergency close already in progress")
            return {}
        
        self._emergency_active = True
        results = {}
        
        try:
            logger.critical("EMERGENCY CLOSE PROCEDURE INITIATED")
            
            # Validate market conditions before closing
            market_validation = await self._validate_market_conditions(
                data_manager, symbols
            )
            
            if not market_validation['markets_open']:
                logger.error("Markets are not open - cannot execute emergency close")
                return {symbol: False for symbol in positions.keys()}
            
            # Close positions one by one with validation
            for symbol, position in positions.items():
                if position.current_side == Side.FLAT:
                    results[symbol] = True  # Already flat
                    continue
                
                try:
                    success = await self._emergency_close_position(
                        symbol, position, execution_engine, data_manager
                    )
                    results[symbol] = success
                    
                    if success:
                        logger.info(f"Emergency close successful for {symbol}")
                    else:
                        logger.error(f"Emergency close failed for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error in emergency close for {symbol}: {e}")
                    results[symbol] = False
            
            # Log summary
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            logger.critical(
                f"EMERGENCY CLOSE COMPLETE: {successful}/{total} positions closed"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error in emergency close: {e}")
            return {symbol: False for symbol in positions.keys()}
        
        finally:
            self._emergency_active = False
    
    async def _validate_market_conditions(
        self,
        data_manager,
        symbols: List[str]
    ) -> Dict[str, bool]:
        """Validate market conditions for emergency close."""
        validation = {
            'markets_open': True,
            'exchange_healthy': True,
            'liquidity_acceptable': True,
            'symbols': {}
        }
        
        try:
            # Check each symbol
            for symbol in symbols:
                try:
                    market_state = await data_manager.get_market_state(symbol, '1m')
                    
                    # Check if market is open (basic check)
                    is_open = self._is_market_open(market_state)
                    validation['symbols'][symbol] = {
                        'open': is_open,
                        'liquidity': market_state.liquidity_score,
                        'spread': market_state.spread,
                    }
                    
                    # Update overall validation
                    if not is_open:
                        validation['markets_open'] = False
                    
                    if market_state.liquidity_score < 0.1:
                        validation['liquidity_acceptable'] = False
                        
                except Exception as e:
                    logger.error(f"Error validating {symbol}: {e}")
                    validation['symbols'][symbol] = {'error': str(e)}
                    validation['markets_open'] = False
            
        except Exception as e:
            logger.error(f"Error in market validation: {e}")
            validation['exchange_healthy'] = False
        
        return validation
    
    def _is_market_open(self, market_state: MarketState) -> bool:
        """Check if market is open based on market state."""
        # Basic checks - in production, you'd check exchange-specific hours
        if market_state.current_price <= 0:
            return False
        
        if market_state.liquidity_score <= 0:
            return False
        
        # Check if data is recent (within last 5 minutes)
        if not market_state.ohlcv.empty:
            latest_time = market_state.ohlcv.index[-1]
            time_diff = datetime.now() - latest_time
            if time_diff.total_seconds() > 300:  # 5 minutes
                return False
        
        return True
    
    async def _emergency_close_position(
        self,
        symbol: str,
        position: PositionState,
        execution_engine,
        data_manager
    ) -> bool:
        """
        Emergency close a single position with validation.
        
        Args:
            symbol: Trading symbol
            position: Position to close
            execution_engine: Execution engine
            data_manager: Data manager
            
        Returns:
            True if close was successful
        """
        for attempt in range(self.max_emergency_attempts):
            try:
                # Get current market state
                market_state = await data_manager.get_market_state(symbol, '1m')
                
                # Validate position can be closed
                if not self._can_close_position(position, market_state):
                    logger.error(f"Cannot close position {symbol}: validation failed")
                    return False
                
                # Create emergency close action
                close_action = self._create_emergency_close_action(position, market_state)
                
                # Execute with market order (for immediate execution)
                order = await execution_engine.execute_action(
                    close_action, symbol, market_state.current_price
                )
                
                if order:
                    logger.info(f"Emergency close order placed for {symbol}: {order.id}")
                    return True
                else:
                    logger.warning(f"Failed to place emergency close order for {symbol}")
                    
            except Exception as e:
                logger.error(f"Emergency close attempt {attempt + 1} failed for {symbol}: {e}")
                
                if attempt < self.max_emergency_attempts - 1:
                    await asyncio.sleep(1)  # Brief delay before retry
        
        return False
    
    def _can_close_position(self, position: PositionState, market_state: MarketState) -> bool:
        """Validate if position can be safely closed."""
        # Check position state
        if position.current_side == Side.FLAT:
            return True  # Already closed
        
        if position.position_size <= 0:
            return True  # No position
        
        # Check market conditions
        if market_state.liquidity_score < 0.05:
            logger.error("Liquidity too low for emergency close")
            return False
        
        if market_state.spread > 0.01:  # 1% spread
            logger.error("Spread too wide for emergency close")
            return False
        
        return True
    
    def _create_emergency_close_action(
        self,
        position: PositionState,
        market_state: MarketState
    ) -> TradingAction:
        """Create emergency close trading action."""
        # Determine close side
        close_side = Action.SELL if position.current_side == Side.LONG else Action.BUY
        
        # Use full position size
        close_size = abs(position.position_size)
        
        # Emergency close uses market order - no stop loss needed
        return TradingAction(
            action=close_side,
            size=close_size,
            stop_loss=None,  # No stop loss for emergency close
            take_profit=None,
            expected_return=0.0,  # Not applicable for emergency close
            expected_risk=0.0,
            confidence=1.0  # Maximum confidence for emergency
        )
    
    def is_emergency_active(self) -> bool:
        """Check if emergency procedure is currently active."""
        return self._emergency_active
    
    async def validate_emergency_conditions(
        self,
        account_state,
        risk_state,
        market_states: Dict[str, MarketState]
    ) -> Dict[str, bool]:
        """
        Validate if emergency conditions are met.
        
        Returns:
            Dictionary of condition -> status
        """
        conditions = {
            'drawdown_exceeded': False,
            'daily_loss_exceeded': False,
            'consecutive_losses_exceeded': False,
            'system_healthy': True,
            'markets_healthy': True,
        }
        
        # Check drawdown
        lock_threshold = 0.20  # 20% drawdown
        if account_state.current_drawdown >= lock_threshold:
            conditions['drawdown_exceeded'] = True
            logger.critical(f"DRAWDOWN EMERGENCY: {account_state.current_drawdown:.2%}")
        
        # Check daily loss
        daily_loss_threshold = 0.10  # 10% daily loss
        if account_state.daily_loss_pct >= daily_loss_threshold:
            conditions['daily_loss_exceeded'] = True
            logger.critical(f"DAILY LOSS EMERGENCY: {account_state.daily_loss_pct:.2%}")
        
        # Check consecutive losses
        if risk_state.consecutive_losses >= 5:
            conditions['consecutive_losses_exceeded'] = True
            logger.critical(f"CONSECUTIVE LOSSES EMERGENCY: {risk_state.consecutive_losses}")
        
        # Check market health
        for symbol, market_state in market_states.items():
            if market_state.liquidity_score < 0.1:
                conditions['markets_healthy'] = False
                logger.error(f"Market health issue: {symbol} liquidity {market_state.liquidity_score:.2f}")
        
        return conditions
