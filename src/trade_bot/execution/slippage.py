"""Real-time slippage protection and order size scaling."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..core.models import MarketState, OrderType, Side

logger = logging.getLogger(__name__)


@dataclass
class SlippageAnalysis:
    """Analysis of slippage conditions."""
    acceptable: bool
    recommended_size: float
    max_safe_size: float
    market_depth_score: float
    spread_impact: float
    liquidity_score: float


class SlippageProtector:
    """Real-time slippage protection system."""
    
    def __init__(self, max_slippage_bps: float = 5.0, min_liquidity_score: float = 0.3):
        """Initialize slippage protector."""
        self.max_slippage_bps = max_slippage_bps
        self.min_liquidity_score = min_liquidity_score
        
        # Market depth history
        self._depth_history: List[Tuple[datetime, Dict]] = []
        self._max_history_size = 100
        
        logger.info(f"Slippage protector initialized (max_slippage: {max_slippage_bps} bps)")
    
    async def analyze_slippage_conditions(
        self,
        market_state: MarketState,
        order_size: float,
        order_type: OrderType,
        side: Side
    ) -> SlippageAnalysis:
        """
        Analyze current slippage conditions for an order.
        
        Args:
            market_state: Current market state
            order_size: Desired order size
            order_type: Type of order
            side: Order side
            
        Returns:
            Slippage analysis with recommendations
        """
        try:
            # Calculate spread impact
            spread_impact = self._calculate_spread_impact(market_state, order_size)
            
            # Calculate market depth score
            depth_score = self._calculate_market_depth_score(market_state, order_size)
            
            # Calculate liquidity score
            liquidity_score = market_state.liquidity_score
            
            # Determine if conditions are acceptable
            acceptable = (
                spread_impact <= self.max_slippage_bps and
                liquidity_score >= self.min_liquidity_score and
                depth_score >= 0.3  # Minimum depth score
            )
            
            # Calculate recommended order size
            recommended_size = self._calculate_recommended_size(
                market_state, order_size, side
            )
            
            # Calculate maximum safe size
            max_safe_size = self._calculate_max_safe_size(market_state, side)
            
            analysis = SlippageAnalysis(
                acceptable=acceptable,
                recommended_size=recommended_size,
                max_safe_size=max_safe_size,
                market_depth_score=depth_score,
                spread_impact=spread_impact,
                liquidity_score=liquidity_score
            )
            
            logger.debug(
                f"Slippage analysis: acceptable={acceptable}, "
                f"spread_impact={spread_impact:.2f} bps, "
                f"depth_score={depth_score:.2f}, "
                f"rec_size={recommended_size:.6f}"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in slippage analysis: {e}")
            # Return conservative analysis on error
            return SlippageAnalysis(
                acceptable=False,
                recommended_size=0.0,
                max_safe_size=0.0,
                market_depth_score=0.0,
                spread_impact=999.0,
                liquidity_score=0.0
            )
    
    def _calculate_spread_impact(self, market_state: MarketState, order_size: float) -> float:
        """Calculate spread impact in basis points."""
        if market_state.current_price <= 0:
            return 999.0
        
        # Spread impact as percentage of price
        spread_pct = (market_state.spread / market_state.current_price) * 100
        
        # Adjust for order size (larger orders have more impact)
        size_factor = min(order_size / 1.0, 5.0)  # Cap at 5x impact
        adjusted_spread_pct = spread_pct * (1 + size_factor * 0.2)
        
        return adjusted_spread_pct * 100  # Convert to basis points
    
    def _calculate_market_depth_score(self, market_state: MarketState, order_size: float) -> float:
        """Calculate market depth score based on order book imbalance and volume."""
        # Base score from liquidity
        base_score = market_state.liquidity_score
        
        # Adjust for order book imbalance
        imbalance_factor = abs(market_state.order_book_imbalance)
        imbalance_adjustment = 1.0 - (imbalance_factor * 0.3)  # Reduce score for imbalance
        
        # Adjust for volume delta
        volume_factor = min(abs(market_state.volume_delta) / 10000, 1.0)  # Normalize
        volume_adjustment = 1.0 + (volume_factor * 0.2)  # Increase score for volume
        
        # Calculate final score
        depth_score = base_score * imbalance_adjustment * volume_adjustment
        
        return max(0.0, min(1.0, depth_score))
    
    def _calculate_recommended_size(
        self,
        market_state: MarketState,
        desired_size: float,
        side: Side
    ) -> float:
        """Calculate recommended order size based on market conditions."""
        # Base size adjustment based on liquidity
        liquidity_adjustment = market_state.liquidity_score
        
        # Adjust for spread
        spread_adjustment = 1.0
        if market_state.spread > 0.001:  # 0.1% spread threshold
            spread_adjustment = 0.5  # Reduce size by 50% for wide spreads
        
        # Adjust for volatility
        volatility_adjustment = 1.0
        if market_state.volatility_percentile > 80:
            volatility_adjustment = 0.7  # Reduce size in high volatility
        
        # Calculate recommended size
        recommended = desired_size * (
            liquidity_adjustment * 
            spread_adjustment * 
            volatility_adjustment
        )
        
        return max(0.0, recommended)
    
    def _calculate_max_safe_size(
        self,
        market_state: MarketState,
        side: Side
    ) -> float:
        """Calculate maximum safe order size based on market depth."""
        # This is a simplified calculation
        # In production, you'd analyze actual order book depth
        
        # Base maximum from liquidity score
        base_max = market_state.liquidity_score * 10.0  # Arbitrary scaling
        
        # Adjust for spread
        if market_state.spread > 0.002:  # 0.2% spread
            base_max *= 0.3  # Reduce to 30%
        
        # Adjust for volatility
        if market_state.volatility_percentile > 90:
            base_max *= 0.5  # Reduce to 50%
        
        return max(0.0, base_max)
    
    async def record_execution(
        self,
        expected_price: float,
        actual_price: float,
        order_size: float,
        symbol: str
    ) -> float:
        """
        Record actual execution and calculate slippage.
        
        Args:
            expected_price: Expected execution price
            actual_price: Actual execution price
            order_size: Order size
            symbol: Trading symbol
            
        Returns:
            Slippage in basis points
        """
        if expected_price == 0:
            return 0.0
        
        # Calculate slippage
        slippage_pct = ((actual_price - expected_price) / expected_price) * 100
        slippage_bps = slippage_pct * 100
        
        logger.info(
            f"Execution recorded for {symbol}: "
            f"expected={expected_price:.6f}, actual={actual_price:.6f}, "
            f"size={order_size:.6f}, slippage={slippage_bps:.2f} bps"
        )
        
        # Alert on high slippage
        if abs(slippage_bps) > self.max_slippage_bps:
            logger.warning(
                f"High slippage detected for {symbol}: {slippage_bps:.2f} bps "
                f"(threshold: {self.max_slippage_bps} bps)"
            )
        
        return slippage_bps
    
    def should_use_market_order(
        self,
        market_state: MarketState,
        order_size: float,
        urgency: str = "normal"
    ) -> bool:
        """
        Determine if market order should be used based on conditions.
        
        Args:
            market_state: Current market state
            order_size: Order size
            urgency: Order urgency (low, normal, high)
            
        Returns:
            True if market order is recommended
        """
        # Always use limit orders for large orders
        if order_size > 5.0:
            return False
        
        # Use market orders for high urgency
        if urgency == "high":
            return True
        
        # Use limit orders if spread is wide
        if market_state.spread > 0.001:  # 0.1%
            return False
        
        # Use limit orders if liquidity is low
        if market_state.liquidity_score < 0.5:
            return False
        
        # Default to market order for normal conditions
        return True
