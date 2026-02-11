"""Core data models for the trading system."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

from .enums import Action, Side, Regime, OrderType, OrderStatus, RiskLevel


class MarketState(BaseModel):
    """Market state data."""
    
    # Price data
    ohlcv: pd.DataFrame = Field(description="OHLCV data for last N bars")
    current_price: float = Field(description="Current market price")
    
    # Technical indicators
    atr: float = Field(description="Average True Range")
    realized_volatility: float = Field(description="Realized volatility")
    spread: float = Field(description="Current bid-ask spread")
    
    # Market microstructure
    order_book_imbalance: float = Field(description="Order book imbalance ratio")
    volume_delta: float = Field(description="Volume delta (buy - sell volume)")
    liquidity_score: float = Field(description="Liquidity score (0-1)")
    
    # Regime information
    regime_label: Regime = Field(description="Current market regime")
    volatility_percentile: float = Field(description="Volatility percentile (0-100)")
    
    class Config:
        arbitrary_types_allowed = True


class PositionState(BaseModel):
    """Position state data."""
    
    # Basic position info
    current_side: Side = Field(default=Side.FLAT, description="Current position side")
    position_size: float = Field(default=0.0, description="Current position size")
    entry_price: Optional[float] = Field(default=None, description="Entry price")
    
    # Time metrics
    time_in_position: timedelta = Field(default=timedelta(0), description="Time in current position")
    entry_time: Optional[datetime] = Field(default=None, description="Position entry time")
    
    # Performance metrics
    mae: float = Field(default=0.0, description="Maximum adverse excursion")
    mfe: float = Field(default=0.0, description="Maximum favorable excursion")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L")
    
    # Risk management
    stop_loss: Optional[float] = Field(default=None, description="Stop loss price")
    take_profit: Optional[float] = Field(default=None, description="Take profit price")
    trailing_stop: Optional[float] = Field(default=None, description="Trailing stop price")


class AccountState(BaseModel):
    """Account state data."""
    
    # Capital
    equity: float = Field(description="Total account equity")
    balance: float = Field(description="Available balance")
    used_margin: float = Field(default=0.0, description="Used margin")
    
    # Exposure
    exposure_pct: float = Field(default=0.0, description="Total exposure as % of equity")
    leverage: float = Field(default=1.0, description="Current leverage")
    
    # Performance
    unrealized_pnl: float = Field(default=0.0, description="Total unrealized P&L")
    realized_pnl: float = Field(default=0.0, description="Total realized P&L")
    current_drawdown: float = Field(default=0.0, description="Current drawdown %")
    max_drawdown: float = Field(default=0.0, description="Maximum drawdown %")
    
    # Daily metrics
    daily_loss_pct: float = Field(default=0.0, description="Daily loss %")
    daily_pnl: float = Field(default=0.0, description="Daily P&L")
    consecutive_losses: int = Field(default=0, description="Consecutive losing days")


class RiskState(BaseModel):
    """Risk state data."""
    
    # Risk budget
    risk_budget_left: float = Field(description="Risk budget remaining")
    max_daily_loss_remaining: float = Field(description="Maximum daily loss remaining")
    
    # Risk metrics
    consecutive_losses: int = Field(default=0, description="Consecutive losing trades")
    volatility_percentile: float = Field(description="Current volatility percentile")
    
    # Risk limits
    max_risk_per_trade: float = Field(default=0.01, description="Maximum risk per trade (% of equity)")
    max_exposure_per_asset: float = Field(default=0.30, description="Maximum exposure per asset")
    max_total_exposure: float = Field(default=0.50, description="Maximum total exposure")
    max_leverage: float = Field(default=2.0, description="Maximum leverage")
    
    # Risk level
    current_risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="Current risk level")
    safe_mode_active: bool = Field(default=False, description="Safe mode status")


class TradingAction(BaseModel):
    """Trading action schema."""

    action: Action = Field(description="Trading action")
    size: float = Field(description="Position size")
    stop_loss: Optional[float] = Field(default=None, description="Stop loss price")
    take_profit: Optional[float] = Field(default=None, description="Take profit price")
    expected_return: float = Field(description="Expected return")
    expected_risk: float = Field(description="Expected risk")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level (0-1)")
    is_close: bool = Field(default=False, description="True if this action closes existing position")
    
    @validator('expected_return', 'expected_risk')
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Expected return and risk must be positive")
        return v
    
    @validator('size')
    def validate_size(cls, v, values):
        if v < 0:
            raise ValueError("Size must be non-negative")
        return v


class Order(BaseModel):
    """Order model."""
    
    id: str = Field(description="Order ID")
    symbol: str = Field(description="Trading symbol")
    side: Side = Field(description="Order side")
    order_type: OrderType = Field(description="Order type")
    quantity: float = Field(description="Order quantity")
    price: Optional[float] = Field(default=None, description="Order price")
    stop_price: Optional[float] = Field(default=None, description="Stop price")
    
    # Status and timing
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    filled_at: Optional[datetime] = Field(default=None, description="Fill time")
    
    # Execution details
    filled_quantity: float = Field(default=0.0, description="Filled quantity")
    average_fill_price: Optional[float] = Field(default=None, description="Average fill price")
    commission: float = Field(default=0.0, description="Commission paid")


class Trade(BaseModel):
    """Trade model for completed transactions."""
    
    id: str = Field(description="Trade ID")
    symbol: str = Field(description="Trading symbol")
    side: Side = Field(description="Trade side")
    quantity: float = Field(description="Trade quantity")
    entry_price: float = Field(description="Entry price")
    exit_price: float = Field(description="Exit price")
    
    # Timing
    entry_time: datetime = Field(description="Entry time")
    exit_time: datetime = Field(description="Exit time")
    duration: timedelta = Field(description="Trade duration")
    
    # Performance
    pnl: float = Field(description="Trade P&L")
    pnl_pct: float = Field(description="Trade P&L %")
    r_multiple: float = Field(description="R multiple (return/risk)")
    
    # Risk metrics
    mae: float = Field(description="Maximum adverse excursion")
    mfe: float = Field(description="Maximum favorable excursion")
    stop_loss: float = Field(description="Stop loss price")
    take_profit: Optional[float] = Field(default=None, description="Take profit price")
    
    # Context
    regime: Regime = Field(description="Market regime during trade")
    volatility_percentile: float = Field(description="Volatility percentile")
    liquidity_score: float = Field(description="Liquidity score")
    slippage: float = Field(default=0.0, description="Slippage in basis points")
    confidence: float = Field(description="Decision confidence")

    # Status and costs
    status: str = Field(default="open", description="Trade status: open, closed")
    funding_cost: float = Field(default=0.0, description="Total funding rate costs")
    commission_entry: float = Field(default=0.0, description="Entry commission")
    commission_exit: float = Field(default=0.0, description="Exit commission")
