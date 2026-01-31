"""Pytest configuration and fixtures."""

import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd

from trade_bot.core.models import MarketState, PositionState, AccountState, RiskState
from trade_bot.core.enums import Side, Regime


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_market_state():
    """Create a sample market state for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    
    ohlcv = pd.DataFrame({
        'open': [100.0] * 100,
        'high': [102.0] * 100,
        'low': [98.0] * 100,
        'close': [100.0] * 100,
        'volume': [1000] * 100,
    }, index=dates)
    
    return MarketState(
        ohlcv=ohlcv,
        current_price=100.0,
        atr=2.0,
        realized_volatility=0.2,
        spread=0.001,
        order_book_imbalance=0.1,
        volume_delta=100.0,
        liquidity_score=0.8,
        regime_label=Regime.TRENDING,
        volatility_percentile=50.0,
    )


@pytest.fixture
def sample_position_state():
    """Create a sample position state for testing."""
    return PositionState(
        current_side=Side.FLAT,
        position_size=0.0,
        entry_price=None,
        time_in_position=timedelta(0),
        entry_time=None,
        mae=0.0,
        mfe=0.0,
        unrealized_pnl=0.0,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
    )


@pytest.fixture
def sample_account_state():
    """Create a sample account state for testing."""
    return AccountState(
        equity=10000.0,
        balance=10000.0,
        used_margin=0.0,
        exposure_pct=0.0,
        leverage=1.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        current_drawdown=0.0,
        max_drawdown=0.0,
        daily_loss_pct=0.0,
        daily_pnl=0.0,
        consecutive_losses=0,
    )


@pytest.fixture
def sample_risk_state():
    """Create a sample risk state for testing."""
    return RiskState(
        risk_budget_left=100.0,
        max_daily_loss_remaining=500.0,
        consecutive_losses=0,
        volatility_percentile=50.0,
        max_risk_per_trade=0.01,
        max_exposure_per_asset=0.30,
        max_total_exposure=0.50,
        max_leverage=2.0,
        current_risk_level='low',
        safe_mode_active=False,
    )
