"""Unit tests for risk validator."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

import pandas as pd

from trade_bot.core.models import (
    MarketState, PositionState, AccountState, RiskState,
    TradingAction, RiskLevel
)
from trade_bot.core.enums import Action, Side, Regime
from trade_bot.risk.validator import RiskValidator, RiskViolation


def _dummy_ohlcv(n=50, price=100.0):
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": [price]*n, "high": [price*1.01]*n, "low": [price*0.99]*n,
         "close": [price]*n, "volume": [1000]*n},
        index=dates,
    )


class TestRiskValidator:
    """Test cases for RiskValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = RiskValidator()

        # Create test market state
        self.market_state = MarketState(
            ohlcv=_dummy_ohlcv(),
            current_price=100.0,
            atr=2.0,
            realized_volatility=0.2,
            spread=0.001,
            order_book_imbalance=0.1,
            volume_delta=1000,
            liquidity_score=0.8,
            regime_label=Regime.TRENDING,
            volatility_percentile=50.0,
        )
        
        # Create test position state
        self.position_state = PositionState(
            current_side=Side.FLAT,
            position_size=0.0,
            entry_price=None,
        )
        
        # Create test account state
        self.account_state = AccountState(
            equity=10000.0,
            balance=10000.0,
            exposure_pct=0.0,
            unrealized_pnl=0.0,
            current_drawdown=0.0,
            max_drawdown=0.0,
            daily_loss_pct=0.0,
            daily_pnl=0.0,
            consecutive_losses=0,
        )
        
        # Create test risk state
        self.risk_state = RiskState(
            risk_budget_left=100.0,
            max_daily_loss_remaining=500.0,
            consecutive_losses=0,
            volatility_percentile=50.0,
            current_risk_level=RiskLevel.LOW,
            safe_mode_active=False,
        )
    
    def test_validate_valid_buy_action(self):
        """Test validation of a valid buy action."""
        action = TradingAction(
            action=Action.BUY,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            expected_return=4.0,
            expected_risk=2.0,
            confidence=0.8
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is True
        assert reason is None
    
    def test_validate_invalid_risk_reward_ratio(self):
        """Test rejection of action with poor R:R ratio."""
        action = TradingAction(
            action=Action.BUY,
            size=1.0,
            stop_loss=98.0,
            take_profit=99.0,
            expected_return=1.0,
            expected_risk=2.0,
            confidence=0.8
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is False
        assert "R:R ratio" in reason
    
    def test_validate_missing_stop_loss(self):
        """Test rejection of action without stop loss."""
        action = TradingAction(
            action=Action.BUY,
            size=1.0,
            stop_loss=None,
            take_profit=104.0,
            expected_return=4.0,
            expected_risk=2.0,
            confidence=0.8
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is False
        assert "Stop loss" in reason
    
    def test_validate_excessive_risk_per_trade(self):
        """Test rejection of action with excessive risk."""
        action = TradingAction(
            action=Action.BUY,
            size=100.0,  # Very large position
            stop_loss=90.0,  # 10% stop loss
            take_profit=110.0,
            expected_return=10.0,
            expected_risk=10.0,
            confidence=0.8
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is False
        assert "risk per trade" in reason.lower()
    
    def test_validate_daily_loss_limit(self):
        """Test rejection when daily loss limit is reached."""
        # Set daily loss to exceed limit
        self.account_state.daily_loss_pct = 0.06  # 6% > 5% limit
        
        action = TradingAction(
            action=Action.BUY,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            expected_return=4.0,
            expected_risk=2.0,
            confidence=0.8
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is False
        assert "daily loss" in reason.lower()
    
    def test_validate_drawdown_pause(self):
        """Test rejection when drawdown triggers pause."""
        # Set drawdown to trigger pause
        self.account_state.current_drawdown = 0.16  # 16% > 15% threshold
        
        action = TradingAction(
            action=Action.BUY,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            expected_return=4.0,
            expected_risk=2.0,
            confidence=0.8
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is False
        assert "pause" in reason.lower()
    
    def test_validate_volatility_guard(self):
        """Test rejection when volatility is too high."""
        # Set high volatility
        self.market_state.volatility_percentile = 96.0  # > 95% threshold
        
        action = TradingAction(
            action=Action.BUY,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            expected_return=4.0,
            expected_risk=2.0,
            confidence=0.8
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is False
        assert "volatility" in reason.lower()
    
    def test_validate_liquidity_guard(self):
        """Test rejection when liquidity is too low."""
        # Set low liquidity
        self.market_state.liquidity_score = 0.2  # < 0.3 threshold
        
        action = TradingAction(
            action=Action.BUY,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            expected_return=4.0,
            expected_risk=2.0,
            confidence=0.8
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is False
        assert "liquidity" in reason.lower()
    
    def test_validate_safe_mode_high_confidence(self):
        """Test that high confidence trades are allowed in safe mode."""
        # Enable safe mode
        self.risk_state.safe_mode_active = True
        
        action = TradingAction(
            action=Action.BUY,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            expected_return=4.0,
            expected_risk=2.0,
            confidence=0.9  # High confidence
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is True
        assert reason is None
    
    def test_validate_safe_mode_low_confidence(self):
        """Test that low confidence trades are rejected in safe mode."""
        # Enable safe mode
        self.risk_state.safe_mode_active = True
        
        action = TradingAction(
            action=Action.BUY,
            size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            expected_return=4.0,
            expected_risk=2.0,
            confidence=0.7  # Low confidence
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is False
        assert "Safe mode" in reason
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        stop_distance = 2.0  # 2% stop loss
        
        size = self.validator.calculate_position_size(
            self.account_state, self.market_state, stop_distance
        )
        
        # Expected: 1% of 10000 = 100 risk amount
        # risk-based size = 100 / 2 = 50
        # exposure cap = 30% * 10000 / 100 = 30
        expected_size = 30.0
        assert abs(size - expected_size) < 0.01
    
    def test_calculate_position_size_with_drawdown(self):
        """Test position size calculation with drawdown reduction."""
        # Set drawdown to trigger reduction
        self.account_state.current_drawdown = 0.12  # 12% > 10% threshold
        
        stop_distance = 2.0
        
        size = self.validator.calculate_position_size(
            self.account_state, self.market_state, stop_distance
        )
        
        # Exposure cap = 30% * 10000 / 100 = 30, then 50% drawdown reduction = 15
        expected_size = 15.0
        assert abs(size - expected_size) < 0.01
    
    def test_update_risk_state(self):
        """Test risk state update."""
        # Add some losing trades
        losing_trades = []
        for i in range(3):
            trade = type('Trade', (), {'pnl': -10.0})()
            losing_trades.append(trade)
        
        risk_state = self.validator.update_risk_state(
            self.account_state, self.market_state, losing_trades
        )
        
        assert risk_state.consecutive_losses == 3
        assert risk_state.volatility_percentile == 50.0
    
    def test_update_risk_state_with_drawdown(self):
        """Test risk state update with high drawdown."""
        # Set high drawdown
        self.account_state.current_drawdown = 0.16  # 16% > 15% threshold
        
        risk_state = self.validator.update_risk_state(
            self.account_state, self.market_state, []
        )
        
        assert risk_state.current_risk_level == RiskLevel.CRITICAL
        assert risk_state.safe_mode_active is True
    
    def test_validate_averaging_down_rejected(self):
        """Test that averaging down is rejected."""
        self.position_state.current_side = Side.LONG
        self.position_state.position_size = 1.0
        self.position_state.entry_price = 105.0
        self.position_state.unrealized_pnl = -5.0

        action = TradingAction(
            action=Action.BUY, size=1.0, stop_loss=98.0, take_profit=104.0,
            expected_return=4.0, expected_risk=2.0, confidence=0.8,
        )

        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state,
            self.account_state, self.risk_state,
        )
        assert is_valid is False
        assert "Averaging down" in reason

    def test_validate_drawdown_lock(self):
        """Test system lock at 20% drawdown."""
        self.account_state.current_drawdown = 0.21
        action = TradingAction(
            action=Action.BUY, size=1.0, stop_loss=98.0, take_profit=104.0,
            expected_return=4.0, expected_risk=2.0, confidence=0.8,
        )
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state,
            self.account_state, self.risk_state,
        )
        assert is_valid is False
        assert "lock" in reason.lower()

    def test_validate_consecutive_losses(self):
        """Test rejection at max consecutive losses."""
        self.risk_state.consecutive_losses = 5
        action = TradingAction(
            action=Action.BUY, size=1.0, stop_loss=98.0, take_profit=104.0,
            expected_return=4.0, expected_risk=2.0, confidence=0.8,
        )
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state,
            self.account_state, self.risk_state,
        )
        assert is_valid is False
        assert "consecutive" in reason.lower()

    def test_update_risk_state_with_red_days(self):
        """Test safe mode activation on 3 red days."""
        recent_daily = [{"pnl": -10}, {"pnl": -20}, {"pnl": -5}]
        risk_state = self.validator.update_risk_state(
            self.account_state, self.market_state, [],
            recent_daily_pnls=recent_daily,
        )
        assert risk_state.safe_mode_active is True

    def test_update_risk_state_no_red_days(self):
        """Test no safe mode when recent days are profitable."""
        recent_daily = [{"pnl": 10}, {"pnl": -20}, {"pnl": 5}]
        risk_state = self.validator.update_risk_state(
            self.account_state, self.market_state, [],
            recent_daily_pnls=recent_daily,
        )
        assert risk_state.safe_mode_active is False

    def test_validate_exposure_limit(self):
        """Test rejection when exposure limit exceeded."""
        self.position_state.position_size = 100
        self.position_state.entry_price = 100.0
        action = TradingAction(
            action=Action.BUY, size=1.0, stop_loss=98.0, take_profit=104.0,
            expected_return=4.0, expected_risk=2.0, confidence=0.8,
        )
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state,
            self.account_state, self.risk_state,
        )
        assert is_valid is False
        assert "exposure" in reason.lower() or "leverage" in reason.lower()

    def test_validate_hold_action(self):
        """Test that HOLD actions are always valid."""
        action = TradingAction(
            action=Action.HOLD,
            size=0.0,
            expected_return=0.0,
            expected_risk=0.0,
            confidence=1.0
        )
        
        is_valid, reason = self.validator.validate_trading_action(
            action, self.market_state, self.position_state, 
            self.account_state, self.risk_state
        )
        
        assert is_valid is True
        assert reason is None
