"""Unit tests for the reward function."""

import pytest

from trade_bot.learning.reward import RewardFunction, RewardComponents


class TestRewardFunction:
    def setup_method(self):
        self.rf = RewardFunction()

    def test_positive_equity_positive_reward(self):
        result = self.rf.calculate(
            delta_equity=100.0,
            drawdown_increase=0.0,
            volatility_exposure=0.3,
            position_size_pct=0.01,
            transaction_cost=1.0,
            daily_loss_pct=0.0,
        )
        assert result.total > 0
        assert result.delta_equity == 100.0
        assert not result.risk_violation

    def test_drawdown_penalty(self):
        base = self.rf.calculate(
            delta_equity=100.0, drawdown_increase=0.0,
            volatility_exposure=0.3, position_size_pct=0.01,
            transaction_cost=0.0, daily_loss_pct=0.0,
        )
        with_dd = self.rf.calculate(
            delta_equity=100.0, drawdown_increase=0.05,
            volatility_exposure=0.3, position_size_pct=0.01,
            transaction_cost=0.0, daily_loss_pct=0.0,
        )
        assert with_dd.total < base.total
        assert with_dd.drawdown_penalty > 0

    def test_volatility_penalty(self):
        low_vol = self.rf.calculate(
            delta_equity=50.0, drawdown_increase=0.0,
            volatility_exposure=0.1, position_size_pct=0.01,
            transaction_cost=0.0, daily_loss_pct=0.0,
        )
        high_vol = self.rf.calculate(
            delta_equity=50.0, drawdown_increase=0.0,
            volatility_exposure=0.9, position_size_pct=0.01,
            transaction_cost=0.0, daily_loss_pct=0.0,
        )
        assert high_vol.total < low_vol.total

    def test_position_size_penalty_only_above_threshold(self):
        small = self.rf.calculate(
            delta_equity=50.0, drawdown_increase=0.0,
            volatility_exposure=0.3, position_size_pct=0.01,
            transaction_cost=0.0, daily_loss_pct=0.0,
        )
        assert small.position_size_penalty == 0.0

        large = self.rf.calculate(
            delta_equity=50.0, drawdown_increase=0.0,
            volatility_exposure=0.3, position_size_pct=0.05,
            transaction_cost=0.0, daily_loss_pct=0.0,
        )
        assert large.position_size_penalty > 0

    def test_transaction_cost_penalty(self):
        result = self.rf.calculate(
            delta_equity=50.0, drawdown_increase=0.0,
            volatility_exposure=0.3, position_size_pct=0.01,
            transaction_cost=10.0, daily_loss_pct=0.0,
        )
        assert result.transaction_cost_penalty == 10.0

    def test_daily_loss_violation_penalty(self):
        result = self.rf.calculate(
            delta_equity=10.0, drawdown_increase=0.0,
            volatility_exposure=0.3, position_size_pct=0.01,
            transaction_cost=0.0, daily_loss_pct=0.06,  # > 5%
        )
        assert result.daily_loss_violation_penalty < 0
        assert result.total < -50  # Large negative

    def test_risk_violation_severe_penalty(self):
        result = self.rf.calculate(
            delta_equity=10.0, drawdown_increase=0.0,
            volatility_exposure=0.3, position_size_pct=0.01,
            transaction_cost=0.0, daily_loss_pct=0.0,
            risk_violation=True,
        )
        assert result.risk_violation is True
        assert result.total < -100

    def test_components_structure(self):
        result = self.rf.calculate(
            delta_equity=0.0, drawdown_increase=0.0,
            volatility_exposure=0.0, position_size_pct=0.0,
            transaction_cost=0.0, daily_loss_pct=0.0,
        )
        assert isinstance(result, RewardComponents)
        assert hasattr(result, "total")
        assert hasattr(result, "risk_violation")
