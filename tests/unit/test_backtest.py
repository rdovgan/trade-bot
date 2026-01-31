"""Unit tests for walk-forward validation."""

import pytest
import numpy as np
import pandas as pd

from trade_bot.learning.backtest import (
    SlippageModel,
    LatencySimulator,
    CommissionModel,
    WalkForwardValidator,
    BacktestResult,
    ValidationResult,
)


def _make_data(n=800):
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1h")
    prices = 100 + np.cumsum(np.random.normal(0, 0.5, n))
    prices = np.maximum(prices, 10)  # Keep positive
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(500, 1500, n),
        },
        index=dates,
    )


def _simple_signal_fn(data):
    """Simple signal: buy if last close > previous close."""
    if len(data) < 2:
        return []
    last = data["close"].iloc[-1]
    prev = data["close"].iloc[-2]
    if last > prev:
        return [{
            "bar_index": len(data) - 1,
            "action": "BUY",
            "size": 1.0,
            "stop_loss": last - 2.0,
            "take_profit": last + 3.0,
            "confidence": 0.6,
        }]
    return []


class TestSlippageModel:
    def test_base_slippage(self):
        model = SlippageModel(base_slippage_bps=5.0)
        slip = model.estimate(price=100.0, volatility=0.0, liquidity_score=1.0)
        assert slip > 0

    def test_higher_volatility_more_slippage(self):
        model = SlippageModel()
        low = model.estimate(100.0, 0.1, 0.8)
        high = model.estimate(100.0, 0.5, 0.8)
        assert high > low

    def test_lower_liquidity_more_slippage(self):
        model = SlippageModel()
        liquid = model.estimate(100.0, 0.2, 0.9)
        illiquid = model.estimate(100.0, 0.2, 0.1)
        assert illiquid > liquid


class TestLatencySimulator:
    def test_get_delay_bars(self):
        sim = LatencySimulator(base_delay_ms=50, jitter_ms=10)
        delay = sim.get_delay_bars(60_000)  # 1-minute bars
        assert delay >= 0
        assert delay <= 2  # Typically 0 or 1

    def test_zero_bar_duration(self):
        sim = LatencySimulator()
        delay = sim.get_delay_bars(0)
        assert delay == 0


class TestCommissionModel:
    def test_calculate(self):
        model = CommissionModel(rate_bps=10.0)
        comm = model.calculate(10_000.0)
        assert comm == 10.0  # 10 bps of 10000

    def test_zero_notional(self):
        model = CommissionModel()
        assert model.calculate(0.0) == 0.0


class TestWalkForwardValidator:
    def test_validate_returns_result(self):
        data = _make_data(800)
        validator = WalkForwardValidator(
            train_bars=200, test_bars=50, step_bars=50,
        )
        result = validator.validate(data, _simple_signal_fn)
        assert isinstance(result, ValidationResult)
        assert len(result.windows) > 0

    def test_validate_insufficient_data(self):
        data = _make_data(50)
        validator = WalkForwardValidator(
            train_bars=200, test_bars=50, step_bars=50,
        )
        result = validator.validate(data, _simple_signal_fn)
        assert len(result.windows) == 0
        assert result.passed is False

    def test_validate_no_signals(self):
        data = _make_data(800)
        validator = WalkForwardValidator(
            train_bars=200, test_bars=50, step_bars=50,
        )
        result = validator.validate(data, lambda d: [])
        assert isinstance(result, ValidationResult)

    def test_window_metrics(self):
        data = _make_data(800)
        validator = WalkForwardValidator(
            train_bars=200, test_bars=50, step_bars=50,
        )
        result = validator.validate(data, _simple_signal_fn)
        for w in result.windows:
            assert isinstance(w, BacktestResult)
            assert w.trade_count >= 0
            assert 0 <= w.win_rate <= 1.0

    def test_calc_max_drawdown_empty(self):
        assert WalkForwardValidator._calc_max_drawdown([]) == 0.0

    def test_calc_sharpe_insufficient(self):
        assert WalkForwardValidator._calc_sharpe([]) == 0.0
        assert WalkForwardValidator._calc_sharpe([1.0]) == 0.0
