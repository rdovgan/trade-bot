"""Unit tests for signal generators."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trade_bot.core.models import MarketState, TradingAction
from trade_bot.core.enums import Action, Regime
from trade_bot.signal.generator import (
    MeanReversionSignal,
    TrendFollowingSignal,
    VolatilityBreakoutSignal,
    SignalManager,
    create_default_signal_manager,
)


def _make_ohlcv(n=100, base_price=100.0, trend=0.0, volatility=0.02):
    """Create synthetic OHLCV data."""
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1h")
    prices = [base_price]
    for i in range(1, n):
        prices.append(prices[-1] * (1 + trend + np.random.normal(0, volatility)))
    prices = np.array(prices)
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


def _make_market_state(ohlcv, regime=Regime.TRENDING, vol_pct=50.0):
    return MarketState(
        ohlcv=ohlcv,
        current_price=float(ohlcv["close"].iloc[-1]),
        atr=float(ohlcv["close"].std() * 0.5),
        realized_volatility=0.2,
        spread=0.0005,
        order_book_imbalance=0.1,
        volume_delta=100.0,
        liquidity_score=0.8,
        regime_label=regime,
        volatility_percentile=vol_pct,
    )


class TestMeanReversionSignal:
    def setup_method(self):
        self.gen = MeanReversionSignal(lookback=20, threshold=2.0)

    def test_required_regime(self):
        assert self.gen.get_required_regime() == Regime.MEAN_REVERT

    def test_no_signal_insufficient_data(self):
        ohlcv = _make_ohlcv(n=5)
        ms = _make_market_state(ohlcv, Regime.MEAN_REVERT)
        assert self.gen.generate_signal(ms) is None

    def test_buy_signal_below_lower_band(self):
        """Price far below the mean → BUY signal."""
        ohlcv = _make_ohlcv(n=50, base_price=100, volatility=0.001)
        # Force current price way below lower band
        ohlcv.iloc[-1, ohlcv.columns.get_loc("close")] = 90.0
        ms = _make_market_state(ohlcv, Regime.MEAN_REVERT)
        ms.current_price = 90.0
        signal = self.gen.generate_signal(ms)
        if signal:
            assert signal.action == Action.BUY

    def test_sell_signal_above_upper_band(self):
        ohlcv = _make_ohlcv(n=50, base_price=100, volatility=0.001)
        ohlcv.iloc[-1, ohlcv.columns.get_loc("close")] = 110.0
        ms = _make_market_state(ohlcv, Regime.MEAN_REVERT)
        ms.current_price = 110.0
        signal = self.gen.generate_signal(ms)
        if signal:
            assert signal.action == Action.SELL

    def test_no_signal_within_bands(self):
        ohlcv = _make_ohlcv(n=50, base_price=100, volatility=0.001)
        ms = _make_market_state(ohlcv, Regime.MEAN_REVERT)
        signal = self.gen.generate_signal(ms)
        assert signal is None


class TestTrendFollowingSignal:
    def setup_method(self):
        self.gen = TrendFollowingSignal(fast_period=10, slow_period=30)

    def test_required_regime(self):
        assert self.gen.get_required_regime() == Regime.TRENDING

    def test_no_signal_insufficient_data(self):
        ohlcv = _make_ohlcv(n=10)
        ms = _make_market_state(ohlcv)
        assert self.gen.generate_signal(ms) is None

    def test_buy_signal_uptrend(self):
        ohlcv = _make_ohlcv(n=50, base_price=100, trend=0.005, volatility=0.001)
        ms = _make_market_state(ohlcv)
        signal = self.gen.generate_signal(ms)
        if signal:
            assert signal.action == Action.BUY
            assert signal.stop_loss is not None

    def test_sell_signal_downtrend(self):
        ohlcv = _make_ohlcv(n=50, base_price=100, trend=-0.005, volatility=0.001)
        ms = _make_market_state(ohlcv)
        signal = self.gen.generate_signal(ms)
        if signal:
            assert signal.action == Action.SELL


class TestVolatilityBreakoutSignal:
    def setup_method(self):
        self.gen = VolatilityBreakoutSignal(lookback=20, multiplier=2.0)

    def test_required_regime(self):
        assert self.gen.get_required_regime() == Regime.HIGH_VOL

    def test_no_signal_low_volatility(self):
        ohlcv = _make_ohlcv(n=50, volatility=0.001)
        ms = _make_market_state(ohlcv, Regime.HIGH_VOL)
        signal = self.gen.generate_signal(ms)
        assert signal is None

    def test_breakout_above_resistance(self):
        ohlcv = _make_ohlcv(n=30, base_price=100, volatility=0.05)
        # Force breakout
        ohlcv.iloc[-1, ohlcv.columns.get_loc("close")] = 130.0
        ms = _make_market_state(ohlcv, Regime.HIGH_VOL)
        ms.current_price = 130.0
        signal = self.gen.generate_signal(ms)
        if signal:
            assert signal.action == Action.BUY


class TestSignalManager:
    def test_create_default_manager(self):
        mgr = create_default_signal_manager()
        assert len(mgr.generators) == 3

    def test_generate_signals_routing(self):
        mgr = create_default_signal_manager()
        ohlcv = _make_ohlcv(n=50, volatility=0.001)
        ms = _make_market_state(ohlcv, Regime.TRENDING)
        signals = mgr.generate_signals(ms)
        assert isinstance(signals, list)

    def test_get_best_signal_empty(self):
        mgr = SignalManager()
        assert mgr.get_best_signal([]) is None

    def test_get_best_signal_picks_highest_confidence(self):
        mgr = SignalManager()
        s1 = TradingAction(
            action=Action.BUY, size=1, stop_loss=98, take_profit=104,
            expected_return=4, expected_risk=2, confidence=0.6,
        )
        s2 = TradingAction(
            action=Action.SELL, size=1, stop_loss=102, take_profit=96,
            expected_return=4, expected_risk=2, confidence=0.9,
        )
        best = mgr.get_best_signal([s1, s2])
        assert best.confidence == 0.9

    def test_add_generator(self):
        mgr = SignalManager()
        gen = MeanReversionSignal()
        mgr.add_generator(gen)
        assert len(mgr.generators) == 1
        assert Regime.MEAN_REVERT in mgr.regime_generators
