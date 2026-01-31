"""Unit tests for TradeJournal."""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from trade_bot.core.models import Trade, MarketState
from trade_bot.core.enums import Side, Regime
from trade_bot.learning.journal import TradeJournal

import pandas as pd


@pytest.fixture
def journal(tmp_path):
    db_path = tmp_path / "test_journal.db"
    return TradeJournal(db_path=str(db_path))


def _make_trade(trade_id="t1", pnl=10.0, regime=Regime.TRENDING, entry_price=100.0):
    return Trade(
        id=trade_id,
        symbol="BTC/USDT",
        side=Side.LONG,
        quantity=1.0,
        entry_price=entry_price,
        exit_price=entry_price + pnl,
        entry_time=datetime.now() - timedelta(hours=1),
        exit_time=datetime.now(),
        duration=timedelta(hours=1),
        pnl=pnl,
        pnl_pct=pnl / entry_price * 100,
        r_multiple=pnl / 2.0 if pnl != 0 else 0,
        mae=abs(min(0, pnl)),
        mfe=max(0, pnl),
        stop_loss=entry_price - 2.0,
        take_profit=entry_price + 4.0,
        regime=regime,
        volatility_percentile=50.0,
        liquidity_score=0.8,
        slippage=1.0,
        confidence=0.8,
    )


def _make_market_state(price=100.0, regime=Regime.TRENDING):
    dates = pd.date_range(start="2023-01-01", periods=50, freq="1h")
    ohlcv = pd.DataFrame(
        {"open": [price]*50, "high": [price*1.01]*50, "low": [price*0.99]*50,
         "close": [price]*50, "volume": [1000]*50},
        index=dates,
    )
    return MarketState(
        ohlcv=ohlcv, current_price=price, atr=2.0,
        realized_volatility=0.2, spread=0.001,
        order_book_imbalance=0.1, volume_delta=100,
        liquidity_score=0.8, regime_label=regime,
        volatility_percentile=50.0,
    )


class TestTradeJournal:
    def test_record_and_get_trade(self, journal):
        trade = _make_trade()
        ms = _make_market_state()
        journal.record_trade(trade, ms)

        trades = journal.get_trades(symbol="BTC/USDT")
        assert len(trades) == 1
        assert trades[0].id == "t1"

    def test_get_trades_with_regime_filter(self, journal):
        ms = _make_market_state()
        journal.record_trade(_make_trade("t1", regime=Regime.TRENDING), ms)
        journal.record_trade(_make_trade("t2", regime=Regime.HIGH_VOL), ms)

        trades = journal.get_trades(regime=Regime.TRENDING)
        assert len(trades) == 1

    def test_get_trades_with_limit(self, journal):
        ms = _make_market_state()
        for i in range(5):
            journal.record_trade(_make_trade(f"t{i}"), ms)

        trades = journal.get_trades(limit=3)
        assert len(trades) == 3

    def test_record_market_context(self, journal):
        ms = _make_market_state()
        journal.record_market_context(ms, "BTC/USDT")
        # No error means success

    def test_update_daily_performance(self, journal):
        trades = [_make_trade("t1", pnl=10), _make_trade("t2", pnl=-5)]
        journal.update_daily_performance(
            datetime.now(), 10000.0, 10000.0, 5.0, trades
        )

    def test_get_performance_stats_empty(self, journal):
        stats = journal.get_performance_stats()
        assert "error" in stats

    def test_get_performance_stats(self, journal):
        ms = _make_market_state()
        for i in range(5):
            pnl = 10.0 if i % 2 == 0 else -5.0
            journal.record_trade(_make_trade(f"t{i}", pnl=pnl), ms)
        stats = journal.get_performance_stats()
        assert stats["total_trades"] == 5

    def test_get_regime_performance(self, journal):
        ms = _make_market_state()
        journal.record_trade(_make_trade("t1", regime=Regime.TRENDING), ms)
        journal.record_trade(_make_trade("t2", regime=Regime.HIGH_VOL), ms)
        perf = journal.get_regime_performance()
        assert len(perf) == 2

    def test_get_recent_daily_performance(self, journal):
        trades = [_make_trade("t1", pnl=10)]
        journal.update_daily_performance(
            datetime.now(), 10000.0, 10000.0, 10.0, trades
        )
        recent = journal.get_recent_daily_performance(days=3)
        assert len(recent) == 1
        assert recent[0]["pnl"] == 10.0

    def test_export_to_csv(self, journal, tmp_path):
        ms = _make_market_state()
        journal.record_trade(_make_trade("t1"), ms)
        csv_path = str(tmp_path / "export.csv")
        journal.export_to_csv(csv_path, "trades")
        assert Path(csv_path).exists()
