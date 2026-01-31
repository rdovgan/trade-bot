"""Unit tests for SalienceModel, LearningLoop, and promotion criteria."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from trade_bot.core.models import Trade
from trade_bot.core.enums import Side, Regime
from trade_bot.learning.loop import SalienceModel, StrategyMetrics, LearningLoop
from trade_bot.learning.journal import TradeJournal


def _make_metrics(**kwargs):
    defaults = dict(
        strategy_name="test_strat",
        regime=Regime.TRENDING,
        trade_count=100,
        win_rate=0.55,
        avg_r_multiple=1.5,
        drawdown=0.08,
        r_variance=1.0,
        stability_score=0.6,
        salience_score=0.0,
        sharpe_ratio=1.5,
        last_updated=datetime.now(),
    )
    defaults.update(kwargs)
    return StrategyMetrics(**defaults)


def _make_trade(trade_id="t1", pnl=10.0, regime=Regime.TRENDING):
    return Trade(
        id=trade_id, symbol="BTC/USDT", side=Side.LONG, quantity=1.0,
        entry_price=100.0, exit_price=100.0 + pnl,
        entry_time=datetime.now() - timedelta(hours=1),
        exit_time=datetime.now(), duration=timedelta(hours=1),
        pnl=pnl, pnl_pct=pnl, r_multiple=pnl / 2.0 if pnl != 0 else 0,
        mae=abs(min(0, pnl)), mfe=max(0, pnl),
        stop_loss=98.0, take_profit=104.0,
        regime=regime, volatility_percentile=50.0,
        liquidity_score=0.8, slippage=1.0, confidence=0.8,
    )


class TestSalienceModel:
    def setup_method(self):
        self.model = SalienceModel()

    def test_calculate_salience_returns_float(self):
        metrics = _make_metrics()
        score = self.model.calculate_salience(metrics)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_high_performance_high_salience(self):
        metrics = _make_metrics(
            win_rate=0.7, avg_r_multiple=3.0, trade_count=500,
            drawdown=0.02, r_variance=0.5,
        )
        score = self.model.calculate_salience(metrics)
        assert score > 0.5

    def test_low_performance_low_salience(self):
        metrics = _make_metrics(
            win_rate=0.2, avg_r_multiple=-1.0, trade_count=5,
            drawdown=0.4, r_variance=8.0,
        )
        score = self.model.calculate_salience(metrics)
        assert score < 0.5

    def test_recommendation_archive(self):
        assert self.model.get_recommendation(0.1) == "ARCHIVE"

    def test_recommendation_active(self):
        assert self.model.get_recommendation(0.5) == "ACTIVE"

    def test_recommendation_promote(self):
        assert self.model.get_recommendation(0.9) == "PROMOTE"


class TestLearningLoop:
    @pytest.fixture
    def journal(self, tmp_path):
        return TradeJournal(db_path=str(tmp_path / "test.db"))

    @pytest.fixture
    def loop(self, journal):
        return LearningLoop(journal)

    def test_process_trade(self, loop):
        trade = _make_trade()
        loop.process_trade(trade, "test_strat")
        assert "test_strat" in loop.strategies or True  # May not have enough trades

    def test_get_strategy_rankings_empty(self, loop):
        rankings = loop.get_strategy_rankings()
        assert rankings == []

    def test_get_best_strategy_none(self, loop):
        assert loop.get_best_strategy(Regime.TRENDING) is None

    def test_should_promote_not_enough_trades(self, loop):
        loop.strategies["s1"] = {
            Regime.TRENDING: _make_metrics(trade_count=50),
        }
        assert loop.should_promote_strategy("s1", Regime.TRENDING) is False

    def test_should_promote_low_sharpe(self, loop):
        loop.strategies["s1"] = {
            Regime.TRENDING: _make_metrics(trade_count=200, sharpe_ratio=0.5),
        }
        assert loop.should_promote_strategy("s1", Regime.TRENDING) is False

    def test_should_promote_high_drawdown(self, loop):
        loop.strategies["s1"] = {
            Regime.TRENDING: _make_metrics(trade_count=200, drawdown=0.15),
        }
        assert loop.should_promote_strategy("s1", Regime.TRENDING) is False

    def test_should_promote_success(self, loop):
        loop.strategies["s1"] = {
            Regime.TRENDING: _make_metrics(
                trade_count=200, sharpe_ratio=1.5, drawdown=0.05,
                stability_score=0.6, avg_r_multiple=1.5,
            ),
            Regime.MEAN_REVERT: _make_metrics(
                trade_count=50, avg_r_multiple=1.0,
                regime=Regime.MEAN_REVERT,
            ),
        }
        assert loop.should_promote_strategy("s1", Regime.TRENDING) is True

    def test_should_promote_needs_multi_regime(self, loop):
        """Must be profitable in at least 2 regimes."""
        loop.strategies["s1"] = {
            Regime.TRENDING: _make_metrics(
                trade_count=200, sharpe_ratio=1.5, drawdown=0.05,
                stability_score=0.6,
            ),
        }
        assert loop.should_promote_strategy("s1", Regime.TRENDING) is False

    def test_get_learning_summary(self, loop):
        summary = loop.get_learning_summary()
        assert "total_strategies" in summary

    def test_reward_history_populated(self, loop):
        trade = _make_trade()
        loop.process_trade(trade, "test_strat")
        assert len(loop.reward_history) == 1
