"""Unit tests for DeploymentManager."""

import pytest
from datetime import datetime

from trade_bot.deployment.manager import (
    DeploymentManager,
    DeploymentStage,
    StageMetrics,
    STAGE_CAPITAL_PCT,
)


def _make_metrics(**kwargs):
    defaults = dict(
        sharpe_ratio=1.5,
        max_drawdown=0.05,
        avg_slippage_bps=3.0,
        trade_count=100,
        entered_at=datetime.now(),
    )
    defaults.update(kwargs)
    return StageMetrics(**defaults)


class TestDeploymentManager:
    def test_initial_stage(self):
        dm = DeploymentManager()
        assert dm.current_stage == DeploymentStage.STAGE_1
        assert dm.capital_pct == 0.05

    def test_set_baseline(self):
        dm = DeploymentManager()
        dm.set_baseline(_make_metrics())
        assert dm.baseline_metrics is not None

    def test_should_advance_true(self):
        dm = DeploymentManager(min_trades_per_stage=10, min_sharpe=1.0)
        dm.set_baseline(_make_metrics())
        assert dm.should_advance(_make_metrics()) is True

    def test_should_advance_not_enough_trades(self):
        dm = DeploymentManager(min_trades_per_stage=200)
        assert dm.should_advance(_make_metrics(trade_count=50)) is False

    def test_should_advance_low_sharpe(self):
        dm = DeploymentManager(min_sharpe=2.0)
        assert dm.should_advance(_make_metrics(sharpe_ratio=1.0)) is False

    def test_should_advance_at_max_stage(self):
        dm = DeploymentManager()
        dm.current_stage = DeploymentStage.STAGE_4
        assert dm.should_advance(_make_metrics()) is False

    def test_should_advance_dd_exceeds_baseline(self):
        dm = DeploymentManager(min_trades_per_stage=10, max_dd_multiplier=1.5)
        dm.set_baseline(_make_metrics(max_drawdown=0.05))
        assert dm.should_advance(_make_metrics(max_drawdown=0.10)) is False

    def test_advance(self):
        dm = DeploymentManager()
        dm.advance()
        assert dm.current_stage == DeploymentStage.STAGE_2
        assert dm.capital_pct == 0.10

    def test_advance_through_all_stages(self):
        dm = DeploymentManager()
        dm.advance()  # -> STAGE_2
        dm.advance()  # -> STAGE_3
        dm.advance()  # -> STAGE_4
        assert dm.capital_pct == 1.00
        dm.advance()  # No-op
        assert dm.current_stage == DeploymentStage.STAGE_4

    def test_should_rollback_at_stage_1(self):
        dm = DeploymentManager()
        assert dm.should_rollback(_make_metrics(sharpe_ratio=0.1)) is False

    def test_should_rollback_sharpe_degraded(self):
        dm = DeploymentManager(min_sharpe=1.0)
        dm.current_stage = DeploymentStage.STAGE_2
        assert dm.should_rollback(_make_metrics(sharpe_ratio=0.3)) is True

    def test_should_rollback_dd_exceeded(self):
        dm = DeploymentManager(max_dd_multiplier=1.5)
        dm.set_baseline(_make_metrics(max_drawdown=0.05))
        dm.current_stage = DeploymentStage.STAGE_3
        assert dm.should_rollback(_make_metrics(max_drawdown=0.10)) is True

    def test_should_rollback_slippage_increase(self):
        dm = DeploymentManager(max_slippage_increase_bps=5.0)
        dm.set_baseline(_make_metrics(avg_slippage_bps=2.0))
        dm.current_stage = DeploymentStage.STAGE_2
        assert dm.should_rollback(_make_metrics(avg_slippage_bps=10.0)) is True

    def test_rollback(self):
        dm = DeploymentManager()
        dm.advance()  # -> STAGE_2
        dm.rollback()  # -> STAGE_1
        assert dm.current_stage == DeploymentStage.STAGE_1

    def test_rollback_at_stage_1_noop(self):
        dm = DeploymentManager()
        dm.rollback()
        assert dm.current_stage == DeploymentStage.STAGE_1

    def test_evaluate_and_act_advance(self):
        dm = DeploymentManager(min_trades_per_stage=10, min_sharpe=1.0)
        dm.set_baseline(_make_metrics())
        action = dm.evaluate_and_act(_make_metrics())
        assert action == "advance"
        assert dm.current_stage == DeploymentStage.STAGE_2

    def test_evaluate_and_act_rollback(self):
        dm = DeploymentManager(min_sharpe=1.0)
        dm.current_stage = DeploymentStage.STAGE_2
        action = dm.evaluate_and_act(_make_metrics(sharpe_ratio=0.2))
        assert action == "rollback"
        assert dm.current_stage == DeploymentStage.STAGE_1

    def test_evaluate_and_act_hold(self):
        dm = DeploymentManager(min_trades_per_stage=200)
        action = dm.evaluate_and_act(_make_metrics(trade_count=50))
        assert action == "hold"

    def test_get_status(self):
        dm = DeploymentManager()
        status = dm.get_status()
        assert status["current_stage"] == "stage_1"
        assert status["capital_pct"] == 0.05
