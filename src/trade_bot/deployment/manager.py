"""Production deployment manager (§15).

Rollout stages: 5% → 10% → 25% → 100% capital.

Rollback if:
- Sharpe degrades below threshold
- DD exceeds historical max
- Slippage increases materially
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Capital allocation stages."""
    STAGE_1 = "stage_1"  # 5%
    STAGE_2 = "stage_2"  # 10%
    STAGE_3 = "stage_3"  # 25%
    STAGE_4 = "stage_4"  # 100%


STAGE_CAPITAL_PCT: Dict[DeploymentStage, float] = {
    DeploymentStage.STAGE_1: 0.05,
    DeploymentStage.STAGE_2: 0.10,
    DeploymentStage.STAGE_3: 0.25,
    DeploymentStage.STAGE_4: 1.00,
}

STAGE_ORDER = [
    DeploymentStage.STAGE_1,
    DeploymentStage.STAGE_2,
    DeploymentStage.STAGE_3,
    DeploymentStage.STAGE_4,
]


@dataclass
class StageMetrics:
    """Metrics recorded at a deployment stage."""
    sharpe_ratio: float
    max_drawdown: float
    avg_slippage_bps: float
    trade_count: int
    entered_at: datetime


class DeploymentManager:
    """Manages staged deployment of trading strategies.

    Controls what fraction of capital is allocated based on
    live performance at each stage.
    """

    def __init__(
        self,
        min_trades_per_stage: int = 50,
        min_sharpe: float = 1.0,
        max_dd_multiplier: float = 1.5,
        max_slippage_increase_bps: float = 5.0,
    ):
        self.min_trades_per_stage = min_trades_per_stage
        self.min_sharpe = min_sharpe
        self.max_dd_multiplier = max_dd_multiplier
        self.max_slippage_increase_bps = max_slippage_increase_bps

        self.current_stage = DeploymentStage.STAGE_1
        self.stage_history: Dict[DeploymentStage, StageMetrics] = {}
        self.baseline_metrics: Optional[StageMetrics] = None

        logger.info(f"Deployment manager initialized at {self.current_stage.value}")

    @property
    def capital_pct(self) -> float:
        """Current capital allocation percentage."""
        return STAGE_CAPITAL_PCT[self.current_stage]

    def set_baseline(self, metrics: StageMetrics):
        """Set baseline metrics (from paper trading / backtest)."""
        self.baseline_metrics = metrics
        logger.info(
            f"Baseline set: Sharpe={metrics.sharpe_ratio:.2f}, "
            f"DD={metrics.max_drawdown:.2%}, slippage={metrics.avg_slippage_bps:.1f}bps"
        )

    def record_stage_metrics(self, metrics: StageMetrics):
        """Record metrics for the current stage."""
        self.stage_history[self.current_stage] = metrics

    def should_advance(self, current_metrics: StageMetrics) -> bool:
        """Check if we should advance to the next stage."""
        if self.current_stage == DeploymentStage.STAGE_4:
            return False  # Already at max

        if current_metrics.trade_count < self.min_trades_per_stage:
            return False

        if current_metrics.sharpe_ratio < self.min_sharpe:
            return False

        # DD must not exceed baseline * multiplier
        if self.baseline_metrics:
            max_allowed_dd = self.baseline_metrics.max_drawdown * self.max_dd_multiplier
            if current_metrics.max_drawdown > max_allowed_dd:
                return False

        return True

    def should_rollback(self, current_metrics: StageMetrics) -> bool:
        """Check if we should rollback to a previous stage.

        Rollback if:
        - Sharpe degrades significantly
        - DD exceeds historical max
        - Slippage increases materially
        """
        if self.current_stage == DeploymentStage.STAGE_1:
            return False  # Can't go lower

        if current_metrics.sharpe_ratio < self.min_sharpe * 0.5:
            logger.warning(
                f"Sharpe degraded to {current_metrics.sharpe_ratio:.2f} — rollback"
            )
            return True

        if self.baseline_metrics:
            max_dd = self.baseline_metrics.max_drawdown * self.max_dd_multiplier
            if current_metrics.max_drawdown > max_dd:
                logger.warning(
                    f"DD {current_metrics.max_drawdown:.2%} exceeds "
                    f"threshold {max_dd:.2%} — rollback"
                )
                return True

            slippage_increase = (
                current_metrics.avg_slippage_bps
                - self.baseline_metrics.avg_slippage_bps
            )
            if slippage_increase > self.max_slippage_increase_bps:
                logger.warning(
                    f"Slippage increased by {slippage_increase:.1f}bps — rollback"
                )
                return True

        return False

    def advance(self):
        """Advance to the next deployment stage."""
        idx = STAGE_ORDER.index(self.current_stage)
        if idx < len(STAGE_ORDER) - 1:
            old = self.current_stage
            self.current_stage = STAGE_ORDER[idx + 1]
            logger.info(
                f"Advanced deployment: {old.value} -> {self.current_stage.value} "
                f"({self.capital_pct:.0%} capital)"
            )

    def rollback(self):
        """Rollback to the previous deployment stage."""
        idx = STAGE_ORDER.index(self.current_stage)
        if idx > 0:
            old = self.current_stage
            self.current_stage = STAGE_ORDER[idx - 1]
            logger.info(
                f"Rolled back deployment: {old.value} -> {self.current_stage.value} "
                f"({self.capital_pct:.0%} capital)"
            )

    def evaluate_and_act(self, current_metrics: StageMetrics) -> str:
        """Evaluate metrics and advance/rollback/hold. Returns action taken."""
        self.record_stage_metrics(current_metrics)

        if self.should_rollback(current_metrics):
            self.rollback()
            return "rollback"

        if self.should_advance(current_metrics):
            self.advance()
            return "advance"

        return "hold"

    def get_status(self) -> Dict:
        """Get deployment status."""
        return {
            "current_stage": self.current_stage.value,
            "capital_pct": self.capital_pct,
            "stages_completed": [
                s.value for s in STAGE_ORDER
                if s in self.stage_history
            ],
        }
