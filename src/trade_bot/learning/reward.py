"""Reward function for reinforcement learning / strategy evaluation.

Implements the reward function from plan §7:

    reward = Δequity
             - λ1 * drawdown_increase
             - λ2 * volatility_exposure
             - λ3 * position_size_penalty
             - λ4 * transaction_cost

Additional penalties:
    - Daily loss violation  → large negative reward
    - Risk violation        → episode termination signal
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Breakdown of reward into its components."""
    delta_equity: float
    drawdown_penalty: float
    volatility_penalty: float
    position_size_penalty: float
    transaction_cost_penalty: float
    daily_loss_violation_penalty: float
    risk_violation: bool
    total: float


class RewardFunction:
    """Risk-adjusted reward function.

    Goal: maximize risk-adjusted return, not raw PnL.
    """

    def __init__(
        self,
        lambda_drawdown: float = 2.0,
        lambda_volatility: float = 1.0,
        lambda_position_size: float = 0.5,
        lambda_transaction: float = 1.0,
        daily_loss_violation_penalty: float = -100.0,
        max_daily_loss_pct: float = 0.05,
        max_position_size_pct: float = 0.02,
    ):
        self.lambda_drawdown = lambda_drawdown
        self.lambda_volatility = lambda_volatility
        self.lambda_position_size = lambda_position_size
        self.lambda_transaction = lambda_transaction
        self.daily_loss_violation_penalty = daily_loss_violation_penalty
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_size_pct = max_position_size_pct

    def calculate(
        self,
        delta_equity: float,
        drawdown_increase: float,
        volatility_exposure: float,
        position_size_pct: float,
        transaction_cost: float,
        daily_loss_pct: float,
        risk_violation: bool = False,
    ) -> RewardComponents:
        """Calculate the reward for a single step / trade.

        Args:
            delta_equity: Change in account equity (positive = profit).
            drawdown_increase: Increase in drawdown since last step (≥0).
            volatility_exposure: Current volatility percentile / 100 (0-1).
            position_size_pct: Position risk as fraction of equity (0-1).
            transaction_cost: Absolute cost of the transaction.
            daily_loss_pct: Current daily loss as fraction (0-1, positive = loss).
            risk_violation: Whether a hard risk rule was violated.

        Returns:
            RewardComponents with breakdown and total.
        """
        # Base reward
        reward = delta_equity

        # Penalties
        dd_penalty = self.lambda_drawdown * max(0.0, drawdown_increase)
        vol_penalty = self.lambda_volatility * max(0.0, volatility_exposure)

        size_excess = max(0.0, position_size_pct - self.max_position_size_pct)
        size_penalty = self.lambda_position_size * size_excess

        tx_penalty = self.lambda_transaction * abs(transaction_cost)

        reward -= dd_penalty + vol_penalty + size_penalty + tx_penalty

        # Daily loss violation
        daily_violation = 0.0
        if daily_loss_pct >= self.max_daily_loss_pct:
            daily_violation = self.daily_loss_violation_penalty
            reward += daily_violation

        # Risk violation → signal episode termination
        if risk_violation:
            reward += self.daily_loss_violation_penalty * 2

        return RewardComponents(
            delta_equity=delta_equity,
            drawdown_penalty=dd_penalty,
            volatility_penalty=vol_penalty,
            position_size_penalty=size_penalty,
            transaction_cost_penalty=tx_penalty,
            daily_loss_violation_penalty=daily_violation,
            risk_violation=risk_violation,
            total=reward,
        )
