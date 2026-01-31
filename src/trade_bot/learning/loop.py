"""Self-learning loop with salience model and strategy evaluation."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..core.models import Trade, MarketState, AccountState, Regime
from ..core.enums import Side
from .journal import TradeJournal
from .reward import RewardFunction, RewardComponents

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    strategy_name: str
    regime: Regime
    trade_count: int
    win_rate: float
    avg_r_multiple: float
    drawdown: float
    r_variance: float
    stability_score: float
    salience_score: float
    sharpe_ratio: float = 0.0
    last_updated: datetime = None


class SalienceModel:
    """
    Salience model for strategy evaluation.
    
    Salience is not win/loss-based but considers multiple factors:
    - Average R multiple
    - Win rate
    - Trade count (experience)
    - Drawdown
    - Regime consistency
    - Stability (variance of R)
    """
    
    def __init__(self):
        """Initialize salience model."""
        # Weights for different factors
        self.weights = {
            'avg_r_multiple': 0.4,
            'win_rate': 0.3,
            'trade_count': 0.2,
            'drawdown': -0.3,
            'r_variance': -0.2,
        }
        
        # Normalization parameters
        self.norm_params = {
            'avg_r_multiple': {'min': -2.0, 'max': 5.0},
            'win_rate': {'min': 0.0, 'max': 1.0},
            'trade_count': {'min': 0.0, 'max': 1000.0},  # Log scale
            'drawdown': {'min': 0.0, 'max': 0.5},
            'r_variance': {'min': 0.0, 'max': 10.0},
        }
        
        logger.info("Salience model initialized")
    
    def calculate_salience(self, metrics: StrategyMetrics) -> float:
        """
        Calculate salience score for a strategy.
        
        Args:
            metrics: Strategy performance metrics
            
        Returns:
            Salience score (0-1)
        """
        try:
            # Normalize each factor
            normalized_factors = {}
            
            # Average R multiple
            norm_r = self._normalize(
                metrics.avg_r_multiple,
                self.norm_params['avg_r_multiple']
            )
            normalized_factors['avg_r_multiple'] = norm_r
            
            # Win rate
            norm_win_rate = self._normalize(
                metrics.win_rate,
                self.norm_params['win_rate']
            )
            normalized_factors['win_rate'] = norm_win_rate
            
            # Trade count (log scale)
            log_trade_count = np.log1p(metrics.trade_count)
            norm_trade_count = self._normalize(
                log_trade_count,
                self.norm_params['trade_count']
            )
            normalized_factors['trade_count'] = norm_trade_count
            
            # Drawdown (negative factor)
            norm_drawdown = self._normalize(
                metrics.drawdown,
                self.norm_params['drawdown']
            )
            normalized_factors['drawdown'] = norm_drawdown
            
            # R variance (negative factor)
            norm_r_variance = self._normalize(
                metrics.r_variance,
                self.norm_params['r_variance']
            )
            normalized_factors['r_variance'] = norm_r_variance
            
            # Calculate weighted sum
            salience = 0.0
            for factor, value in normalized_factors.items():
                weight = self.weights[factor]
                salience += weight * value
            
            # Apply sigmoid to get 0-1 range
            salience = 1 / (1 + np.exp(-salience))
            
            return float(salience)
            
        except Exception as e:
            logger.error(f"Error calculating salience: {e}")
            return 0.5  # Neutral score
    
    def _normalize(self, value: float, params: Dict[str, float]) -> float:
        """Normalize value to 0-1 range."""
        min_val, max_val = params['min'], params['max']
        
        if max_val <= min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def get_recommendation(self, salience: float) -> str:
        """Get recommendation based on salience score."""
        if salience < 0.2:
            return "ARCHIVE"
        elif salience < 0.8:
            return "ACTIVE"
        else:
            return "PROMOTE"


class LearningLoop:
    """
    Self-learning loop for strategy improvement.
    
    Implements: TRADE → JOURNAL → EXTRACT → SCORE → APPLY
    """
    
    def __init__(self, journal: TradeJournal, reward_function: Optional[RewardFunction] = None):
        """Initialize learning loop."""
        self.journal = journal
        self.salience_model = SalienceModel()
        self.reward_function = reward_function or RewardFunction()

        # Strategy registry
        self.strategies: Dict[str, Dict[Regime, StrategyMetrics]] = {}

        # Reward history
        self.reward_history: List[RewardComponents] = []

        # Learning parameters
        self.min_trades_for_evaluation = 20
        self.decay_factor = 0.95  # Decay for old strategies

        logger.info("Learning loop initialized")
    
    def process_trade(
        self,
        trade: Trade,
        strategy_name: str,
        account_state: Optional[AccountState] = None,
        drawdown_increase: float = 0.0,
        daily_loss_pct: float = 0.0,
        risk_violation: bool = False,
    ):
        """Process a new trade, compute reward, and update learning."""
        try:
            # Record trade in journal
            market_state = self._reconstruct_market_state(trade)
            self.journal.record_trade(trade, market_state)

            # Compute reward
            equity = account_state.equity if account_state else 10000.0
            position_size_pct = (
                (trade.quantity * trade.entry_price) / equity if equity > 0 else 0
            )
            reward = self.reward_function.calculate(
                delta_equity=trade.pnl,
                drawdown_increase=drawdown_increase,
                volatility_exposure=trade.volatility_percentile / 100.0,
                position_size_pct=position_size_pct,
                transaction_cost=trade.slippage * trade.quantity * trade.entry_price / 10000,
                daily_loss_pct=daily_loss_pct,
                risk_violation=risk_violation,
            )
            self.reward_history.append(reward)

            # Update strategy metrics
            self._update_strategy_metrics(strategy_name, trade)

            # Evaluate strategy
            self._evaluate_strategy(strategy_name, trade.regime)

            logger.info(
                f"Processed trade {trade.id} for strategy {strategy_name}, "
                f"reward={reward.total:.2f}"
            )

        except Exception as e:
            logger.error(f"Error processing trade: {e}")
    
    def _reconstruct_market_state(self, trade: Trade) -> MarketState:
        """Reconstruct market state from trade context."""
        # In a real implementation, this would fetch the actual market state
        # For now, create a minimal market state from trade data
        from ..core.models import MarketState
        import pandas as pd
        
        # Create minimal OHLCV data
        dates = pd.date_range(
            start=trade.entry_time - timedelta(days=1),
            end=trade.entry_time,
            freq='1h'
        )
        
        ohlcv = pd.DataFrame({
            'open': [trade.entry_price] * len(dates),
            'high': [trade.entry_price * 1.01] * len(dates),
            'low': [trade.entry_price * 0.99] * len(dates),
            'close': [trade.entry_price] * len(dates),
            'volume': [1000] * len(dates),
        }, index=dates)
        
        return MarketState(
            ohlcv=ohlcv,
            current_price=trade.entry_price,
            atr=trade.stop_loss * 0.02 if trade.stop_loss else 0.01,
            realized_volatility=0.2,
            spread=0.0001,
            order_book_imbalance=0.0,
            volume_delta=0.0,
            liquidity_score=trade.liquidity_score,
            regime_label=trade.regime,
            volatility_percentile=trade.volatility_percentile,
        )
    
    def _update_strategy_metrics(self, strategy_name: str, trade: Trade):
        """Update strategy metrics with new trade."""
        regime = trade.regime
        
        # Initialize strategy if not exists
        if strategy_name not in self.strategies:
            self.strategies[strategy_name] = {}
        
        if regime not in self.strategies[strategy_name]:
            self.strategies[strategy_name][regime] = StrategyMetrics(
                strategy_name=strategy_name,
                regime=regime,
                trade_count=0,
                win_rate=0.0,
                avg_r_multiple=0.0,
                drawdown=0.0,
                r_variance=0.0,
                stability_score=0.0,
                salience_score=0.0,
                last_updated=datetime.now(),
            )
        
        # Get recent trades for this strategy and regime
        recent_trades = self.journal.get_trades(
            regime=regime,
            limit=100  # Last 100 trades
        )
        
        # Filter by strategy (in a real implementation, trades would have strategy field)
        strategy_trades = recent_trades  # Simplified
        
        if not strategy_trades:
            return
        
        # Calculate metrics
        trade_count = len(strategy_trades)
        winning_trades = len([t for t in strategy_trades if t.pnl > 0])
        win_rate = winning_trades / trade_count if trade_count > 0 else 0
        
        r_multiples = [t.r_multiple for t in strategy_trades if t.r_multiple != 0]
        avg_r_multiple = np.mean(r_multiples) if r_multiples else 0
        r_variance = np.var(r_multiples) if r_multiples else 0
        
        # Calculate drawdown
        max_drawdown = self._calculate_strategy_drawdown(strategy_trades)
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(strategy_trades)
        
        # Calculate Sharpe ratio from trade PnLs
        sharpe_ratio = self._calculate_strategy_sharpe(strategy_trades)

        # Update metrics
        metrics = self.strategies[strategy_name][regime]
        metrics.trade_count = trade_count
        metrics.win_rate = win_rate
        metrics.avg_r_multiple = avg_r_multiple
        metrics.drawdown = max_drawdown
        metrics.r_variance = r_variance
        metrics.stability_score = stability_score
        metrics.sharpe_ratio = sharpe_ratio
        metrics.last_updated = datetime.now()
    
    def _evaluate_strategy(self, strategy_name: str, regime: Regime):
        """Evaluate strategy and calculate salience."""
        if strategy_name not in self.strategies:
            return
        
        if regime not in self.strategies[strategy_name]:
            return
        
        metrics = self.strategies[strategy_name][regime]
        
        # Only evaluate if we have enough trades
        if metrics.trade_count < self.min_trades_for_evaluation:
            return
        
        # Calculate salience
        salience = self.salience_model.calculate_salience(metrics)
        metrics.salience_score = salience
        
        # Get recommendation
        recommendation = self.salience_model.get_recommendation(salience)
        
        logger.info(
            f"Strategy {strategy_name} in {regime.value} regime: "
            f"salience={salience:.3f}, recommendation={recommendation}"
        )
        
        # Store recommendation in journal
        self._store_strategy_recommendation(metrics, recommendation)
    
    def _calculate_strategy_drawdown(self, trades: List[Trade]) -> float:
        """Calculate drawdown for strategy trades."""
        if not trades:
            return 0.0
        
        # Sort by entry time
        sorted_trades = sorted(trades, key=lambda t: t.entry_time)
        
        # Calculate cumulative P&L
        cumulative_pnl = []
        running_pnl = 0
        
        for trade in sorted_trades:
            running_pnl += trade.pnl
            cumulative_pnl.append(running_pnl)
        
        if not cumulative_pnl:
            return 0.0
        
        # Calculate drawdown
        peak = cumulative_pnl[0]
        max_drawdown = 0.0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            
            drawdown = (peak - pnl) / peak if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_strategy_sharpe(self, trades: List[Trade], annualization: float = 252**0.5) -> float:
        """Calculate Sharpe ratio from trade PnLs."""
        if len(trades) < 2:
            return 0.0
        pnls = np.array([t.pnl for t in trades])
        mean = pnls.mean()
        std = pnls.std()
        if std == 0:
            return 0.0
        return float(mean / std * annualization)

    def _calculate_stability_score(self, trades: List[Trade]) -> float:
        """Calculate stability score based on consistency."""
        if len(trades) < 10:
            return 0.5
        
        # Get R multiples
        r_multiples = [t.r_multiple for t in trades if t.r_multiple != 0]
        if len(r_multiples) < 10:
            return 0.5
        
        # Calculate consistency metrics
        avg_r = np.mean(r_multiples)
        std_r = np.std(r_multiples)
        
        # Stability is inverse of coefficient of variation
        cv = std_r / avg_r if avg_r != 0 else float('inf')
        stability = 1 / (1 + cv)  # Convert to 0-1 scale
        
        return float(stability)
    
    def _store_strategy_recommendation(self, metrics: StrategyMetrics, recommendation: str):
        """Store strategy recommendation in journal."""
        try:
            # This would update the strategy_performance table in the journal
            # For now, just log the recommendation
            logger.info(
                f"Strategy recommendation stored: {metrics.strategy_name} "
                f"({metrics.regime.value}) -> {recommendation}"
            )
            
        except Exception as e:
            logger.error(f"Error storing strategy recommendation: {e}")
    
    def get_strategy_rankings(self, regime: Optional[Regime] = None) -> List[StrategyMetrics]:
        """Get ranked strategies by salience."""
        all_metrics = []
        
        for strategy_name, regime_dict in self.strategies.items():
            for reg, metrics in regime_dict.items():
                if regime is None or reg == regime:
                    if metrics.trade_count >= self.min_trades_for_evaluation:
                        all_metrics.append(metrics)
        
        # Sort by salience score
        ranked = sorted(all_metrics, key=lambda m: m.salience_score, reverse=True)
        return ranked
    
    def get_best_strategy(self, regime: Regime) -> Optional[StrategyMetrics]:
        """Get best strategy for a specific regime."""
        rankings = self.get_strategy_rankings(regime)
        return rankings[0] if rankings else None
    
    def should_promote_strategy(self, strategy_name: str, regime: Regime) -> bool:
        """Check if strategy should be promoted to live trading (§8.2).

        Criteria:
        - Sharpe > 1.2
        - Max DD < 10%
        - >= 200 trades
        - Stable monthly returns (stability_score > 0.4)
        - Profit across multiple regimes (at least 2 regimes profitable)
        """
        if strategy_name not in self.strategies:
            return False

        if regime not in self.strategies[strategy_name]:
            return False

        metrics = self.strategies[strategy_name][regime]

        # Minimum trade count
        if metrics.trade_count < 200:
            return False

        # Sharpe ratio threshold
        if metrics.sharpe_ratio < 1.2:
            return False

        # Max drawdown threshold
        if metrics.drawdown > 0.10:
            return False

        # Stability (low coefficient of variation)
        if metrics.stability_score < 0.4:
            return False

        # Must be profitable across multiple regimes
        profitable_regimes = 0
        for reg, m in self.strategies.get(strategy_name, {}).items():
            if m.avg_r_multiple > 0 and m.trade_count >= 20:
                profitable_regimes += 1
        if profitable_regimes < 2:
            return False

        return True
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning status."""
        total_strategies = len(self.strategies)
        total_regime_combinations = sum(
            len(regime_dict) for regime_dict in self.strategies.values()
        )
        
        # Count strategies by recommendation
        recommendation_counts = {"ARCHIVE": 0, "ACTIVE": 0, "PROMOTE": 0}
        
        for strategy_name, regime_dict in self.strategies.items():
            for regime, metrics in regime_dict.items():
                if metrics.trade_count >= self.min_trades_for_evaluation:
                    rec = self.salience_model.get_recommendation(metrics.salience_score)
                    recommendation_counts[rec] += 1
        
        return {
            'total_strategies': total_strategies,
            'total_regime_combinations': total_regime_combinations,
            'recommendation_counts': recommendation_counts,
            'min_trades_for_evaluation': self.min_trades_for_evaluation,
        }
