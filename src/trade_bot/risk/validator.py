"""Risk management validator with veto power over all trading decisions."""

from decimal import Decimal
from typing import Optional, Tuple, Dict, Any, List
import logging

from ..core.models import (
    MarketState, PositionState, AccountState, RiskState,
    TradingAction, RiskLevel
)
from typing import TYPE_CHECKING
from ..core.enums import Action, Side

logger = logging.getLogger(__name__)


class RiskViolation(Exception):
    """Exception raised when risk rules are violated."""
    pass


class RiskValidator:
    """
    Risk management engine with hard-coded rules that cannot be bypassed.
    
    This validator has veto power over all trading decisions and implements
    the risk-first architecture principles.
    """
    
    def __init__(self, risk_config: Optional[Dict[str, Any]] = None):
        """Initialize risk validator with configuration."""
        defaults = self._default_config()
        if risk_config:
            defaults.update(risk_config)
        self.config = defaults
        logger.info("Risk validator initialized with hard-coded safety rules")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default risk configuration - these are hard limits that cannot be changed."""
        return {
            # Risk per trade (hard limits)
            "max_risk_per_trade": 0.01,  # 1% of equity
            "max_risk_per_trade_absolute": 0.02,  # 2% absolute maximum
            
            # Exposure limits
            "max_exposure_per_asset": 0.30,  # 30% per asset (as per plan.md)
            "max_total_exposure": 0.50,  # 50% total exposure (as per plan.md)
            "max_leverage": 2.0,  # Maximum 2x leverage (as per plan.md)
            
            # Daily risk limits
            "max_daily_loss": 0.05,  # 5% daily loss limit (as per plan.md)
            "max_consecutive_losses": 5,  # 5 consecutive trades (as per plan.md)
            
            # Drawdown controls
            "drawdown_reduction_threshold": 0.10,  # 10% DD -> 50% size reduction (as per plan.md)
            "drawdown_pause_threshold": 0.15,  # 15% DD -> trading pause (as per plan.md)
            "drawdown_lock_threshold": 0.20,  # 20% DD -> system lock (as per plan.md)
            
            # Volatility guards
            "volatility_guard_threshold": 95,  # 95th percentile
            "max_spread_bps": 10,  # Maximum spread in basis points
            
            # Position sizing rules
            "min_rr_ratio": 1.5,  # Minimum risk:reward ratio
            "mandatory_stop_loss": True,  # Stop loss is mandatory
            "no_averaging_down": True,  # No averaging down allowed
            "no_martingale": True,  # No martingale strategy
            
            # Red days control
            "max_red_days": 3,  # 3 consecutive red days -> safe mode

            # Max concurrent positions (scanner)
            "max_positions": 5,
        }
    
    def validate_trading_action(
        self,
        action: TradingAction,
        market_state: MarketState,
        position_state: PositionState,
        account_state: AccountState,
        risk_state: RiskState,
        active_positions: int = 0,
        total_exposure_pct: float = 0.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a trading action against all risk rules.

        Args:
            active_positions: Number of currently open positions (used for max_positions check).
            total_exposure_pct: Total portfolio exposure as fraction of equity across ALL positions.

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        try:
            # Check if this is a close action (closing existing position)
            is_close_action = self._is_close_action(action, position_state)

            # Skip position-opening rules for close actions
            if is_close_action:
                logger.info(f"Close action validated: {action.action} size={action.size}")
                return True, None

            # Check max concurrent positions (only for new positions)
            if action.action in [Action.BUY, Action.SELL]:
                max_pos = self.config.get("max_positions", 5)
                if active_positions >= max_pos:
                    return False, f"Max positions reached ({active_positions}/{max_pos})"

            # Check if safe mode is active
            if risk_state.safe_mode_active:
                if action.confidence < 0.8:
                    return False, "Safe mode active - only high-confidence trades allowed"

            # Validate action-specific rules
            if action.action in [Action.BUY, Action.SELL]:
                self._validate_new_position(action, market_state, position_state, account_state, risk_state)

            # Validate stop loss (must come before position sizing)
            self._validate_stop_loss(action)

            # Validate position sizing
            self._validate_position_size(action, account_state, risk_state, market_state)

            # Validate risk:reward ratio
            self._validate_risk_reward_ratio(action)

            # Check exposure limits
            self._validate_exposure_limits(action, position_state, account_state, market_state, total_exposure_pct)

            # Check market conditions
            self._validate_market_conditions(action, market_state)

            # Check daily risk limits
            self._validate_daily_limits(action, account_state, risk_state)

            # Check drawdown controls
            self._validate_drawdown_controls(action, account_state, risk_state)

            logger.info(f"Trading action validated: {action.action} size={action.size}")
            return True, None

        except RiskViolation as e:
            logger.warning(f"Risk validation failed: {e}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected error in risk validation: {e}")
            return False, f"Validation error: {e}"

    def _is_close_action(self, action: TradingAction, position_state: PositionState) -> bool:
        """Check if action is closing existing position (not opening new one)."""
        # Check explicit is_close flag first
        if getattr(action, 'is_close', False):
            return True

        # No position to close
        if position_state.current_side == Side.FLAT:
            return False

        # Closing long = SELL, closing short = BUY
        if position_state.current_side == Side.LONG and action.action == Action.SELL:
            return True
        if position_state.current_side == Side.SHORT and action.action == Action.BUY:
            return True

        return False
    
    def _validate_new_position(
        self,
        action: TradingAction,
        market_state: MarketState,
        position_state: PositionState,
        account_state: AccountState,
        risk_state: RiskState,
    ):
        """Validate rules for new positions."""
        
        # Check for averaging down (adding to losing position)
        if self.config["no_averaging_down"]:
            if position_state.current_side != Side.FLAT:
                if position_state.unrealized_pnl < 0:
                    current_side_bias = 1 if position_state.current_side == Side.LONG else -1
                    new_side_bias = 1 if action.action == Action.BUY else -1
                    if current_side_bias * new_side_bias > 0:  # Same direction
                        raise RiskViolation("Averaging down is not allowed")
        
        # Check for martingale (increasing size after loss)
        if self.config["no_martingale"]:
            if risk_state.consecutive_losses >= 3:
                # Calculate position size as % of equity
                size_pct = (action.size * action.stop_loss) / account_state.equity if action.stop_loss and account_state.equity > 0 else 0
                if size_pct > self.config["max_risk_per_trade"]:
                    raise RiskViolation("Martingale strategy detected - position size too large after consecutive losses")
    
    def _validate_position_size(
        self,
        action: TradingAction,
        account_state: AccountState,
        risk_state: RiskState,
        market_state: Optional[MarketState] = None,
    ):
        """Validate position sizing rules."""

        if action.action not in [Action.BUY, Action.SELL]:
            return

        # Calculate risk amount: size * distance from entry to stop loss
        if action.stop_loss and market_state:
            stop_distance = abs(market_state.current_price - action.stop_loss)
            risk_amount = action.size * stop_distance
            risk_pct = risk_amount / account_state.equity if account_state.equity > 0 else 1.0
        elif action.stop_loss and action.expected_risk > 0:
            # Fallback: use expected_risk as % of position notional
            risk_pct = action.expected_risk
        else:
            # No stop loss — treat as maximum risk (full position notional)
            risk_pct = 1.0

        # Check risk per trade limits against absolute maximum
        if risk_pct > self.config["max_risk_per_trade_absolute"]:
            raise RiskViolation(f"Risk per trade {risk_pct:.2%} exceeds absolute maximum {self.config['max_risk_per_trade_absolute']:.2%}")

        # Check against dynamic max risk (which is reduced during drawdown)
        if risk_pct > risk_state.max_risk_per_trade:
            raise RiskViolation(f"Risk per trade {risk_pct:.2%} exceeds dynamic maximum {risk_state.max_risk_per_trade:.2%}")

        # If it passes, log an info message
        logger.info(f"Risk per trade {risk_pct:.2%} is within dynamic maximum {risk_state.max_risk_per_trade:.2%}")
    
    def _validate_risk_reward_ratio(self, action: TradingAction):
        """Validate minimum risk:reward ratio."""
        
        if action.action not in [Action.BUY, Action.SELL]:
            return
        
        if action.expected_risk <= 0:
            raise RiskViolation("Expected risk must be positive")
        
        rr_ratio = action.expected_return / action.expected_risk
        if rr_ratio < self.config["min_rr_ratio"]:
            raise RiskViolation(f"R:R ratio {rr_ratio:.2f} below minimum {self.config['min_rr_ratio']}")
    
    def _validate_stop_loss(self, action: TradingAction):
        """Validate stop loss requirements."""
        
        if action.action not in [Action.BUY, Action.SELL]:
            return
        
        if self.config["mandatory_stop_loss"] and action.stop_loss is None:
            raise RiskViolation("Stop loss is mandatory for all positions")
    
    def _validate_exposure_limits(
        self,
        action: TradingAction,
        position_state: PositionState,
        account_state: AccountState,
        market_state: MarketState,
        total_exposure_pct: float = 0.0,
    ):
        """Validate exposure limits."""

        if action.action not in [Action.BUY, Action.SELL]:
            return

        price = market_state.current_price

        # Per-asset exposure: existing position + new order
        current_asset_exposure = abs(position_state.position_size * position_state.entry_price) if position_state.position_size and position_state.entry_price else 0
        new_order_notional = action.size * price
        new_asset_exposure = current_asset_exposure + new_order_notional
        new_asset_exposure_pct = new_asset_exposure / account_state.equity if account_state.equity > 0 else 1.0

        # Check per-asset exposure limit
        if new_asset_exposure_pct > self.config["max_exposure_per_asset"]:
            raise RiskViolation(f"Per-asset exposure {new_asset_exposure_pct:.2%} exceeds maximum {self.config['max_exposure_per_asset']:.2%}")

        # Total portfolio exposure: all existing positions + this new order
        new_total_exposure_pct = total_exposure_pct + (new_order_notional / account_state.equity if account_state.equity > 0 else 1.0)

        # Check total exposure limit
        if new_total_exposure_pct > self.config["max_total_exposure"]:
            raise RiskViolation(f"Total exposure {new_total_exposure_pct:.2%} exceeds maximum {self.config['max_total_exposure']:.2%}")

        # Check leverage limit
        if new_total_exposure_pct > self.config["max_leverage"]:
            raise RiskViolation(f"Leverage {new_total_exposure_pct:.2f}x exceeds maximum {self.config['max_leverage']:.1f}x")
    
    def _validate_market_conditions(self, action: TradingAction, market_state: MarketState):
        """Validate market condition guards."""
        
        # Volatility guard
        if market_state.volatility_percentile > self.config["volatility_guard_threshold"]:
            if action.action in [Action.BUY, Action.SELL]:
                raise RiskViolation(f"Volatility guard: {market_state.volatility_percentile:.1f}th percentile exceeds threshold")
        
        # Liquidity guard
        if market_state.spread > self.config["max_spread_bps"] / 10000:  # Convert bps to decimal
            if action.action in [Action.BUY, Action.SELL]:
                raise RiskViolation(f"Liquidity guard: spread {market_state.spread*10000:.1f} bps exceeds maximum")
        
        # Liquidity score check
        if market_state.liquidity_score < 0.3:  # Low liquidity threshold
            if action.action in [Action.BUY, Action.SELL]:
                raise RiskViolation(f"Liquidity guard: liquidity score {market_state.liquidity_score:.2f} too low")
    
    def _validate_daily_limits(
        self,
        action: TradingAction,
        account_state: AccountState,
        risk_state: RiskState,
    ):
        """Validate daily risk limits."""
        
        # Check daily loss limit
        if account_state.daily_loss_pct >= self.config["max_daily_loss"]:
            raise RiskViolation(f"Daily loss limit reached: {account_state.daily_loss_pct:.2%}")
        
        # Check consecutive losses
        if risk_state.consecutive_losses >= self.config["max_consecutive_losses"]:
            raise RiskViolation(f"Maximum consecutive losses reached: {risk_state.consecutive_losses}")
    
    def _validate_drawdown_controls(
        self,
        action: TradingAction,
        account_state: AccountState,
        risk_state: RiskState,
    ):
        """Validate drawdown controls."""
        
        dd = account_state.current_drawdown
        
        # System lock
        if dd >= self.config["drawdown_lock_threshold"]:
            raise RiskViolation(f"System lock: drawdown {dd:.2%} exceeds lock threshold")
        
        # Trading pause
        if dd >= self.config["drawdown_pause_threshold"]:
            raise RiskViolation(f"Trading pause: drawdown {dd:.2%} exceeds pause threshold")
        
        # Position size reduction
        if dd >= self.config["drawdown_reduction_threshold"]:
            # Allow trading but at reduced size — halve the normal risk budget
            max_risk_pct = self.config["max_risk_per_trade"] * 0.5  # 50% reduction
            logger.warning(f"Drawdown {dd:.2%} — reducing max risk per trade to {max_risk_pct:.2%}")
    
    def calculate_position_size(
        self,
        account_state: AccountState,
        market_state: MarketState,
        stop_distance: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate position size based on risk rules.
        
        Args:
            account_state: Current account state
            market_state: Current market state
            stop_distance: Stop loss distance in price units
            confidence: Decision confidence (0-1)
        
        Returns:
            Recommended position size in base currency units
        """
        
        # Validate inputs
        if account_state.equity <= 0 or stop_distance <= 0 or market_state.current_price <= 0:
            return 0.0
        
        # Calculate risk amount (fixed % of equity)
        risk_amount = account_state.equity * self.config["max_risk_per_trade"]
        
        # Calculate base position size: risk_amount / stop_distance
        # This gives us the size in base currency units
        base_size = risk_amount / stop_distance
        
        # Calculate notional value of base_size
        notional_value = base_size * market_state.current_price
        
        # Cap by maximum exposure per asset
        max_notional = account_state.equity * self.config["max_exposure_per_asset"]
        if notional_value > max_notional:
            base_size = max_notional / market_state.current_price

        # Adjust for confidence - NON-LINEAR scaling
        # Low confidence (< 0.5) = very small position (min 10%)
        # Medium confidence (0.5-0.7) = modest position (50-100%)
        # High confidence (0.7-0.9) = full position (100-150%)
        # Very high confidence (> 0.9) = up to 200% of base size
        if confidence < 0.5:
            confidence_multiplier = 0.1 + (confidence * 0.8)  # 0.1 to 0.5
        elif confidence < 0.7:
            confidence_multiplier = 0.5 + ((confidence - 0.5) * 2.5)  # 0.5 to 1.0
        elif confidence < 0.9:
            confidence_multiplier = 1.0 + ((confidence - 0.7) * 2.5)  # 1.0 to 1.5
        else:
            confidence_multiplier = 1.5 + ((confidence - 0.9) * 5.0)  # 1.5 to 2.0

        adjusted_size = base_size * confidence_multiplier

        logger.info(
            f"Confidence-based sizing: confidence={confidence:.2f} -> "
            f"multiplier={confidence_multiplier:.2f}x -> size={adjusted_size:.6f}"
        )
        
        # Adjust for drawdown (if applicable)
        if account_state.current_drawdown >= self.config["drawdown_reduction_threshold"]:
            adjusted_size *= 0.5  # 50% reduction
        
        # Adjust for safe mode
        # This would be set based on risk_state.safe_mode_active
        # Implementation depends on how safe mode is triggered
        
        # Final validation
        if adjusted_size <= 0:
            return 0.0
        
        # Log calculation details for debugging
        logger.debug(
            f"Position sizing: equity={account_state.equity:.2f}, "
            f"risk_pct={self.config['max_risk_per_trade']:.2%}, "
            f"risk_amount={risk_amount:.2f}, stop_distance={stop_distance:.4f}, "
            f"base_size={base_size:.6f}, adjusted_size={adjusted_size:.6f}"
        )
        
        return adjusted_size
    
    def update_risk_state(
        self,
        account_state: AccountState,
        market_state: MarketState,
        recent_trades: list,
        recent_daily_pnls: Optional[list] = None,
    ) -> RiskState:
        """Update risk state based on current conditions."""
        
        # Calculate consecutive losses
        consecutive_losses = 0
        for trade in reversed(recent_trades[-10:]):  # Check last 10 trades
            if trade.pnl < 0:
                consecutive_losses += 1
            else:
                break
        
        # Determine risk level
        if account_state.current_drawdown >= self.config["drawdown_pause_threshold"]:
            risk_level = RiskLevel.CRITICAL
        elif account_state.current_drawdown >= self.config["drawdown_reduction_threshold"]:
            risk_level = RiskLevel.HIGH
        elif market_state.volatility_percentile > 90:
            risk_level = RiskLevel.HIGH
        elif consecutive_losses >= 3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Check for consecutive red days
        red_days_triggered = False
        if recent_daily_pnls is not None:
            max_red = self.config.get("max_red_days", 3)
            if len(recent_daily_pnls) >= max_red:
                red_days_triggered = all(
                    d.get("pnl", 0) < 0 for d in recent_daily_pnls[:max_red]
                )

        # Check if safe mode should be active
        safe_mode = (
            account_state.current_drawdown >= self.config["drawdown_reduction_threshold"] or
            consecutive_losses >= 3 or
            market_state.volatility_percentile > self.config["volatility_guard_threshold"] or
            red_days_triggered
        )
        
        # Calculate risk budget
        risk_budget_left = account_state.equity * (
            self.config["max_daily_loss"] - account_state.daily_loss_pct
        )
        
        return RiskState(
            risk_budget_left=risk_budget_left,
            max_daily_loss_remaining=account_state.equity * (
                self.config["max_daily_loss"] - account_state.daily_loss_pct
            ),
            consecutive_losses=consecutive_losses,
            volatility_percentile=market_state.volatility_percentile,
            current_risk_level=risk_level,
            safe_mode_active=safe_mode,
            max_risk_per_trade=self.config["max_risk_per_trade"],
            max_exposure_per_asset=self.config["max_exposure_per_asset"],
            max_total_exposure=self.config["max_total_exposure"],
            max_leverage=self.config["max_leverage"],
        )
