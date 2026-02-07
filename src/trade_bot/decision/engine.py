"""Decision engine for trading actions."""

import json
from typing import Optional, Dict, Any, List
import logging
import asyncio
from datetime import datetime

import aiohttp

from ..core.models import (
    MarketState, PositionState, AccountState, RiskState,
    TradingAction, Trade
)
from ..core.enums import Action, Side
from ..risk.validator import RiskValidator
from ..signal.generator import SignalManager

logger = logging.getLogger(__name__)


class LLMAdvisor:
    """LLM advisory system for structured market insights.

    LLM may:
    - Generate hypotheses
    - Describe macro narrative
    - Detect news-based risk
    - Suggest regime shift

    LLM may NOT:
    - Set position size
    - Override risk limits
    - Remove stop-loss
    - Change leverage rules
    """

    # Fields the LLM is allowed to return
    ALLOWED_FIELDS = {
        'regime_confirmation', 'risk_factors', 'opportunity_score',
        'confidence_adjustment', 'reasoning',
        'take_profit_target', 'stop_loss_target',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM advisor."""
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        self.api_key = self.config.get('api_key', '')
        self.model = self.config.get('model', 'gpt-4')
        self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')
        self.timeout = self.config.get('timeout', 30)
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"LLM advisor initialized (enabled: {self.enabled})")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_market_insights(
        self,
        market_state: MarketState,
        position_state: PositionState
    ) -> Dict[str, Any]:
        """Get structured market insights from LLM."""
        if not self.enabled or not self.api_key:
            return self._fallback_insights(market_state)

        try:
            return await self._call_llm(market_state, position_state)
        except Exception as e:
            logger.warning(f"LLM call failed, using fallback: {e}")
            return self._fallback_insights(market_state)

    async def _call_llm(
        self,
        market_state: MarketState,
        position_state: PositionState
    ) -> Dict[str, Any]:
        """Call LLM API for market insights."""
        session = await self._get_session()

        prompt = self._build_prompt(market_state, position_state)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a market analysis advisor. Provide structured analysis "
                        "in JSON format with these fields only: regime_confirmation (string), "
                        "risk_factors (list of strings), opportunity_score (float 0-1), "
                        "confidence_adjustment (float -0.3 to 0.3), reasoning (string), "
                        "take_profit_target (float/null), stop_loss_target (float/null). "
                        "Use the targets to suggest a dynamic exit based on observed market manipulation or specific chart patterns. "
                        "You must NOT suggest position sizes, leverage changes, or stop-loss removal."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"LLM API error {response.status}: {error_text}")

            data = await response.json()
            content = data["choices"][0]["message"]["content"]
            raw_insights = json.loads(content)

            # Enforce policy: strip any disallowed fields
            return self._sanitize_response(raw_insights, market_state)

    def _build_prompt(
        self, market_state: MarketState, position_state: PositionState
    ) -> str:
        """Build analysis prompt from market state."""
        return (
            f"Analyze the current market conditions:\n"
            f"- Current price: {market_state.current_price}\n"
            f"- ATR: {market_state.atr:.4f}\n"
            f"- Realized volatility: {market_state.realized_volatility:.4f}\n"
            f"- Spread: {market_state.spread:.6f}\n"
            f"- Order book imbalance: {market_state.order_book_imbalance:.4f}\n"
            f"- Volume delta: {market_state.volume_delta:.2f}\n"
            f"- Liquidity score: {market_state.liquidity_score:.2f}\n"
            f"- Current regime: {market_state.regime_label.value}\n"
            f"- Volatility percentile: {market_state.volatility_percentile:.1f}\n"
            f"- Position side: {position_state.current_side.value}\n"
            f"- Unrealized PnL: {position_state.unrealized_pnl:.2f}\n\n"
            f"Provide your structured analysis as JSON."
        )

    def _sanitize_response(
        self, raw: Dict[str, Any], market_state: MarketState
    ) -> Dict[str, Any]:
        """Sanitize LLM response — remove disallowed fields, clamp values."""
        result = {
            'regime_confirmation': str(raw.get('regime_confirmation', market_state.regime_label.value)),
            'risk_factors': list(raw.get('risk_factors', [])),
            'opportunity_score': max(0.0, min(1.0, float(raw.get('opportunity_score', 0.5)))),
            'confidence_adjustment': max(-0.3, min(0.3, float(raw.get('confidence_adjustment', 0.0)))),
            'reasoning': str(raw.get('reasoning', '')),
        }
        return result

    def _fallback_insights(self, market_state: MarketState) -> Dict[str, Any]:
        """Generate fallback insights without LLM."""
        return {
            'regime_confirmation': market_state.regime_label.value,
            'risk_factors': self._identify_risk_factors(market_state),
            'opportunity_score': self._calculate_opportunity_score(market_state),
            'confidence_adjustment': 0.0,
            'reasoning': 'LLM advisor disabled or unavailable'
        }

    def _identify_risk_factors(self, market_state: MarketState) -> List[str]:
        """Identify risk factors from market state."""
        risk_factors = []

        if market_state.volatility_percentile > 90:
            risk_factors.append('High volatility detected')

        if market_state.liquidity_score < 0.3:
            risk_factors.append('Low liquidity conditions')

        if market_state.spread > 0.001:
            risk_factors.append('Wide market spread')

        return risk_factors

    def _calculate_opportunity_score(self, market_state: MarketState) -> float:
        """Calculate opportunity score (0-1)."""
        score = 0.5

        if 30 <= market_state.volatility_percentile <= 70:
            score += 0.2

        score += market_state.liquidity_score * 0.2

        if market_state.spread < 0.0005:
            score += 0.1

        return min(score, 1.0)


class DecisionEngine:
    """Main decision engine for trading actions."""

    def __init__(
        self,
        risk_validator: RiskValidator,
        signal_manager: SignalManager,
        llm_advisor: Optional[LLMAdvisor] = None
    ):
        """Initialize decision engine."""
        self.risk_validator = risk_validator
        self.signal_manager = signal_manager
        self.llm_advisor = llm_advisor or LLMAdvisor()

        # Decision history
        self.decision_history: List[Dict[str, Any]] = []

        logger.info("Decision engine initialized")

    async def make_decision(
        self,
        market_state: MarketState,
        position_state: PositionState,
        account_state: AccountState,
        risk_state: RiskState,
        active_positions: int = 0,
    ) -> Optional[TradingAction]:
        """
        Make a trading decision based on all available information.

        This follows the architecture: SCAN -> SIGNAL -> RESEARCH -> DECIDE -> RISK VALIDATOR
        """
        try:
            # CRITICAL: If Safe Mode is active and we have a losing position, CLOSE IT
            if risk_state.safe_mode_active and position_state.current_side != Side.FLAT:
                if position_state.entry_price and position_state.entry_price > 0:
                    # Calculate unrealized PnL
                    if position_state.current_side == Side.LONG:
                        pnl_pct = ((market_state.current_price - position_state.entry_price) / position_state.entry_price) * 100
                    else:
                        pnl_pct = ((position_state.entry_price - market_state.current_price) / position_state.entry_price) * 100

                    # If position is losing, force close it to free capital
                    if pnl_pct < 0:
                        logger.critical(
                            f"SAFE MODE FORCE CLOSE: Position has {pnl_pct:.2f}% unrealized loss. "
                            f"Generating CLOSE action to free capital and exit safe mode."
                        )
                        return self._create_close_action(position_state, market_state)

                    # Even if profitable, close to reduce exposure in safe mode
                    logger.warning(
                        f"SAFE MODE: Closing position with {pnl_pct:.2f}% profit to reduce exposure."
                    )
                    return self._create_close_action(position_state, market_state)

            # Step 1: Generate signals
            logger.info("Generating trading signals...")
            signals = self.signal_manager.generate_signals(market_state)

            if not signals:
                logger.info("No signals generated")
                return self._create_hold_action()

            # Step 2: Get best signal
            best_signal = self.signal_manager.get_best_signal(signals)
            if not best_signal:
                return self._create_hold_action()

            # Step 3: Get LLM insights (advisory only)
            logger.info("Getting LLM market insights...")
            llm_insights = await self.llm_advisor.get_market_insights(
                market_state, position_state
            )

            # Step 4: Adjust signal based on insights
            adjusted_signal = self._adjust_signal_with_insights(
                best_signal, llm_insights
            )

            # Step 5: Risk validation (hard gate)
            logger.info("Validating with risk engine...")
            is_valid, rejection_reason = self.risk_validator.validate_trading_action(
                adjusted_signal, market_state, position_state, account_state, risk_state,
                active_positions=active_positions,
                total_exposure_pct=account_state.exposure_pct,
            )

            if not is_valid:
                logger.warning(f"Trading action rejected by risk validator: {rejection_reason}")
                self._record_decision(adjusted_signal, False, rejection_reason)
                return self._create_hold_action()

            # Step 6: Calculate proper position size
            if adjusted_signal.action in [Action.BUY, Action.SELL]:
                stop_distance = abs(adjusted_signal.stop_loss - market_state.current_price) if adjusted_signal.stop_loss else market_state.atr
                proper_size = self.risk_validator.calculate_position_size(
                    account_state, market_state, stop_distance, adjusted_signal.confidence
                )
                adjusted_signal.size = proper_size

            # Step 7: Final validation
            if adjusted_signal.size <= 0:
                logger.warning("Position size is zero or negative")
                self._record_decision(adjusted_signal, False, "Invalid position size")
                return self._create_hold_action()

            # Record successful decision
            self._record_decision(adjusted_signal, True, "Approved")

            logger.info(
                f"Decision made: {adjusted_signal.action.value} "
                f"size={adjusted_signal.size:.6f} "
                f"confidence={adjusted_signal.confidence:.2f}"
            )

            return adjusted_signal

        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            self._record_decision(None, False, f"Error: {e}")
            return self._create_hold_action()

    def _adjust_signal_with_insights(
        self,
        signal: TradingAction,
        insights: Dict[str, Any]
    ) -> TradingAction:
        """Adjust signal based on LLM insights."""
        adjusted = TradingAction(**signal.model_dump())

        # Adjust confidence based on insights
        confidence_adjustment = insights.get('confidence_adjustment', 0.0)
        adjusted.confidence = max(0.0, min(1.0, adjusted.confidence + confidence_adjustment))

        # Dynamic Stop Loss / Take Profit adjustment from LLM
        llm_sl = insights.get('stop_loss_target')
        llm_tp = insights.get('take_profit_target')

        if llm_sl is not None and llm_sl > 0:
            adjusted.stop_loss = llm_sl
            # Re-calculate expected risk based on new SL
            market_price = self.signal_manager.get_last_price() # Assumes signal manager can cache this
            if market_price and adjusted.stop_loss:
                adjusted.expected_risk = abs(adjusted.stop_loss - market_price)
            logger.debug(f"LLM adjusted Stop Loss to {llm_sl}")

        if llm_tp is not None and llm_tp > 0:
            adjusted.take_profit = llm_tp
            # Re-calculate expected return based on new TP
            market_price = self.signal_manager.get_last_price() # Assumes signal manager can cache this
            if market_price and adjusted.take_profit:
                adjusted.expected_return = abs(adjusted.take_profit - market_price)
            logger.debug(f"LLM adjusted Take Profit to {llm_tp}")

        logger.debug(
            f"Adjusted signal confidence from {signal.confidence:.2f} "
            f"to {adjusted.confidence:.2f} based on LLM insights"
        )

        return adjusted

    def _create_hold_action(self) -> TradingAction:
        """Create a hold action."""
        return TradingAction(
            action=Action.HOLD,
            size=0.0,
            expected_return=0.0,
            expected_risk=0.0,
            confidence=1.0
        )

    def _create_close_action(self, position_state: PositionState, market_state: MarketState) -> TradingAction:
        """Create a close action for existing position."""
        # Determine close direction (opposite of current position)
        if position_state.current_side == Side.LONG:
            close_action = Action.SELL
        else:
            close_action = Action.BUY

        close_size = abs(position_state.position_size)

        logger.info(
            f"Creating CLOSE action: {close_action.value} size={close_size:.6f} "
            f"to close {position_state.current_side.value} position"
        )

        return TradingAction(
            action=close_action,
            size=close_size,
            expected_return=0.0,
            expected_risk=0.0,
            confidence=1.0,  # High confidence for forced close
            stop_loss=None,
            take_profit=None,
        )

    def _record_decision(
        self,
        action: Optional[TradingAction],
        approved: bool,
        reason: str
    ):
        """Record decision for learning and analysis."""
        decision_record = {
            'timestamp': datetime.now(),
            'action': action.model_dump() if action else None,
            'approved': approved,
            'reason': reason,
        }

        self.decision_history.append(decision_record)

        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision statistics."""
        if not self.decision_history:
            return {'total_decisions': 0}

        total = len(self.decision_history)
        approved = sum(1 for d in self.decision_history if d['approved'])

        action_counts = {}
        for decision in self.decision_history:
            if decision['action']:
                action = decision['action']['action']
                action_counts[action] = action_counts.get(action, 0) + 1

        return {
            'total_decisions': total,
            'approved_decisions': approved,
            'approval_rate': approved / total if total > 0 else 0,
            'action_distribution': action_counts,
        }

    async def should_close_position(
        self,
        market_state: MarketState,
        position_state: PositionState,
        account_state: AccountState,
        risk_state: RiskState
    ) -> bool:
        """Determine if current position should be closed."""
        if position_state.current_side == Side.FLAT:
            return False

        # Check stop loss
        if position_state.stop_loss:
            if position_state.current_side == Side.LONG:
                if market_state.current_price <= position_state.stop_loss:
                    logger.info("Stop loss triggered for long position")
                    return True
            else:
                if market_state.current_price >= position_state.stop_loss:
                    logger.info("Stop loss triggered for short position")
                    return True

        # Check take profit
        if position_state.take_profit:
            if position_state.current_side == Side.LONG:
                if market_state.current_price >= position_state.take_profit:
                    logger.info("Take profit triggered for long position")
                    return True
            else:
                if market_state.current_price <= position_state.take_profit:
                    logger.info("Take profit triggered for short position")
                    return True

        # Check trailing stop
        if position_state.trailing_stop:
            if position_state.current_side == Side.LONG:
                if market_state.current_price <= position_state.trailing_stop:
                    logger.info("Trailing stop triggered for long position")
                    return True
            else:
                if market_state.current_price >= position_state.trailing_stop:
                    logger.info("Trailing stop triggered for short position")
                    return True

        # AGGRESSIVE CLOSE: Time-based rules
        if position_state.entry_time:
            from datetime import datetime, timedelta
            age = datetime.now() - position_state.entry_time
            age_hours = age.total_seconds() / 3600

            # Force close after 8 hours regardless of PnL
            if age > timedelta(hours=8):
                logger.info(f"Position age limit triggered: {age_hours:.1f} hours old (max 8h)")
                return True

            # Aggressive rules apply only for positions older than 4 hours
            if age > timedelta(hours=4):
                if position_state.entry_price and position_state.position_size != 0:
                    pnl_pct = 0.0
                    if position_state.current_side == Side.LONG:
                        pnl_pct = ((market_state.current_price - position_state.entry_price) / position_state.entry_price) * 100
                    else:
                        pnl_pct = ((position_state.entry_price - market_state.current_price) / position_state.entry_price) * 100

                    # Close if loss > 1.5%
                    if pnl_pct < -1.5:
                        logger.warning(f"Aggressive loss cut triggered for old position ({age_hours:.1f}h): {pnl_pct:.2f}% loss")
                        return True

                    # Close if profit > 2.5%
                    if pnl_pct > 2.5:
                        logger.info(f"Aggressive profit take triggered for old position ({age_hours:.1f}h): {pnl_pct:.2f}% profit")
                        return True

        # Check risk limits — safe mode forces closure of low-confidence positions
        if risk_state.safe_mode_active:
            logger.info("Safe mode active - considering position closure")

        return False
