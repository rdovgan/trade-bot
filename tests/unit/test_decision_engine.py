"""Unit tests for DecisionEngine and LLMAdvisor."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from trade_bot.core.models import (
    MarketState, PositionState, AccountState, RiskState, TradingAction,
)
from trade_bot.core.enums import Action, Side, Regime, RiskLevel
from trade_bot.decision.engine import LLMAdvisor, DecisionEngine
from trade_bot.risk.validator import RiskValidator
from trade_bot.signal.generator import SignalManager


@pytest.fixture
def market_state(sample_market_state):
    return sample_market_state


@pytest.fixture
def position_state(sample_position_state):
    return sample_position_state


@pytest.fixture
def account_state(sample_account_state):
    return sample_account_state


@pytest.fixture
def risk_state(sample_risk_state):
    return sample_risk_state


class TestLLMAdvisor:
    def test_disabled_returns_fallback(self):
        advisor = LLMAdvisor({"enabled": False})
        loop = asyncio.get_event_loop()
        ms = MagicMock(spec=MarketState)
        ms.regime_label = Regime.TRENDING
        ms.volatility_percentile = 50.0
        ms.liquidity_score = 0.8
        ms.spread = 0.0005
        ps = MagicMock(spec=PositionState)
        ps.current_side = Side.FLAT
        ps.unrealized_pnl = 0.0

        result = loop.run_until_complete(advisor.get_market_insights(ms, ps))
        assert result["reasoning"] == "LLM advisor disabled or unavailable"
        assert "regime_confirmation" in result

    def test_no_api_key_returns_fallback(self):
        advisor = LLMAdvisor({"enabled": True, "api_key": ""})
        loop = asyncio.get_event_loop()
        ms = MagicMock(spec=MarketState)
        ms.regime_label = Regime.TRENDING
        ms.volatility_percentile = 50.0
        ms.liquidity_score = 0.8
        ms.spread = 0.0005
        ps = MagicMock(spec=PositionState)
        ps.current_side = Side.FLAT
        ps.unrealized_pnl = 0.0

        result = loop.run_until_complete(advisor.get_market_insights(ms, ps))
        assert "regime_confirmation" in result

    def test_sanitize_response_clamps_values(self):
        advisor = LLMAdvisor()
        ms = MagicMock(spec=MarketState)
        ms.regime_label = Regime.TRENDING

        raw = {
            "regime_confirmation": "trending",
            "risk_factors": ["test"],
            "opportunity_score": 5.0,  # Should be clamped to 1.0
            "confidence_adjustment": 1.0,  # Should be clamped to 0.3
            "reasoning": "test",
            "position_size": 999,  # Should be ignored
        }
        result = advisor._sanitize_response(raw, ms)
        assert result["opportunity_score"] == 1.0
        assert result["confidence_adjustment"] == 0.3
        assert "position_size" not in result

    def test_identify_risk_factors(self):
        advisor = LLMAdvisor()
        ms = MagicMock(spec=MarketState)
        ms.volatility_percentile = 95.0
        ms.liquidity_score = 0.2
        ms.spread = 0.002
        factors = advisor._identify_risk_factors(ms)
        assert len(factors) == 3

    def test_calculate_opportunity_score(self):
        advisor = LLMAdvisor()
        ms = MagicMock(spec=MarketState)
        ms.volatility_percentile = 50.0
        ms.liquidity_score = 0.9
        ms.spread = 0.0003
        score = advisor._calculate_opportunity_score(ms)
        assert 0.0 <= score <= 1.0


class TestDecisionEngine:
    def _make_engine(self):
        rv = RiskValidator()
        sm = SignalManager()
        la = LLMAdvisor()
        return DecisionEngine(rv, sm, la)

    @pytest.mark.asyncio
    async def test_no_signals_returns_hold(
        self, market_state, position_state, account_state, risk_state
    ):
        engine = self._make_engine()
        action = await engine.make_decision(
            market_state, position_state, account_state, risk_state
        )
        assert action.action == Action.HOLD

    @pytest.mark.asyncio
    async def test_should_close_flat_position(
        self, market_state, position_state, account_state, risk_state
    ):
        engine = self._make_engine()
        result = await engine.should_close_position(
            market_state, position_state, account_state, risk_state
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_should_close_stop_loss_triggered(
        self, market_state, account_state, risk_state
    ):
        engine = self._make_engine()
        ps = PositionState(
            current_side=Side.LONG,
            position_size=1.0,
            entry_price=105.0,
            stop_loss=101.0,
        )
        market_state.current_price = 100.0
        result = await engine.should_close_position(
            market_state, ps, account_state, risk_state
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_close_take_profit_triggered(
        self, market_state, account_state, risk_state
    ):
        engine = self._make_engine()
        ps = PositionState(
            current_side=Side.LONG,
            position_size=1.0,
            entry_price=95.0,
            take_profit=99.0,
        )
        market_state.current_price = 100.0
        result = await engine.should_close_position(
            market_state, ps, account_state, risk_state
        )
        assert result is True

    def test_create_hold_action(self):
        engine = self._make_engine()
        hold = engine._create_hold_action()
        assert hold.action == Action.HOLD
        assert hold.size == 0.0

    def test_record_decision(self):
        engine = self._make_engine()
        action = TradingAction(
            action=Action.BUY, size=1, stop_loss=98, take_profit=104,
            expected_return=4, expected_risk=2, confidence=0.8,
        )
        engine._record_decision(action, True, "Approved")
        assert len(engine.decision_history) == 1

    def test_get_decision_stats_empty(self):
        engine = self._make_engine()
        stats = engine.get_decision_stats()
        assert stats["total_decisions"] == 0

    def test_get_decision_stats(self):
        engine = self._make_engine()
        action = TradingAction(
            action=Action.BUY, size=1, stop_loss=98, take_profit=104,
            expected_return=4, expected_risk=2, confidence=0.8,
        )
        engine._record_decision(action, True, "Approved")
        engine._record_decision(action, False, "Rejected")
        stats = engine.get_decision_stats()
        assert stats["total_decisions"] == 2
        assert stats["approved_decisions"] == 1
