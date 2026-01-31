"""Unit tests for MarketScanner."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from trade_bot.scanner.market_scanner import MarketScanner
from trade_bot.scanner.models import CoinCandidate


def _make_markets(symbols):
    """Create a fake markets dict."""
    return {
        s: {"active": True, "quote": "USDT", "type": "spot"}
        for s in symbols
    }


def _make_ticker(quote_volume=5_000_000, last=100.0, pct=2.0, spread=0.05):
    bid = last - spread / 2
    ask = last + spread / 2
    return {
        "quoteVolume": quote_volume,
        "last": last,
        "bid": bid,
        "ask": ask,
        "percentage": pct,
        "high": last * 1.03,
        "low": last * 0.97,
    }


class TestMarketScanner:
    def setup_method(self):
        self.scanner = MarketScanner({
            "quote_currency": "USDT",
            "min_volume_24h": 1_000_000,
            "max_positions": 3,
            "blacklist": ["SCAM/USDT"],
            "weight_volume": 0.30,
            "weight_momentum": 0.25,
            "weight_spread": 0.25,
            "weight_volatility": 0.20,
            "vol_min": 0.10,
            "vol_max": 1.50,
        })

    def test_filter_pairs_basic(self):
        markets = {
            "BTC/USDT": {"active": True, "quote": "USDT", "type": "spot"},
            "ETH/BTC": {"active": True, "quote": "BTC", "type": "spot"},
            "SOL/USDT": {"active": False, "quote": "USDT", "type": "spot"},
            "DOGE/USDT": {"active": True, "quote": "USDT", "type": "future"},
        }
        result = self.scanner._filter_pairs(markets)
        assert result == ["BTC/USDT"]

    def test_filter_pairs_blacklist(self):
        markets = _make_markets(["BTC/USDT", "SCAM/USDT", "ETH/USDT"])
        result = self.scanner._filter_pairs(markets)
        assert "SCAM/USDT" not in result
        assert "BTC/USDT" in result
        assert "ETH/USDT" in result

    def test_build_candidate_low_volume_filtered(self):
        ticker = _make_ticker(quote_volume=500_000)
        result = self.scanner._build_candidate("LOW/USDT", ticker)
        assert result is None

    def test_build_candidate_ok(self):
        ticker = _make_ticker(quote_volume=2_000_000, last=50.0, pct=3.5)
        c = self.scanner._build_candidate("OK/USDT", ticker)
        assert c is not None
        assert c.symbol == "OK/USDT"
        assert c.volume_24h == 2_000_000

    def test_normalise_and_score(self):
        candidates = [
            CoinCandidate("A/USDT", 0, 10_000_000, 5.0, 0.01, 0.5),
            CoinCandidate("B/USDT", 0, 1_000_000, 1.0, 0.10, 0.05),
        ]
        scored = self.scanner._normalise_and_score(candidates)
        # A should score higher (more volume, momentum, tighter spread, better vol)
        assert scored[0].score > 0
        assert scored[1].score > 0
        assert scored[0].symbol == "A/USDT"

    def test_get_top_candidates_respects_max(self):
        self.scanner._last_candidates = [
            CoinCandidate(f"C{i}/USDT", score=10 - i, volume_24h=0, price_change_pct=0, spread_pct=0, volatility=0)
            for i in range(10)
        ]
        top = self.scanner.get_top_candidates()
        assert len(top) == 3  # max_positions=3

    @pytest.mark.asyncio
    async def test_scan_market_integration(self):
        exchange = AsyncMock()
        exchange.load_markets = AsyncMock(return_value=_make_markets(["BTC/USDT", "ETH/USDT"]))
        exchange.fetch_tickers = AsyncMock(return_value={
            "BTC/USDT": _make_ticker(quote_volume=50_000_000, last=60000, pct=1.5),
            "ETH/USDT": _make_ticker(quote_volume=20_000_000, last=3000, pct=3.0),
        })

        candidates = await self.scanner.scan_market(exchange)
        assert len(candidates) == 2
        # Both should have positive scores
        assert all(c.score > 0 for c in candidates)

    @pytest.mark.asyncio
    async def test_scan_market_empty(self):
        exchange = AsyncMock()
        exchange.load_markets = AsyncMock(return_value={})
        result = await self.scanner.scan_market(exchange)
        assert result == []

    def test_coin_candidate_ordering(self):
        a = CoinCandidate("A", score=0.8, volume_24h=0, price_change_pct=0, spread_pct=0, volatility=0)
        b = CoinCandidate("B", score=0.5, volume_24h=0, price_change_pct=0, spread_pct=0, volatility=0)
        assert b < a


class TestRiskValidatorMaxPositions:
    """Test max_positions enforcement in RiskValidator."""

    def test_max_positions_rejects(self):
        from trade_bot.risk.validator import RiskValidator
        from trade_bot.core.models import (
            MarketState, PositionState, AccountState, RiskState, TradingAction, RiskLevel,
        )
        from trade_bot.core.enums import Action, Side, Regime
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=50, freq="1h")
        ohlcv = pd.DataFrame(
            {"open": [100]*50, "high": [101]*50, "low": [99]*50, "close": [100]*50, "volume": [1000]*50},
            index=dates,
        )

        validator = RiskValidator()
        validator.config["max_positions"] = 2
        market = MarketState(
            ohlcv=ohlcv, current_price=100.0, atr=2.0, realized_volatility=0.2,
            spread=0.001, order_book_imbalance=0.1, volume_delta=1000,
            liquidity_score=0.8, regime_label=Regime.TRENDING, volatility_percentile=50.0,
        )
        position = PositionState(current_side=Side.FLAT, position_size=0.0, entry_price=None)
        account = AccountState(equity=10000, balance=10000)
        risk = RiskState(
            risk_budget_left=100, max_daily_loss_remaining=500,
            consecutive_losses=0, volatility_percentile=50,
            current_risk_level=RiskLevel.LOW, safe_mode_active=False,
        )
        action = TradingAction(
            action=Action.BUY, size=1.0, stop_loss=98.0, take_profit=104.0,
            expected_return=4.0, expected_risk=2.0, confidence=0.8,
        )

        is_valid, reason = validator.validate_trading_action(
            action, market, position, account, risk, active_positions=2,
        )
        assert is_valid is False
        assert "Max positions" in reason

        # Under limit should pass
        is_valid2, _ = validator.validate_trading_action(
            action, market, position, account, risk, active_positions=1,
        )
        assert is_valid2 is True
