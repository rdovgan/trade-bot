"""Walk-forward validation framework (§8.1).

Features:
- Walk-forward data splitting (no look-ahead bias)
- Realistic slippage model
- Commission model
- Latency simulator
- Validation metrics (Sharpe, max DD, trade count)
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SlippageModel:
    """Configurable slippage model based on volatility and liquidity."""
    base_slippage_bps: float = 2.0
    volatility_multiplier: float = 1.0
    liquidity_multiplier: float = 1.0

    def estimate(self, price: float, volatility: float, liquidity_score: float) -> float:
        """Estimate slippage in price units.

        Args:
            price: Current price.
            volatility: Realized volatility (annualized fraction).
            liquidity_score: 0-1 (higher = more liquid).
        """
        vol_factor = 1.0 + self.volatility_multiplier * volatility
        liq_factor = 1.0 + self.liquidity_multiplier * max(0, 1.0 - liquidity_score)
        slippage_bps = self.base_slippage_bps * vol_factor * liq_factor
        return price * slippage_bps / 10_000


@dataclass
class LatencySimulator:
    """Simulates execution latency."""
    base_delay_ms: float = 50.0
    jitter_ms: float = 20.0

    def get_delay_bars(self, bar_duration_ms: float) -> int:
        """Return how many bars of delay to apply.

        For most intraday strategies with 1-minute bars (60 000 ms),
        typical latency is sub-bar → returns 0 or 1.
        """
        delay = self.base_delay_ms + np.random.uniform(0, self.jitter_ms)
        return int(np.ceil(delay / bar_duration_ms)) if bar_duration_ms > 0 else 0


@dataclass
class CommissionModel:
    """Simple commission model."""
    rate_bps: float = 5.0  # 5 bps per side

    def calculate(self, notional: float) -> float:
        """Calculate commission for a trade."""
        return notional * self.rate_bps / 10_000


@dataclass
class BacktestResult:
    """Result of a single walk-forward window."""
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    trade_count: int
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_r_multiple: float
    profit_factor: float


@dataclass
class ValidationResult:
    """Aggregated walk-forward validation result."""
    windows: List[BacktestResult]
    overall_sharpe: float
    overall_max_dd: float
    overall_trade_count: int
    overall_win_rate: float
    passed: bool


class WalkForwardValidator:
    """Walk-forward validation engine.

    Splits historical data into rolling train/test windows,
    applies a signal function on each test window using only
    data available at that point (no look-ahead bias).
    """

    def __init__(
        self,
        train_bars: int = 500,
        test_bars: int = 100,
        step_bars: int = 100,
        slippage_model: Optional[SlippageModel] = None,
        latency_simulator: Optional[LatencySimulator] = None,
        commission_model: Optional[CommissionModel] = None,
        bar_duration_ms: float = 60_000,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.slippage = slippage_model or SlippageModel()
        self.latency = latency_simulator or LatencySimulator()
        self.commission = commission_model or CommissionModel()
        self.bar_duration_ms = bar_duration_ms

    def validate(
        self,
        data: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame], List[Dict]],
        initial_equity: float = 10_000.0,
    ) -> ValidationResult:
        """Run walk-forward validation.

        Args:
            data: OHLCV DataFrame with columns [open, high, low, close, volume].
            signal_fn: Function that takes a DataFrame (train+current bar)
                       and returns a list of signal dicts with keys:
                       {bar_index, action, size, stop_loss, take_profit, confidence}.
            initial_equity: Starting equity.

        Returns:
            ValidationResult with per-window and aggregate metrics.
        """
        n = len(data)
        windows: List[BacktestResult] = []
        all_pnls: List[float] = []
        window_idx = 0

        start = 0
        while start + self.train_bars + self.test_bars <= n:
            train_end = start + self.train_bars
            test_end = train_end + self.test_bars

            train_data = data.iloc[start:train_end]
            test_data = data.iloc[train_end:test_end]

            result = self._run_window(
                window_idx, train_data, test_data, signal_fn, initial_equity
            )
            windows.append(result)
            all_pnls.append(result.total_pnl)

            start += self.step_bars
            window_idx += 1

        if not windows:
            return ValidationResult(
                windows=[], overall_sharpe=0, overall_max_dd=0,
                overall_trade_count=0, overall_win_rate=0, passed=False,
            )

        overall_trade_count = sum(w.trade_count for w in windows)
        overall_win_rate = (
            np.mean([w.win_rate for w in windows if w.trade_count > 0])
            if any(w.trade_count > 0 for w in windows) else 0.0
        )
        overall_sharpe = self._calc_sharpe(all_pnls)
        overall_max_dd = max(w.max_drawdown for w in windows) if windows else 0.0

        passed = (
            overall_sharpe > 1.2
            and overall_max_dd < 0.10
            and overall_trade_count >= 200
        )

        return ValidationResult(
            windows=windows,
            overall_sharpe=overall_sharpe,
            overall_max_dd=overall_max_dd,
            overall_trade_count=overall_trade_count,
            overall_win_rate=float(overall_win_rate),
            passed=passed,
        )

    def _run_window(
        self,
        window_idx: int,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        signal_fn: Callable,
        initial_equity: float,
    ) -> BacktestResult:
        """Run a single train/test window."""
        equity = initial_equity
        pnls: List[float] = []

        # Generate signals on test data (signal_fn receives train + bar-so-far)
        for i in range(len(test_data)):
            # Data available up to this bar (train + test bars seen so far)
            available = pd.concat([train_data, test_data.iloc[: i + 1]])
            signals = signal_fn(available)

            if not signals:
                continue

            bar = test_data.iloc[i]
            volatility = (
                test_data["close"].iloc[max(0, i - 20) : i + 1].pct_change().std()
                if i > 1 else 0.01
            )

            for sig in signals:
                # Apply latency
                delay = self.latency.get_delay_bars(self.bar_duration_ms)
                exec_idx = i + delay
                if exec_idx >= len(test_data):
                    continue
                exec_bar = test_data.iloc[exec_idx]

                entry_price = exec_bar["close"]
                slip = self.slippage.estimate(entry_price, volatility, 0.7)
                if sig.get("action") == "BUY":
                    entry_price += slip
                else:
                    entry_price -= slip

                notional = sig.get("size", 1.0) * entry_price
                comm = self.commission.calculate(notional) * 2  # round trip

                # Simplified P&L using stop/take profit
                sl = sig.get("stop_loss", 0)
                tp = sig.get("take_profit", 0)
                if sl and tp:
                    risk = abs(entry_price - sl)
                    reward = abs(tp - entry_price)
                    # Simple outcome model: win if confidence > 0.5, lose otherwise
                    if sig.get("confidence", 0.5) > 0.5:
                        raw_pnl = reward * sig.get("size", 1.0)
                    else:
                        raw_pnl = -risk * sig.get("size", 1.0)
                else:
                    raw_pnl = 0

                net_pnl = raw_pnl - comm
                pnls.append(net_pnl)
                equity += net_pnl

        trade_count = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / trade_count if trade_count > 0 else 0

        total_pnl = sum(pnls)
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        max_dd = self._calc_max_drawdown(pnls)
        sharpe = self._calc_sharpe(pnls)

        avg_r = float(np.mean(pnls)) / (float(np.std(pnls)) + 1e-9) if pnls else 0.0

        return BacktestResult(
            window_index=window_idx,
            train_start=str(train_data.index[0]),
            train_end=str(train_data.index[-1]),
            test_start=str(test_data.index[0]),
            test_end=str(test_data.index[-1]),
            trade_count=trade_count,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            avg_r_multiple=avg_r,
            profit_factor=profit_factor,
        )

    @staticmethod
    def _calc_max_drawdown(pnls: List[float]) -> float:
        if not pnls:
            return 0.0
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative)
        # Normalise by peak (avoid div-by-zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd_pct = np.where(peak > 0, drawdowns / peak, 0.0)
        return float(np.max(dd_pct)) if len(dd_pct) > 0 else 0.0

    @staticmethod
    def _calc_sharpe(pnls: List[float], annualization: float = 252**0.5) -> float:
        if len(pnls) < 2:
            return 0.0
        arr = np.array(pnls)
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return 0.0
        return float(mean / std * annualization)
