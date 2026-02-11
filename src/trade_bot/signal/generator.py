"""Signal generation framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

from ..core.models import MarketState, TradingAction
from ..core.enums import Action, Regime

logger = logging.getLogger(__name__)


class SignalGenerator(ABC):
    """Abstract base class for signal generators."""
    
    @abstractmethod
    def generate_signal(
        self, 
        market_state: MarketState
    ) -> Optional[TradingAction]:
        """Generate trading signal."""
        pass
    
    @abstractmethod
    def get_required_regime(self) -> Regime:
        """Get the regime this generator works best in."""
        pass


class MeanReversionSignal(SignalGenerator):
    """Mean reversion signal generator."""
    
    def __init__(self, lookback: int = 20, threshold: float = 2.0):
        """Initialize mean reversion signal generator."""
        self.lookback = lookback
        self.threshold = threshold
        self.required_regime = Regime.MEAN_REVERT
    
    def generate_signal(self, market_state: MarketState) -> Optional[TradingAction]:
        """Generate mean reversion signal."""
        try:
            df = market_state.ohlcv
            if len(df) < self.lookback:
                return None
            
            # Calculate Bollinger Bands
            sma = df['close'].rolling(window=self.lookback).mean()
            std = df['close'].rolling(window=self.lookback).std()
            upper_band = sma + (self.threshold * std)
            lower_band = sma - (self.threshold * std)
            
            current_price = market_state.current_price
            current_sma = sma.iloc[-1]
            current_std = std.iloc[-1]
            
            logger.debug(
                f"MeanReversion: price={current_price:.2f} lower={lower_band.iloc[-1]:.2f} "
                f"upper={upper_band.iloc[-1]:.2f} sma={current_sma:.2f}"
            )

            # Generate signals based on Bollinger Band penetration
            if current_price <= lower_band.iloc[-1]:
                # Price is below lower band - potential buy signal
                stop_distance = current_std * self.threshold
                expected_return = current_sma - current_price
                expected_risk = stop_distance
                
                if expected_return > 0 and expected_risk > 0:
                    return TradingAction(
                        action=Action.BUY,
                        size=1.0,  # Will be calculated by risk manager
                        stop_loss=current_price - stop_distance,
                        take_profit=current_sma,
                        expected_return=expected_return,
                        expected_risk=expected_risk,
                        confidence=min(abs(current_price - lower_band.iloc[-1]) / (current_std * self.threshold), 1.0)
                    )
            
            elif current_price >= upper_band.iloc[-1]:
                # Price is above upper band - potential sell signal
                stop_distance = current_std * self.threshold
                expected_return = current_price - current_sma
                expected_risk = stop_distance
                
                if expected_return > 0 and expected_risk > 0:
                    return TradingAction(
                        action=Action.SELL,
                        size=1.0,  # Will be calculated by risk manager
                        stop_loss=current_price + stop_distance,
                        take_profit=current_sma,
                        expected_return=expected_return,
                        expected_risk=expected_risk,
                        confidence=min(abs(current_price - upper_band.iloc[-1]) / (current_std * self.threshold), 1.0)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in mean reversion signal generation: {e}")
            return None
    
    def get_required_regime(self) -> Regime:
        """Get required regime."""
        return self.required_regime


class TrendFollowingSignal(SignalGenerator):
    """Trend following signal generator."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        """Initialize trend following signal generator."""
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.required_regime = Regime.TRENDING
    
    def generate_signal(self, market_state: MarketState) -> Optional[TradingAction]:
        """Generate trend following signal."""
        try:
            df = market_state.ohlcv
            if len(df) < self.slow_period:
                return None
            
            # Calculate moving averages
            fast_ma = df['close'].rolling(window=self.fast_period).mean()
            slow_ma = df['close'].rolling(window=self.slow_period).mean()
            
            current_price = market_state.current_price
            current_fast = fast_ma.iloc[-1]
            current_slow = slow_ma.iloc[-1]
            
            # Calculate ATR for stop loss
            atr = market_state.atr
            
            prev_fast = fast_ma.iloc[-2]
            prev_slow = slow_ma.iloc[-2]

            # Generate signals based on moving average crossover
            if current_fast > current_slow and (prev_fast <= prev_slow or current_price > current_fast):
                # Uptrend - potential buy signal
                stop_distance = atr * 2.0
                expected_return = atr * 3.0  # 3:1 R:R ratio
                expected_risk = stop_distance
                
                return TradingAction(
                    action=Action.BUY,
                    size=1.0,  # Will be calculated by risk manager
                    stop_loss=current_price - stop_distance,
                    take_profit=current_price + expected_return,
                    expected_return=expected_return,
                    expected_risk=expected_risk,
                    confidence=min((current_fast - current_slow) / (current_slow * 0.02), 1.0)
                )
            
            elif current_fast < current_slow and (prev_fast >= prev_slow or current_price < current_fast):
                # Downtrend - potential sell signal
                stop_distance = atr * 2.0
                expected_return = atr * 3.0  # 3:1 R:R ratio
                expected_risk = stop_distance
                
                return TradingAction(
                    action=Action.SELL,
                    size=1.0,  # Will be calculated by risk manager
                    stop_loss=current_price + stop_distance,
                    take_profit=current_price - expected_return,
                    expected_return=expected_return,
                    expected_risk=expected_risk,
                    confidence=min((current_slow - current_fast) / (current_slow * 0.02), 1.0)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in trend following signal generation: {e}")
            return None
    
    def get_required_regime(self) -> Regime:
        """Get required regime."""
        return self.required_regime


class VolatilityBreakoutSignal(SignalGenerator):
    """Volatility breakout signal generator."""
    
    def __init__(self, lookback: int = 20, multiplier: float = 2.0):
        """Initialize volatility breakout signal generator."""
        self.lookback = lookback
        self.multiplier = multiplier
        self.required_regime = Regime.HIGH_VOL
    
    def generate_signal(self, market_state: MarketState) -> Optional[TradingAction]:
        """Generate volatility breakout signal."""
        try:
            df = market_state.ohlcv
            if len(df) < self.lookback:
                return None
            
            # Calculate volatility bands
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=self.lookback).std()
            avg_volatility = volatility.mean()
            
            current_price = market_state.current_price
            current_vol = volatility.iloc[-1]
            
            # Calculate price levels
            resistance = df['high'].rolling(window=self.lookback).max().iloc[-1]
            support = df['low'].rolling(window=self.lookback).min().iloc[-1]
            
            atr = market_state.atr
            stop_distance = atr * self.multiplier
            
            # Generate breakout signals
            if current_price > resistance:
                # Breakout above resistance
                expected_return = atr * 2.5
                expected_risk = stop_distance
                
                return TradingAction(
                    action=Action.BUY,
                    size=1.0,
                    stop_loss=current_price - stop_distance,
                    take_profit=current_price + expected_return,
                    expected_return=expected_return,
                    expected_risk=expected_risk,
                    confidence=min(current_vol / (avg_volatility * 2), 1.0)
                )
            
            elif current_price < support:
                # Breakout below support
                expected_return = atr * 2.5
                expected_risk = stop_distance
                
                return TradingAction(
                    action=Action.SELL,
                    size=1.0,
                    stop_loss=current_price + stop_distance,
                    take_profit=current_price - expected_return,
                    expected_return=expected_return,
                    expected_risk=expected_risk,
                    confidence=min(current_vol / (avg_volatility * 2), 1.0)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in volatility breakout signal generation: {e}")
            return None
    
    def get_required_regime(self) -> Regime:
        """Get required regime."""
        return self.required_regime


class SignalManager:
    """Manages multiple signal generators."""

    def __init__(self):
        """Initialize signal manager."""
        self.generators: List[SignalGenerator] = []
        self.regime_generators: Dict[Regime, List[SignalGenerator]] = {}
        self._last_price: Optional[float] = None  # Cache for LLM adjustments
        logger.info("Signal manager initialized")
    
    def add_generator(self, generator: SignalGenerator):
        """Add a signal generator."""
        self.generators.append(generator)
        regime = generator.get_required_regime()
        if regime not in self.regime_generators:
            self.regime_generators[regime] = []
        self.regime_generators[regime].append(generator)
        logger.info(f"Added signal generator for {regime.value} regime")
    
    def generate_signals(
        self,
        market_state: MarketState
    ) -> List[TradingAction]:
        """Generate signals from all generators. Regime match boosts confidence."""
        # Cache last price for LLM adjustments
        self._last_price = market_state.current_price

        signals = []
        current_regime = market_state.regime_label

        for generator in self.generators:
            try:
                signal = generator.generate_signal(market_state)
                if signal:
                    # Boost confidence if regime matches, penalize if not
                    if generator.get_required_regime() == current_regime:
                        signal.confidence = min(signal.confidence * 1.2, 1.0)
                    else:
                        signal.confidence *= 0.7
                    signals.append(signal)
                    logger.debug(
                        f"{generator.__class__.__name__}: {signal.action.value} "
                        f"conf={signal.confidence:.2f} (regime={current_regime.value})"
                    )
            except Exception as e:
                logger.error(f"Error generating signal from {generator.__class__.__name__}: {e}")

        logger.info(f"Generated {len(signals)} signals for {current_regime.value} regime")
        return signals
    
    def get_best_signal(self, signals: List[TradingAction]) -> Optional[TradingAction]:
        """Select the best signal from multiple signals."""
        if not signals:
            return None

        # Sort by confidence and return the highest confidence signal
        best_signal = max(signals, key=lambda s: s.confidence)
        logger.info(f"Selected best signal: {best_signal.action.value} with confidence {best_signal.confidence:.2f}")
        return best_signal

    def get_last_price(self) -> Optional[float]:
        """Get the last cached market price."""
        return self._last_price


def create_default_signal_manager() -> SignalManager:
    """Create a signal manager with default generators."""
    manager = SignalManager()
    
    # Add default signal generators (with looser thresholds for 1m timeframe)
    manager.add_generator(MeanReversionSignal(lookback=14, threshold=1.5))
    manager.add_generator(TrendFollowingSignal(fast_period=5, slow_period=15))
    manager.add_generator(VolatilityBreakoutSignal(lookback=14, multiplier=1.5))
    
    logger.info("Created default signal manager")
    return manager
