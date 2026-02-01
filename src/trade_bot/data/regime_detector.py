"""Advanced regime detection with statistical methods."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..core.models import MarketState
from ..core.enums import Regime

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Statistical regime detection system."""
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        """Initialize regime detector."""
        self.lookback_periods = lookback_periods or {
            'short': 20,   # Short-term (20 periods)
            'medium': 50,  # Medium-term (50 periods)
            'long': 100,   # Long-term (100 periods)
        }
        
        # Regime detection parameters
        self.trend_threshold = 0.02  # 2% trend threshold
        self.volatility_threshold = 0.15  # 15% volatility threshold
        self.mean_reversion_threshold = 0.3  # Mean reversion score threshold
        
        logger.info("Advanced regime detector initialized")
    
    def detect_regime(self, df: pd.DataFrame, current_price: float) -> Regime:
        """
        Detect market regime using statistical analysis.
        
        Args:
            df: OHLCV data
            current_price: Current market price
            
        Returns:
            Detected regime
        """
        try:
            if len(df) < max(self.lookback_periods.values()):
                logger.warning("Insufficient data for regime detection")
                return Regime.LOW_VOL
            
            # Calculate statistical indicators
            indicators = self._calculate_indicators(df)
            
            # Calculate regime scores
            trend_score = self._calculate_trend_score(indicators, current_price)
            volatility_score = self._calculate_volatility_score(indicators)
            mean_reversion_score = self._calculate_mean_reversion_score(indicators)
            
            # Determine regime based on scores
            regime = self._determine_regime(trend_score, volatility_score, mean_reversion_score)
            
            logger.debug(
                f"Regime detection: trend={trend_score:.3f}, "
                f"volatility={volatility_score:.3f}, "
                f"mean_reversion={mean_reversion_score:.3f} -> {regime.value}"
            )
            
            return regime
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return Regime.LOW_VOL
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical indicators for regime detection."""
        indicators = {}
        
        # Price-based indicators
        indicators['returns'] = df['close'].pct_change().dropna()
        indicators['log_returns'] = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        # Trend indicators
        for period_name, period in self.lookback_periods.items():
            if len(df) >= period:
                # Moving averages
                indicators[f'ma_{period_name}'] = df['close'].rolling(window=period).mean()
                indicators[f'ma_slope_{period_name}'] = self._calculate_slope(
                    indicators[f'ma_{period_name}'].dropna()
                )
                
                # Price relative to MA
                current_ma = indicators[f'ma_{period_name}'].iloc[-1]
                indicators[f'price_to_ma_{period_name}'] = (df['close'].iloc[-1] - current_ma) / current_ma
        
        # Volatility indicators
        for period_name, period in self.lookback_periods.items():
            if len(indicators['returns']) >= period:
                indicators[f'volatility_{period_name}'] = indicators['returns'].rolling(window=period).std()
                indicators[f'volatility_percentile_{period_name}'] = (
                    indicators[f'volatility_{period_name}'].rank(pct=True).iloc[-1]
                )
        
        # Mean reversion indicators
        indicators['hurst_exponent'] = self._calculate_hurst_exponent(indicators['log_returns'])
        indicators['autocorrelation'] = self._calculate_autocorrelation(indicators['returns'])
        
        return indicators
    
    def _calculate_trend_score(self, indicators: Dict[str, float], current_price: float) -> float:
        """Calculate trend strength score (0-1)."""
        trend_components = []
        
        # Moving average slopes
        for period_name in self.lookback_periods.keys():
            slope_key = f'ma_slope_{period_name}'
            if slope_key in indicators:
                slope = indicators[slope_key]
                # Normalize slope (assuming reasonable range)
                normalized_slope = min(abs(slope) * 1000, 1.0)  # Scale and cap
                trend_components.append(normalized_slope)
        
        # Price distance from moving averages
        for period_name in self.lookback_periods.keys():
            price_to_ma_key = f'price_to_ma_{period_name}'
            if price_to_ma_key in indicators:
                distance = abs(indicators[price_to_ma_key])
                # Normalize distance (assuming 5% is significant)
                normalized_distance = min(distance / 0.05, 1.0)
                trend_components.append(normalized_distance)
        
        # Calculate average trend score
        if trend_components:
            return np.mean(trend_components)
        return 0.0
    
    def _calculate_volatility_score(self, indicators: Dict[str, float]) -> float:
        """Calculate volatility score (0-1)."""
        volatility_components = []
        
        # Current volatility percentiles
        for period_name in self.lookback_periods.keys():
            vol_percentile_key = f'volatility_percentile_{period_name}'
            if vol_percentile_key in indicators:
                percentile = indicators[vol_percentile_key]
                volatility_components.append(percentile)
        
        # Calculate average volatility score
        if volatility_components:
            return np.mean(volatility_components)
        return 0.5  # Neutral if no data
    
    def _calculate_mean_reversion_score(self, indicators: Dict[str, float]) -> float:
        """Calculate mean reversion score (0-1)."""
        reversion_components = []
        
        # Hurst exponent (H < 0.5 indicates mean reversion)
        if 'hurst_exponent' in indicators:
            hurst = indicators['hurst_exponent']
            # Convert to mean reversion score (closer to 0 = more mean reverting)
            reversion_score = 1.0 - min(abs(hurst - 0.5) * 2, 1.0)
            reversion_components.append(reversion_score)
        
        # Autocorrelation (negative autocorrelation suggests mean reversion)
        if 'autocorrelation' in indicators:
            autocorr = indicators['autocorrelation']
            # Convert negative autocorrelation to positive score
            reversion_score = max(0, -autocorr)
            reversion_components.append(reversion_score)
        
        # Calculate average mean reversion score
        if reversion_components:
            return np.mean(reversion_components)
        return 0.5  # Neutral if no data
    
    def _determine_regime(
        self,
        trend_score: float,
        volatility_score: float,
        mean_reversion_score: float
    ) -> Regime:
        """Determine regime based on statistical scores."""
        
        # High volatility regime
        if volatility_score > 0.8:
            return Regime.HIGH_VOL
        
        # Low volatility regime
        if volatility_score < 0.2:
            return Regime.LOW_VOL
        
        # Trending regime
        if trend_score > 0.6 and mean_reversion_score < 0.4:
            return Regime.TRENDING
        
        # Mean reversion regime
        if mean_reversion_score > 0.6 and trend_score < 0.4:
            return Regime.MEAN_REVERT
        
        # Default based on dominant factor
        if trend_score > mean_reversion_score:
            return Regime.TRENDING
        else:
            return Regime.MEAN_REVERT
    
    def _calculate_slope(self, series: pd.Series) -> float:
        """Calculate linear slope of a series."""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Calculate linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by series mean
        normalized_slope = slope / series.mean() if series.mean() != 0 else 0
        
        return normalized_slope
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for mean reversion detection."""
        try:
            if len(returns) < 50:
                return 0.5  # Random walk assumption
            
            # Simplified Hurst exponent calculation
            lags = range(2, min(20, len(returns) // 4))
            tau = [np.sqrt(np.std(np.subtract(returns.iloc[l:], returns.iloc[:-l]))) for l in lags]
            
            # Linear fit on log-log plot
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
            
            return hurst
            
        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5
    
    def _calculate_autocorrelation(self, returns: pd.Series, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag."""
        try:
            if len(returns) < lag + 10:
                return 0.0
            
            return returns.autocorr(lag=lag)
            
        except Exception as e:
            logger.error(f"Error calculating autocorrelation: {e}")
            return 0.0
    
    def get_regime_confidence(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> Tuple[Regime, float]:
        """
        Get regime with confidence score.
        
        Returns:
            Tuple of (regime, confidence_score)
        """
        try:
            if len(df) < max(self.lookback_periods.values()):
                return Regime.LOW_VOL, 0.0
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            
            # Calculate scores
            trend_score = self._calculate_trend_score(indicators, current_price)
            volatility_score = self._calculate_volatility_score(indicators)
            mean_reversion_score = self._calculate_mean_reversion_score(indicators)
            
            # Determine regime
            regime = self._determine_regime(trend_score, volatility_score, mean_reversion_score)
            
            # Calculate confidence based on score separation
            scores = [trend_score, volatility_score, mean_reversion_score]
            max_score = max(scores)
            second_max = sorted(scores)[-2]
            
            # Confidence based on how much the top score stands out
            confidence = (max_score - second_max) / max_score if max_score > 0 else 0.0
            confidence = max(0.0, min(1.0, confidence))
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return Regime.LOW_VOL, 0.0
