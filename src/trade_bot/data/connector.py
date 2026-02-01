"""Market data connector interface and implementations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator, Sequence
import asyncio
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt

from ..core.models import MarketState
from ..core.enums import Regime

logger = logging.getLogger(__name__)



def _apply_sandbox_urls(exchange, exchange_name: str):
    """Apply sandbox/demo URLs for supported exchanges."""
    if hasattr(exchange, 'enable_demo_trading'):
        try:
            exchange.enable_demo_trading(True)
            logger.info(f"{exchange_name} demo trading enabled")
            return
        except Exception:
            pass
    exchange.set_sandbox_mode(True)
    logger.info(f"{exchange_name} sandbox mode enabled")


class DataConnector(ABC):
    """Abstract base class for market data connectors."""
    
    @abstractmethod
    async def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 100
    ) -> pd.DataFrame:
        """Get OHLCV data for a symbol."""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data."""
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book data."""
        pass
    
    @abstractmethod
    async def stream_trades(
        self, 
        symbol: str
    ) -> AsyncGenerator[Dict, None]:
        """Stream real-time trades."""
        pass
    
    @abstractmethod
    async def get_all_tickers(self, symbols: Optional[List[str]] = None) -> Dict:
        """Fetch all tickers at once."""
        pass

    @abstractmethod
    async def get_markets(self) -> Dict:
        """Fetch market info (all available instruments)."""
        pass

    @abstractmethod
    async def close(self):
        """Close connection."""
        pass


class CCXTConnector(DataConnector):
    """CCXT-based data connector for cryptocurrency exchanges."""
    
    def __init__(self, exchange_name: str, config: Optional[Dict] = None):
        """Initialize CCXT connector."""
        self.exchange_name = exchange_name
        self.config = config or {}
        
        # Initialize exchange
        is_sandbox = self.config.get('sandbox', False)
        # Binance uses 'future', Bybit/OKX use 'linear'
        type_map = {'binance': 'future', 'binanceusdm': 'future'}
        default_type = type_map.get(exchange_name, 'linear')
        ccxt_keys = {k: v for k, v in self.config.items() if k != 'sandbox'}
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {'defaultType': default_type},
            **ccxt_keys
        })
        if is_sandbox:
            _apply_sandbox_urls(self.exchange, exchange_name)

        logger.info(f"Initialized CCXT connector for {exchange_name} (sandbox={is_sandbox}, type={default_type})")
    
    async def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '1m', 
        limit: int = 100
    ) -> pd.DataFrame:
        """Get OHLCV data."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f"Retrieved {len(df)} OHLCV bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last': ticker.get('last', 0),
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'volume': ticker.get('baseVolume', 0),
                'timestamp': ticker.get('timestamp', 0),
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book data."""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit)

            # Calculate order book imbalance
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            bids_total = sum(bid[1] for bid in bids[:10]) if bids else 0
            asks_total = sum(ask[1] for ask in asks[:10]) if asks else 0
            imbalance = (bids_total - asks_total) / (bids_total + asks_total) if (bids_total + asks_total) > 0 else 0

            return {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'imbalance': imbalance,
                'spread': asks[0][0] - bids[0][0] if bids and asks else 0,
                'timestamp': orderbook.get('timestamp', 0),
            }
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise
    
    async def stream_trades(
        self, 
        symbol: str
    ) -> AsyncGenerator[Dict, None]:
        """Stream real-time trades."""
        while True:
            try:
                trades = await self.exchange.fetch_trades(symbol, limit=10)
                for trade in trades:
                    yield {
                        'symbol': symbol,
                        'price': trade['price'],
                        'amount': trade['amount'],
                        'side': trade['side'],
                        'timestamp': trade['timestamp'],
                    }
                await asyncio.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error in trade stream for {symbol}: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def get_all_tickers(self, symbols: Optional[List[str]] = None) -> Dict:
        """Fetch all tickers at once via CCXT fetch_tickers."""
        try:
            tickers = await self.exchange.fetch_tickers(symbols)
            logger.debug(f"Fetched {len(tickers)} tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            raise

    async def get_markets(self) -> Dict:
        """Fetch market info via CCXT load_markets."""
        try:
            markets = await self.exchange.load_markets()
            logger.debug(f"Loaded {len(markets)} markets")
            return markets
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            raise

    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for perpetual."""
        try:
            if hasattr(self.exchange, 'fetch_funding_rate'):
                funding = await self.exchange.fetch_funding_rate(symbol)
                return float(funding.get('fundingRate', 0))
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return 0.0

    async def get_funding_rate_history(self, symbol: str, since: int, limit: int = 100) -> List[Dict]:
        """Get funding rate history."""
        try:
            if hasattr(self.exchange, 'fetch_funding_rate_history'):
                return await self.exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
            return []
        except Exception as e:
            logger.error(f"Error fetching funding history for {symbol}: {e}")
            return []

    async def close(self):
        """Close exchange connection."""
        await self.exchange.close()
        logger.info(f"Closed connection to {self.exchange_name}")


class MarketDataProcessor:
    """Processes market data and calculates indicators."""

    def __init__(self, lookback_periods: Dict[str, int] = None, max_data_age_multiplier: float = 1.0):
        """Initialize market data processor.

        Args:
            lookback_periods: Number of periods to use for each indicator
            max_data_age_multiplier: Multiplier for max acceptable data age.
                                    Use higher values (e.g., 100) for sandbox/testnet.
                                    Default 1.0 for production strict validation.
        """
        self.max_data_age_multiplier = max_data_age_multiplier
        self.lookback_periods = lookback_periods or {
            'atr': 14,
            'volatility': 20,
            'volume_delta': 10,
            'regime': 50,
        }
        logger.info("Market data processor initialized")
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period + 1:
            return 0.0
        
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not atr.empty else 0.0
    
    def calculate_realized_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate realized volatility."""
        if len(df) < period:
            return 0.0
        
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(window=period).std() * (252 ** 0.5)  # Annualized
        
        return float(volatility.iloc[-1]) if not volatility.empty else 0.0
    
    def calculate_volume_delta(self, df: pd.DataFrame, period: int = 10) -> float:
        """Calculate volume delta (buy volume - sell volume)."""
        if len(df) < period:
            return 0.0
        
        # Approximate buy/sell volume using price direction
        price_change = df['close'].diff()
        volume = df['volume']
        
        # If price went up, assume more buying; if down, more selling
        buy_volume = volume[price_change > 0].sum()
        sell_volume = volume[price_change < 0].sum()
        
        delta = buy_volume - sell_volume
        return float(delta)
    
    def calculate_liquidity_score(self, order_book: Dict) -> float:
        """Calculate liquidity score from order book."""
        if not order_book.get('bids') or not order_book.get('asks'):
            return 0.0
        
        # Sum volumes at top of book
        bid_volume = sum(bid[1] for bid in order_book['bids'][:10])
        ask_volume = sum(ask[1] for ask in order_book['asks'][:10])
        total_volume = bid_volume + ask_volume
        
        # Normalize score (0-1)
        # This is a simplified calculation - in production, you'd want more sophisticated metrics
        max_volume = 1000000  # Adjust based on typical market conditions
        score = min(total_volume / max_volume, 1.0)
        
        return float(score)
    
    def detect_regime(self, df: pd.DataFrame, period: int = 50) -> Regime:
        """Detect market regime."""
        if len(df) < period:
            return Regime.LOW_VOL
        
        # Calculate trend strength
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(window=min(period, len(returns))).std()
        
        # Calculate trend direction
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period]
        
        # Determine regime
        current_vol = volatility.iloc[-1] if not volatility.empty else 0
        vol_percentile = (volatility.rank(pct=True).iloc[-1] * 100) if not volatility.empty else 50
        
        if vol_percentile > 80:
            return Regime.HIGH_VOL
        elif vol_percentile < 20:
            return Regime.LOW_VOL
        elif abs(price_change) > 0.05:  # 5% move over period
            return Regime.TRENDING
        else:
            return Regime.MEAN_REVERT

    def _validate_ohlcv_freshness(self, ohlcv: pd.DataFrame, timeframe: str) -> bool:
        """Validate that market data is fresh."""
        if ohlcv.empty:
            return False

        latest_timestamp = ohlcv.index[-1]
        now = datetime.now()

        # Map timeframe to base max acceptable age
        base_max_age_seconds = {
            '1m': 120,   # 2 minutes
            '5m': 600,   # 10 minutes
            '15m': 1800, # 30 minutes
            '1h': 7200,  # 2 hours
        }.get(timeframe, 300)

        # Apply multiplier (higher for sandbox/testnet)
        max_age_seconds = base_max_age_seconds * self.max_data_age_multiplier

        age = (now - latest_timestamp).total_seconds()

        if age > max_age_seconds:
            logger.warning(f"Market data is stale: age={age:.0f}s, max={max_age_seconds:.0f}s")
            return False

        return True

    def _validate_ohlcv_data(self, ohlcv: pd.DataFrame) -> bool:
        """Validate OHLCV data quality."""
        if ohlcv.empty:
            return False

        # Check for NaN/inf
        if ohlcv.isnull().any().any() or np.isinf(ohlcv.values).any():
            logger.error("OHLCV contains NaN or inf values")
            return False

        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        if (ohlcv[price_cols] <= 0).any().any():
            logger.error("OHLCV contains zero or negative prices")
            return False

        # Check for time gaps
        try:
            expected_interval = pd.infer_freq(ohlcv.index)
            if expected_interval is None:
                logger.warning("Cannot infer OHLCV frequency - possible gaps")
        except Exception as e:
            logger.warning(f"Could not check OHLCV frequency: {e}")

        return True

    async def create_market_state(
        self,
        symbol: str,
        connector: DataConnector,
        timeframe: str = '1m'
    ) -> MarketState:
        """Create comprehensive market state with validation."""
        try:
            # Get market data
            ohlcv = await connector.get_ohlcv(symbol, timeframe, limit=200)

            # Validate data quality
            if not self._validate_ohlcv_data(ohlcv):
                raise ValueError(f"Invalid OHLCV data for {symbol}")

            # Validate freshness
            if not self._validate_ohlcv_freshness(ohlcv, timeframe):
                raise ValueError(f"Stale market data for {symbol}")

            ticker = await connector.get_ticker(symbol)
            order_book = await connector.get_order_book(symbol)
            
            # Calculate indicators
            atr = self.calculate_atr(ohlcv, self.lookback_periods['atr'])
            volatility = self.calculate_realized_volatility(ohlcv, self.lookback_periods['volatility'])
            volume_delta = self.calculate_volume_delta(ohlcv, self.lookback_periods['volume_delta'])
            liquidity_score = self.calculate_liquidity_score(order_book)
            regime = self.detect_regime(ohlcv, self.lookback_periods['regime'])
            
            # Calculate volatility percentile
            returns = ohlcv['close'].pct_change().dropna()
            if len(returns) > 20:
                vol_percentile = (returns.rolling(20).std().rank(pct=True).iloc[-1] * 100)
            else:
                vol_percentile = 50
            
            # Calculate spread
            bid = ticker.get('bid', 0) or 0
            ask = ticker.get('ask', 0) or 0
            spread = ask - bid if bid and ask else 0

            market_state = MarketState(
                ohlcv=ohlcv,
                current_price=ticker.get('last', 0) or 0,
                atr=atr,
                realized_volatility=volatility,
                spread=spread,
                order_book_imbalance=order_book['imbalance'],
                volume_delta=volume_delta,
                liquidity_score=liquidity_score,
                regime_label=regime,
                volatility_percentile=vol_percentile,
            )
            
            logger.debug(f"Created market state for {symbol}: {regime.value} regime")
            return market_state
            
        except Exception as e:
            logger.error(f"Error creating market state for {symbol}: {e}")
            raise


class DataManager:
    """Manages market data collection and processing."""

    def __init__(self, connector: DataConnector, max_data_age_multiplier: float = 1.0):
        """Initialize data manager.

        Args:
            connector: Data connector for fetching market data
            max_data_age_multiplier: Multiplier for max acceptable data age.
                                    Use higher values (e.g., 100) for sandbox/testnet.
        """
        self.connector = connector
        self.processor = MarketDataProcessor(max_data_age_multiplier=max_data_age_multiplier)
        self._running = False
        
    async def start(self):
        """Start data collection."""
        self._running = True
        logger.info("Data manager started")
    
    async def stop(self):
        """Stop data collection."""
        self._running = False
        await self.connector.close()
        logger.info("Data manager stopped")
    
    async def get_market_state(self, symbol: str, timeframe: str = '1m') -> MarketState:
        """Get current market state for a symbol."""
        return await self.processor.create_market_state(symbol, self.connector, timeframe)
