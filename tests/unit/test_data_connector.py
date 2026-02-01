"""Unit tests for data connector and validation (Phase 5)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from trade_bot.data.connector import MarketDataProcessor, CCXTConnector


class MockDataConnector:
    """Mock data connector for testing."""

    def __init__(self):
        self.ohlcv_data = None
        self.ticker_data = {}
        self.order_book_data = {'imbalance': 0.0}

    async def get_ohlcv(self, symbol, timeframe, limit=100):
        if self.ohlcv_data is not None:
            return self.ohlcv_data
        # Default valid data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1min')
        return pd.DataFrame({
            'open': [100.0] * limit,
            'high': [101.0] * limit,
            'low': [99.0] * limit,
            'close': [100.5] * limit,
            'volume': [1000.0] * limit,
        }, index=dates)

    async def get_ticker(self, symbol):
        return self.ticker_data or {'last': 100.0, 'bid': 99.5, 'ask': 100.5}

    async def get_order_book(self, symbol, limit=100):
        return self.order_book_data


@pytest.fixture
def mock_connector():
    return MockDataConnector()


@pytest.fixture
def processor():
    return MarketDataProcessor()


class TestDataValidation:
    """Tests for OHLCV data validation."""

    def test_validate_ohlcv_data_valid(self, processor):
        """Test validation passes for valid data."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        ohlcv = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates)

        assert processor._validate_ohlcv_data(ohlcv) is True

    def test_validate_ohlcv_data_empty(self, processor):
        """Test validation fails for empty dataframe."""
        ohlcv = pd.DataFrame()
        assert processor._validate_ohlcv_data(ohlcv) is False

    def test_validate_ohlcv_data_nan_values(self, processor):
        """Test validation fails for NaN values."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        ohlcv = pd.DataFrame({
            'open': [100.0] * 99 + [np.nan],
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates)

        assert processor._validate_ohlcv_data(ohlcv) is False

    def test_validate_ohlcv_data_inf_values(self, processor):
        """Test validation fails for infinite values."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        ohlcv = pd.DataFrame({
            'open': [100.0] * 99 + [np.inf],
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates)

        assert processor._validate_ohlcv_data(ohlcv) is False

    def test_validate_ohlcv_data_zero_prices(self, processor):
        """Test validation fails for zero prices."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        ohlcv = pd.DataFrame({
            'open': [100.0] * 99 + [0.0],
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates)

        assert processor._validate_ohlcv_data(ohlcv) is False

    def test_validate_ohlcv_data_negative_prices(self, processor):
        """Test validation fails for negative prices."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        ohlcv = pd.DataFrame({
            'open': [100.0] * 99 + [-1.0],
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates)

        assert processor._validate_ohlcv_data(ohlcv) is False


class TestDataFreshness:
    """Tests for data freshness validation."""

    def test_validate_freshness_recent_data(self, processor):
        """Test validation passes for recent data."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        ohlcv = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates)

        assert processor._validate_ohlcv_freshness(ohlcv, '1m') is True

    def test_validate_freshness_stale_data(self, processor):
        """Test validation fails for stale data."""
        # Data from 10 minutes ago for 1m timeframe (max age 2 minutes)
        dates = pd.date_range(end=datetime.now() - timedelta(minutes=10), periods=100, freq='1min')
        ohlcv = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates)

        assert processor._validate_ohlcv_freshness(ohlcv, '1m') is False

    def test_validate_freshness_empty_data(self, processor):
        """Test validation fails for empty data."""
        ohlcv = pd.DataFrame()
        assert processor._validate_ohlcv_freshness(ohlcv, '1m') is False

    def test_validate_freshness_5m_timeframe(self, processor):
        """Test validation with 5m timeframe (10 minute max age)."""
        # Data from 5 minutes ago should pass for 5m timeframe
        dates = pd.date_range(end=datetime.now() - timedelta(minutes=5), periods=100, freq='5min')
        ohlcv = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates)

        assert processor._validate_ohlcv_freshness(ohlcv, '5m') is True

        # Data from 15 minutes ago should fail
        dates_old = pd.date_range(end=datetime.now() - timedelta(minutes=15), periods=100, freq='5min')
        ohlcv_old = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates_old)

        assert processor._validate_ohlcv_freshness(ohlcv_old, '5m') is False


class TestMarketStateValidation:
    """Tests for market state creation with validation."""

    @pytest.mark.asyncio
    async def test_create_market_state_valid_data(self, processor, mock_connector):
        """Test market state creation with valid data."""
        market_state = await processor.create_market_state("BTC/USDT", mock_connector, "1m")

        assert market_state is not None
        assert market_state.current_price == 100.0
        assert market_state.atr > 0

    @pytest.mark.asyncio
    async def test_create_market_state_invalid_data_raises(self, processor, mock_connector):
        """Test market state creation raises on invalid data."""
        # Set invalid data (NaN values)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        mock_connector.ohlcv_data = pd.DataFrame({
            'open': [100.0] * 99 + [np.nan],
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates)

        with pytest.raises(ValueError, match="Invalid OHLCV data"):
            await processor.create_market_state("BTC/USDT", mock_connector, "1m")

    @pytest.mark.asyncio
    async def test_create_market_state_stale_data_raises(self, processor, mock_connector):
        """Test market state creation raises on stale data."""
        # Set stale data
        dates = pd.date_range(end=datetime.now() - timedelta(minutes=10), periods=100, freq='1min')
        mock_connector.ohlcv_data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100,
        }, index=dates)

        with pytest.raises(ValueError, match="Stale market data"):
            await processor.create_market_state("BTC/USDT", mock_connector, "1m")


class TestFundingRates:
    """Tests for funding rate fetching (Phase 4)."""

    @pytest.mark.asyncio
    async def test_get_funding_rate_success(self):
        """Test fetching funding rate successfully."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_funding_rate = AsyncMock(return_value={'fundingRate': 0.0001})

        connector = CCXTConnector.__new__(CCXTConnector)
        connector.exchange = mock_exchange

        rate = await connector.get_funding_rate("BTC/USDT")

        assert rate == 0.0001
        mock_exchange.fetch_funding_rate.assert_called_once_with("BTC/USDT")

    @pytest.mark.asyncio
    async def test_get_funding_rate_no_support(self):
        """Test handling when exchange doesn't support funding rates."""
        mock_exchange = MagicMock()
        del mock_exchange.fetch_funding_rate  # Exchange doesn't have this method

        connector = CCXTConnector.__new__(CCXTConnector)
        connector.exchange = mock_exchange

        rate = await connector.get_funding_rate("BTC/USDT")

        assert rate == 0.0

    @pytest.mark.asyncio
    async def test_get_funding_rate_error(self):
        """Test handling errors when fetching funding rate."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_funding_rate = AsyncMock(side_effect=Exception("Network error"))

        connector = CCXTConnector.__new__(CCXTConnector)
        connector.exchange = mock_exchange

        rate = await connector.get_funding_rate("BTC/USDT")

        assert rate == 0.0
