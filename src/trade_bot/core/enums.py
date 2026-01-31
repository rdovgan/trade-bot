"""Core enumerations for the trading system."""

from enum import Enum


class Action(str, Enum):
    """Trading action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Side(str, Enum):
    """Position sides."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Regime(str, Enum):
    """Market regimes."""
    TRENDING = "trending"
    MEAN_REVERT = "mean_revert"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RiskLevel(str, Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
