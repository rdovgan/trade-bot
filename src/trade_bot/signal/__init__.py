"""Signal generation module."""

from .generator import (
    SignalGenerator, 
    MeanReversionSignal, 
    TrendFollowingSignal, 
    VolatilityBreakoutSignal,
    SignalManager,
    create_default_signal_manager
)

__all__ = [
    "SignalGenerator",
    "MeanReversionSignal",
    "TrendFollowingSignal", 
    "VolatilityBreakoutSignal",
    "SignalManager",
    "create_default_signal_manager",
]
