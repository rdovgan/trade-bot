"""
AI Trading Agent with Risk-First Architecture

A production-ready trading system that prioritizes risk management above all else.
The risk validator has veto power over all trading decisions.
"""

__version__ = "0.1.0"
__author__ = "Trade Bot Team"

from .core.models import MarketState, PositionState, AccountState, RiskState
from .core.enums import Action, Regime, Side
from .risk.validator import RiskValidator
from .execution.engine import ExecutionEngine

__all__ = [
    "MarketState",
    "PositionState", 
    "AccountState",
    "RiskState",
    "Action",
    "Regime",
    "Side",
    "RiskValidator",
    "ExecutionEngine",
]
