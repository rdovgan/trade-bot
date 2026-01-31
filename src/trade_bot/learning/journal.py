"""Trade journal and performance tracking."""

import json
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pandas as pd
import numpy as np

from ..core.models import Trade, MarketState, Regime
from ..core.enums import Side

logger = logging.getLogger(__name__)


class TradeJournal:
    """Comprehensive trade journal for performance tracking and learning."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize trade journal."""
        if db_path is None:
            db_path = Path.home() / ".trade_bot" / "journal.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Trade journal initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    duration_seconds INTEGER,
                    pnl REAL,
                    pnl_pct REAL,
                    r_multiple REAL,
                    mae REAL,
                    mfe REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    regime TEXT,
                    volatility_percentile REAL,
                    liquidity_score REAL,
                    slippage REAL,
                    confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Daily performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    equity REAL,
                    balance REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    trades_count INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Strategy performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy_name TEXT,
                    regime TEXT,
                    trade_count INTEGER,
                    win_rate REAL,
                    avg_r_multiple REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    salience_score REAL,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (strategy_name, regime)
                )
            """)
            
            # Market context table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_context (
                    timestamp TEXT PRIMARY KEY,
                    symbol TEXT,
                    regime TEXT,
                    volatility_percentile REAL,
                    liquidity_score REAL,
                    spread REAL,
                    atr REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def record_trade(self, trade: Trade, market_state: MarketState):
        """Record a completed trade in the journal."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO trades (
                        id, symbol, side, quantity, entry_price, exit_price,
                        entry_time, exit_time, duration_seconds, pnl, pnl_pct,
                        r_multiple, mae, mfe, stop_loss, take_profit, regime,
                        volatility_percentile, liquidity_score, slippage, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.id,
                    trade.symbol,
                    trade.side.value,
                    trade.quantity,
                    trade.entry_price,
                    trade.exit_price,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    int(trade.duration.total_seconds()) if trade.duration else None,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.r_multiple,
                    trade.mae,
                    trade.mfe,
                    trade.stop_loss,
                    trade.take_profit,
                    trade.regime.value,
                    trade.volatility_percentile,
                    trade.liquidity_score,
                    trade.slippage,
                    trade.confidence,
                ))
                
                conn.commit()
                logger.info(f"Recorded trade {trade.id} in journal")
                
        except Exception as e:
            logger.error(f"Error recording trade {trade.id}: {e}")
            raise
    
    def record_market_context(self, market_state: MarketState, symbol: str):
        """Record market context for learning."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                timestamp = datetime.now().isoformat()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO market_context (
                        timestamp, symbol, regime, volatility_percentile,
                        liquidity_score, spread, atr
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    symbol,
                    market_state.regime_label.value,
                    market_state.volatility_percentile,
                    market_state.liquidity_score,
                    market_state.spread,
                    market_state.atr,
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording market context: {e}")
    
    def update_daily_performance(
        self,
        date: datetime,
        equity: float,
        balance: float,
        pnl: float,
        trades: List[Trade]
    ):
        """Update daily performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate metrics
                trades_count = len(trades)
                winning_trades = len([t for t in trades if t.pnl > 0])
                losing_trades = len([t for t in trades if t.pnl < 0])
                win_rate = winning_trades / trades_count if trades_count > 0 else 0
                
                # Calculate daily return percentage
                pnl_pct = (pnl / (equity - pnl)) * 100 if (equity - pnl) != 0 else 0
                
                # Calculate max drawdown for the day
                max_drawdown = self._calculate_max_drawdown(trades)
                
                # Calculate Sharpe ratio (simplified)
                sharpe_ratio = self._calculate_sharpe_ratio(trades)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_performance (
                        date, equity, balance, pnl, pnl_pct, trades_count,
                        winning_trades, losing_trades, max_drawdown, sharpe_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date.date().isoformat(),
                    equity,
                    balance,
                    pnl,
                    pnl_pct,
                    trades_count,
                    winning_trades,
                    losing_trades,
                    max_drawdown,
                    sharpe_ratio,
                ))
                
                conn.commit()
                logger.info(f"Updated daily performance for {date.date()}")
                
        except Exception as e:
            logger.error(f"Error updating daily performance: {e}")
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        regime: Optional[Regime] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Trade]:
        """Get trades from journal with optional filters."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM trades WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if regime:
                    query += " AND regime = ?"
                    params.append(regime.value)
                
                if start_date:
                    query += " AND entry_time >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND entry_time <= ?"
                    params.append(end_date.isoformat())
                
                query += " ORDER BY entry_time DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                trades = []
                for row in cursor.fetchall():
                    trade = self._row_to_trade(row)
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get performance statistics for the last N days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get trades for the period
                start_date = datetime.now() - timedelta(days=days)
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE entry_time >= ? 
                    ORDER BY entry_time DESC
                """, (start_date.isoformat(),))
                
                trades = [self._row_to_trade(row) for row in cursor.fetchall()]
                
                if not trades:
                    return {'error': 'No trades found in period'}
                
                # Calculate statistics
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t.pnl > 0])
                losing_trades = len([t for t in trades if t.pnl < 0])
                win_rate = winning_trades / total_trades
                
                total_pnl = sum(t.pnl for t in trades)
                avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if winning_trades > 0 else 0
                avg_loss = np.mean([t.pnl for t in trades if t.pnl < 0]) if losing_trades > 0 else 0
                
                # R-multiple statistics
                r_multiples = [t.r_multiple for t in trades if t.r_multiple != 0]
                avg_r_multiple = np.mean(r_multiples) if r_multiples else 0
                
                # Drawdown
                max_drawdown = self._calculate_max_drawdown(trades)
                
                # Sharpe ratio
                sharpe_ratio = self._calculate_sharpe_ratio(trades)
                
                # Profit factor
                gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
                gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                return {
                    'period_days': days,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'avg_r_multiple': avg_r_multiple,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'profit_factor': profit_factor,
                }
                
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return {'error': str(e)}
    
    def get_regime_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by market regime."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT regime, 
                           COUNT(*) as trade_count,
                           SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                           AVG(r_multiple) as avg_r_multiple,
                           SUM(pnl) as total_pnl,
                           AVG(pnl) as avg_pnl
                    FROM trades 
                    GROUP BY regime
                """)
                
                results = {}
                for row in cursor.fetchall():
                    regime, trade_count, wins, avg_r_multiple, total_pnl, avg_pnl = row
                    
                    if trade_count > 0:
                        win_rate = wins / trade_count
                        results[regime] = {
                            'trade_count': trade_count,
                            'win_rate': win_rate,
                            'avg_r_multiple': avg_r_multiple or 0,
                            'total_pnl': total_pnl or 0,
                            'avg_pnl': avg_pnl or 0,
                        }
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting regime performance: {e}")
            return {}
    
    def _row_to_trade(self, row) -> Trade:
        """Convert database row to Trade object."""
        return Trade(
            id=row[0],
            symbol=row[1],
            side=Side(row[2]),
            quantity=row[3],
            entry_price=row[4],
            exit_price=row[5],
            entry_time=datetime.fromisoformat(row[6]),
            exit_time=datetime.fromisoformat(row[7]) if row[7] else None,
            duration=timedelta(seconds=row[8]) if row[8] else timedelta(0),
            pnl=row[9],
            pnl_pct=row[10],
            r_multiple=row[11],
            mae=row[12],
            mfe=row[13],
            stop_loss=row[14],
            take_profit=row[15],
            regime=Regime(row[16]) if row[16] else Regime.LOW_VOL,
            volatility_percentile=row[17],
            liquidity_score=row[18],
            slippage=row[19],
            confidence=row[20],
        )
    
    def _calculate_max_drawdown(self, trades: List[Trade]) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0.0
        
        # Sort trades by entry time
        sorted_trades = sorted(trades, key=lambda t: t.entry_time)
        
        # Calculate cumulative P&L
        cumulative_pnl = []
        running_pnl = 0
        
        for trade in sorted_trades:
            running_pnl += trade.pnl
            cumulative_pnl.append(running_pnl)
        
        if not cumulative_pnl:
            return 0.0
        
        # Calculate drawdown
        peak = cumulative_pnl[0]
        max_drawdown = 0.0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            
            drawdown = (peak - pnl) / peak if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, trades: List[Trade]) -> float:
        """Calculate Sharpe ratio from trades."""
        if len(trades) < 2:
            return 0.0
        
        # Get daily returns
        daily_returns = {}
        for trade in trades:
            date = trade.entry_time.date()
            if date not in daily_returns:
                daily_returns[date] = 0
            daily_returns[date] += trade.pnl
        
        returns = list(daily_returns.values())
        if not returns:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        return avg_return / std_return if std_return != 0 else 0.0
    
    def get_recent_daily_performance(self, days: int = 3) -> List[Dict[str, Any]]:
        """Get recent daily performance records.

        Returns a list of dicts with 'date' and 'pnl' keys, ordered most recent first.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT date, pnl FROM daily_performance ORDER BY date DESC LIMIT ?",
                    (days,),
                )
                return [{"date": row[0], "pnl": row[1]} for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting recent daily performance: {e}")
            return []

    def export_to_csv(self, filepath: str, table: str = "trades"):
        """Export journal table to CSV."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                df.to_csv(filepath, index=False)
                logger.info(f"Exported {table} to {filepath}")
                
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
