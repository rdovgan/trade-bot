"""Main trading bot application."""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional
import os
from pathlib import Path

from .core.models import MarketState, PositionState, AccountState, RiskState
from .core.enums import Side, Regime
from .risk.validator import RiskValidator
from .data.connector import CCXTConnector, DataManager
from .signal.generator import create_default_signal_manager
from .decision.engine import DecisionEngine, LLMAdvisor
from .execution.engine import CCXTExchangeConnector, PositionMonitor, ExecutionEngine
from .learning.journal import TradeJournal
from .learning.loop import LearningLoop
from .monitoring.monitor import MonitoringEngine
from .deployment.manager import DeploymentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot class."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize trading bot."""
        self.config = config or self._default_config()
        self._running = False
        
        # Initialize components
        self._init_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Trading bot initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'exchange': {
                'name': 'binance',  # Default to Binance
                'sandbox': True,    # Use testnet by default
                'api_key': os.getenv('EXCHANGE_API_KEY'),
                'secret': os.getenv('EXCHANGE_SECRET'),
            },
            'trading': {
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'timeframe': '1m',
                'max_position_size': 0.1,
            },
            'risk': {
                'max_risk_per_trade': 0.01,  # 1%
                'max_daily_loss': 0.05,      # 5%
            },
            'llm': {
                'enabled': False,  # Disabled by default
            },
        }
    
    def _init_components(self):
        """Initialize all bot components."""
        try:
            # Initialize data connector
            exchange_config = self.config['exchange'].copy()
            if exchange_config['sandbox']:
                exchange_config['sandboxMode'] = True
            
            self.data_connector = CCXTConnector(
                exchange_config['name'],
                exchange_config
            )
            self.data_manager = DataManager(self.data_connector)
            
            # Initialize risk validator
            self.risk_validator = RiskValidator(self.config['risk'])
            
            # Initialize signal manager
            self.signal_manager = create_default_signal_manager()
            
            # Initialize decision engine
            self.llm_advisor = LLMAdvisor(self.config['llm'])
            self.decision_engine = DecisionEngine(
                self.risk_validator,
                self.signal_manager,
                self.llm_advisor
            )
            
            # Initialize execution engine
            self.exchange_connector = CCXTExchangeConnector(
                exchange_config['name'],
                exchange_config
            )
            self.position_monitor = PositionMonitor(self.exchange_connector)
            self.execution_engine = ExecutionEngine(
                self.exchange_connector,
                self.position_monitor
            )
            
            # Initialize learning system
            self.journal = TradeJournal()
            self.learning_loop = LearningLoop(self.journal)

            # Initialize monitoring
            self.monitoring_engine = MonitoringEngine()

            # Initialize deployment manager
            self.deployment_manager = DeploymentManager()
            
            # Initialize state
            self.account_state = AccountState(
                equity=10000.0,  # Starting equity
                balance=10000.0,
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    async def start(self):
        """Start the trading bot."""
        try:
            logger.info("Starting trading bot...")
            
            # Start data collection
            await self.data_manager.start()
            
            # Start position monitoring
            symbols = self.config['trading']['symbols']
            await self.position_monitor.start_monitoring(symbols)
            
            # Set running flag
            self._running = True
            
            # Start main trading loop
            await self._trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            raise
    
    async def stop(self):
        """Stop the trading bot."""
        try:
            logger.info("Stopping trading bot...")
            
            self._running = False
            
            # Cancel all pending orders
            await self.execution_engine.cancel_all_orders()
            
            # Stop components
            await self.data_manager.stop()
            await self.position_monitor.stop_monitoring()
            await self.execution_engine.close()
            
            logger.info("Trading bot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
    
    async def _trading_loop(self):
        """Main trading loop."""
        symbols = self.config['trading']['symbols']
        timeframe = self.config['trading']['timeframe']
        
        logger.info(f"Starting trading loop for {len(symbols)} symbols")
        
        while self._running:
            try:
                # Process each symbol
                for symbol in symbols:
                    await self._process_symbol(symbol, timeframe)
                
                # Update daily performance
                await self._update_daily_performance()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _process_symbol(self, symbol: str, timeframe: str):
        """Process a single trading symbol."""
        try:
            # Get market state
            market_state = await self.data_manager.get_market_state(symbol, timeframe)
            
            # Get position state
            position_state = self.position_monitor.get_position(symbol) or PositionState()
            
            # Update account state (simplified)
            await self._update_account_state()
            
            # Update risk state
            recent_trades = self.journal.get_trades(symbol=symbol, limit=10)
            recent_daily_pnls = self.journal.get_recent_daily_performance(days=3)
            risk_state = self.risk_validator.update_risk_state(
                self.account_state, market_state, recent_trades,
                recent_daily_pnls=recent_daily_pnls,
            )
            
            # Record market context
            self.journal.record_market_context(market_state, symbol)

            # Run monitoring checks
            mon_result = await self.monitoring_engine.run_monitoring_cycle(
                self.exchange_connector,
                {symbol: market_state},
                {symbol: position_state},
            )
            if not mon_result["exchange_healthy"]:
                logger.warning(f"Exchange unhealthy — skipping {symbol}")
                return

            # Check if we should close existing position
            should_close = await self.decision_engine.should_close_position(
                market_state, position_state, self.account_state, risk_state
            )
            
            if should_close:
                await self._close_position(symbol, market_state)
                return
            
            # Make trading decision
            action = await self.decision_engine.make_decision(
                market_state, position_state, self.account_state, risk_state
            )
            
            # Execute action if not HOLD
            if action and action.action.value != 'HOLD':
                await self._execute_action(action, symbol, market_state)
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    
    async def _execute_action(self, action, symbol: str, market_state: MarketState):
        """Execute a trading action."""
        try:
            # Execute the action
            order = await self.execution_engine.execute_action(
                action, symbol, market_state.current_price
            )
            
            if order:
                logger.info(
                    f"Executed {action.action.value} order for {symbol}: "
                    f"size={action.size:.6f}, price={market_state.current_price}"
                )
            
        except Exception as e:
            logger.error(f"Error executing action for {symbol}: {e}")
    
    async def _close_position(self, symbol: str, market_state: MarketState):
        """Close existing position."""
        try:
            position = self.position_monitor.get_position(symbol)
            if not position or position.current_side == Side.FLAT:
                return
            
            # Create closing action
            from .core.models import TradingAction
            from .core.enums import Action
            
            close_action = TradingAction(
                action=Action.SELL if position.current_side == Side.LONG else Action.BUY,
                size=abs(position.position_size),
                expected_return=0,
                expected_risk=0,
                confidence=1.0
            )
            
            # Execute close
            await self.execution_engine.execute_action(
                close_action, symbol, market_state.current_price
            )
            
            logger.info(f"Closed position for {symbol}")
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    async def _update_account_state(self):
        """Update account state from exchange."""
        try:
            # Get balance from exchange
            balance = await self.exchange_connector.get_balance()
            
            # Update account state (simplified)
            total_balance = sum(balance.values())
            self.account_state.balance = total_balance
            self.account_state.equity = total_balance + self.account_state.unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error updating account state: {e}")
    
    async def _update_daily_performance(self):
        """Update daily performance metrics."""
        try:
            # Get today's trades
            today = datetime.now().date()
            start_of_day = datetime.combine(today, datetime.min.time())
            
            trades = self.journal.get_trades(
                start_date=start_of_day,
                end_date=datetime.now()
            )
            
            # Calculate daily P&L
            daily_pnl = sum(trade.pnl for trade in trades)
            
            # Update daily performance in journal
            self.journal.update_daily_performance(
                datetime.now(),
                self.account_state.equity,
                self.account_state.balance,
                daily_pnl,
                trades
            )
            
        except Exception as e:
            logger.error(f"Error updating daily performance: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())
    
    def get_status(self) -> Dict:
        """Get bot status."""
        return {
            'running': self._running,
            'account_equity': self.account_state.equity,
            'account_balance': self.account_state.balance,
            'current_positions': len(self.position_monitor.get_all_positions()),
            'pending_orders': len(self.execution_engine.get_pending_orders()),
            'total_trades': len(self.journal.get_trades()),
        }


async def main():
    """Main entry point."""
    # Load configuration from environment or file
    config = None  # Could load from config file
    
    # Create and start bot
    bot = TradingBot(config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
