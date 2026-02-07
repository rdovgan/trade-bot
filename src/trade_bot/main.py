"""Main trading bot application."""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from .core.models import MarketState, PositionState, AccountState, RiskState, TradingAction
from .core.enums import Action, Side, Regime
from .core.state_lock import state_manager
from .risk.validator import RiskValidator
from .data.connector import CCXTConnector, DataManager
from .signal.generator import create_default_signal_manager
from .decision.engine import DecisionEngine, LLMAdvisor
from .execution.engine import CCXTExchangeConnector, PositionMonitor, ExecutionEngine
from .learning.journal import TradeJournal
from .learning.loop import LearningLoop
from .monitoring.monitor import MonitoringEngine
from .deployment.manager import DeploymentManager
from .scanner.market_scanner import MarketScanner

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
        defaults = self._default_config()
        if config:
            for key, val in config.items():
                if isinstance(val, dict) and key in defaults and isinstance(defaults[key], dict):
                    defaults[key].update(val)
                else:
                    defaults[key] = val
        self.config = defaults
        self._running = False
        self._funding_task: Optional[asyncio.Task] = None

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
                'name': 'bybit',  # Default to Bybit
                'sandbox': True,    # Use testnet by default
                'apiKey': os.getenv('EXCHANGE_API_KEY'),
                'secret': os.getenv('EXCHANGE_SECRET'),
            },
            'trading': {
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'timeframe': '1m',
                'max_position_size': 0.1,
            },
            'data': {
                'max_data_age_multiplier': 100.0,  # Lenient for sandbox (100x normal threshold)
            },
            'risk': {
                'max_risk_per_trade': 0.01,  # 1%
                'max_daily_loss': 0.03,      # 3%
            },
            'llm': {
                'enabled': False,  # Disabled by default
            },
            'scanner': {
                'enabled': True,
                'scan_interval_minutes': 15,
                'quote_currency': 'USDT',
                'min_volume_24h': 1_000_000,
                'max_positions': 3,
                'portfolio_pct': 0.30,
                'blacklist': [],
            },
        }
    
    def _init_components(self):
        """Initialize all bot components."""
        try:
            # Initialize data connector
            exchange_config = self.config['exchange'].copy()
            exchange_name = exchange_config.pop('name')
            is_sandbox = exchange_config.pop('sandbox', False)

            # Build clean CCXT config (remove keys CCXT doesn't understand)
            ccxt_config = {
                k: v for k, v in exchange_config.items()
                if k in ('api_key', 'secret', 'apiKey', 'password', 'uid', 'options')
            }
            # Remap our key names to CCXT names
            if 'api_key' in ccxt_config:
                ccxt_config['apiKey'] = ccxt_config.pop('api_key')

            if is_sandbox:
                ccxt_config['sandbox'] = True

            self.data_connector = CCXTConnector(
                exchange_name,
                ccxt_config.copy()
            )

            # Get data age multiplier (higher for sandbox/testnet)
            data_config = self.config.get('data', {})
            max_data_age_multiplier = data_config.get('max_data_age_multiplier', 1.0)

            self.data_manager = DataManager(
                self.data_connector,
                max_data_age_multiplier=max_data_age_multiplier
            )

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

            # Initialize learning system (before execution engine needs it)
            self.journal = TradeJournal()
            self.learning_loop = LearningLoop(self.journal)

            # Initialize execution engine
            self.exchange_connector = CCXTExchangeConnector(
                exchange_name,
                ccxt_config.copy()
            )
            self.position_monitor = PositionMonitor(self.exchange_connector)
            self.execution_engine = ExecutionEngine(
                self.exchange_connector,
                self.position_monitor,
                journal=self.journal
            )

            # Initialize monitoring
            self.monitoring_engine = MonitoringEngine()

            # Initialize deployment manager
            self.deployment_manager = DeploymentManager()

            # Initialize market scanner
            scanner_cfg = self.config.get('scanner', {})
            self.scanner = MarketScanner(scanner_cfg) if scanner_cfg.get('enabled') else None
            self._scanned_symbols: list = []
            self._last_scan_time: Optional[datetime] = None

            # Initialize state
            self.account_state = AccountState(
                equity=10000.0,  # Starting equity
                balance=10000.0,
            )

            # Peak equity tracking for drawdown calculation
            # These start as None and get set on first successful balance fetch
            self._peak_equity: Optional[float] = None
            self._day_start_equity: Optional[float] = None
            self._current_day: Optional[datetime] = None
            self._system_locked: bool = self._load_lock_state()

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

            # Fetch real balance before anything else
            balance = await self.exchange_connector.get_balance()
            total_balance = sum(balance.values())
            logger.info(f"Exchange balance at startup: {balance} (total: {total_balance:.2f})")

            if total_balance <= 0:
                logger.critical("Account balance is zero — cannot start trading")
                return

            self.account_state.balance = total_balance
            self.account_state.equity = total_balance
            self._peak_equity = total_balance
            self._day_start_equity = total_balance

            # Start position monitoring
            symbols = self.config['trading']['symbols']
            await self.position_monitor.start_monitoring(symbols)

            # Start order monitoring (for stop-loss fills)
            await self.execution_engine.start_monitoring()

            # Set running flag
            self._running = True

            # Start funding cost accumulation task
            self._funding_task = asyncio.create_task(self._accumulate_funding_costs())

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

            # Stop funding cost task
            if self._funding_task:
                self._funding_task.cancel()
                try:
                    await self._funding_task
                except asyncio.CancelledError:
                    pass

            # Cancel all pending orders
            await self.execution_engine.cancel_all_orders()
            
            # Stop components
            await self.data_manager.stop()
            await self.position_monitor.stop_monitoring()
            await self.execution_engine.stop_monitoring()
            await self.execution_engine.close()
            
            logger.info("Trading bot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
    
    async def _run_market_scan(self):
        """Run market scanner and update the active symbol list."""
        if self.scanner is None:
            return

        scanner_cfg = self.config.get('scanner', {})
        interval = scanner_cfg.get('scan_interval_minutes', 15)

        now = datetime.now()
        if self._last_scan_time and (now - self._last_scan_time) < timedelta(minutes=interval):
            return

        try:
            exchange = self.data_connector.exchange

            # Pass open positions to scanner for penalty calculation
            open_positions = self.position_monitor.get_all_positions()
            candidates = await self.scanner.scan_market(exchange, open_positions=open_positions)

            max_pos = scanner_cfg.get('max_positions', 5)
            top = self.scanner.get_top_candidates(max_pos)

            new_symbols = [c.symbol for c in top]
            if new_symbols != self._scanned_symbols:
                added = set(new_symbols) - set(self._scanned_symbols)
                removed = set(self._scanned_symbols) - set(new_symbols)
                if added:
                    logger.info(f"Scanner added symbols: {added}")
                if removed:
                    logger.info(f"Scanner removed symbols: {removed}")
                self._scanned_symbols = new_symbols

            self._last_scan_time = now
            logger.info(f"Market scan complete — top {len(new_symbols)} candidates")

        except Exception as e:
            logger.error(f"Error during market scan: {e}")

    async def _trading_loop(self):
        """Main trading loop."""
        user_symbols = self.config['trading']['symbols']
        timeframe = self.config['trading']['timeframe']
        loop_count = 0

        logger.info(f"Starting trading loop for {len(user_symbols)} user symbols")

        while self._running:
            try:
                if self._system_locked:
                    logger.warning("System is locked — no trading until manual restart")
                    await asyncio.sleep(60)
                    continue

                # Run market scan if enabled
                await self._run_market_scan()

                # Merge user-specified symbols with scanner results
                all_symbols = list(dict.fromkeys(user_symbols + self._scanned_symbols))

                # Process each symbol
                for symbol in all_symbols:
                    if self._system_locked:
                        break
                    await self._process_symbol(symbol, timeframe)

                # Update daily performance
                await self._update_daily_performance()

                # Periodic position reconciliation (every 5 minutes)
                loop_count += 1
                if loop_count % 5 == 0:
                    discrepancies = await self.execution_engine.reconcile_positions()
                    if discrepancies:
                        logger.warning(f"Position discrepancies found: {discrepancies}")

                # Wait before next iteration
                await asyncio.sleep(60)  # Process every minute

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _process_symbol(self, symbol: str, timeframe: str):
        """Process a single trading symbol."""
        try:
            # Get market state with validation
            try:
                market_state = await self.data_manager.get_market_state(symbol, timeframe)
            except ValueError as e:
                logger.warning(f"Skipping {symbol}: {e}")
                return  # Skip this symbol if data is invalid
            
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
            
            # Make trading decision and execute with locking to prevent concurrent approvals
            async with state_manager.lock_state("risk_validation"):
                all_positions = self.position_monitor.get_all_positions()
                active_count = sum(
                    1 for p in all_positions.values() if p.current_side != Side.FLAT
                )
                action = await self.decision_engine.make_decision(
                    market_state, position_state, self.account_state, risk_state,
                    active_positions=active_count,
                )

                # CLOSE/REPLACE LOGIC: If max positions reached and new signal, close worst position
                if action and action.action.value != 'HOLD':
                    scanner_cfg = self.config.get('scanner', {})
                    max_positions = scanner_cfg.get('max_positions', 3)

                    if active_count >= max_positions:
                        # Find worst performing position (most negative unrealized PnL)
                        worst_symbol = None
                        worst_pnl_pct = 0.0

                        for pos_symbol, pos in all_positions.items():
                            if pos.current_side == Side.FLAT:
                                continue
                            if not pos.entry_price or pos.entry_price == 0:
                                continue

                            # Calculate unrealized PnL %
                            current_price = await self._get_current_price(pos_symbol)
                            if current_price:
                                if pos.current_side == Side.LONG:
                                    pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                                else:
                                    pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100

                                # Track worst (most negative)
                                if pnl_pct < worst_pnl_pct:
                                    worst_pnl_pct = pnl_pct
                                    worst_symbol = pos_symbol

                        # Close worst position if losing > 1%
                        if worst_symbol and worst_pnl_pct < -1.0:
                            logger.warning(
                                f"Max positions ({max_positions}) reached. Closing worst position "
                                f"{worst_symbol} (PnL: {worst_pnl_pct:.2f}%) to make room for {symbol}"
                            )
                            worst_market_state = await self.data_manager.get_market_state(worst_symbol, timeframe)
                            await self._close_position(worst_symbol, worst_market_state)
                        else:
                            logger.warning(
                                f"Max positions ({max_positions}) reached but no losing position to close. "
                                f"Skipping new signal for {symbol}"
                            )
                            return

                    # Execute action
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
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            ticker = await self.data_connector.exchange.fetch_ticker(symbol)
            return ticker.get('last')
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    async def _close_position(self, symbol: str, market_state: MarketState):
        """Close existing position."""
        try:
            position = self.position_monitor.get_position(symbol)
            if not position or position.current_side == Side.FLAT:
                return

            # Create closing action
            close_action = TradingAction(
                action=Action.SELL if position.current_side == Side.LONG else Action.BUY,
                size=abs(position.position_size),
                expected_return=0,
                expected_risk=0,
                confidence=1.0
            )
            
            # Execute close
            order = await self.execution_engine.execute_action(
                close_action, symbol, market_state.current_price
            )

            # Complete the trade
            if order:
                await self.execution_engine.complete_trade_on_manual_close(symbol, order)

            logger.info(f"Closed position for {symbol}")
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    async def _update_account_state(self):
        """Update account state from exchange."""
        async with state_manager.lock_state("account"):
            try:
                # Get balance from exchange
                balance = await self.exchange_connector.get_balance()

                # Calculate unrealized PnL across all positions
                all_positions = self.position_monitor.get_all_positions()
                total_unrealized_pnl = sum(
                    pos.unrealized_pnl for pos in all_positions.values()
                    if pos.current_side != Side.FLAT
                )
                self.account_state.unrealized_pnl = total_unrealized_pnl

                # Calculate total exposure across all positions
                total_exposure = 0.0
                for pos in all_positions.values():
                    if pos.current_side != Side.FLAT and pos.entry_price:
                        total_exposure += abs(pos.position_size * pos.entry_price)

                # Update balance and equity
                total_balance = sum(balance.values())
                self.account_state.balance = total_balance
                equity = total_balance + total_unrealized_pnl
                self.account_state.equity = equity

                # Skip drawdown calculations if equity is zero or negative
                # (exchange may return 0 balance during API issues)
                if equity <= 0:
                    logger.warning(f"Equity is {equity:.2f} — skipping drawdown calculation")
                    self.account_state.exposure_pct = 0.0
                    self.account_state.leverage = 0.0
                    return

                # Update exposure and leverage
                self.account_state.exposure_pct = total_exposure / equity
                self.account_state.leverage = total_exposure / equity

                # Initialize peak/day-start equity on first valid balance
                if self._peak_equity is None:
                    self._peak_equity = equity
                    logger.info(f"Initial peak equity set to {equity:.2f}")
                if self._day_start_equity is None:
                    self._day_start_equity = equity
                    logger.info(f"Initial day-start equity set to {equity:.2f}")

                # Reset day-start equity at day boundary
                today = datetime.now().date()
                if self._current_day is None or self._current_day != today:
                    self._current_day = today
                    self._day_start_equity = equity
                    logger.info(f"New trading day — start equity: {self._day_start_equity:.2f}")

                # Update peak equity (high-water mark)
                if equity > self._peak_equity:
                    self._peak_equity = equity

                # Calculate current drawdown from peak
                self.account_state.current_drawdown = (
                    (self._peak_equity - equity) / self._peak_equity
                )

                # Track max drawdown
                if self.account_state.current_drawdown > self.account_state.max_drawdown:
                    self.account_state.max_drawdown = self.account_state.current_drawdown

                # Calculate daily loss (including unrealized PnL)
                daily_change = equity - self._day_start_equity
                self.account_state.daily_pnl = daily_change
                self.account_state.daily_loss_pct = max(0.0, -daily_change / self._day_start_equity)

                # Emergency: if drawdown exceeds lock threshold, close all and lock
                lock_threshold = self.risk_validator.config.get("drawdown_lock_threshold", 0.10)
                if not self._system_locked and self.account_state.current_drawdown >= lock_threshold:
                    logger.critical(
                        f"EMERGENCY: Drawdown {self.account_state.current_drawdown:.2%} "
                        f"exceeds lock threshold {lock_threshold:.2%} — closing all positions and locking system"
                    )
                    await self._emergency_close_all()
                    self._system_locked = True

            except Exception as e:
                logger.error(f"Error updating account state: {e}")

    async def _emergency_close_all(self):
        """Emergency close all open positions with proper order cleanup."""
        try:
            # 1. Cancel ALL pending orders first
            logger.critical("Cancelling all pending orders...")
            await self.execution_engine.cancel_all_orders()

            # 2. Close all positions
            all_positions = self.position_monitor.get_all_positions()
            for symbol, position in all_positions.items():
                if position.current_side != Side.FLAT:
                    logger.critical(f"Emergency closing position: {symbol}")
                    # Build a dummy market state just for the close
                    close_action = TradingAction(
                        action=Action.SELL if position.current_side == Side.LONG else Action.BUY,
                        size=abs(position.position_size),
                        expected_return=0,
                        expected_risk=0,
                        confidence=1.0,
                    )
                    try:
                        await self.execution_engine.execute_action(
                            close_action, symbol, position.entry_price or 0
                        )
                    except Exception as e:
                        logger.error(f"Failed to emergency close {symbol}: {e}")

            # 3. Set lock flag and persist
            self._system_locked = True
            self._persist_lock_state(True)

        except Exception as e:
            logger.error(f"Error in emergency close all: {e}")

    def _persist_lock_state(self, locked: bool):
        """Persist system lock to file."""
        import json
        lock_file = Path("trade_bot_lock.json")
        lock_file.write_text(json.dumps({
            "locked": locked,
            "timestamp": datetime.now().isoformat(),
            "reason": "emergency_drawdown"
        }))
        logger.info(f"Persisted lock state: locked={locked}")

    def _load_lock_state(self) -> bool:
        """Load lock state on startup."""
        import json
        lock_file = Path("trade_bot_lock.json")
        if lock_file.exists():
            data = json.loads(lock_file.read_text())
            if data.get("locked"):
                logger.warning(f"System was locked at {data.get('timestamp')}")
                return True
        return False

    async def _accumulate_funding_costs(self):
        """Background task to track funding rate costs every 8 hours."""
        while self._running:
            try:
                all_positions = self.position_monitor.get_all_positions()

                for symbol, position in all_positions.items():
                    if position.current_side != Side.FLAT:
                        # Get funding rate
                        funding_rate = await self.data_connector.get_funding_rate(symbol)

                        # Calculate funding cost
                        notional = abs(position.position_size * position.entry_price) if position.entry_price else 0
                        funding_cost = notional * funding_rate

                        # Find open trade and add to funding_cost
                        open_trade = next((t for t in self.execution_engine._trades
                                         if t.symbol == symbol and t.status == "open"), None)
                        if open_trade:
                            open_trade.funding_cost += funding_cost
                            logger.info(f"Accumulated funding cost for {symbol}: ${funding_cost:.4f} (rate: {funding_rate:.6f})")

                # Wait 8 hours (28800 seconds)
                await asyncio.sleep(28800)
            except Exception as e:
                logger.error(f"Error accumulating funding costs: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour on error

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


def _config_from_env() -> Dict:
    """Build config dict from environment variables."""
    config: Dict = {}

    # Exchange
    exchange_name = os.getenv('EXCHANGE_NAME', '').strip()
    sandbox_raw = os.getenv('EXCHANGE_SANDBOX', '').strip().lower()
    if exchange_name or sandbox_raw:
        config['exchange'] = {}
        if exchange_name:
            config['exchange']['name'] = exchange_name
        if sandbox_raw in ('0', 'false', 'no'):
            config['exchange']['sandbox'] = False
        elif sandbox_raw in ('1', 'true', 'yes'):
            config['exchange']['sandbox'] = True

    # Trading
    symbols_raw = os.getenv('TRADING_SYMBOLS', '').strip()
    timeframe = os.getenv('TRADING_TIMEFRAME', '').strip()
    if symbols_raw or timeframe:
        config['trading'] = {}
        if symbols_raw:
            config['trading']['symbols'] = [s.strip() for s in symbols_raw.split(',') if s.strip()]
        if timeframe:
            config['trading']['timeframe'] = timeframe

    # Risk
    risk_per_trade = os.getenv('MAX_RISK_PER_TRADE', '').strip()
    daily_loss = os.getenv('MAX_DAILY_LOSS', '').strip()
    if risk_per_trade or daily_loss:
        config['risk'] = {}
        if risk_per_trade:
            config['risk']['max_risk_per_trade'] = float(risk_per_trade)
        if daily_loss:
            config['risk']['max_daily_loss'] = float(daily_loss)

    # Scanner
    scanner_enabled = os.getenv('SCANNER_ENABLED', '').strip().lower()
    scanner_max_pos = os.getenv('SCANNER_MAX_POSITIONS', '').strip()
    scanner_portfolio = os.getenv('SCANNER_PORTFOLIO_PCT', '').strip()
    scanner_min_vol = os.getenv('SCANNER_MIN_VOLUME_24H', '').strip()
    scanner_blacklist = os.getenv('SCANNER_BLACKLIST', '').strip()
    scanner_interval = os.getenv('SCANNER_INTERVAL_MINUTES', '').strip()
    if any([scanner_enabled, scanner_max_pos, scanner_portfolio]):
        config['scanner'] = {}
        if scanner_enabled in ('0', 'false', 'no'):
            config['scanner']['enabled'] = False
        elif scanner_enabled in ('1', 'true', 'yes'):
            config['scanner']['enabled'] = True
        if scanner_max_pos:
            config['scanner']['max_positions'] = int(scanner_max_pos)
        if scanner_portfolio:
            config['scanner']['portfolio_pct'] = float(scanner_portfolio)
        if scanner_min_vol:
            config['scanner']['min_volume_24h'] = float(scanner_min_vol)
        if scanner_blacklist:
            config['scanner']['blacklist'] = [s.strip() for s in scanner_blacklist.split(',') if s.strip()]
        if scanner_interval:
            config['scanner']['scan_interval_minutes'] = int(scanner_interval)

    # LLM
    llm_enabled = os.getenv('LLM_ENABLED', '').strip().lower()
    if llm_enabled in ('1', 'true', 'yes'):
        config['llm'] = {'enabled': True, 'api_key': os.getenv('LLM_API_KEY', '')}

    return config


async def main():
    """Main entry point."""
    config = _config_from_env()

    bot = TradingBot(config if config else None)

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
