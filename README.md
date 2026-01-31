# AI Trading Agent

Crypto trading bot with risk-first architecture. The risk engine has veto power over all decisions — LLM and signals are advisory only.

```
SCAN → SIGNAL → RESEARCH (LLM) → DECIDE → RISK VALIDATOR → EXECUTE → MONITOR → CLOSE → JOURNAL → LEARN
```

## Quickstart — Live Trading

### 1. Install

```bash
pip install -e .
```

### 2. Create API keys

| Exchange | Where to create keys |
|---|---|
| **Bybit** | https://www.bybit.com/app/user/api-management |
| Binance | https://www.binance.com/en/my/settings/api-management |
| OKX | https://www.okx.com/account/my-api |

Required permissions: **Read + Trade (Futures)**. Do NOT enable Withdraw.

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```bash
EXCHANGE_NAME=bybit
EXCHANGE_API_KEY=xxxxxxxxxxxxxxxx
EXCHANGE_SECRET=xxxxxxxxxxxxxxxx
EXCHANGE_SANDBOX=false              # false = REAL MONEY
TRADING_SYMBOLS=BTC/USDT:USDT      # linear futures use :USDT suffix
```

### 4. Test with demo first (recommended)

```bash
# Bybit demo keys: https://testnet.bybit.com/app/user/api-management
EXCHANGE_SANDBOX=true
```

```bash
python -m trade_bot.main
```

Verify in logs that the bot connects, scans, generates signals, and places orders on testnet. Then switch `EXCHANGE_SANDBOX=false` when ready.

### 5. Go live

```bash
EXCHANGE_SANDBOX=false python -m trade_bot.main
```

Or set `EXCHANGE_SANDBOX=false` in `.env` and run:

```bash
python -m trade_bot.main
```

The bot will:
1. Scan all USDT perpetual pairs, rank by volume/momentum/spread/volatility
2. Pick the top N coins (default 5) automatically
3. Generate signals (mean reversion, trend, breakout) every 60s
4. Validate every trade through the risk engine before execution
5. Place market orders with stop-losses on Bybit linear futures

### `.env` Reference

| Variable | Default | Description |
|---|---|---|
| `EXCHANGE_NAME` | `bybit` | Exchange id (bybit, binance, okx) |
| `EXCHANGE_API_KEY` | — | API key |
| `EXCHANGE_SECRET` | — | API secret |
| `EXCHANGE_SANDBOX` | `true` | `true` = demo/testnet, `false` = live |
| `TRADING_SYMBOLS` | `BTC/USDT,ETH/USDT` | Manual symbol list (merged with scanner) |
| `TRADING_TIMEFRAME` | `1m` | Candle timeframe |
| `MAX_RISK_PER_TRADE` | `0.01` | 1% equity risk per trade |
| `MAX_DAILY_LOSS` | `0.05` | 5% daily loss halt |
| `SCANNER_ENABLED` | `true` | Auto coin selection |
| `SCANNER_MAX_POSITIONS` | `5` | Max simultaneous positions |
| `SCANNER_PORTFOLIO_PCT` | `0.50` | % of equity for auto-trading |
| `SCANNER_MIN_VOLUME_24H` | `1000000` | Min $1M daily volume filter |
| `SCANNER_BLACKLIST` | — | Comma-separated symbols to skip |
| `LLM_ENABLED` | `false` | Optional LLM market analysis |
| `LLM_API_KEY` | — | OpenAI-compatible API key |

## Run Tests

```bash
pip install -e ".[dev]"
pytest
```

## How It Works

### Signal Generation

Three built-in strategies, each activated in its matching market regime:

| Strategy | Regime | Method |
|---|---|---|
| Mean Reversion | `mean_revert` | Bollinger Band penetration |
| Trend Following | `trending` | MA crossover (10/30) |
| Volatility Breakout | `high_vol` | Support/resistance breakout |

### Risk Rules (hard-coded, cannot be bypassed)

- Max 1% equity risk per trade (2% absolute ceiling)
- Min 1.5:1 reward-to-risk ratio
- Mandatory stop-loss on every trade
- Max 30% exposure per asset, 50% total, 2x leverage
- Daily loss > 5% → trading halted until next day
- 5 consecutive losses → 24h cooldown
- 3 red days in a row → safe mode (high-confidence trades only)
- Drawdown > 10% → position sizes cut 50%
- Drawdown > 15% → trading paused
- Drawdown > 20% → system locked
- Volatility > 95th percentile → no new positions
- No averaging down, no martingale

### Deployment Stages

Capital is allocated gradually when going live:

```
5% → 10% → 25% → 100%
```

Automatic rollback if Sharpe degrades, drawdown exceeds historical max, or slippage increases.

### Monitoring

Every cycle checks:
- Exchange health (response times, consecutive failures)
- Slippage vs expected (alerts if excessive)
- Regime shift since position entry

### Learning Loop

```
TRADE → JOURNAL → EXTRACT → SCORE → APPLY
```

Strategies are scored by a **salience model** (not just win/loss):
- 40% avg R-multiple, 30% win rate, 20% trade count
- Penalized for drawdown and R variance
- Score < 0.2 → archived, 0.2–0.8 → active, > 0.8 → promoted

Promotion to live requires: Sharpe > 1.2, DD < 10%, 200+ trades, stable returns, profitable in 2+ regimes.

### LLM Advisory (optional)

When enabled, calls an OpenAI-compatible API for market analysis. The LLM **can** suggest regime shifts and risk factors. It **cannot** set position sizes, remove stop-losses, or override risk limits.

## Project Structure

```
src/trade_bot/
  core/           # Enums, Pydantic models
  data/           # CCXT market data connector
  signal/         # Signal generators (mean reversion, trend, breakout)
  decision/       # Decision engine + LLM advisor
  risk/           # Risk validator (hard gate)
  execution/      # Order execution + position monitor
  monitoring/     # Exchange health, slippage, regime shift detection
  learning/       # Journal, salience model, reward function, walk-forward backtest
  deployment/     # Staged capital rollout manager
  scanner/         # Market scanner (auto coin selection)
  main.py         # Bot orchestrator
tests/unit/       # 157 unit tests
```

## License

MIT
