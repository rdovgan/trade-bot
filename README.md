# AI Trading Agent

Crypto trading bot with risk-first architecture. The risk engine has veto power over all decisions — LLM and signals are advisory only.

```
SCAN → SIGNAL → RESEARCH (LLM) → DECIDE → RISK VALIDATOR → EXECUTE → MONITOR → CLOSE → JOURNAL → LEARN
```

## Setup

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env with your exchange API keys
```

### Required `.env` variables

| Variable | Description |
|---|---|
| `EXCHANGE_API_KEY` | Exchange API key (Binance, etc.) |
| `EXCHANGE_SECRET` | Exchange API secret |

### Optional `.env` variables

| Variable | Default | Description |
|---|---|---|
| `TRADING_SYMBOLS` | `BTC/USDT,ETH/USDT` | Comma-separated trading pairs |
| `TRADING_TIMEFRAME` | `1m` | Candle timeframe |
| `MAX_RISK_PER_TRADE` | `0.01` | Max risk per trade (1% of equity) |
| `MAX_DAILY_LOSS` | `0.05` | Max daily loss before halt (5%) |
| `LLM_ENABLED` | `false` | Enable LLM market analysis |
| `LLM_API_KEY` | — | OpenAI-compatible API key |

## Run

```bash
# Start trading (sandbox mode by default)
python -m trade_bot.main
```

The bot starts in **sandbox/testnet mode** by default. To trade live, set `sandbox: False` in the config.

## Run Tests

```bash
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
  main.py         # Bot orchestrator
tests/unit/       # 147 unit tests
```

## License

MIT
