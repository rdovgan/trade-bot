# AI Trading Agent — Training & Risk Policy (Production Version)

---

# 1. System Principles

1. **Risk-first architecture**
2. Deterministic risk layer (cannot be bypassed)
3. LLM is advisory, not authoritative
4. All decisions must be structured and testable
5. No deployment without statistical validation

---

# 2. System Architecture

```
SCAN
  ↓
SIGNAL
  ↓
RESEARCH (LLM optional)
  ↓
DECIDE (Agent)
  ↓
RISK VALIDATOR (Hard Gate)
  ↓
EXECUTE
  ↓
MONITOR
  ↓
CLOSE
  ↓
JOURNAL
  ↓
LEARNING LOOP
```

Risk Validator has veto power.

---

# 3. State Definition (Agent Input)

## 3.1 Market State

* OHLCV (last N bars)
* ATR
* Realized volatility
* Spread
* Order book imbalance
* Volume delta
* Liquidity score
* Regime label (trend / mean-revert / high-vol / low-vol)

## 3.2 Position State

* Current side
* Position size
* Entry price
* Time in position
* MAE
* MFE

## 3.3 Account State

* Equity
* Balance
* Exposure %
* Unrealized PnL
* Current drawdown
* Daily loss %

## 3.4 Risk State

* Risk budget left
* Max daily loss remaining
* Consecutive losses
* Volatility percentile

---

# 4. Action Schema

```
{
  action: BUY | SELL | HOLD,
  size: float,
  stop_loss: price,
  take_profit: price,
  expected_return: float,
  expected_risk: float,
  confidence: 0–1
}
```

Hard rule:

* R:R must be ≥ 1.5
* Stop-loss mandatory

---

# 5. Risk Management Rules

## 5.1 Risk per Trade

```
max_risk_per_trade = 1% equity (max 2%)
```

Position size:

```
position_size = (equity * risk%) / stop_distance
```

No fixed % capital allocation.

---

## 5.2 Exposure Limits

* Max 30% equity per asset
* Max 50% total exposure
* Max leverage: 2x
* No martingale
* No averaging down

---

## 5.3 Daily Risk

* Max daily loss: 3–5%
* If hit → trading disabled until next day

---

## 5.4 Drawdown Controls

* DD > 10% → position size reduced 50%
* DD > 15% → trading paused
* DD > 20% → system lock

---

## 5.5 Consecutive Loss Control

* 5 losses in a row → 24h cooldown
* 3 red days → safe mode

---

## 5.6 Volatility Guard

If volatility percentile > 95%:

* No new positions
* Existing positions reduced

---

## 5.7 Liquidity Guard

If spread or slippage exceeds threshold:

* Block execution

---

# 6. Monitoring Rules

Every monitoring cycle:

* Check trailing stop
* Check volatility spike
* Check exchange health
* Check slippage vs expected
* Check regime shift

If regime shifts → optional early exit.

---

# 7. Reward Function (RL / Learning)

Do NOT use raw PnL.

```
reward =
  Δequity
  - λ1 * drawdown_increase
  - λ2 * volatility_exposure
  - λ3 * position_size_penalty
  - λ4 * transaction_cost
```

Additional penalties:

* Daily loss violation → large negative reward
* Risk violation → episode termination

Goal:
Maximize risk-adjusted return.

---

# 8. Learning Framework

## 8.1 Training

* Walk-forward validation
* No look-ahead bias
* Realistic slippage model
* Commission included
* Latency simulated

## 8.2 Promotion Criteria (Paper → Live)

* Sharpe > 1.2
* Max DD < 10%
* ≥ 200 trades
* Stable monthly returns
* Profit across multiple regimes

---

# 9. Self-Learning Loop

```
TRADE → JOURNAL → EXTRACT → SCORE → APPLY
```

---

# 10. Journal Requirements

For every trade:

* Entry / exit price
* Stop / TP
* R multiple
* MAE
* MFE
* Slippage
* Regime
* Volatility percentile
* Liquidity condition
* Decision confidence

---

# 11. Salience Model

Salience is not win/loss-based.

## 11.1 Inputs

* win_rate
* avg_R
* trade_count
* drawdown
* regime_consistency
* stability (variance of R)

## 11.2 Formula

```
salience =
  sigmoid(
    0.4 * normalized_avg_R +
    0.3 * win_rate +
    0.2 * log(trade_count) -
    0.3 * drawdown -
    0.2 * R_variance
  )
```

## 11.3 Thresholds

* < 0.2 → ARCHIVE
* 0.2–0.8 → ACTIVE
* > 0.8 → PROMOTE

Decay depends on trade frequency, not time only.

---

# 12. Regime Awareness

Strategies must be tagged by regime:

* Trending
* Mean-reverting
* High volatility
* Low volatility

Salience is evaluated per regime.

No cross-regime promotion.

---

# 13. Safe Mode

Activated when:

* DD > 10%
* 3 red days
* Volatility spike
* API instability

Safe Mode:

* 50% position size
* Only high-confidence trades
* Or full halt

---

# 14. LLM Usage Policy

LLM may:

* Generate hypotheses
* Describe macro narrative
* Detect news-based risk
* Suggest regime shift

LLM may NOT:

* Set position size
* Override risk limits
* Remove stop-loss
* Change leverage rules

All LLM outputs must be structured.

---

# 15. Production Deployment Policy

Rollout stages:

1. 5% capital
2. 10%
3. 25%
4. 100%

Rollback if:

* Sharpe degrades
* DD exceeds historical max
* Slippage increases materially

---

# 16. Hard Prohibitions

* No martingale
* No unlimited averaging
* No trading without SL
* No bypassing risk validator
* No immediate size increase after loss
* No trade during exchange instability

---

# 17. System Hierarchy

```
Risk Engine
  ↓
Statistical Edge
  ↓
Agent Intelligence
  ↓
LLM Advisory
```

Risk always dominates.

---

If needed, next step can be:

* Formal RL environment specification
* Mathematical risk budget model
* Execution-layer safety specification
* Capital allocation model for multi-asset trading
