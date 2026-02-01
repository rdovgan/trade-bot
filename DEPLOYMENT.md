# Deployment Guide

This guide covers deployment of the AI Trading Agent in production environments.

## Architecture Overview

The trading bot follows a risk-first architecture with these key components:

```
SCAN → SIGNAL → RESEARCH → DECIDE → RISK VALIDATOR → EXECUTE → MONITOR → CLOSE → JOURNAL → LEARNING
```

## Risk Management Rules

### Hard-coded Safety Limits
- **Maximum risk per trade**: 1% of equity (absolute maximum 2%)
- **Maximum exposure per asset**: 30% of equity
- **Maximum total exposure**: 50% of equity
- **Maximum leverage**: 2x
- **Daily loss limit**: 5% (trading halt when reached)
- **Drawdown controls**:
  - >10% DD: 50% position size reduction
  - >15% DD: Trading pause
  - >20% DD: System lock

### Mandatory Requirements
- Stop loss is mandatory for all positions
- Minimum R:R ratio: 1.5
- No martingale strategies
- No averaging down

## Pre-Deployment Checklist

### 1. Environment Setup
- [ ] Python 3.9+ installed
- [ ] All dependencies installed (`pip install -e ".[dev]"`)
- [ ] Environment variables configured (see `.env.example`)
- [ ] Exchange API credentials configured and tested
- [ ] LLM Advisory (`LLM_ENABLED`) checked: ensure it's disabled unless fully tested/justified for live use.
- [ ] Database directory created with proper permissions

### 2. Risk Configuration Review
- [ ] Risk limits appropriate for account size
- [ ] Position sizing parameters verified
- [ ] Daily loss limits set appropriately
- [ ] Drawdown thresholds reviewed

### 3. Exchange Setup
- [ ] API keys have appropriate permissions (read-only for testing)
- [ ] Testnet/sandbox environment configured
- [ ] Rate limits understood and configured
- [ ] Withdrawal restrictions in place (recommended)

### 4. Testing
- [ ] Unit tests pass (`pytest`)
- [ ] Integration tests pass
- [ ] Paper trading validation completed
- [ ] Risk validation tests pass
- [ ] Performance benchmarks met

## Deployment Stages

### Stage 1: Paper Trading (5% Capital Allocation)
1. **Configuration**
   ```bash
   export TRADING_MODE=paper
   export CAPITAL_ALLOCATION=0.05
   export EXCHANGE_SANDBOX=true
   ```

2. **Monitoring Requirements**
   - Daily performance review
   - Risk rule validation
   - Trade execution verification
   - System health checks

3. **Success Criteria**
   - Sharpe ratio > 1.0
   - Maximum drawdown < 8%
   - No risk rule violations
   - Stable performance over 100+ trades

### Stage 2: Small Live Trading (10% Capital Allocation)
1. **Configuration**
   ```bash
   export TRADING_MODE=live
   export CAPITAL_ALLOCATION=0.10
   export EXCHANGE_SANDBOX=false
   ```

2. **Additional Monitoring**
   - Real-time P&L tracking
   - Slippage analysis
   - Latency monitoring
   - Exchange connectivity alerts

3. **Success Criteria**
   - Performance matches paper trading results
   - Slippage within acceptable limits (<5 bps)
   - No system errors or crashes
   - Risk rules functioning correctly

### Stage 3: Medium Live Trading (25% Capital Allocation)
1. **Enhanced Monitoring**
   - Automated alerts for all risk events
   - Real-time dashboard
   - Backup systems tested
   - Disaster recovery procedures verified

2. **Success Criteria**
   - Consistent performance across market conditions
   - Risk management functioning under stress
   - System stability maintained

### Stage 4: Full Production (100% Capital Allocation)
1. **Final Requirements**
   - All monitoring systems operational
   - Backup systems in place
   - 24/7 monitoring capability
   - Emergency procedures documented

## Monitoring and Alerting

### Key Metrics to Monitor
1. **Performance Metrics**
   - Sharpe ratio
   - Maximum drawdown
   - Win rate
   - Average R multiple
   - Profit factor

2. **Risk Metrics**
   - Current exposure
   - Daily loss percentage
   - Consecutive losses
   - Volatility percentile
   - Safe mode status

3. **System Metrics**
   - API response times
   - Error rates
   - Order execution success
   - Database performance

### Alert Thresholds
- **Critical**: System lock, daily loss limit reached, exchange connectivity lost
- **Warning**: Drawdown > 10%, consecutive losses > 3, high volatility
- **Info**: New trades executed, daily performance summary

## Rollback Procedures

### Automatic Rollback Triggers
- Sharpe ratio degradation > 20%
- Drawdown exceeds historical maximum
- Slippage increases materially
- Risk rule violations

### Manual Rollback Steps
1. Cancel all pending orders
2. Close all positions (if safe)
3. Reduce capital allocation to previous stage
4. Investigate root cause
5. Implement fixes
6. Resume with reduced allocation

## Security Considerations

### API Security
- Use API keys with minimal permissions
- Regularly rotate API keys
- IP whitelisting where supported
- Monitor API usage for anomalies

### System Security
- Regular security updates
- Firewall configuration
- Access logging and monitoring
- Secure credential storage

## Maintenance

### Daily Tasks
- Review performance metrics
- Check system logs for errors
- Verify risk rule compliance
- Monitor market conditions

### Weekly Tasks
- Update market data models
- Review strategy performance
- Backup databases
- Update risk parameters if needed

### Monthly Tasks
- Comprehensive performance review
- Strategy evaluation and retraining
- System health check
- Security audit

## Troubleshooting

### Common Issues
1. **API Connectivity**
   - Check API credentials
   - Verify exchange status
   - Review rate limits

2. **Risk Rule Violations**
   - Review position sizing
   - Check market conditions
   - Verify risk parameters

3. **Performance Degradation**
   - Analyze recent trades
   - Check market regime changes
   - Review strategy effectiveness

### Emergency Procedures
1. **System Lock**
   - Investigate cause
   - Fix underlying issue
   - Reset safe mode
   - Resume with caution

2. **Exchange Issues**
   - Switch to backup exchange
   - Cancel pending orders
   - Monitor for recovery

3. **Market Extreme Conditions**
   - Activate safe mode
   - Reduce position sizes
   - Increase monitoring frequency

## Documentation

### Required Documentation
- System architecture diagram
- Risk management procedures
- Emergency contact procedures
- Performance benchmarks
- Change log

### Operational Documentation
- Daily operations checklist
- Monitoring procedures
- Troubleshooting guide
- Contact information

This deployment guide ensures safe and controlled rollout of the trading bot with proper risk management and monitoring.
