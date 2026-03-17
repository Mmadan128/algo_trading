## Strategy Overview

**ML Volatility Momentum + Monte Carlo Optimised**

We built an autonomous trading agent that uses a Gradient Boosting model (XGBoost) trained on 116,000 bars of 1-minute OHLCV data to predict short-term upward price movements, combined with Monte Carlo simulation to optimise entry threshold and position sizing for the 3-hour live session.

The core insight: **large upward moves in Asset Alpha are preceded by volatility expansion (high-low range spikes), not compression.** This is the opposite of the standard "volatility squeeze" strategy and was discovered empirically by analysing feature correlations with forward returns.

---

## What We Built

### Day 1 Pipeline

```
Raw CSV (116k bars)
    ↓
Feature Engineering (27 features)
    ↓
XGBoost Classifier (AUC = 0.635)
    ↓
Trade Distribution Extraction
    ↓
Monte Carlo Simulation (10,000 sessions)
    ↓
Threshold Optimisation → CONF_THR = 0.25
    ↓
agent.py + model.pkl
```

---

## Key Findings from Data Analysis

### 1. Fees Kill Everything

The first thing we proved was that frequent trading is impossible to make profitable:

- Fee per round trip: **0.2%** (0.1% each way)
- Oracle P&L (trading every bar): **-99%** — even perfect prediction loses to fees
- SMA crossover baseline: **-22%** on 10k bars, **-79%** on full data (4,800 trades)
- **Conclusion:** must minimise trade count, maximise hold time

### 2. Cash Decay Dominates Short Sessions

- Cash decays at **0.02%/min** on idle capital during the live run
- Over 3 hours: **-3.6%** on fully uninvested cash
- Any strategy that stays mostly in cash loses 3.6% regardless of signal quality
- **Conclusion:** must be invested for most of the session

### 3. The Real Pattern: Volatility Expansion

The starter notebook deliberately showed three features before the price chart:
- **Rolling 5-bar volatility** (`roll_vol`)
- **Volume ratio** vs 10-bar average (`vol_ratio`)
- **30-bar Z-score** (`zscore`)

After testing every combination, we found that large upward moves (>0.5% in 10 bars) are preceded by **high volatility (1.69× the baseline average)**, not low volatility. The compression/breakout thesis is wrong for this asset.

### 4. Hurst Exponent = 0.49

The asset is essentially a random walk at the 1-minute level. Trend-following strategies (SMA crossover) systematically lose because there is no persistent trend at that granularity.

### 5. Candle Shape Predicts Better Than Vol Alone

Feature importance from XGBoost:

| Feature | Importance |
|---------|-----------|
| `hl_range` (high-low / open) | 0.385 |
| `wick_up` (upper wick size) | 0.109 |
| `rv_20` (20-bar rolling vol) | 0.071 |
| `sma10_30` (trend ratio) | 0.046 |
| `ret_20` (20-bar return) | 0.046 |

The high-low range of a candle is by far the strongest predictor — it captures the total energy of the bar, including both directions, which the model uses to identify when the market is "deciding something."

---

## Feature Engineering

27 features computed from OHLCV data:

```python
# Volatility (8 features)
rv_3, rv_5, rv_10, rv_20          # rolling std of close
rv_exp_3, rv_exp_5, rv_exp_10, rv_exp_20  # expansion ratio vs prev bar

# Volume (3 features)
vr_5, vr_10, vr_20                # volume / rolling mean

# Momentum (6 features)
ret_1, ret_2, ret_3, ret_5, ret_10, ret_20  # pct returns

# Trend (2 features)
sma5_20, sma10_30                  # SMA ratio (trend direction)

# Mean reversion (3 features)
z_10, z_20, z_30                   # z-scores at multiple windows

# Candle shape (4 features)
body, wick_up, wick_down, hl_range  # candle anatomy

# Momentum oscillator (1 feature)
rsi14                               # manual RSI
```

All features are normalised (divided by price or prior value) to be scale-invariant across the asset's price range (63–155).

---

## Model

**XGBoost Classifier**

```python
XGBClassifier(
    n_estimators    = 300,
    max_depth       = 4,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    min_child_weight= 30,
    eval_metric     = 'logloss',
)
```

**Target:** binary — will price be >0.15% higher in the next 10 bars?

**Train/test split:** 80/20, time-ordered (no shuffling — critical for time series)

**Results:**
- AUC: **0.635** (random = 0.50, meaningful = 0.60+)
- Precision at threshold 0.40: **42.8%**
- Training time: **~2 seconds** (XGBoost C++ histogram algorithm)

---

## Monte Carlo Simulation

After extracting the historical trade distribution from model signals on the test set, we simulated 10,000 live sessions to find the optimal configuration.

### Trade Distribution (at CONF_THR=0.38)
- Trades extracted: 61
- Win rate: 42.6%
- Average win: +1.215%
- Average loss: -1.008%
- EV net of fees: -0.060%

### Session Simulation (Approach B winner)

We tested three approaches:

| Approach | Description | P(profit) | Mean P&L |
|----------|-------------|-----------|---------|
| A — loose threshold | Many short trades | 0.0% | -3.12% |
| B — one entry, hold long | Single entry, hold 150 bars | **3.2%** | **-2.49%** |
| C — medium frequency | 3–5 trades, 25-bar hold | 0.0% | -3.13% |

**Approach B wins** because it minimises both fee drag and cash decay simultaneously.

### Final Monte Carlo (timing-aware)

With signal firing at median bar 15 of 180 (confirmed from training data analysis):

```
Mean P&L:    -0.30%
Median P&L:  -0.36%
P(profit):   34.1%
P(> +1%):     8.3%
P(< -5%):     0.0%
Best 5%:     +1.19%
Worst 5%:    -1.58%
```

The timing-aware simulation accounts for the fact that the signal fires early (median bar 15 = ~2.5 minutes into session), leaving ~25 minutes of invested time vs only ~2.5 minutes of cash decay.

---

## Signal Logic

```python
# Entry: model confidence >= CONF_THR
conf = model.predict_proba(features)[0, 1]
if conf >= CONF_THR:
    BUY

# Exit conditions (checked every tick):
if ret < -0.03:          # stop loss
if trail < -0.02:        # trailing stop (2% below peak)
if ret > +0.05:          # take profit
if hold_bars >= 150:     # time exit (~25 min at 10s ticks)
```

---

## Final Parameters

```python
CONF_THR = 0.25          # entry confidence threshold

PARAMS = {
    'pos_pct':  0.58,    # 58% of capital per trade (near 60% limit)
    'stop':     0.03,    # 3% hard stop loss
    'profit':   0.05,    # 5% take profit
    'trail':    0.02,    # 2% trailing stop from peak
    'max_hold': 150,     # hold for 150 ticks (~25 minutes)
    'cooldown': 180,     # one trade per session
    'fee':      0.001,   # 0.1% per side
}
```

**Why 58% position size?** Cash decay at 0.02%/min means idle capital loses 3.6% over 3 hours. Deploying 58% of capital immediately after a signal fires (median ~2.5 min in) means only ~2.5% of that cash sits idle — total decay impact drops from 3.6% to under 0.5%.

**Why cooldown=180?** One high-quality entry and hold beats multiple fee-eroding trades. The Monte Carlo confirmed this decisively.

---

## Sandbox Results

Tested on the live sandbox before submission:

- Signal fired at tick 9 (conf=0.437)
- Entry: 100.4587
- Position peaked at +0.39% (price 101.29)
- Leaderboard rank: **4th** at +0.43%

---

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Autonomous trading agent |
| `model.pkl` | Trained XGBoost model (116k rows, full data) |
| `README.txt` | This file |
| `requirements.txt` | Python dependencies |

---

## Running the Agent

```bash
API_URL=http://live-server TEAM_API_KEY=your-key python agent.py
```

The agent will:
1. Warm up feature buffers from `/api/history`
2. Print `HOLD | price | conf=X.XXX` each tick
3. Print `BUY N @ price conf=X.XXX` when signal fires
4. Print `SELL @ price ret=+X.XX%` on exit
5. Stop cleanly on `Ctrl+C` or when phase = "closed"

---

## Requirements

```
numpy
pandas
scikit-learn
xgboost
requests
joblib
```

---
