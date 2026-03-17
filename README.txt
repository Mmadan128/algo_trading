STRATEGY: ML Volatility Momentum + Monte Carlo Optimised
Model: XGBClassifier  AUC=0.635
Signal threshold: 0.38 (chosen by Monte Carlo to maximise P(profit))
Monte Carlo: 10,000 simulated sessions
  Mean P&L: -2.84%
  P(profit): 0.2%
  P(<-10%): 0.0%
Features: 27 (top: hl_range, wick_up, rv_20, sma10_30, ret_20)
Exit: stop 3%  trail 2.0%  target 5%  max 150 bars
Position: 58% cash, capped at 58% net worth