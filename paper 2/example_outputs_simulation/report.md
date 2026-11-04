# Paper 2 model report (simulated data)

## 1. Data check
```
Rows: 2330  (2017-11-10 to 2026-10-15)
asset   first date   missing  biggest daily move
BTC     2017-11-10      0.0%                 39%
ETH     2017-11-10      0.0%                 39%
SPX     2017-11-10      0.0%                  7%
GOLD    2017-11-10      0.0%                  7%
US10Y   2017-11-10      0.0%             0.38 pp
VIX     2017-11-10      0.0%                 50%
```

Basket: BTC, ETH, SPX, GOLD, US10Y, VIX. Target: BTC. Aligned on SPX trading days: 2330 days (2017-11-10 to 2026-10-15).

## 2. Model setup
- Rolling covariance window: 21 business days, recomputed daily
- Feature set: adapted (22 features per day), reduced to 3 principal components
- Ward hierarchical clustering into 2 regimes, refitted every 21 days on past data only (first fit after 504 days)
- Smoothing: a new regime must last 5 days before the label switches
- Labelled out of sample: 1805 days (2019-11-15 to 2026-10-15)

## 3. Latest answer
BTC on 2026-10-15: **Low volatility**, P(high volatility) = 0.04. Full JSON in latest_regime.json.

## 4. Key numbers

|                                          | adapted   | paper   |
|:-----------------------------------------|:----------|:--------|
| high-vol share of days                   | 36%       | 35%     |
| label switches per year                  | 3.63      | 4.33    |
| next-20d volatility after high-vol days  | 97%       | 96%     |
| next-20d volatility after calm days      | 63%       | 64%     |
| PCA variance explained (avg)             | 62%       | 59%     |
| cluster silhouette (avg)                 | 0.58      | 0.49    |
| accuracy - smoothed                      | 80%       | 78%     |
| missed stress (share of days) - smoothed | 5%        | 7%      |
| false alarms (share of days) - smoothed  | 15%       | 15%     |
| accuracy - raw (no smoothing)            | 82%       | 79%     |

Silhouette runs from -1 to 1 (higher = clearer clusters). Volatility figures are annualised.

## 5. Regimes vs. what the price did (backtest)

|          | time_share   |   avg_length_days | ann_return   | ann_volatility   | next_20d_volatility   | avg_drawdown_from_peak   | worst_drawdown_from_peak   |
|:---------|:-------------|------------------:|:-------------|:-----------------|:----------------------|:-------------------------|:---------------------------|
| high_vol | 36%          |             50.23 | 16%          | 107%             | 97%                   | -38%                     | -87%                       |
| low_vol  | 64%          |             82.29 | 52%          | 65%              | 63%                   | -30%                     | -67%                       |

## 6. Transition matrix (chance of tomorrow's regime, given today's)

| regime   |   high_vol |   low_vol |
|:---------|-----------:|----------:|
| high_vol |      0.98  |     0.02  |
| low_vol  |      0.011 |     0.989 |

## 7. Simulation check

Accuracy against the true regimes is in section 4.

## 8. Trading test (the paper's idea: skip the trend rule in high volatility)

|                            | ann_return   | ann_volatility   |   sharpe | max_drawdown   | time_invested   |   trades_per_year |
|:---------------------------|:-------------|:-----------------|---------:|:---------------|:----------------|------------------:|
| Buy and hold               | 47%          | 83%              |     0.88 | -87%           | 100%            |              0    |
| Trend rule (30/100 MA)     | 17%          | 63%              |     0.56 | -81%           | 62%             |              3.35 |
| Trend rule + regime filter | 25%          | 41%              |     0.75 | -62%           | 40%             |              4.05 |
| Hold only in calm regime   | 38%          | 52%              |     0.88 | -77%           | 64%             |              3.77 |

## 9. Notes
- p_bull / p_bear are empty: this model only separates calm from highly volatile markets.
- Daily data and a 21-day window replace the paper's hourly data and monthly matrices.
- scikit-learn's Ward linkage uses Euclidean distance; the paper used Manhattan.
- Past results do not guarantee future results; this is a first prototype.
