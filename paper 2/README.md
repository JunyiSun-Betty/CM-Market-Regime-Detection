# Paper 2 model: market regimes from co-movement (covariance + clustering)

Part of the Crypto Matter × Frankfurt School project "AI-Driven Market Regime Detection".
Owner: Mainak. Paper: Bucci & Ciciretti (2022), *Market regime detection via realized
covariances*, Economic Modelling 111 (preprint: arXiv 2104.03667).

## The idea in one paragraph
In a crisis, assets swing harder **and** start moving together. The paper measures that
with covariance matrices of a basket of assets, compresses them with PCA, and uses
hierarchical clustering to split time into two regimes: **calm** and **highly volatile**.
We do the same on daily data, with Bitcoin inside a small basket, and turn the result
into the team's standard output: `date, p_bull, p_bear, p_low_vol, p_high_vol`.

See `docs/architecture_team_pipeline.png` and `docs/architecture_paper2_model.png`.

## Files
| File | What it does |
|---|---|
| `config.py` | Every setting: basket, window, refit schedule, smoothing, costs |
| `fetch_data.py` | Step 1: downloads daily prices from Yahoo Finance into `data/` |
| `simulate_data.py` | Optional: makes fake data with known calm/stress periods (a test) |
| `data_prep.py` | Loads, checks and aligns prices; turns them into daily returns |
| `features.py` | Rolling 21-day covariance, turned into volatilities + correlations |
| `model.py` | PCA + Ward clustering, monthly walk-forward, probabilities, smoothing |
| `backtest.py` | Regime vs. what the price did, transitions, crash-period check, trading test |
| `plots.py` | The three charts |
| `run_paper2.py` | Step 2: runs everything and writes all outputs |
| `experiments.py` | Step 3 (optional): compares model variants with a simple BTC-volatility benchmark |

## How to run (VS Code)
1. Open this folder in VS Code and open a terminal (Terminal → New Terminal).
2. Optional but clean: `python -m venv .venv` and activate it
   (Windows: `.venv\Scripts\activate`, Mac: `source .venv/bin/activate`).
3. `pip install -r requirements.txt`
4. `python fetch_data.py`  (needs internet; about 10 seconds)
5. `python run_paper2.py --compare`  (about 30 seconds)
6. Open `outputs/report.md` (in VS Code: right-click → Open Preview) and the PNG charts.
7. Optional: `python experiments.py` writes `outputs/experiments.md` (variants vs. benchmark).

Test without internet: `python simulate_data.py` then
`python run_paper2.py --prices data/sim_prices.csv --truth data/sim_truth.csv --out outputs_sim --compare`.
`example_outputs_simulation/` already contains the result of this test.

When the Crypto Matter API is ready, save its prices in the same format as
`data/raw_prices.csv` (a `date` column + one column per asset) and run
`python run_paper2.py --prices <your file>`.

## Outputs (in `outputs/`)
| File | Meaning |
|---|---|
| `paper2_model_output.csv` | The team data contract. p_bull/p_bear are empty on purpose: this model has no direction |
| `paper2_details.csv` | Daily probability, raw label, smoothed label, BTC price |
| `latest_regime.json` | Today's answer in the team's JSON format |
| `refit_diagnostics.csv` | Quality of each monthly refit (PCA variance explained, silhouette) |
| `report.md` | All checks and tables in plain English |
| `chart_regimes.png` | BTC price with the high-volatility periods shaded, plus P(high vol) |
| `chart_clusters.png` | How the last fit splits past days into the two groups |
| `chart_strategy.png` | The paper's trading test: trend rule with and without the regime filter |

## How to read the report
- **Section 5, next_20d_volatility**: the honest test. BTC's volatility over the 20 days
  *after* a high-vol label should be clearly higher than after a calm label.
- **Section 7, crash periods**: COVID (Mar 2020), May 2021, Terra/LUNA (2022) and FTX
  (Nov 2022) should be mostly labelled high volatility.
- **Section 4, label switches per year**: a few per year is healthy; dozens means noise.
- **Section 8, trading test**: the paper found that skipping the trend rule in high
  volatility improved the Sharpe ratio. Check if that holds for BTC.
- **`--compare`** adds the paper's own feature set (correlation distances only) next to ours.

## What we kept and what we changed
| Paper | Ours | Why |
|---|---|---|
| Hourly prices, monthly realized covariance | Daily prices, rolling 21-day covariance, updated daily | We only have daily data, and we need a daily answer |
| 9 (preprint) / 7 (published) futures | BTC, ETH, S&P 500, gold, US 10Y yield, VIX | Crypto plus the market around it |
| 3 principal components, Ward clustering, 2 regimes | Same | Faithful to the paper |
| Hard labels | Probability from distance to each cluster centre | The team output needs probabilities |
| Clusters fitted once | Refitted monthly on past data only | No look-ahead in the backtest |
| Manhattan distance | Euclidean (scikit-learn's Ward only supports Euclidean) | Library limit |

## Known limits
- No bull/bear: the model only separates calm from highly volatile markets.
- The rolling window reacts with a delay, so regimes start and end a little late.
- The first fits have only about 2 years of history, so early labels are less stable.
- The published paper (clustering best) and the preprint (VLSTAR best) disagree; the
  VLSTAR model exists only in R (`starvars` package) and is not included here.
