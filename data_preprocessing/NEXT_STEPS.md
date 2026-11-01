# Next steps for the market regime detection project

## Current starting point

`data/btc_daily_preprocessed.csv` contains daily BTC closes, market cap, volume,
quality flags, daily changes and returns, and two strict 30-day features. Its
metadata is in `data/btc_daily_preprocessed.metadata.json`. The current CSV runs
from 2013-04-29 through 2026-10-31. It has 138 dates with no API close, six API
closes of zero, and 239 observed zero-volume values. These counts describe this
snapshot and may change when the API is refreshed.

The prepared CSV is a source table, not a final input matrix for all four models.
Do not forward-fill missing BTC prices or volume in this source table: doing so
would create artificial returns or trading activity. Keep raw values and flags.

## 1. Set the research question and evaluation clock

Decide whether the first output is (a) a regime estimate available after each
daily close, (b) a forecast of tomorrow's regime, or (c) a trading decision.
Record when a signal becomes available and when an action could be taken. These
choices determine the labels, feature cutoffs, and backtest timing.

Use chronological train, validation, and final-test periods. An initial daily
plan is training through 2021, validation in 2022-2023, and final testing from
2024 onward. Recheck the cutoffs after measuring the coverage of hourly prices
and on-chain metrics; for a four-model comparison, evaluate on their shared
out-of-sample dates. Never fit a scaler, imputer, PCA, or regime threshold on
validation or test observations.

## 2. Finish the daily market-data layer

1. Check the six zero BTC closes and significant missing-date runs against an
   independent source. Record any correction and its provenance; do not silently
   replace the API values.
2. Adapt the BTC pipeline for ETH, LTC, and XMR. Save one prepared file per coin,
   then join the four on calendar date while retaining each coin's quality flags.
3. Verify that prices use compatible quote currencies and that market cap and
   volume units have the expected meaning. The current endpoint does not make
   those conventions explicit in its response.
4. Decide, using training and validation data, how rolling features should
   handle incomplete windows. The current 30-day volatility and drawdown require
   complete windows; a missing 2026-10-13 close leaves both blank at the latest
   CSV date. If a minimum-coverage rule is adopted, add the valid-observation
   count as a separate feature and document the rule.

## 3. Obtain the inputs the daily endpoint does not provide

| Method | Additional input | Key checks |
| --- | --- | --- |
| Realized covariance adapted to crypto | Synchronized hourly close prices for a liquid basket such as BTC, ETH, LTC, and XMR | Same quote currency, venue/index definition, hourly cutoff, historical coverage, and missing-hour policy. The cited paper used seven traditional-market futures, not ETH. |
| 2025 regime forecasting | Historical BTC network metrics such as active addresses, transaction counts, hash rate, fees, and mining difficulty | Publication/availability timestamp, revisions, units, and gaps. Its separate portfolio-allocation stage also needs hourly market data. |

Do not approximate hourly realized covariance with daily candles and call it a
replication. If the extra sources are unavailable, document a reduced model as
an adaptation.

## 4. Build separate model input tables

| Model family | Initial features or observations |
| --- | --- |
| 2021 crypto Markov switching | Daily price differences for BTC, ETH, LTC, and XMR; compare individual and common states. |
| Gaussian HMM adapted to crypto | Daily returns, trailing volatility, drawdowns, momentum, and carefully checked market-cap/volume features; fit scaling and PCA on training data only. |
| Realized covariance | Hourly returns aligned across coins; sum return outer products within completed months, then fit PCA and a detector on monthly observations. |
| 2025 regime forecast | BTC regime labels defined from past-only volatility and return information, plus available point-in-time network features; predict the next period. |

Keep each feature's earliest valid date, missingness flag, and construction
rule. A trailing feature may use earlier dates for warm-up, but its parameters
must be learned from the training period only.

## 5. Evaluate without future information

Select model settings on validation data, then freeze them for the final test.
For live-style detection, use filtered state probabilities calculated with
observations available through that date; smoothed probabilities use later
observations and belong only in retrospective analysis. For a trading backtest,
generate the signal after the data cutoff and execute at the next genuinely
available price. Include costs and a simple benchmark. Report regime duration,
transition stability, next-period return/volatility separation, and results by
market period. A backtest is the evaluation procedure on held-out dates, not a
substitute for a separate final test set.

## Immediate task

Create the ETH, LTC, and XMR daily source tables with the BTC cleaning rules,
then produce a dated coverage report for the four-coin panel. That will show
the usable common history before choosing model features or final split dates.
