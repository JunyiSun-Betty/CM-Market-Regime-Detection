# CryptoMatter API connector

This project fetches complete daily market-data histories from CryptoMatter without putting the API key in source code.

## Configure and run

Put the key in a local `.env` file:

```text
CM_API_KEY=your-key
```

Alternatively, set `CM_API_KEY` in your shell. A shell value takes priority over `.env`.
Then run:

```powershell
$env:CM_API_KEY = "your-key"
python .\crypto_matter.py bitcoin --column Close
```

Supported columns are `Close`, `Market_cap`, and `Trade_volume`. The key is read from `CM_API_KEY` and sent only in the `X-API-Key` request header.

Run the local tests with:

```powershell
python -m unittest -v
```

The client loads the complete history, so select date ranges after the fetch. Keep the results cached and refresh no more than daily.

## Prepare BTC for regime models

Run:

```powershell
python .\preprocess_btc.py
```

This writes `data/btc_daily_preprocessed.csv` with one row per calendar day. It joins
the API's `Close`, `Market_cap`, and `Trade_volume` series by `CreationDate`.
`close_raw` preserves the API value; `close` is blank when that value is missing
or nonpositive. `invalid_close` flags nonpositive source prices. Missing calendar
dates remain blank, and `has_*` columns distinguish missing values from zero.
`zero_trade_volume` flags an observed zero without changing it.

`price_change`, `simple_return`, and `log_return` use the previous calendar day
only when both closes are valid. `volatility_30d_annualized` is the sample
standard deviation of 30 consecutive daily log returns times `sqrt(365)`.
`drawdown_from_30d_peak` is current close divided by the maximum close in
the last 30 consecutive calendar days, minus one. Rolling features stay blank
until their complete windows are available. No missing prices are filled.
