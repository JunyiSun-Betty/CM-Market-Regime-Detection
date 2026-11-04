"""
STEP 1 - Download daily prices for the basket (free Yahoo Finance data).

Run:   python fetch_data.py

Creates:
  data/raw_prices.csv   one close-price column per asset (date index)
  data/btc_input.csv    the team's input format for the target coin:
                        date, close, volume, market_cap
                        (market_cap stays empty until we have the Crypto Matter API)

Tickers are downloaded one at a time, with retries, so yfinance's cache
cannot lock up ("database is locked").
"""
import os
import time
import pandas as pd
import yfinance as yf
import config


def fetch_one(ticker, start, tries=4):
    """Daily history for one ticker, with simple retries."""
    last_error = "no data returned"
    for attempt in range(1, tries + 1):
        try:
            df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
            if df is not None and not df.empty:
                idx = pd.DatetimeIndex(df.index)
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                df.index = idx.normalize()
                return df[~df.index.duplicated(keep="last")]
        except Exception as e:                     # network hiccup, rate limit, cache lock
            last_error = str(e)
        print(f"   {ticker}: attempt {attempt} failed ({last_error}), retrying...")
        time.sleep(3 * attempt)
    return None


def main():
    os.makedirs("data", exist_ok=True)
    try:                                           # fresh cache folder avoids old lock files
        yf.set_tz_cache_location(os.path.abspath("data/yf_cache"))
    except Exception:
        pass

    closes, volumes, failed = {}, {}, []
    for name, ticker in config.BASKET.items():
        print(f"Downloading {name:6s} ({ticker}) ...")
        df = fetch_one(ticker, config.START)
        if df is None:
            failed.append(ticker)
            continue
        closes[name], volumes[name] = df["Close"], df["Volume"]
        time.sleep(1)                              # be gentle with Yahoo

    if failed:
        raise SystemExit(f"\nCould not download: {', '.join(failed)}.\n"
                         "Nothing was saved. Wait a few minutes and run "
                         "'python fetch_data.py' again.")

    close = pd.DataFrame(closes)[list(config.BASKET)].sort_index().loc[config.START:]
    close.index.name = "date"
    close.to_csv("data/raw_prices.csv")

    coin = pd.DataFrame({"close": closes[config.TARGET], "volume": volumes[config.TARGET]})
    coin["market_cap"] = float("nan")
    coin.index.name = "date"
    target_file = f"data/{config.TARGET.lower()}_input.csv"
    coin.sort_index().loc[config.START:].to_csv(target_file)

    print(f"\nSaved data/raw_prices.csv and {target_file}\n")
    print(f"{'asset':7s} {'first date':12s} {'last date':12s} {'rows':>6s}")
    for c in close:
        s = close[c].dropna()
        print(f"{c:7s} {str(s.index[0].date()):12s} {str(s.index[-1].date()):12s} {len(s):6d}")


if __name__ == "__main__":
    main()
