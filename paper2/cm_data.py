"""
Crypto Matter API: test the connection, download and cache the data, check it,
and build the files our models use.

Before running, put the key in the terminal (PowerShell, once per terminal):
    $env:CM_API_KEY = "<your key>"
Then:
    python fetch_data.py        (refreshes the Yahoo market data: S&P 500, gold, yield, VIX)
    python cm_data.py

Options:
    --coins bitcoin ethereum    Crypto Matter coin IDs to download (default: bitcoin ethereum)
    --refresh                   download again even if today's files already exist

Creates in data/cm/:
    <coin>_Close.csv, <coin>_Market_cap.csv, <coin>_Trade_volume.csv   the downloads (cache)
    cm_data_check.md     plain-English check of every series
    cm_overview.png      price, market cap and volume for each coin
    cm_vs_yahoo.png      Bitcoin close, Crypto Matter vs Yahoo (needs data/raw_prices.csv)
Creates in data/:
    btc_input_cm.csv     team input format: date, close, volume, market_cap (Crypto Matter)
    cm_prices.csv        Paper 2 basket: BTC and ETH from Crypto Matter, the rest from Yahoo

The key is read from the CM_API_KEY environment variable only. Never write it into
a script, a notebook, Git or a chat.
"""
import argparse
import datetime as dt
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://backend.crypto-matter.com/api"
COLUMNS = ["Close", "Market_cap", "Trade_volume"]
CACHE = os.path.join("data", "cm")
SHORT = {"bitcoin": "BTC", "ethereum": "ETH"}   # names used in raw_prices.csv
RETRY_WAIT = 60                                 # seconds, as Tobias suggests for 500/503
_KEY = None


# ---------------------------------------------------------------- download
def api_key():
    global _KEY
    if _KEY is None:
        _KEY = os.environ.get("CM_API_KEY", "").strip()
        if not _KEY:
            sys.exit("CM_API_KEY is not set. In PowerShell run:\n"
                     '    $env:CM_API_KEY = "<your key>"\n'
                     "and then run this script again in the SAME terminal.")
    return _KEY


def fetch(coin, column, tries=3):
    """Full daily history of one series as a pandas Series indexed by date."""
    msg = ""
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(f"{BASE_URL}/coins/marketdata",
                             params={"coin": coin, "column": column},
                             headers={"X-API-Key": api_key()}, timeout=30)
        except requests.RequestException as e:
            msg = f"network problem ({e.__class__.__name__})"
        else:
            if r.status_code == 200:
                df = pd.DataFrame(r.json())
                if df.empty or column not in df:
                    sys.exit(f"{coin} {column}: the API answered but returned no data.")
                s = pd.to_numeric(df[column], errors="coerce")          # null -> NaN
                s.index = pd.to_datetime(df["CreationDate"]).dt.normalize()
                s.index.name = "date"
                return s.rename(column).sort_index()
            hints = {400: "unknown column (column names are case-sensitive)",
                     403: "key missing, invalid or not active: check CM_API_KEY",
                     404: f"no data for coin ID '{coin}': ask Tobias for the right ID"}
            if r.status_code in hints:
                sys.exit(f"{coin} {column}: HTTP {r.status_code}, {hints[r.status_code]}.\n"
                         f"Server said: {r.text[:300]}")
            msg = f"HTTP {r.status_code}"
        if attempt < tries:
            print(f"   {coin} {column}: {msg}, trying again in {RETRY_WAIT} s ...")
            time.sleep(RETRY_WAIT)
    sys.exit(f"{coin} {column}: still failing ({msg}). Try later; if it persists, tell Tobias.")


def load(coin, column, refresh):
    """Use today's cached file if there is one, otherwise call the API once."""
    path = os.path.join(CACHE, f"{coin}_{column}.csv")
    fresh = os.path.exists(path) and \
        dt.date.fromtimestamp(os.path.getmtime(path)) == dt.date.today()
    if fresh and not refresh:
        s = pd.read_csv(path, parse_dates=["date"], index_col="date")[column]
        return s, "cache"
    s = fetch(coin, column)
    s.to_csv(path)
    time.sleep(1)                           # be gentle with the API
    return s, "API"


# ---------------------------------------------------------------- checks
def clean(s):
    """Prices, market caps and volumes can't be zero or negative: treat those as missing."""
    return s.where(s > 0)


def check(s):
    """Plain-English quality check of one series (on the raw download)."""
    v = s.dropna()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    missing = full.difference(s.index)
    nulls = s[s.isna()].index
    zeros = v[v <= 0].index
    gaps = s.index.to_series().diff().dt.days - 1          # missing days before each row
    longest = (f"{int(gaps.max())} days before {gaps.idxmax().date()}"
               if len(gaps.dropna()) and gaps.max() > 0 else "-")
    moves = np.log(v[v > 0]).diff().dropna()
    big = moves.abs().idxmax() if len(moves) else None
    return {
        "first date": str(s.index.min().date()),
        "last date": str(s.index.max().date()),
        "rows": len(s),
        "empty values (null)": int(s.isna().sum()),
        "first empty dates": ", ".join(str(d.date()) for d in nulls[:3]) or "-",
        "zero or negative values": len(zeros),
        "first zero dates": ", ".join(str(d.date()) for d in zeros[:3]) or "-",
        "missing calendar days": len(missing),
        "missing days since 2017-11-10": int((missing >= "2017-11-10").sum()),
        "longest gap": longest,
        "duplicate dates": int(s.index.duplicated().sum()),
        "biggest move between rows (zeros skipped)":
            (f"{100 * (np.exp(moves[big]) - 1):+.0f}% on {big.date()}" if big is not None else "-"),
    }


def compare_with_yahoo(cm_close, yahoo_close):
    """How close is Crypto Matter's Bitcoin close to Yahoo's? Also checks whether the
    two 'days' are shifted against each other (different closing times)."""
    cal = pd.date_range(max(cm_close.index.min(), yahoo_close.index.min()),
                        min(cm_close.index.max(), yahoo_close.index.max()), freq="D")
    both = pd.concat({"cm": clean(cm_close), "yahoo": clean(yahoo_close)}, axis=1).reindex(cal)
    d = both.dropna()
    if len(d) < 30:
        return None, d
    diff = (d["cm"] / d["yahoo"] - 1).abs()
    # daily returns on the full calendar, so a missing day is never bridged
    r_cm, r_y = np.log(both["cm"]).diff(), np.log(both["yahoo"]).diff()
    corr = {lag: r_cm.corr(r_y.shift(lag)) for lag in (-1, 0, 1)}
    names = {0: "same day", 1: "Crypto Matter one day later",
             -1: "Crypto Matter one day earlier"}
    # same date, and Crypto Matter's value compared with Yahoo's previous / next day
    diff_prev = (both["cm"] / both["yahoo"].shift(1) - 1).abs().median()
    diff_next = (both["cm"] / both["yahoo"].shift(-1) - 1).abs().median()
    out = {
        "overlapping days": len(d),
        "period": f"{d.index[0].date()} to {d.index[-1].date()}",
        "median price difference, same date": f"{diff.median():.2%}",
        "... vs Yahoo's previous day": f"{diff_prev:.2%}",
        "... vs Yahoo's next day": f"{diff_next:.2%}",
        "days more than 2% apart": f"{(diff > 0.02).mean():.1%}",
        "daily return correlation, same day": f"{corr[0]:.3f}",
        "... if Crypto Matter is one day later": f"{corr[1]:.3f}",
        "... if Crypto Matter is one day earlier": f"{corr[-1]:.3f}",
        "best alignment": names[max(corr, key=lambda k: -1 if pd.isna(corr[k]) else corr[k])],
    }
    return out, d


# ---------------------------------------------------------------- charts
def plot_overview(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1", "#555555"]
    titles = {"Close": "Daily close (USD)", "Market_cap": "Market cap (USD)",
              "Trade_volume": "24-hour trading volume (USD)"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, col in zip(axes, COLUMNS):
        for i, coin in enumerate(data):
            s = data[coin][col].dropna()
            s = s[s > 0]
            if len(s):
                ax.plot(s.index, s, lw=1.2, color=colours[i % len(colours)], label=coin)
        ax.set_yscale("log")
        ax.set_title(titles[col], loc="left", fontsize=11)
        ax.set_facecolor("#fcfcfb")
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Crypto Matter API data (log scale)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_vs_yahoo(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    a1.plot(d.index, d["yahoo"], color="#2a78d6", lw=1.4, label="Yahoo")
    a1.plot(d.index, d["cm"], color="#eb6834", lw=1, label="Crypto Matter")
    a1.set_yscale("log")
    a1.set_title("Bitcoin daily close", loc="left", fontsize=11)
    a1.legend(frameon=False)
    a2.plot(d.index, 100 * (d["cm"] / d["yahoo"] - 1), color="#eb6834", lw=0.8)
    a2.axhline(0, color="#888", lw=0.8)
    a2.set_title("Difference, Crypto Matter vs Yahoo (%)", loc="left", fontsize=11)
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum"])
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    # 1. connection test + downloads
    print("Step 1: downloading from the Crypto Matter API")
    data = {}
    for coin in a.coins:
        data[coin] = {}
        for col in COLUMNS:
            s, source = load(coin, col, a.refresh)
            data[coin][col] = s
            print(f"   {coin:10s} {col:13s} {len(s):6d} days  "
                  f"{s.index.min().date()} to {s.index.max().date()}  (from {source})")
    print("   Connection OK.\n")

    # 2. checks
    print("Step 2: checking the data")
    checks = pd.DataFrame({f"{coin} {col}": check(data[coin][col])
                           for coin in data for col in COLUMNS})
    short = checks.T[["first date", "last date", "empty values (null)", "zero or negative values",
                      "missing calendar days", "missing days since 2017-11-10", "longest gap"]]
    short.columns = ["first date", "last date", "nulls", "zeros", "missing days",
                     "missing since 2017-11", "longest gap"]
    print(short.to_string())
    for name, row in checks.T.iterrows():
        if row["zero or negative values"]:
            print(f"   {name}: zero values on {row['first zero dates']}"
                  f"{' ...' if row['zero or negative values'] > 3 else ''}")
    print("   Zeros are treated as missing in every file this script builds.")
    print("   (full table in data/cm/cm_data_check.md)\n")

    yahoo = None
    if os.path.exists("data/raw_prices.csv"):
        yahoo = pd.read_csv("data/raw_prices.csv", parse_dates=["date"], index_col="date")
    comparison = None
    if yahoo is not None and "bitcoin" in data and "BTC" in yahoo:
        comparison, both = compare_with_yahoo(data["bitcoin"]["Close"], yahoo["BTC"])
        if comparison:
            print("Step 3: Bitcoin close, Crypto Matter vs Yahoo")
            for k, v in comparison.items():
                print(f"   {k:42s} {v}")
            plot_vs_yahoo(both, os.path.join(CACHE, "cm_vs_yahoo.png"))
            print()

    # 3. files for the models
    made = []
    if "bitcoin" in data:
        b = data["bitcoin"]
        coin = pd.DataFrame({"close": clean(b["Close"]), "volume": clean(b["Trade_volume"]),
                             "market_cap": clean(b["Market_cap"])})
        coin.index.name = "date"
        coin.sort_index().to_csv("data/btc_input_cm.csv")
        made.append("data/btc_input_cm.csv  (team input format: date, close, volume, market_cap)")
    if yahoo is not None and all(c in data for c in SHORT):
        px = yahoo.copy()
        for coin, name in SHORT.items():
            px = px.drop(columns=name, errors="ignore")
        cm = pd.DataFrame({name: clean(data[coin]["Close"]) for coin, name in SHORT.items()})
        px = cm.join(px, how="outer").sort_index()
        px = px[["BTC", "ETH"] + [c for c in px if c not in ("BTC", "ETH")]]
        px.index.name = "date"
        px.to_csv("data/cm_prices.csv")
        made.append("data/cm_prices.csv     (Paper 2 basket: BTC + ETH from Crypto Matter, rest from Yahoo)")
    elif yahoo is None:
        print("data/raw_prices.csv not found: run 'python fetch_data.py' first to build cm_prices.csv.\n")

    plot_overview(data, os.path.join(CACHE, "cm_overview.png"))

    # 4. report
    L = ["# Crypto Matter data check", "",
         f"Run on {dt.date.today()}. Coins: {', '.join(data)}. Columns: {', '.join(COLUMNS)}.", "",
         "## 1. Every series", "", checks.to_markdown(), "",
         "Missing calendar days = dates with no row at all. Empty values = rows whose value is null. "
         "Zero values are impossible for prices and are treated as missing in every file built here. "
         "Longest gap = the most days in a row without a row.", ""]
    if comparison:
        L += ["## 2. Bitcoin close: Crypto Matter vs Yahoo", "",
              pd.Series(comparison).to_frame("value").to_markdown(), "",
              "Small price differences are normal (different exchanges and closing times). "
              "If the difference is smaller against Yahoo's previous or next day than on the "
              "same date, or 'best alignment' is not 'same day', the two sources label or close "
              "the day differently: ask Tobias what CreationDate means and at what time (UTC) "
              "the daily close is taken.", ""]
    L += ["## 3. Files created", ""] + [f"- {m}" for m in made] + [
          "", "Charts: data/cm/cm_overview.png" +
          (", data/cm/cm_vs_yahoo.png" if comparison else "")]
    with open(os.path.join(CACHE, "cm_data_check.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print("Files created:")
    for m in made:
        print("   " + m)
    print("   data/cm/cm_data_check.md, data/cm/cm_overview.png"
          + (", data/cm/cm_vs_yahoo.png" if comparison else ""))
    if comparison and comparison["best alignment"] != "same day":
        print(f"\nNOTE: the dates look shifted ({comparison['best alignment']}) compared with Yahoo.\n"
              "Ask Tobias what CreationDate means and when the daily close is taken, before\n"
              "mixing Crypto Matter prices with stock-market data from Yahoo.")
    if any("cm_prices" in m for m in made):
        print("\nNext: python run_paper2.py --prices data/cm_prices.csv --out outputs_cm")


if __name__ == "__main__":
    main()
