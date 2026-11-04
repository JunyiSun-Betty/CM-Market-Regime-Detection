"""
Check how Crypto Matter dates its daily closes, compared with Yahoo Finance.

Yahoo's crypto day closes at 00:00 UTC at the END of the date. For each period we
test whether Crypto Matter's value for a date matches Yahoo's close of the SAME date
or of the PREVIOUS date (that is, labelled one day late).

Uses the files cm_data.py already saved (no API call, no key needed):
    data/cm/<coin>_Close.csv
Downloads the matching Yahoo history (free, no key).

Run:
    python check_cm_dates.py                       (bitcoin and ethereum)
    python check_cm_dates.py --coins bitcoin litecoin

Creates:
    data/cm/date_check.md        table by year + the dates where the convention switches
    data/cm/date_check_<coin>.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

YAHOO = {"bitcoin": "BTC-USD", "ethereum": "ETH-USD", "litecoin": "LTC-USD",
         "binancecoin": "BNB-USD", "monero": "XMR-USD", "solana": "SOL-USD"}
CACHE = os.path.join("data", "cm")


def load_cm(coin):
    path = os.path.join(CACHE, f"{coin}_Close.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run: python cm_data.py --coins {coin}")
    s = pd.read_csv(path, parse_dates=["date"], index_col="date")["Close"]
    return s.where(s > 0)                                   # zeros = missing


def load_yahoo(coin, csv=None):
    if csv:                                                 # offline test option
        df = pd.read_csv(csv, parse_dates=["date"], index_col="date")
        return df[csv_col(coin)].dropna() if csv_col(coin) in df else pd.Series(dtype=float)
    import yfinance as yf
    try:                                        # own cache folder avoids "database is locked"
        yf.set_tz_cache_location(os.path.abspath(os.path.join(CACHE, "yf_cache")))
    except Exception:
        pass
    df = yf.Ticker(YAHOO[coin]).history(period="max", auto_adjust=True)
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return df["Close"][~df.index.duplicated(keep="last")]


def csv_col(coin):
    return {"bitcoin": "BTC", "ethereum": "ETH"}.get(coin, coin)


def analyse(cm, y):
    if len(y) == 0 or len(cm.dropna()) == 0:
        return None, None, None
    start, end = max(cm.index.min(), y.index.min()), min(cm.index.max(), y.index.max())
    if end - start < pd.Timedelta(days=60):
        return None, None, None
    cal = pd.date_range(start, end)
    d = pd.DataFrame({"cm": cm, "y": y}).reindex(cal)
    same = (d.cm / d.y - 1).abs()
    late = (d.cm / d.y.shift(1) - 1).abs()                  # CM date = Yahoo's previous day
    r = np.log(d).diff()
    rows = []
    for yr, g in d.groupby(d.index.year):
        rg = r.loc[g.index]
        c_same, c_late = rg.cm.corr(rg.y), rg.cm.corr(rg.y.shift(1))
        if g.cm.notna().sum() < 30 or pd.isna(c_same) or pd.isna(c_late):
            verdict = "not enough data"
        elif max(c_same, c_late) < 0.5:
            verdict = "unclear (prices don't match well)"
        else:
            verdict = "same day" if c_same > c_late else "one day late"
        rows.append({"year": yr, "days": int(g.cm.notna().sum()),
                     "return corr, same day": round(c_same, 2),
                     "return corr, one day late": round(c_late, 2),
                     "median diff, same day": f"{same.loc[g.index].median():.2%}",
                     "median diff, one day late": f"{late.loc[g.index].median():.2%}",
                     "verdict": verdict})
    # day-level vote, smoothed over 31 days, to locate the switch dates
    vote = pd.Series(np.where(late < same, 1.0, 0.0), index=cal).where(same.notna() & late.notna())
    smooth = vote.rolling(31, center=True, min_periods=10).mean()
    state = pd.Series(np.where(smooth > 0.5, "one day late", "same day"), index=cal)
    state = state[smooth.notna()]
    # ignore blips shorter than 14 days (calm days where both prices are almost equal)
    run = (state != state.shift()).cumsum()
    lengths = state.groupby(run).transform("size")
    state = state.where(lengths >= 14).ffill().bfill()
    switches = state[state != state.shift()]
    table = pd.DataFrame(rows).set_index("year")
    # sanity check, judged day by day and year by year (a long history mixes both conventions)
    best_gap = pd.concat([same, late], axis=1).min(axis=1).median()   # gap on the better-fitting day
    clear_years = table["verdict"].isin(["same day", "one day late"]).mean()
    table.attrs["best_gap"], table.attrs["clear_years"] = best_gap, clear_years
    # only refuse when the prices clearly belong to something else (e.g. a wrong coin ID):
    # a typical gap above 20% even on the better-fitting day, or no clear year at all
    if best_gap > 0.20 or clear_years < 0.5:
        switches = pd.Series(dtype=object)
    return table, switches, smooth


def plot(smooth, coin, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(smooth.index, smooth, color="#eb6834", lw=1.2)
    ax.axhline(0.5, color="#888", lw=0.8, ls="--")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["same day", "", "one day late"])
    ax.set_title(f"{coin}: does Crypto Matter's date match Yahoo's close of the same day? "
                 "(31-day average)", loc="left", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum"])
    ap.add_argument("--yahoo-csv", default=None, help="offline test: use a saved Yahoo file")
    a = ap.parse_args()
    L = ["# Crypto Matter date check", "",
         "Crypto Matter's close for a date is compared with Yahoo's close of the same date and "
         "of the previous date. 'One day late' means Crypto Matter labels the 00:00 UTC snapshot "
         "with the next date.", ""]
    for coin in a.coins:
        cm, y = load_cm(coin), load_yahoo(coin, a.yahoo_csv)
        table, switches, smooth = analyse(cm, y)
        if table is None:
            print(f"\n{coin}: Crypto Matter and Yahoo overlap too little to compare, skipped.")
            L += [f"## {coin}", "", "Not enough overlapping data to compare.", ""]
            continue
        print(f"\n{coin}: compared {table['days'].sum()} days "
              f"({table.index.min()} to {table.index.max()})")
        print(table[["return corr, same day", "return corr, one day late", "verdict"]].to_string())
        print(f"Typical price gap on the better-fitting day: {table.attrs['best_gap']:.2%}  "
              f"(years with a clear verdict: {table.attrs['clear_years']:.0%})  [script version 3]")
        if switches.empty:
            print("WARNING: Crypto Matter's prices don't match Yahoo's on either day, so no "
                  "periods are given. Check the coin ID and the data.")
            L += [f"## {coin}", "", table.to_markdown(), "",
                  "Prices don't match Yahoo on either day: no periods given.", ""]
            continue
        print("Convention by period (switch dates are approximate, +/- a few days):")
        ends = list(switches.index[1:]) + [smooth.dropna().index[-1]]
        for (start, label), end in zip(switches.items(), ends):
            print(f"   {start.date()} to {end.date()}: {label}")
        plot(smooth, coin, os.path.join(CACHE, f"date_check_{coin}.png"))
        L += [f"## {coin}", "", table.to_markdown(), "", "Periods:", ""]
        L += [f"- {s.date()} to {e.date()}: **{lab}**" for (s, lab), e in zip(switches.items(), ends)]
        L += ["", f"Chart: data/cm/date_check_{coin}.png", ""]
    with open(os.path.join(CACHE, "date_check.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\nSaved data/cm/date_check.md and the charts in data/cm/")


if __name__ == "__main__":
    main()
