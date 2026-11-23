"""Step 1: label BTC daily regimes following Agakishiev et al. (2025), Table 2.

Choices agreed with Betty (2026-11-22):
- Data: btc_daily_date_fixed_filled.csv, rows from 2014-04-28 onward.
- p_sigma: percentile rank of volatility_30d_annualized.
- p_r:     percentile rank of the 1-day log_return.
- Percentile cutoffs come from the training period only (2014-04-28 to 2021-12-31)
  and are applied unchanged to every day.
- Distress score p_d = (p_sigma + 1 - p_r) / 2.

Rules, checked in this order each day:
  1. p_d > 0.85                     -> Distress
  2. p_sigma < 0.3                  -> Low
  3. 0.4 < p_sigma <= 0.6           -> High
  4. anything else (buffer zones 0.3-0.4 and 0.6-0.7, or p_sigma > 0.7
     without distress)              -> keep the previous day's regime
The first day, if it falls in a buffer, gets Low when p_sigma <= 0.5, else High.

Usage (from the repo root):
    python regime_lstm/label_regimes.py
Defaults: --data data_preprocessing/data/btc_daily_date_fixed_filled.csv, --out regime_lstm/outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = "2014-04-28"
TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"
LOW, HIGH, DISTRESS = "Low", "High", "Distress"
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data_preprocessing" / "data" / "btc_daily_date_fixed_filled.csv"


def percentile_rank(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Share of reference (training) values <= each value, in [0, 1]."""
    ref = np.sort(reference.dropna().to_numpy())
    return pd.Series(np.searchsorted(ref, values.to_numpy(), side="right") / len(ref),
                     index=values.index)


def assign_regimes(p_sigma: pd.Series, p_d: pd.Series) -> pd.Series:
    regimes = []
    previous = None
    for ps, pd_ in zip(p_sigma, p_d):
        if pd_ > 0.85:
            current = DISTRESS
        elif ps < 0.3:
            current = LOW
        elif 0.4 < ps <= 0.6:
            current = HIGH
        elif previous is None:
            current = LOW if ps <= 0.5 else HIGH
        else:
            current = previous
        regimes.append(current)
        previous = current
    return pd.Series(regimes, index=p_sigma.index)


def split_name(date: pd.Timestamp) -> str:
    if date <= pd.Timestamp(TRAIN_END):
        return "train"
    if date <= pd.Timestamp(VAL_END):
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(HERE / "outputs"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["date"])
    df = df[df["date"] >= START].reset_index(drop=True)
    features = ["log_return", "volatility_30d_annualized", "drawdown_from_30d_peak"]
    missing = df[features].isna().sum()
    if missing.any():
        raise ValueError(f"Blank feature values after {START}:\n{missing[missing > 0]}")

    train = df["date"] <= TRAIN_END
    df["p_sigma"] = percentile_rank(df["volatility_30d_annualized"],
                                    df.loc[train, "volatility_30d_annualized"])
    df["p_r"] = percentile_rank(df["log_return"], df.loc[train, "log_return"])
    df["p_d"] = (df["p_sigma"] + 1 - df["p_r"]) / 2
    df["regime"] = assign_regimes(df["p_sigma"], df["p_d"])
    df["switch"] = (df["regime"] != df["regime"].shift()).astype(int)
    df.loc[0, "switch"] = 0
    df["split"] = df["date"].map(split_name)

    cols = ["date", "close", *features, "p_sigma", "p_r", "p_d", "regime", "switch", "split"]
    df[cols].to_csv(out / "btc_regimes.csv", index=False, date_format="%Y-%m-%d")

    # Summary: days per regime and switches per split
    counts = df.pivot_table(index="split", columns="regime", values="date",
                            aggfunc="count", fill_value=0)
    counts = counts.reindex(index=["train", "validation", "test"],
                            columns=[LOW, HIGH, DISTRESS], fill_value=0)
    counts["switches"] = df.groupby("split")["switch"].sum()
    counts["days"] = df.groupby("split").size()
    runs = df.groupby((df["switch"] == 1).cumsum())
    run_table = pd.DataFrame({"regime": runs["regime"].first(), "length": runs.size()})
    durations = run_table.groupby("regime")["length"].agg(["count", "median", "mean", "max"])
    one_day = (run_table["length"] == 1).groupby(run_table["regime"]).sum()
    durations["one_day_runs"] = one_day

    lines = [
        f"Rows: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}",
        "",
        "Days per regime and switches by split:",
        counts.to_string(),
        "",
        "Regime run lengths (days):",
        durations.round(1).to_string(),
    ]
    summary = "\n".join(lines)
    (out / "regime_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)

    plot(df, out / "btc_regimes.png")


def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {LOW: "#2e7d32", HIGH: "#f9a825", DISTRESS: "#c62828"}
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    for regime, color in colors.items():
        y = df["close"].where(df["regime"] == regime)
        ax.scatter(df["date"], y, s=2, color=color, label=regime)
    ax.plot(df["date"], df["close"], color="#9e9e9e", lw=0.4, zorder=0)
    for boundary in (TRAIN_END, VAL_END):
        ax.axvline(pd.Timestamp(boundary), color="black", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("BTC close (log scale)")
    ax.set_title("BTC regimes (dashed lines: train | validation | test)")
    ax.legend(markerscale=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
