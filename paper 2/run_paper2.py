"""
STEP 2 - Run the Paper 2 model end to end and write every output.

Real data (run fetch_data.py first):
    python run_paper2.py --compare

Test on simulated data with known regimes (no internet needed):
    python simulate_data.py
    python run_paper2.py --prices data/sim_prices.csv --truth data/sim_truth.csv --out outputs_sim --compare

Outputs (in the --out folder):
    paper2_model_output.csv   team data contract: date, p_bull, p_bear, p_low_vol, p_high_vol
    paper2_details.csv        probability, raw and smoothed regime label, target price
    latest_regime.json        today's answer in the team's JSON format
    refit_diagnostics.csv     quality of each monthly refit
    report.md                 everything in plain English: checks, tables, comparison
    chart_regimes.png, chart_clusters.png, chart_strategy.png
"""
import argparse
import json
import os
import numpy as np
import pandas as pd

import config
from data_prep import load_prices, check_data, align, to_returns
from features import rolling_features
from model import walk_forward, smooth, team_output
from backtest import regime_table, transition_matrix, stress_check, strategy_test
from plots import plot_regimes, plot_clusters, plot_strategies

NV = f"next_{config.HORIZON}d_volatility"


def run_model(rets, feature_set):
    feats, aux = rolling_features(rets, config.WINDOW, feature_set)
    p_high, diags, model = walk_forward(feats, aux, config.MIN_TRAIN, config.REFIT_EVERY,
                                        config.N_PCS, config.N_REGIMES)
    raw, regime = smooth(p_high, config.MIN_STAY)
    return {"feats": feats, "p_high": p_high, "diags": diags, "model": model,
            "raw": raw, "regime": regime}


def accuracy(labels, truth):
    t = truth.reindex(labels.index).dropna()
    pred = (labels.reindex(t.index) == "high_vol").astype(int)
    return {"accuracy": (pred == t).mean(),
            "missed stress (share of days)": ((pred == 0) & (t == 1)).mean(),
            "false alarms (share of days)": ((pred == 1) & (t == 0)).mean()}


def summary(res, close, truth=None):
    reg = res["regime"]
    tab = regime_table(reg, close, config.HORIZON, config.DAYS_PER_YEAR)
    row = {
        "high-vol share of days": (reg == "high_vol").mean(),
        "label switches per year": ((reg != reg.shift()).sum() - 1) / (len(reg) / config.DAYS_PER_YEAR),
        "next-20d volatility after high-vol days": tab[NV].get("high_vol", np.nan),
        "next-20d volatility after calm days": tab[NV].get("low_vol", np.nan),
        "PCA variance explained (avg)": res["diags"]["explained_variance"].mean(),
        "cluster silhouette (avg)": res["diags"]["silhouette"].mean(),
    }
    if truth is not None:
        row.update({f"{k} - smoothed": v for k, v in accuracy(reg, truth).items()})
        row["accuracy - raw (no smoothing)"] = accuracy(res["raw"], truth)["accuracy"]
    else:
        sc = stress_check(reg, config.STRESS_WINDOWS)
        row["high-vol share inside known crash periods"] = sc.mean() if len(sc) else np.nan
    return row, tab


def pct(x):
    return "" if pd.isna(x) else f"{x:.0%}"


def fmt_table(df, pct_cols=(), num_cols=()):
    d = df.copy()
    for c in d.columns:
        if c in pct_cols:
            d[c] = d[c].map(pct)
        elif c in num_cols:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return d.to_markdown() if hasattr(d, "to_markdown") else d.to_string()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/raw_prices.csv")
    ap.add_argument("--truth", default=None, help="simulation only: file with the true regimes")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--compare", action="store_true", help="also run the paper's feature set and compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---------------- data
    px_raw = load_prices(a.prices)
    data_report = check_data(px_raw, config.YIELD_COLS)
    px = align(px_raw[list(config.BASKET)], config.CALENDAR)
    rets = to_returns(px, config.YIELD_COLS)
    close = px[config.TARGET].rename(config.TARGET)
    truth = None
    if a.truth:
        truth = pd.read_csv(a.truth, parse_dates=["date"], index_col="date")["true_high_vol"]

    # ---------------- model
    main_set = config.FEATURE_SET
    res = run_model(rets, main_set)
    row_main, tab = summary(res, close, truth)

    other = None
    if a.compare:
        other_set = "paper" if main_set == "adapted" else "adapted"
        other = run_model(rets, other_set)
        row_other, _ = summary(other, close, truth)

    # ---------------- outputs: data contract + details
    team_output(res["p_high"]).to_csv(f"{a.out}/paper2_model_output.csv")
    details = pd.concat([res["p_high"], res["raw"], res["regime"],
                         close.reindex(res["p_high"].index).rename(f"{config.TARGET}_close")], axis=1)
    details.to_csv(f"{a.out}/paper2_details.csv")
    res["diags"].to_csv(f"{a.out}/refit_diagnostics.csv")

    # ---------------- backtest
    trans = transition_matrix(res["regime"])
    strat, curves = strategy_test(close, res["regime"], config.FAST_MA, config.SLOW_MA,
                                  config.COST, config.DAYS_PER_YEAR)
    stress = stress_check(res["regime"], config.STRESS_WINDOWS) if truth is None else None

    # ---------------- latest answer in the team's JSON format
    last = res["p_high"].index[-1]
    p = float(res["p_high"].iloc[-1])
    label = res["regime"].iloc[-1]
    latest = {
        "asset": config.TARGET,
        "date": str(last.date()),
        "model": f"paper2_covariance_clustering ({main_set})",
        "regime": "High volatility" if label == "high_vol" else "Low volatility",
        "confidence": round(p if label == "high_vol" else 1 - p, 3),
        "trend": {"bull": None, "bear": None},
        "volatility": {"low": round(1 - p, 3), "high": round(p, 3)},
        "backtest": {k: {c: (None if pd.isna(v) else round(float(v), 4)) for c, v in r.items()}
                     for k, r in tab.iterrows()},
    }
    with open(f"{a.out}/latest_regime.json", "w") as f:
        json.dump(latest, f, indent=2)

    # ---------------- charts
    source = "simulated data" if truth is not None else "real data"
    plot_regimes(close, res["p_high"], res["regime"], f"{a.out}/chart_regimes.png",
                 f"Paper 2 model on {source}: {config.TARGET} with high-volatility regimes", truth)
    plot_clusters(res["model"], f"{a.out}/chart_clusters.png")
    plot_strategies(curves, f"{a.out}/chart_strategy.png",
                    f"Trading test on {config.TARGET} ({source}, 0.1% cost per trade)")

    # ---------------- report
    L = [f"# Paper 2 model report ({source})", "",
         "## 1. Data check", "```", data_report, "```", "",
         f"Basket: {', '.join(config.BASKET)}. Target: {config.TARGET}. "
         f"Aligned on {config.CALENDAR} trading days: {len(px)} days "
         f"({px.index[0].date()} to {px.index[-1].date()}).", "",
         "## 2. Model setup",
         f"- Rolling covariance window: {config.WINDOW} business days, recomputed daily",
         f"- Feature set: {main_set} ({res['feats'].shape[1]} features per day), "
         f"reduced to {config.N_PCS} principal components",
         f"- Ward hierarchical clustering into {config.N_REGIMES} regimes, refitted every "
         f"{config.REFIT_EVERY} days on past data only (first fit after {config.MIN_TRAIN} days)",
         f"- Smoothing: a new regime must last {config.MIN_STAY} days before the label switches",
         f"- Labelled out of sample: {len(res['p_high'])} days "
         f"({res['p_high'].index[0].date()} to {last.date()})", "",
         "## 3. Latest answer",
         f"{config.TARGET} on {last.date()}: **{latest['regime']}**, "
         f"P(high volatility) = {p:.2f}. Full JSON in latest_regime.json.", "",
         "## 4. Key numbers", ""]
    key = pd.DataFrame({main_set: row_main})
    if other is not None:
        key[other_set] = row_other
    plain_rows = {"label switches per year", "cluster silhouette (avg)"}
    shown = key.copy().astype(object)
    for r in shown.index:
        shown.loc[r] = [("" if pd.isna(v) else (f"{v:.2f}" if r in plain_rows else f"{v:.0%}"))
                        for v in key.loc[r]]
    L += [shown.to_markdown(), "",
          "Silhouette runs from -1 to 1 (higher = clearer clusters). "
          "Volatility figures are annualised.", ""]
    L += ["## 5. Regimes vs. what the price did (backtest)", "",
          fmt_table(tab, pct_cols=["time_share", "ann_return", "ann_volatility", NV,
                                   "avg_drawdown_from_peak", "worst_drawdown_from_peak"],
                    num_cols=["avg_length_days"]), "",
          "## 6. Transition matrix (chance of tomorrow's regime, given today's)", "",
          trans.to_markdown(), ""]
    if stress is not None:
        L += ["## 7. Known crash periods: share of days labelled high volatility", "",
              stress.map(pct).to_frame("high-vol share").to_markdown(), ""]
    else:
        L += ["## 7. Simulation check", "",
              "Accuracy against the true regimes is in section 4.", ""]
    L += ["## 8. Trading test (the paper's idea: skip the trend rule in high volatility)", "",
          fmt_table(strat, pct_cols=["ann_return", "ann_volatility", "max_drawdown", "time_invested"],
                    num_cols=["sharpe", "trades_per_year"]), "",
          "## 9. Notes",
          "- p_bull / p_bear are empty: this model only separates calm from highly volatile markets.",
          "- Daily data and a 21-day window replace the paper's hourly data and monthly matrices.",
          "- scikit-learn's Ward linkage uses Euclidean distance; the paper used Manhattan.",
          "- Past results do not guarantee future results; this is a first prototype."]
    with open(f"{a.out}/report.md", "w") as f:
        f.write("\n".join(L) + "\n")

    print("\n".join(L[:2]))
    print(key.to_string())
    print(f"\nLatest: {config.TARGET} {last.date()} -> {latest['regime']} (P(high vol) = {p:.2f})")
    print(f"All outputs written to {a.out}/")


if __name__ == "__main__":
    main()
