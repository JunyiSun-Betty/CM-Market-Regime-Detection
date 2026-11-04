"""
STEP 3 (optional) - Compare variants of the Paper 2 model against a simple benchmark.

Run (after fetch_data.py):   python experiments.py

Why: the first real-data run showed that clustering on the full basket mostly
separates "crypto moves with stocks" periods from "crypto moves on its own" periods,
not calm vs. wild Bitcoin. Each variant tests one reason for that:
  A  as built: 22 features, "high" = the cluster where the whole basket is most volatile
  B  same, but "high" = the cluster where BTC itself is most volatile
  C  B with balanced inputs: 6 volatilities + 2 average correlations + BTC~ETH + factor share
  D  C with every input measured against its own past year (removes slow era changes)
  N  benchmark, no clustering: BTC's 21-day volatility above its own past median
All variants use past data only and are evaluated on the same dates.
Writes outputs/experiments.md
"""
import os
import numpy as np
import pandas as pd

import config
from data_prep import load_prices, align, to_returns
from features import rolling_features
from model import CovClusterModel, smooth
from backtest import regime_table, stress_check, strategy_test

CRYPTO = ["BTC", "ETH"]
TRAD = ["SPX", "GOLD", "US10Y", "VIX"]


def balanced(f):
    out = f[[c for c in f if c.startswith("logvol_")]].copy()
    cross = [f"corr_{a}~{b}" for a in CRYPTO for b in TRAD]
    trad = [f"corr_{a}~{b}" for i, a in enumerate(TRAD) for b in TRAD[i + 1:]]
    out["avg_corr_crypto_vs_markets"] = f[cross].abs().mean(axis=1)
    out["avg_corr_markets"] = f[trad].abs().mean(axis=1)
    out["corr_BTC~ETH"] = f["corr_BTC~ETH"]
    out["top_factor_share"] = f["top_factor_share"]
    return out


def relative(f, win=252):
    mean = f.rolling(win, min_periods=126).mean().shift(1)   # past year only
    std = f.rolling(win, min_periods=126).std().shift(1)
    return ((f - mean) / std).dropna()


def cluster_regimes(feats, name_by):
    parts = []
    for start in range(config.MIN_TRAIN, len(feats), config.REFIT_EVERY):
        tr, te = feats.iloc[:start], feats.iloc[start:start + config.REFIT_EVERY]
        m = CovClusterModel(config.N_PCS, config.N_REGIMES).fit(tr, name_by.reindex(tr.index))
        parts.append(pd.Series(m.predict_proba(te), index=te.index))
    return smooth(pd.concat(parts), config.MIN_STAY)[1]


def main(prices="data/raw_prices.csv"):
    os.makedirs("outputs", exist_ok=True)
    px = align(load_prices(prices)[list(config.BASKET)], config.CALENDAR)
    rets = to_returns(px, config.YIELD_COLS)
    close = px[config.TARGET].rename(config.TARGET)
    full, aux = rolling_features(rets, config.WINDOW, "adapted")

    regimes = {
        "A as built": cluster_regimes(full, aux["avg_logvol"]),
        "B name by BTC vol": cluster_regimes(full, aux["target_logvol"]),
        "C balanced inputs": cluster_regimes(balanced(full), aux["target_logvol"]),
        "D vs. own past year": cluster_regimes(relative(balanced(full)), aux["target_logvol"]),
    }
    vol = rets[config.TARGET].rolling(config.WINDOW).std()
    usual = vol.expanding(config.MIN_TRAIN).median().shift(1)
    regimes["N benchmark: BTC vol only"] = smooth((vol > usual).astype(float).dropna(), config.MIN_STAY)[1]

    start = max(r.index[0] for r in regimes.values())
    rows = {}
    for name, reg in regimes.items():
        reg = reg.loc[start:]
        tab = regime_table(reg, close, config.HORIZON, config.DAYS_PER_YEAR)
        st, _ = strategy_test(close, reg, config.FAST_MA, config.SLOW_MA, config.COST, config.DAYS_PER_YEAR)
        row = {
            "high-vol share of days": f"{(reg == 'high_vol').mean():.0%}",
            "next-20d BTC vol after HIGH": f"{tab['next_20d_volatility'].get('high_vol', np.nan):.0%}",
            "next-20d BTC vol after CALM": f"{tab['next_20d_volatility'].get('low_vol', np.nan):.0%}",
        }
        for k, v in stress_check(reg, config.STRESS_WINDOWS).items():
            row[f"crash: {k.split(' (')[0]}"] = f"{v:.0%}"
        row["label switches per year"] = f"{((reg != reg.shift()).sum() - 1) / (len(reg) / config.DAYS_PER_YEAR):.1f}"
        row["Sharpe: trend rule alone"] = f"{st.loc['Trend rule (30/100 MA)', 'sharpe']:.2f}"
        row["Sharpe: trend rule + filter"] = f"{st.loc['Trend rule + regime filter', 'sharpe']:.2f}"
        row["max drawdown: trend + filter"] = f"{st.loc['Trend rule + regime filter', 'max_drawdown']:.0%}"
        rows[name] = row
    table = pd.DataFrame(rows)

    corr = rets[config.TARGET].rolling(63).corr(rets["SPX"])
    yearly = corr.groupby(corr.index.year).mean().round(2)

    text = [f"# Paper 2 experiments ({start.date()} to {close.index[-1].date()})", "",
            table.to_markdown(), "",
            "Good model = clearly higher next-20d volatility after HIGH than after CALM, "
            "most crash days labelled HIGH, few switches.", "",
            "## BTC vs. S&P 500 correlation (63-day, yearly average)", "",
            yearly.to_frame("corr").to_markdown(), "",
            "Note: variants B-D were designed after seeing the first real run, so treat "
            "them as exploration, not as a final test."]
    with open("outputs/experiments.md", "w") as f:
        f.write("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw_prices.csv")
