"""
Backtest: how do the regimes compare to what the target coin's price actually did?

regime_table():     per regime - time share, length, return, volatility, drawdown,
                    and the volatility of the NEXT 20 days (the honest check)
transition_matrix(): chance of moving from one regime to another the next day
stress_check():     share of high-vol days inside well-known crash periods
strategy_test():    the paper's simple trading rule, with and without the regime filter
"""
import numpy as np
import pandas as pd


def _spells(labels):
    """Lengths (in days) of each uninterrupted run of a regime."""
    block = (labels != labels.shift()).cumsum()
    runs = labels.groupby(block).agg(["first", "size"])
    return runs.groupby("first")["size"].mean()


def regime_table(regime, close, horizon=20, days_per_year=252):
    r = np.log(close).diff().reindex(regime.index)
    fwd_vol = (np.log(close).diff()[::-1].rolling(horizon).std()[::-1].shift(-1)
               * np.sqrt(days_per_year)).reindex(regime.index)
    drawdown = (close / close.cummax() - 1).reindex(regime.index)
    df = pd.DataFrame({"regime": regime, "r": r, "fwd_vol": fwd_vol, "dd": drawdown})
    spell = _spells(regime)
    rows = {}
    for k, g in df.groupby("regime"):
        rows[k] = {
            "time_share": len(g) / len(df),
            "avg_length_days": spell.get(k, np.nan),
            "ann_return": g["r"].mean() * days_per_year,
            "ann_volatility": g["r"].std() * np.sqrt(days_per_year),
            f"next_{horizon}d_volatility": g["fwd_vol"].mean(),
            "avg_drawdown_from_peak": g["dd"].mean(),
            "worst_drawdown_from_peak": g["dd"].min(),
        }
    return pd.DataFrame(rows).T


def transition_matrix(regime):
    return pd.crosstab(regime, regime.shift(-1), normalize="index").round(3)


def stress_check(regime, windows):
    out = {}
    for name, (a, b) in windows.items():
        s = regime.loc[a:b]
        if len(s):
            out[name] = (s == "high_vol").mean()
    return pd.Series(out, dtype=float)


def _metrics(r, position, days_per_year):
    curve = (1 + r).cumprod()
    years = len(r) / days_per_year
    return {
        "ann_return": curve.iloc[-1] ** (1 / years) - 1,
        "ann_volatility": r.std() * np.sqrt(days_per_year),
        "sharpe": r.mean() / r.std() * np.sqrt(days_per_year) if r.std() > 0 else np.nan,
        "max_drawdown": (curve / curve.cummax() - 1).min(),
        "time_invested": position.mean(),
        "trades_per_year": position.diff().abs().sum() / years,
    }


def strategy_test(close, regime, fast=30, slow=100, cost=0.001, days_per_year=252):
    """The paper's idea: take a simple trend rule (buy when the 30-day average is
    above the 100-day average) and skip it when the model says high volatility.
    Positions are taken the NEXT day, and every trade pays `cost`."""
    ret = close.pct_change()
    ma_signal = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)
    calm = (regime == "low_vol").astype(float).reindex(close.index)
    signals = {
        "Buy and hold": pd.Series(1.0, index=close.index),
        "Trend rule (30/100 MA)": ma_signal,
        "Trend rule + regime filter": ma_signal * calm,
        "Hold only in calm regime": calm,
    }
    idx = regime.index
    results, curves = {}, {}
    for name, sig in signals.items():
        pos = sig.shift(1).reindex(idx).fillna(0)        # act the next day
        r = ret.reindex(idx).fillna(0)
        net = pos * r - pos.diff().abs().fillna(0) * cost
        results[name] = _metrics(net, pos, days_per_year)
        curves[name] = (1 + net).cumprod()
    return pd.DataFrame(results).T, pd.DataFrame(curves)
