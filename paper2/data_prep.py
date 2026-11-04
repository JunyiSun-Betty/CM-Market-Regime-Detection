"""
Load, check and align the price data, then turn prices into daily changes.
"""
import numpy as np
import pandas as pd


def load_prices(path):
    px = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    return px[~px.index.duplicated(keep="last")]


def check_data(px, yield_cols=()):
    """Plain-English data checks: coverage, gaps and suspicious jumps."""
    lines = [f"Rows: {len(px)}  ({px.index[0].date()} to {px.index[-1].date()})",
             f"{'asset':7s} {'first date':11s} {'missing':>8s} {'biggest daily move':>19s}"]
    for c in px:
        s = px[c].dropna()
        moves = s.diff() if c in yield_cols else np.log(s).diff()
        big = moves.abs().max()
        unit = " pp" if c in yield_cols else "%"
        big_txt = f"{big:.2f}{unit}" if c in yield_cols else f"{100 * (np.exp(big) - 1):.0f}{unit}"
        lines.append(f"{c:7s} {str(s.index[0].date()):11s} "
                     f"{px[c].isna().mean():8.1%} {big_txt:>19s}")
    return "\n".join(lines)


def align(px, calendar):
    """Keep only the days the calendar market trades, fill short gaps (max 3 days),
    and start on the first day every asset has data."""
    if calendar in px:
        px = px[px[calendar].notna()]
    px = px.ffill(limit=3)
    return px.loc[px.dropna().index[0]:].dropna()


def to_returns(px, yield_cols=()):
    """Log returns for prices, change in level for yields."""
    out = pd.DataFrame(index=px.index)
    for c in px:
        out[c] = px[c].diff() if c in yield_cols else np.log(px[c]).diff()
    return out.dropna()
