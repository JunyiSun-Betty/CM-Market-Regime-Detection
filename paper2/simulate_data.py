"""
OPTIONAL - Make a simulated basket with KNOWN calm / stress periods,
in exactly the same file format as fetch_data.py. This is the same kind of
test the paper uses: if the model cannot find regimes we planted, we should
not trust it on real data.

Run:   python simulate_data.py
Creates data/sim_prices.csv and data/sim_truth.csv
"""
import os
import numpy as np
import pandas as pd
import config


def simulate(n_days=2330, seed=7):
    rng = np.random.default_rng(seed)
    # a slow-moving hidden driver decides calm vs. stress (as in the paper's simulation)
    b = np.zeros(n_days)
    for t in range(1, n_days):
        b[t] = 0.985 * b[t - 1] + rng.normal(0, 0.1)
    b = (b - b.min()) / (b.max() - b.min())
    stress = (b > 0.7).astype(int)

    names = list(config.BASKET)                       # BTC, ETH, SPX, GOLD, US10Y, VIX
    calm_vol = np.array([0.035, 0.045, 0.009, 0.008, 0.05, 0.05])
    sign = np.array([1, 1, 1, -1, -1, -1])            # gold, yields, VIX move against risk assets in stress
    drift_calm = np.array([0.0012, 0.0012, 0.0004, 0.0001, 0.0, 0.0])
    drift_stress = np.array([-0.002, -0.0025, -0.001, 0.0003, 0.0, 0.0])

    r = np.zeros((n_days, len(names)))
    for t in range(n_days):
        s = stress[t]
        vol = calm_vol * (2.5 if s else 1.0)          # volatility jumps in stress
        rho = 0.65 if s else 0.15                     # and assets move together more
        f, e = rng.normal(), rng.normal(size=len(names))
        r[t] = (drift_stress if s else drift_calm) + vol * (np.sqrt(rho) * sign * f + np.sqrt(1 - rho) * e)

    idx = pd.bdate_range(config.START, periods=n_days, name="date")
    px = pd.DataFrame(index=idx)
    for i, n in enumerate(names):
        if n in config.YIELD_COLS:
            px[n] = np.clip(2.0 + np.cumsum(r[:, i]), 0.1, None)   # a yield level in %
        else:
            px[n] = 100 * np.exp(np.cumsum(r[:, i]))
    return px, pd.Series(stress, index=idx, name="true_high_vol")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    px, truth = simulate()
    px.to_csv("data/sim_prices.csv")
    truth.to_csv("data/sim_truth.csv")
    print(f"Saved data/sim_prices.csv ({len(px)} days, {truth.mean():.0%} of days in stress) "
          "and data/sim_truth.csv")
