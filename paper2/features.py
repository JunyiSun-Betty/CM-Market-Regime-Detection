"""
Turn daily returns into one row of "covariance features" per day.

The paper builds monthly realized covariance matrices from hourly data.
We only have daily data, so each day we take the covariance of the last
WINDOW business days instead. Only past days are used, so nothing leaks
from the future.

Two feature sets:
  "adapted": log volatility of each asset + every pairwise correlation
             + the share of movement driven by one common factor.
             (covariance = volatility x correlation x volatility, so this keeps
             all the covariance information in a scale-friendly form)
  "paper":   correlation distances sqrt(2 * (1 - correlation)) for every pair,
             which is how the paper prepares data for clustering.
"""
import numpy as np
import pandas as pd


def rolling_features(rets, window, feature_set="adapted"):
    cols = list(rets.columns)
    n = len(cols)
    iu = np.triu_indices(n, k=1)
    pair_names = [f"{cols[i]}~{cols[j]}" for i, j in zip(*iu)]
    values = rets.to_numpy()

    rows, aux, dates = [], [], []
    for t in range(window - 1, len(rets)):
        w = values[t - window + 1:t + 1]
        cov = np.cov(w.T)
        sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = np.clip(cov / np.outer(sd, sd), -1, 1)
        if feature_set == "adapted":
            top_share = np.linalg.eigvalsh(corr)[-1] / n
            rows.append(list(np.log(sd)) + list(corr[iu]) + [top_share])
        elif feature_set == "paper":
            rows.append(list(np.sqrt(2 * (1 - corr[iu]))))
        else:
            raise ValueError("feature_set must be 'adapted' or 'paper'")
        # kept aside to name the regimes (not used for clustering in "paper" mode)
        aux.append([np.log(sd).mean(), np.log(sd[0])])
        dates.append(rets.index[t])

    if feature_set == "adapted":
        names = [f"logvol_{c}" for c in cols] + [f"corr_{p}" for p in pair_names] + ["top_factor_share"]
    else:
        names = [f"dist_{p}" for p in pair_names]
    feats = pd.DataFrame(rows, index=dates, columns=names)
    aux = pd.DataFrame(aux, index=dates, columns=["avg_logvol", "target_logvol"])
    feats.index.name = aux.index.name = "date"
    return feats, aux
