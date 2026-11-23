"""Step 2: build LSTM inputs from the step-1 regime labels.

- Inputs: log_return, volatility_30d_annualized, drawdown_from_30d_peak only
  (no regime history as input).
- Window: features on days t-2, t-1, t  ->  target = regime on day t+1.
- A sample belongs to the split of its TARGET day (train <= 2021-12-31,
  validation 2022-2023, test 2024+). Windows may use earlier days' features,
  which are already known at prediction time.
- Standardization: mean and std from training feature rows only.
- Loss weights (paper Eq. 2), computed on training targets only:
      w = (1 - lam) * T / n_class(g)
          + lam * S / n_switch_class(g) * I{target is a switch}
  T = training samples, n_class(g) = training samples whose target is g,
  S = training switch samples, n_switch_class(g) = switch samples whose target is g.
  A switch sample is one whose target regime differs from the regime on day t.
  Note: the printed Eq. 2 omits the indicator on t; without it every day of a
  class would get the same weight and switches would not be emphasised, which
  contradicts the paper's text, so the indicator is included here.

Usage (from the repo root, after label_regimes.py):
    python regime_lstm/build_dataset.py
Defaults: --labels regime_lstm/outputs/btc_regimes.csv, --out regime_lstm/outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["log_return", "volatility_30d_annualized", "drawdown_from_30d_peak"]
CLASSES = ["Low", "High", "Distress"]          # class index 0, 1, 2
WINDOW = 3
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
HERE = Path(__file__).resolve().parent


def eq2_weights(y: np.ndarray, is_switch: np.ndarray, lam: float) -> np.ndarray:
    T = len(y)
    S = is_switch.sum()
    w = np.zeros(T)
    for g in range(len(CLASSES)):
        in_g = y == g
        n_g = in_g.sum()
        if n_g == 0:
            continue
        w[in_g] += (1 - lam) * T / n_g
        n_sw_g = (in_g & is_switch).sum()
        if n_sw_g > 0:
            w[in_g & is_switch] += lam * S / n_sw_g
    return w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(HERE / "outputs" / "btc_regimes.csv"))
    parser.add_argument("--out", default=str(HERE / "outputs"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels, parse_dates=["date"])
    label = df["regime"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    train_rows = df["split"] == "train"
    mean = df.loc[train_rows, FEATURES].mean()
    std = df.loc[train_rows, FEATURES].std(ddof=0)
    X_all = ((df[FEATURES] - mean) / std).to_numpy(dtype=np.float32)

    # sample i: window ends at day t = i + WINDOW - 1, target day t + 1
    X, y, prev, tdate, split = [], [], [], [], []
    for t in range(WINDOW - 1, len(df) - 1):
        X.append(X_all[t - WINDOW + 1: t + 1])
        y.append(label[t + 1])
        prev.append(label[t])
        tdate.append(df["date"].iloc[t + 1])
        split.append(df["split"].iloc[t + 1])
    X = np.stack(X)
    y = np.array(y)
    prev = np.array(prev)
    tdate = np.array(tdate, dtype="datetime64[D]")
    split = np.array(split)
    is_switch = y != prev

    tr = split == "train"
    weights = {lam: eq2_weights(y[tr], is_switch[tr], lam) for lam in LAMBDAS}

    arrays = {}
    for name in ["train", "validation", "test"]:
        m = split == name
        arrays[f"X_{name}"] = X[m]
        arrays[f"y_{name}"] = y[m]
        arrays[f"prev_{name}"] = prev[m]          # regime on day t, for baselines
        arrays[f"date_{name}"] = tdate[m].astype(str)
    for lam, w in weights.items():
        arrays[f"w_train_lambda_{lam}"] = w.astype(np.float32)
    np.savez(out / "lstm_dataset.npz", **arrays)
    (out / "scaler.json").write_text(json.dumps(
        {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist()}, indent=2))

    # Summary
    lines = [f"Window {WINDOW} days x {len(FEATURES)} features; target = next-day regime", ""]
    rows = []
    for name in ["train", "validation", "test"]:
        m = split == name
        counts = [(y[m] == g).sum() for g in range(3)]
        rows.append([name, m.sum(), *counts, is_switch[m].sum(),
                     str(tdate[m][0]), str(tdate[m][-1])])
    lines.append(pd.DataFrame(rows, columns=["split", "samples", *CLASSES, "switches",
                                              "first target", "last target"]).to_string(index=False))
    lines += ["", "Scaler (training rows):",
              pd.DataFrame({"mean": mean, "std": std}).to_string(), ""]

    ytr, swtr = y[tr], is_switch[tr]
    wrows = []
    for lam, w in weights.items():
        r = {"lambda": lam}
        for g, c in enumerate(CLASSES):
            r[f"{c} no-switch"] = w[(ytr == g) & ~swtr].mean() if ((ytr == g) & ~swtr).any() else np.nan
            r[f"{c} switch"] = w[(ytr == g) & swtr].mean() if ((ytr == g) & swtr).any() else np.nan
        r["share of total weight on switch days"] = w[swtr].sum() / w.sum()
        wrows.append(r)
    lines += ["Eq. 2 weights per training sample (mean by target class and switch status):",
              pd.DataFrame(wrows).round(3).to_string(index=False)]
    summary = "\n".join(lines)
    (out / "dataset_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
