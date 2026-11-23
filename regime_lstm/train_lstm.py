"""Step 3: train the regime-forecasting LSTM (paper Fig. 2 / Table 3) for each lambda.

Model:   LSTM(18 units, ReLU cell activation) on 3 days x 3 features
         -> Dense(18, ReLU) -> Dropout(0.4) -> Dense(3, softmax)
Train:   Adam, lr 3e-4, batch 32, 100 epochs, cross-entropy weighted per sample by
         the Eq. 2 weights from build_dataset.py.
Runs:    lambda in {0, 0.25, 0.5, 0.75, 1} x 5 seeds. Each run keeps the epoch with
         the best validation macro-F1. The test set is NOT used here.

Validation metrics
- macro_f1:       macro-averaged F1 over Low / High / Distress, all validation days
- accuracy
- switch_recall:  share of true switch days (target regime != today's regime)
                  where the model predicted the new regime correctly
- false_switch:   share of non-switch days where the model predicted a change
- baseline "persistence" = predict that tomorrow's regime equals today's.

Keras' LSTM(activation="relu") replaces the cell's tanh with ReLU; PyTorch's nn.LSTM
always uses tanh, so the cell is written out here to match the paper.

Usage (from the repo root, after build_dataset.py):
    python regime_lstm/train_lstm.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

HERE = Path(__file__).resolve().parent
CLASSES = ["Low", "High", "Distress"]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]
HIDDEN = 18
DROPOUT = 0.4
LR = 3e-4
BATCH = 32
EPOCHS = 100


class ReLULSTM(nn.Module):
    """Single-layer LSTM whose candidate and output activations are ReLU (Keras style)."""

    def __init__(self, n_in: int, n_hidden: int):
        super().__init__()
        self.n_hidden = n_hidden
        self.x2g = nn.Linear(n_in, 4 * n_hidden)
        self.h2g = nn.Linear(n_hidden, 4 * n_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = x.new_zeros(b, self.n_hidden)
        c = x.new_zeros(b, self.n_hidden)
        for t in range(x.shape[1]):
            i, f, g, o = (self.x2g(x[:, t]) + self.h2g(h)).chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f + 1.0), torch.sigmoid(o)  # forget bias 1, as Keras
            c = f * c + i * torch.relu(g)
            h = o * torch.relu(c)
        return h  # last time step only


class RegimeLSTM(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = ReLULSTM(n_features, HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.lstm(x))  # logits; softmax is applied in the loss / at predict


def metrics(y: np.ndarray, pred: np.ndarray, prev: np.ndarray) -> dict:
    sw = y != prev
    return {
        "macro_f1": f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "switch_recall": float((pred[sw] == y[sw]).mean()) if sw.any() else float("nan"),
        "false_switch": float((pred[~sw] != prev[~sw]).mean()),
    }


def predict_proba(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(X), dim=1).numpy()


def train_one(data, lam: float, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr = torch.tensor(data["X_train"])
    ytr = torch.tensor(data["y_train"], dtype=torch.long)
    wtr = torch.tensor(data[f"w_train_lambda_{lam}"])
    Xva = torch.tensor(data["X_validation"])
    yva, prev_va = data["y_validation"], data["prev_validation"]

    model = RegimeLSTM(Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss(reduction="none")
    rng = np.random.default_rng(seed)
    history, best, best_state = [], None, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(Xtr))
        total = 0.0
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            loss = (ce(model(Xtr[idx]), ytr[idx]) * wtr[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        pred = predict_proba(model, Xva).argmax(1)
        m = metrics(yva, pred, prev_va)
        history.append({"epoch": epoch, "train_loss": total / len(Xtr), **m})
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, **m}
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, pd.DataFrame(history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(HERE / "outputs" / "lstm_dataset.npz"))
    parser.add_argument("--out", default=str(HERE / "outputs" / "step3_training"))
    args = parser.parse_args()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    data = dict(np.load(args.data))
    yva, prev_va = data["y_validation"], data["prev_validation"]
    Xva = torch.tensor(data["X_validation"])

    rows, curves, best_per_lambda = [], [], {}
    for lam in LAMBDAS:
        for seed in SEEDS:
            model, best, hist = train_one(data, lam, seed)
            rows.append({"lambda": lam, "seed": seed, **best})
            curves.append(hist.assign(**{"lambda": lam, "seed": seed}))
            print(f"lambda={lam:<4} seed={seed} best epoch {best['epoch']:>3}  "
                  f"macroF1 {best['macro_f1']:.3f}  switch recall {best['switch_recall']:.3f}  "
                  f"false switch {best['false_switch']:.3f}", flush=True)
            if lam not in best_per_lambda or best["macro_f1"] > best_per_lambda[lam][1]["macro_f1"]:
                best_per_lambda[lam] = (seed, best, model)

    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    pd.concat(curves).to_csv(out / "training_curves.csv", index=False)

    agg = runs.groupby("lambda")[["macro_f1", "accuracy", "switch_recall", "false_switch", "epoch"]]
    summary = agg.mean().round(3).add_suffix("_mean").join(agg.std().round(3).add_suffix("_std"))
    base = metrics(yva, prev_va, prev_va)

    lines = ["Validation results (mean and std over 5 seeds, best epoch per run)", "",
             summary.to_string(), "",
             "Persistence baseline (tomorrow = today): " +
             ", ".join(f"{k} {v:.3f}" for k, v in base.items()), ""]

    for lam, (seed, best, model) in best_per_lambda.items():
        pred = predict_proba(model, Xva).argmax(1)
        cm = pd.DataFrame(confusion_matrix(yva, pred, labels=[0, 1, 2]),
                          index=[f"true {c}" for c in CLASSES],
                          columns=[f"pred {c}" for c in CLASSES])
        lines += [f"Best seed for lambda={lam}: seed {seed}, epoch {best['epoch']}, "
                  f"macro-F1 {best['macro_f1']:.3f}", cm.to_string(), ""]
        torch.save(model.state_dict(), out / "models" / f"lstm_lambda_{lam}_seed_{seed}.pt")

    cm = pd.DataFrame(confusion_matrix(yva, prev_va, labels=[0, 1, 2]),
                      index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES])
    lines += ["Persistence baseline confusion matrix", cm.to_string()]
    text = "\n".join(lines)
    (out / "validation_summary.txt").write_text(text + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps(
        {"hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "batch": BATCH, "epochs": EPOCHS,
         "lambdas": LAMBDAS, "seeds": SEEDS, "selection": "best validation macro-F1"}, indent=2))
    print(text)


if __name__ == "__main__":
    main()
