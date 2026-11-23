# LSTM regime forecasting (BTC)

Following Agakishiev, Härdle, Becker & Zuo (2025), *Regime switching forecasting for
cryptocurrencies*, Digital Finance 7:107–131 (`literature/` folder). This folder
covers step 1 of the paper (regime forecasting); the reinforcement-learning trader is
out of scope.

## Run

From the repo root (`CM-Market-Regime-Detection/`), with pandas, numpy and matplotlib installed (step 3 also needs torch and scikit-learn):

```powershell
python regime_lstm/label_regimes.py    # step 1: regime labels + plot
python regime_lstm/build_dataset.py    # step 2: LSTM inputs + loss weights
python regime_lstm/train_lstm.py       # step 3: train 5 lambdas x 5 seeds (~6 min on CPU, needs torch + scikit-learn)
```

Input: `data_preprocessing/data/btc_daily_date_fixed_filled.csv`. All outputs go to `regime_lstm/outputs/`.

## Status

| Step | Script | Status |
| --- | --- | --- |
| 1. Label regimes | `label_regimes.py` | Done |
| 2. Build LSTM inputs and Eq. 2 weights | `build_dataset.py` | Done |
| 3. Train LSTM for each λ, evaluate on validation | `train_lstm.py` | Done (3-day window). LSTM is below the persistence baseline; see results |
| 4. Final test and LIME | — | Not started |

## Decisions so far

**Data**
- Rows from 2014-04-28 to 2026-10-31 (4,570 days). Before 2014-04-28 the 30-day features have blanks.
- Splits: train to 2021-12-31, validation 2022–2023, test 2024 onward.

**Regime labels (step 1)**
- p_σ = percentile rank of `volatility_30d_annualized`; p_r = percentile rank of the 1-day `log_return`.
- Percentile cutoffs come from the training period (2014-04-28 to 2021-12-31) only and are applied to every day.
- p_d = (p_σ + 1 − p_r) / 2.
- Rules (paper Table 2), checked in this order each day:
  1. p_d > 0.85 → Distress
  2. p_σ < 0.3 → Low
  3. 0.4 < p_σ ≤ 0.6 → High
  4. otherwise (buffer 1: 0.3 ≤ p_σ ≤ 0.4; buffer 2: p_σ > 0.6 and p_d ≤ 0.85) → keep the previous day's regime.
  A buffer zone is not a regime; the label stays until another regime is properly reached.
  p_d > 0.85 is only possible when p_σ > 0.7, so Distress never conflicts with Low or High.
- If the very first day is in a buffer, it gets Low when p_σ ≤ 0.5, else High.

**LSTM inputs (step 2)**
- Features: `log_return`, `volatility_30d_annualized`, `drawdown_from_30d_peak` only (no regime history as input).
- Window: days t−2, t−1, t → target = regime on day t+1 (paper Table 3 / Fig. 2 / Table 8).
- A sample belongs to the split of its target day.
- Standardization uses the mean and std of training rows only (`outputs/scaler.json`).
- Eq. 2 loss weights, computed on training targets for λ ∈ {0, 0.25, 0.5, 0.75, 1}:
  `w = (1−λ)·T/n_class + λ·S/n_switch_class · I{switch}`.
  The printed Eq. 2 has no switch indicator on the second term; without it switch days get no extra weight,
  which contradicts the paper's text, so the indicator is included.

## Results so far

| Split | Days | Low | High | Distress | Switches |
| --- | --- | --- | --- | --- | --- |
| Train | 2,805 | 1,090 | 933 | 782 | 78 |
| Validation | 730 | 480 | 184 | 66 | 17 |
| Test | 1,035 | 845 | 190 | 0 | 12 |

- Typical regime run lasts 24–35 days; only 4 runs last a single day.
- Distress is persistent because buffer 2 keeps it while volatility stays high (e.g. much of 2017).
  This follows the paper's rule as written.
- The test period has no Distress days: BTC volatility is lower since 2024 (median 0.43 vs 0.62 in training),
  and the cutoffs are fixed on 2014–2021.
- At λ = 0.5 switch days get roughly 2× the weight of normal days (5% of total weight);
  at λ = 1 non-switch days get zero weight, so only 78 training samples count.

## Step 3 results (validation 2022–2023, 730 days, 17 switches)

Model: LSTM(18, ReLU cell) → Dense(18, ReLU) → Dropout 0.4 → Dense(3, softmax); Adam lr 3e-4,
batch 32, 100 epochs, Eq. 2 weighted cross-entropy; best epoch by validation macro-F1; 5 seeds per λ.

| Model | Macro-F1 | Accuracy | Switches caught | Days with a false switch |
| --- | --- | --- | --- | --- |
| Persistence (tomorrow = today) | 0.966 | 0.977 | 0 of 17 | 0% |
| LSTM λ = 0 | 0.865 ± 0.004 | 0.911 | 39% | 7.6% |
| LSTM λ = 0.25 | 0.863 ± 0.003 | 0.911 | 44% | 7.8% |
| LSTM λ = 0.5 | 0.861 ± 0.005 | 0.909 | 40% | 7.9% |
| LSTM λ = 0.75 | 0.855 ± 0.008 | 0.904 | 41% | 8.4% |
| LSTM λ = 1 | 0.475 ± 0.181 | 0.514 | 57% | 48.7% |

- With 3 features only, the LSTM cannot see the current regime. Because of the buffer zones the label
  depends on history, so the same feature values can belong to different regimes. The most common
  error is predicting Distress on Low days (28 of 480).
- λ between 0 and 0.75 makes little difference; λ = 1 trains on the 78 switch samples only and is unstable.
- The test set has not been used.

## Outputs

| File | Contents |
| --- | --- |
| `outputs/btc_regimes.csv` | Date, close, 3 features, p_σ, p_r, p_d, regime, switch flag, split |
| `outputs/btc_regimes.png` | BTC price colored by regime, split boundaries dashed |
| `outputs/regime_summary.txt` | Regime counts, switches and run lengths per split |
| `outputs/lstm_dataset.npz` | `X_*` (n×3×3), `y_*` (0 Low, 1 High, 2 Distress), `prev_*` (regime on day t), `date_*` (target dates), `w_train_lambda_*` |
| `outputs/scaler.json` | Training mean and std per feature |
| `outputs/dataset_summary.txt` | Sample counts and weight table |
| `outputs/step3_training/validation_summary.txt` | Mean/std per λ, baseline, confusion matrices |
| `outputs/step3_training/runs.csv` | Best epoch and metrics for each of the 25 runs |
| `outputs/step3_training/training_curves.csv` | Loss and validation metrics per epoch per run |
| `outputs/step3_training/models/*.pt` | Best-seed weights per λ (PyTorch state_dict) |
| `outputs/step3_training/config.json` | Hyperparameters |
