"""
The Paper 2 model: PCA + hierarchical (Ward) clustering into calm / highly volatile.

fit():            learn on past data only
predict_proba():  give each new day a probability of the high-volatility regime
walk_forward():   refit every month and label the next month (no look-ahead)
smooth():         a new regime must last MIN_STAY days before the label switches
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


class CovClusterModel:
    def __init__(self, n_pcs=3, n_regimes=2):
        if n_regimes != 2:
            raise ValueError("This version supports 2 regimes (calm / highly volatile), as in the paper.")
        self.n_pcs, self.n_regimes = n_pcs, n_regimes

    def fit(self, X, avg_vol):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.pca = PCA(n_components=self.n_pcs).fit(Z)
        Z = self.pca.transform(Z)
        # Ward linkage (the paper's choice). scikit-learn only allows Euclidean
        # distance with Ward; the paper used Manhattan distance.
        labels = AgglomerativeClustering(n_clusters=self.n_regimes, linkage="ward").fit_predict(Z)
        self.centres = np.vstack([Z[labels == k].mean(axis=0) for k in range(self.n_regimes)])
        vol = np.array([np.asarray(avg_vol)[labels == k].mean() for k in range(self.n_regimes)])
        self.high = int(np.argmax(vol))          # the more volatile cluster = "high vol"
        own = np.linalg.norm(Z - self.centres[labels], axis=1)
        self.scale = max(float(own.mean()), 1e-9)  # typical distance, used to turn distances into probabilities
        self.diag = {
            "explained_variance": float(self.pca.explained_variance_ratio_.sum()),
            "silhouette": float(silhouette_score(Z, labels)) if len(set(labels)) > 1 else np.nan,
            "high_share_train": float((labels == self.high).mean()),
        }
        self.Z_train, self.labels_train = Z, labels
        return self

    def predict_proba(self, X):
        Z = self.pca.transform(self.scaler.transform(X))
        d = np.linalg.norm(Z[:, None, :] - self.centres[None, :, :], axis=2) / self.scale
        w = np.exp(-(d - d.min(axis=1, keepdims=True)))   # closer centre -> higher probability
        return w[:, self.high] / w.sum(axis=1)


def walk_forward(feats, aux, min_train, refit_every, n_pcs=3, n_regimes=2):
    """Fit on all data before each month, label that month, move on."""
    parts, diags, model = [], [], None
    for start in range(min_train, len(feats), refit_every):
        train, test = feats.iloc[:start], feats.iloc[start:start + refit_every]
        model = CovClusterModel(n_pcs, n_regimes).fit(train, aux["avg_logvol"].iloc[:start])
        parts.append(pd.Series(model.predict_proba(test), index=test.index))
        diags.append({"refit_date": test.index[0], "train_days": start, **model.diag})
    if not parts:
        raise ValueError("Not enough data: need more than MIN_TRAIN days of features.")
    p_high = pd.concat(parts).rename("p_high_vol")
    p_high.index.name = "date"
    return p_high, pd.DataFrame(diags).set_index("refit_date"), model


def smooth(p_high, min_stay):
    """Raw label = high vol if p > 0.5. The label only switches after the new
    state has held for `min_stay` days in a row (uses past days only)."""
    raw = (p_high.to_numpy() > 0.5).astype(int)
    out = np.empty_like(raw)
    state, run = raw[0], 0
    for i, r in enumerate(raw):
        if r != state:
            run += 1
            if run >= min_stay:
                state, run = r, 0
        else:
            run = 0
        out[i] = state
    names = np.where(out == 1, "high_vol", "low_vol")
    raw_names = np.where(raw == 1, "high_vol", "low_vol")
    return pd.Series(raw_names, index=p_high.index, name="regime_raw"), \
        pd.Series(names, index=p_high.index, name="regime")


def team_output(p_high):
    """The team's data contract: date, p_bull, p_bear, p_low_vol, p_high_vol.
    This model has no direction, so p_bull / p_bear stay empty."""
    out = pd.DataFrame({"p_bull": np.nan, "p_bear": np.nan,
                        "p_low_vol": 1 - p_high, "p_high_vol": p_high}, index=p_high.index)
    out.index.name = "date"
    return out
