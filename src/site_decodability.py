"""Site-decodability diagnostic (RQ0(b)) — deliberately separate from the
disease classifier.

Question: how strongly does the frozen FC representation encode the acquisition
site? This is a leakage/heterogeneity diagnostic, so the *target* here is the
site label. Site is never a predictor of diagnosis anywhere in this project;
this module is the one place site appears as `y`, and it shares no fitted object
with the diagnostic pipeline.

Protocol: stratified K-fold over acquisitions (each acquisition appears in
exactly one test fold), StandardScaler + PCA fit on the training fold only,
multinomial L2 logistic regression with `C` chosen by an inner stratified CV.

Primary metric: balanced accuracy (spec §18; spec §8 permits BAcc or macro-F1).
Macro-F1 is reported alongside as a secondary number. The 1/23 chance level is
an interpretive reference only.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

N_SPLITS = 5
INNER_SPLITS = 3
PCA_DIM = 200
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _fit_pre(X_tr: np.ndarray, seed: int):
    sc = StandardScaler().fit(X_tr)
    pca = PCA(n_components=min(PCA_DIM, min(X_tr.shape) - 1),
              svd_solver="randomized", random_state=seed).fit(sc.transform(X_tr))
    return lambda A: pca.transform(sc.transform(A))


def run_site_decodability(X: np.ndarray, sites: np.ndarray, seed: int) -> dict:
    """Cross-validated multi-class site decoding. Returns metrics + predictions."""
    y = np.asarray(sites)
    pred = np.empty(len(y), dtype=object)
    chosen_c = []

    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        assert not set(tr) & set(te)
        proj = _fit_pre(X[tr], seed)
        Ztr = proj(X[tr])

        # inner selection of C, training-fold only
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True,
                                random_state=seed)
        inner_scores = {c: [] for c in C_GRID}
        for itr, iva in inner.split(Ztr, y[tr]):
            # preprocessing is refit inside the inner split as well
            iproj = _fit_pre(X[tr][itr], seed)
            Zitr, Ziva = iproj(X[tr][itr]), iproj(X[tr][iva])
            for c in C_GRID:
                clf = LogisticRegression(C=c, max_iter=1000, random_state=seed)
                clf.fit(Zitr, y[tr][itr])
                inner_scores[c].append(
                    balanced_accuracy_score(y[tr][iva], clf.predict(Ziva)))
        best = max(C_GRID, key=lambda c: float(np.mean(inner_scores[c])))
        chosen_c.append(best)

        clf = LogisticRegression(C=best, max_iter=1000, random_state=seed)
        clf.fit(Ztr, y[tr])
        pred[te] = clf.predict(proj(X[te]))

    pred = pred.astype(str)
    n_sites = len(np.unique(y))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float((pred == y).mean()),
        "n_sites": int(n_sites),
        "chance_reference": float(1.0 / n_sites),
        "n_acquisitions": int(len(y)),
        "selected_C_per_fold": chosen_c,
        "per_site_recall": {
            str(s): float((pred[y == s] == s).mean()) for s in np.unique(y)
        },
        "predictions": pred,
    }
