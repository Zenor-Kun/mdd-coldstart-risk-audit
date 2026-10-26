"""Frozen cohort + frozen FC feature loading for Gate 1.

Entry point for `X`, `y` and the split key `site`. Two hard rules, both from
`docs/DATA_CONTRACT_v1.md`:

* the cohort comes from `analysis_manifest_v1.csv` filtered to
  `inclusion_status == "INCLUDED"` — never from the workbook, never from a
  directory listing (§7 assertion 27);
* features are read through `feature_source_id` — never `canonical_id`,
  `source_records` or a glob (§7 assertion 33).

The FC extraction functions themselves are *imported* from
`scripts/validate_fc_input.py` so exactly one copy of the frozen §3.7–§3.8 rule
exists in the codebase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from validate_fc_input import (  # noqa: E402
    AAL116_START,
    AAL116_STOP,
    FISHER_CLIP,
    N_EDGES,
    PRIMARY_BRANCH,
    fc_vector,
    load_aal116,
)

import contracts  # noqa: E402

MANIFEST = ROOT / "results" / "data_contract" / "analysis_manifest_v1.csv"
CACHE = ROOT / "results" / "gate1" / "cache" / "fc_aal116_z_included.npz"


def load_cohort() -> pd.DataFrame:
    """The frozen 2232-acquisition cohort, sorted by (site, canonical_id)."""
    m = pd.read_csv(MANIFEST)
    m = m[m["inclusion_status"] == "INCLUDED"].copy()
    m = m.sort_values(["site", "canonical_id"], key=_site_key).reset_index(drop=True)
    contracts.assert_frozen_cohort(m)
    contracts.assert_unique_acquisitions(m)
    assert m["diagnosis"].isin([0, 1]).all(), "diagnosis is not binary"
    assert m["fc_qc_pass"].all(), "an FC_QC_FAIL row reached the cohort"
    return m


def _site_key(col: pd.Series) -> pd.Series:
    """Numeric site ordering (S1 < S2 < S10), identity for other columns."""
    if col.name == "site":
        return col.str.slice(1).astype(int)
    return col


def build_fc_matrix(cohort: pd.DataFrame, cache: Path = CACHE,
                    verbose: bool = True) -> np.ndarray:
    """(n, 6670) Fisher-z FC matrix, row-aligned to `cohort`.

    Cached on disk keyed by the exact ordered `feature_source_id` list, so a
    stale cache can never be silently reused for a different cohort ordering.
    """
    ids = cohort["feature_source_id"].to_numpy().astype("U32")
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["feature_source_id"]) == list(ids):
            X = z["X"]
            contracts.assert_feature_matrix(X, len(cohort))
            if verbose:
                print(f"[fc] cache hit: {cache}")
            return X
        if verbose:
            print("[fc] cache key mismatch -> rebuilding")

    X = np.empty((len(ids), N_EDGES), dtype=np.float64)
    for i, sid in enumerate(ids):
        # Loader reads ROISignals_<feature_source_id>.mat and nothing else.
        ts = load_aal116(str(sid))
        X[i] = fc_vector(ts)          # pure function of the ROI block
        if verbose and (i + 1) % 250 == 0:
            print(f"[fc] {i + 1}/{len(ids)}")
    contracts.assert_feature_matrix(X, len(cohort))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, feature_source_id=ids)
    return X


def build_xy(verbose: bool = True):
    """Returns (X, y, sites, cohort). `y` is diagnosis; `sites` is a split key."""
    cohort = load_cohort()
    X = build_fc_matrix(cohort, verbose=verbose)
    y = cohort["diagnosis"].to_numpy().astype(int)     # diagnosis is y ONLY
    sites = cohort["site"].to_numpy().astype(str)      # split/aggregation key ONLY
    # X is a bare float ndarray built by a single-argument pure function, so no
    # identifier or phenotype column can exist in it; assert on the manifest
    # columns that were *not* consumed as well.
    contracts.assert_no_forbidden_columns(["fc_edge"])  # X carries no names
    contracts.assert_feature_matrix(X, len(cohort))
    return X, y, sites, cohort


FROZEN_INPUT = {
    "branch": PRIMARY_BRANCH,
    "atlas": "AAL116",
    "roi_slice": f"[:, {AAL116_START}:{AAL116_STOP}]",
    "connectivity": "Pearson",
    "transform": f"arctanh(clip(r, +/-{FISHER_CLIP}))",
    "vectorisation": "strict upper triangle (k=1)",
    "n_features": N_EDGES,
}
