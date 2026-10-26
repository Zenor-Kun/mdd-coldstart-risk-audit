"""Leakage / identity assertions for Gate 1.

Every assertion here is a re-statement of a rule that was frozen *before* any
model existed (`docs/DATA_CONTRACT_v1.md` §7, `docs/EXPERIMENT_LOG.md`). None of
them may be relaxed on the basis of a downstream number.

The functions are deliberately cheap so they can be called inside every fold.
Each returns nothing and raises `AssertionError` on violation; the collecting
helper `AssertionLedger` records PASS/FAIL for the report.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# frozen constants (mirrored from DATA_CONTRACT_v1.md; never computed from data)
# ---------------------------------------------------------------------------
N_FEATURES = 6670                 # 116 * 115 / 2, strict upper triangle
FROZEN_N_ACQUISITIONS = 2232
FROZEN_N_MDD = 1182
FROZEN_N_HC = 1050
FROZEN_N_SITES = 23
FROZEN_SITES = (
    "S1", "S2", "S3", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13",
    "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23",
    "S24", "S25",
)

# DATA_CONTRACT_v1.md §7 assertion 34. Mirrored from
# scripts/build_analysis_manifest.py.
FORBIDDEN_IN_X = (
    "canonical_id",
    "feature_source_id",
    "source_records",
    "original_component_id",
    "site",
    "diagnosis",
    # widened for Gate 1 by explicit task instruction: no phenotype of any kind
    "age",
    "sex",
    "mean_fd",
)


class AssertionLedger:
    """Collects named leakage checks so the report can print PASS/FAIL."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append({"assertion": name, "status": "PASS" if ok else "FAIL",
                          "detail": detail})
        if not ok:
            raise AssertionError(f"{name}: {detail}")

    @property
    def all_pass(self) -> bool:
        return all(r["status"] == "PASS" for r in self.rows)


# ---------------------------------------------------------------------------
# cohort / identity
# ---------------------------------------------------------------------------
def assert_frozen_cohort(manifest) -> None:
    """DATA_CONTRACT_v1.md §3.12.7 — the cohort is an asserted constant."""
    assert (manifest["inclusion_status"] == "INCLUDED").all(), \
        "manifest was not filtered to inclusion_status == 'INCLUDED'"
    n = len(manifest)
    n_mdd = int((manifest["diagnosis"] == 1).sum())
    n_hc = int((manifest["diagnosis"] == 0).sum())
    sites = sorted(manifest["site"].unique())
    assert n == FROZEN_N_ACQUISITIONS, f"N={n} != {FROZEN_N_ACQUISITIONS}"
    assert n_mdd == FROZEN_N_MDD, f"MDD={n_mdd} != {FROZEN_N_MDD}"
    assert n_hc == FROZEN_N_HC, f"HC={n_hc} != {FROZEN_N_HC}"
    assert len(sites) == FROZEN_N_SITES, f"sites={len(sites)} != {FROZEN_N_SITES}"
    assert set(sites) == set(FROZEN_SITES), "eligible-site set moved"


def assert_unique_acquisitions(manifest) -> None:
    """§3.12.8(1) — one row is one independent acquisition, never an alias."""
    assert manifest["canonical_id"].is_unique, "duplicate canonical_id in X"
    assert manifest["feature_source_id"].is_unique, "duplicate feature_source_id"
    assert manifest["feature_source_id"].notna().all(), "null feature_source_id"
    # No dropped alias may appear as a row of its own.
    canon = set(manifest["canonical_id"])
    aliases: set[str] = set()
    for rec in manifest["source_records"]:
        members = [m for m in str(rec).split(";") if m]
        aliases.update(members[1:] if len(members) > 1 else [])
    # members are sorted in the manifest; the representative is members[0]
    leaked = aliases & canon
    assert not leaked, f"dropped alias present as a row: {sorted(leaked)[:3]}"


def assert_no_forbidden_columns(columns) -> None:
    """§7 assertion 34 — nothing identity- or phenotype-derived enters X."""
    bad = sorted(set(map(str, columns)) & set(FORBIDDEN_IN_X))
    assert not bad, f"forbidden columns in X: {bad}"


def assert_feature_matrix(X: np.ndarray, n_rows: int) -> None:
    """§7 assertions 16, 19, 20."""
    assert X.ndim == 2, f"X.ndim == {X.ndim}"
    assert X.shape == (n_rows, N_FEATURES), \
        f"X.shape == {X.shape}, expected ({n_rows}, {N_FEATURES})"
    assert np.isfinite(X).all(), "X contains NaN or Inf"


# ---------------------------------------------------------------------------
# protocol (spec §7, §10)
# ---------------------------------------------------------------------------
def assert_held_out_site_invisible(fit_sites, held_out_site) -> None:
    """§10 checklist item 1 — the outer test site enters no fit of any kind."""
    assert held_out_site not in set(fit_sites), \
        f"held-out site {held_out_site} appeared in a fitting partition"


def assert_disjoint(train_idx, test_idx) -> None:
    assert not (set(np.asarray(train_idx).tolist())
                & set(np.asarray(test_idx).tolist())), \
        "train/test index overlap"


def assert_site_weights(weights: np.ndarray, sites: np.ndarray) -> None:
    """§10 checklist item 9 — training weights make sites equal.

    w_i = (n_train / n_sites_train) / n_{s(i)}  ==>  every site's weights sum to
    the same constant, and the mean weight is exactly 1.
    """
    sums = {s: float(weights[sites == s].sum()) for s in np.unique(sites)}
    v = np.array(list(sums.values()))
    assert np.allclose(v, v[0]), f"site weight mass unequal: {sums}"
    assert abs(float(weights.mean()) - 1.0) < 1e-9, \
        f"mean sample weight {weights.mean()} != 1"


def assert_permutation_within_site(y_orig: np.ndarray, y_perm: np.ndarray,
                                   sites: np.ndarray) -> None:
    """RQ0(c) — site membership, site size and site prevalence are preserved."""
    for s in np.unique(sites):
        m = sites == s
        assert int(m.sum()) > 0
        assert int(y_orig[m].sum()) == int(y_perm[m].sum()), \
            f"permutation changed prevalence at site {s}"
