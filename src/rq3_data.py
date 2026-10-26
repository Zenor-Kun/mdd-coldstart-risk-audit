"""RQ3 populations, outcomes and design matrices — `RQ3_PROTOCOL.md` §2, §5, §8.

Three things live here:

    populations   the two frozen complete-case subsets, 2222/23 and 484/9,
                  derived through the validator's own §5.1-§5.3 functions so
                  the production cohort cannot drift from the validated one;
    outcomes      the six registered `R_i` cells, reconstructed by §2.1 from the
                  frozen S0 threshold table and bound to the published S0
                  operating-point coverage (H5);
    design        the §8 coding and transforms, with centering and scaling
                  estimated on an explicitly named fitting partition — the
                  training sites only, inside cross-validation (§8.1, H9).

The row order is fixed once, here: sorted by site (numerically, `S1 ... S25`)
then `canonical_id`. Every model in a family is built from the same frame in
that order, which is what H8 — identical row index across a family — rests on,
and it is also what lets the GLMM group sites by contiguous blocks.

Identity never enters a design matrix: `canonical_id` and `site` are keys, and
`site` reaches the model only as the random-effect grouping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import rq3_protocol as proto

NOT_ESTIMABLE = proto.NOT_ESTIMABLE

FULL_SAMPLE = "full_sample"
MDD_ONLY = "mdd_only"
POPULATIONS = (FULL_SAMPLE, MDD_ONLY)

#: §8 — the exhaustive transform list. `log1p` is applied on the natural scale
#: before standardisation; binary variables are never scaled.
LOG1P_THEN_Z = ("mean_fd", "illness_duration_months")
Z_ONLY = ("age", "hamd")
BINARY_CODING = {
    "sex": proto.SEX_MAP,                       # 1 -> 0, 2 -> 1
    "diagnosis": proto.DIAGNOSIS_MAP,           # HC = 0, MDD = 1
    "first_episode": proto.BINARY_CLINICAL_MAP,  # recurrent -1 -> 0, first 1 -> 1
    "on_medication": proto.BINARY_CLINICAL_MAP,  # no -1 -> 0, yes 1 -> 1
}
CONTINUOUS_VARS = tuple(sorted(set(LOG1P_THEN_Z) | set(Z_ONLY)))
KNOWN_TERMS = tuple(sorted(set(CONTINUOUS_VARS) | set(BINARY_CODING)))

INTERCEPT = "(Intercept)"

#: Columns that may never enter `X` (mirrors the data contract's `FORBIDDEN_IN_X`
#: and the protocol's site/identity rules). `site` is a grouping key only.
FORBIDDEN_IN_X = (
    "canonical_id", "feature_source_id", "source_records",
    "original_component_id", "site", "inclusion_status",
)


class RQ3DataError(AssertionError):
    """Structural corruption of an RQ3 dataset. Never coerced to NOT_ESTIMABLE."""


def _site_order_key(site: str) -> int:
    return int(str(site)[1:])


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["__site_order"] = ordered.site.map(_site_order_key)
    ordered = ordered.sort_values(["__site_order", "canonical_id"], kind="mergesort")
    return ordered.drop(columns="__site_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# §5 populations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Population:
    """One frozen complete-case subset, in its fixed row order."""

    name: str
    frame: pd.DataFrame

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def sites(self) -> tuple:
        return tuple(dict.fromkeys(self.frame.site))

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def canonical_ids(self) -> tuple:
        return tuple(self.frame.canonical_id)


def assert_mean_fd_binding(cohort: pd.DataFrame) -> None:
    """The manifest's `mean_fd` is the frozen per-subject Jenkinson FD.

    §1 consumes both files; this binds them rather than trusting that they agree.
    The key is `feature_source_id`, the frozen representative of the acquisition
    (`DATA_CONTRACT_v1.md` §3.12), never `canonical_id`.
    """
    per_subject = pd.read_csv(
        proto.ROOT / "results/data_contract/mean_fd_per_subject.csv"
    ).set_index("ID").mean_FD_Jenkinson
    mapped = cohort.feature_source_id.map(per_subject)
    if mapped.isna().any():
        missing = int(mapped.isna().sum())
        raise RQ3DataError(
            f"H1: {missing} acquisitions have no frozen mean-FD row"
        )
    discrepancy = float((cohort.mean_fd.astype(float) - mapped.astype(float)).abs().max())
    if discrepancy != 0.0:
        raise RQ3DataError(
            f"H1: manifest mean_fd disagrees with the frozen FD file by {discrepancy!r}"
        )


def build_population(name: str, verify_mean_fd: bool = True) -> Population:
    """Derive a frozen population subset and hard-fail on any count drift (H2-H4)."""
    if name not in POPULATIONS:
        raise RQ3DataError(f"unknown population {name!r}")

    cohort = proto.load_base_cohort()
    if len(cohort) != proto.BASE_COHORT_N or cohort.site.nunique() != proto.BASE_COHORT_SITES:
        raise RQ3DataError(
            f"H2: base cohort is {len(cohort)} rows / {cohort.site.nunique()} sites, "
            f"frozen is {proto.BASE_COHORT_N} / {proto.BASE_COHORT_SITES}"
        )
    if verify_mean_fd:
        assert_mean_fd_binding(cohort)
    cohort = proto.apply_anomaly_rule(cohort)      # §5.2, frozen

    if name == FULL_SAMPLE:
        subset = proto.full_sample_subset(cohort)
        if len(subset) != proto.FULL_CC_N:
            raise RQ3DataError(
                f"H3: full-sample complete case is {len(subset)}, frozen is {proto.FULL_CC_N}"
            )
        if subset.site.nunique() != proto.FULL_CC_SITES:
            raise RQ3DataError(
                f"H3: full-sample site count is {subset.site.nunique()}, "
                f"frozen is {proto.FULL_CC_SITES}"
            )
        n_mdd = int((subset.diagnosis == 1).sum())
        n_hc = int((subset.diagnosis == 0).sum())
        if (n_mdd, n_hc) != (proto.FULL_CC_MDD, proto.FULL_CC_HC):
            raise RQ3DataError(
                f"H3: full-sample composition is {n_mdd}/{n_hc}, frozen is "
                f"{proto.FULL_CC_MDD}/{proto.FULL_CC_HC}"
            )
    else:
        subset = proto.mdd_only_subset(cohort)
        if len(subset) != proto.MDD_CC_N:
            raise RQ3DataError(
                f"H4: MDD-only complete case is {len(subset)}, frozen is {proto.MDD_CC_N}"
            )
        observed = tuple(sorted(subset.site.unique(), key=_site_order_key))
        if observed != tuple(proto.MDD_CC_SITES):
            raise RQ3DataError(f"H4: MDD-only site set is {observed}")
        per_site = {s: int(n) for s, n in subset.groupby("site").size().items()}
        if per_site != dict(proto.MDD_CC_PER_SITE):
            raise RQ3DataError(f"H4: MDD-only per-site distribution changed: {per_site}")

    frame = _sorted_frame(subset)
    if frame.canonical_id.duplicated().any():
        raise RQ3DataError("a canonical_id appears twice — the acquisition unit is broken")
    return Population(name=name, frame=frame)


# ---------------------------------------------------------------------------
# §2 outcomes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellOutcome:
    """`R_i` for one registered cell, aligned to a population's row order."""

    population: str
    operating_point: str
    method: str
    R: np.ndarray

    @property
    def cell_index(self) -> int:
        return proto.CELL_REGISTRY.index((self.operating_point, self.method))


def registered_cells() -> tuple:
    """§2.4.1 — the six cells in their frozen registry order."""
    return tuple(proto.CELL_REGISTRY)


def build_cell_outcome(population: Population, operating_point: str,
                       method: str) -> CellOutcome:
    """§2.1 — `R_i = 1{confidence_M(i) < tau[s(i), M, q(OP)]}`, aligned to rows.

    The threshold is selected by the frozen table's own `operating_point`
    column; `q` is never re-derived (§2.2).
    """
    if (operating_point, method) not in proto.CELL_REGISTRY:
        raise RQ3DataError(f"unregistered cell {(operating_point, method)!r}")
    rows = proto.reconstruct_R(method, operating_point)
    if rows.canonical_id.duplicated().any():
        raise RQ3DataError("the reconstructed outcome holds a duplicate canonical_id")
    lookup = rows.set_index("canonical_id").R
    missing = [cid for cid in population.canonical_ids if cid not in lookup.index]
    if missing:
        raise RQ3DataError(
            f"{len(missing)} population rows have no reconstructed R_i"
        )
    R = lookup.reindex(list(population.canonical_ids)).to_numpy(dtype=float)
    if not np.all((R == 0.0) | (R == 1.0)):
        raise RQ3DataError("reconstructed R_i is not binary")
    return CellOutcome(population.name, operating_point, method, R)


def assert_outcome_reproduces_s0(tolerance: float | None = None) -> dict:
    """H5 — the six reconstructed cells reproduce the frozen S0 coverage.

    The compared quantity is a **published RQ1/S0 result**, reproduced; it is
    not an RQ3 quantity. Returns the per-cell discrepancies for provenance.
    """
    tol = proto.COVERAGE_TOL if tolerance is None else float(tolerance)
    discrepancies = {}
    for operating_point, method in proto.CELL_REGISTRY:
        rows = proto.reconstruct_R(method, operating_point)
        per_site = rows.groupby("site").R.agg(["size", "sum"])
        coverage = float((1.0 - per_site["sum"] / per_site["size"]).mean())
        frozen = proto.frozen_site_balanced_coverage(operating_point, method)
        discrepancy = abs(coverage - frozen)
        if discrepancy > tol:
            raise RQ3DataError(
                f"H5: {operating_point}/{method} site-balanced coverage {coverage!r} "
                f"!= frozen {frozen!r} (|delta| {discrepancy!r} > {tol!r})"
            )
        discrepancies[f"{operating_point}|{method}"] = discrepancy
    return discrepancies


def rejected_fraction(population: Population, outcome: CellOutcome) -> float:
    """§15.1 — the registered site-balanced rejected fraction.

    Delegates to the frozen mechanism, with the frozen `S = 23 / 9` denominator
    enforced, so an acquisition-pooled value cannot be returned by accident (H14).
    """
    expected = proto.SITE_BALANCED_DENOMINATOR[population.name]
    return float(proto.site_balanced_rejected_fraction(
        population.frame.site, outcome.R, expected_sites=expected
    ))


# ---------------------------------------------------------------------------
# §8 coding and transforms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformFit:
    """Centering/scaling parameters estimated on one explicitly named partition.

    §8.1: inside cross-validation these come from the **training sites only**
    and are applied unchanged to the held-out site; for a full-data inferential
    fit they come from that population's whole frozen complete-case subset and
    are shared by every model in the family, so nested models differ only in
    their fixed-effect terms.
    """

    terms: tuple
    location: dict
    scale: dict
    fitted_on: str
    n_fitting_rows: int

    def to_payload(self) -> dict:
        return {
            "terms": list(self.terms),
            "location": {k: float(v) for k, v in sorted(self.location.items())},
            "scale": {k: float(v) for k, v in sorted(self.scale.items())},
            "fitted_on": self.fitted_on,
            "n_fitting_rows": int(self.n_fitting_rows),
        }


def _raw_continuous(frame: pd.DataFrame, term: str) -> np.ndarray:
    values = frame[term].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise RQ3DataError(f"H6: {term} holds a missing value inside a complete-case subset")
    if term in LOG1P_THEN_Z:
        if np.any(values < 0.0):
            raise RQ3DataError(f"H7: negative log1p argument in {term}")
        values = np.log1p(values)
    return values


def _coded_binary(frame: pd.DataFrame, term: str) -> np.ndarray:
    mapping = BINARY_CODING[term]
    coded = frame[term].map(mapping)
    if coded.isna().any():
        offenders = sorted(set(frame[term][coded.isna()].tolist()))
        raise RQ3DataError(
            f"H6: the {term} coding map is not total — unmapped values {offenders}"
        )
    return coded.to_numpy(dtype=float)


def fit_transform(frame: pd.DataFrame, terms, fitted_on: str) -> TransformFit:
    """Estimate the §8 location/scale of every continuous term on these rows only."""
    location, scale = {}, {}
    for term in terms:
        if term not in KNOWN_TERMS:
            raise RQ3DataError(f"§8.2: {term!r} is not one of the frozen covariates")
        if term in BINARY_CODING:
            _coded_binary(frame, term)     # totality is checked on the fitting rows too
            continue                        # binary variables are never scaled (§8)
        values = _raw_continuous(frame, term)
        sd = float(values.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0.0:
            raise RQ3DataError(
                f"§10.5: {term} has zero variance on the {fitted_on} partition"
            )
        location[term] = float(values.mean())
        scale[term] = sd
    return TransformFit(tuple(terms), location, scale, fitted_on, len(frame))


def design_matrix(frame: pd.DataFrame, transform: TransformFit) -> np.ndarray:
    """Apply a fitted transform to these rows; the intercept is column 0.

    The transform is applied **unchanged** — a held-out site never contributes
    to its own scaler (§8.1, H9).
    """
    columns = [np.ones(len(frame), dtype=float)]
    for term in transform.terms:
        if term in BINARY_CODING:
            columns.append(_coded_binary(frame, term))
            continue
        values = _raw_continuous(frame, term)
        columns.append((values - transform.location[term]) / transform.scale[term])
    design = np.column_stack(columns)
    if not np.all(np.isfinite(design)):
        raise RQ3DataError("the design matrix holds a non-finite entry")
    return design


def design_columns(terms) -> tuple:
    """The design matrix's column names, in their fixed order."""
    return (INTERCEPT,) + tuple(terms)


def assert_no_identity_in_design(terms) -> None:
    """Identity, site and outcome may never be fixed effects."""
    for term in terms:
        if term in FORBIDDEN_IN_X:
            raise RQ3DataError(f"{term!r} may never enter a design matrix")
        if term not in KNOWN_TERMS:
            raise RQ3DataError(f"§8.2: {term!r} is not one of the frozen covariates")
