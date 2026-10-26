#!/usr/bin/env python3
"""Stage E.1 — mechanical validator for the frozen RQ3 protocol.

Verifies that `docs/RQ3_PROTOCOL.md` is mechanically executable against the
frozen upstream, and that every count it freezes is re-derivable. It checks
exactly the layer the freeze needs and nothing beyond it:

    artifacts    every consumed file exists, matches its frozen SHA256, and the
                 four S0 files are additionally bound to the frozen S0 ledger
    cohort       2232 INCLUDED acquisitions across 23 eligible sites
    anomaly      the §5.2 rule fires on exactly the documented rows
    subsets      full-sample complete case = 2222 / 23 sites,
                 MDD-only complete case = 484 / the frozen 9 sites
    outcome      R_i reconstructed by §2.1 reproduces the *already published*
                 RQ1/S0 OP50/OP30 site-balanced coverage in all 6 cells
    coding       the §8 maps are total on both frozen subsets; log1p arguments
                 are non-negative
    folds        every LOSO training fold is predictor-non-degenerate
    mechanisms   the reference protocol mechanisms behave as frozen: the
                 site-balanced rejected fraction, the fixed six-slot Holm
                 procedure, the deterministic initializer, the fail-closed
                 seven-of-seven 40-vs-60 validation and the best-start rule
    constants    every execution-controlling constant appears in both this file
                 and the protocol document

**It fits no model.** No likelihood is evaluated, no optimizer is invoked, no
coefficient, p-value, pseudo-R2, log-loss or variance component is computed, and
no association between R_i and any covariate is examined. The only quantity
derived from R_i is site-balanced coverage, which is a frozen RQ1/S0 result
being reproduced, not an RQ3 result being produced.

**It writes nothing.** The process opens every path read-only and does not
create `results/rq3/`.

    python3 scripts/validate_rq3_protocol_freeze.py
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Uncommitted, untagged pre-result draft. No `rq3-protocol-v2` tag exists.
PROTOCOL_VERSION = "RQ3_PROTOCOL/2.0.1"   # E.1 / E.1a-d / E.2a / E.2a.1

# The superseded freeze is immutable and stays readable at its tag; §21.1 of the
# document records the amendment. Nothing here may move or rewrite that tag.
SUPERSEDED_PROTOCOL_VERSION = "RQ3_PROTOCOL/1.0.4"
SUPERSEDED_PROTOCOL_TAG = "rq3-protocol-v1"

# ---------------------------------------------------------------- §1 artifacts

FROZEN_INPUTS = {
    "results/rq1_s0/confidence_target.csv":
        "99cafb3a90bfb9e8c339458407b56ecf4f666cde4f97dfb9a7c88b28653d6b37",
    "results/rq1_s0/thresholds_source_oof.csv":
        "78f332e5255994d4069f9783dff25d5bd53d3cb54bcc3039287011dec3793a9d",
    "results/rq1_s0/op50_summary.csv":
        "31601e0ad3238c2154ca19f3985a861825d91d20fb42b4eb4f9ef21f90ed6b94",
    "results/rq1_s0/op30_summary.csv":
        "3d4e2b7248a7581691e1ee0bfc7480f325371fbdbbc9fbb4d455d0b03730b024",
    "results/data_contract/analysis_manifest_v1.csv":
        "bb582babe5a46e42503f6682140dfc7b0e7834fa813ca5dbb8bdae42dc6c1ff2",
    "results/data_contract/mean_fd_per_subject.csv":
        "7fc40ba94e2709b8d3b04d512cf2ffc94dca7c501d551b0f6a90cd1477cd1ebb",
}

S0_LEDGER = "results/rq1_s0/RESULT_CHECKSUMS.sha256"
S0_LEDGER_BOUND = (
    "confidence_target.csv",
    "thresholds_source_oof.csv",
    "op50_summary.csv",
    "op30_summary.csv",
)

# ------------------------------------------------------------ frozen constants

BASE_COHORT_N = 2232
BASE_COHORT_SITES = 23

FULL_CC_N = 2222
FULL_CC_SITES = 23
FULL_CC_MDD = 1180
FULL_CC_HC = 1042

MDD_CC_N = 484
MDD_CC_SITES = ("S2", "S6", "S7", "S8", "S11", "S14", "S19", "S20", "S23")
MDD_CC_PER_SITE = {
    "S2": 29, "S6": 14, "S7": 38, "S8": 38, "S11": 2,
    "S14": 62, "S19": 33, "S20": 236, "S23": 32,
}

N_AGE_ZERO = 3          # §5.2 age == 0 -> missing
N_SEX_INVALID = 1       # §5.2 sex not in {1,2} -> missing (non-null offenders)
N_HAMD_ZERO_KEPT = 14   # §5.2 HAMD == 0 retained as observed
N_DURATION_ZERO_KEPT = 4

CONFIDENCE_METHODS = ("probability", "bootstrap", "mahalanobis")
OPERATING_POINTS = ("OP50", "OP30")
COVERAGE_TOL = 1e-9

# §2.4.1 frozen cell registry order — serialization order and the deterministic
# secondary sort key of the Holm procedure. Carries no precedence claim.
CELL_REGISTRY = (
    ("OP50", "probability"),
    ("OP50", "bootstrap"),
    ("OP50", "mahalanobis"),
    ("OP30", "probability"),
    ("OP30", "bootstrap"),
    ("OP30", "mahalanobis"),
)

# §15.1 rejected_fraction is site-balanced, uniquely
REJECTED_FRACTION_MODE = "site_balanced"
SITE_BALANCED_DENOMINATOR = {"full_sample": 23, "mdd_only": 9}

FULL_VARS = ("age", "sex", "mean_fd", "diagnosis")
MDD_VARS = ("mean_fd", "hamd", "first_episode",
            "on_medication", "illness_duration_months")

# §8 coding maps
SEX_MAP = {1.0: 0, 2.0: 1}
DIAGNOSIS_MAP = {0.0: 0, 1.0: 1}
BINARY_CLINICAL_MAP = {-1.0: 0, 1.0: 1}   # first_episode, on_medication
LOG1P_VARS = ("mean_fd", "illness_duration_months")

# §12.3 / §12.4 multiplicity
ALPHA = 0.05
HOLM_FAMILY_SIZE = 6
NOT_ESTIMABLE = "NOT_ESTIMABLE"
HOLM_PLACEHOLDER_P = 1.0   # internal only, never serialized (§12.4 step 2)
HOLM_FAMILIES = (
    ("full_sample", "motion", 6),
    ("full_sample", "diagnosis", 6),
    ("mdd_only", "motion", 6),
    ("mdd_only", "clinical", 6),
)
BLOCK_DF = {
    ("full_sample", "motion"): 1,
    ("full_sample", "diagnosis"): 1,
    ("mdd_only", "motion"): 1,
    ("mdd_only", "clinical"): 4,
}

# §9 numerics
GH_NODES_PRIMARY = 40
GH_NODES_VALIDATION = 60
LOG_SIGMA_BOUNDS = (np.log(1e-4), np.log(10.0))
OPTIMIZER = "L-BFGS-B"
# §9.4.5 — AMENDED at Stage E.2a.1 (was 1e-12). `ftol` is L-BFGS-B's *relative*
# objective criterion; §9.5(3) is an *absolute* gradient criterion. At n = 2222
# the old value fired before `gtol` could, stopping at a point the frozen
# eligibility rule then rejected. 1e-16 is at double-precision round-off, so
# `gtol` becomes the operative stopping criterion. The eligibility threshold
# itself (GRADIENT_TOL) was NOT relaxed.
OPT_FTOL = 1e-16
OPT_GTOL = 1e-8
OPT_MAXITER = 20000
OPT_MAXFUN = 50000
GRADIENT_TOL = 1e-5

# §9.4 deterministic initializer (Stage E.1a) — no secondary solver
INIT_PREVALENCE_CLIP = 1e-6
SIGMA_STARTS = (0.1, 0.5, 1.0, 2.0)

# §9.4.1 the four registered start slots, index -> frozen sigma start.
# The index is the tie-break key of the best-start rule; it is never a priority.
START_INDEX_TO_SIGMA = {0: 0.1, 1: 0.5, 2: 1.0, 3: 2.0}
N_REGISTERED_STARTS = 4

# §9.6 40-vs-60 numerical validation
AUTHORITATIVE_NODES = GH_NODES_PRIMARY      # the 40-node fit is the science
SIGMA2_RELATIVE_DENOMINATOR_FLOOR = 1e-12   # numerical only, never a threshold
AGREEMENT_TOL = {
    "loglik": 1e-4,
    "lrt_statistic": 1e-3,
    "lrt_pvalue": 1e-4,
    "mcfadden": 1e-4,
    "sigma_site_sq_relative": 1e-3,
    "cv_logloss": 1e-6,
    "fixed_effect_coefficients": 1e-4,      # §9.6.3, newly declared at E.1a
}
AGREEMENT_RELATIVE = ("sigma_site_sq_relative",)

# §9.6.4 the explicit required-criterion registry. Exactly seven, no more, no
# fewer: a numerical validation may PASS only when every one of them has been
# constructed from finite inputs and has met its frozen tolerance. Adding a key
# to AGREEMENT_TOL without adding it here — or the reverse — is a hard failure.
REQUIRED_AGREEMENT_CRITERIA = (
    "loglik",
    "lrt_statistic",
    "lrt_pvalue",
    "mcfadden",
    "sigma_site_sq_relative",
    "cv_logloss",
    "fixed_effect_coefficients",
)
N_REQUIRED_AGREEMENT_CRITERIA = 7

# the solution fields each criterion is built from
AGREEMENT_INPUT_FIELDS = {
    "loglik": ("loglik",),
    "lrt_statistic": ("lrt_statistic",),
    "lrt_pvalue": ("lrt_pvalue",),
    "mcfadden": ("mcfadden",),
    "sigma_site_sq_relative": ("sigma_site",),
    "cv_logloss": ("cv_logloss",),
    "fixed_effect_coefficients": ("beta",),
}


# ------------------------------------------- §9.2 amended quadrature (E.2a)
# The site integral is mode/curvature-adaptive Gauss-Hermite. The Stage E.1
# fixed-grid rule is WITHDRAWN: it survives only as the §9.2.4 reduction
# identity and in the §21 amendment record, never as a live instruction.
QUADRATURE_RULE = "mode_curvature_adaptive_gauss_hermite"
WITHDRAWN_QUADRATURE_RULE = "non_adaptive_gauss_hermite"
QUADRATURE_IS_ADAPTIVE = True

# §9.2.3 the four structural terms of the transformation. Dropping any one of
# them silently rescales every likelihood in the protocol (H20).
ADAPTIVE_TRANSFORM_TERMS = (
    "mode_relocation",        # u_{s,k} = u_hat_s + sqrt(2)*tau_s*z_k
    "curvature_scaling",      # tau_s = H_hat_s ** -0.5
    "jacobian",               # + log(sqrt(2) * tau_s)
    "kernel_deweighting",     # + z_k ** 2 inside the logsumexp
)

# §9.2.5 mode solver — every constant is registered; none may be retuned (H22)
MODE_SOLVER = "safeguarded_newton"
MODE_INIT = 0.0                     # frozen start, never data-dependent
MODE_CONVERGENCE_TOL = 1e-8         # |step| * sqrt(H); costs <= 5e-17 in logLik
MODE_STAGNATION_TOL = 1e-4          # fail-closed cap; costs <= 5e-9 in logLik
MODE_MAXITER = 200
MODE_MAX_HALVINGS = 50
CURVATURE_FLOOR = 1e-12
MODE_FALLBACK_SOLVER_PERMITTED = False

# §9.4.4 gradient — exact total derivative of the ADAPTIVE objective. The Stage
# E.1 fixed-node gradient is not the gradient of this objective (H21).
GRADIENT_RULE = "exact_total_derivative"
GRADIENT_FINITE_DIFFERENCE_PERMITTED = False
GRADIENT_TERMS = (
    "d_log_tau",              # + d log tau_s / d theta
    "partial_h_at_fixed_u",   # + sum_k P_sk * dh/dtheta
    "mode_movement",          # + (sum_k P_sk h'(u_sk)) * d u_hat_s / d theta
    "scale_movement",         # + sqrt(2) tau_s (d log tau_s/dtheta) sum_k P_sk z_k h'
)

# §9.6.5 the granularity at which a seven-of-seven verdict is produced
VALIDATION_UNIT = ("population", "cell", "block")
N_VALIDATION_UNITS = 24     # 2 populations x 6 cells x 2 blocks

# §9.5.1 the closed registry of status / failure codes.
STATUS_OK = "ESTIMABLE"
START_FAILURE_CODES = (
    "OPTIMIZER_NOT_CONVERGED",
    "NON_FINITE_OBJECTIVE",
    "GRADIENT_NOT_ZERO",
    "SIGMA_AT_OR_OUTSIDE_BOUND",
    "ADAPTIVE_MODE_NOT_FOUND",
    "ADAPTIVE_CURVATURE_INVALID",
)
AGGREGATE_FAILURE_CODES = (
    "NO_ELIGIBLE_START",
    "QUADRATURE_DISAGREEMENT",
    "VALIDATION_MALFORMED",
    "NESTED_LIKELIHOOD_INVERSION",
    "CV_FOLD_NOT_ESTIMABLE",
)
NUMERICAL_FAILURE_CODES = START_FAILURE_CODES + AGGREGATE_FAILURE_CODES
STATUS_CODES = (STATUS_OK,) + NUMERICAL_FAILURE_CODES

# §10.6 / §12.1.1 semantics accepted at Stage E.2a
CV_AGGREGATION_POLICY = "all_or_nothing"
CV_PARTIAL_FOLD_AVERAGE_PERMITTED = False
NEGATIVE_LRT_POLICY = "NESTED_LIKELIHOOD_INVERSION"
NEGATIVE_LRT_CLAMP_PERMITTED = False
SIGMA2_REQUIRES_BOTH_UNITS_PASSING = True

# §16.1 / §16.2 formal execution invariants
AUTHORITATIVE_RESULT_ROOT = "results/rq3"
COMPLETION_MARKER = "COMPLETION.json"
CALLER_CONTROLLED_ROOT_PERMITTED = False
STALE_ROOT_POLICY = "fail_closed"
COMPLETION_ORDER = (
    "write_artifacts",
    "write_checksum_ledger",
    "flush_and_fsync",
    "reread_from_disk",
    "recompute_and_compare_digests",
    "verify_artifact_set_exact",
    "write_completion_marker_atomically",
)


class ValidationFailure(AssertionError):
    """Raised when a frozen protocol invariant does not hold."""


# --------------------------------------------------------------------- loaders

def sha256_of(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load_base_cohort() -> pd.DataFrame:
    """The §5.1 base cohort: 2232 INCLUDED acquisitions, manifest only."""
    manifest = pd.read_csv(ROOT / "results/data_contract/analysis_manifest_v1.csv")
    return manifest[manifest.inclusion_status == "INCLUDED"].copy()


def apply_anomaly_rule(cohort: pd.DataFrame) -> pd.DataFrame:
    """§5.2 — frozen before outcome modelling.

    age == 0 and sex not in {1, 2} become missing. HAMD == 0 and
    illness_duration_months == 0 are retained as observed. Education is never
    consulted.
    """
    out = cohort.copy()
    out.loc[out.age == 0, "age"] = np.nan
    out.loc[~out.sex.isin([1, 2]), "sex"] = np.nan
    return out


def full_sample_subset(cohort: pd.DataFrame) -> pd.DataFrame:
    return cohort[cohort[list(FULL_VARS)].notna().all(axis=1)].copy()


def mdd_only_subset(cohort: pd.DataFrame) -> pd.DataFrame:
    mdd = cohort[cohort.diagnosis == 1]
    return mdd[mdd[list(MDD_VARS)].notna().all(axis=1)].copy()


def reconstruct_R(method: str, operating_point: str) -> pd.DataFrame:
    """§2.1 — R_i = 1{confidence_M(i) < tau[site, M, q(OP)]}.

    Returns one row per acquisition with its site and its rejection indicator.
    The operating point is taken from the frozen table's own `operating_point`
    column; `q` is never re-derived.
    """
    conf = pd.read_csv(ROOT / "results/rq1_s0/confidence_target.csv")
    thresholds = pd.read_csv(ROOT / "results/rq1_s0/thresholds_source_oof.csv")
    tau = thresholds[
        (thresholds.operating_point == operating_point)
        & (thresholds.method == method)
    ].set_index("held_out_site").tau
    if len(tau) != BASE_COHORT_SITES:
        raise ValidationFailure(
            f"{operating_point}/{method}: expected {BASE_COHORT_SITES} thresholds, got {len(tau)}"
        )
    return pd.DataFrame({
        "canonical_id": conf.canonical_id,
        "site": conf.held_out_site,
        "R": (conf["confidence_" + method] < conf.held_out_site.map(tau)).astype(int),
    })


def frozen_site_balanced_coverage(operating_point: str, method: str) -> float:
    """The published S0 value this stage must reproduce."""
    name = "op50_summary.csv" if operating_point == "OP50" else "op30_summary.csv"
    summary = pd.read_csv(ROOT / "results/rq1_s0" / name)
    row = summary[(summary.operating_point == operating_point)
                  & (summary.method == method)]
    return float(row.actual_coverage_site_balanced.iloc[0])


# ------------------------------------------- reference protocol mechanisms
#
# The four mechanisms closed at Stage E.1a, made executable so the freeze tests
# can drive them. None of them fits a model, evaluates a likelihood, performs
# quadrature or reads the real `R_i` table: they are protocol *definitions*, and
# Stage E.2 must still implement the GLMM itself.


def site_balanced_rejected_fraction(sites, rejections, expected_sites=None) -> float:
    """§15.1 — the registered `rejected_fraction`.

    `(1/S) * sum_s (n_rejected_s / n_s)`, the mean of per-site rejection rates,
    with every represented site weighted equally regardless of size.
    """
    frame = pd.DataFrame({"site": list(sites), "R": list(rejections)})
    if frame.empty:
        raise ValidationFailure("§15.1: rejected_fraction is undefined on an empty subset")
    per_site = frame.groupby("site").R.mean()
    if expected_sites is not None and len(per_site) != expected_sites:
        raise ValidationFailure(
            f"H14: rejected_fraction denominator is {len(per_site)}, frozen value is {expected_sites}"
        )
    return float(per_site.mean())


def pooled_rejected_fraction(rejections) -> float:
    """Acquisition-pooled `n_rejected / N`.

    Mechanical audit quantity only (§15.1). It may never be reported in place of,
    or beside as a competitor to, the site-balanced value.
    """
    values = np.asarray(list(rejections), dtype=float)
    if values.size == 0:
        raise ValidationFailure("pooled audit fraction is undefined on an empty subset")
    return float(values.mean())


def holm_adjust_family(pvalues) -> list:
    """§12.4 — Holm step-down across a fixed family of six pre-registered slots.

    `pvalues` holds six entries, each a float or the `NOT_ESTIMABLE` sentinel.
    Unestimable slots keep their slot and contribute an internal placeholder of
    1.0, so `m` stays 6; they are returned as `NOT_ESTIMABLE`, never as a number.
    Ties break by the §2.4.1 registry index, making the order deterministic.
    """
    entries = list(pvalues)
    if len(entries) != HOLM_FAMILY_SIZE:
        raise ValidationFailure(
            f"H12: a Holm family holds {len(entries)} slots, frozen size is {HOLM_FAMILY_SIZE}"
        )
    internal = []
    for entry in entries:
        if entry == NOT_ESTIMABLE:
            internal.append(HOLM_PLACEHOLDER_P)
            continue
        value = float(entry)
        if not (0.0 <= value <= 1.0):
            raise ValidationFailure(f"§12.4: raw p-value outside [0, 1]: {value!r}")
        internal.append(value)

    m = HOLM_FAMILY_SIZE
    order = sorted(range(m), key=lambda i: (internal[i], i))
    adjusted = [None] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * internal[index]))
        adjusted[index] = running
    return [
        NOT_ESTIMABLE if entry == NOT_ESTIMABLE else adjusted[i]
        for i, entry in enumerate(entries)
    ]


def holm_is_significant(raw_p, adjusted_p, alpha: float = ALPHA) -> bool:
    """§12.4 step 8 — a `NOT_ESTIMABLE` cell is never significant."""
    if raw_p == NOT_ESTIMABLE or adjusted_p == NOT_ESTIMABLE:
        return False
    return float(adjusted_p) <= alpha


def deterministic_initial_points(fitting_rejections, n_non_intercept: int) -> list:
    """§9.4 — the four deterministic starts, from the fitting rows alone.

    `fitting_rejections` is `R` restricted to the rows the fit is trained on; for
    a LOSO fold that is the training sites only. Because the function sees
    nothing but the rows handed to it, a fold physically cannot reach the
    held-out site's outcomes through it.
    """
    values = np.asarray(list(fitting_rejections), dtype=float)
    if values.size == 0:
        raise ValidationFailure("§9.4: the initializer needs at least one fitting row")
    if n_non_intercept < 0:
        raise ValidationFailure("§9.4: negative fixed-effect count")
    prevalence = float(values.mean())
    clipped = min(max(prevalence, INIT_PREVALENCE_CLIP), 1.0 - INIT_PREVALENCE_CLIP)
    intercept = float(np.log(clipped / (1.0 - clipped)))
    beta = np.concatenate([[intercept], np.zeros(n_non_intercept, dtype=float)])
    return [(beta.copy(), float(np.log(sigma))) for sigma in SIGMA_STARTS]


def sigma2_relative_difference(sigma_40: float, sigma_60: float) -> float:
    """§9.6.2 — the one relative criterion, denominated on the 60-node variance."""
    sigma2_40 = float(sigma_40) ** 2
    sigma2_60 = float(sigma_60) ** 2
    return abs(sigma2_40 - sigma2_60) / max(abs(sigma2_60), SIGMA2_RELATIVE_DENOMINATOR_FLOOR)


def coefficient_discrepancy(beta_40, beta_60) -> float:
    """§9.6.2 — sup-norm over the fixed-effect vector, no denominator.

    Fail-closed (§9.6.4): both vectors must be one-dimensional, numeric,
    non-empty, equal in length and finite throughout.
    """
    a = _require_finite_vector(beta_40, "beta (40-node)")
    b = _require_finite_vector(beta_60, "beta (60-node)")
    if a.shape != b.shape:
        raise ValidationFailure(
            f"§9.6.4: coefficient vectors have different lengths: {a.shape} vs {b.shape}"
        )
    return float(np.max(np.abs(a - b)))


def _require_finite_scalar(solution, key: str, label: str) -> float:
    """§9.6.4 — a required scalar must exist, be numeric and be finite."""
    if not isinstance(solution, dict):
        raise ValidationFailure(f"§9.6.4: {label} solution is not a mapping")
    if key not in solution:
        raise ValidationFailure(f"§9.6.4: {label} solution is missing required field {key!r}")
    value = solution[key]
    if value is None:
        raise ValidationFailure(f"§9.6.4: {label} field {key!r} is None")
    if isinstance(value, bool):
        raise ValidationFailure(f"§9.6.4: {label} field {key!r} is a bool, not a measurement")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(
            f"§9.6.4: {label} field {key!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(numeric):
        raise ValidationFailure(f"§9.6.4: {label} field {key!r} is not finite: {numeric!r}")
    return numeric


def _require_finite_vector(vector, label: str) -> np.ndarray:
    """§9.6.4 — a required coefficient vector: 1-D, numeric, non-empty, finite."""
    if vector is None:
        raise ValidationFailure(f"§9.6.4: {label} is None")
    if isinstance(vector, (str, bytes, dict)):
        raise ValidationFailure(f"§9.6.4: {label} is not an array: {type(vector).__name__}")
    try:
        array = np.asarray(vector, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"§9.6.4: {label} is not a numeric array") from exc
    if array.ndim != 1:
        raise ValidationFailure(f"§9.6.4: {label} is not one-dimensional (ndim {array.ndim})")
    if array.size == 0:
        raise ValidationFailure(f"§9.6.4: {label} is empty")
    if not np.all(np.isfinite(array)):
        raise ValidationFailure(f"§9.6.4: {label} holds a non-finite element")
    return array


def agreement_checks(solution_40: dict, solution_60: dict) -> dict:
    """§9.6 — the seven registered 40-vs-60 criteria, evaluated fail-closed.

    Returns `{criterion: (discrepancy, tolerance, passed)}` with **exactly** the
    seven entries of `REQUIRED_AGREEMENT_CRITERIA`. Any missing, `None`,
    non-numeric or non-finite required input raises `ValidationFailure`; nothing
    is ever skipped, so a criterion cannot be silently dropped (§9.6.4).

    The 40-node solution stays authoritative whatever the verdict: a failure
    marks the cell `NOT_ESTIMABLE`, it never substitutes the 60-node numbers.
    """
    verdict = {}
    for criterion in REQUIRED_AGREEMENT_CRITERIA:
        tolerance = AGREEMENT_TOL[criterion]
        if criterion == "sigma_site_sq_relative":
            sigma_40 = _require_finite_scalar(solution_40, "sigma_site", "40-node")
            sigma_60 = _require_finite_scalar(solution_60, "sigma_site", "60-node")
            discrepancy = sigma2_relative_difference(sigma_40, sigma_60)
        elif criterion == "fixed_effect_coefficients":
            if not isinstance(solution_40, dict) or not isinstance(solution_60, dict):
                raise ValidationFailure("§9.6.4: a solution is not a mapping")
            if "beta" not in solution_40 or "beta" not in solution_60:
                raise ValidationFailure("§9.6.4: a solution is missing required field 'beta'")
            discrepancy = coefficient_discrepancy(solution_40["beta"], solution_60["beta"])
        else:
            value_40 = _require_finite_scalar(solution_40, criterion, "40-node")
            value_60 = _require_finite_scalar(solution_60, criterion, "60-node")
            discrepancy = abs(value_40 - value_60)
        if not math.isfinite(discrepancy):
            raise ValidationFailure(f"§9.6.4: {criterion} discrepancy is not finite")
        verdict[criterion] = (discrepancy, tolerance, discrepancy <= tolerance)

    if tuple(verdict) != REQUIRED_AGREEMENT_CRITERIA:
        raise ValidationFailure(
            f"§9.6.4: evaluated criteria {tuple(verdict)} != required "
            f"{REQUIRED_AGREEMENT_CRITERIA}"
        )
    return verdict


def numerical_validation_verdict(solution_40: dict, solution_60: dict) -> dict:
    """§9.6.4 — the seven-of-seven verdict, with its counts made explicit.

    `passed` is true **only** when `n_evaluated == n_passed == 7`. The counts are
    computed and compared as integers, never inferred from the truth value of an
    empty collection.
    """
    try:
        criteria = agreement_checks(solution_40, solution_60)
    except ValidationFailure as exc:
        return {
            "passed": False,
            "n_evaluated": 0,
            "n_required": N_REQUIRED_AGREEMENT_CRITERIA,
            "n_passed": 0,
            "reason": str(exc),
            "criteria": {},
        }
    n_evaluated = len(criteria)
    n_passed = sum(1 for _, _, passed in criteria.values() if passed)
    complete = n_evaluated == N_REQUIRED_AGREEMENT_CRITERIA
    all_passed = n_passed == N_REQUIRED_AGREEMENT_CRITERIA
    reason = None
    if not complete:
        reason = f"only {n_evaluated} of {N_REQUIRED_AGREEMENT_CRITERIA} criteria evaluated"
    elif not all_passed:
        breached = [name for name, (_, _, ok) in criteria.items() if not ok]
        reason = f"criteria beyond tolerance: {breached}"
    return {
        "passed": bool(complete and all_passed),
        "n_evaluated": n_evaluated,
        "n_required": N_REQUIRED_AGREEMENT_CRITERIA,
        "n_passed": n_passed,
        "reason": reason,
        "criteria": criteria,
    }


def passes_numerical_validation(solution_40: dict, solution_60: dict) -> bool:
    """§9.6.1 step 7 / §9.6.4 — PASS iff all seven criteria are evaluated and met."""
    return numerical_validation_verdict(solution_40, solution_60)["passed"]


# ------------------------------------------- §9.4.1 best-start selection (reference)

def _strict_start_index(value, label: str) -> int:
    """§9.4.1 — a registered start index is a true integer in {0, 1, 2, 3}.

    `bool` is rejected explicitly: it subclasses `int`, so `True == 1` would
    otherwise smuggle a malformed identity into slot 1 (and `{1: a, True: b}`
    would silently collapse to one mapping entry).
    """
    if isinstance(value, bool):
        raise ValidationFailure(f"H19: {label} is a bool, not an integer start index")
    if not isinstance(value, (int, np.integer)):
        raise ValidationFailure(f"H19: {label} is not an integer start index: {value!r}")
    index = int(value)
    if index not in START_INDEX_TO_SIGMA:
        raise ValidationFailure(
            f"H19: {label} {index} is not one of the registered {sorted(START_INDEX_TO_SIGMA)}"
        )
    return index


def _strict_start_sigma(record: dict, index: int) -> float:
    """§9.4.1 — `sigma_start` must exist, be a finite real, and be canonical.

    Finiteness is established **before** the equality comparison, so a `NaN`
    cannot slip through on the falsity of every comparison it takes part in.
    """
    if "sigma_start" not in record:
        raise ValidationFailure(f"H19: start {index} carries no sigma_start")
    value = record["sigma_start"]
    if value is None:
        raise ValidationFailure(f"H19: start {index} has sigma_start None")
    if isinstance(value, bool):
        raise ValidationFailure(f"H19: start {index} has a bool sigma_start")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValidationFailure(
            f"H19: start {index} has a non-numeric sigma_start: {value!r}"
        )
    numeric = float(value)
    if not math.isfinite(numeric):                     # checked BEFORE equality
        raise ValidationFailure(
            f"H19: start {index} has a non-finite sigma_start: {numeric!r}"
        )
    canonical = START_INDEX_TO_SIGMA[index]
    if numeric != canonical:
        raise ValidationFailure(
            f"H19: start {index} declares sigma {numeric!r}, registered value is {canonical!r}"
        )
    return numeric


def select_best_start(start_records):
    """§9.4.1 — apply the frozen best-start rule to the four registered starts.

    Pure protocol mechanics: it **fits nothing**, runs no optimizer and reads no
    RQ3 outcome. It receives the four already-computed start results and decides
    which one the protocol selects.

    `start_records` is either a mapping `{index: record}` or a sequence of
    exactly four records. Identity is validated strictly and fails **the
    selector**, not merely the start: the four `(index, sigma_start)` pairs must
    be exactly the registered ones, and bool / non-numeric / non-finite /
    non-canonical values are rejected.

    The two input forms carry the registered index differently (§9.4.2):

    * **mapping** — the *key* is the registered index; an embedded `index` is
      optional and must equal the key;
    * **sequence** — the *position* is the registered index; an embedded
      `index` is optional and must equal the position. The selector never
      reorders a sequence by embedded index, so a permuted sequence is a
      contradiction, not a reordering instruction.

    A record whose *identity* is sound but whose *likelihood* is malformed is
    merely unselectable (unchanged Stage E.1b policy). Returns the winning
    index, or `NOT_ESTIMABLE` when no start may be selected.
    """
    pairs = []
    if isinstance(start_records, dict):
        for key, record in start_records.items():
            index = _strict_start_index(key, "mapping key")
            pairs.append((index, record))
    else:
        records = list(start_records)
        if len(records) != N_REGISTERED_STARTS:
            raise ValidationFailure(
                f"H19: {len(records)} start records, the registry holds "
                f"{N_REGISTERED_STARTS}"
            )
        for position, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValidationFailure(f"H19: start record at position {position} is not a mapping")
            # position — never the embedded field — defines the registered index
            index = _strict_start_index(position, f"start record position {position}")
            if "index" in record:
                embedded = _strict_start_index(
                    record["index"], f"embedded index at position {position}"
                )
                if embedded != index:
                    raise ValidationFailure(
                        f"H19: start identity contradiction — position {index} "
                        f"vs embedded index {embedded}"
                    )
            pairs.append((index, record))

    indexed = {}
    for index, record in pairs:
        if not isinstance(record, dict):
            raise ValidationFailure(f"H19: start {index} is not a mapping")
        # a mapping key and an embedded index must agree exactly; neither one
        # silently wins over the other
        if "index" in record:
            embedded = _strict_start_index(record["index"], f"embedded index of start {index}")
            if embedded != index:
                raise ValidationFailure(
                    f"H19: start identity contradiction — key {index} vs embedded index {embedded}"
                )
        if index in indexed:
            raise ValidationFailure(f"H19: duplicate registered start index {index}")
        indexed[index] = record

    if set(indexed) != set(START_INDEX_TO_SIGMA):
        raise ValidationFailure(
            f"H19: start indices {sorted(indexed)} != the registered "
            f"{sorted(START_INDEX_TO_SIGMA)}"
        )

    eligible = {}
    for index in sorted(START_INDEX_TO_SIGMA):
        record = indexed[index]
        _strict_start_sigma(record, index)              # identity, fails the selector
        if record.get("eligible") is not True:
            continue                                    # ineligible can never win
        value = record.get("loglik")
        if value is None or isinstance(value, bool):
            continue                                    # malformed => cannot win
        try:
            loglik = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(loglik):
            continue                                    # NaN / ±inf => cannot win
        eligible[index] = loglik

    if not eligible:
        return NOT_ESTIMABLE                            # §9.4 / §9.6.1 step 6
    # maximum marginal log-likelihood; exact ties break to the lowest index
    return min(eligible, key=lambda index: (-eligible[index], index))


# ------------------------------- §9.2.5 / §9.5.1 / §16 reference mechanisms
# These are protocol *decisions* made executable, exactly like the Stage E.1
# mechanisms above. As stated in §20 they contain no likelihood, no optimizer
# call and no quadrature: `mode_solver_verdict` receives an already-computed
# offset and curvature and only rules on them; it never locates a mode.


def mode_solver_verdict(offset_in_scale_units, curvature, step_accepted,
                        iterations_used) -> str:
    """§9.2.5 — rule on one solver iterate. Fail-closed on every path."""
    if not isinstance(curvature, (int, float, np.floating, np.integer)) \
            or isinstance(curvature, bool):
        raise ValidationFailure("§9.2.5: curvature must be a real number")
    curvature = float(curvature)
    if not np.isfinite(curvature) or curvature <= CURVATURE_FLOOR:
        return "ADAPTIVE_CURVATURE_INVALID"

    d = float(offset_in_scale_units)
    if not np.isfinite(d):
        return "ADAPTIVE_MODE_NOT_FOUND"
    if d <= MODE_CONVERGENCE_TOL:
        return STATUS_OK
    if not step_accepted:
        # machine-precision stationarity of a strictly concave h, but only
        # inside the fail-closed cap
        return STATUS_OK if d <= MODE_STAGNATION_TOL else "ADAPTIVE_MODE_NOT_FOUND"
    if int(iterations_used) >= MODE_MAXITER:
        return "ADAPTIVE_MODE_NOT_FOUND"
    return "CONTINUE"


def validation_units(cell_registry=CELL_REGISTRY) -> tuple:
    """§9.6.5 — the 24 `(population, cell, block)` units, in registry order."""
    units = []
    for population in ("full_sample", "mdd_only"):
        for index, cell in enumerate(cell_registry):
            for pop, block in BLOCK_DF:
                if pop == population:
                    units.append((population, index, cell, block))
    if len(units) != N_VALIDATION_UNITS:
        raise ValidationFailure(
            f"§9.6.5: expected {N_VALIDATION_UNITS} validation units, built {len(units)}"
        )
    return tuple(units)


def is_numerical_failure(code: str) -> bool:
    """§9.5.1 — every registered non-OK code is numerical, never scientific."""
    if code not in STATUS_CODES:
        raise ValidationFailure(f"§9.5.1: {code!r} is not a registered status code")
    return code != STATUS_OK


def may_be_reported_as_null(code: str) -> bool:
    """§9.5.1 / F11 — no numerical failure may ever be read as a null."""
    return not is_numerical_failure(code)


def cv_logloss_status(fold_statuses) -> str:
    """§10.6 — all-or-nothing; a subset average is never the registered value."""
    statuses = list(fold_statuses)
    if not statuses:
        raise ValidationFailure("§10.6: an empty fold set is a failure, not a pass")
    if all(s == STATUS_OK for s in statuses):
        return STATUS_OK
    return "CV_FOLD_NOT_ESTIMABLE"


def lrt_status(statistic) -> str:
    """§12.1.1 — a negative statistic is numerical, and is never clamped."""
    value = float(statistic)
    if not np.isfinite(value):
        return "NON_FINITE_OBJECTIVE"
    if value < 0.0:
        return NEGATIVE_LRT_POLICY
    return STATUS_OK          # exact 0.0 is the admissible boundary, not an inversion


def is_authoritative_root(path) -> bool:
    """§16.1 — the formal root is a constant, compared as a normalized path."""
    from pathlib import PurePosixPath
    candidate = PurePosixPath(str(path).replace("\\", "/"))
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(PurePosixPath(str(ROOT)))
        except ValueError:
            return False
    return str(candidate).rstrip("/") == AUTHORITATIVE_RESULT_ROOT


def completion_barrier_order() -> tuple:
    """§16.2 — the fixed persistence order; the marker is always terminal."""
    return COMPLETION_ORDER


def completion_marker_is_permitted(steps_completed) -> bool:
    """§16.2 — the marker may be written only after every prior step passed."""
    completed = list(steps_completed)
    return completed == list(COMPLETION_ORDER[:-1])


# ---------------------------------------------------------------------- checks

def check_artifacts(report: list) -> None:
    for rel, expected in FROZEN_INPUTS.items():
        path = ROOT / rel
        if not path.exists():
            raise ValidationFailure(f"H1: missing consumed artifact {rel}")
        got = sha256_of(rel)
        if got != expected:
            raise ValidationFailure(f"H1: checksum mismatch for {rel}: {got}")
    ledger = {}
    for line in (ROOT / S0_LEDGER).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("#"):
            ledger[parts[-1].lstrip("*")] = parts[0]
    for base in S0_LEDGER_BOUND:
        rel = "results/rq1_s0/" + base
        if base not in ledger:
            raise ValidationFailure(f"H1: {base} absent from the frozen S0 ledger")
        if ledger[base] != FROZEN_INPUTS[rel]:
            raise ValidationFailure(f"H1: {base} disagrees with the frozen S0 ledger")
    report.append(f"artifacts: {len(FROZEN_INPUTS)} checksums match; "
                  f"{len(S0_LEDGER_BOUND)} bound to the S0 ledger")


def check_cohort(cohort: pd.DataFrame, report: list) -> None:
    if len(cohort) != BASE_COHORT_N or cohort.site.nunique() != BASE_COHORT_SITES:
        raise ValidationFailure(
            f"H2: base cohort is {len(cohort)} rows / {cohort.site.nunique()} sites"
        )
    report.append(f"cohort: {BASE_COHORT_N} INCLUDED acquisitions, "
                  f"{BASE_COHORT_SITES} eligible sites")


def check_anomaly_rule(cohort: pd.DataFrame, report: list) -> None:
    n_age0 = int((cohort.age == 0).sum())
    n_sexbad = int((~cohort.sex.isin([1, 2]) & cohort.sex.notna()).sum())
    n_hamd0 = int((cohort.hamd == 0).sum())
    n_dur0 = int((cohort.illness_duration_months == 0).sum())
    if (n_age0, n_sexbad) != (N_AGE_ZERO, N_SEX_INVALID):
        raise ValidationFailure(
            f"§5.2: anomaly counts changed — age==0 {n_age0}, sex invalid {n_sexbad}"
        )
    if (n_hamd0, n_dur0) != (N_HAMD_ZERO_KEPT, N_DURATION_ZERO_KEPT):
        raise ValidationFailure(
            f"§5.2: retained-zero counts changed — HAMD==0 {n_hamd0}, duration==0 {n_dur0}"
        )
    report.append(f"anomaly rule: {n_age0} age==0 and {n_sexbad} sex-invalid set missing; "
                  f"{n_hamd0} HAMD==0 and {n_dur0} duration==0 retained as observed")


def check_subsets(cohort: pd.DataFrame, report: list) -> tuple:
    full = full_sample_subset(cohort)
    if len(full) != FULL_CC_N:
        raise ValidationFailure(
            f"H3: full-sample complete case is {len(full)}, frozen value is {FULL_CC_N}"
        )
    if full.site.nunique() != FULL_CC_SITES:
        raise ValidationFailure(
            f"H3: full-sample site count is {full.site.nunique()}, frozen value is {FULL_CC_SITES}"
        )
    n_mdd = int((full.diagnosis == 1).sum())
    n_hc = int((full.diagnosis == 0).sum())
    if (n_mdd, n_hc) != (FULL_CC_MDD, FULL_CC_HC):
        raise ValidationFailure(
            f"H3: full-sample composition is {n_mdd}/{n_hc}, frozen is {FULL_CC_MDD}/{FULL_CC_HC}"
        )

    mdd = mdd_only_subset(cohort)
    if len(mdd) != MDD_CC_N:
        raise ValidationFailure(
            f"H4: MDD-only complete case is {len(mdd)}, frozen value is {MDD_CC_N}"
        )
    if tuple(sorted(mdd.site.unique(), key=lambda s: int(s[1:]))) != MDD_CC_SITES:
        raise ValidationFailure(f"H4: MDD-only site set is {sorted(mdd.site.unique())}")
    per_site = {s: int(n) for s, n in mdd.groupby("site").size().items()}
    if per_site != MDD_CC_PER_SITE:
        raise ValidationFailure(f"H4: MDD-only per-site distribution changed: {per_site}")

    report.append(f"full-sample complete case: {FULL_CC_N}/{BASE_COHORT_N}, "
                  f"{FULL_CC_SITES} sites, {FULL_CC_MDD} MDD / {FULL_CC_HC} HC")
    report.append(f"MDD-only complete case: {MDD_CC_N}/1182, "
                  f"{len(MDD_CC_SITES)} sites, S20 share "
                  f"{MDD_CC_PER_SITE['S20'] / MDD_CC_N:.3f}")
    return full, mdd


def check_outcome_reproduces_s0(full: pd.DataFrame, report: list) -> None:
    """§2.4 / H5 — reproduce the frozen S0 operating-point coverage.

    This is the one place R_i is touched at Stage E.1. The quantity compared is
    a published RQ1/S0 result; no RQ3 quantity is produced.
    """
    degenerate = []
    for op in OPERATING_POINTS:
        for method in CONFIDENCE_METHODS:
            rows = reconstruct_R(method, op)
            per_site = rows.groupby("site").R.agg(["size", "sum"])
            coverage = (1.0 - per_site["sum"] / per_site["size"]).mean()
            frozen = frozen_site_balanced_coverage(op, method)
            if abs(coverage - frozen) > COVERAGE_TOL:
                raise ValidationFailure(
                    f"H5: {op}/{method} coverage {coverage!r} != frozen {frozen!r}"
                )
            in_full = rows[rows.canonical_id.isin(set(full.canonical_id))]
            g = in_full.groupby("site").R.agg(["size", "sum"])
            deg = g[(g["sum"] == 0) | (g["sum"] == g["size"])]
            degenerate.extend((op, method, s) for s in deg.index)
    if degenerate:
        raise ValidationFailure(
            f"full-sample subset has all-rejected/all-retained sites: {degenerate}"
        )
    report.append(f"outcome: R_i reproduces frozen OP50/OP30 site-balanced coverage in "
                  f"all {len(OPERATING_POINTS) * len(CONFIDENCE_METHODS)} cells "
                  f"(tol {COVERAGE_TOL:g}); no degenerate site in the full-sample subset")


def check_coding(full: pd.DataFrame, mdd: pd.DataFrame, report: list) -> None:
    if full.sex.map(SEX_MAP).isna().any():
        raise ValidationFailure("H6: sex map is not total on the full-sample subset")
    if full.diagnosis.map(DIAGNOSIS_MAP).isna().any():
        raise ValidationFailure("H6: diagnosis map is not total on the full-sample subset")
    for var in ("first_episode", "on_medication"):
        if mdd[var].map(BINARY_CLINICAL_MAP).isna().any():
            raise ValidationFailure(f"H6: {var} map is not total on the MDD-only subset")
    for frame, name in ((full, "full-sample"), (mdd, "MDD-only")):
        for var in LOG1P_VARS:
            if var not in frame.columns:
                continue
            values = frame[var].dropna()
            if name == "full-sample" and var == "illness_duration_months":
                continue  # not a covariate of the full-sample model
            if (values < 0).any():
                raise ValidationFailure(f"H7: negative log1p argument in {name}/{var}")
    report.append("coding: §8 maps total on both frozen subsets; log1p arguments non-negative")


def check_loso_folds(full: pd.DataFrame, mdd: pd.DataFrame, report: list) -> None:
    """§10.5 — predictor-only degeneracy screen. R_i is not read here."""
    def screen(frame, continuous, binaries, label):
        bad = []
        for site in sorted(frame.site.unique(), key=lambda s: int(s[1:])):
            train = frame[frame.site != site]
            if train.site.nunique() < 2:
                bad.append((site, "site", "<2 training sites"))
            for var, transform in continuous:
                if transform(train[var]).std(ddof=0) <= 0:
                    bad.append((site, var, "zero SD"))
            for var, mapping in binaries:
                if train[var].map(mapping).nunique() < 2:
                    bad.append((site, var, "single level"))
        if bad:
            raise ValidationFailure(f"§10.5: degenerate {label} training folds: {bad}")
        return frame.site.nunique()

    identity = lambda s: s
    n_full = screen(
        full,
        [("age", identity), ("mean_fd", np.log1p)],
        [("sex", SEX_MAP), ("diagnosis", DIAGNOSIS_MAP)],
        "full-sample",
    )
    n_mdd = screen(
        mdd,
        [("hamd", identity), ("mean_fd", np.log1p),
         ("illness_duration_months", np.log1p)],
        [("first_episode", BINARY_CLINICAL_MAP),
         ("on_medication", BINARY_CLINICAL_MAP)],
        "MDD-only",
    )
    report.append(f"LOSO folds: {n_full} full-sample and {n_mdd} MDD-only training folds, "
                  f"none predictor-degenerate")


def check_multiplicity_shape(report: list) -> None:
    if len(HOLM_FAMILIES) != 4 or any(size != 6 for *_, size in HOLM_FAMILIES):
        raise ValidationFailure("H12: Holm family structure is not 4 families of 6")
    expected_cells = len(CONFIDENCE_METHODS) * len(OPERATING_POINTS)
    if expected_cells != 6:
        raise ValidationFailure(f"§2.4: expected 6 cells, got {expected_cells}")
    report.append(f"multiplicity: alpha={ALPHA}, 4 Holm families of 6; "
                  f"block df {dict((f'{p}/{b}', d) for (p, b), d in BLOCK_DF.items())}")


def check_rejected_fraction_rule(report: list) -> None:
    """§15.1 — the registered fraction is site-balanced, provably not pooled."""
    # deliberately unequal site sizes, as in the real subsets (24..531, 2..236)
    sites = ["A"] * 10 + ["B"] * 2
    rejections = [1] * 10 + [0] * 2          # pooled 10/12; site-balanced (1.0 + 0.0)/2
    balanced = site_balanced_rejected_fraction(sites, rejections, expected_sites=2)
    pooled = pooled_rejected_fraction(rejections)
    if abs(balanced - 0.5) > 1e-12:
        raise ValidationFailure(f"H14: site-balanced fraction is {balanced!r}, expected 0.5")
    if abs(pooled - 10.0 / 12.0) > 1e-12:
        raise ValidationFailure(f"pooled audit fraction is {pooled!r}")
    if abs(balanced - pooled) < 1e-6:
        raise ValidationFailure("§15.1: the two fractions must be distinguishable here")
    report.append(f"rejected_fraction: site-balanced ({REJECTED_FRACTION_MODE}); "
                  f"unequal site sizes give {balanced:.4f}, not the pooled {pooled:.4f}; "
                  f"denominators S = {SITE_BALANCED_DENOMINATOR}")


def check_holm_semantics(report: list) -> None:
    """§12.4 — six slots always, placeholder internal, NOT_ESTIMABLE never significant."""
    estimable = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]
    adjusted = holm_adjust_family(estimable)
    expected = []
    running = 0.0
    for rank, value in enumerate(sorted(estimable)):
        running = max(running, min(1.0, (HOLM_FAMILY_SIZE - rank) * value))
        expected.append(running)
    # the fixture is already ascending, so slot order coincides with sorted order
    if any(abs(a - e) > 1e-12 for a, e in zip(adjusted, expected)):
        raise ValidationFailure(f"§12.4: six-slot Holm arithmetic is wrong: {adjusted}")
    if any(a > b for a, b in zip(adjusted, adjusted[1:])):
        raise ValidationFailure("§12.4: adjusted p-values are not monotone in sorted order")

    with_gaps = [0.001, NOT_ESTIMABLE, 0.02, NOT_ESTIMABLE, 0.04, NOT_ESTIMABLE]
    mixed = holm_adjust_family(with_gaps)
    for i, entry in enumerate(with_gaps):
        if entry == NOT_ESTIMABLE and mixed[i] != NOT_ESTIMABLE:
            raise ValidationFailure("H15: an unestimable slot was serialized as a number")
        if entry == NOT_ESTIMABLE and holm_is_significant(entry, mixed[i]):
            raise ValidationFailure("H15: an unestimable cell was declared significant")
    # m must not shrink: the smallest p is still multiplied by 6, not by 3
    if abs(mixed[0] - min(1.0, HOLM_FAMILY_SIZE * 0.001)) > 1e-12:
        raise ValidationFailure(f"H12: m shrank below {HOLM_FAMILY_SIZE}: {mixed[0]!r}")

    all_gaps = holm_adjust_family([NOT_ESTIMABLE] * HOLM_FAMILY_SIZE)
    if any(entry != NOT_ESTIMABLE for entry in all_gaps):
        raise ValidationFailure("§12.4: an all-unestimable family produced numbers")
    report.append(f"Holm: fixed {HOLM_FAMILY_SIZE}-slot families; internal placeholder "
                  f"p={HOLM_PLACEHOLDER_P} never serialized; NOT_ESTIMABLE never significant")


def check_initializer(report: list) -> None:
    """§9.4 — deterministic, closed-form, no secondary solver."""
    starts = deterministic_initial_points([1, 0, 1, 0], n_non_intercept=4)
    if len(starts) != len(SIGMA_STARTS):
        raise ValidationFailure(f"H16: {len(starts)} starts, frozen count is {len(SIGMA_STARTS)}")
    betas = [beta for beta, _ in starts]
    if any(not np.array_equal(beta, betas[0]) for beta in betas):
        raise ValidationFailure("H16: the four starts differ in more than log_sigma")
    if any(abs(beta[1:]).max() != 0.0 for beta in betas):
        raise ValidationFailure("H16: a non-intercept slope did not start at exactly 0")
    if abs(betas[0][0] - 0.0) > 1e-12:      # prevalence 0.5 -> log-odds 0
        raise ValidationFailure(f"H16: intercept start is {betas[0][0]!r}, expected 0.0")
    if [np.exp(ls) for _, ls in starts] != pytest_approx_list(SIGMA_STARTS):
        raise ValidationFailure("H16: sigma starts are not the frozen set")

    # degenerate fitting samples are handled by the clip, not by a fallback
    all_rejected = deterministic_initial_points([1, 1, 1], n_non_intercept=0)
    expected = np.log((1.0 - INIT_PREVALENCE_CLIP) / INIT_PREVALENCE_CLIP)
    if abs(all_rejected[0][0][0] - expected) > 1e-9:
        raise ValidationFailure("H16: the prevalence clip was not applied at exactly 1e-6")
    report.append(f"initializer: clip {INIT_PREVALENCE_CLIP:g}, intercept = clipped empirical "
                  f"log-odds, zero slopes, sigma starts {SIGMA_STARTS}")


def pytest_approx_list(values, tol: float = 1e-12) -> list:
    """Small helper so the initializer check reads as an exact-set comparison."""
    return [_Approx(v, tol) for v in values]


class _Approx:
    def __init__(self, value: float, tol: float):
        self.value = float(value)
        self.tol = tol

    def __eq__(self, other) -> bool:
        return abs(float(other) - self.value) <= self.tol

    def __repr__(self) -> str:
        return f"~{self.value}"


def check_agreement_procedure(report: list) -> None:
    """§9.6 — every registered criterion has a defined formula and both boundaries."""
    base = {
        "loglik": -100.0, "lrt_statistic": 4.0, "lrt_pvalue": 0.05,
        "mcfadden": 0.10, "cv_logloss": 0.60,
        "sigma_site": 0.5, "beta": np.array([0.1, -0.2, 0.3]),
    }
    identical = agreement_checks(base, dict(base))
    if tuple(identical) != REQUIRED_AGREEMENT_CRITERIA:
        raise ValidationFailure(
            f"§9.6.4: evaluated criteria {tuple(identical)} != required registry"
        )
    verdict = numerical_validation_verdict(base, dict(base))
    if not (verdict["passed"] and verdict["n_evaluated"] == N_REQUIRED_AGREEMENT_CRITERIA
            and verdict["n_passed"] == N_REQUIRED_AGREEMENT_CRITERIA):
        raise ValidationFailure(f"§9.6.4: identical solutions did not pass 7-of-7: {verdict}")

    # ---- fail-closed: nothing may pass by being absent (§9.6.4) ----
    if passes_numerical_validation({}, {}):
        raise ValidationFailure("§9.6.4: empty solutions passed numerical validation")
    if passes_numerical_validation({"loglik": -100.0}, {"loglik": -100.0}):
        raise ValidationFailure("§9.6.4: a partial solution passed numerical validation")
    for criterion in REQUIRED_AGREEMENT_CRITERIA:
        for field in AGREEMENT_INPUT_FIELDS[criterion]:
            stripped = {k: val for k, val in base.items() if k != field}
            if passes_numerical_validation(stripped, dict(base)):
                raise ValidationFailure(
                    f"§9.6.4: dropping {field!r} still passed — {criterion} was skipped"
                )
            if numerical_validation_verdict(stripped, dict(base))["n_evaluated"] != 0:
                raise ValidationFailure(
                    f"§9.6.4: dropping {field!r} produced a partial evaluation"
                )
    for bad in (None, float("nan"), float("inf"), float("-inf"), "x"):
        broken = dict(base)
        broken["loglik"] = bad
        if passes_numerical_validation(broken, dict(base)):
            raise ValidationFailure(f"§9.6.4: a {bad!r} log-likelihood passed")
    for bad_beta in (None, np.array([]), np.array([[0.1, 0.2]]), np.array([0.1, np.nan])):
        broken = dict(base)
        broken["beta"] = bad_beta
        if passes_numerical_validation(broken, dict(base)):
            raise ValidationFailure("§9.6.4: a malformed coefficient vector passed")
    mismatched = dict(base)
    mismatched["beta"] = np.array([0.1, -0.2])
    if passes_numerical_validation(mismatched, dict(base)):
        raise ValidationFailure("§9.6.4: mismatched coefficient lengths passed")

    # the relative sigma^2 formula, denominated on the 60-node value
    if abs(sigma2_relative_difference(0.0, 0.0)) != 0.0:
        raise ValidationFailure("§9.6.2: the 1e-12 floor did not protect a zero denominator")
    if abs(sigma2_relative_difference(2.0, 1.0) - 3.0) > 1e-12:
        raise ValidationFailure("§9.6.2: relative sigma^2 formula is wrong")

    # each criterion must fail on its own when nudged past its tolerance
    for criterion, tolerance in AGREEMENT_TOL.items():
        perturbed = dict(base)
        perturbed["beta"] = np.array(base["beta"], copy=True)
        if criterion == "sigma_site_sq_relative":
            # sigma^2 must move by more than `tolerance` relatively
            perturbed["sigma_site"] = float(np.sqrt(base["sigma_site"] ** 2 * (1 + 10 * tolerance)))
        elif criterion == "fixed_effect_coefficients":
            perturbed["beta"] = base["beta"] + np.array([0.0, 0.0, 10 * tolerance])
        else:
            perturbed[criterion] = base[criterion] + 10 * tolerance
        breached = agreement_checks(base, perturbed)
        if breached[criterion][2]:
            raise ValidationFailure(f"§9.6: {criterion} did not fail past its tolerance")
        if passes_numerical_validation(base, perturbed):
            raise ValidationFailure(f"§9.6.1: a {criterion} breach did not fail the fit")
    report.append(f"40-vs-60: fail-closed {N_REQUIRED_AGREEMENT_CRITERIA}-of-"
                  f"{N_REQUIRED_AGREEMENT_CRITERIA} registry; empty, partial, non-finite and "
                  f"malformed inputs all FAIL; {AUTHORITATIVE_NODES}-node stays authoritative")


def check_best_start_rule(report: list) -> None:
    """§9.4.1 — the frozen best-start rule, machine-enforced."""
    def start(index, eligible, loglik):
        return {"index": index, "sigma_start": START_INDEX_TO_SIGMA[index],
                "eligible": eligible, "loglik": loglik}

    four = [start(0, True, -101.0), start(1, True, -100.0),
            start(2, True, -102.0), start(3, True, -103.0)]
    if select_best_start(four) != 1:
        raise ValidationFailure("§9.4.1: the maximum marginal log-likelihood did not win")

    # an ineligible start may never win, however large its log-likelihood
    tempting = [start(0, True, -101.0), start(1, False, 0.0),
                start(2, True, -102.0), start(3, True, -103.0)]
    if select_best_start(tempting) != 0:
        raise ValidationFailure("H16: an ineligible start won the selection")

    # exact ties break to the lowest registered index
    tied = [start(i, True, -100.0) for i in range(4)]
    if select_best_start(tied) != 0:
        raise ValidationFailure("§9.4.1: an exact tie did not break to the lowest index")

    # all ineligible => NOT_ESTIMABLE, never a least-bad fit
    if select_best_start([start(i, False, -100.0) for i in range(4)]) != NOT_ESTIMABLE:
        raise ValidationFailure("§9.4.1: an all-ineligible family did not return NOT_ESTIMABLE")

    # a non-finite log-likelihood cannot win even when the start claims eligibility
    for bad in (float("nan"), float("inf"), float("-inf"), None, "x"):
        poisoned = [start(0, True, bad), start(1, True, -100.0),
                    start(2, False, -1.0), start(3, False, -1.0)]
        if select_best_start(poisoned) != 1:
            raise ValidationFailure(f"§9.4.1: a {bad!r} log-likelihood was selectable")

    # ---- structural identity violations fail the selector (§9.4.1, H19) ----
    def canonical():
        return {i: start(i, True, -100.0) for i in range(4)}

    def contradicted(slot, **overrides):
        family = canonical()
        family[slot] = {**family[slot], **overrides}
        return family

    identity_violations = [
        ({i: start(i, True, -100.0) for i in range(3)}, "missing slot"),
        ({**canonical(), 4: {"index": 4, "sigma_start": 0.1,
                             "eligible": True, "loglik": 0.0}}, "fifth slot"),
        (contradicted(0, index=1), "mapping key vs embedded index"),
        (contradicted(1, index=0), "embedded index vs mapping key"),
        (contradicted(1, index=True), "bool embedded index"),
        (contradicted(1, index=False), "bool embedded index (False)"),
        (contradicted(0, sigma_start=float("nan")), "NaN sigma_start"),
        (contradicted(0, sigma_start=float("inf")), "+inf sigma_start"),
        (contradicted(0, sigma_start=float("-inf")), "-inf sigma_start"),
        (contradicted(0, sigma_start=None), "None sigma_start"),
        (contradicted(2, sigma_start=True), "bool sigma_start"),
        (contradicted(2, sigma_start=False), "bool sigma_start (False)"),
        (contradicted(0, sigma_start="0.1"), "string sigma_start"),
        (contradicted(0, sigma_start=0.5), "index 0 with sigma 0.5"),
        (contradicted(3, sigma_start=1.0), "index 3 with sigma 1.0"),
        ([start(0, True, -100.0)] * 4, "duplicate registered index"),
    ]
    for broken, label in identity_violations:
        try:
            select_best_start(broken)
        except ValidationFailure:
            continue
        raise ValidationFailure(f"H19: a {label} did not fail closed")

    missing_sigma = canonical()
    missing_sigma[1] = {k: val for k, val in missing_sigma[1].items() if k != "sigma_start"}
    try:
        select_best_start(missing_sigma)
    except ValidationFailure:
        pass
    else:
        raise ValidationFailure("H19: an absent sigma_start did not fail closed")

    # ---- positional sequence identity (§9.4.2): position is the index ----
    def bare(index, loglik=-100.0):
        """a positional record that declares no embedded index"""
        return {"sigma_start": START_INDEX_TO_SIGMA[index], "eligible": True, "loglik": loglik}

    # a canonical positional sequence carrying no embedded index is valid, and
    # position alone supplies the registered identity
    if select_best_start([bare(0, -101.0), bare(1, -100.0),
                          bare(2, -102.0), bare(3, -103.0)]) != 1:
        raise ValidationFailure("§9.4.2: a canonical index-free positional sequence was rejected")

    positional_violations = [
        ([start(1, True, -100.0), start(0, True, -101.0),
          start(2, True, -102.0), start(3, True, -103.0)], "positions 0 and 1 swapped"),
        ([start(i, True, -100.0) for i in (3, 2, 1, 0)], "fully reversed sequence"),
        ([{**bare(0), "index": 1}, start(1, True, -100.0),
          start(2, True, -100.0), start(3, True, -100.0)], "position 0 declaring index 1"),
        ([start(0, True, -100.0), {**bare(1), "index": 0},
          start(2, True, -100.0), start(3, True, -100.0)], "position 1 declaring index 0"),
        ([start(0, True, -100.0), start(1, True, -100.0),
          {**bare(2), "index": 3}, start(3, True, -100.0)], "position 2 declaring index 3"),
        ([start(0, True, -100.0), start(1, True, -100.0),
          start(2, True, -100.0), {**bare(3), "index": 2}], "position 3 declaring index 2"),
        ([bare(1), bare(0), bare(2), bare(3)], "index-free sequence with permuted sigmas"),
        ([bare(0), bare(1), bare(2)], "sequence of length 3"),
        ([bare(i) for i in range(4)] + [bare(0)], "sequence of length 5"),
    ]
    for broken, label in positional_violations:
        try:
            select_best_start(broken)
        except ValidationFailure:
            continue
        raise ValidationFailure(f"H19: a {label} did not fail closed")

    # order of presentation cannot change the answer — for a *mapping*, whose
    # identity travels in the key rather than in the position
    shuffled = {3: four[3], 1: four[1], 0: four[0], 2: four[2]}
    if select_best_start(shuffled) != 1:
        raise ValidationFailure("§9.4.1: the result depended on mapping insertion order")

    report.append(f"best start: {N_REGISTERED_STARTS} registered identities "
                  f"{START_INDEX_TO_SIGMA} enforced exactly "
                  f"({len(identity_violations) + len(positional_violations) + 1} identity "
                  f"corruptions fail closed, sequence position binding); max eligible logLik "
                  f"wins, ties → lowest index; all-ineligible → {NOT_ESTIMABLE}")


DOC_REQUIRED_TOKENS = {
    "OP50": "`OP50`",
    "OP30": "`OP30`",
    "source-intended semantics": "source-intended coverage of 50 % / 70 %",
    "confidence method probability": "probability",
    "confidence method bootstrap": "bootstrap",
    "confidence method mahalanobis": "mahalanobis",
    "full-sample complete case": "2222",
    "MDD-only complete case": "484",
    "primary quadrature": "`K = 40`",
    "validation quadrature": "`K = 60`",
    "sigma starts": "{0.1, 0.5, 1.0, 2.0}",
    "initializer clip": "1e-6",
    "sigma^2 denominator floor": "1e-12",
    "optimizer": "L-BFGS-B",
    # pinned as the whole table row, so the value cannot drift while the
    # historical `1e-12` mentions in §9.4.5 / §21.3 keep a bare literal matching
    "optimizer ftol": "| `ftol` | `1e-16`",
    "optimizer gtol row": "| `gtol` | `1e-8`",
    "ftol amendment rationale": "Why `ftol` is `1e-16` — FROZEN (Stage E.2a.1)",
    "gtol is operative": "**`gtol` becomes the operative stopping criterion**",
    "eligibility not relaxed": "**The eligibility criterion\nwas not relaxed to admit more fits**",
    "optimizer gtol": "`1e-8`",
    "maxiter": "`20000`",
    "maxfun": "`50000`",
    "gradient tolerance": "1e-5",
    "alpha": "`alpha = 0.05`",
    "Holm family size": "m = 6",
    "rejected fraction rule": "site-balanced",
    "unestimable sentinel": "NOT_ESTIMABLE",
    # the §9.6.3 rows are pinned whole, so a comparison type cannot drift away
    # from its value without failing here
    "loglik tolerance": "| marginal log-likelihood | absolute | `1e-4` |",
    "LRT statistic tolerance": "| LRT statistic | absolute | `1e-3` |",
    "LRT p-value tolerance": "| LRT p-value | absolute | `1e-4` |",
    "McFadden tolerance": "| McFadden R² and ΔR² | absolute | `1e-4` |",
    "relative sigma^2 tolerance": "| `sigma_site^2` | relative, §9.6.2 | `1e-3` |",
    "CV log-loss tolerance": "| CV log-loss and Δ CV log-loss | absolute | `1e-6` |",
    "coefficient tolerance": "| fixed-effect coefficients | sup-norm, §9.6.2 | `1e-4` |",
    "block df full motion": "| full-sample | motion | 1 |",
    "block df full diagnosis": "| full-sample | diagnosis | 1 |",
    "block df MDD motion": "| MDD-only | motion | 1 |",
    "block df MDD clinical": "| MDD-only | clinical | 4 |",
    # Stage E.1b enforcement rules
    "seven-of-seven rule": "n_evaluated == 7  AND  n_passed == 7",
    "fail-closed heading": "The validation is fail-closed",
    "inclusive boundary": "`discrepancy <= tolerance`",
    "best-start heading": "Best-start selection — FROZEN (Stage E.1b)",
    "start slot 0": "| 0 | `0.1` |",
    "start slot 1": "| 1 | `0.5` |",
    "start slot 2": "| 2 | `1.0` |",
    "start slot 3": "| 3 | `2.0` |",
    "lowest-index tie-break": "select the **lowest registered start index**",
    # Stage E.1c start-identity rules
    "identity heading": "Registered start identity — FROZEN (Stage E.1c)",
    "canonical registry": "single canonical registry of registered start identities",
    "key/embedded agreement": "**must agree exactly**",
    "bool index rejected": "`bool` is **rejected explicitly**",
    "finiteness before equality": "established **before** any equality comparison",
    "identity vs likelihood": "Identity corruption fails the **selector**",
    # Stage E.1d positional sequence rules
    "positional heading": "Positional sequence identity — FROZEN (Stage E.1d)",
    "position defines index": "the sequence position defines the",
    "sequence length": "The sequence length must be **exactly 4**",
    "embedded equals position": "that field **must equal its position",
    "no reordering": "**never reorders a sequence by embedded index**",
    "permutation is corruption": "identity**, not a reordering instruction",
    "positional slot 0": "| 0 | 0 | `0.1` |",
    "positional slot 1": "| 1 | 1 | `0.5` |",
    "positional slot 2": "| 2 | 2 | `1.0` |",
    "positional slot 3": "| 3 | 3 | `2.0` |",
    "mapping unchanged": "**Mapping input is unchanged** by this section",
    # ---------------------------------------------- Stage E.2a amended numerics
    "adaptive heading": "mode/curvature-adaptive Gauss–Hermite",
    "adaptive mode definition": "u_hat_s = argmax_u h_s(u)",
    "adaptive scale definition": "tau_s   = H_hat_s^(−1/2)",
    "curvature rule": "H_s(u)   = −h_s''(u) = S2_s(u) + 1/sigma²",
    "curvature positivity": "`H_s(u) ≥ 1/sigma² > 0` for every `u`",
    "node transformation": "u_{s,k} = u_hat_s + sqrt(2)·tau_s·z_k",
    "jacobian and deweighting": "log( sqrt(2)·tau_s ) + logsumexp_k [ log w_k + z_k² + h_s( u_{s,k} ) ]",
    "reduction identity": "Reduction to the superseded rule",
    "non-adaptive withdrawn": "non-adaptive rule is withdrawn",
    "mode solver heading": "The mode solver — FROZEN, deterministic",
    "mode solver init": "u^(0) = 0",
    "mode solver tol": "if d_j <= 1e-8:",
    "mode solver stagnation cap": "if d_j <= 1e-4:",
    "mode solver budget": "(iteration budget = 200)",
    "mode solver halvings": "for at most 50 halvings:",
    "mode solver curvature floor": "if H_j is not finite, or H_j <= 1e-12:",
    "mode tolerances derived": "Both tolerances are derived, not tuned",
    "gradient heading": "The gradient of the adaptive objective — FROZEN (Stage E.2a)",
    "gradient is exact total derivative": "**exact total derivative**",
    "gradient mode movement": "∂u_hat_s/∂beta_j     = −Q_{s,j} / H_hat_s",
    "gradient scale movement": "∂log tau_s/∂log_sigma = −0.5·( S3_s·∂u_hat_s/∂log_sigma − 2/sigma² ) / H_hat_s",
    "no finite differencing": "**No numerical differentiation is registered anywhere.**",
    "validation unit heading": "The validation unit — FROZEN (Stage E.2a)",
    "validation unit triple": "( population , cell , block )",
    "validation unit count": "**24 validation units in total** (2 × 6 × 2)",
    "status code registry": "Registered status and failure codes — FROZEN (Stage E.2a)",
    "numerical not scientific": "**Every code in this registry is a numerical outcome.**",
    "structural abort separate": "**Structural corruption is never mapped onto these codes.**",
    "cv all or nothing": "A CV quantity is all-or-nothing — FROZEN (Stage E.2a)",
    "cv subset forbidden": "**Averaging over the surviving folds is forbidden**",
    "lrt inversion": "Likelihood inversion — FROZEN (Stage E.2a)",
    "lrt no clamp": "may not be clamped, floored, truncated at zero",
    "authoritative root": "**The authoritative RQ3 result root is exactly `results/rq3/`.**",
    "no caller root": "no**\n   `--store-root`",
    "stale root fails closed": "**A stale root fails closed.**",
    "completion terminal": "`COMPLETION.json` is the **terminal** artifact of a formal run",
    "completion reread": "**re-open and re-read the persisted artifacts and the ledger from disk**",
    "completion atomic": "**atomically**",
    "amendment history": "## 21. Amendment history",
    "v1 immutable": "remains immutable",
    "amendment pre-result": "no real RQ3 model had ever been fitted",
}


def check_document_constants(report: list) -> None:
    """§18.2 — the constants are in the document, not only in the code."""
    document = (ROOT / "docs/RQ3_PROTOCOL.md").read_text()
    missing = [label for label, token in DOC_REQUIRED_TOKENS.items() if token not in document]
    if missing:
        raise ValidationFailure(f"§18.2: protocol document omits {missing}")
    if PROTOCOL_VERSION not in document:
        raise ValidationFailure(f"§18.2: document does not carry {PROTOCOL_VERSION}")
    # the withdrawn Stage E.1 initializer may only appear where its withdrawal is
    # recorded, never as a live instruction
    withdrawn = "plain logistic-regression MLE"
    removal_markers = ("removed", "depended on", "Ambiguity", "no longer", "not itself pinned")
    position = document.find(withdrawn)
    while position != -1:
        window = document[max(0, position - 400):position + 400]
        if not any(marker in window for marker in removal_markers):
            raise ValidationFailure(
                "§9.4: the withdrawn initializer phrase appears outside a removal record"
            )
        position = document.find(withdrawn, position + 1)
    report.append(f"constants: {len(DOC_REQUIRED_TOKENS)} execution-controlling values present "
                  f"in both the document and the validator")


def check_amended_quadrature(report: list) -> None:
    """§9.2 — the site integral is adaptive, and the old rule stays withdrawn."""
    if not QUADRATURE_IS_ADAPTIVE or QUADRATURE_RULE != "mode_curvature_adaptive_gauss_hermite":
        raise ValidationFailure("§9.2: the registered quadrature rule is not adaptive")
    if GH_NODES_PRIMARY != 40 or GH_NODES_VALIDATION != 60:
        raise ValidationFailure("§9.2: node counts must remain 40 authoritative / 60 validation")
    if AUTHORITATIVE_NODES != GH_NODES_PRIMARY:
        raise ValidationFailure("§9.6.1: the 40-node fit must remain authoritative")
    if len(ADAPTIVE_TRANSFORM_TERMS) != 4:
        raise ValidationFailure("§9.2.3: the transformation has exactly four structural terms")

    document = (ROOT / "docs/RQ3_PROTOCOL.md").read_text()
    # every structural term of §9.2.3 must be visible in the document
    for token in ("u_hat_s + sqrt(2)·tau_s·z_k", "tau_s   = H_hat_s^(−1/2)",
                  "log( sqrt(2)·tau_s ) + logsumexp_k [ log w_k + z_k² + h_s( u_{s,k} ) ]"):
        if token not in document:
            raise ValidationFailure(f"§9.2.3: the document omits {token!r}")
    # the withdrawn rule may appear only where it is recorded as withdrawn
    if "**The\nnon-adaptive rule is withdrawn**" not in document \
            and "non-adaptive rule is withdrawn" not in document:
        raise ValidationFailure("§9.2.4: the withdrawal of the non-adaptive rule is not recorded")
    report.append(
        f"quadrature: {QUADRATURE_RULE}, K={GH_NODES_PRIMARY} authoritative / "
        f"{GH_NODES_VALIDATION} validation; all four §9.2.3 terms present; "
        f"{WITHDRAWN_QUADRATURE_RULE} withdrawn"
    )


def check_mode_solver(report: list) -> None:
    """§9.2.5 — the solver is deterministic, and fail-closed on every path."""
    if MODE_INIT != 0.0:
        raise ValidationFailure("§9.2.5: the mode solver starts at exactly u = 0")
    if MODE_FALLBACK_SOLVER_PERMITTED:
        raise ValidationFailure("§9.2.5: no fallback solver is permitted")
    if not (MODE_CONVERGENCE_TOL < MODE_STAGNATION_TOL):
        raise ValidationFailure("§9.2.5: the stagnation cap must be looser than the primary tol")
    # the derived cost of each tolerance must stay below the tightest §9.6.3 value
    tightest = min(AGREEMENT_TOL.values())
    for tol in (MODE_CONVERGENCE_TOL, MODE_STAGNATION_TOL):
        if 0.5 * tol ** 2 >= tightest:
            raise ValidationFailure(
                f"§9.2.5: a mode offset of {tol} costs {0.5 * tol ** 2:.3g} in logLik, "
                f"which is not below the tightest registered tolerance {tightest}"
            )

    # curvature invalidity dominates, whatever the offset
    if mode_solver_verdict(0.0, 0.0, True, 0) != "ADAPTIVE_CURVATURE_INVALID":
        raise ValidationFailure("§9.2.5: zero curvature must fail closed")
    if mode_solver_verdict(0.0, np.nan, True, 0) != "ADAPTIVE_CURVATURE_INVALID":
        raise ValidationFailure("§9.2.5: non-finite curvature must fail closed")
    if mode_solver_verdict(0.0, CURVATURE_FLOOR, True, 0) != "ADAPTIVE_CURVATURE_INVALID":
        raise ValidationFailure("§9.2.5: curvature at the floor must fail closed")
    # convergence, continuation, stagnation and budget exhaustion
    if mode_solver_verdict(MODE_CONVERGENCE_TOL, 1.0, True, 0) != STATUS_OK:
        raise ValidationFailure("§9.2.5: the tolerance boundary is inclusive")
    if mode_solver_verdict(1e-6, 1.0, True, 0) != "CONTINUE":
        raise ValidationFailure("§9.2.5: an unconverged accepted step must continue")
    if mode_solver_verdict(1e-6, 1.0, False, 0) != STATUS_OK:
        raise ValidationFailure("§9.2.5: stagnation inside the cap is convergence")
    if mode_solver_verdict(1e-3, 1.0, False, 0) != "ADAPTIVE_MODE_NOT_FOUND":
        raise ValidationFailure("§9.2.5: stagnation beyond the cap must fail closed")
    if mode_solver_verdict(1e-6, 1.0, True, MODE_MAXITER) != "ADAPTIVE_MODE_NOT_FOUND":
        raise ValidationFailure("§9.2.5: an exhausted iteration budget must fail closed")
    if mode_solver_verdict(np.nan, 1.0, True, 0) != "ADAPTIVE_MODE_NOT_FOUND":
        raise ValidationFailure("§9.2.5: a non-finite offset must fail closed")
    report.append(
        f"mode solver: {MODE_SOLVER} from u={MODE_INIT}, tol {MODE_CONVERGENCE_TOL} "
        f"(cap {MODE_STAGNATION_TOL}), <= {MODE_MAXITER} iterations, "
        f"<= {MODE_MAX_HALVINGS} halvings, curvature floor {CURVATURE_FLOOR}; "
        f"9 fail-closed paths enforced"
    )


def check_gradient_rule(report: list) -> None:
    """§9.4.4 — the exact total derivative, with node movement, no differencing."""
    if GRADIENT_RULE != "exact_total_derivative":
        raise ValidationFailure("§9.4.4: the gradient must be the exact total derivative")
    if GRADIENT_FINITE_DIFFERENCE_PERMITTED:
        raise ValidationFailure("§9.4.4: finite differencing is not registered")
    if len(GRADIENT_TERMS) != 4:
        raise ValidationFailure("§9.4.4: the total derivative has exactly four terms")
    document = (ROOT / "docs/RQ3_PROTOCOL.md").read_text()
    for token in ("∂u_hat_s/∂beta_j     = −Q_{s,j} / H_hat_s",
                  "∂log tau_s/∂beta_j    = −0.5·( T_{s,j} + S3_s·∂u_hat_s/∂beta_j ) / H_hat_s",
                  "∂h_s(u)/∂log_sigma = u²/sigma² − 1"):
        if token not in document:
            raise ValidationFailure(f"§9.4.4: the document omits {token!r}")
    if OPTIMIZER != "L-BFGS-B":
        raise ValidationFailure("§9.4: the optimizer identity is unchanged by the amendment")
    report.append(
        f"gradient: {GRADIENT_RULE} of the adaptive objective, all 4 terms "
        f"({', '.join(GRADIENT_TERMS)}); finite differencing prohibited; "
        f"optimizer still {OPTIMIZER}"
    )


def check_validation_unit(report: list) -> None:
    """§9.6.5 — exactly 24 `(population, cell, block)` units, no other grain."""
    if VALIDATION_UNIT != ("population", "cell", "block"):
        raise ValidationFailure("§9.6.5: the validation unit is (population, cell, block)")
    units = validation_units()
    if len(set(units)) != N_VALIDATION_UNITS:
        raise ValidationFailure("§9.6.5: validation units are not distinct")
    per_population = {p: sum(1 for u in units if u[0] == p) for p in ("full_sample", "mdd_only")}
    if per_population != {"full_sample": 12, "mdd_only": 12}:
        raise ValidationFailure(f"§9.6.5: wrong unit distribution {per_population}")
    if not SIGMA2_REQUIRES_BOTH_UNITS_PASSING:
        raise ValidationFailure("§13: sigma_site^2 requires both of a cell's units to pass")
    # the seven criteria are still seven, and still un-retuned
    if len(REQUIRED_AGREEMENT_CRITERIA) != N_REQUIRED_AGREEMENT_CRITERIA != 7:
        raise ValidationFailure("§9.6.4: the amendment must not change the criterion count")
    report.append(
        f"validation unit: {VALIDATION_UNIT} — {N_VALIDATION_UNITS} units "
        f"(12 per population); seven-of-seven registry and tolerances unchanged"
    )


def check_status_codes(report: list) -> None:
    """§9.5.1 — numerical failure is never a scientific null."""
    if len(set(STATUS_CODES)) != len(STATUS_CODES):
        raise ValidationFailure("§9.5.1: duplicate status code")
    if STATUS_OK not in STATUS_CODES:
        raise ValidationFailure("§9.5.1: the OK status is missing")
    for code in NUMERICAL_FAILURE_CODES:
        if not is_numerical_failure(code):
            raise ValidationFailure(f"§9.5.1: {code} must be a numerical failure")
        if may_be_reported_as_null(code):
            raise ValidationFailure(f"§9.5.1 / F11: {code} may never be read as a null")
    if is_numerical_failure(STATUS_OK) or not may_be_reported_as_null(STATUS_OK):
        raise ValidationFailure("§9.5.1: ESTIMABLE is not a failure")
    for bogus in ("NOT_ESTIMABLE", "FAILED", "", "estimable"):
        try:
            is_numerical_failure(bogus)
        except ValidationFailure:
            continue
        raise ValidationFailure(f"§9.5.1: {bogus!r} must not be accepted as a code")

    # §10.6 all-or-nothing CV
    if CV_PARTIAL_FOLD_AVERAGE_PERMITTED:
        raise ValidationFailure("§10.6: a partial-fold average is not the registered quantity")
    if cv_logloss_status([STATUS_OK] * 23) != STATUS_OK:
        raise ValidationFailure("§10.6: all-estimable folds give an estimable CV log-loss")
    if cv_logloss_status([STATUS_OK] * 22 + ["NO_ELIGIBLE_START"]) != "CV_FOLD_NOT_ESTIMABLE":
        raise ValidationFailure("§10.6: one unestimable fold makes the CV quantity unestimable")
    try:
        cv_logloss_status([])
        raise ValidationFailure("§10.6: an empty fold set must fail, not pass")
    except ValidationFailure as exc:
        if "empty fold set is a failure" not in str(exc):
            raise

    # §12.1.1 negative LRT
    if NEGATIVE_LRT_CLAMP_PERMITTED:
        raise ValidationFailure("§12.1.1: a negative LRT statistic may not be clamped")
    if lrt_status(-1e-9) != "NESTED_LIKELIHOOD_INVERSION":
        raise ValidationFailure("§12.1.1: a negative statistic is an inversion")
    if lrt_status(0.0) != STATUS_OK:
        raise ValidationFailure("§12.1.1: exact zero is the admissible boundary")
    if lrt_status(3.84) != STATUS_OK:
        raise ValidationFailure("§12.1.1: a positive statistic is reported normally")
    report.append(
        f"status codes: {len(STATUS_CODES)} registered, closed set; "
        f"{len(NUMERICAL_FAILURE_CODES)} numerical failures, none reportable as a null; "
        f"CV {CV_AGGREGATION_POLICY}; negative LRT -> {NEGATIVE_LRT_POLICY}"
    )


def check_execution_invariants(report: list) -> None:
    """§16.1 / §16.2 — authoritative root, stale root, completion barrier."""
    if AUTHORITATIVE_RESULT_ROOT != "results/rq3":
        raise ValidationFailure("§16.1: the authoritative root is exactly results/rq3/")
    if CALLER_CONTROLLED_ROOT_PERMITTED:
        raise ValidationFailure("§16.1: a caller-controlled formal root is forbidden")
    if STALE_ROOT_POLICY != "fail_closed":
        raise ValidationFailure("§16.1: a stale formal root must fail closed")
    if not is_authoritative_root("results/rq3") or not is_authoritative_root("results/rq3/"):
        raise ValidationFailure("§16.1: the authoritative root is not recognised")
    for other in ("results/rq3_tmp", "results/rq3/sub", "results", "/tmp/rq3",
                  "results/rq2_s2", "../results/rq3"):
        if is_authoritative_root(other):
            raise ValidationFailure(f"§16.1: {other!r} must not count as the formal root")

    if COMPLETION_ORDER[-1] != "write_completion_marker_atomically":
        raise ValidationFailure("§16.2: the completion marker must be terminal")
    if COMPLETION_ORDER.index("reread_from_disk") <= COMPLETION_ORDER.index("write_checksum_ledger"):
        raise ValidationFailure("§16.2: the ledger must be re-read after it is written")
    if COMPLETION_ORDER.index("recompute_and_compare_digests") <= COMPLETION_ORDER.index("reread_from_disk"):
        raise ValidationFailure("§16.2: digests must be recomputed from the re-read bytes")
    if not completion_marker_is_permitted(COMPLETION_ORDER[:-1]):
        raise ValidationFailure("§16.2: the full barrier must permit the marker")
    for truncated in (COMPLETION_ORDER[:2], COMPLETION_ORDER[:5], (), COMPLETION_ORDER):
        if completion_marker_is_permitted(truncated):
            raise ValidationFailure("§16.2: an incomplete barrier must not permit the marker")
    report.append(
        f"execution: formal root pinned to {AUTHORITATIVE_RESULT_ROOT}/ (no caller override, "
        f"stale root {STALE_ROOT_POLICY}); {COMPLETION_MARKER} terminal after an "
        f"independent re-read, {len(COMPLETION_ORDER)}-step barrier"
    )


def check_optimizer_termination(report: list) -> None:
    """§9.4 / §9.4.5 — Stage E.2a.1: `ftol` is 1e-16 and `gtol` is operative."""
    if OPT_FTOL != 1e-16:
        raise ValidationFailure(
            f"§9.4: ftol must be 1e-16 (Stage E.2a.1), found {OPT_FTOL!r}"
        )
    if OPT_GTOL != 1e-8:
        raise ValidationFailure(f"§9.4: gtol is unchanged at 1e-8, found {OPT_GTOL!r}")
    if GRADIENT_TOL != 1e-5:
        raise ValidationFailure(
            f"§9.5(3): the eligibility threshold is unchanged at 1e-5, found {GRADIENT_TOL!r}"
        )
    # the whole point of the amendment: the relative criterion must no longer be
    # able to pre-empt the gradient criterion
    if not OPT_FTOL < OPT_GTOL:
        raise ValidationFailure(
            "§9.4.5: ftol must be tighter than gtol so that gtol is the operative "
            "stopping criterion"
        )
    # everything else about the optimizer is untouched by Stage E.2a.1
    if OPTIMIZER != "L-BFGS-B":
        raise ValidationFailure("§9.4: the optimizer family is unchanged")
    if (OPT_MAXITER, OPT_MAXFUN) != (20000, 50000):
        raise ValidationFailure("§9.4: the iteration/evaluation budgets are unchanged")
    if SIGMA_STARTS != (0.1, 0.5, 1.0, 2.0) or N_REGISTERED_STARTS != 4:
        raise ValidationFailure("§9.4: the four registered starts are unchanged")
    if START_INDEX_TO_SIGMA != {0: 0.1, 1: 0.5, 2: 1.0, 3: 2.0}:
        raise ValidationFailure("§9.4.1: the start registry is unchanged")
    if LOG_SIGMA_BOUNDS != (np.log(1e-4), np.log(10.0)):
        raise ValidationFailure("§9.3: the parameter bounds are unchanged")
    if INIT_PREVALENCE_CLIP != 1e-6:
        raise ValidationFailure("§9.4: the initializer is unchanged")
    if MODE_FALLBACK_SOLVER_PERMITTED:
        raise ValidationFailure("§9.2.5: no fallback solver — unchanged by Stage E.2a.1")

    document = (ROOT / "docs/RQ3_PROTOCOL.md").read_text()
    if "| `ftol` | `1e-16`" not in document:
        raise ValidationFailure("§9.4: the document does not register ftol = 1e-16")
    # the §21.3 discovery record must survive, marked resolved rather than deleted
    if "Status: RESOLVED at Stage E.2a.1" not in document:
        raise ValidationFailure("§21.3: the discovery record is not marked resolved")
    for token in ("21.4", "Stage E.2a.1", "pre-result optimizer-termination"):
        if token not in document:
            raise ValidationFailure(f"§21.4: the amendment record omits {token!r}")
    report.append(
        f"optimizer termination: ftol {OPT_FTOL} (Stage E.2a.1, was 1e-12) < "
        f"gtol {OPT_GTOL}, so gtol is operative; eligibility bound still "
        f"{GRADIENT_TOL}; {OPTIMIZER}, {N_REGISTERED_STARTS} starts, budgets and "
        f"bounds unchanged"
    )


def check_amendment_history(report: list) -> None:
    """§21 — the amendment is recorded as pre-result, and v1 stays immutable."""
    document = (ROOT / "docs/RQ3_PROTOCOL.md").read_text()
    if SUPERSEDED_PROTOCOL_TAG not in document:
        raise ValidationFailure(f"§21.1: the document does not name {SUPERSEDED_PROTOCOL_TAG}")
    for token in ("remains immutable", "no real RQ3 model had ever been fitted",
                  "independent reviewer reproduced", "No tolerance was loosened"):
        if token not in document:
            raise ValidationFailure(f"§21.1: the amendment record omits {token!r}")
    # v1's version string may appear only where it is marked as superseded
    position = document.find(SUPERSEDED_PROTOCOL_VERSION)
    while position != -1:
        window = document[max(0, position - 300):position + 300]
        if not any(m in window for m in ("Superseded", "superseded", "→", "amendment",
                                         "against version")):
            raise ValidationFailure(
                "§21.1: the superseded version string appears outside a supersession record"
            )
        position = document.find(SUPERSEDED_PROTOCOL_VERSION, position + 1)
    report.append(
        f"amendment: {SUPERSEDED_PROTOCOL_VERSION} -> {PROTOCOL_VERSION}, pre-result, "
        f"{SUPERSEDED_PROTOCOL_TAG} immutable; no tolerance retuned"
    )


def check_no_rq3_results(report: list) -> None:
    if (ROOT / "results/rq3").exists():
        raise ValidationFailure(
            "Stage E.1 must not create results/rq3/; it already exists"
        )
    report.append("scope: results/rq3/ does not exist; no model fitted, nothing written")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    report: list = []
    try:
        check_artifacts(report)
        cohort = load_base_cohort()
        check_cohort(cohort, report)
        check_anomaly_rule(cohort, report)
        cohort = apply_anomaly_rule(cohort)
        full, mdd = check_subsets(cohort, report)
        check_outcome_reproduces_s0(full, report)
        check_coding(full, mdd, report)
        check_loso_folds(full, mdd, report)
        check_multiplicity_shape(report)
        check_rejected_fraction_rule(report)
        check_holm_semantics(report)
        check_initializer(report)
        check_agreement_procedure(report)
        check_best_start_rule(report)
        check_amended_quadrature(report)
        check_mode_solver(report)
        check_gradient_rule(report)
        check_validation_unit(report)
        check_status_codes(report)
        check_execution_invariants(report)
        check_optimizer_termination(report)
        check_amendment_history(report)
        check_document_constants(report)
        check_no_rq3_results(report)
    except ValidationFailure as exc:
        for line in report:
            print(f"  PASS  {line}")
        print(f"\nFAIL  {exc}")
        return 1

    print(f"RQ3 protocol freeze validation — {PROTOCOL_VERSION}\n")
    for line in report:
        print(f"  PASS  {line}")
    print("\nPASS  docs/RQ3_PROTOCOL.md is mechanically executable against the frozen upstream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
