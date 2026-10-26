"""Stage-F frozen constants, output registry and CI scope (Stage F.2).

Every constant here is a **transcription** of frozen protocol authority — either
`docs/STAGE_F_PROTOCOL.md` (the prose authority) or `docs/stage_f_constants.json`
(its machine-readable subordinate, §L.7). Nothing here is a choice made at
implementation time, and nothing here may be revised in the light of a Stage-F
result (§A.3).

The module is deliberately inert: it fits nothing, reads no scientific artifact
and writes nothing. `assert_no_constant_drift()` performs the **two-way
structured comparison** against the JSON registry — every mirrored constant must
be present in the JSON with an equal value, and every JSON constant must be
present here with an equal value. A substring test is not accepted as mirroring
(§L.3 item 18).

The three module namespaces are kept disjoint by construction: `stage_f_conformal`
(F-RQ4), `stage_f_sensitivities` (F-SENS) and `stage_f_svm` (F-SVM) each import
this module and never each other.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# protocol identity — the frozen `stage-f-protocol-v2` anchor (§0.3, §L.7)
# ---------------------------------------------------------------------------
#: `STAGE_F_PROTOCOL/2.0.0`, tagged `stage-f-protocol-v2`, is the **only**
#: runtime authority. The `1.0.2` anchor below is retained as *superseded
#: identity*, never as a fallback: no code path may bind to it, and the two are
#: never resolved by preference order (§H.1a).
PROTOCOL_VERSION = "STAGE_F_PROTOCOL/2.0.0"
PROTOCOL_TAG = "stage-f-protocol-v2"
PROTOCOL_COMMIT = "10a23aeb02ccdbf3fd93ed327f058ac5c71ec72a"

#: §0.3.4 / §H.1a — the superseded `1.0.2` identity. Immutable historical
#: evidence; readable for provenance bookkeeping and for nothing else.
SUPERSEDED_PROTOCOL_VERSION = "STAGE_F_PROTOCOL/1.0.2"
SUPERSEDED_PROTOCOL_TAG = "stage-f-protocol-v1"
SUPERSEDED_PROTOCOL_COMMIT = "d2451ff7f719770c40cc6a3ac0fb9e3e7f0dd7dd"

PROTOCOL_DOC = "docs/STAGE_F_PROTOCOL.md"
CONSTANTS_DOC = "docs/stage_f_constants.json"
PROTOCOL_VALIDATOR = "scripts/validate_stage_f_protocol_freeze.py"

#: The three protocol-authority digests, recomputed at the `2.0.0` freeze.
EXPECTED_PROTOCOL_DOC_SHA256 = (
    "2dfef8351ad5efb498292034ef768701dfbde4c0f3a9a2722f4394d1a06611fb")
EXPECTED_CONSTANTS_SHA256 = (
    "3451db81fd315e1da79bbcee0874d6f086c65e4049278a3596371e223a8da28a")
EXPECTED_PROTOCOL_VALIDATOR_SHA256 = (
    "b6d86f3ac8e4cde375cfd70b0c964d1ba05e35d234964700671ebb653e990fe7")


class StageFProtocolDriftError(AssertionError):
    """A frozen Stage-F constant no longer matches its protocol authority."""


class StageFRegistryError(AssertionError):
    """A Stage-F output or CI quantity is not registered."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# cohort and registry counts (§B.1, §D.9)
# ---------------------------------------------------------------------------
BASE_COHORT_N = 2232
BASE_COHORT_SITES = 23
SOURCE_OOF_ROWS = 49104
TARGET_ROWS = 2232
SPLIT_REGISTRY_ROWS = 7700
SPLIT_REGISTRY_ROWS_N_GE_10 = 5400
SITES_PER_N = {5: 23, 10: 22, 20: 21, 40: 11}
R_SPLIT = 100
COMMON_SITES = ("S1", "S7", "S8", "S9", "S10", "S14", "S15", "S17", "S20",
                "S21", "S25")
COMMON_SITE_REGISTRY_ROWS = 4400
COMMON_SITE_REGISTRY_ROWS_N_GE_10 = 3300


def sites_per_n_canonical() -> dict:
    """`SITES_PER_N` rendered in the canonical JSON key form, `{"5": 23, ...}`.

    `SITES_PER_N` is keyed by the integer label budget, because that is what the
    grid *is*; JSON object keys are strings, so a mapping that enters run
    provenance with integer keys cannot survive its own round trip and fails the
    §I.1 step-5 provenance-equality barrier. The canonical in-memory
    representation of this fact is therefore the string-keyed one, derived here
    from the single frozen constant rather than restated anywhere.
    """
    return {str(n): SITES_PER_N[n] for n in sorted(SITES_PER_N)}


# ---------------------------------------------------------------------------
# F-RQ4 (§D)
# ---------------------------------------------------------------------------
ALPHA_GRID = (0.1, 0.2, 0.3, 0.4)
N_GRID = (10, 20, 40)                     # §D.6 — N = 5 is excluded
S2_N_GRID = (5, 10, 20, 40)               # the parent's full label budget
CONFORMAL_TOLERANCE = 0.05                # descriptive band only (§D.5)
NONCONFORMITY_SCORE = "a(x, y) = 1 - p_y(x)"
INDEX_RULE = "k = ceil((n + 1) * (1 - alpha)); t = a_(k) if k <= n else +inf"
INTERPOLATION_PERMITTED = False
TIE_INCLUSION = "non_strict"

METHOD_SOURCE_POOLED = "source_pooled_split"
METHOD_WEIGHTED = "weighted"
METHOD_TARGET_LOCAL = "target_local"
CONFORMAL_METHODS = (METHOD_SOURCE_POOLED, METHOD_WEIGHTED, METHOD_TARGET_LOCAL)

#: §D.3 — the literal frozen guarantee label of each method. No implementation
#: may upgrade an empirical label, and `target_local` is the only method
#: permitted a finite-sample coverage claim.
GUARANTEE_LABELS = {
    METHOD_SOURCE_POOLED: "EMPIRICAL_SOURCE_REFERENCE",
    METHOD_WEIGHTED: "EMPIRICAL",
    METHOD_TARGET_LOCAL: "EXACT_WITHIN_TARGET_SITE",
}
METHOD_INFORMATION_BUDGET = {
    METHOD_SOURCE_POOLED: "S0",
    METHOD_WEIGHTED: "S1",
    METHOD_TARGET_LOCAL: "S2",
}
EXACT_GUARANTEE_PERMITTED_FOR = (METHOD_TARGET_LOCAL,)

WEIGHT_COLUMN = "w_final"
WEIGHTED_TEST_POINT_ATOM_INCLUDED = False   # §D.8, disclosed omission

REGISTERED_RATES = ("coverage", "avg_set_size", "singleton_rate",
                    "empty_rate", "doubleton_rate")
PRIMARY_RATES = ("coverage", "avg_set_size", "singleton_rate")
DIAGNOSTIC_RATES = ("empty_rate", "doubleton_rate")

#: §H.2 — the 20 registered conformal cells, enumerated rather than counted.
REGISTERED_CELLS = (
    tuple((METHOD_SOURCE_POOLED, a, None) for a in ALPHA_GRID)
    + tuple((METHOD_WEIGHTED, a, None) for a in ALPHA_GRID)
    + tuple((METHOD_TARGET_LOCAL, a, n) for n in N_GRID for a in ALPHA_GRID)
)
N_REGISTERED_CELLS = len(REGISTERED_CELLS)

# ---------------------------------------------------------------------------
# the closed RQ4 CI registry (§D.12)
# ---------------------------------------------------------------------------
SITE_BOOTSTRAP_B = 5000
CI_LEVEL = 0.95
CI_QUANTILES = (0.025, 0.975)
CI_METHOD = "percentile"
QUANTILE_METHOD = "linear"
MIN_VALID_REPLICATES = 4750
BOOTSTRAP_ENTROPY = (20260807, 1102)
RESAMPLING_UNIT = "site"
BOOTSTRAP_SITE_SET_SIZES = (23, 22, 21, 11)

F_CI_REQUIRED = (
    "rq4_site_macro_coverage",
    "rq4_site_macro_avg_set_size",
    "rq4_site_macro_singleton_rate",
    "rq4_marginal_coverage",
)

F_CI_PROHIBITED = (
    "rq4_per_site_value",
    "rq4_per_repeat_value",
    "rq4_worst_site_coverage",
    "rq4_site_macro_empty_rate",
    "rq4_site_macro_doubleton_rate",
    "rq4_marginal_avg_set_size",
    "rq4_marginal_singleton_rate",
    "rq4_marginal_empty_rate",
    "rq4_marginal_doubleton_rate",
    "rq4_tau",
    "rq4_degeneracy_count",
    "rq4_method_contrast",
    "rq4_alpha_contrast",
    "rq4_n_contrast",
    "svm_site_macro_bacc",
    "svm_per_site_bacc",
    "sensitivity_overall_error_ucb",
    "sensitivity_q_step_002",
    "sensitivity_r_star_shift",
    "sensitivity_common_sites",
)

HYPOTHESIS_TEST = False
MULTIPLICITY_CORRECTION = False
TOST = False

#: The 80 interval slots, enumerated one by one — 4 quantities x 20 cells.
REQUIRED_INTERVALS = tuple(
    (q, method, alpha, n_labels)
    for q in F_CI_REQUIRED
    for method, alpha, n_labels in REGISTERED_CELLS
)
N_CI_SLOTS = len(REQUIRED_INTERVALS)


def requires_ci(quantity: str) -> bool:
    """True iff `quantity` carries a 95% site-bootstrap CI in the Stage-F report.

    Raises on an unregistered name: §D.12.4 is closed in both directions and
    there is no default-off category.
    """
    if quantity in F_CI_REQUIRED:
        return True
    if quantity in F_CI_PROHIBITED:
        return False
    raise StageFRegistryError(
        f"{quantity!r} is not in the closed Stage-F CI registry "
        f"(docs/STAGE_F_PROTOCOL.md §D.12.2 / §D.12.4); classify it there first")


# ---------------------------------------------------------------------------
# status, degeneracy and composition (§G)
# ---------------------------------------------------------------------------
ESTIMABLE = "ESTIMABLE"
NOT_ESTIMABLE = "NOT_ESTIMABLE"
STATUS_VALUES = (ESTIMABLE, NOT_ESTIMABLE)

REASON_CODES = (
    "SITE_NOT_ELIGIBLE_AT_N",
    "NO_EVALUATION_POINTS",
    "COVERAGE_MATCH_UNATTAINABLE",
    "CI_INSUFFICIENT_VALID_REPLICATES",
    "NO_ESTIMABLE_SITE_IN_CELL",
    "SVM_NOT_CONVERGED",
    "SVM_FOLD_FAILED",
)
REASON_CODES_REUSED_FROM_RQ3 = (
    "NO_ELIGIBLE_START",
    "CV_FOLD_NOT_ESTIMABLE",
    "QUADRATURE_DISAGREEMENT",
    "NESTED_LIKELIHOOD_INVERSION",
    "VALIDATION_MALFORMED",
)
REASON_CODES_REUSED_BY = ("sens5_s1_attribution", "sens6_realized_coverage_matched")

#: Withdrawn at F.1a. Neither may ever be emitted as a live `NOT_ESTIMABLE`
#: reason: an all-inclusive threshold and a single-class evaluation set both
#: leave every registered quantity computable.
WITHDRAWN_REASON_CODES = ("CONFORMAL_LEVEL_UNATTAINABLE", "SINGLE_CLASS_EVALUATION")

DEGENERACY_CODES = (
    "ALL_INCLUSIVE_THRESHOLD",
    "ALL_INCLUSIVE_REALIZED",
    "ALL_EMPTY_REALIZED",
)
DEGENERACY_IMPLIES_NOT_ESTIMABLE = False

CLASS_COMPOSITION_FLAGS = ("BOTH_CLASSES", "SINGLE_CLASS_MDD", "SINGLE_CLASS_HC")
CLASS_COMPOSITION_IMPLIES_NOT_ESTIMABLE = False
CLASS_CONDITIONAL_CONFORMAL_ENDPOINT_REGISTERED = False

#: The metrics an all-inclusive threshold must produce (§D.10, §G.3). Registered
#: as a **structural consequence** of the score, not as an expected result.
ALL_INCLUSIVE_METRICS = {
    "coverage": 1.0,
    "avg_set_size": 2.0,
    "singleton_rate": 0.0,
    "empty_rate": 0.0,
    "doubleton_rate": 1.0,
}


def assert_reason_code(code: str) -> str:
    """Raise on an unregistered or withdrawn `NOT_ESTIMABLE` reason."""
    if code in WITHDRAWN_REASON_CODES:
        raise StageFRegistryError(
            f"{code!r} was withdrawn as an estimability reason at F.1a "
            "(docs/STAGE_F_PROTOCOL.md §G.2); it is a degeneracy code or a "
            "composition flag, and the row stays ESTIMABLE")
    if code not in REASON_CODES + REASON_CODES_REUSED_FROM_RQ3:
        raise StageFRegistryError(
            f"{code!r} is not a registered Stage-F reason code (§G.2)")
    return code


# ---------------------------------------------------------------------------
# F-SENS (§E, §H.4)
# ---------------------------------------------------------------------------
N_SENSITIVITIES = 6
SENSITIVITY_LABELS = (
    "sens1_overall_error_ucb",
    "sens2_q_step_002",
    "sens3_rstar_pm_005",
    "sens4_common_site",
    "sens5_s1_attribution",
    "sens6_realized_coverage_matched",
)
COMBAT_IN_MANDATORY_CORE = False

SENS1_ALPHA_UCB = 0.10
SENS1_BONFERRONI = False
SENS2_Q_STEP = 0.02
SENS2_N_CANDIDATES = 41
SENS3_R_STAR_OFFSETS = (-0.05, 0.05)

PARENT_STRATEGIES_LABEL_CONSUMING = ("target_local", "pooled_local_shrinkage")
PARENT_STRATEGIES_ZERO_LABEL = ("pooled_source", "target_unlabeled_weighted")
PARENT_CONFIDENCE_METHODS = ("probability", "bootstrap", "mahalanobis")

CARDINALITY = {
    "sens1_decisions": 46200,
    "sens1_evaluations": 78600,
    "sens2_decisions": 46200,
    "sens2_evaluations": 78600,
    "sens3_decisions": 92400,
    "sens3_evaluations": 157200,
    "sens4_decisions": 26400,
    "sens4_evaluations": 46200,
    "sens4_zero_label_evaluations": 19800,
    "rq3_cells": 12,
    "rq3_validation_units": 24,
    "rq3_families": 4,
    "sens6_threshold_selection_rows": 192,
}

# --- the SENS-5/6 schema, corrected at F.1b (§H.4.3) -----------------------
SENS56_APPLIES_TO = ("sens5_s1_attribution", "sens6_realized_coverage_matched")
SENS56_POPULATIONS = ("full_sample", "mdd_only")
SENS56_CELLS_FILE = "cells.csv"
SENS56_CELLS_LOGICAL_KEY = ("population", "operating_point", "method")
SENS56_CELLS_EXPECTED_ROWS = 12
SENS56_BLOCK_IN_CELLS_LOGICAL_KEY = False
SENS56_VALIDATION_UNITS_FILE = "validation_units.csv"
SENS56_VALIDATION_UNITS_LOGICAL_KEY = ("population", "operating_point", "method",
                                       "block")
SENS56_VALIDATION_UNITS_EXPECTED_ROWS = 24
SENS56_FAMILIES_FILE = "families.csv"
SENS56_FAMILIES_LOGICAL_KEY = ("population", "block")
SENS56_FAMILIES_EXPECTED_ROWS = 4
SENS56_HOLM_FAMILY_SIZE = 6
SENS56_BLOCKS_PER_POPULATION = 2
SENS56_VALIDATION_UNITS_PER_CELL = 2
SENS56_BLOCKS_BY_POPULATION = {
    "full_sample": ("motion", "diagnosis"),
    "mdd_only": ("motion", "clinical"),
}
SENS56_BLOCKS_CARRIED_WITHIN_CELL_RECORD = True
SENS56_PARENT_CELL_REGISTRY_SIZE = 6
SENS56_PARENT_VALIDATION_UNIT = ("population", "cell", "block")

# --- the Gate-1 OOF binding, transcribed from src/s2_inputs (§B.1a) --------
GATE1_BINDING_HELPER = "src/s2_inputs.assert_gate1_binds_to_checksummed_s0"
GATE1_UNLEDGERED_FILE = "results/gate1/source_oof_predictions.csv"
GATE1_LEDGER_BOUND_FILE = "results/rq1_s0/confidence_source_oof.csv"
GATE1_LOGICAL_KEY = ("outer_fold_held_out_site", "canonical_id")
GATE1_EXACT_COLUMNS = ("outer_fold_held_out_site", "canonical_id", "source_site",
                       "y_true", "oof_pred", "selected_pca", "selected_C")
GATE1_N_EXACT_COLUMNS = 7
GATE1_TOLERANCE_COLUMNS = ("oof_prob_mdd",)
GATE1_OOF_PROB_MDD_ATOL = 4e-16
GATE1_TOLERANCE_RULE = "max_abs_difference <= atol"
GATE1_OOF_PROB_MDD_BIT_IDENTICAL_REQUIRED = False
GATE1_ROW_COUNT_EQUALITY_REQUIRED = True
GATE1_DUPLICATE_LOGICAL_KEYS_PERMITTED = False
GATE1_BINDING_IS_ONE_TO_ONE = True
GATE1_CSV_FLOAT_PRECISION = "round_trip"
GATE1_N_OOF_ROWS = 49104
GATE1_R_STAR_CROSSCHECK_ATOL = 1e-15
GATE1_R_STAR_CROSSCHECK_SITES = 23

# --- SENS-6 (§E.7.1) -------------------------------------------------------
SENS6_REQUESTED_COVERAGE = {"OP50": 0.50, "OP30": 0.70}
SENS6_OPERATING_POINTS = ("OP50", "OP30")
SENS6_CANDIDATE_SET = "distinct observed confidence values of the group, ascending"
SENS6_COVERAGE_DEFINITION = "cov(t) = #{i : conf_i >= t} / n"
SENS6_SELECTION_RULE = "tau = min { t in T : cov(t) <= c_star }"
SENS6_RETENTION_COMPARISON = ">="
SENS6_ABSTENTION_COMPARISON = "<"
SENS6_INTERPOLATION_PERMITTED = False
SENS6_RANDOMISATION_PERMITTED = False
SENS6_USES_TARGET_LABELS = False
SENS6_NOT_ESTIMABLE_CODES = ("NO_EVALUATION_POINTS", "COVERAGE_MATCH_UNATTAINABLE")
SENS6_FULL_SAMPLE_N = 2222
SENS6_FULL_SAMPLE_SITES = 23
SENS6_MDD_ONLY_N = 484
SENS6_MDD_ONLY_SITES = 9

#: §E.7.2 — four arithmetic demonstrations of determinism. Not results: they use
#: no cohort value and are re-derived by the tests and by both validators.
SENS6_WORKED_EXAMPLES = (
    {"c_star": 0.50, "conf": (0.10, 0.20, 0.30, 0.30, 0.30, 0.70, 0.80, 0.90),
     "tau": 0.70, "realized_coverage": 0.375},
    {"c_star": 0.50, "conf": (0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88),
     "tau": 0.55, "realized_coverage": 0.500},
    {"c_star": 0.70, "conf": (0.10, 0.10, 0.10, 0.40, 0.40, 0.60, 0.70, 0.80,
                              0.90, 0.95),
     "tau": 0.40, "realized_coverage": 0.700},
    {"c_star": 0.70, "conf": (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
     "tau": None, "reason_code": "COVERAGE_MATCH_UNATTAINABLE"},
)

# ---------------------------------------------------------------------------
# F-SVM (§F)
# ---------------------------------------------------------------------------
SVM_ESTIMATOR = "sklearn.svm.LinearSVC"
SVM_ESTIMAND = "site_macro_outer_loso_balanced_accuracy"
SVM_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
SVM_PCA_CANDIDATES = (None, 50, 100, 200)
SVM_PENALTY = "l2"
SVM_LOSS = "squared_hinge"
SVM_DUAL = False
SVM_CLASS_WEIGHT = None
SVM_FIT_INTERCEPT = True
SVM_TOL = 1e-4
SVM_MAX_ITER = 2000
SVM_RANDOM_STATE = 20260807
#: Under `dual=False` liblinear's primal path is not random; sklearn 1.7.2's own
#: docstring says `random_state` has no effect. The literal is retained for
#: constructor explicitness and is never described as an operative seed.
SVM_RANDOM_STATE_STATUS = "OPERATIONALLY_INERT"
SVM_DECISION_RULE = "decision_function(x) > 0"
SVM_PROBABILITY_CALIBRATION = False
SVM_PLATT_SCALING = False
SVM_SAMPLE_WEIGHT = "src/nested_loso.site_equal_weights"
SVM_C_SCALE_EQUIVALENCE_CLAIMED = False
#: §F.4 / §F.5 — the two splits, named for the mirror. The estimand, the splits
#: and the metric are §0.3.2 items and are unchanged by `2.0.0`.
SVM_OUTER_SPLIT = "leave-one-site-out"
SVM_INNER_SPLIT = "leave-one-source-site-out"

# --- §F.7.1 / §F.7.1a / §F.7.2 / §F.7.3 / §F.7.4 / §F.7.5 --------------------
#: §F.7.1 — the any-fit fail-closed rule, stated as the implementation must
#: satisfy it. An **inner-grid** fit is a fit in the outer fold's lifecycle, so
#: its `ConvergenceWarning` invalidates the whole outer fold. Clarified at F.7,
#: never weakened: under `2.0.0` the rule bites harder, not softer.
SVM_ANY_FIT_WARNING_INVALIDATES_FOLD = True
SVM_INNER_FIT_IS_FOLD_LIFECYCLE_FIT = True
#: §F.7.1 — the two explicit prohibitions. Selecting over the candidates that
#: happened not to warn is exactly what Attempt #5 did; refitting an already
#: invalid fold could only manufacture a number §F.7 forbids reporting.
SVM_REDUCED_GRID_SELECTION_PERMITTED = False
SVM_FINAL_REFIT_AFTER_INNER_WARNING_PERMITTED = False
#: §F.7.2 — selection is permitted **only** when all 20 candidate rows carry
#: `status = ESTIMABLE` with a finite score.
SVM_SELECTION_REQUIRES_ALL_CANDIDATES_ESTIMABLE = True
#: §F.7.5 — the total, deterministic selection order. `no-PCA` is the *weakest*
#: compression and is therefore **last**; the `1.0.2` implementation ranked by
#: `SVM_PCA_CANDIDATES.index`, which put it first. This tuple is the tie-break
#: order and is deliberately **not** the candidate set: `SVM_PCA_CANDIDATES` is
#: unchanged, and reordering it would change what is searched, not how ties
#: break.
SVM_TIE_BREAK_PCA_ORDER = (50, 100, 200, None)
SVM_TIE_BREAK_ORDER = ("inner_macro_bacc_desc", "C_asc",
                       "compression_strength_desc")
#: §F.7.3 — `converged` is a **fold-lifecycle** statement, never a statement
#: about the last fit only.
SVM_CONVERGED_REQUIRES_ALL_INNER_FITS_CLEAN = True
SVM_CONVERGED_REQUIRES_CLEAN_FINAL_REFIT = True
SVM_CONVERGED_REQUIRES_FOLD_ESTIMABLE = True
#: §F.7.4 — `n_iter_max` records the configured `max_iter` cap and nothing else.
#: It is `2000` whenever the final refit was **attempted** (warned or not) and
#: `null` when the refit was skipped because the fold was already invalid. It is
#: never redefined as an observed `n_iter_`, and no observed iteration count is
#: added as a new scientific output.
SVM_N_ITER_MAX_IS_CONFIGURED_CAP = True
SVM_N_ITER_MAX_IS_OBSERVED_N_ITER = False
SVM_N_ITER_MAX_WHEN_REFIT_ATTEMPTED = SVM_MAX_ITER
SVM_N_ITER_MAX_WHEN_REFIT_SKIPPED = None
SVM_OBSERVED_ITERATION_COUNT_PERSISTED = False

# --- §G.2a — the closed `SVM_FOLD_FAILED` scope ------------------------------
#: The **positive, closed, exhaustive** allow-list. An exclusion list alone is
#: not a specification: it says what the code is *not* without ever saying what
#: it *is*, which leaves the narrowness unenforceable. Exactly these two types,
#: and nothing else, may ever be spelled `SVM_FOLD_FAILED`.
#:
#: The names are transcribed here; `stage_f_svm` resolves them to the live type
#: objects and asserts the resolution, so the registry never holds a type object
#: and the two can never drift into naming different classes.
SVM_FOLD_FAILED_ALLOWED_TYPES = ("FloatingPointError", "numpy.linalg.LinAlgError")
SVM_FOLD_FAILED_N_ALLOWED_TYPES = 2
SVM_FOLD_FAILED_ALLOW_LIST_IS_CLOSED = True
#: Membership is by **exact type identity** — `type(exc) is X` — and never by
#: `isinstance`, so no subclass of a listed type can widen the list. This is not
#: pedantry: `numpy.linalg.LinAlgError` is a subclass of `ValueError` in the
#: installed numpy 2.2, while a bare `ValueError` is expressly excluded, and
#: only exact-type matching keeps both statements true at once.
SVM_FOLD_FAILED_TYPE_MATCH_RULE = "exact_type_identity"
SVM_FOLD_FAILED_ISINSTANCE_PERMITTED = False
SVM_FOLD_FAILED_SUBCLASS_WIDENING_PERMITTED = False
SVM_FOLD_FAILED_BROAD_CATCHING_PERMITTED = False
#: `scipy.linalg.LinAlgError` **is the same class object** in the installed
#: environment, so it is an identity-equivalent alias and adds no third type.
SVM_FOLD_FAILED_IDENTITY_ALIASES = ("scipy.linalg.LinAlgError",)
SVM_FOLD_FAILED_N_SEMANTIC_TYPES = 2
#: Condition (2): the exception must have been raised **inside** the frozen
#: estimator/backend fit boundary — the `LinearSVC.fit(...)` call for one
#: candidate together with the `decision_function(...)` on that same fitted
#: estimator, and nothing else. The same exception type raised anywhere else is
#: **not** `SVM_FOLD_FAILED`.
SVM_FOLD_FAILED_REQUIRES_BOUNDARY = True
SVM_FOLD_FAILED_BOUNDARY_INSIDE = ("LinearSVC.fit", "decision_function")
SVM_FOLD_FAILED_BOUNDARY_OUTSIDE = (
    "Data assembly", "the scaler and PCA transforms", "weight construction",
    "scoring", "selection", "aggregation", "schema handling", "persistence")
SVM_FOLD_FAILED_OUTSIDE_BOUNDARY_IS_HARD_FAILURE = True
#: The exclusions, retained and extended. Each is a defect in the run, not a
#: property of the science: each still fails hard (§J.3), aborts the Formal run,
#: and must **not** be converted into a candidate row or a fold row.
SVM_FOLD_FAILED_MUST_NOT_ABSORB = (
    "exception_catch_all", "programming_errors", "invariant_failures",
    "schema_errors", "contract_or_registry_invariant_failures",
    "AssertionError", "KeyError", "TypeError", "IndexError", "RuntimeError",
    "ValueError", "invalid_shapes", "nonfinite_frozen_inputs",
    "missing_classes", "off_grid_parameters", "impossible_internal_states",
    "MemoryError", "KeyboardInterrupt", "SystemExit",
    "leakage_or_data_contract_violations")
SVM_FOLD_FAILED_N_MUST_NOT_ABSORB = 20
SVM_FOLD_FAILED_IS_CATCH_ALL = False
SVM_FOLD_FAILED_UNEXPECTED_ERRORS_FAIL_HARD = True
#: §G.2a — a convergence warning has its own code and its own fold-level
#: consequence, and is never spelled `SVM_FOLD_FAILED`.
SVM_FOLD_FAILED_SUBSTITUTES_FOR_NOT_CONVERGED = False
SVM_FOLD_FAILED_ADMITS_ONLY = (
    "narrowly classified expected estimator/backend failure")
SVM_FOLD_FAILED_EXACT_TYPE_PREDICATES = (
    "type(exc) is FloatingPointError",
    "type(exc) is numpy.linalg.LinAlgError")
SVM_FOLD_FAILED_ALIASES_ADD_NO_TYPE = True
SVM_FOLD_FAILED_BOUNDARY_DESCRIPTION = (
    "the frozen LinearSVC.fit(...) call for one candidate and the "
    "decision_function(...) call on that same fitted estimator")
SVM_FOLD_FAILED_LINALGERROR_SUBCLASSES_VALUEERROR = True
SVM_FOLD_FAILED_BARE_VALUEERROR_PERMITTED = False
#: §0.3.2 / §G.2a — making a failure go away by changing the estimator would be
#: exactly the post-result revision §A.3 prohibits.
SVM_ESTIMATOR_SETTINGS_ALTERED_TO_AVOID_FAILURES = False

# --- §F.7.1 / §F.7.1a — the two distinct per-site warning representations ----
#: Both are `NOT_ESTIMABLE` / `SVM_NOT_CONVERGED` on the existing §G.1 / §G.2
#: vocabulary — **no new status value and no new reason code**. They differ only
#: in which fields carry a value, and neither may be written where the other
#: applies. `RETAINED_SELECTED_VALUE` is a *sentinel*, never a persisted cell:
#: it marks a field the §F.7.1a row must carry at its lawfully selected value,
#: so that a deleted field can never read as `null`.
SVM_RETAINED_VALUE_TOKEN = "RETAINED_SELECTED_VALUE"
SVM_COHORT_FIELDS_ALWAYS_PERSISTED = ("n_acquisitions", "n_mdd", "n_hc")
#: §F.7.1 — an **inner-grid** fit warned. The §F.7.2 selection precondition was
#: not met, so no lawful selection occurred, there is no selected pair and no
#: selected inner score to retain, and the final refit was **skipped** rather
#: than attempted — which is why `n_iter_max` is `null` under §F.7.4.
SVM_INNER_GRID_WARNING_REPRESENTATION = {
    "section": "F.7.1",
    "lawful_selection_occurred": False,
    "status": "NOT_ESTIMABLE",
    "reason_code": "SVM_NOT_CONVERGED",
    "bacc": None,
    "selected_pca": None,
    "selected_C": None,
    "inner_macro_bacc": None,
    "converged": False,
    "n_iter_max": None,
    "final_refit_attempted": False,
    "selection_metadata_retained": False,
}
#: §F.7.1a — the complete 20-candidate grid was valid, §F.7.5 lawfully selected,
#: and the **selected final outer refit** warned. The fold is invalidated
#: exactly as §F.7.1 requires; what differs is only the persisted evidence,
#: which really exists and is therefore retained rather than erased. The
#: retained metadata is provenance, never a result: it may never report a
#: number, rescue or re-refit the fold, or enter any aggregate.
SVM_FINAL_REFIT_WARNING_REPRESENTATION = {
    "section": "F.7.1a",
    "lawful_selection_occurred": True,
    "all_inner_candidate_rows_estimable": True,
    "status": "NOT_ESTIMABLE",
    "reason_code": "SVM_NOT_CONVERGED",
    "bacc": None,
    "selected_pca": SVM_RETAINED_VALUE_TOKEN,
    "selected_C": SVM_RETAINED_VALUE_TOKEN,
    "inner_macro_bacc": SVM_RETAINED_VALUE_TOKEN,
    "selected_pca_retained": True,
    "selected_C_retained": True,
    "inner_macro_bacc_retained": True,
    "selected_inner_score_finite": True,
    "converged": False,
    "n_iter_max": SVM_MAX_ITER,
    "final_refit_attempted": True,
    "selection_metadata_retained": True,
    "retained_metadata_usable_as_result": False,
    #: §F.7.1a — the final outer refit is **not** a candidate and never appears
    #: in `fold_selection.csv`, so a fold in this state is not identifiable from
    #: that table alone. That is exactly why the per-site row must carry
    #: `converged = false` and the `NOT_ESTIMABLE` status.
    "appears_in_fold_selection_csv": False,
}
SVM_WARNING_REPRESENTATIONS_ARE_DISTINCT = True
SVM_WARNING_INTRODUCES_NEW_STATUS_OR_REASON = False

# --- §E.2a — the SENS-1 `score_type` metadata correction ---------------------
#: A **persisted-label** defect, not a definition change. §E.2's scientific
#: definition is unchanged: SENS-1 is, and remains, one overall-error upper
#: confidence bound at `alpha = 0.10` with no Bonferroni, computed by
#: `certification.overall_error_upper`. What was wrong is that the rows
#: inherited the *parent's* class-conditional label verbatim, so a persisted
#: label contradicted the persisted computation.
#:
#: **No numeric recomputation is authorized by this**, and rows produced by a
#: genuinely class-conditional functional keep the parent's own label.
SENS1_ACTUAL_FUNCTIONAL = "certification.overall_error_upper"
SENS1_PERSISTED_LABEL_IN_1_0_2 = "exact_classwise_clopper_pearson_ucb"
SENS1_OVERALL_ERROR_SCORE_TYPE = "exact_overall_error_clopper_pearson_ucb"
SENS1_SCORE_TYPE_NUMERIC_RECOMPUTATION_AUTHORIZED = False
SENS1_SCORE_TYPE_SCIENTIFIC_DEFINITION_CHANGED = False
#: §E.2a's own field: the `2.0.0` **document** recorded the correction as
#: required future work and did not implement it. That statement is about the
#: protocol version, not about this runtime, so it stays `False` here.
SENS1_IMPLEMENTED_BY_PROTOCOL_VERSION = False

# --- §0.3 — what the `2.0.0` amendment was, mirrored ------------------------
AMENDMENT_TRIGGER = ("independently audited section F.7 / section H.3 "
                     "representability contradiction found after Attempt #5")
AMENDMENT_ATTEMPT_NUMBER = 5
#: §0.3.2, restated as a machine-checkable claim: no solver, grid, tolerance,
#: iteration cap, estimator or metric was altered, and none may be altered to
#: rescue the Attempt #5 SVM number.
SVM_DEFINITION_CHANGED_BY_2_0_0 = False
AMENDMENT_FAIL_CLOSED_RULE_WEAKENED = False
AMENDMENT_SOLVER_OR_GRID_CHANGED = False
AMENDMENT_IS_POST_RESULT_RELAXATION = False
AMENDMENT_SCOPE = ("representability", "status", "root", "version")
AMENDMENT_RUNTIME_CHANGED = False
AMENDMENT_FORMAL_RUN_AUTHORIZED = False
AMENDMENT_STAGE_G_AUTHORIZED = False
AMENDMENT_NEW_RESULT_TAG_CREATED = False
AMENDMENT_DRAFT_AUDIT_VERDICT = "FAIL"
AMENDMENT_DRAFT_AUDIT_CRITICAL = 4
AMENDMENT_DRAFT_AUDIT_NON_CRITICAL = 2
AMENDMENT_DRAFT_CORRECTIONS_ARE_SCIENTIFIC = False
AMENDMENT_OLD_IMPLEMENTATION_FAILURES_EXPECTED = True
AMENDMENT_PROTOCOL_PINNING_WEAKENED = False
AMENDMENT_COVERAGE_AUDIT_VERDICT = "FAIL"
AMENDMENT_COVERAGE_AUDIT_CRITICAL = 1
AMENDMENT_COVERAGE_AUDIT_ACCEPTED_CONTENT = True
AMENDMENT_HARDENING_IS_SCIENTIFIC_CHANGE = False
AMENDMENT_HARDENING_CHANGED_RUNTIME = False
SVM_FORBIDDEN_OUTPUTS = ("auc", "f1", "accuracy", "selective_risk_curve",
                         "calibrated_probability",
                         "significance_test_vs_logistic_regression")

# ---------------------------------------------------------------------------
# the 17 consumed artifacts (§B.1)
# ---------------------------------------------------------------------------
CONSUMED_ARTIFACTS = {
    "results/data_contract/analysis_manifest_v1.csv":
        "bb582babe5a46e42503f6682140dfc7b0e7834fa813ca5dbb8bdae42dc6c1ff2",
    "results/data_contract/mean_fd_per_subject.csv":
        "7fc40ba94e2709b8d3b04d512cf2ffc94dca7c501d551b0f6a90cd1477cd1ebb",
    "results/gate1/cache/fc_aal116_z_included.npz":
        "0598371c9b0a29f29004d0480739b631e18a971a0ad77bfefd888c7e3370d992",
    "results/gate1/source_full_coverage_error_by_fold.csv":
        "7ec3cb7fb66d00caa71d26eb8af93a682ef5617ce277ae2660b1fe8aecc64384",
    "results/gate1/source_oof_predictions.csv":
        "6ac19a48f2fcdeb9ab871d5799d356d81c43939f62620bc5c8d8eb026f0672a3",
    "results/rq1_s0/confidence_source_oof.csv":
        "03a87f11cdb95be50e3ca43bd8217f27e26ed6bb25731a6dd6fc9a8733d46390",
    "results/rq1_s0/risk_coverage_all_q.csv":
        "dd5a266adf632bc889f8282d02adc65dc6d793af94bb6f42e6d330b6be3adb8e",
    "results/rq1_s0/confidence_target.csv":
        "99cafb3a90bfb9e8c339458407b56ecf4f666cde4f97dfb9a7c88b28653d6b37",
    "results/rq1_s0/thresholds_source_oof.csv":
        "78f332e5255994d4069f9783dff25d5bd53d3cb54bcc3039287011dec3793a9d",
    "results/rq1_s0/op50_summary.csv":
        "31601e0ad3238c2154ca19f3985a861825d91d20fb42b4eb4f9ef21f90ed6b94",
    "results/rq1_s0/op30_summary.csv":
        "3d4e2b7248a7581691e1ee0bfc7480f325371fbdbbc9fbb4d455d0b03730b024",
    "results/rq1_s1/density_weights.csv":
        "2f6c464487867ea94e1205182332b9f106aabcb283ae4670633f76689bd76db3",
    "results/rq1_s1/thresholds_weighted_source.csv":
        "a908c08c608b374671589c713d9a0a79c90c673839d2d7cd2a735e1ca4fdf57f",
    "results/rq1_s1/weighted_source_risk.csv":
        "b414312a6ddf42dae50f774c6f3eb51d1e879d00eddd1cd54f6fb06616f7e83a",
    "results/rq1_s1/ess_by_site.csv":
        "74a6b54852a7ef74e1bea3fb9124df0be9ee00cdea6e84ba050e92177fb8d646",
    "results/rq1_s1/selected_domain_C.csv":
        "80141d3aa49f44471a9f729f8778d12cbe3f9e7799e7b3f064c7588a0373739c",
    "results/rq2_s2/production_split_registry.csv":
        "9cdf4bd8cd6a273fb17332afbe4946f79263fcb63713d2910cd03694dade03f4",
}
N_CONSUMED_ARTIFACTS = 17
RECOMPUTABLE_ARTIFACTS = ("results/gate1/cache/fc_aal116_z_included.npz",)

# ---------------------------------------------------------------------------
# the output registry (§H.1-§H.4)
# ---------------------------------------------------------------------------
#: §H.1 — the authoritative `2.0.0` store root. It moved because
#: `results/stage_f/` holds the tagged, frozen `1.0.2` Attempt #5 store and §I.2
#: refuses to start a Formal run against a non-empty authoritative root: moving
#: the root is the mechanical consequence of keeping the historical store
#: immutable, not a scientific change. The namespaces, filenames and schemas are
#: otherwise identical.
STORE_ROOT = "results/stage_f_v2/"
#: §H.1 — where a future, separately authorized freeze of that store will live.
FREEZE_ROOT = "results/stage_f_v2_freeze/"
#: §L.4 — the result tag a future Formal run would earn. Declared so nothing
#: invents one; **not** created by this stage.
INTENDED_RESULT_TAG = "stage-f-results-v2"

#: §H.1a — the immutable historical roots. They are **never** written to,
#: regenerated, re-checksummed, merged, moved or deleted, and they **do not
#: block** the `2.0.0` authority: the §I.2 non-empty-root refusal and the
#: §J.1(7) preflight apply to `STORE_ROOT` alone. Their byte-level binding to
#: the `stage-f-results-v1` tag is verified by the frozen protocol validator
#: (§L.3 item 17a), which this implementation neither modifies nor duplicates.
HISTORICAL_STORE_ROOT = "results/stage_f/"
HISTORICAL_FREEZE_ROOT = "results/stage_f_freeze/"
HISTORICAL_RESULT_TAG = "stage-f-results-v1"
HISTORICAL_RESULT_COMMIT = "90bc5ddca720fac080698fb1872c59e48595f9cd"
HISTORICAL_ROOTS = (HISTORICAL_STORE_ROOT, HISTORICAL_FREEZE_ROOT)
#: §0.3.6 — mechanically valid, scientifically superseded. No `1.0.2` value may
#: be consumed as an input, a comparison, a baseline or a tie-break.
HISTORICAL_BLOCKS_NEW_AUTHORITY = False
HISTORICAL_WRITABLE_BY_NEW_AUTHORITY = False
HISTORICAL_READABLE_AS_SCIENTIFIC_INPUT = False
#: §H.1a — the byte-level immutability binding. "Immutable" is only enforceable
#: if drift is mechanically detectable, so the historical freeze is pinned by
#: content and bound to the **tagged** authority rather than to working-tree
#: metadata. These values are transcribed for mirroring only: the byte-for-byte
#: verification against the tagged tree is the frozen protocol validator's job
#: (§L.3 item 17a), and nothing here reads, regenerates or re-checksums them.
HISTORICAL_FREEZE_BINDING_FILE = (
    "results/stage_f_freeze/authoritative_run_binding.json")
HISTORICAL_RESULT_TAG_OBJECT = "c3300d10896b094021cd63297868b249d240169b"
HISTORICAL_FREEZE_FILES = {
    "authoritative_run_binding.json":
        "8d167aed8a37b398f6d72b468a176aef32547676524b4a937b681f97698e52cb",
    "RESULT_CHECKSUMS.sha256":
        "a3f7e448b9f46a3baa91c04afedba901300ff5bbfa7eb91bb3884f44a478ce0a",
}
HISTORICAL_N_FREEZE_FILES = 2
HISTORICAL_FREEZE_FILES_TRACKED_IN_GIT = True
HISTORICAL_ANCHOR_PROVENANCE_SHA256 = (
    "468e939fa86d131afb2ea381c4e586f8ebcf086b8ad5fabe04c8428ed5f2bda7")
HISTORICAL_ANCHOR_CHECKSUMS_SHA256 = (
    "99b677b0cd4145399416b7e6e95a28c941fde1b0e4b1aab7c12fd71414749e5a")
HISTORICAL_ANCHOR_COMPLETION_SHA256 = (
    "bdf01f9356902b969b73574f60a723f6e36f5b254f39a094d23345615eeeb1e6")
HISTORICAL_STORE_CONTENT_N_FILES = 35
HISTORICAL_STORE_CONTENT_DIGEST = (
    "50d3e275e668642a6a83089a70387ee261d6e2aae29d4ed5bfd7d3b4be8769a4")
HISTORICAL_BYTE_DRIFT_IS_FAIL = True
HISTORICAL_IMMUTABLE = True
HISTORICAL_MECHANICALLY_COMPLETE = True
HISTORICAL_SCIENTIFICALLY_SUPERSEDED = True
HISTORICAL_IS_FINAL_SCIENTIFIC_AUTHORITY = False

NAMESPACE_RQ4 = "rq4"
NAMESPACE_SENSITIVITIES = "sensitivities"
NAMESPACE_SVM = "svm"
NAMESPACES = (NAMESPACE_RQ4, NAMESPACE_SENSITIVITIES, NAMESPACE_SVM)

PROVENANCE_FILE = "PROVENANCE.json"
CHECKSUM_FILE = "RESULT_CHECKSUMS.sha256"
COMPLETION_FILE = "COMPLETION.json"
BARRIER_ARTIFACTS = (PROVENANCE_FILE, CHECKSUM_FILE, COMPLETION_FILE)

#: §I.1 — the seven barrier steps, in the one permitted order.
BARRIER_STEPS = (
    "write_artifacts",
    "write_checksum_ledger",
    "flush_and_fsync",
    "reread_from_disk",
    "recompute_and_compare_digests",
    "verify_artifact_set_exact",
    "write_completion_marker_atomically",
)

RUN_KIND_FORMAL = "FORMAL"
RUN_KIND_SMOKE = "SMOKE"

#: `rq4/` — path -> (logical key, expected row count or None for a JSON file).
RQ4_ARTIFACTS = {
    "rq4/per_site_metrics.csv":
        (("method", "alpha", "n_labels", "held_out_site"), 400),
    "rq4/per_repeat_target_local.csv":
        (("alpha", "n_labels", "held_out_site", "repeat"), 21600),
    "rq4/site_macro_metrics.csv": (("method", "alpha", "n_labels"), 20),
    "rq4/marginal_metrics.csv": (("method", "alpha", "n_labels"), 20),
    "rq4/site_bootstrap_intervals.csv":
        (("quantity", "method", "alpha", "n_labels"), 80),
    "rq4/degeneracy.csv": (("method", "alpha", "n_labels"), 20),
    "rq4/conformal_config.json": ((), None),
}

RQ4_PER_SITE_COLUMNS = (
    "method", "alpha", "n_labels", "held_out_site", "guarantee_label",
    "n_eval", "n_cal", "tau", "coverage", "avg_set_size", "singleton_rate",
    "empty_rate", "doubleton_rate", "status", "reason_code", "degeneracy_flag",
    "degeneracy_code", "class_composition_flag",
)
RQ4_PER_REPEAT_COLUMNS = (
    "alpha", "n_labels", "held_out_site", "repeat", "guarantee_label",
    "n_eval", "n_cal", "tau", "coverage", "avg_set_size", "singleton_rate",
    "empty_rate", "doubleton_rate", "status", "reason_code", "degeneracy_flag",
    "degeneracy_code", "class_composition_flag",
)
RQ4_SITE_MACRO_COLUMNS = (
    "method", "alpha", "n_labels", "guarantee_label", "n_sites",
    "n_estimable_sites", "site_macro_coverage", "worst_site_coverage",
    "worst_site", "site_macro_avg_set_size", "site_macro_singleton_rate",
    "site_macro_empty_rate", "site_macro_doubleton_rate", "n_degenerate_sites",
    "status", "reason_code",
)
RQ4_MARGINAL_COLUMNS = (
    "method", "alpha", "n_labels", "guarantee_label", "n_acquisitions",
    "marginal_coverage", "marginal_avg_set_size", "marginal_singleton_rate",
    "marginal_empty_rate", "marginal_doubleton_rate", "status", "reason_code",
)
RQ4_INTERVAL_COLUMNS = (
    "quantity", "method", "alpha", "n_labels", "point_estimate", "ci_low",
    "ci_high", "ci_method", "ci_level", "n_sites", "n_estimable_sites",
    "total_bootstrap_replicates", "valid_bootstrap_replicates", "status",
    "reason_code",
)
RQ4_DEGENERACY_COLUMNS = (
    "method", "alpha", "n_labels", "n_cells", "n_all_inclusive_threshold",
    "n_all_inclusive_realized", "n_all_empty_realized",
)

#: `svm/` — §H.3.
SVM_ARTIFACTS = {
    "svm/per_site_bacc.csv": (("held_out_site",), 23),
    "svm/site_macro_bacc.json": ((), None),
    "svm/fold_selection.csv": (("held_out_site", "pca", "C"), 460),
    "svm/svm_config.json": ((), None),
}
SVM_PER_SITE_COLUMNS = (
    "held_out_site", "n_acquisitions", "n_mdd", "n_hc", "bacc", "selected_pca",
    "selected_C", "inner_macro_bacc", "n_iter_max", "converged", "status",
    "reason_code",
)
#: §H.3.1 — amended at F.7. The key `(held_out_site, pca, C)` and the 460-row
#: cardinality are **unchanged**; what the amendment adds is the one lawful
#: spelling of a candidate score that does not exist. `status` / `reason_code`
#: are the **existing** §G.1 / §G.2 vocabulary — no `candidate_status` /
#: `candidate_reason` fields are registered, and no new status value or reason
#: code is introduced.
SVM_FOLD_SELECTION_COLUMNS = ("held_out_site", "pca", "C", "inner_macro_bacc",
                              "status", "reason_code")
#: §H.3.1 — `inner_macro_bacc` is the candidate **value** field: present and
#: finite exactly when the candidate row is `ESTIMABLE`, and at the missing
#: representation exactly when it is `NOT_ESTIMABLE` with a registered reason.
#: An empty cell without a status is no longer a lawful spelling anywhere in
#: this table (the Attempt #5 defect).
SVM_CANDIDATE_VALUE_FIELD = "inner_macro_bacc"
#: §H.3.1 cases 2 and 3 — the closed candidate-level reason vocabulary.
SVM_CANDIDATE_REASON_CODES = ("SVM_NOT_CONVERGED", "SVM_FOLD_FAILED")
#: §F.7.2 — 4 PCA candidates x 5 `C` values, per outer fold. All 20 rows are
#: mandatory, including every row of an invalidated fold.
SVM_N_CANDIDATES_PER_FOLD = len(SVM_PCA_CANDIDATES) * len(SVM_C_GRID)
#: §H.3.1 / §F.7.2 — all 460 rows are mandatory, including every row of an
#: invalidated fold. Dropping rows to avoid a null is a cardinality violation.
SVM_FOLD_SELECTION_ALL_ROWS_MANDATORY = True
SVM_REDUNDANT_CANDIDATE_STATUS_FIELDS_REGISTERED = False
SVM_USES_EXISTING_STATUS_VOCABULARY = True
#: The Attempt #5 spelling: a blank score with no status at all.
SVM_NULL_SCORE_WITHOUT_STATUS_PERMITTED = False
#: §F.7.2 — the grid keeps being evaluated for deterministic diagnostic
#: completeness wherever it is safe to continue, and successful candidates keep
#: their finite scores; but those scores are an **audit trail, not a candidate
#: pool**, and may never rescue or select a model for an invalidated fold.
SVM_ALL_CANDIDATES_EVALUATED_WHEN_SAFE = True
SVM_DIAGNOSTIC_SCORES_RETAINED = True
SVM_DIAGNOSTIC_SCORES_USABLE_FOR_SELECTION = False
#: §F.7.2 rule 6 — invariant and programming failures still abort the Formal
#: run and must **not** be converted into candidate rows.
SVM_INVARIANT_FAILURES_BECOME_CANDIDATE_ROWS = False

#: §H.3 — the two SVM table paths, named here rather than at their first use so
#: that the status, reason and cross-artifact registries below can key on them
#: instead of repeating the literals. `svm/per_site_bacc.csv` is one row per
#: outer LOSO fold, and the outer folds are the 23 frozen eligible sites, no
#: more and no fewer.
SVM_PER_SITE_TABLE = "svm/per_site_bacc.csv"
SVM_FOLD_SELECTION_TABLE = "svm/fold_selection.csv"
SVM_SITE_FIELD = "held_out_site"

#: `sensitivities/` — §H.4.2. One directory per sensitivity; the tables each one
#: registers, with its logical key and its frozen cardinality.
SENSITIVITY_ARTIFACTS = {
    "sens1_overall_error_ucb": {
        "decisions.csv": (("site", "repeat", "n_labels", "strategy", "method"),
                          46200),
        "evaluations.csv": (("site", "repeat", "n_labels", "strategy", "method"),
                            78600),
        "manifest.json": ((), None),
    },
    "sens2_q_step_002": {
        "decisions.csv": (("site", "repeat", "n_labels", "strategy", "method"),
                          46200),
        "evaluations.csv": (("site", "repeat", "n_labels", "strategy", "method"),
                            78600),
        "manifest.json": ((), None),
    },
    "sens3_rstar_pm_005": {
        "decisions.csv": (("site", "repeat", "n_labels", "strategy", "method",
                           "r_star_offset"), 92400),
        "evaluations.csv": (("site", "repeat", "n_labels", "strategy", "method",
                             "r_star_offset"), 157200),
        "manifest.json": ((), None),
    },
    "sens4_common_site": {
        "decisions.csv": (("site", "repeat", "n_labels", "strategy", "method"),
                          26400),
        "evaluations.csv": (("site", "repeat", "n_labels", "strategy", "method"),
                            46200),
        "manifest.json": ((), None),
    },
    "sens5_s1_attribution": {
        "cells.csv": (SENS56_CELLS_LOGICAL_KEY, 12),
        "validation_units.csv": (SENS56_VALIDATION_UNITS_LOGICAL_KEY, 24),
        "families.csv": (SENS56_FAMILIES_LOGICAL_KEY, 4),
        "manifest.json": ((), None),
    },
    "sens6_realized_coverage_matched": {
        "cells.csv": (SENS56_CELLS_LOGICAL_KEY, 12),
        "validation_units.csv": (SENS56_VALIDATION_UNITS_LOGICAL_KEY, 24),
        "families.csv": (SENS56_FAMILIES_LOGICAL_KEY, 4),
        "threshold_selection.csv": (("population", "held_out_site", "method",
                                     "operating_point"), 192),
        "manifest.json": ((), None),
    },
}

SENS6_THRESHOLD_SELECTION_COLUMNS = (
    "population", "held_out_site", "method", "operating_point",
    "requested_coverage", "n_rows", "n_distinct_conf", "tau_selected",
    "realized_coverage", "coverage_gap", "status", "reason_code",
)


#: `sensitivities/sens{1..4}/decisions.csv` — the parent RQ2/S2 decision schema
#: reproduced field for field, plus the `sensitivity_label` stamp and the
#: `r_star_offset` axis SENS-3 varies. Bound to `stage_f_sensitivities`'s own
#: row builder by test, so a parent-side schema change breaks loudly.
SENS_DECISIONS_COLUMNS = (
    "sensitivity_label", "site", "repeat", "n_labels", "strategy", "method",
    "r_star_offset", "guarantee_tier", "score_type", "guarantee", "state",
    "reason", "selected_q", "selected_candidate_index", "selected_tau",
    "selected_score", "n_eligible", "first_eligible_index",
    "n_passed_candidates", "stopped_at_index", "n_keep_cal", "n_keep_cal_MDD",
    "n_keep_cal_HC", "x_cal_MDD", "x_cal_HC", "n0", "r_star",
)
#: `sensitivities/sens{1..4}/evaluations.csv` — the realized outcome on the
#: disjoint test split, again in the parent's own field names.
SENS_EVALUATIONS_COLUMNS = (
    "sensitivity_label", "site", "repeat", "n_labels", "strategy", "method",
    "r_star_offset", "guarantee_tier", "state", "selected_q", "selected_tau",
    "selected_candidate_index", "realized_field", "r_star", "coverage_test",
    "n_keep_test", "n_keep_test_MDD", "n_keep_test_HC", "ber_test",
    "risk_excess", "realized_outcome",
)

#: The three RQ3-derived tables carry the **parent's** record schema verbatim
#: (`rq3_inference.CellSummary.to_payload`, the validation record written by
#: `rq3_execution._persist_cell_summary`, and `rq3_inference.holm_family`),
#: with only the `sensitivity_label` stamp prepended. Nothing is renamed,
#: dropped or re-derived here, and each tuple is bound to the parent's own
#: literal by test.
SENS56_CELLS_COLUMNS = (
    "sensitivity_label", "population", "operating_point", "method",
    "cell_index", "n_subjects", "n_sites", "rejected_fraction",
    "rejected_fraction_mode", "sigma_site_sq", "sigma_site_sq_condition",
    "sigma_site_sq_interpretation", "blocks",
)
SENS56_VALIDATION_UNITS_COLUMNS = (
    "sensitivity_label", "population", "operating_point", "method", "block",
    "status", "failing_condition", "authoritative_nodes", "validation_nodes",
    "numerical_validation",
)
SENS56_FAMILIES_COLUMNS = (
    "sensitivity_label", "population", "block", "m", "alpha",
    "n_not_estimable", "slots",
)


def _sensitivity_table_columns() -> dict:
    """Every registered sensitivity CSV -> its exact frozen column tuple."""
    out = {}
    for label in SENSITIVITY_LABELS:
        stem = f"{NAMESPACE_SENSITIVITIES}/{label}"
        if label in SENS56_APPLIES_TO:
            out[f"{stem}/{SENS56_CELLS_FILE}"] = SENS56_CELLS_COLUMNS
            out[f"{stem}/{SENS56_VALIDATION_UNITS_FILE}"] = \
                SENS56_VALIDATION_UNITS_COLUMNS
            out[f"{stem}/{SENS56_FAMILIES_FILE}"] = SENS56_FAMILIES_COLUMNS
        else:
            out[f"{stem}/decisions.csv"] = SENS_DECISIONS_COLUMNS
            out[f"{stem}/evaluations.csv"] = SENS_EVALUATIONS_COLUMNS
    out[f"{NAMESPACE_SENSITIVITIES}/sens6_realized_coverage_matched/"
        "threshold_selection.csv"] = ("sensitivity_label",) + \
        SENS6_THRESHOLD_SELECTION_COLUMNS
    return out


#: **Every registered CSV -> its exact, ordered column tuple.** This is the
#: schema the store enforces before an artifact is allowed to reach disk: no
#: missing column, no extra column, no reordering. JSON artifacts carry no
#: tabular schema and are absent from this map by construction.
TABLE_COLUMNS = {
    "rq4/per_site_metrics.csv": RQ4_PER_SITE_COLUMNS,
    "rq4/per_repeat_target_local.csv": RQ4_PER_REPEAT_COLUMNS,
    "rq4/site_macro_metrics.csv": RQ4_SITE_MACRO_COLUMNS,
    "rq4/marginal_metrics.csv": RQ4_MARGINAL_COLUMNS,
    "rq4/site_bootstrap_intervals.csv": RQ4_INTERVAL_COLUMNS,
    "rq4/degeneracy.csv": RQ4_DEGENERACY_COLUMNS,
    "svm/per_site_bacc.csv": SVM_PER_SITE_COLUMNS,
    "svm/fold_selection.csv": SVM_FOLD_SELECTION_COLUMNS,
    **_sensitivity_table_columns(),
}

#: Projection is **never** authorised: a table that does not present exactly its
#: registered columns is rejected, not trimmed. Recorded as a named constant so
#: the prohibition is testable rather than merely absent.
COLUMN_PROJECTION_PERMITTED = False
DUPLICATE_LOGICAL_KEYS_PERMITTED = False

#: The **exact** set of paths `RESULT_CHECKSUMS.sha256` covers, derived from the
#: output registry rather than maintained as a second artifact list. Two frozen
#: exclusions, and no others: the ledger cannot checksum itself, and
#: `COMPLETION.json` is terminal — it is written at barrier step 7, long after
#: the ledger is sealed at step 2, so it can never appear inside it (§I.1).
LEDGER_COVERS_ITSELF = False
LEDGER_COVERS_COMPLETION = False
LEDGER_EXCLUDED = (CHECKSUM_FILE, COMPLETION_FILE)


def expected_checksum_paths() -> tuple:
    """Every registered path the checksum ledger must cover, exactly."""
    return tuple(sorted(set(registered_artifact_paths()) - set(LEDGER_EXCLUDED)))


#: The **missing representation** for a scientific quantity the protocol defines
#: as unavailable. `None` (serialised as an empty CSV field) throughout, plus the
#: literal `"NOT_ESTIMABLE"` used by the bootstrap layer for an interval
#: endpoint. Zero is never a missing representation (§G.4).
MISSING_REPRESENTATIONS = (None, NOT_ESTIMABLE)
#: The two *string* spellings of the missing representation, so a column can be
#: screened for them vectorised without falling back to a per-cell test.
MISSING_STRING_VALUES = ("", NOT_ESTIMABLE)


def is_missing(value) -> bool:
    """Whether `value` is a registered missing representation."""
    if value is None or value is NOT_ESTIMABLE:
        return True
    if isinstance(value, str):
        return value in MISSING_STRING_VALUES
    try:
        import math

        return isinstance(value, float) and math.isnan(value)
    except TypeError:  # pragma: no cover - non-numeric, non-string
        return False


#: Per-artifact status semantics (§G, §H). `value_fields` are the scientific
#: quantities that must be present when the row is `ESTIMABLE` and must carry the
#: missing representation when it is `NOT_ESTIMABLE`; counts, sizes and
#: thresholds are deliberately excluded — a zero count is a fact, and `tau` may
#: legitimately be `+inf`. `estimable_reasons` names the one registered reason a
#: still-`ESTIMABLE` row may carry: the §D.12.3 validity floor, which invalidates
#: the interval endpoints alone and never the point estimate.
STATUS_SEMANTICS = {
    "rq4/per_site_metrics.csv": {
        "vocabulary": "stage_f", "value_fields": REGISTERED_RATES,
        "estimable_reasons": ()},
    "rq4/per_repeat_target_local.csv": {
        "vocabulary": "stage_f", "value_fields": REGISTERED_RATES,
        "estimable_reasons": ()},
    # §D.11 / §G.4 — the worst-site pair is governed by the aggregate's own
    # status: `site_macro` computes `worst_site` / `worst_site_coverage` from the
    # estimable sites of the cell, so when no site is estimable both go missing
    # with the rest. A stale worst site may never survive an inestimable row.
    "rq4/site_macro_metrics.csv": {
        "vocabulary": "stage_f",
        "value_fields": (tuple(f"site_macro_{r}" for r in REGISTERED_RATES)
                         + ("worst_site_coverage", "worst_site")),
        "estimable_reasons": ()},
    "rq4/marginal_metrics.csv": {
        "vocabulary": "stage_f",
        "value_fields": tuple(f"marginal_{r}" for r in REGISTERED_RATES),
        "estimable_reasons": ()},
    "rq4/site_bootstrap_intervals.csv": {
        "vocabulary": "stage_f", "value_fields": ("point_estimate",),
        "estimable_reasons": ("CI_INSUFFICIENT_VALID_REPLICATES",),
        "interval_fields": ("ci_low", "ci_high")},
    "svm/per_site_bacc.csv": {
        "vocabulary": "stage_f", "value_fields": ("bacc",),
        "estimable_reasons": ()},
    # §H.3.1 — amended at F.7. `inner_macro_bacc` is the candidate row's one
    # scientific value: finite when the candidate is ESTIMABLE, at the missing
    # representation when it is not. This entry is what makes the Attempt #5
    # spelling — a blank score with no status at all — unrepresentable.
    SVM_FOLD_SELECTION_TABLE: {
        "vocabulary": "stage_f", "value_fields": (SVM_CANDIDATE_VALUE_FIELD,),
        "estimable_reasons": ()},
    f"{NAMESPACE_SENSITIVITIES}/sens6_realized_coverage_matched/"
    "threshold_selection.csv": {
        "vocabulary": "stage_f",
        "value_fields": ("tau_selected", "realized_coverage", "coverage_gap"),
        "estimable_reasons": ()},
}
#: The RQ3-derived validation records carry the **parent's** 12-code status
#: registry, not Stage-F's two-valued one, and are validated against it.
RQ3_STATUS_TABLES = tuple(
    f"{NAMESPACE_SENSITIVITIES}/{label}/{SENS56_VALIDATION_UNITS_FILE}"
    for label in SENS56_APPLIES_TO)

#: The SENS-5/6 **cell** tables carry a second parent-governed pair: §13's
#: `sigma_site_sq` and the §9.5.1 code that withdrew it. `summarize_cell` reports
#: the variance only when both of the cell's validation units passed and the full
#: fit was estimable, and otherwise emits the literal `NOT_ESTIMABLE` beside a
#: registered parent condition — so the value and the condition are one state,
#: exactly as `status` and `reason_code` are elsewhere. The value column is
#: deliberately outside `NUMERIC_FIELDS` because it is a number *or* that literal
#: by design, which is why it needs a rule of its own.
RQ3_CELL_TABLES = tuple(
    f"{NAMESPACE_SENSITIVITIES}/{label}/{SENS56_CELLS_FILE}"
    for label in SENS56_APPLIES_TO)
RQ3_CELL_VALUE_FIELD = "sigma_site_sq"
RQ3_CELL_CONDITION_FIELD = "sigma_site_sq_condition"

#: The registered **scientific numeric** fields of each table: every column whose
#: producer emits a real number (or the missing representation). A string is not
#: a number here — `"0.5"`, `"nan"` and `"abc"` are all rejected rather than
#: coerced, because a scientific quantity that arrived as text did not come from
#: the estimator that was supposed to produce it (§H.1).
#:
#: Deliberately **excluded**: every logical-key / design-coordinate column
#: (`alpha`, `n_labels`, `repeat`, `pca`, `C`, `r_star_offset`, …), which
#: identifies a cell rather than reporting a result; every categorical or label
#: column — including `selected_pca`, whose frozen domain is
#: `{no-PCA, 50, 100, 200}` and which is governed by `CLOSED_DOMAIN_FIELDS`
#: below; and `sigma_site_sq`, whose parent serialisation is a number *or* a
#: status string by design.
_SENS_DECISION_NUMERIC = (
    "selected_q", "selected_candidate_index", "selected_tau", "selected_score",
    "n_eligible", "first_eligible_index", "n_passed_candidates",
    "stopped_at_index", "n_keep_cal", "n_keep_cal_MDD", "n_keep_cal_HC",
    "x_cal_MDD", "x_cal_HC", "n0", "r_star",
)
_SENS_EVALUATION_NUMERIC = (
    "selected_q", "selected_tau", "selected_candidate_index", "r_star",
    "coverage_test", "n_keep_test", "n_keep_test_MDD", "n_keep_test_HC",
    "ber_test", "risk_excess", "realized_outcome",
)


def _numeric_fields() -> dict:
    out = {
        "rq4/per_site_metrics.csv": ("n_eval", "n_cal", "tau") + REGISTERED_RATES,
        "rq4/per_repeat_target_local.csv":
            ("n_eval", "n_cal", "tau") + REGISTERED_RATES,
        "rq4/site_macro_metrics.csv": (
            ("n_sites", "n_estimable_sites", "n_degenerate_sites",
             "worst_site_coverage")
            + tuple(f"site_macro_{r}" for r in REGISTERED_RATES)),
        "rq4/marginal_metrics.csv": (
            ("n_acquisitions",)
            + tuple(f"marginal_{r}" for r in REGISTERED_RATES)),
        "rq4/site_bootstrap_intervals.csv": (
            "point_estimate", "ci_low", "ci_high", "ci_level", "n_sites",
            "n_estimable_sites", "total_bootstrap_replicates",
            "valid_bootstrap_replicates"),
        "rq4/degeneracy.csv": ("n_cells", "n_all_inclusive_threshold",
                               "n_all_inclusive_realized", "n_all_empty_realized"),
        # `selected_pca` is deliberately absent: it is a closed-domain
        # categorical model-selection state, not a number (CLOSED_DOMAIN_FIELDS).
        "svm/per_site_bacc.csv": ("n_acquisitions", "n_mdd", "n_hc", "bacc",
                                  "selected_C", "inner_macro_bacc",
                                  "n_iter_max"),
        "svm/fold_selection.csv": ("inner_macro_bacc",),
    }
    for label in SENSITIVITY_LABELS:
        stem = f"{NAMESPACE_SENSITIVITIES}/{label}"
        if label in SENS56_APPLIES_TO:
            out[f"{stem}/{SENS56_CELLS_FILE}"] = (
                "cell_index", "n_subjects", "n_sites", "rejected_fraction")
            out[f"{stem}/{SENS56_VALIDATION_UNITS_FILE}"] = (
                "authoritative_nodes", "validation_nodes")
            out[f"{stem}/{SENS56_FAMILIES_FILE}"] = ("m", "alpha",
                                                     "n_not_estimable")
        else:
            out[f"{stem}/decisions.csv"] = _SENS_DECISION_NUMERIC
            out[f"{stem}/evaluations.csv"] = _SENS_EVALUATION_NUMERIC
    out[f"{NAMESPACE_SENSITIVITIES}/sens6_realized_coverage_matched/"
        "threshold_selection.csv"] = ("requested_coverage", "n_rows",
                                      "n_distinct_conf", "tau_selected",
                                      "realized_coverage", "coverage_gap")
    return out


NUMERIC_FIELDS = _numeric_fields()
NUMERIC_STRINGS_COERCED = False

#: The registered **closed-domain categorical** fields: a column whose value is
#: drawn from a frozen finite vocabulary rather than from a number line, and
#: which is therefore validated by *membership*, never by numeric type (§H.1).
#:
#: `svm/per_site_bacc.csv:selected_pca` is the one such field. §F freezes the
#: inner model-selection grid at `{no-PCA, 50, 100, 200}` (`SVM_PCA_CANDIDATES`),
#: and `no-PCA` is a **completed selection**, not a missing value: the inner
#: leave-one-source-site-out CV really did choose the uncompressed
#: representation. Registering it as numeric was a category error — it refused
#: the producer's own legal output — and there is no numeric stand-in for it:
#: `0` is not "no PCA" (a zero-dimensional projection is not the identity), `-1`
#: is not a dimension at all, and `nan` would convert a completed selection into
#: a missing one, which §G.4 forbids.
#:
#: The domain is **derived** from `SVM_PCA_CANDIDATES` rather than restated, in
#: the same `None -> "no-PCA"` spelling `stage_f_svm.run_outer_fold` emits and
#: `svm/fold_selection.csv:pca` already carries, so the frozen grid and its
#: persisted vocabulary cannot drift apart. Membership is by value **and** kind:
#: `"50"` is refused exactly as a stringified balanced accuracy is refused in a
#: numeric field, because text that spells a dimension did not come from the
#: selector; a bool is never a dimension; and a value outside the grid is a
#: selection the protocol never offered.
SVM_SELECTED_PCA_FIELD = "selected_pca"
SVM_SELECTED_PCA_DOMAIN = tuple("no-PCA" if p is None else int(p)
                                for p in SVM_PCA_CANDIDATES)
CLOSED_DOMAIN_FIELDS = {
    "svm/per_site_bacc.csv": {SVM_SELECTED_PCA_FIELD: SVM_SELECTED_PCA_DOMAIN},
}
#: A closed-domain value is never coerced into its domain, and the domain is
#: never widened to admit whatever arrived.
CLOSED_DOMAIN_STRINGS_COERCED = False
#: A row that reports a result reports the state it selected: a closed-domain
#: field may hold the missing representation only where the row's own status
#: says no selection was reached. An `ESTIMABLE` fold with no `selected_pca`
#: is a fold whose reported `bacc` came from an unnamed model.
CLOSED_DOMAIN_REQUIRED_WHEN_ESTIMABLE = True


def closed_domain_fields(relpath: str) -> dict:
    """The registered closed-domain categorical fields of a table, or `{}`."""
    assert_registered_artifact(relpath)
    return dict(CLOSED_DOMAIN_FIELDS.get(relpath, {}))


def closed_domain(relpath: str, field: str) -> tuple:
    """The frozen vocabulary of one registered closed-domain field."""
    assert_registered_artifact(relpath)
    domain = CLOSED_DOMAIN_FIELDS.get(relpath, {}).get(field)
    if domain is None:
        raise StageFRegistryError(
            f"{field!r} is not a registered closed-domain field of {relpath}")
    return tuple(domain)


def in_closed_domain(relpath: str, field: str, value) -> bool:
    """Whether `value` is a registered member of a closed domain.

    Membership is by kind as well as by value. A bool is never a member (`True`
    is not `1`); a string is a member only of the domain's *string* members, so
    `"50"` is refused while the literal `"no-PCA"` is accepted; and a number is
    a member only if it is integral and equals one of the domain's integer
    members, so `0`, `-1`, `50.5` and `"abc"` are refused rather than coerced
    (`CLOSED_DOMAIN_STRINGS_COERCED=False`).

    Integral **floats** are members because a table that carries `"no-PCA"` or a
    `NOT_ESTIMABLE` fold makes pandas hold the column as `object` or `float64`,
    so a dimension the producer emitted as `int` can reach validation as `100.0`.
    That is a container's storage kind, not a different selection, and refusing
    it would refuse the producer's own legal output for a second time. A
    non-integral number is still refused: `50.5` is not a PCA dimension.
    """
    import numbers

    domain = closed_domain(relpath, field)
    integers = tuple(m for m in domain
                     if isinstance(m, int) and not isinstance(m, bool))
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return value in tuple(m for m in domain if isinstance(m, str))
    if isinstance(value, numbers.Real):
        try:
            if not float(value).is_integer():
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        return int(value) in integers
    return False

#: The **only** registered scientific fields in which a non-finite value is a
#: legal protocol value. `t = +inf` is the all-inclusive conformal threshold
#: (§D.10): it is a real, registered construction whose rates are structurally
#: forced, not a numerical failure. Everywhere else a scientific quantity that
#: reached persistence as `nan`, `+inf` or `-inf` is an estimator failure that
#: was never converted into the registered missing representation, and the store
#: refuses it. There is deliberately **no** global infinity exemption: the
#: permission is per artifact, per column (§H.1).
NONFINITE_PERMITTED_FIELDS = {
    "rq4/per_site_metrics.csv": ("tau",),
    "rq4/per_repeat_target_local.csv": ("tau",),
}
for _label in SENSITIVITY_LABELS:
    if _label not in SENS56_APPLIES_TO:
        NONFINITE_PERMITTED_FIELDS[
            f"{NAMESPACE_SENSITIVITIES}/{_label}/decisions.csv"] = ("selected_tau",)
        NONFINITE_PERMITTED_FIELDS[
            f"{NAMESPACE_SENSITIVITIES}/{_label}/evaluations.csv"] = ("selected_tau",)
del _label
NONFINITE_PERMITTED_ANYWHERE = False
NAN_PERMITTED_ANYWHERE = False


def numeric_field_requires_finite(relpath: str, field: str) -> bool:
    """Whether a registered numeric field must hold a finite value (§H.1)."""
    assert_registered_artifact(relpath)
    if field not in NUMERIC_FIELDS.get(relpath, ()):
        raise StageFRegistryError(
            f"{field!r} is not a registered numeric field of {relpath}")
    return field not in NONFINITE_PERMITTED_FIELDS.get(relpath, ())


#: Scientific outputs that a `NOT_ESTIMABLE` row must surrender **in addition**
#: to its `value_fields` and `interval_fields`. Artifact-specific by
#: construction: `rq4/per_repeat_target_local.csv` reports no threshold at all
#: when the repeat has no evaluation points, whereas a `NOT_ESTIMABLE`
#: `rq4/per_site_metrics.csv` row keeps the `tau` its construction really did
#: select, and a non-converged SVM fold keeps the inner selection diagnostics
#: that led to the failure. Design coordinates, provenance columns and counts are
#: never blanked — a zero count is a fact (§G.4).
NOT_ESTIMABLE_UNAVAILABLE_EXTRA = {
    "rq4/per_repeat_target_local.csv": ("tau",),
}

#: Columns that name a **site**, and must therefore hold one of the 23 frozen
#: eligible sites when the row reports a value at all. The eligible-site set is
#: not restated here: it is read from `contracts.FROZEN_SITES`, the project's one
#: site registry, at validation time.
SITE_IDENTIFIER_FIELDS = {"rq4/site_macro_metrics.csv": ("worst_site",)}

#: §D.12 — the one table whose interval endpoints carry their own arithmetic
#: contract (a valid interval is ordered, finite and above the validity floor).
INTERVAL_TABLE = "rq4/site_bootstrap_intervals.csv"
INTERVAL_ENDPOINTS = ("ci_low", "ci_high")
CI_VALIDITY_FLOOR_REASON = "CI_INSUFFICIENT_VALID_REPLICATES"
#: The two replicate counts every interval row carries. `total` is the frozen
#: `B`; `valid` is how many replicates produced a finite statistic. Both are
#: **counts** — integral, never fractional, never boolean — and the validity
#: floor is a statement about `valid`, so a row may not invoke the floor without
#: actually being below it, nor stay silent while below it (§D.12.3).
INTERVAL_TOTAL_FIELD = "total_bootstrap_replicates"
INTERVAL_VALID_FIELD = "valid_bootstrap_replicates"

#: The JSON artifacts whose payload carries a scientific status. A JSON file is
#: not exempt from §G: the same estimability semantics apply, and the same
#: refusal to coerce a scientific quantity out of a string or an arbitrary
#: object. `svm/site_macro_bacc.json` is the one headline SVM number (§F.9).
SVM_MACRO_JSON = "svm/site_macro_bacc.json"
JSON_STATUS_ARTIFACTS = (SVM_MACRO_JSON,)
SVM_MACRO_JSON_KEYS = (
    "carries_confidence_interval", "comparison_to_logistic_regression",
    "estimand", "n_sites_contributing", "n_sites_not_estimable", "reason_code",
    "site_macro_bacc", "status",
)
#: Genuinely numeric, finite when present. The headline BAcc is the scientific
#: value; the two site counts are integral counts.
SVM_MACRO_JSON_VALUE_FIELDS = ("site_macro_bacc",)
SVM_MACRO_JSON_COUNT_FIELDS = ("n_sites_contributing", "n_sites_not_estimable")
SVM_MACRO_JSON_NUMERIC = SVM_MACRO_JSON_VALUE_FIELDS + SVM_MACRO_JSON_COUNT_FIELDS
SVM_MACRO_JSON_STRING_FIELDS = ("estimand", "comparison_to_logistic_regression")
SVM_MACRO_JSON_BOOL_FIELDS = ("carries_confidence_interval",)
#: The §D.12.4 registry name of the SVM headline number.
SVM_CI_REGISTRY_NAME = "svm_site_macro_bacc"
#: §F.9 — the SVM number never carries an interval, whatever else it carries.
SVM_MACRO_CARRIES_CI = False
#: Arbitrary objects are never stringified into a scientific artifact.
JSON_SERIALISATION_COERCES_OBJECTS = False


# ---------------------------------------------------------------------------
# artifact-specific reason semantics (§G.2)
# ---------------------------------------------------------------------------
#: The **module** column of the frozen §G.2 reason-code table, mirrored exactly.
#: A reason code is not a free-floating vocabulary item: the protocol scopes each
#: one to the module that can raise it, and a code that is globally registered is
#: still illegal in an artifact its own module never produces (§G.2).
REASON_CODE_MODULES = {
    "SITE_NOT_ELIGIBLE_AT_N": ("F-RQ4", "F-SENS"),
    "NO_EVALUATION_POINTS": ("F-RQ4", "F-SENS-6"),
    "COVERAGE_MATCH_UNATTAINABLE": ("F-SENS-6",),
    "CI_INSUFFICIENT_VALID_REPLICATES": ("F-RQ4",),
    "NO_ESTIMABLE_SITE_IN_CELL": ("F-RQ4",),
    "SVM_NOT_CONVERGED": ("F-SVM",),
    "SVM_FOLD_FAILED": ("F-SVM",),
}

#: The exact reason vocabulary of each status-carrying artifact, **by state**.
#:
#: `not_estimable` is the closed set a `NOT_ESTIMABLE` row of *this* artifact may
#: name; `estimable` is the closed set a still-`ESTIMABLE` row may name (empty
#: everywhere but the §D.12.3 validity floor). Each entry is the intersection of
#: two authorities: the frozen §G.2 module scope above, and the artifact's own
#: producer — `stage_f_conformal.evaluate_sets` / `collapse_repeats` /
#: `site_macro` / `marginal` / `interval_for`, `stage_f_svm.fold_result` /
#: `site_macro_bacc`, `stage_f_sensitivities.select_realized_coverage_threshold`.
#: Nothing here widens what a producer can emit.
#:
#: `SITE_NOT_ELIGIBLE_AT_N` is in **no** artifact vocabulary, and that is its
#: §G.2 meaning read literally: "the site is not in the frozen per-`N` eligible
#: set, **so the cell does not exist**". A cell that does not exist is not a row
#: that says so — the registered cardinalities (400, 21 600, 20, 80) already
#: enumerate the eligible cells alone, so persisting an ineligible `(site, N)`
#: row would have to *displace* an eligible one. **Absence is the
#: representation** (`NON_PERSISTABLE_REASON_CODES`).
#:
#: The five RQ3-reused codes appear in **no** Stage-F vocabulary: they live only
#: in the SENS-5/6 validation-unit records, which are governed by the parent's
#: own 12-code registry (`RQ3_STATUS_TABLES`, `rq3_protocol.STATUS_CODES`).
ARTIFACT_REASON_CODES = {
    "rq4/per_site_metrics.csv": {
        "not_estimable": ("NO_EVALUATION_POINTS",), "estimable": ()},
    "rq4/per_repeat_target_local.csv": {
        "not_estimable": ("NO_EVALUATION_POINTS",), "estimable": ()},
    "rq4/site_macro_metrics.csv": {
        "not_estimable": ("NO_ESTIMABLE_SITE_IN_CELL",), "estimable": ()},
    "rq4/marginal_metrics.csv": {
        "not_estimable": ("NO_ESTIMABLE_SITE_IN_CELL",), "estimable": ()},
    "rq4/site_bootstrap_intervals.csv": {
        "not_estimable": ("NO_ESTIMABLE_SITE_IN_CELL",),
        "estimable": ("CI_INSUFFICIENT_VALID_REPLICATES",)},
    # §G.2 scopes **both** F-SVM codes to F-SVM, and the SVM outer fold *is* the
    # per-site row: `SVM_FOLD_FAILED` means "the fold raised before producing
    # predictions", which is a statement about one fold, so the per-fold table is
    # where it can be true. `SVM_NOT_CONVERGED` is the narrower liblinear case.
    "svm/per_site_bacc.csv": {
        "not_estimable": ("SVM_NOT_CONVERGED", "SVM_FOLD_FAILED"),
        "estimable": ()},
    # §H.3.1 cases 2 and 3 — a candidate row may be NOT_ESTIMABLE for exactly
    # the same two F-SVM reasons, and for nothing else. `reason_code` is set
    # **only** when the row is NOT_ESTIMABLE, exactly as §G.2 requires
    # everywhere else in Stage F.
    SVM_FOLD_SELECTION_TABLE: {
        "not_estimable": SVM_CANDIDATE_REASON_CODES, "estimable": ()},
    SVM_MACRO_JSON: {
        "not_estimable": ("SVM_FOLD_FAILED",), "estimable": ()},
    f"{NAMESPACE_SENSITIVITIES}/sens6_realized_coverage_matched/"
    "threshold_selection.csv": {
        "not_estimable": ("NO_EVALUATION_POINTS", "COVERAGE_MATCH_UNATTAINABLE"),
        "estimable": ()},
}


def artifact_reason_codes(relpath: str, status: str) -> tuple:
    """The closed reason vocabulary of one artifact in one state (§G.2)."""
    assert_registered_artifact(relpath)
    if status not in STATUS_VALUES:
        raise StageFRegistryError(f"{status!r} is not one of {STATUS_VALUES}")
    scope = ARTIFACT_REASON_CODES.get(relpath)
    if scope is None:
        raise StageFRegistryError(
            f"{relpath} has no registered reason vocabulary; classify it in "
            "ARTIFACT_REASON_CODES before it may carry a reason_code (§G.2)")
    return tuple(scope["estimable" if status == ESTIMABLE else "not_estimable"])


def assert_reason_code_for(relpath: str, status: str, code: str) -> str:
    """Raise unless `code` is legal **in this artifact, in this state** (§G.2).

    Global registration is necessary and not sufficient. `SVM_NOT_CONVERGED` is
    a registered Stage-F code and is still nonsense on an RQ4 coverage row;
    `CI_INSUFFICIENT_VALID_REPLICATES` is registered and still nonsense anywhere
    but the one interval table; `NO_ELIGIBLE_START` is a registered *parent* code
    and belongs only to a SENS-5/6 validation unit. The artifact, the status and
    the reason are one state, so they are checked as one.
    """
    assert_reason_code(code)                      # closed registry + withdrawals
    allowed = artifact_reason_codes(relpath, status)
    if code not in allowed:
        raise StageFRegistryError(
            f"{relpath}: {code!r} is not a reason this artifact may carry when "
            f"{status}; its registered vocabulary is {allowed or '(none)'} "
            f"(§G.2 scopes {code!r} to "
            f"{REASON_CODE_MODULES.get(code, 'the parent RQ3 registry')}). A "
            "reason code being globally known is not permission to use it here")
    return code


#: Counts a `NOT_ESTIMABLE` row must report as **exactly zero** under a given
#: reason, because the producer's own branch condition was that count being zero.
#: `NO_EVALUATION_POINTS` is emitted from `n == 0`; `NO_ESTIMABLE_SITE_IN_CELL`
#: from an empty estimable-site list, which forces the derived counts with it.
#: A reason may never be asserted over a count that contradicts it (§G.4).
REASON_REQUIRED_ZERO_COUNTS = {
    ("rq4/per_site_metrics.csv", "NO_EVALUATION_POINTS"): ("n_eval",),
    ("rq4/per_repeat_target_local.csv", "NO_EVALUATION_POINTS"): ("n_eval",),
    ("rq4/site_macro_metrics.csv", "NO_ESTIMABLE_SITE_IN_CELL"):
        ("n_estimable_sites", "n_degenerate_sites"),
    ("rq4/marginal_metrics.csv", "NO_ESTIMABLE_SITE_IN_CELL"): ("n_acquisitions",),
    ("rq4/site_bootstrap_intervals.csv", "NO_ESTIMABLE_SITE_IN_CELL"):
        ("n_estimable_sites", "valid_bootstrap_replicates"),
    (f"{NAMESPACE_SENSITIVITIES}/sens6_realized_coverage_matched/"
     "threshold_selection.csv", "NO_EVALUATION_POINTS"):
        ("n_rows", "n_distinct_conf"),
}

#: Counts that must be **strictly positive** under a given reason: the producer
#: reached that branch only after the count was non-zero. `COVERAGE_MATCH_
#: UNATTAINABLE` is raised on a non-empty confidence vector whose candidates all
#: cover more than `c*`; an empty vector is `NO_EVALUATION_POINTS` instead, and
#: the two states are not interchangeable (§E.7.1).
REASON_REQUIRED_POSITIVE_COUNTS = {
    (f"{NAMESPACE_SENSITIVITIES}/sens6_realized_coverage_matched/"
     "threshold_selection.csv", "COVERAGE_MATCH_UNATTAINABLE"):
        ("n_rows", "n_distinct_conf"),
}

#: The mirror image, per artifact: counts an `ESTIMABLE` row must report as
#: strictly positive, because the producer could not have computed the value
#: otherwise. Without this, "no evaluation points" and "an evaluated row" would
#: be distinguishable only by a label the writer chose.
ESTIMABLE_REQUIRED_POSITIVE_COUNTS = {
    "rq4/per_site_metrics.csv": ("n_eval",),
    "rq4/per_repeat_target_local.csv": ("n_eval",),
    "rq4/site_macro_metrics.csv": ("n_estimable_sites",),
    "rq4/marginal_metrics.csv": ("n_acquisitions",),
    "rq4/site_bootstrap_intervals.csv": ("n_estimable_sites",),
    f"{NAMESPACE_SENSITIVITIES}/sens6_realized_coverage_matched/"
    "threshold_selection.csv": ("n_rows", "n_distinct_conf"),
}

#: §F.8 — the SVM per-site table carries its own convergence flag, and the flag
#: is not decorative: `SVM_NOT_CONVERGED` is *defined* as liblinear reporting
#: non-convergence, so the row must actually say so, and an `ESTIMABLE` row must
#: say the opposite. A genuine boolean, never a string and never a 0/1.
CONVERGENCE_FLAG_FIELDS = {"svm/per_site_bacc.csv": "converged"}
#: Both F-SVM failure states leave the fold without a usable final fit, so
#: neither may claim a converged one. `SVM_NOT_CONVERGED` *is* liblinear
#: reporting non-convergence; `SVM_FOLD_FAILED` is a fold that raised **before
#: producing predictions**, which cannot have reached a converged prediction
#: either. An `ESTIMABLE` fold must say the opposite (handled in the store).
CONVERGENCE_FLAG_BY_REASON = {
    ("svm/per_site_bacc.csv", "SVM_NOT_CONVERGED"): False,
    ("svm/per_site_bacc.csv", "SVM_FOLD_FAILED"): False,
}

#: §G.2 — codes that are **registered and never persistable as a row**. The
#: protocol's own wording is the rule: a site that is not eligible at `N` has no
#: cell, and a table row is precisely the assertion that a cell exists. Keeping
#: the code registered matters (it is the frozen §G.2 vocabulary and the
#: validator checks the module column both ways); giving it an artifact would
#: contradict the cardinalities that already exclude it.
NON_PERSISTABLE_REASON_CODES = ("SITE_NOT_ELIGIBLE_AT_N",)

#: Fields a `SVM_FOLD_FAILED` row may not report, over and above its value
#: field: the fold raised **before predictions existed**, so nothing computed
#: from predictions can be on the row. `bacc` is the value field and is already
#: surrendered by the generic `NOT_ESTIMABLE` rule; this names the same fact at
#: the reason level so the two SVM failure states are not interchangeable.
REASON_PREDICTION_DERIVED_FIELDS = {
    ("svm/per_site_bacc.csv", "SVM_FOLD_FAILED"): ("bacc",),
}


# ---------------------------------------------------------------------------
# structural cell registries (§H.2, §H.3, §G.2 `SITE_NOT_ELIGIBLE_AT_N`)
# ---------------------------------------------------------------------------
#: Tables whose logical-key **set** is a closed frozen enumeration, checked for
#: exact equality rather than for cardinality alone: a table of the right length
#: built from the wrong cells is exactly the failure mode `SITE_NOT_ELIGIBLE_AT_N`
#: was being misused to record. Each entry maps a registered artifact to the
#: frozen key tuples it must hold, in no particular order.
EXACT_KEY_SET_TABLES = {
    "rq4/site_macro_metrics.csv": tuple(REGISTERED_CELLS),
    "rq4/marginal_metrics.csv": tuple(REGISTERED_CELLS),
    "rq4/degeneracy.csv": tuple(REGISTERED_CELLS),
    "rq4/site_bootstrap_intervals.csv": tuple(REQUIRED_INTERVALS),
}

#: The two RQ4 tables carrying one row per **site x cell**. The site axis is not
#: enumerable from the frozen constants (the per-`N` eligible *identities* live
#: in the frozen split registry, a run-time input), so what is frozen — and what
#: is enforced — is its exact **shape**: which cells exist, how many distinct
#: sites each cell holds, that every site is one of the 23 frozen eligible sites,
#: that the site set does not move between the `alpha` values of one `N`, and
#: that it does not move between the two tables. An ineligible `(site, N)` cell
#: cannot be added without breaking one of those.
RQ4_SITE_CELL_TABLES = ("rq4/per_site_metrics.csv",
                        "rq4/per_repeat_target_local.csv")
RQ4_SITE_FIELD = "held_out_site"
RQ4_REPEAT_FIELD = "repeat"
#: `rq4/per_repeat_target_local.csv` has no `method` column: §H.2 registers it
#: as the `target_local` per-repeat table, and its 21 600 rows are
#: `54 site-N cells x 100 repeats x 4 alpha`.
RQ4_PER_REPEAT_METHOD = METHOD_TARGET_LOCAL


def expected_sites_in_cell(method: str, n_labels) -> int:
    """How many distinct held-out sites one registered RQ4 cell holds (§H.2).

    The two zero-label methods are evaluated at every site of the base cohort;
    `target_local` needs `N` target calibration labels, so its cell exists only
    at the sites the frozen split registry admits at that `N` — `SITES_PER_N`.
    """
    if method not in CONFORMAL_METHODS:
        raise StageFRegistryError(f"{method!r} is not a registered §D.2 method")
    if method != METHOD_TARGET_LOCAL:
        if n_labels is not None:
            raise StageFRegistryError(
                f"{method} carries no label budget; §H.2 fixes n_labels null "
                f"(NOT_APPLICABLE, §D.6), got {n_labels!r}")
        return BASE_COHORT_SITES
    if n_labels not in N_GRID:
        raise StageFRegistryError(
            f"{METHOD_TARGET_LOCAL} n_labels={n_labels!r} is not one of the "
            f"frozen §D.6 grid {N_GRID}")
    return SITES_PER_N[int(n_labels)]


#: §D.6 — eligibility at `N` is "the site can supply `N` target calibration
#: labels", which cannot become *easier* as `N` grows. The frozen counts
#: `23 >= 22 >= 21 >= 11` are that statement's arithmetic shadow, and the frozen
#: split registry satisfies the set form. Enforcing it catches an ineligible site
#: substituted for an eligible one at a single `N`, which the counts alone would
#: accept.
ELIGIBILITY_IS_MONOTONE_IN_N = True
assert all(SITES_PER_N[a] >= SITES_PER_N[b]
           for a, b in zip(S2_N_GRID, S2_N_GRID[1:]))

#: §H.3 — the headline number is `NOT_ESTIMABLE` exactly when fewer than the
#: 23 folds are estimable, and its two site counts are not free-standing: they
#: partition the fold table. `stage_f_svm.site_macro_bacc` computes them that
#: way, so the JSON and the CSV are one statement in two files.
SVM_MACRO_CONTRIBUTING_FIELD = "n_sites_contributing"
SVM_MACRO_NOT_ESTIMABLE_FIELD = "n_sites_not_estimable"
SVM_MACRO_REQUIRED_SITES = BASE_COHORT_SITES

# ---------------------------------------------------------------------------
# SENS-1..4: the parent RQ2/S2 state vocabulary, inherited unchanged (§H.4)
# ---------------------------------------------------------------------------
#: §H.4 gives SENS-1..4 "the parent's own registered output schema unchanged",
#: and §G.2 adds that "RQ2/S2's `not_testable_reason_split` vocabulary is reused
#: only by SENS-1..4, and only inside its parent's own decision artifacts". These
#: tables therefore carry no Stage-F `status` column and are **not** unrestricted
#: because of it: every state, tier, score, guarantee, reason and realized-field
#: value is the parent's, and the allowed values are read from
#: `src/certification` and `src/s2_decisions` at validation time. No copy is kept
#: here — only which columns are governed.
SENS_DECISION_TABLES = tuple(
    f"{NAMESPACE_SENSITIVITIES}/{label}/decisions.csv"
    for label in SENSITIVITY_LABELS if label not in SENS56_APPLIES_TO)
SENS_EVALUATION_TABLES = tuple(
    f"{NAMESPACE_SENSITIVITIES}/{label}/evaluations.csv"
    for label in SENSITIVITY_LABELS if label not in SENS56_APPLIES_TO)
SENS_INHERITED_STATE_FIELD = "state"
SENS_INHERITED_REASON_FIELD = "reason"
SENS_INHERITED_TIER_FIELD = "guarantee_tier"
SENS_INHERITED_SCORE_FIELD = "score_type"
SENS_INHERITED_GUARANTEE_FIELD = "guarantee"
SENS_INHERITED_REALIZED_FIELD = "realized_field"
SENS_STRATEGY_FIELD = "strategy"
#: `certification.SELECTION_FIELDS` under the parent's own names; the parent's
#: `s2_decisions.assert_decision_field_consistency` is called on every row whose
#: sensitivity leaves the candidate grid alone, so this list exists only to name
#: the columns for the SENS-2 grid, where the parent's literal `Q_GRID` check
#: does not apply (§E.3 changes the grid step, nothing else).
SENS_SELECTION_FIELDS = ("selected_q", "selected_tau",
                         "selected_candidate_index", "selected_score")
SENS2_LABEL = "sens2_q_step_002"
#: §H.4.2 — `decisions.csv` cardinality is `|REG| . S_few . M`: the label-
#: consuming strategies alone. `evaluations.csv` adds the zero-label ones.
SENS_DECISION_STRATEGIES = PARENT_STRATEGIES_LABEL_CONSUMING
SENS_EVALUATION_STRATEGIES = (PARENT_STRATEGIES_LABEL_CONSUMING
                              + PARENT_STRATEGIES_ZERO_LABEL)
#: The five calibration counts `_decision_row` fills from the parent's retained
#: counts. They exist only where the strategy actually held target calibration
#: labels; a zero-label decision reports none of them.
SENS_CALIBRATION_COUNT_FIELDS = ("n_keep_cal", "n_keep_cal_MDD", "n_keep_cal_HC",
                                 "x_cal_MDD", "x_cal_HC")
SENS_EVALUATION_COUNT_FIELDS = ("n_keep_test", "n_keep_test_MDD",
                                "n_keep_test_HC")
SENS_EVALUATION_VALUE_FIELDS = ("ber_test", "risk_excess", "realized_outcome")

# ---------------------------------------------------------------------------
# SENS-5/6: the nested block record and its sibling representations (§H.4.3)
# ---------------------------------------------------------------------------
#: `cells.csv` carries the two block results **inside the row** (§H.4.3), in the
#: `blocks` column, and `validation_units.csv` carries the same two results one
#: row each. Both are transcriptions of the same `rq3_inference.BlockResult`, so
#: they are checked against each other rather than each on its own.
SENS56_CELLS_BLOCKS_FIELD = "blocks"
SENS56_BLOCK_KEYS = (
    "block", "cell_index", "delta_cv_logloss", "delta_mcfadden", "detail",
    "df", "failing_condition", "failure_detail", "failure_is_numerical",
    "lrt_pvalue_raw", "lrt_statistic", "method", "numerical_validation",
    "operating_point", "population", "status", "validation_unit",
)
#: The four block-level scientific quantities, present exactly when the block is
#: estimable and the parent's `NOT_ESTIMABLE` literal otherwise.
SENS56_BLOCK_VALUE_FIELDS = ("lrt_statistic", "lrt_pvalue_raw",
                             "delta_mcfadden", "delta_cv_logloss")
#: The fields a nested block and its `validation_units.csv` row must agree on
#: verbatim. `numerical_validation` is compared separately, because the parent
#: substitutes a canonical empty record on the unit side (§9.6.1 step 6).
SENS56_BLOCK_UNIT_SHARED_FIELDS = ("status", "failing_condition")
SENS56_FAMILIES_SLOTS_FIELD = "slots"
SENS56_FAMILY_SLOT_KEYS = ("cell_index", "method", "operating_point", "p_holm",
                           "p_raw", "significant")
SENS56_FAMILY_PVALUE_FIELD = "p_raw"
#: the same number under its `BlockResult` name (§12.3 draws one from the other)
SENS56_BLOCK_PVALUE_FIELD = "lrt_pvalue_raw"
SENS56_FAMILY_NOT_ESTIMABLE_FIELD = "n_not_estimable"

# ---------------------------------------------------------------------------
# the per-`N` eligible-site identities — declared here, derived at run time
# ---------------------------------------------------------------------------
#: The frozen production split registry is the *only* authority on **which**
#: sites are eligible at each label budget `N`. That is run-time data, not a
#: frozen constant, so nothing here enumerates the identities: a second list
#: kept in this module would be a second authority, free to drift from the
#: registry the run actually verified. What is declared here is only **where**
#: those identities are persisted, so that every such artifact can be bound
#: mechanically to the one verified mapping (§D.6, §D.9).
#:
#: Each entry names the artifact's site column, its label-budget column, the
#: label budgets that must appear in it, an optional restriction of the eligible
#: set (SENS-4 evaluates the 11 common sites alone, §E.5), and an optional
#: `(column, value)` filter selecting the rows whose site axis is the
#: target-local eligible set rather than the base cohort.
SPLIT_IDENTITY_SITE_FIELD = "site_field"
SPLIT_IDENTITY_N_FIELD = "n_field"


def _split_identity_spec(site_field, n_field, n_values, *,
                         restrict_to=None, row_filter=None) -> dict:
    return {SPLIT_IDENTITY_SITE_FIELD: site_field,
            SPLIT_IDENTITY_N_FIELD: n_field,
            "n_values": tuple(n_values),
            "restrict_to": None if restrict_to is None else tuple(restrict_to),
            "row_filter": row_filter}


SPLIT_IDENTITY_TABLES = {
    # F-RQ4: `target_local` is the one method that consumes target labels, so it
    # is the one whose site axis is the split registry's eligible set (§D.6).
    "rq4/per_site_metrics.csv": _split_identity_spec(
        RQ4_SITE_FIELD, "n_labels", N_GRID,
        row_filter=("method", METHOD_TARGET_LOCAL)),
    # the per-repeat table carries no `method` column: §H.2 registers it as the
    # target-local table outright, so every row is governed.
    "rq4/per_repeat_target_local.csv": _split_identity_spec(
        RQ4_SITE_FIELD, "n_labels", N_GRID),
    # F-SENS 1..4 re-run the parent's own S2 protocol over the same registry
    # rows, so their `(site, n_labels)` axis is that registry's, at the parent's
    # full label grid (§H.4.2).
    **{path: _split_identity_spec(
        "site", "n_labels", S2_N_GRID,
        restrict_to=(COMMON_SITES if "sens4_common_site" in path else None))
       for path in SENS_DECISION_TABLES + SENS_EVALUATION_TABLES},
}


def split_identity_spec(relpath: str):
    """How one artifact encodes the split registry's eligible-site identities."""
    assert_registered_artifact(relpath)
    spec = SPLIT_IDENTITY_TABLES.get(relpath)
    return None if spec is None else dict(spec)


def expected_identity_sites(spec: dict, n_labels: int, eligible) -> frozenset:
    """The eligible identities one artifact must show at `N`, from `eligible`.

    `eligible` is the verified run-time mapping. This function narrows it by the
    artifact's own frozen restriction and never widens it, so no caller can
    obtain a site the verified registry did not admit at that `N`.
    """
    sites = frozenset(str(s) for s in eligible)
    restrict = spec.get("restrict_to")
    if restrict is not None:
        sites &= frozenset(restrict)
    return sites


def numeric_fields(relpath: str) -> tuple:
    """The registered scientific numeric columns of a table, or `()`."""
    assert_registered_artifact(relpath)
    return tuple(NUMERIC_FIELDS.get(relpath, ()))


def table_columns(relpath: str):
    """The exact frozen column tuple of a registered CSV, or `None` for JSON."""
    assert_registered_artifact(relpath)
    return TABLE_COLUMNS.get(relpath)


def logical_key(relpath: str):
    """The registered logical key of an artifact, or `()` for a JSON file."""
    assert_registered_artifact(relpath)
    if relpath in RQ4_ARTIFACTS:
        return tuple(RQ4_ARTIFACTS[relpath][0])
    if relpath in SVM_ARTIFACTS:
        return tuple(SVM_ARTIFACTS[relpath][0])
    parts = relpath.split("/")
    if len(parts) == 3 and parts[0] == NAMESPACE_SENSITIVITIES:
        return tuple(SENSITIVITY_ARTIFACTS.get(parts[1], {})
                     .get(parts[2], ((), None))[0])
    return ()


def expected_rows(relpath: str):
    """The registered cardinality of an artifact, or `None` where unbounded."""
    assert_registered_artifact(relpath)
    if relpath in RQ4_ARTIFACTS:
        return RQ4_ARTIFACTS[relpath][1]
    if relpath in SVM_ARTIFACTS:
        return SVM_ARTIFACTS[relpath][1]
    parts = relpath.split("/")
    if len(parts) == 3 and parts[0] == NAMESPACE_SENSITIVITIES:
        return SENSITIVITY_ARTIFACTS.get(parts[1], {}).get(parts[2],
                                                           ((), None))[1]
    return None


def registered_artifact_paths() -> tuple:
    """Every registered relative path in the authoritative store, sorted.

    This is the set barrier step 6 verifies **exactly**: the store must contain
    these paths and nothing else.
    """
    paths = list(RQ4_ARTIFACTS) + list(SVM_ARTIFACTS)
    for label, files in SENSITIVITY_ARTIFACTS.items():
        paths.extend(f"{NAMESPACE_SENSITIVITIES}/{label}/{name}" for name in files)
    paths.extend(BARRIER_ARTIFACTS)
    return tuple(sorted(paths))


def assert_registered_artifact(relpath: str) -> str:
    """Raise unless `relpath` is a registered Stage-F output (§H.1)."""
    if relpath not in registered_artifact_paths():
        raise StageFRegistryError(
            f"{relpath!r} is not in the frozen Stage-F output registry (§H); "
            "no artifact outside the registry may appear in the store")
    return relpath


# ---------------------------------------------------------------------------
# two-way structured mirroring against docs/stage_f_constants.json (§L.7)
# ---------------------------------------------------------------------------
#: The mirrored view of the JSON registry, in the JSON's own shape. Compared in
#: both directions by `assert_no_constant_drift`; `_`-prefixed JSON keys are
#: documentation and are skipped on both sides.
MIRROR = {
    "protocol_version": PROTOCOL_VERSION,
    "supersedes_version": SUPERSEDED_PROTOCOL_VERSION,
    "supersedes_commit": SUPERSEDED_PROTOCOL_COMMIT,
    "supersedes_tag": SUPERSEDED_PROTOCOL_TAG,
    "protocol_doc": PROTOCOL_DOC,
    "validator": PROTOCOL_VALIDATOR,
    "intended_tag": PROTOCOL_TAG,
    # The draft-lifecycle flag. It records the state the *protocol document* was
    # frozen in; Stage F.2 requires the tag to exist and asserts that separately
    # (see `stage_f_inputs.assert_protocol_tag`), never by editing this mirror.
    "tag_exists": False,
    "store_root": STORE_ROOT,
    "freeze_root": FREEZE_ROOT,
    "intended_result_tag": INTENDED_RESULT_TAG,
    # §H.1a — the immutable `1.0.2` roots, mirrored so that a silent edit to
    # either side is a drift failure rather than a quiet divergence.
    "historical_authority": {
        "store_root": HISTORICAL_STORE_ROOT,
        "freeze_root": HISTORICAL_FREEZE_ROOT,
        "protocol_tag": SUPERSEDED_PROTOCOL_TAG,
        "protocol_commit": SUPERSEDED_PROTOCOL_COMMIT,
        "protocol_version": SUPERSEDED_PROTOCOL_VERSION,
        "result_tag": HISTORICAL_RESULT_TAG,
        "result_commit": HISTORICAL_RESULT_COMMIT,
        "freeze_binding_file": HISTORICAL_FREEZE_BINDING_FILE,
        "freeze_store_content_digest": HISTORICAL_STORE_CONTENT_DIGEST,
        "immutable": HISTORICAL_IMMUTABLE,
        "mechanically_complete_and_cryptographically_valid":
            HISTORICAL_MECHANICALLY_COMPLETE,
        "scientifically_superseded": HISTORICAL_SCIENTIFICALLY_SUPERSEDED,
        "is_final_stage_f_scientific_authority":
            HISTORICAL_IS_FINAL_SCIENTIFIC_AUTHORITY,
        "blocks_new_authority": HISTORICAL_BLOCKS_NEW_AUTHORITY,
        "writable_by_new_authority": HISTORICAL_WRITABLE_BY_NEW_AUTHORITY,
        "readable_as_scientific_input":
            HISTORICAL_READABLE_AS_SCIENTIFIC_INPUT,
        "result_tag_object": HISTORICAL_RESULT_TAG_OBJECT,
        "freeze_files": dict(HISTORICAL_FREEZE_FILES),
        "n_freeze_files": HISTORICAL_N_FREEZE_FILES,
        "freeze_files_tracked_in_git": HISTORICAL_FREEZE_FILES_TRACKED_IN_GIT,
        "store_anchor_provenance_sha256": HISTORICAL_ANCHOR_PROVENANCE_SHA256,
        "store_anchor_result_checksums_sha256":
            HISTORICAL_ANCHOR_CHECKSUMS_SHA256,
        "store_anchor_completion_sha256": HISTORICAL_ANCHOR_COMPLETION_SHA256,
        "store_content_n_files": HISTORICAL_STORE_CONTENT_N_FILES,
        "byte_drift_is_fail": HISTORICAL_BYTE_DRIFT_IS_FAIL,
    },
    "svm_fold_selection": {
        "file": SVM_FOLD_SELECTION_TABLE,
        "logical_key": list(SVM_ARTIFACTS[SVM_FOLD_SELECTION_TABLE][0]),
        "fields": [SVM_CANDIDATE_VALUE_FIELD, "status", "reason_code"],
        "expected_rows": SVM_ARTIFACTS[SVM_FOLD_SELECTION_TABLE][1],
        "cardinality_formula":
            "n_sites * n_pca_candidates * n_c_grid = 23 * 4 * 5",
        "all_rows_mandatory": SVM_FOLD_SELECTION_ALL_ROWS_MANDATORY,
        "redundant_candidate_status_fields_registered":
            SVM_REDUNDANT_CANDIDATE_STATUS_FIELDS_REGISTERED,
        "uses_existing_status_vocabulary": SVM_USES_EXISTING_STATUS_VOCABULARY,
        "estimable_status": ESTIMABLE,
        "not_estimable_status": NOT_ESTIMABLE,
        "reason_code_convergence": SVM_CANDIDATE_REASON_CODES[0],
        "reason_code_expected_failure": SVM_CANDIDATE_REASON_CODES[1],
        "reason_code_when_estimable": None,
        "null_score_without_status_permitted":
            SVM_NULL_SCORE_WITHOUT_STATUS_PERMITTED,
        "all_candidates_evaluated_when_safe":
            SVM_ALL_CANDIDATES_EVALUATED_WHEN_SAFE,
        "diagnostic_scores_retained_for_successful_candidates":
            SVM_DIAGNOSTIC_SCORES_RETAINED,
        "diagnostic_scores_usable_for_selection_in_invalidated_fold":
            SVM_DIAGNOSTIC_SCORES_USABLE_FOR_SELECTION,
        "invariant_failures_converted_to_candidate_rows":
            SVM_INVARIANT_FAILURES_BECOME_CANDIDATE_ROWS,
    },
    "svm_fold_failed_scope": {
        "code": "SVM_FOLD_FAILED",
        "is_catch_all": SVM_FOLD_FAILED_IS_CATCH_ALL,
        "admits_only": SVM_FOLD_FAILED_ADMITS_ONLY,
        "must_not_absorb": list(SVM_FOLD_FAILED_MUST_NOT_ABSORB),
        "unexpected_errors_fail_hard":
            SVM_FOLD_FAILED_UNEXPECTED_ERRORS_FAIL_HARD,
        "substitutes_for_svm_not_converged":
            SVM_FOLD_FAILED_SUBSTITUTES_FOR_NOT_CONVERGED,
        "allowed_exception_types": list(SVM_FOLD_FAILED_ALLOWED_TYPES),
        "n_allowed_exception_types": SVM_FOLD_FAILED_N_ALLOWED_TYPES,
        "n_must_not_absorb": SVM_FOLD_FAILED_N_MUST_NOT_ABSORB,
        "allow_list_is_closed_and_exhaustive":
            SVM_FOLD_FAILED_ALLOW_LIST_IS_CLOSED,
        "type_match_rule": SVM_FOLD_FAILED_TYPE_MATCH_RULE,
        "exact_type_predicates": list(SVM_FOLD_FAILED_EXACT_TYPE_PREDICATES),
        "isinstance_matching_permitted":
            SVM_FOLD_FAILED_ISINSTANCE_PERMITTED,
        "subclass_widening_permitted":
            SVM_FOLD_FAILED_SUBCLASS_WIDENING_PERMITTED,
        "broad_exception_catching_permitted":
            SVM_FOLD_FAILED_BROAD_CATCHING_PERMITTED,
        "identity_equivalent_aliases":
            list(SVM_FOLD_FAILED_IDENTITY_ALIASES),
        "aliases_add_no_semantic_type": SVM_FOLD_FAILED_ALIASES_ADD_NO_TYPE,
        "n_semantic_failure_types": SVM_FOLD_FAILED_N_SEMANTIC_TYPES,
        "requires_estimator_backend_boundary":
            SVM_FOLD_FAILED_REQUIRES_BOUNDARY,
        "estimator_backend_boundary": SVM_FOLD_FAILED_BOUNDARY_DESCRIPTION,
        "estimator_backend_boundary_inside":
            list(SVM_FOLD_FAILED_BOUNDARY_INSIDE),
        "estimator_backend_boundary_outside":
            list(SVM_FOLD_FAILED_BOUNDARY_OUTSIDE),
        "outside_boundary_is_hard_failure":
            SVM_FOLD_FAILED_OUTSIDE_BOUNDARY_IS_HARD_FAILURE,
        "linalgerror_subclasses_valueerror":
            SVM_FOLD_FAILED_LINALGERROR_SUBCLASSES_VALUEERROR,
        "bare_valueerror_permitted": SVM_FOLD_FAILED_BARE_VALUEERROR_PERMITTED,
        "estimator_settings_altered_to_avoid_failures":
            SVM_ESTIMATOR_SETTINGS_ALTERED_TO_AVOID_FAILURES,
    },
    "sens1_score_type_correction": {
        "scientific_definition_changed":
            SENS1_SCORE_TYPE_SCIENTIFIC_DEFINITION_CHANGED,
        "numeric_recomputation_authorized":
            SENS1_SCORE_TYPE_NUMERIC_RECOMPUTATION_AUTHORIZED,
        # The protocol document records this as work the *future*, separately
        # authorized implementation stage must do. This runtime **is** that
        # stage, and the flag is the protocol's own field: it describes what
        # `2.0.0` implemented, not what the implementation did, so it stays
        # `false` and is not edited to claim credit.
        "implemented_by_this_version": SENS1_IMPLEMENTED_BY_PROTOCOL_VERSION,
        "actual_functional": SENS1_ACTUAL_FUNCTIONAL,
        "persisted_label_in_1_0_2": SENS1_PERSISTED_LABEL_IN_1_0_2,
        "required_label": SENS1_OVERALL_ERROR_SCORE_TYPE,
    },
    "amendment_2_0_0": {
        "trigger": AMENDMENT_TRIGGER,
        "attempt_number": AMENDMENT_ATTEMPT_NUMBER,
        "scientific_svm_definition_changed": SVM_DEFINITION_CHANGED_BY_2_0_0,
        "fail_closed_rule_weakened": AMENDMENT_FAIL_CLOSED_RULE_WEAKENED,
        "solver_or_grid_or_tolerance_or_max_iter_changed":
            AMENDMENT_SOLVER_OR_GRID_CHANGED,
        "is_post_result_relaxation": AMENDMENT_IS_POST_RESULT_RELAXATION,
        "scope": list(AMENDMENT_SCOPE),
        # The protocol document's own record of what the *amendment* did. The
        # amendment changed no runtime and authorized no run; those flags
        # describe `2.0.0` the document, and are never edited by the runtime
        # that implements it.
        "runtime_implementation_changed": AMENDMENT_RUNTIME_CHANGED,
        "formal_run_authorized": AMENDMENT_FORMAL_RUN_AUTHORIZED,
        "stage_g_authorized": AMENDMENT_STAGE_G_AUTHORIZED,
        "new_result_tag_created": AMENDMENT_NEW_RESULT_TAG_CREATED,
        "draft_audit_verdict": AMENDMENT_DRAFT_AUDIT_VERDICT,
        "draft_audit_critical_findings": AMENDMENT_DRAFT_AUDIT_CRITICAL,
        "draft_audit_non_critical_findings": AMENDMENT_DRAFT_AUDIT_NON_CRITICAL,
        "draft_corrections_are_scientific_changes":
            AMENDMENT_DRAFT_CORRECTIONS_ARE_SCIENTIFIC,
        "old_implementation_failures_expected":
            AMENDMENT_OLD_IMPLEMENTATION_FAILURES_EXPECTED,
        "protocol_pinning_weakened": AMENDMENT_PROTOCOL_PINNING_WEAKENED,
        "validator_coverage_audit_verdict": AMENDMENT_COVERAGE_AUDIT_VERDICT,
        "validator_coverage_audit_critical_findings":
            AMENDMENT_COVERAGE_AUDIT_CRITICAL,
        "validator_coverage_audit_accepted_protocol_content":
            AMENDMENT_COVERAGE_AUDIT_ACCEPTED_CONTENT,
        "validator_hardening_is_scientific_change":
            AMENDMENT_HARDENING_IS_SCIENTIFIC_CHANGE,
        "validator_hardening_changed_runtime":
            AMENDMENT_HARDENING_CHANGED_RUNTIME,
    },
    "svm_per_site_warning_representation": {
        "file": SVM_PER_SITE_TABLE,
        "representations_are_distinct":
            SVM_WARNING_REPRESENTATIONS_ARE_DISTINCT,
        "new_status_or_reason_code_introduced":
            SVM_WARNING_INTRODUCES_NEW_STATUS_OR_REASON,
        "cohort_fields_always_persisted":
            list(SVM_COHORT_FIELDS_ALWAYS_PERSISTED),
        "inner_grid_warning": dict(SVM_INNER_GRID_WARNING_REPRESENTATION),
        "retained_value_token": SVM_RETAINED_VALUE_TOKEN,
        "final_refit_warning": dict(SVM_FINAL_REFIT_WARNING_REPRESENTATION),
    },
    "cohort": {
        "base_cohort_n": BASE_COHORT_N,
        "base_cohort_sites": BASE_COHORT_SITES,
        "source_oof_rows": SOURCE_OOF_ROWS,
        "target_rows": TARGET_ROWS,
        "split_registry_rows": SPLIT_REGISTRY_ROWS,
        "split_registry_rows_n_ge_10": SPLIT_REGISTRY_ROWS_N_GE_10,
        "sites_per_n": sites_per_n_canonical(),
        "r_split": R_SPLIT,
        "common_sites": list(COMMON_SITES),
        "common_site_registry_rows": COMMON_SITE_REGISTRY_ROWS,
        "common_site_registry_rows_n_ge_10": COMMON_SITE_REGISTRY_ROWS_N_GE_10,
    },
    "rq4": {
        "alpha_grid": list(ALPHA_GRID),
        "n_grid": list(N_GRID),
        "s2_n_grid": list(S2_N_GRID),
        "conformal_tolerance": CONFORMAL_TOLERANCE,
        "nonconformity_score": NONCONFORMITY_SCORE,
        "index_rule": INDEX_RULE,
        "interpolation_permitted": INTERPOLATION_PERMITTED,
        "tie_inclusion": TIE_INCLUSION,
        "methods": list(CONFORMAL_METHODS),
        "guarantee_labels": dict(GUARANTEE_LABELS),
        "method_information_budget": dict(METHOD_INFORMATION_BUDGET),
        "exact_guarantee_permitted_for": list(EXACT_GUARANTEE_PERMITTED_FOR),
        "weight_column": WEIGHT_COLUMN,
        "weighted_test_point_atom_included": WEIGHTED_TEST_POINT_ATOM_INCLUDED,
        "registered_rates": list(REGISTERED_RATES),
        "primary_rates": list(PRIMARY_RATES),
        "diagnostic_rates": list(DIAGNOSTIC_RATES),
        "n_registered_cells": N_REGISTERED_CELLS,
    },
    "rq4_ci": {
        "site_bootstrap_b": SITE_BOOTSTRAP_B,
        "ci_level": CI_LEVEL,
        "ci_quantiles": list(CI_QUANTILES),
        "ci_method": CI_METHOD,
        "quantile_method": QUANTILE_METHOD,
        "min_valid_replicates": MIN_VALID_REPLICATES,
        "bootstrap_entropy": list(BOOTSTRAP_ENTROPY),
        "resampling_unit": RESAMPLING_UNIT,
        "required": list(F_CI_REQUIRED),
        "n_ci_slots": N_CI_SLOTS,
        "prohibited": list(F_CI_PROHIBITED),
        "hypothesis_test": HYPOTHESIS_TEST,
        "multiplicity_correction": MULTIPLICITY_CORRECTION,
        "tost": TOST,
        "bootstrap_site_set_sizes": list(BOOTSTRAP_SITE_SET_SIZES),
    },
    "status": {
        "status_values": list(STATUS_VALUES),
        "reason_codes": list(REASON_CODES),
        "reason_codes_reused_from_rq3": list(REASON_CODES_REUSED_FROM_RQ3),
        "reason_codes_reused_by": list(REASON_CODES_REUSED_BY),
        "withdrawn_reason_codes": list(WITHDRAWN_REASON_CODES),
        "degeneracy_codes": list(DEGENERACY_CODES),
        "degeneracy_implies_not_estimable": DEGENERACY_IMPLIES_NOT_ESTIMABLE,
        "class_composition_flags": list(CLASS_COMPOSITION_FLAGS),
        "class_composition_implies_not_estimable":
            CLASS_COMPOSITION_IMPLIES_NOT_ESTIMABLE,
        "class_conditional_conformal_endpoint_registered":
            CLASS_CONDITIONAL_CONFORMAL_ENDPOINT_REGISTERED,
    },
    "sensitivities": {
        "n_sensitivities": N_SENSITIVITIES,
        "labels": list(SENSITIVITY_LABELS),
        "combat_in_mandatory_core": COMBAT_IN_MANDATORY_CORE,
        "sens1_alpha_ucb": SENS1_ALPHA_UCB,
        "sens1_bonferroni": SENS1_BONFERRONI,
        "sens2_q_step": SENS2_Q_STEP,
        "sens2_n_candidates": SENS2_N_CANDIDATES,
        "sens3_r_star_offsets": list(SENS3_R_STAR_OFFSETS),
        "parent_strategies_label_consuming":
            list(PARENT_STRATEGIES_LABEL_CONSUMING),
        "parent_strategies_zero_label": list(PARENT_STRATEGIES_ZERO_LABEL),
        "parent_confidence_methods": list(PARENT_CONFIDENCE_METHODS),
        "cardinality": dict(CARDINALITY),
    },
    "sens56_schema": {
        "applies_to": list(SENS56_APPLIES_TO),
        "populations": list(SENS56_POPULATIONS),
        "cells_file": SENS56_CELLS_FILE,
        "cells_logical_key": list(SENS56_CELLS_LOGICAL_KEY),
        "cells_expected_rows": SENS56_CELLS_EXPECTED_ROWS,
        "block_in_cells_logical_key": SENS56_BLOCK_IN_CELLS_LOGICAL_KEY,
        "validation_units_file": SENS56_VALIDATION_UNITS_FILE,
        "validation_units_logical_key": list(SENS56_VALIDATION_UNITS_LOGICAL_KEY),
        "validation_units_expected_rows": SENS56_VALIDATION_UNITS_EXPECTED_ROWS,
        "families_file": SENS56_FAMILIES_FILE,
        "families_logical_key": list(SENS56_FAMILIES_LOGICAL_KEY),
        "families_expected_rows": SENS56_FAMILIES_EXPECTED_ROWS,
        "holm_family_size": SENS56_HOLM_FAMILY_SIZE,
        "blocks_per_population": SENS56_BLOCKS_PER_POPULATION,
        "validation_units_per_cell": SENS56_VALIDATION_UNITS_PER_CELL,
        "blocks_by_population": {k: list(v) for k, v
                                 in SENS56_BLOCKS_BY_POPULATION.items()},
        "blocks_carried_within_cell_record":
            SENS56_BLOCKS_CARRIED_WITHIN_CELL_RECORD,
        "parent_cell_registry_size": SENS56_PARENT_CELL_REGISTRY_SIZE,
        "parent_validation_unit": list(SENS56_PARENT_VALIDATION_UNIT),
    },
    "gate1_oof_binding": {
        "helper": GATE1_BINDING_HELPER,
        "helper_invoked_by_validator": True,
        "unledgered_file": GATE1_UNLEDGERED_FILE,
        "ledger_bound_file": GATE1_LEDGER_BOUND_FILE,
        "logical_key": list(GATE1_LOGICAL_KEY),
        "exact_columns": list(GATE1_EXACT_COLUMNS),
        "n_exact_columns": GATE1_N_EXACT_COLUMNS,
        "exact_comparison": "pandas.Series.equals",
        "exact_comparison_is_positional": True,
        "tolerance_columns": list(GATE1_TOLERANCE_COLUMNS),
        "oof_prob_mdd_atol": GATE1_OOF_PROB_MDD_ATOL,
        "tolerance_rule": GATE1_TOLERANCE_RULE,
        "oof_prob_mdd_bit_identical_required":
            GATE1_OOF_PROB_MDD_BIT_IDENTICAL_REQUIRED,
        "row_count_equality_required": GATE1_ROW_COUNT_EQUALITY_REQUIRED,
        "duplicate_logical_keys_permitted": GATE1_DUPLICATE_LOGICAL_KEYS_PERMITTED,
        "binding_is_one_to_one": GATE1_BINDING_IS_ONE_TO_ONE,
        "csv_float_precision": GATE1_CSV_FLOAT_PRECISION,
        "n_oof_rows": GATE1_N_OOF_ROWS,
        "r_star_crosscheck_atol": GATE1_R_STAR_CROSSCHECK_ATOL,
        "r_star_crosscheck_sites": GATE1_R_STAR_CROSSCHECK_SITES,
    },
    "sens6": {
        "requested_coverage_by_operating_point": dict(SENS6_REQUESTED_COVERAGE),
        "operating_points": list(SENS6_OPERATING_POINTS),
        "candidate_set": SENS6_CANDIDATE_SET,
        "coverage_definition": SENS6_COVERAGE_DEFINITION,
        "selection_rule": SENS6_SELECTION_RULE,
        "retention_comparison": SENS6_RETENTION_COMPARISON,
        "abstention_comparison": SENS6_ABSTENTION_COMPARISON,
        "interpolation_permitted": SENS6_INTERPOLATION_PERMITTED,
        "randomisation_permitted": SENS6_RANDOMISATION_PERMITTED,
        "uses_target_labels": SENS6_USES_TARGET_LABELS,
        "not_estimable_codes": list(SENS6_NOT_ESTIMABLE_CODES),
        "full_sample_n": SENS6_FULL_SAMPLE_N,
        "full_sample_sites": SENS6_FULL_SAMPLE_SITES,
        "mdd_only_n": SENS6_MDD_ONLY_N,
        "mdd_only_sites": SENS6_MDD_ONLY_SITES,
        "worked_examples": [
            {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in example.items()}
            for example in SENS6_WORKED_EXAMPLES
        ],
    },
    "svm": {
        "estimator": SVM_ESTIMATOR,
        "estimand": SVM_ESTIMAND,
        "c_grid": list(SVM_C_GRID),
        "pca_candidates": list(SVM_PCA_CANDIDATES),
        "penalty": SVM_PENALTY,
        "loss": SVM_LOSS,
        "dual": SVM_DUAL,
        "class_weight": SVM_CLASS_WEIGHT,
        "fit_intercept": SVM_FIT_INTERCEPT,
        "tol": SVM_TOL,
        "max_iter": SVM_MAX_ITER,
        "random_state": SVM_RANDOM_STATE,
        "random_state_status": SVM_RANDOM_STATE_STATUS,
        "decision_rule": SVM_DECISION_RULE,
        "probability_calibration": SVM_PROBABILITY_CALIBRATION,
        "platt_scaling": SVM_PLATT_SCALING,
        "sample_weight": SVM_SAMPLE_WEIGHT,
        "c_scale_equivalence_claimed": SVM_C_SCALE_EQUIVALENCE_CLAIMED,
        "forbidden_outputs": list(SVM_FORBIDDEN_OUTPUTS),
        "outer_split": SVM_OUTER_SPLIT,
        "inner_split": SVM_INNER_SPLIT,
        # §0.3.2 — the absence of change, asserted rather than remembered.
        "definition_changed_by_2_0_0": SVM_DEFINITION_CHANGED_BY_2_0_0,
        "any_fit_convergence_warning_invalidates_outer_fold":
            SVM_ANY_FIT_WARNING_INVALIDATES_FOLD,
        "inner_fit_counts_as_fold_lifecycle_fit":
            SVM_INNER_FIT_IS_FOLD_LIFECYCLE_FIT,
        "reduced_grid_selection_permitted":
            SVM_REDUCED_GRID_SELECTION_PERMITTED,
        "final_refit_after_inner_warning_permitted":
            SVM_FINAL_REFIT_AFTER_INNER_WARNING_PERMITTED,
        "selection_requires_all_candidates_estimable":
            SVM_SELECTION_REQUIRES_ALL_CANDIDATES_ESTIMABLE,
        "n_candidates_per_fold": SVM_N_CANDIDATES_PER_FOLD,
        "tie_break_order": list(SVM_TIE_BREAK_ORDER),
        "tie_break_pca_order": list(SVM_TIE_BREAK_PCA_ORDER),
        "converged_requires_all_inner_fits_clean":
            SVM_CONVERGED_REQUIRES_ALL_INNER_FITS_CLEAN,
        "converged_requires_clean_final_refit":
            SVM_CONVERGED_REQUIRES_CLEAN_FINAL_REFIT,
        "converged_requires_fold_estimable":
            SVM_CONVERGED_REQUIRES_FOLD_ESTIMABLE,
        "n_iter_max_is_configured_cap": SVM_N_ITER_MAX_IS_CONFIGURED_CAP,
        "n_iter_max_is_observed_n_iter": SVM_N_ITER_MAX_IS_OBSERVED_N_ITER,
        "n_iter_max_when_refit_attempted":
            SVM_N_ITER_MAX_WHEN_REFIT_ATTEMPTED,
        "n_iter_max_when_refit_skipped": SVM_N_ITER_MAX_WHEN_REFIT_SKIPPED,
        "observed_iteration_count_persisted":
            SVM_OBSERVED_ITERATION_COUNT_PERSISTED,
    },
    "consumed_artifacts": dict(CONSUMED_ARTIFACTS),
    "n_consumed_artifacts": N_CONSUMED_ARTIFACTS,
    "recomputable_artifacts": list(RECOMPUTABLE_ARTIFACTS),
}


def load_constant_registry(root: Path = ROOT) -> dict:
    """Read `docs/stage_f_constants.json` — the protocol's constant authority."""
    return json.loads((root / CONSTANTS_DOC).read_text())


def _flatten(node, prefix: str = "") -> dict:
    """Flatten to `dotted.key -> scalar`; `_`-prefixed keys are documentation."""
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).startswith("_"):
                continue
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            out.update(_flatten(value, f"{prefix}[{i}]"))
    else:
        out[prefix] = node
    return out


def assert_no_constant_drift(root: Path = ROOT) -> dict:
    """Two-way structured comparison against the JSON registry (§L.3 item 18).

    Every mirrored constant must appear in the registry with an equal value, and
    every registry constant must appear here with an equal value. Keys and
    values are compared as parsed objects, never as substrings of prose.
    """
    registry = _flatten(load_constant_registry(root))
    mirrored = _flatten(MIRROR)

    missing_here = sorted(set(registry) - set(mirrored))
    missing_there = sorted(set(mirrored) - set(registry))
    unequal = sorted(k for k in set(registry) & set(mirrored)
                     if registry[k] != mirrored[k])
    if missing_here or missing_there or unequal:
        raise StageFProtocolDriftError(
            f"Stage-F constant drift against {CONSTANTS_DOC}: "
            f"{len(missing_here)} registry keys absent from the implementation "
            f"{missing_here[:5]}, {len(missing_there)} implementation keys "
            f"absent from the registry {missing_there[:5]}, "
            f"{len(unequal)} unequal values {unequal[:5]}")
    return {"n_constants_compared": len(registry),
            "n_missing_in_implementation": 0,
            "n_missing_in_registry": 0,
            "n_unequal": 0}


def assert_protocol_documents_unchanged(root: Path = ROOT) -> dict:
    """The three protocol-authority files still hash to their frozen digests."""
    observed = {}
    for relpath, expected in (
            (PROTOCOL_DOC, EXPECTED_PROTOCOL_DOC_SHA256),
            (CONSTANTS_DOC, EXPECTED_CONSTANTS_SHA256),
            (PROTOCOL_VALIDATOR, EXPECTED_PROTOCOL_VALIDATOR_SHA256)):
        path = root / relpath
        if not path.exists():
            raise StageFProtocolDriftError(f"{relpath} is not on disk")
        digest = sha256_file(path)
        if digest != expected:
            raise StageFProtocolDriftError(
                f"{relpath} hashes to {digest[:12]}..., the frozen "
                f"{PROTOCOL_TAG} value is {expected[:12]}...")
        observed[relpath] = digest
    return observed


# ---------------------------------------------------------------------------
# structural self-assertions — cheap, and run at import
# ---------------------------------------------------------------------------
EXPECTED_CHECKSUM_PATHS = expected_checksum_paths()
N_EXPECTED_CHECKSUM_PATHS = len(EXPECTED_CHECKSUM_PATHS)

assert len(REGISTERED_CELLS) == 20 == N_REGISTERED_CELLS
assert len(set(REGISTERED_CELLS)) == 20
assert N_CI_SLOTS == 80 == len(F_CI_REQUIRED) * N_REGISTERED_CELLS
assert not (set(F_CI_REQUIRED) & set(F_CI_PROHIBITED)), "CI registry overlaps"
assert 5 not in N_GRID, "N = 5 is excluded from target-local conformal (§D.6)"
assert set(N_GRID) < set(S2_N_GRID)
assert len(SENSITIVITY_LABELS) == N_SENSITIVITIES == 6
assert len(set(SENSITIVITY_LABELS)) == 6, "the sensitivity set is closed"
assert not (set(WITHDRAWN_REASON_CODES) & set(REASON_CODES))
assert GUARANTEE_LABELS[METHOD_TARGET_LOCAL] == "EXACT_WITHIN_TARGET_SITE"
assert tuple(k for k, v in GUARANTEE_LABELS.items()
             if v.startswith("EXACT")) == EXACT_GUARANTEE_PERMITTED_FOR
assert "block" not in SENS56_CELLS_LOGICAL_KEY
assert (tuple(SENS56_VALIDATION_UNITS_LOGICAL_KEY)
        == SENS56_CELLS_LOGICAL_KEY + ("block",))
assert (SENS56_CELLS_EXPECTED_ROWS * SENS56_VALIDATION_UNITS_PER_CELL
        == SENS56_VALIDATION_UNITS_EXPECTED_ROWS == 24)
assert len(COMMON_SITES) == 11 == len(set(COMMON_SITES))
assert len(SVM_C_GRID) == 5 and len(SVM_PCA_CANDIDATES) == 4
assert SVM_N_CANDIDATES_PER_FOLD == 20
assert (SVM_ARTIFACTS[SVM_FOLD_SELECTION_TABLE][1]
        == BASE_COHORT_SITES * SVM_N_CANDIDATES_PER_FOLD == 460)
# §F.7.5 — the tie-break order is a permutation of the candidate set, not a
# second grid: same members, `no-PCA` moved from first to last.
assert set(SVM_TIE_BREAK_PCA_ORDER) == set(SVM_PCA_CANDIDATES)
assert len(SVM_TIE_BREAK_PCA_ORDER) == len(SVM_PCA_CANDIDATES)
assert SVM_TIE_BREAK_PCA_ORDER[-1] is None, "no-PCA is the weakest compression"
assert SVM_TIE_BREAK_PCA_ORDER[:-1] == (50, 100, 200)
# §H.3.1 — the candidate reason vocabulary is the existing F-SVM one, and adds
# no new code to the closed §G.2 registry.
assert set(SVM_CANDIDATE_REASON_CODES) <= set(REASON_CODES)
assert SVM_CANDIDATE_VALUE_FIELD in SVM_FOLD_SELECTION_COLUMNS
assert {"status", "reason_code"} <= set(SVM_FOLD_SELECTION_COLUMNS)
# §G.2a — exactly two permitted types, and none of them appears in the
# exclusion list. An allow-list that overlapped its own exclusions would be
# unenforceable in either direction.
assert len(SVM_FOLD_FAILED_ALLOWED_TYPES) == SVM_FOLD_FAILED_N_ALLOWED_TYPES == 2
assert len(SVM_FOLD_FAILED_MUST_NOT_ABSORB) == SVM_FOLD_FAILED_N_MUST_NOT_ABSORB
assert not (set(SVM_FOLD_FAILED_ALLOWED_TYPES)
            & set(SVM_FOLD_FAILED_MUST_NOT_ABSORB))
assert "SVMNotConverged" not in SVM_FOLD_FAILED_ALLOWED_TYPES
# §F.7.1 vs §F.7.1a — the two representations agree on the fold verdict and
# disagree on exactly the retained-evidence fields.
assert (SVM_INNER_GRID_WARNING_REPRESENTATION["status"]
        == SVM_FINAL_REFIT_WARNING_REPRESENTATION["status"] == NOT_ESTIMABLE)
assert (SVM_INNER_GRID_WARNING_REPRESENTATION["reason_code"]
        == SVM_FINAL_REFIT_WARNING_REPRESENTATION["reason_code"]
        == "SVM_NOT_CONVERGED")
assert (SVM_INNER_GRID_WARNING_REPRESENTATION["converged"]
        is SVM_FINAL_REFIT_WARNING_REPRESENTATION["converged"] is False)
assert (SVM_INNER_GRID_WARNING_REPRESENTATION["bacc"]
        is SVM_FINAL_REFIT_WARNING_REPRESENTATION["bacc"] is None)
assert SVM_INNER_GRID_WARNING_REPRESENTATION["n_iter_max"] is None
assert SVM_FINAL_REFIT_WARNING_REPRESENTATION["n_iter_max"] == SVM_MAX_ITER
for _field in ("selected_pca", "selected_C", "inner_macro_bacc"):
    assert SVM_INNER_GRID_WARNING_REPRESENTATION[_field] is None
    assert (SVM_FINAL_REFIT_WARNING_REPRESENTATION[_field]
            == SVM_RETAINED_VALUE_TOKEN)
del _field
# §E.2a — the corrected label is a new string, distinct from the parent's.
assert SENS1_OVERALL_ERROR_SCORE_TYPE != SENS1_PERSISTED_LABEL_IN_1_0_2
assert len(registered_artifact_paths()) == len(set(registered_artifact_paths()))
# Every registered CSV has an exact schema, and only CSVs do.
assert set(TABLE_COLUMNS) == {p for p in registered_artifact_paths()
                              if p.endswith(".csv")}
assert all(len(c) == len(set(c)) for c in TABLE_COLUMNS.values())
# A logical key is a subset of its own table's columns, so uniqueness is checkable.
assert all(set(logical_key(p)) <= set(cols) for p, cols in TABLE_COLUMNS.items())
assert not COLUMN_PROJECTION_PERMITTED and not DUPLICATE_LOGICAL_KEYS_PERMITTED
# Status semantics may only be declared for tables that actually carry a status.
assert all("status" in TABLE_COLUMNS[p] for p in STATUS_SEMANTICS)
assert all("reason_code" in TABLE_COLUMNS[p] for p in STATUS_SEMANTICS)
assert all(set(s["value_fields"]) <= set(TABLE_COLUMNS[p])
           for p, s in STATUS_SEMANTICS.items())
assert all(not (set(s["estimable_reasons"]) & set(WITHDRAWN_REASON_CODES))
           for s in STATUS_SEMANTICS.values())
assert all("status" in TABLE_COLUMNS[p] for p in RQ3_STATUS_TABLES)
assert all("failing_condition" in TABLE_COLUMNS[p] for p in RQ3_STATUS_TABLES)
assert 0 not in MISSING_REPRESENTATIONS and not is_missing(0.0)
# The checksum-covered set is the registry minus exactly the two frozen
# exclusions — never a hand-maintained second list of scientific artifacts.
assert set(EXPECTED_CHECKSUM_PATHS) == set(registered_artifact_paths()) - {
    CHECKSUM_FILE, COMPLETION_FILE}
assert CHECKSUM_FILE not in EXPECTED_CHECKSUM_PATHS
assert COMPLETION_FILE not in EXPECTED_CHECKSUM_PATHS
assert PROVENANCE_FILE in EXPECTED_CHECKSUM_PATHS
assert N_EXPECTED_CHECKSUM_PATHS == len(registered_artifact_paths()) - 2
assert not LEDGER_COVERS_ITSELF and not LEDGER_COVERS_COMPLETION
# Every registered CSV declares its numeric fields, and each names real columns
# that are not part of the table's own logical key (a coordinate is not a result).
assert set(NUMERIC_FIELDS) == set(TABLE_COLUMNS)
assert all(set(cols) <= set(TABLE_COLUMNS[p]) for p, cols in NUMERIC_FIELDS.items())
assert all(not (set(cols) & set(logical_key(p)))
           for p, cols in NUMERIC_FIELDS.items())
assert not NUMERIC_STRINGS_COERCED
# A closed-domain field is a real column of a real table, is never *also* a
# numeric field (the two rules are alternatives, not layers), is never a logical
# key coordinate, and holds a non-empty frozen vocabulary of hashable members.
# `selected_pca`'s vocabulary is the frozen §F PCA grid in the producer's own
# spelling, so the grid cannot be widened without widening the domain with it.
assert all(p in TABLE_COLUMNS for p in CLOSED_DOMAIN_FIELDS)
assert all(set(fields) <= set(TABLE_COLUMNS[p])
           for p, fields in CLOSED_DOMAIN_FIELDS.items())
assert all(not (set(fields) & set(NUMERIC_FIELDS.get(p, ())))
           for p, fields in CLOSED_DOMAIN_FIELDS.items())
assert all(not (set(fields) & set(logical_key(p)))
           for p, fields in CLOSED_DOMAIN_FIELDS.items())
assert all(len(domain) and len(set(domain)) == len(domain)
           for fields in CLOSED_DOMAIN_FIELDS.values()
           for domain in fields.values())
assert not CLOSED_DOMAIN_STRINGS_COERCED and CLOSED_DOMAIN_REQUIRED_WHEN_ESTIMABLE
assert SVM_SELECTED_PCA_DOMAIN == ("no-PCA", 50, 100, 200)
assert len(SVM_SELECTED_PCA_DOMAIN) == len(SVM_PCA_CANDIDATES)
assert closed_domain(SVM_PER_SITE_TABLE, SVM_SELECTED_PCA_FIELD) \
    == SVM_SELECTED_PCA_DOMAIN
assert all(in_closed_domain(SVM_PER_SITE_TABLE, SVM_SELECTED_PCA_FIELD, m)
           for m in SVM_SELECTED_PCA_DOMAIN)
assert not any(in_closed_domain(SVM_PER_SITE_TABLE, SVM_SELECTED_PCA_FIELD, v)
               for v in ("abc", "50", "100", 0, -1, 1, 150, 50.5, 0.0, -1.0,
                         float("nan"), float("inf"), True, False, None, ""))
# a pandas float upcast of a real member is the same selection, not a new one
assert all(in_closed_domain(SVM_PER_SITE_TABLE, SVM_SELECTED_PCA_FIELD, float(m))
           for m in SVM_SELECTED_PCA_DOMAIN if isinstance(m, int))
# The uncompressed candidate is a *selection*, never a missing representation.
assert "no-PCA" in SVM_SELECTED_PCA_DOMAIN and not is_missing("no-PCA")
assert not any(is_missing(m) for m in SVM_SELECTED_PCA_DOMAIN)
# Every status-governed numeric value field is also declared numeric.
assert all(set(s["value_fields"]) - set(SITE_IDENTIFIER_FIELDS.get(p, ()))
           <= set(NUMERIC_FIELDS[p]) for p, s in STATUS_SEMANTICS.items())
assert all(set(cols) <= set(TABLE_COLUMNS[p])
           for p, cols in SITE_IDENTIFIER_FIELDS.items())
assert set(INTERVAL_ENDPOINTS) == set(STATUS_SEMANTICS[INTERVAL_TABLE]
                                      ["interval_fields"])
assert CI_VALIDITY_FLOOR_REASON in REASON_CODES
assert INTERVAL_TOTAL_FIELD in NUMERIC_FIELDS[INTERVAL_TABLE]
assert INTERVAL_VALID_FIELD in NUMERIC_FIELDS[INTERVAL_TABLE]
# The non-finite permission is a closed, per-column list over real numeric
# columns, and it never reaches a status-governed scientific value field.
assert not NONFINITE_PERMITTED_ANYWHERE and not NAN_PERMITTED_ANYWHERE
assert all(set(cols) <= set(NUMERIC_FIELDS[p])
           for p, cols in NONFINITE_PERMITTED_FIELDS.items())
assert all(not (set(cols) & set(STATUS_SEMANTICS.get(p, {}).get("value_fields", ())))
           for p, cols in NONFINITE_PERMITTED_FIELDS.items())
assert all(numeric_field_requires_finite(p, f)
           for p in NUMERIC_FIELDS for f in NUMERIC_FIELDS[p]
           if f not in NONFINITE_PERMITTED_FIELDS.get(p, ()))
assert not any(numeric_field_requires_finite(p, f)
               for p, cols in NONFINITE_PERMITTED_FIELDS.items() for f in cols)
# The extra NOT_ESTIMABLE surrenders name real columns that are not already
# governed, and never a logical-key coordinate.
assert all(set(cols) <= set(TABLE_COLUMNS[p])
           for p, cols in NOT_ESTIMABLE_UNAVAILABLE_EXTRA.items())
assert all(not (set(cols) & set(logical_key(p)))
           for p, cols in NOT_ESTIMABLE_UNAVAILABLE_EXTRA.items())
assert all(not (set(cols) & set(STATUS_SEMANTICS[p]["value_fields"]))
           for p, cols in NOT_ESTIMABLE_UNAVAILABLE_EXTRA.items())
assert set(NOT_ESTIMABLE_UNAVAILABLE_EXTRA) <= set(STATUS_SEMANTICS)
# The one status-carrying JSON artifact is registered, and its schema is exact.
assert all(p in registered_artifact_paths() for p in JSON_STATUS_ARTIFACTS)
assert all(p.endswith(".json") for p in JSON_STATUS_ARTIFACTS)
assert SVM_MACRO_JSON_KEYS == tuple(sorted(SVM_MACRO_JSON_KEYS))
assert set(SVM_MACRO_JSON_NUMERIC + SVM_MACRO_JSON_STRING_FIELDS
           + SVM_MACRO_JSON_BOOL_FIELDS + ("status", "reason_code")) \
    == set(SVM_MACRO_JSON_KEYS)
assert not SVM_MACRO_CARRIES_CI and not requires_ci(SVM_CI_REGISTRY_NAME)
assert not JSON_SERIALISATION_COERCES_OBJECTS

# Every status-carrying artifact has an exact, closed reason vocabulary, and the
# two states never share a code: a reason that only invalidates a derived
# quantity may not also declare the whole row inestimable, and vice versa.
assert set(ARTIFACT_REASON_CODES) == set(STATUS_SEMANTICS) | {SVM_MACRO_JSON}
assert all(set(v) == {"estimable", "not_estimable"}
           for v in ARTIFACT_REASON_CODES.values())
assert all(not (set(v["estimable"]) & set(v["not_estimable"]))
           for v in ARTIFACT_REASON_CODES.values())
assert all(set(v["estimable"]) | set(v["not_estimable"]) <= set(REASON_CODES)
           for v in ARTIFACT_REASON_CODES.values())
# The per-artifact vocabularies partition the **persistable** part of the
# Stage-F registry: every registered code either has at least one home or is
# declared non-persistable, never both, and none of them is a parent RQ3 code.
assert set(NON_PERSISTABLE_REASON_CODES) <= set(REASON_CODES)
assert set().union(*(set(v["estimable"]) | set(v["not_estimable"])
                     for v in ARTIFACT_REASON_CODES.values())) \
    == set(REASON_CODES) - set(NON_PERSISTABLE_REASON_CODES)
assert all(not (set(NON_PERSISTABLE_REASON_CODES)
                & (set(v["estimable"]) | set(v["not_estimable"])))
           for v in ARTIFACT_REASON_CODES.values())
assert all(not (set(v["estimable"]) | set(v["not_estimable"])
                ) & set(REASON_CODES_REUSED_FROM_RQ3)
           for v in ARTIFACT_REASON_CODES.values())
# The Stage-F vocabulary agrees with the frozen §G.2 module column in both
# directions, and the RQ3-reused codes are scoped by the parent, not here.
assert set(REASON_CODE_MODULES) == set(REASON_CODES)
assert not (set(REASON_CODE_MODULES) & set(REASON_CODES_REUSED_FROM_RQ3))
assert all(p in ARTIFACT_REASON_CODES
           for p in STATUS_SEMANTICS) and SVM_MACRO_JSON in ARTIFACT_REASON_CODES
# `estimable_reasons` and the artifact vocabulary are one statement, not two.
assert all(tuple(s["estimable_reasons"])
           == tuple(ARTIFACT_REASON_CODES[p]["estimable"])
           for p, s in STATUS_SEMANTICS.items())
# Every reason-scoped count rule names a real column of a real artifact, is
# scoped to a reason that artifact may actually carry, and is a count rather
# than a scientific value field.
for _scope, _expect in ((REASON_REQUIRED_ZERO_COUNTS, "not_estimable"),
                        (REASON_REQUIRED_POSITIVE_COUNTS, "not_estimable")):
    for (_p, _code), _cols in _scope.items():
        assert _code in ARTIFACT_REASON_CODES[_p][_expect]
        assert set(_cols) <= set(TABLE_COLUMNS[_p])
        assert set(_cols) <= set(NUMERIC_FIELDS[_p])
        assert not (set(_cols) & set(STATUS_SEMANTICS[_p]["value_fields"]))
assert not (set(REASON_REQUIRED_ZERO_COUNTS) & set(REASON_REQUIRED_POSITIVE_COUNTS))
for _p, _cols in ESTIMABLE_REQUIRED_POSITIVE_COUNTS.items():
    assert set(_cols) <= set(TABLE_COLUMNS[_p]) & set(NUMERIC_FIELDS[_p])
    assert not (set(_cols) & set(STATUS_SEMANTICS[_p]["value_fields"]))
del _scope, _expect, _p, _cols, _code
# The convergence flag is a real column, and the reason it is bound to is one
# the SVM per-site table may actually carry.
assert all(f in TABLE_COLUMNS[p] for p, f in CONVERGENCE_FLAG_FIELDS.items())
assert all(f not in NUMERIC_FIELDS.get(p, ())
           for p, f in CONVERGENCE_FLAG_FIELDS.items())
assert all(p in CONVERGENCE_FLAG_FIELDS
           and code in ARTIFACT_REASON_CODES[p]["not_estimable"]
           and isinstance(flag, bool)
           for (p, code), flag in CONVERGENCE_FLAG_BY_REASON.items())

# The parent-governed cell pair names real columns of real registered tables, and
# the value column is deliberately not a Stage-F numeric field.
assert all(p in registered_artifact_paths() for p in RQ3_CELL_TABLES)
assert all({RQ3_CELL_VALUE_FIELD, RQ3_CELL_CONDITION_FIELD} <= set(TABLE_COLUMNS[p])
           for p in RQ3_CELL_TABLES)
assert all(RQ3_CELL_VALUE_FIELD not in NUMERIC_FIELDS.get(p, ())
           for p in RQ3_CELL_TABLES)
assert not (set(RQ3_CELL_TABLES) & set(RQ3_STATUS_TABLES))

# A prediction-derived surrender names a real column of a real artifact and is
# scoped to a reason that artifact may actually carry.
for _key, _cols in REASON_PREDICTION_DERIVED_FIELDS.items():
    assert _key[1] in ARTIFACT_REASON_CODES[_key[0]]["not_estimable"]
    assert set(_cols) <= set(TABLE_COLUMNS[_key[0]])
del _key, _cols

# Every exact-key-set table is registered, its enumeration is the table's own
# frozen cardinality, and each frozen tuple has one entry per key field.
for _p, _keys in EXACT_KEY_SET_TABLES.items():
    assert _p in registered_artifact_paths()
    assert len(_keys) == len(set(_keys)) == expected_rows(_p)
    assert all(len(_key) == len(logical_key(_p)) for _key in _keys)
del _p, _keys
assert all(p in registered_artifact_paths() for p in RQ4_SITE_CELL_TABLES)
assert all(RQ4_SITE_FIELD in logical_key(p) for p in RQ4_SITE_CELL_TABLES)
assert not (set(EXACT_KEY_SET_TABLES) & set(RQ4_SITE_CELL_TABLES))
# The shape of the two site x cell tables re-derives their frozen cardinalities
# from `SITES_PER_N` — the numbers are consequences, not assertions (§H.4.2).
assert (sum(expected_sites_in_cell(m, n) for m, _a, n in REGISTERED_CELLS)
        == expected_rows("rq4/per_site_metrics.csv"))
assert (sum(SITES_PER_N[n] for n in N_GRID) * R_SPLIT * len(ALPHA_GRID)
        == expected_rows("rq4/per_repeat_target_local.csv"))
assert expected_sites_in_cell(METHOD_SOURCE_POOLED, None) == BASE_COHORT_SITES
assert expected_sites_in_cell(METHOD_TARGET_LOCAL, 40) == SITES_PER_N[40]

# Every declared split-identity artifact really carries the two columns it
# names, and the two RQ4 site x cell tables are both covered — the shape checks
# and the identity binding govern the same files, not overlapping subsets.
assert all(p in registered_artifact_paths() for p in SPLIT_IDENTITY_TABLES)
assert all(spec[SPLIT_IDENTITY_SITE_FIELD] in TABLE_COLUMNS[p]
           and spec[SPLIT_IDENTITY_N_FIELD] in TABLE_COLUMNS[p]
           for p, spec in SPLIT_IDENTITY_TABLES.items())
assert all(spec["row_filter"] is None or spec["row_filter"][0] in TABLE_COLUMNS[p]
           for p, spec in SPLIT_IDENTITY_TABLES.items())
assert set(RQ4_SITE_CELL_TABLES) <= set(SPLIT_IDENTITY_TABLES)
assert (set(SENS_DECISION_TABLES + SENS_EVALUATION_TABLES)
        <= set(SPLIT_IDENTITY_TABLES))
assert len(SPLIT_IDENTITY_TABLES) == 10
# No identity is enumerated here: the registry declares *where* the eligible
# sets are persisted, never *which* sites they hold.
assert all("sites" not in spec or spec["sites"] is None
           for spec in SPLIT_IDENTITY_TABLES.values())
assert all(set(spec["n_values"]) <= set(S2_N_GRID)
           for spec in SPLIT_IDENTITY_TABLES.values())
assert SVM_PER_SITE_TABLE in registered_artifact_paths()
assert expected_rows(SVM_PER_SITE_TABLE) == BASE_COHORT_SITES
assert (expected_rows(SVM_FOLD_SELECTION_TABLE)
        == BASE_COHORT_SITES * len(SVM_PCA_CANDIDATES) * len(SVM_C_GRID))
assert SVM_MACRO_REQUIRED_SITES == BASE_COHORT_SITES
assert {SVM_MACRO_CONTRIBUTING_FIELD,
        SVM_MACRO_NOT_ESTIMABLE_FIELD} <= set(SVM_MACRO_JSON_COUNT_FIELDS)

# The SENS-1..4 governed columns are real columns of real registered tables, and
# the two table families are disjoint from the RQ3-derived ones.
assert all(p in registered_artifact_paths()
           for p in SENS_DECISION_TABLES + SENS_EVALUATION_TABLES)
assert len(SENS_DECISION_TABLES) == len(SENS_EVALUATION_TABLES) == 4
assert not (set(SENS_DECISION_TABLES + SENS_EVALUATION_TABLES)
            & (set(RQ3_STATUS_TABLES) | set(RQ3_CELL_TABLES)))
assert all({SENS_INHERITED_STATE_FIELD, SENS_INHERITED_REASON_FIELD,
            SENS_INHERITED_TIER_FIELD, SENS_INHERITED_SCORE_FIELD,
            SENS_INHERITED_GUARANTEE_FIELD, SENS_STRATEGY_FIELD}
           <= set(TABLE_COLUMNS[p]) for p in SENS_DECISION_TABLES)
assert all({SENS_INHERITED_STATE_FIELD, SENS_INHERITED_TIER_FIELD,
            SENS_INHERITED_REALIZED_FIELD, SENS_STRATEGY_FIELD}
           <= set(TABLE_COLUMNS[p]) for p in SENS_EVALUATION_TABLES)
assert all(set(SENS_SELECTION_FIELDS) <= set(TABLE_COLUMNS[p])
           for p in SENS_DECISION_TABLES)
assert all(set(SENS_CALIBRATION_COUNT_FIELDS) <= set(TABLE_COLUMNS[p])
           for p in SENS_DECISION_TABLES)
assert all(set(SENS_EVALUATION_COUNT_FIELDS + SENS_EVALUATION_VALUE_FIELDS)
           <= set(TABLE_COLUMNS[p]) for p in SENS_EVALUATION_TABLES)
assert SENS2_LABEL in SENSITIVITY_LABELS and SENS2_LABEL not in SENS56_APPLIES_TO
assert len(SENS_DECISION_STRATEGIES) == 2 and len(SENS_EVALUATION_STRATEGIES) == 4
assert set(SENS_DECISION_STRATEGIES) < set(SENS_EVALUATION_STRATEGIES)

# The nested block record names the cell table's own column, its key list is
# sorted (it is compared to `sorted(payload)`), and the shared fields are real.
assert all(SENS56_CELLS_BLOCKS_FIELD in TABLE_COLUMNS[p] for p in RQ3_CELL_TABLES)
assert SENS56_BLOCK_KEYS == tuple(sorted(SENS56_BLOCK_KEYS))
assert SENS56_FAMILY_SLOT_KEYS == tuple(sorted(SENS56_FAMILY_SLOT_KEYS))
assert set(SENS56_BLOCK_UNIT_SHARED_FIELDS) <= set(SENS56_BLOCK_KEYS)
assert set(SENS56_BLOCK_VALUE_FIELDS) <= set(SENS56_BLOCK_KEYS)
assert all(set(SENS56_BLOCK_UNIT_SHARED_FIELDS) <= set(TABLE_COLUMNS[p])
           for p in RQ3_STATUS_TABLES)
assert SENS56_FAMILY_PVALUE_FIELD in SENS56_FAMILY_SLOT_KEYS
assert SENS56_BLOCK_PVALUE_FIELD in SENS56_BLOCK_KEYS
assert all({SENS56_FAMILIES_SLOTS_FIELD, SENS56_FAMILY_NOT_ESTIMABLE_FIELD}
           <= set(TABLE_COLUMNS[f"{NAMESPACE_SENSITIVITIES}/{label}/"
                                f"{SENS56_FAMILIES_FILE}"])
           for label in SENS56_APPLIES_TO)
