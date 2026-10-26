"""The production binding to the frozen RQ3 protocol — aligned at Stage E.2b.

Every RQ3 implementation module imports its protocol mechanisms and its
execution-controlling constants **from here**, and this module imports them
from `scripts/validate_rq3_protocol_freeze.py` — the reference layer that
`docs/RQ3_PROTOCOL.md` freezes and `tests/test_rq3_protocol_freeze.py`
enforces.

The authority is `RQ3_PROTOCOL/2.0.1`, frozen as tag `rq3-protocol-v2` at
commit `aee75c3b00e3361f134c0abab278475b13683b8c`. `rq3-protocol-v1`
(`RQ3_PROTOCOL/1.0.4`) is the superseded historical freeze: Stage E.2a
withdrew its non-adaptive quadrature rule and its fixed-node gradient, and
Stage E.2a.1 corrected its `ftol`. Neither is a runtime target here.

The point is that production cannot quietly acquire its own copy of a frozen
rule. There is exactly one `select_best_start`, one seven-of-seven 40-vs-60
verdict, one six-slot Holm procedure, one site-balanced rejected fraction and
one deterministic initializer in this repository, and Stage E.2 calls those and
no others. A re-implementation that "does the same thing" would be a second
authority, which is precisely what the freeze exists to prevent.

`assert_protocol_binding()` additionally re-checks, at import time, that the
frozen module still carries the protocol version and the constant values Stage
E.2 was written against. Protocol drift is therefore a hard failure at the
first import, not a silent divergence discovered after a run.

### The root of trust is bytes, not names (Stage E.2b.1)

A binding that only asks whether a mechanism is `callable` binds nothing —
a substitute is callable too. Three layers now stand behind every name below:

1. `scripts/validate_rq3_protocol_freeze.py` is hashed against
   `EXPECTED_VALIDATOR_SHA256` **before it is imported**, and again afterwards;
2. `assert_mechanism_binding()` recompiles that verified source into a private
   namespace and compares the *compiled body* of every re-exported mechanism
   against it, so a replaced, wrapped or re-implemented function fails even if
   it carries the right name, module and file;
3. `assert_frozen_provenance()` resolves `rq3-protocol-v2` with git, checks that
   it names `aee75c3b…`, and re-hashes the protocol document and the mechanism
   source **out of that commit** — so the tag, the commit and the bytes are
   enforced identities rather than strings this file asserts about itself. It is
   mandatory before an authoritative store may be created.

**This module fits nothing and reads nothing.** It is a name-binding layer.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The repository, as resolved at import. `ROOT` is what the rest of the code
#: reads and writes relative to, and a test fixture may redirect it into a
#: sandbox; the freeze-provenance checks below are anchored here instead, so
#: they always interrogate the real repository and cannot be sandboxed away.
_REPO_ROOT = ROOT

_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

#: The implementation was aligned at Stage E.2b against exactly this protocol.
EXPECTED_PROTOCOL_VERSION = "RQ3_PROTOCOL/2.0.1"

#: The freeze this implementation claims provenance from. `rq3-protocol-v2` is
#: the authoritative tag; `rq3-protocol-v1` (`RQ3_PROTOCOL/1.0.4`) is the
#: superseded historical freeze and is never a runtime target.
EXPECTED_PROTOCOL_TAG = "rq3-protocol-v2"
EXPECTED_PROTOCOL_COMMIT = "aee75c3b00e3361f134c0abab278475b13683b8c"

#: SHA256 of `docs/RQ3_PROTOCOL.md` at that freeze commit. Version strings are
#: cheap to leave stale, so the *contents* are bound too: an edited protocol
#: document fails the binding even if its version string is untouched.
EXPECTED_PROTOCOL_DOC_SHA256 = (
    "fd51cea84d12a25972526682be0adda5b3df3b6625872dcc84c3d94abd0b914c"
)
PROTOCOL_DOC = "docs/RQ3_PROTOCOL.md"

#: SHA256 of the frozen mechanism source at that same commit. The mechanisms
#: below are *executable* protocol, so binding the version string, the module
#: name or `callable(...)` binds nothing: an equally callable replacement passes
#: all three. What is pinned here is the **source bytes that define them**, and
#: it is checked *before* the module is imported (H30).
VALIDATOR_SOURCE = "scripts/validate_rq3_protocol_freeze.py"
EXPECTED_VALIDATOR_SHA256 = (
    "cf3dc2d35a536df31337ade6e04fd962cb635b09e89cf400561538b92801a793"
)


class ProtocolDriftError(AssertionError):
    """Raised when the frozen protocol is not the one Stage E.2 was built on."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_frozen_source() -> tuple:
    """Re-read the frozen mechanism source **from disk**, and verify its digest.

    There is deliberately **no cached copy of these bytes**. A module-level
    source string is mutable state, and mutable state is not authority: an
    attacker who edits the cache and recompiles a mechanism from it produces a
    "verified" function that never came from the freeze. Every caller that needs
    the frozen source re-reads and re-hashes it here, so the authority is always
    the file on disk at the moment the question is asked.
    """
    path = _REPO_ROOT / VALIDATOR_SOURCE
    if not path.exists():
        raise ProtocolDriftError(f"the frozen mechanism source {VALIDATOR_SOURCE} is missing")
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != EXPECTED_VALIDATOR_SHA256:
        raise ProtocolDriftError(
            f"{VALIDATOR_SOURCE} hashes to {digest}, the frozen "
            f"{EXPECTED_PROTOCOL_TAG} @ {EXPECTED_PROTOCOL_COMMIT[:12]} value is "
            f"{EXPECTED_VALIDATOR_SHA256}. Production does not import a mechanism "
            f"source it cannot recognise."
        )
    return blob.decode("utf-8"), digest


# The bytes are checked first; only then is the module allowed to load.
_read_frozen_source()

import validate_rq3_protocol_freeze as _frozen  # noqa: E402

# --------------------------------------------------------------------------
# frozen mechanisms — re-exported, never re-implemented
# --------------------------------------------------------------------------
ValidationFailure = _frozen.ValidationFailure

select_best_start = _frozen.select_best_start
numerical_validation_verdict = _frozen.numerical_validation_verdict
passes_numerical_validation = _frozen.passes_numerical_validation
agreement_checks = _frozen.agreement_checks
sigma2_relative_difference = _frozen.sigma2_relative_difference
coefficient_discrepancy = _frozen.coefficient_discrepancy

holm_adjust_family = _frozen.holm_adjust_family
holm_is_significant = _frozen.holm_is_significant
site_balanced_rejected_fraction = _frozen.site_balanced_rejected_fraction
pooled_rejected_fraction = _frozen.pooled_rejected_fraction
deterministic_initial_points = _frozen.deterministic_initial_points

# frozen Stage E.2a mechanisms — the amended numerics' rule layer
mode_solver_verdict = _frozen.mode_solver_verdict
validation_units = _frozen.validation_units
is_numerical_failure = _frozen.is_numerical_failure
may_be_reported_as_null = _frozen.may_be_reported_as_null
cv_logloss_status = _frozen.cv_logloss_status
lrt_status = _frozen.lrt_status
is_authoritative_root = _frozen.is_authoritative_root
completion_barrier_order = _frozen.completion_barrier_order
completion_marker_is_permitted = _frozen.completion_marker_is_permitted

# frozen data-side derivations (§5.1, §5.2, §5.3, §2.1) — the validator's own
# functions, so the production cohort cannot drift from the validated one
load_base_cohort = _frozen.load_base_cohort
apply_anomaly_rule = _frozen.apply_anomaly_rule
full_sample_subset = _frozen.full_sample_subset
mdd_only_subset = _frozen.mdd_only_subset
reconstruct_R = _frozen.reconstruct_R
frozen_site_balanced_coverage = _frozen.frozen_site_balanced_coverage

# --------------------------------------------------------------------------
# frozen constants — re-exported (§2, §5, §8, §9, §12, §15)
# --------------------------------------------------------------------------
PROTOCOL_VERSION = _frozen.PROTOCOL_VERSION

BASE_COHORT_N = _frozen.BASE_COHORT_N
BASE_COHORT_SITES = _frozen.BASE_COHORT_SITES
FULL_CC_N = _frozen.FULL_CC_N
FULL_CC_SITES = _frozen.FULL_CC_SITES
FULL_CC_MDD = _frozen.FULL_CC_MDD
FULL_CC_HC = _frozen.FULL_CC_HC
MDD_CC_N = _frozen.MDD_CC_N
MDD_CC_SITES = _frozen.MDD_CC_SITES
MDD_CC_PER_SITE = _frozen.MDD_CC_PER_SITE

CONFIDENCE_METHODS = _frozen.CONFIDENCE_METHODS
OPERATING_POINTS = _frozen.OPERATING_POINTS
CELL_REGISTRY = _frozen.CELL_REGISTRY
COVERAGE_TOL = _frozen.COVERAGE_TOL

FULL_VARS = _frozen.FULL_VARS
MDD_VARS = _frozen.MDD_VARS
SEX_MAP = _frozen.SEX_MAP
DIAGNOSIS_MAP = _frozen.DIAGNOSIS_MAP
BINARY_CLINICAL_MAP = _frozen.BINARY_CLINICAL_MAP
LOG1P_VARS = _frozen.LOG1P_VARS

ALPHA = _frozen.ALPHA
HOLM_FAMILY_SIZE = _frozen.HOLM_FAMILY_SIZE
HOLM_FAMILIES = _frozen.HOLM_FAMILIES
BLOCK_DF = _frozen.BLOCK_DF
NOT_ESTIMABLE = _frozen.NOT_ESTIMABLE

GH_NODES_PRIMARY = _frozen.GH_NODES_PRIMARY
GH_NODES_VALIDATION = _frozen.GH_NODES_VALIDATION
AUTHORITATIVE_NODES = _frozen.AUTHORITATIVE_NODES
LOG_SIGMA_BOUNDS = _frozen.LOG_SIGMA_BOUNDS
OPTIMIZER = _frozen.OPTIMIZER
OPT_FTOL = _frozen.OPT_FTOL
OPT_GTOL = _frozen.OPT_GTOL
OPT_MAXITER = _frozen.OPT_MAXITER
OPT_MAXFUN = _frozen.OPT_MAXFUN
GRADIENT_TOL = _frozen.GRADIENT_TOL

INIT_PREVALENCE_CLIP = _frozen.INIT_PREVALENCE_CLIP
SIGMA_STARTS = _frozen.SIGMA_STARTS
START_INDEX_TO_SIGMA = _frozen.START_INDEX_TO_SIGMA
N_REGISTERED_STARTS = _frozen.N_REGISTERED_STARTS

AGREEMENT_TOL = _frozen.AGREEMENT_TOL
REQUIRED_AGREEMENT_CRITERIA = _frozen.REQUIRED_AGREEMENT_CRITERIA
N_REQUIRED_AGREEMENT_CRITERIA = _frozen.N_REQUIRED_AGREEMENT_CRITERIA
SIGMA2_RELATIVE_DENOMINATOR_FLOOR = _frozen.SIGMA2_RELATIVE_DENOMINATOR_FLOOR

SITE_BALANCED_DENOMINATOR = _frozen.SITE_BALANCED_DENOMINATOR
REJECTED_FRACTION_MODE = _frozen.REJECTED_FRACTION_MODE

# ---- §9.2 amended quadrature (Stage E.2a) --------------------------------
QUADRATURE_RULE = _frozen.QUADRATURE_RULE
WITHDRAWN_QUADRATURE_RULE = _frozen.WITHDRAWN_QUADRATURE_RULE
QUADRATURE_IS_ADAPTIVE = _frozen.QUADRATURE_IS_ADAPTIVE
ADAPTIVE_TRANSFORM_TERMS = _frozen.ADAPTIVE_TRANSFORM_TERMS

# ---- §9.2.5 mode solver ---------------------------------------------------
MODE_SOLVER = _frozen.MODE_SOLVER
MODE_INIT = _frozen.MODE_INIT
MODE_CONVERGENCE_TOL = _frozen.MODE_CONVERGENCE_TOL
MODE_STAGNATION_TOL = _frozen.MODE_STAGNATION_TOL
MODE_MAXITER = _frozen.MODE_MAXITER
MODE_MAX_HALVINGS = _frozen.MODE_MAX_HALVINGS
CURVATURE_FLOOR = _frozen.CURVATURE_FLOOR
MODE_FALLBACK_SOLVER_PERMITTED = _frozen.MODE_FALLBACK_SOLVER_PERMITTED

# ---- §9.4.4 gradient ------------------------------------------------------
GRADIENT_RULE = _frozen.GRADIENT_RULE
GRADIENT_FINITE_DIFFERENCE_PERMITTED = _frozen.GRADIENT_FINITE_DIFFERENCE_PERMITTED
GRADIENT_TERMS = _frozen.GRADIENT_TERMS

# ---- §9.5.1 / §9.6.5 / §10.6 / §12.1.1 -----------------------------------
VALIDATION_UNIT = _frozen.VALIDATION_UNIT
N_VALIDATION_UNITS = _frozen.N_VALIDATION_UNITS
STATUS_OK = _frozen.STATUS_OK
START_FAILURE_CODES = _frozen.START_FAILURE_CODES
AGGREGATE_FAILURE_CODES = _frozen.AGGREGATE_FAILURE_CODES
NUMERICAL_FAILURE_CODES = _frozen.NUMERICAL_FAILURE_CODES
STATUS_CODES = _frozen.STATUS_CODES
CV_AGGREGATION_POLICY = _frozen.CV_AGGREGATION_POLICY
CV_PARTIAL_FOLD_AVERAGE_PERMITTED = _frozen.CV_PARTIAL_FOLD_AVERAGE_PERMITTED
NEGATIVE_LRT_POLICY = _frozen.NEGATIVE_LRT_POLICY
NEGATIVE_LRT_CLAMP_PERMITTED = _frozen.NEGATIVE_LRT_CLAMP_PERMITTED
SIGMA2_REQUIRES_BOTH_UNITS_PASSING = _frozen.SIGMA2_REQUIRES_BOTH_UNITS_PASSING

# ---- §16.1 / §16.2 formal execution invariants ---------------------------
AUTHORITATIVE_RESULT_ROOT = _frozen.AUTHORITATIVE_RESULT_ROOT
COMPLETION_MARKER = _frozen.COMPLETION_MARKER
CALLER_CONTROLLED_ROOT_PERMITTED = _frozen.CALLER_CONTROLLED_ROOT_PERMITTED
STALE_ROOT_POLICY = _frozen.STALE_ROOT_POLICY
COMPLETION_ORDER = _frozen.COMPLETION_ORDER

FROZEN_INPUTS = _frozen.FROZEN_INPUTS
S0_LEDGER = _frozen.S0_LEDGER
S0_LEDGER_BOUND = _frozen.S0_LEDGER_BOUND
sha256_of = _frozen.sha256_of


#: Constants Stage E.2 hard-codes an expectation for. Each pair is
#: `(attribute, expected value)` and every one is re-checked at import.
_PINNED = (
    ("PROTOCOL_VERSION", EXPECTED_PROTOCOL_VERSION),
    ("GH_NODES_PRIMARY", 40),
    ("GH_NODES_VALIDATION", 60),
    ("AUTHORITATIVE_NODES", 40),
    ("OPTIMIZER", "L-BFGS-B"),
    # AMENDED at Stage E.2a.1 (§9.4.5): `ftol` is the *relative* objective
    # criterion and at n = 2222 the superseded `1e-12` fired before `gtol`,
    # leaving `max abs gradient` at the §9.5(3) threshold itself. `1e-16` makes
    # `gtol` operative. The eligibility bound below is unchanged at `1e-5`.
    ("OPT_FTOL", 1e-16),
    ("OPT_GTOL", 1e-8),
    ("OPT_MAXITER", 20000),
    ("OPT_MAXFUN", 50000),
    ("GRADIENT_TOL", 1e-5),
    ("INIT_PREVALENCE_CLIP", 1e-6),
    ("SIGMA_STARTS", (0.1, 0.5, 1.0, 2.0)),
    ("START_INDEX_TO_SIGMA", {0: 0.1, 1: 0.5, 2: 1.0, 3: 2.0}),
    ("N_REGISTERED_STARTS", 4),
    ("N_REQUIRED_AGREEMENT_CRITERIA", 7),
    ("SIGMA2_RELATIVE_DENOMINATOR_FLOOR", 1e-12),
    ("ALPHA", 0.05),
    ("HOLM_FAMILY_SIZE", 6),
    ("NOT_ESTIMABLE", "NOT_ESTIMABLE"),
    ("BASE_COHORT_N", 2232),
    ("BASE_COHORT_SITES", 23),
    ("FULL_CC_N", 2222),
    ("FULL_CC_SITES", 23),
    ("MDD_CC_N", 484),
    ("COVERAGE_TOL", 1e-9),
    ("CONFIDENCE_METHODS", ("probability", "bootstrap", "mahalanobis")),
    ("OPERATING_POINTS", ("OP50", "OP30")),
    ("REJECTED_FRACTION_MODE", "site_balanced"),
    ("SITE_BALANCED_DENOMINATOR", {"full_sample": 23, "mdd_only": 9}),
    ("REQUIRED_AGREEMENT_CRITERIA", (
        "loglik", "lrt_statistic", "lrt_pvalue", "mcfadden",
        "sigma_site_sq_relative", "cv_logloss", "fixed_effect_coefficients",
    )),
    ("AGREEMENT_TOL", {
        "loglik": 1e-4,
        "lrt_statistic": 1e-3,
        "lrt_pvalue": 1e-4,
        "mcfadden": 1e-4,
        "sigma_site_sq_relative": 1e-3,
        "cv_logloss": 1e-6,
        "fixed_effect_coefficients": 1e-4,
    }),
    ("BLOCK_DF", {
        ("full_sample", "motion"): 1,
        ("full_sample", "diagnosis"): 1,
        ("mdd_only", "motion"): 1,
        ("mdd_only", "clinical"): 4,
    }),
    ("CELL_REGISTRY", (
        ("OP50", "probability"), ("OP50", "bootstrap"), ("OP50", "mahalanobis"),
        ("OP30", "probability"), ("OP30", "bootstrap"), ("OP30", "mahalanobis"),
    )),
    ("SEX_MAP", {1.0: 0, 2.0: 1}),
    ("DIAGNOSIS_MAP", {0.0: 0, 1.0: 1}),
    ("BINARY_CLINICAL_MAP", {-1.0: 0, 1.0: 1}),
    ("LOG1P_VARS", ("mean_fd", "illness_duration_months")),

    # ---- Stage E.2a amended numerics -------------------------------------
    ("QUADRATURE_RULE", "mode_curvature_adaptive_gauss_hermite"),
    ("WITHDRAWN_QUADRATURE_RULE", "non_adaptive_gauss_hermite"),
    ("QUADRATURE_IS_ADAPTIVE", True),
    ("ADAPTIVE_TRANSFORM_TERMS", (
        "mode_relocation", "curvature_scaling", "jacobian", "kernel_deweighting",
    )),
    ("MODE_SOLVER", "safeguarded_newton"),
    ("MODE_INIT", 0.0),
    ("MODE_CONVERGENCE_TOL", 1e-8),
    ("MODE_STAGNATION_TOL", 1e-4),
    ("MODE_MAXITER", 200),
    ("MODE_MAX_HALVINGS", 50),
    ("CURVATURE_FLOOR", 1e-12),
    ("MODE_FALLBACK_SOLVER_PERMITTED", False),
    ("GRADIENT_RULE", "exact_total_derivative"),
    ("GRADIENT_FINITE_DIFFERENCE_PERMITTED", False),
    ("GRADIENT_TERMS", (
        "d_log_tau", "partial_h_at_fixed_u", "mode_movement", "scale_movement",
    )),
    ("VALIDATION_UNIT", ("population", "cell", "block")),
    ("N_VALIDATION_UNITS", 24),
    ("STATUS_OK", "ESTIMABLE"),
    ("START_FAILURE_CODES", (
        "OPTIMIZER_NOT_CONVERGED", "NON_FINITE_OBJECTIVE", "GRADIENT_NOT_ZERO",
        "SIGMA_AT_OR_OUTSIDE_BOUND", "ADAPTIVE_MODE_NOT_FOUND",
        "ADAPTIVE_CURVATURE_INVALID",
    )),
    ("AGGREGATE_FAILURE_CODES", (
        "NO_ELIGIBLE_START", "QUADRATURE_DISAGREEMENT", "VALIDATION_MALFORMED",
        "NESTED_LIKELIHOOD_INVERSION", "CV_FOLD_NOT_ESTIMABLE",
    )),
    ("CV_AGGREGATION_POLICY", "all_or_nothing"),
    ("CV_PARTIAL_FOLD_AVERAGE_PERMITTED", False),
    ("NEGATIVE_LRT_POLICY", "NESTED_LIKELIHOOD_INVERSION"),
    ("NEGATIVE_LRT_CLAMP_PERMITTED", False),
    ("SIGMA2_REQUIRES_BOTH_UNITS_PASSING", True),
    ("AUTHORITATIVE_RESULT_ROOT", "results/rq3"),
    ("COMPLETION_MARKER", "COMPLETION.json"),
    ("CALLER_CONTROLLED_ROOT_PERMITTED", False),
    ("STALE_ROOT_POLICY", "fail_closed"),
    ("COMPLETION_ORDER", (
        "write_artifacts", "write_checksum_ledger", "flush_and_fsync",
        "reread_from_disk", "recompute_and_compare_digests",
        "verify_artifact_set_exact", "write_completion_marker_atomically",
    )),
)


#: Every frozen mechanism this module re-exports. Each one is bound to the
#: *compiled body* of the verified source, not merely to a name (H30).
_FROZEN_MECHANISMS = (
    "sha256_of",
    "load_base_cohort", "apply_anomaly_rule", "full_sample_subset",
    "mdd_only_subset", "reconstruct_R", "frozen_site_balanced_coverage",
    "site_balanced_rejected_fraction", "pooled_rejected_fraction",
    "holm_adjust_family", "holm_is_significant",
    "deterministic_initial_points",
    "sigma2_relative_difference", "coefficient_discrepancy",
    "agreement_checks", "numerical_validation_verdict",
    "passes_numerical_validation", "select_best_start",
    "mode_solver_verdict", "validation_units", "is_numerical_failure",
    "may_be_reported_as_null", "cv_logloss_status", "lrt_status",
    "is_authoritative_root", "completion_barrier_order",
    "completion_marker_is_permitted",
)

_FROZEN_MODULE_NAME = "validate_rq3_protocol_freeze"


def _code_fingerprint(code: types.CodeType) -> tuple:
    """A structural digest of one compiled function body, nested bodies included.

    `co_filename` and `co_firstlineno` are deliberately absent: they are the two
    fields a forged replacement can set freely, and neither says anything about
    what the function *does*. What is compared is the bytecode, the names it
    touches, its argument shape and its constants.
    """
    constants = []
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            constants.append(("<code>", _code_fingerprint(constant)))
        else:
            constants.append(repr(constant))
    return (
        code.co_name, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_nlocals, code.co_flags,
        tuple(code.co_names), tuple(code.co_varnames),
        tuple(code.co_freevars), tuple(code.co_cellvars),
        code.co_code.hex(), tuple(constants),
    )


def _referenced_names(code: types.CodeType, into: set) -> set:
    """Every global name a function body can reach, nested bodies included."""
    into.update(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            _referenced_names(constant, into)
    return into


#: Sentinel for "this name is not bound in the namespace it is resolved in".
_ABSENT = object()


def _value_fingerprint(alias, value, seen: set, anchor: dict, owner: dict):
    """Bind one referenced dependency by *what it is*.

    Binding a mechanism's own bytecode is not enough: `select_best_start` is a
    thin shell over `_strict_start_index` / `_strict_start_sigma` and over frozen
    tables like `START_INDEX_TO_SIGMA`, so replacing a helper or handing the
    mechanism altered globals changes what it does without touching its body.
    The dependency closure is therefore bound too.

    Two rules matter here and were both wrong before Stage E.2b.3:

    * a **function** dependency is fingerprinted through *its own* `__globals__`
      (`_closure_fingerprint` below), never through the namespace of whatever
      referred to it. Flattening every helper through the root mechanism's
      namespace made an identical-code helper carrying a doctored private
      `__globals__` indistinguishable from the frozen one;
    * a **module** dependency is compared *pairwise on the referenced alias
      slot*: what `np` is bound to here, against what the verified reference
      namespace binds to `np`. Re-deriving the lookup key from `value.__name__`
      asked whether `namespace["numpy"]` matched — a name the freeze never binds
      — so a forged module that merely calls itself `"numpy"` compared equal on
      both sides and could change `lrt_status(NaN)`.
    """
    if value is _ABSENT:
        return ("<absent>",)
    if isinstance(value, types.FunctionType):
        # `owner` is the namespace that *referred* to this function; a frozen
        # helper shares its referrer's module namespace, an injected clone with
        # a private doctored namespace does not
        return ("<function>", value.__globals__ is owner,
                _closure_fingerprint(value, seen, anchor))
    if isinstance(value, types.ModuleType):
        # identity of the actual referenced alias slot, plus identity against
        # the imported module of that name — never `value.__name__` as a key
        return ("<module>", alias,
                value is anchor.get(alias),
                value is sys.modules.get(value.__name__))
    if isinstance(value, type):
        return ("<class>", value.__qualname__,
                tuple(base.__qualname__ for base in value.__mro__[1:]))
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return ("<scalar>", repr(value))
    if isinstance(value, (tuple, list)):
        return ("<sequence>", tuple(repr(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("<set>", tuple(sorted(repr(item) for item in value)))
    if isinstance(value, dict):
        return ("<mapping>", tuple(sorted((repr(k), repr(v)) for k, v in value.items())))
    return ("<opaque>", type(value).__qualname__, repr(value))


def _closure_fingerprint(function, seen: set, anchor: dict) -> tuple:
    """A function, its body, its defaults, its cells and its whole dependency closure.

    The namespace every name is resolved in is `function.__globals__` — the
    function's own. `anchor` is the verified reference namespace and is used for
    one purpose only: comparing third-party module aliases by identity.
    """
    key = id(function)
    if key in seen:                       # recursive or mutually recursive helper
        return ("<cycle>", function.__qualname__)
    seen = seen | {key}

    namespace = function.__globals__      # its OWN namespace, never its caller's

    globals_used = []
    for name in sorted(_referenced_names(function.__code__, set())):
        if name.startswith("__") and name.endswith("__"):
            # module metadata (`__name__`, `__file__`, …) differs between the
            # imported module and the private reference namespace by
            # construction, and carries no protocol content
            continue
        globals_used.append((name, _value_fingerprint(
            name, namespace.get(name, _ABSENT), seen, anchor, namespace)))

    cells = []
    for index, cell in enumerate(function.__closure__ or ()):
        try:
            contents = cell.cell_contents
        except ValueError:                                    # pragma: no cover
            cells.append(("<empty-cell>",))
            continue
        cells.append(_value_fingerprint(
            f"<cell:{index}>", contents, seen, anchor, namespace))

    return (
        function.__qualname__,
        _code_fingerprint(function.__code__),
        tuple(_value_fingerprint(f"<default:{index}>", default, seen, anchor, namespace)
              for index, default in enumerate(function.__defaults__ or ())),
        tuple(sorted((name, _value_fingerprint(name, value, seen, anchor, namespace))
                     for name, value in (function.__kwdefaults__ or {}).items())),
        tuple(cells),
        tuple(globals_used),
    )


def _mechanism_digest(function, anchor: dict) -> str:
    """The digest actually compared: body **plus** dependency closure."""
    return hashlib.sha256(
        repr(_closure_fingerprint(function, set(), anchor)).encode("utf-8")
    ).hexdigest()


def _reachable_module_aliases(reference: dict) -> tuple:
    """Every third-party alias the frozen mechanisms actually resolve against.

    Collected from the **verified reference side**, so the set of slots that must
    be checked is a property of the freeze rather than of whatever production
    currently happens to hold.
    """
    aliases: set = set()
    walked: set = set()

    def walk(function) -> None:
        if id(function) in walked:
            return
        walked.add(id(function))
        namespace = function.__globals__
        for name in _referenced_names(function.__code__, set()):
            if name.startswith("__") and name.endswith("__"):
                continue
            value = namespace.get(name, _ABSENT)
            if isinstance(value, types.ModuleType):
                aliases.add(name)
            elif isinstance(value, types.FunctionType):
                walk(value)

    for name in _FROZEN_MECHANISMS:
        candidate = reference.get(name)
        if isinstance(candidate, types.FunctionType):
            walk(candidate)
    return tuple(sorted(aliases))


def _assert_module_identity(reference: dict) -> tuple:
    """Compare each referenced module alias **pairwise**, by object identity.

    `np` is checked against the reference namespace's `np`, `pd` against its
    `pd`, `math` against its `math`. A forged module in the `np` slot fails even
    when it reports `__name__ == "numpy"`, and an unrelated real module in that
    slot fails too — because the frozen requirement is the identity of the
    imported module, not the name it is willing to answer to.
    """
    checked = []
    for alias in _reachable_module_aliases(reference):
        expected = reference.get(alias)
        actual = _frozen.__dict__.get(alias, _ABSENT)
        if actual is not expected:
            raise ProtocolDriftError(
                f"the frozen mechanism source resolves `{alias}` against the "
                f"imported {getattr(expected, '__name__', expected)!r} module; "
                f"production binds "
                f"{getattr(actual, '__name__', actual)!r} in that slot. A module "
                f"dependency is bound by identity, never by the name it reports."
            )
        if expected is not sys.modules.get(expected.__name__):
            raise ProtocolDriftError(
                f"`{alias}` is not the imported {expected.__name__} module"
            )
        checked.append(alias)
    return tuple(checked)


def trusted_mechanism_namespace() -> dict:
    """Compile a reference namespace from **freshly re-read, verified** bytes.

    Nothing here is cached. The source is read from disk and hashed on every
    call, so the reference can never be a stale or edited copy, and a mutated
    module-level source string is not an input to it at all.
    """
    source, _ = _read_frozen_source()
    namespace = {
        "__name__": "_rq3_frozen_reference",
        "__file__": str(_REPO_ROOT / VALIDATOR_SOURCE),
        "__builtins__": __builtins__,
    }
    exec(compile(source, str(_REPO_ROOT / VALIDATOR_SOURCE), "exec"), namespace)
    return namespace


def assert_mechanism_binding(trusted: dict | None = None) -> str:
    """Bind every re-exported mechanism, and its dependency closure, to the freeze.

    `callable(...)`, a matching `__name__` and a matching `__module__` are all
    satisfiable by a substitute. So is identical bytecode: a function is its code
    **plus the globals it resolves against**, so an identical body handed altered
    globals, or a body whose helper has been swapped, behaves differently while
    looking identical. Three things are therefore compared, per mechanism:

    * the compiled body (as before);
    * every global name the body can reach, resolved through the function's own
      `__globals__` — recursively for helper functions **through each helper's
      own namespace**, by value for frozen tables and scalars, by pairwise alias
      identity for third-party modules;
    * defaults, keyword defaults and closure cells, by the same rule.

    The reference side is compiled from bytes re-read and re-hashed **now**, and
    the production aliases in this module are checked to still point at the
    verified objects. Returns a digest of the whole closure, which the formal
    store records and re-derives at completion.
    """
    reference = trusted if trusted is not None else trusted_mechanism_namespace()
    validator_path = (_REPO_ROOT / VALIDATOR_SOURCE).resolve()

    # third-party dependencies first, pairwise on the referenced alias slot, so
    # a forged `np` or `pd` produces the module error rather than an opaque
    # closure mismatch
    _assert_module_identity(reference)

    digests = []

    for name in _FROZEN_MECHANISMS:
        bound = getattr(_frozen, name, None)
        if bound is None:
            raise ProtocolDriftError(f"frozen protocol mechanism {name} is missing")
        if not isinstance(bound, types.FunctionType):
            raise ProtocolDriftError(
                f"frozen protocol mechanism {name} is a "
                f"{type(bound).__name__}, not a function of the freeze"
            )
        if bound.__module__ != _FROZEN_MODULE_NAME or bound.__qualname__ != name:
            raise ProtocolDriftError(
                f"frozen protocol mechanism {name} declares itself "
                f"{bound.__module__}.{bound.__qualname__}"
            )
        try:
            defined_in = Path(bound.__code__.co_filename).resolve()
        except OSError:                                    # pragma: no cover
            defined_in = None
        if defined_in != validator_path:
            raise ProtocolDriftError(
                f"frozen protocol mechanism {name} is defined in {defined_in}, "
                f"not in {VALIDATOR_SOURCE}"
            )
        expected = reference.get(name)
        if not isinstance(expected, types.FunctionType):
            raise ProtocolDriftError(
                f"the verified frozen source defines no function {name}"
            )
        # every function in the walk resolves through *its own* globals, so a
        # clone handed a doctored namespace — at any depth — is not mistaken for
        # the original. The reference namespace is passed only as the identity
        # anchor for module aliases.
        digest = _mechanism_digest(bound, reference)
        if digest != _mechanism_digest(expected, reference):
            raise ProtocolDriftError(
                f"frozen protocol mechanism {name} is not the one defined by "
                f"{VALIDATOR_SOURCE} at {EXPECTED_PROTOCOL_TAG} — its compiled "
                f"body or its dependency closure differs. Production executes "
                f"the freeze, not a substitute."
            )
        digests.append((name, digest))

        # and the alias this module re-exports must still be that same object
        alias = globals().get(name)
        if alias is not bound:
            raise ProtocolDriftError(
                f"the production alias rq3_protocol.{name} no longer points at "
                f"the verified frozen mechanism"
            )

    failure = getattr(_frozen, "ValidationFailure", None)
    if not (isinstance(failure, type) and issubclass(failure, AssertionError)
            and failure.__module__ == _FROZEN_MODULE_NAME):
        raise ProtocolDriftError("frozen ValidationFailure is not the freeze's own class")
    if ValidationFailure is not failure:
        raise ProtocolDriftError(
            "the production alias rq3_protocol.ValidationFailure was rebound")

    return hashlib.sha256(repr(tuple(digests)).encode("utf-8")).hexdigest()


def _git_text(*arguments: str) -> str:
    result = subprocess.run(("git", "-C", str(_REPO_ROOT)) + arguments,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ProtocolDriftError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()!r}. The "
            f"freeze provenance cannot be established, so execution stops."
        )
    return result.stdout.strip()


def _git_blob(commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(_REPO_ROOT), "cat-file", "blob", f"{commit}:{relative}"),
        capture_output=True,
    )
    if result.returncode != 0:
        raise ProtocolDriftError(
            f"{relative} is absent from {commit[:12]}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()!r}"
        )
    return result.stdout


#: This module's own name in `sys.modules`, as a literal.
_MODULE_NAME = "rq3_protocol"


def _live_module_dict() -> dict:
    """The **live** module namespace, read deliberately so drift can be *refused*.

    The sealed verifier (bottom of this file) resolves every dependency it
    executes through a private namespace. This is the one place it looks at the
    mutable module, and it looks only in order to compare: reading a value so as
    to reject it is not the same as resolving authority through it. Written as a
    literal `sys.modules` lookup so that sealing cannot accidentally turn the
    comparison into a comparison of the snapshot against itself.
    """
    module = sys.modules.get(_MODULE_NAME)
    if module is None:                                        # pragma: no cover
        raise ProtocolDriftError(f"the {_MODULE_NAME} module is not imported")
    return module.__dict__


def _literal_revision_constants() -> tuple:
    """`(tag, commit)` as literals, with the module constants checked against them.

    The git-free half of `assert_literal_revision_identity`, for the non-formal
    path that writes no authoritative artifact and deliberately does not shell
    out. The values returned are the literals, never the module attributes — and
    the attributes compared against them are read from the **live** module, so a
    rebound constant fails here on the sealed path too.
    """
    tag = "rq3-protocol-v2"
    commit = "aee75c3b00e3361f134c0abab278475b13683b8c"
    declared = _live_module_dict()
    declared_tag = declared.get("EXPECTED_PROTOCOL_TAG")
    declared_commit = declared.get("EXPECTED_PROTOCOL_COMMIT")
    if declared_tag != tag:
        raise ProtocolDriftError(
            f"the authoritative revision is the tag {tag!r}; "
            f"EXPECTED_PROTOCOL_TAG now reads {declared_tag!r}"
        )
    if declared_commit != commit:
        raise ProtocolDriftError(
            f"the authoritative freeze commit is {commit}; "
            f"EXPECTED_PROTOCOL_COMMIT now reads {declared_commit!r}"
        )
    return tag, commit


def assert_literal_revision_identity() -> tuple:
    """The one authoritative revision, written as **literals in this body**.

    `EXPECTED_PROTOCOL_TAG` and `EXPECTED_PROTOCOL_COMMIT` are module attributes,
    and a module attribute is a runtime string a caller can rebind. Rebinding the
    tag to the freeze commit's own SHA used to pass, because git resolves both
    revisions to the same object — but *revision identity* is part of what the
    freeze pins. `refs/tags/rq3-protocol-v2` is the authority; a commit SHA,
    `HEAD`, an abbreviated SHA, a branch, or a second tag pointing at the same
    commit are all different identities and none of them may substitute for it.

    The literals below are what gets resolved and what gets persisted. The module
    constants are checked *against* them and are never the thing looked up, so a
    rebound constant fails loudly instead of silently choosing the revision.
    """
    tag = "rq3-protocol-v2"
    tag_ref = "refs/tags/rq3-protocol-v2"
    commit = "aee75c3b00e3361f134c0abab278475b13683b8c"

    declared = _live_module_dict()
    declared_tag = declared.get("EXPECTED_PROTOCOL_TAG")
    declared_commit = declared.get("EXPECTED_PROTOCOL_COMMIT")
    if declared_tag != tag:
        raise ProtocolDriftError(
            f"the authoritative revision is the tag {tag!r}; "
            f"EXPECTED_PROTOCOL_TAG now reads {declared_tag!r}. The "
            f"revision that gets verified is a literal, not a runtime string."
        )
    if declared_commit != commit:
        raise ProtocolDriftError(
            f"the authoritative freeze commit is {commit}; "
            f"EXPECTED_PROTOCOL_COMMIT now reads {declared_commit!r}"
        )

    # `show-ref --verify` accepts only a full, existing ref path: it fails for a
    # SHA, for an abbreviated SHA and for anything that is not literally a ref.
    tag_object = _git_text("show-ref", "--verify", "--hash", tag_ref)
    if _git_text("cat-file", "-t", tag_object) != "tag":
        raise ProtocolDriftError(
            f"{tag_ref} does not name an annotated tag object")
    resolved = _git_text("rev-parse", "--verify", "--end-of-options",
                         f"{tag_ref}^{{commit}}")
    if resolved != commit:
        raise ProtocolDriftError(
            f"{tag_ref} resolves to {resolved}, the frozen authority is {commit}"
        )
    return tag, commit


def assert_frozen_provenance() -> dict:
    """Resolve tag -> commit -> bytes with git, rather than recording strings.

    The tag is resolved from its **literal ref path**, the commit it names is
    compared against the literal freeze commit, and the two frozen authority
    files are read **out of that commit** and hashed — so a worktree that no
    longer corresponds to the freeze, a tag that has been moved, or a runtime
    constant rebound to some other revision that happens to point at the same
    commit, all fail before an authoritative run begins.

    This is deliberately not an import-time check: it shells out, and the
    binding above already refuses drifted bytes. It is mandatory at the one
    place where provenance is asserted on disk — the formal store.
    """
    tag, commit = assert_literal_revision_identity()

    digests = {}
    for relative, expected in ((PROTOCOL_DOC, EXPECTED_PROTOCOL_DOC_SHA256),
                               (VALIDATOR_SOURCE, EXPECTED_VALIDATOR_SHA256)):
        at_freeze = hashlib.sha256(_git_blob(commit, relative)).hexdigest()
        if at_freeze != expected:
            raise ProtocolDriftError(
                f"{relative} at {tag} hashes to {at_freeze}, "
                f"this implementation is pinned to {expected}"
            )
        in_worktree = _sha256_file(_REPO_ROOT / relative)
        if in_worktree != at_freeze:
            raise ProtocolDriftError(
                f"{relative} in the worktree ({in_worktree}) is not the frozen "
                f"{tag} file ({at_freeze})"
            )
        digests[relative] = at_freeze

    return {
        "protocol_tag": tag,
        "protocol_commit": commit,
        "protocol_doc_sha256": digests[PROTOCOL_DOC],
        "protocol_validator_sha256": digests[VALIDATOR_SOURCE],
    }


#: The two node counts, written as literals rather than read from anything.
#: Formal boundaries assert these directly, so no module attribute — here or in
#: the store — is the authority for which count is the science and which is the
#: check.
AUTHORITATIVE_NODE_IDENTITY = 40
VALIDATION_NODE_IDENTITY = 60


def verified_authority(*, include_git: bool = True) -> types.MappingProxyType:
    """Verify the whole frozen authority **now**, and return what was verified.

    This is the object formal provenance is derived from. It is not a copy of
    this module's constants: every field in it has just been re-established from
    the bytes on disk (and, for a formal run, from the git object store), so a
    persisted `PROVENANCE.json` records what was checked rather than what the
    runtime happened to be claiming.

    Called immediately before authoritative-root creation, before any formal fit
    or scientific write, and again inside the completion barrier.
    """
    trusted = trusted_mechanism_namespace()
    source_digest = _read_frozen_source()[1]
    mechanism_digest = assert_protocol_binding(trusted=trusted)

    # the persisted revision identity comes from the literal authority, not from
    # rebindable module constants (E.2b.3 §4). For a formal run the literal ref
    # `refs/tags/rq3-protocol-v2` is resolved through git and the frozen bytes
    # are read out of the commit it names; a non-formal run does not shell out,
    # but the literals are still the source and the constants are still checked
    # against them.
    resolved = assert_frozen_provenance() if include_git else None
    tag, commit = ((resolved["protocol_tag"], resolved["protocol_commit"])
                   if resolved is not None else _literal_revision_constants())

    authority = {
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_tag": tag,
        "protocol_commit": commit,
        "protocol_doc_sha256": _sha256_file(_REPO_ROOT / PROTOCOL_DOC),
        "protocol_validator_sha256": source_digest,
        "mechanism_closure_sha256": mechanism_digest,
        "quadrature_rule": trusted["QUADRATURE_RULE"],
        "mode_solver": trusted["MODE_SOLVER"],
        "gradient_rule": trusted["GRADIENT_RULE"],
        "authoritative_nodes": AUTHORITATIVE_NODE_IDENTITY,
        "validation_nodes": VALIDATION_NODE_IDENTITY,
        "node_registry": [AUTHORITATIVE_NODE_IDENTITY, VALIDATION_NODE_IDENTITY],
        "n_validation_units": int(trusted["N_VALIDATION_UNITS"]),
        "git_verified": bool(include_git),
    }

    # the roles, asserted against the freshly compiled frozen source and against
    # the literals — not against a rebindable module attribute
    if (int(trusted["AUTHORITATIVE_NODES"]) != 40
            or int(trusted["GH_NODES_PRIMARY"]) != 40
            or int(trusted["GH_NODES_VALIDATION"]) != 60
            or tuple(authority["node_registry"]) != (40, 60)):
        raise ProtocolDriftError(
            "§9.6: the frozen node roles are not 40 authoritative / 60 validation"
        )
    if authority["protocol_doc_sha256"] != EXPECTED_PROTOCOL_DOC_SHA256:
        raise ProtocolDriftError(f"{PROTOCOL_DOC} is not the frozen document")
    if trusted["PROTOCOL_VERSION"] != EXPECTED_PROTOCOL_VERSION:
        raise ProtocolDriftError(
            f"the frozen source declares {trusted['PROTOCOL_VERSION']!r}")

    if resolved is not None:
        for field in ("protocol_tag", "protocol_commit", "protocol_doc_sha256",
                      "protocol_validator_sha256"):
            if resolved[field] != authority[field]:
                raise ProtocolDriftError(
                    f"{field} differs between the worktree and {tag}"
                )

    return types.MappingProxyType(authority)


def assert_protocol_binding(trusted: dict | None = None) -> str:
    """Re-check that the frozen protocol is the one Stage E.2 was written for.

    Called at import, and again at every formal boundary. A drifted constant, a
    renamed mechanism or a bumped protocol version raises here rather than
    producing a run whose provenance claims a freeze it no longer matches.
    Returns the mechanism dependency-closure digest.
    """
    for attribute, expected in _PINNED:
        actual = getattr(_frozen, attribute)
        if actual != expected:
            raise ProtocolDriftError(
                f"frozen protocol constant {attribute} is {actual!r}, "
                f"Stage E.2 was built against {expected!r}"
            )

    # the log_sigma box of §9.3, compared as floats rather than as expressions
    lower, upper = _frozen.LOG_SIGMA_BOUNDS
    if not (math.isclose(float(lower), math.log(1e-4), rel_tol=0.0, abs_tol=1e-15)
            and math.isclose(float(upper), math.log(10.0), rel_tol=0.0, abs_tol=1e-15)):
        raise ProtocolDriftError(
            f"frozen log_sigma bounds are {(lower, upper)!r}, expected "
            f"{(math.log(1e-4), math.log(10.0))!r}"
        )

    # the four Holm families of six (§12.3), in their frozen order
    expected_families = (
        ("full_sample", "motion", 6),
        ("full_sample", "diagnosis", 6),
        ("mdd_only", "motion", 6),
        ("mdd_only", "clinical", 6),
    )
    if tuple(_frozen.HOLM_FAMILIES) != expected_families:
        raise ProtocolDriftError(
            f"frozen Holm families are {tuple(_frozen.HOLM_FAMILIES)!r}"
        )

    # the protocol document's own bytes (§16.1 provenance). A version string is
    # cheap to leave stale; the hash is not.
    document = _REPO_ROOT / PROTOCOL_DOC
    if not document.exists():
        raise ProtocolDriftError(f"the frozen protocol document {PROTOCOL_DOC} is missing")
    digest = _sha256_file(document)
    if digest != EXPECTED_PROTOCOL_DOC_SHA256:
        raise ProtocolDriftError(
            f"{PROTOCOL_DOC} hashes to {digest}, the frozen "
            f"{EXPECTED_PROTOCOL_TAG} @ {EXPECTED_PROTOCOL_COMMIT[:12]} value is "
            f"{EXPECTED_PROTOCOL_DOC_SHA256}"
        )

    # and the mechanism source itself, re-checked after import (the pre-import
    # check above guards the load; this one guards the rest of the process)
    validator = _sha256_file(_REPO_ROOT / VALIDATOR_SOURCE)
    if validator != EXPECTED_VALIDATOR_SHA256:
        raise ProtocolDriftError(
            f"{VALIDATOR_SOURCE} hashes to {validator}, the frozen "
            f"{EXPECTED_PROTOCOL_TAG} value is {EXPECTED_VALIDATOR_SHA256}"
        )

    # every re-exported mechanism, bound to the compiled body **and the
    # dependency closure** of the freshly verified frozen source, rather than to
    # a name that anything callable can satisfy
    mechanism_digest = assert_mechanism_binding(trusted)

    # the twelve-code registry of §9.5.1, as a set and as a size — a code added,
    # removed or re-spelled changes what a result file is allowed to say
    if tuple(_frozen.STATUS_CODES) != (
            (_frozen.STATUS_OK,) + tuple(_frozen.START_FAILURE_CODES)
            + tuple(_frozen.AGGREGATE_FAILURE_CODES)):
        raise ProtocolDriftError("the §9.5.1 status registry is not its frozen composition")
    if len(_frozen.STATUS_CODES) != 12:
        raise ProtocolDriftError(
            f"the §9.5.1 registry holds {len(_frozen.STATUS_CODES)} codes, frozen count is 12"
        )

    return mechanism_digest


assert_protocol_binding()


# ---------------------------------------------------------------------------
# Stage E.2b.4 — the sealed fresh-authority verifier
# ---------------------------------------------------------------------------
#
# Everything above resolves its helpers through this module's namespace, which
# is exactly what a formal boundary must not do. `verified_authority` reaches
# `assert_protocol_binding`, `assert_frozen_provenance`,
# `trusted_mechanism_namespace`, `_read_frozen_source` and
# `_literal_revision_constants` by name, so replacing any one of them with a
# callable that returns the *previously valid* result made a stale answer look
# like a fresh verification — and a formal write went to disk after real
# mechanism drift.
#
# The repair below builds **one** genuine verifier whose complete
# execution-controlling dependency closure is captured at import time:
#
#   * every function in the chain is recompiled against a private namespace
#     (`sealed`), a snapshot taken here and reachable from no module attribute,
#     so `rq3_protocol.assert_protocol_binding = <stale>` cannot change what the
#     sealed chain calls;
#   * the chain is checked to be *closed*: any module-level function reachable
#     from it must itself be in the chain, so a helper cannot be forgotten and
#     silently left resolvable through the live module;
#   * the genuine objects are also remembered, and the live module is compared
#     against them on every call. A substituted dependency is therefore not
#     merely ignored — it fails the run closed, loudly, before any verification
#     result is produced;
#   * the two revision constants are deliberately *not* in that comparison: they
#     are checked against the literals by `assert_literal_revision_identity` /
#     `_literal_revision_constants`, which read the live module for exactly that
#     purpose and report the precise revision message.
#
# This is a bounded repair, and deliberately so: it defeats ordinary
# repository-visible module-name rebinding. It does not attempt whole-program
# immutability, and it does not pretend to survive an interpreter-level attacker
# who rewrites `__globals__`, cell contents or class objects — the same bounded
# threat model Stage E.2b.3 was audited under.

#: Every function the fresh-authority verification actually executes.
_SEALED_VERIFIER_CHAIN = (
    "_sha256_file", "_read_frozen_source",
    "_code_fingerprint", "_referenced_names",
    "_value_fingerprint", "_closure_fingerprint", "_mechanism_digest",
    "_reachable_module_aliases", "_assert_module_identity",
    "trusted_mechanism_namespace", "assert_mechanism_binding",
    "_git_text", "_git_blob",
    "_live_module_dict",
    "_literal_revision_constants", "assert_literal_revision_identity",
    "assert_frozen_provenance",
    "assert_protocol_binding", "verified_authority",
)

#: Every non-function binding those functions resolve against. Rebinding any of
#: them changes what "verified" means, so each is pinned by identity.
_SEALED_VERIFIER_VALUES = (
    "_frozen", "ValidationFailure",
    "hashlib", "subprocess", "sys", "types", "math", "Path",
    "ProtocolDriftError", "_ABSENT", "_MODULE_NAME",
    "EXPECTED_PROTOCOL_VERSION", "EXPECTED_PROTOCOL_DOC_SHA256",
    "EXPECTED_VALIDATOR_SHA256",
    "PROTOCOL_DOC", "VALIDATOR_SOURCE", "_REPO_ROOT",
    "_PINNED", "_FROZEN_MECHANISMS", "_FROZEN_MODULE_NAME",
    "AUTHORITATIVE_NODE_IDENTITY", "VALIDATION_NODE_IDENTITY",
) + _FROZEN_MECHANISMS


def _assert_sealed_chain_is_closed(namespace: dict, chain: tuple) -> tuple:
    """Every module function the chain can reach must itself be in the chain.

    A capture that misses one helper is worse than no capture at all: the
    verifier would look sealed while still resolving that helper through the
    mutable module. This walks the chain the same way the mechanism fingerprint
    walks the freeze, and refuses to seal an incomplete closure.
    """
    reached: set = set()
    walked: set = set()

    def walk(function) -> None:
        if id(function) in walked:
            return
        walked.add(id(function))
        for name in _referenced_names(function.__code__, set()):
            if name.startswith("__") and name.endswith("__"):
                continue
            value = namespace.get(name, _ABSENT)
            if isinstance(value, types.FunctionType) \
                    and value.__module__ == _MODULE_NAME:
                reached.add(name)
                if name in chain:
                    walk(value)

    for name in chain:
        candidate = namespace.get(name, _ABSENT)
        if not isinstance(candidate, types.FunctionType):
            raise ProtocolDriftError(
                f"the sealed verifier chain names {name}, which is not a "
                f"function of this module"
            )
        walk(candidate)

    omitted = sorted(reached - set(chain))
    if omitted:
        raise ProtocolDriftError(
            f"the sealed fresh-authority verifier would still resolve {omitted} "
            f"through the mutable module namespace; the captured dependency "
            f"closure is incomplete"
        )
    return tuple(sorted(reached))


def _seal_fresh_authority_verifier(chain: tuple, values: tuple):
    """Build the one sealed verifier and return it. Its namespace never escapes."""
    module_globals = globals()
    _assert_sealed_chain_is_closed(module_globals, chain)

    sealed = dict(module_globals)          # private; no module attribute holds it
    clones: dict = {}
    genuine: dict = {}

    for name in chain:
        original = module_globals[name]
        clone = types.FunctionType(original.__code__, sealed, original.__name__,
                                   original.__defaults__, original.__closure__)
        clone.__kwdefaults__ = original.__kwdefaults__
        clone.__qualname__ = original.__qualname__
        clone.__module__ = original.__module__
        clone.__doc__ = original.__doc__
        sealed[name] = clone               # the chain calls the clones
        clones[name] = clone
        genuine[name] = original           # the live module must still hold this

    for name in values:
        if name not in module_globals:
            raise ProtocolDriftError(
                f"the sealed verifier pins {name}, which this module does not define"
            )
        genuine[name] = module_globals[name]

    def sealed_verified_authority(*, include_git: bool = True):
        """Establish the whole frozen authority **now**, through sealed dependencies.

        `clones`, `genuine` and `sealed` are closure cells of this function, and
        every global it resolves — including `verified_authority` itself —
        resolves in the private sealed namespace. There is no module attribute
        whose rebinding changes which code runs here, and any rebinding of one
        of the pinned names is refused rather than quietly ignored.
        """
        live = _live_module_dict()
        for name, value in genuine.items():
            private = sealed.get(name, _ABSENT)
            if private is not (clones[name] if name in clones else value):
                raise ProtocolDriftError(   # pragma: no cover - namespace surgery
                    f"the sealed fresh-authority verifier's private binding for "
                    f"{name} was replaced; authority is not established"
                )
            if live.get(name, _ABSENT) is not value:
                raise ProtocolDriftError(
                    f"rq3_protocol.{name} is not the object the sealed "
                    f"fresh-authority verifier was built from. A formal boundary "
                    f"never establishes authority through a substituted "
                    f"dependency, and never accepts a stale result in place of a "
                    f"fresh verification."
                )
        return verified_authority(include_git=include_git)

    entry = types.FunctionType(sealed_verified_authority.__code__, sealed,
                               "sealed_verified_authority",
                               sealed_verified_authority.__defaults__,
                               sealed_verified_authority.__closure__)
    entry.__kwdefaults__ = sealed_verified_authority.__kwdefaults__
    entry.__qualname__ = "sealed_verified_authority"
    entry.__module__ = __name__
    entry.__doc__ = sealed_verified_authority.__doc__
    return entry


#: The single fresh-authority verifier every formal boundary uses. Callers
#: capture **this object** at definition time; nothing on a formal path looks it
#: up by name at call time.
sealed_verified_authority = _seal_fresh_authority_verifier(
    _SEALED_VERIFIER_CHAIN, _SEALED_VERIFIER_VALUES)

#: What the seal pinned, for tests and audit — the names, never the objects and
#: never the private namespace.
SEALED_VERIFIER_PINNED_NAMES = tuple(sorted(
    set(_SEALED_VERIFIER_CHAIN) | set(_SEALED_VERIFIER_VALUES)))
