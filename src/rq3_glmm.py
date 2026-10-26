"""Frequentist logistic random-intercept GLMM by marginal maximum likelihood.

`docs/RQ3_PROTOCOL.md` §9, implemented exactly as frozen at `RQ3_PROTOCOL/2.0.1`
(`rq3-protocol-v2`). One scalar site random intercept, **mode/curvature-adaptive
Gauss-Hermite quadrature** (§9.2, amended at Stage E.2a), log-sum-exp arithmetic
throughout, `log_sigma` parameterization, the **exact total derivative** of the
adaptive objective as an analytic gradient (§9.4.4), L-BFGS-B with the frozen
settings, the frozen deterministic four-start scheme, and the frozen §9.5.1
status codes.

    L_s(beta, sigma) = int prod_{i in s} p_i(u)^{R_i} (1-p_i(u))^{1-R_i}
                       phi(u; 0, sigma^2) du
    p_i(u)           = logistic(eta_i + u)

is evaluated by writing the whole integrand — the normal density included — as
one log-integrand (§9.2.1)

    h_s(u) = sum_{i in s} logsigmoid( c_i (eta_i + u) )
             - u^2/(2 sigma^2) - log(sigma) - 0.5 log(2 pi)        c_i = 2R_i - 1

so that `L_s = int exp(h_s(u)) du` exactly, locating that site's conditional
mode `u_hat_s` with the frozen safeguarded-Newton solver of §9.2.5, and placing
the nodes there at the local curvature scale (§9.2.2, §9.2.3):

    H_s(u) = S2_s(u) + 1/sigma^2        tau_s = H_s(u_hat_s) ** -0.5
    u_{s,k} = u_hat_s + sqrt(2) tau_s z_k

    log L_s = log(sqrt(2) tau_s) + logsumexp_k [ log w_k + z_k^2 + h_s(u_{s,k}) ]

The `log(sqrt(2) tau_s)` Jacobian and the `z_k^2` kernel de-weighting are both
**mandatory**; dropping either silently rescales every likelihood in the
protocol (H20). Setting `u_hat_s = 0` and `tau_s = sigma` collapses the `z_k^2`
terms against `-u_k^2/(2 sigma^2)` and reproduces the superseded Stage E.1 rule
exactly (§9.2.4) — that identity is a unit-test target, never a runtime option.
**The non-adaptive rule is withdrawn**: it is not selected, restored, offered as
an option, or used as a fallback when the mode solver fails (§17.14, H20).

Because the nodes are functions of the parameters, the Stage E.1 fixed-node
gradient is **not** the gradient of this objective and is not reused (H21).
`_marginal_loglik_and_grad` supplies the §9.4.4 exact total derivative, node
movement included, from one mode solve per site per evaluation.

No probability is exponentiated before summation and **no clipping is applied
anywhere** (§9.2). The environment has no frequentist binomial GLMM to defer to
(§9): `statsmodels.MixedLM` is Gaussian-response, `BinomialBayesMixedGLM` is
variational and yields no likelihood, and `lme4`/`glmmTMB`/`rpy2` are absent —
and three of the five registered outputs need a likelihood.

Nothing here is a preference. The quadrature, the mode solver and its two
derived tolerances, the optimizer, its tolerances, the starts and the validity
conditions are frozen, and §17.8 / §17.15 forbid changing any of them on
substantive grounds — least of all because a fit failed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

import rq3_protocol as proto

NOT_ESTIMABLE = proto.NOT_ESTIMABLE

#: §9.5.1 — the registered codes this module can emit. Every one of them is a
#: **numerical** outcome; none of them is ever a scientific null (§9.5.1, H26).
COND_CONVERGED = "OPTIMIZER_NOT_CONVERGED"
COND_OBJECTIVE_FINITE = "NON_FINITE_OBJECTIVE"
COND_GRADIENT = "GRADIENT_NOT_ZERO"
COND_SIGMA = "SIGMA_AT_OR_OUTSIDE_BOUND"
COND_MODE_NOT_FOUND = "ADAPTIVE_MODE_NOT_FOUND"
COND_CURVATURE_INVALID = "ADAPTIVE_CURVATURE_INVALID"
COND_NO_ELIGIBLE_START = "NO_ELIGIBLE_START"
#: §9.6.1 step 7 — the unit-level consequence of a 40-vs-60 disagreement.
COND_QUADRATURE = "QUADRATURE_DISAGREEMENT"

_SQRT2 = math.sqrt(2.0)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class GLMMStructuralError(AssertionError):
    """Raised on structural corruption of a fit's inputs.

    Distinct from `NOT_ESTIMABLE` (§9.5, §9.5.1): a numerical failure to
    estimate is a *registered outcome*, while a malformed design matrix, a
    non-binary outcome or non-contiguous site grouping is corruption and must
    hard-fail rather than be coerced into one (§9.5.1, H26).
    """


class AdaptiveQuadratureFailure(RuntimeError):
    """The §9.2.5 mode solver could not produce a mode for some site.

    Carries the solver's own registered §9.5.1 code. The objective is then
    unevaluable at that `theta`, which makes the start ineligible under §9.5
    condition 2 **with this code**, not a generic one (§9.5). There is no
    fallback solver and no fallback quadrature rule (§17.14, §17.15, H22).
    """

    def __init__(self, code: str, site_index: int):
        if code not in (COND_MODE_NOT_FOUND, COND_CURVATURE_INVALID):
            raise GLMMStructuralError(f"{code!r} is not a registered mode-solver code")
        super().__init__(f"{code} at site index {site_index}")
        self.code = code
        self.site_index = int(site_index)


@lru_cache(maxsize=8)
def gauss_hermite(n_nodes: int) -> tuple:
    """§9.2 — nodes and log-weights of the frozen `hermgauss` rule.

    Cached because the rule depends on nothing but the node count; the returned
    arrays are made read-only so a caller cannot mutate the shared rule.
    """
    if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) or n_nodes < 1:
        raise GLMMStructuralError(f"node count must be a positive integer, got {n_nodes!r}")
    z, w = np.polynomial.hermite.hermgauss(n_nodes)
    log_w = np.log(w)
    z.flags.writeable = False
    log_w.flags.writeable = False
    return z, log_w


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable `log(1/(1+exp(-x)))`, with no clipping (§9.2.1)."""
    return -np.logaddexp(0.0, -np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# site grouping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteGrouping:
    """Contiguous row blocks, one per site.

    Rows must already be grouped by site (the RQ3 datasets are built sorted by
    site, then `canonical_id`, so this holds by construction). Requiring it
    rather than sorting here keeps every array in one row order end to end,
    which is what H8 — identical row index across the models of a family —
    depends on.
    """

    labels: tuple
    starts: np.ndarray
    sizes: np.ndarray
    row_site: np.ndarray

    @property
    def n_sites(self) -> int:
        return len(self.labels)

    @property
    def n_obs(self) -> int:
        return int(self.sizes.sum())


def group_by_site(site_labels) -> SiteGrouping:
    """Build the contiguous grouping, hard-failing if rows are interleaved."""
    labels = np.asarray(list(site_labels), dtype=object)
    if labels.size == 0:
        raise GLMMStructuralError("a GLMM fit needs at least one row")
    boundaries = [0]
    for position in range(1, labels.size):
        if labels[position] != labels[position - 1]:
            boundaries.append(position)
    starts = np.asarray(boundaries, dtype=np.intp)
    ordered = [labels[i] for i in starts]
    if len(set(ordered)) != len(ordered):
        raise GLMMStructuralError(
            "site rows are not contiguous — a site appears in more than one block"
        )
    sizes = np.diff(np.append(starts, labels.size)).astype(np.intp)
    row_site = np.repeat(np.arange(len(starts), dtype=np.intp), sizes)
    return SiteGrouping(tuple(ordered), starts, sizes, row_site)


def singleton_grouping(n_rows: int) -> SiteGrouping:
    """One row per group — the §10.3 one-row sets, not a site structure.

    §10.3 evaluates each subject's two prediction branches by applying §9.2.3 to
    the single-row set `{i}`. That is the same rule with a one-row group, so it
    reuses the same machinery rather than introducing a second quadrature path.
    """
    if not isinstance(n_rows, (int, np.integer)) or isinstance(n_rows, bool) or n_rows < 1:
        raise GLMMStructuralError(f"a one-row grouping needs >= 1 row, got {n_rows!r}")
    count = int(n_rows)
    index = np.arange(count, dtype=np.intp)
    return SiteGrouping(
        labels=tuple(range(count)),
        starts=index,
        sizes=np.ones(count, dtype=np.intp),
        row_site=index,
    )


# ---------------------------------------------------------------------------
# §9.2.1 the conditional log-integrand and its first two derivatives
# ---------------------------------------------------------------------------

def h_terms(u: np.ndarray, eta: np.ndarray, R: np.ndarray, sigma: float,
            grouping: SiteGrouping) -> tuple:
    """§9.2.1 — `h_s(u_s)`, `h_s'(u_s)` and `H_s(u_s)` for every site at once.

    `u` carries one offset per site. `H_s = S2_s + 1/sigma^2 >= 1/sigma^2 > 0`,
    so `h_s` is strictly concave and its curvature is positive **by
    construction, not by assumption** (§9.2.1).
    """
    offsets = np.asarray(u, dtype=float)
    row_offset = offsets[grouping.row_site]
    linear = eta + row_offset
    sign = 2.0 * R - 1.0

    site_log_sigmoid = np.add.reduceat(log_sigmoid(sign * linear), grouping.starts)
    inverse_variance = 1.0 / (sigma * sigma)
    h = (site_log_sigmoid - 0.5 * offsets * offsets * inverse_variance
         - math.log(sigma) - _HALF_LOG_2PI)

    p = expit(linear)
    s1 = np.add.reduceat(R - p, grouping.starts)
    s2 = np.add.reduceat(p * (1.0 - p), grouping.starts)
    return h, s1 - offsets * inverse_variance, s2 + inverse_variance


def _first_failure(mask: np.ndarray, code: str) -> None:
    """Raise for the lowest-indexed failing site, so the code is deterministic."""
    if mask.any():
        raise AdaptiveQuadratureFailure(code, int(np.flatnonzero(mask)[0]))


def solve_modes(eta: np.ndarray, R: np.ndarray, sigma: float,
                grouping: SiteGrouping) -> tuple:
    """§9.2.5 — the frozen safeguarded-Newton mode solver, run for every site.

    Every site follows the frozen scalar iteration independently; they are
    stepped together only because the arithmetic vectorizes. The initialization
    (`u = 0`), the two tolerances, the halving limit and the iteration budget are
    registered constants and are read from the frozen protocol — none may be
    retuned, and there is no fallback solver (§9.2.5, §17.15, H22).

    Returns `(u_hat, H_hat)`. Any site that cannot be solved raises
    `AdaptiveQuadratureFailure` with its registered §9.5.1 code; the solver never
    continues past a non-finite value and never substitutes a default mode
    (§9.2.6).
    """
    n_sites = grouping.n_sites
    u = np.full(n_sites, float(proto.MODE_INIT))
    h, gradient, curvature = h_terms(u, eta, R, sigma, grouping)

    settled = np.zeros(n_sites, dtype=bool)

    for _iteration in range(proto.MODE_MAXITER):
        active = ~settled
        if not active.any():
            return u, curvature

        _first_failure(
            active & (~np.isfinite(curvature) | (curvature <= proto.CURVATURE_FLOOR)),
            COND_CURVATURE_INVALID,
        )
        _first_failure(
            active & (~np.isfinite(h) | ~np.isfinite(gradient)), COND_MODE_NOT_FOUND
        )

        step = np.where(active, gradient / curvature, 0.0)
        # the remaining distance to the mode, in units of the adaptive scale
        offset = np.abs(step) * np.sqrt(curvature)

        settled |= active & (offset <= proto.MODE_CONVERGENCE_TOL)
        active = ~settled
        if not active.any():
            return u, curvature

        # monotone safeguard: at most the frozen number of halvings, per site
        scale = np.ones(n_sites)
        moved = np.zeros(n_sites, dtype=bool)
        stalled = np.zeros(n_sites, dtype=bool)
        for _halving in range(proto.MODE_MAX_HALVINGS + 1):
            trying = active & ~moved & ~stalled
            if not trying.any():
                break
            candidate = np.where(trying, u + scale * step, u)
            # a step below representable resolution is not a step
            stalled |= trying & (candidate == u)
            trying = trying & ~stalled
            if not trying.any():
                break
            h_try, g_try, H_try = h_terms(candidate, eta, R, sigma, grouping)
            accepted = trying & np.isfinite(h_try) & (h_try >= h)
            u = np.where(accepted, candidate, u)
            h = np.where(accepted, h_try, h)
            gradient = np.where(accepted, g_try, gradient)
            curvature = np.where(accepted, H_try, curvature)
            moved |= accepted
            scale = np.where(trying & ~accepted, scale * 0.5, scale)

        stuck = active & ~moved
        # a maximum of a strictly concave h that no representable step improves
        # on *is* the numerical maximum — accepted only inside the frozen
        # fail-closed cap, beyond which it is a registered failure (§9.2.5)
        settled |= stuck & (offset <= proto.MODE_STAGNATION_TOL)
        _first_failure(stuck & ~settled, COND_MODE_NOT_FOUND)

    _first_failure(~settled, COND_MODE_NOT_FOUND)     # budget exhausted
    return u, curvature


# ---------------------------------------------------------------------------
# §9.2.3 the adaptive rule, and §9.4.4 its exact total derivative
# ---------------------------------------------------------------------------

def _validate_inputs(X: np.ndarray, R: np.ndarray) -> tuple:
    design = np.asarray(X, dtype=float)
    outcome = np.asarray(R, dtype=float)
    if design.ndim != 2:
        raise GLMMStructuralError(f"design matrix must be 2-D, got ndim {design.ndim}")
    if outcome.ndim != 1:
        raise GLMMStructuralError(f"outcome must be 1-D, got ndim {outcome.ndim}")
    if design.shape[0] != outcome.shape[0]:
        raise GLMMStructuralError(
            f"design has {design.shape[0]} rows, outcome has {outcome.shape[0]}"
        )
    if design.shape[0] == 0:
        raise GLMMStructuralError("a GLMM fit needs at least one row")
    if not np.all(np.isfinite(design)):
        raise GLMMStructuralError("design matrix holds a non-finite entry")
    if not np.all((outcome == 0.0) | (outcome == 1.0)):
        raise GLMMStructuralError("outcome R must be binary 0/1")
    return design, outcome


def adaptive_site_loglik(eta: np.ndarray, R: np.ndarray, sigma: float,
                         grouping: SiteGrouping, n_nodes: int) -> np.ndarray:
    """§9.2.3 — `log L_s` for every group, by the frozen adaptive rule.

    Both structural terms are present and neither is optional: the Jacobian
    `log(sqrt(2) tau_s)` of the change of variables, and the `z_k^2`
    de-weighting of the Gauss-Hermite kernel (H20).
    """
    z, log_w = gauss_hermite(n_nodes)
    u_hat, H_hat = solve_modes(eta, R, sigma, grouping)
    tau = H_hat ** -0.5

    nodes = u_hat[:, None] + _SQRT2 * tau[:, None] * z[None, :]      # (S, K)
    row_nodes = nodes[grouping.row_site]                             # (n, K)
    sign = 2.0 * R - 1.0
    per_row = log_sigmoid(sign[:, None] * (eta[:, None] + row_nodes))
    site_terms = np.add.reduceat(per_row, grouping.starts, axis=0)   # (S, K)

    h_nodes = (site_terms - 0.5 * nodes * nodes / (sigma * sigma)
               - math.log(sigma) - _HALF_LOG_2PI)
    weighted = log_w[None, :] + (z * z)[None, :] + h_nodes
    return np.log(_SQRT2 * tau) + logsumexp(weighted, axis=1)


def marginal_loglik_and_grad(theta: np.ndarray, X: np.ndarray, R: np.ndarray,
                             grouping: SiteGrouping, n_nodes: int) -> tuple:
    """The §9.1 marginal log-likelihood by §9.2.3, with the §9.4.4 gradient.

    `theta = [beta, log_sigma]`. Returns `(loglik, gradient)` — the likelihood
    itself, not the objective; `_objective` negates both for the minimizer.

    The gradient is the **exact total derivative** of the adaptive objective:
    the nodes move with the parameters, so the withdrawn fixed-node gradient is
    the gradient of a different objective and is not reused (§9.4.4, H21). With
    `P_{s,k}` the log-sum-exp node weights of §9.4.4,

        d logL_s/d theta =   d log tau_s / d theta
                           + sum_k P_sk dh_s(u_sk)/d theta
                           + ( sum_k P_sk h_s'(u_sk) ) du_hat_s/d theta
                           + sqrt(2) tau_s (d log tau_s/d theta) sum_k P_sk z_k h_s'(u_sk)

    where the mode-movement derivatives come from implicit differentiation of
    `h_s'(u_hat_s) = 0`. The objective and its gradient are computed in one pass,
    at one `theta`, from **one** mode solve per site. No numerical
    differentiation is used anywhere (H21).
    """
    theta = np.asarray(theta, dtype=float)
    n_beta = X.shape[1]
    beta = theta[:n_beta]
    log_sigma = float(theta[n_beta])
    sigma = math.exp(log_sigma)
    inverse_variance = 1.0 / (sigma * sigma)
    z, log_w = gauss_hermite(n_nodes)

    eta = X @ beta
    u_hat, H_hat = solve_modes(eta, R, sigma, grouping)
    tau = H_hat ** -0.5

    # ---- §9.2.3 the objective ------------------------------------------
    nodes = u_hat[:, None] + _SQRT2 * tau[:, None] * z[None, :]      # (S, K)
    row_nodes = nodes[grouping.row_site]                             # (n, K)
    linear = eta[:, None] + row_nodes                                # (n, K)
    sign = 2.0 * R - 1.0
    site_terms = np.add.reduceat(log_sigmoid(sign[:, None] * linear),
                                 grouping.starts, axis=0)            # (S, K)
    h_nodes = (site_terms - 0.5 * nodes * nodes * inverse_variance
               - math.log(sigma) - _HALF_LOG_2PI)
    weighted = log_w[None, :] + (z * z)[None, :] + h_nodes
    site_norm = logsumexp(weighted, axis=1)                          # (S,)
    loglik = float(np.sum(np.log(_SQRT2 * tau) + site_norm))

    node_weights = np.exp(weighted - site_norm[:, None])             # (S, K)

    # ---- partial derivatives at fixed u --------------------------------
    residual = R[:, None] - expit(linear)                            # (n, K)
    row_weights = node_weights[grouping.row_site]                    # (n, K)
    grad_beta = X.T @ np.einsum("nk,nk->n", residual, row_weights)
    grad_log_sigma = float(np.sum(
        node_weights * (nodes * nodes * inverse_variance - 1.0)
    ))

    # ---- the node-movement contribution (§9.4.4) ------------------------
    h_prime = (np.add.reduceat(residual, grouping.starts, axis=0)
               - nodes * inverse_variance)                           # (S, K)
    weighted_h_prime = np.sum(node_weights * h_prime, axis=1)                 # (S,)
    weighted_z_h_prime = np.sum(node_weights * z[None, :] * h_prime, axis=1)  # (S,)

    at_mode = expit(eta + u_hat[grouping.row_site])                  # (n,)
    variance = at_mode * (1.0 - at_mode)
    asymmetry = variance * (1.0 - 2.0 * at_mode)
    s3 = np.add.reduceat(asymmetry, grouping.starts)                 # (S,)
    q = np.add.reduceat(X * variance[:, None], grouping.starts, axis=0)   # (S, p)
    t = np.add.reduceat(X * asymmetry[:, None], grouping.starts, axis=0)  # (S, p)

    d_mode_d_beta = -q / H_hat[:, None]                              # (S, p)
    d_mode_d_log_sigma = 2.0 * u_hat * inverse_variance / H_hat      # (S,)
    d_log_tau_d_beta = -0.5 * (t + s3[:, None] * d_mode_d_beta) / H_hat[:, None]
    d_log_tau_d_log_sigma = -0.5 * (s3 * d_mode_d_log_sigma
                                    - 2.0 * inverse_variance) / H_hat

    movement = _SQRT2 * tau * weighted_z_h_prime                     # (S,)
    grad_beta = grad_beta + np.sum(
        d_log_tau_d_beta
        + weighted_h_prime[:, None] * d_mode_d_beta
        + movement[:, None] * d_log_tau_d_beta,
        axis=0,
    )
    grad_log_sigma += float(np.sum(
        d_log_tau_d_log_sigma
        + weighted_h_prime * d_mode_d_log_sigma
        + movement * d_log_tau_d_log_sigma
    ))

    return loglik, np.concatenate([grad_beta, [grad_log_sigma]])


def _objective(theta, X, R, grouping, n_nodes, failure=None):
    """Negative marginal log-likelihood and its gradient, for `jac=True`.

    A `theta` at which the frozen mode solver fails has no evaluable objective.
    The optimizer is handed a large finite value with a zero gradient rather
    than a `NaN` it would propagate, and the solver's registered §9.5.1 code is
    recorded so the start's rejection can name it rather than a generic reason
    (§9.5). Nothing is retried and no fallback rule is substituted.
    """
    try:
        loglik, gradient = marginal_loglik_and_grad(theta, X, R, grouping, n_nodes)
    except AdaptiveQuadratureFailure as exc:
        if failure is not None:
            failure["code"] = exc.code
        return np.inf, np.zeros_like(np.asarray(theta, dtype=float))
    if not math.isfinite(loglik) or not np.all(np.isfinite(gradient)):
        # §9.5 conditions 2-3; the validity check rejects the start
        return np.inf, np.zeros_like(np.asarray(theta, dtype=float))
    return -loglik, -gradient


# ---------------------------------------------------------------------------
# one start, one fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StartResult:
    """One of the four registered starts of §9.4, after optimization."""

    index: int
    sigma_start: float
    eligible: bool
    loglik: float | None
    failing_condition: str | None
    beta: tuple = ()
    log_sigma: float | None = None
    sigma_site: float | None = None
    max_abs_gradient: float | None = None
    converged: bool = False
    n_iterations: int = 0
    optimizer_message: str = ""

    def to_record(self) -> dict:
        """The record shape the frozen `select_best_start` consumes (§9.4.1).

        It carries the registered identity pair `(index, sigma_start)`, the
        eligibility verdict and the marginal log-likelihood, and nothing the
        selector is not entitled to see.
        """
        return {
            "index": int(self.index),
            "sigma_start": float(self.sigma_start),
            "eligible": bool(self.eligible),
            "loglik": None if self.loglik is None else float(self.loglik),
        }

    def to_payload(self) -> dict:
        return {
            "index": int(self.index),
            "sigma_start": float(self.sigma_start),
            "eligible": bool(self.eligible),
            "loglik": None if self.loglik is None else float(self.loglik),
            "failing_condition": self.failing_condition,
            "beta": [float(v) for v in self.beta],
            "log_sigma": None if self.log_sigma is None else float(self.log_sigma),
            "sigma_site": None if self.sigma_site is None else float(self.sigma_site),
            "max_abs_gradient": (None if self.max_abs_gradient is None
                                 else float(self.max_abs_gradient)),
            "converged": bool(self.converged),
            "n_iterations": int(self.n_iterations),
        }


@dataclass(frozen=True)
class GLMMFit:
    """The selected solution for one model at one node count."""

    status: str                       # "ESTIMABLE" or NOT_ESTIMABLE
    n_nodes: int
    n_obs: int
    n_sites: int
    n_params: int
    selected_start_index: object = NOT_ESTIMABLE
    loglik: float | None = None
    beta: tuple = ()
    log_sigma: float | None = None
    sigma_site: float | None = None
    max_abs_gradient: float | None = None
    failing_condition: str | None = None
    starts: tuple = field(default_factory=tuple)

    @property
    def estimable(self) -> bool:
        return self.status == proto.STATUS_OK

    @property
    def sigma_site_sq(self) -> float | None:
        return None if self.sigma_site is None else float(self.sigma_site) ** 2

    def require(self) -> "GLMMFit":
        if not self.estimable:
            raise GLMMStructuralError(
                f"fit is {self.status} ({self.failing_condition}) and has no solution"
            )
        return self

    def to_payload(self) -> dict:
        return {
            "status": self.status,
            "quadrature_rule": proto.QUADRATURE_RULE,
            "n_nodes": int(self.n_nodes),
            "n_obs": int(self.n_obs),
            "n_sites": int(self.n_sites),
            "n_params": int(self.n_params),
            "selected_start_index": (self.selected_start_index
                                     if isinstance(self.selected_start_index, str)
                                     else int(self.selected_start_index)),
            "loglik": None if self.loglik is None else float(self.loglik),
            "beta": [float(v) for v in self.beta],
            "log_sigma": None if self.log_sigma is None else float(self.log_sigma),
            "sigma_site": None if self.sigma_site is None else float(self.sigma_site),
            "sigma_site_sq": self.sigma_site_sq,
            "max_abs_gradient": (None if self.max_abs_gradient is None
                                 else float(self.max_abs_gradient)),
            "failing_condition": self.failing_condition,
            "starts": [s.to_payload() for s in self.starts],
        }


def _sigma_is_valid(log_sigma: float) -> bool:
    """§9.5 condition 4 / §9.3 — positive, finite, and off both box bounds.

    L-BFGS-B returns a clamped coordinate *exactly* at its bound, so exact
    comparison detects contact without inventing a tolerance the protocol did
    not freeze. §9.2.6 leaves this condition untouched by the amendment.
    """
    lower, upper = proto.LOG_SIGMA_BOUNDS
    if not math.isfinite(log_sigma):
        return False
    if log_sigma <= float(lower) or log_sigma >= float(upper):
        return False
    return math.isfinite(math.exp(log_sigma)) and math.exp(log_sigma) > 0.0


def _run_one_start(index: int, beta_start: np.ndarray, log_sigma_start: float,
                   X: np.ndarray, R: np.ndarray, grouping: SiteGrouping,
                   n_nodes: int) -> StartResult:
    """One L-BFGS-B run under the frozen §9.4 settings, then §9.5 conditions 1-4.

    `ftol = 1e-16` is the Stage E.2a.1 amended value (§9.4.5): it is the
    *relative* objective criterion, and at the frozen sample sizes the
    superseded `1e-12` fired before `gtol = 1e-8` could, stopping the optimizer
    at a point the already-frozen `max abs gradient <= 1e-5` eligibility rule
    then rejected. The eligibility threshold itself is unchanged.
    """
    sigma_start = float(proto.START_INDEX_TO_SIGMA[index])
    theta0 = np.concatenate([np.asarray(beta_start, dtype=float), [float(log_sigma_start)]])
    lower, upper = proto.LOG_SIGMA_BOUNDS
    bounds = [(None, None)] * (theta0.size - 1) + [(float(lower), float(upper))]
    failure = {"code": None}

    result = minimize(
        _objective, theta0,
        args=(X, R, grouping, n_nodes, failure),
        method=proto.OPTIMIZER,
        jac=True,                       # the §9.4.4 exact total derivative (H21)
        bounds=bounds,
        options={
            "ftol": proto.OPT_FTOL,
            "gtol": proto.OPT_GTOL,
            "maxiter": proto.OPT_MAXITER,
            "maxfun": proto.OPT_MAXFUN,
        },
    )

    theta = np.asarray(result.x, dtype=float)
    beta = theta[:-1]
    log_sigma = float(theta[-1])

    mode_failure = None
    if np.all(np.isfinite(theta)):
        try:
            loglik, gradient = marginal_loglik_and_grad(theta, X, R, grouping, n_nodes)
            max_abs_grad = (float(np.max(np.abs(gradient)))
                            if np.all(np.isfinite(gradient)) else None)
        except AdaptiveQuadratureFailure as exc:
            mode_failure = exc.code
            loglik, max_abs_grad = float("nan"), None
    else:
        loglik, max_abs_grad = float("nan"), None

    def reject(condition: str) -> StartResult:
        return StartResult(
            index=index, sigma_start=sigma_start, eligible=False, loglik=None,
            failing_condition=condition, beta=tuple(float(v) for v in beta),
            log_sigma=log_sigma if math.isfinite(log_sigma) else None,
            sigma_site=None, max_abs_gradient=max_abs_grad, converged=bool(result.success),
            n_iterations=int(getattr(result, "nit", 0)),
            optimizer_message=str(getattr(result, "message", "")),
        )

    # §9.5, conditions 1-4, in their frozen order. A start whose objective could
    # not be evaluated because the §9.2.5 solver failed is ineligible under
    # condition 2 carrying the solver's own registered code (§9.5, §9.5.1).
    if not bool(result.success):
        return reject(mode_failure or failure["code"] or COND_CONVERGED)
    if mode_failure is not None:
        return reject(mode_failure)
    if not math.isfinite(loglik):
        return reject(COND_OBJECTIVE_FINITE)
    if max_abs_grad is None or not math.isfinite(max_abs_grad) \
            or max_abs_grad > proto.GRADIENT_TOL:
        return reject(COND_GRADIENT)
    if not _sigma_is_valid(log_sigma):
        return reject(COND_SIGMA)

    return StartResult(
        index=index, sigma_start=sigma_start, eligible=True, loglik=float(loglik),
        failing_condition=None, beta=tuple(float(v) for v in beta),
        log_sigma=log_sigma, sigma_site=float(math.exp(log_sigma)),
        max_abs_gradient=max_abs_grad, converged=True,
        n_iterations=int(getattr(result, "nit", 0)),
        optimizer_message=str(getattr(result, "message", "")),
    )


def fit_glmm(X, R, site_labels, n_nodes: int) -> GLMMFit:
    """Fit the §9.1 GLMM by marginal ML, under the frozen four-start scheme.

    `X` already carries its intercept column; `R` is the binary outcome; rows
    are grouped by site. `n_nodes` is 40 (authoritative) or 60 (validation) —
    the initializer does not depend on it (§9.6.1 step 2), so both node counts
    start from the same four points, and neither the mode solver nor the
    adaptive transformation depends on it either.

    The rows handed in **are** the fitting rows: for a LOSO fold they are the
    training sites only, so the initializer physically cannot reach the held-out
    site's outcomes (§9.4, H17).

    Returns a `GLMMFit`; a fit with no eligible start is `NOT_ESTIMABLE` with
    `NO_ELIGIBLE_START`, and no fallback solver, penalty, extra start, node-count
    change or 60-node substitution is attempted (§9.4.1 step 6, §9.6.1 step 5).
    """
    design, outcome = _validate_inputs(X, R)
    grouping = group_by_site(site_labels)
    if grouping.n_obs != design.shape[0]:
        raise GLMMStructuralError(
            f"{grouping.n_obs} grouped rows vs {design.shape[0]} design rows"
        )

    n_non_intercept = design.shape[1] - 1
    if n_non_intercept < 0:
        raise GLMMStructuralError("design matrix has no intercept column")
    # the frozen initializer, computed from the fitting rows alone
    starts = proto.deterministic_initial_points(outcome, n_non_intercept)
    if len(starts) != proto.N_REGISTERED_STARTS:
        raise GLMMStructuralError(
            f"initializer produced {len(starts)} starts, registry holds "
            f"{proto.N_REGISTERED_STARTS}"
        )

    results = []
    for index, (beta_start, log_sigma_start) in enumerate(starts):
        # the initializer emits starts in registered index order; the frozen
        # selector re-checks that each position carries its registered sigma
        expected = float(proto.START_INDEX_TO_SIGMA[index])
        if not math.isclose(math.exp(log_sigma_start), expected, rel_tol=1e-12, abs_tol=0.0):
            raise GLMMStructuralError(
                f"start {index} carries sigma {math.exp(log_sigma_start)!r}, "
                f"registered value is {expected!r}"
            )
        results.append(
            _run_one_start(index, beta_start, log_sigma_start,
                           design, outcome, grouping, n_nodes)
        )

    # the frozen selector — position defines the registered index (§9.4.3)
    selected = proto.select_best_start([r.to_record() for r in results])

    common = {
        "n_nodes": int(n_nodes),
        "n_obs": int(design.shape[0]),
        "n_sites": int(grouping.n_sites),
        "n_params": int(design.shape[1]) + 1,   # fixed effects + log_sigma
        "starts": tuple(results),
    }
    if selected == NOT_ESTIMABLE:
        return GLMMFit(
            status=NOT_ESTIMABLE,
            selected_start_index=NOT_ESTIMABLE,
            failing_condition=COND_NO_ELIGIBLE_START,
            **common,
        )

    winner = results[int(selected)]
    return GLMMFit(
        status=proto.STATUS_OK,
        selected_start_index=int(selected),
        loglik=float(winner.loglik),
        beta=winner.beta,
        log_sigma=winner.log_sigma,
        sigma_site=winner.sigma_site,
        max_abs_gradient=winner.max_abs_gradient,
        failing_condition=None,
        **common,
    )


# ---------------------------------------------------------------------------
# §10.3 marginalized prediction for an unobserved site
# ---------------------------------------------------------------------------

def marginal_log_probabilities(eta, sigma_site: float, n_nodes: int) -> tuple:
    """§10.3 — `log p_i` and `log(1-p_i)`, marginalized over `u ~ N(0, sigma^2)`.

    **AMENDED (Stage E.2a).** Each branch is §9.2.3 applied to the one-row set
    `{i}`, with the outcome argument of `h` set to the branch rather than to the
    observed `R_i`, so each branch gets its own conditional mode and curvature
    from the frozen §9.2.5 solver. This introduces no new mechanism, no new
    constant and no new solver, and leaves the estimand untouched: the same
    marginal probability under the same `N(0, sigma_hat^2)` prior.

    The held-out site's intercept is **unobserved**: it is integrated out under
    the training fit's random-effect distribution, never estimated, tuned,
    shrunk or empirical-Bayes-updated from held-out outcomes (§10.3, H9). The
    observed `R_i` never reaches this function.

    Both branches are formed in log space, so no clipping, flooring or epsilon
    is applied — and none may be introduced.
    """
    linear_predictor = np.asarray(eta, dtype=float)
    if linear_predictor.ndim != 1:
        raise GLMMStructuralError("eta must be 1-D")
    if linear_predictor.size == 0:
        raise GLMMStructuralError("a prediction needs at least one row")
    if not np.all(np.isfinite(linear_predictor)):
        raise GLMMStructuralError("eta holds a non-finite entry")
    sigma = float(sigma_site)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise GLMMStructuralError(f"sigma_site must be finite and positive, got {sigma!r}")

    grouping = singleton_grouping(linear_predictor.size)
    ones = np.ones(linear_predictor.size)
    log_p = adaptive_site_loglik(linear_predictor, ones, sigma, grouping, n_nodes)
    log_1mp = adaptive_site_loglik(linear_predictor, 1.0 - ones, sigma, grouping, n_nodes)
    return log_p, log_1mp


def marginal_log_loss(R, eta, sigma_site: float, n_nodes: int) -> np.ndarray:
    """Per-row `-( R log p + (1-R) log(1-p) )` under §10.3.

    `R` selects which branch enters the loss **and nothing else**; it never
    reaches the mode solver, the quadrature, `beta_hat` or `sigma_hat` (§10.3).
    """
    outcome = np.asarray(R, dtype=float)
    if outcome.ndim != 1:
        raise GLMMStructuralError("R must be 1-D")
    if not np.all((outcome == 0.0) | (outcome == 1.0)):
        raise GLMMStructuralError("outcome R must be binary 0/1")
    log_p, log_1mp = marginal_log_probabilities(eta, sigma_site, n_nodes)
    if log_p.shape != outcome.shape:
        raise GLMMStructuralError(
            f"{outcome.size} outcomes vs {log_p.size} predictions"
        )
    return -(outcome * log_p + (1.0 - outcome) * log_1mp)
