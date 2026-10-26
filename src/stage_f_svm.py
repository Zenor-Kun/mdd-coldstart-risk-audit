"""F-SVM — the linear-SVM classification-robustness comparator (§F).

**One number, one sentence** (spec §6 L397): the site-macro outer-LOSO balanced
accuracy of a linear-SVM pipeline that is otherwise identical to the frozen
ridge-logistic one. The model family changes; the estimand does not.

F-SVM produces **no** confidence score, threshold, abstention decision,
calibrated probability or conformal set, and it enters neither S0 selective
prediction, S1 weighting, S2 certification, RQ3 attribution nor RQ4 conformal
prediction (§F.1). AUC, F1, plain accuracy, selective-risk curves, Platt scaling
and any significance test against logistic regression are forbidden outputs
(§F.9) and no function here can produce one — `LinearSVC` has no `predict_proba`
at all, so probability calibration is structurally unavailable rather than
merely switched off.

Everything except the estimator is reused verbatim from the audited Gate-1
pipeline: `nested_loso.Preprocessor` (StandardScaler + one randomized-SVD PCA
per candidate, fit on the training partition only), `nested_loso.PCA_CANDIDATES`,
`nested_loso.C_GRID`, `nested_loso.site_equal_weights`,
`nested_loso.macro_site_bacc` and `contracts`' leakage assertions. That is what
keeps the comparison single-factor.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

import contracts
import nested_loso as nl
import stage_f_registry as reg


class SVMScopeError(AssertionError):
    """An F-SVM call left the frozen scope of §F."""


def assert_grids_match_frozen_pipeline() -> dict:
    """The `C` and PCA grids are the frozen ridge-logistic ones, reused exactly.

    Reusing the same numerical candidate grid is a **pre-result fairness and
    design choice** — the same search space, fixed before any Stage-F number
    existed. It is *not* an assertion that the two `C` axes lie on a common
    scale: the objectives differ (logistic loss vs squared hinge) and every
    claim of scale equivalence between them is forbidden (§F.6). What keeps the
    comparison single-factor is that the search procedure and the selection
    metric are identical, not that the two `C` parameters are.
    """
    if tuple(nl.C_GRID) != reg.SVM_C_GRID:
        raise SVMScopeError(
            f"nested_loso.C_GRID is {nl.C_GRID}, the frozen SVM grid is "
            f"{reg.SVM_C_GRID}")
    if tuple(nl.PCA_CANDIDATES) != reg.SVM_PCA_CANDIDATES:
        raise SVMScopeError(
            f"nested_loso.PCA_CANDIDATES is {nl.PCA_CANDIDATES}, the frozen SVM "
            f"grid is {reg.SVM_PCA_CANDIDATES}")
    if nl.MAX_ITER != reg.SVM_MAX_ITER:
        raise SVMScopeError(
            f"the frozen pipeline max_iter is {nl.MAX_ITER}, §F.7 mirrors "
            f"{reg.SVM_MAX_ITER}")
    return {"c_grid": list(reg.SVM_C_GRID),
            "pca_candidates": list(reg.SVM_PCA_CANDIDATES),
            "c_scale_equivalence_claimed": reg.SVM_C_SCALE_EQUIVALENCE_CLAIMED}


def make_estimator(C: float) -> LinearSVC:
    """The frozen §F.7 estimator. Every parameter is stated, none defaulted.

    `random_state` is `OPERATIONALLY_INERT` under `dual=False` — sklearn's own
    `LinearSVC` docstring states the primal implementation is not random and the
    parameter has no effect on the results. The literal is passed for
    constructor explicitness only and is never described as an operative seed;
    the PCA fits in the same pipeline do use `20260807` as a genuine seed, and
    the two uses are never conflated.
    """
    if float(C) not in reg.SVM_C_GRID:
        raise SVMScopeError(f"C={C} is off the frozen grid {reg.SVM_C_GRID}")
    return LinearSVC(
        penalty=reg.SVM_PENALTY,
        loss=reg.SVM_LOSS,
        dual=reg.SVM_DUAL,
        C=float(C),
        class_weight=reg.SVM_CLASS_WEIGHT,
        fit_intercept=reg.SVM_FIT_INTERCEPT,
        tol=reg.SVM_TOL,
        max_iter=reg.SVM_MAX_ITER,
        random_state=reg.SVM_RANDOM_STATE,
    )


def assert_no_probability_path(estimator) -> None:
    """`LinearSVC` exposes no `predict_proba`, and none may be attached (§F.9)."""
    for attribute in ("predict_proba", "predict_log_proba"):
        if hasattr(estimator, attribute):
            raise SVMScopeError(
                f"the SVM comparator exposes {attribute}; F-SVM produces no "
                "calibrated probability and Platt scaling is forbidden")


class SVMNotConverged(RuntimeError):
    """liblinear reported non-convergence. Fail-visible, never silent (§F.7)."""


class SVMExpectedEstimatorFailure(RuntimeError):
    """A §G.2a-permitted estimator/backend failure inside the fit boundary.

    Raised **only** by `fit_predict`, and only after both §G.2a conditions have
    been checked: the exception's type is exactly one of the two registered
    ones, and it was raised inside the frozen estimator/backend boundary. It
    carries the original exception so the audit trail keeps what actually
    happened; nothing reconstructs a `SVM_FOLD_FAILED` row from anything else.
    """

    def __init__(self, original: BaseException, where: str):
        super().__init__(f"{type(original).__name__} inside {where}: {original}")
        self.original = original
        self.where = where


def _permitted_failure_types() -> tuple:
    """Resolve §G.2a's two registered type **names** to the live class objects.

    The registry transcribes names, never type objects, so the protocol
    authority and the runtime cannot drift into naming different classes. The
    resolution is asserted rather than assumed: an environment in which either
    name no longer resolves to a class is a hard failure, not a silently shorter
    allow-list.

    `scipy.linalg.LinAlgError` is deliberately **not** resolved: §G.2a records
    it as an identity-equivalent alias of the numpy class in this environment,
    which adds no third semantic type, and resolving it would invite a future
    environment to turn one entry into two.
    """
    resolved = {"FloatingPointError": FloatingPointError,
                "numpy.linalg.LinAlgError": np.linalg.LinAlgError}
    if tuple(resolved) != reg.SVM_FOLD_FAILED_ALLOWED_TYPES:
        raise SVMScopeError(
            f"the resolved allow-list names {tuple(resolved)}, §G.2a freezes "
            f"{reg.SVM_FOLD_FAILED_ALLOWED_TYPES}")
    types_ = tuple(resolved.values())
    if len(set(types_)) != reg.SVM_FOLD_FAILED_N_ALLOWED_TYPES:
        raise SVMScopeError(
            "§G.2a registers exactly "
            f"{reg.SVM_FOLD_FAILED_N_ALLOWED_TYPES} distinct permitted types")
    return types_


#: The two live class objects, resolved once at import. `LinAlgError` is a
#: subclass of `ValueError` in the installed numpy, which is precisely why
#: membership below is by exact type identity and never by `isinstance`.
PERMITTED_ESTIMATOR_FAILURES = _permitted_failure_types()


def is_permitted_estimator_failure(exc: BaseException) -> bool:
    """§G.2a condition (1): membership by **exact type identity**.

    `type(exc) is X`, never `isinstance(exc, X)`, so no subclass of a listed
    type can widen the list. This is not pedantry: `numpy.linalg.LinAlgError`
    subclasses `ValueError`, and an arbitrary `ValueError` is expressly
    excluded — only exact-type matching keeps both statements true at once.
    """
    return any(type(exc) is permitted            # noqa: E721 — deliberate
               for permitted in PERMITTED_ESTIMATOR_FAILURES)


def fit_predict(X_tr, y_tr, w_tr, X_te, C: float) -> np.ndarray:
    """Fit one SVM under the frozen settings and return hard labels on `X_te`.

    `sample_weight` is **mandatory**: unweighted fitting is forbidden as a silent
    fallback (§F.8). The decision rule is `decision_function(x) > 0`, the SVM
    analogue of the frozen `0.5` probability threshold; no other cut-point is
    considered.

    A convergence warning raises `SVMNotConverged`, which the fold turns into
    `SVM_NOT_CONVERGED` and a `NOT_ESTIMABLE` BAcc. A non-converged fit is never
    reported as a result, and `max_iter` is never raised after seeing a result.

    **The §G.2a estimator/backend boundary is exactly the two calls below** —
    the frozen `LinearSVC.fit(...)` for one candidate and the
    `decision_function(...)` on that same fitted estimator, and nothing else.
    The weight check, the constructor, the probability-path assertion and every
    caller-side step are **outside** it: the same exception type raised there is
    not `SVM_FOLD_FAILED` and still fails hard. Nothing here catches `Exception`,
    and no other exception type is converted into an estimability code.

    When a fit both warns and then raises a permitted failure, the **warning
    wins**: §H.3.1 case 2 is "any inner fit for this candidate emits a
    `ConvergenceWarning`", and §G.2a states that `SVM_FOLD_FAILED` is never a
    substitute for the more specific `SVM_NOT_CONVERGED`.
    """
    if w_tr is None:
        raise SVMScopeError(
            "an SVM fit was attempted without sample_weight; site-balanced "
            "training weights are mandatory (§F.8)")
    estimator = make_estimator(C)
    assert_no_probability_path(estimator)

    failure = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            estimator.fit(X_tr, y_tr, sample_weight=w_tr)
        except PERMITTED_ESTIMATOR_FAILURES as exc:
            # Caught by the two registered types only. `isinstance` semantics
            # got us here; exact-type identity decides whether it may be
            # converted, and a subclass is re-raised untouched.
            if not is_permitted_estimator_failure(exc):
                raise
            failure = SVMExpectedEstimatorFailure(exc, "LinearSVC.fit")
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise SVMNotConverged(f"liblinear did not converge at C={C}")
    if failure is not None:
        raise failure

    try:
        margin = estimator.decision_function(X_te)
    except PERMITTED_ESTIMATOR_FAILURES as exc:
        if not is_permitted_estimator_failure(exc):
            raise
        raise SVMExpectedEstimatorFailure(exc, "decision_function") from exc
    return (margin > 0).astype(int)


def _pca_label(pca):
    """The producer's one spelling of a PCA candidate: `None -> "no-PCA"`."""
    return "no-PCA" if pca is None else int(pca)


def _candidate_row(held_out: str, pca, C: float, *, score=None, reason=None
                   ) -> dict:
    """One `svm/fold_selection.csv` row, in exactly one of the §H.3.1 states.

    Case 1 — a successful complete inner score: finite `inner_macro_bacc`,
    `ESTIMABLE`, `reason_code = null`.
    Cases 2 and 3 — the score genuinely does not exist: `inner_macro_bacc =
    null`, `NOT_ESTIMABLE`, and the registered reason that says which.

    There is no fourth spelling. An empty cell without a status — the Attempt #5
    representation — cannot be produced by this function at all, because the
    score and the state are set together or not at all.
    """
    if reason is None:
        if score is None or not np.isfinite(score):
            raise SVMScopeError(
                f"{held_out}/{_pca_label(pca)}/C={C}: an ESTIMABLE candidate "
                f"row was built with inner_macro_bacc={score!r}. §H.3.1 case 1 "
                "is a finite score; a missing one is case 2 or case 3 and "
                "needs its registered reason")
        return {"held_out_site": held_out, "pca": _pca_label(pca),
                "C": float(C), "inner_macro_bacc": float(score),
                "status": reg.ESTIMABLE, "reason_code": None}
    return {"held_out_site": held_out, "pca": _pca_label(pca), "C": float(C),
            "inner_macro_bacc": None, "status": reg.NOT_ESTIMABLE,
            "reason_code": reg.assert_reason_code_for(
                reg.SVM_FOLD_SELECTION_TABLE, reg.NOT_ESTIMABLE, reason)}


def select_candidate(scores: dict):
    """§F.7.5 — the deterministic argmax over the **complete** 20-candidate grid.

    1. highest `inner_macro_bacc`;
    2. then smaller numeric `C` — `0.01 < 0.1 < 1.0 < 10.0 < 100.0`;
    3. then stronger compression — `50`, then `100`, then `200`, then `no-PCA`.

    `no-PCA` is the *weakest* compression and is therefore **last**. The `1.0.2`
    implementation ranked by position in `SVM_PCA_CANDIDATES`, whose declaration
    order begins at `None`, so an exact cross-PCA tie silently preferred the
    uncompressed candidate — the inverse of the frozen rule. The order is read
    from `SVM_TIE_BREAK_PCA_ORDER`, which is a permutation of the candidate set
    and never a second grid.

    The precondition is §F.7.2's: this is called only when all 20 candidates are
    `ESTIMABLE` with finite scores, and it refuses to rank a partial grid rather
    than quietly selecting from one.
    """
    if len(scores) != reg.SVM_N_CANDIDATES_PER_FOLD:
        raise SVMScopeError(
            f"selection was offered {len(scores)} candidates; §F.7.2 permits "
            f"it only over the complete grid of "
            f"{reg.SVM_N_CANDIDATES_PER_FOLD}")
    if not all(np.isfinite(v) for v in scores.values()):
        raise SVMScopeError(
            "selection was offered a non-finite candidate score; §F.7.2 "
            "permits selection only when all 20 rows are ESTIMABLE with a "
            "finite score, and a reduced grid is never selected from")
    c_order = list(reg.SVM_C_GRID)
    pca_order = list(reg.SVM_TIE_BREAK_PCA_ORDER)

    def rank(key):
        pca, C = key
        return (-scores[key], c_order.index(C), pca_order.index(pca))

    return min(scores, key=rank)


def run_outer_fold(X, y, sites, held_out: str, seed: int = reg.SVM_RANDOM_STATE) -> dict:
    """One outer LOSO fold: inner CV -> joint (PCA, C) selection -> refit -> score.

    The held-out site is fully removed and is invisible to the scaler, the PCA,
    the `C` selection, the dimension selection and every fit
    (`contracts.assert_held_out_site_invisible`). Its labels are read **only** to
    score balanced accuracy after that fold's model is final — exactly as Gate 1
    does — so F-SVM is S0-equivalent in information budget and no target label
    ever influences a fitted parameter or a selected hyperparameter.

    Selection is by **site-macro balanced accuracy across the inner held-out
    source sites**, never pooled accuracy, with the frozen §F.7.5 tie-break.

    **The §F.7.1 any-fit rule, as the implementation satisfies it.** Any fit
    belonging to this fold's lifecycle — *any inner-grid fit* or the final outer
    refit — that emits a `ConvergenceWarning` invalidates the whole fold. §F.7
    draws no inner/outer distinction and admits no reduced-grid fallback, so:

    * the grid keeps being swept for **diagnostic completeness** (§F.7.2), and
      all 20 candidate rows are emitted whatever happened to each of them;
    * successful candidates keep their finite scores as an **audit trail, never
      a candidate pool** — they may not rescue or select a model for this fold;
    * **no selection** and **no final refit** occur;
    * the per-site row is the §F.7.1 representation: `NOT_ESTIMABLE` /
      `SVM_NOT_CONVERGED`, every selection field `null`, `converged = false`,
      and `n_iter_max = null` because no refit was attempted.

    The distinct §F.7.1a case — a complete valid grid, a lawful selection, and
    then a warning from the *selected final refit* — keeps the selection
    metadata it really produced and records `n_iter_max = 2000`. That retained
    metadata is provenance, never a result.

    The cohort counts `n_acquisitions`, `n_mdd`, `n_hc` are facts about the
    held-out site rather than model outputs, and are persisted in every state.
    """
    te = sites == held_out
    tr = ~te
    contracts.assert_disjoint(np.flatnonzero(tr), np.flatnonzero(te))
    X_src, y_src, s_src = X[tr], y[tr], sites[tr]
    source_sites = np.unique(s_src)
    contracts.assert_held_out_site_invisible(source_sites, held_out)

    pca_candidates, c_grid = reg.SVM_PCA_CANDIDATES, reg.SVM_C_GRID
    oof = {(p, c): np.full(len(y_src), -1, dtype=int)
           for p in pca_candidates for c in c_grid}
    #: Candidate -> the registered reason its score does not exist. A candidate
    #: that both warned and raised a permitted failure keeps `SVM_NOT_CONVERGED`:
    #: §H.3.1 case 2 is "**any** inner fit for this candidate emits a
    #: ConvergenceWarning", and §G.2a forbids `SVM_FOLD_FAILED` from ever
    #: standing in for the more specific code.
    reasons: dict = {}

    def record(key, reason: str) -> None:
        if reasons.get(key) != "SVM_NOT_CONVERGED":
            reasons[key] = reason

    cohort = {"n_acquisitions": int(te.sum()),
              "n_mdd": int((y[te] == 1).sum()),
              "n_hc": int((y[te] == 0).sum())}

    for inner in source_sites:
        iva = s_src == inner
        itr = ~iva
        contracts.assert_held_out_site_invisible(np.unique(s_src[itr]), held_out)
        contracts.assert_held_out_site_invisible(np.unique(s_src[itr]), inner)
        pre = nl.Preprocessor(pca_candidates).fit(X_src[itr], seed)
        w = nl.site_equal_weights(s_src[itr])
        for p in pca_candidates:
            Ztr = pre.transform(X_src[itr], p)
            Zva = pre.transform(X_src[iva], p)
            for c in c_grid:
                # Only the two registered candidate-level outcomes are caught.
                # Everything else — invariant, schema and programming failures —
                # propagates and aborts the Formal run (§F.7.2 rule 6, §J.3).
                try:
                    oof[(p, c)][iva] = fit_predict(Ztr, y_src[itr], w, Zva, c)
                except SVMNotConverged:
                    record((p, c), "SVM_NOT_CONVERGED")
                except SVMExpectedEstimatorFailure:
                    record((p, c), "SVM_FOLD_FAILED")

    selection_rows, scores = [], {}
    for (p, c), yhat in oof.items():
        reason = reasons.get((p, c))
        if reason is None and (yhat < 0).any():
            # Structurally unreachable: every inner site writes its own slice,
            # and a slice can be left unwritten only when that candidate's fit
            # was skipped, which is exactly what puts it in `reasons`.
            raise SVMScopeError(
                f"{held_out}/{_pca_label(p)}/C={c}: the inner OOF vector is "
                "incomplete but no candidate-level outcome was recorded")
        if reason is not None:
            selection_rows.append(_candidate_row(held_out, p, c, reason=reason))
            continue
        score = nl.macro_site_bacc(y_src, yhat, s_src)
        scores[(p, c)] = float(score)
        selection_rows.append(_candidate_row(held_out, p, c, score=score))

    if len(selection_rows) != reg.SVM_N_CANDIDATES_PER_FOLD:
        raise SVMScopeError(
            f"{held_out}: {len(selection_rows)} candidate rows, §H.3.1 makes "
            f"all {reg.SVM_N_CANDIDATES_PER_FOLD} mandatory")

    if reasons:
        # §F.7.1 — the fold is invalid. No selection, no refit, and the four
        # selection fields are `null` because no such event occurred. The fold
        # reason is the more specific code whenever a warning happened anywhere
        # in this fold's lifecycle (§G.2a).
        fold_reason = ("SVM_NOT_CONVERGED"
                       if "SVM_NOT_CONVERGED" in reasons.values()
                       else "SVM_FOLD_FAILED")
        return {"held_out_site": held_out, "status": reg.NOT_ESTIMABLE,
                "reason_code": reg.assert_reason_code_for(
                    reg.SVM_PER_SITE_TABLE, reg.NOT_ESTIMABLE, fold_reason),
                "bacc": None, "selected_pca": None, "selected_C": None,
                "inner_macro_bacc": None, "converged": False,
                "n_iter_max": reg.SVM_N_ITER_MAX_WHEN_REFIT_SKIPPED,
                **cohort, "_selection": selection_rows}

    best_pca, best_c = select_candidate(scores)
    selected = {"selected_pca": _pca_label(best_pca),
                "selected_C": float(best_c),
                "inner_macro_bacc": float(scores[(best_pca, best_c)])}

    pre = nl.Preprocessor((best_pca,)).fit(X_src, seed)
    w = nl.site_equal_weights(s_src)
    try:
        yhat_te = fit_predict(pre.transform(X_src, best_pca), y_src, w,
                              pre.transform(X[te], best_pca), best_c)
    except (SVMNotConverged, SVMExpectedEstimatorFailure) as exc:
        # §F.7.1a — the grid was complete, the selection was lawful, and only
        # the selected final refit failed. That evidence exists and is true, so
        # it is retained rather than erased; `n_iter_max` is the configured cap
        # because the refit **was** attempted. `bacc` does not exist, the fold
        # contributes nothing to the headline, and the retained metadata may
        # never report a number or re-refit the fold.
        refit_reason = ("SVM_NOT_CONVERGED" if isinstance(exc, SVMNotConverged)
                        else "SVM_FOLD_FAILED")
        return {"held_out_site": held_out, "status": reg.NOT_ESTIMABLE,
                "reason_code": reg.assert_reason_code_for(
                    reg.SVM_PER_SITE_TABLE, reg.NOT_ESTIMABLE, refit_reason),
                "bacc": None, **selected, "converged": False,
                "n_iter_max": reg.SVM_N_ITER_MAX_WHEN_REFIT_ATTEMPTED,
                **cohort, "_selection": selection_rows}

    y_te = y[te]
    return {"held_out_site": held_out, **cohort,
            "bacc": float(balanced_accuracy_score(y_te, yhat_te)),
            **selected,
            "n_iter_max": reg.SVM_N_ITER_MAX_WHEN_REFIT_ATTEMPTED,
            # §F.7.3 — `converged = true` is a fold-lifecycle statement: every
            # inner-grid fit across all 20 candidates was clean (`reasons` is
            # empty), the selected final refit was clean (we are past its
            # handler), and the fold is ESTIMABLE. It is never set merely
            # because the last fit succeeded.
            "converged": True,
            "status": reg.ESTIMABLE, "reason_code": None,
            "_selection": selection_rows}


def site_macro_bacc(per_site: list) -> dict:
    """`svm/site_macro_bacc.json` — the single headline number.

    The mean is taken over sites with `status = ESTIMABLE`, and
    `n_sites_not_estimable` is always reported beside it. If fewer than 23 sites
    are estimable the headline number is itself `NOT_ESTIMABLE`: the one number
    may not be quietly computed over a reduced site set (§H.3).
    """
    estimable = [r for r in per_site if r["status"] == reg.ESTIMABLE]
    not_estimable = len(per_site) - len(estimable)
    base = {"estimand": reg.SVM_ESTIMAND,
            "n_sites_contributing": len(estimable),
            "n_sites_not_estimable": not_estimable,
            "carries_confidence_interval": False,
            "comparison_to_logistic_regression": "NONE — no significance test "
                                                 "and no superiority claim (§F.9)"}
    if not_estimable or len(estimable) != reg.BASE_COHORT_SITES:
        return {**base, "site_macro_bacc": None, "status": reg.NOT_ESTIMABLE,
                "reason_code": reg.assert_reason_code("SVM_FOLD_FAILED")}
    return {**base,
            "site_macro_bacc": float(np.mean([r["bacc"] for r in estimable])),
            "status": reg.ESTIMABLE, "reason_code": None}


def svm_config() -> dict:
    """`svm/svm_config.json` — every §F.7 parameter, each one explicit."""
    return {
        "protocol_version": reg.PROTOCOL_VERSION,
        "protocol_tag": reg.PROTOCOL_TAG,
        "estimator": reg.SVM_ESTIMATOR,
        "estimand": reg.SVM_ESTIMAND,
        "penalty": reg.SVM_PENALTY,
        "loss": reg.SVM_LOSS,
        "dual": reg.SVM_DUAL,
        "class_weight": reg.SVM_CLASS_WEIGHT,
        "fit_intercept": reg.SVM_FIT_INTERCEPT,
        "tol": reg.SVM_TOL,
        "max_iter": reg.SVM_MAX_ITER,
        "random_state": reg.SVM_RANDOM_STATE,
        "random_state_status": reg.SVM_RANDOM_STATE_STATUS,
        "c_grid": list(reg.SVM_C_GRID),
        "pca_candidates": list(reg.SVM_PCA_CANDIDATES),
        "decision_rule": reg.SVM_DECISION_RULE,
        "sample_weight": reg.SVM_SAMPLE_WEIGHT,
        "probability_calibration": reg.SVM_PROBABILITY_CALIBRATION,
        "platt_scaling": reg.SVM_PLATT_SCALING,
        "c_scale_equivalence_claimed": reg.SVM_C_SCALE_EQUIVALENCE_CLAIMED,
        "forbidden_outputs": list(reg.SVM_FORBIDDEN_OUTPUTS),
        "inner_cv": "leave-one-source-site-out over the 22 source sites",
        "tuning_metric": "site-macro balanced accuracy across inner held-out "
                         "source sites (nested_loso.macro_site_bacc)",
        "tie_break": "highest inner macro site-BAcc, then smaller C, then "
                     "stronger compression",
        # §F.7.5 written out: "stronger compression" needs an unambiguous order
        # for a mechanical check, and `no-PCA` is the weakest and therefore last.
        "tie_break_order": list(reg.SVM_TIE_BREAK_ORDER),
        "tie_break_pca_order": [_pca_label(p)
                                for p in reg.SVM_TIE_BREAK_PCA_ORDER],
        # §F.7.1 / §F.7.2 — the fail-closed rule and the diagnostic policy, as
        # persisted facts about the run rather than as prose in this module.
        "any_fit_convergence_warning_invalidates_outer_fold":
            reg.SVM_ANY_FIT_WARNING_INVALIDATES_FOLD,
        "inner_fit_counts_as_fold_lifecycle_fit":
            reg.SVM_INNER_FIT_IS_FOLD_LIFECYCLE_FIT,
        "reduced_grid_selection_permitted":
            reg.SVM_REDUCED_GRID_SELECTION_PERMITTED,
        "final_refit_after_inner_warning_permitted":
            reg.SVM_FINAL_REFIT_AFTER_INNER_WARNING_PERMITTED,
        "selection_requires_all_candidates_estimable":
            reg.SVM_SELECTION_REQUIRES_ALL_CANDIDATES_ESTIMABLE,
        "n_candidates_per_fold": reg.SVM_N_CANDIDATES_PER_FOLD,
        # §F.7.4 — the configured cap, never an observed `n_iter_`.
        "n_iter_max_is_configured_cap": reg.SVM_N_ITER_MAX_IS_CONFIGURED_CAP,
        "observed_iteration_count_persisted":
            reg.SVM_OBSERVED_ITERATION_COUNT_PERSISTED,
        # §G.2a — the closed allow-list and the boundary it must be raised in.
        "svm_fold_failed_allowed_types":
            list(reg.SVM_FOLD_FAILED_ALLOWED_TYPES),
        "svm_fold_failed_type_match_rule": reg.SVM_FOLD_FAILED_TYPE_MATCH_RULE,
        "svm_fold_failed_boundary": reg.SVM_FOLD_FAILED_BOUNDARY_DESCRIPTION,
        "consumed_inputs": {p: d for p, d in sorted(reg.CONSUMED_ARTIFACTS.items())},
    }


def run(X=None, y=None, sites=None, *, n_jobs: int = 1, progress=None) -> dict:
    """All 23 outer folds. Returns the four `svm/` artifacts in memory."""
    def say(message):
        if progress is not None:
            progress(message)

    assert_grids_match_frozen_pipeline()
    if X is None or y is None or sites is None:
        import fc_features
        X, y, sites, _ = fc_features.build_xy(verbose=False)
    contracts.assert_feature_matrix(np.asarray(X), len(y))

    order = [s for s in contracts.FROZEN_SITES if s in set(map(str, sites))]
    if n_jobs == 1:
        folds = []
        for site in order:
            say(f"svm outer fold {site}")
            folds.append(run_outer_fold(X, y, sites, site))
    else:
        from joblib import Parallel, delayed
        folds = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_outer_fold)(X, y, sites, site) for site in order)

    selection = [row for fold in folds for row in fold["_selection"]]
    per_site = [{k: v for k, v in fold.items() if k != "_selection"}
                for fold in folds]
    expected = len(order) * reg.SVM_N_CANDIDATES_PER_FOLD
    if len(selection) != expected:
        raise SVMScopeError(
            f"fold_selection holds {len(selection)} rows, the registry expects "
            f"{expected}")
    # §F.7.2 / §H.3.1 — every one of the 460 rows carries a state, in every
    # fold, whether or not that fold survived. A blank score with no status is
    # the Attempt #5 spelling and is unrepresentable here.
    stateless = [row for row in selection
                 if row["status"] not in reg.STATUS_VALUES]
    if stateless:
        raise SVMScopeError(
            f"{len(stateless)} candidate rows carry no registered status; "
            "§H.3.1 gives a non-existent score exactly one lawful spelling")
    return {"per_site_bacc": per_site, "site_macro_bacc": site_macro_bacc(per_site),
            "fold_selection": selection, "svm_config": svm_config()}
