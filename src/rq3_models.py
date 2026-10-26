"""The frozen RQ3 model families and LOSO-site cross-validation.

`RQ3_PROTOCOL.md` §6 (full-sample, primary), §7 (MDD-only, secondary),
§10 (leave-one-site-out CV), §11 (the McFadden null).

The registries below are the whole model specification: a model is a name, a
population and an ordered tuple of fixed-effect terms. `(1 | site)` is present
in every one of them and is never a listed term, because it is never optional —
§6.2 forbids removing it to form a comparison, and §12.2 forbids testing it.

Cross-validation is leave-one-**site**-out. Subject-level random k-fold is
forbidden (§10.1): the site is the inference unit throughout the project, and
subject-level folds would leak site-specific intercept information across the
split. The fold worker below is handed the training rows only, so the held-out
site cannot reach the scaler, the initializer or the likelihood; the held-out
outcome is touched exactly once, as the target of the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import rq3_data as data
import rq3_glmm as glmm
import rq3_protocol as proto

NOT_ESTIMABLE = proto.NOT_ESTIMABLE


class RQ3ModelError(AssertionError):
    """Structural corruption of a model family, a fold or a comparison."""


@dataclass(frozen=True)
class ModelSpec:
    """One frozen model. `role` records what it is for, never what it may become."""

    name: str
    population: str
    terms: tuple
    role: str          # "full" | "reduced" | "backbone" | "null"

    @property
    def n_fixed_effects(self) -> int:
        return len(self.terms) + 1     # + intercept

    @property
    def formula(self) -> str:
        right = " + ".join(self.terms) if self.terms else "1"
        return f"R ~ {right} + (1 | site)"


# --------------------------------------------------------------------- §6
# Full-sample family, fitted on the same 2222-row subset. `M0_full` is a
# declared backbone (§6.3): it fixes the demographic baseline that both reduced
# models extend, appears in no registered output of §15, and may not be promoted
# into one or used as a McFadden reference — the reference is §11's null, a
# different object. It is therefore registered here and **not fitted**.
FULL_SAMPLE_FAMILY = (
    ModelSpec("M0_full", data.FULL_SAMPLE, ("age", "sex"), "backbone"),
    ModelSpec("M_motion_full", data.FULL_SAMPLE, ("age", "sex", "mean_fd"), "reduced"),
    ModelSpec("M_clin_full", data.FULL_SAMPLE, ("age", "sex", "diagnosis"), "reduced"),
    ModelSpec("M_full", data.FULL_SAMPLE, ("age", "sex", "mean_fd", "diagnosis"), "full"),
)

#: §11 — `R ~ 1 + (1 | site)`, identical across all six cells and both
#: operating points, re-fitted per cell because `R_i` differs per cell (F4).
NULL_FULL = ModelSpec("M_null_full", data.FULL_SAMPLE, (), "null")

# --------------------------------------------------------------------- §7
# MDD-only family, fitted on the 484-row subset only. §7.1: `M0_mdd` coincides
# in form with the §11 McFadden null for this population, and is fitted once and
# used for both purposes.
MDD_ONLY_FAMILY = (
    ModelSpec("M0_mdd", data.MDD_ONLY, (), "null"),
    ModelSpec("M_motion_mdd", data.MDD_ONLY, ("mean_fd",), "reduced"),
    ModelSpec("M_clin_mdd", data.MDD_ONLY,
              ("hamd", "first_episode", "on_medication", "illness_duration_months"),
              "reduced"),
    ModelSpec("M_full_mdd", data.MDD_ONLY,
              ("mean_fd", "hamd", "first_episode", "on_medication",
               "illness_duration_months"),
              "full"),
)
NULL_MDD = MDD_ONLY_FAMILY[0]

MODEL_FAMILIES = {
    data.FULL_SAMPLE: FULL_SAMPLE_FAMILY,
    data.MDD_ONLY: MDD_ONLY_FAMILY,
}
NULL_MODEL = {data.FULL_SAMPLE: NULL_FULL, data.MDD_ONLY: NULL_MDD}


@dataclass(frozen=True)
class BlockSpec:
    """One registered block comparison (§6.1, §7.1).

    `df` is frozen in `BLOCK_DF` and equals the number of added terms; both are
    checked, because a df that merely *happens* to match today would drift
    silently.
    """

    population: str
    block: str
    full_model: str
    reduced_model: str
    added_terms: tuple
    df: int


BLOCKS = (
    BlockSpec(data.FULL_SAMPLE, "motion", "M_full", "M_clin_full", ("mean_fd",), 1),
    BlockSpec(data.FULL_SAMPLE, "diagnosis", "M_full", "M_motion_full", ("diagnosis",), 1),
    BlockSpec(data.MDD_ONLY, "motion", "M_full_mdd", "M_clin_mdd", ("mean_fd",), 1),
    BlockSpec(data.MDD_ONLY, "clinical", "M_full_mdd", "M_motion_mdd",
              ("hamd", "first_episode", "on_medication", "illness_duration_months"), 4),
)


def model_by_name(population: str, name: str) -> ModelSpec:
    for spec in MODEL_FAMILIES[population] + (NULL_MODEL[population],):
        if spec.name == name:
            return spec
    raise RQ3ModelError(f"{name!r} is not a registered model of {population!r}")


def blocks_for(population: str) -> tuple:
    return tuple(b for b in BLOCKS if b.population == population)


def models_to_fit(population: str) -> tuple:
    """The models a registered output actually needs, in a deterministic order.

    The §6.3 backbone is excluded: fitting it would create a record no output
    may read, and §6.3 forbids promoting it into one.
    """
    needed = {NULL_MODEL[population].name: NULL_MODEL[population]}
    for block in blocks_for(population):
        for name in (block.full_model, block.reduced_model):
            needed[name] = model_by_name(population, name)
    return tuple(sorted(needed.values(), key=lambda spec: spec.name))


def models_to_cross_validate(population: str) -> tuple:
    """Only the models a Delta CV log-loss needs (§10.4): full and both reduced."""
    needed = {}
    for block in blocks_for(population):
        for name in (block.full_model, block.reduced_model):
            needed[name] = model_by_name(population, name)
    return tuple(sorted(needed.values(), key=lambda spec: spec.name))


def assert_family_is_frozen() -> None:
    """Guard the registries against silent edits (§6, §7, §12.1)."""
    expected_terms = {
        "M0_full": ("age", "sex"),
        "M_motion_full": ("age", "sex", "mean_fd"),
        "M_clin_full": ("age", "sex", "diagnosis"),
        "M_full": ("age", "sex", "mean_fd", "diagnosis"),
        "M_null_full": (),
        "M0_mdd": (),
        "M_motion_mdd": ("mean_fd",),
        "M_clin_mdd": ("hamd", "first_episode", "on_medication", "illness_duration_months"),
        "M_full_mdd": ("mean_fd", "hamd", "first_episode", "on_medication",
                       "illness_duration_months"),
    }
    registered = {}
    for population in data.POPULATIONS:
        for spec in MODEL_FAMILIES[population] + (NULL_MODEL[population],):
            registered[spec.name] = spec.terms
            data.assert_no_identity_in_design(spec.terms)
            if len(set(spec.terms)) != len(spec.terms):
                raise RQ3ModelError(f"{spec.name} repeats a term")
    if registered != expected_terms:
        raise RQ3ModelError(f"the frozen model families changed: {registered}")

    for block in BLOCKS:
        full = model_by_name(block.population, block.full_model)
        reduced = model_by_name(block.population, block.reduced_model)
        if set(reduced.terms) - set(full.terms):
            raise RQ3ModelError(
                f"{block.reduced_model} is not nested in {block.full_model}"
            )
        added = tuple(t for t in full.terms if t not in set(reduced.terms))
        if added != block.added_terms:
            raise RQ3ModelError(
                f"{block.population}/{block.block} adds {added}, registered {block.added_terms}"
            )
        if block.df != len(block.added_terms):
            raise RQ3ModelError(f"{block.population}/{block.block} df != added-term count")
        frozen_df = proto.BLOCK_DF[(block.population, block.block)]
        if block.df != frozen_df:
            raise RQ3ModelError(
                f"{block.population}/{block.block} df {block.df} != frozen {frozen_df}"
            )
    if len(BLOCKS) != len(proto.BLOCK_DF):
        raise RQ3ModelError("the block registry does not match the frozen df table")


assert_family_is_frozen()


# ---------------------------------------------------------------------------
# full-data inferential fits (§8.1, §11, §12, §13)
# ---------------------------------------------------------------------------

def family_transform(population: data.Population, models=None) -> data.TransformFit:
    """One transform per population family, shared by every model in it (§8.1).

    Estimated once on that population's fixed complete-case subset, so nested
    models differ only in their fixed-effect terms and never in their scaling.
    """
    specs = MODEL_FAMILIES[population.name] if models is None else models
    terms = sorted({t for spec in specs for t in spec.terms})
    return data.fit_transform(population.frame, terms, fitted_on=population.name)


def fit_full_data_model(spec: ModelSpec, population: data.Population,
                        outcome: data.CellOutcome, transform: data.TransformFit,
                        n_nodes: int) -> glmm.GLMMFit:
    """Fit one model on the whole frozen complete-case subset of its population."""
    if spec.population != population.name:
        raise RQ3ModelError(f"{spec.name} belongs to {spec.population}, not {population.name}")
    if outcome.population != population.name:
        raise RQ3ModelError("outcome and population disagree")
    if len(outcome.R) != population.n:
        raise RQ3ModelError(
            f"H8: outcome has {len(outcome.R)} rows, population has {population.n}"
        )
    model_transform = data.TransformFit(
        terms=spec.terms,
        location={k: v for k, v in transform.location.items() if k in spec.terms},
        scale={k: v for k, v in transform.scale.items() if k in spec.terms},
        fitted_on=transform.fitted_on,
        n_fitting_rows=transform.n_fitting_rows,
    )
    X = data.design_matrix(population.frame, model_transform)
    return glmm.fit_glmm(X, outcome.R, population.frame.site, n_nodes)


# ---------------------------------------------------------------------------
# §10 leave-one-site-out cross-validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LosoFold:
    """One held-out site. The training and held-out partitions are disjoint.

    The held-out outcome `test_R` is carried here so it can be the target of the
    loss and nothing else: `fit_fold` never receives it (§10.3, H9, H17).
    """

    held_out_site: str
    train_frame: pd.DataFrame
    train_R: np.ndarray
    test_frame: pd.DataFrame
    test_R: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_frame)

    @property
    def n_test(self) -> int:
        return len(self.test_frame)


def loso_folds(population: data.Population, outcome: data.CellOutcome) -> tuple:
    """§10.1 — one fold per site present in the frozen complete-case subset."""
    if len(outcome.R) != population.n:
        raise RQ3ModelError("outcome and population row counts disagree")
    frame = population.frame
    folds = []
    for site in population.sites:
        held_out = (frame.site == site).to_numpy()
        train_frame = frame[~held_out].reset_index(drop=True)
        test_frame = frame[held_out].reset_index(drop=True)
        if train_frame.site.nunique() < 2:
            raise RQ3ModelError(f"§10.5: fold {site} leaves fewer than 2 training sites")
        if len(test_frame) == 0:
            raise RQ3ModelError(f"fold {site} holds out no rows")
        if set(train_frame.canonical_id) & set(test_frame.canonical_id):
            raise RQ3ModelError(f"H9: fold {site} partitions overlap")
        folds.append(LosoFold(
            held_out_site=site,
            train_frame=train_frame, train_R=outcome.R[~held_out],
            test_frame=test_frame, test_R=outcome.R[held_out],
        ))
    if len(folds) != population.n_sites:
        raise RQ3ModelError(f"{len(folds)} folds for {population.n_sites} sites")
    return tuple(folds)


def fit_fold(spec: ModelSpec, held_out_site: str, train_frame: pd.DataFrame,
             train_R: np.ndarray, n_nodes: int) -> tuple:
    """Fit one fold on the training sites alone (§10.2 steps 1-2).

    The signature is the firewall: this function is handed the training frame
    and the training outcome and has no reference to the held-out rows at all,
    so neither the scaler nor the initializer nor the likelihood can reach them.
    """
    if held_out_site in set(train_frame.site):
        raise RQ3ModelError(
            f"H9: held-out site {held_out_site} is present in its own training partition"
        )
    transform = data.fit_transform(
        train_frame, spec.terms, fitted_on=f"train_without_{held_out_site}"
    )
    X = data.design_matrix(train_frame, transform)
    fit = glmm.fit_glmm(X, train_R, train_frame.site, n_nodes)
    return fit, transform


def score_fold(fit: glmm.GLMMFit, transform: data.TransformFit, fold: LosoFold,
               n_nodes: int) -> float:
    """§10.3-§10.4 — the held-out site's mean marginalized log-loss.

    The training transform is applied unchanged, the held-out random intercept
    is integrated out under §9.2.3 rather than estimated, and `test_R` enters
    only as the target of the loss.
    """
    fit.require()
    X = data.design_matrix(fold.test_frame, transform)
    eta = X @ np.asarray(fit.beta, dtype=float)
    losses = glmm.marginal_log_loss(fold.test_R, eta, fit.sigma_site, n_nodes)
    return float(np.mean(losses))


@dataclass(frozen=True)
class CVResult:
    """Cross-validated log-loss for one model, one cell, one node count."""

    model: str
    population: str
    n_nodes: int
    status: str
    macro_log_loss: object = NOT_ESTIMABLE
    folds: tuple = ()
    failing_condition: str | None = None
    n_registered_folds: int = 0
    unestimable_folds: tuple = ()

    @property
    def estimable(self) -> bool:
        return self.status == proto.STATUS_OK

    def to_payload(self) -> dict:
        return {
            "model": self.model,
            "population": self.population,
            "n_nodes": int(self.n_nodes),
            "status": self.status,
            "aggregation_policy": proto.CV_AGGREGATION_POLICY,
            "n_registered_folds": int(self.n_registered_folds),
            "macro_log_loss": (self.macro_log_loss if isinstance(self.macro_log_loss, str)
                               else float(self.macro_log_loss)),
            "failing_condition": self.failing_condition,
            "n_folds": len(self.folds),
            "unestimable_folds": list(self.unestimable_folds),
            "folds": [dict(f) for f in self.folds],
        }


def cross_validate(spec: ModelSpec, population: data.Population,
                   outcome: data.CellOutcome, n_nodes: int) -> CVResult:
    """§10 — LOSO-site CV, macro-averaged over held-out sites.

    Every site weighs the same regardless of size, because the site — not the
    subject — is the inference unit (§10.4). The consequence flagged as F2, that
    the 2-subject S11 fold of the MDD-only population carries the weight of the
    236-subject S20 fold, is recorded rather than corrected.

    **§10.6 — a CV quantity is all-or-nothing.** The registered value is the
    macro-average over **all** registered folds (23 full-sample, 9 MDD-only). If
    any one of them is `NOT_ESTIMABLE`, the quantity is `NOT_ESTIMABLE` with
    `CV_FOLD_NOT_ESTIMABLE`, and every Δ CV log-loss depending on it in turn.
    Averaging over the surviving folds is forbidden (H28): the fold set is part
    of the estimand, and which sites survive would itself depend on the fitting
    behaviour the quantity is meant to summarise. No fold is dropped, re-run
    under different settings, replaced, imputed, back-filled from another node
    count, or reweighted.

    The verdict is delegated to the frozen `cv_logloss_status`, so "all folds
    estimable" is decided in exactly one place — and an empty fold set is a
    failure there, never a vacuous pass.
    """
    folds = loso_folds(population, outcome)
    records, per_fold = [], []
    for fold in folds:
        fit, transform = fit_fold(
            spec, fold.held_out_site, fold.train_frame, fold.train_R, n_nodes
        )
        condition = None if fit.estimable else fit.failing_condition
        site_loss = None
        if fit.estimable:
            try:
                site_loss = score_fold(fit, transform, fold, n_nodes)
            except glmm.AdaptiveQuadratureFailure as failure:
                # §10.3 evaluates each prediction branch by §9.2.3; a mode the
                # frozen solver cannot find is a registered numerical failure of
                # this fold, never a silently substituted rule (§9.2.5, H22)
                condition = failure.code
            else:
                if not np.isfinite(site_loss):
                    condition, site_loss = glmm.COND_OBJECTIVE_FINITE, None

        if condition is not None:
            records.append({
                "held_out_site": fold.held_out_site,
                "status": NOT_ESTIMABLE,
                "failing_condition": condition,
                "n_train": fold.n_train, "n_test": fold.n_test,
                "site_mean_log_loss": NOT_ESTIMABLE,
                "selected_start_index": fit.selected_start_index,
            })
            continue

        per_fold.append(site_loss)
        records.append({
            "held_out_site": fold.held_out_site,
            "status": proto.STATUS_OK,
            "failing_condition": None,
            "n_train": fold.n_train, "n_test": fold.n_test,
            "site_mean_log_loss": float(site_loss),
            "selected_start_index": int(fit.selected_start_index),
            "loglik": float(fit.loglik),
            "sigma_site": float(fit.sigma_site),
        })

    common = {
        "model": spec.name, "population": population.name, "n_nodes": n_nodes,
        "folds": tuple(records), "n_registered_folds": len(folds),
        "unestimable_folds": tuple(r["held_out_site"] for r in records
                                   if r["status"] == NOT_ESTIMABLE),
    }
    # the frozen §10.6 verdict — all registered folds, or nothing
    status = proto.cv_logloss_status(r["status"] for r in records)
    if status != proto.STATUS_OK:
        return CVResult(status=NOT_ESTIMABLE, macro_log_loss=NOT_ESTIMABLE,
                        failing_condition=status, **common)

    if len(per_fold) != len(folds):
        raise RQ3ModelError(
            f"H28: {len(per_fold)} fold losses for {len(folds)} registered folds"
        )
    macro = float(np.mean(per_fold))
    if not np.isfinite(macro):
        return CVResult(status=NOT_ESTIMABLE, macro_log_loss=NOT_ESTIMABLE,
                        failing_condition=glmm.COND_OBJECTIVE_FINITE, **common)
    return CVResult(status=proto.STATUS_OK, macro_log_loss=macro, **common)
