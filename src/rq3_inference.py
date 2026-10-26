"""RQ3 registered inferential outputs — `RQ3_PROTOCOL.md` §9.6, §11, §12, §13, §15.

One cell is executed twice, independently, at `K = 40` and `K = 60` (§9.6.1
steps 1-3). The **40-node solution is the scientific, authoritative one**; the
60-node solution is numerical validation, produces no reported row, is not a
sensitivity analysis, and may never replace the 40-node result — not on
disagreement, not on a larger likelihood, not on any other ground
(§9.6.1 steps 4-5).

The registered quantities assembled here are exactly those of §15 and no others:
site-balanced rejected fraction, Delta CV log-loss, marginal log-likelihood,
McFadden pseudo-R2 against the §11 random-intercept-only null, block-wise
Delta McFadden, fixed-effect block LRTs at their frozen dfs, Holm-adjusted
p-values over four fixed families of six slots, and the site random-intercept
variance.

`sigma_site^2` is **site-associated acquisition heterogeneity remaining after
the registered fixed effects**, and nothing else (§13).

### The unit at which the 40-vs-60 validation is applied

§9.6.3 registers seven criteria: marginal log-likelihood, LRT statistic, LRT
p-value, McFadden R2/DeltaR2, `sigma_site^2`, CV log-loss/Delta CV log-loss, and
the fixed-effect coefficient sup-norm. Four of those are properties of a *block
comparison* rather than of a single fitted model, so the smallest unit at which
all seven simultaneously exist is one **(population, cell, block)** triple.
**§9.6.5 freezes exactly that unit** — 24 of them, 2 populations x 6 cells x 2
blocks — and forbids splitting, merging, skipping or re-granularizing it (H27).
Each block's solution record therefore carries its own LRT, Delta McFadden and
Delta CV log-loss, together with the log-likelihood, `sigma_site` and
coefficient vector of the block's **full** model. A unit whose seven-of-seven
verdict fails makes **its block** `NOT_ESTIMABLE` with `QUADRATURE_DISAGREEMENT`
or `VALIDATION_MALFORMED`; the cell's other block is a separate unit with its
own verdict, and the 60-node numbers are never substituted (§9.6.1 step 5).

### Status codes

Every recorded outcome carries exactly one code from the closed §9.5.1 registry.
No code is invented here and none is reused for a different meaning: the
free-text explanation of *which* model or fold failed lives in a separate
`failure_detail` field, never in the code itself. Every registered code is a
**numerical** statement — it never means "no association", "no effect" or "the
null was supported", and a slot carrying one may not be described, plotted,
tabulated or summarised as a null result (§9.5.1, H26, F8, F11).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2

import rq3_data as data
import rq3_glmm as glmm
import rq3_models as models
import rq3_protocol as proto

NOT_ESTIMABLE = proto.NOT_ESTIMABLE

#: §9.5.1 registered codes this module can emit. All of them come from the
#: frozen registry; none is invented here.
CODE_QUADRATURE = glmm.COND_QUADRATURE                  # QUADRATURE_DISAGREEMENT
CODE_VALIDATION_MALFORMED = "VALIDATION_MALFORMED"
CODE_INVERSION = proto.NEGATIVE_LRT_POLICY              # NESTED_LIKELIHOOD_INVERSION
CODE_CV = "CV_FOLD_NOT_ESTIMABLE"


class RQ3InferenceError(AssertionError):
    """Structural corruption of an inferential assembly. Never NOT_ESTIMABLE.

    §9.5.1 keeps the two layers apart deliberately: a numerical failure is a
    registered outcome that a result file may carry, while corruption aborts the
    run so that a corrupted run cannot present itself as a scientifically
    inconclusive one.
    """


def _registered(code: str) -> str:
    """Refuse to emit anything outside the closed §9.5.1 registry."""
    if code not in proto.STATUS_CODES:
        raise RQ3InferenceError(f"§9.5.1: {code!r} is not a registered status code")
    return code


# ---------------------------------------------------------------------------
# one cell at one node count
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellFits:
    """Every fit and CV result a cell needs, at one node count."""

    population: str
    operating_point: str
    method: str
    n_nodes: int
    fits: dict
    cross_validations: dict
    transform: data.TransformFit

    @property
    def cell_index(self) -> int:
        return proto.CELL_REGISTRY.index((self.operating_point, self.method))


def run_cell(population: data.Population, outcome: data.CellOutcome,
             n_nodes: int, progress=None) -> CellFits:
    """Fit every registered model of a cell, and cross-validate those §10.4 needs.

    Nothing here selects, drops or reorders a model on the basis of a result.
    """
    if outcome.population != population.name:
        raise RQ3InferenceError("outcome and population disagree")
    specs = models.models_to_fit(population.name)
    transform = models.family_transform(population, specs)

    fits, cvs = {}, {}
    for spec in specs:
        if progress is not None:
            progress(f"fit {spec.name} K={n_nodes}")
        fits[spec.name] = models.fit_full_data_model(
            spec, population, outcome, transform, n_nodes
        )
    # H8 — every model of the family was fitted on the same rows, in one order
    row_counts = {spec.name: fits[spec.name].n_obs for spec in specs}
    if set(row_counts.values()) != {population.n}:
        raise RQ3InferenceError(
            f"H8: models were fitted on different row counts: {row_counts}"
        )

    for spec in models.models_to_cross_validate(population.name):
        if progress is not None:
            progress(f"cv {spec.name} K={n_nodes}")
        cvs[spec.name] = models.cross_validate(spec, population, outcome, n_nodes)

    return CellFits(
        population=population.name, operating_point=outcome.operating_point,
        method=outcome.method, n_nodes=n_nodes, fits=fits,
        cross_validations=cvs, transform=transform,
    )


# ---------------------------------------------------------------------------
# §11 McFadden and §12.1 LRT
# ---------------------------------------------------------------------------

def mcfadden_r2(model_loglik: float, null_loglik: float) -> float:
    """§11 — `1 - logLik(model)/logLik(null)`, both **marginal** (§9.1).

    Conditional likelihood is forbidden (§11); both arguments come from the
    marginal ML fit that integrates over the site random intercept.

    A marginal Bernoulli log-likelihood is a sum of `log p` over a finite,
    non-degenerate outcome (H5 fixes the frozen operating points, so both
    classes are present in every registered cell): each term is strictly
    negative and finite, so the null log-likelihood is strictly negative on any
    valid formal path. A non-negative value is therefore **structural or
    numerical corruption**, not a scientific outcome — §9.5.1 has no code for it
    and forbids inventing one, so this aborts rather than being reported.
    """
    null = float(null_loglik)
    if not np.isfinite(null) or null >= 0.0:
        raise RQ3InferenceError(f"§11: the null log-likelihood is {null!r}")
    return 1.0 - float(model_loglik) / null


def lrt(full_loglik: float, reduced_loglik: float, df: int) -> tuple:
    """§12.1 — `2(logLik(full) - logLik(reduced)) ~ chi2_df`.

    The random-intercept structure is identical on both sides, so the registered
    LRT compares fixed-effect blocks only and the standard chi-square reference
    applies with no boundary correction. No test of `sigma_site = 0` is ever
    performed (§12.2).

    The statistic is returned **exactly as computed**. §12.1.1 forbids clamping,
    flooring, truncating at zero or taking an absolute value (H29): a negative
    statistic is a numerical statement about the optimizer or the quadrature and
    is handled by the caller as `NESTED_LIKELIHOOD_INVERSION`, never smoothed
    into a `p = 1.0` that would assert "tested, no evidence against the null"
    for a comparison that was never validly computed.
    """
    if int(df) < 1:
        raise RQ3InferenceError(f"§12.1: df must be positive, got {df!r}")
    statistic = 2.0 * (float(full_loglik) - float(reduced_loglik))
    return statistic, float(chi2.sf(statistic, int(df)))


# ---------------------------------------------------------------------------
# block solutions and the 40-vs-60 validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockSolution:
    """The seven §9.6.3 quantities for one block at one node count, or a reason."""

    available: bool
    reason: str | None = None            # a registered §9.5.1 code
    solution: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)
    reason_detail: str | None = None     # which model or fold, never a code


def block_solution(cell: CellFits, block: models.BlockSpec) -> BlockSolution:
    """Assemble one block's seven registered numerical-validation quantities."""
    if block.population != cell.population:
        raise RQ3InferenceError("block and cell populations disagree")
    null_name = models.NULL_MODEL[cell.population].name
    needed = (block.full_model, block.reduced_model, null_name)
    for name in needed:
        if name not in cell.fits:
            raise RQ3InferenceError(f"cell is missing the fit for {name}")
    full, reduced = cell.fits[block.full_model], cell.fits[block.reduced_model]
    null = cell.fits[null_name]

    # a model that could not be fitted carries its own registered §9.5.1 code;
    # `needed` is in a fixed order, so the reported code is deterministic
    unestimable = [n for n in needed if not cell.fits[n].estimable]
    if unestimable:
        return BlockSolution(
            False,
            _registered(cell.fits[unestimable[0]].failing_condition),
            reason_detail=f"models: {','.join(unestimable)}",
        )

    for name in (block.full_model, block.reduced_model):
        if name not in cell.cross_validations:
            raise RQ3InferenceError(f"cell is missing the CV result for {name}")
    cv_full = cell.cross_validations[block.full_model]
    cv_reduced = cell.cross_validations[block.reduced_model]
    cv_missing = [n for n, cv in ((block.full_model, cv_full),
                                  (block.reduced_model, cv_reduced)) if not cv.estimable]
    if cv_missing:
        # §10.6 — the CV quantity is all-or-nothing, so a single unestimable
        # registered fold makes it, and every Delta depending on it, unestimable
        return BlockSolution(False, _registered(CODE_CV),
                             reason_detail=f"cv: {','.join(cv_missing)}")

    r2_full = mcfadden_r2(full.loglik, null.loglik)
    r2_reduced = mcfadden_r2(reduced.loglik, null.loglik)

    statistic, pvalue = lrt(full.loglik, reduced.loglik, block.df)
    # §12.1.1 via the frozen rule: a nested full model cannot have a lower
    # marginal likelihood than the model it contains, so a negative statistic is
    # a numerical statement about the fits, never evidence about the block. It
    # is not clamped, floored or absolute-valued (H29); exact 0.0 is the
    # admissible boundary and is reported normally.
    lrt_code = proto.lrt_status(statistic)
    if lrt_code != proto.STATUS_OK:
        return BlockSolution(False, _registered(lrt_code),
                             reason_detail=f"lrt_statistic={statistic!r}",
                             detail={"lrt_statistic": float(statistic)})

    # §10.4 — positive means the block improved prediction
    delta_cv = float(cv_reduced.macro_log_loss) - float(cv_full.macro_log_loss)

    solution = {
        "loglik": float(full.loglik),
        "lrt_statistic": float(statistic),
        "lrt_pvalue": float(pvalue),
        "mcfadden": float(r2_full - r2_reduced),
        "sigma_site": float(full.sigma_site),
        "cv_logloss": float(delta_cv),
        "beta": np.asarray(full.beta, dtype=float),
    }
    detail = {
        "mcfadden_full": float(r2_full),
        "mcfadden_reduced": float(r2_reduced),
        "loglik_full": float(full.loglik),
        "loglik_reduced": float(reduced.loglik),
        "loglik_null": float(null.loglik),
        "cv_logloss_full": float(cv_full.macro_log_loss),
        "cv_logloss_reduced": float(cv_reduced.macro_log_loss),
        "sigma_site_sq": float(full.sigma_site) ** 2,
        "selected_start_index_full": int(full.selected_start_index),
        "selected_start_index_reduced": int(reduced.selected_start_index),
        "selected_start_index_null": int(null.selected_start_index),
        "beta_columns": list(data.design_columns(
            models.model_by_name(cell.population, block.full_model).terms
        )),
    }
    return BlockSolution(True, None, solution, detail)


@dataclass(frozen=True)
class BlockResult:
    """One registered block comparison of one cell, after numerical validation."""

    population: str
    block: str
    operating_point: str
    method: str
    df: int
    status: str
    failing_condition: str | None = None     # a registered §9.5.1 code
    failure_detail: str | None = None        # which model or fold, never a code
    lrt_statistic: object = NOT_ESTIMABLE
    lrt_pvalue_raw: object = NOT_ESTIMABLE
    delta_mcfadden: object = NOT_ESTIMABLE
    delta_cv_logloss: object = NOT_ESTIMABLE
    numerical_validation: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    @property
    def estimable(self) -> bool:
        return self.status == proto.STATUS_OK

    @property
    def cell_index(self) -> int:
        return proto.CELL_REGISTRY.index((self.operating_point, self.method))

    def to_payload(self) -> dict:
        def render(value):
            return value if isinstance(value, str) else float(value)
        return {
            "population": self.population,
            "block": self.block,
            "operating_point": self.operating_point,
            "method": self.method,
            "cell_index": self.cell_index,
            "validation_unit": list(proto.VALIDATION_UNIT),
            "df": int(self.df),
            "status": self.status,
            "failing_condition": self.failing_condition,
            "failure_detail": self.failure_detail,
            "failure_is_numerical": (
                None if self.failing_condition is None
                else bool(proto.is_numerical_failure(self.failing_condition))
            ),
            "lrt_statistic": render(self.lrt_statistic),
            "lrt_pvalue_raw": render(self.lrt_pvalue_raw),
            "delta_mcfadden": render(self.delta_mcfadden),
            "delta_cv_logloss": render(self.delta_cv_logloss),
            "numerical_validation": self.numerical_validation,
            "detail": self.detail,
        }


def _validation_payload(verdict: dict) -> dict:
    """Serialize a frozen 40-vs-60 verdict, criteria included."""
    return {
        "passed": bool(verdict["passed"]),
        "n_evaluated": int(verdict["n_evaluated"]),
        "n_required": int(verdict["n_required"]),
        "n_passed": int(verdict["n_passed"]),
        "reason": verdict["reason"],
        "criteria": {
            name: {"discrepancy": float(d), "tolerance": float(t), "passed": bool(ok)}
            for name, (d, t, ok) in sorted(verdict["criteria"].items())
        },
    }


def evaluate_block(cell_40: CellFits, cell_60: CellFits,
                   block: models.BlockSpec) -> BlockResult:
    """§9.5 condition 5 / §9.6 — validate 40 against 60 and emit the block result.

    Both node counts are fitted independently and each selects its own best
    eligible start; the 60-node run may legitimately select a different start
    index, and that alone is not a failure (§9.6.1 step 3).
    """
    if cell_40.n_nodes != proto.GH_NODES_PRIMARY:
        raise RQ3InferenceError(
            f"the authoritative solution must be K={proto.GH_NODES_PRIMARY}, "
            f"got K={cell_40.n_nodes}"
        )
    if cell_60.n_nodes != proto.GH_NODES_VALIDATION:
        raise RQ3InferenceError(
            f"the validation solution must be K={proto.GH_NODES_VALIDATION}, "
            f"got K={cell_60.n_nodes}"
        )
    if (cell_40.population, cell_40.operating_point, cell_40.method) != \
            (cell_60.population, cell_60.operating_point, cell_60.method):
        raise RQ3InferenceError("the two node counts describe different cells")

    common = {
        "population": block.population, "block": block.block,
        "operating_point": cell_40.operating_point, "method": cell_40.method,
        "df": block.df,
    }
    solution_40 = block_solution(cell_40, block)
    solution_60 = block_solution(cell_60, block)

    if not solution_40.available or not solution_60.available:
        # §9.6.1 step 6 — either node count failing makes the block unestimable;
        # the other node count's success does not rescue it. The authoritative
        # 40-node failure names the code when both failed.
        failed = [(candidate, nodes)
                  for candidate, nodes in ((solution_40, proto.GH_NODES_PRIMARY),
                                           (solution_60, proto.GH_NODES_VALIDATION))
                  if not candidate.available]
        primary = failed[0][0]
        detail = "; ".join(
            f"K{nodes}: {candidate.reason}"
            + (f" ({candidate.reason_detail})" if candidate.reason_detail else "")
            for candidate, nodes in failed
        )
        return BlockResult(status=NOT_ESTIMABLE,
                           failing_condition=_registered(primary.reason),
                           failure_detail=detail,
                           detail=dict(solution_40.detail), **common)

    verdict = proto.numerical_validation_verdict(solution_40.solution, solution_60.solution)
    payload = _validation_payload(verdict)
    if not verdict["passed"]:
        # §9.5 condition 5 fails for this unit. A breached tolerance is
        # QUADRATURE_DISAGREEMENT; a required input that could not be
        # constructed at all is VALIDATION_MALFORMED (§9.6.4, §9.5.1). Either
        # way the unit's block is NOT_ESTIMABLE and the 60-node numbers are
        # still never substituted (§9.6.1 step 5).
        malformed = int(verdict["n_evaluated"]) != proto.N_REQUIRED_AGREEMENT_CRITERIA
        code = CODE_VALIDATION_MALFORMED if malformed else CODE_QUADRATURE
        return BlockResult(status=NOT_ESTIMABLE, failing_condition=_registered(code),
                           failure_detail=str(verdict["reason"]),
                           numerical_validation=payload,
                           detail=dict(solution_40.detail), **common)

    authoritative = solution_40.solution        # §9.6.1 step 4
    return BlockResult(
        status=proto.STATUS_OK,
        lrt_statistic=float(authoritative["lrt_statistic"]),
        lrt_pvalue_raw=float(authoritative["lrt_pvalue"]),
        delta_mcfadden=float(authoritative["mcfadden"]),
        delta_cv_logloss=float(authoritative["cv_logloss"]),
        numerical_validation=payload,
        detail=dict(solution_40.detail),
        **common,
    )


# ---------------------------------------------------------------------------
# §15 cell summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellSummary:
    """The §15 registered outputs of one cell (Holm columns filled in later)."""

    population: str
    operating_point: str
    method: str
    n_subjects: int
    n_sites: int
    rejected_fraction: float
    sigma_site_sq: object
    sigma_site_sq_condition: str | None
    blocks: dict

    @property
    def cell_index(self) -> int:
        return proto.CELL_REGISTRY.index((self.operating_point, self.method))

    def to_payload(self) -> dict:
        return {
            "population": self.population,
            "operating_point": self.operating_point,
            "method": self.method,
            "cell_index": self.cell_index,
            "n_subjects": int(self.n_subjects),
            "n_sites": int(self.n_sites),
            "rejected_fraction": float(self.rejected_fraction),
            "rejected_fraction_mode": proto.REJECTED_FRACTION_MODE,
            "sigma_site_sq": (self.sigma_site_sq if isinstance(self.sigma_site_sq, str)
                              else float(self.sigma_site_sq)),
            "sigma_site_sq_condition": self.sigma_site_sq_condition,
            "sigma_site_sq_interpretation": (
                "site-associated acquisition heterogeneity remaining after the "
                "registered fixed effects"
            ),
            "blocks": {name: result.to_payload() for name, result in sorted(self.blocks.items())},
        }


def summarize_cell(population: data.Population, outcome: data.CellOutcome,
                   cell_40: CellFits, cell_60: CellFits) -> CellSummary:
    """§15 — assemble one cell's registered outputs from its validated blocks."""
    blocks = {}
    for block in models.blocks_for(population.name):
        blocks[block.block] = evaluate_block(cell_40, cell_60, block)

    # §13 reporting condition — FROZEN at Stage E.2a. `sigma_site^2` comes from
    # the cell's *full* model and is reported **only when both of that cell's
    # §9.6.5 validation units passed**. Both units carry the same `M_full`, so a
    # quadrature failure in either one impugns the very variance estimate being
    # reported (H27). The 60-node variance is never substituted, and no
    # `sigma_site^2` is ever carried over from another cell, node count or model.
    full_names = {block.full_model for block in models.blocks_for(population.name)}
    if len(full_names) != 1:
        raise RQ3InferenceError(
            f"{population.name} blocks name more than one full model: {sorted(full_names)}"
        )
    if len(blocks) != len(models.blocks_for(population.name)):
        raise RQ3InferenceError(
            f"H27: {population.name} produced {len(blocks)} units, not its registered set"
        )
    full_fit = cell_40.fits[full_names.pop()]
    units_passed = all(result.estimable for result in blocks.values())
    if proto.SIGMA2_REQUIRES_BOTH_UNITS_PASSING and units_passed and full_fit.estimable:
        sigma_sq, condition = float(full_fit.sigma_site_sq), None
    else:
        # report that unit's own registered §9.5.1 code, in registered block order
        failing = [blocks[b.block].failing_condition
                   for b in models.blocks_for(population.name)
                   if not blocks[b.block].estimable and blocks[b.block].failing_condition]
        sigma_sq = NOT_ESTIMABLE
        condition = _registered(failing[0]) if failing else _registered(
            full_fit.failing_condition or glmm.COND_NO_ELIGIBLE_START
        )

    return CellSummary(
        population=population.name,
        operating_point=outcome.operating_point,
        method=outcome.method,
        n_subjects=population.n,
        n_sites=population.n_sites,
        rejected_fraction=data.rejected_fraction(population, outcome),
        sigma_site_sq=sigma_sq,
        sigma_site_sq_condition=condition,
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# §12.3 / §12.4 multiplicity
# ---------------------------------------------------------------------------

def holm_family(summaries, population: str, block: str) -> dict:
    """§12.3-§12.4 — one fixed family of six pre-registered slots.

    The family always contains exactly six slots in the §2.4.1 registry order.
    A `NOT_ESTIMABLE` cell keeps its slot and contributes an internal
    placeholder of 1.0 to the Holm arithmetic only; `m` is always 6, the
    placeholder is never serialized, and the cell can never be significant
    (§12.4, H12, H15, F8).
    """
    by_cell = {(s.operating_point, s.method): s for s in summaries
               if s.population == population}
    slots, raw = [], []
    for operating_point, method in proto.CELL_REGISTRY:
        summary = by_cell.get((operating_point, method))
        if summary is None:
            raise RQ3InferenceError(
                f"H12: family {population}/{block} is missing cell "
                f"{operating_point}/{method}"
            )
        if block not in summary.blocks:
            raise RQ3InferenceError(f"cell {operating_point}/{method} has no {block} block")
        result = summary.blocks[block]
        slots.append((operating_point, method))
        raw.append(result.lrt_pvalue_raw if result.estimable else NOT_ESTIMABLE)

    if len(raw) != proto.HOLM_FAMILY_SIZE:
        raise RQ3InferenceError(
            f"H12: family {population}/{block} holds {len(raw)} slots"
        )
    adjusted = proto.holm_adjust_family(raw)

    rows = []
    for index, ((operating_point, method), raw_p, adj_p) in enumerate(
            zip(slots, raw, adjusted)):
        significant = proto.holm_is_significant(raw_p, adj_p)
        if (raw_p == NOT_ESTIMABLE or adj_p == NOT_ESTIMABLE) and significant:
            raise RQ3InferenceError("H15: an unestimable cell was declared significant")
        rows.append({
            "cell_index": index,
            "operating_point": operating_point,
            "method": method,
            "p_raw": raw_p if isinstance(raw_p, str) else float(raw_p),
            "p_holm": adj_p if isinstance(adj_p, str) else float(adj_p),
            "significant": bool(significant),
        })
    return {
        "population": population,
        "block": block,
        "m": proto.HOLM_FAMILY_SIZE,
        "alpha": proto.ALPHA,
        "n_not_estimable": sum(1 for r in rows if r["p_raw"] == NOT_ESTIMABLE),
        "slots": rows,
    }


def holm_families(summaries) -> tuple:
    """The four registered families (§12.3), in their frozen order.

    No cross-family pooling, post-hoc or otherwise; the family definition is
    fixed by the protocol and is not widened or narrowed here.
    """
    families = []
    for population, block, size in proto.HOLM_FAMILIES:
        if size != proto.HOLM_FAMILY_SIZE:
            raise RQ3InferenceError(f"H12: family {population}/{block} declares size {size}")
        families.append(holm_family(summaries, population, block))
    if len(families) != 4:
        raise RQ3InferenceError(f"H12: {len(families)} Holm families, frozen count is 4")
    return tuple(families)
