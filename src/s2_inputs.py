"""Production input derivation for the frozen RQ2/S2 runner (Stage 2.1a).

Stage 2.0.8 froze `scripts/run_rq2_s2.py::run_formal(context, inputs)` and every
scientific primitive it calls, but it never built the `inputs` object. The only
provider that existed was `tests/_s2_fixtures.candidate_inputs()`, which returns
`[400, 120, 400, 120]` for every candidate of every fold. This module is the
production replacement, and it is **only** that: a derivation and orchestration
layer.

    frozen upstream artifact  ->  deterministic derivation  ->  run_formal kwarg

Four things it deliberately does not do.

* **It re-implements no statistic.** Clopper-Pearson, `U_BER`, the ESS score,
  the pseudo-count score, the fixed-sequential mechanism, the meta objective and
  the meta split RNG are all called from `certification` and `shrinkage`. There
  is no second copy of a formula here.
* **It fits nothing.** No domain classifier, no `C_domain`, no threshold, no
  weight and no `r*` is re-estimated. `tau_q^S0`, `tau_q^S1`, the density
  weights and `e_full,-s` are read.
* **It reads no target outcome.** Every quantity below is a function of the
  frozen source-OOF block of an outer fold, which by construction excludes that
  fold's held-out site. The production outcome source is never opened here, and
  the module names neither that file nor the post-barrier loader that owns it —
  `audit_inputs_never_open_target_outcomes` checks the source text for both.
* **It has no fallback.** A missing artifact, a checksum mismatch, a missing
  join key or a duplicated key raises; nothing degrades to a default.

Three derivation conventions are forced by §13 and are **not** stated there in
so many words. Each is bound to a frozen rule rather than chosen, each is
independent of `n0` (so the common-support requirement is untouched), and each
is recorded in the driver's own diagnostics so an auditor can see it:

1. `outer_index` / `pseudo_index` of `shrinkage.meta_split_permutation` are
   `contracts.FROZEN_SITES.index(...)`, which is the convention
   `s2_splitting.split_permutation` already froze for the real split RNG.
2. the pseudo-target's rows are permuted in frozen cohort-roster order
   (`sorted(canonical_id)`), asserted equal to the roster
   `s2_production.load_frozen_cohort_roster()` derives.
3. a pseudo-target hosts a label budget `N` iff `s2_splitting.site_eligible_at_n`
   accepts it — the frozen §4 rule, applied to the pseudo-target's own size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import certification as cert
import contracts
import s2_decisions as sd
import s2_production as prod
import s2_splitting as spl
import shrinkage as shr

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# the authoritative upstream artifacts
# ---------------------------------------------------------------------------
GATE1 = ROOT / "results" / "gate1"
RQ1_S0 = ROOT / "results" / "rq1_s0"
RQ1_S1 = ROOT / "results" / "rq1_s1"

#: The frozen provenance ledgers. Every S0/S1 artifact this module consumes is
#: verified against the checksum recorded in one of these at freeze time.
CHECKSUM_LEDGERS = {"rq1_s0": RQ1_S0 / "RESULT_CHECKSUMS.sha256",
                    "rq1_s1": RQ1_S1 / "RESULT_CHECKSUMS.sha256"}

#: `stage` is the ledger a file's checksum is verified against. Gate 1 froze no
#: checksum ledger, so its two artifacts carry `stage=None` and are instead
#: **content-bound** to checksummed S0 artifacts by exact equality
#: (`assert_gate1_binds_to_checksummed_s0`): `source_oof_predictions.csv` must
#: agree column-for-column with `confidence_source_oof.csv`, and
#: `source_full_coverage_error_by_fold.csv` must reproduce the `r_star_s` of
#: `risk_coverage_all_q.csv`. Neither is trusted on the strength of its path.
UPSTREAM_ARTIFACTS = (
    ("gate1_e_full", GATE1 / "source_full_coverage_error_by_fold.csv", None),
    ("gate1_source_oof", GATE1 / "source_oof_predictions.csv", None),
    ("s0_thresholds", RQ1_S0 / "thresholds_source_oof.csv", "rq1_s0"),
    ("s0_source_confidence", RQ1_S0 / "confidence_source_oof.csv", "rq1_s0"),
    ("s0_risk_coverage", RQ1_S0 / "risk_coverage_all_q.csv", "rq1_s0"),
    ("s1_thresholds", RQ1_S1 / "thresholds_weighted_source.csv", "rq1_s1"),
    ("s1_density_weights", RQ1_S1 / "density_weights.csv", "rq1_s1"),
    ("s1_weighted_source_risk", RQ1_S1 / "weighted_source_risk.csv", "rq1_s1"),
    ("s1_ess_by_site", RQ1_S1 / "ess_by_site.csv", "rq1_s1"),
    ("s1_selected_domain_c", RQ1_S1 / "selected_domain_C.csv", "rq1_s1"),
)

#: Columns of `risk_coverage_all_q.csv` the `r*` cross-check may decode. The
#: file also carries `BER_sel_target` and `realized_risk_violation__descriptive`;
#: `usecols` means the parser never decodes them, rather than the caller
#: dropping them afterwards.
R_STAR_CROSSCHECK_COLUMNS = ("held_out_site", "r_star_s")

#: `r*_s = e_full,-s - 0.10` recomputed in binary floating point need not be the
#: last decimal digit `risk_coverage_all_q.csv` serialized. One unit in the last
#: place of a quantity near 0.3 is 5.6e-17; this admits a few of them and
#: nothing wider.
R_STAR_CROSSCHECK_ATOL = 1e-15

#: The literal string a threshold table uses for the full-coverage operating
#: point. §7: it is **not** one of the 17 candidates.
NO_ABSTENTION = "NO_ABSTENTION"

#: Which frozen threshold grid each strategy deploys (§7). Not a choice made
#: here: three strategies operate on the S0 source-OOF confidence quantiles and
#: the weighted arm operates on the S1 density-weighted ones, so an oracle
#: diagnostic computed for a cell must sweep *that* strategy's own frozen
#: candidate sequence. Asserted total over `certification.STRATEGIES` below.
S0_GRID = "S0"
S1_GRID = "S1"
STRATEGY_THRESHOLD_GRID = {
    "target_local": S0_GRID,
    "pooled_local_shrinkage": S0_GRID,
    "pooled_source": S0_GRID,
    "target_unlabeled_weighted": S1_GRID,
}
assert set(STRATEGY_THRESHOLD_GRID) == set(cert.STRATEGIES)

#: Structural facts, re-derived at load and asserted — not scientific constants.
N_SITES = len(contracts.FROZEN_SITES)
N_METHODS = len(cert.CONFIDENCE_METHODS)
N_CANDIDATES = len(cert.Q_GRID)
N_ZERO_LABEL_DECISIONS = N_SITES * N_METHODS * len(cert.ZERO_LABEL_STRATEGIES)


class FrozenInputError(AssertionError):
    """Raised when a frozen upstream artifact cannot support a derivation."""


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def _read_checksum_ledger(path: Path) -> dict:
    if not path.exists():
        raise FrozenInputError(f"the frozen checksum ledger {path} is missing")
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        if not digest or not name:
            raise FrozenInputError(f"{path}: malformed ledger line {line!r}")
        out[name.strip()] = digest
    if not out:
        raise FrozenInputError(f"{path}: the ledger records no artifact")
    return out


def verify_frozen_upstream() -> dict:
    """Existence and checksum of every artifact this module consumes.

    Returns one row per artifact. Raises on a missing file, a file absent from
    its stage's frozen ledger, or a checksum that has moved since the freeze.
    """
    ledgers = {stage: _read_checksum_ledger(p)
               for stage, p in CHECKSUM_LEDGERS.items()}
    rows = []
    for key, path, stage in UPSTREAM_ARTIFACTS:
        if not path.exists():
            raise FrozenInputError(f"{key}: {path} is not on disk")
        digest = sd.sha256_file(path)
        recorded = None
        if stage is not None:
            recorded = ledgers[stage].get(path.name)
            if recorded is None:
                raise FrozenInputError(
                    f"{key}: {path.name} is not recorded in the frozen "
                    f"{stage} checksum ledger")
            if recorded != digest:
                raise FrozenInputError(
                    f"{key}: {path.name} checksum {digest[:12]}... does not "
                    f"match the frozen ledger {recorded[:12]}...")
        rows.append({"artifact": key, "path": str(path.relative_to(ROOT)),
                     "sha256": digest, "checksum_ledger": stage,
                     "ledger_verified": recorded is not None})
    return {"artifacts": rows,
            "n_ledger_verified": sum(r["ledger_verified"] for r in rows),
            "n_content_bound": sum(not r["ledger_verified"] for r in rows)}


def _path_of(key: str) -> Path:
    for k, path, _ in UPSTREAM_ARTIFACTS:
        if k == key:
            return path
    raise KeyError(key)


def _read(key: str, **kw) -> pd.DataFrame:
    return pd.read_csv(_path_of(key), float_precision="round_trip", **kw)


# ---------------------------------------------------------------------------
# gate 1 is bound to the checksummed S0 artifacts, never trusted by path
# ---------------------------------------------------------------------------
#: Bit-identical between the two files, and the only OOF columns any S2
#: derivation consumes: the identity keys, the label and the frozen prediction.
GATE1_OOF_EXACT_COLUMNS = ("outer_fold_held_out_site", "canonical_id",
                           "source_site", "y_true", "oof_pred",
                           "selected_pca", "selected_C")
#: Written by two independent `to_csv` calls, so the decimal round-trip may
#: differ in the last place. S2 consumes `oof_pred` and the three confidence
#: columns, never this probability, so an agreement requirement is enough.
GATE1_OOF_ULP_COLUMNS = ("oof_prob_mdd",)
#: One unit in the last place of a probability is at most 2**-53 ~ 1.1e-16.
GATE1_OOF_ULP_ATOL = 4e-16


def assert_gate1_binds_to_checksummed_s0() -> dict:
    """Bind both Gate 1 inputs to artifacts that *are* under a frozen ledger."""
    pred = _read("gate1_source_oof")
    conf = _read("s0_source_confidence")
    shared = GATE1_OOF_EXACT_COLUMNS + GATE1_OOF_ULP_COLUMNS
    missing = [c for c in shared if c not in conf.columns]
    if missing:
        raise FrozenInputError(
            f"the checksummed S0 confidence table lacks {missing}; the Gate 1 "
            "OOF predictions cannot be bound to it")
    if len(pred) != len(conf):
        raise FrozenInputError(
            f"Gate 1 OOF has {len(pred)} rows, S0 confidence has {len(conf)}")
    for col in GATE1_OOF_EXACT_COLUMNS:
        if not pred[col].equals(conf[col]):
            raise FrozenInputError(
                f"Gate 1 OOF column {col!r} differs from the checksummed S0 "
                "confidence table; the two are not the same frozen object")
    worst = 0.0
    for col in GATE1_OOF_ULP_COLUMNS:
        delta = float(np.abs(pred[col].to_numpy()
                             - conf[col].to_numpy()).max())
        worst = max(worst, delta)
        if delta > GATE1_OOF_ULP_ATOL:
            raise FrozenInputError(
                f"Gate 1 OOF column {col!r} disagrees with the checksummed S0 "
                f"confidence table by {delta:g}, beyond a decimal round-trip")

    e_full = _read("gate1_e_full")
    rc = _read("s0_risk_coverage", usecols=list(R_STAR_CROSSCHECK_COLUMNS))
    rc = rc.drop_duplicates()
    if len(rc) != N_SITES or rc["held_out_site"].duplicated().any():
        raise FrozenInputError(
            f"the r* cross-check table resolves {len(rc)} site rows, not "
            f"{N_SITES} unique ones")
    return {"n_oof_rows": int(len(pred)),
            "n_exact_columns": len(GATE1_OOF_EXACT_COLUMNS),
            "n_ulp_columns": len(GATE1_OOF_ULP_COLUMNS),
            "max_ulp_delta": worst,
            "n_e_full_rows": int(len(e_full)),
            "n_r_star_crosscheck_rows": int(len(rc))}


# ---------------------------------------------------------------------------
# the risk target
# ---------------------------------------------------------------------------
def derive_r_star() -> dict:
    """`r*_s = e_full,-s - 0.10`, recomputed, then cross-checked.

    The Gate 1 fold table is the authoritative source of `e_full,-s`; `r*_s` is
    **recomputed** from it rather than read off any artifact that already
    carries the answer. `risk_coverage_all_q.csv` does carry `r_star_s`, and it
    is used as an independent cross-check only — an agreement requirement, never
    the value.
    """
    e_full = _read("gate1_e_full")
    for col in ("site", "e_full_minus_s"):
        if col not in e_full.columns:
            raise FrozenInputError(f"the Gate 1 fold table lacks {col!r}")
    if e_full["site"].duplicated().any():
        raise FrozenInputError("duplicate outer fold in the Gate 1 fold table")
    sites = set(e_full["site"].astype(str))
    if sites != set(contracts.FROZEN_SITES):
        raise FrozenInputError(
            f"the Gate 1 fold table covers {len(sites)} folds, not the frozen "
            f"{N_SITES} eligible sites")

    r_star = {str(r.site): float(r.e_full_minus_s) - 0.10
              for r in e_full.itertuples()}
    if not all(np.isfinite(v) for v in r_star.values()):
        raise FrozenInputError("a recomputed r*_s is not finite")

    rc = _read("s0_risk_coverage", usecols=list(R_STAR_CROSSCHECK_COLUMNS))
    rc = rc.drop_duplicates()
    crosscheck = {str(r.held_out_site): float(r.r_star_s) for r in rc.itertuples()}
    if set(crosscheck) != set(r_star):
        raise FrozenInputError("the r* cross-check covers a different site set")
    deltas = {s: abs(r_star[s] - crosscheck[s]) for s in r_star}
    worst = max(deltas.values())
    if worst > R_STAR_CROSSCHECK_ATOL:
        bad = sorted((d, s) for s, d in deltas.items())[-3:]
        raise FrozenInputError(
            f"the recomputed r*_s disagrees with the independently frozen "
            f"r_star_s by up to {worst:g} (worst {bad})")
    return {"r_star": r_star, "n_sites": len(r_star),
            "max_abs_crosscheck_delta": float(worst),
            "n_exactly_equal": int(sum(1 for d in deltas.values() if d == 0.0)),
            "crosscheck_columns_read": list(R_STAR_CROSSCHECK_COLUMNS)}


# ---------------------------------------------------------------------------
# the candidate threshold grids
# ---------------------------------------------------------------------------
def _load_threshold_grid(key: str, tau_column: str) -> dict:
    """`{(site, method): (tau_0 .. tau_16)}` joined on explicit keys.

    `NO_ABSTENTION` is dropped before anything else happens, the 17 frozen `q`
    are then required to be present exactly once each, and the sequence is
    materialized **by looking up `Q_GRID` in order** — never by trusting the
    order the CSV happens to have.
    """
    frame = _read(key)
    for col in ("held_out_site", "method", "q", tau_column):
        if col not in frame.columns:
            raise FrozenInputError(f"{key}: threshold table lacks {col!r}")
    q_raw = frame["q"].astype(str)
    frame = frame.loc[q_raw != NO_ABSTENTION].copy()
    if (frame["q"].astype(str) == NO_ABSTENTION).any():
        raise FrozenInputError(f"{key}: NO_ABSTENTION survived the filter")
    frame["q_value"] = frame["q"].astype(str).astype(float)
    off_grid = sorted(set(frame["q_value"]) - set(cert.Q_GRID))
    if off_grid:
        raise FrozenInputError(f"{key}: off-grid q values {off_grid}")

    out = {}
    for (site, method), block in frame.groupby(["held_out_site", "method"]):
        site, method = str(site), str(method)
        if site not in contracts.FROZEN_SITES:
            raise FrozenInputError(f"{key}: {site!r} is not a frozen site")
        if method not in cert.CONFIDENCE_METHODS:
            raise FrozenInputError(f"{key}: {method!r} is not a frozen method")
        if block["q_value"].duplicated().any():
            raise FrozenInputError(f"{key}: duplicate q for {site}/{method}")
        by_q = dict(zip(block["q_value"], block[tau_column]))
        missing = [q for q in cert.Q_GRID if q not in by_q]
        if missing:
            raise FrozenInputError(f"{key}: {site}/{method} is missing q={missing}")
        if len(by_q) != N_CANDIDATES:
            raise FrozenInputError(
                f"{key}: {site}/{method} holds {len(by_q)} candidates, not "
                f"{N_CANDIDATES}")
        taus = tuple(float(by_q[q]) for q in cert.Q_GRID)
        if not all(np.isfinite(taus)):
            raise FrozenInputError(f"{key}: {site}/{method} has a non-finite tau")
        if any(b > a for a, b in zip(taus, taus[1:])):
            raise FrozenInputError(
                f"{key}: {site}/{method} tau is not non-increasing along the "
                "frozen candidate order (§25.24)")
        out[(site, method)] = taus

    expected = {(s, m) for s in contracts.FROZEN_SITES
                for m in cert.CONFIDENCE_METHODS}
    if set(out) != expected:
        missing = sorted(expected - set(out))
        extra = sorted(set(out) - expected)
        raise FrozenInputError(
            f"{key}: {len(missing)} missing and {len(extra)} unexpected "
            f"(site, method) blocks")
    return out


def source_threshold_grid() -> dict:
    """`tau_q^S0` — the grid `pooled_source`, `target_local` and shrinkage use."""
    return _load_threshold_grid("s0_thresholds", "tau")


def weighted_threshold_grid() -> dict:
    """`tau_q^S1` — the grid `target_unlabeled_weighted` deploys."""
    return _load_threshold_grid("s1_thresholds", "tau_S1")


# ---------------------------------------------------------------------------
# the fold's source-OOF calibration block
# ---------------------------------------------------------------------------
class SourceOofBlocks:
    """The frozen source-OOF rows of every outer fold, keyed and verified.

    One outer fold's block holds the 22 source sites of that fold and never the
    held-out site. Rows are ordered by `canonical_id` inside each site, which is
    the order `s2_production.load_frozen_cohort_roster()` derives, so a meta
    permutation drawn here indexes the same acquisitions the real split RNG
    would.
    """

    def __init__(self) -> None:
        frame = _read("s0_source_confidence")
        required = ("outer_fold_held_out_site", "canonical_id", "source_site",
                    "y_true", "oof_pred") + tuple(
                        f"confidence_{m}" for m in cert.CONFIDENCE_METHODS)
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise FrozenInputError(f"the source-OOF table lacks {missing}")
        frame = frame.sort_values(["outer_fold_held_out_site", "canonical_id"],
                                  kind="mergesort").reset_index(drop=True)
        if frame.duplicated(["outer_fold_held_out_site", "canonical_id"]).any():
            raise FrozenInputError(
                "duplicate (outer fold, canonical_id) in the source-OOF table")

        weights = _read("s1_density_weights")
        for col in ("held_out_site", "canonical_id", "w_final"):
            if col not in weights.columns:
                raise FrozenInputError(f"the density-weight table lacks {col!r}")
        weights = weights.sort_values(["held_out_site", "canonical_id"],
                                      kind="mergesort").reset_index(drop=True)
        if weights.duplicated(["held_out_site", "canonical_id"]).any():
            raise FrozenInputError(
                "duplicate (outer fold, canonical_id) in the density weights")
        if len(weights) != len(frame):
            raise FrozenInputError(
                f"the density weights hold {len(weights)} rows and the "
                f"source-OOF table {len(frame)}")
        key_left = frame["outer_fold_held_out_site"].astype(str) + "|" + \
            frame["canonical_id"].astype(str)
        key_right = weights["held_out_site"].astype(str) + "|" + \
            weights["canonical_id"].astype(str)
        if not key_left.equals(key_right):
            raise FrozenInputError(
                "the density weights and the source-OOF rows do not align on "
                "(outer fold, canonical_id); a positional join is not permitted")
        w = weights["w_final"].to_numpy(dtype=float)
        if not np.isfinite(w).all() or (w < 0).any():
            raise FrozenInputError("a frozen density weight is negative or non-finite")

        roster = prod.load_frozen_cohort_roster()
        self._roster = roster
        self._blocks = {}
        for site, block in frame.groupby("outer_fold_held_out_site"):
            site = str(site)
            if site not in contracts.FROZEN_SITES:
                raise FrozenInputError(f"{site!r} is not a frozen outer fold")
            idx = block.index.to_numpy()
            source_sites = block["source_site"].astype(str).to_numpy()
            if site in set(source_sites):
                raise FrozenInputError(
                    f"fold {site}: the held-out site appears in its own source "
                    "calibration pool")
            present = sorted(set(source_sites))
            if len(present) != N_SITES - 1:
                raise FrozenInputError(
                    f"fold {site}: {len(present)} source sites, not {N_SITES - 1}")
            rows_by_site = {}
            for j in present:
                pos = np.flatnonzero(source_sites == j)
                ids = block["canonical_id"].astype(str).to_numpy()[pos]
                if tuple(ids) != roster[j]:
                    raise FrozenInputError(
                        f"fold {site}, source site {j}: the block's row order is "
                        "not the frozen cohort roster order")
                rows_by_site[j] = pos
            self._blocks[site] = {
                "y": block["y_true"].to_numpy(dtype=int),
                "pred": block["oof_pred"].to_numpy(dtype=int),
                "weight": w[idx],
                "source_sites": source_sites,
                "rows_by_site": rows_by_site,
                "confidence": {m: block[f"confidence_{m}"].to_numpy(dtype=float)
                               for m in cert.CONFIDENCE_METHODS},
                "n_rows": int(len(block)),
            }
        if set(self._blocks) != set(contracts.FROZEN_SITES):
            raise FrozenInputError("the source-OOF table does not cover 23 folds")

    @property
    def folds(self) -> tuple:
        return contracts.FROZEN_SITES

    def block(self, site: str) -> dict:
        site = str(site)
        if site not in self._blocks:
            raise KeyError(f"no source-OOF block for outer fold {site!r}")
        return self._blocks[site]

    def roster_size(self, site: str) -> int:
        return len(self._roster[str(site)])


# ---------------------------------------------------------------------------
# retained source-side class counts, per candidate
# ---------------------------------------------------------------------------
def retained_source_counts(block: dict, method: str, taus) -> dict:
    """`[n_MDD, x_MDD, n_HC, x_HC]` per candidate on one fold's source-OOF rows.

    The order is the one `s2_cell_worker.decide_from_frame` unpacks. Counts come
    from `certification.retained_class_counts`, the same function every decision
    cell uses, so the source curve and the local curve are computed by one
    definition of "retained" and one definition of "wrong".
    """
    conf = block["confidence"][method]
    y, pred = block["y"], block["pred"]
    counts, coverage = [], []
    for tau in taus:
        keep = conf >= float(tau)
        c = cert.retained_class_counts(y, pred, keep)
        counts.append([int(c["n_keep_MDD"]), int(c["x_MDD"]),
                       int(c["n_keep_HC"]), int(c["x_HC"])])
        coverage.append(float(keep.mean()) if keep.size else 0.0)
    return {"source_class_counts": counts, "coverage": coverage,
            "n_source_oof": int(len(y))}


# ---------------------------------------------------------------------------
# §13 — the source meta-LOSO n0 driver
# ---------------------------------------------------------------------------
def select_n0_for_fold(blocks, source_taus: dict, site: str) -> dict:
    """The §13 meta-LOSO for exactly one outer fold. A pure function.

    Kept module-level and single-fold so the construction can be exercised on
    one fold — the whole 23-fold sweep is just this in a loop — and so nothing
    about the selection depends on which folds ran before it.
    """
    block = blocks.block(site)
    outer_index = contracts.FROZEN_SITES.index(site)
    y, pred = block["y"], block["pred"]
    wrong = pred != y
    is_mdd, is_hc = y == 1, y == 0

    terms = {n0: [] for n0 in shr.N0_GRID}
    local_terms, source_terms = [], []
    pseudo_targets, n_by_pseudo = [], {}

    for j, rows in sorted(block["rows_by_site"].items()):
        n_j = len(rows)
        if n_j != blocks.roster_size(j):
            raise FrozenInputError(
                f"fold {site}: pseudo-target {j} contributes {n_j} rows but "
                f"the frozen roster holds {blocks.roster_size(j)}")
        feasible = spl.feasible_n_grid(n_j)
        if not feasible:
            continue
        pseudo_targets.append(j)
        n_by_pseudo[j] = list(feasible)
        pseudo_index = contracts.FROZEN_SITES.index(j)
        width = max(feasible)
        perms = [shr.meta_split_permutation(outer_index, pseudo_index, rm, n_j)
                 for rm in range(shr.R_META)]

        for method in cert.CONFIDENCE_METHODS:
            conf = block["confidence"][method]
            taus = source_taus[(site, method)]
            keep = conf[None, :] >= np.asarray(taus, dtype=float)[:, None]
            k_mdd, k_hc = keep & is_mdd, keep & is_hc
            e_mdd, e_hc = k_mdd & wrong, k_hc & wrong

            # pseudo-source = the fold's block minus the pseudo-target
            n_src = {"MDD": k_mdd.sum(1) - k_mdd[:, rows].sum(1),
                     "HC": k_hc.sum(1) - k_hc[:, rows].sum(1)}
            x_src = {"MDD": e_mdd.sum(1) - e_mdd[:, rows].sum(1),
                     "HC": e_hc.sum(1) - e_hc[:, rows].sum(1)}
            # the pseudo-target's own totals, for the remainder by difference
            n_tot = {"MDD": k_mdd[:, rows].sum(1), "HC": k_hc[:, rows].sum(1)}
            x_tot = {"MDD": e_mdd[:, rows].sum(1), "HC": e_hc[:, rows].sum(1)}

            for perm in perms:
                head = rows[perm[:width]]
                cum_keep = np.cumsum(keep[:, head], axis=1)
                cum_n = {"MDD": np.cumsum(k_mdd[:, head], axis=1),
                         "HC": np.cumsum(k_hc[:, head], axis=1)}
                cum_x = {"MDD": np.cumsum(e_mdd[:, head], axis=1),
                         "HC": np.cumsum(e_hc[:, head], axis=1)}
                for n_labels in feasible:
                    k = n_labels - 1
                    coverage = cum_keep[:, k] / float(n_labels)
                    # §8.3, the pooled_local_shrinkage row: local coverage,
                    # both local class counts and both source class counts.
                    eligible = ((coverage >= cert.C_MIN)
                                & (cum_n["MDD"][:, k] >= cert.M_MIN)
                                & (cum_n["HC"][:, k] >= cert.M_MIN)
                                & (n_src["MDD"] >= cert.M_MIN)
                                & (n_src["HC"] >= cert.M_MIN))
                    for cls in ("MDD", "HC"):
                        n_loc = cum_n[cls][:, k]
                        x_loc = cum_x[cls][:, k]
                        n_rem = n_tot[cls] - n_loc
                        x_rem = x_tot[cls] - x_loc
                        use = eligible & (n_rem >= 1)
                        if not use.any():
                            continue
                        r_ref = x_rem[use] / n_rem[use]
                        r_src = x_src[cls][use] / n_src[cls][use]
                        n_c = n_loc[use].astype(float)
                        x_c = x_loc[use].astype(float)
                        for n0 in shr.N0_GRID:
                            terms[n0].append(
                                (x_c + n0 * r_src) / (n_c + n0) - r_ref)
                        # §24.9 references: lambda == 1 is the pure local curve,
                        # lambda == 0 the pure source curve. Neither is a grid
                        # candidate; both are reported.
                        local_terms.append(x_c / n_c - r_ref)
                        source_terms.append(r_src - r_ref)

    stacked = {n0: (np.concatenate(v) if v else np.array([], dtype=float))
               for n0, v in terms.items()}
    sizes = {n0: int(arr.size) for n0, arr in stacked.items()}
    if len(set(sizes.values())) != 1:
        raise FrozenInputError(
            f"fold {site}: the n0 candidates were scored on different term "
            f"counts {sizes}; §13 requires a common support")
    selection = shr.meta_objective(stacked)
    local = np.concatenate(local_terms) if local_terms else np.array([])
    source = np.concatenate(source_terms) if source_terms else np.array([])
    return {
        "outer_site": site,
        "n0": int(selection["selected_n0"]),
        "d_by_n0": {int(k): float(v) for k, v in selection["d_by_n0"].items()},
        "n_terms": int(selection["n_terms"]),
        "n_ties_at_argmin": int(selection["n_ties_at_argmin"]),
        "pseudo_targets": tuple(pseudo_targets),
        "term_counts_by_n0": dict(sizes),
        "feasible_n_by_pseudo_target": n_by_pseudo,
        "d_lambda_1_pure_local": float(np.sqrt(np.mean(local ** 2))),
        "d_lambda_0_pure_source": float(np.sqrt(np.mean(source ** 2))),
    }


class MetaLosoN0Driver:
    """One `n0` per outer fold, selected inside that fold's 22 source sites.

    The construction is §13 verbatim: pseudo-target `j`, pseudo-source the other
    21, `R_META = 20` meta-splits per `(outer, pseudo)` from the frozen
    `SeedSequence`, the shrunk class-risk curve against the class error rate at
    the same `tau` on the pseudo-target's held-out remainder, RMS over
    `(j, N, rm, method, tau, c)`, `argmin` over the frozen grid, ties to the
    smallest.

    Everything that decides *which* terms enter is independent of `n0` —
    §8.3 eligibility, and whether the remainder holds a class-`c` row at all —
    so the five candidates are scored on one common support, which
    `shrinkage.meta_objective` re-asserts before it selects.

    The held-out target site contributes nothing: its rows are not in the fold's
    source-OOF block, and no target label, split or outcome is opened here.
    """

    def __init__(self, blocks: SourceOofBlocks, source_taus: dict) -> None:
        self._blocks = blocks
        self._taus = source_taus
        self._by_fold = {}
        self._objective_rows = []
        self._support_rows = []
        for site in contracts.FROZEN_SITES:
            self._by_fold[site] = self._record(
                select_n0_for_fold(blocks, source_taus, site))

    def _record(self, fold: dict) -> dict:
        site = fold["outer_site"]
        for n0 in shr.N0_GRID:
            self._objective_rows.append({
                "outer_site": site, "n0": int(n0),
                "d_s_n0": float(fold["d_by_n0"][int(n0)]),
                "n_terms": int(fold["n_terms"]),
                "selected": bool(int(n0) == fold["n0"]),
            })
        self._support_rows.append({
            "outer_site": site,
            "n_pseudo_targets": len(fold["pseudo_targets"]),
            "pseudo_target_sites": ",".join(fold["pseudo_targets"]),
            "held_out_target_in_meta_world": bool(site in fold["pseudo_targets"]),
            "meta_repeats": int(shr.R_META),
            "n_terms": int(fold["n_terms"]),
            "common_support": True,
            "term_counts_by_n0": dict(fold["term_counts_by_n0"]),
            "n0_grid": list(shr.N0_GRID),
            "selected_n0": int(fold["n0"]),
            "n_ties_at_argmin": int(fold["n_ties_at_argmin"]),
            "tie_break_rule": shr.N0_TIE_BREAK,
            "d_lambda_1_pure_local": float(fold["d_lambda_1_pure_local"]),
            "d_lambda_0_pure_source": float(fold["d_lambda_0_pure_source"]),
            "feasible_n_by_pseudo_target": fold["feasible_n_by_pseudo_target"],
            "target_outcome_used": False,
        })
        return fold

    # -- results ------------------------------------------------------------
    def n0_for(self, site: str) -> float:
        site = str(site)
        if site not in self._by_fold:
            raise KeyError(f"no meta-LOSO selection for outer fold {site!r}")
        return float(self._by_fold[site]["n0"])

    @property
    def selected(self) -> dict:
        return {s: v["n0"] for s, v in self._by_fold.items()}

    def objective_table(self) -> pd.DataFrame:
        """The `len(FROZEN_SITES)` x `len(N0_GRID)` `d(s, n0)` table of §24.9."""
        frame = pd.DataFrame(self._objective_rows)
        if frame.shape[0] != N_SITES * len(shr.N0_GRID):
            raise FrozenInputError(
                f"the d(s, n0) table has {frame.shape[0]} rows, not "
                f"{N_SITES * len(shr.N0_GRID)}")
        return frame

    def support_diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame(self._support_rows)


# ---------------------------------------------------------------------------
# phase 0 — the 138 zero-label decisions
# ---------------------------------------------------------------------------
def _not_testable_reason(coverage) -> str:
    """Name the constraint that bit, exactly as the decision cell does.

    Coverage is non-decreasing along the frozen order, so the loosest candidate
    decides whether coverage or a class count was the binding constraint.
    """
    return (sd.REASON_COVERAGE if float(coverage[-1]) < cert.C_MIN
            else sd.REASON_CLASS_COUNT)


def _decision_payload(site, method, strategy, outcome, taus, coverage,
                      r_star, extra) -> dict:
    """Shape one zero-label artifact from a frozen sequence result."""
    sel = outcome["selected_index"]
    reason = outcome["reason"]
    if outcome["state"] == cert.NOT_TESTABLE:
        reason = _not_testable_reason(coverage)
    payload = {
        "protocol_version": sd.PROTOCOL_VERSION,
        "site": str(site), "method": str(method), "strategy": str(strategy),
        "guarantee_tier": outcome["guarantee_tier"],
        "score_type": outcome["score_type"],
        "guarantee": outcome["guarantee"],
        "state": outcome["state"], "reason": reason,
        "selected_q": (float(cert.Q_GRID[sel]) if sel is not None else None),
        "selected_candidate_index": (int(sel) if sel is not None else None),
        "selected_tau": (float(taus[sel]) if sel is not None else None),
        "selected_score": (float(outcome["score_by_index"][sel])
                           if sel is not None else None),
        "n_eligible": int(outcome["n_eligible"]),
        "first_eligible_index": (None if outcome["first_eligible_index"] is None
                                 else int(outcome["first_eligible_index"])),
        "n_passed_candidates": int(outcome["n_passed_candidates"]),
        "stopped_at_index": (None if outcome["stopped_at_index"] is None
                             else int(outcome["stopped_at_index"])),
        "evaluated_q": [float(cert.Q_GRID[i]) for i in outcome["evaluated_indices"]],
        "score_by_q": {str(cert.Q_GRID[i]): float(v)
                       for i, v in outcome["score_by_index"].items()},
        "r_star": float(r_star),
        "repeat": None, "n_labels": None,
    }
    payload.update(extra)
    return payload


def _pooled_source_decision(site, method, taus, counts, coverage, r_star) -> dict:
    """Tier B: the exact class-conditional bound on the fold's source-OOF counts."""
    eligible = [cert.candidate_eligible(coverage[i], (counts[i][0], counts[i][2]))
                for i in range(N_CANDIDATES)]

    def u_ber_fn(i):
        return cert.u_ber(cert.clopper_pearson_upper(counts[i][1], counts[i][0]),
                          cert.clopper_pearson_upper(counts[i][3], counts[i][2]))

    outcome = cert.source_reference_sequence(cert.Q_GRID, eligible, u_ber_fn,
                                             r_star)
    sel = outcome["selected_index"]
    extra = {
        "scope": outcome["scope"],
        "eligible_q": [float(cert.Q_GRID[i]) for i, f in enumerate(eligible) if f],
        "n_keep_source": (int(counts[sel][0] + counts[sel][2])
                          if sel is not None else None),
        "n_keep_source_MDD": int(counts[sel][0]) if sel is not None else None,
        "n_keep_source_HC": int(counts[sel][2]) if sel is not None else None,
        "x_source_MDD": int(counts[sel][1]) if sel is not None else None,
        "x_source_HC": int(counts[sel][3]) if sel is not None else None,
        "source_coverage_selected": (float(coverage[sel]) if sel is not None
                                     else None),
    }
    return _decision_payload(site, method, "pooled_source", outcome, taus,
                             coverage, r_star, extra)


def _weighted_decision(site, method, taus, diagnostics, r_star) -> dict:
    """Tier C: the ESS screening score on the frozen density-weighted rows."""
    eligible = [cert.candidate_eligible(diagnostics["weighted_coverage"][i],
                                        (diagnostics["n_keep_MDD"][i],
                                         diagnostics["n_keep_HC"][i]))
                for i in range(N_CANDIDATES)]

    def score_fn(i):
        return cert.mean_class_score(diagnostics["score_MDD"][i],
                                     diagnostics["score_HC"][i])

    outcome = cert.empirical_screening_sequence(
        cert.Q_GRID, eligible, score_fn, r_star, "target_unlabeled_weighted")
    sel = outcome["selected_index"]
    extra = {
        "screening_reference": float(outcome["screening_reference"]),
        "eligible_q": [float(cert.Q_GRID[i]) for i, f in enumerate(eligible) if f],
        "n_keep_source_MDD": (int(diagnostics["n_keep_MDD"][sel])
                              if sel is not None else None),
        "n_keep_source_HC": (int(diagnostics["n_keep_HC"][sel])
                             if sel is not None else None),
        "n_eff_MDD": (float(diagnostics["n_eff_MDD"][sel]) if sel is not None
                      else None),
        "n_eff_HC": (float(diagnostics["n_eff_HC"][sel]) if sel is not None
                     else None),
        "ess_ratio_MDD": (float(diagnostics["ess_ratio_MDD"][sel])
                          if sel is not None else None),
        "ess_ratio_HC": (float(diagnostics["ess_ratio_HC"][sel])
                         if sel is not None else None),
        "weighted_coverage_selected": (float(diagnostics["weighted_coverage"][sel])
                                       if sel is not None else None),
    }
    return _decision_payload(site, method, "target_unlabeled_weighted", outcome,
                             taus, diagnostics["weighted_coverage"], r_star,
                             extra)


# ---------------------------------------------------------------------------
# the weighted (S1) candidate inputs
# ---------------------------------------------------------------------------
class WeightedCandidateInputs:
    """Per-candidate weighted screening inputs, consumed from frozen S1.

    Nothing is re-fitted and no `C_domain` is re-selected: the deployed weights
    `w_final` and the deployed thresholds `tau_q^S1` are read, the retained set
    is formed with the same `conf >= tau` rule every other S2 path uses, and the
    score comes from `certification.empirical_ess_upper_score`.

    Every derived quantity that the frozen `weighted_source_risk.csv` also
    carries — retained class counts, class weight sums, weighted class error
    rates, weighted and unweighted coverage — is required to reproduce it, so
    the derivation is checked against the S1 run rather than merely resembling
    it.
    """

    #: `weighted_source_risk.csv` serializes to ~17 significant digits; a
    #: re-derived sum of ~2000 floats may differ in the last few ulp.
    REPRODUCTION_RTOL = 1e-12

    def __init__(self, blocks: SourceOofBlocks, weighted_taus: dict) -> None:
        self._diagnostics = {}
        frozen = _read("s1_weighted_source_risk")
        frozen = frozen.loc[frozen["q"].astype(str) != NO_ABSTENTION].copy()
        frozen["q_value"] = frozen["q"].astype(str).astype(float)
        frozen = frozen.set_index(["held_out_site", "method", "q_value"])
        if frozen.index.duplicated().any():
            raise FrozenInputError(
                "duplicate (site, method, q) in the frozen weighted source risk")
        self._reproduction = []

        for site in contracts.FROZEN_SITES:
            block = blocks.block(site)
            y, pred, w = block["y"], block["pred"], block["weight"]
            wrong = pred != y
            is_mdd, is_hc = y == 1, y == 0
            w_total = float(w.sum())
            if w_total <= 0:
                raise FrozenInputError(f"fold {site}: the weight sum is not positive")
            for method in cert.CONFIDENCE_METHODS:
                conf = block["confidence"][method]
                taus = weighted_taus[(site, method)]
                d = {k: [] for k in (
                    "n_keep", "n_keep_MDD", "n_keep_HC", "weighted_coverage",
                    "unweighted_coverage", "score_MDD", "score_HC",
                    "n_eff_MDD", "n_eff_HC", "ess_ratio_MDD", "ess_ratio_HC",
                    "w_sum_MDD", "w_sum_HC", "err_weighted_MDD",
                    "err_weighted_HC")}
                for tau in taus:
                    keep = conf >= float(tau)
                    counts = cert.retained_class_counts(y, pred, keep)
                    d["n_keep"].append(int(counts["n_keep"]))
                    d["n_keep_MDD"].append(int(counts["n_keep_MDD"]))
                    d["n_keep_HC"].append(int(counts["n_keep_HC"]))
                    d["weighted_coverage"].append(float(w[keep].sum() / w_total))
                    d["unweighted_coverage"].append(
                        float(keep.mean()) if keep.size else 0.0)
                    for cls, mask in (("MDD", is_mdd), ("HC", is_hc)):
                        sel = keep & mask
                        scored = cert.empirical_ess_upper_score(w[sel], wrong[sel])
                        d[f"score_{cls}"].append(float(scored["score"]))
                        d[f"n_eff_{cls}"].append(float(scored["n_eff"]))
                        d[f"ess_ratio_{cls}"].append(float(scored["ess_ratio"]))
                        d[f"w_sum_{cls}"].append(float(scored["weight_sum"]))
                        d[f"err_weighted_{cls}"].append(float(scored["err_weighted"]))
                self._assert_reproduces_frozen(site, method, frozen, d)
                self._diagnostics[(site, method)] = d

    def _assert_reproduces_frozen(self, site, method, frozen, d) -> None:
        for i, q in enumerate(cert.Q_GRID):
            try:
                row = frozen.loc[(site, method, q)]
            except KeyError as exc:
                raise FrozenInputError(
                    f"the frozen weighted source risk has no row for "
                    f"{site}/{method}/q={q}") from exc
            checks = (
                ("n_keep_source", d["n_keep"][i], row["n_keep_source"]),
                ("n_keep_source_MDD", d["n_keep_MDD"][i], row["n_keep_source_MDD"]),
                ("n_keep_source_HC", d["n_keep_HC"][i], row["n_keep_source_HC"]),
                ("W_MDD", d["w_sum_MDD"][i], row["W_MDD"]),
                ("W_HC", d["w_sum_HC"][i], row["W_HC"]),
                ("weighted_error_MDD", d["err_weighted_MDD"][i],
                 row["weighted_error_MDD"]),
                ("weighted_error_HC", d["err_weighted_HC"][i],
                 row["weighted_error_HC"]),
                ("weighted_source_coverage", d["weighted_coverage"][i],
                 row["weighted_source_coverage"]),
                ("unweighted_source_coverage", d["unweighted_coverage"][i],
                 row["unweighted_source_coverage"]),
            )
            for name, got, want in checks:
                if not np.isclose(float(got), float(want),
                                  rtol=self.REPRODUCTION_RTOL, atol=0.0):
                    raise FrozenInputError(
                        f"{site}/{method}/q={q}: re-derived {name}={got!r} does "
                        f"not reproduce the frozen S1 value {want!r}")
            self._reproduction.append({"site": site, "method": method, "q": q,
                                       "n_checks": len(checks)})

    def for_cell(self, site: str, method: str) -> dict:
        key = (str(site), str(method))
        if key not in self._diagnostics:
            raise KeyError(f"no weighted candidate inputs for {key}")
        return self._diagnostics[key]

    @property
    def n_reproduction_checks(self) -> int:
        return sum(r["n_checks"] for r in self._reproduction)


# ---------------------------------------------------------------------------
# the production provider
# ---------------------------------------------------------------------------
class FrozenS2Inputs:
    """The production `inputs` object `run_formal(context, inputs)` consumes.

    Four methods, and each of them is exactly what the frozen runner calls:

    ==============================  ==========================================
    `zero_label_decisions()`        138 phase-0 artifacts, `record_zero_label_decision`
    `candidate_inputs(site, method)` `q_values`, `taus`, `source_class_counts`, `r_star`, `n0`
    `evaluation_inputs(site, method)` `taus`, `r_star`, `oracle`
    `point_sections()`              the input-derived point-estimate sections
    ==============================  ==========================================

    Construction performs every derivation eagerly, so a provider that exists is
    a provider whose upstream verified, whose joins closed and whose 138
    decisions validated. There is no lazy path on which a missing key is
    discovered halfway through a formal run.
    """

    #: Every column of the frozen `ess_by_site.csv` the §24.10 diagnostics
    #: section must carry. Read from the artifact, never re-estimated: the
    #: clipping fractions, the post-clip weight mean, the overall ESS and its
    #: fraction, the per-method retained-ESS summaries and the frozen exclusion
    #: rule are all S1 outputs. `_load_ess_by_site` asserts the artifact still
    #: holds exactly this set, so a column added or dropped upstream fails here
    #: rather than silently shortening the report.
    ESS_BY_SITE_COLUMNS = (
        "held_out_site", "n_target", "n_source", "c_domain", "ess",
        "ess_fraction", "w_min", "w_median", "w_mean", "w_max",
        "frac_clipped_low", "frac_clipped_high", "w_mean_post_clip",
        "ess_retained_OP50_probability", "ess_retained_OP30_probability",
        "ess_retained_OP50_bootstrap", "ess_retained_OP30_bootstrap",
        "ess_retained_OP50_mahalanobis", "ess_retained_OP30_mahalanobis",
        "ess_exclusion_rule",
    )

    #: Per-candidate weight diagnostics carried for every `(site, method)` cell.
    PER_CANDIDATE_WEIGHT_FIELDS = (
        "n_keep", "n_keep_MDD", "n_keep_HC", "n_eff_MDD", "n_eff_HC",
        "ess_ratio_MDD", "ess_ratio_HC", "w_sum_MDD", "w_sum_HC",
        "err_weighted_MDD", "err_weighted_HC", "weighted_coverage",
        "unweighted_coverage",
    )

    @classmethod
    def from_frozen_context(cls, context) -> "FrozenS2Inputs":
        """**The authoritative constructor the formal runner uses.**

        The production runner takes no input provider (§2 of the Stage 2.1b
        closure): it calls this, and this reads only frozen artifacts whose
        paths are module constants. `context` is authority, not data — no path,
        no threshold, no `n0` and no decision travels through it. It is
        re-derived here so a provider cannot be built under a context that has
        stopped being authoritative.
        """
        from s2_production import ProductionAuthorityError, ProductionS2Context

        if not isinstance(context, ProductionS2Context):
            raise ProductionAuthorityError(
                "frozen S2 inputs are constructed under a ProductionS2Context")
        context.assert_still_authoritative()
        inputs = cls()
        inputs._context_run_id = context.run_id
        return inputs

    def __init__(self) -> None:
        self._context_run_id = None
        self.provenance = verify_frozen_upstream()
        self.gate1_binding = assert_gate1_binds_to_checksummed_s0()
        self._r_star_result = derive_r_star()
        self._r_star = self._r_star_result["r_star"]
        self._source_taus = source_threshold_grid()
        self._weighted_taus = weighted_threshold_grid()
        self._blocks = SourceOofBlocks()
        self._source_counts = {}
        for site in contracts.FROZEN_SITES:
            block = self._blocks.block(site)
            for method in cert.CONFIDENCE_METHODS:
                self._source_counts[(site, method)] = retained_source_counts(
                    block, method, self._source_taus[(site, method)])
        self._weighted = WeightedCandidateInputs(self._blocks, self._weighted_taus)
        self._meta = MetaLosoN0Driver(self._blocks, self._source_taus)
        self._domain_c = self._load_domain_c()
        self._ess_by_site = self._load_ess_by_site()
        self._zero_label = self._derive_zero_label_decisions()

    # -- S1 provenance ------------------------------------------------------
    def _load_domain_c(self) -> dict:
        frame = _read("s1_selected_domain_c")
        for col in ("held_out_site", "c_domain", "target_outcome_used",
                    "real_target_site_present_in_meta_world"):
            if col not in frame.columns:
                raise FrozenInputError(f"selected_domain_C.csv lacks {col!r}")
        if set(frame["held_out_site"].astype(str)) != set(contracts.FROZEN_SITES):
            raise FrozenInputError("selected_domain_C.csv covers another site set")
        if frame["target_outcome_used"].any():
            raise FrozenInputError(
                "a frozen S1 domain selection records target_outcome_used")
        if frame["real_target_site_present_in_meta_world"].any():
            raise FrozenInputError(
                "a frozen S1 domain selection put the real target in its meta world")
        return {str(r.held_out_site): float(r.c_domain) for r in frame.itertuples()}

    def _load_ess_by_site(self) -> dict:
        """The frozen per-fold ESS, clipping and weight diagnostics (§24.10).

        Read whole and asserted whole: the §24.10 section must carry the
        clipping fractions and the overall ESS summaries the S1 run froze, not a
        hand-picked subset of them, and nothing here recomputes a weight.
        """
        frame = _read("s1_ess_by_site")
        columns = tuple(str(c) for c in frame.columns)
        if columns != self.ESS_BY_SITE_COLUMNS:
            missing = [c for c in self.ESS_BY_SITE_COLUMNS if c not in columns]
            extra = [c for c in columns if c not in self.ESS_BY_SITE_COLUMNS]
            raise FrozenInputError(
                f"ess_by_site.csv does not carry the frozen S1 diagnostic "
                f"schema: missing {missing}, unexpected {extra}")
        if set(frame["held_out_site"].astype(str)) != set(contracts.FROZEN_SITES):
            raise FrozenInputError("ess_by_site.csv covers another site set")
        if frame["held_out_site"].duplicated().any():
            raise FrozenInputError("ess_by_site.csv repeats an outer fold")
        out = {}
        for row in frame.to_dict("records"):
            site = str(row["held_out_site"])
            record = {}
            for column in self.ESS_BY_SITE_COLUMNS:
                value = row[column]
                record[column] = (str(value) if isinstance(value, str)
                                  else float(value))
            record["held_out_site"] = site
            if not np.isclose(record["c_domain"], self._domain_c[site],
                              rtol=0.0, atol=1e-12):
                raise FrozenInputError(
                    f"{site}: ess_by_site.csv reports C_domain "
                    f"{record['c_domain']!r}, selected_domain_C.csv "
                    f"{self._domain_c[site]!r}")
            out[site] = record
        return out

    # -- phase 0 ------------------------------------------------------------
    def _derive_zero_label_decisions(self) -> tuple:
        payloads = []
        for site in contracts.FROZEN_SITES:
            r_star = self._r_star[site]
            for method in cert.CONFIDENCE_METHODS:
                src = self._source_counts[(site, method)]
                payloads.append(_pooled_source_decision(
                    site, method, self._source_taus[(site, method)],
                    src["source_class_counts"], src["coverage"], r_star))
                payloads.append(_weighted_decision(
                    site, method, self._weighted_taus[(site, method)],
                    self._weighted.for_cell(site, method), r_star))
        for payload in payloads:
            sd.validate_decision_artifact(payload, zero_label=True)
        keys = [(p["site"], p["method"], p["strategy"]) for p in payloads]
        if len(set(keys)) != len(keys):
            raise FrozenInputError("a zero-label decision key is duplicated")
        expected = {(s, m, st) for s in contracts.FROZEN_SITES
                    for m in cert.CONFIDENCE_METHODS
                    for st in cert.ZERO_LABEL_STRATEGIES}
        if set(keys) != expected:
            raise FrozenInputError(
                f"the zero-label decision set is not 23 x 3 x 2: "
                f"{len(set(keys))} distinct keys")
        if len(payloads) != N_ZERO_LABEL_DECISIONS:
            raise FrozenInputError(
                f"{len(payloads)} zero-label decisions, not {N_ZERO_LABEL_DECISIONS}")
        return tuple(payloads)

    # -- the run_formal interface -------------------------------------------
    def zero_label_decisions(self) -> tuple:
        """The 138 phase-0 artifacts, in frozen (site, method, strategy) order."""
        return self._zero_label

    def candidate_inputs(self, site: str, method: str) -> dict:
        """Exactly the keyword arguments `run_decision_cell` declares."""
        key = (str(site), str(method))
        if key not in self._source_counts:
            raise KeyError(f"no candidate inputs for {key}")
        src = self._source_counts[key]
        return {
            "q_values": list(cert.Q_GRID),
            "taus": [float(t) for t in self._source_taus[key]],
            "source_class_counts": [list(map(int, row))
                                    for row in src["source_class_counts"]],
            "r_star": float(self._r_star[key[0]]),
            "n0": self._meta.n0_for(key[0]),
        }

    def candidate_grid(self, site: str, method: str, strategy: str) -> list:
        """The frozen candidate threshold sequence *that strategy* deploys.

        §7 makes the grid strategy-dependent, and the mapping is frozen, not
        chosen here: `target_local`, `pooled_local_shrinkage` and
        `pooled_source` sweep `tau_q^S0`; `target_unlabeled_weighted` sweeps
        `tau_q^S1`. Both come from a frozen artifact — neither is re-estimated,
        and neither is reconstructed from a selected threshold.
        """
        key = (str(site), str(method))
        strategy = str(strategy)
        if strategy not in STRATEGY_THRESHOLD_GRID:
            raise KeyError(f"{strategy!r} is not a frozen S2 strategy")
        grids = {S0_GRID: self._source_taus, S1_GRID: self._weighted_taus}
        table = grids[STRATEGY_THRESHOLD_GRID[strategy]]
        if key not in table:
            raise KeyError(f"no {STRATEGY_THRESHOLD_GRID[strategy]} grid for {key}")
        return [float(t) for t in table[key]]

    def evaluation_inputs(self, site: str, method: str, strategy: str) -> dict:
        """The non-outcome metadata `run_evaluation_cell` declares.

        The strategy axis is not cosmetic: `taus` is consumed by the §17 oracle
        diagnostic alone, and a cell evaluated on another strategy's grid would
        answer the feasibility question about thresholds that strategy never
        deploys. `oracle` is **on** — the diagnostic is required by §24.5 — and
        it stays a diagnostic: `run_evaluation_cell` is reachable only after
        `open_phase2()`, it re-verifies the frozen decision checksum before and
        after, and no oracle field enters a decision, a `selected_q`, a state, a
        threshold, a certification or an empirical screening sequence.
        """
        key = (str(site), str(method))
        if key not in self._source_taus:
            raise KeyError(f"no evaluation inputs for {key}")
        return {
            "taus": self.candidate_grid(key[0], key[1], strategy),
            "r_star": float(self._r_star[key[0]]),
            "oracle": True,
        }

    # -- authoritative accessors for the post-phase-2 section derivation ------
    def r_star_for(self, site: str) -> float:
        return float(self._r_star[str(site)])

    def n0_for(self, site: str) -> float:
        return float(self._meta.n0_for(str(site)))

    def source_class_counts(self, site: str, method: str) -> list:
        key = (str(site), str(method))
        if key not in self._source_counts:
            raise KeyError(f"no source class counts for {key}")
        return [list(map(int, row))
                for row in self._source_counts[key]["source_class_counts"]]

    def point_sections(self) -> dict:
        """The two point-estimate sections that are functions of the **inputs**.

        The other eighteen registered names are functions of the formal
        decisions and evaluations, which an input provider has not seen and may
        not invent; `s2_point_sections` derives those after phase 2 and no
        placeholder is emitted for them here.

        This is **not** a runner argument. Stage 2.1b removed the report's
        `point_sections` parameter, so nothing a caller holds reaches a section:
        `s2_point_sections.derive_authoritative_formal_point_sections` builds
        its own provider through `FrozenS2Inputs.from_frozen_context` and calls
        the two builders below.
        """
        return {"meta_objective_table": self.meta_objective_section(),
                "ess_and_weight_diagnostics":
                    self.ess_and_weight_diagnostics_section()}

    def meta_objective_section(self) -> dict:
        """§24.9 — the `d(s, n0)` table, per-fold `n0`, references, support."""
        return {
            "d_by_site_and_n0": self._meta.objective_table().to_dict("records"),
            "selected_n0_by_site": dict(self._meta.selected),
            "n0_grid": list(shr.N0_GRID),
            "tie_break_rule": shr.N0_TIE_BREAK,
            "meta_repeats": int(shr.R_META),
            "meta_split_entropy": list(shr.META_SPLIT_ENTROPY),
            "common_support": self._meta.support_diagnostics().to_dict("records"),
        }

    def ess_and_weight_diagnostics_section(self) -> dict:
        """§24.10 — the complete frozen S1 ESS and weight diagnostics.

        Three layers, none of them recomputed: the per-fold artifact row of
        `ess_by_site.csv` in full (overall ESS and its fraction, the weight
        order statistics, the low and high clipping fractions, the post-clip
        weight mean, the per-method retained-ESS summaries at OP50/OP30 and the
        frozen exclusion rule); the selected `C_domain` per fold; and the
        per-candidate retained counts, effective sample sizes, ESS ratios,
        class weight sums, weighted class error rates and weighted/unweighted
        coverage that the deployed `tau_q^S1` sequence produces — each of which
        was checked against `weighted_source_risk.csv` at construction.
        """
        return {
            "ess_by_site": {site: dict(row)
                            for site, row in self._ess_by_site.items()},
            "ess_by_site_columns": list(self.ESS_BY_SITE_COLUMNS),
            "per_candidate_by_site_and_method": {
                f"{site}|{method}": {
                    field: list(self._weighted.for_cell(site, method)[field])
                    for field in self.PER_CANDIDATE_WEIGHT_FIELDS}
                for site in contracts.FROZEN_SITES
                for method in cert.CONFIDENCE_METHODS},
            "per_candidate_fields": list(self.PER_CANDIDATE_WEIGHT_FIELDS),
            "q_values": list(cert.Q_GRID),
            "weighted_taus_by_site_and_method": {
                f"{site}|{method}": [float(t)
                                     for t in self._weighted_taus[(site, method)]]
                for site in contracts.FROZEN_SITES
                for method in cert.CONFIDENCE_METHODS},
            "c_domain_by_site": dict(self._domain_c),
            "n_frozen_s1_reproduction_checks":
                self._weighted.n_reproduction_checks,
        }

    # -- mechanical summary --------------------------------------------------
    def summary(self) -> dict:
        """One machine-checkable record of everything the provider derived."""
        states = {}
        for payload in self._zero_label:
            k = (payload["strategy"], payload["state"])
            states[k] = states.get(k, 0) + 1
        counts = {k: v["source_class_counts"] for k, v in self._source_counts.items()}
        flat = [n for rows in counts.values() for row in rows for n in row]
        return {
            "upstream": self.provenance,
            "gate1_binding": self.gate1_binding,
            "r_star": {k: v for k, v in self._r_star_result.items()
                       if k != "r_star"},
            "r_star_values": dict(self._r_star),
            "n_threshold_cells": len(self._source_taus),
            "n_weighted_threshold_cells": len(self._weighted_taus),
            "n_candidates_per_cell": N_CANDIDATES,
            "source_class_count_cells": len(counts),
            "source_class_count_rows": sum(len(v) for v in counts.values()),
            "source_class_count_min": int(min(flat)),
            "source_class_count_max": int(max(flat)),
            "n_zero_label_decisions": len(self._zero_label),
            "zero_label_states": {f"{s}|{st}": n for (s, st), n in
                                  sorted(states.items())},
            "n0_by_site": dict(self._meta.selected),
            "n0_objective_table_shape": list(self._meta.objective_table().shape),
            "n_weighted_reproduction_checks": self._weighted.n_reproduction_checks,
            "point_sections": sorted(self.point_sections()),
        }


# ---------------------------------------------------------------------------
# static audit — production inputs may not reach a test fixture
# ---------------------------------------------------------------------------
#: The formal import graph. A production input provider that imported a test
#: fixture would put `[400, 120, 400, 120]` back inside a formal run.
PRODUCTION_INPUT_MODULES = ("src/s2_inputs.py", "scripts/run_rq2_s2.py",
                            "scripts/execute_s2_input_preflight.py")
FORBIDDEN_FIXTURE_MODULES = ("_s2_fixtures", "_s2_test_worker", "tests")


def audit_production_inputs_have_no_fixture_import(root=ROOT) -> dict:
    """No formal input module may name a tests-only provider."""
    import ast

    findings = []
    for rel in PRODUCTION_INPUT_MODULES:
        path = Path(root) / rel
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = {a.name for a in node.names}
                for bad in FORBIDDEN_FIXTURE_MODULES:
                    if module.split(".")[0] == bad or bad in names:
                        findings.append((rel, bad))
    return {"ok": not findings, "findings": findings}


def audit_inputs_never_open_target_outcomes(root=ROOT) -> dict:
    """`s2_inputs` may not name the production target-outcome source."""
    path = Path(root) / "src" / "s2_inputs.py"
    text = path.read_text()
    findings = []
    # spelled from parts so the audit does not flag its own token list
    tokens = ("confidence_" + "target.csv",
              "load_authoritative_target_outcomes" + "_for_phase2",
              "_materialize_outcome" + "_table", "s2_" + "outcomes")
    for token in tokens:
        if token in text:
            findings.append(("src/s2_inputs.py", token))
    return {"ok": not findings, "findings": findings}


def run_input_static_audits(root=ROOT) -> list:
    audits = (("production_inputs_have_no_fixture_import",
               audit_production_inputs_have_no_fixture_import),
              ("inputs_never_open_target_outcomes",
               audit_inputs_never_open_target_outcomes))
    rows = []
    for name, fn in audits:
        result = fn(root)
        rows.append({"audit": name,
                     "status": "PASS" if result["ok"] else "FAIL",
                     "findings": result["findings"]})
    return rows


# ---------------------------------------------------------------------------
# static audits — the Stage 2.1b execution-wiring closure
# ---------------------------------------------------------------------------
#: Kept as its own table rather than appended to `s2_firewall.run_static_audits`,
#: whose count and membership §24.13 froze at sixteen.
def _function_def(path: Path, name: str):
    import ast

    tree = __import__("ast").parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _signature_names(node) -> dict:
    return {
        "positional": [a.arg for a in
                       list(getattr(node.args, "posonlyargs", []))
                       + list(node.args.args)],
        "keyword_only": [a.arg for a in node.args.kwonlyargs],
        "vararg": node.args.vararg.arg if node.args.vararg else None,
        "kwarg": node.args.kwarg.arg if node.args.kwarg else None,
    }


#: Names that would put a caller-controlled scientific provider back into the
#: production entry points.
FORBIDDEN_PROVIDER_PARAMS = (
    "inputs", "input", "provider", "inputs_provider", "candidate_provider",
    "evaluation_provider", "point_sections", "sections", "section_dict",
    "fixture", "fixtures", "input_factory", "factory", "candidate_inputs",
    "evaluation_inputs", "oracle_summary", "diagnostics", "point_estimates",
)


def audit_production_runner_takes_no_input_provider(root=ROOT) -> dict:
    """`run_formal` must construct its provider, not receive one."""
    path = Path(root) / "scripts" / "run_rq2_s2.py"
    node = _function_def(path, "run_formal")
    if node is None:
        return {"ok": False, "findings": [("scripts/run_rq2_s2.py",
                                           "run_formal is missing")]}
    sig = _signature_names(node)
    findings = []
    if sig["positional"] != ["context"]:
        findings.append(("run_formal positional", sig["positional"]))
    if sig["keyword_only"] or sig["vararg"] or sig["kwarg"]:
        findings.append(("run_formal extra parameters",
                         [sig["keyword_only"], sig["vararg"], sig["kwarg"]]))
    text = path.read_text()
    if "FrozenS2Inputs.from_frozen_context" not in text:
        findings.append(("scripts/run_rq2_s2.py",
                         "run_formal does not construct FrozenS2Inputs"))
    return {"ok": not findings, "findings": findings}


def audit_report_takes_no_point_sections(root=ROOT) -> dict:
    """The formal report may not ingest caller-computed sections."""
    path = Path(root) / "src" / "s2_reporting.py"
    node = _function_def(path, "build_formal_s2_report")
    if node is None:
        return {"ok": False, "findings": [("src/s2_reporting.py",
                                           "build_formal_s2_report is missing")]}
    sig = _signature_names(node)
    findings = []
    if sig["positional"] != ["context", "ledger"]:
        findings.append(("build_formal_s2_report positional", sig["positional"]))
    if sig["keyword_only"] or sig["vararg"] or sig["kwarg"]:
        findings.append(("build_formal_s2_report extra parameters",
                         [sig["keyword_only"], sig["vararg"], sig["kwarg"]]))
    if "derive_authoritative_formal_point_sections" not in path.read_text():
        findings.append(("src/s2_reporting.py",
                         "the report does not derive its own sections"))
    return {"ok": not findings, "findings": findings}


def audit_evaluation_inputs_are_strategy_aware(root=ROOT) -> dict:
    """`evaluation_inputs` must carry the strategy axis §7 makes it need."""
    path = Path(root) / "src" / "s2_inputs.py"
    node = _function_def(path, "evaluation_inputs")
    if node is None:
        return {"ok": False, "findings": [("src/s2_inputs.py",
                                           "evaluation_inputs is missing")]}
    sig = _signature_names(node)
    findings = []
    if sig["positional"] != ["self", "site", "method", "strategy"]:
        findings.append(("evaluation_inputs", sig["positional"]))
    if set(STRATEGY_THRESHOLD_GRID) != set(cert.STRATEGIES):
        findings.append(("STRATEGY_THRESHOLD_GRID",
                         sorted(STRATEGY_THRESHOLD_GRID)))
    return {"ok": not findings, "findings": findings}


#: The only public names `s2_point_sections` exposes. A public function taking a
#: payload would be a second way into a formal section.
POINT_SECTION_PUBLIC_API = (
    "derive_authoritative_formal_point_sections", "assert_sections_complete",
    "preflight_section_schema", "section_registry_table",
)


def audit_point_sections_are_internally_derived(root=ROOT) -> dict:
    """No caller array, frame or section dict may reach a formal section."""
    import ast

    path = Path(root) / "src" / "s2_point_sections.py"
    if not path.exists():
        return {"ok": False, "findings": [("src/s2_point_sections.py",
                                           "missing")]}
    tree = ast.parse(path.read_text())
    findings = []
    public = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            public.append(node.name)
            if node.name == "derive_authoritative_formal_point_sections":
                sig = _signature_names(node)
                if (sig["positional"] != ["context", "ledger"]
                        or sig["keyword_only"] or sig["vararg"]
                        or sig["kwarg"]):
                    findings.append(("derive_authoritative_formal_point_sections",
                                     sig))
    if sorted(public) != sorted(POINT_SECTION_PUBLIC_API):
        findings.append(("public api", sorted(public)))
    for other in (ast.Import, ast.ImportFrom):
        for node in ast.walk(tree):
            if isinstance(node, other):
                module = getattr(node, "module", "") or ""
                names = {a.name for a in node.names}
                for bad in FORBIDDEN_FIXTURE_MODULES:
                    if module.split(".")[0] == bad or bad in names:
                        findings.append(("src/s2_point_sections.py", bad))
    return {"ok": not findings, "findings": findings}


def run_execution_wiring_audits(root=ROOT) -> list:
    audits = (
        ("production_runner_takes_no_input_provider",
         audit_production_runner_takes_no_input_provider),
        ("report_takes_no_point_sections", audit_report_takes_no_point_sections),
        ("evaluation_inputs_are_strategy_aware",
         audit_evaluation_inputs_are_strategy_aware),
        ("point_sections_are_internally_derived",
         audit_point_sections_are_internally_derived),
    )
    rows = []
    for name, fn in audits:
        result = fn(root)
        rows.append({"audit": name,
                     "status": "PASS" if result["ok"] else "FAIL",
                     "findings": result["findings"]})
    return rows
