"""Stage-F consumed-input binding and preflight (§B.1, §B.1a, §J.1).

Fail-closed by construction. Every one of the **17** frozen artifacts of §B.1 is
verified before any Stage-F computation begins, and **path existence is never
accepted as evidence**:

* digest-pinned artifacts must hash to the value frozen in the protocol;
* the S0/S1 artifacts must additionally match their own frozen result ledgers,
  which is done by invoking the frozen upstream verifier
  `src/s2_inputs.verify_frozen_upstream` rather than by re-implementing it;
* the two Gate-1 artifacts are **content-bound** to ledger-bound S0 artifacts by
  the genuine frozen helper `src/s2_inputs.assert_gate1_binds_to_checksummed_s0`
  — seven bit-identical columns plus `oof_prob_mdd` within `4e-16` (§B.1a). The
  helper is *invoked*, never approximated, and its own frozen constants are
  checked against the protocol transcription first;
* the SVM feature cache must match its digest **and** its `feature_source_id`
  content key, in the cohort's own order.

This module fits nothing, computes no Stage-F scientific quantity and writes
nothing.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

import certification as cert
import contracts
import fc_features
import s2_inputs as si
import s2_splitting as spl
import stage_f_registry as reg

ROOT = reg.ROOT


class StageFInputError(AssertionError):
    """A frozen Stage-F input cannot support a Stage-F derivation."""


# ---------------------------------------------------------------------------
# protocol identity
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    out = subprocess.run(("git", "-C", str(ROOT)) + args,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise StageFInputError(
            f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def assert_protocol_tag() -> dict:
    """`stage-f-protocol-v2` exists, is annotated and targets the frozen commit.

    The runtime **requires** the tag — the opposite of the protocol validator's
    pre-freeze lifecycle guard, which required its absence while the draft was
    unaudited. That guard belongs to the pre-tag lifecycle and is not a
    scientific check; it is neither consulted nor modified here.

    The authority is `STAGE_F_PROTOCOL/2.0.0` at `stage-f-protocol-v2`, and it
    is the **only** one: `stage-f-protocol-v1` is immutable historical evidence
    (§H.1a) and is never accepted as a fallback, never resolved by preference
    order, and never consulted when the `2.0.0` tag is missing. A missing or
    moved tag is a hard failure, not a reason to reach for the older one.
    """
    tags = _git("tag", "-l", reg.PROTOCOL_TAG).splitlines()
    if reg.PROTOCOL_TAG not in tags:
        raise StageFInputError(
            f"{reg.PROTOCOL_TAG} does not exist; Stage F.2 implements a frozen "
            "protocol and may not run against an untagged draft")
    kind = _git("cat-file", "-t", reg.PROTOCOL_TAG)
    if kind != "tag":
        raise StageFInputError(
            f"{reg.PROTOCOL_TAG} is a {kind}, not an annotated tag object")
    target = _git("rev-parse", f"{reg.PROTOCOL_TAG}^{{}}")
    if target != reg.PROTOCOL_COMMIT:
        raise StageFInputError(
            f"{reg.PROTOCOL_TAG} targets {target}, the frozen protocol commit "
            f"is {reg.PROTOCOL_COMMIT}")
    return {"tag": reg.PROTOCOL_TAG, "annotated": True, "target": target}


# ---------------------------------------------------------------------------
# the 17 consumed artifacts
# ---------------------------------------------------------------------------
def verify_consumed_artifacts() -> dict:
    """Existence + pinned SHA256 for all 17 §B.1 artifacts. Hard-fails on drift."""
    rows = []
    for relpath, expected in sorted(reg.CONSUMED_ARTIFACTS.items()):
        path = ROOT / relpath
        if not path.exists():
            raise StageFInputError(f"consumed artifact {relpath} is not on disk")
        digest = reg.sha256_file(path)
        if digest != expected:
            raise StageFInputError(
                f"{relpath}: SHA256 {digest[:12]}... does not match the frozen "
                f"protocol pin {expected[:12]}...")
        rows.append({"path": relpath, "sha256": digest,
                     "recomputable": relpath in reg.RECOMPUTABLE_ARTIFACTS})
    if len(rows) != reg.N_CONSUMED_ARTIFACTS:
        raise StageFInputError(
            f"the consumed registry holds {len(rows)} artifacts, the protocol "
            f"freezes {reg.N_CONSUMED_ARTIFACTS}")
    return {"artifacts": rows, "n_artifacts": len(rows)}


def verify_upstream_ledgers() -> dict:
    """The S0/S1 artifacts additionally match their own frozen result ledgers.

    Delegated to the frozen upstream verifier; Stage F re-implements no ledger
    parsing of its own.
    """
    return si.verify_frozen_upstream()


def assert_gate1_binding() -> dict:
    """Invoke the genuine frozen Gate-1 binding helper, then bind its constants.

    The protocol transcription of the rule (§B.1a) is checked against the
    helper's own frozen constants **before** the helper runs, so a silently
    widened tolerance or a shortened exact-column list cannot pass unnoticed.
    """
    if tuple(si.GATE1_OOF_EXACT_COLUMNS) != reg.GATE1_EXACT_COLUMNS:
        raise StageFInputError(
            "src/s2_inputs.GATE1_OOF_EXACT_COLUMNS no longer equals the §B.1a "
            f"transcription: {si.GATE1_OOF_EXACT_COLUMNS}")
    if tuple(si.GATE1_OOF_ULP_COLUMNS) != reg.GATE1_TOLERANCE_COLUMNS:
        raise StageFInputError(
            "src/s2_inputs.GATE1_OOF_ULP_COLUMNS no longer equals the §B.1a "
            f"transcription: {si.GATE1_OOF_ULP_COLUMNS}")
    if float(si.GATE1_OOF_ULP_ATOL) != reg.GATE1_OOF_PROB_MDD_ATOL:
        raise StageFInputError(
            f"the frozen oof_prob_mdd tolerance moved to {si.GATE1_OOF_ULP_ATOL}; "
            f"§B.1a transcribes {reg.GATE1_OOF_PROB_MDD_ATOL}")
    if float(si.R_STAR_CROSSCHECK_ATOL) != reg.GATE1_R_STAR_CROSSCHECK_ATOL:
        raise StageFInputError(
            f"the frozen r* cross-check tolerance moved to "
            f"{si.R_STAR_CROSSCHECK_ATOL}")

    stats = si.assert_gate1_binds_to_checksummed_s0()

    if int(stats["n_oof_rows"]) != reg.GATE1_N_OOF_ROWS:
        raise StageFInputError(
            f"the Gate-1 binding covers {stats['n_oof_rows']} rows, the frozen "
            f"count is {reg.GATE1_N_OOF_ROWS}")
    if int(stats["n_exact_columns"]) != reg.GATE1_N_EXACT_COLUMNS:
        raise StageFInputError(
            f"the binding compared {stats['n_exact_columns']} exact columns, "
            f"not {reg.GATE1_N_EXACT_COLUMNS}")
    if float(stats["max_ulp_delta"]) > reg.GATE1_OOF_PROB_MDD_ATOL:
        raise StageFInputError(
            f"oof_prob_mdd disagrees by {stats['max_ulp_delta']:g}, beyond the "
            f"frozen {reg.GATE1_OOF_PROB_MDD_ATOL:g}")
    if int(stats["n_r_star_crosscheck_rows"]) != reg.GATE1_R_STAR_CROSSCHECK_SITES:
        raise StageFInputError(
            f"the r* cross-check resolves {stats['n_r_star_crosscheck_rows']} "
            f"sites, not {reg.GATE1_R_STAR_CROSSCHECK_SITES}")

    # The logical key must be duplicate-free and the binding one-to-one. The
    # helper obtains this implicitly (both key columns are inside the seven
    # bit-identical columns, so positional equality *is* row alignment); it is
    # asserted explicitly here because an implicit guarantee is not an audited
    # one (§B.1a).
    pred = pd.read_csv(ROOT / reg.GATE1_UNLEDGERED_FILE,
                       usecols=list(reg.GATE1_LOGICAL_KEY),
                       float_precision=reg.GATE1_CSV_FLOAT_PRECISION)
    conf = pd.read_csv(ROOT / reg.GATE1_LEDGER_BOUND_FILE,
                       usecols=list(reg.GATE1_LOGICAL_KEY),
                       float_precision=reg.GATE1_CSV_FLOAT_PRECISION)
    for name, frame in (("gate1", pred), ("s0", conf)):
        if frame.duplicated(subset=list(reg.GATE1_LOGICAL_KEY)).any():
            raise StageFInputError(
                f"the {name} table carries a duplicate "
                f"{reg.GATE1_LOGICAL_KEY} logical key")
    for col in reg.GATE1_LOGICAL_KEY:
        if not pred[col].equals(conf[col]):
            raise StageFInputError(
                f"the Gate-1 binding is not row-aligned on {col!r}")

    return {**{k: (float(v) if k == "max_ulp_delta" else int(v))
               for k, v in stats.items()},
            "duplicate_free": True, "one_to_one": True}


def assert_fc_cache_identity(cohort: pd.DataFrame | None = None) -> dict:
    """The SVM feature cache matches its digest **and** its content key.

    A `feature_source_id` mismatch is a HARD FAIL, never a silent rebuild: the
    protocol registers the cache as recomputable but forbids regenerating it
    while a valid cache exists (§B.1 #3).
    """
    relpath = "results/gate1/cache/fc_aal116_z_included.npz"
    path = ROOT / relpath
    if not path.exists():
        raise StageFInputError(
            f"the frozen FC cache {relpath} is absent; deterministic "
            "reconstruction is permitted only under the frozen feature "
            "identity rules of src/fc_features.build_fc_matrix, and never "
            "with a redefined feature")
    digest = reg.sha256_file(path)
    if digest != reg.CONSUMED_ARTIFACTS[relpath]:
        raise StageFInputError(
            f"{relpath}: SHA256 {digest[:12]}... does not match the frozen pin")

    frame = fc_features.load_cohort() if cohort is None else cohort
    ids = frame["feature_source_id"].to_numpy().astype("U32")
    with np.load(path, allow_pickle=False) as blob:
        cached = list(blob["feature_source_id"])
        shape = tuple(blob["X"].shape)
    if list(ids) != cached:
        raise StageFInputError(
            "the FC cache content key does not equal the cohort's ordered "
            "feature_source_id list; this is a hard failure, not a rebuild")
    if shape != (reg.BASE_COHORT_N, contracts.N_FEATURES):
        raise StageFInputError(f"the FC cache holds X of shape {shape}")
    return {"path": relpath, "sha256": digest, "n_rows": shape[0],
            "n_features": shape[1], "content_key_verified": True}


# ---------------------------------------------------------------------------
# cohort, splits and probability hygiene
# ---------------------------------------------------------------------------
def load_cohort() -> pd.DataFrame:
    """The frozen 2232 / 23 analysed cohort, re-derived and asserted (C1)."""
    cohort = fc_features.load_cohort()
    contracts.assert_frozen_cohort(cohort)
    contracts.assert_unique_acquisitions(cohort)
    return cohort


def assert_common_sites_present(cohort: pd.DataFrame) -> tuple:
    missing = [s for s in reg.COMMON_SITES if s not in set(cohort["site"])]
    if missing:
        raise StageFInputError(f"SENS-4 common sites absent from the cohort: {missing}")
    return reg.COMMON_SITES


#: The one authority on which sites are eligible at each label budget `N`.
SPLIT_REGISTRY_FILE = "results/rq2_s2/production_split_registry.csv"


def load_split_registry() -> spl.SplitRegistry:
    """The frozen RQ2/S2 production split registry, checksum-bound (§D.9)."""
    return spl.SplitRegistry.load(ROOT / SPLIT_REGISTRY_FILE,
                                  reg.CONSUMED_ARTIFACTS[SPLIT_REGISTRY_FILE])


def _registry_frame(registry) -> pd.DataFrame:
    return registry.frame if hasattr(registry, "frame") else registry._frame


def verify_split_registry(registry: spl.SplitRegistry | None = None) -> dict:
    """7700 rows, 23/22/21/11 sites per `N`, 100 repeats, every row disjoint.

    `cal ∩ test = ∅`, `|cal| = N` and `|cal| + |test| = n_acquisitions` are
    re-asserted for **every** row, not a sample: `SplitRegistry.cell` performs
    exactly those checks and is called once per registered cell.
    """
    registry = load_split_registry() if registry is None else registry
    frame = _registry_frame(registry)
    if len(frame) != reg.SPLIT_REGISTRY_ROWS:
        raise StageFInputError(
            f"the split registry holds {len(frame)} rows, frozen is "
            f"{reg.SPLIT_REGISTRY_ROWS}")
    per_n = {int(n): int(block["site"].nunique())
             for n, block in frame.groupby("n_labels")}
    if per_n != reg.SITES_PER_N:
        raise StageFInputError(
            f"per-N site counts are {per_n}, frozen is {reg.SITES_PER_N}")
    repeats = sorted(frame["repeat"].unique())
    if repeats != list(range(reg.R_SPLIT)):
        raise StageFInputError(
            f"the registry holds {len(repeats)} repeats, frozen is {reg.R_SPLIT}")

    n_cells = 0
    for row in frame.itertuples():
        registry.cell(str(row.site), int(row.repeat), int(row.n_labels))
        n_cells += 1
    return {"n_rows": int(len(frame)), "sites_per_n": per_n,
            "n_repeats": len(repeats), "n_cells_verified": n_cells}


class VerifiedSplitIdentities(NamedTuple):
    """Which sites the *verified* split registry admits at each label budget.

    A canonical, immutable representation of one successful
    `verify_split_registry` call. It exists because verifying the registry and
    then persisting some other site set is not a contradiction any count, any
    universe-membership test or any cross-table comparison can detect: all four
    survive substituting one eligible site for another. The identities have to
    be carried forward from the verification itself and compared.

    It is a `NamedTuple` of tuples, so there is no field to reassign and no
    container to insert into after the fact, and `sites_at` narrows — it never
    widens — the mapping it was built from.
    """

    sites_by_n: tuple          # ((n, (site, ...)), ...), sorted, canonical
    n_rows: int
    n_cells_verified: int
    registry_path: str
    registry_sha256: str
    verified_utc: str

    def n_values(self) -> tuple:
        return tuple(n for n, _ in self.sites_by_n)

    def sites_at(self, n_labels) -> frozenset:
        for n, sites in self.sites_by_n:
            if n == int(n_labels):
                return frozenset(sites)
        raise StageFInputError(
            f"the verified split registry admits no label budget N={n_labels!r}; "
            f"it holds {self.n_values()}")

    def as_report(self) -> dict:
        return {"registry_path": self.registry_path,
                "registry_sha256": self.registry_sha256,
                "n_rows": self.n_rows,
                "n_cells_verified": self.n_cells_verified,
                "verified_utc": self.verified_utc,
                "eligible_sites_by_n": {str(n): list(sites)
                                        for n, sites in self.sites_by_n}}


#: Derived identities, keyed by the **content digest** of the registry they came
#: from. This is a memo of a pure function of that content, not a mutable global
#: the run depends on: a key can never name identities derived from different
#: bytes, and `derive_verified_split_identities(use_cache=False)` — which the
#: completion barrier uses — ignores it entirely and re-verifies from disk.
_VERIFIED_IDENTITY_CACHE: dict = {}


def derive_verified_split_identities(registry=None, *, use_cache: bool = True
                                     ) -> VerifiedSplitIdentities:
    """Verify the production split registry, then derive its eligible identities.

    There is no path here that derives identities without verifying: the full
    `verify_split_registry` gate — row count, per-`N` site counts, repeat count
    and every row's disjointness — runs first, and the identities are read from
    the very frame it verified. A caller cannot supply the identities.
    """
    path = ROOT / SPLIT_REGISTRY_FILE
    digest = reg.sha256_file(path) if registry is None else None
    if digest is not None and use_cache:
        cached = _VERIFIED_IDENTITY_CACHE.get(digest)
        if cached is not None:
            return cached
    if registry is None:
        registry = load_split_registry()       # re-checks the frozen pin itself
        pinned = reg.CONSUMED_ARTIFACTS[SPLIT_REGISTRY_FILE]
        if digest != pinned:                   # defence in depth
            raise StageFInputError(
                f"{SPLIT_REGISTRY_FILE}: SHA256 {digest[:12]}... is not the "
                f"frozen pin {pinned[:12]}...")

    report = verify_split_registry(registry)
    frame = _registry_frame(registry)
    sites_by_n = tuple(sorted(
        (int(n), tuple(sorted(set(block["site"].astype(str)))))
        for n, block in frame.groupby("n_labels")))

    observed = {n: sites for n, sites in sites_by_n}
    if sorted(observed) != sorted(reg.SITES_PER_N):
        raise StageFInputError(
            f"the split registry holds label budgets {sorted(observed)}; the "
            f"frozen grid is {sorted(reg.SITES_PER_N)}")
    universe = set(contracts.FROZEN_SITES)
    for n, sites in sites_by_n:
        if len(sites) != len(set(sites)) or len(sites) != reg.SITES_PER_N[n]:
            raise StageFInputError(
                f"the split registry admits {len(set(sites))} distinct sites at "
                f"N={n}; the frozen count is {reg.SITES_PER_N[n]}")
        stray = sorted(set(sites) - universe)
        if stray:
            raise StageFInputError(
                f"the split registry admits {stray} at N={n}, which are not "
                f"among the {len(universe)} frozen eligible sites")
    for (small, wide), (large, narrow) in zip(sites_by_n, sites_by_n[1:]):
        if not set(narrow) <= set(wide):
            raise StageFInputError(
                f"the split registry admits {sorted(set(narrow) - set(wide))} at "
                f"N={large} but not at N={small}; eligibility at a larger label "
                "budget cannot be easier (§D.6)")

    identities = VerifiedSplitIdentities(
        sites_by_n=sites_by_n,
        n_rows=int(report["n_rows"]),
        n_cells_verified=int(report["n_cells_verified"]),
        registry_path=SPLIT_REGISTRY_FILE,
        registry_sha256=str(digest) if digest is not None else "",
        verified_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if digest is not None:
        _VERIFIED_IDENTITY_CACHE[digest] = identities
    return identities


def assert_density_weight_keys_match() -> dict:
    """The density-weight and source-OOF key sets are exactly equal (§D.8)."""
    weights = pd.read_csv(ROOT / "results/rq1_s1/density_weights.csv",
                          float_precision="round_trip")
    oof = pd.read_csv(ROOT / "results/rq1_s0/confidence_source_oof.csv",
                      usecols=["outer_fold_held_out_site", "canonical_id"],
                      float_precision="round_trip")
    left = set(zip(weights["held_out_site"].astype(str),
                   weights["canonical_id"].astype(str)))
    right = set(zip(oof["outer_fold_held_out_site"].astype(str),
                    oof["canonical_id"].astype(str)))
    if left != right:
        raise StageFInputError(
            f"the density-weight and source-OOF key sets differ: "
            f"{len(left - right)} weight-only, {len(right - left)} OOF-only")
    if len(left) != reg.SOURCE_OOF_ROWS:
        raise StageFInputError(
            f"the shared key set holds {len(left)} rows, frozen is "
            f"{reg.SOURCE_OOF_ROWS}")
    w = weights[reg.WEIGHT_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise StageFInputError("a frozen density weight is non-positive or non-finite")
    return {"n_keys": len(left), "weight_column": reg.WEIGHT_COLUMN}


def assert_probabilities_finite() -> dict:
    """Every consumed probability column is finite and inside `[0, 1]` (§J.1.6)."""
    checked = {}
    for relpath, column, expected_rows in (
            ("results/rq1_s0/confidence_source_oof.csv", "oof_prob_mdd",
             reg.SOURCE_OOF_ROWS),
            ("results/rq1_s0/confidence_target.csv", "prob_mdd", reg.TARGET_ROWS)):
        frame = pd.read_csv(ROOT / relpath, usecols=[column],
                            float_precision="round_trip")
        values = frame[column].to_numpy(dtype=float)
        if len(values) != expected_rows:
            raise StageFInputError(
                f"{relpath} holds {len(values)} rows, frozen is {expected_rows}")
        if not np.isfinite(values).all():
            raise StageFInputError(f"{relpath}:{column} holds a non-finite value")
        if (values < 0.0).any() or (values > 1.0).any():
            raise StageFInputError(f"{relpath}:{column} leaves [0, 1]")
        checked[f"{relpath}:{column}"] = int(len(values))
    return checked


def load_target_confidence() -> pd.DataFrame:
    """`confidence_target.csv` — the target-side predictions Stage F consumes."""
    frame = pd.read_csv(ROOT / "results/rq1_s0/confidence_target.csv",
                        float_precision="round_trip")
    required = ("held_out_site", "canonical_id", "y_true", "prob_mdd", "pred") \
        + tuple(f"confidence_{m}" for m in cert.CONFIDENCE_METHODS)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise StageFInputError(f"confidence_target.csv lacks {missing}")
    if frame.duplicated(subset=["canonical_id"]).any():
        raise StageFInputError("confidence_target.csv holds a duplicate canonical_id")
    return frame


def load_source_oof_confidence() -> pd.DataFrame:
    """`confidence_source_oof.csv` — the frozen source-OOF calibration scores."""
    frame = pd.read_csv(ROOT / "results/rq1_s0/confidence_source_oof.csv",
                        float_precision="round_trip")
    required = ("outer_fold_held_out_site", "canonical_id", "source_site",
                "y_true", "oof_prob_mdd")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise StageFInputError(f"confidence_source_oof.csv lacks {missing}")
    return frame


#: The relative path `scripts/run_rq1_s0.py` opens to build the frozen S0
#: threshold table, and the parent grouping key of `build_thresholds`.
PARENT_S0_CONFIDENCE_FILE = "results/rq1_s0/confidence_source_oof.csv"
PARENT_S0_FOLD_COLUMN = "outer_fold_held_out_site"


def load_parent_s0_threshold_confidence() -> pd.DataFrame:
    """The source-OOF frame **exactly as the RQ1/S0 threshold path read it**.

    `scripts/run_rq1_s0.py:771` reads this table with `pandas.read_csv(path)` —
    the parser's *default* float conversion — and hands the result straight to
    `build_thresholds`, which is what produced the frozen
    `thresholds_source_oof.csv`. That default conversion is not exact: it can
    land one to four ULP away from the decimal in the file. Reading the same
    file with `float_precision="round_trip"` therefore yields a *different*
    float64 vector, and quantiles of it disagree with the frozen table in the
    last place.

    Neither vector is "more correct" — but only one of them is the mechanism
    that produced the frozen parent output, and §E.8 requires a sensitivity to
    reuse the parent's own parameterised mechanism. So this helper reproduces
    the parent read verbatim and is the **only** admissible source of
    confidences for re-deriving an S0 threshold.

    The distinction is confined to threshold derivation. Every other Stage-F
    consumer of this table keeps `float_precision="round_trip"`, because that
    is what *its* parent does: `s2_inputs` reads the confidences round-trip and
    the frozen thresholds round-trip, and RQ1/S1 derives `tau_S1` from
    round-trip confidences (`s1_data_access._read_rows`). Re-deriving an S1
    threshold from this frame would break the S1 grid exactly as re-deriving an
    S0 threshold from the round-trip frame breaks the S0 grid.
    """
    return pd.read_csv(ROOT / PARENT_S0_CONFIDENCE_FILE)


def parent_s0_threshold_confidence() -> dict:
    """`{(site, method): confidence}` in the parent's own fold grouping.

    `run_rq1_s0.build_thresholds` selects a fold's rows with
    `oof[oof["outer_fold_held_out_site"] == site]` and takes the
    `confidence_<method>` column as it stands, so the rows keep file order and
    no re-sort is applied here either. Order is immaterial to
    `numpy.quantile`, but reproducing the parent means not editorialising.
    """
    frame = load_parent_s0_threshold_confidence()
    required = (PARENT_S0_FOLD_COLUMN,) + tuple(
        f"confidence_{m}" for m in cert.CONFIDENCE_METHODS)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise StageFInputError(
            f"{PARENT_S0_CONFIDENCE_FILE} lacks {missing}; the RQ1/S0 threshold "
            "path cannot be reproduced")
    folds = frame[PARENT_S0_FOLD_COLUMN].astype(str).to_numpy()
    if set(folds) != set(contracts.FROZEN_SITES):
        raise StageFInputError(
            "the parent S0 confidence table does not cover exactly the 23 "
            "frozen outer folds")
    out = {}
    for site in contracts.FROZEN_SITES:
        rows = np.flatnonzero(folds == site)
        for method in cert.CONFIDENCE_METHODS:
            out[(site, method)] = \
                frame[f"confidence_{method}"].to_numpy(dtype=float)[rows]
    return out


def load_density_weights() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "results/rq1_s1/density_weights.csv",
                        float_precision="round_trip")
    for col in ("held_out_site", "canonical_id", reg.WEIGHT_COLUMN):
        if col not in frame.columns:
            raise StageFInputError(f"density_weights.csv lacks {col!r}")
    return frame


def operating_point_coverage_map() -> dict:
    """`{OP50: 0.50, OP30: 0.70}`, read from the frozen OP summaries (§E.7.1).

    Mechanically established, never chosen here: the map is the frozen
    `source_intended_coverage` column of the two OP summaries, and the frozen
    transcription is asserted against it.
    """
    observed = {}
    for op, name in (("OP50", "op50_summary.csv"), ("OP30", "op30_summary.csv")):
        frame = pd.read_csv(ROOT / "results/rq1_s0" / name)
        rows = frame[frame["operating_point"] == op]
        if rows.empty:
            raise StageFInputError(f"{name} carries no {op} row")
        values = sorted(set(rows["source_intended_coverage"].astype(float)))
        if len(values) != 1:
            raise StageFInputError(
                f"{name} carries {len(values)} distinct source_intended_coverage "
                f"values for {op}")
        observed[op] = float(values[0])
    if observed != dict(reg.SENS6_REQUESTED_COVERAGE):
        raise StageFInputError(
            f"the frozen OP summaries give {observed}; §E.7.1 transcribes "
            f"{dict(reg.SENS6_REQUESTED_COVERAGE)}")
    return observed


# ---------------------------------------------------------------------------
# §J.1 preflight — the complete ordered gate
# ---------------------------------------------------------------------------
def preflight(*, require_empty_store: bool = True,
              store_root: Path | None = None) -> dict:
    """The §J.1 preflight, in the frozen order. Any failure aborts the run.

    Returns a provenance-shaped report. It writes nothing and computes no
    Stage-F scientific quantity.
    """
    report = {}
    report["protocol_documents"] = reg.assert_protocol_documents_unchanged()
    report["protocol_tag"] = assert_protocol_tag()
    report["constant_drift"] = reg.assert_no_constant_drift()

    report["consumed_artifacts"] = verify_consumed_artifacts()          # 1
    report["upstream_ledgers"] = verify_upstream_ledgers()              # 2
    report["gate1_binding"] = assert_gate1_binding()                    # 2 (cont.)

    cohort = load_cohort()                                              # 3
    report["cohort"] = {"n": int(len(cohort)),
                        "n_sites": int(cohort["site"].nunique())}
    report["common_sites"] = list(assert_common_sites_present(cohort))
    report["fc_cache"] = assert_fc_cache_identity(cohort)

    identities = derive_verified_split_identities()                     # 4
    report["split_registry"] = {"n_rows": identities.n_rows,
                                "sites_per_n": reg.sites_per_n_canonical(),
                                "n_repeats": reg.R_SPLIT,
                                "n_cells_verified": identities.n_cells_verified}
    report["eligible_site_identities"] = identities.as_report()
    report["density_weight_keys"] = assert_density_weight_keys_match()  # 5
    report["probabilities"] = assert_probabilities_finite()             # 6
    report["operating_point_coverage"] = operating_point_coverage_map()

    # §I.2 / §J.1(7) — the non-empty-root refusal applies to the **`2.0.0`
    # authoritative root alone**. The historical `results/stage_f/` holds the
    # tagged, frozen `1.0.2` Attempt #5 store; under §H.1a its existence is the
    # *expected* state and is explicitly **not** a `2.0.0` failure condition, so
    # it is neither checked here nor written to, moved, cleaned or merged.
    root = (ROOT / reg.STORE_ROOT) if store_root is None else Path(store_root)
    if require_empty_store:                                             # 7
        if root.exists() and any(root.iterdir()):
            raise StageFInputError(
                f"{root} exists and is non-empty; a Formal Stage-F run never "
                "overwrites, merges, resumes or cleans an existing store "
                "(§I.2). Archive it explicitly before running.")
    report["store_root_clear"] = bool(
        not root.exists() or not any(root.iterdir()))
    report["authority"] = {
        "protocol_version": reg.PROTOCOL_VERSION,
        "protocol_tag": reg.PROTOCOL_TAG,
        "store_root": reg.STORE_ROOT,
        "freeze_root": reg.FREEZE_ROOT,
        "supersedes_version": reg.SUPERSEDED_PROTOCOL_VERSION,
        "supersedes_tag": reg.SUPERSEDED_PROTOCOL_TAG,
        "historical_roots": list(reg.HISTORICAL_ROOTS),
        "historical_roots_block_this_run": reg.HISTORICAL_BLOCKS_NEW_AUTHORITY,
        "historical_roots_written_by_this_run":
            reg.HISTORICAL_WRITABLE_BY_NEW_AUTHORITY,
    }
    return report
