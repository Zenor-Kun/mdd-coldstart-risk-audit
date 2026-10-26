#!/usr/bin/env python3
"""Stage F.2 implementation preflight — mechanical, read-only, writes nothing.

This is the **implementation** validator. It is a different object from
`scripts/validate_stage_f_protocol_freeze.py`, which is the frozen *protocol*
validator and is never modified here.

**Protocol-validator lifecycle.** The frozen pre-tag protocol validator contains
one lifecycle guard that asserts `stage-f-protocol-v1` does **not** exist — the
rule that kept the draft from claiming a tag before its independent re-audit.
The freeze ended that lifecycle stage, so that single guard now fails by design.
It is not protocol drift, it is not a scientific check, and it is not edited.
This validator takes the opposite and now-correct position: the tag is
**REQUIRED**, and it must be annotated and target the frozen protocol commit.

What it verifies:

  identity   the protocol tag, its annotation, its target commit, and the three
             protocol-authority SHA256s;
  constants  true two-way structured mirroring of every execution-controlling
             constant against `docs/stage_f_constants.json`;
  scope      no guarantee-label upgrade, no seventh sensitivity, no grid change,
             no added CI slot, no SVM probability path, no withdrawn reason code
             back in service, no output beyond the frozen registry;
  inputs     all 17 consumed artifacts, the genuine Gate-1 content binding by
             value, and the FC cache identity key;
  runtime    the Formal CLI's confirmation flag and the absence of any
             caller-controlled authoritative root;
  authority  that no module-visible callable mints authority at all, that the
             direct constructor is refused outright, that authority copied from
             a genuine store (or inherited by a subclass) confers nothing, and
             that a mutated root or run kind is caught at completion —
             exercised behaviourally against a temporary root, never asserted
             from source text;
  completion that the checksum-covered set is derived from the output registry,
             that registered / written / validated / ledgered are *identical*
             before the marker, that a written-but-unvalidated artifact blocks
             the run, and that the completion phases are observed — through
             spies, not source order — to run authority -> ledger -> reread ->
             set equality -> authority -> marker;
  durability that every registered CSV is file-fsynced before its atomic
             rename, that the nested directory chain is fsynced up to the store
             root, and that the checksum ledger is fsynced and independently
             reread;
  integrity  that exact schemas, registered logical-key uniqueness, genuinely
             numeric scientific fields and status / reason / degeneracy
             semantics are enforced *at write time*; that the ledger parser
             refuses duplicate, self-referential, absolute, traversing, aliased
             and unregistered entries; and that CI endpoints, worst-site fields
             and the parent's RQ3 status/failing-condition pair are enforced —
             all probed with malformed tables that must be refused;
  store      no authoritative `results/stage_f/` exists, and no preserved
             failure store under `results/failed_runs/` carries the terminal
             completion marker — which barrier artifact that is, and which are
             merely pre-terminal, is derived from the store lifecycle itself;
  tests      the Stage-F test modules pass.

    python3 scripts/validate_stage_f_implementation.py
    python3 scripts/validate_stage_f_implementation.py --skip-tests
"""

from __future__ import annotations

import argparse
import ast
import inspect
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                      # noqa: E402

import certification as cert            # noqa: E402
import nested_loso as nl                # noqa: E402
import rq3_protocol as proto            # noqa: E402
import s2_inputs as si                  # noqa: E402
import site_bootstrap as sb             # noqa: E402
import stage_f_conformal as conformal   # noqa: E402
import stage_f_execution as execution   # noqa: E402
import stage_f_inputs as inputs         # noqa: E402
import stage_f_registry as reg          # noqa: E402
import stage_f_sensitivities as sens    # noqa: E402
import stage_f_store as store_module    # noqa: E402
import stage_f_svm as svm               # noqa: E402

STAGE_F_MODULES = ("stage_f_registry", "stage_f_inputs", "stage_f_conformal",
                   "stage_f_sensitivities", "stage_f_svm", "stage_f_store",
                   "stage_f_execution")
STAGE_F_TESTS = ("tests/test_stage_f_conformal.py",
                 "tests/test_stage_f_sensitivities.py",
                 "tests/test_stage_f_sens2_threshold_parent_path.py",
                 "tests/test_stage_f_svm.py",
                 "tests/test_stage_f_svm_persistence.py",
                 "tests/test_stage_f_svm_protocol_v2.py",
                 "tests/test_stage_f_protocol_validator_enforcement.py",
                 "tests/test_stage_f_inputs.py",
                 "tests/test_stage_f_store.py",
                 "tests/test_stage_f_rq3_scratch_provenance.py",
                 "tests/test_stage_f_genuine_provenance.py",
                 "tests/test_stage_f_failed_store_lifecycle.py")

_FAILURES: list[str] = []
_NOTES: list[str] = []


def ok(message: str) -> None:
    _NOTES.append(message)
    print(f"  ok    {message}")


def fail(message: str) -> None:
    _FAILURES.append(message)
    print(f"  FAIL  {message}")


def _git(*args: str) -> str:
    out = subprocess.run(("git", "-C", str(ROOT)) + args,
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


# ---------------------------------------------------------------------------
# 1. protocol identity — the tag is REQUIRED at Stage F.2
# ---------------------------------------------------------------------------
def check_protocol_identity() -> None:
    try:
        tag = inputs.assert_protocol_tag()
        ok(f"{tag['tag']} is an annotated tag at {tag['target'][:12]}")
    except AssertionError as exc:
        fail(f"protocol tag: {exc}")
        return
    try:
        digests = reg.assert_protocol_documents_unchanged()
    except AssertionError as exc:
        fail(f"protocol authority bytes: {exc}")
        return
    for relpath, digest in sorted(digests.items()):
        ok(f"{relpath} = {digest[:12]}... (frozen)")
    if reg.PROTOCOL_VERSION != "STAGE_F_PROTOCOL/2.0.0":
        fail(f"the implementation targets {reg.PROTOCOL_VERSION}, the frozen "
             "runtime authority is STAGE_F_PROTOCOL/2.0.0")
    elif reg.PROTOCOL_TAG != "stage-f-protocol-v2":
        fail(f"the implementation binds to {reg.PROTOCOL_TAG}")
    else:
        ok(f"implementation targets {reg.PROTOCOL_VERSION} @ {reg.PROTOCOL_TAG}")

    # §H.1a — the superseded identity is retained as *evidence*, never as a
    # fallback. It must still be named, and must not be what anything binds to.
    if reg.SUPERSEDED_PROTOCOL_TAG != "stage-f-protocol-v1" \
            or reg.SUPERSEDED_PROTOCOL_VERSION != "STAGE_F_PROTOCOL/1.0.2":
        fail("the superseded 1.0.2 identity was edited")
    elif reg.PROTOCOL_TAG == reg.SUPERSEDED_PROTOCOL_TAG:
        fail("the runtime authority and the superseded tag are the same tag")
    else:
        ok(f"{reg.SUPERSEDED_PROTOCOL_TAG} / "
           f"{reg.SUPERSEDED_PROTOCOL_VERSION} is retained as immutable "
           "historical identity and is never a fallback authority (§H.1a)")

    # §H.1 / §H.1a — the roots moved, and the historical ones are untouched and
    # non-blocking. Their presence is the *expected* state under 2.0.0.
    if reg.STORE_ROOT != "results/stage_f_v2/" \
            or reg.FREEZE_ROOT != "results/stage_f_v2_freeze/":
        fail(f"the authoritative roots are {reg.STORE_ROOT} / "
             f"{reg.FREEZE_ROOT}; §H.1 freezes results/stage_f_v2/ and "
             "results/stage_f_v2_freeze/")
    elif reg.STORE_ROOT in reg.HISTORICAL_ROOTS \
            or reg.FREEZE_ROOT in reg.HISTORICAL_ROOTS:
        fail("an authoritative root collides with an immutable historical root")
    elif reg.HISTORICAL_BLOCKS_NEW_AUTHORITY \
            or reg.HISTORICAL_WRITABLE_BY_NEW_AUTHORITY \
            or reg.HISTORICAL_READABLE_AS_SCIENTIFIC_INPUT:
        fail("the historical roots are declared blocking, writable or "
             "readable as a scientific input; §H.1a forbids all three")
    else:
        ok(f"the authority root is {reg.STORE_ROOT} and the freeze root "
           f"{reg.FREEZE_ROOT}; {list(reg.HISTORICAL_ROOTS)} are immutable, "
           "non-blocking historical evidence and are never written to")


def check_protocol_validator_lifecycle() -> None:
    """The frozen protocol validator is present, unmodified, and not edited here.

    Its pre-freeze guard now fails by design; that is a lifecycle fact, not
    protocol drift, and this stage neither modifies nor re-runs it as a gate.
    """
    path = ROOT / reg.PROTOCOL_VALIDATOR
    if reg.sha256_file(path) != reg.EXPECTED_PROTOCOL_VALIDATOR_SHA256:
        fail("the frozen protocol validator was modified")
        return
    ok("the frozen protocol validator is unmodified (its pre-freeze tag guard "
       "is expected to fail after the freeze and is not a scientific check)")
    if reg.MIRROR["tag_exists"] is not False:
        fail("the mirrored draft-lifecycle flag was edited")
    else:
        ok("the mirrored `tag_exists` draft flag is untouched; Stage F.2 "
           "requires the tag through stage_f_inputs.assert_protocol_tag")


# ---------------------------------------------------------------------------
# 2. constants — two-way structured mirroring
# ---------------------------------------------------------------------------
def check_constant_mirroring() -> None:
    try:
        report = reg.assert_no_constant_drift()
    except AssertionError as exc:
        fail(f"constant drift: {exc}")
        return
    ok(f"two-way structured mirroring: {report['n_constants_compared']} "
       "constants, 0 unequal, 0 missing in either direction")

    # a negative self-test: the comparison must detect a changed *value*
    original = reg.MIRROR["rq4"]["alpha_grid"]
    reg.MIRROR["rq4"]["alpha_grid"] = [0.1, 0.2, 0.3]
    try:
        reg.assert_no_constant_drift()
    except reg.StageFProtocolDriftError:
        ok("the mirroring rejects a changed value (negative self-test)")
    else:
        fail("the mirroring accepted a changed alpha grid")
    finally:
        reg.MIRROR["rq4"]["alpha_grid"] = original


def check_upstream_constants() -> None:
    """Every borrowed constant still equals its frozen upstream definition."""
    pairs = (
        ("site bootstrap B", reg.SITE_BOOTSTRAP_B, sb.SITE_BOOTSTRAP_B),
        ("CI level", reg.CI_LEVEL, sb.CI_LEVEL),
        ("quantile method", reg.QUANTILE_METHOD, sb.QUANTILE_METHOD),
        ("validity floor", reg.MIN_VALID_REPLICATES, sb.MIN_VALID_REPLICATES),
        ("bootstrap namespace", list(reg.BOOTSTRAP_ENTROPY),
         list(sb.SITE_BOOTSTRAP_ENTROPY)),
        ("SVM C grid", list(reg.SVM_C_GRID), list(nl.C_GRID)),
        ("PCA candidates", list(reg.SVM_PCA_CANDIDATES), list(nl.PCA_CANDIDATES)),
        ("max_iter", reg.SVM_MAX_ITER, nl.MAX_ITER),
        ("SENS-1 alpha", reg.SENS1_ALPHA_UCB, cert.ALPHA_OVERALL),
        ("SENS-2 candidates", reg.SENS2_N_CANDIDATES,
         len(cert.Q_GRID_SENSITIVITY)),
        ("gate-1 exact columns", list(reg.GATE1_EXACT_COLUMNS),
         list(si.GATE1_OOF_EXACT_COLUMNS)),
        ("gate-1 tolerance", reg.GATE1_OOF_PROB_MDD_ATOL, si.GATE1_OOF_ULP_ATOL),
        ("RQ3 cell registry", reg.SENS56_PARENT_CELL_REGISTRY_SIZE,
         len(proto.CELL_REGISTRY)),
        ("RQ3 validation units", reg.SENS56_VALIDATION_UNITS_EXPECTED_ROWS,
         int(proto.N_VALIDATION_UNITS)),
        ("RQ3 Holm families", reg.SENS56_FAMILIES_EXPECTED_ROWS,
         len(proto.HOLM_FAMILIES)),
        ("RQ3 Holm family size", reg.SENS56_HOLM_FAMILY_SIZE,
         int(proto.HOLM_FAMILY_SIZE)),
    )
    bad = [name for name, mine, theirs in pairs if mine != theirs]
    if bad:
        fail(f"borrowed constants drifted from their frozen source: {bad}")
    else:
        ok(f"all {len(pairs)} borrowed constants equal their frozen upstream "
           "definitions")


# ---------------------------------------------------------------------------
# 3. scope — nothing may expand
# ---------------------------------------------------------------------------
def check_guarantee_labels() -> None:
    expected = {"source_pooled_split": "EMPIRICAL_SOURCE_REFERENCE",
                "weighted": "EMPIRICAL",
                "target_local": "EXACT_WITHIN_TARGET_SITE"}
    if dict(reg.GUARANTEE_LABELS) != expected:
        fail(f"guarantee labels moved: {dict(reg.GUARANTEE_LABELS)}")
        return
    exact = tuple(m for m, label in reg.GUARANTEE_LABELS.items()
                  if "EXACT" in label)
    if exact != ("target_local",) != reg.EXACT_GUARANTEE_PERMITTED_FOR:
        fail(f"an empirical method acquired an exact guarantee: {exact}")
        return
    ok("guarantee labels are the frozen three; target_local is the only method "
       "permitted a finite-sample coverage claim")

    source = (ROOT / "src" / "stage_f_conformal.py").read_text()
    if '"EXACT_WITHIN_TARGET_SITE"' in source or '"EMPIRICAL"' in source:
        fail("a guarantee label is rebuilt locally instead of being read from "
             "the registry, so an upgrade could not be detected by drift")
    else:
        ok("every emitted guarantee label is read from the frozen registry")


def check_scientific_scope() -> None:
    checks = (
        ("alpha grid", tuple(reg.ALPHA_GRID), (0.1, 0.2, 0.3, 0.4)),
        ("N grid", tuple(reg.N_GRID), (10, 20, 40)),
        ("registered conformal cells", reg.N_REGISTERED_CELLS, 20),
        ("CI slots", reg.N_CI_SLOTS, 80),
        ("sensitivities", reg.N_SENSITIVITIES, 6),
        ("consumed artifacts", reg.N_CONSUMED_ARTIFACTS, 17),
        ("SENS-5/6 cells", reg.SENS56_CELLS_EXPECTED_ROWS, 12),
        ("SENS-5/6 validation units", reg.SENS56_VALIDATION_UNITS_EXPECTED_ROWS, 24),
        ("SENS-5/6 families", reg.SENS56_FAMILIES_EXPECTED_ROWS, 4),
    )
    bad = [name for name, got, want in checks if got != want]
    if bad:
        fail(f"a frozen grid or registry size moved: {bad}")
    else:
        ok("no grid, registry size or cardinality expanded")

    if 5 in reg.N_GRID:
        fail("N = 5 re-entered the target-local grid")
    else:
        ok("N = 5 remains excluded from target-local conformal (§D.6)")

    if set(reg.WITHDRAWN_REASON_CODES) & set(reg.REASON_CODES):
        fail("a withdrawn reason code returned to service")
    else:
        ok("CONFORMAL_LEVEL_UNATTAINABLE and SINGLE_CLASS_EVALUATION are still "
           "withdrawn as estimability reasons")

    if reg.HYPOTHESIS_TEST or reg.MULTIPLICITY_CORRECTION or reg.TOST:
        fail("a hypothesis test, multiplicity correction or TOST was registered")
    else:
        ok("no hypothesis test, no multiplicity correction, no TOST, no Holm "
           "family for RQ4")

    try:
        sens.assert_closed_set()
        ok("the sensitivity set is closed at exactly six")
    except AssertionError as exc:
        fail(f"sensitivity closure: {exc}")


def check_svm_scope() -> None:
    try:
        svm.assert_grids_match_frozen_pipeline()
        ok("the SVM C and PCA grids equal the frozen ridge-logistic pipeline")
    except AssertionError as exc:
        fail(f"SVM grids: {exc}")
    estimator = svm.make_estimator(1.0)
    if hasattr(estimator, "predict_proba"):
        fail("the SVM estimator exposes a probability path")
    else:
        ok("LinearSVC exposes no predict_proba; Platt scaling is structurally "
           "unavailable")
    if reg.SVM_RANDOM_STATE_STATUS != "OPERATIONALLY_INERT" or reg.SVM_DUAL:
        fail("random_state is no longer recorded as OPERATIONALLY_INERT under "
             "dual=False")
    else:
        ok("random_state is OPERATIONALLY_INERT under the frozen dual=False")
    if reg.SVM_C_SCALE_EQUIVALENCE_CLAIMED:
        fail("a C-scale equivalence with logistic regression is claimed")
    else:
        ok("no scale equivalence between the two C parameters is claimed")

    body = (ROOT / "src" / "stage_f_svm.py").read_text()
    banned = [t for t in ("roc_auc_score", "f1_score", "CalibratedClassifierCV",
                          "predict_proba(") if t in body]
    if banned:
        fail(f"a forbidden SVM output path is present: {banned}")
    else:
        ok("no AUC, F1, calibration or probability path exists in F-SVM")


def check_svm_warning_propagation() -> None:
    """§F.7.1 / §F.7.1a / §F.7.2 — the fold-lifecycle rule, exercised.

    Behavioural, never textual: the fold is driven on a synthetic four-site
    fixture with one inner-grid fit made to warn, and the persisted shape is
    read off the result. No real cohort is touched and no Stage-F quantity is
    produced.
    """
    import warnings as _warnings

    rng = np.random.default_rng(reg.SVM_RANDOM_STATE)
    sites = np.repeat(["S1", "S2", "S3", "S5"], 16)
    y = np.tile([1, 0], len(sites) // 2)
    X = rng.normal(size=(len(sites), 12)) + y[:, None] * 0.8

    clean = svm.run_outer_fold(X, y, sites, "S1")
    rows = clean["_selection"]
    if len(rows) != reg.SVM_N_CANDIDATES_PER_FOLD:
        fail(f"a clean fold emitted {len(rows)} candidate rows, not "
             f"{reg.SVM_N_CANDIDATES_PER_FOLD}")
        return
    if clean["status"] != reg.ESTIMABLE or clean["converged"] is not True \
            or clean["n_iter_max"] != reg.SVM_MAX_ITER:
        fail(f"a clean fold is not the ESTIMABLE representation: {clean}")
        return
    ok(f"a clean outer fold is ESTIMABLE with converged=True, "
       f"n_iter_max={reg.SVM_MAX_ITER} and all "
       f"{reg.SVM_N_CANDIDATES_PER_FOLD} candidate rows ESTIMABLE")

    original = svm.fit_predict
    seen = {"n": 0}

    def warn_on_third(X_tr, y_tr, w_tr, X_te, C):
        seen["n"] += 1
        if seen["n"] == 3:
            raise svm.SVMNotConverged("validator probe")
        return original(X_tr, y_tr, w_tr, X_te, C)

    refits = {"n": 0}

    def count_refit(X_tr, y_tr, w_tr, X_te, C):
        refits["n"] += 1
        return original(X_tr, y_tr, w_tr, X_te, C)

    try:
        svm.fit_predict = warn_on_third
        warned = svm.run_outer_fold(X, y, sites, "S1")
    finally:
        svm.fit_predict = original

    rows = warned["_selection"]
    expected = dict(reg.SVM_INNER_GRID_WARNING_REPRESENTATION)
    mismatch = [f for f in ("status", "reason_code", "bacc", "selected_pca",
                            "selected_C", "inner_macro_bacc", "converged",
                            "n_iter_max")
                if warned[f] != expected[f]]
    if mismatch:
        fail(f"an inner-grid ConvergenceWarning did not produce the §F.7.1 "
             f"row; disagreeing fields: {mismatch}")
        return
    if len(rows) != reg.SVM_N_CANDIDATES_PER_FOLD:
        fail(f"an invalidated fold emitted {len(rows)} candidate rows; §F.7.2 "
             f"keeps all {reg.SVM_N_CANDIDATES_PER_FOLD}")
        return
    survivors = [r for r in rows if r["status"] == reg.ESTIMABLE]
    failed_rows = [r for r in rows if r["status"] == reg.NOT_ESTIMABLE]
    if len(failed_rows) != 1 or failed_rows[0]["reason_code"] != "SVM_NOT_CONVERGED":
        fail(f"the warned candidate is not the single NOT_ESTIMABLE row: "
             f"{failed_rows}")
        return
    if any(r["inner_macro_bacc"] is None for r in survivors):
        fail("a surviving diagnostic candidate lost its finite score; §F.7.2 "
             "rule 4 retains them")
        return
    if any(r["inner_macro_bacc"] is not None for r in failed_rows):
        fail("a NOT_ESTIMABLE candidate row still carries a score")
        return
    for field in ("n_acquisitions", "n_mdd", "n_hc"):
        if not isinstance(warned[field], int):
            fail(f"the cohort fact {field} was not persisted on an "
                 "invalidated fold")
            return
    ok("one inner-grid ConvergenceWarning invalidates the whole outer fold: "
       f"NOT_ESTIMABLE / SVM_NOT_CONVERGED, all four selection fields null, "
       f"converged=False, n_iter_max=null, all "
       f"{reg.SVM_N_CANDIDATES_PER_FOLD} candidate rows retained "
       f"({len(survivors)} diagnostic scores kept, 1 warned row null)")

    # No final refit after an inner warning: the refit is the 21st distinct
    # `fit_predict` call in a clean fold, and must simply never happen here.
    seen["n"] = 0

    def warn_then_count(X_tr, y_tr, w_tr, X_te, C):
        seen["n"] += 1
        if seen["n"] == 3:
            raise svm.SVMNotConverged("validator probe")
        return count_refit(X_tr, y_tr, w_tr, X_te, C)

    try:
        svm.fit_predict = warn_then_count
        before = refits["n"]
        svm.run_outer_fold(X, y, sites, "S1")
        inner_calls = refits["n"] - before
        svm.fit_predict = count_refit
        refits["n"] = 0
        svm.run_outer_fold(X, y, sites, "S1")
        clean_calls = refits["n"]
    finally:
        svm.fit_predict = original
    # a clean fold does (inner sweep) + 1 refit; the warned one does the same
    # sweep minus the one that raised, and **no** refit.
    if clean_calls - inner_calls != 2:
        fail(f"a final refit was attempted after an inner warning: the clean "
             f"fold made {clean_calls} fits and the warned fold {inner_calls} "
             "(expected two fewer: the raising candidate and the skipped refit)")
    else:
        ok("no final outer refit is attempted after an inner-grid warning; the "
           "fold is already invalid and refitting could only manufacture a "
           "number §F.7 forbids reporting")

    # §F.7.1a — a complete valid grid, a lawful selection, then a warning from
    # the selected final refit only.
    calls = {"n": 0}
    total = clean_calls

    def warn_on_refit(X_tr, y_tr, w_tr, X_te, C):
        calls["n"] += 1
        if calls["n"] == total:
            raise svm.SVMNotConverged("validator refit probe")
        return original(X_tr, y_tr, w_tr, X_te, C)

    try:
        svm.fit_predict = warn_on_refit
        refit = svm.run_outer_fold(X, y, sites, "S1")
    finally:
        svm.fit_predict = original

    if any(r["status"] != reg.ESTIMABLE for r in refit["_selection"]):
        fail("the §F.7.1a probe did not leave all 20 candidate rows ESTIMABLE")
        return
    if refit["status"] != reg.NOT_ESTIMABLE \
            or refit["reason_code"] != "SVM_NOT_CONVERGED" \
            or refit["bacc"] is not None \
            or refit["converged"] is not False \
            or refit["n_iter_max"] != reg.SVM_MAX_ITER:
        fail(f"the §F.7.1a final-refit representation is wrong: {refit}")
        return
    if refit["selected_pca"] is None or refit["selected_C"] is None \
            or refit["inner_macro_bacc"] is None:
        fail("the §F.7.1a row erased the lawful selection metadata it really "
             "produced")
        return
    if (refit["selected_pca"], refit["selected_C"]) != (clean["selected_pca"],
                                                        clean["selected_C"]):
        fail("the §F.7.1a row retained a different selection than the lawful "
             "one")
        return
    ok("a final-refit ConvergenceWarning after a lawful selection gives the "
       "distinct §F.7.1a row: NOT_ESTIMABLE / SVM_NOT_CONVERGED, bacc null, "
       f"converged=False, n_iter_max={reg.SVM_MAX_ITER}, and selected_pca / "
       "selected_C / inner_macro_bacc retained as provenance")

    if warned["n_iter_max"] == refit["n_iter_max"] \
            or warned["selected_pca"] == refit["selected_pca"]:
        fail("the §F.7.1 and §F.7.1a representations are not distinguishable")
    else:
        ok("the inner-grid and final-refit representations remain distinct on "
           "n_iter_max and on the three retained selection fields")
    del _warnings


def check_svm_selection_and_tie_break() -> None:
    """§F.7.2 / §F.7.5 — complete-grid precondition and the corrected order."""
    grid = [(p, c) for p in reg.SVM_PCA_CANDIDATES for c in reg.SVM_C_GRID]
    if len(grid) != reg.SVM_N_CANDIDATES_PER_FOLD:
        fail("the candidate grid is no longer 20 points")
        return
    smallest_c = reg.SVM_C_GRID[0]

    # The PCA preference, read off one member at a time: give every candidate
    # at the smallest `C` an identical top score, take the winner, drop that
    # PCA dimension from the tied set, and repeat. The grid handed to
    # `select_candidate` stays complete throughout — §F.7.2 forbids ranking a
    # partial one — so only the *scores* change between probes.
    order, remaining = [], list(reg.SVM_PCA_CANDIDATES)
    while remaining:
        tied = {key: (0.9 if key[0] in remaining and key[1] == smallest_c
                      else 0.1)
                for key in grid}
        winner = svm.select_candidate(tied)
        order.append(winner[0])
        remaining.remove(winner[0])
    expected_order = list(reg.SVM_TIE_BREAK_PCA_ORDER)
    if order != expected_order:
        fail(f"the PCA tie-break order is {order}, §F.7.5 freezes "
             f"{expected_order}")
        return
    if order[-1] is not None:
        fail("`no-PCA` is not last in the compression order")
        return
    ok(f"an exact cross-PCA tie resolves in the frozen §F.7.5 order "
       f"{['no-PCA' if p is None else p for p in order]} — `no-PCA` is the "
       "weakest compression and is last, not first (the `1.0.2` implementation "
       "ranked by declaration order and preferred it first)")

    # `C` dominates the compression tie-break: at equal scores throughout, the
    # smallest `C` wins first and only then does 50 beat 100.
    flat = {key: 0.7 for key in grid}
    if svm.select_candidate(flat) != (50, smallest_c):
        fail(f"a fully flat grid selected {svm.select_candidate(flat)}, not "
             f"(50, {smallest_c})")
    else:
        ok("with every score equal, the smaller C wins before the compression "
           "tie-break is consulted, and the score dominates both")

    # A strictly better score outranks both tie-breaks.
    best = (200, reg.SVM_C_GRID[-1])
    dominant = {key: (0.95 if key == best else 0.7) for key in grid}
    if svm.select_candidate(dominant) != best:
        fail("the highest inner_macro_bacc did not win outright")
    else:
        ok("the highest inner macro site-BAcc is the primary criterion; the "
           "two tie-breaks are consulted only on an exact tie")

    # §F.7.2 — a reduced grid is refused outright, not silently selected from.
    probes = 0
    for label, partial in (
            ("a 19-candidate grid", {k: 0.6 for k in grid[:-1]}),
            ("a grid carrying a non-finite score",
             {**{k: 0.6 for k in grid[:-1]}, grid[-1]: float("nan")}),
            ("an empty grid", {})):
        try:
            svm.select_candidate(partial)
        except svm.SVMScopeError:
            probes += 1
        else:
            fail(f"selection accepted {label}")
            return
    ok(f"selection over a reduced or partly non-finite candidate set is "
       f"refused outright ({probes} probes); §F.7.2 permits it only when all "
       f"{reg.SVM_N_CANDIDATES_PER_FOLD} rows are ESTIMABLE with finite scores")


def check_svm_fold_failed_scope() -> None:
    """§G.2a — the closed allow-list, exact-type identity, and the boundary."""
    if list(reg.SVM_FOLD_FAILED_ALLOWED_TYPES) != ["FloatingPointError",
                                                   "numpy.linalg.LinAlgError"]:
        fail(f"the §G.2a allow-list is {reg.SVM_FOLD_FAILED_ALLOWED_TYPES}")
        return
    resolved = svm.PERMITTED_ESTIMATOR_FAILURES
    if resolved != (FloatingPointError, np.linalg.LinAlgError):
        fail(f"the allow-list resolved to {resolved}")
        return
    ok("the §G.2a allow-list is closed at exactly two types — "
       "FloatingPointError and numpy.linalg.LinAlgError — resolved from the "
       "registry's names to the live classes")

    class SubclassedLinAlg(np.linalg.LinAlgError):
        pass

    accepted = [name for name, exc in (
        ("FloatingPointError", FloatingPointError("probe")),
        ("numpy.linalg.LinAlgError", np.linalg.LinAlgError("probe")))
        if svm.is_permitted_estimator_failure(exc)]
    rejected = [type(exc).__name__ for exc in (
        ValueError("probe"), RuntimeError("probe"), AssertionError("probe"),
        KeyError("probe"), TypeError("probe"), IndexError("probe"),
        MemoryError(), SubclassedLinAlg("probe"),
        svm.SVMNotConverged("probe"), svm.SVMScopeError("probe"))
        if not svm.is_permitted_estimator_failure(exc)]
    if len(accepted) != 2:
        fail(f"the exact-type test rejected a registered type: {accepted}")
        return
    if len(rejected) != 10:
        fail("the exact-type test accepted an unregistered type")
        return
    if svm.is_permitted_estimator_failure(SubclassedLinAlg("probe")):
        fail("a subclass of a permitted type widened the allow-list")
        return
    ok(f"membership is by exact type identity: the 2 registered types are "
       f"accepted and {len(rejected)} others are refused — including a "
       "**subclass** of LinAlgError and a bare ValueError, which LinAlgError "
       "itself inherits from")

    if reg.SVM_FOLD_FAILED_ISINSTANCE_PERMITTED \
            or reg.SVM_FOLD_FAILED_SUBCLASS_WIDENING_PERMITTED \
            or reg.SVM_FOLD_FAILED_BROAD_CATCHING_PERMITTED:
        fail("isinstance matching, subclass widening or broad catching is "
             "declared permitted")
    else:
        ok("isinstance matching, subclass widening and broad exception "
           "catching are all declared forbidden")

    source = (ROOT / "src" / "stage_f_svm.py").read_text()
    tree = ast.parse(source)
    broad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                broad.append("bare except")
            elif isinstance(node.type, ast.Name) and node.type.id in (
                    "Exception", "BaseException"):
                broad.append(f"except {node.type.id}")
    if broad:
        fail(f"F-SVM catches broadly: {broad}; §G.2a forbids it outright")
    else:
        ok("no bare `except:` and no `except Exception` exists anywhere in "
           "F-SVM — broad catching is forbidden whatever it is then spelled as")

    # The boundary: the permitted types are caught around the two frozen calls
    # and nowhere else.
    handlers = [h for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler)]
    permitted_handlers = [
        h for h in handlers
        if isinstance(h.type, ast.Name)
        and h.type.id == "PERMITTED_ESTIMATOR_FAILURES"]
    if len(permitted_handlers) != 2:
        fail(f"{len(permitted_handlers)} handlers guard the permitted types; "
             "the boundary is exactly LinearSVC.fit and decision_function")
        return
    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(isinstance(h.type, ast.Name)
                   and h.type.id == "PERMITTED_ESTIMATOR_FAILURES"
                   for h in node.handlers):
            continue
        body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        guarded.append(("fit" if "'fit'" in body else
                        "decision_function" if "decision_function" in body
                        else "?"))
    if sorted(guarded) != ["decision_function", "fit"]:
        fail(f"the guarded regions are {guarded}; §G.2a's boundary is the "
             "frozen LinearSVC.fit(...) and the decision_function(...) on that "
             "same fitted estimator, and nothing else")
    else:
        ok("the estimator/backend boundary is exactly the two frozen calls: "
           "`LinearSVC.fit(...)` and `decision_function(...)` on that same "
           "estimator; data assembly, transforms, weights, scoring, selection, "
           "aggregation and persistence are all outside it")

    if reg.SVM_FOLD_FAILED_N_MUST_NOT_ABSORB != 20 \
            or reg.SVM_FOLD_FAILED_IS_CATCH_ALL \
            or reg.SVM_FOLD_FAILED_SUBSTITUTES_FOR_NOT_CONVERGED:
        fail("the §G.2a exclusion set or its catch-all flags were weakened")
    else:
        ok(f"the {reg.SVM_FOLD_FAILED_N_MUST_NOT_ABSORB} fail-hard exclusions "
           "stand, SVM_FOLD_FAILED is not a catch-all, and it is never a "
           "substitute for SVM_NOT_CONVERGED")


def check_svm_fold_selection_schema() -> None:
    """§H.3.1 — the amended candidate-row contract, key and cardinality intact."""
    columns = reg.table_columns(reg.SVM_FOLD_SELECTION_TABLE)
    if set(columns) != {"held_out_site", "pca", "C", "inner_macro_bacc",
                        "status", "reason_code"}:
        fail(f"the fold_selection schema is {columns}")
        return
    key = reg.logical_key(reg.SVM_FOLD_SELECTION_TABLE)
    if tuple(key) != ("held_out_site", "pca", "C"):
        fail(f"the fold_selection logical key moved to {key}")
        return
    rows = reg.SVM_ARTIFACTS[reg.SVM_FOLD_SELECTION_TABLE][1]
    if rows != 460 or rows != reg.BASE_COHORT_SITES * reg.SVM_N_CANDIDATES_PER_FOLD:
        fail(f"the fold_selection cardinality is {rows}, not 23 x 4 x 5 = 460")
        return
    ok("svm/fold_selection.csv keeps its key (held_out_site, pca, C) and its "
       "460-row cardinality, and gains exactly `status` and `reason_code` — no "
       "candidate_status / candidate_reason duplicates")

    if any(f in columns for f in ("candidate_status", "candidate_reason")):
        fail("redundant candidate status fields were registered")
        return
    semantics = reg.STATUS_SEMANTICS.get(reg.SVM_FOLD_SELECTION_TABLE)
    if semantics is None \
            or tuple(semantics["value_fields"]) != ("inner_macro_bacc",):
        fail("inner_macro_bacc is not registered as the candidate value field")
        return
    allowed = reg.artifact_reason_codes(reg.SVM_FOLD_SELECTION_TABLE,
                                        reg.NOT_ESTIMABLE)
    if tuple(allowed) != ("SVM_NOT_CONVERGED", "SVM_FOLD_FAILED"):
        fail(f"the candidate reason vocabulary is {allowed}")
        return
    if reg.artifact_reason_codes(reg.SVM_FOLD_SELECTION_TABLE, reg.ESTIMABLE):
        fail("an ESTIMABLE candidate row may carry a reason code")
        return
    if reg.SVM_NULL_SCORE_WITHOUT_STATUS_PERMITTED:
        fail("a null score without a status is declared permissible")
        return
    ok("inner_macro_bacc is the registered candidate value field, the "
       "NOT_ESTIMABLE vocabulary is exactly {SVM_NOT_CONVERGED, "
       "SVM_FOLD_FAILED}, an ESTIMABLE row may carry no reason, and a blank "
       "score with no status — the Attempt #5 spelling — is unrepresentable")


def check_sens1_score_type_label() -> None:
    """§E.2a — the label names the functional that actually ran. Metadata only."""
    label = reg.SENSITIVITY_LABELS[0]
    if reg.SENS1_OVERALL_ERROR_SCORE_TYPE != \
            "exact_overall_error_clopper_pearson_ucb":
        fail(f"the corrected SENS-1 label is "
             f"{reg.SENS1_OVERALL_ERROR_SCORE_TYPE!r}")
        return
    corrected = [s for s in cert.STRATEGY_SCORE_TYPE
                 if sens.persisted_score_type(label, s)
                 != cert.STRATEGY_SCORE_TYPE[s]]
    exact = [s for s, t in cert.STRATEGY_SCORE_TYPE.items()
             if t == cert.SCORE_EXACT_CP]
    if sorted(corrected) != sorted(exact):
        fail(f"SENS-1 relabels {sorted(corrected)}; the strategies whose "
             f"functional is certification.overall_error_upper are "
             f"{sorted(exact)}")
        return
    ok(f"SENS-1 relabels exactly the rows whose functional really is "
       f"{reg.SENS1_ACTUAL_FUNCTIONAL} ({sorted(corrected)}) and no others")

    other = [(lab, s) for lab in reg.SENSITIVITY_LABELS[1:4]
             for s in cert.STRATEGY_SCORE_TYPE
             if sens.persisted_score_type(lab, s)
             != cert.STRATEGY_SCORE_TYPE[s]]
    if other:
        fail(f"a non-SENS-1 sensitivity's score_type moved: {other}")
        return
    ok("no other sensitivity's score_type moves; SENS-2..4 inherit the "
       "parent's SCORE_EXACT_CP meaning unchanged")

    # The guarantee tier and the numeric level are untouched.
    if cert.GUARANTEE_BY_SCORE[cert.SCORE_EXACT_CP] != \
            "EXACT_CP_ON_ITS_OWN_CALIBRATION_SAMPLE":
        fail("the parent guarantee vocabulary moved")
    elif reg.SENS1_ALPHA_UCB != 0.10 or reg.SENS1_BONFERRONI:
        fail("the SENS-1 level or Bonferroni flag changed")
    elif reg.SENS1_SCORE_TYPE_NUMERIC_RECOMPUTATION_AUTHORIZED \
            or reg.SENS1_SCORE_TYPE_SCIENTIFIC_DEFINITION_CHANGED:
        fail("the SENS-1 correction claims a numeric or definitional change")
    else:
        ok("the correction is metadata only: alpha = 0.10, no Bonferroni, the "
           "parent guarantee tier and certification.overall_error_upper are "
           "all unchanged, and no value is recomputed")


def check_output_registry() -> None:
    paths = reg.registered_artifact_paths()
    if len(paths) != len(set(paths)):
        fail("the output registry holds a duplicate path")
        return
    ok(f"the output registry is closed at {len(paths)} paths across "
       f"{len(reg.NAMESPACES)} namespaces plus the three barrier artifacts")
    try:
        reg.assert_registered_artifact("rq4/extra.csv")
        fail("an unregistered artifact was accepted")
    except reg.StageFRegistryError:
        ok("an unregistered artifact is refused by the writer's guard")
    if list(store_module.reg.BARRIER_STEPS) != list(reg.BARRIER_STEPS):
        fail("the barrier step order drifted")
    elif reg.BARRIER_STEPS[-1] != "write_completion_marker_atomically":
        fail("COMPLETION.json is not the terminal barrier step")
    else:
        ok("the seven barrier steps are in the frozen order, COMPLETION.json last")


def check_ci_registry() -> None:
    if len(reg.REQUIRED_INTERVALS) != 80:
        fail(f"the CI registry enumerates {len(reg.REQUIRED_INTERVALS)} slots")
        return
    if set(reg.F_CI_REQUIRED) & set(reg.F_CI_PROHIBITED):
        fail("the CI registry is not disjoint")
        return
    try:
        reg.requires_ci("rq4_unregistered_quantity")
        fail("requires_ci accepted an unregistered quantity")
    except reg.StageFRegistryError:
        ok("the CI registry is closed in both directions and raises on an "
           "unregistered quantity")
    ok("80 interval slots = 4 registered quantities x 20 registered cells")


# ---------------------------------------------------------------------------
# 4. §E.8 reachability and the sensitivity design
# ---------------------------------------------------------------------------
def check_parent_reachability() -> None:
    try:
        report = sens.reachability_report()
    except sens.SensitivityBlocked as exc:
        fail(f"a parent mechanism is not parameterisable — the sensitivity is "
             f"BLOCKED, not forked: {exc}")
        return
    except AssertionError as exc:
        fail(f"reachability probe: {exc}")
        return
    ok(f"RQ2/S2 changed factors reach {report['s2_family']['entry_point']} "
       f"as {report['s2_family']['parameters']}")
    ok(f"RQ3 changed factors reach {report['rq3_family']['entry_point']} as "
       f"`{report['rq3_family']['parameter']}`")
    for label in reg.SENSITIVITY_LABELS:
        entry = sens.manifest(label)
        if entry["n_changed_factors"] != 1 or entry["forked"]:
            fail(f"{label} declares {entry['n_changed_factors']} changed "
                 f"factors / forked={entry['forked']}")
            return
    ok("all six sensitivities declare exactly one changed factor and no fork")


def check_sens6_algorithm() -> None:
    try:
        sens.assert_selector_is_label_free()
        ok("the SENS-6 threshold selector receives no label column")
    except AssertionError as exc:
        fail(f"SENS-6 label-freeness: {exc}")
    try:
        checked = sens.verify_worked_examples()
        ok(f"all {len(checked)} frozen SENS-6 worked examples re-derive, ties "
           "included")
    except AssertionError as exc:
        fail(f"SENS-6 worked examples: {exc}")
    try:
        sens.assert_sens56_cardinality({})
        ok("SENS-5/6 12 / 24 / 4 re-derive from src/rq3_protocol; `block` is a "
           "validation-unit key field, never a cell key field")
    except AssertionError as exc:
        fail(f"SENS-5/6 schema: {exc}")


def check_sensitivity_cardinalities() -> None:
    bad = []
    for label in reg.SENSITIVITY_LABELS[:4]:
        try:
            sens.assert_expected_cardinality(label)
        except AssertionError as exc:
            bad.append(str(exc))
    if bad:
        fail(f"sensitivity cardinality: {bad}")
    else:
        ok("SENS-1..4 cardinalities re-derive from the frozen split registry "
           "(46200 / 78600 / 92400 / 157200 / 26400 / 46200)")


# ---------------------------------------------------------------------------
# 5. consumed inputs
# ---------------------------------------------------------------------------
def check_consumed_inputs() -> None:
    try:
        report = inputs.verify_consumed_artifacts()
        ok(f"all {report['n_artifacts']} consumed artifacts match their frozen "
           "SHA256")
    except AssertionError as exc:
        fail(f"consumed artifacts: {exc}")
        return
    try:
        ledgers = inputs.verify_upstream_ledgers()
        ok(f"{ledgers['n_ledger_verified']} artifacts additionally verified "
           f"against their frozen result ledgers, {ledgers['n_content_bound']} "
           "content-bound")
    except AssertionError as exc:
        fail(f"upstream ledgers: {exc}")
    try:
        binding = inputs.assert_gate1_binding()
        ok(f"gate-1 binding by value: {binding['n_oof_rows']} rows, "
           f"{binding['n_exact_columns']} bit-identical columns, observed max "
           f"|delta oof_prob_mdd| = {binding['max_ulp_delta']:.6e} <= "
           f"{reg.GATE1_OOF_PROB_MDD_ATOL:g}")
    except AssertionError as exc:
        fail(f"gate-1 binding: {exc}")
    try:
        cache = inputs.assert_fc_cache_identity()
        ok(f"FC cache digest and feature_source_id content key verified "
           f"({cache['n_rows']} x {cache['n_features']})")
    except AssertionError as exc:
        fail(f"FC cache: {exc}")

    body = (ROOT / "src" / "stage_f_inputs.py").read_text()
    if "si.assert_gate1_binds_to_checksummed_s0()" not in body:
        fail("the genuine frozen Gate-1 helper is not invoked")
    else:
        ok("the genuine frozen helper src/s2_inputs."
           "assert_gate1_binds_to_checksummed_s0 is invoked, not approximated")


# ---------------------------------------------------------------------------
# 6. runtime hardening
# ---------------------------------------------------------------------------
def check_formal_cli() -> None:
    import run_stage_f

    parser = run_stage_f.build_parser()
    dests = {a.dest for a in parser._actions}
    if "confirmed" not in dests or "formal" not in dests:
        fail("the Formal entry point has no confirmation flag")
        return
    ok("the Formal entry point requires --formal and "
       "--i-understand-this-is-authoritative")
    leaky = [d for d in ("out", "root", "store_root", "path", "directory")
             if d in dests]
    if leaky:
        fail(f"the Formal CLI exposes a caller-controlled root: {leaky}")
    else:
        ok("no caller-controlled authoritative store root exists")

    result = subprocess.run(
        (sys.executable, str(ROOT / "scripts" / "run_stage_f.py"), "--formal"),
        capture_output=True, text=True)
    if result.returncode == 0:
        fail("`--formal` without confirmation did not refuse")
    else:
        ok("`--formal` without confirmation refuses with a non-zero exit")

    result = subprocess.run(
        (sys.executable, str(ROOT / "scripts" / "run_stage_f.py")),
        capture_output=True, text=True)
    if result.returncode != 0 or "No action taken" not in result.stdout:
        fail("the default invocation is not inert")
    else:
        ok("the default invocation takes no action")

    signature = inspect.signature(
        store_module.FormalStageFStore.create_authoritative)
    if any(p in signature.parameters
           for p in ("root", "out", "store_root", "path")):
        fail("the formal store factory accepts a root argument")
    else:
        ok("the formal store factory takes no root argument (§I.2)")
    if hasattr(store_module.SmokeStageFStore, "complete_run"):
        fail("the smoke store can complete a run")
    else:
        ok("the smoke store has no completion barrier and cannot write "
           "COMPLETION.json")


# ---------------------------------------------------------------------------
# 6a. the store lifecycle — what a *preserved failure* store may contain
# ---------------------------------------------------------------------------
#: The calls that put bytes on disk. A store function that names a barrier
#: artifact's registry constant *and* makes one of these is a writer of it.
_WRITE_CALLS = ("_atomic_write", "write_json", "write_table",
                "write_checksum_ledger", "write_text", "write_bytes")


def _store_functions() -> dict:
    """`{qualified name: (registry constants named, calls made)}`.

    Every function defined in `src/stage_f_store.py`, at module scope or in a
    class body, with whatever it nests folded into it. Read structurally so the
    lifecycle facts below move with the store rather than with this validator.
    """
    tree = ast.parse((ROOT / "src" / "stage_f_store.py").read_text())
    functions: dict = {}

    def visit(node, prefix: str) -> None:
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                constants, calls = set(), set()
                for inner in ast.walk(child):
                    if isinstance(inner, ast.Attribute) \
                            and isinstance(inner.value, ast.Name) \
                            and inner.value.id == "reg":
                        constants.add(inner.attr)
                    if isinstance(inner, ast.Call):
                        func = inner.func
                        name = (func.attr if isinstance(func, ast.Attribute)
                                else getattr(func, "id", None))
                        if name:
                            calls.add(name)
                functions[f"{prefix}{child.name}"] = (constants, calls)

    visit(tree, "")
    return functions


def lifecycle_partition() -> dict:
    """Split the frozen barrier artifacts by *when* the lifecycle writes them.

    Derived, not listed. The completion barrier is the one store function held
    against the frozen `reg.BARRIER_STEPS` order; an artifact is a **terminal
    authority marker** when that barrier is the only thing that can write it,
    and **pre-terminal** otherwise. On the frozen lifecycle that makes
    `COMPLETION.json` the marker — `complete_run` alone writes it, and the
    smoke store has no `complete_run` — while `PROVENANCE.json` (written at
    construction, by both store classes) and `RESULT_CHECKSUMS.sha256` (written
    by `write_checksum_ledger`, which lives on the shared record store and is
    reached by the deliberately non-authoritative `seal_smoke`) are pre-terminal.

    That asymmetry is the whole rule: a store can hold a ledger and never have
    completed, so a ledger is step-2 evidence and never a claim of authority.
    """
    functions = _store_functions()
    barrier = {name for name, (constants, _) in functions.items()
               if "BARRIER_STEPS" in constants}
    constant_of = {value: name for name, value in vars(reg).items()
                   if isinstance(value, str) and name.endswith("_FILE")}
    terminal, preterminal, writers_of = [], [], {}
    for artifact in reg.BARRIER_ARTIFACTS:
        constant = constant_of.get(artifact)
        writers = {name for name, (constants, calls) in functions.items()
                   if constant in constants and set(calls) & set(_WRITE_CALLS)}
        writers_of[artifact] = tuple(sorted(writers))
        (terminal if writers and writers <= barrier
         else preterminal).append(artifact)
    return {"terminal": tuple(terminal), "preterminal": tuple(preterminal),
            "barrier_functions": tuple(sorted(barrier)), "writers": writers_of}


def classify_stage_f_store(path) -> dict:
    """Where on the Stage-F lifecycle a store directory on disk sits.

    Directory entries only. A store is authoritative **iff** it carries a
    terminal completion marker; a pre-terminal artifact — including a complete,
    internally valid `RESULT_CHECKSUMS.sha256` — never promotes it, because the
    ledger is sealed at barrier step 2 and five further steps can still refuse.
    Ledger *integrity* is not softened anywhere by this: it is enforced, strictly
    and unchanged, by `stage_f_store._StageFRecordStore.read_checksum_ledger`,
    which is what a run must survive to complete. This classifier deliberately
    does not consult it, so a valid ledger cannot buy authority and a corrupt
    one cannot be argued into it either.
    """
    path = Path(path)
    partition = lifecycle_partition()
    if len(partition["terminal"]) != 1:
        # Fail closed: if the lifecycle ever stops having exactly one terminal
        # marker, this validator does not get to guess which file means "done".
        raise AssertionError(
            "the store lifecycle no longer yields exactly one terminal "
            f"completion marker: {partition['terminal']}")
    markers = [name for name in partition["terminal"] if (path / name).exists()]
    preterminal = [name for name in partition["preterminal"]
                   if (path / name).exists()]
    return {"authoritative": bool(markers),
            "markers_present": markers,
            "preterminal_present": preterminal,
            "terminal_markers": list(partition["terminal"]),
            "reason": ("carries the terminal completion marker "
                       f"{markers}" if markers else
                       "no terminal completion marker; pre-terminal evidence "
                       f"only {preterminal}")}


def check_store_absent(repo=None) -> None:
    """The pre-run lifecycle rule, over one repository tree.

    `repo` defaults to this repository and is a parameter only so the regression
    tests can drive the same code over a synthetic tree; the lifecycle facts
    themselves are always read from the real `src/stage_f_store.py`.
    """
    repo = ROOT if repo is None else Path(repo)
    root = repo / reg.STORE_ROOT
    if root.exists() and any(root.iterdir()):
        fail(f"{reg.STORE_ROOT} exists and is non-empty; Stage F has been run")
    else:
        ok(f"no authoritative {reg.STORE_ROOT} exists — Stage F has not been run")

    try:
        partition = lifecycle_partition()
    except SyntaxError as exc:                      # pragma: no cover
        fail(f"the store lifecycle could not be read: {exc}")
        return
    if len(partition["terminal"]) != 1:
        fail("the store lifecycle does not yield exactly one terminal "
             f"completion marker: {partition['terminal']}")
        return
    if reg.CHECKSUM_FILE in partition["terminal"]:
        fail(f"{reg.CHECKSUM_FILE} is derived as a terminal completion marker; "
             "the ledger is sealed at barrier step 2, five steps before the run "
             "can complete")
        return
    ok(f"the terminal completion marker is derived from the lifecycle as "
       f"{partition['terminal'][0]} (written by {partition['barrier_functions']}"
       f" alone); {list(partition['preterminal'])} are pre-terminal")

    # A preserved failure store is *evidence*, not a result. Stage F.4 Attempts
    # #1-#4 produced them and they are retained deliberately, so their mere
    # presence is not a finding — nor is a pre-terminal artifact inside one.
    # Attempt #4 reached barrier step 5 and legitimately left its ledger behind.
    # What must hold is that no preserved store claims a completion it never
    # reached.
    failed = repo / "results" / "failed_runs"
    preserved = sorted(p for p in failed.glob("stage_f_*")) if failed.exists() else []
    if not preserved:
        ok("no preserved Stage-F failure store is present")
        return
    promoted, evidence = [], []
    for store in preserved:
        verdict = classify_stage_f_store(store)
        if verdict["authoritative"]:
            promoted += [f"{store.name}/{name}"
                         for name in verdict["markers_present"]]
        else:
            evidence.append(f"{store.name}"
                            f"{verdict['preterminal_present'] or []}")
    if promoted:
        fail("a preserved Stage-F failure store carries the terminal "
             f"completion marker: {promoted}")
    else:
        ok(f"{len(preserved)} preserved Stage-F failure store(s) carry no "
           f"{partition['terminal'][0]}; they are pre-terminal evidence and "
           f"cannot be read as authoritative results: {evidence}")


def check_modules_write_nothing_at_import() -> None:
    """No Stage-F module may open a file for writing at module scope."""
    offenders = []
    writers = ("to_csv", "write_text", "write_bytes", "mkdir", "to_json",
               "savez", "savez_compressed")
    for name in STAGE_F_MODULES:
        tree = ast.parse((ROOT / "src" / f"{name}.py").read_text())
        # module scope only: a call inside a def or a class body runs when the
        # function is called, not when the module is imported
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and \
                        isinstance(inner.func, ast.Attribute) and \
                        inner.func.attr in writers:
                    offenders.append(f"{name}:{inner.lineno}")
    if offenders:
        fail(f"a Stage-F module writes at import time: {offenders}")
    else:
        ok("no Stage-F module writes anything at import time")


# ---------------------------------------------------------------------------
# 6b. Formal authority — behavioural, not textual
# ---------------------------------------------------------------------------
def _reachable_from(root, depth: int = 4) -> list:
    """Everything an ordinary caller reaches from an authority-related object.

    Deliberately mirrors the introspection an attacker performs rather than
    reading the source: closure cells, `__defaults__`, `__kwdefaults__`,
    function attributes and the contents of any container found along the way,
    followed recursively. Module dictionaries and instance dictionaries are not
    followed — the question is what the *mechanism* holds.
    """
    seen, found, queue = set(), [], [(root, 0)]
    while queue:
        value, level = queue.pop()
        if level > depth or id(value) in seen:
            continue
        seen.add(id(value))
        found.append(value)
        children = []
        for cell in (getattr(value, "__closure__", None) or ()):
            try:
                children.append(cell.cell_contents)
            except ValueError:
                continue
        children.extend(getattr(value, "__defaults__", None) or ())
        children.extend((getattr(value, "__kwdefaults__", None) or {}).values())
        if isinstance(value, types.FunctionType):
            children.extend(vars(value).values())
        if isinstance(value, (list, tuple, set, frozenset)):
            children.extend(value)
        if isinstance(value, dict):
            children.extend(list(value.keys()) + list(value.values()))
        queue.extend((child, level + 1) for child in children)
    return found


def check_formal_authority() -> None:
    """Direct construction, copied state and mutated stores all fail closed.

    Exercised against a temporary root, so nothing authoritative can result:
    the point of every probe below is that it **raises**.
    """
    import tempfile

    # A — the attack itself, not a search for suspicious names. An underscore
    # prefix is a naming convention and a keyword blacklist is a spelling test;
    # what matters is whether an *insertable authorisation capability* is
    # reachable. Walk closure cells, defaults, keyword defaults, function
    # attributes and the contents of whatever they hold, from every
    # authority-related module-visible callable, and require that no mutable
    # container turns up at all.
    holders = [value for name, value in vars(store_module).items()
               if callable(value)
               and any(word in name.lower()
                       for word in ("authorit", "verify", "create", "formal"))]
    holders += list(store_module._AUTHORITY_CALLABLES)
    containers = [value for holder in holders
                  for value in _reachable_from(holder)
                  if isinstance(value, (dict, list, set))]
    if containers:
        fail(f"an insertable authorization container is reachable: "
             f"{[type(c).__name__ for c in containers]}")
        return
    ok(f"no mutable authorization container is reachable from any of the "
       f"{len(holders)} authority-related callables (closure cells, defaults, "
       "keyword defaults, function attributes, recursively)")
    try:
        inspected = store_module.assert_no_authorization_container()
    except store_module.FormalAuthorityError as error:      # pragma: no cover
        fail(f"the authority mechanism holds insertable state: {error}")
        return
    ok(f"the authority mechanism holds one immutable record and {inspected} "
       "inert references, and the barrier re-asserts it")
    system = inspect.getsource(store_module._make_formal_authority_system)
    if "return _create, _verify" not in system:
        fail("the authority closure returns more than the factory and verifier")
    else:
        ok("the authority closure returns only the genuine factory and the "
           "verifier complete_run needs")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "stage_f"
        root.mkdir()

        # B — the direct constructor is refused outright, at any root
        for kwargs in ({"root": root, "run_kind": reg.RUN_KIND_FORMAL},
                       {"root": Path(tmp) / "elsewhere",
                        "run_kind": reg.RUN_KIND_FORMAL},
                       {"root": root, "run_kind": reg.RUN_KIND_SMOKE}):
            try:
                store_module.FormalStageFStore(**kwargs)
            except store_module.FormalAuthorityError:
                continue
            fail(f"direct construction produced a Formal store with {kwargs}")
            return
        ok("the Formal constructor is refused outright (3 probes)")

        genuine_root = Path(tmp) / "authoritative"
        real = store_module.authoritative_root
        store_module.authoritative_root = lambda *_: genuine_root
        try:
            genuine = store_module.FormalStageFStore.create_authoritative()
            report = store_module._verify_formal_authority(genuine)
            if report["authoritative_root"] != str(genuine_root.resolve()) \
                    or report["bound_to_instance"] is not True:
                fail("the authorized root is not the derived authoritative root")
            else:
                ok("only the genuine factory instance passes the authority "
                   "check, bound to the derived root")

            # C — authority is bound to the object, not to any copyable state
            impostor = store_module.FormalStageFStore.__new__(
                store_module.FormalStageFStore)
            for name, value in vars(genuine).items():
                object.__setattr__(impostor, name, value)
            try:
                impostor.complete_run()
                fail("a store carrying every attribute of the genuine one "
                     "completed a Formal run")
            except store_module.FormalAuthorityError:
                ok("authority state copied from a genuine store authorizes "
                   "nothing: the binding is by object identity")

            class Convenience(store_module.FormalStageFStore):
                def __post_init__(self):
                    pass

            subclass_root = Path(tmp) / "subclass"
            store_module.authoritative_root = lambda *_: subclass_root
            try:
                Convenience.create_authoritative()
                fail("a subclass inherited authoritative completion")
            except store_module.FormalAuthorityError:
                ok("a subclass cannot inherit authoritative completion, even "
                   "through the genuine factory")
            finally:
                store_module.authoritative_root = lambda *_: genuine_root

            # D — the audit's own forgery: find the binding structure and
            # insert a separately allocated store into it. There is nothing to
            # insert into, and the twin is refused regardless.
            twin = store_module.FormalStageFStore.__new__(
                store_module.FormalStageFStore)
            for name, value in vars(genuine).items():
                object.__setattr__(twin, name, value)
            for holder in list(store_module._AUTHORITY_CALLABLES):
                for value in _reachable_from(holder):
                    if isinstance(value, dict):
                        value[id(twin)] = twin
                    elif isinstance(value, list):
                        value.append(twin)
                    elif isinstance(value, set):
                        value.add(twin)
            try:
                store_module._verify_formal_authority(twin)
                fail("a separately allocated store was inserted into the "
                     "authority and passed verification")
                return
            except store_module.FormalAuthorityError:
                ok("a separately allocated twin cannot be inserted into the "
                   "authority: the mechanism holds no container to insert into")

            # E — mutating the exposed closure cell is not enough either: the
            # authorised store carries the same record and both are compared.
            cells = list(store_module._verify_formal_authority.__closure__ or ())
            original = [cell.cell_contents for cell in cells]
            object.__delattr__(twin, store_module._AUTHORITY_ATTR)
            try:
                for cell in cells:
                    cell.cell_contents = twin
                try:
                    store_module._verify_formal_authority(twin)
                    fail("rewriting a closure cell authorized another store")
                    return
                except store_module.FormalAuthorityError:
                    ok("rewriting the exposed closure cell authorizes no other "
                       "store")
            finally:
                for cell, value in zip(cells, original):
                    cell.cell_contents = value

            # F — the immutable record cannot be edited into naming another
            record = vars(genuine)[store_module._AUTHORITY_ATTR]
            for setter in (setattr, object.__setattr__):
                try:
                    setter(record, "store", twin)
                    fail("the authority record is mutable")
                    return
                except (AttributeError, TypeError):
                    continue
            ok("the authority record is immutable: neither setattr nor "
               "object.__setattr__ can redirect it")

            # G — object.__new__ / a bypassed __post_init__ confer nothing
            for build in (lambda: object.__new__(store_module.FormalStageFStore),
                          lambda: store_module.FormalStageFStore.__new__(
                              store_module.FormalStageFStore)):
                impostor = build()
                for name, value in vars(genuine).items():
                    if name != store_module._AUTHORITY_ATTR:
                        object.__setattr__(impostor, name, value)
                try:
                    store_module._verify_formal_authority(impostor)
                    fail("object.__new__ produced an authorized store")
                    return
                except store_module.FormalAuthorityError:
                    continue
            ok("object.__new__ and a bypassed __post_init__ confer no "
               "authority (2 probes)")

            # completion revalidates the exact root and the run kind
            genuine.root = Path(tmp) / "redirected"
            try:
                genuine.complete_run()
                fail("complete_run accepted a root mutated after construction")
            except store_module.FormalAuthorityError:
                ok("complete_run revalidates the exact authoritative root")
            genuine.root = genuine_root
            genuine.run_kind = reg.RUN_KIND_SMOKE
            try:
                genuine.complete_run()
                fail("complete_run accepted a mutated run kind")
            except store_module.FormalAuthorityError:
                ok("complete_run revalidates the Formal run kind")
        finally:
            store_module.authoritative_root = real

    signature = tuple(inspect.signature(
        store_module.FormalStageFStore.create_authoritative).parameters)
    if any(name in signature for name in ("root", "out", "store_root", "path")):
        fail(f"the Formal factory takes a root argument: {signature}")
    else:
        ok(f"the Formal factory takes no root argument {signature}")
    if hasattr(store_module.SmokeStageFStore, "complete_run"):
        fail("the smoke store exposes complete_run")
    else:
        ok("the smoke store has no completion path at all")

    # H — a redirected authoritative root, refused on the lexical form before
    # any canonical comparison can make a link look like itself.
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "elsewhere").mkdir()
        (base / "real_results").mkdir()
        link = base / "stage_f"
        link.symlink_to(base / "elsewhere")
        (base / "results").symlink_to(base / "real_results")
        dangling = base / "dangling"
        dangling.symlink_to(base / "never_created")
        redirected = {"a symlinked root": link,
                      "a symlinked component": base / "results" / "stage_f",
                      "a dangling symlinked root": dangling}
        real = store_module.authoritative_root
        try:
            for label, candidate in redirected.items():
                store_module.authoritative_root = lambda *_, c=candidate: c
                try:
                    store_module.FormalStageFStore.create_authoritative()
                except store_module.RootRedirectionError:
                    continue
                fail(f"{label} was accepted as the authoritative root")
                return
        finally:
            store_module.authoritative_root = real
        if any((base / "elsewhere").iterdir()):
            fail("a store was written through a symbolic link")
            return
        ok(f"a redirected authoritative root is refused before any canonical "
           f"comparison ({len(redirected)} probes), and nothing is written "
           "through the link")
    lexical = store_module.authoritative_root()
    if lexical != ROOT / "results/stage_f_v2" or lexical.is_symlink():
        fail(f"the authoritative root is not the lexical repository path: "
             f"{lexical}")
    else:
        ok(f"the authoritative root is the lexical repository path {lexical}")


# ---------------------------------------------------------------------------
# 6c. durability and the write-time schema / key / status barrier
# ---------------------------------------------------------------------------
def _verdict(n_evaluated: int | None = None, n_passed: int | None = None,
             passed: bool | None = None) -> dict:
    """The parent's own seven-of-seven agreement record (§9.6.4).

    Built from `rq3_protocol`'s frozen criterion tuple **and** its frozen
    per-criterion tolerances, so the validator cannot drift away from the
    definition it is supposed to be checking against. Each discrepancy sits on
    the side of its own tolerance that matches its verdict.
    """
    required = int(proto.N_REQUIRED_AGREEMENT_CRITERIA)
    n_evaluated = required if n_evaluated is None else n_evaluated
    n_passed = n_evaluated if n_passed is None else n_passed
    names = list(proto.REQUIRED_AGREEMENT_CRITERIA)[:n_evaluated]
    complete = n_evaluated == required and n_passed == required
    criteria = {}
    for index, name in enumerate(names):
        tolerance = float(proto.AGREEMENT_TOL[name])
        met = index < n_passed
        criteria[name] = {"discrepancy": tolerance / 2 if met else tolerance * 10,
                          "tolerance": tolerance, "passed": met}
    return {"passed": complete if passed is None else passed,
            "n_evaluated": n_evaluated, "n_required": required,
            "n_passed": n_passed,
            "reason": None if complete else "criteria beyond tolerance",
            "criteria": criteria}


def _valid_json(relpath: str) -> dict:
    """A schema-valid payload for one registered JSON artifact."""
    if relpath == reg.SVM_MACRO_JSON:
        return {"estimand": reg.SVM_ESTIMAND,
                "n_sites_contributing": reg.BASE_COHORT_SITES,
                "n_sites_not_estimable": 0,
                "carries_confidence_interval": reg.SVM_MACRO_CARRIES_CI,
                "comparison_to_logistic_regression": "NONE (§F.9)",
                "site_macro_bacc": 0.61, "status": reg.ESTIMABLE,
                "reason_code": None}
    return {"placeholder": True}


def _v_identities():
    """The run-time eligible-site identities, from the real verified registry."""
    import stage_f_inputs as inputs

    return inputs.derive_verified_split_identities()


def _v_eligible(n_labels) -> list:
    return sorted(_v_identities().sites_at(int(n_labels)))


def _v_ineligible(n_labels) -> list:
    import contracts

    return sorted(set(contracts.FROZEN_SITES)
                  - _v_identities().sites_at(int(n_labels)))


def _v_prefix(n_labels) -> list:
    """The prefix approximation this validator's fixture used before F.2c-4."""
    import contracts

    return list(contracts.FROZEN_SITES[:reg.SITES_PER_N[int(n_labels)]])


def _v_sites(method: str, n_labels):
    """One RQ4 cell's eligible sites, as the verified split registry has them."""
    import contracts

    if method != reg.METHOD_TARGET_LOCAL:
        return tuple(contracts.FROZEN_SITES)
    return tuple(_v_eligible(n_labels))


def _v_registry_rows(*, common_only: bool = False) -> list:
    """`(site, repeat, n_labels)` of the verified registry: `sum_N |elig(N)| . R`."""
    identities = _v_identities()
    return [(site, repeat, int(n))
            for n in identities.n_values()
            for site in sorted(identities.sites_at(n))
            if not (common_only and site not in reg.COMMON_SITES)
            for repeat in range(reg.R_SPLIT)]


def _v_block(population, block, operating_point, method, *, status=None,
             condition=None, verdict=None, p_raw=0.04) -> dict:
    """One `rq3_inference.BlockResult.to_payload()` record."""
    ok_status = proto.STATUS_OK if status is None else status
    estimable = ok_status == proto.STATUS_OK
    if verdict is None:
        # `BlockResult.numerical_validation` defaults to `{}` and `to_payload`
        # carries that empty mapping straight into the nested cell record.
        verdict = _verdict() if estimable else {}
    values = ({"lrt_statistic": 3.2, "lrt_pvalue_raw": p_raw,
               "delta_mcfadden": 0.01, "delta_cv_logloss": 0.02} if estimable
              else {f: proto.NOT_ESTIMABLE
                    for f in reg.SENS56_BLOCK_VALUE_FIELDS})
    return {"population": population, "block": block,
            "operating_point": operating_point, "method": method,
            "cell_index": proto.CELL_REGISTRY.index((operating_point, method)),
            "validation_unit": list(proto.VALIDATION_UNIT),
            "df": proto.BLOCK_DF[(population, block)], "status": ok_status,
            "failing_condition": condition,
            "failure_detail": (None if estimable
                               else (str(verdict["reason"]) if verdict
                                     else f"K40: {condition}")),
            "failure_is_numerical": (None if condition is None else
                                     bool(proto.is_numerical_failure(condition))),
            **values, "numerical_validation": verdict, "detail": {}}


def _v_blocks_by_cell() -> dict:
    """One deterministic block map shared by cells, units and families."""
    out = {}
    for population in reg.SENS56_POPULATIONS:
        names = [b for pop, b in proto.BLOCK_DF if pop == population]
        for index, (operating_point, method) in enumerate(proto.CELL_REGISTRY):
            out[(population, operating_point, method)] = {
                name: _v_block(population, name, operating_point, method,
                               p_raw=round(0.01 + 0.003 * index + 0.001 * j, 4))
                for j, name in enumerate(names)}
    return out


def _v_keys(relpath: str) -> list:
    """The table's **registered** logical keys, enumerated from the registry."""
    import contracts

    if relpath == "rq4/per_site_metrics.csv":
        return [{"method": m, "alpha": a, "n_labels": n, "held_out_site": s}
                for m, a, n in reg.REGISTERED_CELLS for s in _v_sites(m, n)]
    if relpath == "rq4/per_repeat_target_local.csv":
        return [{"alpha": a, "n_labels": n, "held_out_site": s, "repeat": r}
                for m, a, n in reg.REGISTERED_CELLS
                if m == reg.METHOD_TARGET_LOCAL for r in range(reg.R_SPLIT)
                for s in _v_sites(m, n)]
    if relpath in reg.EXACT_KEY_SET_TABLES:
        fields = reg.logical_key(relpath)
        return [dict(zip(fields, key))
                for key in reg.EXACT_KEY_SET_TABLES[relpath]]
    if relpath == reg.SVM_PER_SITE_TABLE:
        return [{"held_out_site": s} for s in contracts.FROZEN_SITES]
    if relpath == reg.SVM_FOLD_SELECTION_TABLE:
        return [{"held_out_site": s,
                 "pca": "no-PCA" if p is None else int(p), "C": float(c)}
                for s in contracts.FROZEN_SITES
                for p in reg.SVM_PCA_CANDIDATES for c in reg.SVM_C_GRID]
    if relpath in reg.RQ3_CELL_TABLES:
        return [{"population": p, "operating_point": o, "method": m}
                for p, o, m in _v_blocks_by_cell()]
    if relpath in reg.RQ3_STATUS_TABLES:
        return [{"population": p, "operating_point": o, "method": m, "block": b}
                for (p, o, m), blocks in _v_blocks_by_cell().items()
                for b in blocks]
    if relpath.endswith(reg.SENS56_FAMILIES_FILE) \
            and relpath.split("/")[1] in reg.SENS56_APPLIES_TO:
        return [{"population": p, "block": b} for p, b, _s in proto.HOLM_FAMILIES]
    if relpath in reg.SENS_DECISION_TABLES + reg.SENS_EVALUATION_TABLES:
        rows = _v_registry_rows(
            common_only=relpath.split("/")[1] == "sens4_common_site")
        offsets = (tuple(reg.SENS3_R_STAR_OFFSETS)
                   if "r_star_offset" in reg.logical_key(relpath) else (None,))
        tiers = [(reg.PARENT_STRATEGIES_LABEL_CONSUMING, rows)]
        if relpath in reg.SENS_EVALUATION_TABLES:
            tiers.append((reg.PARENT_STRATEGIES_ZERO_LABEL,
                          [t for t in rows if t[2] >= 10]))
        keys = []
        for offset in offsets:
            for strategies, subset in tiers:
                for site, repeat, n_labels in subset:
                    for strategy in strategies:
                        for method in reg.PARENT_CONFIDENCE_METHODS:
                            key = {"site": site, "repeat": repeat,
                                   "n_labels": n_labels, "strategy": strategy,
                                   "method": method}
                            if offset is not None:
                                key["r_star_offset"] = float(offset)
                            keys.append(key)
        return keys
    return []


def _sens_label_of(relpath: str) -> str:
    """The sensitivity directory name of a `sensitivities/<label>/...` path."""
    return relpath.split("/")[1]


def _v_sens_state(relpath: str, key: dict) -> dict:
    """A coherent inherited parent state for one SENS-1..4 row."""
    import certification as cert

    strategy = key["strategy"]
    grid = (cert.Q_GRID_SENSITIVITY
            if _sens_label_of(relpath) == reg.SENS2_LABEL else cert.Q_GRID)
    first, n_passed = 2, 3
    selected = first + n_passed - 1
    row = {"guarantee_tier": cert.tier_of(strategy),
           "state": cert.state_for(strategy, True),
           "selected_q": float(grid[selected]),
           "selected_candidate_index": selected,
           "selected_tau": 0.55, "r_star": 0.3}
    if relpath in reg.SENS_DECISION_TABLES:
        # §E.2a — the persisted label names the functional that actually ran,
        # which for SENS-1's exact-CP tiers is the overall-error bound. The
        # guarantee still derives from the *parent's* score type: the
        # correction renames the score, not the tier.
        parent_score = cert.STRATEGY_SCORE_TYPE[strategy]
        row.update({"score_type": sens.persisted_score_type(
                        _sens_label_of(relpath), strategy),
                    "guarantee": cert.GUARANTEE_BY_SCORE[parent_score],
                    "reason": None,
                    "selected_score": 0.25, "n_eligible": 10,
                    "first_eligible_index": first,
                    "n_passed_candidates": n_passed,
                    "stopped_at_index": selected + 1,
                    **{f: 4 for f in reg.SENS_CALIBRATION_COUNT_FIELDS}})
        return row
    row.update({"realized_field": cert.REALIZED_FIELD[strategy],
                "coverage_test": 0.8, "ber_test": 0.34,
                "risk_excess": 0.34 - 0.3, "realized_outcome": 1,
                **{f: 5 for f in reg.SENS_EVALUATION_COUNT_FIELDS}})
    return row


_ROW_CACHE: dict = {}


def _valid_rows(relpath: str, n: int | None = None) -> list:
    """Schema-valid rows for one registered table (memoised, shallow-copied)."""
    if n is None:
        if relpath not in _ROW_CACHE:
            _ROW_CACHE[relpath] = _build_valid_rows(relpath, None)
        return list(_ROW_CACHE[relpath])
    return _build_valid_rows(relpath, n)


def _build_valid_rows(relpath: str, n: int | None = None) -> list:
    """Schema-valid placeholder rows for one registered table.

    Written independently of the test suite's generator on purpose: the
    validator is a second opinion, not a re-run of the tests. Only the *shape*
    matters here — every frozen column, in the frozen order, with a distinct
    logical key and internally coherent status fields. No number below is, or
    resembles, a scientific result.
    """
    import contracts

    columns = reg.table_columns(relpath)
    key = reg.logical_key(relpath)
    sites = reg.SITE_IDENTIFIER_FIELDS.get(relpath, ())
    values = set(reg.STATUS_SEMANTICS.get(relpath, {}).get("value_fields", ()))
    fixed = {"reason_code": None, "failing_condition": None,
             "degeneracy_code": None, "class_composition_flag": None,
             "degeneracy_flag": False, "ci_low": 0.4, "ci_high": 0.6,
             "numerical_validation": _verdict(),
             "authoritative_nodes": proto.AUTHORITATIVE_NODES,
             "validation_nodes": proto.GH_NODES_VALIDATION,
             "total_bootstrap_replicates": reg.SITE_BOOTSTRAP_B,
             "valid_bootstrap_replicates": reg.MIN_VALID_REPLICATES,
             "n_sites": 23, "n_estimable_sites": 23}
    registered = _v_keys(relpath)
    blocks_by_cell = _v_blocks_by_cell()
    families = tuple(f"{reg.NAMESPACE_SENSITIVITIES}/{lab}/"
                     f"{reg.SENS56_FAMILIES_FILE}"
                     for lab in reg.SENS56_APPLIES_TO)
    rows = []
    for i in range(reg.expected_rows(relpath) if n is None else n):
        keys = registered[i] if i < len(registered) else {}
        row = {}
        for column in columns:
            if column in keys:
                row[column] = keys[column]
            elif column in key:
                row[column] = f"{column}_{i}"
            elif column in sites:
                row[column] = contracts.FROZEN_SITES[i % len(contracts.FROZEN_SITES)]
            elif column in values:
                row[column] = 0.5
            elif column == "status":
                row[column] = (proto.STATUS_OK
                               if relpath in reg.RQ3_STATUS_TABLES
                               else reg.ESTIMABLE)
            elif column in fixed:
                row[column] = fixed[column]
            elif column in reg.ESTIMABLE_REQUIRED_POSITIVE_COUNTS.get(relpath, ()):
                row[column] = i + 1          # an ESTIMABLE row came from data
            elif column == reg.CONVERGENCE_FLAG_FIELDS.get(relpath):
                row[column] = True
            elif column in reg.closed_domain_fields(relpath):
                # a registered categorical state, cycled through the frozen
                # vocabulary so `no-PCA` — the real F-SVM output — is exercised
                domain = reg.closed_domain(relpath, column)
                row[column] = domain[i % len(domain)]
            elif relpath in reg.RQ3_CELL_TABLES \
                    and column == reg.RQ3_CELL_VALUE_FIELD:
                row[column] = 0.25          # a reported §13 variance …
            elif relpath in reg.RQ3_CELL_TABLES \
                    and column == reg.RQ3_CELL_CONDITION_FIELD:
                row[column] = None          # … carries no failing condition
            else:
                row[column] = i
        cell = (row.get("population"), row.get("operating_point"),
                row.get("method"))
        if relpath in reg.RQ3_CELL_TABLES:
            row[reg.SENS56_CELLS_BLOCKS_FIELD] = blocks_by_cell[cell]
        elif relpath in reg.RQ3_STATUS_TABLES:
            record = blocks_by_cell[cell][row["block"]]
            row["status"] = record["status"]
            row["failing_condition"] = record["failing_condition"]
            row["numerical_validation"] = (
                record["numerical_validation"]
                or store_module._canonical_empty_verdict(
                    record["failing_condition"]))
        elif relpath in families:
            row["m"] = reg.SENS56_HOLM_FAMILY_SIZE
            row["alpha"] = proto.ALPHA
            row[reg.SENS56_FAMILY_NOT_ESTIMABLE_FIELD] = 0
            row[reg.SENS56_FAMILIES_SLOTS_FIELD] = [
                {"cell_index": index, "operating_point": op, "method": meth,
                 "p_raw": blocks_by_cell[(row["population"], op, meth)]
                          [row["block"]]["lrt_pvalue_raw"],
                 "p_holm": 1.0, "significant": False}
                for index, (op, meth) in enumerate(proto.CELL_REGISTRY)]
        elif keys and relpath in (reg.SENS_DECISION_TABLES
                                  + reg.SENS_EVALUATION_TABLES):
            row.update(_v_sens_state(relpath, keys))
        rows.append(row)
    if relpath in (reg.SVM_PER_SITE_TABLE, reg.SVM_FOLD_SELECTION_TABLE):
        _v_svm_coherence(relpath, rows)
    return rows


def _v_svm_winner(site: str):
    """The one candidate this synthetic fold selects, per site.

    A **strict** argmax, rotated across the grid so the fixture does not lean on
    the tie-break: the winner's score is unique, and the §F.7.5 order is probed
    separately by `check_svm_selection_and_tie_break`.
    """
    import contracts

    grid = [("no-PCA" if p is None else int(p), float(c))
            for p in reg.SVM_PCA_CANDIDATES for c in reg.SVM_C_GRID]
    return grid[contracts.FROZEN_SITES.index(site) % len(grid)]


def _v_svm_score(site: str, pca, penalty: float) -> float:
    return 0.9 if (pca, float(penalty)) == _v_svm_winner(site) else 0.5


def _v_svm_coherence(relpath: str, rows: list) -> None:
    """Make the two SVM fixtures one coherent run, in place.

    The cross-artifact rules derive the fold verdict from the candidate rows, so
    an incoherent placeholder pair is now itself a violation. These fixtures are
    the *lawful* ESTIMABLE state: 20 ESTIMABLE candidates per site, a strict
    argmax, and a per-site row whose selection, retained score, `converged` flag
    and `n_iter_max` all agree with it. No number here is a scientific result.
    """
    if relpath == reg.SVM_FOLD_SELECTION_TABLE:
        for row in rows:
            row["inner_macro_bacc"] = _v_svm_score(
                row["held_out_site"], row["pca"], row["C"])
            row["status"] = reg.ESTIMABLE
            row["reason_code"] = None
        return
    for row in rows:
        pca, penalty = _v_svm_winner(row["held_out_site"])
        row["selected_pca"] = pca
        row["selected_C"] = penalty
        row["inner_macro_bacc"] = _v_svm_score(row["held_out_site"],
                                               pca, penalty)
        row["n_iter_max"] = reg.SVM_MAX_ITER
        row["converged"] = True
        row["status"] = reg.ESTIMABLE
        row["reason_code"] = None
        row["bacc"] = 0.55


def _not_estimable(relpath: str, reason: str | None = None) -> dict:
    """The updates that make one artifact's row coherently `NOT_ESTIMABLE`.

    Read from the registry per artifact: after F.2c-2 there is no single reason
    code, and no single field state, that fits every status-carrying table.
    """
    semantics = reg.STATUS_SEMANTICS[relpath]
    reason = reason or reg.artifact_reason_codes(relpath, reg.NOT_ESTIMABLE)[0]
    update = {"status": reg.NOT_ESTIMABLE, "reason_code": reason}
    for field in (tuple(semantics["value_fields"])
                  + tuple(semantics.get("interval_fields", ()))
                  + tuple(reg.NOT_ESTIMABLE_UNAVAILABLE_EXTRA.get(relpath, ()))):
        update[field] = None
    for field in reg.REASON_REQUIRED_ZERO_COUNTS.get((relpath, reason), ()):
        update[field] = 0
    for field in reg.REASON_REQUIRED_POSITIVE_COUNTS.get((relpath, reason), ()):
        update[field] = 7
    flag = reg.CONVERGENCE_FLAG_FIELDS.get(relpath)
    if flag is not None:
        update[flag] = reg.CONVERGENCE_FLAG_BY_REASON.get((relpath, reason), False)
    return update


def _fill_store(store) -> None:
    """Every registered artifact, each one schema-valid. Writes to a temp root."""
    for relpath in reg.registered_artifact_paths():
        if relpath in reg.BARRIER_ARTIFACTS:
            continue
        if relpath.endswith(".json"):
            store.write_json(relpath, _valid_json(relpath))
        else:
            store.write_table(relpath, _valid_rows(relpath))


def check_split_registry_identity_binding() -> None:
    """F.2c-4 — the persisted site x `N` identities are the verified registry's.

    Every probe writes a real table through the real write barrier, or runs a
    real completion barrier against a temporary root. Nothing here reads source
    text: the substituted identities are constructed, offered and refused.
    """
    import tempfile

    import contracts
    import stage_f_inputs as inputs

    per_site = "rq4/per_site_metrics.csv"
    per_repeat = "rq4/per_repeat_target_local.csv"
    decisions = reg.SENS_DECISION_TABLES[0]

    identities = inputs.derive_verified_split_identities(use_cache=False)
    if identities.registry_sha256 != \
            reg.CONSUMED_ARTIFACTS[inputs.SPLIT_REGISTRY_FILE]:
        fail("the derived identities did not come from the pinned registry")
        return
    if {n: len(v) for n, v in identities.sites_by_n} != reg.SITES_PER_N:
        fail("the verified registry does not hold the frozen per-N counts")
        return
    differing = [n for n in reg.SITES_PER_N
                 if set(_v_prefix(n)) != set(_v_eligible(n))]
    if differing != [10, 20, 40]:
        fail(f"the prefix approximation differs at {differing}, expected "
             "[10, 20, 40] — the F.2c-4 finding no longer reproduces")
        return
    ok("the eligible identities are derived from a registry that verified, and "
       f"differ from a same-count frozen-site prefix at N={differing}")

    def swap_map(n_labels):
        victim = _v_eligible(n_labels)[0]
        smaller = {5: 5, 10: 5, 20: 10, 40: 20}[int(n_labels)]
        return {victim: sorted(set(_v_eligible(smaller))
                               - set(_v_eligible(n_labels)))[0]}

    def substitute(relpath, n_labels, site_field="held_out_site"):
        """One identity traded, at `N` **and every larger budget**.

        Applying it only at `N` would leave the displaced site inside a nested
        larger-`N` set and the shape layer would catch it as a nesting break.
        Carried upward, the counts, the universe, the per-`alpha` agreement, the
        nesting and the cross-table agreement are all still satisfied — and only
        the identities have moved.
        """
        swap = swap_map(n_labels)
        out = []
        for row in _valid_rows(relpath):
            row = dict(row)
            budget = row.get("n_labels")
            if budget is not None and int(budget) >= int(n_labels):
                row[site_field] = swap.get(row[site_field], row[site_field])
            out.append(row)
        return out

    with tempfile.TemporaryDirectory() as tmp:
        made = [0]

        def fresh():
            made[0] += 1
            return store_module.SmokeStageFStore.create(
                Path(tmp) / f"identity_{made[0]}")

        def refuses(label, relpath, rows, target=None, expect=None):
            target = target or fresh()
            try:
                target.write_table(relpath, rows)
            except store_module.StageFStoreError as error:
                if expect is not None and expect not in str(error):
                    fail(f"{relpath} refused {label} for the wrong reason: {error}")
                    return False
                return True
            fail(f"{relpath} accepted {label}")
            return False

        # 1 — the exact verified identities are accepted
        store = fresh()
        try:
            store.write_table(per_site, _valid_rows(per_site))
            store.write_table(per_repeat, _valid_rows(per_repeat))
            store.write_table(decisions, _valid_rows(decisions))
        except Exception as error:
            fail(f"the verified identities were refused: {error}")
            return
        recorded = store._validated[per_repeat]["split_identity"]
        if dict(recorded["sites_by_n"]) != {n: tuple(_v_eligible(n))
                                            for n in reg.N_GRID}:
            fail("the accepted table recorded identities other than the "
                 "verified ones")
            return
        probes = 3

        # 2 — the prefix approximation, which every shape check accepts
        import pandas as pd

        for relpath in (per_site, per_repeat):
            rows = []
            for row in _valid_rows(relpath):
                row = dict(row)
                n = row["n_labels"]
                if n in (10, 20, 40):
                    order = _v_eligible(n)
                    row["held_out_site"] = _v_prefix(n)[
                        order.index(row["held_out_site"])]
                rows.append(row)
            frame = pd.DataFrame(rows)[list(reg.table_columns(relpath))]
            if store_module._validate_structural_cells(relpath, frame) != len(rows):
                fail(f"{relpath}: the shape layer already refused the prefix "
                     "approximation; the identity probe proves nothing")
                return
            if not refuses("prefix-derived site sets", relpath, rows,
                           expect="verified split registry"):
                return
            probes += 1

        # 3, 5, 6 — one substitution: count, universe and nesting all intact
        for n_labels in (10, 20, 40):
            rows = substitute(per_site, n_labels)
            persisted = {r["held_out_site"] for r in rows
                         if r["method"] == reg.METHOD_TARGET_LOCAL
                         and r["n_labels"] == n_labels}
            if len(persisted) != reg.SITES_PER_N[n_labels] \
                    or not persisted <= set(contracts.FROZEN_SITES):
                fail(f"the N={n_labels} substitution changed the count or left "
                     "the frozen universe; it is not the bypass under audit")
                return
            frame = pd.DataFrame(rows)[list(reg.table_columns(per_site))]
            if store_module._validate_structural_cells(per_site, frame) != len(rows):
                fail(f"the N={n_labels} substitution was caught by a shape "
                     "check, so it does not exercise the identity binding")
                return
            if not refuses(f"a substituted identity at N={n_labels}", per_site,
                           rows, expect=f"at N={n_labels}"):
                return
            probes += 1

        # 4 — both target-local tables substituted identically
        store = fresh()
        left = substitute(per_site, 40)
        right = substitute(per_repeat, 40)
        seen = ({r["held_out_site"] for r in left
                 if r["method"] == reg.METHOD_TARGET_LOCAL and r["n_labels"] == 40},
                {r["held_out_site"] for r in right if r["n_labels"] == 40})
        if seen[0] != seen[1]:
            fail("the two substituted tables do not agree, so cross-table "
                 "consistency is not what is being bypassed")
            return
        for relpath, rows in ((per_site, left), (per_repeat, right)):
            if not refuses("a cross-table-consistent substitution", relpath,
                           rows, target=store):
                return
            probes += 1

        # 7 — no binding at all
        unbound = fresh()
        object.__setattr__(unbound, "_splits", None)
        for relpath in (per_site, per_repeat, decisions):
            if not refuses("a write with no verified binding", relpath,
                           _valid_rows(relpath), target=unbound,
                           expect="no verified split-registry binding"):
                return
            probes += 1
        forged = fresh()
        object.__setattr__(forged, "_splits",
                           {n: _v_eligible(n) for n in reg.SITES_PER_N})
        if not refuses("a hand-built mapping in place of a binding", per_site,
                       _valid_rows(per_site), target=forged,
                       expect="only the immutable record"):
            return
        probes += 1

        # 8 — verified for one identity set, artifacts written from another
        moved = identities._replace(sites_by_n=tuple(
            (n, tuple(sorted({swap_map(40).get(x, x) for x in v})) if n == 40
             else v) for n, v in identities.sites_by_n))
        elsewhere = store_module.SmokeStageFStore.create(
            Path(tmp) / "elsewhere", split_identities=moved)
        if not refuses("the real identities under a foreign binding", per_site,
                       _valid_rows(per_site), target=elsewhere, expect="at N=40"):
            return
        probes += 1

        # §7 — the SENS tables carry the same axis
        for relpath in (reg.SENS_DECISION_TABLES[0],
                        reg.SENS_EVALUATION_TABLES[0],
                        "sensitivities/sens4_common_site/decisions.csv"):
            if not refuses("a substituted SENS site axis", relpath,
                           substitute(relpath, 40, site_field="site"),
                           expect="at N=40"):
                return
            probes += 1
        ok(f"the persisted site x N identities must be the verified split "
           f"registry's, not a same-count same-universe nested substitution "
           f"({probes} probes plus three positive controls)")

    # 10 / 11 — the complete synthetic persistence barrier
    with tempfile.TemporaryDirectory() as tmp:
        real_root = store_module.authoritative_root
        genuine = Path(tmp) / "stage_f"
        store_module.authoritative_root = lambda *_: genuine
        try:
            store = store_module.FormalStageFStore.create_authoritative()
            if type(store._splits) is not inputs.VerifiedSplitIdentities \
                    or store._splits.sites_by_n != identities.sites_by_n:
                fail("the Formal factory did not bind the verified identities")
                return
            _fill_store(store)
            object.__setattr__(store, "_splits", moved)
            try:
                store.complete_run()
            except store_module.CompletionBarrierError as error:
                if "re-verified at the barrier" not in str(error):
                    fail(f"the barrier refused for the wrong reason: {error}")
                    return
            else:
                fail("a complete 33-artifact store sealed against substituted "
                     "identities")
                return
            if (genuine / reg.COMPLETION_FILE).exists():
                fail("COMPLETION.json was written despite the identity refusal")
                return
            object.__setattr__(store, "_splits", None)
            try:
                store.complete_run()
            except store_module.CompletionBarrierError:
                pass
            else:
                fail("a store that lost its binding still completed")
                return
            object.__setattr__(store, "_splits", identities)
            completion = store.complete_run()
        finally:
            store_module.authoritative_root = real_root
        binding = completion.get("split_registry_identity_binding", {})
        if binding.get("n_identity_bound_artifacts") != \
                len(reg.SPLIT_IDENTITY_TABLES) \
                or binding.get("registry_sha256") != identities.registry_sha256 \
                or binding.get("eligible_sites_by_n") != {
                    str(n): list(v) for n, v in identities.sites_by_n}:
            fail(f"the completion marker does not record the binding: {binding}")
            return
        ok("a complete 33-artifact barrier seals only against the re-derived "
           f"identities, and records all {len(reg.SPLIT_IDENTITY_TABLES)} "
           "identity-bound artifacts (3 barrier runs)")


def check_formal_wiring_binds_identities() -> None:
    """The genuine Formal call graph cannot reach persistence without a binding.

    `execute_formal` has one store-construction path and it is the authority
    factory; the factory derives the identities itself and exposes no argument
    through which another set could be supplied; and the barrier re-derives.
    """
    import ast
    import inspect

    import stage_f_execution as execution
    import stage_f_inputs as inputs

    if list(inspect.signature(
            store_module.FormalStageFStore.create_authoritative
            ).parameters) != ["provenance"]:
        fail("create_authoritative gained a parameter beyond `provenance`")
        return
    tree = ast.parse(inspect.getsource(execution))
    constructors = {
        f"{ast.unparse(node.func)}"
        for node in ast.walk(tree) if isinstance(node, ast.Call)
        and "FormalStageFStore" in ast.unparse(node.func)}
    if constructors != {"store_module.FormalStageFStore.create_authoritative"}:
        fail(f"stage_f_execution reaches the Formal store by {constructors}")
        return
    ok("the only Formal store construction in stage_f_execution is the "
       "authority factory, and it takes no eligible-site argument")

    # the factory really runs the derivation, not a stored copy
    calls = 0
    genuine = inputs.derive_verified_split_identities

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return genuine(*args, **kwargs)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        real_root = store_module.authoritative_root
        store_module.authoritative_root = lambda *_: Path(tmp) / "stage_f"
        inputs.derive_verified_split_identities = counting
        try:
            store = store_module.FormalStageFStore.create_authoritative()
        finally:
            inputs.derive_verified_split_identities = genuine
            store_module.authoritative_root = real_root
    if calls != 1 or store._splits is None:
        fail(f"the authority factory called the derivation {calls} times")
        return
    if "_splits" not in {f.name for f in
                         __import__("dataclasses").fields(
                             store_module.FormalStageFStore)}:
        fail("the binding is not a declared field of the Formal store")
        return
    ok("the authority factory derives the binding on every genuine Formal "
       "construction, and the barrier re-derives it without the memo")


def check_genuine_provenance_round_trip() -> None:
    """The **genuine** §J.1 preflight provenance survives its own persistence.

    Stage F.4 Attempt #4 failed here and nowhere else: the run provenance
    carried `preflight.split_registry.sites_per_n` keyed by the integer label
    budgets, canonical JSON persists object keys as strings, and §I.1 step 5
    compares the in-memory payload with the re-read file for exact equality. The
    earlier probes missed it because they drove the barrier with the factory's
    *default* provenance, which carries no preflight block at all. This one
    builds the real report, from the real frozen inputs, and completes a
    disposable store against it. Nothing scientific is computed and
    `results/stage_f/` is never touched.
    """
    import json
    import tempfile

    try:
        preflight = inputs.preflight(require_empty_store=True)
    except AssertionError as exc:
        fail(f"the genuine preflight did not run: {exc}")
        return

    sites_per_n = preflight["split_registry"]["sites_per_n"]
    if sites_per_n != reg.sites_per_n_canonical() \
            or any(not isinstance(key, str) for key in sites_per_n):
        fail(f"preflight.split_registry.sites_per_n is {sites_per_n!r}; the "
             "canonical in-memory form is the string-keyed "
             f"{reg.sites_per_n_canonical()!r}")
        return
    ok("preflight.split_registry.sites_per_n carries canonical string keys, "
       "derived from the frozen SITES_PER_N")

    provenance = store_module.build_provenance(reg.RUN_KIND_FORMAL,
                                               {"preflight": preflight})
    if json.loads(store_module.canonical_json(provenance)) != provenance:
        fail("the genuine run provenance does not survive the canonical JSON "
             "round trip; §I.1 step 5 would refuse every Formal run")
        return
    ok("the genuine run provenance round-trips through canonical JSON exactly")

    with tempfile.TemporaryDirectory() as tmp:
        genuine_root = Path(tmp) / "authoritative"
        real_root = store_module.authoritative_root
        store_module.authoritative_root = lambda *_: genuine_root
        try:
            store = store_module.FormalStageFStore.create_authoritative(
                provenance)
            persisted = json.loads(
                (store.root / reg.PROVENANCE_FILE).read_bytes())
            if persisted != store.provenance:
                fail("the persisted PROVENANCE.json differs from the run's own "
                     "provenance before the barrier even runs")
                return
            _fill_store(store)
            completion = store.complete_run()
        finally:
            store_module.authoritative_root = real_root

    if completion.get("steps") != list(reg.BARRIER_STEPS):
        fail(f"the barrier ran {completion.get('steps')} on the genuine "
             "provenance")
        return
    ok(f"a complete store carrying the genuine preflight provenance passes all "
       f"{len(reg.BARRIER_STEPS)} barrier steps and writes COMPLETION.json last")


def check_completion_integrity() -> None:
    """The four artifact books, the write-time gate, and the observed order.

    Driven against a **temporary** root: a store is genuinely built, filled with
    schema-valid placeholders, refused once, then completed, and the phase order
    is read off spies rather than off the source text. Nothing scientific is
    computed and `results/stage_f/` is never touched.
    """
    import tempfile

    covered = set(reg.expected_checksum_paths())
    if covered != set(reg.registered_artifact_paths()) - {reg.CHECKSUM_FILE,
                                                          reg.COMPLETION_FILE}:
        fail("the checksum-covered set is not derived from the output registry")
        return
    ok(f"the checksum-covered set is derived from the registry "
       f"({len(covered)} paths; the ledger and the marker are excluded)")

    events: list = []
    completion: dict = {}
    genuine_write = store_module._atomic_write
    genuine_verify = store_module._verify_formal_authority
    genuine_reread = store_module._StageFRecordStore.read_checksum_ledger
    genuine_sets = reg.expected_checksum_paths

    with tempfile.TemporaryDirectory() as tmp:
        genuine_root = Path(tmp) / "authoritative"
        real_root = store_module.authoritative_root
        store_module.authoritative_root = lambda *_: genuine_root
        try:
            store = store_module.FormalStageFStore.create_authoritative()
            _fill_store(store)

            # E — an artifact that skipped the write-time gate blocks completion
            held = store._validated.pop("rq4/degeneracy.csv")
            try:
                store.complete_run()
                fail("completion accepted a written-but-unvalidated artifact")
                return
            except store_module.CompletionBarrierError:
                ok("a written-but-unvalidated artifact blocks completion")
            store._validated["rq4/degeneracy.csv"] = held

            store_module._atomic_write = lambda t, b, fsync_up_to=None: (
                events.append(("write", Path(t).name))
                or genuine_write(t, b, fsync_up_to=fsync_up_to))
            store_module._verify_formal_authority = lambda s: (
                events.append(("authority", None)) or genuine_verify(s))
            store_module._StageFRecordStore.read_checksum_ledger = (
                lambda self, **kw: (events.append(("ledger_reread", None))
                                    or genuine_reread(self, **kw)))
            reg.expected_checksum_paths = lambda: (
                events.append(("expected_set", None)) or genuine_sets())
            try:
                completion = store.complete_run()
            finally:
                store_module._atomic_write = genuine_write
                store_module._verify_formal_authority = genuine_verify
                store_module._StageFRecordStore.read_checksum_ledger = \
                    genuine_reread
                reg.expected_checksum_paths = genuine_sets
        finally:
            store_module.authoritative_root = real_root

    # D — the four books agreed exactly
    if completion.get("artifact_sets_identical") != ["ledgered", "registered",
                                                     "validated", "written"] \
            or completion.get("n_checksum_covered_artifacts") != len(covered):
        fail("completion did not record the four-way artifact-set equality")
    else:
        ok("registered == written == validated == ledgered, exactly "
           f"({len(covered)} artifacts)")

    # P — the observed completion order
    kinds = [kind for kind, _ in events]
    try:
        ledger_written = events.index(("write", reg.CHECKSUM_FILE))
        reread = kinds.index("ledger_reread")
        four_way = len(kinds) - 1 - kinds[::-1].index("expected_set")
        last_authority = len(kinds) - 1 - kinds[::-1].index("authority")
        marker = events.index(("write", reg.COMPLETION_FILE))
    except ValueError as error:
        fail(f"a completion phase never ran: {error}")
        return
    ordered = (kinds[0] == "authority" and ledger_written < reread < four_way
               < last_authority < marker == len(events) - 1)
    if not ordered:
        fail(f"the observed completion order is wrong: {kinds}")
    else:
        ok("observed completion order: authority -> ledger write -> ledger "
           "reread -> four-way set equality -> authority -> COMPLETION last")


def check_persistence_hardening() -> None:
    """Every registered write is fsynced, and nothing bypasses the validators."""
    import os as real_os
    import stat
    import tempfile

    # D/E/F — file fsync, nested directory fsync, ledger fsync, by observation
    events: list[str] = []
    genuine_fsync = real_os.fsync

    def spy(fd):
        try:
            kind = ("dir" if stat.S_ISDIR(real_os.fstat(fd).st_mode) else "file")
        except OSError:                                    # pragma: no cover
            kind = "unknown"
        events.append(kind)
        return genuine_fsync(fd)

    relpath = "sensitivities/sens5_s1_attribution/families.csv"
    columns = reg.table_columns(relpath)
    key = reg.logical_key(relpath)
    rows = [{c: (f"{c}_{i}" if c in key else i) for c in columns}
            for i in range(reg.expected_rows(relpath))]
    with tempfile.TemporaryDirectory() as tmp:
        store_module.os.fsync = spy
        try:
            store = store_module.SmokeStageFStore.create(Path(tmp) / "probe")
            events.clear()
            store.write_table(relpath, rows)
            csv_events = list(events)
            events.clear()
            store.write_checksum_ledger()
            ledger_events = list(events)
            reread = store.read_checksum_ledger(require_exact=False)
        finally:
            store_module.os.fsync = genuine_fsync

    if not csv_events or csv_events[0] != "file":
        fail(f"a registered CSV is not file-fsynced before its rename: {csv_events}")
    else:
        ok("every registered CSV is file-fsynced before the atomic rename")
    if csv_events.count("dir") < 3:
        fail("the nested directory chain is not fsynced up to the store root "
             f"({csv_events.count('dir')} directory fsyncs)")
    else:
        ok(f"the nested directory chain is fsynced up to the store root "
           f"({csv_events.count('dir')} directory fsyncs)")
    if "file" not in ledger_events:
        fail("the checksum ledger is not file-fsynced")
    elif relpath not in reread["entries"]:
        fail("the reread ledger does not carry the artifact it ledgered")
    else:
        ok("the checksum ledger is file-fsynced and independently reread")

    source = (ROOT / "src" / "stage_f_store.py").read_text()
    tree = ast.parse(source)
    csv_calls = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "to_csv"]
    direct = [line for line in csv_calls
              if "_atomic_write" not in source.splitlines()[line - 1]
              and "encode" not in source.splitlines()[line - 1]]
    if direct:
        fail(f"a CSV is serialised outside the durable write path: {direct}")
    else:
        ok("no CSV reaches disk outside the atomic, fsynced write path")

    # G/H/I/J — schema, logical key and status semantics, by negative probe
    with tempfile.TemporaryDirectory() as tmp:
        store = store_module.SmokeStageFStore.create(Path(tmp) / "probe")
        probes = (
            ("an extra column", store_module.SchemaViolationError,
             [dict(r, unregistered_extra=1) for r in rows]),
            ("a missing column", store_module.SchemaViolationError,
             [{k: v for k, v in r.items() if k != "alpha"} for r in rows]),
            ("a duplicate logical key", store_module.DuplicateLogicalKeyError,
             [dict(rows[0]) for _ in rows]),
        )
        for label, expected, payload in probes:
            try:
                store.write_table(relpath, payload)
            except expected:
                continue
            except Exception as error:                     # pragma: no cover
                fail(f"{label} raised {type(error).__name__}, not {expected.__name__}")
                break
            fail(f"{label} was accepted at write time")
            break
        else:
            ok("malformed tables fail closed at write time "
               "(extra column, missing column, duplicate logical key)")

        per_site = "rq4/per_site_metrics.csv"
        # the registered site x cell enumeration: §H.2 fixes which cells exist,
        # so a hand-rolled key would no longer be a valid table at all
        status_rows = [
            {**{c: ("BOTH_CLASSES" if c == "class_composition_flag"
                    else reg.ESTIMABLE if c == "status"
                    else None if c in ("reason_code", "degeneracy_code")
                    else False if c == "degeneracy_flag"
                    else 0.5 if c in reg.REGISTERED_RATES
                    else i)
                for c in reg.table_columns(per_site)}, **keys}
            for i, keys in enumerate(_v_keys(per_site))]
        status_probes = (
            ("ESTIMABLE with a missing value", {"coverage": None}),
            ("NOT_ESTIMABLE with no reason",
             {"status": reg.NOT_ESTIMABLE, "reason_code": None,
              **{r: None for r in reg.REGISTERED_RATES}}),
            ("NOT_ESTIMABLE with a zero substitute",
             {"status": reg.NOT_ESTIMABLE, "reason_code": "NO_EVALUATION_POINTS",
              **{r: 0.0 for r in reg.REGISTERED_RATES}}),
            ("degeneracy serialised as NOT_ESTIMABLE",
             {"status": reg.NOT_ESTIMABLE, "reason_code": "NO_EVALUATION_POINTS",
              "degeneracy_flag": True,
              "degeneracy_code": "ALL_INCLUSIVE_THRESHOLD",
              **{r: None for r in reg.REGISTERED_RATES}}),
        )
        for label, patch in status_probes:
            payload = [dict(r) for r in status_rows]
            payload[0].update(patch)
            try:
                store.write_table(per_site, payload)
            except store_module.StatusSemanticsError:
                continue
            except Exception as error:                     # pragma: no cover
                fail(f"{label} raised {type(error).__name__}")
                break
            fail(f"{label} was accepted at write time")
            break
        else:
            ok("status/reason semantics fail closed at write time "
               "(4 probes, incl. degeneracy wrongly marked NOT_ESTIMABLE)")

        for withdrawn in reg.WITHDRAWN_REASON_CODES:
            payload = [dict(r) for r in status_rows]
            payload[0].update({"status": reg.NOT_ESTIMABLE,
                               "reason_code": withdrawn,
                               **{r: None for r in reg.REGISTERED_RATES}})
            try:
                store.write_table(per_site, payload)
                fail(f"the withdrawn reason {withdrawn} was accepted")
                break
            except reg.StageFRegistryError:
                continue
        else:
            ok(f"the {len(reg.WITHDRAWN_REASON_CODES)} withdrawn reason codes "
               "remain rejected at write time")

    # J — every sensitivity CSV has a fixed schema, none is free-form
    unschematised = [p for p in reg.registered_artifact_paths()
                     if p.endswith(".csv") and not reg.table_columns(p)]
    if unschematised:
        fail(f"registered CSVs with no frozen schema: {unschematised}")
    else:
        ok(f"all {len(reg.TABLE_COLUMNS)} registered CSVs carry an exact frozen "
           "schema, including every sensitivity table")
    if reg.COLUMN_PROJECTION_PERMITTED or reg.DUPLICATE_LOGICAL_KEYS_PERMITTED:
        fail("projection or duplicate logical keys are declared permitted")
    else:
        ok("column projection and duplicate logical keys are both prohibited")

    # F/G/H/I/J — the ledger parser refuses every unsafe entry, line by line
    with tempfile.TemporaryDirectory() as tmp:
        probe = store_module.SmokeStageFStore.create(Path(tmp) / "probe")
        digest = "c" * 64
        good = f"{digest}  rq4/degeneracy.csv"
        ledger_probes = (
            ("a duplicate entry", [good, good]),
            ("a self-entry", [f"{digest}  {reg.CHECKSUM_FILE}"]),
            ("a completion-marker entry", [f"{digest}  {reg.COMPLETION_FILE}"]),
            ("an absolute path", [f"{digest}  /etc/passwd"]),
            ("a traversing path", [f"{digest}  ../outside.csv"]),
            ("an interior traversal", [f"{digest}  rq4/../rq4/degeneracy.csv"]),
            ("an unregistered path", [f"{digest}  rq4/unknown_table.csv"]),
            ("a canonical alias", [f"{digest}  RQ4/degeneracy.csv"]),
            ("a malformed digest", ["not-a-digest  rq4/degeneracy.csv"]),
            ("a malformed line", ["no-separator"]),
        )
        refused = 0
        for label, lines in ledger_probes:
            (probe.root / reg.CHECKSUM_FILE).write_text("\n".join(lines) + "\n")
            try:
                probe.read_checksum_ledger(require_exact=False)
            except store_module.CompletionBarrierError:
                refused += 1
                continue
            fail(f"the checksum ledger accepted {label}")
            break
        else:
            ok(f"the ledger parser refuses all {refused} unsafe entry forms "
               "(duplicate, self, marker, absolute, traversal, alias, "
               "unregistered, malformed)")

        # a ledger missing an expected artifact is refused as well
        (probe.root / reg.CHECKSUM_FILE).write_text(good + "\n")
        try:
            probe.read_checksum_ledger()
            fail("the checksum ledger accepted an incomplete artifact set")
        except store_module.CompletionBarrierError:
            ok("a ledger that omits a registered artifact is refused")

    # K/L/M/N/O — scientific value semantics, by negative probe
    with tempfile.TemporaryDirectory() as tmp:
        probe = store_module.SmokeStageFStore.create(Path(tmp) / "semantics")
        intervals, macro = reg.INTERVAL_TABLE, "rq4/site_macro_metrics.csv"
        units = reg.RQ3_STATUS_TABLES[0]
        floor = reg.MIN_VALID_REPLICATES
        semantic_probes = (
            ("a stringified coverage", "rq4/per_site_metrics.csv",
             {"coverage": "0.5"}, store_module.NumericSemanticsError),
            ("a stringified balanced accuracy", "svm/per_site_bacc.csv",
             {"bacc": "0.61"}, store_module.NumericSemanticsError),
            ("a valid CI row with no lower endpoint", intervals,
             {"ci_low": None}, store_module.StatusSemanticsError),
            ("an inverted CI", intervals, {"ci_low": 0.9, "ci_high": 0.1},
             store_module.StatusSemanticsError),
            ("a valid CI below the validity floor", intervals,
             {"valid_bootstrap_replicates": floor - 1},
             store_module.StatusSemanticsError),
            ("the validity floor invoked above the floor", intervals,
             {"reason_code": reg.CI_VALIDITY_FLOOR_REASON,
              "ci_low": reg.NOT_ESTIMABLE, "ci_high": reg.NOT_ESTIMABLE},
             store_module.StatusSemanticsError),
            ("a NOT_ESTIMABLE aggregate keeping its worst site", macro,
             {"status": reg.NOT_ESTIMABLE,
              "reason_code": "NO_ESTIMABLE_SITE_IN_CELL",
              "worst_site_coverage": None,
              **{f"site_macro_{r}": None for r in reg.REGISTERED_RATES}},
             store_module.StatusSemanticsError),
            ("an ESTIMABLE aggregate naming a site outside the cohort", macro,
             {"worst_site": "S999"}, store_module.StatusSemanticsError),
            ("an RQ3 unit both ESTIMABLE and failing", units,
             {"failing_condition": "QUADRATURE_DISAGREEMENT"},
             store_module.StatusSemanticsError),
            ("an RQ3 failure with no condition", units,
             {"status": proto.NOT_ESTIMABLE}, store_module.StatusSemanticsError),
            ("an RQ3 condition belonging to another status", units,
             {"status": proto.NOT_ESTIMABLE,
              "failing_condition": proto.STATUS_OK},
             store_module.StatusSemanticsError),
            ("an RQ3 unit missing a parent-required output", units,
             {"numerical_validation": None}, store_module.StatusSemanticsError),
            # the closed-domain rule: `selected_pca` is a categorical model
            # state, so it is validated by membership rather than by numeric
            # type — and membership is narrower than numeric type, not weaker
            ("a stringified PCA dimension", reg.SVM_PER_SITE_TABLE,
             {"selected_pca": "50"}, store_module.ClosedDomainError),
            ("a PCA dimension off the frozen grid", reg.SVM_PER_SITE_TABLE,
             {"selected_pca": 0}, store_module.ClosedDomainError),
            ("a PCA dimension given as a bool", reg.SVM_PER_SITE_TABLE,
             {"selected_pca": True}, store_module.ClosedDomainError),
            ("an ESTIMABLE fold that names no selected model",
             reg.SVM_PER_SITE_TABLE, {"selected_pca": None},
             store_module.ClosedDomainError),
        )
        for label, relpath, patch, expected in semantic_probes:
            payload = [dict(r) for r in _valid_rows(relpath)]
            payload[0].update(patch)
            try:
                probe.write_table(relpath, payload)
            except expected:
                continue
            except Exception as error:                     # pragma: no cover
                fail(f"{label} raised {type(error).__name__}, "
                     f"not {expected.__name__}")
                break
            fail(f"{label} was accepted at write time")
            break
        else:
            ok(f"scientific value semantics fail closed at write time "
               f"({len(semantic_probes)} probes: nonnumeric values, CI "
               "endpoints, the validity floor, worst-site fields, the "
               "parent's RQ3 status/failing-condition pair and the "
               "`selected_pca` closed domain)")

    if reg.NUMERIC_STRINGS_COERCED:
        fail("numeric scientific strings are declared coercible")
    else:
        ok("a scientific numeric field is never coerced from a string")

    # `selected_pca` is a categorical selection state, and the registry must say
    # so exactly once: a closed domain that is *also* a numeric field would
    # reinstate the type contradiction this rule was added to remove.
    if reg.CLOSED_DOMAIN_STRINGS_COERCED:
        fail("closed-domain values are declared coercible")
    elif reg.SVM_SELECTED_PCA_FIELD in reg.numeric_fields(reg.SVM_PER_SITE_TABLE):
        fail(f"{reg.SVM_SELECTED_PCA_FIELD} is registered as a scientific "
             "numeric field, but 'no-PCA' is one of its frozen values")
    elif reg.closed_domain(reg.SVM_PER_SITE_TABLE, reg.SVM_SELECTED_PCA_FIELD) \
            != tuple("no-PCA" if p is None else int(p)
                     for p in reg.SVM_PCA_CANDIDATES):
        fail("the selected_pca domain is not the frozen §F PCA grid")
    else:
        ok("selected_pca is a closed categorical domain "
           f"{reg.closed_domain(reg.SVM_PER_SITE_TABLE, reg.SVM_SELECTED_PCA_FIELD)} "
           "derived from the frozen PCA grid, never a numeric field and never "
           "coerced")

    # K — completion depends on the write-time barrier having run
    barrier_source = inspect.getsource(store_module.FormalStageFStore.complete_run)
    for needle in ("_verify_formal_authority", "_validated",
                   "read_checksum_ledger", "expected_checksum_paths"):
        if needle not in barrier_source:
            fail(f"complete_run does not invoke {needle}")
            return
    ok("complete_run re-proves authority, requires the write-time barrier, "
       "rereads the ledger and compares all four artifact books before the "
       "marker")


# ---------------------------------------------------------------------------
# 6d. scientific value semantics — the JSON headline, intervals, NOT_ESTIMABLE
#     and the parent's structured numerical-validation verdict
# ---------------------------------------------------------------------------
class _Arbitrary:
    """An object whose `str()` looks like a number — the audit's own probe."""

    def __str__(self):                          # pragma: no cover - never used
        return "0.87"


def check_scientific_semantics() -> None:
    """Every value the audit persisted through F.2b, refused at write time.

    Driven against a temporary smoke root: each probe below must **raise**, and
    each positive control must be accepted, so the group proves behaviour rather
    than reading the registry back to itself.
    """
    import tempfile

    intervals = reg.INTERVAL_TABLE
    units = reg.RQ3_STATUS_TABLES[0]
    macro = "rq4/site_macro_metrics.csv"
    per_repeat = "rq4/per_repeat_target_local.csv"
    folds = "svm/fold_selection.csv"

    def interval(**overrides):
        rows = [dict(r) for r in _valid_rows(intervals)]
        rows[0].update(overrides)
        return rows

    def unit(**overrides):
        rows = [dict(r) for r in _valid_rows(units)]
        rows[0].update(overrides)
        return rows

    def macro_row(**overrides):
        rows = [dict(r) for r in _valid_rows(macro)]
        rows[0].update(overrides)
        return rows

    def json_payload(**overrides):
        return {**_valid_json(reg.SVM_MACRO_JSON), **overrides}

    with tempfile.TemporaryDirectory() as tmp:
        store = store_module.SmokeStageFStore.create(Path(tmp) / "semantics")
        made = [0]

        def positive():
            """A fresh store for each positive control.

            A negative probe never reaches disk, so they can share one store; an
            accepted write consumes its registered path for that run, so each
            positive control gets its own.
            """
            made[0] += 1
            return store_module.SmokeStageFStore.create(
                Path(tmp) / f"accepted_{made[0]}")

        # C — the SVM headline number: schema, genuine numbers, status coherence
        json_probes = [
            ("a stringified number", json_payload(site_macro_bacc="0.5")),
            ("the string 'nan'", json_payload(site_macro_bacc="nan")),
            ("the string 'inf'", json_payload(site_macro_bacc="inf")),
            ("the string 'abc'", json_payload(site_macro_bacc="abc")),
            ("an arbitrary object", json_payload(site_macro_bacc=_Arbitrary())),
            ("NaN", json_payload(site_macro_bacc=float("nan"))),
            ("+inf", json_payload(site_macro_bacc=float("inf"))),
            ("-inf", json_payload(site_macro_bacc=float("-inf"))),
            ("a bool", json_payload(site_macro_bacc=True)),
            ("a missing headline number", json_payload(site_macro_bacc=None)),
            ("an ESTIMABLE payload with a reason",
             json_payload(reason_code="SVM_FOLD_FAILED")),
            ("a NOT_ESTIMABLE payload that kept its value",
             json_payload(status=reg.NOT_ESTIMABLE,
                          reason_code="SVM_FOLD_FAILED")),
            ("a NOT_ESTIMABLE payload with no reason",
             json_payload(status=reg.NOT_ESTIMABLE, site_macro_bacc=None,
                          reason_code=None)),
            ("a fractional site count",
             json_payload(n_sites_contributing=22.5,
                          n_sites_not_estimable=0.5)),
            ("a site count that is not the frozen cohort",
             json_payload(n_sites_contributing=22)),
            ("a declared confidence interval",
             json_payload(carries_confidence_interval=True)),
            ("a substituted estimand", json_payload(estimand="something_else")),
        ]
        for label, payload in json_probes:
            try:
                store.write_json(reg.SVM_MACRO_JSON, payload)
            except store_module.StageFStoreError:
                continue
            fail(f"{reg.SVM_MACRO_JSON} accepted {label}")
            return
        ok(f"{reg.SVM_MACRO_JSON} enforces its exact schema, genuine finite "
           f"numeric science and coherent status semantics "
           f"({len(json_probes)} probes)")

        try:
            store_module.canonical_json({"value": _Arbitrary()})
            fail("the serialiser stringified an arbitrary object")
            return
        except store_module.JsonSerialisationError:
            pass
        if reg.JSON_SERIALISATION_COERCES_OBJECTS:
            fail("the JSON serialiser is declared to coerce objects")
            return
        ok("the JSON serialiser has no default=str path: an arbitrary object "
           "stops the write instead of becoming acceptable scientific text")

        # C2 — non-finite values in fields that require finiteness
        nonfinite = []
        for value in (float("inf"), float("-inf")):
            rows = [dict(r) for r in _valid_rows(folds)]
            rows[0]["inner_macro_bacc"] = value
            nonfinite.append((f"inner_macro_bacc={value}", folds, rows))
        # `nan` in a CSV column is indistinguishable from an empty cell, so the
        # registry treats it as the missing representation; it is refused where
        # a scientific *value* is required, which is what this probe asserts.
        rows = [dict(r) for r in _valid_rows(macro)]
        rows[0]["site_macro_coverage"] = float("nan")
        nonfinite.append(("site_macro_coverage=nan", macro, rows))
        rows = [dict(r) for r in _valid_rows(per_repeat)]
        rows[0]["n_eval"] = float("inf")
        nonfinite.append(("n_eval=+inf", per_repeat, rows))
        rows = [dict(r) for r in _valid_rows(macro)]
        rows[0]["site_macro_coverage"] = float("inf")
        nonfinite.append(("site_macro_coverage=+inf", macro, rows))
        for label, relpath, probe in nonfinite:
            try:
                store.write_table(relpath, probe)
            except store_module.StageFStoreError:
                continue
            fail(f"{relpath} accepted {label}")
            return
        ok(f"a non-finite value never reaches a field the registry requires to "
           f"be finite ({len(nonfinite)} probes)")

        permitted = {field for fields in reg.NONFINITE_PERMITTED_FIELDS.values()
                     for field in fields}
        if reg.NONFINITE_PERMITTED_ANYWHERE or reg.NAN_PERMITTED_ANYWHERE \
                or permitted != {"tau", "selected_tau"}:
            fail(f"the non-finite permission is not the registered threshold "
                 f"exception: {sorted(permitted)}")
            return
        rows = [dict(r) for r in _valid_rows("rq4/per_site_metrics.csv")]
        for row in rows:
            row["tau"] = float("inf")
        try:
            positive().write_table("rq4/per_site_metrics.csv", rows)
        except store_module.StageFStoreError as error:
            fail(f"the registered all-inclusive threshold was refused: {error}")
            return
        ok("the frozen all-inclusive threshold tau = +inf stays legal exactly "
           "where it is registered, and nowhere else")

        # D — the interval contract
        interval_probes = [
            ("a missing total replicate count",
             interval(total_bootstrap_replicates=None)),
            ("a fractional total replicate count",
             interval(total_bootstrap_replicates=4999.5)),
            ("a total that is not the frozen B",
             interval(total_bootstrap_replicates=4999)),
            ("a boolean total", interval(total_bootstrap_replicates=True)),
            ("a stringified total", interval(total_bootstrap_replicates="5000")),
            ("a missing valid replicate count",
             interval(valid_bootstrap_replicates=None)),
            ("a fractional valid replicate count",
             interval(valid_bootstrap_replicates=4750.25)),
            ("more valid replicates than total",
             interval(valid_bootstrap_replicates=5001)),
            ("a valid interval below the floor",
             interval(valid_bootstrap_replicates=reg.MIN_VALID_REPLICATES - 1)),
            ("the validity floor claimed at or above the floor",
             interval(reason_code=reg.CI_VALIDITY_FLOOR_REASON, ci_low=None,
                      ci_high=None,
                      valid_bootstrap_replicates=reg.MIN_VALID_REPLICATES)),
            ("the validity floor converting the row to NOT_ESTIMABLE",
             interval(status=reg.NOT_ESTIMABLE,
                      reason_code=reg.CI_VALIDITY_FLOOR_REASON,
                      point_estimate=None, ci_low=None, ci_high=None,
                      valid_bootstrap_replicates=10)),
            ("an interval with one endpoint", interval(ci_high=None)),
            ("an unordered interval", interval(ci_low=0.7, ci_high=0.3)),
            ("a non-finite endpoint", interval(ci_high=float("inf"))),
        ]
        for label, probe in interval_probes:
            try:
                store.write_table(intervals, probe)
            except store_module.StageFStoreError:
                continue
            fail(f"{intervals} accepted {label}")
            return
        ok(f"the §D.12 interval contract fails closed ({len(interval_probes)} "
           "probes: replicate counts, the validity floor and endpoint order)")

        try:
            positive().write_table(intervals, interval(
                reason_code=reg.CI_VALIDITY_FLOOR_REASON, ci_low=None,
                ci_high=None,
                valid_bootstrap_replicates=reg.MIN_VALID_REPLICATES - 1))
        except store_module.StageFStoreError as error:
            fail(f"the validity floor was refused on a legitimate row: {error}")
            return
        ok("CI_INSUFFICIENT_VALID_REPLICATES invalidates the interval alone: "
           "the point estimate survives and the row stays ESTIMABLE")

        # E — artifact-specific NOT_ESTIMABLE semantics
        retained = []
        for relpath in sorted(reg.STATUS_SEMANTICS):
            semantics = reg.STATUS_SEMANTICS[relpath]
            blanked = (tuple(semantics["value_fields"])
                       + tuple(semantics.get("interval_fields", ()))
                       + tuple(reg.NOT_ESTIMABLE_UNAVAILABLE_EXTRA.get(relpath,
                                                                       ())))
            for field in blanked:
                rows = [dict(r) for r in _valid_rows(relpath)]
                rows[0].update(_not_estimable(relpath))
                for name in blanked:
                    rows[0][name] = None
                rows[0][field] = "S20" if field == "worst_site" else 0.5
                retained.append((relpath, field, rows))
        for relpath, field, probe in retained:
            try:
                store.write_table(relpath, probe)
            except store_module.StatusSemanticsError:
                continue
            fail(f"a NOT_ESTIMABLE {relpath} row retained {field}")
            return
        ok(f"every registered status-carrying table surrenders every field it "
           f"declares unavailable when NOT_ESTIMABLE ({len(retained)} probes "
           f"across {len(reg.STATUS_SEMANTICS)} artifacts)")

        try:
            positive().write_table(macro, macro_row(**_not_estimable(macro)))
        except store_module.StageFStoreError as error:
            fail(f"a coherent NOT_ESTIMABLE aggregate was refused: {error}")
            return
        ok("a NOT_ESTIMABLE aggregate keeps its design coordinates and counts: "
           "a zero count is a fact, not a missing value")

        # F — the parent's structured numerical-validation verdict
        verdict_probes = [
            ("an ESTIMABLE unit with a failed verdict",
             unit(numerical_validation=_verdict(n_passed=6))),
            ("an ESTIMABLE unit with a truncated verdict",
             unit(numerical_validation={"passed": True})),
            ("an ESTIMABLE unit with a scalar verdict",
             unit(numerical_validation="PASSED")),
            ("a verdict declaring the wrong required count",
             unit(numerical_validation={**_verdict(), "n_required": 5})),
            ("a verdict claiming a pass below seven of seven",
             unit(numerical_validation=_verdict(n_passed=6, passed=True))),
            ("a NOT_ESTIMABLE unit with a passing verdict",
             unit(status=proto.NOT_ESTIMABLE,
                  failing_condition="QUADRATURE_DISAGREEMENT",
                  numerical_validation=_verdict())),
            ("VALIDATION_MALFORMED beside a complete verdict",
             unit(status=proto.NOT_ESTIMABLE,
                  failing_condition="VALIDATION_MALFORMED",
                  numerical_validation=_verdict(n_passed=6))),
            ("QUADRATURE_DISAGREEMENT beside an incomplete verdict",
             unit(status=proto.NOT_ESTIMABLE,
                  failing_condition="QUADRATURE_DISAGREEMENT",
                  numerical_validation=_verdict(n_evaluated=3, n_passed=3,
                                                passed=False))),
            ("a criterion record that is not the parent's",
             unit(numerical_validation={
                 **_verdict(),
                 "criteria": {"loglik": {"discrepancy": "0.0",
                                         "tolerance": 1e-6, "passed": True}}})),
        ]
        for label, probe in verdict_probes:
            try:
                store.write_table(units, probe)
            except store_module.StatusSemanticsError:
                continue
            fail(f"{units} accepted {label}")
            return
        ok(f"the parent's structured 40-vs-60 verdict is validated in full, "
           f"and is one decision with the unit's status ({len(verdict_probes)} "
           "probes)")

        if int(proto.N_REQUIRED_AGREEMENT_CRITERIA) != 7 \
                or len(proto.REQUIRED_AGREEMENT_CRITERIA) != 7:
            fail("the parent's agreement registry is not seven of seven")
            return
        source = inspect.getsource(store_module._validate_numerical_verdict)
        if "proto.N_REQUIRED_AGREEMENT_CRITERIA" not in source:
            fail("the verdict check does not bind to the parent's own count")
            return
        ok("the verdict semantics bind to rq3_protocol's own tuples, not to a "
           "list copied into the Stage-F layer")


# ---------------------------------------------------------------------------
# 6e. artifact-specific reason semantics and parent-bound criterion identity
# ---------------------------------------------------------------------------
def check_reason_scoping() -> None:
    """A reason code is legal in one artifact, in one state, or nowhere.

    Every probe here is behavioural: a real row through the real write barrier.
    The previous round passed a name-based check while accepting a forged
    criterion registry, a borrowed tolerance and a reason code lifted into an
    artifact whose module never runs, so nothing below is satisfied by reading
    the registry back to itself.
    """
    import tempfile

    units = reg.RQ3_STATUS_TABLES[0]
    tables = [p for p in reg.ARTIFACT_REASON_CODES if p != reg.SVM_MACRO_JSON]

    def unit(**overrides):
        rows = [dict(r) for r in _valid_rows(units)]
        rows[0].update(overrides)
        return rows

    def refuses(store, label, relpath, rows, expected=store_module.StageFStoreError):
        try:
            if isinstance(rows, dict):
                store.write_json(relpath, rows)
            else:
                store.write_table(relpath, rows)
        except (expected, reg.StageFRegistryError):
            return True
        except Exception as error:                          # pragma: no cover
            fail(f"{label} raised {type(error).__name__} instead of refusing")
            return False
        fail(f"{relpath} accepted {label}")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        store = store_module.SmokeStageFStore.create(Path(tmp) / "scoping")
        made = [0]

        def positive():
            made[0] += 1
            return store_module.SmokeStageFStore.create(
                Path(tmp) / f"accepted_{made[0]}")

        # A — cross-artifact misuse, every registered code against every
        #     status-carrying artifact that does not register it
        misuse = 0
        for relpath in tables:
            legal = set(reg.artifact_reason_codes(relpath, reg.NOT_ESTIMABLE))
            foreign = [c for c in reg.REASON_CODES + reg.REASON_CODES_REUSED_FROM_RQ3
                       if c not in legal]
            for code in foreign:
                rows = [dict(r) for r in _valid_rows(relpath)]
                rows[0].update({**_not_estimable(relpath), "reason_code": code})
                if not refuses(store, f"the foreign reason {code}", relpath, rows):
                    return
                misuse += 1
        for code in ("SVM_NOT_CONVERGED", "NO_EVALUATION_POINTS",
                     "NO_ESTIMABLE_SITE_IN_CELL",
                     "CI_INSUFFICIENT_VALID_REPLICATES", "NO_ELIGIBLE_START"):
            payload = {**_valid_json(reg.SVM_MACRO_JSON),
                       "status": reg.NOT_ESTIMABLE, "reason_code": code,
                       "site_macro_bacc": None, "n_sites_contributing": 22,
                       "n_sites_not_estimable": 1}
            if not refuses(store, f"the foreign reason {code}",
                           reg.SVM_MACRO_JSON, payload):
                return
            misuse += 1
        ok(f"a reason code is refused in every artifact its own module never "
           f"produces ({misuse} probes across {len(tables) + 1} artifacts)")

        # B — and each artifact's own vocabulary still works
        accepted = 0
        for relpath in tables:
            for code in reg.artifact_reason_codes(relpath, reg.NOT_ESTIMABLE):
                rows = [dict(r) for r in _valid_rows(relpath)]
                rows[0].update(_not_estimable(relpath, code))
                try:
                    positive().write_table(relpath, rows)
                except store_module.StageFStoreError as error:
                    fail(f"{relpath} refused its own registered {code}: {error}")
                    return
                accepted += 1
        try:
            positive().write_json(reg.SVM_MACRO_JSON, {
                **_valid_json(reg.SVM_MACRO_JSON), "status": reg.NOT_ESTIMABLE,
                "reason_code": "SVM_FOLD_FAILED", "site_macro_bacc": None,
                "n_sites_contributing": 22, "n_sites_not_estimable": 1})
        except store_module.StageFStoreError as error:
            fail(f"the SVM headline JSON refused SVM_FOLD_FAILED: {error}")
            return
        ok(f"every artifact still accepts the reasons §G.2 does give it "
           f"({accepted + 1} positive controls)")

        # C — the field state each reason asserts
        state_probes = [
            ("NO_EVALUATION_POINTS beside 12 evaluated cases",
             "rq4/per_site_metrics.csv",
             {**_not_estimable("rq4/per_site_metrics.csv",
                               "NO_EVALUATION_POINTS"), "n_eval": 12}),
            ("NO_EVALUATION_POINTS beside a coverage",
             "rq4/per_site_metrics.csv",
             {**_not_estimable("rq4/per_site_metrics.csv",
                               "NO_EVALUATION_POINTS"), "coverage": 0.93}),
            ("an ESTIMABLE row computed from an empty evaluation set",
             "rq4/per_site_metrics.csv", {"n_eval": 0}),
            ("NO_ESTIMABLE_SITE_IN_CELL beside three estimable sites",
             "rq4/site_macro_metrics.csv",
             {**_not_estimable("rq4/site_macro_metrics.csv"),
              "n_estimable_sites": 3}),
            ("NO_ESTIMABLE_SITE_IN_CELL beside acquisitions",
             "rq4/marginal_metrics.csv",
             {**_not_estimable("rq4/marginal_metrics.csv"),
              "n_acquisitions": 40}),
            ("NO_ESTIMABLE_SITE_IN_CELL beside valid replicates",
             reg.INTERVAL_TABLE,
             {**_not_estimable(reg.INTERVAL_TABLE),
              "valid_bootstrap_replicates": 5000}),
            ("an ESTIMABLE aggregate over no estimable site",
             "rq4/site_macro_metrics.csv", {"n_estimable_sites": 0}),
            ("SVM_NOT_CONVERGED beside converged=True", "svm/per_site_bacc.csv",
             {**_not_estimable("svm/per_site_bacc.csv"), "converged": True}),
            ("SVM_NOT_CONVERGED beside a balanced accuracy",
             "svm/per_site_bacc.csv",
             {**_not_estimable("svm/per_site_bacc.csv"), "bacc": 0.71}),
            ("an ESTIMABLE fold that never converged", "svm/per_site_bacc.csv",
             {"converged": False}),
            ("a convergence flag that is a string", "svm/per_site_bacc.csv",
             {"converged": "False"}),
            ("a convergence flag that is a 0/1", "svm/per_site_bacc.csv",
             {"converged": 1}),
        ]
        sens6 = ("sensitivities/sens6_realized_coverage_matched/"
                 "threshold_selection.csv")
        state_probes += [
            ("NO_EVALUATION_POINTS beside 40 confidence rows", sens6,
             {**_not_estimable(sens6, "NO_EVALUATION_POINTS"), "n_rows": 40}),
            ("COVERAGE_MATCH_UNATTAINABLE over an empty candidate set", sens6,
             {**_not_estimable(sens6, "COVERAGE_MATCH_UNATTAINABLE"),
              "n_rows": 0, "n_distinct_conf": 0}),
        ]
        for label, relpath, overrides in state_probes:
            rows = [dict(r) for r in _valid_rows(relpath)]
            rows[0].update(overrides)
            if not refuses(store, label, relpath, rows):
                return
        ok(f"a reason's own field state is enforced, in both directions "
           f"({len(state_probes)} probes: evaluation counts, cell counts, "
           "replicate counts and the SVM convergence flag)")

        # D — criterion identity and arithmetic, bound to the parent
        registry = tuple(proto.REQUIRED_AGREEMENT_CRITERIA)
        forged = _verdict()
        forged["criteria"]["fake_criterion"] = \
            forged["criteria"].pop(registry[0])
        missing = _verdict()
        missing["criteria"].pop(registry[-1])
        extra = _verdict()
        extra["criteria"]["an_eighth"] = {"discrepancy": 0.0,
                                          "tolerance": 1e-4, "passed": True}
        duplicate = _verdict()
        duplicate["criteria"].pop(registry[-1])
        duplicate["criteria"][f"{registry[0]} "] = \
            dict(duplicate["criteria"][registry[0]])
        invented = _verdict()
        invented["criteria"][registry[0]]["tolerance"] = 0.1
        borrowed = _verdict()
        borrowed["criteria"]["loglik"]["tolerance"] = \
            float(proto.AGREEMENT_TOL["cv_logloss"])
        breached = _verdict()
        breached["criteria"][registry[0]]["discrepancy"] = \
            float(proto.AGREEMENT_TOL[registry[0]]) * 1000
        understated = _verdict(n_passed=6)
        understated["criteria"][registry[-1]]["discrepancy"] = 0.0
        negative = _verdict()
        negative["criteria"][registry[0]]["discrepancy"] = -1e-9
        criterion_probes = [
            ("a forged criterion registry", unit(numerical_validation=forged)),
            ("a criterion registry of its own",
             unit(numerical_validation={**_verdict(), "criteria": {
                 f"criterion_{i}": dict(record) for i, record
                 in enumerate(_verdict()["criteria"].values())}})),
            ("a missing criterion", unit(numerical_validation=missing)),
            ("an extra criterion", unit(numerical_validation=extra)),
            ("a duplicated criterion", unit(numerical_validation=duplicate)),
            ("an invented tolerance", unit(numerical_validation=invented)),
            ("a tolerance borrowed from another criterion",
             unit(numerical_validation=borrowed)),
            ("a breached criterion reported as passed",
             unit(numerical_validation=breached)),
            ("a met criterion reported as failed",
             unit(status=proto.NOT_ESTIMABLE,
                  failing_condition="QUADRATURE_DISAGREEMENT",
                  numerical_validation=understated)),
            ("a negative discrepancy", unit(numerical_validation=negative)),
            ("a criteria collection that is not a mapping",
             unit(numerical_validation={**_verdict(),
                                        "criteria": list(registry)})),
        ]
        for label, probe in criterion_probes:
            if not refuses(store, label, units, probe,
                           store_module.StatusSemanticsError):
                return
        ok(f"the parent's seven-criterion registry, its frozen per-criterion "
           f"tolerances and `passed == (discrepancy <= tolerance)` are all "
           f"enforced ({len(criterion_probes)} probes)")

        # E — which parent codes carry a verdict at all
        shape_probes = [
            ("NO_ELIGIBLE_START beside a constructed verdict",
             unit(status=proto.NOT_ESTIMABLE,
                  failing_condition="NO_ELIGIBLE_START",
                  numerical_validation=_verdict(n_passed=6))),
            ("NO_ELIGIBLE_START beside a passing verdict",
             unit(status=proto.NOT_ESTIMABLE,
                  failing_condition="NO_ELIGIBLE_START",
                  numerical_validation=_verdict())),
            ("CV_FOLD_NOT_ESTIMABLE beside a verdict",
             unit(status=proto.NOT_ESTIMABLE,
                  failing_condition="CV_FOLD_NOT_ESTIMABLE",
                  numerical_validation=_verdict(n_evaluated=0, n_passed=0,
                                                passed=False))),
            ("QUADRATURE_DISAGREEMENT beside the empty record",
             unit(status=proto.NOT_ESTIMABLE,
                  failing_condition="QUADRATURE_DISAGREEMENT",
                  numerical_validation={})),
            ("an ESTIMABLE unit with the empty record",
             unit(status=proto.STATUS_OK, failing_condition=None,
                  numerical_validation={})),
        ]
        for label, probe in shape_probes:
            if not refuses(store, label, units, probe,
                           store_module.StatusSemanticsError):
                return
        try:
            positive().write_table(units, unit(
                status=proto.NOT_ESTIMABLE, failing_condition="NO_ELIGIBLE_START",
                numerical_validation=store_module._canonical_empty_verdict(
                    "NO_ELIGIBLE_START")))
        except store_module.StageFStoreError as error:
            fail(f"the parent's own start-failure record was refused: {error}")
            return
        ok(f"only the two parent codes that reach 40-vs-60 carry a verdict; a "
           f"start failure carries the empty form its own producer writes — "
           f"`BlockResult`'s `{{}}` inside cells.csv, "
           f"`_persist_cell_summary`'s canonical record in "
           f"validation_units.csv ({len(shape_probes)} probes plus its "
           f"positive control)")

        # F — §13: the cell variance and the code that withdrew it
        cells = reg.RQ3_CELL_TABLES[0]

        def cell_row(**overrides):
            rows = [dict(r) for r in _valid_rows(cells)]
            rows[0].update(overrides)
            return rows

        cell_probes = [("a withdrawn variance with no condition",
                        cell_row(sigma_site_sq=reg.NOT_ESTIMABLE,
                                 sigma_site_sq_condition=None)),
                       ("a reported variance beside a condition",
                        cell_row(sigma_site_sq_condition="NO_ELIGIBLE_START")),
                       ("a stringified variance", cell_row(sigma_site_sq="0.25")),
                       ("a non-finite variance",
                        cell_row(sigma_site_sq=float("inf"))),
                       ("a negative variance", cell_row(sigma_site_sq=-0.1))]
        first = (reg.SENS56_POPULATIONS[0],) + tuple(proto.CELL_REGISTRY[0])
        withdrawn_blocks = dict(_v_blocks_by_cell()[first])
        _failed = list(withdrawn_blocks)[0]
        withdrawn_blocks[_failed] = _v_block(first[0], _failed, first[1],
                                             first[2],
                                             status=proto.NOT_ESTIMABLE,
                                             condition="NO_ELIGIBLE_START")
        cell_probes += [(f"the Stage-F reason {code} withdrawing a variance",
                         cell_row(blocks=withdrawn_blocks,
                                  sigma_site_sq=reg.NOT_ESTIMABLE,
                                  sigma_site_sq_condition=code))
                        for code in reg.REASON_CODES]
        for label, probe in cell_probes:
            if not refuses(store, label, cells, probe):
                return
        try:
            first = (reg.SENS56_POPULATIONS[0],) + tuple(proto.CELL_REGISTRY[0])
            withdrawn = dict(_v_blocks_by_cell()[first])
            block = list(withdrawn)[0]
            withdrawn[block] = _v_block(first[0], block, first[1], first[2],
                                        status=proto.NOT_ESTIMABLE,
                                        condition="NO_ELIGIBLE_START")
            positive().write_table(cells, cell_row(
                blocks=withdrawn, sigma_site_sq=reg.NOT_ESTIMABLE,
                sigma_site_sq_condition="NO_ELIGIBLE_START"))
        except store_module.StageFStoreError as error:
            fail(f"a legitimate §13 withdrawal was refused: {error}")
            return
        ok(f"the SENS-5/6 cell variance and the §9.5.1 code that withdrew it are "
           f"one state, governed by the parent's registry ({len(cell_probes)} "
           "probes plus its positive control)")

    # G — the binding itself: no Stage-F copy of the parent's definitions
    source = inspect.getsource(store_module._validate_numerical_verdict)
    for needle in ("proto.REQUIRED_AGREEMENT_CRITERIA", "proto.AGREEMENT_TOL",
                   "parent.CODE_QUADRATURE", "parent.CODE_VALIDATION_MALFORMED"):
        if needle not in source:
            fail(f"the verdict check does not bind to {needle}")
            return
    for name in registry:
        if f'"{name}"' in source or f"'{name}'" in source:
            fail(f"the verdict check names the criterion {name} itself")
            return
    for tolerance in set(proto.AGREEMENT_TOL.values()):
        if repr(tolerance) in source:
            fail(f"the verdict check restates the tolerance {tolerance}")
            return
    ok("criterion identity and tolerance are read from rq3_protocol at "
       "validation time; the Stage-F layer holds no copy to drift")


# ---------------------------------------------------------------------------
# 7. tests
# ---------------------------------------------------------------------------
def check_structural_semantics() -> None:
    """F.2c-3 — which cells may exist, per-fold SVM failure, nested blocks,
    inherited RQ2/S2 state, and the agreements between the files that repeat
    them. Every probe writes a real table through the real barrier.
    """
    import tempfile

    import certification as cert

    per_site = "rq4/per_site_metrics.csv"
    per_repeat = "rq4/per_repeat_target_local.csv"
    cells = reg.RQ3_CELL_TABLES[0]
    units = reg.RQ3_STATUS_TABLES[0]
    fams = f"{reg.NAMESPACE_SENSITIVITIES}/{reg.SENS56_APPLIES_TO[0]}/" \
           f"{reg.SENS56_FAMILIES_FILE}"
    decisions = reg.SENS_DECISION_TABLES[0]
    evaluations = reg.SENS_EVALUATION_TABLES[0]
    first_cell = (reg.SENS56_POPULATIONS[0],) + tuple(proto.CELL_REGISTRY[0])

    def edited(relpath, index=0, **overrides):
        rows = [dict(r) for r in _valid_rows(relpath)]
        rows[index].update(overrides)
        return rows

    def failed_blocks(condition="NO_ELIGIBLE_START", which=0):
        blocks = dict(_v_blocks_by_cell()[first_cell])
        name = list(blocks)[which]
        blocks[name] = _v_block(first_cell[0], name, first_cell[1],
                                first_cell[2], status=proto.NOT_ESTIMABLE,
                                condition=condition)
        return blocks

    with tempfile.TemporaryDirectory() as tmp:
        store = store_module.SmokeStageFStore.create(Path(tmp) / "structural")
        made = [0]

        def fresh():
            made[0] += 1
            return store_module.SmokeStageFStore.create(
                Path(tmp) / f"fresh_{made[0]}")

        def refuses(label, relpath, rows, target=None):
            target = target or store
            try:
                if isinstance(rows, dict):
                    target.write_json(relpath, rows)
                else:
                    target.write_table(relpath, rows)
            except (store_module.StageFStoreError, reg.StageFRegistryError):
                return True
            fail(f"{relpath} accepted {label}")
            return False

        def accepts(label, relpath, rows, target=None):
            target = target or fresh()
            try:
                if isinstance(rows, dict):
                    target.write_json(relpath, rows)
                else:
                    target.write_table(relpath, rows)
            except Exception as error:
                fail(f"{relpath} refused {label}: {error}")
                return False
            return True

        # A — SVM_FOLD_FAILED is a per-fold state (§G.2, §H.3)
        svm = reg.SVM_PER_SITE_TABLE
        if not accepts("a fold that raised before predictions", svm,
                       edited(svm, **_not_estimable(svm, "SVM_FOLD_FAILED"))):
            return
        if not accepts("a successful fold", svm, _valid_rows(svm)):
            return
        probes = 2
        for label, update in (
                ("a failed fold reporting a balanced accuracy",
                 {**_not_estimable(svm, "SVM_FOLD_FAILED"), "bacc": 0.63}),
                ("a failed fold claiming a converged fit",
                 {**_not_estimable(svm, "SVM_FOLD_FAILED"), "converged": True})):
            if not refuses(label, svm, edited(svm, **update)):
                return
            probes += 1
        for relpath in (per_site, per_repeat, "rq4/site_macro_metrics.csv",
                        "rq4/marginal_metrics.csv", reg.INTERVAL_TABLE):
            rows = edited(relpath, **{**_not_estimable(relpath),
                                      "reason_code": "SVM_FOLD_FAILED"})
            if not refuses("SVM_FOLD_FAILED outside F-SVM", relpath, rows):
                return
            probes += 1
        if not refuses("SVM_FOLD_FAILED as a parent §9.5.1 code", units,
                       edited(units, status=proto.NOT_ESTIMABLE,
                              failing_condition="SVM_FOLD_FAILED",
                              numerical_validation={})):
            return
        probes += 1
        ok(f"a fold that raised before predictions is persistable on the fold "
           f"table and nowhere else ({probes} probes)")

        # B — an ineligible cell is absent, never a row (§G.2)
        import contracts

        probes = 0
        for relpath in (per_site, per_repeat):
            rows = edited(relpath, **{**_not_estimable(relpath),
                                      "reason_code": "SITE_NOT_ELIGIBLE_AT_N"})
            if not refuses("SITE_NOT_ELIGIBLE_AT_N as a row", relpath, rows):
                return
            probes += 1
        if reg.NON_PERSISTABLE_REASON_CODES != ("SITE_NOT_ELIGIBLE_AT_N",) \
                or "SITE_NOT_ELIGIBLE_AT_N" not in reg.REASON_CODES:
            fail("SITE_NOT_ELIGIBLE_AT_N left the frozen §G.2 vocabulary")
            return
        rows = [dict(r) for r in _valid_rows(per_site)]
        for row in rows:
            if row["n_labels"] == 40 \
                    and row["held_out_site"] == contracts.FROZEN_SITES[0]:
                row["held_out_site"] = contracts.FROZEN_SITES[reg.SITES_PER_N[20]]
        if not refuses("a site eligible at N=40 but not at N=20", per_site, rows):
            return
        probes += 1
        rows = [dict(r) for r in _valid_rows(per_site)]
        for row in rows:
            if row["n_labels"] == 20 and row["alpha"] == reg.ALPHA_GRID[1] \
                    and row["held_out_site"] == contracts.FROZEN_SITES[0]:
                row["held_out_site"] = contracts.FROZEN_SITES[reg.SITES_PER_N[20]]
        if not refuses("an eligible site set that moves with alpha", per_site,
                       rows):
            return
        probes += 1
        rows = [dict(r) for r in _valid_rows(per_site)]
        for row in rows:
            if row["alpha"] == reg.ALPHA_GRID[0]:
                row["alpha"] = 0.15
        if not refuses("an unregistered conformal cell", per_site, rows):
            return
        probes += 1
        for relpath in reg.EXACT_KEY_SET_TABLES:
            if not refuses("a substituted cell", relpath,
                           edited(relpath, alpha=0.15)):
                return
            probes += 1
        if not refuses("an outer fold that is not a frozen eligible site", svm,
                       edited(svm, held_out_site="S999")):
            return
        probes += 1
        if not refuses("an inner grid point off the frozen (pca, C) grid",
                       reg.SVM_FOLD_SELECTION_TABLE,
                       edited(reg.SVM_FOLD_SELECTION_TABLE, C=0.123)):
            return
        probes += 1
        if not accepts("the registered site x cell enumeration", per_site,
                       _valid_rows(per_site)):
            return
        ok(f"an ineligible site x N cell is refused rather than recorded, and "
           f"the enumerated cell tables must be the frozen registry "
           f"({probes} probes plus its positive control)")

        # C — the nested block records and their siblings (§H.4.3)
        probes = 0
        for label, blob in (("an opaque nested object", {"anything": "at all"}),
                            ("a nested block with no record",
                             {"motion": {"note": "opaque"}}),
                            ("a scalar in the blocks column", 7),
                            ("no blocks at all", None)):
            if not refuses(label, cells, edited(cells, blocks=blob)):
                return
            probes += 1
        base = dict(_v_blocks_by_cell()[first_cell])
        renamed = dict(base)
        renamed["clinical"] = renamed.pop("diagnosis")
        duplicated = dict(base, diagnosis=dict(base["motion"]))
        missing = {"motion": base["motion"]}
        extra = dict(base, clinical=dict(base["motion"], block="clinical"))
        for label, blocks in (("a wrong block name", renamed),
                              ("the same block twice", duplicated),
                              ("a missing block", missing),
                              ("an extra block", extra)):
            if not refuses(label, cells, edited(cells, blocks=blocks)):
                return
            probes += 1
        for field, value in (("cell_index", 5), ("df", 4),
                             ("validation_unit", ["population", "cell"]),
                             ("failure_is_numerical", True),
                             ("operating_point", "OP30"),
                             ("lrt_statistic", proto.NOT_ESTIMABLE)):
            blocks = dict(base, motion={**base["motion"], field: value})
            if not refuses(f"a nested block misdescribing its {field}", cells,
                           edited(cells, blocks=blocks)):
                return
            probes += 1
        verdict = _verdict()
        verdict["criteria"]["sigma_site_sq_relative"]["passed"] = False
        blocks = dict(base, motion={**base["motion"],
                                    "numerical_validation": verdict})
        if not refuses("a nested verdict that fails the parent's arithmetic",
                       cells, edited(cells, blocks=blocks)):
            return
        probes += 1
        blocks = failed_blocks()
        blocks["motion"] = {**blocks["motion"], "numerical_validation":
                            store_module._canonical_empty_verdict(
                                "NO_ELIGIBLE_START")}
        if not refuses("the units' canonical record inside a nested block",
                       cells, edited(cells, blocks=blocks,
                                     sigma_site_sq=reg.NOT_ESTIMABLE,
                                     sigma_site_sq_condition="NO_ELIGIBLE_START")):
            return
        probes += 1
        if not refuses("the blocks' bare default inside a validation unit",
                       units, edited(units, status=proto.NOT_ESTIMABLE,
                                     failing_condition="NO_ELIGIBLE_START",
                                     numerical_validation={})):
            return
        probes += 1
        # cells <-> validation_units
        for label, update in (
                ("a unit whose status contradicts its own block",
                 {"status": proto.NOT_ESTIMABLE,
                  "failing_condition": "NO_ELIGIBLE_START",
                  "numerical_validation":
                      store_module._canonical_empty_verdict("NO_ELIGIBLE_START")}),
                ("a unit with no cell of its own", {"block": "clinical"})):
            target = fresh()
            target.write_table(cells, _valid_rows(cells))
            if not refuses(label, units, edited(units, **update), target=target):
                return
            probes += 1
        target = fresh()
        target.write_table(cells, _valid_rows(cells))
        forged = _verdict()
        forged["criteria"]["sigma_site_sq_relative"]["discrepancy"] = 0.0
        if not refuses("a unit verdict that is not its block's", units,
                       edited(units, numerical_validation=forged), target=target):
            return
        probes += 1
        target = fresh()
        target.write_table(cells, _valid_rows(cells))
        if not accepts("a cells / validation_units pair that agrees", units,
                       _valid_rows(units), target=target):
            return
        # cells <-> families
        for label, mutate in (
                ("a family slot that is not its block's p-value",
                 lambda f: f[0]["slots"].__setitem__(
                     0, {**f[0]["slots"][0], "p_raw": 0.999})),
                ("a family slot out of the parent's cell order",
                 lambda f: f[0].__setitem__(
                     "slots", [f[0]["slots"][1], f[0]["slots"][0]]
                     + list(f[0]["slots"][2:]))),
                ("a family that miscounts its unestimable slots",
                 lambda f: f[0].__setitem__(
                     reg.SENS56_FAMILY_NOT_ESTIMABLE_FIELD, 2))):
            target = fresh()
            target.write_table(cells, _valid_rows(cells))
            rows = [dict(r) for r in _valid_rows(fams)]
            rows[0]["slots"] = [dict(x) for x in rows[0]["slots"]]
            mutate(rows)
            if not refuses(label, fams, rows, target=target):
                return
            probes += 1
        ok(f"the two block records inside a cell are structurally validated "
           f"and bound to validation_units.csv and families.csv "
           f"({probes} probes plus its positive control)")

        # D — the quadrature reason is parent-derived (§9.6.4)
        def breached(*names):
            verdict = _verdict()
            for name in names:
                entry = verdict["criteria"][name]
                entry["discrepancy"] = entry["tolerance"] * 10
                entry["passed"] = False
            verdict["n_passed"] = sum(1 for e in verdict["criteria"].values()
                                      if e["passed"])
            verdict["passed"] = False
            failed = [n for n in proto.REQUIRED_AGREEMENT_CRITERIA
                      if not verdict["criteria"][n]["passed"]]
            verdict["reason"] = f"criteria beyond tolerance: {failed}"
            return verdict

        if not accepts("a disagreement naming its own breached criterion",
                       units, edited(units, status=proto.NOT_ESTIMABLE,
                                     failing_condition="QUADRATURE_DISAGREEMENT",
                                     numerical_validation=breached(
                                         "sigma_site_sq_relative"))):
            return
        probes = 0
        passing = next(n for n in proto.REQUIRED_AGREEMENT_CRITERIA
                       if n != "sigma_site_sq_relative")
        for label, reason in (
                ("free prose", "the 60-node run disagreed"),
                ("the bare sentence", "criteria beyond tolerance"),
                ("an empty breach list", "criteria beyond tolerance: []"),
                ("an unregistered criterion",
                 "criteria beyond tolerance: ['loglik']"),
                ("a criterion this record reports as passing",
                 f"criteria beyond tolerance: {[passing]}")):
            verdict = breached("sigma_site_sq_relative")
            verdict["reason"] = reason
            if not refuses(f"a quadrature reason that is {label}", units,
                           edited(units, status=proto.NOT_ESTIMABLE,
                                  failing_condition="QUADRATURE_DISAGREEMENT",
                                  numerical_validation=verdict)):
                return
            probes += 1
        verdict = breached(*proto.REQUIRED_AGREEMENT_CRITERIA[:2])
        verdict["reason"] = ("criteria beyond tolerance: "
                             f"{list(reversed(proto.REQUIRED_AGREEMENT_CRITERIA[:2]))}")
        if not refuses("a breach list out of the parent's registry order", units,
                       edited(units, status=proto.NOT_ESTIMABLE,
                              failing_condition="QUADRATURE_DISAGREEMENT",
                              numerical_validation=verdict)):
            return
        probes += 1
        ok(f"a QUADRATURE_DISAGREEMENT names the criteria its own record "
           f"reports as breached, in the parent's order ({probes} probes plus "
           "its positive control)")

        # E — SENS-1..4 inherit RQ2/S2's state machine (§H.4, §G.2)
        probes = 0
        if not accepts("a legal inherited decision state", decisions,
                       _valid_rows(decisions)):
            return
        if not accepts("a legal inherited evaluation state", evaluations,
                       _valid_rows(evaluations)):
            return
        for label, update in (
                ("an invented state", {"state": "CERTIFIED"}),
                ("another tier's state", {"state": cert.SOURCE_REFERENCE_PASS}),
                ("an invented reason", {"reason": "coverage too low"}),
                ("a Stage-F reason", {"reason": "NO_EVALUATION_POINTS"}),
                ("a fail reason on a pass", {"reason": "NO_PASSING_CANDIDATE"}),
                ("an untestable row that still selected",
                 {"state": cert.NOT_TESTABLE,
                  "reason": "NO_ELIGIBLE_CANDIDATE__COVERAGE"}),
                ("a pass with no threshold", {"selected_tau": None}),
                ("a selected index off the parent's arithmetic",
                 {"n_passed_candidates": 2}),
                ("a substituted guarantee",
                 {"guarantee": "EXACT_CP_ON_ITS_OWN_CALIBRATION_SAMPLE"})):
            # located by strategy, never by position: the shrinkage tier is
            # the one whose guarantee is the empirical one
            shrinkage = next(
                i for i, row in enumerate(_valid_rows(decisions))
                if row["strategy"] == "pooled_local_shrinkage")
            if not refuses(label, decisions,
                           edited(decisions, index=shrinkage, **update)):
                return
            probes += 1
        for label, update in (
                ("a borrowed realized field",
                 {"realized_field": cert.REALIZED_FIELD["pooled_source"]}),
                ("an outcome that contradicts its own BER",
                 {"realized_outcome": 0}),
                ("a risk excess that is not ber - r_star",
                 {"risk_excess": 0.5}),
                ("science with no threshold behind it",
                 {"state": cert.NOT_TESTABLE, "selected_tau": None,
                  "selected_q": None, "selected_candidate_index": None}),
                ("a missing realized coverage", {"coverage_test": None}),
                ("a derived quantity outliving its BER", {"ber_test": None})):
            if not refuses(label, evaluations, edited(evaluations, **update)):
                return
            probes += 1
        ok(f"SENS-1..4 carry the parent's own states, reasons, tiers and "
           f"realized-outcome fields, and nothing else ({probes} probes plus "
           "two positive controls)")

        # F — sigma is bound to the block that withdrew it (§13)
        probes = 0
        for label, rows in (
                ("a reported variance beside a failed block",
                 edited(cells, blocks=failed_blocks(), sigma_site_sq=0.25,
                        sigma_site_sq_condition=None)),
                ("a withdrawn variance beside two passing blocks",
                 edited(cells, sigma_site_sq=reg.NOT_ESTIMABLE,
                        sigma_site_sq_condition="NO_ELIGIBLE_START")),
                ("a condition that is not the failing block's own code",
                 edited(cells, blocks=failed_blocks(),
                        sigma_site_sq=reg.NOT_ESTIMABLE,
                        sigma_site_sq_condition="CV_FOLD_NOT_ESTIMABLE"))):
            if not refuses(label, cells, rows):
                return
            probes += 1
        if not accepts("a variance withdrawn by a block that really failed",
                       cells, edited(cells, blocks=failed_blocks(),
                                     sigma_site_sq=reg.NOT_ESTIMABLE,
                                     sigma_site_sq_condition="NO_ELIGIBLE_START")):
            return
        ok(f"sigma_site_sq and its condition are computed from the cell's own "
           f"validated block records ({probes} probes plus its positive control)")

        # G — duplicated science across files
        target = fresh()
        target.write_table(svm, edited(svm, **_not_estimable(
            svm, "SVM_FOLD_FAILED")))
        if not refuses("a headline count that does not match the fold table",
                       reg.SVM_MACRO_JSON, _valid_json(reg.SVM_MACRO_JSON),
                       target=target):
            return
        target = fresh()
        target.write_table(per_site, _valid_rows(per_site))
        rows = [dict(r) for r in _valid_rows(per_repeat)]
        for row in rows:
            if row["n_labels"] == 40 \
                    and row["held_out_site"] == contracts.FROZEN_SITES[0]:
                row["held_out_site"] = contracts.FROZEN_SITES[reg.SITES_PER_N[40]]
        if not refuses("two target-local tables disagreeing about eligibility",
                       per_repeat, rows, target=target):
            return
        if set(store_module.CROSS_ARTIFACT_GROUPS) != {
                ("svm",), ("rq4",)} | {("sens56", lab)
                                       for lab in reg.SENS56_APPLIES_TO}:
            fail("the cross-file consistency groups moved")
            return
        try:
            store_module._assert_cross_bindings_complete({})
        except store_module.CrossArtifactError:
            pass
        else:
            fail("the barrier accepted a run in which no group was proved")
            return
        ok("duplicated persisted science is compared across files, and the "
           "barrier refuses a run in which a group was never proved "
           "(2 probes plus the group registry)")


def check_tests(skip: bool) -> None:
    if skip:
        ok("test execution skipped by --skip-tests")
        return
    result = subprocess.run(
        (sys.executable, "-m", "pytest", "-q", *STAGE_F_TESTS),
        cwd=ROOT, capture_output=True, text=True)
    tail = result.stdout.strip().splitlines()[-1] if result.stdout else ""
    if result.returncode != 0:
        fail(f"the Stage-F test suite failed: {tail}")
    else:
        ok(f"the Stage-F test suite passes: {tail}")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    print("Stage F.2 implementation preflight — "
          f"{reg.PROTOCOL_VERSION} @ {reg.PROTOCOL_TAG}\n")
    groups = (
        ("protocol identity", check_protocol_identity),
        ("protocol-validator lifecycle", check_protocol_validator_lifecycle),
        ("constant mirroring", check_constant_mirroring),
        ("borrowed upstream constants", check_upstream_constants),
        ("guarantee labels", check_guarantee_labels),
        ("scientific scope", check_scientific_scope),
        ("SVM scope", check_svm_scope),
        ("SVM warning propagation (§F.7.1, §F.7.1a, §F.7.2)",
         check_svm_warning_propagation),
        ("SVM selection and tie-break (§F.7.5)",
         check_svm_selection_and_tie_break),
        ("SVM_FOLD_FAILED closed scope (§G.2a)", check_svm_fold_failed_scope),
        ("SVM fold_selection schema (§H.3.1)", check_svm_fold_selection_schema),
        ("SENS-1 score_type label (§E.2a)", check_sens1_score_type_label),
        ("output registry", check_output_registry),
        ("RQ4 CI registry", check_ci_registry),
        ("parent reachability (§E.8)", check_parent_reachability),
        ("SENS-6 algorithm and SENS-5/6 schema", check_sens6_algorithm),
        ("sensitivity cardinalities", check_sensitivity_cardinalities),
        ("consumed inputs", check_consumed_inputs),
        ("formal CLI hardening", check_formal_cli),
        ("Formal authority", check_formal_authority),
        ("completion integrity", check_completion_integrity),
        ("genuine provenance round trip", check_genuine_provenance_round_trip),
        ("scientific value semantics", check_scientific_semantics),
        ("artifact-specific reason semantics", check_reason_scoping),
        ("structural cells, nested blocks and inherited state",
         check_structural_semantics),
        ("split-registry identity binding",
         check_split_registry_identity_binding),
        ("formal wiring", check_formal_wiring_binds_identities),
        ("persistence hardening", check_persistence_hardening),
        ("authoritative store", check_store_absent),
        ("import purity", check_modules_write_nothing_at_import),
        ("stage-f tests", lambda: check_tests(args.skip_tests)),
    )
    for title, check in groups:
        print(f"[{title}]")
        check()
        print()

    print(f"{len(_NOTES)} checks across {len(groups)} groups")
    if _FAILURES:
        print(f"VALIDATOR: FAIL ({len(_FAILURES)} failures)")
        for message in _FAILURES:
            print(f"  - {message}")
        return 1
    print(f"VALIDATOR: PASS ({len(_NOTES)} checks)")
    print("Nothing was written. No Stage-F scientific quantity was computed.")
    return 0


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())
