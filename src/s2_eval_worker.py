"""Phase-2 evaluation — a **pure computation worker** (RQ2/S2, Stage 2.0.5).

Stage 2.0.4 tried to stop a direct worker call with a capability token and an
HMAC secret. That is not a real trust boundary in Python: any caller with import
rights can reach a module-private token or secret. The fix is architectural, not
cryptographic.

**This module has no ability to obtain target outcomes.** It contains no outcome
source resolver, no dataset path, no label reader, no phenotype loader and no
file access of any kind. It computes on arrays its caller supplies. A caller who
imports it directly and calls it can only evaluate labels they already had, so a
direct call is not a protocol bypass — there is nothing here to bypass.

Materializing the authoritative target outcomes is the parent's job and happens
in `src/s2_outcomes.py`, **after** the global barrier: all 46,200 decisions
frozen, the release authority re-derived, and phase 2 open. That ordering is
what makes reading them legitimate, and the ledger records it.

What this worker may not do, and structurally cannot: change a threshold,
re-run certification or screening, select a `q`, or reach a file.
"""

from __future__ import annotations

import os

import numpy as np

import certification as cert

#: The complete input contract. Anything not in here is unavailable to the
#: computation, which is the point: there is no argument through which a
#: dataset, a path or a reader could arrive.
EVALUATION_INPUT_FIELDS = (
    "site", "repeat", "n_labels", "strategy", "method", "state",
    "selected_q", "selected_tau", "test_canonical_ids", "y_test", "pred_test",
    "confidence_test", "r_star", "taus", "oracle",
)

#: Everything the worker may return. Counts, rates and states only.
EVALUATION_OUTPUT_FIELDS = frozenset({
    "site", "repeat", "n_labels", "strategy", "method", "guarantee_tier",
    "state", "selected_q", "selected_tau", "realized_field", "coverage_test",
    "n_keep_test", "n_keep_test_MDD", "n_keep_test_HC", "ber_test",
    "risk_excess", "realized_outcome", "jsd_keep_vs_all", "oracle_infeasible",
    "n_oracle_feasible_candidates", "n_test", "worker_pid", "r_star",
    "decision_checksum", "split_checksum",
})

NOT_ESTIMABLE = cert.NOT_ESTIMABLE


class EvaluationInputError(AssertionError):
    """Raised when the supplied evaluation inputs are not self-consistent."""


def evaluate_frozen_cell(inputs: dict) -> dict:
    """Evaluate one frozen cell on the labels the caller supplies.

    `inputs` carries the frozen decision summary (state, `selected_q`,
    `selected_tau`), the cell's exact test `canonical_id`s, and the aligned
    `y_test`, `pred_test` and `confidence_test` arrays. The threshold is
    **read, never chosen**: this function performs no certification, no
    screening, no candidate selection and no re-ranking.
    """
    missing = [f for f in EVALUATION_INPUT_FIELDS if f not in inputs]
    if missing:
        raise EvaluationInputError(f"evaluation inputs are missing {missing}")

    strategy = inputs["strategy"]
    state = inputs["state"]
    cert.assert_state_allowed(strategy, state)

    ids = [str(i) for i in inputs["test_canonical_ids"]]
    y = np.asarray(inputs["y_test"]).ravel().astype(int)
    pred = np.asarray(inputs["pred_test"]).ravel().astype(int)
    conf = np.asarray(inputs["confidence_test"]).ravel().astype(float)
    if not (len(ids) == y.size == pred.size == conf.size):
        raise EvaluationInputError(
            f"the supplied test arrays disagree: {len(ids)} ids, {y.size} "
            f"labels, {pred.size} predictions, {conf.size} confidences")
    if len(set(ids)) != len(ids):
        raise EvaluationInputError("duplicate canonical_id in the test partition")
    if y.size == 0:
        raise EvaluationInputError("an empty test partition cannot be evaluated")

    r_star = float(inputs["r_star"])
    tau = inputs["selected_tau"]
    payload = {
        "site": str(inputs["site"]), "repeat": int(inputs["repeat"]),
        "n_labels": int(inputs["n_labels"]), "strategy": strategy,
        "method": str(inputs["method"]),
        "guarantee_tier": cert.tier_of(strategy), "state": state,
        "selected_q": inputs["selected_q"], "selected_tau": tau,
        "realized_field": cert.realized_outcome_field(strategy),
        "n_test": int(y.size), "r_star": r_star,
        "decision_checksum": inputs.get("decision_checksum"),
        "split_checksum": inputs.get("split_checksum"),
        "worker_pid": os.getpid(),
    }

    if tau is None:
        payload.update({"coverage_test": None, "n_keep_test": 0,
                        "n_keep_test_MDD": 0, "n_keep_test_HC": 0,
                        "ber_test": None, "risk_excess": None,
                        "realized_outcome": None, "jsd_keep_vs_all": None})
    else:
        keep = conf >= float(tau)
        counts = cert.retained_class_counts(y, pred, keep)
        ber = cert.selective_ber_from_counts(counts)
        payload.update({
            "coverage_test": float(keep.mean()),
            "n_keep_test": counts["n_keep"],
            "n_keep_test_MDD": counts["n_keep_MDD"],
            "n_keep_test_HC": counts["n_keep_HC"],
            "ber_test": (None if ber == NOT_ESTIMABLE else float(ber)),
            "risk_excess": (None if ber == NOT_ESTIMABLE else float(ber) - r_star),
            "realized_outcome": (None if ber == NOT_ESTIMABLE
                                 else int(float(ber) > r_star)),
            "jsd_keep_vs_all": _jsd(counts, y),
        })

    if inputs.get("oracle"):
        oracle = _oracle(inputs["taus"], conf, y, pred, r_star)
        payload["oracle_infeasible"] = oracle["oracle_infeasible"]
        payload["n_oracle_feasible_candidates"] = oracle["n_feasible_candidates"]

    assert_evaluation_payload_clean(payload)
    return payload


def assert_evaluation_payload_clean(payload: dict) -> None:
    extra = sorted(set(payload) - EVALUATION_OUTPUT_FIELDS)
    if extra:
        raise EvaluationInputError(f"evaluation payload has extra keys: {extra}")
    for key, value in payload.items():
        if isinstance(value, (list, tuple, np.ndarray, dict)):
            raise EvaluationInputError(
                f"evaluation payload field {key!r} is a container; only counts, "
                "rates and states may be returned")


def _jsd(counts, y):
    from s0_evaluation import jsd_composition
    val = jsd_composition(counts["n_keep_MDD"], counts["n_keep_HC"],
                          int((y == 1).sum()), int((y == 0).sum()))
    return None if val == NOT_ESTIMABLE else float(val)


def _oracle(taus, conf, y, pred, r_star) -> dict:
    coverage, pairs, bers = [], [], []
    for tau in taus:
        keep = conf >= float(tau)
        c = cert.retained_class_counts(y, pred, keep)
        coverage.append(float(keep.mean()) if keep.size else 0.0)
        pairs.append((c["n_keep_MDD"], c["n_keep_HC"]))
        bers.append(cert.selective_ber_from_counts(c))
    return cert.oracle_infeasible(coverage, pairs, bers, r_star)


def _pool_entry(inputs: dict) -> dict:
    """The picklable callable a parent pool maps over. Pure, like the rest."""
    return evaluate_frozen_cell(inputs)


#: Deliberately absent from this module, and audited: `main()`, an argparse
#: parser, a `__main__` block, any dataset path, any outcome-source resolver,
#: any label reader, and any file access whatsoever.
