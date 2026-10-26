"""The isolated `(site, repeat, N)` cell worker (RQ2/S2, Stage 2.0.2).

One process, one cell, one label partition. Stage 2.0.1 established the process
boundary; Stage 2.0.2 closes the three ways a caller could still steer what
happened inside it:

* **cell membership** is no longer supplied. A job names `(site, repeat, N)` and
  a registered split manifest; the worker resolves the exact calibration and
  test `canonical_id`s from that registry itself. A job carrying rows,
  confidences or predictions is rejected outright.
* **the outcome source** is an internal immutable constant. It cannot be passed
  in, overridden by the job, or redirected by the parent.
* **what was read is verified**, not assumed: exact `canonical_id` set equality,
  exact site, no extra row, no missing row, no duplicate — and the raw
  identifiers and labels stay inside the process.

Two mutually exclusive modes, and the job schema is what makes them exclusive:
a `decision` job never names, and cannot resolve, its test partition; an
`evaluation` job carries a frozen decision artifact and a global release
artifact and may not alter either.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import certification as cert
import s2_decisions as sd
import s2_production
import shrinkage as shr
from s2_splitting import SplitRegistry

ROOT = Path(__file__).resolve().parents[1]

MODE_DECISION = "decision"
MODE_EVALUATION = "evaluation"

# ---------------------------------------------------------------------------
# the outcome source is an internal immutable dependency (Stage 2.0.3)
# ---------------------------------------------------------------------------
# Stage 2.0.2 honoured two environment variables here, and the parent forwarded
# them, so a production run's outcome source could be replaced from outside.
# Both are gone. This module reads no environment variable, accepts no path and
# has no override branch: the source comes from
# `s2_production.resolve_authoritative_outcome_source()` and nowhere else.
# Test injection lives entirely outside production code, in a tests-only reader
# and a tests-only worker script that the production import graph never names.

#: Keys a caller may never place in a job: each of them would let the caller,
#: rather than the registry, decide what the worker reads.
FORBIDDEN_JOB_KEYS = (
    "cal_rows", "test_rows", "cal_ids", "test_ids", "cal_confidence",
    "cal_pred", "test_confidence", "test_pred", "label_source",
    "outcome_source", "labels", "y", "y_true", "rows", "dataframe",
    "test_outcome_source",
)

LABEL_COLUMN = "y_true"
ID_COLUMN = "canonical_id"
SITE_COLUMN = "held_out_site"
PRED_COLUMN = "pred"


def resolve_outcome_source() -> Path:
    """The authoritative outcome source. Takes no argument, reads no env."""
    return s2_production.resolve_authoritative_outcome_source()


#: The only keys a decision artifact may carry across the process boundary.
DECISION_PAYLOAD_KEYS = frozenset({
    "protocol_version", "mode", "site", "repeat", "n_labels", "method",
    "strategy", "guarantee_tier", "score_type", "guarantee", "state", "reason",
    "selected_q", "selected_tau", "selected_score", "n_eligible",
    "first_eligible_index", "n_passed_candidates", "n_rejected_hypotheses",
    "stopped_at_index", "nominal_alpha_ucb", "screening_reference", "scope",
    "n_keep_cal", "n_keep_cal_MDD", "n_keep_cal_HC", "x_cal_MDD", "x_cal_HC",
    "n0", "lambda_local_selected", "score_by_q", "evaluated_q", "eligible_q",
    "selected_candidate_index",
    "worker_run_id", "worker_pid", "cal_label_rows_read",
    "test_label_rows_read_during_decision", "split_checksum",
    "registry_checksum", "cal_id_set_checksum", "outcome_source_authoritative",
    "r_star",
})

EVALUATION_PAYLOAD_KEYS = frozenset({
    "protocol_version", "mode", "site", "repeat", "n_labels", "method",
    "strategy", "guarantee_tier", "state", "selected_q", "selected_tau",
    "realized_field", "selected_candidate_index", "selected_score",
    "coverage_test", "n_keep_test", "n_keep_test_MDD",
    "n_keep_test_HC", "ber_test", "risk_excess", "realized_outcome",
    "jsd_keep_vs_all", "oracle_infeasible", "n_oracle_feasible_candidates",
    "worker_run_id", "worker_pid", "test_label_rows_read", "decision_checksum",
    "decision_checksum_unchanged", "split_checksum", "registry_checksum",
    "test_id_set_checksum", "outcome_source_authoritative", "release_checksum",
    "r_star",
})

RAW_LABEL_KEY_TOKENS = ("y_true", "y_cal", "y_test", "diagnosis",
                        "label_vector", "raw_label", "y_rows", "label_values",
                        "canonical_ids", "cal_ids", "test_ids")

NOT_ESTIMABLE = cert.NOT_ESTIMABLE


def sha256_file(path) -> str:
    return sd.sha256_file(path)


def id_set_checksum(ids) -> str:
    """A checksum over the identifier set — never the identifiers themselves."""
    return hashlib.sha256(";".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# job hygiene
# ---------------------------------------------------------------------------
def assert_job_clean(job: dict, mode: str) -> None:
    """A job may name a cell and a registry. It may not name data."""
    bad = sorted(k for k in job if k in FORBIDDEN_JOB_KEYS)
    if bad:
        raise AssertionError(
            f"{mode} job carries caller-controlled data selection {bad}; cell "
            "membership comes from the registered split manifest and the "
            "outcome source is an internal constant "
            "(docs/RQ2_S2_PROTOCOL.md §17.4)")
    for field in ("site", "repeat", "n_labels", "method", "strategy",
                  "split_registry_path", "split_registry_checksum"):
        if field not in job:
            raise AssertionError(f"{mode} job is missing {field!r}")
    for field in ("repeat", "n_labels"):
        if isinstance(job[field], bool) or not isinstance(job[field], int):
            raise AssertionError(
                f"{mode} job field {field!r} must be a single integer; the "
                "isolation unit is one (site, repeat, N) cell")


# ---------------------------------------------------------------------------
# the only door to a target label, bound to a registered cell
# ---------------------------------------------------------------------------
class ProductionCellLabelReader:
    """Reads exactly the rows of one registered partition, then proves it did.

    The allow-list is a set of `canonical_id`s resolved from the split registry
    — not a row range, not a length. After the read the reader checks the row
    count, the exact identifier set, the site of every row and the absence of
    duplicates. "N rows came back, so it must be right" is precisely the
    assumption this class exists to remove.
    """

    def __init__(self, source_path, site: str, allowed_ids) -> None:
        self._path = Path(source_path)
        self._site = str(site)
        self._allowed = tuple(str(i) for i in allowed_ids)
        if len(set(self._allowed)) != len(self._allowed):
            raise AssertionError("duplicate canonical_id in the cell allowance")
        key = pd.read_csv(self._path, usecols=[SITE_COLUMN, ID_COLUMN])
        ids = key[ID_COLUMN].astype(str).to_numpy()
        sites = key[SITE_COLUMN].astype(str).to_numpy()
        self._row_of = {}
        for i, (cid, st) in enumerate(zip(ids, sites)):
            if cid in self._row_of:
                raise AssertionError(f"duplicate canonical_id {cid} in the source")
            self._row_of[cid] = (i, st)
        self.rows_read = 0

    def read(self) -> pd.DataFrame:
        missing = [i for i in self._allowed if i not in self._row_of]
        if missing:
            raise AssertionError(
                f"{len(missing)} registered canonical_ids are absent from the "
                f"outcome source, e.g. {missing[:3]}")
        foreign = [i for i in self._allowed if self._row_of[i][1] != self._site]
        if foreign:
            raise AssertionError(
                f"{len(foreign)} registered ids belong to another site, e.g. "
                f"{[self._row_of[i][1] for i in foreign[:3]]}; a cell may only "
                f"read rows of {self._site}")
        want = [self._row_of[i][0] for i in self._allowed]
        keep = {r + 1 for r in want}                    # +1 for the header line
        df = pd.read_csv(self._path, float_precision="round_trip",
                         skiprows=lambda i: i != 0 and i not in keep)

        if len(df) != len(self._allowed):
            raise AssertionError(
                f"row projection returned {len(df)}/{len(self._allowed)} rows")
        got = df[ID_COLUMN].astype(str)
        if got.duplicated().any():
            raise AssertionError("the projection returned a duplicated row")
        if set(got) != set(self._allowed):
            extra = sorted(set(got) - set(self._allowed))
            absent = sorted(set(self._allowed) - set(got))
            raise AssertionError(
                f"the projection is not the registered cell: {len(extra)} extra, "
                f"{len(absent)} missing")
        if not (df[SITE_COLUMN].astype(str) == self._site).all():
            raise AssertionError(f"the projection leaked a row of another site")
        # canonicalise the order to the registry's, so nothing downstream
        # depends on CSV row order
        df = df.set_index(ID_COLUMN).loc[list(self._allowed)].reset_index()
        self.rows_read = len(df)
        return df


def _resolve_cell(job: dict, partition: str) -> dict:
    """Bind the job to exactly one registered cell of the frozen manifest."""
    registry = SplitRegistry.load(job["split_registry_path"],
                                 job["split_registry_checksum"])
    cell = registry.cell(job["site"], job["repeat"], job["n_labels"])
    if job.get("cell_checksum") not in (None, cell["cell_checksum"]):
        raise AssertionError(
            f"cell checksum mismatch for {job['site']}/{job['repeat']}/"
            f"{job['n_labels']}")
    cell["ids"] = cell["cal_ids"] if partition == "cal" else cell["test_ids"]
    return cell


# ---------------------------------------------------------------------------
# decision mode
# ---------------------------------------------------------------------------
def run_decision_cell(job: dict) -> dict:
    """One `(site, repeat, N, strategy, method)` decision, in this process.

    Reads through the production reader from the authoritative source, then
    delegates the arithmetic to `decide_from_frame`. The split is deliberate:
    the compute half takes an already-read frame, so tests exercise it without
    any source indirection existing in production code.
    """
    assert_job_clean(job, MODE_DECISION)
    cell = _resolve_cell(job, "cal")
    source = resolve_outcome_source()
    s2_production.assert_authoritative_outcome_source(source)
    reader = ProductionCellLabelReader(source, cell["site"], cell["ids"])
    frame = reader.read()
    payload = decide_from_frame(job, cell, frame, reader.rows_read)
    payload["outcome_source_authoritative"] = bool(
        Path(source).resolve() == s2_production.resolve_authoritative_outcome_source().resolve())
    return payload


def decide_from_frame(job: dict, cell: dict, frame, rows_read: int) -> dict:
    """The decision arithmetic, on an already-read calibration partition."""
    strategy = job["strategy"]
    if strategy not in cert.LABEL_CONSUMING_STRATEGIES:
        raise AssertionError(
            f"{strategy!r} consumes no target calibration label and must be "
            "decided in the target-label-free phase 0, not in a decision cell")
    y = frame[LABEL_COLUMN].to_numpy().astype(int)
    pred = frame[PRED_COLUMN].to_numpy().astype(int)
    conf = frame[f"confidence_{job['method']}"].to_numpy().astype(float)

    taus = [float(t) for t in job["taus"]]
    qs = [float(q) for q in job["q_values"]]
    if tuple(qs) != tuple(cert.Q_GRID):
        raise AssertionError(
            "a cell decision must run the frozen 17-candidate grid in the "
            "frozen order; the candidate index is a grid index")
    r_star = float(job["r_star"])
    src = job["source_class_counts"]
    n0 = float(job.get("n0", 0.0))

    coverage, eligible, cell_counts, scores = [], [], [], []
    for i, tau in enumerate(taus):
        keep = conf >= tau
        c = cert.retained_class_counts(y, pred, keep)
        cell_counts.append(c)
        cov = float(keep.mean()) if keep.size else 0.0
        coverage.append(cov)
        n_ms, x_ms, n_hs, x_hs = (int(v) for v in src[i])
        counts_for_eligibility = ((c["n_keep_MDD"], c["n_keep_HC"])
                                  if strategy == "target_local"
                                  else (c["n_keep_MDD"], c["n_keep_HC"], n_ms, n_hs))
        eligible.append(cert.candidate_eligible(cov, counts_for_eligibility))
        if strategy == "target_local":
            scores.append(None)
        else:
            scores.append((
                shr.empirical_pseudocount_beta_upper_score(
                    c["x_MDD"], c["n_keep_MDD"], (x_ms / n_ms) if n_ms else 1.0, n0),
                shr.empirical_pseudocount_beta_upper_score(
                    c["x_HC"], c["n_keep_HC"], (x_hs / n_hs) if n_hs else 1.0, n0)))

    if strategy == "target_local":
        def score_fn(i):
            u_m = cert.clopper_pearson_upper(cell_counts[i]["x_MDD"],
                                             cell_counts[i]["n_keep_MDD"])
            u_h = cert.clopper_pearson_upper(cell_counts[i]["x_HC"],
                                             cell_counts[i]["n_keep_HC"])
            return cert.u_ber(u_m, u_h)
    else:
        def score_fn(i):
            return cert.mean_class_score(scores[i][0]["score"],
                                         scores[i][1]["score"])

    out = cert.run_sequence_for_strategy(strategy, qs, eligible, score_fn, r_star)
    sel = out["selected_index"]
    reason = out["reason"]
    if out["state"] == cert.NOT_TESTABLE:
        # name the constraint that bit, rather than a generic code: coverage is
        # monotone along the order, so the loosest candidate decides which.
        reason = (sd.REASON_COVERAGE if coverage[-1] < cert.C_MIN
                  else sd.REASON_CLASS_COUNT)
    counts_sel = cell_counts[sel] if sel is not None else {
        "n_keep": 0, "n_keep_MDD": 0, "n_keep_HC": 0, "x_MDD": 0, "x_HC": 0}
    lam = None
    if strategy == "pooled_local_shrinkage" and sel is not None:
        lam = 0.5 * (scores[sel][0]["lambda_local"] + scores[sel][1]["lambda_local"])

    payload = {
        "protocol_version": sd.PROTOCOL_VERSION,
        "mode": MODE_DECISION,
        "site": cell["site"], "repeat": cell["repeat"],
        "n_labels": cell["n_labels"], "method": job["method"],
        "strategy": strategy, "guarantee_tier": out["guarantee_tier"],
        "score_type": out["score_type"], "guarantee": out["guarantee"],
        "state": out["state"], "reason": reason,
        "selected_q": out["selected_q"],
        "selected_candidate_index": sel,
        "selected_tau": (taus[sel] if sel is not None else None),
        "selected_score": (out["score_by_index"].get(sel) if sel is not None
                           else None),
        "n_eligible": out["n_eligible"],
        "first_eligible_index": out["first_eligible_index"],
        "n_passed_candidates": out["n_passed_candidates"],
        "stopped_at_index": out["stopped_at_index"],
        "n_keep_cal": counts_sel["n_keep"],
        "n_keep_cal_MDD": counts_sel["n_keep_MDD"],
        "n_keep_cal_HC": counts_sel["n_keep_HC"],
        "x_cal_MDD": counts_sel["x_MDD"], "x_cal_HC": counts_sel["x_HC"],
        "score_by_q": {str(qs[i]): v for i, v in out["score_by_index"].items()},
        "evaluated_q": [qs[i] for i in out["evaluated_indices"]],
        "eligible_q": [qs[i] for i, f in enumerate(eligible) if f],
        "worker_run_id": job["run_id"], "worker_pid": os.getpid(),
        "cal_label_rows_read": int(rows_read),
        "test_label_rows_read_during_decision": 0,
        "split_checksum": cell["cell_checksum"],
        "registry_checksum": cell["registry_checksum"],
        "cal_id_set_checksum": id_set_checksum(cell["ids"]),
        "outcome_source_authoritative": False,
        "r_star": r_star,
    }
    if strategy == "target_local":
        payload["n_rejected_hypotheses"] = out["n_rejected_hypotheses"]
        payload["nominal_alpha_ucb"] = out["nominal_alpha_ucb"]
    else:
        payload["screening_reference"] = out["screening_reference"]
        payload["n0"] = n0
        payload["lambda_local_selected"] = lam

    assert_payload_clean(payload, MODE_DECISION)
    sd.validate_decision_artifact(payload, zero_label=False)
    return payload


# ---------------------------------------------------------------------------
# evaluation mode
# ---------------------------------------------------------------------------
def evaluate_from_frame(job: dict, cell: dict, decision: dict, frame, release,
                        decision_checksum: str, rows_read: int) -> dict:
    """The evaluation arithmetic, on an already-read test partition."""
    strategy = decision["strategy"]
    y = frame[LABEL_COLUMN].to_numpy().astype(int)
    pred = frame[PRED_COLUMN].to_numpy().astype(int)
    conf = frame[f"confidence_{job['method']}"].to_numpy().astype(float)
    r_star = float(job["r_star"])
    tau = decision["selected_tau"]
    frozen = {"sha256": decision_checksum}

    payload = {
        "protocol_version": sd.PROTOCOL_VERSION,
        "mode": MODE_EVALUATION,
        "site": cell["site"], "repeat": cell["repeat"],
        "n_labels": cell["n_labels"], "method": job["method"],
        "strategy": strategy, "guarantee_tier": decision["guarantee_tier"],
        "state": decision["state"], "selected_q": decision["selected_q"],
        "selected_tau": tau,
        "selected_candidate_index": decision["selected_candidate_index"],
        "selected_score": decision["selected_score"],
        "realized_field": cert.realized_outcome_field(strategy),
        "worker_run_id": job["run_id"], "worker_pid": os.getpid(),
        "decision_checksum": frozen["sha256"],
        "split_checksum": cell["cell_checksum"],
        "registry_checksum": cell["registry_checksum"],
        "test_id_set_checksum": id_set_checksum(cell["ids"]),
        "outcome_source_authoritative": True,
        "release_checksum": release["release_checksum"],
        "r_star": r_star,
    }

    if tau is None:
        payload.update({"coverage_test": None, "n_keep_test": 0,
                        "n_keep_test_MDD": 0, "n_keep_test_HC": 0,
                        "ber_test": None, "risk_excess": None,
                        "realized_outcome": None, "jsd_keep_vs_all": None})
    else:
        keep = conf >= float(tau)
        c = cert.retained_class_counts(y, pred, keep)
        ber = cert.selective_ber_from_counts(c)
        payload.update({
            "coverage_test": float(keep.mean()) if keep.size else 0.0,
            "n_keep_test": c["n_keep"], "n_keep_test_MDD": c["n_keep_MDD"],
            "n_keep_test_HC": c["n_keep_HC"],
            "ber_test": (None if ber == NOT_ESTIMABLE else float(ber)),
            "risk_excess": (None if ber == NOT_ESTIMABLE else float(ber) - r_star),
            "realized_outcome": (None if ber == NOT_ESTIMABLE
                                 else int(float(ber) > r_star)),
            "jsd_keep_vs_all": _jsd(c, y),
        })

    if job.get("oracle", False):
        oracle = _oracle(job, conf, y, pred, r_star)
        payload["oracle_infeasible"] = oracle["oracle_infeasible"]
        payload["n_oracle_feasible_candidates"] = oracle["n_feasible_candidates"]

    payload["test_label_rows_read"] = int(rows_read)
    return payload


def _jsd(counts, y):
    from s0_evaluation import jsd_composition
    val = jsd_composition(counts["n_keep_MDD"], counts["n_keep_HC"],
                          int((y == 1).sum()), int((y == 0).sum()))
    return None if val == NOT_ESTIMABLE else float(val)


def _oracle(job, conf, y, pred, r_star) -> dict:
    coverage, pairs, bers = [], [], []
    for tau in job["taus"]:
        keep = conf >= float(tau)
        c = cert.retained_class_counts(y, pred, keep)
        coverage.append(float(keep.mean()) if keep.size else 0.0)
        pairs.append((c["n_keep_MDD"], c["n_keep_HC"]))
        bers.append(cert.selective_ber_from_counts(c))
    return cert.oracle_infeasible(coverage, pairs, bers, r_star)


# ---------------------------------------------------------------------------
# payload hygiene
# ---------------------------------------------------------------------------
def assert_payload_clean(payload: dict, mode: str) -> None:
    """Whitelist the artifact schema and refuse anything label- or id-shaped."""
    allowed = (DECISION_PAYLOAD_KEYS if mode == MODE_DECISION
               else EVALUATION_PAYLOAD_KEYS)
    extra = sorted(set(payload) - allowed)
    if extra:
        raise AssertionError(f"{mode} payload carries unregistered keys: {extra}")
    for key, value in payload.items():
        if any(tok in key.lower() for tok in RAW_LABEL_KEY_TOKENS):
            raise AssertionError(f"{mode} payload key {key!r} is label-shaped")
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) > len(cert.Q_GRID_SENSITIVITY):
                raise AssertionError(
                    f"{mode} payload value {key!r} is longer than the candidate "
                    "grid; only per-candidate summaries may cross the boundary")
        elif isinstance(value, dict):
            if len(value) > len(cert.Q_GRID_SENSITIVITY):
                raise AssertionError(
                    f"{mode} payload mapping {key!r} is longer than the grid")


# ---------------------------------------------------------------------------
# process entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: s2_cell_worker <job.json> <out.json>", file=sys.stderr)
        return 2
    job = json.loads(Path(argv[0]).read_text())
    job.setdefault("run_id", uuid.uuid4().hex)
    mode = job["mode"]
    if mode == MODE_DECISION:
        payload = run_decision_cell(job)
        sd.freeze_decision_artifact(payload, argv[1], zero_label=False)
        return 0
    # Stage 2.0.4: there is no standalone evaluation executable. A phase-2
    # evaluation runs only inside a pool created by `open_phase2()`, whose
    # workers hold a capability that cannot be obtained from a command line.
    raise ValueError(
        f"mode {mode!r} has no command-line entry point; phase-2 evaluation "
        "requires a Phase2CellPermit issued by an open phase 2 "
        "(docs/RQ2_S2_PROTOCOL.md §17.17)")


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())
