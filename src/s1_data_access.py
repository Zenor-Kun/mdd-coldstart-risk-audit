"""Outer-fold-scoped data access for RQ1/S1 (`docs/RQ1_S1_PROTOCOL.md` §1, §11).

This module exists because Stage 1.3 attempt 1 could not be frozen. Its
dataflow was clean — no target label reached a `C_domain`, a density weight or
a threshold — but its *access protocol* was not: the pre-decision path called
the labelled cohort loader and so materialized the whole diagnosis vector, and
it held one source-OOF frame carrying `y_true` for every site. A ledger built
on top of that cannot support the claim it was making.

The rule this module enforces is **outer-fold specific**, which is the only
form the claim can honestly take under LOSO:

    for outer target site s, y_s must not be materialized, deserialized or
    exposed before release_s.

The other 22 sites' labels are legitimate source information for fold `s` —
the weighted source calibration of §10 is defined on them. What is forbidden is
the held-out site's own diagnosis, its MDD/HC counts, its prevalence, and
anything derived from them, entering any object that exists before that fold's
decisions are frozen.

Three doors, and no others:

* `build_features_only()` — X, site membership and identity, never an outcome;
* `SourceOofProvider.for_fold(s)` — the calibration rows of fold `s` only,
  selected on a non-outcome key *before* any label is decoded;
* `TargetOutcomeGuard.load_target_y(s)` — the single entry point to target
  diagnosis, which raises unless `release(s)` has already happened.

`scripts/run_rq1_s1.py` asserts statically that no other door exists in the S1
reachable graph.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import contracts
import fc_features

ROOT = Path(__file__).resolve().parents[1]

#: The three outcome-bearing artifacts of the S1 stage live here and nowhere
#: else. Owning the paths in this module is what lets the static audit in the
#: runner assert that no other S1 file can even name them.
MANIFEST = ROOT / "results" / "data_contract" / "analysis_manifest_v1.csv"
S0_SOURCE_OOF = ROOT / "results" / "rq1_s0" / "confidence_source_oof.csv"
S0_TARGET_CONFIDENCE = ROOT / "results" / "rq1_s0" / "confidence_target.csv"

#: Manifest columns the pre-decision path may deserialize. `diagnosis` is
#: absent by construction, and so is every phenotype field: `usecols` means the
#: parser never decodes them, rather than the caller dropping them afterwards.
IDENTITY_COLUMNS = (
    "canonical_id",
    "feature_source_id",
    "original_component_id",
    "site",
    "fc_qc_pass",
    "inclusion_status",
    "n_source_records",
    "source_records",
)

#: Substrings that mark a column as carrying, or being derived from, an
#: outcome. Checked against every frame the pre-decision path is handed.
OUTCOME_TOKENS = ("diagnosis", "y_true", "label", "prevalence", "n_mdd", "n_hc",
                  "bacc", "auroc", "ber_sel", "violation")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_no_outcome_columns(frame, context: str) -> None:
    """Raise if any column of `frame` is, or is derived from, an outcome."""
    bad = sorted(c for c in map(str, frame.columns)
                 if any(t in c.lower() for t in OUTCOME_TOKENS))
    if bad:
        raise AssertionError(
            f"{context}: outcome-derived columns reached a pre-decision "
            f"object: {bad}")
    forbidden = sorted(set(map(str, frame.columns)) & set(contracts.FORBIDDEN_IN_X))
    if "site" in forbidden:
        forbidden.remove("site")            # a split key, never a feature
    for key in ("canonical_id", "feature_source_id", "source_records",
                "original_component_id"):
        if key in forbidden:
            forbidden.remove(key)           # join keys, never features
    if forbidden:
        raise AssertionError(f"{context}: phenotype columns present: {forbidden}")


# ---------------------------------------------------------------------------
# door 1 — features, site membership and identity, with no outcome anywhere
# ---------------------------------------------------------------------------
def load_cohort_identity() -> pd.DataFrame:
    """The frozen 2232-acquisition cohort **without its diagnosis column**.

    Row order is identical to the labelled loader's — sorted by numeric site
    then `canonical_id` — so the frozen FC cache keyed on the ordered
    `feature_source_id` list still hits. Every check here is identity- or
    QC-based; the cohort's MDD/HC composition is *not* verified at this point,
    because verifying it means reading `y`. That assertion is deferred to the
    evaluation stage, where every fold's labels have been released.
    """
    m = pd.read_csv(MANIFEST, usecols=list(IDENTITY_COLUMNS))
    m = m[m["inclusion_status"] == "INCLUDED"].copy()
    m = m.sort_values(["site", "canonical_id"],
                      key=fc_features._site_key).reset_index(drop=True)
    assert_no_outcome_columns(m, "cohort identity frame")
    contracts.assert_unique_acquisitions(m)
    assert len(m) == contracts.FROZEN_N_ACQUISITIONS, \
        f"N={len(m)} != {contracts.FROZEN_N_ACQUISITIONS}"
    sites = sorted(m["site"].unique())
    assert len(sites) == contracts.FROZEN_N_SITES == 23, "site count moved"
    assert set(sites) == set(contracts.FROZEN_SITES), "eligible-site set moved"
    assert m["fc_qc_pass"].all(), "an FC_QC_FAIL row reached the cohort"
    return m


def build_features_only(verbose: bool = True):
    """`(X, sites, cohort_identity)` — the pre-decision view of the data.

    Deliberately **not** the labelled loader: nothing in the returned tuple is
    a diagnosis, an MDD/HC indicator, a class count or anything derived from
    one. `sites` is a split/aggregation key, exactly as the data contract
    allows, and `cohort_identity` carries only join keys and QC flags.
    """
    cohort = load_cohort_identity()
    X = fc_features.build_fc_matrix(cohort, verbose=verbose)
    contracts.assert_feature_matrix(X, len(cohort))
    sites = cohort["site"].to_numpy().astype(str)
    assert isinstance(X, np.ndarray) and X.dtype.kind == "f" and X.dtype.names is None
    return X, sites, cohort


# ---------------------------------------------------------------------------
# door 2 — the fold's own source-OOF calibration rows, and only those
# ---------------------------------------------------------------------------
def _row_index_by_key(path: Path, key_column: str) -> dict:
    """Map each key value to its CSV data-row positions, decoding nothing else.

    `usecols` restricts the parser to one non-outcome column, so this pass
    cannot decode a label even in principle.
    """
    key = pd.read_csv(path, usecols=[key_column])[key_column].to_numpy().astype(str)
    return {v: np.flatnonzero(key == v) for v in np.unique(key)}


def _read_rows(path: Path, rows: np.ndarray, **kw) -> pd.DataFrame:
    """Read exactly `rows` (0-based data rows) — other rows are never decoded."""
    keep = set(int(r) + 1 for r in rows)                # +1 for the header line
    df = pd.read_csv(path, float_precision="round_trip",
                     skiprows=lambda i: i != 0 and i not in keep, **kw)
    assert len(df) == len(rows), f"row projection returned {len(df)}/{len(rows)}"
    return df


class SourceOofProvider:
    """Fold-scoped access to the frozen source-OOF calibration table.

    The frozen table holds one block per outer fold; a fold's block already
    excludes that fold's held-out site by construction. This provider makes
    that structural fact operational: the legal rows are chosen from the
    non-outcome key `outer_fold_held_out_site` **first**, and only then is
    `y_true` decoded, for those rows alone. No object in this class ever holds
    the labels of all 23 sites at once.
    """

    def __init__(self, path=S0_SOURCE_OOF) -> None:
        self._path = Path(path)
        self._rows_of = _row_index_by_key(self._path, "outer_fold_held_out_site")
        self._served: dict[str, int] = {}

    @property
    def folds(self) -> tuple:
        return tuple(sorted(self._rows_of))

    def for_fold(self, held_out_site: str, expected_rows: int | None = None
                 ) -> pd.DataFrame:
        s = str(held_out_site)
        if s not in self._rows_of:
            raise KeyError(f"no source-OOF block for outer fold {s!r}")
        df = _read_rows(self._path, self._rows_of[s])
        if not (df["outer_fold_held_out_site"] == s).all():
            raise AssertionError(f"fold {s}: row projection leaked another fold")
        present = set(df["source_site"].astype(str))
        if s in present:
            raise AssertionError(
                f"fold {s}: the held-out site appears in its own source "
                "calibration pool")
        if len(present) != 22:
            raise AssertionError(f"fold {s}: {len(present)} source sites, not 22")
        if expected_rows is not None and len(df) != expected_rows:
            raise AssertionError(
                f"fold {s}: {len(df)} source rows, expected {expected_rows}")
        self._served[s] = len(df)
        return df

    def audit(self, site: str, df: pd.DataFrame) -> dict:
        """The per-fold evidence that the source label object is clean."""
        return {
            "outer_site": site,
            "n_source_label_rows": int(len(df)),
            "n_source_sites": int(df["source_site"].nunique()),
            "target_site_rows_in_source_labels":
                int((df["source_site"].astype(str) == str(site)).sum()),
            "every_row_carries_a_source_site": bool(
                df["source_site"].notna().all()),
        }


# ---------------------------------------------------------------------------
# door 3 — the only entry point to target diagnosis
# ---------------------------------------------------------------------------
class TargetOutcomeGuard:
    """Per-outer-fold firewall around the held-out site's diagnosis.

    `released[s]` starts `False` for all 23 folds. `load_target_y(s)` raises
    unless that fold's decisions — `C_domain`, the deployed domain fit, the
    density weights, the ESS, every `tau_q^S1` and their checksums — are
    already on disk and `release(s)` has been called. Releasing one fold says
    nothing about any other: fold `s'` stays closed until its own decisions
    are frozen.

    Every event carries a monotonic sequence number, so the ledger proves the
    ordering rather than asserting it.
    """

    REQUIRED_DECISIONS = ("selected_domain_C", "domain_fit_diagnostics",
                          "density_weights", "ess", "thresholds_weighted_source",
                          "weighted_source_risk", "decision_checksums")

    def __init__(self, sites, target_confidence_path=S0_TARGET_CONFIDENCE) -> None:
        self._path = Path(target_confidence_path)
        self._sites = tuple(str(s) for s in sites)
        self._released = {s: False for s in self._sites}
        self._decisions: dict[str, dict[str, str]] = {s: {} for s in self._sites}
        self._events: list[dict] = []
        self._seq = 0
        self._t0 = time.perf_counter()
        # non-outcome pass: which CSV rows belong to which held-out site
        self._rows_of = _row_index_by_key(self._path, "held_out_site")
        missing = sorted(set(self._sites) - set(self._rows_of))
        if missing:
            raise AssertionError(f"no target block for {missing}")

    # -- ledger -------------------------------------------------------------
    def _stamp(self, site: str, event: str, **detail) -> dict:
        self._seq += 1
        row = {"seq": self._seq, "outer_site": site, "event": event,
               "utc": datetime.now(timezone.utc).isoformat(),
               "monotonic_seconds": time.perf_counter() - self._t0, **detail}
        self._events.append(row)
        return row

    def record_decision(self, site: str, artifact: str, path) -> dict:
        s = str(site)
        if self._released[s]:
            raise RuntimeError(
                f"fold {s}: decision {artifact!r} recorded after release; every "
                "S1 decision must be frozen before the target labels open")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"fold {s}/{artifact}: {p} is not on disk")
        digest = sha256_file(p)
        self._decisions[s][artifact] = digest
        return self._stamp(s, f"decision_frozen__{artifact}", path=str(p),
                           sha256=digest, bytes=int(p.stat().st_size))

    def release(self, site: str) -> dict:
        s = str(site)
        if self._released[s]:
            raise RuntimeError(f"fold {s}: already released")
        missing = sorted(set(self.REQUIRED_DECISIONS) - set(self._decisions[s]))
        if missing:
            raise RuntimeError(
                f"fold {s}: cannot release the target labels, {missing} are not "
                "frozen on disk")
        self._released[s] = True
        return self._stamp(s, "target_labels_released",
                           n_decisions=len(self._decisions[s]))

    def load_target_y(self, site: str, expected_rows: int | None = None
                      ) -> pd.DataFrame:
        """The single door to target diagnosis. Raises before `release(site)`."""
        s = str(site)
        if s not in self._released:
            raise KeyError(f"{s!r} is not an outer fold of this run")
        if not self._released[s]:
            raise RuntimeError(
                f"fold {s}: target diagnosis requested before the fold's "
                "C_domain, density weights and thresholds were frozen on disk "
                "(docs/RQ1_S1_PROTOCOL.md §11)")
        df = _read_rows(self._path, self._rows_of[s])
        if not (df["held_out_site"] == s).all():
            raise AssertionError(f"fold {s}: target projection leaked another site")
        if expected_rows is not None and len(df) != expected_rows:
            raise AssertionError(
                f"fold {s}: {len(df)} target rows, expected {expected_rows}")
        self._stamp(s, "target_labels_loaded", n_rows=int(len(df)))
        return df

    # -- aggregate stage ----------------------------------------------------
    def require_all_released(self, reason: str) -> None:
        closed = sorted(s for s, r in self._released.items() if not r)
        if closed:
            raise RuntimeError(
                f"{reason}: folds {closed} have not released their target "
                "labels; a cross-site aggregate may not precede them")
        self._stamp("", "all_folds_released", reason=reason)

    @property
    def all_released(self) -> bool:
        return all(self._released.values())

    # -- the per-fold ledger of §8 ------------------------------------------
    def fold_ledger(self, extra: dict | None = None) -> pd.DataFrame:
        extra = extra or {}
        rows = []
        for s in self._sites:
            ev = [e for e in self._events if e["outer_site"] == s]

            def seq(prefix, evs=ev):
                hits = [e["seq"] for e in evs if e["event"].startswith(prefix)]
                return min(hits) if hits else None

            ev_seq = seq("target_evaluation_completed")
            c = seq("decision_frozen__selected_domain_C")
            w = seq("decision_frozen__density_weights")
            t = seq("decision_frozen__thresholds_weighted_source")
            k = seq("decision_frozen__decision_checksums")
            rel = seq("target_labels_released")
            first_y = seq("target_labels_loaded")
            pre = [e for e in ev if e["event"] == "target_labels_loaded"
                   and rel is not None and e["seq"] < rel]
            decisions = [e["seq"] for e in ev
                         if e["event"].startswith("decision_frozen__")]
            ok = (rel is not None and first_y is not None and not pre
                  and None not in (c, w, t, k, ev_seq)
                  and max(decisions) < rel < first_y <= ev_seq)
            rows.append({
                "outer_site": s,
                "prebarrier_X_loaded": True,
                "source_labels_loaded": extra.get(s, {}).get(
                    "n_source_label_rows"),
                "source_label_target_rows": extra.get(s, {}).get(
                    "target_site_rows_in_source_labels"),
                "target_labels_visible_prebarrier": bool(pre),
                "c_domain_frozen_seq": c,
                "weights_frozen_seq": w,
                "thresholds_frozen_seq": t,
                "decision_checksums_seq": k,
                "n_decision_artifacts": len(decisions),
                "last_decision_seq": max(decisions) if decisions else None,
                "release_seq": rel,
                "first_target_y_seq": first_y,
                "evaluation_seq": ev_seq,
                "ordering_ok": bool(ok),
            })
        return pd.DataFrame(rows)

    def record_evaluation(self, site: str) -> int:
        return self._stamp(str(site), "target_evaluation_completed")["seq"]

    def write(self, path, fold_ledger: pd.DataFrame) -> None:
        payload = {
            "firewall": "outer-fold-scoped target diagnosis",
            "rule": "for outer target s, y_s is not materialized before release_s",
            "n_folds": len(self._sites),
            "n_folds_ordering_ok": int(fold_ledger["ordering_ok"].sum()),
            "decision_checksums": self._decisions,
            "fold_ledger": fold_ledger.to_dict(orient="records"),
            "events": self._events,
        }
        Path(path).write_text(json.dumps(payload, indent=2, default=str))
