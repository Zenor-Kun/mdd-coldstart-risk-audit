#!/usr/bin/env python3
"""RQ2/S2 final provenance freeze — checksums and mechanical re-derivation.

Documentation / provenance only. This script **reads** the frozen authoritative
Formal S2 result tree and writes nothing inside it. It fits no model, runs no
certification, reads no target outcome and recomputes no scientific quantity —
it re-derives the closure's headline counts *arithmetically from the published
registered artifacts* to prove they follow from the tree, and fails loudly if
any of them does not.

Why this stage does not write into `results/rq2_s2/` (the one place where the
S0/S1 convention needed a decision rather than a mechanical copy): unlike S0 and
S1, the S2 runner emits its own `RESULT_CHECKSUMS.sha256` **and** binds its
SHA256 into `formal_run_completion.json["checksum_manifest_checksum"]`. That
name is therefore already taken by an authoritative scientific artifact, and the
authoritative tree audited by the independent result closure must stay identical
file-for-file, not merely byte-for-byte in its existing files. The freeze
artifacts are written to the sibling `results/rq2_s2_freeze/` instead; the
ledger addresses the authoritative files by relative path so that

    cd results/rq2_s2_freeze && sha256sum -c RESULT_CHECKSUMS.sha256

still verifies them in place. Everything else follows S0/S1 mechanically: the
small authoritative artifacts are committed, the 171 149-file per-cell store is
recorded by checksum and deliberately NOT committed.

Usage:
    python3 scripts/freeze_rq2_s2_provenance.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "rq2_s2"
OUT = ROOT / "results" / "rq2_s2_freeze"
LEDGER = OUT / "RESULT_CHECKSUMS.sha256"
BINDING = OUT / "authoritative_run_binding.json"

#: The authoritative run this freeze is bound to. A mismatch is a hard failure:
#: the freeze may not silently re-point at a different run.
RUN_ID = "4ad32c2bc302426f832132574def81d4"
LEDGER_ID = "0c92625dfb89469abf09d93588f6f784"
PROTOCOL_TAG = "rq2-s2-protocol-v1"
PROTOCOL_SHA = "57fa1ded9777699b4b632ad9e03f3761e14925ea"
EXECUTION_TAG = "rq2-s2-execution-v3"
EXECUTION_SHA = "5d077a63bd22a4a1f13a5776b8b6134665d0c341"
PROTOCOL_VERSION = "RQ2_S2_PROTOCOL/2.0.8"

#: Frozen counts (spec + protocol, fixed long before this run existed).
N_ZERO_LABEL = 138
N_DECISIONS = 46_200
N_EVALUATIONS = 78_600
N_SECTIONS = 16
N_REGISTERED_SECTIONS = 20
N_INTERVALS = 216
CI_FAMILIES = {"formal": 39, "empirical_few_label": 27, "zero_label": 6,
               "descriptive": 108, "descriptive_contrast": 36}
PER_N_SITES = {"5": 23, "10": 22, "20": 21, "40": 11}

#: Small authoritative artifacts — committed in place, per the S0/S1 rule that
#: the closure/summary layer is reviewable in Git.
COMMITTED = (
    "formal_run_provenance.json",
    "authoritative_ledger_binding.json",
    "global_release.json",
    "registered_intervals.csv",
    "RESULT_CHECKSUMS.sha256",
    "formal_run_completion.json",
)
#: Recorded by checksum but deliberately NOT committed (bulk per-cell store and
#: the two large aggregate artifacts).
BULK_TOP = ("evaluation_manifest.json", "formal_point_sections.json",
            "global_run_ledger.csv", "production_split_registry.csv")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_digest(root: Path) -> tuple[str, int]:
    """Relocation-invariant digest over (relpath, sha256(bytes)), sorted."""
    files = sorted((p for p in root.rglob("*") if p.is_file()),
                   key=lambda p: str(p.relative_to(root)))
    h = hashlib.sha256()
    for path in files:
        rel = str(path.relative_to(root))
        h.update(rel.encode())
        h.update(bytes.fromhex(sha256(path)))
    return h.hexdigest(), len(files)


def recompute_headlines() -> tuple[list[str], dict]:
    """Re-derive the closure from the published artifacts. Read-only."""
    notes, bad = [], []
    facts: dict = {}

    completion = json.loads((RUN / "formal_run_completion.json").read_text())
    provenance = json.loads((RUN / "formal_run_provenance.json").read_text())
    binding = json.loads((RUN / "authoritative_ledger_binding.json").read_text())
    rp = completion["report_provenance"]

    # ---- identity ---------------------------------------------------------
    ids = {"completion.run_id": completion["run_id"],
           "provenance.run_id": provenance["run_id"],
           "binding.run_id": binding["run_id"],
           "report_provenance.run_id": rp["run_id"]}
    if set(ids.values()) != {RUN_ID}:
        bad.append(f"run_id disagreement: {ids}")
    notes.append(f"run_id {RUN_ID} agreed by {len(ids)} independent artifacts")

    led = {"completion": completion["authoritative_ledger_id"],
           "binding": binding["authoritative_ledger_id"],
           "ledger_ordering": completion["ledger_ordering"]["ledger_id"],
           "report_provenance": rp["global_ledger_id"]}
    if set(led.values()) != {LEDGER_ID}:
        bad.append(f"ledger id disagreement: {led}")
    notes.append(f"authoritative ledger {LEDGER_ID} governs all events")

    repo = provenance["repository"]
    for key, want in (("protocol_tag", PROTOCOL_TAG),
                      ("protocol_tag_sha", PROTOCOL_SHA),
                      ("head_sha", EXECUTION_SHA),
                      ("protocol_version", PROTOCOL_VERSION)):
        if repo[key] != want:
            bad.append(f"provenance.repository.{key} = {repo[key]} != {want}")
    if repo["working_tree_dirty"]:
        bad.append("the run was made from a dirty working tree")
    notes.append(f"protocol {PROTOCOL_TAG} {PROTOCOL_SHA[:12]} · execution HEAD "
                 f"{EXECUTION_SHA[:12]} · clean tree")

    # ---- mechanical counts ------------------------------------------------
    on_disk = {
        "decisions": sum(1 for _ in (RUN / "decisions").glob("*.json")),
        "evaluations": sum(1 for p in (RUN / "evaluations").rglob("*.json")),
        "zero_label": sum(1 for _ in RUN.glob("zero__*.json")),
    }
    for name, got, want in (("zero-label", on_disk["zero_label"], N_ZERO_LABEL),
                            ("decisions", on_disk["decisions"], N_DECISIONS),
                            ("evaluations", on_disk["evaluations"], N_EVALUATIONS)):
        if got != want:
            bad.append(f"{name}: {got} artifacts on disk != {want}")
    for name, got, want in (("rp.n_zero_label_decisions",
                             rp["n_zero_label_decisions"], N_ZERO_LABEL),
                            ("rp.n_decision_cells", rp["n_decision_cells"], N_DECISIONS),
                            ("rp.n_evaluation_cells", rp["n_evaluation_cells"], N_EVALUATIONS),
                            ("completion.n_evaluations", completion["n_evaluations"], N_EVALUATIONS)):
        if got != want:
            bad.append(f"{name} = {got} != {want}")
    notes.append(f"counts reproduce on disk and in the report provenance: "
                 f"{N_ZERO_LABEL} zero-label / {N_DECISIONS} decisions / "
                 f"{N_EVALUATIONS} evaluations")
    facts["counts"] = on_disk

    if {str(k): v for k, v in rp["per_n_site_counts"].items()} != PER_N_SITES:
        bad.append(f"per-N site counts {rp['per_n_site_counts']} != {PER_N_SITES}")
    notes.append(f"per-N eligible sites {PER_N_SITES}")

    # ---- sections and intervals -------------------------------------------
    rc = completion["report_completeness"]
    if (rc["n_point_sections"], rc["n_registered_point_sections"],
            rc["n_intervals"]) != (N_SECTIONS, N_REGISTERED_SECTIONS, N_INTERVALS):
        bad.append(f"report_completeness {rc}")
    sections = json.loads((RUN / "formal_point_sections.json").read_text())
    if sorted(sections) != sorted(rp["required_point_sections"]):
        bad.append("formal_point_sections.json keys != required_point_sections")
    if len(sections) != N_SECTIONS:
        bad.append(f"{len(sections)} point sections != {N_SECTIONS}")
    later = set(rp["later_stage_point_sections"])
    if later & set(sections):
        bad.append(f"§23 sensitivity sections present in the freeze: {later & set(sections)}")
    notes.append(f"{len(sections)}/{N_SECTIONS} primary sections derived, "
                 f"{len(later)} §23 sensitivity sections excluded")

    with (RUN / "registered_intervals.csv").open(newline="") as fh:
        body = list(csv.DictReader(fh))
    if len(body) != N_INTERVALS:
        bad.append(f"registered_intervals.csv has {len(body)} rows != {N_INTERVALS}")
    fams: dict[str, int] = {}
    for row in body:
        fams[row["family"]] = fams.get(row["family"], 0) + 1
    if fams != CI_FAMILIES:
        bad.append(f"CI family decomposition {fams} != {CI_FAMILIES}")
    notes.append(f"{len(body)}/{N_INTERVALS} registered CIs, families {fams}")
    facts["ci_families"] = fams

    # ---- the eligibility ledger the execution-v3 fix closed ---------------
    cl = sections["candidate_ledger"]["payload"]
    if (cl["n_cells"], cl["n_zero_label"]) != (N_DECISIONS, N_ZERO_LABEL):
        bad.append(f"candidate_ledger {cl['n_cells']}/{cl['n_zero_label']}")
    if cl["n_suffix_checks"] != N_DECISIONS + N_ZERO_LABEL:
        bad.append(f"{cl['n_suffix_checks']} suffix checks != "
                   f"{N_DECISIONS + N_ZERO_LABEL}")
    ledger_rows = cl["cells"] + cl["zero_label"]
    if not all(r["suffix_property_ok"] for r in ledger_rows):
        bad.append("a decision violates the contiguous-suffix property")
    empty = [r for r in ledger_rows if r["n_eligible"] == 0]
    if not all(r["first_eligible_index"] is None for r in empty):
        bad.append("an empty eligible set records a first_eligible_index")
    notes.append(f"{cl['n_suffix_checks']} suffix checks pass, "
                 f"{len(empty)} with an empty eligible set recorded as None")
    facts["n_suffix_checks"] = cl["n_suffix_checks"]
    facts["n_empty_eligible"] = len(empty)

    reasons: dict[str, int] = {}
    for counts in sections["not_testable_reason_split"]["payload"]["by_site"].values():
        for reason, n in counts.items():
            reasons[reason] = reasons.get(reason, 0) + n
    if sum(reasons.values()) != N_DECISIONS:
        bad.append(f"reason split sums to {sum(reasons.values())} != {N_DECISIONS}")
    notes.append(f"NOT_TESTABLE/no-pass reason split sums to {N_DECISIONS}: {reasons}")
    facts["reason_split"] = reasons

    # ---- barrier ----------------------------------------------------------
    ev = completion["ledger_ordering"]["critical_event_seqs"]
    order = ["ALL_DECISIONS_FROZEN", "RELEASE_AUTHORITY_VERIFIED",
             "PHASE2_OPEN", "FIRST_TARGET_OUTCOME_READ"]
    seqs = [ev[k] for k in order]
    if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
        bad.append(f"barrier out of order: {dict(zip(order, seqs))}")
    oo = completion["outcome_ordering"]
    if not oo["ordering_ok"] or oo["target_outcomes_materialized_pre_phase2"]:
        bad.append(f"outcome ordering: {oo['ordering_ok']}, "
                   f"pre-phase2 {oo['target_outcomes_materialized_pre_phase2']}")
    if oo["first_outcome_row_materialized_seq"] <= ev["PHASE2_OPEN"]:
        bad.append("a target outcome row materialised before PHASE2_OPEN")
    notes.append("barrier " + " < ".join(f"{k} {v}" for k, v in zip(order, seqs)))
    facts["barrier"] = dict(zip(order, seqs))
    facts["target_outcome_rows_read"] = completion["target_outcome_rows_read"]

    n_events = completion["ledger_ordering"]["n_events"]
    with (RUN / "global_run_ledger.csv").open() as fh:
        n_lines = sum(1 for _ in fh) - 1
    if n_lines != n_events:
        bad.append(f"global_run_ledger.csv has {n_lines} rows != {n_events} events")
    notes.append(f"global ledger persisted: {n_lines} rows == n_events")

    # ---- checksum chain ---------------------------------------------------
    manifest = RUN / "RESULT_CHECKSUMS.sha256"
    entries = [ln.split("  ", 1) for ln in manifest.read_text().splitlines()
               if ln and not ln.startswith("#")]
    for digest, name in entries:
        actual = sha256(RUN / name)
        if actual != digest:
            bad.append(f"{name}: {actual} != recorded {digest}")
        if completion["artifact_checksums"].get(name) != digest:
            bad.append(f"{name}: completion artifact_checksums disagree")
    notes.append(f"RESULT_CHECKSUMS.sha256: {len(entries)}/{len(entries)} "
                 f"entries verify against the persisted bytes")

    manifest_sha = sha256(manifest)
    if completion["checksum_manifest_checksum"] != manifest_sha:
        bad.append(f"completion checksum_manifest_checksum "
                   f"{completion['checksum_manifest_checksum']} != {manifest_sha}")
    notes.append(f"completion -> manifest chain valid ({manifest_sha[:16]}…)")
    facts["checksum_manifest_checksum"] = manifest_sha

    em = json.loads((RUN / "evaluation_manifest.json").read_text())
    if em["evaluation_manifest_checksum"] != rp["evaluation_manifest_checksum"]:
        bad.append("evaluation manifest identity disagrees with report provenance")
    if em["n_evaluated"] != N_EVALUATIONS:
        bad.append(f"evaluation manifest n_evaluated {em['n_evaluated']}")
    if em["evaluation_root_identity"] != rp["evaluation_root_identity"]:
        bad.append("evaluation root identity disagrees with report provenance")
    notes.append(f"evaluation manifest {em['evaluation_manifest_checksum'][:16]}… "
                 f"over {em['n_evaluated']} evaluations, root identity "
                 f"{em['evaluation_root_identity'][:16]}…")
    facts["evaluation_manifest_checksum"] = em["evaluation_manifest_checksum"]
    facts["evaluation_root_identity"] = em["evaluation_root_identity"]
    facts["decision_manifest_checksum"] = rp["decision_manifest_checksum"]
    facts["split_registry_checksum"] = rp["split_registry_checksum"]
    facts["ci_registry_checksum"] = rp["ci_registry_checksum"]

    if bad:
        raise AssertionError("closure re-derivation FAILED:\n  " + "\n  ".join(bad))
    return notes, facts


def main() -> int:
    print("=" * 78)
    print("RQ2/S2 FINAL PROVENANCE FREEZE — checksums + mechanical re-derivation")
    print("=" * 78)

    notes, facts = recompute_headlines()
    for n in notes:
        print(f"  [ok] {n}")

    digest, n_files = content_digest(RUN)
    print(f"  [ok] authoritative root: {n_files} files, content digest "
          f"{digest[:16]}…")

    committed = [(name, sha256(RUN / name)) for name in COMMITTED]
    bulk_top = [(name, sha256(RUN / name)) for name in BULK_TOP]
    store = {
        "decisions/": facts["counts"]["decisions"],
        "evaluations/": facts["counts"]["evaluations"],
        "zero__*.json": facts["counts"]["zero_label"],
        "job_*__decision.json": sum(1 for _ in RUN.glob("job_*.json")),
    }

    OUT.mkdir(exist_ok=True)
    BINDING.write_text(json.dumps({
        "binding_version": "S2_RESULT_FREEZE_BINDING/1",
        "stage": "2.2 — authoritative Formal RQ2/S2 result freeze",
        "result_tag": "rq2-s2-results-v1",
        "run_id": RUN_ID,
        "authoritative_ledger_id": LEDGER_ID,
        "protocol": {"tag": PROTOCOL_TAG, "sha": PROTOCOL_SHA,
                     "version": PROTOCOL_VERSION},
        "execution": {"tag": EXECUTION_TAG, "sha": EXECUTION_SHA},
        "counts": {
            "zero_label_decisions": N_ZERO_LABEL,
            "phase1_decisions": N_DECISIONS,
            "phase2_evaluations": N_EVALUATIONS,
            "primary_point_sections": N_SECTIONS,
            "registered_point_sections": N_REGISTERED_SECTIONS,
            "registered_intervals": N_INTERVALS,
            "ci_families": facts["ci_families"],
            "per_n_eligible_sites": PER_N_SITES,
            "suffix_checks": facts["n_suffix_checks"],
            "empty_eligible_decisions": facts["n_empty_eligible"],
            "not_testable_reason_split": facts["reason_split"],
        },
        "barrier": facts["barrier"],
        "target_outcome_rows_read": facts["target_outcome_rows_read"],
        "identities": {
            "evaluation_manifest_checksum": facts["evaluation_manifest_checksum"],
            "evaluation_root_identity": facts["evaluation_root_identity"],
            "decision_manifest_checksum": facts["decision_manifest_checksum"],
            "split_registry_checksum": facts["split_registry_checksum"],
            "ci_registry_checksum": facts["ci_registry_checksum"],
            "checksum_manifest_checksum": facts["checksum_manifest_checksum"],
            "formal_run_completion_sha256": sha256(RUN / "formal_run_completion.json"),
            "result_checksums_sha256": sha256(RUN / "RESULT_CHECKSUMS.sha256"),
            "authoritative_root_content_digest": digest,
            "authoritative_root_n_files": n_files,
        },
        "independent_result_closure": {
            "reviewer": "Codex",
            "verdict": "PASS",
            "critical_issues": 0,
            "non_critical_issues": 0,
            "closure_decisions": "30/30 YES",
        },
        "superseded_failed_run": {
            "run_id": "2f57a960b57649ee85d5730a05a186d9",
            "execution_tag": "rq2-s2-execution-v2",
            "status": "FAILED_INCOMPLETE",
            "archive": "results/failed_runs/"
                       "rq2_s2_execution_v2_run_2f57a960b57649ee85d5730a05a186d9",
            "note": "forensic evidence only; never an input to this freeze",
        },
    }, indent=2, sort_keys=True) + "\n")

    # The ledger is written last so that it can cover the binding it describes.
    lines = [
        "# RQ2/S2 formal result checksums (SHA256)",
        "#",
        "# Stage 2.2 final provenance freeze. Verify with:",
        "#     cd results/rq2_s2_freeze && sha256sum -c RESULT_CHECKSUMS.sha256",
        "#",
        f"# Authoritative Formal run   {RUN_ID}",
        f"# Authoritative ledger       {LEDGER_ID}",
        f"# Protocol                   {PROTOCOL_TAG} {PROTOCOL_SHA}",
        f"# Execution                  {EXECUTION_TAG} {EXECUTION_SHA}",
        "#",
        "# This freeze writes nothing inside the authoritative tree. Paths below",
        "# are relative to this directory so the in-place artifacts still verify.",
        "#",
        "# --- authoritative closure artifacts (committed) ---",
    ]
    for name, dig in committed:
        lines.append(f"{dig}  ../rq2_s2/{name}")
    lines += ["#", "# --- freeze artifacts (committed) ---"]
    for path in sorted(p for p in OUT.iterdir()
                       if p.is_file() and p.name != LEDGER.name):
        lines.append(f"{sha256(path)}  {path.name}")
    lines += ["#",
              "# --- NOT COMMITTED: bulk authoritative artifacts, recorded for audit ---",
              f"# authoritative root: {n_files} files, "
              f"{sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()) / 1e6:.1f} MB",
              f"# content digest, sha256 over sorted (relpath, sha256(bytes)): {digest}",
              "#"]
    for name, dig in bulk_top:
        lines.append(f"# {dig}  ../rq2_s2/{name}")
    lines += ["#", "# per-cell stores, bound by the manifest checksums above:"]
    for name, count in store.items():
        lines.append(f"#   {name:24s} {count} artifacts")
    LEDGER.write_text("\n".join(lines) + "\n")

    print(f"\n[ledger]  {LEDGER}")
    print(f"[ledger]  {len(committed)} authoritative closure artifacts committed, "
          f"{len(bulk_top)} bulk artifacts + {n_files} tree files recorded only")
    print(f"[binding] {BINDING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
