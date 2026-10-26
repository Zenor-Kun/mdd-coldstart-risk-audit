#!/usr/bin/env python3
"""RQ1/S0 final provenance freeze — checksums and mechanical re-derivation.

Documentation / provenance only. This script **reads** the frozen S0 outputs
and writes exactly one new file, `results/rq1_s0/RESULT_CHECKSUMS.sha256`. It
runs no model, refits nothing, and recomputes no scientific quantity from the
feature matrix — it re-derives the headline aggregates *arithmetically from the
published per-site CSVs* to prove the report's numbers follow from the
artifacts, and fails loudly if any of them does not.

Usage:
    python3 scripts/freeze_rq1_s0_provenance.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import s0_evaluation as s0e  # noqa: E402
import site_bootstrap as sb  # noqa: E402
from contracts import FROZEN_SITES  # noqa: E402

OUT = ROOT / "results" / "rq1_s0"
LEDGER = OUT / "RESULT_CHECKSUMS.sha256"
METHODS = sb.CONFIDENCE_METHODS

#: Formal result artifacts. Small, authoritative, and intended for Git.
FORMAL_SUFFIXES = (".csv", ".json", ".yaml", ".md")
#: Recorded by checksum but deliberately NOT committed (bulk derived data).
BULK = ("cache",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recompute_headlines() -> list[str]:
    """Re-derive every headline aggregate from the published per-site CSVs."""
    notes, bad = [], []
    order = list(FROZEN_SITES)

    aurc = pd.read_csv(OUT / "per_site_aurc.csv").set_index(["method", "held_out_site"])
    summary = pd.read_csv(OUT / "aurc_summary.csv").set_index("method")
    iv = pd.read_csv(OUT / "site_bootstrap_intervals.csv").set_index(
        ["quantity", "method_or_contrast"])
    rc = pd.read_csv(OUT / "risk_coverage_all_q.csv")
    counts = np.load(OUT / "site_bootstrap_draws.npy")

    if not (counts.shape == (sb.SITE_BOOTSTRAP_B, 23)
            and (counts.sum(axis=1) == 23).all()
            and np.array_equal(counts, sb.site_bootstrap_counts(23))):
        bad.append("site_bootstrap_draws.npy does not reproduce")
    notes.append(f"site bootstrap draws reproduce bit-identically: "
                 f"{counts.shape}, row sums 23")

    for m in METHODS:
        a = np.array([aurc.loc[(m, s), "AURC_s"] for s in order], dtype=float)
        n = np.array([aurc.loc[(m, s), "n_s"] for s in order], dtype=float)
        if n.sum() != 2232:
            bad.append(f"{m}: sum n_s = {n.sum()} != 2232")
        for est, val in (("patient_weighted_aurc", s0e.patient_weighted_aurc(a, n)),
                         ("site_balanced_aurc", s0e.site_balanced_aurc(a))):
            for src, ref in ((f"aurc_summary.csv:{est}", summary.loc[m, est]),
                             (f"site_bootstrap_intervals.csv:{est}",
                              iv.loc[(est, m), "point_estimate"])):
                if abs(val - float(ref)) > 1e-12:
                    bad.append(f"{m} {src}: {ref} != recomputed {val}")
            notes.append(f"{m:12s} {est:22s} = {val:.6f} recomputed from "
                         f"per_site_aurc.csv")

    # paired contrasts: A - B on the common estimable set, same draws
    for estimand, A, B in sb.PAIRED_AURC_CONTRASTS:
        w = (np.array([aurc.loc[(A, s), "n_s"] for s in order], dtype=float)
             if estimand == "patient_weighted_aurc" else None)
        va = np.array([aurc.loc[(A, s), "AURC_s"] for s in order], dtype=float)
        vb = np.array([aurc.loc[(B, s), "AURC_s"] for s in order], dtype=float)
        got = sb.paired_contrast(va, vb, counts, weights=w)
        ref = iv.loc[("paired_aurc_contrast", str((estimand, A, B)))]
        for k in ("point_estimate", "ci_low", "ci_high"):
            if abs(float(got[k]) - float(ref[k])) > 1e-12:
                bad.append(f"contrast {estimand} {A}-{B} {k}: "
                           f"{ref[k]} != recomputed {got[k]}")
        notes.append(f"contrast {estimand[:7]:7s} {A[:4]}-{B[:4]} = "
                     f"{got['point_estimate']:+.6f} "
                     f"[{got['ci_low']:+.6f}, {got['ci_high']:+.6f}] recomputed")

    # operating points: site-balanced coverage / BER / violation rate
    for op, q in s0e.OPERATING_POINTS.items():
        for m in METHODS:
            g = rc[(rc["method"] == m) & (rc["q"].astype(str) == str(q))] \
                .set_index("held_out_site")
            arrays = {
                "site_balanced_coverage": np.array(
                    [float(g.loc[s, "coverage_target"]) for s in order]),
                "site_balanced_selective_ber": np.array(
                    [s0e.as_float(g.loc[s, "BER_sel_target"]) for s in order]),
                "descriptive_violation_rate": np.array(
                    [s0e.as_float(g.loc[s, "realized_risk_violation__descriptive"])
                     for s in order]),
            }
            for name, arr in arrays.items():
                got = sb.bootstrap_summary(arr, counts)
                ref = iv.loc[(f"{op}_{name}", m)]
                for k in ("point_estimate", "ci_low", "ci_high",
                          "n_estimable_sites", "valid_bootstrap_replicates"):
                    if abs(float(got[k]) - float(ref[k])) > 1e-12:
                        bad.append(f"{op} {m} {name} {k}: "
                                   f"{ref[k]} != recomputed {got[k]}")
            notes.append(f"{op} {m:12s} coverage/BER/violation reproduce from "
                         f"risk_coverage_all_q.csv")

    led = pd.read_csv(OUT / "leakage_assertions.csv")
    if not (len(led) == 41 and (led["status"] == "PASS").all()):
        bad.append(f"leakage ledger: {len(led)} rows, "
                   f"{(led['status'] == 'PASS').sum()} PASS")
    notes.append(f"leakage_assertions.csv: {len(led)} rows, "
                 f"{(led['status'] == 'PASS').sum()} PASS, "
                 f"{(led['status'] != 'PASS').sum()} FAIL")

    report = (OUT / "RQ1_S0_REPORT.md").read_text()
    stated = [ln for ln in report.splitlines() if "leakage assertions PASS" in ln]
    if not stated or "41" not in stated[0]:
        bad.append(f"report assertion count line: {stated}")
    notes.append(f"report states: {stated[0].strip() if stated else 'MISSING'}")

    if len(iv) != sb.N_REQUIRED_INTERVALS:
        bad.append(f"{len(iv)} intervals != {sb.N_REQUIRED_INTERVALS}")
    notes.append(f"site_bootstrap_intervals.csv: {len(iv)}/"
                 f"{sb.N_REQUIRED_INTERVALS} required intervals")

    if bad:
        raise AssertionError("headline re-derivation FAILED:\n  "
                             + "\n  ".join(bad))
    return notes


def main() -> int:
    print("=" * 78)
    print("RQ1/S0 FINAL PROVENANCE FREEZE — checksums + mechanical re-derivation")
    print("=" * 78)

    notes = recompute_headlines()
    for n in notes:
        print(f"  [ok] {n}")

    formal = sorted(p for p in OUT.iterdir()
                    if p.is_file() and p.suffix in FORMAL_SUFFIXES
                    and p.name != LEDGER.name and not p.name.startswith("_"))
    other = sorted(p for p in OUT.iterdir()
                   if p.is_file() and p not in formal and p.name != LEDGER.name)
    bulk = sorted(p for d in BULK for p in (OUT / d).rglob("*") if p.is_file())

    lines = [
        "# RQ1/S0 formal result checksums (SHA256)",
        "#",
        "# Stage 1.1 final provenance freeze. Verify with:",
        "#     cd results/rq1_s0 && sha256sum -c RESULT_CHECKSUMS.sha256",
        "# (bulk entries below the marker are recorded but intentionally NOT",
        "#  committed to Git -- they are large regenerable derived data.)",
        "#",
        "# --- formal result artifacts (committed) ---",
    ]
    for p in formal:
        lines.append(f"{sha256(p)}  {p.name}")
    if other:
        lines += ["#", "# --- other committed-size artifacts ---"]
        for p in other:
            lines.append(f"{sha256(p)}  {p.name}")
    lines += ["#",
              "# --- NOT COMMITTED: bulk derived cache, recorded for audit only ---",
              f"# {len(bulk)} files, "
              f"{sum(p.stat().st_size for p in bulk) / 1e6:.1f} MB total"]
    for p in bulk:
        lines.append(f"# {sha256(p)}  {p.relative_to(OUT)}")

    LEDGER.write_text("\n".join(lines) + "\n")
    print(f"\n[ledger] {LEDGER}")
    print(f"[ledger] {len(formal)} formal artifacts + {len(other)} other "
          f"checksummed and committed")
    print(f"[ledger] {len(bulk)} bulk cache files checksummed as comments, "
          f"NOT committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
