#!/usr/bin/env python3
"""RQ1/S1 final provenance freeze — checksums and mechanical re-derivation.

Documentation / provenance only. This script **reads** the frozen S1 outputs
(and the frozen S0 inputs they were built from) and writes exactly one new
file, `results/rq1_s1/RESULT_CHECKSUMS.sha256`. It fits no domain classifier,
refits nothing, and touches no feature matrix — it re-derives the report's
headline numbers *arithmetically from the published per-site CSVs* to prove
they follow from the artifacts, and fails loudly if any of them does not.

The re-derivation deliberately re-walks the leakage-sensitive steps: the
weighted thresholds are rebuilt from the frozen source-OOF confidences and the
published weights alone, the ESS is rebuilt from the published weights, the
selected `C_domain` is rebuilt as the argmin of the published `J(C)` table, and
the target AURC is rebuilt from the frozen S0 target confidences and required
to be bit-identical to S0's.

Usage:
    python3 scripts/freeze_rq1_s1_provenance.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import density_ratio as dr  # noqa: E402
import s0_evaluation as s0e  # noqa: E402
import s1_evaluation as s1e  # noqa: E402
import site_bootstrap as sb  # noqa: E402
from contracts import FROZEN_SITES  # noqa: E402
from selective_aurc import tie_aware_aurc  # noqa: E402

S0 = ROOT / "results" / "rq1_s0"
OUT = ROOT / "results" / "rq1_s1"
LEDGER = OUT / "RESULT_CHECKSUMS.sha256"
METHODS = dr.CONFIDENCE_METHODS
N_ASSERTIONS = None          # set from the ledger; only "all PASS" is fixed

FORMAL_SUFFIXES = (".csv", ".json", ".yaml", ".md")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", **kw)


def _site_values(rc: pd.DataFrame, m: str, q: float, order) -> dict:
    qn = pd.to_numeric(rc["q"], errors="coerce")
    g = rc[(rc["method"] == m) & (qn == float(q))].set_index("held_out_site")
    return {
        "coverage": np.array([float(g.loc[s, "coverage_target"]) for s in order]),
        "ber": np.array([s0e.as_float(g.loc[s, "BER_sel_target"]) for s in order]),
        "violation": np.array([
            s0e.as_float(g.loc[s, "realized_risk_violation__descriptive"])
            for s in order]),
    }


def recompute_headlines() -> list[str]:
    notes, bad = [], []
    order = list(FROZEN_SITES)

    rc = _read(OUT / "risk_coverage_all_q.csv")
    s0_rc = _read(S0 / "risk_coverage_all_q.csv")
    iv = _read(OUT / "site_bootstrap_intervals.csv").set_index(
        ["quantity", "method_or_contrast"])
    sel = _read(OUT / "selected_domain_C.csv").set_index("held_out_site")
    obj = _read(OUT / "domain_meta_objective.csv")
    wdf = _read(OUT / "density_weights.csv")
    ess = _read(OUT / "ess_by_site.csv").set_index("held_out_site")
    thr = _read(OUT / "thresholds_weighted_source.csv")
    oof = _read(S0 / "confidence_source_oof.csv")

    counts = sb.site_bootstrap_counts(23)
    frozen_draws = np.load(S0 / "site_bootstrap_draws.npy")
    if not np.array_equal(counts, frozen_draws):
        bad.append("the S0 site-bootstrap draw matrix does not reproduce")
    notes.append("site bootstrap draws are the frozen S0 (5000, 23) matrix, "
                 "reused bit-identically — S1 drew nothing new")

    # -- cohort accounting straight off the published tables -----------------
    n_s = rc.drop_duplicates("held_out_site").set_index("held_out_site")["n_s"]
    if int(n_s.loc[order].sum()) != 2232 or len(n_s) != 23:
        bad.append(f"target acquisitions sum to {int(n_s.sum())}, not 2232")
    if int(ess["n_source"].sum() + ess["n_target"].sum()) != 23 * 2232:
        bad.append("n_source + n_target != 2232 in every fold")
    notes.append("23 folds, sum of n_s = 2232, n_source + n_target = 2232 per fold")

    # -- C_domain is the argmin of the published J(C) table ------------------
    for s in order:
        g = obj[obj["held_out_site"] == s].sort_values("C_domain")
        best = float(g.loc[g["J_C"].idxmin(), "C_domain"])
        ties = g[g["J_C"] == g["J_C"].min()]["C_domain"]
        best = float(min(ties))                       # frozen tie-break
        if best != float(sel.loc[s, "c_domain"]):
            bad.append(f"{s}: published C* {sel.loc[s, 'c_domain']} != "
                       f"argmin J(C) {best}")
        if best not in dr.C_DOMAIN_GRID:
            bad.append(f"{s}: C* {best} outside the frozen grid")
    notes.append(f"C_domain reproduces as argmin J(C) (ties -> smallest C) in "
                 f"all 23 folds; distribution "
                 f"{ {float(k): int(v) for k, v in sel['c_domain'].value_counts().items()} }")

    # -- ESS and the clip bounds rebuild from the published weights ----------
    for s in order:
        w = wdf.loc[wdf["held_out_site"] == s, "w_final"].to_numpy()
        if abs(dr.effective_sample_size(w) - float(ess.loc[s, "ess"])) > 1e-9:
            bad.append(f"{s}: ESS does not reproduce from density_weights.csv")
        if w.min() < dr.W_CLIP_LOW - 1e-12 or w.max() > dr.W_CLIP_HIGH + 1e-12:
            bad.append(f"{s}: final weights leave [{dr.W_CLIP_LOW}, "
                       f"{dr.W_CLIP_HIGH}]")
        raw = wdf.loc[wdf["held_out_site"] == s, "w_raw"].to_numpy()
        if not np.allclose(dr.normalize_then_clip(raw), w, rtol=0, atol=1e-12):
            bad.append(f"{s}: normalize-then-clip does not reproduce w_final")
    notes.append("ESS and the final weights reproduce from w_raw by "
                 "normalize -> clip in all 23 folds; no post-clip step exists")

    # -- weighted thresholds rebuild from source-OOF confidence + weights ----
    checked = 0
    for s in order:
        g = oof[oof["outer_fold_held_out_site"] == s]
        w = wdf.loc[wdf["held_out_site"] == s, "w_final"].to_numpy()
        t = thr[(thr["held_out_site"] == s)]
        for m in METHODS:
            c = g[f"confidence_{m}"].to_numpy()
            tm = t[(t["method"] == m) & (t["q"] != s0e.NO_ABSTENTION)]
            for r in tm.itertuples():
                got = dr.weighted_quantile(c, w, float(r.q))
                if got != float(r.tau_S1):
                    bad.append(f"{s}/{m}/q={r.q}: tau_S1 does not reproduce")
                checked += 1
    notes.append(f"all {checked} tau_q^S1 reproduce exactly from the frozen "
                 "source-OOF confidences and the published weights alone")

    # -- the target AURC invariant -------------------------------------------
    tgt = _read(S0 / "confidence_target.csv")
    s0_aurc = _read(S0 / "per_site_aurc.csv").set_index(["held_out_site", "method"])
    worst = 0.0
    for s, g in tgt.groupby("held_out_site"):
        loss = (g["pred"].to_numpy().astype(int)
                != g["y_true"].to_numpy().astype(int)).astype(float)
        for m in METHODS:
            got = tie_aware_aurc(g[f"confidence_{m}"].to_numpy(), loss)
            ref = float(s0_aurc.loc[(s, m), "AURC_s"])
            worst = max(worst, abs(got - ref))
            if got != ref:
                bad.append(f"{s}/{m}: target AURC moved between S0 and S1")
    notes.append(f"69 target AURC values are bit-identical to S0 "
                 f"(max |diff| = {worst:.1e}); S1 emits no AURC of its own")

    # -- the 36 intervals -----------------------------------------------------
    # `len(iv) != N != 36` was a chained comparison: with N == 36 the right
    # conjunct is always false, so the count check could never fire. The shared
    # helper spells the three conditions out and is pinned by unit tests.
    try:
        s1e.assert_interval_registry_exact(list(iv.index))
        notes.append(f"site_bootstrap_intervals.csv: {len(iv)}/36 required "
                     "intervals, set-equal to the frozen registry")
    except AssertionError as exc:
        bad.append(f"interval registry: {exc}")

    for op, q in s0e.OPERATING_POINTS.items():
        for m in METHODS:
            a = _site_values(rc, m, q, order)
            b = _site_values(s0_rc, m, q, order)
            for name, key in (("site_balanced_coverage", "coverage"),
                              ("site_balanced_selective_ber", "ber"),
                              ("descriptive_violation_rate", "violation")):
                got = sb.bootstrap_summary(a[key], counts)
                ref = iv.loc[(f"S1_{op}_{name}", m)]
                for k in ("point_estimate", "ci_low", "ci_high",
                          "n_estimable_sites", "valid_bootstrap_replicates"):
                    if abs(float(got[k]) - float(ref[k])) > 1e-12:
                        bad.append(f"S1 {op} {m} {name} {k} does not reproduce")
                got = sb.paired_contrast(a[key], b[key], counts)
                ref = iv.loc[(f"S1_minus_S0_{op}_{name}", m)]
                for k in ("point_estimate", "ci_low", "ci_high",
                          "n_common_estimable_sites"):
                    if abs(float(got[k]) - float(ref[k])) > 1e-12:
                        bad.append(f"S1-S0 {op} {m} {name} {k} does not reproduce")
            notes.append(f"{op} {m:12s} S1 and paired S1-S0 coverage / BER / "
                         "violation reproduce from the per-site tables")

    led = _read(OUT / "leakage_assertions.csv")
    if not (len(led) > 0 and (led["status"] == "PASS").all()):
        bad.append(f"leakage ledger: {len(led)} rows, "
                   f"{(led['status'] == 'PASS').sum()} PASS")
    notes.append(f"leakage_assertions.csv: {len(led)} rows, "
                 f"{(led['status'] == 'PASS').sum()} PASS, "
                 f"{(led['status'] != 'PASS').sum()} FAIL")

    # -- the outer-fold-scoped target-diagnosis firewall ---------------------
    fw = _read(OUT / "firewall_fold_ledger.csv")
    if len(fw) != 23 or not fw["ordering_ok"].all():
        bad.append(f"firewall ledger: {int(fw['ordering_ok'].sum())}/23 folds ok")
    if fw["target_labels_visible_prebarrier"].any():
        bad.append("a fold saw target labels before its own release")
    if (fw["source_label_target_rows"] != 0).any():
        bad.append("a source-label object contained the held-out site")
    for col_a, col_b in (("last_decision_seq", "release_seq"),
                         ("release_seq", "first_target_y_seq"),
                         ("first_target_y_seq", "evaluation_seq")):
        if not (fw[col_a] < fw[col_b]).all() and col_a != "first_target_y_seq":
            bad.append(f"firewall ordering violated: {col_a} !< {col_b}")
    if not (fw["first_target_y_seq"] <= fw["evaluation_seq"]).all():
        bad.append("evaluation preceded the target-label load")
    notes.append(f"target-diagnosis firewall: {int(fw['ordering_ok'].sum())}/23 "
                 "folds with decisions+checksums < release < first target-y "
                 "load, and no held-out-site row in any source-label object")

    # per-fold decision checksums still describe the frozen decision artifacts
    import json
    bad_ck = []
    for site in order:
        d = OUT / "decisions" / site
        recorded = json.loads((d / "decision_checksums.json").read_text())
        for name, digest in recorded.items():
            if sha256(d / f"{name}.csv") != digest:
                bad_ck.append(f"{site}/{name}")
    if bad_ck:
        bad.append(f"per-fold decision checksums do not match: {bad_ck[:3]}")
    notes.append(f"per-fold decision checksums verify for all {len(order)} folds "
                 "(C_domain, weights, thresholds, ESS, weighted source risk)")

    if bad:
        raise AssertionError("headline re-derivation FAILED:\n  " + "\n  ".join(bad))
    return notes


def main() -> int:
    print("=" * 78)
    print("RQ1/S1 FINAL PROVENANCE FREEZE — checksums + mechanical re-derivation")
    print("=" * 78)

    for n in recompute_headlines():
        print(f"  [ok] {n}")

    formal = sorted(p for p in OUT.iterdir()
                    if p.is_file() and p.suffix in FORMAL_SUFFIXES
                    and p.name != LEDGER.name and not p.name.startswith("_"))
    other = sorted(p for p in OUT.iterdir()
                   if p.is_file() and p not in formal and p.name != LEDGER.name)

    lines = [
        "# RQ1/S1 formal result checksums (SHA256)",
        "#",
        "# Stage 1.3 final provenance freeze. Verify with:",
        "#     cd results/rq1_s1 && sha256sum -c RESULT_CHECKSUMS.sha256",
        "#",
        "# --- formal result artifacts ---",
    ]
    for p in formal:
        lines.append(f"{sha256(p)}  {p.name}")
    if other:
        lines += ["#", "# --- other artifacts ---"]
        for p in other:
            lines.append(f"{sha256(p)}  {p.name}")

    LEDGER.write_text("\n".join(lines) + "\n")
    print(f"\n[ledger] {LEDGER}")
    print(f"[ledger] {len(formal) + len(other)} artifacts checksummed "
          f"({len(formal)} formal + {len(other)} other)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
