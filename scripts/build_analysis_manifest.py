#!/usr/bin/env python3
"""Stage 0.8 -- Acquisition-level de-duplication contract freeze (Strategy D).

Builds the acquisition-level analysis manifest that every downstream stage
must consume. The statistical unit stops being "a phenotype row" and becomes
"one independent fMRI acquisition": the 41 duplicate connected components
frozen in Stage 0.7 are collapsed to a single observational unit each, and the
2 components whose two records carry contradictory diagnoses are removed
outright.

DETERMINISTIC IDENTITY / QC ONLY -- NO MODELLING BY CONSTRUCTION. This script
must never fit a classifier, run LOSO, run PCA or feature selection, permute
labels, compute BAcc/AUROC, implement S0/S1/S2, or write anything under data/.
Nothing in it consults a downstream performance quantity; every decision is a
function of identity, site, and the frozen label-blind FC QC rule.

Duplicate identity is NOT re-derived here. It is read from the frozen Stage 0.7
component table, which rests on label-independent upstream imaging evidence
(identical scan length, bit-identical realignment parameters, identical
FD_Jenkinson traces), and is bound to that stage by exact unordered-pair set
equality against duplicate_pair_reconciliation.csv. No similarity cutoff is
read, recomputed or re-tuned.

Each collapsed acquisition also names one frozen feature representative,
`feature_source_id` -- the single original record whose derivative supplies the
acquisition's FC vector. The two reprocessed copies in a component are
near-identical but not bit-identical, so the choice is real and is made here,
label- and performance-blind, before any model exists.

Usage:
    python3 scripts/build_analysis_manifest.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finalize_data_contract import (  # noqa: E402
    PKG,
    assert_read_only,
    eligibility,
    load_phenotype,
    site_sort,
)
from validate_fc_input import (  # noqa: E402
    N_EDGES,
    PRIMARY_BRANCH,
    fc_qc,
    fc_vector,
    load_aal116,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "data_contract"

# ---------------------------------------------------------------------------
# FROZEN STAGE 0.8 CONSTANTS
# ---------------------------------------------------------------------------
# Strategy D: one independent fMRI acquisition = one observational unit.
STRATEGY = "D_acquisition_level_collapse"

# Frozen Stage 0.7 evidence. Read, never recomputed: duplicate identity rests
# on upstream imaging evidence that is already frozen, and re-deriving it here
# would reopen a settled decision.
COMPONENTS_CSV = OUT / "duplicate_component_summary.csv"
PAIRS_CSV = OUT / "duplicate_pair_reconciliation.csv"
MEAN_FD_CSV = OUT / "mean_fd_per_subject.csv"

# Expected Stage 0.7 structure, asserted rather than assumed.
EXPECT_COMPONENTS = 41
EXPECT_MEMBERS = 82
EXPECT_COMPONENT_SIZE = 2
EXPECT_PAIRS = 41

# Frozen Stage 0.8 cohort. Asserted so that a later edit to this script cannot
# silently move the analysed cohort; changing these is a protocol decision, not
# a code change.
EXPECT_INCLUDED = 2232
EXPECT_INCLUDED_MDD = 1182
EXPECT_INCLUDED_HC = 1050
EXPECT_ELIGIBLE_SITES = 23

# Manifest columns that carry identity, site or label. None of them may ever
# enter a feature matrix X. Declared here so the Gate 1 loader has one
# authoritative list to exclude; enforcing it is that loader's job (§7.27-33).
FORBIDDEN_IN_X = [
    "canonical_id", "feature_source_id", "source_records",
    "original_component_id", "site", "diagnosis",
]

# The consortium-documented duplicate of S14. It has no phenotype row, so it
# carries no label and never enters any diagnostic experiment.
S4_SITE = "S4"

# Core demographics, resolved under the Stage 0.8 §4 rule.
DEMOGRAPHIC_VARS = ["Age", "Sex"]
# Renamed to stable snake_case keys in the manifest.
RENAME = {
    "Age": "age",
    "Sex": "sex",
    "Education (years)": "education_years",
    "If first episode?": "first_episode",
    "On medication?": "on_medication",
    "Illness duration (months)": "illness_duration_months",
    "HAMD": "hamd",
    "HAMA": "hama",
}
# Two members of a collapsed acquisition agree on the mean FD of the shared
# scan to within floating-point noise (Stage 0.7: max 1.4e-9). Anything larger
# would mean the component is not one acquisition.
FD_TOL = 1e-6


# ---------------------------------------------------------------------------
# frozen resolution rule for a collapsed acquisition's phenotype value
# ---------------------------------------------------------------------------
def resolve_value(va, vb) -> tuple[object, str]:
    """Resolve one phenotype variable across the two records of a component.

    Frozen Stage 0.8 rule (§4 for demographics, §5 for clinical variables):

      both present and equal   -> keep the value                 ("agree")
      both present and differ  -> AMBIGUOUS, value set missing   ("conflict")
      exactly one present      -> AMBIGUOUS, value set missing   ("single_source")
      neither present          -> missing, not ambiguous         ("both_missing")

    The "single_source" branch is deliberately conservative. At least 7 of the
    39 diagnosis-concordant components demonstrably describe two different
    people's demographics, so a value carried by only one of the two records
    cannot be attributed to the acquisition. Choosing that record -- or the
    canonical record, or the official-list record -- would be exactly the
    arbitrary pick this stage forbids.
    """
    na, nb = pd.notna(va), pd.notna(vb)
    if na and nb:
        return (va, "agree") if va == vb else (np.nan, "conflict")
    if na or nb:
        return np.nan, "single_source"
    return np.nan, "both_missing"


AMBIGUOUS_STATUSES = {"conflict", "single_source"}


def canonical_key(members: list[str]) -> str:
    """Deterministic, phenotype-blind and performance-blind identity rule.

    canonical_id = lexicographically smallest original subject_id. All 41
    components are within-site and running IDs are zero-padded, so this is a
    stable acquisition-level key.

    It is TWO things, and the second must not be understated: it is the
    manifest's primary key, AND -- via feature_representative() below, which
    returns the same ID by construction -- the pre-specified raw derivative
    that supplies the acquisition's FC vector. It carries NO claim that the
    chosen record is the more correct description of the person scanned: the
    two records' phenotypes are reconciled by resolve_value(), never inherited
    from the canonical row.
    """
    return min(members)


def feature_representative(members: list[str]) -> str:
    """Frozen representative-selection rule: which record supplies the features.

    The two members of a diagnosis-concordant component are the same raw
    acquisition reprocessed twice, so their FC vectors are near-identical but
    NOT bit-identical (Stage 0.7: fc_r 0.918 .. 0.99998). Exactly one of them
    must therefore be named, before any model exists, as the acquisition's
    feature source:

        feature_source_id = lexicographically smallest original subject_id
                          = canonical_id

    The rule is label- and performance-independent by construction:

      - diagnosis is used only to decide whether the component is RETAINED
        (concordant) or EXCLUDED (conflict) -- never to pick the member;
      - age, sex and clinical variables are not consulted;
      - FC magnitude, FC similarity, motion quality, phenotype completeness,
        official-list membership, BAcc and AUROC are not consulted.

    Forbidden alternatives: averaging the two FC vectors, taking the member
    with higher pair correlation, the member with lower motion, or the member
    with the more complete phenotype. It asserts nothing about which
    reprocessing run is intrinsically more correct.
    """
    return min(members)


# ---------------------------------------------------------------------------
# Stage 0.7 evidence
# ---------------------------------------------------------------------------
def load_components() -> pd.DataFrame:
    """Load the frozen Stage 0.7 components, bound to the Stage 0.7 pair table.

    Structural checks (count 41, size 2, within-site) are NOT sufficient
    provenance: a table with 41 different within-site pairs would satisfy all
    of them. The binding check is exact set equality of the unordered member
    pairs against duplicate_pair_reconciliation.csv -- the table that carries
    the label-independent imaging evidence Stage 0.7 adjudicated. A missing,
    extra or substituted pair is a hard failure and no manifest is produced.
    """
    comp = pd.read_csv(COMPONENTS_CSV)
    comp["members"] = comp["members"].str.split(";")
    assert len(comp) == EXPECT_COMPONENTS, "Stage 0.7 component count changed"
    assert (comp["size"] == EXPECT_COMPONENT_SIZE).all(), "component size != 2"
    assert comp["members"].map(len).eq(EXPECT_COMPONENT_SIZE).all()
    flat = [m for ms in comp["members"] for m in ms]
    assert len(flat) == EXPECT_MEMBERS, "member count changed"
    assert len(set(flat)) == EXPECT_MEMBERS, "a subject appears in two components"
    # Every component is within-site; a cross-site component would break the
    # site-level protocol and must stop the stage.
    for _, r in comp.iterrows():
        sites = {m.split("-")[0] for m in r["members"]}
        assert sites == {r["site"]}, f"component {r['component_id']} spans sites"

    # --- exact provenance binding to the Stage 0.7 pair table --------------
    pairs = pd.read_csv(PAIRS_CSV)
    assert len(pairs) == EXPECT_PAIRS, \
        f"Stage 0.7 pair table has {len(pairs)} rows, expected {EXPECT_PAIRS}"
    stage_0_7_pairs = {tuple(sorted((a, b)))
                       for a, b in zip(pairs.subject_a, pairs.subject_b)}
    current_pairs = {tuple(sorted(ms)) for ms in comp["members"]}
    missing = stage_0_7_pairs - current_pairs
    extra = current_pairs - stage_0_7_pairs
    assert len(stage_0_7_pairs) == EXPECT_PAIRS, "Stage 0.7 pair table has a repeat"
    assert not missing, f"{len(missing)} Stage 0.7 pair(s) absent from the components"
    assert not extra, f"{len(extra)} component pair(s) not adjudicated in Stage 0.7"
    assert current_pairs == stage_0_7_pairs, "component pair set != Stage 0.7 pair set"
    # Diagnosis conflict must also agree between the two frozen tables.
    conflict_0_7 = {tuple(sorted((r.subject_a, r.subject_b)))
                    for r in pairs.itertuples() if not r.diagnosis_compatible}
    conflict_now = {tuple(sorted(r["members"])) for _, r in comp.iterrows()
                    if bool(r["diagnosis_conflict"])}
    assert conflict_now == conflict_0_7, \
        "diagnosis-conflict components disagree with the Stage 0.7 pair table"
    print(f"  [provenance] component pair set == Stage 0.7 pair set "
          f"({len(current_pairs)} unordered pairs, 0 missing, 0 extra, "
          f"{len(conflict_0_7)} conflict)")
    return comp


# ---------------------------------------------------------------------------
# acquisition assembly
# ---------------------------------------------------------------------------
def build_acquisitions(phen: pd.DataFrame, comp: pd.DataFrame,
                       value_cols: list[str],
                       passing: set[str]) -> tuple[pd.DataFrame, pd.DataFrame,
                                                   pd.DataFrame]:
    """Collapse phenotype records to acquisitions. Returns (acq, resolution, ambi).

    Singleton records pass through unchanged. Component members are collapsed
    into one row whose phenotype is reconciled variable by variable.
    """
    p = phen.set_index("ID")
    member_of = {m: int(r.component_id) for _, r in comp.iterrows() for m in r["members"]}

    rows, resolution, ambi_events = [], [], []

    # --- singletons -------------------------------------------------------
    for sid, r in p.iterrows():
        if sid in member_of:
            continue
        rec = {
            "canonical_id": sid,
            # A singleton record is its own feature source; the derivative
            # file read downstream is ROISignals_<feature_source_id>.mat.
            "feature_source_id": sid,
            "n_source_records": 1,
            "source_records": sid,
            "component_id": pd.NA,
            "duplicate_component": False,
            "site": r["site"],
            "diagnosis": int(r["dx"]),
            "diagnosis_conflict": False,
        }
        for c in value_cols:
            rec[RENAME.get(c, c)] = r[c]
            rec[f"{RENAME.get(c, c)}__ambiguous"] = False
        rec["age_recorded"] = bool(pd.notna(r["Age"]))
        rec["sex_recorded"] = bool(pd.notna(r["Sex"]))
        rows.append(rec)

    # --- components -------------------------------------------------------
    for _, cr in comp.iterrows():
        members = sorted(cr["members"])
        a, b = members
        ra, rb = p.loc[a], p.loc[b]
        # Site must be identical -- asserted, not assumed (Stage 0.8 §2).
        assert ra["site"] == rb["site"], f"component {cr.component_id} site mismatch"
        conflict = int(ra["dx"]) != int(rb["dx"])
        assert conflict == bool(cr["diagnosis_conflict"]), \
            f"component {cr.component_id} conflict flag disagrees with phenotype"
        canon = canonical_key(members)
        feat = feature_representative(members)

        # Preconditions for collapsing a component onto ONE feature source.
        # A conflict component is excluded downstream and supplies no features.
        if not conflict:
            cid = cr.component_id
            assert len(members) == EXPECT_COMPONENT_SIZE, f"component {cid} size != 2"
            assert ra["site"] == rb["site"], f"component {cid} site mismatch"
            assert int(ra["dx"]) == int(rb["dx"]), f"component {cid} diagnosis mismatch"
            assert set(members) <= passing, \
                f"component {cid} has a member without FC_QC_PASS"
            assert feat in members, f"component {cid} representative outside component"
            assert feat == min(members), \
                f"component {cid} representative is not the lexicographic minimum"

        rec = {
            "canonical_id": canon,
            # A diagnosis-conflict component supplies no diagnostic record and
            # therefore no features: left null so nothing can load it by
            # accident.
            "feature_source_id": pd.NA if conflict else feat,
            "n_source_records": len(members),
            "source_records": ";".join(members),
            "component_id": int(cr["component_id"]),
            "duplicate_component": True,
            "site": ra["site"],
            # A conflict component has no defensible diagnosis; it is left
            # missing and the acquisition is excluded in step 5.
            "diagnosis": pd.NA if conflict else int(ra["dx"]),
            "diagnosis_conflict": conflict,
        }
        amb_vars = []
        for c in value_cols:
            key = RENAME.get(c, c)
            val, status = resolve_value(ra[c], rb[c])
            is_amb = status in AMBIGUOUS_STATUSES
            rec[key] = val
            rec[f"{key}__ambiguous"] = is_amb
            if is_amb:
                amb_vars.append(key)
            ambi_events.append({
                "component_id": int(cr["component_id"]),
                "site": ra["site"],
                "diagnosis_conflict": conflict,
                "variable": key,
                "status": status,
            })
        rec["age_recorded"] = bool(pd.notna(ra["Age"]) and pd.notna(rb["Age"]))
        rec["sex_recorded"] = bool(pd.notna(ra["Sex"]) and pd.notna(rb["Sex"]))
        rows.append(rec)

        resolution.append({
            "component_id": int(cr["component_id"]),
            "site": ra["site"],
            "size": len(members),
            "dx_a": int(ra["dx"]),
            "dx_b": int(rb["dx"]),
            "diagnosis_conflict": conflict,
            "resolution": "EXCLUDE_DIAGNOSIS_CONFLICT" if conflict else "COLLAPSE",
            "canonical_id": canon,
            "feature_source_id": pd.NA if conflict else feat,
            "dropped_alias_id": ";".join(m for m in members if m != canon),
            "records_in": len(members),
            "records_out": 0 if conflict else 1,
            "age_ambiguous": rec["age__ambiguous"],
            "sex_ambiguous": rec["sex__ambiguous"],
            "demographic_ambiguous": rec["age__ambiguous"] or rec["sex__ambiguous"],
            "n_clinical_ambiguous": sum(
                1 for v in amb_vars if v not in ("age", "sex")),
            "ambiguous_vars": ";".join(amb_vars),
        })

    acq = pd.DataFrame(rows)
    acq = acq.sort_values(
        ["site", "canonical_id"],
        key=lambda s: s.map(lambda v: int(v[1:]) if str(v).startswith("S")
                            and str(v)[1:].isdigit() else v) if s.name == "site" else s,
    ).reset_index(drop=True)
    return acq, pd.DataFrame(resolution), pd.DataFrame(ambi_events)


def acquisition_eligibility(acq: pd.DataFrame) -> pd.DataFrame:
    """Frozen §5.4 site rule (>=10 MDD, >=10 HC, age/sex complete) on acquisitions.

    Completeness is evaluated on whether age and sex are RECORDED in the source
    phenotype, not on whether the two records of a collapsed acquisition agree.
    Concordance is a separate concern carried by demographic_ambiguous, and per
    Stage 0.8 §4 it must never remove an acquisition from the diagnostic cohort.

    Age/Sex below are availability probes (0.0 = recorded, NaN = not recorded),
    not ages -- eligibility() reads them only through isna(). Delegating keeps
    exactly one implementation of the inclusion rule in the codebase.
    """
    probe = acq.assign(
        dx=acq["diagnosis"].astype(int),
        Age=np.where(acq["age_recorded"], 0.0, np.nan),
        Sex=np.where(acq["sex_recorded"], 0.0, np.nan),
    )
    return eligibility(probe).rename_axis("site").reset_index()


# ---------------------------------------------------------------------------
def main() -> int:
    assert_read_only()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 120)

    print("=" * 74)
    print("STAGE 0.8 -- ACQUISITION-LEVEL DEDUPLICATION CONTRACT FREEZE")
    print(f"strategy      : {STRATEGY}")
    print(f"primary branch: {PRIMARY_BRANCH}  (frozen, never swapped)")
    print("no classifier / no LOSO / no PCA / no permutation / no BAcc / no S0-S2")
    print("=" * 74)

    phen = load_phenotype()
    comp = load_components()
    value_cols = [c for c in phen.columns if c not in ("ID", "site", "dx")]

    # =====================================================================
    # STEP 1 -- consortium official S4 exclusion
    # =====================================================================
    print("\n[STEP 1] consortium official S4 exclusion")
    n_s4_phen = int((phen["site"] == S4_SITE).sum())
    n_s4_files = len(list((PKG / "Results" / PRIMARY_BRANCH)
                          .glob(f"ROISignals_{S4_SITE}-*.mat")))
    assert n_s4_phen == 0, "S4 must never carry a phenotype row"
    print(f"  S4 phenotype rows            : {n_s4_phen}  (never label-backfilled)")
    print(f"  S4 imaging files present     : {n_s4_files}  (excluded: no label, "
          f"documented duplicate of S14)")
    print(f"  phenotype records entering   : {len(phen)}")

    # =====================================================================
    # STEP 2 -- original phenotype / site validity (frozen spec §5.4)
    # =====================================================================
    print("\n[STEP 2] phenotype/site validity -- frozen §5.4 rule on records")
    elig0 = eligibility(phen)
    eligible_sites = site_sort(elig0.index[elig0.eligible].tolist())
    excluded_sites = site_sort(elig0.index[~elig0.eligible].tolist())
    keep_site = phen["site"].isin(eligible_sites)
    print(f"  sites in phenotype           : {len(elig0)}")
    print(f"  eligible sites               : {len(eligible_sites)}")
    print(f"  excluded sites               : {excluded_sites}")
    print(f"  records on eligible sites    : {int(keep_site.sum())}")

    # =====================================================================
    # STEP 3 -- frozen AAL116 constant-ROI FC QC (label-blind)
    # =====================================================================
    print(f"\n[STEP 3] frozen FC QC + FC extraction on {int(keep_site.sum())} records")
    qc_rows = []
    for sid in phen.loc[keep_site, "ID"]:
        x = load_aal116(sid)
        q = fc_qc(x)
        rec = {"ID": sid, **{k: q[k] for k in
                             ("n_timepoints", "n_nonfinite_roi", "n_constant_roi",
                              "fc_qc_pass")}}
        if q["fc_qc_pass"]:
            v = fc_vector(x)
            rec.update(fc_dim=int(v.size),
                       fc_all_finite=bool(np.isfinite(v).all()),
                       fc_n_nan=int(np.isnan(v).sum()),
                       fc_n_inf=int(np.isinf(v).sum()))
        else:
            rec.update(fc_dim=pd.NA, fc_all_finite=pd.NA,
                       fc_n_nan=pd.NA, fc_n_inf=pd.NA)
        qc_rows.append(rec)
    qc = pd.DataFrame(qc_rows)
    n_pass = int(qc.fc_qc_pass.sum())
    print(f"  FC_QC_PASS                   : {n_pass}")
    print(f"  FC_QC_FAIL                   : {len(qc) - n_pass}")

    # Stage 0.7 detected duplicates inside the post-QC cohort, so every
    # component member must be a QC-passing record on an eligible site.
    passing = set(qc.loc[qc.fc_qc_pass, "ID"])
    members = {m for ms in comp["members"] for m in ms}
    assert members <= passing, "a duplicate component member is not in the post-QC cohort"

    # =====================================================================
    # STEP 4 -- acquisition-level duplicate collapse (Strategy D)
    # =====================================================================
    print("\n[STEP 4] acquisition-level collapse of duplicate components")
    acq, resolution, ambi = build_acquisitions(phen, comp, value_cols, passing)
    fd = pd.read_csv(MEAN_FD_CSV).set_index("ID")["mean_FD_Jenkinson"]
    # One acquisition has one motion estimate. Verify the two records of a
    # collapsed component agree, which is a direct consequence of them being
    # the same scan.
    for _, r in resolution.iterrows():
        a, b = sorted(acq.loc[acq.canonical_id == r.canonical_id,
                              "source_records"].iat[0].split(";"))
        assert abs(float(fd[a]) - float(fd[b])) < FD_TOL, \
            f"component {r.component_id}: mean FD differs across the pair"
    # Motion and FC are read from the frozen representative, not the key.
    acq["mean_fd"] = acq["feature_source_id"].map(fd)
    acq = acq.merge(qc[["ID", "n_timepoints", "fc_qc_pass", "fc_dim",
                        "fc_all_finite", "fc_n_nan", "fc_n_inf"]],
                    left_on="feature_source_id", right_on="ID",
                    how="left").drop(columns="ID")
    acq["fc_qc_pass"] = acq["fc_qc_pass"].fillna(False).astype(bool)
    acq = acq.rename(columns={"component_id": "original_component_id"})
    acq_columns = list(acq.columns)

    n_collapsed = int((~resolution.diagnosis_conflict).sum())
    n_conflict = int(resolution.diagnosis_conflict.sum())
    print(f"  components                   : {len(resolution)}")
    print(f"  diagnosis-concordant         : {n_collapsed}  -> collapsed 2 records -> 1")
    print(f"  diagnosis-conflict           : {n_conflict}  -> both records dropped")
    print(f"  manifest rows (acquisitions) : {len(acq)}")

    # =====================================================================
    # STEP 5 -- inclusion status, incl. diagnosis-conflict exclusion
    # =====================================================================
    print("\n[STEP 5] inclusion status")
    status = np.where(
        ~acq["site"].isin(eligible_sites), "EXCLUDED_SITE_INELIGIBLE",
        np.where(acq["diagnosis_conflict"], "EXCLUDED_DIAGNOSIS_CONFLICT",
                 np.where(~acq["fc_qc_pass"], "EXCLUDED_FC_QC_FAIL", "INCLUDED")))
    reason = {
        "EXCLUDED_SITE_INELIGIBLE": "site fails frozen §5.4 rule (>=10 MDD, >=10 HC)",
        "EXCLUDED_DIAGNOSIS_CONFLICT": "duplicate component carries contradictory diagnoses",
        "EXCLUDED_FC_QC_FAIL": "non-finite or constant AAL116 ROI (frozen FC QC)",
        "INCLUDED": "",
    }
    acq["inclusion_status"] = status
    acq["exclusion_reason"] = acq["inclusion_status"].map(reason)
    print(acq["inclusion_status"].value_counts().to_string())

    acq["demographic_ambiguous"] = acq["age__ambiguous"] | acq["sex__ambiguous"]
    clin_cols = [f"{RENAME.get(c, c)}__ambiguous" for c in value_cols
                 if c not in DEMOGRAPHIC_VARS]
    acq["n_clinical_ambiguous"] = acq[clin_cols].sum(axis=1).astype(int)
    acq["clinical_ambiguous"] = acq["n_clinical_ambiguous"] > 0
    acq["clinical_ambiguous_vars"] = [
        ";".join(c[:-len("__ambiguous")] for c in clin_cols if row[c])
        for _, row in acq.iterrows()
    ]

    # =====================================================================
    # STEP 6 -- recompute site eligibility on independent acquisitions
    # =====================================================================
    print("\n[STEP 6] site eligibility recomputed on independent acquisitions")
    kept = acq[acq.inclusion_status == "INCLUDED"].copy()
    elig1 = acquisition_eligibility(kept)
    final_sites = site_sort(elig1.loc[elig1.eligible, "site"].tolist())
    dropped = sorted(set(eligible_sites) - set(final_sites))
    n_mdd = int((kept.diagnosis == 1).sum())
    n_hc = int((kept.diagnosis == 0).sum())
    print(f"  retained independent acquisitions : {len(kept)}")
    print(f"  MDD / HC                          : {n_mdd} / {n_hc}")
    print(f"  eligible sites after collapse     : {len(final_sites)}")
    print(f"  sites falling below >=10/>=10     : {dropped if dropped else 'none'}")

    # Strict reading of the completeness clause, reported for auditability:
    # it would additionally penalise sites merely for containing an ambiguous
    # acquisition. Not adopted (Stage 0.8 §4 forbids dropping them).
    strict = site_sort(kept.loc[kept.demographic_ambiguous, "site"].unique().tolist())
    print(f"  sites holding a demographic-ambiguous acquisition : {strict}")
    print("  (flag only -- these acquisitions stay in the diagnostic cohort)")

    # =====================================================================
    # per-site accounting
    # =====================================================================
    rec_pass = phen[keep_site & phen.ID.isin(passing)]
    per_site = pd.DataFrame({
        "n_MDD_records_postqc": rec_pass.groupby("site").dx.sum().astype(int),
        "n_HC_records_postqc": rec_pass.groupby("site").dx.apply(
            lambda s: int((s == 0).sum())),
    }).reindex(eligible_sites).fillna(0).astype(int)
    per_site["n_components"] = resolution.groupby("site").size().reindex(
        eligible_sites).fillna(0).astype(int)
    per_site["n_records_collapsed_away"] = (
        resolution[~resolution.diagnosis_conflict].groupby("site").size()
        .reindex(eligible_sites).fillna(0).astype(int))
    per_site["n_records_conflict_excluded"] = (
        resolution[resolution.diagnosis_conflict].groupby("site").size()
        .reindex(eligible_sites).fillna(0).astype(int) * 2)
    per_site["n_MDD_acq"] = kept[kept.diagnosis == 1].groupby("site").size().reindex(
        eligible_sites).fillna(0).astype(int)
    per_site["n_HC_acq"] = kept[kept.diagnosis == 0].groupby("site").size().reindex(
        eligible_sites).fillna(0).astype(int)
    per_site["n_total_acq"] = per_site.n_MDD_acq + per_site.n_HC_acq
    per_site["n_demographic_ambiguous"] = kept[kept.demographic_ambiguous].groupby(
        "site").size().reindex(eligible_sites).fillna(0).astype(int)
    per_site["n_clinical_ambiguous"] = kept[kept.clinical_ambiguous].groupby(
        "site").size().reindex(eligible_sites).fillna(0).astype(int)
    per_site["eligible_post_dedup"] = per_site.index.isin(final_sites)
    per_site["dropped_below_threshold"] = ~per_site["eligible_post_dedup"]
    per_site = per_site.rename_axis("site").reset_index()
    print("\n  per-site acquisition counts")
    print(per_site[["site", "n_MDD_records_postqc", "n_HC_records_postqc",
                    "n_components", "n_MDD_acq", "n_HC_acq", "n_total_acq",
                    "eligible_post_dedup"]].to_string(index=False))

    # =====================================================================
    # ambiguity summary
    # =====================================================================
    print("\n[AMBIGUITY] phenotype reconciliation across collapsed components")
    if len(ambi):
        summ = (ambi[~ambi.diagnosis_conflict]
                .pivot_table(index="variable", columns="status", aggfunc="size",
                             fill_value=0)
                .reindex(columns=["agree", "conflict", "single_source",
                                  "both_missing"], fill_value=0)
                .rename_axis(None, axis=1).reset_index())
        summ["n_components_checked"] = n_collapsed
        summ["n_acquisitions_flagged"] = summ["conflict"] + summ["single_source"]
        summ["variable_class"] = np.where(
            summ.variable.isin(["age", "sex"]), "demographic", "clinical")
        summ["resolution"] = np.where(
            summ.n_acquisitions_flagged > 0,
            "value set missing + AMBIGUOUS flag; never picked from one record",
            "concordant -- value retained")
        summ = summ.sort_values(["variable_class", "n_acquisitions_flagged"],
                                ascending=[True, False])
    else:  # pragma: no cover - components are frozen at 41
        summ = pd.DataFrame()
    print(summ[summ.n_acquisitions_flagged > 0][
        ["variable_class", "variable", "agree", "conflict", "single_source",
         "both_missing", "n_acquisitions_flagged"]].to_string(index=False))
    n_demo_amb = int(kept.demographic_ambiguous.sum())
    n_clin_amb = int(kept.clinical_ambiguous.sum())
    print(f"\n  acquisitions DEMOGRAPHIC_AMBIGUOUS : {n_demo_amb}")
    print(f"  acquisitions clinical-ambiguous    : {n_clin_amb}")

    # =====================================================================
    # manifest assertions -- one row per independent acquisition
    # =====================================================================
    print("\n[ASSERT] one-row-per-acquisition invariants")
    checks = []

    def check(name: str, ok: bool) -> None:
        checks.append({"assertion": name, "result": "PASS" if ok else "FAIL"})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        assert ok, name

    check("canonical_id is unique across the manifest", acq.canonical_id.is_unique)
    all_src = [s for row in acq.source_records for s in row.split(";")]
    check("every phenotype record maps to exactly one manifest row",
          len(all_src) == len(phen) and set(all_src) == set(phen.ID))
    check("no phenotype record appears in two manifest rows",
          len(all_src) == len(set(all_src)))
    check("manifest rows == records - collapsed - (conflict merges)",
          len(acq) == len(phen) - EXPECT_MEMBERS + EXPECT_COMPONENTS)
    check("no duplicate component contributes >1 retained record",
          int(kept[kept.duplicate_component].n_source_records.count()) == n_collapsed
          and kept[kept.duplicate_component].canonical_id.is_unique)
    check("diagnosis-conflict components contribute 0 retained records",
          int((kept.diagnosis_conflict).sum()) == 0)
    check("no S4 record anywhere in the manifest",
          not any(s.split("-")[0] == S4_SITE for s in all_src))
    check("retained diagnosis is non-null and binary",
          kept.diagnosis.notna().all() and set(kept.diagnosis.unique()) <= {0, 1})
    check("retained acquisitions all on eligible sites",
          set(kept.site) <= set(eligible_sites))
    check("retained record count == collapse arithmetic",
          len(kept) == n_pass - n_collapsed - 2 * n_conflict)
    check("mean FD present for every retained acquisition",
          kept.mean_fd.notna().all())

    print("\n[ASSERT] frozen cohort constants")
    check(f"INCLUDED rows == {EXPECT_INCLUDED}", len(kept) == EXPECT_INCLUDED)
    check(f"INCLUDED MDD == {EXPECT_INCLUDED_MDD}", n_mdd == EXPECT_INCLUDED_MDD)
    check(f"INCLUDED HC == {EXPECT_INCLUDED_HC}", n_hc == EXPECT_INCLUDED_HC)
    check(f"eligible sites == {EXPECT_ELIGIBLE_SITES}",
          len(final_sites) == EXPECT_ELIGIBLE_SITES
          and kept.site.nunique() == EXPECT_ELIGIBLE_SITES)

    print("\n[ASSERT] acquisition identity and feature representative")
    check("feature_source_id is unique across INCLUDED rows",
          kept.feature_source_id.is_unique and kept.feature_source_id.notna().all())
    check("every INCLUDED feature_source_id is one of that row's source_records",
          all(r.feature_source_id in r.source_records.split(";")
              for r in kept.itertuples()))
    check("feature_source_id == lexicographic min of the source records",
          all(r.feature_source_id == min(r.source_records.split(";"))
              for r in kept.itertuples()))
    check("feature_source_id == canonical_id on every INCLUDED row",
          bool((kept.feature_source_id == kept.canonical_id).all()))
    check(f"each of the {n_collapsed} concordant components has exactly one "
          "retained representative",
          int(kept[kept.duplicate_component].shape[0]) == n_collapsed
          and kept[kept.duplicate_component].feature_source_id.is_unique)
    check("diagnosis-conflict components carry no feature_source_id",
          acq.loc[acq.diagnosis_conflict, "feature_source_id"].isna().all())
    check("every manifest identity column exists for the Gate 1 exclusion list",
          all(c in acq_columns for c in FORBIDDEN_IN_X))

    print("\n[ASSERT] 6670-D FC validation on retained acquisitions")
    check(f"exactly one FC vector per acquisition, read via feature_source_id "
          f"({len(kept)})",
          len(kept) == kept.feature_source_id.nunique()
          and all((PKG / "Results" / PRIMARY_BRANCH /
                   f"ROISignals_{s}.mat").exists() for s in kept.feature_source_id))
    check(f"every retained FC vector has {N_EDGES} dimensions",
          bool((kept.fc_dim == N_EDGES).all()))
    check("every retained FC vector is all-finite",
          bool(kept.fc_all_finite.all()))
    check("zero NaN across retained FC vectors", int(kept.fc_n_nan.sum()) == 0)
    check("zero Inf across retained FC vectors", int(kept.fc_n_inf.sum()) == 0)
    check("every retained acquisition passed the frozen FC QC",
          bool(kept.fc_qc_pass.all()))
    fc_verdict = "PASS" if all(c["result"] == "PASS" for c in checks) else "FAIL"

    # =====================================================================
    # outputs
    # =====================================================================
    front = ["canonical_id", "feature_source_id", "original_component_id", "site",
             "diagnosis", "age", "sex", "mean_fd", "fc_qc_pass",
             "duplicate_component", "demographic_ambiguous", "clinical_ambiguous",
             "n_clinical_ambiguous", "clinical_ambiguous_vars", "inclusion_status",
             "exclusion_reason", "n_source_records", "source_records",
             "diagnosis_conflict", "age_recorded", "sex_recorded", "n_timepoints",
             "fc_dim", "fc_all_finite", "fc_n_nan", "fc_n_inf"]
    acq_out = acq[front + [c for c in acq.columns if c not in front]]

    writes = {
        "analysis_manifest_v1.csv": acq_out,
        "post_dedup_site_counts.csv": per_site,
        "duplicate_component_resolution.csv": resolution,
        "phenotype_ambiguity_summary.csv": summ,
        "manifest_assertions.csv": pd.DataFrame(checks),
    }
    print()
    for name, df in writes.items():
        df.to_csv(OUT / name, index=False)
        print(f"  wrote {name}  ({len(df)} rows)")

    print("\n" + "=" * 74)
    print("STAGE 0.8 SUMMARY")
    print(f"  strategy                       : {STRATEGY}")
    print(f"  duplicate components           : {len(resolution)} "
          f"({n_collapsed} concordant / {n_conflict} conflict)")
    print(f"  feature representative         : feature_source_id = "
          f"min(member ids), frozen; {n_collapsed} collapsed components each "
          f"contribute exactly 1")
    print(f"  forbidden in X (Gate 1 loader) : {', '.join(FORBIDDEN_IN_X)}")
    print(f"  retained independent acquisitions : {len(kept)} "
          f"({n_mdd} MDD / {n_hc} HC)")
    print(f"  eligible sites                 : {len(final_sites)}")
    print(f"  DEMOGRAPHIC_AMBIGUOUS          : {n_demo_amb}")
    print(f"  clinical-ambiguous             : {n_clin_amb}")
    print(f"  diagnosis-conflict exclusions  : {n_conflict} acquisitions "
          f"/ {2 * n_conflict} records")
    print(f"  6670-D FC validation           : {fc_verdict}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
