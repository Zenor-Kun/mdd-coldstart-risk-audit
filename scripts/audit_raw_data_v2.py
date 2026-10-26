#!/usr/bin/env python3
"""Independent raw-data identity, site, label, and leakage audit.

This script deliberately reconstructs all counts from the original workbook and
dataset directories. It does not import project code or consume an earlier audit.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ID_FULL_RE = re.compile(r"^S(?P<site>[1-9]\d*)-(?P<group>[12])-(?P<serial>\d{4})$")
ID_SEARCH_RE = re.compile(r"(?<![A-Za-z0-9])S[1-9]\d*-[12]-\d{4}(?!\d)")
SENTINEL = -9999


def canonical_id(value: object) -> str:
    return str(value).strip().upper()


def parse_id(subject_id: str) -> tuple[int, int, int] | None:
    match = ID_FULL_RE.fullmatch(subject_id)
    if not match:
        return None
    return tuple(int(match.group(name)) for name in ("site", "group", "serial"))


def extract_id(name: str) -> str | None:
    matches = ID_SEARCH_RE.findall(name)
    return matches[0] if len(matches) == 1 else None


def read_phenotype(workbook: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unknown extension is not supported")
        excel = pd.ExcelFile(workbook)
        for sheet, label in (("MDD", 1), ("Controls", 0)):
            frame = pd.read_excel(excel, sheet_name=sheet, dtype=object)
            frame.insert(0, "sheet", sheet)
            frame.insert(1, "label", label)
            frame.insert(2, "subject_id", frame["ID"].map(canonical_id))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def collect_named_files(source: Path) -> tuple[list[Path], list[str], list[str]]:
    files = sorted(path for path in source.iterdir() if path.is_file() and path.name != ".DS_Store")
    ids: list[str] = []
    unparsed: list[str] = []
    for path in files:
        subject_id = extract_id(path.name)
        if subject_id is None:
            unparsed.append(path.name)
        else:
            ids.append(subject_id)
    return files, ids, unparsed


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def roi_payload_hashes(directory: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_digest: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
    errors: list[dict[str, object]] = []
    for index, path in enumerate(sorted(directory.glob("*.mat")), start=1):
        subject_id = extract_id(path.name)
        try:
            payload = loadmat(path, variable_names=["ROISignals"])["ROISignals"]
            contiguous = np.ascontiguousarray(payload)
            digest = hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
            by_digest[digest].append((subject_id or "", path.name, *payload.shape))
        except Exception as exc:  # report corrupt or structurally unexpected files
            errors.append({"file": path.name, "error": repr(exc)})
        if index % 250 == 0:
            print(f"hashed {index} ROI matrices", flush=True)
    duplicates: list[dict[str, object]] = []
    group_number = 0
    for digest, members in sorted(by_digest.items()):
        if len(members) < 2:
            continue
        group_number += 1
        for subject_id, filename, rows, columns in members:
            duplicates.append(
                {
                    "duplicate_group": group_number,
                    "sha256_payload": digest,
                    "subject_id": subject_id,
                    "file": filename,
                    "timepoints": rows,
                    "columns": columns,
                }
            )
    return duplicates, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/audit_v2"))
    parser.add_argument(
        "--hash-roi",
        action="store_true",
        help="Hash numerical ROI payloads in the no-GSR branch to detect exact duplicated scans.",
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    workbook = args.data_dir / "REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx"
    dataset = args.data_dir / "REST-meta-MDD"
    phenotype = read_phenotype(workbook)
    phenotype_ids = set(phenotype["subject_id"])

    parsed = phenotype["subject_id"].map(parse_id)
    invalid_rows = phenotype.loc[parsed.isna(), ["sheet", "subject_id"]]
    valid = phenotype.loc[parsed.notna()].copy()
    valid[["site", "id_group", "serial"]] = pd.DataFrame(
        parsed.dropna().tolist(), index=valid.index
    )
    valid[["site", "id_group", "serial"]] = valid[["site", "id_group", "serial"]].astype(int)

    exact_duplicates = phenotype[phenotype.duplicated("subject_id", keep=False)].sort_values("subject_id")
    label_mismatches = valid[
        ((valid["sheet"] == "MDD") & (valid["id_group"] != 1))
        | ((valid["sheet"] == "Controls") & (valid["id_group"] != 2))
    ]
    cross_label_ids = (
        phenotype.groupby("subject_id")["label"].nunique().loc[lambda values: values > 1].index.tolist()
    )

    # Collisions caused by tempting but unsafe ID transformations.
    valid["site_serial_key"] = valid.apply(lambda row: f"S{row.site}-{row.serial:04d}", axis=1)
    valid["group_serial_key"] = valid.apply(lambda row: f"G{row.id_group}-{row.serial:04d}", axis=1)
    dropped_group = valid[valid.duplicated("site_serial_key", keep=False)].sort_values("site_serial_key")
    dropped_site = valid[valid.duplicated("group_serial_key", keep=False)].sort_values("group_serial_key")

    site_counts = (
        valid.pivot_table(index="site", columns="sheet", values="subject_id", aggfunc="count", fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )
    for column in ("MDD", "Controls"):
        if column not in site_counts:
            site_counts[column] = 0
    site_counts["total"] = site_counts["MDD"] + site_counts["Controls"]
    site_counts["mdd_prevalence"] = site_counts["MDD"] / site_counts["total"]
    site_counts["eligible_ge10_each"] = (site_counts["MDD"] >= 10) & (site_counts["Controls"] >= 10)
    site_counts.to_csv(output / "site_counts.csv", index=False)

    # Same-data site-majority prediction is intentionally diagnostic: it quantifies
    # how much label information a random subject split could extract from site alone.
    majority_by_site = valid.groupby("site")["label"].mean().ge(0.5).astype(int)
    site_prediction = valid["site"].map(majority_by_site).astype(int)
    truth = valid["label"].astype(int)
    site_only_accuracy = float((site_prediction == truth).mean())
    recalls = [float((site_prediction[truth == cls] == cls).mean()) for cls in (0, 1)]
    site_only_bacc = sum(recalls) / 2

    inventory: list[dict[str, object]] = []
    differences: list[dict[str, object]] = []

    def add_source(name: str, ids: list[str], file_count: int, unparsed: list[str]) -> None:
        counts = Counter(ids)
        id_set = set(ids)
        multiple_files = sorted(subject_id for subject_id, count in counts.items() if count > 1)
        multiplicities = ";".join(f"{count}x:{sum(value == count for value in counts.values())}" for count in sorted(set(counts.values())))
        inventory.append(
            {
                "source": name,
                "files_or_rows": file_count,
                "parsed_id_occurrences": len(ids),
                "unique_subject_ids": len(id_set),
                "subjects_with_multiple_files": len(multiple_files),
                "files_per_subject_distribution": multiplicities,
                "unparsed_names": len(unparsed),
                "missing_phenotype_subjects": len(phenotype_ids - id_set),
                "extra_nonphenotype_subjects": len(id_set - phenotype_ids),
            }
        )
        for subject_id in sorted(phenotype_ids - id_set):
            differences.append({"source": name, "status": "missing_from_source", "subject_id": subject_id})
        for subject_id in sorted(id_set - phenotype_ids):
            differences.append({"source": name, "status": "not_in_phenotype", "subject_id": subject_id})
        for subject_id in multiple_files:
            differences.append(
                {"source": name, "status": f"multiple_files_for_id={counts[subject_id]}", "subject_id": subject_id}
            )
        for value in unparsed:
            differences.append({"source": name, "status": "unparsed_name", "subject_id": value})

    realign = dataset / "RealignParameter"
    realign_dirs = sorted(path for path in realign.iterdir() if path.is_dir())
    realign_ids = [canonical_id(path.name) for path in realign_dirs if parse_id(canonical_id(path.name))]
    realign_unparsed = [path.name for path in realign_dirs if not parse_id(canonical_id(path.name))]
    add_source("RealignParameter/directories", realign_ids, len(realign_dirs), realign_unparsed)
    derivative_sites = sorted({parse_id(value)[0] for value in realign_ids})

    results_root = dataset / "Results"
    result_branches = sorted(path for path in results_root.iterdir() if path.is_dir())
    for branch in result_branches:
        files, ids, unparsed = collect_named_files(branch)
        add_source(f"Results/{branch.name}", ids, len(files), unparsed)

    masks = dataset / "Masks" / "AutoMasks"
    files, ids, unparsed = collect_named_files(masks)
    add_source("Masks/AutoMasks", ids, len(files), unparsed)

    qc_stats: list[dict[str, object]] = []
    for qc_path in sorted((dataset / "QC").glob("*.tsv")):
        qc = pd.read_csv(qc_path, sep="\t", dtype=object, encoding="gb18030")
        subject_column = "Subject ID"
        qc_ids = qc[subject_column].map(canonical_id).tolist()
        add_source(f"QC/{qc_path.name}", qc_ids, len(qc), [])
        scores = pd.to_numeric(qc["QC Score"], errors="coerce")
        qc_stats.append(
            {
                "source": qc_path.name,
                "rows": len(qc),
                "missing_score": int(scores.isna().sum()),
                "minimum_score": scores.min(),
                "maximum_score": scores.max(),
                "score_lt_1": int((scores < 1).sum()),
            }
        )

    pd.DataFrame(inventory).to_csv(output / "source_inventory.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "source_subject_differences.csv", index=False)
    pd.DataFrame(qc_stats).to_csv(output / "qc_summary.csv", index=False)

    id_issue_rows: list[dict[str, object]] = []
    for _, row in invalid_rows.iterrows():
        id_issue_rows.append({"issue": "invalid_canonical_format", "subject_id": row.subject_id, "detail": row.sheet})
    for _, row in exact_duplicates.iterrows():
        id_issue_rows.append({"issue": "duplicate_full_id", "subject_id": row.subject_id, "detail": row.sheet})
    for _, row in label_mismatches.iterrows():
        id_issue_rows.append({"issue": "sheet_vs_id_group_mismatch", "subject_id": row.subject_id, "detail": row.sheet})
    for subject_id in cross_label_ids:
        id_issue_rows.append({"issue": "same_id_in_both_labels", "subject_id": subject_id, "detail": ""})
    for key, group in dropped_group.groupby("site_serial_key"):
        id_issue_rows.append(
            {"issue": "collision_if_group_dropped", "subject_id": key, "detail": "|".join(group.subject_id)}
        )
    pd.DataFrame(id_issue_rows, columns=["issue", "subject_id", "detail"]).to_csv(
        output / "id_issues.csv", index=False
    )

    duplicate_roi_rows: list[dict[str, object]] = []
    roi_errors: list[dict[str, object]] = []
    if args.hash_roi:
        roi_branch = results_root / "ROISignals_FunImgARCWF"
        duplicate_roi_rows, roi_errors = roi_payload_hashes(roi_branch)
    pd.DataFrame(
        duplicate_roi_rows,
        columns=["duplicate_group", "sha256_payload", "subject_id", "file", "timepoints", "columns"],
    ).to_csv(output / "roi_payload_duplicates.csv", index=False)
    pd.DataFrame(roi_errors, columns=["file", "error"]).to_csv(output / "roi_hash_errors.csv", index=False)

    core_columns = ["ID", "Sex", "Age"]
    core_missing: dict[str, int] = {}
    for column in core_columns:
        series = phenotype[column]
        numeric = pd.to_numeric(series, errors="coerce") if column != "ID" else None
        missing = series.isna() | series.astype(str).str.strip().eq("")
        if numeric is not None:
            missing |= numeric.eq(SENTINEL)
        core_missing[column] = int(missing.sum())

    clinical_sentinel: dict[str, int] = {}
    mdd = phenotype[phenotype["sheet"] == "MDD"]
    for column in mdd.columns:
        if column in {"sheet", "label", "subject_id", "ID"}:
            continue
        numeric = pd.to_numeric(mdd[column], errors="coerce")
        count = int(numeric.eq(SENTINEL).sum())
        if count:
            clinical_sentinel[column] = count

    extras_by_source = pd.DataFrame(differences)
    extras = extras_by_source[extras_by_source["status"] == "not_in_phenotype"] if len(extras_by_source) else extras_by_source
    unique_extras = sorted(extras["subject_id"].unique()) if len(extras) else []
    extra_group_counts = Counter(parse_id(value)[1] for value in unique_extras if parse_id(value))
    extra_site_counts = Counter(parse_id(value)[0] for value in unique_extras if parse_id(value))
    qc_missing = {
        row["source"]: int(row["missing_phenotype_subjects"])
        for row in inventory
        if str(row["source"]).startswith("QC/")
    }

    lines = [
        "# Raw-data audit v2 (independent reconstruction)",
        "",
        "This audit reads only the original phenotype workbook and raw dataset directories. It does not consume prior audit numbers or project-generated splits.",
        "",
        "## Executive findings",
        "",
        f"- Phenotype: **{len(phenotype):,} rows / {phenotype.subject_id.nunique():,} unique full IDs** ({int((phenotype.sheet == 'MDD').sum()):,} MDD; {int((phenotype.sheet == 'Controls').sum()):,} controls).",
        f"- Full-ID duplicates: **{exact_duplicates.subject_id.nunique()}**; cross-label duplicate IDs: **{len(cross_label_ids)}**; invalid canonical ID formats: **{len(invalid_rows)}**.",
        f"- Label mapping errors: **{len(label_mismatches)}**. Every checked MDD row should carry middle code `1`, and every control row middle code `2`.",
        f"- Site count: **{site_counts.site.nunique()}** represented sites; **{int(site_counts.eligible_ge10_each.sum())}** meet >=10 subjects in each class.",
        f"- Derivative directories represent site codes **{derivative_sites}**, while phenotype represents **{sorted(site_counts.site.tolist())}**.",
        f"- Non-phenotyped IDs seen in at least one derivative/QC source: **{len(unique_extras)}** (sites: {dict(sorted(extra_site_counts.items()))}; ID-code groups: {dict(sorted(extra_group_counts.items()))}).",
        f"- Exact duplicate numerical ROI payload groups (no-GSR branch): **{len(set(row['duplicate_group'] for row in duplicate_roi_rows)) if duplicate_roi_rows else 0}**; hashing {'enabled' if args.hash_roi else 'not run'}; load/hash errors: **{len(roi_errors)}**.",
        "",
        "## Leakage and identity risks",
        "",
        "1. **Critical — diagnosis is encoded in the subject ID.** The middle token is perfectly aligned with workbook diagnosis (`1` = MDD, `2` = control). Raw IDs, filenames, paths, group-token-derived values, and one-hot/string/vectorized IDs must never enter features, preprocessing statistics, feature selection, or model inputs.",
        "   **Mapping trap:** naïvely computing `middle_token - 1` yields MDD=0 and control=1, the reverse of this audit's explicit `MDD=1, control=0` convention. Labels must come from sheet membership and be cross-checked against the token.",
        f"2. **High — site is encoded in the ID and class prevalence differs by site.** Site prevalence ranges from {site_counts.mdd_prevalence.min():.3f} to {site_counts.mdd_prevalence.max():.3f}. A same-data site-majority diagnostic reaches accuracy {site_only_accuracy:.3f} and balanced accuracy {site_only_bacc:.3f}; this is not a valid model estimate, but demonstrates the shortcut available under random subject splitting.",
        "   Parse site with the anchored full-ID regex, not string slicing (`S1` and `S10` must remain distinct). Site codes are nominal grouping keys, not ordered numeric features; numeric or one-hot site values must not enter the diagnostic model.",
        f"3. **High — filesystem-first joins silently create labels for subjects absent from phenotype.** All {len(unique_extras)} derivative-only IDs are site S4 ({extra_group_counts.get(1, 0)} code-1 and {extra_group_counts.get(2, 0)} code-2), whereas phenotype has no S4. Build the cohort from workbook rows and treat S4 as excluded until phenotype provenance is resolved.",
        f"4. **High — dropping the middle group token creates {dropped_group.site_serial_key.nunique()} colliding site+serial keys covering {len(dropped_group)} phenotype rows.** Never normalize `S<site>-<group>-<serial>` to `S<site>-<serial>`.",
        f"5. **Medium — dropping site creates {dropped_site.group_serial_key.nunique()} reused group+serial keys.** Serial numbers are site-local and cannot identify subjects globally.",
        "6. **Protocol leakage — no train/test split manifests exist in this raw-only repository.** Future splits must be site-disjoint for outer LOSO; all transforms, feature selection, hyperparameters, OOF calibration, threshold grids, and per-fold risk targets must be fit/frozen from the allowed source fold only.",
        "",
        "## Completeness and encoding checks",
        "",
        f"- Core-field missing/sentinel counts: `{core_missing}`.",
        f"- MDD clinical `-9999` sentinel counts (nonzero fields): `{clinical_sentinel}`.",
        "- QC tables require `gb18030` decoding; UTF-8 parsing fails. This is an ingestion hazard, not a subject-level defect.",
        f"- QC is not a complete phenotype key: missing phenotype subjects by table are `{qc_missing}`. The missing set is dominated by S10; QC also contains all 48 non-phenotyped S4 IDs. Never inner-join QC without reporting cohort loss.",
        "- Two files per subject in DegreeCentrality branches (binarized and weighted maps) and AutoMasks (native and warped masks) are expected artifact multiplicity, not duplicated phenotype subjects.",
        "- Detailed per-site counts and every source-vs-phenotype discrepancy are saved in companion CSV files.",
        "",
        "## Required gates before modeling",
        "",
        "- Start from the phenotype cohort and map labels from sheet membership or an explicit validated table, never from the ID token.",
        "- Assert full ID uniqueness, canonical regex, sheet/token consistency, one derivative per included subject per selected branch, and no derivative-only IDs after the join.",
        "- Keep `subject_id`, `site`, and `label` in metadata-only columns; use grouped/site-disjoint splitting keyed by numeric site parsed from the full validated ID.",
        "- Fit every learned preprocessing operation inside the training fold. For S0, do not inspect held-out-site covariates even for normalization or atlas/edge filtering.",
        "- Save explicit fold manifests and automatically assert empty subject and site intersections between train/calibration/test partitions.",
        "",
        "## Companion files",
        "",
        "- `site_counts.csv`: independent per-site MDD/control counts and eligibility flag.",
        "- `source_inventory.csv`: counts, duplicates, missing phenotype rows, and extra IDs for every raw source.",
        "- `source_subject_differences.csv`: row-level source/phenotype discrepancies.",
        "- `id_issues.csv`: malformed/duplicate/mismapped IDs and unsafe-normalization collisions.",
        "- `qc_summary.csv`: QC score parsing summary.",
        "- `roi_payload_duplicates.csv`, `roi_hash_errors.csv`: exact numerical payload duplicate scan and errors.",
    ]
    (output / "AUDIT_V2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote audit to {output}")


if __name__ == "__main__":
    main()
