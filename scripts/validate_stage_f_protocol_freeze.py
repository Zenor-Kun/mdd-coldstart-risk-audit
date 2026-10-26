#!/usr/bin/env python3
"""Stage F.7 — mechanical validator for the Stage-F Protocol v2 amendment.

The **protocol-authority** validator for `STAGE_F_PROTOCOL/2.0.0`, currently an
**untagged amendment DRAFT** awaiting an independent audit before
`stage-f-protocol-v2` may be created (§L.4 condition 8). It is not the
implementation validator (`scripts/validate_stage_f_implementation.py`), it
validates no runtime behaviour, and it authorizes no run.

It covers the whole Stage-F amendment-validation lifecycle: the original F.1
protocol audit, the F.1a and F.1b re-audits, the F.7 `1.0.2 -> 2.0.0`
amendment, and the corrections made to the `2.0.0` draft after the independent
Protocol-v2 draft audit returned FAIL (`STAGE_F_PROTOCOL.md` §0.4).

Verifies that `docs/STAGE_F_PROTOCOL.md` (`STAGE_F_PROTOCOL/2.0.0`) is
mechanically executable against the frozen upstream, that every constant and
count it freezes is re-derivable, and that every finding of those audits is
actually closed in the document rather than described as closed.

    artifacts    all 17 consumed files exist and match their frozen SHA256; the
                 RQ1/S0 + RQ1/S1 files are additionally bound to their own
                 frozen result ledgers; the two Gate-1 files are content-bound
                 to ledger-bound S0 files (Gate 1 froze no ledger); the SVM
                 feature cache matches its digest and its content key
    gate1        the Gate-1 OOF binding is verified BY VALUE, by invoking the
                 genuine frozen helper
                 src/s2_inputs.assert_gate1_binds_to_checksummed_s0 read-only:
                 seven bit-identical columns, oof_prob_mdd within the frozen
                 4e-16, no duplicate logical key, one-to-one row alignment.
                 Negative synthetic self-tests prove the reproduction of that
                 helper rejects a column mutation, an over-tolerance
                 perturbation, a duplicate key, a misalignment and a row-count
                 mismatch. Shared-column presence and equal row counts are NOT
                 accepted as a binding (F.1a re-audit finding 3)
    sens56       the SENS-5/6 output-registry keys are structurally consistent
                 with the frozen RQ3 parent: cells.csv keyed on
                 (population, operating_point, method) with 12 rows and NO block
                 field, validation_units.csv keyed on that plus block with 24
                 rows, exactly 2 block-level units per cell, families 4 x 6
    cohort       2232 INCLUDED acquisitions across the 23 frozen eligible sites
    splits       the RQ2/S2 production split registry re-derives exactly:
                 7700 rows, per-N site counts 23/22/21/11, 100 repeats, and
                 EVERY row is a disjoint partition with |cal| == N
    weights      density-weight and source-OOF key sets are exactly equal
    scores       every consumed probability column is finite and in [0, 1]
    conformal    the frozen order-statistic index rule is attainable at every
                 (N, alpha) in {10,20,40} x {0.1,0.2,0.3,0.4} and unattainable
                 at N=5, alpha=0.1; the guarantee labels are correctly bound
                 and no exact-target-coverage claim survives
    ci           the closed 80-slot RQ4 CI registry matches src/site_bootstrap's
                 frozen machinery, is disjoint from the prohibited list, and the
                 draw matrix is deterministic at every Stage-F site-set size
    status       estimability, degeneracy and class composition are separate;
                 the two withdrawn 1.0.0 reason codes no longer invalidate
                 anything
    sens6        the coverage map agrees with the frozen OP summaries and the
                 frozen selection rule reproduces all four worked examples
    cardinality  every sensitivity cardinality formula recomputes from the
                 frozen registries to its registered value
    svm          the frozen C / PCA grids equal src/nested_loso's, LinearSVC
                 accepts sample_weight under the frozen parameter set, exposes
                 no predict_proba, is deterministic, and random_state is
                 confirmed inert under dual=False by the installed sklearn
    svm 2.0.0    the amended fold_selection representation uses only the
                 existing status/reason vocabulary, keeps the key and the 460-row
                 cardinality, registers the complete-grid diagnostic policy, and
                 leaves every frozen scientific SVM definition untouched
    svm warnings the two per-site ConvergenceWarning representations (§F.7.1
                 inner-grid, §F.7.1a final-refit-after-lawful-selection) are both
                 frozen, complete and DISTINCT. Every mandated field is checked
                 for KEY PRESENCE, type and value independently, so a deleted
                 key can never read as a present null and false / 0 / "" are
                 never accepted as null; both persisted-row tables and the
                 field-by-field difference table are PARSED out of the protocol
                 document and compared for exact row-set and value equality;
                 named mutation probes prove a deleted field, a null<->2000 flip
                 in either direction, a conflation of the two cases, a smuggled
                 unregistered field and a removed or reworded table are rejected
    svm failures SVM_FOLD_FAILED carries a POSITIVE, closed, exhaustive
                 allow-list (FloatingPointError, numpy.linalg.LinAlgError),
                 exact-type matching -- `type(exc) is T`, verified BY BEHAVIOUR
                 to reject a subclass that isinstance would admit -- an
                 estimator/backend boundary limited to LinearSVC.fit and
                 decision_function, and the full 20-class fail-hard exclusion
                 set. All four are DERIVED STRUCTURALLY from §G.2a's own
                 normative table, numbered conditions and bullet list and
                 compared for equality: token presence is never accepted.
                 Exclusions alone are NOT a specification, the scipy alias may
                 never become a third semantic type, and named mutation probes
                 prove that a third permitted type, a removed type, isinstance
                 widening, a deleted or reworded exclusion, and a boundary
                 broadened to preprocessing/PCA are all rejected in the registry
                 AND in the document
    store        no authoritative 2.0.0 store, no 2.0.0 freeze, no protocol tag
                 and no result tag exist
    historical   results/stage_f/ still reproduces its frozen content digest and
                 stage-f-protocol-v1 / stage-f-results-v1 still resolve to their
                 recorded commits -- the historical authority is untouched
    freeze bytes results/stage_f_freeze/ is bound to the TAGGED authority, not to
                 working-tree metadata: the stage-f-results-v1 tag object and its
                 target commit are pinned, the expected file set is read from the
                 tagged tree, and every freeze file is compared BYTE-FOR-BYTE
                 with its tagged blob and by SHA-256 with its frozen digest. No
                 extra file, no missing file, and the three results/stage_f/
                 store anchors plus the 35-file count agree with both the
                 protocol and the freeze binding. Any byte drift is a FAIL
    mirroring    TRUE two-way structured comparison between this file's mirrored
                 constants and docs/stage_f_constants.json -- key sets equal in
                 both directions and every value equal. Substring presence is
                 never accepted as mirroring.
    stale        no prohibited 1.0.0 claim survives outside explicit historical
                 audit text

**It fits no real-data model and computes no Stage-F scientific quantity.** No
conformal quantile is taken on cohort data, no prediction set is built, no
coverage, set size or singleton rate is computed, no sensitivity is executed, no
GLMM is fitted and no SVM is fitted on real data. The only estimator call is a
capability probe on synthetic Gaussian noise. The only arithmetic performed on
frozen predictions is finiteness and range checking.

**It writes nothing.** Every path is opened read-only. `results/stage_f_v2/`
(the 2.0.0 authoritative root) must be absent or empty; the historical
`results/stage_f/` and `results/stage_f_freeze/` are expected to exist, are
re-verified byte-for-byte against their frozen freeze binding, and are never
written to or treated as blockers (§H.1a). No store root is ever created.

    python3 scripts/validate_stage_f_protocol_freeze.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROTOCOL_DOC = ROOT / "docs" / "STAGE_F_PROTOCOL.md"
CONSTANTS_JSON = ROOT / "docs" / "stage_f_constants.json"
PROTOCOL_VERSION = "STAGE_F_PROTOCOL/2.0.0"
SUPERSEDED_VERSION = "STAGE_F_PROTOCOL/1.0.2"
SUPERSEDED_COMMIT = "d2451ff7f719770c40cc6a3ac0fb9e3e7f0dd7dd"
SUPERSEDED_TAG = "stage-f-protocol-v1"
#: The 2.0.0 authoritative root. Advanced at F.7 because `results/stage_f/` holds
#: the 1.0.2 Attempt #5 store, which is frozen, tagged and immutable (§H.1a).
STAGE_F_STORE = ROOT / "results" / "stage_f_v2"
INTENDED_TAG = "stage-f-protocol-v2"

#: Immutable historical authority (§H.1a). Verified read-only; NEVER a blocker.
HISTORICAL_STORE = ROOT / "results" / "stage_f"
HISTORICAL_FREEZE = ROOT / "results" / "stage_f_freeze"
HISTORICAL_BINDING = HISTORICAL_FREEZE / "authoritative_run_binding.json"
HISTORICAL_RESULT_TAG = "stage-f-results-v1"

# ---------------------------------------------------------------------------
# Frozen constants, mirrored from the protocol document.
#
# `check_constant_mirroring` compares this structure against
# docs/stage_f_constants.json by FLATTENED KEY AND VALUE in both directions.
# A missing key, an extra key or an unequal value is a failure. This replaced
# the 1.0.0 substring check, which could not detect a changed value.
# ---------------------------------------------------------------------------
MIRROR: dict = {
    "protocol_version": PROTOCOL_VERSION,
    "supersedes_version": SUPERSEDED_VERSION,
    "supersedes_commit": SUPERSEDED_COMMIT,
    "protocol_doc": "docs/STAGE_F_PROTOCOL.md",
    "validator": "scripts/validate_stage_f_protocol_freeze.py",
    "intended_tag": INTENDED_TAG,
    "tag_exists": False,
    "store_root": "results/stage_f_v2/",

    "cohort": {
        "base_cohort_n": 2232,
        "base_cohort_sites": 23,
        "source_oof_rows": 49104,
        "target_rows": 2232,
        "split_registry_rows": 7700,
        "split_registry_rows_n_ge_10": 5400,
        "sites_per_n": {"5": 23, "10": 22, "20": 21, "40": 11},
        "r_split": 100,
        "common_sites": ["S1", "S7", "S8", "S9", "S10", "S14",
                         "S15", "S17", "S20", "S21", "S25"],
        "common_site_registry_rows": 4400,
        "common_site_registry_rows_n_ge_10": 3300,
    },

    "rq4": {
        "alpha_grid": [0.1, 0.2, 0.3, 0.4],
        "n_grid": [10, 20, 40],
        "s2_n_grid": [5, 10, 20, 40],
        "conformal_tolerance": 0.05,
        "nonconformity_score": "a(x, y) = 1 - p_y(x)",
        "index_rule": "k = ceil((n + 1) * (1 - alpha)); t = a_(k) if k <= n else +inf",
        "interpolation_permitted": False,
        "tie_inclusion": "non_strict",
        "methods": ["source_pooled_split", "weighted", "target_local"],
        "guarantee_labels": {
            "source_pooled_split": "EMPIRICAL_SOURCE_REFERENCE",
            "weighted": "EMPIRICAL",
            "target_local": "EXACT_WITHIN_TARGET_SITE",
        },
        "method_information_budget": {
            "source_pooled_split": "S0",
            "weighted": "S1",
            "target_local": "S2",
        },
        "exact_guarantee_permitted_for": ["target_local"],
        "weight_column": "w_final",
        "weighted_test_point_atom_included": False,
        "registered_rates": ["coverage", "avg_set_size", "singleton_rate",
                             "empty_rate", "doubleton_rate"],
        "primary_rates": ["coverage", "avg_set_size", "singleton_rate"],
        "diagnostic_rates": ["empty_rate", "doubleton_rate"],
        "n_registered_cells": 20,
    },

    "rq4_ci": {
        "site_bootstrap_b": 5000,
        "ci_level": 0.95,
        "ci_quantiles": [0.025, 0.975],
        "ci_method": "percentile",
        "quantile_method": "linear",
        "min_valid_replicates": 4750,
        "bootstrap_entropy": [20260807, 1102],
        "resampling_unit": "site",
        "required": ["rq4_site_macro_coverage",
                     "rq4_site_macro_avg_set_size",
                     "rq4_site_macro_singleton_rate",
                     "rq4_marginal_coverage"],
        "n_ci_slots": 80,
        "prohibited": [
            "rq4_per_site_value",
            "rq4_per_repeat_value",
            "rq4_worst_site_coverage",
            "rq4_site_macro_empty_rate",
            "rq4_site_macro_doubleton_rate",
            "rq4_marginal_avg_set_size",
            "rq4_marginal_singleton_rate",
            "rq4_marginal_empty_rate",
            "rq4_marginal_doubleton_rate",
            "rq4_tau",
            "rq4_degeneracy_count",
            "rq4_method_contrast",
            "rq4_alpha_contrast",
            "rq4_n_contrast",
            "svm_site_macro_bacc",
            "svm_per_site_bacc",
            "sensitivity_overall_error_ucb",
            "sensitivity_q_step_002",
            "sensitivity_r_star_shift",
            "sensitivity_common_sites",
        ],
        "hypothesis_test": False,
        "multiplicity_correction": False,
        "tost": False,
        "bootstrap_site_set_sizes": [23, 22, 21, 11],
    },

    "status": {
        "status_values": ["ESTIMABLE", "NOT_ESTIMABLE"],
        "reason_codes": [
            "SITE_NOT_ELIGIBLE_AT_N",
            "NO_EVALUATION_POINTS",
            "COVERAGE_MATCH_UNATTAINABLE",
            "CI_INSUFFICIENT_VALID_REPLICATES",
            "NO_ESTIMABLE_SITE_IN_CELL",
            "SVM_NOT_CONVERGED",
            "SVM_FOLD_FAILED",
        ],
        "reason_codes_reused_from_rq3": [
            "NO_ELIGIBLE_START",
            "CV_FOLD_NOT_ESTIMABLE",
            "QUADRATURE_DISAGREEMENT",
            "NESTED_LIKELIHOOD_INVERSION",
            "VALIDATION_MALFORMED",
        ],
        "reason_codes_reused_by": ["sens5_s1_attribution",
                                   "sens6_realized_coverage_matched"],
        "withdrawn_reason_codes": ["CONFORMAL_LEVEL_UNATTAINABLE",
                                   "SINGLE_CLASS_EVALUATION"],
        "degeneracy_codes": ["ALL_INCLUSIVE_THRESHOLD",
                             "ALL_INCLUSIVE_REALIZED",
                             "ALL_EMPTY_REALIZED"],
        "degeneracy_implies_not_estimable": False,
        "class_composition_flags": ["BOTH_CLASSES", "SINGLE_CLASS_MDD",
                                    "SINGLE_CLASS_HC"],
        "class_composition_implies_not_estimable": False,
        "class_conditional_conformal_endpoint_registered": False,
    },

    "sensitivities": {
        "n_sensitivities": 6,
        "labels": ["sens1_overall_error_ucb", "sens2_q_step_002",
                   "sens3_rstar_pm_005", "sens4_common_site",
                   "sens5_s1_attribution", "sens6_realized_coverage_matched"],
        "combat_in_mandatory_core": False,
        "sens1_alpha_ucb": 0.10,
        "sens1_bonferroni": False,
        "sens2_q_step": 0.02,
        "sens2_n_candidates": 41,
        "sens3_r_star_offsets": [-0.05, 0.05],
        "parent_strategies_label_consuming": ["target_local",
                                              "pooled_local_shrinkage"],
        "parent_strategies_zero_label": ["pooled_source",
                                         "target_unlabeled_weighted"],
        "parent_confidence_methods": ["probability", "bootstrap", "mahalanobis"],
        "cardinality": {
            "sens1_decisions": 46200,
            "sens1_evaluations": 78600,
            "sens2_decisions": 46200,
            "sens2_evaluations": 78600,
            "sens3_decisions": 92400,
            "sens3_evaluations": 157200,
            "sens4_decisions": 26400,
            "sens4_evaluations": 46200,
            "sens4_zero_label_evaluations": 19800,
            "rq3_cells": 12,
            "rq3_validation_units": 24,
            "rq3_families": 4,
            "sens6_threshold_selection_rows": 192,
        },
    },

    # §H.4.3 — corrected at F.1b. `block` is a validation-unit key field and
    # NEVER a cell key field: 12 cells x 2 blocks = 24 validation units.
    "sens56_schema": {
        "applies_to": ["sens5_s1_attribution",
                       "sens6_realized_coverage_matched"],
        "populations": ["full_sample", "mdd_only"],
        "cells_file": "cells.csv",
        "cells_logical_key": ["population", "operating_point", "method"],
        "cells_expected_rows": 12,
        "block_in_cells_logical_key": False,
        "validation_units_file": "validation_units.csv",
        "validation_units_logical_key": ["population", "operating_point",
                                         "method", "block"],
        "validation_units_expected_rows": 24,
        "families_file": "families.csv",
        "families_logical_key": ["population", "block"],
        "families_expected_rows": 4,
        "holm_family_size": 6,
        "blocks_per_population": 2,
        "validation_units_per_cell": 2,
        "blocks_by_population": {
            "full_sample": ["motion", "diagnosis"],
            "mdd_only": ["motion", "clinical"],
        },
        "blocks_carried_within_cell_record": True,
        "parent_cell_registry_size": 6,
        "parent_validation_unit": ["population", "cell", "block"],
    },

    # §B.1a — transcribed at F.1b from src/s2_inputs. NOT chosen here: the
    # column list and the tolerance are read back from the frozen helper below
    # and must equal these values.
    "gate1_oof_binding": {
        "helper": "src/s2_inputs.assert_gate1_binds_to_checksummed_s0",
        "helper_invoked_by_validator": True,
        "unledgered_file": "results/gate1/source_oof_predictions.csv",
        "ledger_bound_file": "results/rq1_s0/confidence_source_oof.csv",
        "logical_key": ["outer_fold_held_out_site", "canonical_id"],
        "exact_columns": ["outer_fold_held_out_site", "canonical_id",
                          "source_site", "y_true", "oof_pred",
                          "selected_pca", "selected_C"],
        "n_exact_columns": 7,
        "exact_comparison": "pandas.Series.equals",
        "exact_comparison_is_positional": True,
        "tolerance_columns": ["oof_prob_mdd"],
        "oof_prob_mdd_atol": 4e-16,
        "tolerance_rule": "max_abs_difference <= atol",
        "oof_prob_mdd_bit_identical_required": False,
        "row_count_equality_required": True,
        "duplicate_logical_keys_permitted": False,
        "binding_is_one_to_one": True,
        "csv_float_precision": "round_trip",
        "n_oof_rows": 49104,
        "r_star_crosscheck_atol": 1e-15,
        "r_star_crosscheck_sites": 23,
    },

    "sens6": {
        "requested_coverage_by_operating_point": {"OP50": 0.50, "OP30": 0.70},
        "operating_points": ["OP50", "OP30"],
        "candidate_set": "distinct observed confidence values of the group, ascending",
        "coverage_definition": "cov(t) = #{i : conf_i >= t} / n",
        "selection_rule": "tau = min { t in T : cov(t) <= c_star }",
        "retention_comparison": ">=",
        "abstention_comparison": "<",
        "interpolation_permitted": False,
        "randomisation_permitted": False,
        "uses_target_labels": False,
        "not_estimable_codes": ["NO_EVALUATION_POINTS",
                                "COVERAGE_MATCH_UNATTAINABLE"],
        "full_sample_n": 2222,
        "full_sample_sites": 23,
        "mdd_only_n": 484,
        "mdd_only_sites": 9,
        "worked_examples": [
            {"c_star": 0.50,
             "conf": [0.10, 0.20, 0.30, 0.30, 0.30, 0.70, 0.80, 0.90],
             "tau": 0.70, "realized_coverage": 0.375},
            {"c_star": 0.50,
             "conf": [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88],
             "tau": 0.55, "realized_coverage": 0.500},
            {"c_star": 0.70,
             "conf": [0.10, 0.10, 0.10, 0.40, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95],
             "tau": 0.40, "realized_coverage": 0.700},
            {"c_star": 0.70, "conf": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
             "tau": None, "reason_code": "COVERAGE_MATCH_UNATTAINABLE"},
        ],
    },

    "svm": {
        "estimator": "sklearn.svm.LinearSVC",
        "estimand": "site_macro_outer_loso_balanced_accuracy",
        "c_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
        "pca_candidates": [None, 50, 100, 200],
        "penalty": "l2",
        "loss": "squared_hinge",
        "dual": False,
        "class_weight": None,
        "fit_intercept": True,
        "tol": 0.0001,
        "max_iter": 2000,
        "random_state": 20260807,
        "random_state_status": "OPERATIONALLY_INERT",
        "decision_rule": "decision_function(x) > 0",
        "probability_calibration": False,
        "platt_scaling": False,
        "sample_weight": "src/nested_loso.site_equal_weights",
        "c_scale_equivalence_claimed": False,
        "forbidden_outputs": ["auc", "f1", "accuracy", "selective_risk_curve",
                              "calibrated_probability",
                              "significance_test_vs_logistic_regression"],
        # --- added at F.7; none of these changes a scientific definition ---
        "outer_split": "leave-one-site-out",
        "inner_split": "leave-one-source-site-out",
        "definition_changed_by_2_0_0": False,
        "any_fit_convergence_warning_invalidates_outer_fold": True,
        "inner_fit_counts_as_fold_lifecycle_fit": True,
        "reduced_grid_selection_permitted": False,
        "final_refit_after_inner_warning_permitted": False,
        "selection_requires_all_candidates_estimable": True,
        "n_candidates_per_fold": 20,
        "tie_break_order": ["inner_macro_bacc_desc", "C_asc",
                            "compression_strength_desc"],
        "tie_break_pca_order": [50, 100, 200, None],
        "converged_requires_all_inner_fits_clean": True,
        "converged_requires_clean_final_refit": True,
        "converged_requires_fold_estimable": True,
        "n_iter_max_is_configured_cap": True,
        "n_iter_max_is_observed_n_iter": False,
        "n_iter_max_when_refit_attempted": 2000,
        "n_iter_max_when_refit_skipped": None,
        "observed_iteration_count_persisted": False,
    },

    "consumed_artifacts": {},          # filled from CONSUMED below
    "n_consumed_artifacts": 17,
    "recomputable_artifacts": ["results/gate1/cache/fc_aal116_z_included.npz"],

    # -----------------------------------------------------------------------
    # Added at F.7 -- the 2.0.0 amendment. Representability, status, root and
    # version contracts only; no scientific definition appears here.
    # -----------------------------------------------------------------------
    "supersedes_tag": SUPERSEDED_TAG,
    "freeze_root": "results/stage_f_v2_freeze/",
    "intended_result_tag": "stage-f-results-v2",

    "historical_authority": {
        "store_root": "results/stage_f/",
        "freeze_root": "results/stage_f_freeze/",
        "protocol_tag": "stage-f-protocol-v1",
        "protocol_commit": "d2451ff7f719770c40cc6a3ac0fb9e3e7f0dd7dd",
        "protocol_version": "STAGE_F_PROTOCOL/1.0.2",
        "result_tag": "stage-f-results-v1",
        "result_commit": "90bc5ddca720fac080698fb1872c59e48595f9cd",
        "freeze_binding_file":
            "results/stage_f_freeze/authoritative_run_binding.json",
        "freeze_store_content_digest":
            "50d3e275e668642a6a83089a70387ee261d6e2aae29d4ed5bfd7d3b4be8769a4",
        "immutable": True,
        "mechanically_complete_and_cryptographically_valid": True,
        "scientifically_superseded": True,
        "is_final_stage_f_scientific_authority": False,
        "blocks_new_authority": False,
        "writable_by_new_authority": False,
        "readable_as_scientific_input": False,
        # --- byte-level binding to the TAGGED authority [§0.4 correction] ---
        "result_tag_object": "c3300d10896b094021cd63297868b249d240169b",
        "freeze_files": {
            "authoritative_run_binding.json":
                "8d167aed8a37b398f6d72b468a176aef32547676524b4a937b681f97698e52cb",
            "RESULT_CHECKSUMS.sha256":
                "a3f7e448b9f46a3baa91c04afedba901300ff5bbfa7eb91bb3884f44a478ce0a",
        },
        "n_freeze_files": 2,
        "freeze_files_tracked_in_git": True,
        "store_anchor_provenance_sha256":
            "468e939fa86d131afb2ea381c4e586f8ebcf086b8ad5fabe04c8428ed5f2bda7",
        "store_anchor_result_checksums_sha256":
            "99b677b0cd4145399416b7e6e95a28c941fde1b0e4b1aab7c12fd71414749e5a",
        "store_anchor_completion_sha256":
            "bdf01f9356902b969b73574f60a723f6e36f5b254f39a094d23345615eeeb1e6",
        "store_content_n_files": 35,
        "byte_drift_is_fail": True,
    },

    # -----------------------------------------------------------------------
    # The two per-site ConvergenceWarning representations, §F.7.1 / §F.7.1a.
    # Added at the §0.4 draft-audit correction. Both are NOT_ESTIMABLE /
    # SVM_NOT_CONVERGED; they differ ONLY in which fields carry a value.
    # -----------------------------------------------------------------------
    "svm_per_site_warning_representation": {
        "file": "svm/per_site_bacc.csv",
        "representations_are_distinct": True,
        "new_status_or_reason_code_introduced": False,
        "cohort_fields_always_persisted": ["n_acquisitions", "n_mdd", "n_hc"],
        "inner_grid_warning": {
            "section": "F.7.1",
            "lawful_selection_occurred": False,
            "status": "NOT_ESTIMABLE",
            "reason_code": "SVM_NOT_CONVERGED",
            "bacc": None,
            "selected_pca": None,
            "selected_C": None,
            "inner_macro_bacc": None,
            "converged": False,
            "n_iter_max": None,
            "final_refit_attempted": False,
            "selection_metadata_retained": False,
        },
        # The §F.7.1a row freezes the RETAINEDNESS of the three selection
        # fields, not a number: the value is per-fold. They are STRINGS here,
        # never null -- that is exactly what separates §F.7.1a from §F.7.1,
        # and it is why deleting one of them can never read as "null".
        "retained_value_token": "RETAINED_SELECTED_VALUE",
        "final_refit_warning": {
            "section": "F.7.1a",
            "lawful_selection_occurred": True,
            "all_inner_candidate_rows_estimable": True,
            "status": "NOT_ESTIMABLE",
            "reason_code": "SVM_NOT_CONVERGED",
            "bacc": None,
            "selected_pca": "RETAINED_SELECTED_VALUE",
            "selected_C": "RETAINED_SELECTED_VALUE",
            "inner_macro_bacc": "RETAINED_SELECTED_VALUE",
            "selected_pca_retained": True,
            "selected_C_retained": True,
            "inner_macro_bacc_retained": True,
            "selected_inner_score_finite": True,
            "converged": False,
            "n_iter_max": 2000,
            "final_refit_attempted": True,
            "selection_metadata_retained": True,
            "retained_metadata_usable_as_result": False,
            "appears_in_fold_selection_csv": False,
        },
    },

    "svm_fold_selection": {
        "file": "svm/fold_selection.csv",
        "logical_key": ["held_out_site", "pca", "C"],
        "fields": ["inner_macro_bacc", "status", "reason_code"],
        "expected_rows": 460,
        "cardinality_formula":
            "n_sites * n_pca_candidates * n_c_grid = 23 * 4 * 5",
        "all_rows_mandatory": True,
        "redundant_candidate_status_fields_registered": False,
        "uses_existing_status_vocabulary": True,
        "estimable_status": "ESTIMABLE",
        "not_estimable_status": "NOT_ESTIMABLE",
        "reason_code_convergence": "SVM_NOT_CONVERGED",
        "reason_code_expected_failure": "SVM_FOLD_FAILED",
        "reason_code_when_estimable": None,
        "null_score_without_status_permitted": False,
        "all_candidates_evaluated_when_safe": True,
        "diagnostic_scores_retained_for_successful_candidates": True,
        "diagnostic_scores_usable_for_selection_in_invalidated_fold": False,
        "invariant_failures_converted_to_candidate_rows": False,
    },

    "svm_fold_failed_scope": {
        "code": "SVM_FOLD_FAILED",
        "is_catch_all": False,
        "admits_only":
            "narrowly classified expected estimator/backend failure",
        "must_not_absorb": [
            "exception_catch_all", "programming_errors", "invariant_failures",
            "schema_errors", "contract_or_registry_invariant_failures",
            "AssertionError", "KeyError", "TypeError", "IndexError",
            "RuntimeError", "ValueError", "invalid_shapes",
            "nonfinite_frozen_inputs", "missing_classes",
            "off_grid_parameters", "impossible_internal_states",
            "MemoryError", "KeyboardInterrupt", "SystemExit",
            "leakage_or_data_contract_violations"],
        "unexpected_errors_fail_hard": True,
        "substitutes_for_svm_not_converged": False,
        # --- the POSITIVE allow-list [§0.4 draft-audit correction] ---------
        "allowed_exception_types": ["FloatingPointError",
                                    "numpy.linalg.LinAlgError"],
        "n_allowed_exception_types": 2,
        "n_must_not_absorb": 20,
        "allow_list_is_closed_and_exhaustive": True,
        "type_match_rule": "exact_type_identity",
        # `type(exc) is T`, never `isinstance(exc, T)` -- transcribed from
        # §G.2a condition 1 so the semantics, not just the words, are checked.
        "exact_type_predicates": ["type(exc) is FloatingPointError",
                                  "type(exc) is numpy.linalg.LinAlgError"],
        "isinstance_matching_permitted": False,
        "subclass_widening_permitted": False,
        "broad_exception_catching_permitted": False,
        # scipy.linalg.LinAlgError IS numpy.linalg.LinAlgError -- an
        # identity-equivalent explanatory note, never a third semantic type.
        "identity_equivalent_aliases": ["scipy.linalg.LinAlgError"],
        "aliases_add_no_semantic_type": True,
        "n_semantic_failure_types": 2,
        "requires_estimator_backend_boundary": True,
        "estimator_backend_boundary":
            "the frozen LinearSVC.fit(...) call for one candidate and the "
            "decision_function(...) call on that same fitted estimator",
        "estimator_backend_boundary_inside": ["LinearSVC.fit",
                                              "decision_function"],
        "estimator_backend_boundary_outside": [
            "Data assembly", "the scaler and PCA transforms",
            "weight construction", "scoring", "selection", "aggregation",
            "schema handling", "persistence"],
        "outside_boundary_is_hard_failure": True,
        "linalgerror_subclasses_valueerror": True,
        "bare_valueerror_permitted": False,
        "estimator_settings_altered_to_avoid_failures": False,
    },

    "sens1_score_type_correction": {
        "scientific_definition_changed": False,
        "numeric_recomputation_authorized": False,
        "implemented_by_this_version": False,
        "actual_functional": "certification.overall_error_upper",
        "persisted_label_in_1_0_2": "exact_classwise_clopper_pearson_ucb",
        "required_label": "exact_overall_error_clopper_pearson_ucb",
    },

    "amendment_2_0_0": {
        "trigger": "independently audited section F.7 / section H.3 "
                   "representability contradiction found after Attempt #5",
        "attempt_number": 5,
        "scientific_svm_definition_changed": False,
        "fail_closed_rule_weakened": False,
        "solver_or_grid_or_tolerance_or_max_iter_changed": False,
        "is_post_result_relaxation": False,
        "scope": ["representability", "status", "root", "version"],
        "runtime_implementation_changed": False,
        "formal_run_authorized": False,
        "stage_g_authorized": False,
        "new_result_tag_created": False,
        # --- the independent Protocol-v2 DRAFT audit [§0.4] ----------------
        "draft_audit_verdict": "FAIL",
        "draft_audit_critical_findings": 4,
        "draft_audit_non_critical_findings": 2,
        "draft_corrections_are_scientific_changes": False,
        "old_implementation_failures_expected": True,
        "protocol_pinning_weakened": False,
        # --- the independent re-audit of the CORRECTED draft [§0.5] --------
        # It accepted the scientific / specification content and returned FAIL
        # on ONE blocker: incomplete protocol-validator enforcement.
        "validator_coverage_audit_verdict": "FAIL",
        "validator_coverage_audit_critical_findings": 1,
        "validator_coverage_audit_accepted_protocol_content": True,
        "validator_hardening_is_scientific_change": False,
        "validator_hardening_changed_runtime": False,
    },
}

#: path -> (SHA256, ledger file or None, name inside that ledger)
CONSUMED = {
    "results/data_contract/analysis_manifest_v1.csv": (
        "bb582babe5a46e42503f6682140dfc7b0e7834fa813ca5dbb8bdae42dc6c1ff2",
        None, None),
    "results/data_contract/mean_fd_per_subject.csv": (
        "7fc40ba94e2709b8d3b04d512cf2ffc94dca7c501d551b0f6a90cd1477cd1ebb",
        None, None),
    "results/gate1/cache/fc_aal116_z_included.npz": (
        "0598371c9b0a29f29004d0480739b631e18a971a0ad77bfefd888c7e3370d992",
        None, None),
    "results/gate1/source_full_coverage_error_by_fold.csv": (
        "7ec3cb7fb66d00caa71d26eb8af93a682ef5617ce277ae2660b1fe8aecc64384",
        None, None),
    "results/gate1/source_oof_predictions.csv": (
        "6ac19a48f2fcdeb9ab871d5799d356d81c43939f62620bc5c8d8eb026f0672a3",
        None, None),
    "results/rq1_s0/confidence_source_oof.csv": (
        "03a87f11cdb95be50e3ca43bd8217f27e26ed6bb25731a6dd6fc9a8733d46390",
        "results/rq1_s0/RESULT_CHECKSUMS.sha256", "confidence_source_oof.csv"),
    "results/rq1_s0/risk_coverage_all_q.csv": (
        "dd5a266adf632bc889f8282d02adc65dc6d793af94bb6f42e6d330b6be3adb8e",
        "results/rq1_s0/RESULT_CHECKSUMS.sha256", "risk_coverage_all_q.csv"),
    "results/rq1_s0/confidence_target.csv": (
        "99cafb3a90bfb9e8c339458407b56ecf4f666cde4f97dfb9a7c88b28653d6b37",
        "results/rq1_s0/RESULT_CHECKSUMS.sha256", "confidence_target.csv"),
    "results/rq1_s0/thresholds_source_oof.csv": (
        "78f332e5255994d4069f9783dff25d5bd53d3cb54bcc3039287011dec3793a9d",
        "results/rq1_s0/RESULT_CHECKSUMS.sha256", "thresholds_source_oof.csv"),
    "results/rq1_s0/op50_summary.csv": (
        "31601e0ad3238c2154ca19f3985a861825d91d20fb42b4eb4f9ef21f90ed6b94",
        "results/rq1_s0/RESULT_CHECKSUMS.sha256", "op50_summary.csv"),
    "results/rq1_s0/op30_summary.csv": (
        "3d4e2b7248a7581691e1ee0bfc7480f325371fbdbbc9fbb4d455d0b03730b024",
        "results/rq1_s0/RESULT_CHECKSUMS.sha256", "op30_summary.csv"),
    "results/rq1_s1/density_weights.csv": (
        "2f6c464487867ea94e1205182332b9f106aabcb283ae4670633f76689bd76db3",
        "results/rq1_s1/RESULT_CHECKSUMS.sha256", "density_weights.csv"),
    "results/rq1_s1/thresholds_weighted_source.csv": (
        "a908c08c608b374671589c713d9a0a79c90c673839d2d7cd2a735e1ca4fdf57f",
        "results/rq1_s1/RESULT_CHECKSUMS.sha256",
        "thresholds_weighted_source.csv"),
    "results/rq1_s1/weighted_source_risk.csv": (
        "b414312a6ddf42dae50f774c6f3eb51d1e879d00eddd1cd54f6fb06616f7e83a",
        "results/rq1_s1/RESULT_CHECKSUMS.sha256", "weighted_source_risk.csv"),
    "results/rq1_s1/ess_by_site.csv": (
        "74a6b54852a7ef74e1bea3fb9124df0be9ee00cdea6e84ba050e92177fb8d646",
        "results/rq1_s1/RESULT_CHECKSUMS.sha256", "ess_by_site.csv"),
    "results/rq1_s1/selected_domain_C.csv": (
        "80141d3aa49f44471a9f729f8778d12cbe3f9e7799e7b3f064c7588a0373739c",
        "results/rq1_s1/RESULT_CHECKSUMS.sha256", "selected_domain_C.csv"),
    "results/rq2_s2/production_split_registry.csv": (
        "9cdf4bd8cd6a273fb17332afbe4946f79263fcb63713d2910cd03694dade03f4",
        None, None),
}
MIRROR["consumed_artifacts"] = {k: v[0] for k, v in CONSUMED.items()}

# Convenience aliases used throughout the checks.
C = MIRROR["cohort"]
Q4 = MIRROR["rq4"]
CI = MIRROR["rq4_ci"]
ST = MIRROR["status"]
SENS = MIRROR["sensitivities"]
S6 = MIRROR["sens6"]
SVM = MIRROR["svm"]
FS = MIRROR["svm_fold_selection"]
FF = MIRROR["svm_fold_failed_scope"]
WREP = MIRROR["svm_per_site_warning_representation"]
HIST = MIRROR["historical_authority"]
AMEND = MIRROR["amendment_2_0_0"]
SENS1FIX = MIRROR["sens1_score_type_correction"]
S56 = MIRROR["sens56_schema"]
G1 = MIRROR["gate1_oof_binding"]

ALPHA_GRID = tuple(Q4["alpha_grid"])
RQ4_N_GRID = tuple(Q4["n_grid"])
COMMON_SITES = tuple(C["common_sites"])

FAILURES: list[str] = []
NOTES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)


def ok(msg: str) -> None:
    NOTES.append(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ledger(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        out[rel.strip()] = digest
    return out


def flatten(obj, prefix: str = "") -> dict:
    """Flatten a nested dict to {dotted.path: value}; lists become tuples."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        out[prefix] = tuple(
            tuple(sorted(flatten(e).items())) if isinstance(e, (dict, list)) else e
            for e in obj)
    else:
        out[prefix] = obj
    return out


# ---------------------------------------------------------------------------
# 1. consumed artifacts  (audit finding 3)
# ---------------------------------------------------------------------------
def check_artifacts() -> None:
    n_ledger = 0
    for rel, (digest, ledger_rel, ledger_name) in CONSUMED.items():
        path = ROOT / rel
        if not path.is_file():
            fail(f"consumed artifact missing: {rel}")
            continue
        got = sha256(path)
        if got != digest:
            fail(f"{rel}: SHA256 {got} != frozen {digest}")
            continue
        if ledger_rel is not None:
            ledger = read_ledger(ROOT / ledger_rel)
            if ledger.get(ledger_name) != digest:
                fail(f"{rel}: not bound to {ledger_rel} (entry "
                     f"{ledger.get(ledger_name)!r})")
                continue
            n_ledger += 1
    if FAILURES:
        return
    if len(CONSUMED) != MIRROR["n_consumed_artifacts"]:
        fail(f"consumed registry has {len(CONSUMED)} entries != "
             f"{MIRROR['n_consumed_artifacts']}")
        return
    ok(f"all {len(CONSUMED)} consumed artifacts verified by SHA256 "
       f"({n_ledger} additionally ledger-bound)")

    # The Gate-1 OOF binding is verified BY VALUE in check_gate1_oof_binding
    # below -- shared-column presence and equal row counts were the 1.0.1 weak
    # check the F.1a re-audit rejected, and are not repeated here.
    e_full = pd.read_csv(ROOT / "results/gate1/source_full_coverage_error_by_fold.csv")
    rc = pd.read_csv(ROOT / "results/rq1_s0/risk_coverage_all_q.csv",
                     usecols=["held_out_site", "r_star_s"])
    r_star = {str(r.site): float(r.e_full_minus_s) - 0.10
              for r in e_full.itertuples()}
    bad = [s for s, v in rc.groupby(rc["held_out_site"].astype(str))["r_star_s"]
           .first().items() if abs(float(v) - r_star.get(str(s), float("nan"))) > 1e-15]
    if bad:
        fail(f"gate1 e_full does not reproduce r_star_s for sites {bad[:5]}")
        return
    ok(f"gate1 source_full_coverage_error_by_fold.csv reproduces r*_s = "
       f"e_full,-s - 0.10 for all {len(r_star)} folds within 1e-15")

    # the SVM feature cache: digest AND content key, never path trust
    import contracts
    import fc_features as fcf
    z = np.load(ROOT / "results/gate1/cache/fc_aal116_z_included.npz",
                allow_pickle=False)
    cohort = fcf.load_cohort()
    ids = list(cohort["feature_source_id"].to_numpy().astype("U32"))
    if list(z["feature_source_id"]) != ids:
        fail("SVM feature cache key does not match the frozen cohort ordering")
        return
    if z["X"].shape != (C["base_cohort_n"], contracts.N_FEATURES):
        fail(f"SVM feature cache shape {z['X'].shape} != "
             f"({C['base_cohort_n']}, {contracts.N_FEATURES})")
        return
    ok(f"SVM feature cache: digest pinned, content key matches the frozen "
       f"cohort ordering, shape {z['X'].shape}")


# ---------------------------------------------------------------------------
# 1a. the Gate-1 OOF binding, BY VALUE  (F.1a re-audit finding 3)
#
# The authority is the frozen upstream helper
# `src/s2_inputs.assert_gate1_binds_to_checksummed_s0`. It is INVOKED directly,
# read-only, on the real frozen files. `_gate1_binding` below reproduces its
# checks exactly -- same column set, same `pandas.Series.equals` semantics, same
# 4e-16 tolerance, same row-count requirement -- so that negative synthetic
# self-tests can prove those checks actually reject bad input. The reproduction
# is cross-checked against the genuine helper's own reported statistics on the
# real files; it never replaces it and never weakens it.
#
# Two hardened requirements are ADDED on top of the frozen helper, never
# substituted for it: the logical key must be duplicate-free in both tables, and
# the binding must be one-to-one and row-aligned.
# ---------------------------------------------------------------------------
class Gate1BindingError(AssertionError):
    """Raised by the in-memory reproduction of the frozen binding rule."""


def _gate1_binding(pred: pd.DataFrame, conf: pd.DataFrame, *,
                   exact_columns, ulp_columns, atol: float,
                   logical_key) -> dict:
    """Exact reproduction of the frozen helper, plus the two added checks."""
    shared = tuple(exact_columns) + tuple(ulp_columns)

    # (helper) every bound column must exist in the ledger-bound table
    missing = [c for c in shared if c not in conf.columns]
    if missing:
        raise Gate1BindingError(
            f"the checksummed S0 confidence table lacks {missing}")
    missing = [c for c in shared if c not in pred.columns]
    if missing:
        raise Gate1BindingError(f"the Gate-1 OOF table lacks {missing}")

    # (helper) equal row counts
    if len(pred) != len(conf):
        raise Gate1BindingError(
            f"Gate 1 OOF has {len(pred)} rows, S0 confidence has {len(conf)}")

    # (added) the logical key must be duplicate-free in both tables
    for label, frame in (("Gate-1 OOF", pred), ("S0 confidence", conf)):
        dup = int(frame.duplicated(subset=list(logical_key)).sum())
        if dup:
            raise Gate1BindingError(
                f"{label} carries {dup} duplicate logical key(s) "
                f"{tuple(logical_key)}; the binding cannot be one-to-one")

    # (added) the binding must be one-to-one AND row-aligned
    kp = list(map(tuple, pred[list(logical_key)].to_numpy()))
    kc = list(map(tuple, conf[list(logical_key)].to_numpy()))
    if set(kp) != set(kc):
        raise Gate1BindingError(
            f"the two tables do not describe the same key set "
            f"(only_gate1={len(set(kp) - set(kc))}, "
            f"only_s0={len(set(kc) - set(kp))})")
    if kp != kc:
        n_bad = sum(1 for a, b in zip(kp, kc) if a != b)
        raise Gate1BindingError(
            f"the two tables are not row-aligned: {n_bad} positions carry "
            f"different logical keys")

    # (helper) the exact columns are bit-identical, positionally and dtype-aware
    for col in exact_columns:
        if not pred[col].equals(conf[col]):
            raise Gate1BindingError(
                f"Gate 1 OOF column {col!r} differs from the checksummed S0 "
                "confidence table; the two are not the same frozen object")

    # (helper) the probability column agrees within the frozen tolerance
    worst = 0.0
    for col in ulp_columns:
        delta = float(np.abs(pred[col].to_numpy(float)
                             - conf[col].to_numpy(float)).max())
        worst = max(worst, delta)
        if delta > atol:
            raise Gate1BindingError(
                f"Gate 1 OOF column {col!r} disagrees with the checksummed S0 "
                f"confidence table by {delta:g}, beyond {atol:g}")
    return {"n_rows": int(len(pred)), "max_abs_delta": worst}


def _synthetic_gate1_frames():
    """Four-row in-memory frames in the frozen schema. No cohort value is used."""
    frame = pd.DataFrame({
        "outer_fold_held_out_site": ["S1", "S1", "S2", "S2"],
        "canonical_id": ["A-1", "A-2", "B-1", "B-2"],
        "source_site": ["S3", "S4", "S3", "S4"],
        "y_true": [1, 0, 1, 0],
        "oof_pred": [1, 0, 0, 1],
        "selected_pca": [100, 100, 200, 200],
        "selected_C": [1.0, 1.0, 0.1, 0.1],
        "oof_prob_mdd": [0.25, 0.75, 0.5, 0.125],
    })
    return frame.copy(), frame.copy()


def _rejects(pred, conf, **kw) -> bool:
    try:
        _gate1_binding(pred, conf, **kw)
    except Gate1BindingError:
        return True
    return False


def check_gate1_oof_binding() -> None:
    import s2_inputs as si

    # 1. the frozen helper's own constants still equal the §B.1a transcription
    if tuple(si.GATE1_OOF_EXACT_COLUMNS) != tuple(G1["exact_columns"]):
        fail(f"src/s2_inputs.GATE1_OOF_EXACT_COLUMNS "
             f"{tuple(si.GATE1_OOF_EXACT_COLUMNS)} != the protocol's "
             f"{tuple(G1['exact_columns'])}")
        return
    if len(si.GATE1_OOF_EXACT_COLUMNS) != G1["n_exact_columns"]:
        fail(f"the frozen helper binds {len(si.GATE1_OOF_EXACT_COLUMNS)} exact "
             f"columns, not {G1['n_exact_columns']}")
        return
    if tuple(si.GATE1_OOF_ULP_COLUMNS) != tuple(G1["tolerance_columns"]):
        fail(f"src/s2_inputs.GATE1_OOF_ULP_COLUMNS "
             f"{tuple(si.GATE1_OOF_ULP_COLUMNS)} != "
             f"{tuple(G1['tolerance_columns'])}")
        return
    if si.GATE1_OOF_ULP_ATOL != G1["oof_prob_mdd_atol"]:
        fail(f"src/s2_inputs.GATE1_OOF_ULP_ATOL {si.GATE1_OOF_ULP_ATOL!r} != "
             f"the protocol's frozen {G1['oof_prob_mdd_atol']!r}")
        return
    if si.R_STAR_CROSSCHECK_ATOL != G1["r_star_crosscheck_atol"]:
        fail(f"src/s2_inputs.R_STAR_CROSSCHECK_ATOL "
             f"{si.R_STAR_CROSSCHECK_ATOL!r} != "
             f"{G1['r_star_crosscheck_atol']!r}")
        return
    ok(f"the frozen helper {G1['helper']} binds exactly the "
       f"{G1['n_exact_columns']} transcribed bit-identical columns "
       f"{tuple(G1['exact_columns'])}, with {tuple(G1['tolerance_columns'])} "
       f"tolerance-bound at {G1['oof_prob_mdd_atol']:g}")

    # 2. invoke the GENUINE helper on the real frozen files, read-only
    try:
        report = si.assert_gate1_binds_to_checksummed_s0()
    except Exception as exc:                                   # noqa: BLE001
        fail(f"the genuine frozen Gate-1 binding helper rejected the frozen "
             f"upstream: {exc}")
        return
    observed = float(report["max_ulp_delta"])
    if report["n_exact_columns"] != G1["n_exact_columns"] or \
            report["n_ulp_columns"] != len(G1["tolerance_columns"]):
        fail(f"the helper reported {report['n_exact_columns']} exact / "
             f"{report['n_ulp_columns']} tolerance columns, not "
             f"{G1['n_exact_columns']} / {len(G1['tolerance_columns'])}")
        return
    if int(report["n_oof_rows"]) != G1["n_oof_rows"]:
        fail(f"the helper bound {report['n_oof_rows']} OOF rows, not "
             f"{G1['n_oof_rows']}")
        return
    if not (observed <= G1["oof_prob_mdd_atol"]):
        fail(f"observed max |delta oof_prob_mdd| {observed:g} exceeds the "
             f"frozen {G1['oof_prob_mdd_atol']:g}")
        return
    if int(report["n_r_star_crosscheck_rows"]) != G1["r_star_crosscheck_sites"]:
        fail(f"the r* cross-check resolved "
             f"{report['n_r_star_crosscheck_rows']} sites, not "
             f"{G1['r_star_crosscheck_sites']}")
        return
    ok(f"the GENUINE frozen helper {G1['helper']} passes on the frozen "
       f"upstream: {report['n_oof_rows']} rows, {G1['n_exact_columns']} "
       f"bit-identical columns, observed max |delta oof_prob_mdd| = "
       f"{observed:.6e} <= {G1['oof_prob_mdd_atol']:g}, r* cross-check over "
       f"{report['n_r_star_crosscheck_rows']} sites")

    # 3. the in-memory reproduction agrees with the genuine helper on the real
    #    files, and additionally enforces duplicate-freeness and one-to-one
    #    row alignment
    kw = dict(exact_columns=tuple(G1["exact_columns"]),
              ulp_columns=tuple(G1["tolerance_columns"]),
              atol=G1["oof_prob_mdd_atol"],
              logical_key=tuple(G1["logical_key"]))
    pred = pd.read_csv(ROOT / G1["unledgered_file"], float_precision="round_trip")
    conf = pd.read_csv(ROOT / G1["ledger_bound_file"], float_precision="round_trip")
    try:
        mine = _gate1_binding(pred, conf, **kw)
    except Gate1BindingError as exc:
        fail(f"the reproduced binding rule rejected the frozen upstream, which "
             f"the genuine helper accepted: {exc}")
        return
    if mine["n_rows"] != int(report["n_oof_rows"]) or \
            mine["max_abs_delta"] != observed:
        fail(f"the reproduction disagrees with the genuine helper "
             f"({mine} vs rows={report['n_oof_rows']}, delta={observed!r})")
        return
    ok(f"the reproduced rule agrees with the genuine helper exactly "
       f"(same {mine['n_rows']} rows, identical max delta "
       f"{mine['max_abs_delta']:.6e}) and additionally certifies the logical "
       f"key {tuple(G1['logical_key'])} duplicate-free and one-to-one "
       f"row-aligned in both tables")

    # 4. negative synthetic self-tests -- purely in memory, no cohort value, no
    #    result written. Each proves the rule REJECTS a specific defect.
    caught = {}

    p, c = _synthetic_gate1_frames()
    if _rejects(p, c, **kw):
        fail("the binding rule rejects two identical synthetic tables")
        return

    for col in G1["exact_columns"]:
        p, c = _synthetic_gate1_frames()
        s = c[col]
        c.loc[c.index[0], col] = (s.iloc[1] if s.iloc[1] != s.iloc[0]
                                  else ("MUTATED" if s.dtype == object
                                        else s.iloc[0] + 1))
        if not _rejects(p, c, **kw):
            fail(f"the binding rule does NOT reject a mutation of the exact "
                 f"column {col!r}")
            return
    caught["exact_column_mutation"] = len(G1["exact_columns"])

    p, c = _synthetic_gate1_frames()
    c.loc[c.index[0], "oof_prob_mdd"] += 1e-15
    delta_big = abs(float(c["oof_prob_mdd"].iloc[0]) - float(p["oof_prob_mdd"].iloc[0]))
    if delta_big <= G1["oof_prob_mdd_atol"]:
        fail(f"the synthetic over-tolerance perturbation realized {delta_big:g}, "
             f"which is not above {G1['oof_prob_mdd_atol']:g}")
        return
    if not _rejects(p, c, **kw):
        fail(f"the binding rule does NOT reject an oof_prob_mdd perturbation of "
             f"{delta_big:g} > {G1['oof_prob_mdd_atol']:g}")
        return
    caught["oof_prob_mdd_over_tolerance"] = delta_big

    p, c = _synthetic_gate1_frames()
    c.loc[c.index[0], "oof_prob_mdd"] += 2e-16
    delta_small = abs(float(c["oof_prob_mdd"].iloc[0]) - float(p["oof_prob_mdd"].iloc[0]))
    if delta_small > G1["oof_prob_mdd_atol"] or delta_small == 0.0:
        fail(f"the synthetic within-tolerance perturbation realized "
             f"{delta_small:g}, which is not a usable probe of the tolerance")
        return
    if _rejects(p, c, **kw):
        fail(f"the binding rule rejects an oof_prob_mdd difference of "
             f"{delta_small:g} <= {G1['oof_prob_mdd_atol']:g}; the tolerance is "
             f"not the frozen one")
        return
    caught["oof_prob_mdd_within_tolerance_accepted"] = delta_small

    p, c = _synthetic_gate1_frames()
    p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    c = pd.concat([c, c.iloc[[0]]], ignore_index=True)
    if not _rejects(p, c, **kw):
        fail("the binding rule does NOT reject a duplicated logical key even "
             "though both tables were duplicated identically")
        return
    caught["duplicate_logical_key"] = True

    p, c = _synthetic_gate1_frames()
    c = c.iloc[[1, 0, 2, 3]].reset_index(drop=True)
    if not _rejects(p, c, **kw):
        fail("the binding rule does NOT reject a row misalignment")
        return
    caught["row_misalignment"] = True

    p, c = _synthetic_gate1_frames()
    if not _rejects(p, c.iloc[:3].reset_index(drop=True), **kw):
        fail("the binding rule does NOT reject a row-count mismatch")
        return
    caught["row_count_mismatch"] = True

    p, c = _synthetic_gate1_frames()
    if not _rejects(p, c.drop(columns=["oof_prob_mdd"]), **kw):
        fail("the binding rule does NOT reject a missing bound column")
        return
    caught["missing_bound_column"] = True

    ok("negative synthetic self-tests: the binding rule rejects a mutation in "
       "each of the 7 exact columns, an oof_prob_mdd perturbation of "
       f"{caught['oof_prob_mdd_over_tolerance']:.3e} > "
       f"{G1['oof_prob_mdd_atol']:g}, a duplicated logical key, a row "
       "misalignment, a row-count mismatch and a missing bound column, while "
       f"accepting a difference of "
       f"{caught['oof_prob_mdd_within_tolerance_accepted']:.3e}. No cohort "
       "value was used and nothing was written")


# ---------------------------------------------------------------------------
# 1b. the SENS-5/6 key and cardinality registry  (F.1a re-audit finding 1)
# ---------------------------------------------------------------------------
_KEY_CELL_RE = re.compile(r"`([^`]+)`")


def _parse_doc_keys(doc: str) -> dict:
    """Parse the §H.4.2 SENS-5/6 rows into {(sens, file): (key fields...)}.

    Structured parse of the markdown table: the row is split on `|`, the third
    field is the logical key, backticks and bold markers are stripped and the
    key is split on commas into an ordered tuple. Never a substring test.

    The cardinality *formula* column legitimately contains `|` (set-size bars
    such as `|CELL_REGISTRY|`), so the value is taken from the LAST field rather
    than by counting columns.
    """
    out = {}
    for line in doc.splitlines():
        if not line.startswith("| SENS-5 |") and not line.startswith("| SENS-6 |"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 5:
            continue
        sens, path, key_field, value = parts[0], parts[1], parts[2], parts[-1]
        fname = path.strip("`").split("/")[-1]
        if fname not in ("cells.csv", "validation_units.csv", "families.csv"):
            continue
        m = _KEY_CELL_RE.search(key_field)
        if not m:
            out[(sens, fname)] = (None, None)
            continue
        fields = tuple(f.strip() for f in m.group(1).split(",") if f.strip())
        digits = re.findall(r"\*\*(\d+)", value)
        n = int(digits[0]) if digits else None
        out[(sens, fname)] = (fields, n)
    return out


def check_sens56_schema() -> None:
    import rq3_protocol as rp

    # 1. the parent representation, re-derived rather than asserted
    if tuple(rp.VALIDATION_UNIT) != tuple(S56["parent_validation_unit"]):
        fail(f"rq3_protocol.VALIDATION_UNIT {tuple(rp.VALIDATION_UNIT)} != "
             f"{tuple(S56['parent_validation_unit'])}")
        return
    if len(rp.CELL_REGISTRY) != S56["parent_cell_registry_size"]:
        fail(f"rq3_protocol.CELL_REGISTRY has {len(rp.CELL_REGISTRY)} entries, "
             f"not {S56['parent_cell_registry_size']}")
        return
    # the parent cell is exactly (operating_point, method): the registry is the
    # full product of the two frozen axes, in the frozen order
    product = tuple((op, m) for op in rp.OPERATING_POINTS
                    for m in rp.CONFIDENCE_METHODS)
    if tuple(tuple(c) for c in rp.CELL_REGISTRY) != product:
        fail("the RQ3 cell registry is not the (operating_point x method) "
             "product, so the SENS-5/6 cell key cannot be derived from it")
        return
    ok(f"the parent cell is exactly (operating_point, method) — "
       f"{len(rp.OPERATING_POINTS)} x {len(rp.CONFIDENCE_METHODS)} = "
       f"{len(rp.CELL_REGISTRY)} registry entries — and the parent validation "
       f"unit is {tuple(rp.VALIDATION_UNIT)}")

    # 2. blocks per population, from BLOCK_DF, and exactly two per population
    blocks: dict = {}
    for (pop, block) in rp.BLOCK_DF:
        blocks.setdefault(pop, []).append(block)
    if sorted(blocks) != sorted(S56["populations"]):
        fail(f"rq3_protocol.BLOCK_DF covers populations {sorted(blocks)}, not "
             f"{sorted(S56['populations'])}")
        return
    if {p: sorted(b) for p, b in blocks.items()} != \
            {p: sorted(b) for p, b in S56["blocks_by_population"].items()}:
        fail(f"the frozen blocks per population {blocks} differ from the "
             f"registry {S56['blocks_by_population']}")
        return
    sizes = {p: len(b) for p, b in blocks.items()}
    if set(sizes.values()) != {S56["blocks_per_population"]}:
        fail(f"the number of blocks per population is {sizes}, not uniformly "
             f"{S56['blocks_per_population']}; the 12 x 2 = 24 identity fails")
        return
    ok(f"every population carries exactly {S56['blocks_per_population']} "
       f"blocks ({blocks}); the two scientific blocks are unchanged")

    # 3. the three cardinalities, recomputed
    n_cells = len(S56["populations"]) * len(rp.CELL_REGISTRY)
    n_units = n_cells * S56["blocks_per_population"]
    n_families = len(rp.HOLM_FAMILIES)
    if n_cells != S56["cells_expected_rows"]:
        fail(f"recomputed SENS-5/6 cells {n_cells} != "
             f"{S56['cells_expected_rows']}")
        return
    if n_units != S56["validation_units_expected_rows"] or \
            n_units != rp.N_VALIDATION_UNITS:
        fail(f"recomputed validation units {n_units} != registered "
             f"{S56['validation_units_expected_rows']} / parent "
             f"{rp.N_VALIDATION_UNITS}")
        return
    if n_units // n_cells != S56["validation_units_per_cell"]:
        fail(f"{n_units // n_cells} validation units per cell, not "
             f"{S56['validation_units_per_cell']}")
        return
    if n_families != S56["families_expected_rows"]:
        fail(f"recomputed Holm families {n_families} != "
             f"{S56['families_expected_rows']}")
        return
    if rp.HOLM_FAMILY_SIZE != S56["holm_family_size"] or \
            any(int(f[2]) != S56["holm_family_size"] for f in rp.HOLM_FAMILIES):
        fail(f"the Holm family size is not uniformly {S56['holm_family_size']}")
        return
    ok(f"recomputed from src/rq3_protocol: {n_cells} cells, "
       f"{n_cells} x {S56['blocks_per_population']} = {n_units} validation "
       f"units (= N_VALIDATION_UNITS), {n_families} Holm families of "
       f"{rp.HOLM_FAMILY_SIZE}")

    # 4. the KEYS themselves, structurally
    cells_key = tuple(S56["cells_logical_key"])
    units_key = tuple(S56["validation_units_logical_key"])
    fams_key = tuple(S56["families_logical_key"])
    if "block" in cells_key:
        fail(f"'block' appears in the live cells.csv logical key {cells_key}; "
             f"a {S56['cells_expected_rows']}-row registry cannot be keyed on "
             f"a field with {S56['blocks_per_population']} values per cell")
        return
    if S56["block_in_cells_logical_key"]:
        fail("the registry declares block_in_cells_logical_key = true")
        return
    if cells_key != ("population", "operating_point", "method"):
        fail(f"the cells.csv logical key {cells_key} is not "
             f"(population, operating_point, method)")
        return
    if units_key != cells_key + ("block",):
        fail(f"the validation-unit key {units_key} is not the cell key plus "
             f"'block'")
        return
    if fams_key != ("population", "block"):
        fail(f"the families key {fams_key} is not (population, block)")
        return
    ok(f"keys: cells.csv {cells_key} ({S56['cells_expected_rows']} rows, no "
       f"'block' field), validation_units.csv {units_key} "
       f"({S56['validation_units_expected_rows']} rows), families.csv "
       f"{fams_key} ({S56['families_expected_rows']} rows)")

    # 5. the document's own §H.4.2 table must state the same thing, parsed
    doc = PROTOCOL_DOC.read_text()
    parsed = _parse_doc_keys(doc)
    expected = {
        "cells.csv": (cells_key, S56["cells_expected_rows"]),
        "validation_units.csv": (units_key,
                                 S56["validation_units_expected_rows"]),
        "families.csv": (fams_key, S56["families_expected_rows"]),
    }
    for sens in ("SENS-5", "SENS-6"):
        for fname, (want_key, want_n) in expected.items():
            got = parsed.get((sens, fname))
            if got is None:
                fail(f"the §H.4.2 table has no parseable {sens} {fname} row")
                return
            got_key, got_n = got
            if got_key != want_key:
                fail(f"§H.4.2 {sens} {fname} states the key {got_key}, the "
                     f"registry states {want_key}")
                return
            if got_n != want_n:
                fail(f"§H.4.2 {sens} {fname} states {got_n} rows, the registry "
                     f"states {want_n}")
                return
    ok(f"the §H.4.2 table, docs/stage_f_constants.json and src/rq3_protocol "
       f"agree on all {len(parsed)} parsed SENS-5/6 key/cardinality rows "
       f"(structured tuple comparison, not substring)")

    # 6. the cardinality block already registered must not drift from this one
    card = SENS["cardinality"]
    if (card["rq3_cells"], card["rq3_validation_units"], card["rq3_families"]) \
            != (S56["cells_expected_rows"],
                S56["validation_units_expected_rows"],
                S56["families_expected_rows"]):
        fail("sensitivities.cardinality and sens56_schema disagree on the "
             "12 / 24 / 4 cardinalities")
        return
    ok("the sensitivities.cardinality block and the sens56_schema block agree "
       "on 12 cells / 24 validation units / 4 families")


# ---------------------------------------------------------------------------
# 2. cohort
# ---------------------------------------------------------------------------
def check_cohort() -> None:
    import contracts
    m = pd.read_csv(ROOT / "results/data_contract/analysis_manifest_v1.csv")
    inc = m[m["inclusion_status"] == "INCLUDED"]
    if len(inc) != C["base_cohort_n"]:
        fail(f"cohort {len(inc)} != {C['base_cohort_n']}")
        return
    sites = sorted(set(inc["site"].astype(str)))
    if len(sites) != C["base_cohort_sites"]:
        fail(f"cohort sites {len(sites)} != {C['base_cohort_sites']}")
        return
    if set(sites) != set(contracts.FROZEN_SITES):
        fail("cohort site set differs from contracts.FROZEN_SITES")
        return
    missing = [s for s in COMMON_SITES if s not in sites]
    if missing:
        fail(f"SENS-4 common sites absent from the cohort: {missing}")
        return
    ok(f"cohort {C['base_cohort_n']} acquisitions / {C['base_cohort_sites']} "
       f"frozen sites; the 11 SENS-4 common sites are all present")


# ---------------------------------------------------------------------------
# 3. the RQ2/S2 split registry — every row, not a sample
# ---------------------------------------------------------------------------
def check_splits() -> None:
    reg = pd.read_csv(ROOT / "results/rq2_s2/production_split_registry.csv")
    if len(reg) != C["split_registry_rows"]:
        fail(f"split registry {len(reg)} rows != {C['split_registry_rows']}")
        return
    per_n = reg.groupby("n_labels")["site"].nunique().to_dict()
    if {str(int(k)): int(v) for k, v in per_n.items()} != C["sites_per_n"]:
        fail(f"per-N site counts {per_n} != {C['sites_per_n']}")
        return
    reps = reg.groupby("n_labels")["repeat"].nunique().unique()
    if list(reps) != [C["r_split"]]:
        fail(f"repeat counts {reps} != [{C['r_split']}]")
        return

    bad = 0
    for row in reg.itertuples(index=False):
        cal = row.cal_ids.split(";")
        test = row.test_ids.split(";")
        cs, ts = set(cal), set(test)
        if (len(cal) != row.n_labels or len(cs) != len(cal)
                or len(ts) != len(test) or (cs & ts)
                or len(cal) + len(test) != row.n_acquisitions):
            bad += 1
    if bad:
        fail(f"{bad} split-registry rows are not disjoint partitions with "
             f"|cal| == N")
        return
    ok(f"split registry: {C['split_registry_rows']} rows, per-N sites "
       f"{C['sites_per_n']}, {C['r_split']} repeats, all rows disjoint "
       f"partitions with |cal| == N")

    sub = reg[reg["n_labels"].isin(RQ4_N_GRID)]
    cells = sub.groupby(["site", "n_labels"]).ngroups
    expected_cells = sum(C["sites_per_n"][str(n)] for n in RQ4_N_GRID)
    if cells != expected_cells:
        fail(f"RQ4 target-local site-N cells {cells} != {expected_cells}")
        return
    if len(sub) != C["split_registry_rows_n_ge_10"]:
        fail(f"registry rows at N>=10 {len(sub)} != "
             f"{C['split_registry_rows_n_ge_10']}")
        return
    ok(f"RQ4 target-local subset: {expected_cells} site-N cells at "
       f"N in {RQ4_N_GRID}, {len(sub)} registry rows")

    # the number of registered RQ4 cells must equal the H.2 cardinality
    n_cells = 2 * len(ALPHA_GRID) + len(RQ4_N_GRID) * len(ALPHA_GRID)
    if n_cells != Q4["n_registered_cells"]:
        fail(f"registered RQ4 cells {n_cells} != {Q4['n_registered_cells']}")
        return
    ok(f"the RQ4 registry has exactly {n_cells} cells "
       f"(2 methods x 4 alpha + 3 N x 4 alpha)")


# ---------------------------------------------------------------------------
# 4. weights, keys and scores
# ---------------------------------------------------------------------------
def check_weights_and_scores() -> None:
    oof = pd.read_csv(ROOT / "results/rq1_s0/confidence_source_oof.csv")
    dw = pd.read_csv(ROOT / "results/rq1_s1/density_weights.csv")
    tgt = pd.read_csv(ROOT / "results/rq1_s0/confidence_target.csv")

    if len(oof) != C["source_oof_rows"]:
        fail(f"source OOF rows {len(oof)} != {C['source_oof_rows']}")
        return
    if len(tgt) != C["target_rows"]:
        fail(f"target rows {len(tgt)} != {C['target_rows']}")
        return

    a = set(map(tuple, oof[["outer_fold_held_out_site", "canonical_id"]].values))
    b = set(map(tuple, dw[["held_out_site", "canonical_id"]].values))
    if a != b:
        fail(f"density-weight / source-OOF key sets differ "
             f"(only_oof={len(a - b)}, only_dw={len(b - a)})")
        return
    ok(f"density-weight and source-OOF key sets are exactly equal "
       f"({len(a)} keys)")

    import density_ratio as dr
    w = dw[Q4["weight_column"]].to_numpy(float)
    if not np.isfinite(w).all():
        fail(f"density_weights.{Q4['weight_column']} contains non-finite values")
        return
    if w.min() < dr.W_CLIP_LOW - 1e-12 or w.max() > dr.W_CLIP_HIGH + 1e-12:
        fail(f"w_final range [{w.min()}, {w.max()}] outside the frozen clip "
             f"[{dr.W_CLIP_LOW}, {dr.W_CLIP_HIGH}]")
        return
    if w.min() <= 0:
        fail("w_final contains a non-positive weight; the normalised weighted "
             "quantile would be undefined")
        return
    ok(f"w_final finite, positive, within the frozen clip "
       f"[{dr.W_CLIP_LOW}, {dr.W_CLIP_HIGH}]")

    for name, frame, col in (("source OOF", oof, "oof_prob_mdd"),
                             ("target", tgt, "prob_mdd")):
        p = frame[col].to_numpy(float)
        if not np.isfinite(p).all():
            fail(f"{name} {col} contains non-finite values")
            return
        if p.min() < 0.0 or p.max() > 1.0:
            fail(f"{name} {col} range [{p.min()}, {p.max()}] outside [0, 1]")
            return
        s_mdd, s_hc = 1.0 - p, p
        if abs(float((s_mdd + s_hc).max()) - 1.0) > 1e-12:
            fail(f"{name}: a(x,MDD) + a(x,HC) != 1")
            return
        ok(f"{name} {col}: {len(p)} finite values in [0, 1]; the frozen score "
           f"a(x,y) = 1 - p_y(x) is total and sums to 1 over the two labels")

    ytgt = sorted(set(tgt["y_true"].astype(int)))
    if ytgt != [0, 1]:
        fail(f"target y_true values {ytgt} != [0, 1]")
        return
    ok("target labels are binary {0, 1}")


# ---------------------------------------------------------------------------
# 5. the order-statistic threshold construction and the guarantee labels
#    (audit findings 1 and 4)
# ---------------------------------------------------------------------------
def check_conformal() -> None:
    def k(n: int, alpha: float) -> int:
        return math.ceil((n + 1) * (1.0 - alpha))

    for n in RQ4_N_GRID:
        for alpha in ALPHA_GRID:
            if k(n, alpha) > n:
                fail(f"conformal level unattainable at N={n}, alpha={alpha}: "
                     f"ceil((N+1)(1-alpha)) = {k(n, alpha)} > {n}")
                return
    ok(f"the order-statistic rule is attainable at every (N, alpha) in "
       f"{RQ4_N_GRID} x {ALPHA_GRID}")

    if k(5, 0.1) <= 5:
        fail("N=5, alpha=0.1 is attainable; the D.6 argument for excluding "
             "N=5 does not hold")
        return
    ok(f"N=5, alpha=0.1 is unattainable (ceil(6*0.9) = {k(5, 0.1)} > 5), "
       f"confirming the frozen N >= 10 floor")

    m = pd.read_csv(ROOT / "results/data_contract/analysis_manifest_v1.csv")
    inc = m[m["inclusion_status"] == "INCLUDED"]
    sizes = inc.groupby(inc["site"].astype(str)).size()
    for site, n_s in sizes.items():
        n_cal = C["base_cohort_n"] - int(n_s)
        for alpha in ALPHA_GRID:
            if k(n_cal, alpha) > n_cal:
                fail(f"source-pooled index attainable check failed for fold "
                     f"{site} (n_cal={n_cal}, alpha={alpha})")
                return
    ok("the source-pooled order-statistic index exists in all 23 outer folds "
       "at every alpha (a threshold construction, not a coverage guarantee)")

    doc = PROTOCOL_DOC.read_text()

    # (a) the guarantee labels are bound to the right methods
    for method, label in Q4["guarantee_labels"].items():
        if label not in doc:
            fail(f"guarantee label {label} for {method} absent from the "
                 f"protocol document")
            return
    ok("all three guarantee labels are stated: "
       f"{', '.join(Q4['guarantee_labels'].values())}")

    # (b) only target_local may claim exactness
    if Q4["exact_guarantee_permitted_for"] != ["target_local"]:
        fail("more than one method is permitted an exact guarantee")
        return
    if "EXACT_WITHIN_TARGET_SITE" not in doc:
        fail("the target-local exactness label is missing")
        return
    ok("target_local is the only method permitted a finite-sample coverage "
       "claim")

    # (c) the 1.0.0 exactness claim about source_pooled is gone
    stale = _stale_lines(doc, [
        re.compile(r"source[- ]exchangeability"),
        re.compile(r"exact split[- ]conformal.{0,80}source", re.I),
        re.compile(r"source_pooled_split.{0,60}exact split", re.I),
    ])
    if stale:
        fail(f"an exact-target-coverage claim for source_pooled survives at "
             f"line(s) {stale}")
        return
    ok("no exact-target-coverage claim is attached to source_pooled_split "
       "outside the historical audit record")


# ---------------------------------------------------------------------------
# 6. the closed RQ4 CI registry  (audit finding 2)
# ---------------------------------------------------------------------------
def check_ci_registry() -> None:
    import site_bootstrap as sb

    checks = (
        ("SITE_BOOTSTRAP_B", sb.SITE_BOOTSTRAP_B, CI["site_bootstrap_b"]),
        ("CI_LEVEL", sb.CI_LEVEL, CI["ci_level"]),
        ("CI_QUANTILES", tuple(sb.CI_QUANTILES), tuple(CI["ci_quantiles"])),
        ("QUANTILE_METHOD", sb.QUANTILE_METHOD, CI["quantile_method"]),
        ("MIN_VALID_REPLICATES", sb.MIN_VALID_REPLICATES,
         CI["min_valid_replicates"]),
        ("SITE_BOOTSTRAP_ENTROPY", list(sb.SITE_BOOTSTRAP_ENTROPY),
         list(CI["bootstrap_entropy"])),
    )
    for name, got, want in checks:
        if got != want:
            fail(f"site_bootstrap.{name} = {got!r} != frozen Stage-F value "
                 f"{want!r}")
            return
    ok("the Stage-F CI machinery equals src/site_bootstrap's frozen constants: "
       f"B={CI['site_bootstrap_b']}, level={CI['ci_level']}, "
       f"{CI['ci_method']} (numpy method='{CI['quantile_method']}'), floor="
       f"{CI['min_valid_replicates']}, namespace={CI['bootstrap_entropy']}")

    req, pro = set(CI["required"]), set(CI["prohibited"])
    if req & pro:
        fail(f"CI registry is not disjoint: {sorted(req & pro)}")
        return
    if len(req) != 4:
        fail(f"F_CI_REQUIRED has {len(req)} quantities != 4")
        return
    slots = len(req) * Q4["n_registered_cells"]
    if slots != CI["n_ci_slots"]:
        fail(f"CI slots {len(req)} x {Q4['n_registered_cells']} = {slots} != "
             f"registered {CI['n_ci_slots']}")
        return
    ok(f"the RQ4 CI registry is closed and disjoint: {len(req)} quantities x "
       f"{Q4['n_registered_cells']} cells = {slots} intervals, "
       f"{len(pro)} quantities explicitly prohibited")

    # the four RQ2 sensitivities must be prohibited BY INHERITANCE, not by
    # a fresh Stage-F opinion
    import s2_aggregation as s2a
    inherited = {"sensitivity_overall_error_ucb", "sensitivity_q_step_002",
                 "sensitivity_r_star_shift", "sensitivity_common_sites"}
    if not inherited <= set(s2a.S2_CI_PROHIBITED):
        fail("the four RQ2 sensitivities are not CI-prohibited upstream; the "
             "Stage-F inheritance claim is wrong")
        return
    if not inherited <= pro:
        fail("the four inherited sensitivity prohibitions are missing from "
             "F_CI_PROHIBITED")
        return
    ok("the four sensitivity CI prohibitions are inherited verbatim from "
       "src/s2_aggregation.S2_CI_PROHIBITED")

    # the draw matrix is deterministic and correctly shaped at every Stage-F
    # site-set size
    for n_sites in CI["bootstrap_site_set_sizes"]:
        counts = sb.site_bootstrap_counts(int(n_sites))
        if counts.shape != (CI["site_bootstrap_b"], n_sites):
            fail(f"bootstrap matrix shape {counts.shape} at n_sites={n_sites}")
            return
        if not (counts.sum(axis=1) == n_sites).all():
            fail(f"bootstrap row sums != n_sites at n_sites={n_sites}")
            return
        if not np.array_equal(counts, sb.site_bootstrap_counts(int(n_sites))):
            fail(f"bootstrap matrix is not deterministic at n_sites={n_sites}")
            return
    ok(f"the site-bootstrap matrix is deterministic, shaped "
       f"({CI['site_bootstrap_b']}, n_sites) with row sums n_sites, at every "
       f"Stage-F site-set size {tuple(CI['bootstrap_site_set_sizes'])}")

    # the validity floor semantics: interval NOT_ESTIMABLE, point estimate kept
    res = sb.percentile_ci(np.concatenate(
        [np.full(CI["min_valid_replicates"] - 1, 0.5),
         np.full(CI["site_bootstrap_b"] - CI["min_valid_replicates"] + 1,
                 np.nan)]))
    if res["ci_low"] != sb.NOT_ESTIMABLE or res["ci_high"] != sb.NOT_ESTIMABLE:
        fail("percentile_ci does not return NOT_ESTIMABLE below the floor")
        return
    ok(f"below {CI['min_valid_replicates']} valid replicates the interval is "
       f"NOT_ESTIMABLE while the point estimate is untouched")

    for flag in ("hypothesis_test", "multiplicity_correction", "tost"):
        if CI[flag]:
            fail(f"RQ4 registers {flag}, which the protocol forbids")
            return
    ok("RQ4 registers no hypothesis test, no multiplicity correction and no "
       "TOST")


# ---------------------------------------------------------------------------
# 7. estimability vs degeneracy vs class composition  (audit finding 4)
# ---------------------------------------------------------------------------
def check_status_semantics() -> None:
    doc = PROTOCOL_DOC.read_text()

    for code in ST["withdrawn_reason_codes"]:
        if code in ST["reason_codes"]:
            fail(f"{code} is still a Stage-F estimability reason code")
            return
    ok(f"the two 1.0.0 codes {ST['withdrawn_reason_codes']} are withdrawn from "
       f"the estimability registry")

    if "ALL_INCLUSIVE_THRESHOLD" not in ST["degeneracy_codes"]:
        fail("ALL_INCLUSIVE_THRESHOLD is not a registered degeneracy code")
        return
    if ST["degeneracy_implies_not_estimable"]:
        fail("degeneracy is registered as implying NOT_ESTIMABLE")
        return
    if ST["class_composition_implies_not_estimable"]:
        fail("class composition is registered as implying NOT_ESTIMABLE")
        return
    if ST["class_conditional_conformal_endpoint_registered"]:
        fail("a class-conditional conformal endpoint is registered, so "
             "SINGLE_CLASS_EVALUATION would have a referent")
        return
    ok("degeneracy and class composition are separate from estimability and "
       "neither implies NOT_ESTIMABLE; no class-conditional conformal "
       "endpoint is registered")

    # every surviving mention of a withdrawn code must be historical text
    stale = _stale_lines(doc, [re.compile(re.escape(c))
                               for c in ST["withdrawn_reason_codes"]])
    if stale:
        fail(f"a withdrawn 1.0.0 reason code is used as live protocol text at "
             f"line(s) {stale}")
        return
    ok("every surviving mention of a withdrawn reason code is explicit "
       "historical audit text")

    if "degeneracy_flag" not in doc or "degeneracy_code" not in doc:
        fail("the degeneracy fields are not defined in the protocol document")
        return
    if "class_composition_flag" not in doc:
        fail("the class-composition flag is not defined in the protocol "
             "document")
        return
    ok("the four status/degeneracy fields and the composition flag are defined "
       "in the protocol document")


# ---------------------------------------------------------------------------
# 8. SENS-6: the frozen threshold-selection algorithm  (audit finding 5)
# ---------------------------------------------------------------------------
def _sens6_select(conf, c_star):
    """The frozen SENS-6 rule, implemented exactly as protocol E.7.1 states.

    Pure arithmetic on the caller's numbers. It is never applied to a cohort
    value by this validator.
    """
    conf = np.asarray(conf, dtype=float)
    n = conf.size
    if n == 0:
        return None, None, "NO_EVALUATION_POINTS"
    for t in np.unique(conf):                     # distinct, ascending
        cov = float((conf >= t).sum()) / n
        if cov <= c_star:
            return float(t), cov, None
    return None, None, "COVERAGE_MATCH_UNATTAINABLE"


def check_sens6() -> None:
    for op, want in S6["requested_coverage_by_operating_point"].items():
        rel = f"results/rq1_s0/{op.lower()}_summary.csv"
        s = pd.read_csv(ROOT / rel)
        got = sorted(set(s["source_intended_coverage"].astype(float)))
        if got != [want]:
            fail(f"{rel}: source_intended_coverage {got} != [{want}]")
            return
        if sorted(set(s["operating_point"].astype(str))) != [op]:
            fail(f"{rel}: operating_point column is not exactly [{op}]")
            return
    ok(f"the SENS-6 coverage map {S6['requested_coverage_by_operating_point']} "
       f"agrees with the frozen op50_summary.csv / op30_summary.csv")

    for i, ex in enumerate(S6["worked_examples"], start=1):
        tau, cov, code = _sens6_select(ex["conf"], ex["c_star"])
        if ex["tau"] is None:
            if tau is not None or code != ex["reason_code"]:
                fail(f"SENS-6 worked example {i}: expected "
                     f"{ex['reason_code']}, got tau={tau}, code={code}")
                return
            continue
        if tau is None or abs(tau - ex["tau"]) > 1e-12:
            fail(f"SENS-6 worked example {i}: tau {tau} != {ex['tau']}")
            return
        if abs(cov - ex["realized_coverage"]) > 1e-12:
            fail(f"SENS-6 worked example {i}: realized coverage {cov} != "
                 f"{ex['realized_coverage']}")
            return
    ok(f"the frozen SENS-6 selection rule reproduces all "
       f"{len(S6['worked_examples'])} worked examples, including the three "
       f"tie cases and the unattainable case")

    # determinism and uniqueness on a tie-dense synthetic grid
    rng = np.random.default_rng(7)
    for _ in range(200):
        conf = np.round(rng.random(rng.integers(3, 40)), 1)   # heavy ties
        for c_star in (0.50, 0.70):
            a = _sens6_select(conf, c_star)
            b = _sens6_select(conf[rng.permutation(conf.size)], c_star)
            if a != b:
                fail("SENS-6 selection depends on input order")
                return
            tau, cov, code = a
            if code is None and cov > c_star + 1e-12:
                fail(f"SENS-6 selection returned coverage {cov} > {c_star}")
                return
    ok("SENS-6 selection is order-independent and never exceeds the requested "
       "coverage (200 tie-dense synthetic groups, no cohort value used)")

    import rq3_protocol as rp
    if tuple(rp.OPERATING_POINTS) != tuple(S6["operating_points"]):
        fail(f"rq3_protocol.OPERATING_POINTS {rp.OPERATING_POINTS} != "
             f"{S6['operating_points']}")
        return
    if rp.FULL_CC_N != S6["full_sample_n"] or rp.MDD_CC_N != S6["mdd_only_n"]:
        fail("the SENS-6 complete-case sizes differ from rq3_protocol's")
        return
    if rp.FULL_CC_SITES != S6["full_sample_sites"] or \
            len(rp.MDD_CC_SITES) != S6["mdd_only_sites"]:
        fail("the SENS-6 site counts differ from rq3_protocol's")
        return
    n_rows = (S6["full_sample_sites"] + S6["mdd_only_sites"]) * \
        len(SENS["parent_confidence_methods"]) * len(S6["operating_points"])
    if n_rows != SENS["cardinality"]["sens6_threshold_selection_rows"]:
        fail(f"SENS-6 threshold_selection rows {n_rows} != "
             f"{SENS['cardinality']['sens6_threshold_selection_rows']}")
        return
    ok(f"SENS-6 inherits rq3_protocol's populations ({rp.FULL_CC_N} / "
       f"{rp.MDD_CC_N}) and operating points; threshold_selection.csv has "
       f"exactly {n_rows} rows")


# ---------------------------------------------------------------------------
# 9. sensitivity cardinality formulas  (audit finding 6)
# ---------------------------------------------------------------------------
def check_sensitivity_cardinalities() -> None:
    import rq3_protocol as rp
    reg = pd.read_csv(ROOT / "results/rq2_s2/production_split_registry.csv")
    site = reg["site"].astype(str)

    n_reg = len(reg)
    n_reg10 = int((reg["n_labels"].isin(RQ4_N_GRID)).sum())
    sub = reg[site.isin(COMMON_SITES)]
    n_c = len(sub)
    n_c10 = int((sub["n_labels"].isin(RQ4_N_GRID)).sum())

    if (n_reg, n_reg10, n_c, n_c10) != (
            C["split_registry_rows"], C["split_registry_rows_n_ge_10"],
            C["common_site_registry_rows"], C["common_site_registry_rows_n_ge_10"]):
        fail(f"registry counts ({n_reg}, {n_reg10}, {n_c}, {n_c10}) differ "
             f"from the frozen values")
        return
    if sub["site"].nunique() != len(COMMON_SITES):
        fail("the SENS-4 subset does not cover exactly the 11 common sites")
        return

    s_few = len(SENS["parent_strategies_label_consuming"])
    s_zero = len(SENS["parent_strategies_zero_label"])
    m = len(SENS["parent_confidence_methods"])
    card = SENS["cardinality"]

    derived = {
        "sens1_decisions": n_reg * s_few * m,
        "sens1_evaluations": n_reg * s_few * m + n_reg10 * s_zero * m,
        "sens2_decisions": n_reg * s_few * m,
        "sens2_evaluations": n_reg * s_few * m + n_reg10 * s_zero * m,
        "sens3_decisions": 2 * n_reg * s_few * m,
        "sens3_evaluations": 2 * (n_reg * s_few * m + n_reg10 * s_zero * m),
        "sens4_decisions": n_c * s_few * m,
        "sens4_zero_label_evaluations": n_c10 * s_zero * m,
        "sens4_evaluations": n_c * s_few * m + n_c10 * s_zero * m,
        "rq3_cells": 2 * len(rp.CELL_REGISTRY),
        "rq3_validation_units": rp.N_VALIDATION_UNITS,
        "rq3_families": len(rp.HOLM_FAMILIES),
    }
    for name, got in derived.items():
        if got != card[name]:
            fail(f"cardinality {name}: recomputed {got} != registered "
                 f"{card[name]}")
            return
    ok("every sensitivity cardinality recomputes from the frozen registries: "
       + ", ".join(f"{k}={v}" for k, v in derived.items()))

    # cross-check against the RQ2 parent's own composition, re-derived rather
    # than read as a scientific value
    if derived["sens1_evaluations"] != 78600 or \
            derived["sens1_decisions"] != 46200:
        fail("the recomputed parent cell space does not match RQ2/S2's")
        return
    ok("the recomputed parent cell space (46 200 decisions, 32 400 zero-label "
       "evaluations, 78 600 total) matches the RQ2/S2 structure")

    if SENS["n_sensitivities"] != 6 or len(SENS["labels"]) != 6:
        fail("the sensitivity set is not exactly six")
        return
    if SENS["combat_in_mandatory_core"]:
        fail("ComBat is registered inside the mandatory Stage-F core")
        return
    ok("exactly six sensitivities are registered; ComBat is outside the "
       "mandatory core")


# ---------------------------------------------------------------------------
# 10. SVM  (audit findings 8 and 9)
# ---------------------------------------------------------------------------
def check_svm() -> None:
    import nested_loso as nl

    if list(nl.C_GRID) != SVM["c_grid"]:
        fail(f"nested_loso.C_GRID {tuple(nl.C_GRID)} != frozen {SVM['c_grid']}")
        return
    if list(nl.PCA_CANDIDATES) != SVM["pca_candidates"]:
        fail(f"nested_loso.PCA_CANDIDATES {tuple(nl.PCA_CANDIDATES)} != frozen "
             f"{SVM['pca_candidates']}")
        return
    if int(nl.MAX_ITER) != SVM["max_iter"]:
        fail(f"nested_loso.MAX_ITER {nl.MAX_ITER} != frozen SVM max_iter "
             f"{SVM['max_iter']}")
        return
    ok(f"the SVM reuses the exact frozen grids: C={SVM['c_grid']}, "
       f"PCA={SVM['pca_candidates']}, max_iter={SVM['max_iter']}")

    from sklearn.svm import LinearSVC

    if hasattr(LinearSVC, "predict_proba"):
        fail("LinearSVC exposes predict_proba; the no-calibration guarantee of "
             "F.7 does not hold for this sklearn version")
        return
    ok("LinearSVC exposes no predict_proba — Platt scaling is structurally "
       "unavailable, not merely switched off")

    # random_state is inert under dual=False -- confirmed from the installed
    # sklearn's own documented behaviour, not assumed
    src = inspect.getdoc(LinearSVC) or ""
    if not re.search(r"dual=False.{0,120}random_state.{0,60}no effect",
                     src, re.S):
        fail("the installed sklearn does not document random_state as having "
             "no effect under dual=False; the OPERATIONALLY_INERT claim is "
             "unverified")
        return
    if SVM["random_state_status"] != "OPERATIONALLY_INERT":
        fail("random_state is not recorded as OPERATIONALLY_INERT")
        return
    ok("random_state is OPERATIONALLY_INERT under the frozen dual=False, "
       "confirmed from the installed scikit-learn's documented behaviour")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 40))
    y = (rng.random(300) < 0.5).astype(int)
    sites = np.array(["A"] * 100 + ["B"] * 150 + ["C"] * 50)
    w = nl.site_equal_weights(sites)

    def fit(C_, seed):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LinearSVC(penalty=SVM["penalty"], loss=SVM["loss"],
                             dual=SVM["dual"], C=C_,
                             class_weight=SVM["class_weight"],
                             fit_intercept=SVM["fit_intercept"],
                             tol=SVM["tol"], max_iter=SVM["max_iter"],
                             random_state=seed).fit(X, y, sample_weight=w)

    for C_ in SVM["c_grid"]:
        try:
            fit(C_, SVM["random_state"])
        except Exception as exc:                       # noqa: BLE001
            fail(f"LinearSVC rejects the frozen parameter set at C={C_}: "
                 f"{type(exc).__name__}: {exc}")
            return
    ok(f"LinearSVC accepts the frozen parameter set with site-balanced "
       f"sample_weight at every C in {SVM['c_grid']} (synthetic probe)")

    a = fit(1.0, SVM["random_state"]).coef_
    if not np.array_equal(a, fit(1.0, SVM["random_state"]).coef_):
        fail("LinearSVC is not deterministic under the frozen solver path")
        return
    if not np.array_equal(a, fit(1.0, 12345).coef_):
        fail("changing random_state changed the fit; it is not inert under "
             "dual=False as the protocol states")
        return
    ok("the frozen solver path is deterministic and its result is unchanged by "
       "random_state, as OPERATIONALLY_INERT requires")

    if SVM["c_scale_equivalence_claimed"]:
        fail("the protocol registry still claims C-scale equivalence between "
             "LogisticRegression and LinearSVC")
        return
    doc = PROTOCOL_DOC.read_text()
    stale = _stale_lines(doc, [
        re.compile(r"same units"),
        re.compile(r"equal regularis?zation strength", re.I),
    ])
    if stale:
        fail(f"a C-scale equivalence claim survives at line(s) {stale}")
        return
    ok("no 'same units' or equal-regularisation claim about C survives")


# ---------------------------------------------------------------------------
# 11. no Stage-F result store, no protocol tag
# ---------------------------------------------------------------------------
def _git_tags() -> list[str] | None:
    try:
        return subprocess.run(["git", "tag", "--list"], cwd=ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.split()
    except Exception as exc:                           # noqa: BLE001
        fail(f"could not list git tags: {exc}")
        return None


def _rev(ref: str) -> str | None:
    try:
        return subprocess.run(["git", "rev-list", "-n", "1", ref], cwd=ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                  # noqa: BLE001
        return None


def check_no_result_store() -> None:
    """§L.3 item 17 — the 2.0.0 authoritative root, and the 2.0.0 tag.

    Amended at F.7. The root checked here is `results/stage_f_v2/`. The
    historical `results/stage_f/` is EXPECTED to exist and is never a blocker
    (§H.1a); it is verified separately and read-only by
    `check_historical_authority`.
    """
    if STAGE_F_STORE.exists() and any(STAGE_F_STORE.rglob("*")):
        fail(f"an authoritative 2.0.0 Stage-F store already exists at "
             f"{STAGE_F_STORE.relative_to(ROOT)}; the protocol may not be "
             f"frozen against a non-empty root")
        return
    ok(f"no authoritative {MIRROR['store_root']} result store exists "
       f"(the historical results/stage_f/ is expected and does not block)")

    freeze_v2 = ROOT / "results" / "stage_f_v2_freeze"
    if freeze_v2.exists() and any(freeze_v2.rglob("*")):
        fail(f"a 2.0.0 result freeze already exists at "
             f"{freeze_v2.relative_to(ROOT)}")
        return
    ok(f"no {MIRROR['freeze_root']} freeze directory exists")

    tags = _git_tags()
    if tags is None:
        return
    if INTENDED_TAG in tags:
        fail(f"{INTENDED_TAG} already exists; the amendment draft may not claim "
             f"a tag before the independent protocol audit")
        return
    if MIRROR["intended_result_tag"] in tags:
        fail(f"{MIRROR['intended_result_tag']} already exists; no 2.0.0 result "
             f"has been produced and no result tag may be created")
        return
    if MIRROR["tag_exists"]:
        fail("the constant registry claims a protocol tag exists")
        return
    ok(f"no {INTENDED_TAG} tag and no {MIRROR['intended_result_tag']} tag exist "
       f"({len(tags)} tags in the repository)")


# ---------------------------------------------------------------------------
# 11a. the immutable historical authority is intact  (§L.3 item 17a, F.7)
# ---------------------------------------------------------------------------
def _store_content_digest(root: Path) -> str:
    """The recipe recorded in results/stage_f_freeze/authoritative_run_binding.json:
    sha256 over '<sha256(bytes)>  <relpath>\n' lines, sorted bytewise by relpath.
    """
    lines = []
    for f in sorted((q for q in root.rglob("*") if q.is_file()),
                    key=lambda q: str(q.relative_to(root)).encode()):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  "
                     f"{f.relative_to(root)}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def check_historical_authority() -> None:
    """The 1.0.2 authority must still be there, byte-for-byte, and untouched.

    Everything here is READ-ONLY. `results/stage_f/` and
    `results/stage_f_freeze/` are immutable historical identity/provenance
    evidence (§H.1a): the amendment may neither modify them nor be blocked by
    them.
    """
    for path in (HISTORICAL_STORE, HISTORICAL_FREEZE, HISTORICAL_BINDING):
        if not path.exists():
            fail(f"the immutable historical authority is missing: "
                 f"{path.relative_to(ROOT)}")
            return
    ok("the historical roots results/stage_f/ and results/stage_f_freeze/ are "
       "both present")

    binding = json.loads(HISTORICAL_BINDING.read_text())
    expected = {
        "protocol/tag": HIST["protocol_tag"],
        "protocol/sha": HIST["protocol_commit"],
        "protocol/version": HIST["protocol_version"],
        "result_tag": HIST["result_tag"],
        "identities/freeze_store_content_digest":
            HIST["freeze_store_content_digest"],
    }
    for dotted, want in expected.items():
        node = binding
        for key in dotted.split("/"):
            node = node.get(key) if isinstance(node, dict) else None
        if node != want:
            fail(f"the frozen binding's {dotted} is {node!r}, the protocol "
                 f"records {want!r}")
            return
    ok("the Stage F.5 freeze binding still records protocol "
       f"{HIST['protocol_version']} @ {HIST['protocol_tag']} "
       f"({HIST['protocol_commit'][:8]}…) and result tag {HIST['result_tag']}")

    observed = _store_content_digest(HISTORICAL_STORE)
    if observed != HIST["freeze_store_content_digest"]:
        fail(f"results/stage_f/ no longer reproduces its frozen content digest: "
             f"observed {observed[:16]}…, frozen "
             f"{HIST['freeze_store_content_digest'][:16]}…. The historical "
             f"authority has been modified and the amendment may not proceed")
        return
    ok(f"results/stage_f/ reproduces its frozen freeze_store_content_digest "
       f"{observed[:16]}… byte-for-byte — the historical store is untouched")

    tags = _git_tags()
    if tags is None:
        return
    for tag, want in ((HIST["protocol_tag"], HIST["protocol_commit"]),
                      (HIST["result_tag"], HIST["result_commit"])):
        if tag not in tags:
            fail(f"the immutable tag {tag} no longer exists")
            return
        got = _rev(tag)
        if got != want:
            fail(f"{tag} resolves to {got}, not the immutable {want}; the tag "
                 f"has been moved")
            return
    ok(f"{HIST['protocol_tag']} and {HIST['result_tag']} both still resolve to "
       f"their recorded commits — neither tag was moved")

    if HIST["blocks_new_authority"] or HIST["writable_by_new_authority"] \
            or HIST["readable_as_scientific_input"] \
            or HIST["is_final_stage_f_scientific_authority"] \
            or not HIST["immutable"] or not HIST["scientifically_superseded"]:
        fail("the historical-authority standing flags contradict §H.1a / §0.3.6")
        return
    ok("the historical store is registered as immutable, non-blocking, "
       "not writable, not a scientific input, and scientifically superseded")


# ---------------------------------------------------------------------------
# 11a-bis. results/stage_f_freeze/ is byte-identical to the TAGGED freeze
#          (§L.3 item 17a, strengthened at the §0.4 draft-audit correction)
# ---------------------------------------------------------------------------
def _rev_parse(ref: str) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", ref], cwd=ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                  # noqa: BLE001
        return None


def _git_blob_bytes(commit: str, relpath: str) -> bytes | None:
    """Read one blob out of a tagged commit. READ-ONLY; nothing is checked out."""
    try:
        return subprocess.run(["git", "show", f"{commit}:{relpath}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
    except Exception:                                  # noqa: BLE001
        return None


def _git_tree_files(commit: str, prefix: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit, "--", prefix],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:                                  # noqa: BLE001
        return None
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def _freeze_binding_errors(on_disk: dict, tagged: dict, frozen: dict) -> list:
    """Compare the working-tree freeze against the tagged blobs and the frozen
    digests. Pure and in-memory, so mutation probes can prove it detects drift
    without any file being written -- the historical freeze is never touched.
    """
    errs: list[str] = []
    if sorted(tagged) != sorted(frozen):
        errs.append(f"the tagged file set {sorted(tagged)} is not the frozen "
                    f"{sorted(frozen)}")
    missing = [n for n in frozen if n not in on_disk]
    extra = [n for n in on_disk if n not in frozen]
    if missing:
        errs.append(f"missing freeze file(s): {missing}")
    if extra:
        errs.append(f"extra freeze file(s): {extra}")
    for name, want in frozen.items():
        if name not in on_disk or name not in tagged:
            continue
        if on_disk[name] != tagged[name]:
            errs.append(f"BYTE DRIFT: {name} differs from the content bound by "
                        f"the historical result tag")
        if hashlib.sha256(on_disk[name]).hexdigest() != want:
            errs.append(f"{name} does not match its frozen SHA-256 {want[:16]}…")
        if hashlib.sha256(tagged[name]).hexdigest() != want:
            errs.append(f"the TAGGED {name} does not match the frozen SHA-256 "
                        f"{want[:16]}…")
    return errs


def check_historical_freeze_bytes() -> None:
    """§L.3 item 17a — the historical freeze is bound to the TAGGED authority.

    `check_historical_authority` proves the *store* still reproduces its content
    digest. That is not enough: the freeze record itself
    (`results/stage_f_freeze/`) could drift, and re-deriving it from the working
    tree alone would only prove it agrees with itself. `results/stage_f_freeze/`
    was force-added past `.gitignore` at the Stage F.5 freeze commit, so its
    files exist as blobs inside the `stage-f-results-v1` tag's target commit.
    This check binds the working-tree files to those blobs BYTE-FOR-BYTE.

    Everything is read-only: no checkout, no write, no regeneration.
    """
    tag = HISTORICAL_RESULT_TAG
    tag_obj, target = _rev_parse(tag), _rev(tag)
    if tag_obj is None or target is None:
        fail(f"{tag} could not be resolved; the historical result authority is "
             f"unreachable")
        return
    if tag_obj != HIST["result_tag_object"]:
        fail(f"the {tag} tag OBJECT is {tag_obj}, not the frozen "
             f"{HIST['result_tag_object']}; the tag was recreated or moved")
        return
    if target != HIST["result_commit"]:
        fail(f"{tag} targets {target}, not the frozen {HIST['result_commit']}")
        return
    ok(f"{tag} is unchanged: annotated tag object {tag_obj[:12]}… still targets "
       f"commit {target[:12]}…")

    prefix = HIST["freeze_root"].rstrip("/")
    tagged = _git_tree_files(target, prefix)
    if tagged is None:
        fail(f"could not list {prefix}/ inside the tagged commit {target[:12]}…")
        return
    tagged_rel = sorted(p[len(prefix) + 1:] for p in tagged
                        if p.startswith(prefix + "/"))
    expected = sorted(HIST["freeze_files"])
    if tagged_rel != expected:
        fail(f"the file set bound by {tag} under {prefix}/ is {tagged_rel}, not "
             f"the frozen {expected}")
        return
    if len(expected) != HIST["n_freeze_files"]:
        fail(f"the frozen freeze-file count {HIST['n_freeze_files']} does not "
             f"match the {len(expected)} registered files")
        return
    ok(f"the expected freeze file set is read FROM THE TAG, not from the "
       f"working tree: {tagged_rel} ({HIST['n_freeze_files']} files)")

    on_disk = sorted(str(q.relative_to(HISTORICAL_FREEZE))
                     for q in HISTORICAL_FREEZE.rglob("*") if q.is_file())
    missing = [n for n in expected if n not in on_disk]
    extra = [n for n in on_disk if n not in expected]
    if missing:
        fail(f"the historical freeze is MISSING file(s): {missing}")
        return
    if extra:
        fail(f"the historical freeze carries EXTRA file(s) not bound by {tag}: "
             f"{extra}; the freeze is immutable and may gain nothing")
        return
    ok(f"{prefix}/ holds exactly the bound files — no extra file and no missing "
       f"file")

    frozen = dict(HIST["freeze_files"])
    tagged_bytes, disk_bytes = {}, {}
    for name in frozen:
        blob = _git_blob_bytes(target, f"{prefix}/{name}")
        if blob is None:
            fail(f"could not read the tagged blob {prefix}/{name} @ {tag}")
            return
        tagged_bytes[name] = blob
    for name in on_disk:
        disk_bytes[name] = (HISTORICAL_FREEZE / name).read_bytes()

    drift = _freeze_binding_errors(disk_bytes, tagged_bytes, frozen)
    if drift:
        fail(f"the historical freeze has drifted from the content bound by "
             f"{tag}; it is immutable and may not be modified: {drift}")
        return
    ok(f"every historical freeze file is byte-for-byte identical to the content "
       f"bound by {tag} AND matches its frozen SHA-256 "
       f"({', '.join(f'{n} {h[:8]}…' for n, h in frozen.items())})")

    # --- mutation probes, entirely IN MEMORY: the immutable freeze on disk is
    #     never modified, and the probe proves drift would be caught ---------
    probes = []
    victim = sorted(frozen)[0]
    flipped = dict(disk_bytes)
    flipped[victim] = bytearray(disk_bytes[victim])
    flipped[victim][0] ^= 0x01
    flipped[victim] = bytes(flipped[victim])
    probes.append(("one byte flipped in a freeze file", flipped, tagged_bytes))
    appended = dict(disk_bytes)
    appended[victim] = disk_bytes[victim] + b"\n"
    probes.append(("a trailing newline appended", appended, tagged_bytes))
    truncated = dict(disk_bytes)
    truncated[victim] = disk_bytes[victim][:-1]
    probes.append(("a freeze file truncated by one byte", truncated,
                   tagged_bytes))
    probes.append(("an extra file added to the freeze",
                   {**disk_bytes, "NOTES.md": b"x"}, tagged_bytes))
    probes.append(("a freeze file removed",
                   {k: v for k, v in disk_bytes.items() if k != victim},
                   tagged_bytes))
    probes.append(("the tagged blob itself replaced", disk_bytes,
                   {**tagged_bytes, victim: b"replaced"}))
    survived = [name for name, d, t in probes
                if not _freeze_binding_errors(d, t, frozen)]
    if survived:
        fail(f"the freeze byte-identity check is not sensitive: these mutations "
             f"passed undetected: {survived}")
        return
    ok(f"mutation probes: all {len(probes)} in-memory drift scenarios — a "
       f"flipped byte, an appended newline, a truncation, an extra file, a "
       f"missing file and a substituted tagged blob — are DETECTED. Nothing on "
       f"disk was touched")

    ledger = (HISTORICAL_FREEZE / "RESULT_CHECKSUMS.sha256").read_text()
    binding_digest = HIST["freeze_files"]["authoritative_run_binding.json"]
    if f"{binding_digest}  authoritative_run_binding.json" not in ledger:
        fail("the freeze's own ledger no longer records "
             "authoritative_run_binding.json at its frozen digest")
        return
    ok("the freeze's own RESULT_CHECKSUMS.sha256 still records "
       f"authoritative_run_binding.json at {binding_digest[:16]}…")

    anchors = (("PROVENANCE.json", "store_anchor_provenance_sha256",
                "provenance_sha256"),
               ("RESULT_CHECKSUMS.sha256", "store_anchor_result_checksums_sha256",
                "result_checksums_sha256"),
               ("COMPLETION.json", "store_anchor_completion_sha256",
                "completion_sha256"))
    binding = json.loads(HISTORICAL_BINDING.read_text())
    identities = binding.get("identities", {})
    for fname, mirror_key, binding_key in anchors:
        want = HIST[mirror_key]
        path = HISTORICAL_STORE / fname
        if not path.is_file():
            fail(f"the v1 store anchor results/stage_f/{fname} is missing")
            return
        got = sha256(path)
        if got != want:
            fail(f"the v1 store anchor results/stage_f/{fname} hashes to "
                 f"{got[:16]}…, not the frozen {want[:16]}…")
            return
        if identities.get(binding_key) != want:
            fail(f"the freeze binding's identities.{binding_key} is "
                 f"{identities.get(binding_key)!r}, not the protocol's {want!r}")
            return
    n_files = sum(1 for q in HISTORICAL_STORE.rglob("*") if q.is_file())
    if n_files != HIST["store_content_n_files"] or \
            identities.get("store_content_n_files") != HIST["store_content_n_files"]:
        fail(f"results/stage_f/ holds {n_files} files and the binding records "
             f"{identities.get('store_content_n_files')!r}; the protocol freezes "
             f"{HIST['store_content_n_files']}")
        return
    ok(f"the three v1 result-store anchors (PROVENANCE {HIST['store_anchor_provenance_sha256'][:8]}…, "
       f"ledger {HIST['store_anchor_result_checksums_sha256'][:8]}…, COMPLETION "
       f"{HIST['store_anchor_completion_sha256'][:8]}…) and the "
       f"{HIST['store_content_n_files']}-file count agree with BOTH the protocol "
       f"and the freeze binding")

    doc = PROTOCOL_DOC.read_text()
    pinned = [HIST["result_tag_object"], *HIST["freeze_files"].values(),
              HIST["store_anchor_provenance_sha256"],
              HIST["store_anchor_result_checksums_sha256"],
              HIST["store_anchor_completion_sha256"]]
    absent = [v[:12] for v in pinned if v not in doc]
    if absent:
        fail(f"§H.1a does not pin every historical identity in the protocol "
             f"document; missing {absent}")
        return
    if not HIST["byte_drift_is_fail"] or not HIST["freeze_files_tracked_in_git"]:
        fail("the historical-authority registry no longer declares byte drift a "
             "failure, or no longer records the freeze files as git-tracked")
        return
    ok("§H.1a pins the tag object, both freeze-file digests and all three store "
       "anchors in the protocol document itself, and declares byte drift a FAIL")


# ---------------------------------------------------------------------------
# Markdown helpers -- used to read the frozen contracts OUT OF the document,
# so that removing or mutating a protocol table is itself a validator failure.
# ---------------------------------------------------------------------------
def _md_section(doc: str, heading: str) -> str | None:
    """Body of the section whose heading line starts with `heading`."""
    lines = doc.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(heading):
            start = i + 1
            break
    if start is None:
        return None
    depth = len(heading) - len(heading.lstrip("#"))
    for j in range(start, len(lines)):
        stripped = lines[j].lstrip("#")
        level = len(lines[j]) - len(stripped)
        if lines[j].startswith("#") and 0 < level <= depth:
            return "\n".join(lines[start:j])
    return "\n".join(lines[start:])


def _md_cell(text: str) -> str:
    return text.replace("`", "").replace("**", "").strip()


def _md_tables(body: str) -> list[list[list[str]]]:
    """Every pipe table in `body`, as lists of normalised cell lists."""
    tables, current = [], []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = [_md_cell(c) for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue                                  # separator row
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


# ---------------------------------------------------------------------------
# 11b0. Typed field enforcement  [added at §0.5, validator-coverage audit]
# ---------------------------------------------------------------------------
#: The independent Protocol-v2 validator-coverage audit demonstrated that a
#: required NULL-valued field could be DELETED without failing, because
#: `dict.get()` returns `None` both for a key that is absent and for a key that
#: is present and explicitly null. Those are different facts about a frozen
#: representation, and this validator must never conflate them again.
#:
#: Every registered field is therefore checked on three independent axes:
#:
#:   1. the key EXISTS in the representation                (absent  -> FAIL)
#:   2. the value has the registered TYPE                   (0/""/false != null)
#:   3. the value equals the registered VALUE
#:
#: plus a fourth, structural axis: the field SET is exact, so an unregistered
#: field cannot be smuggled in beside the frozen ones. `dict.get()` is not used
#: anywhere in these checks.
_KIND_NULL, _KIND_BOOL, _KIND_INT = "null", "bool", "int"
_KIND_STR, _KIND_STRLIST = "str", "strlist"

#: Sentinel spelled into the registry for the §F.7.1a fields whose frozen value
#: is "**retained** — the value §F.7.5 selected". The value is per-fold, so the
#: registry freezes its RETAINEDNESS, not a number. It is a string, never null:
#: that is exactly what separates §F.7.1a from §F.7.1.
_RETAINED = "RETAINED_SELECTED_VALUE"


def _typed_field_errors(container, specs: dict, label: str,
                        nested_keys: tuple = ()) -> list[str]:
    """Exact key-presence + type + value + field-set enforcement.

    Pure. `nested_keys` names sub-representations that are validated by their
    own spec and are therefore not "unregistered extra fields" here.
    """
    errs: list[str] = []
    if not isinstance(container, dict):
        return [f"{label} is not a mapping (it is {type(container).__name__})"]
    for name, (kind, expected) in specs.items():
        if name not in container:
            errs.append(
                f"{label}.{name}: the key is ABSENT. The frozen representation "
                f"requires it to be PRESENT with the {kind} value "
                f"{expected!r}; a missing key is not the same fact as a "
                f"present null")
            continue
        val = container[name]
        if kind == _KIND_NULL:
            if val is not None:
                errs.append(f"{label}.{name} is {val!r} "
                            f"({type(val).__name__}); the frozen "
                            f"representation requires an explicit null (not "
                            f"false, not 0, not an empty string)")
        elif kind == _KIND_BOOL:
            if val is not expected:
                errs.append(f"{label}.{name} is {val!r} "
                            f"({type(val).__name__}), not the boolean "
                            f"{expected!r}")
        elif kind == _KIND_INT:
            if type(val) is not int or val != expected:
                errs.append(f"{label}.{name} is {val!r} "
                            f"({type(val).__name__}), not the integer "
                            f"{expected!r} (null, bool and str are all wrong "
                            f"here)")
        elif kind == _KIND_STR:
            if type(val) is not str or val != expected:
                errs.append(f"{label}.{name} is {val!r} "
                            f"({type(val).__name__}), not the string "
                            f"{expected!r}")
        elif kind == _KIND_STRLIST:
            wrong = (not isinstance(val, (list, tuple))
                     or any(type(x) is not str for x in val)
                     or list(val) != list(expected))
            if wrong:
                errs.append(f"{label}.{name} is {val!r}, not the frozen "
                            f"sequence {list(expected)!r} (order included)")
        else:                                              # pragma: no cover
            errs.append(f"{label}.{name}: unregistered kind {kind!r}")
    extra = sorted(set(container) - set(specs) - set(nested_keys))
    if extra:
        errs.append(f"{label} carries unregistered field(s) {extra}; the frozen "
                    f"field set is exact, not a minimum")
    return errs


# ---------------------------------------------------------------------------
# 11b1. Structural document parsing  [added at §0.5, validator-coverage audit]
# ---------------------------------------------------------------------------
#: The same audit demonstrated that token/substring searches over the normative
#: document accept mutations that change its meaning: a third type added to the
#: positive allow-list, a deleted exclusion, a broadened boundary. Every check
#: below therefore DERIVES the document's own structure -- its normative tables
#: and bullet lists -- and compares it for EQUALITY with the frozen structure.
def _table_with_header(body: str, header) -> list[list[str]] | None:
    """Rows (header excluded) of the table whose header row is EXACTLY `header`."""
    for table in _md_tables(body):
        if table and tuple(c.strip() for c in table[0]) == tuple(header):
            return [list(r) for r in table[1:]]
    return None


def _row_key(cell: str) -> str:
    return " ".join(cell.split()).strip()


def _row_token(cell: str) -> str:
    """A table cell's normative token: the value, minus its explanation."""
    s = cell.split("—")[0]
    s = re.sub(r"\(§[^)]*\)?", "", s)
    s = re.sub(r",?\s*exactly as in.*$", "", s)
    return " ".join(s.split()).strip().rstrip(",;.").strip()


def _plain(text: str) -> str:
    """Markdown emphasis and code marks removed, whitespace collapsed."""
    return " ".join(text.replace("`", "").replace("**", "").split()).strip()


def _doc_row_table_errors(body: str, header, label: str,
                          expected: dict) -> list[str]:
    """EQUALITY between a normative field table and a frozen structure."""
    errs: list[str] = []
    rows = _table_with_header(body, header)
    if rows is None:
        return [f"{label}: the normative table whose header is exactly "
                f"{tuple(header)} is missing from the protocol document"]
    got: dict = {}
    for r in rows:
        if len(r) != len(header):
            return [f"{label}: malformed table row {r!r}"]
        key = _row_key(r[0])
        if key in got:
            return [f"{label}: duplicate table row {key!r}"]
        got[key] = tuple(_row_token(c) for c in r[1:])
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    if missing:
        errs.append(f"{label}: the table DROPS the frozen row(s) {missing}")
    if extra:
        errs.append(f"{label}: the table carries unfrozen row(s) {extra}")
    for key in sorted(set(expected) & set(got)):
        want = expected[key]
        want = want if isinstance(want, tuple) else (want,)
        if got[key] != want:
            errs.append(f"{label}: row {key!r} gives {got[key]!r}, not the "
                        f"frozen {want!r}")
    return errs


def _bullet_list(body: str, anchor: str) -> list[str] | None:
    """The bullet list immediately following `anchor`, wrapped lines joined."""
    idx = body.find(anchor)
    if idx < 0:
        return None
    out: list[str] = []
    cur: str | None = None
    started = False
    for line in body[idx:].splitlines():
        if line.startswith("- "):
            started = True
            if cur is not None:
                out.append(cur)
            cur = line[2:].strip()
        elif started and line.startswith("  ") and line.strip():
            cur = f"{cur} {line.strip()}"
        elif started and not line.strip():
            continue
        elif started:
            break
    if cur is not None:
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# 11c. the two per-site ConvergenceWarning representations  (§L.3 item 17e)
# ---------------------------------------------------------------------------
#: The four selection/refit fields that distinguish §F.7.1 from §F.7.1a.
_DISTINGUISHING = ("selected_pca", "selected_C", "inner_macro_bacc",
                   "n_iter_max")

#: The eight representation fields the Protocol-v2 validator-coverage audit
#: requires to be enforced key-by-key, independently, in BOTH cases.
_MANDATORY_WARNING_FIELDS = ("status", "reason_code", "bacc", "selected_pca",
                             "selected_C", "inner_macro_bacc", "converged",
                             "n_iter_max")

#: §F.7.1 -- the inner-grid warning row. No lawful selection occurred, so all
#: four selection/refit fields are explicitly NULL, and every one of them must
#: be PRESENT and null.
_INNER_WARNING_SPEC = {
    "section": (_KIND_STR, "F.7.1"),
    "lawful_selection_occurred": (_KIND_BOOL, False),
    "status": (_KIND_STR, "NOT_ESTIMABLE"),
    "reason_code": (_KIND_STR, "SVM_NOT_CONVERGED"),
    "bacc": (_KIND_NULL, None),
    "selected_pca": (_KIND_NULL, None),
    "selected_C": (_KIND_NULL, None),
    "inner_macro_bacc": (_KIND_NULL, None),
    "converged": (_KIND_BOOL, False),
    "n_iter_max": (_KIND_NULL, None),
    "final_refit_attempted": (_KIND_BOOL, False),
    "selection_metadata_retained": (_KIND_BOOL, False),
}

#: §F.7.1a -- the final-refit warning row after a LAWFUL selection. `bacc` is
#: still null, but the three selection fields are RETAINED and `n_iter_max` is
#: the configured cap 2000.
_FINAL_WARNING_SPEC = {
    "section": (_KIND_STR, "F.7.1a"),
    "lawful_selection_occurred": (_KIND_BOOL, True),
    "all_inner_candidate_rows_estimable": (_KIND_BOOL, True),
    "status": (_KIND_STR, "NOT_ESTIMABLE"),
    "reason_code": (_KIND_STR, "SVM_NOT_CONVERGED"),
    "bacc": (_KIND_NULL, None),
    "selected_pca": (_KIND_STR, _RETAINED),
    "selected_C": (_KIND_STR, _RETAINED),
    "inner_macro_bacc": (_KIND_STR, _RETAINED),
    "selected_pca_retained": (_KIND_BOOL, True),
    "selected_C_retained": (_KIND_BOOL, True),
    "inner_macro_bacc_retained": (_KIND_BOOL, True),
    "selected_inner_score_finite": (_KIND_BOOL, True),
    "converged": (_KIND_BOOL, False),
    "n_iter_max": (_KIND_INT, 2000),
    "final_refit_attempted": (_KIND_BOOL, True),
    "selection_metadata_retained": (_KIND_BOOL, True),
    "retained_metadata_usable_as_result": (_KIND_BOOL, False),
    "appears_in_fold_selection_csv": (_KIND_BOOL, False),
}

_WREP_TOP_SPEC = {
    "file": (_KIND_STR, "svm/per_site_bacc.csv"),
    "representations_are_distinct": (_KIND_BOOL, True),
    "new_status_or_reason_code_introduced": (_KIND_BOOL, False),
    "cohort_fields_always_persisted":
        (_KIND_STRLIST, ["n_acquisitions", "n_mdd", "n_hc"]),
    "retained_value_token": (_KIND_STR, _RETAINED),
}

#: The document's own §F.7.1 / §F.7.1a persisted-row tables, structurally.
_ROW_TABLE_HEADER = ("Field", "Value")
_COHORT_ROW_KEY = "n_acquisitions, n_mdd, n_hc"
_DOC_INNER_ROWS = {
    "status": "NOT_ESTIMABLE", "reason_code": "SVM_NOT_CONVERGED",
    "bacc": "null", "selected_pca": "null", "selected_C": "null",
    "inner_macro_bacc": "null", "converged": "false", "n_iter_max": "null",
    _COHORT_ROW_KEY: "normally persisted",
}
_DOC_FINAL_ROWS = {
    "status": "NOT_ESTIMABLE", "reason_code": "SVM_NOT_CONVERGED",
    "bacc": "null", "selected_pca": "retained", "selected_C": "retained",
    "inner_macro_bacc": "retained", "converged": "false", "n_iter_max": "2000",
    _COHORT_ROW_KEY: "normally persisted",
}

#: The document's own field-by-field difference table, structurally.
_DIFF_TABLE_HEADER = ("Field", "§F.7.1 inner-grid warning",
                      "§F.7.1a final-refit warning")
_DOC_DIFF_ROWS = {
    "lawful selection occurred": ("no", "yes"),
    "selected_pca": ("null", "the selected value"),
    "selected_C": ("null", "the selected value"),
    "inner_macro_bacc": ("null", "the selected candidate's finite inner score"),
    "final outer refit": ("skipped", "attempted, and it warned"),
    "n_iter_max": ("null", "2000"),
    "bacc": ("null", "null"),
    "status": ("NOT_ESTIMABLE", "NOT_ESTIMABLE"),
    "reason_code": ("SVM_NOT_CONVERGED", "SVM_NOT_CONVERGED"),
    "converged": ("false", "false"),
    _COHORT_ROW_KEY: ("persisted", "persisted"),
}


def _warning_representation_errors(wrep: dict, doc: str) -> list[str]:
    """Errors in the §F.7.1 / §F.7.1a contract. Pure -- used by mutation probes.

    Both representations are checked field by field with EXACT key presence,
    type and value (`_typed_field_errors`), and the protocol document's own
    tables are parsed and compared for structural equality. Neither half is
    token-based, and neither accepts a deleted field as a null one.
    """
    errs: list[str] = []
    if not isinstance(wrep, dict):
        return ["the warning-representation registry is not a mapping"]
    for case in ("inner_grid_warning", "final_refit_warning"):
        if case not in wrep:
            errs.append(f"the {case} representation is ABSENT from the registry")
    if errs:
        return errs
    inner, final = wrep["inner_grid_warning"], wrep["final_refit_warning"]

    # --- registry: exact key presence, type, value and field set ----------
    errs += _typed_field_errors(wrep, _WREP_TOP_SPEC,
                                "svm_per_site_warning_representation",
                                nested_keys=("inner_grid_warning",
                                             "final_refit_warning"))
    errs += _typed_field_errors(inner, _INNER_WARNING_SPEC, "§F.7.1 inner_grid_warning")
    errs += _typed_field_errors(final, _FINAL_WARNING_SPEC, "§F.7.1a final_refit_warning")

    # --- the eight mandated fields are present in BOTH, independently ------
    for name, case, spec in (("inner_grid_warning", inner, _INNER_WARNING_SPEC),
                             ("final_refit_warning", final,
                              _FINAL_WARNING_SPEC)):
        for field in _MANDATORY_WARNING_FIELDS:
            if field not in spec:
                errs.append(f"{name}.{field} is not registered at all; the "
                            f"validator itself would not enforce it")
            if isinstance(case, dict) and field not in case:
                errs.append(f"{name}.{field}: the mandated representation field "
                            f"is ABSENT")

    # --- the two must not collapse into one --------------------------------
    if isinstance(inner, dict) and isinstance(final, dict):
        same = [f for f in _DISTINGUISHING
                if f in inner and f in final and inner[f] == final[f]]
        if same:
            errs.append(f"the two representations no longer differ on {same}; "
                        f"§F.7.1 and §F.7.1a would be indistinguishable")
        if "lawful_selection_occurred" in inner and \
                "lawful_selection_occurred" in final and \
                inner["lawful_selection_occurred"] == \
                final["lawful_selection_occurred"]:
            errs.append("the two cases no longer differ on whether a lawful "
                        "selection occurred")

    # --- the document must FREEZE both, in its own tables ------------------
    b1 = _md_section(doc, "#### F.7.1 ")
    b1a = _md_section(doc, "#### F.7.1a ")
    if b1 is None:
        errs.append("§F.7.1 is missing from the protocol document")
    if b1a is None:
        errs.append("§F.7.1a is missing from the protocol document: the "
                    "final-refit-warning row contract is not frozen")
        return errs
    if b1 is not None:
        errs += _doc_row_table_errors(b1, _ROW_TABLE_HEADER,
                                      "§F.7.1's persisted-row table",
                                      _DOC_INNER_ROWS)
    errs += _doc_row_table_errors(b1a, _ROW_TABLE_HEADER,
                                  "§F.7.1a's persisted-row table",
                                  _DOC_FINAL_ROWS)
    errs += _doc_row_table_errors(b1a, _DIFF_TABLE_HEADER,
                                  "§F.7.1a's field-by-field difference table",
                                  _DOC_DIFF_ROWS)

    # --- the document must also state the two rules that make the retained
    #     metadata safe, and the no-new-vocabulary rule --------------------
    flat = " ".join(b1a.split())
    if "No new status value and no new reason code is introduced" not in flat:
        errs.append("§F.7.1a no longer states that no new status value and no "
                    "new reason code is introduced")
    if "never a result" not in flat:
        errs.append("§F.7.1a no longer states that the retained selection "
                    "metadata is provenance, never a result")
    return errs


# ---------------------------------------------------------------------------
# 11d. the SVM_FOLD_FAILED closed scope, structurally  (§L.3 item 17f)
# ---------------------------------------------------------------------------
#: Frozen at §G.2a, in document order. Closed and exhaustive -- not exemplary.
_ALLOWED_FAILURE_TYPES = ("FloatingPointError", "numpy.linalg.LinAlgError")

#: The exact-type predicates §G.2a freezes, derived from the allow-list. The
#: contract is `type(exc) is T`, NEVER `isinstance(exc, T)`: LinAlgError is a
#: ValueError subclass and a bare ValueError is excluded, so only exact-type
#: identity keeps both statements true.
_EXACT_TYPE_PREDICATES = tuple(f"type(exc) is {t}"
                               for t in _ALLOWED_FAILURE_TYPES)

#: `scipy.linalg.LinAlgError` IS `numpy.linalg.LinAlgError` (the same class
#: object). §G.2a may MENTION it as an identity-equivalent explanatory note; it
#: may never appear as a third permitted semantic type.
_G2A_ALIAS_MENTIONS = ("scipy.linalg.LinAlgError",)

#: Exception-ish names §G.2a may name because it EXCLUDES them (or names them
#: inside an exclusion). Any other type name in the section is unregistered.
_G2A_EXCLUSION_TYPE_MENTIONS = (
    "Exception", "AssertionError", "KeyError", "TypeError", "IndexError",
    "RuntimeError", "ValueError", "MemoryError", "KeyboardInterrupt",
    "SystemExit", "SVMScopeError")

#: The estimator/backend boundary, structurally: what is INSIDE it (and
#: nothing else), and every operation §G.2a places OUTSIDE it.
_BOUNDARY_INSIDE = ("LinearSVC.fit", "decision_function")
_BOUNDARY_OUTSIDE = ("Data assembly", "the scaler and PCA transforms",
                     "weight construction", "scoring", "selection",
                     "aggregation", "schema handling", "persistence")

#: The frozen fail-hard exclusion list, bullet by bullet, IN ORDER. Each entry
#: is (exact normalised bullet text, the registry keys that bullet discharges).
#: Text, order and count are all normative: a deleted bullet shortens the list,
#: an added one lengthens it, and a broadened one stops matching.
_EXCLUSION_BULLETS = (
    ("a bare except Exception: catch-all — broad exception catching is "
     "forbidden outright, whatever it is then spelled as",
     ("exception_catch_all",)),
    ("programming errors", ("programming_errors",)),
    ("invariant failures (any contracts / SVMScopeError-class assertion)",
     ("invariant_failures",)),
    ("schema, contract or registry-invariant failures",
     ("schema_errors", "contract_or_registry_invariant_failures")),
    ("AssertionError", ("AssertionError",)),
    ("KeyError", ("KeyError",)),
    ("TypeError", ("TypeError",)),
    ("IndexError", ("IndexError",)),
    ("arbitrary RuntimeError / ValueError", ("RuntimeError", "ValueError")),
    ("invalid shapes", ("invalid_shapes",)),
    ("non-finite values in a frozen input", ("nonfinite_frozen_inputs",)),
    ("a missing class in a training or evaluation partition",
     ("missing_classes",)),
    ("an off-grid C or PCA parameter", ("off_grid_parameters",)),
    ("impossible internal states", ("impossible_internal_states",)),
    ("MemoryError", ("MemoryError",)),
    ("KeyboardInterrupt", ("KeyboardInterrupt",)),
    ("SystemExit", ("SystemExit",)),
    ("leakage or data-contract violations",
     ("leakage_or_data_contract_violations",)),
)

#: Every class §G.2a forbids SVM_FOLD_FAILED from absorbing, derived from the
#: bullet list above so the two can never disagree.
_REQUIRED_EXCLUSIONS = {k for _, keys in _EXCLUSION_BULLETS for k in keys}

#: The registry order of `must_not_absorb`. Order is frozen too, so a silent
#: reordering that hides a substitution cannot pass.
_MUST_NOT_ABSORB_ORDER = (
    "exception_catch_all", "programming_errors", "invariant_failures",
    "schema_errors", "contract_or_registry_invariant_failures",
    "AssertionError", "KeyError", "TypeError", "IndexError", "RuntimeError",
    "ValueError", "invalid_shapes", "nonfinite_frozen_inputs",
    "missing_classes", "off_grid_parameters", "impossible_internal_states",
    "MemoryError", "KeyboardInterrupt", "SystemExit",
    "leakage_or_data_contract_violations")

_G2A_ALLOW_TABLE_HEADER = (
    "#", "Permitted failure type",
    "Why it is an expected numerical failure inside the estimator/backend "
    "boundary")

_G2A_BOUNDARY_STRING = ("the frozen LinearSVC.fit(...) call for one candidate "
                        "and the decision_function(...) call on that same "
                        "fitted estimator")

#: Exact key presence, type, value and field set for the §G.2a registry.
_FF_SPEC = {
    "code": (_KIND_STR, "SVM_FOLD_FAILED"),
    "is_catch_all": (_KIND_BOOL, False),
    "admits_only":
        (_KIND_STR, "narrowly classified expected estimator/backend failure"),
    "must_not_absorb": (_KIND_STRLIST, list(_MUST_NOT_ABSORB_ORDER)),
    "n_must_not_absorb": (_KIND_INT, len(_MUST_NOT_ABSORB_ORDER)),
    "unexpected_errors_fail_hard": (_KIND_BOOL, True),
    "substitutes_for_svm_not_converged": (_KIND_BOOL, False),
    "allowed_exception_types": (_KIND_STRLIST, list(_ALLOWED_FAILURE_TYPES)),
    "n_allowed_exception_types": (_KIND_INT, len(_ALLOWED_FAILURE_TYPES)),
    "allow_list_is_closed_and_exhaustive": (_KIND_BOOL, True),
    "type_match_rule": (_KIND_STR, "exact_type_identity"),
    "exact_type_predicates": (_KIND_STRLIST, list(_EXACT_TYPE_PREDICATES)),
    "isinstance_matching_permitted": (_KIND_BOOL, False),
    "subclass_widening_permitted": (_KIND_BOOL, False),
    "broad_exception_catching_permitted": (_KIND_BOOL, False),
    "identity_equivalent_aliases": (_KIND_STRLIST, list(_G2A_ALIAS_MENTIONS)),
    "aliases_add_no_semantic_type": (_KIND_BOOL, True),
    "n_semantic_failure_types": (_KIND_INT, len(_ALLOWED_FAILURE_TYPES)),
    "requires_estimator_backend_boundary": (_KIND_BOOL, True),
    "estimator_backend_boundary": (_KIND_STR, _G2A_BOUNDARY_STRING),
    "estimator_backend_boundary_inside":
        (_KIND_STRLIST, list(_BOUNDARY_INSIDE)),
    "estimator_backend_boundary_outside":
        (_KIND_STRLIST, list(_BOUNDARY_OUTSIDE)),
    "outside_boundary_is_hard_failure": (_KIND_BOOL, True),
    "linalgerror_subclasses_valueerror": (_KIND_BOOL, True),
    "bare_valueerror_permitted": (_KIND_BOOL, False),
    "estimator_settings_altered_to_avoid_failures": (_KIND_BOOL, False),
}

#: Any exception-ish identifier in §G.2a, dotted names included, plus the bare
#: word `Exception`.
_TYPE_NAME_RE = re.compile(
    r"\b(?:(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Interrupt|Exit)|Exception)\b")


def _parse_g2a(doc: str) -> tuple[dict, list[str]]:
    """DERIVE §G.2a's frozen contract from the document's own structure.

    Returns (structure, errors). Nothing here is a substring search for an
    expected token: the allow-list is read out of the normative table, the
    exclusions out of the normative bullet list, and the boundary out of the
    numbered condition that defines it.
    """
    errs: list[str] = []
    body = _md_section(doc, "### G.2a ")
    if body is None:
        return {}, ["§G.2a is missing from the protocol document"]
    flat = " ".join(body.split())
    plain = _plain(body)
    out: dict = {}

    # -- (1) the positive allow-list, as a TABLE, in order ------------------
    rows = _table_with_header(body, _G2A_ALLOW_TABLE_HEADER)
    if rows is None:
        errs.append("§G.2a carries no positive allow-list table with the frozen "
                    "header; a prose mention is not a closed allow-list")
        out["allow_list"] = None
    else:
        malformed = [r for r in rows if len(r) != 3]
        if malformed:
            errs.append(f"§G.2a's allow-list table has malformed row(s) "
                        f"{malformed}")
        rows = [r for r in rows if len(r) == 3]
        out["allow_list"] = tuple(_plain(r[1]) for r in rows)
        numbering = [_plain(r[0]) for r in rows]
        if numbering != [str(i + 1) for i in range(len(rows))]:
            errs.append(f"§G.2a's allow-list table is numbered {numbering}, not "
                        f"1..{len(rows)}")

    # -- (2) the declared cardinality, cross-checked with the table ---------
    m = re.search(r"Exactly \*\*(\w+)\*\* types are registered", flat)
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    out["declared_count"] = words.get(m.group(1).lower()) if m else None
    if m is None:
        errs.append("§G.2a no longer declares how many types are registered")
    elif out["declared_count"] is None:
        errs.append(f"§G.2a declares an unparseable type count "
                    f"{m.group(1)!r}")

    # -- (3) the fail-hard exclusion list, as a BULLET LIST, in order -------
    bullets = _bullet_list(body, "It must NOT absorb any of the following.")
    if bullets is None:
        errs.append("§G.2a carries no `must NOT absorb` bullet list")
        out["exclusion_bullets"] = None
        out["exclusion_keys"] = None
    else:
        out["exclusion_bullets"] = tuple(
            _plain(b).rstrip(";.").strip() for b in bullets)
        want = tuple(text for text, _ in _EXCLUSION_BULLETS)
        got = out["exclusion_bullets"]
        if got != want:
            dropped = [t for t in want if t not in got]
            added = [t for t in got if t not in want]
            if dropped:
                errs.append(f"§G.2a's exclusion list DROPS or rewords the "
                            f"frozen exclusion(s) {dropped}")
            if added:
                errs.append(f"§G.2a's exclusion list carries unfrozen "
                            f"exclusion(s) {added}")
            if not dropped and not added:
                errs.append("§G.2a's exclusion list has been reordered")
        out["exclusion_keys"] = frozenset(
            k for text, keys in _EXCLUSION_BULLETS if text in got for k in keys)
        if out["exclusion_keys"] != _REQUIRED_EXCLUSIONS:
            errs.append(f"§G.2a's exclusion list no longer discharges "
                        f"{sorted(_REQUIRED_EXCLUSIONS - out['exclusion_keys'])}")

    # -- (4) the two conversion conditions ----------------------------------
    marker = "Both conditions must hold before any conversion."
    conds: dict = {}
    if marker not in flat:
        errs.append("§G.2a no longer requires BOTH conditions before any "
                    "conversion")
    else:
        seg = flat.split(marker, 1)[1]
        seg = seg.split("**Everything outside that closed allow-list", 1)[0]
        parts = re.split(r"(?:^|\s)(\d)\.\s+\*\*", seg)
        for i in range(1, len(parts) - 1, 2):
            conds[int(parts[i])] = "**" + parts[i + 1]
        if set(conds) != {1, 2}:
            errs.append(f"§G.2a's conversion conditions parse as {sorted(conds)}"
                        f", not exactly two numbered conditions")

    # -- (4a) condition 1: EXACT TYPE IDENTITY, never isinstance ------------
    c1 = conds.get(1, "")
    out["exact_predicates"] = tuple(
        f"type(exc) is {t}"
        for t in re.findall(r"type\(exc\) is ([A-Za-z_][\w.]*)", c1))
    c1_plain = _plain(c1)
    if "exact type identity" not in c1_plain:
        errs.append("§G.2a's first condition no longer requires membership by "
                    "exact type identity")
    if "never by isinstance" not in c1_plain:
        errs.append("§G.2a's first condition no longer rules out isinstance "
                    "matching")
    if "no subclass of a listed type can widen the list" not in c1_plain:
        errs.append("§G.2a's first condition no longer forbids subclass "
                    "widening")
    out["isinstance_mentions"] = plain.count("isinstance")
    if out["isinstance_mentions"] != 1:
        errs.append(f"§G.2a mentions `isinstance` {out['isinstance_mentions']} "
                    f"times; the only permitted mention is the one that rules "
                    f"it out")

    # -- (4b) condition 2: the estimator/backend boundary, exactly ----------
    c2 = conds.get(2, "")
    closed = "and **nothing else**" in c2
    out["boundary_closed"] = closed
    if not closed:
        errs.append("§G.2a's boundary is no longer closed with `and nothing "
                    "else`; it admits more than the frozen two operations")
    prefix = c2.split("and **nothing else**", 1)[0] if closed else c2
    out["boundary_inside"] = tuple(
        re.sub(r"\(\.\.\.\)$", "", s)
        for s in re.findall(r"`([^`]+)`", prefix))
    mo = re.search(r"\.\s*([^.]*?) are all \*\*outside\*\* the boundary", c2)
    if mo is None:
        errs.append("§G.2a no longer lists the operations that are OUTSIDE the "
                    "estimator/backend boundary")
        out["boundary_outside"] = None
    else:
        items = [p.strip() for p in mo.group(1).split(",") if p.strip()]
        if items:
            tail = items.pop().split(" and ")
            items += [t.strip() for t in tail if t.strip()]
        out["boundary_outside"] = tuple(items)
    if "the same exception type raised there is **not**" not in c2:
        errs.append("§G.2a no longer states that an allowed type raised outside "
                    "the boundary is NOT SVM_FOLD_FAILED")

    # -- (5) closure, alias handling and the hard-failure residue -----------
    for phrase, message in (
        ("This list is closed: it is exhaustive, not exemplary.",
         "§G.2a no longer declares the allow-list closed and exhaustive"),
        ("no third type may be added without a protocol amendment",
         "§G.2a no longer forbids adding a third type without an amendment"),
        ("is the same class object",
         "§G.2a no longer records scipy.linalg.LinAlgError as the SAME class "
         "object as numpy.linalg.LinAlgError"),
        ("so it adds no third type",
         "§G.2a no longer states that the scipy alias adds NO third type"),
        ("an allowed type raised outside the boundary, or any other type "
         "raised anywhere",
         "§G.2a no longer makes everything outside the closed allow-list a "
         "hard Formal failure"),
        ("There is no residual category and no default conversion.",
         "§G.2a no longer forbids a residual category / default conversion"),
        ("Estimator settings may not be altered to avoid these failures.",
         "§G.2a no longer forbids altering estimator settings to avoid these "
         "failures"),
        ("SVM_FOLD_FAILED is never a substitute for SVM_NOT_CONVERGED.",
         "§G.2a no longer separates SVM_FOLD_FAILED from SVM_NOT_CONVERGED"),
    ):
        if phrase not in plain:
            errs.append(message)

    # -- (6) closure over TYPE MENTIONS: no unregistered type may appear ----
    allowed = set(out.get("allow_list") or ())
    mentioned = set(_TYPE_NAME_RE.findall(plain))
    unregistered = sorted(mentioned - allowed - set(_G2A_ALIAS_MENTIONS)
                          - set(_G2A_EXCLUSION_TYPE_MENTIONS))
    out["unregistered_type_mentions"] = tuple(unregistered)
    if unregistered:
        errs.append(f"§G.2a names exception type(s) {unregistered} that are "
                    f"neither on the closed allow-list, nor an identity-"
                    f"equivalent alias, nor an explicit exclusion")
    for alias in _G2A_ALIAS_MENTIONS:
        if alias in allowed:
            errs.append(f"the identity-equivalent alias {alias} has been "
                        f"promoted onto the positive allow-list; it may never "
                        f"become a third permitted semantic type")
    return out, errs


def _fold_failed_allow_list_errors(ff: dict, doc: str) -> list[str]:
    """Errors in the §G.2a closed scope. Pure -- used by mutation probes.

    Two independent halves, neither token-based:

      * the REGISTRY is checked field by field for exact key presence, type,
        value and field set (`_typed_field_errors`);
      * the DOCUMENT's own structure is parsed (`_parse_g2a`) and compared for
        equality -- allow-list table, exclusion bullet list, boundary
        definition and exact-type predicates.

    A registry that carries only exclusions fails, because an exclusion list
    does not say what the code admits.
    """
    errs: list[str] = []
    errs += _typed_field_errors(ff, _FF_SPEC, "svm_fold_failed_scope")

    # -- registry-internal consistency (defence in depth) -------------------
    if isinstance(ff, dict):
        allowed = (ff["allowed_exception_types"]
                   if "allowed_exception_types" in ff else None)
        if allowed is not None and isinstance(allowed, (list, tuple)):
            if not allowed:
                errs.append("no positive SVM_FOLD_FAILED allow-list is "
                            "registered; exclusions alone are not a "
                            "specification")
            promoted = sorted(set(allowed) & set(_G2A_ALIAS_MENTIONS))
            if promoted:
                errs.append(f"identity-equivalent alias(es) {promoted} are "
                            f"registered as permitted types; they add no "
                            f"semantic type and may not widen the list")
            if "exact_type_predicates" in ff and \
                    isinstance(ff["exact_type_predicates"], (list, tuple)):
                want = [f"type(exc) is {t}" for t in allowed]
                if list(ff["exact_type_predicates"]) != want:
                    errs.append(f"the registered exact-type predicates "
                                f"{list(ff['exact_type_predicates'])} do not "
                                f"match the allow-list {list(allowed)}")
        excl = ff["must_not_absorb"] if "must_not_absorb" in ff else None
        if isinstance(excl, (list, tuple)):
            missing = _REQUIRED_EXCLUSIONS - set(excl)
            extra = set(excl) - _REQUIRED_EXCLUSIONS
            if missing:
                errs.append(f"the exclusion list omits {sorted(missing)}")
            if extra:
                errs.append(f"the exclusion list carries unfrozen entries "
                            f"{sorted(extra)}")

    # -- the document must state the same thing, STRUCTURALLY ---------------
    struct, doc_errs = _parse_g2a(doc)
    errs += doc_errs
    if not struct:
        return errs
    if struct.get("allow_list") is not None and \
            tuple(struct["allow_list"]) != _ALLOWED_FAILURE_TYPES:
        errs.append(f"§G.2a's allow-list table registers "
                    f"{tuple(struct['allow_list'])}, not exactly the frozen "
                    f"{_ALLOWED_FAILURE_TYPES}")
    if struct.get("declared_count") is not None and \
            struct.get("allow_list") is not None and \
            struct["declared_count"] != len(struct["allow_list"]):
        errs.append(f"§G.2a declares {struct['declared_count']} permitted types "
                    f"but tabulates {len(struct['allow_list'])}")
    if struct.get("exact_predicates") is not None and \
            tuple(struct["exact_predicates"]) != _EXACT_TYPE_PREDICATES:
        errs.append(f"§G.2a's exact-type predicates parse as "
                    f"{tuple(struct['exact_predicates'])}, not the frozen "
                    f"{_EXACT_TYPE_PREDICATES}; exact-type semantics are no "
                    f"longer stated for exactly the permitted types")
    if struct.get("boundary_inside") is not None and \
            tuple(struct["boundary_inside"]) != _BOUNDARY_INSIDE:
        errs.append(f"§G.2a's estimator/backend boundary admits "
                    f"{tuple(struct['boundary_inside'])}, not exactly "
                    f"{_BOUNDARY_INSIDE}")
    if struct.get("boundary_outside") is not None:
        got = tuple(s.lower() for s in struct["boundary_outside"])
        want = tuple(s.lower() for s in _BOUNDARY_OUTSIDE)
        if got != want:
            errs.append(f"§G.2a places {got} outside the boundary, not exactly "
                        f"{want}; the operations it drops would become "
                        f"convertible")
    # the registry and the document must not disagree
    if isinstance(ff, dict) and struct.get("allow_list") is not None and \
            "allowed_exception_types" in ff and \
            isinstance(ff["allowed_exception_types"], (list, tuple)) and \
            tuple(ff["allowed_exception_types"]) != tuple(struct["allow_list"]):
        errs.append(f"the registry allow-list "
                    f"{tuple(ff['allowed_exception_types'])} disagrees with "
                    f"§G.2a's table {tuple(struct['allow_list'])}")
    if isinstance(ff, dict) and struct.get("exclusion_keys") is not None and \
            "must_not_absorb" in ff and \
            isinstance(ff["must_not_absorb"], (list, tuple)) and \
            set(ff["must_not_absorb"]) != set(struct["exclusion_keys"]):
        errs.append("the registry exclusion set disagrees with §G.2a's own "
                    "bullet list")
    return errs


# ---------------------------------------------------------------------------
# 11e. named mutation probes  [added at §0.5, validator-coverage audit]
# ---------------------------------------------------------------------------
#: Each probe is a NAMED mutation that MUST be rejected. Document probes carry
#: an `applied` flag: a probe whose anchor text has moved is void, and a void
#: probe is reported as a failure rather than as silent assurance.
def _doc_sub(doc: str, old: str, new: str) -> tuple[str, bool]:
    return (doc.replace(old, new, 1), old in doc)


def _doc_drop_line(doc: str, prefix: str) -> tuple[str, bool]:
    lines = doc.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith(prefix):
            return "".join(lines[:i] + lines[i + 1:]), True
    return doc, False


_ISINSTANCE_DOC_OLD = (
    "   **exact type identity** — `type(exc) is FloatingPointError` or\n"
    "   `type(exc) is numpy.linalg.LinAlgError` — and **never** by "
    "`isinstance`, so\n"
    "   no subclass of a listed type can widen the list; **and**")
_ISINSTANCE_DOC_NEW = (
    "   **isinstance matching** — `isinstance(exc, FloatingPointError)` or\n"
    "   `isinstance(exc, numpy.linalg.LinAlgError)` — so any subclass of a\n"
    "   listed type also qualifies; **and**")
_BOUNDARY_DOC_OLD = (
    "   `decision_function(...)` call on that same fitted estimator, and "
    "**nothing\n   else**.")
_BOUNDARY_DOC_NEW = (
    "   `decision_function(...)` call on that same fitted estimator, the\n"
    "   `PCA.fit_transform(...)` call and any other `sklearn` backend call in\n"
    "   the fold, and **nothing else**.")
_OUTSIDE_DOC_OLD = "Data assembly, the scaler and PCA transforms, weight"
_OUTSIDE_DOC_NEW = "Data assembly, weight"


def build_mutation_probes(doc: str, wrep: dict | None = None,
                          ff: dict | None = None) -> list[dict]:
    """The named mutations that must all FAIL, registry-side and document-side."""
    import copy
    base_w = copy.deepcopy(WREP if wrep is None else wrep)
    base_f = copy.deepcopy(FF if ff is None else ff)
    probes: list[dict] = []

    def w_probe(name, variant, mutate=None, mdoc=None, applied=True):
        m = copy.deepcopy(base_w)
        if mutate is not None:
            mutate(m)
        probes.append({"name": name, "variant": variant, "target": "warning",
                       "wrep": m, "ff": base_f,
                       "doc": doc if mdoc is None else mdoc,
                       "applied": applied})

    def f_probe(name, variant, changes=None, mdoc=None, applied=True):
        m = copy.deepcopy(base_f)
        if changes:
            m.update(changes)
        probes.append({"name": name, "variant": variant,
                       "target": "fold_failed", "wrep": base_w, "ff": m,
                       "doc": doc if mdoc is None else mdoc,
                       "applied": applied})

    def drop(key):
        return lambda m: m["must_not_absorb"].remove(key) \
            if key in m["must_not_absorb"] else None

    def f_drop_exclusion(name, key, line_prefix):
        m = [x for x in base_f["must_not_absorb"] if x != key]
        f_probe(name, "registry",
                {"must_not_absorb": m, "n_must_not_absorb": len(m)})
        d, applied = _doc_drop_line(doc, line_prefix)
        f_probe(name, "document", None, d, applied)

    # -- 1..4: deleting a required field, null-valued or not ---------------
    for name, case, field in (
            ("delete inner_grid_warning.selected_pca",
             "inner_grid_warning", "selected_pca"),
            ("delete inner_grid_warning.n_iter_max",
             "inner_grid_warning", "n_iter_max"),
            ("delete final_refit_warning.bacc",
             "final_refit_warning", "bacc"),
            ("delete final_refit_warning.selected_C",
             "final_refit_warning", "selected_C")):
        w_probe(name, "registry",
                lambda m, c=case, f=field: m[c].pop(f, None))

    # -- 5..6: null <-> value, in both directions --------------------------
    w_probe("change final_refit_warning.n_iter_max from 2000 to null",
            "registry", lambda m: m["final_refit_warning"].__setitem__(
                "n_iter_max", None))
    w_probe("change inner_grid_warning.n_iter_max from null to 2000",
            "registry", lambda m: m["inner_grid_warning"].__setitem__(
                "n_iter_max", 2000))

    # -- 7..9: the positive allow-list -------------------------------------
    f_probe("add RuntimeError to positive allow-list", "registry",
            {"allowed_exception_types": ["FloatingPointError",
                                         "numpy.linalg.LinAlgError",
                                         "RuntimeError"],
             "n_allowed_exception_types": 3,
             "n_semantic_failure_types": 3,
             "exact_type_predicates":
                 ["type(exc) is FloatingPointError",
                  "type(exc) is numpy.linalg.LinAlgError",
                  "type(exc) is RuntimeError"]})
    d, applied = _doc_sub(
        doc, "so it adds no third type |\n",
        "so it adds no third type |\n"
        "| 3 | `RuntimeError` | an unexpected backend failure |\n")
    f_probe("add RuntimeError to positive allow-list", "document", None, d,
            applied)

    f_probe("remove FloatingPointError", "registry",
            {"allowed_exception_types": ["numpy.linalg.LinAlgError"],
             "n_allowed_exception_types": 1, "n_semantic_failure_types": 1,
             "exact_type_predicates":
                 ["type(exc) is numpy.linalg.LinAlgError"]})
    d, applied = _doc_drop_line(doc, "| 1 | `FloatingPointError`")
    f_probe("remove FloatingPointError", "document", None, d, applied)

    f_probe("remove numpy.linalg.LinAlgError", "registry",
            {"allowed_exception_types": ["FloatingPointError"],
             "n_allowed_exception_types": 1, "n_semantic_failure_types": 1,
             "exact_type_predicates": ["type(exc) is FloatingPointError"]})
    d, applied = _doc_drop_line(doc, "| 2 | `numpy.linalg.LinAlgError`")
    f_probe("remove numpy.linalg.LinAlgError", "document", None, d, applied)

    # -- 10: exact-type identity vs isinstance -----------------------------
    f_probe("broaden exact-type semantics to isinstance", "registry",
            {"type_match_rule": "isinstance",
             "isinstance_matching_permitted": True,
             "subclass_widening_permitted": True,
             "exact_type_predicates":
                 ["isinstance(exc, FloatingPointError)",
                  "isinstance(exc, numpy.linalg.LinAlgError)"]})
    d, applied = _doc_sub(doc, _ISINSTANCE_DOC_OLD, _ISINSTANCE_DOC_NEW)
    f_probe("broaden exact-type semantics to isinstance", "document", None, d,
            applied)

    # -- 11..14: deleting a frozen fail-hard exclusion ---------------------
    f_drop_exclusion("delete invalid-shape exclusion", "invalid_shapes",
                     "- invalid shapes;")
    f_drop_exclusion("delete nonfinite-input exclusion",
                     "nonfinite_frozen_inputs",
                     "- non-finite values in a frozen input;")
    f_drop_exclusion("delete leakage/data-contract exclusion",
                     "leakage_or_data_contract_violations",
                     "- leakage or data-contract violations.")
    f_drop_exclusion("delete MemoryError exclusion", "MemoryError",
                     "- `MemoryError`;")

    # -- 15..16: the estimator/backend boundary ----------------------------
    f_probe("broaden estimator boundary beyond fit/decision_function",
            "registry",
            {"estimator_backend_boundary_inside":
                 ["LinearSVC.fit", "decision_function", "PCA.fit_transform"]})
    d, applied = _doc_sub(doc, _BOUNDARY_DOC_OLD, _BOUNDARY_DOC_NEW)
    f_probe("broaden estimator boundary beyond fit/decision_function",
            "document", None, d, applied)

    f_probe("allow preprocessing/PCA failure conversion", "registry",
            {"estimator_backend_boundary_outside":
                 [x for x in _BOUNDARY_OUTSIDE
                  if x != "the scaler and PCA transforms"]})
    d, applied = _doc_sub(doc, _OUTSIDE_DOC_OLD, _OUTSIDE_DOC_NEW)
    f_probe("allow preprocessing/PCA failure conversion", "document", None, d,
            applied)

    # -- 17: collapsing the two warning representations --------------------
    import copy as _copy
    w_probe("conflate inner-warning and final-refit representations",
            "final := inner",
            lambda m: m.__setitem__("final_refit_warning",
                                    _copy.deepcopy(m["inner_grid_warning"])))
    w_probe("conflate inner-warning and final-refit representations",
            "inner := final",
            lambda m: m.__setitem__("inner_grid_warning",
                                    _copy.deepcopy(m["final_refit_warning"])))

    # -- regression probes: everything the earlier audits already closed ---
    for field, value in (("n_iter_max", None), ("selected_pca_retained", False),
                         ("selected_C_retained", False),
                         ("inner_macro_bacc_retained", False),
                         ("converged", True), ("bacc", 0.5),
                         ("status", "ESTIMABLE"),
                         ("reason_code", "SVM_FOLD_FAILED"),
                         ("selected_pca", None), ("selected_C", None),
                         ("inner_macro_bacc", None),
                         ("lawful_selection_occurred", False)):
        w_probe(f"final_refit_warning.{field} := {value!r}", "registry",
                lambda m, f=field, v=value: m["final_refit_warning"]
                .__setitem__(f, v))
    for field, value in (("n_iter_max", 2000), ("selected_pca", 50),
                         ("selected_C", 1.0), ("inner_macro_bacc", 0.6),
                         ("bacc", 0.5), ("converged", True),
                         ("status", "ESTIMABLE")):
        w_probe(f"inner_grid_warning.{field} := {value!r}", "registry",
                lambda m, f=field, v=value: m["inner_grid_warning"]
                .__setitem__(f, v))
    for case in ("inner_grid_warning", "final_refit_warning"):
        for field in ("status", "reason_code", "converged"):
            w_probe(f"delete {case}.{field}", "registry",
                    lambda m, c=case, f=field: m[c].pop(f, None))
    w_probe("representations_are_distinct := False", "registry",
            lambda m: m.__setitem__("representations_are_distinct", False))
    w_probe("delete representations_are_distinct", "registry",
            lambda m: m.pop("representations_are_distinct", None))
    w_probe("delete cohort_fields_always_persisted", "registry",
            lambda m: m.pop("cohort_fields_always_persisted", None))
    w_probe("an unregistered field smuggled into final_refit_warning",
            "registry",
            lambda m: m["final_refit_warning"].__setitem__("bacc_shadow", 0.5))

    body_1a = _md_section(doc, "#### F.7.1a ")
    d, applied = _doc_sub(doc, body_1a, "") if body_1a else (doc, False)
    w_probe("§F.7.1a removed from the document", "document", None, d, applied)
    d, applied = _doc_sub(doc, "| `n_iter_max` | `2000` (§F.7.4",
                          "| `n_iter_max` | `null` (§F.7.4")
    w_probe("§F.7.1a n_iter_max 2000 -> null in the document", "document",
            None, d, applied)
    d, applied = _doc_sub(
        doc, "| `selected_pca` | **retained** — the value §F.7.5 selected |",
        "| `selected_pca` | `null` |")
    w_probe("§F.7.1a selected_pca no longer retained in the document",
            "document", None, d, applied)
    d, applied = _doc_drop_line(doc, "| `selected_C` | `null` |")
    w_probe("§F.7.1 selected_C row deleted from the document", "document",
            None, d, applied)
    d, applied = _doc_drop_line(doc, "| `inner_macro_bacc` | `null` |")
    w_probe("§F.7.1a difference-table row deleted from the document",
            "document", None, d, applied)

    f_probe("allow-list emptied (exclusions alone must NOT pass)", "registry",
            {"allowed_exception_types": [], "n_allowed_exception_types": 0,
             "n_semantic_failure_types": 0, "exact_type_predicates": []})
    f_probe("allow-list broadened with Exception", "registry",
            {"allowed_exception_types": ["FloatingPointError",
                                         "numpy.linalg.LinAlgError",
                                         "Exception"],
             "n_allowed_exception_types": 3, "n_semantic_failure_types": 3,
             "exact_type_predicates":
                 ["type(exc) is FloatingPointError",
                  "type(exc) is numpy.linalg.LinAlgError",
                  "type(exc) is Exception"]})
    f_probe("the scipy alias promoted onto the allow-list", "registry",
            {"allowed_exception_types": ["FloatingPointError",
                                         "numpy.linalg.LinAlgError",
                                         "scipy.linalg.LinAlgError"],
             "n_allowed_exception_types": 3, "n_semantic_failure_types": 3,
             "exact_type_predicates":
                 ["type(exc) is FloatingPointError",
                  "type(exc) is numpy.linalg.LinAlgError",
                  "type(exc) is scipy.linalg.LinAlgError"]})
    f_probe("subclass widening permitted", "registry",
            {"subclass_widening_permitted": True})
    f_probe("broad exception catching permitted", "registry",
            {"broad_exception_catching_permitted": True})
    f_probe("boundary requirement dropped", "registry",
            {"requires_estimator_backend_boundary": False})
    f_probe("boundary definition emptied", "registry",
            {"estimator_backend_boundary": ""})
    f_probe("out-of-boundary failure no longer hard", "registry",
            {"outside_boundary_is_hard_failure": False})
    f_probe("bare ValueError permitted", "registry",
            {"bare_valueerror_permitted": True})
    f_probe("closed/exhaustive flag dropped", "registry",
            {"allow_list_is_closed_and_exhaustive": False})
    f_probe("catch-all reinstated", "registry", {"is_catch_all": True})
    f_probe("unexpected errors no longer fail hard", "registry",
            {"unexpected_errors_fail_hard": False})
    for key in ("allowed_exception_types", "isinstance_matching_permitted",
                "type_match_rule", "outside_boundary_is_hard_failure",
                "must_not_absorb", "estimator_backend_boundary",
                "broad_exception_catching_permitted"):
        m = {k: v for k, v in base_f.items() if k != key}
        probes.append({"name": f"delete svm_fold_failed_scope.{key}",
                       "variant": "registry", "target": "fold_failed",
                       "wrep": base_w, "ff": m, "doc": doc, "applied": True})

    body_g2a = _md_section(doc, "### G.2a ")
    d, applied = _doc_sub(doc, body_g2a, "") if body_g2a else (doc, False)
    f_probe("§G.2a removed from the document", "document", None, d, applied)
    d, applied = _doc_sub(doc, "Exactly **two** types", "Exactly **three** types")
    f_probe("§G.2a's declared type count contradicts its table", "document",
            None, d, applied)
    d, applied = _doc_drop_line(doc, "- `AssertionError`;")
    f_probe("delete AssertionError exclusion", "document", None, d, applied)
    d, applied = _doc_drop_line(doc,
                                "- **a bare `except Exception:` catch-all**")
    f_probe("delete the catch-all prohibition", "document", None, d, applied)
    d, applied = _doc_sub(doc, "- invalid shapes;", "- invalid shapes, unless "
                                                    "the backend raised them;")
    f_probe("weaken the invalid-shape exclusion by rewording", "document", None,
            d, applied)
    return probes


def evaluate_mutation_probes(doc: str) -> list[dict]:
    """Run every named probe; a probe is REJECTED iff it applied and failed."""
    out = []
    for p in build_mutation_probes(doc):
        if p["target"] == "warning":
            errs = _warning_representation_errors(p["wrep"], p["doc"])
        else:
            errs = _fold_failed_allow_list_errors(p["ff"], p["doc"])
        out.append({**p, "n_errors": len(errs),
                    "rejected": bool(errs) and p["applied"]})
    return out


def _report_probes(doc: str, target: str) -> list[str]:
    """Names of probes for `target` that were NOT rejected (empty == good)."""
    bad = []
    for p in evaluate_mutation_probes(doc):
        if p["target"] != target:
            continue
        if not p["applied"]:
            bad.append(f"{p['name']} [{p['variant']}] (VOID — the anchor text "
                       f"moved; the probe never applied)")
        elif not p["rejected"]:
            bad.append(f"{p['name']} [{p['variant']}] (SURVIVED)")
    return bad


def check_svm_warning_representations() -> None:
    """§L.3 item 17e — both per-site warning rows are frozen and distinct."""
    doc = PROTOCOL_DOC.read_text()
    errs = _warning_representation_errors(WREP, doc)
    if errs:
        fail("the per-site ConvergenceWarning representations are incomplete or "
             f"have drifted: {errs[:6]}")
        return
    f = WREP["final_refit_warning"]
    ok("the §F.7.1a final-refit-warning row is FROZEN, in the registry and in "
       f"the protocol document's own table: status={f['status']}, "
       f"reason_code={f['reason_code']}, bacc=null, converged=false, "
       f"n_iter_max={f['n_iter_max']}, with selected_pca / selected_C / "
       f"inner_macro_bacc RETAINED at the lawfully selected values")
    ok("the §F.7.1 inner-grid representation remains DISTINCT: no lawful "
       "selection, all three selection fields null, the final refit skipped and "
       "n_iter_max null; the document carries the field-by-field difference "
       "table")
    ok(f"every one of the {len(_MANDATORY_WARNING_FIELDS)} mandated fields "
       f"{_MANDATORY_WARNING_FIELDS} is checked in BOTH representations for KEY "
       f"PRESENCE, type and value independently: a deleted key fails, a "
       f"present-null passes only where null is frozen, and false / 0 / '' are "
       f"never accepted as null")
    ok("both persisted-row tables and the field-by-field difference table are "
       "PARSED out of §F.7.1 / §F.7.1a and compared for exact row-set and "
       "value equality — a deleted, added or reworded row is a failure; no "
       "token search is used")

    bad = _report_probes(doc, "warning")
    if bad:
        fail(f"the §F.7.1 / §F.7.1a contract check is not sensitive: {bad}")
        return
    n = sum(1 for p in evaluate_mutation_probes(doc)
            if p["target"] == "warning")
    ok(f"mutation probes: all {n} named mutations of the two warning "
       f"representations are REJECTED — including deleting "
       f"inner_grid_warning.selected_pca, deleting inner_grid_warning."
       f"n_iter_max, deleting final_refit_warning.bacc and .selected_C, "
       f"flipping n_iter_max null<->2000 in either direction, conflating the "
       f"two representations, smuggling in an unregistered field, and removing "
       f"or rewriting either table in the document")


def check_svm_fold_failed_allow_list() -> None:
    """§L.3 item 17f — the positive allow-list is frozen, closed and narrow."""
    doc = PROTOCOL_DOC.read_text()

    # the two permitted types must be REAL and must behave as the protocol says
    import numpy as _np
    import scipy.linalg as _sla
    if _sla.LinAlgError is not _np.linalg.LinAlgError:
        fail("scipy.linalg.LinAlgError is not numpy.linalg.LinAlgError in this "
             "environment; §G.2a's claim that it adds no third type is false")
        return
    if not issubclass(_np.linalg.LinAlgError, ValueError):
        fail("numpy.linalg.LinAlgError no longer subclasses ValueError; §G.2a's "
             "justification for exact-type matching must be restated")
        return
    if not issubclass(FloatingPointError, ArithmeticError):
        fail("FloatingPointError no longer subclasses ArithmeticError")
        return
    ok("both permitted types exist in the installed environment: "
       "FloatingPointError (an ArithmeticError) and numpy.linalg.LinAlgError, "
       "which scipy re-exports as the same object and which subclasses "
       "ValueError — exactly why §G.2a matches by exact type identity")

    # the exact-type rule is verified BY BEHAVIOUR on real exception objects:
    # `type(exc) is T` must reject a subclass that `isinstance` would admit.
    class _SubclassOfAllowed(_np.linalg.LinAlgError):
        pass

    exact_ok = type(_np.linalg.LinAlgError("x")) is _np.linalg.LinAlgError
    exact_rejects_subclass = type(_SubclassOfAllowed("x")) is not \
        _np.linalg.LinAlgError
    isinstance_admits_subclass = isinstance(_SubclassOfAllowed("x"),
                                            _np.linalg.LinAlgError)
    isinstance_admits_bare_valueerror = isinstance(
        _np.linalg.LinAlgError("x"), ValueError)
    if not (exact_ok and exact_rejects_subclass and isinstance_admits_subclass
            and isinstance_admits_bare_valueerror):
        fail("exact-type identity and isinstance are not observably different "
             "in this environment; §G.2a's distinction cannot be verified")
        return
    ok("the exact-type rule is verified BY BEHAVIOUR: `type(exc) is T` accepts "
       "the registered class and REJECTS a subclass that `isinstance` would "
       "admit, and numpy.linalg.LinAlgError is isinstance-compatible with the "
       "excluded ValueError — so isinstance matching would demonstrably widen "
       "the closed list")

    errs = _fold_failed_allow_list_errors(FF, doc)
    if errs:
        fail(f"the SVM_FOLD_FAILED positive allow-list is missing, incomplete "
             f"or has drifted: {errs[:6]}")
        return
    struct, _ = _parse_g2a(doc)
    ok(f"the SVM_FOLD_FAILED allow-list is POSITIVE, closed and exhaustive: "
       f"exactly {FF['n_allowed_exception_types']} types "
       f"{tuple(FF['allowed_exception_types'])}, matched by "
       f"{FF['type_match_rule']}, admissible only inside the estimator/backend "
       f"fit boundary, with broad exception catching forbidden and "
       f"{len(FF['must_not_absorb'])} excluded failure classes")
    ok(f"the allow-list is DERIVED from §G.2a's own normative table — "
       f"{tuple(struct['allow_list'])}, numbered 1..{len(struct['allow_list'])}, "
       f"cross-checked against the document's declared count of "
       f"{struct['declared_count']} — and compared for equality, not searched "
       f"for as tokens; the scipy alias is recorded as identity-equivalent and "
       f"is NOT on the list")
    ok(f"the exact-type predicates are parsed out of §G.2a's first conversion "
       f"condition as {tuple(struct['exact_predicates'])} and must equal the "
       f"allow-list exactly; `isinstance` occurs exactly "
       f"{struct['isinstance_mentions']} time in the section — the mention that "
       f"rules it out")
    ok(f"the estimator/backend boundary is parsed out of §G.2a's second "
       f"condition: inside = {tuple(struct['boundary_inside'])} and nothing "
       f"else; outside = {tuple(struct['boundary_outside'])}. Any broadening — "
       f"preprocessing, PCA, scoring, schema handling, arbitrary backend code — "
       f"is a failure")
    ok(f"the fail-hard exclusion set is parsed out of §G.2a's own bullet list: "
       f"{len(struct['exclusion_bullets'])} bullets discharging all "
       f"{len(_REQUIRED_EXCLUSIONS)} frozen exclusion keys, compared bullet by "
       f"bullet in order; §G.2a names no exception type that is neither "
       f"permitted, an identity-equivalent alias, nor explicitly excluded")

    bad = _report_probes(doc, "fold_failed")
    if bad:
        fail(f"the SVM_FOLD_FAILED allow-list check is not sensitive: {bad}")
        return
    n = sum(1 for p in evaluate_mutation_probes(doc)
            if p["target"] == "fold_failed")
    ok(f"mutation probes: all {n} named mutations are REJECTED — adding "
       f"RuntimeError (registry AND document), removing either permitted type, "
       f"broadening to isinstance, deleting the invalid-shape, nonfinite-input, "
       f"leakage/data-contract, MemoryError or AssertionError exclusion, "
       f"rewording an exclusion into a weaker one, broadening the boundary "
       f"beyond fit/decision_function, and admitting preprocessing/PCA failures")


# ---------------------------------------------------------------------------
# 11b. the 2.0.0 SVM contracts  (§L.3 items 17c / 17d, F.7)
# ---------------------------------------------------------------------------
def check_svm_v2_contracts() -> None:
    doc = PROTOCOL_DOC.read_text()

    # -- 17d: the scientific definition did not move ------------------------
    frozen_1_0_2 = {
        "estimator": "sklearn.svm.LinearSVC",
        "penalty": "l2",
        "loss": "squared_hinge",
        "dual": False,
        "class_weight": None,
        "fit_intercept": True,
        "tol": 0.0001,
        "max_iter": 2000,
        "random_state": 20260807,
        "random_state_status": "OPERATIONALLY_INERT",
        "estimand": "site_macro_outer_loso_balanced_accuracy",
        "decision_rule": "decision_function(x) > 0",
        "c_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
        "pca_candidates": [None, 50, 100, 200],
        "probability_calibration": False,
        "platt_scaling": False,
        "sample_weight": "src/nested_loso.site_equal_weights",
    }
    drift = {k: (SVM.get(k), v) for k, v in frozen_1_0_2.items()
             if SVM.get(k) != v}
    if drift:
        fail(f"the 2.0.0 amendment moved a frozen scientific SVM definition — "
             f"this is prohibited (§0.3.2): {drift}")
        return
    if SVM["definition_changed_by_2_0_0"] or \
            AMEND["scientific_svm_definition_changed"] or \
            AMEND["fail_closed_rule_weakened"] or \
            AMEND["solver_or_grid_or_tolerance_or_max_iter_changed"] or \
            AMEND["is_post_result_relaxation"] or \
            AMEND["runtime_implementation_changed"] or \
            AMEND["formal_run_authorized"] or AMEND["stage_g_authorized"] or \
            AMEND["new_result_tag_created"]:
        fail("the amendment registry claims a scientific / runtime / execution "
             "change; 2.0.0 amends only representability, status, root and "
             "version contracts")
        return
    ok(f"every frozen scientific SVM definition is unchanged by 2.0.0 "
       f"({len(frozen_1_0_2)} parameters, grids and the estimand), and the "
       f"amendment registers no scientific, runtime or execution change")

    # -- the §0.5 validator-coverage round is recorded, and is not a
    #    scientific change  [added at §0.5] ---------------------------------
    if AMEND["validator_hardening_is_scientific_change"] or \
            AMEND["validator_hardening_changed_runtime"]:
        fail("the §0.5 validator hardening is registered as a scientific or "
             "runtime change; it is neither")
        return
    if AMEND["validator_coverage_audit_verdict"] != "FAIL" or \
            not AMEND["validator_coverage_audit_accepted_protocol_content"]:
        fail("the §0.5 re-audit record does not match the audit that was "
             "actually returned")
        return
    if "### 0.5 " not in doc:
        fail("the protocol document carries no §0.5 correction record for the "
             "validator-coverage re-audit")
        return
    sec = _md_section(doc, "### 0.5 ")
    for phrase in ("No normative content changed.",
                   "incomplete protocol-validator enforcement",
                   "present and explicitly", "token-based"):
        if phrase not in " ".join((sec or "").split()).replace("**", ""):
            fail(f"§0.5 does not record {phrase!r}")
            return
    ok("§0.5 records the independent re-audit of the corrected draft: the "
       "protocol content was ACCEPTED, the single critical blocker was "
       "incomplete validator enforcement, the hardening changed no normative "
       "content and no runtime code, and no Formal run or tag followed")

    # -- the any-fit fail-closed rule is preserved, not weakened -------------
    if not (SVM["any_fit_convergence_warning_invalidates_outer_fold"]
            and SVM["inner_fit_counts_as_fold_lifecycle_fit"]):
        fail("the any-fit convergence-warning fail-closed rule is not preserved")
        return
    if SVM["reduced_grid_selection_permitted"] or \
            SVM["final_refit_after_inner_warning_permitted"] or \
            not SVM["selection_requires_all_candidates_estimable"]:
        fail("reduced-grid selection or a post-warning final refit is permitted; "
             "§F.7.1 forbids both")
        return
    ok("the any-fit fail-closed rule is preserved and strengthened: an inner "
       "fit is a fold-lifecycle fit, reduced-grid selection is prohibited, a "
       "post-warning final refit is prohibited, and selection requires all "
       f"{SVM['n_candidates_per_fold']} candidates ESTIMABLE")

    # -- 17c: the fold_selection contract -----------------------------------
    if FS["logical_key"] != ["held_out_site", "pca", "C"]:
        fail(f"the fold_selection logical key changed: {FS['logical_key']}")
        return
    if FS["fields"] != ["inner_macro_bacc", "status", "reason_code"]:
        fail(f"the fold_selection field list is {FS['fields']}, not the exact "
             f"amended set")
        return
    derived = (C["base_cohort_sites"] * len(SVM["pca_candidates"])
               * len(SVM["c_grid"]))
    if derived != FS["expected_rows"] or FS["expected_rows"] != 460:
        fail(f"the fold_selection cardinality {FS['expected_rows']} does not "
             f"recompute from the frozen grids ({derived})")
        return
    if SVM["n_candidates_per_fold"] != len(SVM["pca_candidates"]) * len(SVM["c_grid"]):
        fail("n_candidates_per_fold does not equal |PCA| x |C|")
        return
    ok(f"fold_selection keeps the key {tuple(FS['logical_key'])} and recomputes "
       f"to {derived} = {C['base_cohort_sites']} x {len(SVM['pca_candidates'])} "
       f"x {len(SVM['c_grid'])} mandatory rows, "
       f"{SVM['n_candidates_per_fold']} per fold")

    if not FS["uses_existing_status_vocabulary"] or \
            FS["redundant_candidate_status_fields_registered"]:
        fail("the amendment registers redundant candidate_status / "
             "candidate_reason fields, or leaves the existing vocabulary")
        return
    if FS["estimable_status"] not in ST["status_values"] or \
            FS["not_estimable_status"] not in ST["status_values"]:
        fail("the fold_selection statuses are not drawn from the §G.1 list")
        return
    for code in (FS["reason_code_convergence"], FS["reason_code_expected_failure"]):
        if code not in ST["reason_codes"]:
            fail(f"{code} is not in the §G.2 reason-code registry")
            return
    if FS["reason_code_when_estimable"] is not None or \
            FS["null_score_without_status_permitted"]:
        fail("an ESTIMABLE row carries a reason code, or a null score without a "
             "status is still permitted")
        return
    # The names may appear only inside the sentence that PROHIBITS them.
    _prohibiting = re.compile(r"\bno\b|\bnot\b|never|redundant|forbid|prohibit",
                              re.I)
    _dl = doc.splitlines()
    for token in ("candidate_status", "candidate_reason"):
        bad = [i + 1 for i, line in enumerate(_dl)
               if token in line
               and not _prohibiting.search(" ".join(_dl[max(0, i - 1):i + 2]))]
        if bad:
            fail(f"the protocol document registers a redundant {token} field "
                 f"at line(s) {bad}")
            return
    ok("the three §H.3.1 candidate outcomes use only the existing §G.1 status "
       "values and §G.2 reason codes; no candidate_status / candidate_reason "
       "field is registered anywhere")

    if not (FS["all_rows_mandatory"] and FS["all_candidates_evaluated_when_safe"]
            and FS["diagnostic_scores_retained_for_successful_candidates"]):
        fail("the complete-grid diagnostic policy is not registered (§F.7.2)")
        return
    if FS["diagnostic_scores_usable_for_selection_in_invalidated_fold"] or \
            FS["invariant_failures_converted_to_candidate_rows"]:
        fail("diagnostic scores may rescue an invalidated fold, or invariant "
             "failures may become candidate rows; §F.7.2 forbids both")
        return
    ok("the complete-grid diagnostic policy is registered: all 20 candidates "
       "evaluated when safe, all rows persisted, diagnostic scores never "
       "usable for selection, invariant failures never converted to rows")

    # -- tie-break, converged, n_iter_max -----------------------------------
    if SVM["tie_break_pca_order"] != [50, 100, 200, None]:
        fail(f"the §F.7.5 compression order is {SVM['tie_break_pca_order']}, "
             f"not 50 -> 100 -> 200 -> no-PCA")
        return
    if sorted(SVM["tie_break_pca_order"], key=lambda v: (v is None, v)) != \
            [50, 100, 200, None]:
        fail("the compression order is not strongest-first")
        return
    if set(SVM["tie_break_pca_order"]) != set(SVM["pca_candidates"]):
        fail("the tie-break order does not cover exactly the frozen PCA grid")
        return
    if SVM["tie_break_order"] != ["inner_macro_bacc_desc", "C_asc",
                                  "compression_strength_desc"]:
        fail(f"the tie-break order is {SVM['tie_break_order']}")
        return
    ok("the frozen tie-break is spelled out as a total order: highest "
       "inner_macro_bacc, then smaller C, then 50 -> 100 -> 200 -> no-PCA")

    if not (SVM["converged_requires_all_inner_fits_clean"]
            and SVM["converged_requires_clean_final_refit"]
            and SVM["converged_requires_fold_estimable"]):
        fail("`converged` is not registered as a fold-lifecycle predicate")
        return
    ok("`converged = true` iff every inner-grid fit is clean, the selected "
       "final refit is clean, and the fold is ESTIMABLE (§F.7.3)")

    if not SVM["n_iter_max_is_configured_cap"] or \
            SVM["n_iter_max_is_observed_n_iter"] or \
            SVM["observed_iteration_count_persisted"]:
        fail("n_iter_max is redefined as an observed iteration count, or an "
             "observed count is persisted as a new output (§F.7.4)")
        return
    if SVM["n_iter_max_when_refit_attempted"] != SVM["max_iter"] or \
            SVM["n_iter_max_when_refit_skipped"] is not None:
        fail("the n_iter_max values do not match the configured cap / null rule")
        return
    ok(f"n_iter_max records the configured cap: {SVM['max_iter']} when a final "
       f"refit is attempted, null when it is skipped; no observed n_iter_ is "
       f"added as an output")

    # -- SVM_FOLD_FAILED closed scope ---------------------------------------
    if FF["is_catch_all"] or FF["substitutes_for_svm_not_converged"] or \
            not FF["unexpected_errors_fail_hard"]:
        fail("SVM_FOLD_FAILED is registered as a catch-all, or as a substitute "
             "for SVM_NOT_CONVERGED (§G.2a)")
        return
    required_exclusions = {
        "programming_errors", "invariant_failures", "schema_errors", "KeyError",
        "TypeError", "IndexError", "RuntimeError", "ValueError", "MemoryError",
        "KeyboardInterrupt", "SystemExit", "leakage_or_data_contract_violations"}
    missing = required_exclusions - set(FF["must_not_absorb"])
    if missing:
        fail(f"the SVM_FOLD_FAILED exclusion list omits {sorted(missing)}")
        return
    ok(f"SVM_FOLD_FAILED has a closed, narrow scope and may absorb none of "
       f"{len(FF['must_not_absorb'])} listed failure classes; unexpected "
       f"implementation errors still fail hard")

    # -- SENS-1 metadata record ---------------------------------------------
    if SENS1FIX["scientific_definition_changed"] or \
            SENS1FIX["numeric_recomputation_authorized"] or \
            SENS1FIX["implemented_by_this_version"]:
        fail("the SENS-1 record claims a definition change, a recomputation or "
             "an implementation (§E.2a)")
        return
    if SENS1FIX["required_label"] != "exact_overall_error_clopper_pearson_ucb":
        fail(f"the required SENS-1 score_type is "
             f"{SENS1FIX['required_label']!r}")
        return
    if SENS1FIX["required_label"] not in doc:
        fail("the protocol document does not state the required SENS-1 "
             "score_type label")
        return
    ok("the SENS-1 score_type correction is recorded as future implementation "
       f"work with the exact label {SENS1FIX['required_label']!r}; §E.2's "
       f"scientific definition is unchanged and no recomputation is authorized")


# ---------------------------------------------------------------------------
# 12. TRUE two-way structured constant mirroring  (audit finding 7)
# ---------------------------------------------------------------------------
def check_constant_mirroring() -> None:
    if not CONSTANTS_JSON.is_file():
        fail(f"constant registry missing: {CONSTANTS_JSON}")
        return
    registry = json.loads(CONSTANTS_JSON.read_text())

    flat_reg = flatten(registry)
    flat_val = flatten(MIRROR)

    only_reg = sorted(set(flat_reg) - set(flat_val))
    only_val = sorted(set(flat_val) - set(flat_reg))
    if only_reg:
        fail(f"constants in the registry but not in the validator: {only_reg}")
        return
    if only_val:
        fail(f"constants in the validator but not in the registry: {only_val}")
        return

    mismatched = [(k, flat_reg[k], flat_val[k]) for k in flat_reg
                  if flat_reg[k] != flat_val[k]]
    if mismatched:
        fail(f"constant values differ between the registry and the validator: "
             f"{mismatched[:6]}")
        return
    ok(f"true two-way structured mirroring: all {len(flat_val)} flattened "
       f"constants agree in name and value between the validator and "
       f"docs/stage_f_constants.json")

    # the document must declare the same version and identify itself honestly
    doc = PROTOCOL_DOC.read_text()
    for token in (PROTOCOL_VERSION, SUPERSEDED_VERSION, SUPERSEDED_COMMIT,
                  SUPERSEDED_TAG, MIRROR["store_root"], MIRROR["freeze_root"],
                  MIRROR["intended_result_tag"], INTENDED_TAG,
                  HIST["store_root"], HIST["freeze_root"], HIST["result_tag"]):
        if token not in doc:
            fail(f"the protocol document does not state {token!r}")
            return
    ok(f"the protocol document states every version, root and tag identifier: "
       f"{PROTOCOL_VERSION}, superseded {SUPERSEDED_VERSION} @ "
       f"{SUPERSEDED_COMMIT[:8]}… ({SUPERSEDED_TAG}), roots "
       f"{MIRROR['store_root']} / {MIRROR['freeze_root']}, historical roots "
       f"{HIST['store_root']} / {HIST['freeze_root']}, intended tags "
       f"{INTENDED_TAG} / {MIRROR['intended_result_tag']}")

    # self-identification: an untagged amendment draft, awaiting audit,
    # pre-result with respect to its OWN authoritative store  [amended at F.7]
    if "untagged amendment\nDRAFT" not in doc and \
            "untagged amendment" not in doc:
        fail("the protocol document does not identify itself as an untagged "
             "amendment draft")
        return
    if "pre-result with respect to this version's own authoritative" not in doc:
        fail("the protocol document does not state that it is pre-result with "
             "respect to its own authoritative store")
        return
    for phrase in ("scientifically\nsuperseded", "scientifically superseded"):
        if phrase in doc:
            break
    else:
        fail("the protocol document does not record the 1.0.2 result store as "
             "scientifically superseded")
        return
    ok("the protocol document identifies itself as an untagged amendment draft, "
       "pre-result with respect to its own authoritative store, and records the "
       "1.0.2 store as mechanically valid but scientifically superseded")

    for digest in MIRROR["consumed_artifacts"].values():
        if digest not in doc:
            fail(f"consumed-artifact digest {digest[:12]}… not pinned in the "
                 f"protocol document")
            return
    ok(f"all {len(CONSUMED)} consumed-artifact digests are pinned in the "
       f"protocol document")


# ---------------------------------------------------------------------------
# 13. no prohibited 1.0.0 claim survives as live text
# ---------------------------------------------------------------------------
#: A line is historical -- and therefore exempt -- if it names a superseded
#: version, a correction stage, or marks itself as withdrawn/struck through.
#: `1.0.1` is deliberately NOT exempting on its own: the document carries live
#: `**[+1.0.1]**` registry markers, and exempting those would blind the sweep.
_HISTORICAL = re.compile(
    r"1\.0\.0|F\.1a|F\.1b|[Ww]ithdrawn|WITHDRAWN|~~|§0\.1|§0\.2"
    r"|was wrong|no longer")


def _stale_lines(doc: str, patterns) -> list[int]:
    """1-indexed lines matching a prohibited pattern outside historical text."""
    lines = doc.splitlines()
    try:
        end_of_section_0 = next(i for i, l in enumerate(lines)
                                if l.startswith("## A."))
    except StopIteration:
        end_of_section_0 = 0
    out = []
    for i, line in enumerate(lines):
        if i < end_of_section_0 or _HISTORICAL.search(line):
            continue
        if any(p.search(line) for p in patterns):
            out.append(i + 1)
    return out


def check_no_stale_claims() -> None:
    doc = PROTOCOL_DOC.read_text()
    prohibited = {
        "exact split conformal for source_pooled":
            re.compile(r"exact split[- ]conformal.{0,80}source", re.I),
        "source-exchangeability": re.compile(r"source[- ]exchangeability"),
        "RQ4 point-estimate-only":
            re.compile(r"RQ4 is descriptive|reports point estimates only", re.I),
        "same units for C": re.compile(r"same units"),
        "operative random_state":
            re.compile(r"random_state.{0,40}(stochastic seed|operative seed)"),
        "CONFORMAL_LEVEL_UNATTAINABLE as a status":
            re.compile(r"status\s+CONFORMAL_LEVEL_UNATTAINABLE"),
        "SINGLE_CLASS_EVALUATION invalidating":
            re.compile(r"SINGLE_CLASS_EVALUATION"),
        # --- added at F.1b (re-audit findings 1 and 2) ---
        "Gate-1 column-for-column identity":
            re.compile(r"column[- ]for[- ]column", re.I),
        # An affirmative claim only: a negation ("not bit-identical",
        # "never exactly equal") is the correct live statement and is allowed.
        "oof_prob_mdd asserted bit-identical / exactly equal":
            re.compile(r"oof_prob_mdd"
                       r"(?:(?!not |never |rather than |instead of )[^|\n]){0,60}"
                       r"(bit[- ]identical|exactly equal|exact equality)", re.I),
        "block inside the SENS-5/6 cells.csv key":
            re.compile(r"cells\.csv[^|\n]*\|[^|\n]*\bblock\b"),
        "four-field cell key with 12 rows":
            re.compile(r"population,\s*operating_point,\s*method,\s*block"
                       r"[^|\n]*\|[^|\n]*\|[^|\n]*\b12\b"),
    }
    found = {}
    for name, pat in prohibited.items():
        lines = _stale_lines(doc, [pat])
        if lines:
            found[name] = lines
    if found:
        fail(f"prohibited superseded claims survive as live protocol text: "
             f"{found}")
        return
    ok(f"none of the {len(prohibited)} prohibited superseded claims survives "
       f"outside explicit historical audit text")


def main() -> int:
    print(f"Stage F.7 protocol validator — {PROTOCOL_VERSION}")
    print(f"repository root: {ROOT}\n")

    for name, fn in (
        ("artifacts", check_artifacts),
        ("gate-1 OOF binding (by value)", check_gate1_oof_binding),
        ("SENS-5/6 key + cardinality schema", check_sens56_schema),
        ("cohort", check_cohort),
        ("splits", check_splits),
        ("weights + scores", check_weights_and_scores),
        ("conformal + guarantee labels", check_conformal),
        ("RQ4 CI registry", check_ci_registry),
        ("status / degeneracy semantics", check_status_semantics),
        ("SENS-6 algorithm", check_sens6),
        ("sensitivity cardinalities", check_sensitivity_cardinalities),
        ("svm", check_svm),
        ("svm 2.0.0 contracts", check_svm_v2_contracts),
        ("svm warning representations", check_svm_warning_representations),
        ("svm SVM_FOLD_FAILED allow-list", check_svm_fold_failed_allow_list),
        ("result store + tag", check_no_result_store),
        ("historical authority intact", check_historical_authority),
        ("historical freeze byte identity", check_historical_freeze_bytes),
        ("constant mirroring", check_constant_mirroring),
        ("stale claims", check_no_stale_claims),
    ):
        before = len(FAILURES)
        fn()
        status = "FAIL" if len(FAILURES) > before else "PASS"
        print(f"[{status}] {name}")

    print("\n--- checks ---")
    for n in NOTES:
        print("  .", n)

    if FAILURES:
        print("\n--- FAILURES ---")
        for f in FAILURES:
            print("  x", f)
        print(f"\nVALIDATOR: FAIL ({len(FAILURES)} failure(s))")
        return 1

    print(f"\nVALIDATOR: PASS ({len(NOTES)} checks)")
    print("Nothing was written. No Stage-F scientific quantity was computed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
