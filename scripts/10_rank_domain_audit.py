"""Stage 10 — Frozen-model cross-rank audit of the stage 09 transfer discrepancy.

This is a diagnostic, POST-HOC analysis, not independent validation or a
normalization method. The same two SOURCE-fitted pipelines are evaluated under
both target rank conventions WITHOUT retraining, recalibration or target-label
selection. No downloaded data or historical outputs are modified.

From the repository root:
    python scripts/10_rank_domain_audit.py --self-test
    python scripts/10_rank_domain_audit.py --exploratory-only

Requires existing numbered scripts 07, 09, their input reports, and stage 09
output; outputs are written under outputs/rank_domain (gitignored).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE09_SCRIPT = ROOT / "scripts" / "09_feature_rank_ablation.py"
STAGE09_REPORT = ROOT / "outputs" / "ablation" / "09_ablation_summary.json"
OUTPUT = ROOT / "outputs" / "rank_domain"
CONDITIONS = (
    ("source_shared__target_shared", "shared", "shared"),
    ("source_shared__target_full", "shared", "full"),
    ("source_full__target_shared", "full", "shared"),
    ("source_full__target_full", "full", "full"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for buf in iter(lambda: file.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def load_stage09():
    if not STAGE09_SCRIPT.is_file():
        raise FileNotFoundError(STAGE09_SCRIPT)
    spec = importlib.util.spec_from_file_location("stage09_for_stage10", STAGE09_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Stage 09 import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_predictions(model, matrix, actual, ml):
    import numpy as np
    predictions = model.predict(matrix)
    margins = model.decision_function(matrix)
    if margins.shape != (len(actual),):
        raise ValueError("Unexpected classifier decision function shape")
    report = ml.model_metrics(actual, predictions, {
        "confusion_matrix": __import__("sklearn.metrics", fromlist=["confusion_matrix"]).confusion_matrix,
        "balanced_accuracy_score": __import__("sklearn.metrics", fromlist=["balanced_accuracy_score"]).balanced_accuracy_score,
    })
    return predictions, margins, report


def rank_summary(matrix, selected_indices):
    import numpy as np
    selected = matrix[:, selected_indices]
    return {
        "median_selected_rank": float(np.median(selected)),
        "mean_selected_rank": float(np.mean(selected)),
        "fraction_selected_below_0_25": float(np.mean(selected < 0.25)),
        "fraction_selected_above_0_75": float(np.mean(selected > 0.75)),
    }


def analyze():
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not STAGE09_REPORT.is_file():
        raise FileNotFoundError(f"Run stage 09 first; missing {STAGE09_REPORT}")
    previous = json.loads(STAGE09_REPORT.read_text(encoding="utf-8"))
    if previous.get("status") != "EXPLORATORY_ABLATION_NOT_INDEPENDENT_VALIDATION":
        raise ValueError("Stage 09 result is not the expected exploratory report")
    stage09 = load_stage09()
    stage07 = stage09.load07()
    if previous.get("stage07_code_sha256") != sha256(stage09.S07):
        raise ValueError("Stage 07 script changed since stage 09")
    if stage07.FIXED_HYPERPARAMETERS != {"model": "LogisticRegression", "k": 100, "C": 0.01}:
        raise ValueError("Stage 07 frozen model configuration changed")
    paths = [stage07.META / x for x in (
        "evaluation_protocol.json", "cleaning_report.json",
        "expression_alignment.csv", "cohort_review.csv")]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("One or more existing metadata reports are missing")
    protocol = json.loads(paths[0].read_text(encoding="utf-8"))
    if protocol.get("source_study") != "GSE193258" or len(protocol.get("source_gsm_to_label", [])) != 24:
        raise ValueError("Incorrect source protocol")
    if protocol.get("flags", {}).get("biology_approved_for_training") is not False:
        raise ValueError("Only unapproved exploratory source protocol supported")
    psource = stage07.safe_local_path(protocol["source_matrix"], "processed")
    genes, columns, source = stage07.load_matrix(psource, protocol["source_matrix_sha256"], np)
    if columns != [row["gsm"] for row in protocol["source_gsm_to_label"]]:
        raise ValueError("Source GSM ordering mismatch")
    if protocol["source_matrix_sha256"] != previous["source_matrix_sha256"]:
        raise ValueError("Source matrix changed since stage 09")
    if len(genes) != previous["source_genes"]:
        raise ValueError("Gene count changed since stage 09")
    cleaning = json.loads(paths[1].read_text(encoding="utf-8"))
    targets = stage07.prepare_targets(np, cleaning, stage07.csv_rows(paths[2]),
                                      stage07.csv_rows(paths[3]), ROOT)
    common = set(genes)
    for study in stage07.TARGETS:
        common.intersection_update(targets[study]["genes"])
    common_genes = [g for g in genes if g in common]
    if len(common_genes) != previous["exact_gene_name_intersection_across_three_targets"]:
        raise ValueError("Shared feature universe changed since stage 09")
    gene_ids = {g: i for i, g in enumerate(genes)}
    source_common = source[:, [gene_ids[g] for g in common_genes]]
    Xsource_shared = stage09.rank_space(source_common, np, rankdata)
    Xsource_full = stage09.rank_space(source, np, rankdata)[:, [gene_ids[g] for g in common_genes]]
    ysource = np.asarray([stage07.LABELS[r["state"]] for r in protocol["source_gsm_to_label"]], dtype=int)
    ml = {"Pipeline": Pipeline, "VarianceThreshold": VarianceThreshold,
          "SelectKBest": SelectKBest, "f_classif": f_classif,
          "StandardScaler": StandardScaler, "LogisticRegression": LogisticRegression}
    model_shared = stage07.pipeline(ml).fit(Xsource_shared, ysource)
    model_full = stage07.pipeline(ml).fit(Xsource_full, ysource)
    selected_shared = stage09.selected_genes(model_shared, common_genes)
    selected_full = stage09.selected_genes(model_full, common_genes)
    if selected_shared != previous["selected_genes"]["A"] or selected_full != previous["selected_genes"]["B"]:
        raise ValueError("Selected features differ from recorded stage 09; stop to investigate")
    common_index = {g: i for i, g in enumerate(common_genes)}
    selected_ix_shared = [common_index[g] for g in selected_shared]
    selected_ix_full = [common_index[g] for g in selected_full]
    per_study = []
    details = {}
    for gse in stage07.TARGETS:
        target = targets[gse]
        n = len(target["states"])
        yt = np.array([stage07.LABELS[r] for r in target["states"]], dtype=int)
        target_indices = {g: i for i, g in enumerate(target["genes"])}
        values = target["matrix"]
        Xt_shared = stage09.rank_space(values[:, [target_indices[g] for g in common_genes]], np, rankdata)
        Xt_full = stage09.rank_space(values, np, rankdata)[:, [target_indices[g] for g in common_genes]]
        rank_shift = Xt_full[:, selected_ix_shared] - Xt_shared[:, selected_ix_shared]
        rowset = {}
        for condition, source_kind, target_kind in CONDITIONS:
            model = model_shared if source_kind == "shared" else model_full
            Xtarget = Xt_shared if target_kind == "shared" else Xt_full
            pred, margin, metrics = summarize_predictions(model, Xtarget, yt, stage07)
            per_study.append({"study":gse,"condition":condition,**metrics,
                              "mean_margin":float(np.mean(margin)),
                              "median_margin":float(np.median(margin)),
                              "predicted_dtp_fraction":float(np.mean(pred==1))})
            rowset[condition] = {"pred": pred, "margin": margin, "metrics": metrics}
        # Hold model fixed, vary only target rank rule. This is the key diagnostic.
        a1 = rowset["source_shared__target_shared"]
        a2 = rowset["source_shared__target_full"]
        b1 = rowset["source_full__target_full"]
        b2 = rowset["source_full__target_shared"]
        details[gse] = {
            "n_samples":n,
            "source_gene_count":len(genes),
            "target_gene_count":len(target["genes"]),
            "shared_gene_count":len(common_genes),
            "source_selected_gene_count":len(selected_shared),
            "median_selected_rank_shift_full_minus_shared":float(np.median(rank_shift)),
            "median_absolute_selected_rank_shift":float(np.median(np.abs(rank_shift))),
            "mean_selected_rank_shift_full_minus_shared":float(np.mean(rank_shift)),
            "target_zero_fraction_all_genes":float(np.mean(values==0)),
            "target_zero_fraction_common_genes":float(np.mean(values[:,[target_indices[g] for g in common_genes]]==0)),
            "selected_gene_rank_stats_shared":rank_summary(Xt_shared,selected_ix_shared),
            "selected_gene_rank_stats_full":rank_summary(Xt_full,selected_ix_shared),
            "fixed_shared_model_switches_when_target_rank_changes":int(np.sum(a1["pred"]!=a2["pred"])),
            "fixed_full_model_switches_when_target_rank_changes":int(np.sum(b1["pred"]!=b2["pred"])),
            "fixed_shared_model_median_margin_change_target_full_minus_shared":float(np.median(a2["margin"]-a1["margin"])),
            "fixed_full_model_median_margin_change_target_full_minus_shared":float(np.median(b1["margin"]-b2["margin"])),
            "expected_control_count":int(np.sum(yt==0)),
            "expected_dtp_count":int(np.sum(yt==1)),
        }
        # Freeze stage 09 reference comparisons: AA is stage 09 A, BB is stage 09 B.
        ref = [r for r in previous["per_study_metrics"] if r["study"]==gse]
        ref_by = {r["condition"]:r for r in ref}
        if (a1["metrics"]["balanced_accuracy"] != ref_by["A_shared_genes_shared_ranks"]["balanced_accuracy"] or
            b1["metrics"]["balanced_accuracy"] != ref_by["B_shared_genes_full_ranks"]["balanced_accuracy"]):
            raise ValueError(f"Stage 09 baseline mismatch for {gse}")
    report = {
        "status":"EXPLORATORY_FIXED_MODEL_RANK_TRANSFORM_DIAGNOSTIC",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_study":"GSE193258",
        "source_samples":24,
        "source_zero_fraction_all_genes":float(np.mean(source==0)),
        "source_zero_fraction_shared_genes":float(np.mean(source_common==0)),
        "common_gene_count":len(common_genes),
        "A_B_selected_gene_sets_identical":selected_shared==selected_full,
        "A_B_selected_gene_overlap":len(set(selected_shared)&set(selected_full)),
        "models_fit_source_only":True,
        "models_frozen_between_target_rank_conditions":True,
        "target_labels_used_only_for_scoring":True,
        "target_gene_headers_used_for_common_universe":True,
        "stage09_report_sha256":sha256(STAGE09_REPORT),
        "source_matrix_sha256":protocol["source_matrix_sha256"],
        "per_study":details,
        "metrics":per_study,
        "limitations":[
            "Post-hoc test on inspected cohorts; no independent validation or new algorithm claim.",
            "A shared-gene universe was defined using external gene headers (not labels).",
            "Unharmonized gene-name equivalence, treatment-time confounding, and biological sample review remain unresolved.",
            "Cross-rank diagnostics are not a validated normalization; no target-based calibration is performed.",
            "Model was source-fitted, but source and target receive different feature-space coverage under full ranks.",
        ],
    }
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"10_rank_domain_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    with (OUTPUT/"10_rank_domain_metrics.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file, fieldnames=list(per_study[0]));writer.writeheader();writer.writerows(per_study)
    print(f"Stage 10: source n=24, common exact gene names={len(common_genes):,}")
    print(f"Stage 09 A/B selected genes identical: {selected_shared==selected_full} ({len(set(selected_shared)&set(selected_full))}/100 overlap)")
    for gse in stage07.TARGETS:
        d=details[gse]
        rows = {r["condition"]:r for r in per_study if r["study"]==gse}
        fmt=lambda k:f"{rows[k]['balanced_accuracy']:.3f}"
        print(f"{gse}: AA={fmt('source_shared__target_shared')} AB={fmt('source_shared__target_full')} BA={fmt('source_full__target_shared')} BB={fmt('source_full__target_full')}; "
              f"A-fixed prediction switches={d['fixed_shared_model_switches_when_target_rank_changes']}/{d['n_samples']}; "
              f"median selected-rank shift={d['median_selected_rank_shift_full_minus_shared']:+.4f}")
    print("Saved outputs/rank_domain/10_rank_domain_summary.json and 10_rank_domain_metrics.csv")
    print("EXPLORATORY ONLY — NO INDEPENDENT VALIDATION OR BIOLOGICAL APPROVAL")
    return report


def self_test():
    import numpy as np
    from scipy.stats import rankdata
    # Same named genes, but additional target genes change ranks and ties.
    source=np.array([[0,1,2,3,4,5],[5,4,3,2,1,0]],dtype=float)
    target=np.array([[0,1,2,3,4,5,10,11],[5,4,3,2,1,0,10,11]],dtype=float)
    def ranks(x):return rankdata(x,axis=1,method="average")/x.shape[1]
    a=ranks(target[:,:6]);b=ranks(target)[:,:6]
    assert a.shape==b.shape==(2,6)
    assert np.any(a!=b)
    assert np.all(b<a)
    assert 0<=rank_summary(a,[0,1])["mean_selected_rank"]<=1
    assert np.median(b-a)<0
    assert len(CONDITIONS)==4 and len(set(x[0] for x in CONDITIONS))==4
    print("SELF-TEST PASSED (frozen-model four-condition plan, rank-space shift, deterministic summaries)")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--exploratory-only",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test();return 0
    if not args.exploratory_only:
        parser.error("Explicit --exploratory-only required")
    analyze();return 0


if __name__=="__main__":
    raise SystemExit(main())
