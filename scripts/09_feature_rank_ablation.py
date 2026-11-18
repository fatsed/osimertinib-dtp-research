"""09 — Diagnose stage-08 source-only transfer drop without using target labels for fitting.

Run in project root:
    python scripts/09_feature_rank_ablation.py --self-test
    python scripts/09_feature_rank_ablation.py --exploratory-only

Uses the same verified inputs and study loader as numbered stage 07; no source files
are modified and no network request is made. This is an EXPLORATORY ablation;
it is NOT independent external validation and is NOT an approved normalization.

Conditions (source-only model fitting in all cases):
 A: Shared gene headers; percentile ranks within shared gene space (stage 07).
 B: Shared gene headers; percentile ranks over each sample's full gene space.
 C: Source-only gene selection; full-space ranks, missing target genes neutralized
    by inserting source-train feature means (stage 08 source-only sensitivity).
 D: Freeze C's source-selected gene names; retain only shared genes; train and
    evaluate on common-space ranks (isolates feature choice as far as possible).

Outputs: outputs/ablation/09_ablation_summary.json and 09_ablation_metrics.csv.
Only exploratory source-trained transfer; no target label is used in fitting.
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
S07 = ROOT / "scripts" / "07_external_stress_test.py"
OUT = ROOT / "outputs" / "ablation"


def load07():
    if not S07.is_file():
        raise FileNotFoundError(f"Missing prerequisite stage 07: {S07}")
    spec = importlib.util.spec_from_file_location("experiment07", S07)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load stage 07")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if m.FIXED_HYPERPARAMETERS != {"model": "LogisticRegression", "k": 100, "C": 0.01}:
        raise ValueError("The frozen stage-07 model configuration changed")
    return m


def selected_genes(model, genes):
    var_mask = model.named_steps["variance"].get_support()
    if len(var_mask) != len(genes):
        raise ValueError("Model feature universe differs from supplied gene IDs")
    filtered = [name for name, take in zip(genes, var_mask) if take]
    selected = [name for name, take in zip(filtered, model.named_steps["anova"].get_support()) if take]
    if len(selected) != 100:
        raise ValueError(f"Expected 100 selected features, received {len(selected)}")
    return selected


def rank_space(matrix, np, rankdata):
    if matrix.ndim != 2 or matrix.shape[1] < 100:
        raise ValueError("Expected samples x genes with at least 100 genes")
    return rankdata(matrix, axis=1, method="average") / matrix.shape[1]


def evaluate_model(model, X_map, y_map, stage07, ml, method):
    metrics = []
    for gse in stage07.TARGETS:
        pred = model.predict(X_map[gse])
        m = stage07.model_metrics(y_map[gse], pred, ml)
        metrics.append({"condition": method, "study": gse, **m})
    return metrics


def run():
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    stage07 = load07()
    ml = locals()
    meta = stage07.META
    required = [meta / name for name in (
        "evaluation_protocol.json", "cleaning_report.json",
        "expression_alignment.csv", "cohort_review.csv")]
    missing = [str(f) for f in required if not f.is_file()]
    if missing:
        raise FileNotFoundError("Required existing reports missing: " + ", ".join(missing))

    protocol = json.loads(required[0].read_text(encoding="utf-8"))
    if protocol.get("source_study") != "GSE193258" or len(protocol.get("source_gsm_to_label", [])) != 24:
        raise ValueError("Unexpected frozen source protocol")
    if protocol.get("flags", {}).get("biology_approved_for_training") is not False:
        raise ValueError("Only the existing unapproved exploratory protocol is supported")
    p_source = stage07.safe_local_path(protocol["source_matrix"], "processed")
    genes, columns, source = stage07.load_matrix(p_source, protocol["source_matrix_sha256"], np)
    if columns != [row["gsm"] for row in protocol["source_gsm_to_label"]]:
        raise ValueError("Frozen source GSM column order changed")
    if len(genes) != protocol.get("source_gene_count") or len(set(genes)) != len(genes):
        raise ValueError("Source gene count/uniqueness mismatch")
    y_source = np.array([stage07.LABELS[row["state"]] for row in protocol["source_gsm_to_label"]], dtype=int)
    from collections import Counter
    if Counter(y_source) != {0: 12, 1: 12}:
        raise ValueError("Unexpected source label distribution")

    cleaning = json.loads(required[1].read_text(encoding="utf-8"))
    targets = stage07.prepare_targets(np, cleaning, stage07.csv_rows(required[2]), stage07.csv_rows(required[3]), ROOT)
    for gse in stage07.TARGETS:
        if gse not in targets:
            raise ValueError(f"Missing target: {gse}")
    common_set = set(genes)
    for gse in stage07.TARGETS:
        common_set &= set(targets[gse]["genes"])
    common_genes = [g for g in genes if g in common_set]
    if len(common_genes) < 1000:
        raise ValueError("Very small exact-name shared gene universe")
    source_idx = {g: i for i, g in enumerate(genes)}
    shared_src_idx = [source_idx[g] for g in common_genes]
    # A — common genes ranked only against other common genes.
    X_a = rank_space(source[:, shared_src_idx], np, rankdata)
    # B — same feature candidates, but each sample ranked against its whole gene universe.
    X_full_src = rank_space(source, np, rankdata)
    X_b = X_full_src[:, shared_src_idx]
    # C — source-only candidate universe; no target header in feature selection.
    X_c = X_full_src
    X_targets_a, X_targets_b, target_full_ranks, y_targets = {}, {}, {}, {}
    for gse in stage07.TARGETS:
        t = targets[gse]
        idx = {g: i for i, g in enumerate(t["genes"])}
        shared_target_idx = [idx[g] for g in common_genes]
        X_targets_a[gse] = rank_space(t["matrix"][:, shared_target_idx], np, rankdata)
        full = rank_space(t["matrix"], np, rankdata)
        target_full_ranks[gse] = full
        X_targets_b[gse] = full[:, shared_target_idx]
        y_targets[gse] = np.array([stage07.LABELS[s] for s in t["states"]], dtype=int)

    model_a = stage07.pipeline(ml).fit(X_a, y_source)
    model_b = stage07.pipeline(ml).fit(X_b, y_source)
    model_c = stage07.pipeline(ml).fit(X_c, y_source)
    genes_a = selected_genes(model_a, common_genes)
    genes_b = selected_genes(model_b, common_genes)
    genes_c = selected_genes(model_c, genes)
    a_metrics = evaluate_model(model_a, X_targets_a, y_targets, stage07, ml, "A_shared_genes_shared_ranks")
    b_metrics = evaluate_model(model_b, X_targets_b, y_targets, stage07, ml, "B_shared_genes_full_ranks")

    # Reproduce stage 08 method C with explicit coefficient diagnostics.
    train_c_selected_ranks = X_c[:, [source_idx[g] for g in genes_c]]
    source_feature_means = train_c_selected_ranks.mean(axis=0)
    missing_by_target, c_X = {}, {}
    beta = model_c.named_steps["classifier"].coef_[0]
    for gse in stage07.TARGETS:
        idx = {g: i for i, g in enumerate(targets[gse]["genes"])}
        matrix = np.broadcast_to(source_feature_means, (len(y_targets[gse]), len(genes_c))).copy()
        missing = []
        for k, g in enumerate(genes_c):
            if g not in idx:
                missing.append(g)
            else:
                matrix[:, k] = target_full_ranks[gse][:, idx[g]]
        c_X[gse] = matrix
        selected_missing = set(missing)
        abs_total = float(np.sum(np.abs(beta)))
        missing_beta_fraction = float(sum(abs(beta[k]) for k, g in enumerate(genes_c) if g in selected_missing) / abs_total) if abs_total else 0.
        missing_by_target[gse] = {
            "missing_gene_count": len(missing), "missing_genes": missing,
            "missing_absolute_coefficient_fraction": missing_beta_fraction,
        }
    c_X_scaled = {gse: model_c.named_steps["scale"].transform(c_X[gse]) for gse in stage07.TARGETS}
    c_metrics = evaluate_model(
        model_c.named_steps["classifier"], c_X_scaled, y_targets, stage07, ml,
        "C_source_only_selection_full_ranks")

    # D — selected genes are frozen from source-only C. Retain genes present in
    # every target, then fit ONLY source expression, on COMMON-space ranks.
    genes_d = [g for g in genes_c if g in common_set]
    if len(genes_d) < 25:
        raise ValueError("Too few source-only selected genes shared with targets")
    indices = {g: i for i, g in enumerate(common_genes)}
    take = [indices[g] for g in genes_d]
    scaler_d = StandardScaler().fit(X_a[:, take])
    clf_d = LogisticRegression(C=stage07.FIXED_HYPERPARAMETERS["C"], class_weight="balanced",
                               solver="liblinear", max_iter=2000, random_state=0)
    clf_d.fit(scaler_d.transform(X_a[:, take]), y_source)
    x_d = {gse: scaler_d.transform(X_targets_a[gse][:, take]) for gse in stage07.TARGETS}
    d_metrics = evaluate_model(clf_d, x_d, y_targets, stage07, ml,
                               "D_source_selected_shared_rank_refit")

    all_metrics = a_metrics + b_metrics + c_metrics + d_metrics
    if len(all_metrics) != 4 * len(stage07.TARGETS):
        raise AssertionError("Ablation metrics incomplete")
    metrics_by_condition = {}
    for row in all_metrics:
        metrics_by_condition.setdefault(row["condition"], []).append(row["balanced_accuracy"])
    means = {condition: float(np.mean(scores)) for condition, scores in metrics_by_condition.items()}
    overlaps = {
        "A_vs_B":len(set(genes_a)&set(genes_b)),
        "A_vs_C":len(set(genes_a)&set(genes_c)),
        "B_vs_C":len(set(genes_b)&set(genes_c)),
        "source_only_top100_shared_with_all_targets":len(genes_d),
    }
    result = {
        "status":"EXPLORATORY_ABLATION_NOT_INDEPENDENT_VALIDATION",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "stage07_code_sha256":stage07.digest(S07),
        "source_matrix_sha256":protocol["source_matrix_sha256"],
        "source_n":len(y_source), "source_genes":len(genes),
        "exact_gene_name_intersection_across_three_targets":len(common_genes),
        "models_fit_on_source_only":True,
        "external_labels_used_only_for_scoring":True,
        "study_macro_balanced_accuracy":means,
        "selected_gene_overlap":overlaps,
        "source_only_missing_gene_diagnostics":missing_by_target,
        "selected_genes": {"A":genes_a, "B":genes_b, "C":genes_c, "D":genes_d},
        "per_study_metrics": all_metrics,
        "interpretation":{
            "A_vs_B":"Same shared candidate feature set; differences involve full-space vs shared-space percentile ranks and training-dependent feature selection.",
            "B_vs_C":"Same full-space percentile ranks; differences involve selection among shared genes vs all source genes and missing-gene handling.",
            "A_vs_D":"Same common-space ranks; compared source-only-selected shared features against common-universe feature selection, with refitting only on source data.",
            "C_vs_D":"Both use source-only-selected features; differs in rank denominator, missing-gene handling and classifier refit.",
        },
        "cautions":[
            "Exact gene names are not harmonized identifiers; same name need not mean same annotation.",
            "All target datasets have already been inspected; no untouched independent external validation.",
            "The shared-gene feature universe A/B/D consults target gene headers but not target expression values or labels; interpret as a sensitivity test.",
            "Control vs DTP is confounded with dose, time and culture conditions, especially in source GSE193258.",
            "Source-only selected features C: missing genes are filled at source train means; sensitivity result is not a validated normalization.",
            "No condition is claimed to resolve batch effects, select causal DTP features or establish a new method.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    outcsv = OUT / "09_ablation_metrics.csv"
    with outcsv.open("w", encoding="utf-8", newline="") as f:
        fields = ["condition", "study", "n", "balanced_accuracy", "control_recall", "dtp_recall", "tn", "fp", "fn", "tp"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_metrics)
    (OUT / "09_ablation_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Stage 09: source n={len(y_source)}, source genes={len(genes):,}, common genes={len(common_genes):,}")
    for condition in metrics_by_condition:
        print(f"{condition}: study-mean bACC={means[condition]:.3f}")
        for row in all_metrics:
            if row["condition"] == condition:
                print(f"  {row['study']}: bACC={row['balanced_accuracy']:.3f}, Control recall={row['control_recall']:.3f}, DTP recall={row['dtp_recall']:.3f}")
    print(f"Selected-gene overlap: A/B={overlaps['A_vs_B']}, A/C={overlaps['A_vs_C']}, B/C={overlaps['B_vs_C']}")
    for gse in stage07.TARGETS:
        info=missing_by_target[gse]
        print(f"  C missing {gse}: {info['missing_gene_count']} genes, abs coefficient fraction={info['missing_absolute_coefficient_fraction']:.3f}")
    print("Saved outputs/ablation/09_ablation_summary.json and 09_ablation_metrics.csv")
    print("EXPLORATORY ONLY: no biological approval or independent external validation")
    return result


def self_test():
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    stage07=load07()
    ml=locals()
    rng=np.random.default_rng(2109)
    genes=[f"G{i}" for i in range(250)]
    y=np.array(([0]*3+[1]*3)*4,dtype=int)
    X=rng.normal(0,1,(24,250)); X[:,0]+=3*y
    ranks=rank_space(X,np,rankdata)
    assert ranks.shape==(24,250)
    assert np.min(ranks)>=0 and np.max(ranks)<=1
    fit=stage07.pipeline(ml).fit(ranks,y)
    names=selected_genes(fit,genes)
    assert len(names)==len(set(names))==100
    assert "G0" in names
    keep=[i for i in range(250) if i<240]
    B=rank_space(X[:,keep],np,rankdata)
    assert not np.array_equal(B,ranks[:,keep]), "Rank denominator sensitivity fixture failed"
    assert len(set(genes[:240])&set(names))<=100
    assert abs(stage07.model_metrics(np.array([0,0,1,1]),np.array([0,0,1,1]),ml)["balanced_accuracy"]-1.)<1e-12
    print("SELF-TEST PASSED (feature masks, per-sample rank denominators, deterministic source-only selection)")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--exploratory-only",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test();return 0
    if not args.exploratory_only:
        p.error("Explicit --exploratory-only required")
    run();return 0


if __name__=="__main__":
    raise SystemExit(main())
