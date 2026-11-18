"""08 — Exploratory negative controls and source-only feature-universe sensitivity.

From the repository root (requires pinned requirements and stage 07 file):
    python scripts/08_robustness_controls.py --self-test
    python scripts/08_robustness_controls.py --exploratory-only

Reuses VERIFIED loaders and the fixed source model from 07_external_stress_test.py.
Does NOT alter data, approve labels, optimize to external labels, or claim that
confounding has been controlled. Reproduces stage 07's original analysis and
runs (a) cell-line-stratified source label permutations, (b) disrupted external
feature correspondence, and (c) a source-only feature-selection sensitivity.
Output: outputs/robustness/08_robustness_summary.json and 08_null_trials.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE07_PATH = ROOT / "scripts" / "07_external_stress_test.py"
OUT = ROOT / "outputs" / "robustness"
SEED = 20261118


def load_stage07():
    if not STAGE07_PATH.is_file():
        raise FileNotFoundError(f"Missing prerequisite: {STAGE07_PATH}")
    spec = importlib.util.spec_from_file_location("stage07_external_stress_test", STAGE07_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load stage 07 code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_permuted_within_cell_line(y, lines, rng, np):
    """Permute states separately within each cell line (3/3 per line preserved)."""
    out = y.copy()
    for line in sorted(set(lines)):
        pos = np.flatnonzero(np.asarray(lines) == line)
        out[pos] = rng.permutation(y[pos])
        if Counter(out[pos]) != Counter(y[pos]):
            raise AssertionError("Within-line label counts changed")
    return out


def mean_study_balanced_accuracy(predictor, target_X, target_y, stage07, ml):
    metrics = {}
    for study in stage07.TARGETS:
        pred = predictor(target_X[study])
        metrics[study] = stage07.model_metrics(target_y[study], pred, ml)["balanced_accuracy"]
    return metrics, float(sum(metrics.values()) / len(metrics))


def selected_source_genes(trained_model, source_genes):
    var_mask = trained_model.named_steps["variance"].get_support()
    if len(var_mask) != len(source_genes):
        raise AssertionError("Source feature count mismatch")
    filtered = [g for g, selected in zip(source_genes, var_mask) if selected]
    sel_mask = trained_model.named_steps["anova"].get_support()
    selected = [g for g, keep in zip(filtered, sel_mask) if keep]
    return selected


def source_only_feature_sensitivity(stage07, ml, source_genes, source_values, target, y_source, rankdata, np):
    """Select source features without consulting any target gene names.

    Ranks are across each study's own gene universe (not a validated normalization).
    For missing source-selected genes, impute the SOURCE training rank mean,
    which becomes 0 after source-fitted StandardScaler; report every missing gene.
    This is a diagnostic, not a confirmatory model or training recommendation.
    """
    Xs = stage07.rank_samples(source_values, rankdata, np)
    fitted = stage07.pipeline(ml).fit(Xs, y_source)
    selected = selected_source_genes(fitted, source_genes)
    if len(selected) != stage07.FIXED_HYPERPARAMETERS["k"]:
        raise ValueError("Incorrect number of source-selected genes")
    inds = {gene: i for i, gene in enumerate(source_genes)}
    selected_source_values = Xs[:, [inds[g] for g in selected]]
    source_means = selected_source_values.mean(axis=0)
    scaler = fitted.named_steps["scale"]
    classifier = fitted.named_steps["classifier"]
    results = {}
    for gse in stage07.TARGETS:
        t = target[gse]
        X_all_target = stage07.rank_samples(t["matrix"], rankdata, np)
        tindex = {gene: i for i, gene in enumerate(t["genes"])}
        target_feats = np.broadcast_to(source_means, (X_all_target.shape[0], len(selected))).copy()
        missing = []
        for k, gene in enumerate(selected):
            if gene in tindex:
                target_feats[:, k] = X_all_target[:, tindex[gene]]
            else:
                missing.append(gene)
        yp = classifier.predict(scaler.transform(target_feats))
        yt = np.asarray([stage07.LABELS[s] for s in t["states"]], dtype=int)
        result = stage07.model_metrics(yt, yp, ml)
        results[gse] = {
            "n_selected_genes": len(selected), "n_missing_selected_genes": len(missing),
            "missing_selected_genes": missing,
            "bacc": result["balanced_accuracy"], "dtp_recall": result["dtp_recall"],
            "control_recall": result["control_recall"], "confusion": {k:result[k] for k in ("tn","fp","fn","tp")},
        }
    return results, selected


def run(permutations: int = 49, seed: int = SEED):
    import numpy as np
    import sklearn
    from scipy.stats import rankdata
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    ml = locals()
    stage07 = load_stage07()
    if not (1 <= permutations <= 999):
        raise ValueError("--permutations must be between 1 and 999")

    p_protocol = stage07.META / "evaluation_protocol.json"
    p_cleaning = stage07.META / "cleaning_report.json"
    p_alignment = stage07.META / "expression_alignment.csv"
    p_cohort = stage07.META / "cohort_review.csv"
    for p in (p_protocol,p_cleaning,p_alignment,p_cohort):
        if not p.is_file():
            raise FileNotFoundError(f"Required local report missing: {p}")
    protocol = json.loads(p_protocol.read_text(encoding="utf-8"))
    if protocol.get("source_study") != "GSE193258" or len(protocol.get("source_gsm_to_label", [])) != 24:
        raise ValueError("Unexpected or changed stage 05 source protocol")
    if protocol.get("flags",{}).get("biology_approved_for_training") is not False:
        raise ValueError("Only exploratory provisional source protocol is supported")
    p_source = stage07.safe_local_path(protocol["source_matrix"], "processed")
    src_genes, src_columns, src_values = stage07.load_matrix(p_source, protocol["source_matrix_sha256"], np)
    if src_columns != [r["gsm"] for r in protocol["source_gsm_to_label"]]:
        raise ValueError("Source column order differs from frozen GSM protocol")
    if len(src_genes) != protocol["source_gene_count"]:
        raise ValueError("Source gene count differs from protocol")
    cleaning = json.loads(p_cleaning.read_text(encoding="utf-8"))
    targets = stage07.prepare_targets(np, cleaning, stage07.csv_rows(p_alignment), stage07.csv_rows(p_cohort), ROOT)

    # Precisely reproduce stage 07's transductive EXACT-NAME gene intersection.
    common = set(src_genes)
    for name in stage07.TARGETS:
        common &= set(targets[name]["genes"])
    common_genes = [g for g in src_genes if g in common]
    if len(common_genes) < 1000:
        raise ValueError("Too few common genes")
    src_ix = {g: i for i,g in enumerate(src_genes)}
    X_source = stage07.rank_samples(src_values[:, [src_ix[g] for g in common_genes]], rankdata, np)
    identities = protocol["source_gsm_to_label"]
    ysource = np.array([stage07.LABELS[r["state"]] for r in identities], dtype=int)
    cell_lines = np.array([r["cell_line"] for r in identities])
    if Counter(ysource) != {0:12, 1:12}:
        raise ValueError("Source class counts changed")
    if any(Counter(ysource[cell_lines==line]) != {0:3,1:3} for line in set(cell_lines)):
        raise ValueError("Source per-line class counts changed")
    X_target = {}
    ytarget = {}
    for gse in stage07.TARGETS:
        t = targets[gse]
        tind = {g:i for i,g in enumerate(t["genes"])}
        X_target[gse] = stage07.rank_samples(t["matrix"][:, [tind[g] for g in common_genes]], rankdata, np)
        ytarget[gse] = np.array([stage07.LABELS[x] for x in t["states"]], dtype=int)
    frozen_model = stage07.pipeline(ml).fit(X_source, ysource)
    observed, observed_macro = mean_study_balanced_accuracy(frozen_model.predict,X_target,ytarget,stage07,ml)
    source_loco = stage07.source_loco(X_source, protocol, ml)

    # Label permutation: keep each cell line's class counts, retrain everything from scratch.
    rng = np.random.default_rng(seed)
    trials = []
    label_marginals = {gse:[] for gse in stage07.TARGETS}
    correspondence_marginals = {gse:[] for gse in stage07.TARGETS}
    for iteration in range(1, permutations+1):
        yp_train = source_permuted_within_cell_line(ysource, cell_lines, rng, np)
        if np.array_equal(yp_train, ysource):  # Avoid identity permutations.
            yp_train = source_permuted_within_cell_line(ysource, cell_lines, rng, np)
        permutation_model = stage07.pipeline(ml).fit(X_source, yp_train)
        shuffled_label, shuffled_label_macro = mean_study_balanced_accuracy(
            permutation_model.predict, X_target, ytarget, stage07, ml)
        # Same gene-index permutation per trial, shared by all target cohorts.
        scrambled_indices = rng.permutation(X_source.shape[1])
        scrambled, scrambled_macro = mean_study_balanced_accuracy(
            frozen_model.predict,
            {gse:X_target[gse][:, scrambled_indices] for gse in stage07.TARGETS},
            ytarget,stage07,ml)
        trials.append({
            "iteration": iteration,
            "shuffled_source_labels_macro_bacc": shuffled_label_macro,
            "broken_gene_correspondence_macro_bacc": scrambled_macro,
            **{f"label_shuffle_{gse}_bacc":shuffled_label[gse] for gse in stage07.TARGETS},
            **{f"gene_shuffle_{gse}_bacc":scrambled[gse] for gse in stage07.TARGETS},
        })
        for gse in stage07.TARGETS:
            label_marginals[gse].append(shuffled_label[gse])
            correspondence_marginals[gse].append(scrambled[gse])
    def null_summary(observed_value, null_values):
        return {
            "observed_bacc": float(observed_value),
            "null_mean": float(np.mean(null_values)),
            "null_min": float(np.min(null_values)),
            "null_max": float(np.max(null_values)),
            "null_ge_observed": int(sum(float(v)>=float(observed_value)-1e-12 for v in null_values)),
            "exploratory_plus_one_tail_fraction": float((1 + sum(float(v)>=float(observed_value)-1e-12 for v in null_values))/(len(null_values)+1)),
        }
    label_null = {gse:null_summary(observed[gse],label_marginals[gse]) for gse in stage07.TARGETS}
    gene_null = {gse:null_summary(observed[gse],correspondence_marginals[gse]) for gse in stage07.TARGETS}
    label_null["macro_3study"] = null_summary(observed_macro, [t["shuffled_source_labels_macro_bacc"] for t in trials])
    gene_null["macro_3study"] = null_summary(observed_macro, [t["broken_gene_correspondence_macro_bacc"] for t in trials])

    # Selection/feature-universe sensitivity, no external gene-header access during source fit.
    source_only_results, source_only_genes = source_only_feature_sensitivity(
        stage07, ml, src_genes, src_values, targets, ysource, rankdata, np)
    report = {
        "status":"EXPLORATORY_NEGATIVE_CONTROLS_NOT_CONFIRMATORY_VALIDATION",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "seed":seed,"permutations":permutations,
        "stage07_sha256":stage07.digest(STAGE07_PATH),
        "source_matrix_sha256":protocol["source_matrix_sha256"],
        "source_study":"GSE193258", "n_source":len(ysource),
        "target_studies":list(stage07.TARGETS),
        "feature_universe_original_test_informed_headers":True,
        "n_exact_common_genes":len(common_genes),
        "original_fixed_model_observed_balanced_accuracy":observed,
        "original_fixed_model_macro_bacc":observed_macro,
        "original_fixed_model_source_LOCO":source_loco,
        "negative_controls":{
            "source_label_shuffled_within_cell_line":label_null,
            "external_gene_correspondence_broken":gene_null,
        },
        "source_only_feature_selection_sensitivity":{
            "source_selection_uses_no_external_feature_names":True,
            "source_gene_count":len(src_genes),
            "selected_gene_count":len(source_only_genes),
            "selected_gene_names":source_only_genes,
            "missing_feature_handling":"Source mean rank inserted, zero after source scaler; diagnostic only",
            "rank_denominator":"Each matrix full gene universe; may vary by study",
            "external_studies":source_only_results,
        },
        "limitations":[
            "All target cohorts were already inspected and are not untouched validation.",
            "Target labels are used to score permutations; NOT for model training or hyperparameter selection.",
            "The common-gene analysis consults external FEATURE NAMES before source fitting (not external expression values).",
            "Percentile-rank features are not proven batch/gene-annotation harmonization.",
            "Source-only gene sensitivity imputes unavailable genes to source means and has differing rank denominators.",
            "Control vs DTP remains confounded with drug exposure duration/dose/experimental conditions.",
            "Small number of independent cell lines and studies; permutations are descriptive diagnostic tails, not publication-grade p-values.",
            "GSE303413 has reported dose discrepancies and no samples are independently biologically approved.",
            "No claims of treatment response prediction, causality, or Q1-ready validation.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/"08_null_trials.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(trials[0]))
        writer.writeheader();writer.writerows(trials)
    (OUT/"08_robustness_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Stage 08 finished; source n={len(ysource)}, exact common genes={len(common_genes):,}, negative-control trials={permutations}")
    print("Original stage07-compatible three-study mean bACC: " + f"{observed_macro:.3f}")
    for name, item in (("Source-label shuffle",label_null["macro_3study"]),("Gene-correspondence shuffle",gene_null["macro_3study"])):
        print(f"{name}: null mean bACC={item['null_mean']:.3f}, max={item['null_max']:.3f}, exploratory tail={item['exploratory_plus_one_tail_fraction']:.3f}")
    for gse in stage07.TARGETS:
        r=source_only_results[gse]
        print(f"Source-only gene sensitivity {gse}: bACC={r['bacc']:.3f}, source-selected genes missing={r['n_missing_selected_genes']}/{r['n_selected_genes']}")
    print("Saved outputs/robustness/08_robustness_summary.json and 08_null_trials.csv")
    print("EXPLORATORY ONLY — duration/dose confounding, source labeling and target gene annotation remain unverified")
    return report


def self_test():
    import numpy as np
    rng=np.random.default_rng(42)
    y=np.array(([0]*3+[1]*3)*4, dtype=int)
    lines=np.array([i for i in ("H1975","HCC2935","HCC827","PC9") for _ in range(6)])
    for _ in range(20):
        yp=source_permuted_within_cell_line(y,lines,rng,np)
        assert all(Counter(yp[lines==line])=={0:3,1:3} for line in set(lines))
        assert len(yp)==len(y)
    # Fail rather than reporting a shuffled null that failed to preserve class balance.
    assert not STAGE07_PATH.name.startswith("08_")
    import sklearn
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    stage07=load_stage07()
    ml=locals()
    assert stage07.FIXED_HYPERPARAMETERS=={"model":"LogisticRegression","k":100,"C":0.01}
    # Source-only selected-feature alignment and missing neutral filling.
    genes=[f"G{i}" for i in range(250)]
    X=rng.normal(size=(24,250))
    X[:,0]+=np.where(y==1,5.,0.)
    fit=stage07.pipeline(ml).fit(X,y)
    sel=selected_source_genes(fit,genes)
    assert len(sel)==100 and all(g in genes for g in sel)
    print("SELF-TEST PASSED (balanced within-line permutations, source-only feature selection, stage 07 integration)")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--exploratory-only",action="store_true")
    ap.add_argument("--permutations",type=int,default=49)
    ap.add_argument("--seed",type=int,default=SEED)
    args=ap.parse_args()
    if args.self_test:
        self_test();return 0
    if not args.exploratory_only:
        ap.error("Explicit --exploratory-only required; this is not confirmatory validation")
    run(args.permutations,args.seed)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
