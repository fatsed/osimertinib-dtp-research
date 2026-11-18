"""11 | Source-only gene-space perturbation benchmark (EXPLORATORY).

Research goal: compare *the same 100 training-selected genes* under two
representations when held-out test cell lines have incomplete gene coverage:
  - all_gene_percentile: ranks within all genes observed for each sample.
  - source_anchor_ecdf: proportion of source-selected stable anchor genes whose
    expression is lower than the feature gene (ties worth half). The anchors
    and selected features are fitted ONLY on the 3 training cell lines.

IMPORTANT:
  1) This simulates missing genes WITHIN one GEO dataset. It is NOT independent
     cross-study validation and it does not correct batch effects.
  2) With background-only removal, the anchor approach is invariant by design;
     that is a diagnostic positive control, NOT evidence of superior biology.
  3) The meaningful comparison also removes randomly selected feature AND
     anchor genes. Unobserved selected features receive the source-train mean
     representation value (equivalent to 0 after StandardScaler).
  4) This does not establish DTP-specific biology: Control vs 21d DTP labels
     are confounded with exposure duration and treatment conditions.

Requires: existing scripts/07_external_stress_test.py, source matrix and
        data/metadata/evaluation_protocol.json; numpy scipy scikit-learn.
Run from project root:
    python scripts/11_source_only_gene_space_benchmark.py --self-test
    python scripts/11_source_only_gene_space_benchmark.py --exploratory-only
Outputs: outputs/gene_space/11_summary.json and 11_metrics.csv (gitignored).
No downloads, no data or earlier output changes, no new dependencies.
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
SOURCE_PROTOCOL = ROOT / "data/metadata/evaluation_protocol.json"
STAGE07 = ROOT / "scripts/07_external_stress_test.py"
OUTPUT = ROOT / "outputs/gene_space"
LABEL_MAP = {"Control": 0, "DTP": 1}
SEED = 42
N_FEATURES = 100
N_ANCHORS = 256
C = 0.01
SITUATIONS = (("background_30pct", "background", 0.30),
              ("background_50pct", "background", 0.50),
              ("all_10pct", "all", 0.10),
              ("all_25pct", "all", 0.25))


def load_stage07():
    spec = importlib.util.spec_from_file_location("stage07_for_stage11", STAGE07)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot import existing numbered script {STAGE07}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def rank_all(X, np, rankdata):
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Nonempty 2D gene array required")
    return rankdata(X, method="average", axis=1) / X.shape[1]


def select_source_features_and_anchors(Xtr, ytr, rankdata, np, f_classif,
                                        k=N_FEATURES, anchors=N_ANCHORS):
    """Source-train only; deterministic tie breaks based on source gene index."""
    fullranks = rank_all(Xtr, np, rankdata)
    variances = fullranks.var(axis=0)
    usable = np.flatnonzero(variances > 1e-12)
    if len(usable) < k + anchors:
        raise ValueError("Not enough nonconstant genes for features and anchors")
    # Restrict supervised testing to genes with nonzero rank variance.
    scores, _ = f_classif(fullranks[:, usable], ytr)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    sorted_ids = np.lexsort((usable, -scores))
    chosen = usable[sorted_ids[:k]]
    if not np.isfinite(scores[sorted_ids[:k]]).all():
        raise ValueError("Feature-selection scores invalid")
    # Source TRAINING samples only. Stable genes are not chosen with labels.
    means = fullranks.mean(axis=0)
    expressed_fraction = (Xtr > 0).mean(axis=0)
    possible = np.flatnonzero((variances > 1e-12) & (means >= 0.15) &
                               (means <= 0.85) & (expressed_fraction >= 0.75))
    possible = np.setdiff1d(possible, chosen, assume_unique=False)
    if len(possible) < anchors:
        raise ValueError(f"Only {len(possible)} stable-anchor candidates; need {anchors}")
    order = np.lexsort((possible, variances[possible]))
    reference = possible[order[:anchors]]
    if np.intersect1d(chosen, reference).size:
        raise AssertionError("Selected features overlap anchor genes")
    return chosen, reference, fullranks


def ecdf_features(X, features, anchors, np):
    """Per sample feature percentile in reference gene set, with tie handling."""
    f = X[:, features]
    a = X[:, anchors]
    return ((a[:, None, :] < f[:, :, None]).sum(axis=2) +
            0.5 * (a[:, None, :] == f[:, :, None]).sum(axis=2)) / len(anchors)


def transform_missing(Xtest, present, feature_idx, anchor_idx, feature_mean,
                      method, np, rankdata):
    """No target labels or learned target statistics are used.

    Missing selected genes are filled with TRAIN means in representation space.
    The entire held-out fold shares one gene-panel mask.
    """
    n = Xtest.shape[0]
    Z = np.broadcast_to(feature_mean, (n, len(feature_idx))).copy()
    present_index = np.flatnonzero(present)
    feature_available = present[feature_idx]
    if not np.any(feature_available):
        return Z, 0
    target_cols = feature_idx[feature_available]
    if method == "all_gene_percentile":
        ranks = rank_all(Xtest[:, present_index], np, rankdata)
        inverse = np.full(Xtest.shape[1], -1, dtype=int)
        inverse[present_index] = np.arange(len(present_index))
        Z[:, feature_available] = ranks[:, inverse[target_cols]]
        return Z, 0
    if method == "source_anchor_ecdf":
        available_anchors = anchor_idx[present[anchor_idx]]
        if len(available_anchors) < max(2, min(64, len(anchor_idx)//2)):
            raise ValueError("Insufficient source-defined anchor coverage")
        f = Xtest[:, target_cols]
        a = Xtest[:, available_anchors]
        Z[:, feature_available] = ((a[:, None, :] < f[:, :, None]).sum(axis=2) +
                                    0.5 * (a[:, None, :] == f[:, :, None]).sum(axis=2)) / len(available_anchors)
        return Z, len(available_anchors)
    raise ValueError(method)


def metrics(y, pred, np):
    if len(y) != len(pred) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Both classes required in evaluation")
    a = float(np.mean(pred[y == 0] == 0))
    b = float(np.mean(pred[y == 1] == 1))
    return {"balanced_accuracy": (a + b) / 2, "control_recall": a,
            "dtp_recall": b, "predicted_dtp_fraction": float(np.mean(pred == 1))}


def masks_for_fold(gene_count, feature_idx, anchor_idx, fold_idx, trials,
                   seed, np):
    protected = np.zeros(gene_count, dtype=bool)
    protected[feature_idx] = True
    protected[anchor_idx] = True
    background = np.flatnonzero(~protected)
    for setting_i, (situation, scope, fraction) in enumerate(SITUATIONS):
        candidates = background if scope == "background" else np.arange(gene_count)
        removed_n = int(round(len(candidates) * fraction))
        for trial in range(trials):
            rng = np.random.default_rng(np.random.SeedSequence([seed, fold_idx, setting_i, trial]))
            dropped = rng.choice(candidates, size=removed_n, replace=False)
            keep = np.ones(gene_count, dtype=bool)
            keep[dropped] = False
            yield situation, trial, keep


def evaluate_fold(X, labels, groups, fold_line, np, rankdata, f_classif,
                  LogisticRegression, StandardScaler, seed=SEED, trials=12,
                  k=N_FEATURES, anchors=N_ANCHORS):
    train = groups != fold_line
    test = ~train
    if np.sum(test) != 6 or np.sum(train) != 18:
        raise ValueError("Expected 18 train and 6 held-out samples per cell line")
    if Counter(labels[train]) != {0: 9, 1: 9} or Counter(labels[test]) != {0: 3, 1: 3}:
        raise ValueError("Unexpected Control/DTP label balance")
    Xtr, Xte = X[train], X[test]
    ytr, yte = labels[train], labels[test]
    selected, refs, train_ranks = select_source_features_and_anchors(
        Xtr, ytr, rankdata, np, f_classif, k, anchors)
    train_features = {
        "all_gene_percentile": train_ranks[:, selected],
        "source_anchor_ecdf": ecdf_features(Xtr, selected, refs, np),
    }
    # Keep fitted models entirely separate for the two representations.
    models, train_means, clean = {}, {}, {}
    whole = np.ones(X.shape[1], dtype=bool)
    for method, Ftr in train_features.items():
        from sklearn.pipeline import make_pipeline
        fitted = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, solver="liblinear", max_iter=2000, random_state=seed))
        fitted.fit(Ftr, ytr)
        models[method] = fitted
        train_means[method] = np.mean(Ftr, axis=0)
        Fte, _ = transform_missing(Xte, whole, selected, refs,
                                   train_means[method], method, np, rankdata)
        # Verify the missing-aware transform reproduces the unmasked direct transform.
        direct = rank_all(Xte, np, rankdata)[:, selected] if method == "all_gene_percentile" else ecdf_features(Xte, selected, refs, np)
        if not np.allclose(Fte, direct, atol=1e-12):
            raise AssertionError(f"Clean transform mismatch for {method}")
        preds = fitted.predict(Fte)
        clean[method] = {"prediction": preds, **metrics(yte, preds, np)}
    # Identical test-gene masks across both representations.
    rows = []
    for situation, trial, keep in masks_for_fold(X.shape[1], selected, refs,
                       ["H1975", "HCC2935", "HCC827", "PC9"].index(fold_line), trials, seed, np):
        mask_sha = hashlib.sha256(keep.tobytes()).hexdigest()
        for method in models:
            Fte, n_anchor_available = transform_missing(
                Xte, keep, selected, refs, train_means[method], method, np, rankdata)
            pred = models[method].predict(Fte)
            row = {"held_out_line":fold_line,"situation":situation,"trial":trial,
                   "representation":method,"test_n":len(yte),
                   "n_genes_observed":int(np.sum(keep)),
                   "n_selected_features_missing":int(np.sum(~keep[selected])),
                   "n_anchors_missing":int(np.sum(~keep[refs])),
                   "mask_sha256":mask_sha,
                   "prediction_switch_fraction":float(np.mean(pred != clean[method]["prediction"])),
                   **metrics(yte,pred,np)}
            rows.append(row)
    clean_output = {name:{k:float(v) for k,v in c.items() if k!="prediction"}
                    for name,c in clean.items()}
    return {"line":fold_line, "selected_count":len(selected), "anchor_count":len(refs),
            "selected_sha256":hashlib.sha256(selected.tobytes()).hexdigest(),
            "anchor_sha256":hashlib.sha256(refs.tobytes()).hexdigest(),
            "clean":clean_output, "rows":rows}


def report_trial_averages(folds, np):
    rows = [r for f in folds for r in f["rows"]]
    summaries = []
    for situation, _, _ in SITUATIONS:
        for method in ("all_gene_percentile", "source_anchor_ecdf"):
            matching = [r for r in rows if r["situation"] == situation and r["representation"] == method]
            summaries.append({"situation":situation,"representation":method,
                              "mean_balanced_accuracy":float(np.mean([r["balanced_accuracy"] for r in matching])),
                              "mean_dtp_recall":float(np.mean([r["dtp_recall"] for r in matching])),
                              "mean_prediction_switch_fraction":float(np.mean([r["prediction_switch_fraction"] for r in matching])),
                              "worst_line_mean_balanced_accuracy":float(min(
                                  np.mean([r["balanced_accuracy"] for r in matching if r["held_out_line"]==f["line"]])
                                  for f in folds)),
                              "n_fold_mask_trials":len(matching)})
    return rows, summaries


def run(trials, seed):
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import sklearn
    if not SOURCE_PROTOCOL.exists():
        raise FileNotFoundError(f"Run earlier steps to generate {SOURCE_PROTOCOL}")
    protocol = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    if protocol["source_study"] != "GSE193258":
        raise ValueError("Unexpected source study")
    if protocol["flags"]["biology_approved_for_training"] is not False:
        raise ValueError("Stage11 is designed for provisional labels only")
    stage07 = load_stage07()
    matrix_path = stage07.safe_local_path(protocol["source_matrix"], "processed")
    genes, gsm, X = stage07.load_matrix(matrix_path, protocol["source_matrix_sha256"], np)
    metadata = protocol["source_gsm_to_label"]
    if len(gsm) != len(metadata) or gsm != [x["gsm"] for x in metadata]:
        raise ValueError("Source sample order differs from frozen protocol")
    if len(genes) != protocol["source_gene_count"] or len(set(genes)) != len(genes):
        raise ValueError("Source gene count or ID uniqueness differs from protocol")
    labels = np.array([LABEL_MAP[x["state"]] for x in metadata])
    groups = np.array([x["cell_line"] for x in metadata])
    if len(np.unique(groups)) != 4:
        raise ValueError("Four cell lines required")
    folds = [evaluate_fold(X,labels,groups,line,np,rankdata,f_classif,
                           LogisticRegression,StandardScaler,seed=seed,trials=trials)
             for line in ("H1975", "HCC2935", "HCC827", "PC9")]
    rows, summaries = report_trial_averages(folds, np)
    clean_means = {method:float(np.mean([f["clean"][method]["balanced_accuracy"] for f in folds]))
                   for method in ("all_gene_percentile", "source_anchor_ecdf")}
    conditions = {(a["situation"],a["representation"]):a for a in summaries}
    hard = "all_25pct"
    gain = (conditions[(hard,"source_anchor_ecdf")]["mean_balanced_accuracy"] -
            conditions[(hard,"all_gene_percentile")]["mean_balanced_accuracy"])
    switch_gain = (conditions[(hard,"all_gene_percentile")]["mean_prediction_switch_fraction"] -
                   conditions[(hard,"source_anchor_ecdf")]["mean_prediction_switch_fraction"])
    # Gate is exploratory heuristic, not a formal significance test.
    gate = {"clean_bacc_drop_max_0_05": clean_means["source_anchor_ecdf"] >= clean_means["all_gene_percentile"]-0.05,
            "all_25pct_bacc_gain_min_0_05":gain >= 0.05,
            "all_25pct_switch_reduction_min_0_10": switch_gain >= 0.10,
            "all_25pct_dtp_recall_min_0_5":conditions[(hard,"source_anchor_ecdf")]["mean_dtp_recall"]>=0.5}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    report = {"analysis":"source_only_gene_space_masking_exploratory_v1",
              "timestamp_utc":datetime.now(timezone.utc).isoformat(),
              "n_samples":len(labels),"source_genes":len(genes),"folds":4,
              "seed":seed,"trials_per_scenario_per_fold":trials,
              "training_only_feature_selection":True,
              "source_defined_anchor_panel":True,
              "same_gene_mask_across_methods":True,
              "source_matrix_sha256":sha256_file(matrix_path),
              "source_protocol_fingerprint":protocol.get("fingerprint_sha256"),
              "versions":{"numpy":np.__version__,"sklearn":sklearn.__version__},
              "fixed_parameters":{"n_features":N_FEATURES,"n_anchors":N_ANCHORS,"logistic_C":C,
                                   "source_only_anova_features":True},
              "clean_mean_bacc":clean_means,
              "stress_summaries":summaries,
              "per_line_clean_and_hashes":[{k:v for k,v in f.items() if k!="rows"} for f in folds],
              "provisional_success_gate":gate,
              "gate_all_passed":all(gate.values()),
              "limits":["Same-source simulated gene deletion, not actual cross-study shift.",
                        "Background-only invariance of anchors is guaranteed by design; do not interpret as model novelty.",
                        "Feature genes missing at test are filled with source training feature means.",
                        "No target studies or labels accessed for feature choice or tuning.",
                        "Small n, four historically studied lines; no independent validation.",
                        "Control vs 21-day DTP confounded with exposure conditions; biological validation pending.",
                        "This anchor-ECDF prototype is not new by itself; comparison with established methods is required."]}
    (OUTPUT/"11_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (OUTPUT/"11_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(f"Stage 11 source-only: n={len(labels)}, genes={len(genes)}, 4 held-out lines, trials={trials}")
    print("Clean LOCO bACC: all-gene rank=%.3f, source-anchor ECDF=%.3f" %
          (clean_means["all_gene_percentile"],clean_means["source_anchor_ecdf"]))
    for name,_,_ in SITUATIONS:
        b=conditions[(name,"all_gene_percentile")]
        a=conditions[(name,"source_anchor_ecdf")]
        print("%s: full-rank bACC=%.3f switch=%.3f; anchor bACC=%.3f switch=%.3f" %
              (name,b["mean_balanced_accuracy"],b["mean_prediction_switch_fraction"],
               a["mean_balanced_accuracy"],a["mean_prediction_switch_fraction"]))
    print("Exploratory pass gate:",gate,"ALL_PASS=",all(gate.values()))
    print("Saved outputs/gene_space/11_summary.json and 11_metrics.csv")
    print("EXPLORATORY ONLY: simulated missing genes, not cross-study/biological validation")


def self_test():
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    # Test tie-aware, monotonic invariance, missing-mean handling.
    X=np.array([[1.,1.,2.,3.,4.,5.],[2.,3.,3.,4.,5.,6.]])
    assert np.allclose(rank_all(X,np,rankdata),rank_all(3*X+7,np,rankdata))
    select=np.array([2,3]);anchor=np.array([0,1,4,5])
    Z=ecdf_features(X,select,anchor,np)
    assert Z.shape==(2,2) and (Z>=0).all() and (Z<=1).all()
    present=np.array([True,True,True,True,False,True])
    anchor_missing,_=transform_missing(X,present,select,anchor,np.zeros(2),"source_anchor_ecdf",np,rankdata)
    assert not np.allclose(anchor_missing,Z)
    present[2]=False
    M,_=transform_missing(X,present,select,anchor,np.array([0.42,0.71]),"all_gene_percentile",np,rankdata)
    assert np.allclose(M[:,0],0.42)
    # Fit and predict with synthetic source-only data; no GEO inputs needed.
    rng=np.random.default_rng(4242)
    Xsyn=rng.normal(size=(24,1000)) + 3.0
    groups=np.repeat(["H1975","HCC2935","HCC827","PC9"],6)
    labels=np.tile([0]*3+[1]*3,4)
    Xsyn[:,0:20] += labels[:,None]*1.2
    Xsyn[:,20:30] += np.repeat(np.arange(4),6)[:,None]*0.2
    # Source-only feature selection and anchor selection run in each fold.
    folds=[evaluate_fold(Xsyn,labels,groups,line,np,rankdata,f_classif,
                         LogisticRegression,StandardScaler,trials=2,k=20,anchors=64)
           for line in ("H1975","HCC2935","HCC827","PC9")]
    rows,summaries=report_trial_averages(folds,np)
    assert len(rows)==4*4*2*2 and len(summaries)==8
    for f in folds:
        for r in f["rows"]:
            assert 0 <= r["balanced_accuracy"]<=1 and r["test_n"]==6
        # exactly same mask for both representations on each trial
        keyed={}
        for r in f["rows"]:
            key=(r["situation"],r["trial"])
            if key in keyed: assert r["mask_sha256"]==keyed[key]
            keyed[key]=r["mask_sha256"]
    print("SELF-TEST PASSED (training-only selection, anchor ranks, shared masks, 4-fold synthetic run)")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--exploratory-only",action="store_true")
    parser.add_argument("--trials",type=int,default=12)
    parser.add_argument("--seed",type=int,default=SEED)
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if not args.exploratory_only:
        parser.error("Specify --exploratory-only; biology and external validation are NOT approved")
    if not 2<=args.trials<=100:
        parser.error("--trials must be 2..100")
    if args.seed<0:
        parser.error("--seed must be nonnegative")
    run(args.trials,args.seed)

if __name__=="__main__":
    main()
