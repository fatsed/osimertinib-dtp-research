"""13 | Source-defined anchor representation: real cross-study DEVELOPMENT benchmark.

Compare identical source-selected 100 genes with three representations:
 A: whole-reported-gene-universe sample percentile ranks
 B: ECDF against 256 low-variance source-selected anchor genes
 C: percentile ranks within the same 100 source-selected features ONLY

A vs B vs C addresses whether anchor genes are necessary at all.
The 32 random-source-panel control challenges the low-variance selection rule.

Both source models are fitted once on GSE193258 (24 provisional Control/DTP
samples). External GEO labels are consulted ONLY after predictions. No target
expression, headers, labels or outcome metrics influence source feature/anchor
selection, model fitting or hyperparameters. Missing selected genes receive
training-mean representation values; unavailable anchors are excluded from ECDF.

Target studies were already inspected during development; scores are POST-HOC,
NOT independent validation. DTP labels are confounded with exposure conditions.

Run:
    python scripts/13_cross_study_anchor_benchmark.py --self-test
    python scripts/13_cross_study_anchor_benchmark.py --exploratory-only
Outputs (gitignored): outputs/anchor_transfer/13_summary.json,
    13_metrics.csv, 13_predictions.csv, 13_anchor_control_trials.csv,
    13_anchor_control_by_study.csv.
Dependencies: existing numbered scripts 07 and 11; installed requirements.txt.
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
SCRIPT07 = ROOT / "scripts/07_external_stress_test.py"
SCRIPT11 = ROOT / "scripts/11_source_only_gene_space_benchmark.py"
PROTOCOL = ROOT / "data/metadata/evaluation_protocol.json"
METADATA = ROOT / "data/metadata"
OUT = ROOT / "outputs/anchor_transfer"
STUDIES = ("GSE153183", "GSE303413", "GSE341996")
METHODS = ("all_gene_percentile", "source_anchor_ecdf", "selected_feature_percentile")
K, ANCHORS, C, SEED = 100, 256, 0.01, 42


def module_from(path: Path, unique_name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Required prior script missing: {path}")
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_source(X, y, selected, anchors, ranks, stage11, np, LogisticRegression,
               StandardScaler):
    from sklearn.pipeline import make_pipeline
    from scipy.stats import rankdata
    feature_matrices = {
        "all_gene_percentile": ranks[:, selected],
        "source_anchor_ecdf": stage11.ecdf_features(X, selected, anchors, np),
        "selected_feature_percentile": (
            rankdata(X[:, selected], method="average", axis=1) / len(selected)
        ),
    }
    fitted = {}
    for method, features in feature_matrices.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, max_iter=2000, solver="liblinear", random_state=SEED))
        model.fit(features, y)
        fitted[method] = {
            "model": model,
            "train_mean": features.mean(axis=0),
        }
    return fitted


def target_transform(values, target_genes, selected_names, anchor_names,
                     source_means, method, rankdata, np):
    """Projection into source-defined features WITHOUT source-gene padding.

    Full-rank baseline ranks against ACTUAL genes reported by target study.
    Anchor transform ignores other genes, comparing only source-frozen anchors.
    Missing feature values are source-training means (0 after source scaling).
    """
    if values.ndim != 2 or values.shape[1] != len(target_genes):
        raise ValueError("Target matrix columns must equal gene identifier count")
    if len(set(target_genes)) != len(target_genes):
        raise ValueError("Duplicated target gene names are unsupported")
    if len(set(selected_names)) != len(selected_names) or len(set(anchor_names)) != len(anchor_names):
        raise ValueError("Source gene selection contains duplicates")
    if len(source_means) != len(selected_names):
        raise ValueError("Training mean length mismatch")
    index = {g: i for i, g in enumerate(target_genes)}
    feature_present = np.array([name in index for name in selected_names], dtype=bool)
    anchor_present = np.array([name in index for name in anchor_names], dtype=bool)
    feat_cols = np.array([index[name] for name, ok in zip(selected_names, feature_present) if ok], dtype=int)
    anc_cols = np.array([index[name] for name, ok in zip(anchor_names, anchor_present) if ok], dtype=int)
    X = np.broadcast_to(source_means, (values.shape[0], len(selected_names))).copy()
    if len(feat_cols):
        if method == "all_gene_percentile":
            # Rank only observed target gene measurements: no imputed/phantom genes
            # influence the per-sample denominator.
            all_ranks = rankdata(values, method="average", axis=1) / values.shape[1]
            X[:, feature_present] = all_ranks[:, feat_cols]
        elif method == "source_anchor_ecdf":
            if len(anc_cols) < min(64, max(8, len(anchor_names) // 2)):
                raise ValueError(f"Only {len(anc_cols)} of {len(anchor_names)} anchor genes available")
            feats = values[:, feat_cols]
            refs = values[:, anc_cols]
            X[:, feature_present] = ((refs[:, None, :] < feats[:, :, None]).sum(axis=2)
                    + 0.5 * (refs[:, None, :] == feats[:, :, None]).sum(axis=2)) / len(anc_cols)
        elif method == "selected_feature_percentile":
            # Compact, source-defined gene universe: no reference panel at all.
            # Rank ONLY observed selected features, never treat absent genes as zeros.
            X[:, feature_present] = rankdata(
                values[:, feat_cols], method="average", axis=1
            ) / len(feat_cols)
        else:
            raise ValueError(method)
    return X, {"selected_present": int(feature_present.sum()),
               "selected_missing": int((~feature_present).sum()),
               "anchors_present": int(anchor_present.sum()),
               "anchors_missing": int((~anchor_present).sum()),
               "missing_selected_names": [g for g, ok in zip(selected_names, feature_present) if not ok],
               "anchor_coverage_fraction": float(anchor_present.mean())}


def confusion_metrics(y, p, np):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) != len(p) or len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected paired predictions with both classes")
    control = float((p[y == 0] == 0).mean())
    dtp = float((p[y == 1] == 1).mean())
    return {"n": int(len(y)), "balanced_accuracy": (control + dtp) / 2,
            "control_recall": control, "dtp_recall": dtp,
            "correct": int((y == p).sum()), "predicted_dtp": int((p == 1).sum())}


def source_loco(X, y, groups, stage11, np, rankdata, f_classif,
                LogisticRegression, StandardScaler):
    records = []
    for line in ("H1975", "HCC2935", "HCC827", "PC9"):
        train, test = groups != line, groups == line
        if Counter(y[train]) != {0: 9, 1: 9} or Counter(y[test]) != {0: 3, 1: 3}:
            raise ValueError("Source grouping/labels differ from locked protocol")
        selected, anchors, ranks = stage11.select_source_features_and_anchors(
            X[train], y[train], rankdata, np, f_classif, k=K, anchors=ANCHORS)
        fitted = fit_source(X[train], y[train], selected, anchors, ranks, stage11,
                            np, LogisticRegression, StandardScaler)
        clean = {
            "all_gene_percentile": stage11.rank_all(X[test], np, rankdata)[:, selected],
            "source_anchor_ecdf": stage11.ecdf_features(X[test], selected, anchors, np),
            "selected_feature_percentile": (
                rankdata(X[test][:, selected], method="average", axis=1) / len(selected)
            ),
        }
        for method in METHODS:
            pred = fitted[method]["model"].predict(clean[method])
            records.append({"held_out_cell_line": line, "method": method,
                            **confusion_metrics(y[test], pred, np)})
    return records


def compare(X, y, groups, genes, targets, stage11, np, rankdata, f_classif,
            LogisticRegression, StandardScaler):
    if len(genes) != len(set(genes)):
        raise ValueError("Source duplicate gene names")
    chosen, refs, ranks = stage11.select_source_features_and_anchors(
        X, y, rankdata, np, f_classif, k=K, anchors=ANCHORS)
    models = fit_source(X, y, chosen, refs, ranks, stage11, np,
                        LogisticRegression, StandardScaler)
    selected_names = [genes[int(i)] for i in chosen]
    anchor_names = [genes[int(i)] for i in refs]
    fold_records = source_loco(X, y, groups, stage11, np, rankdata, f_classif,
                               LogisticRegression, StandardScaler)
    score_records, predictions, coverage = [], [], {}
    for study in STUDIES:
        t = targets[study]
        yy = np.array([int(x == "DTP") for x in t["states"]], dtype=int)
        if len(yy) != len(t["gsm"]):
            raise ValueError(f"Target GSM/label size mismatch {study}")
        p_by_method = {}
        for method in METHODS:
            Xt, info = target_transform(t["matrix"], t["genes"], selected_names,
                anchor_names, models[method]["train_mean"], method, rankdata, np)
            preds = models[method]["model"].predict(Xt)
            p_by_method[method] = preds
            score_records.append({"study": study, "method": method,
                                 **confusion_metrics(yy, preds, np),
                                 **{k:v for k,v in info.items() if k != "missing_selected_names"}})
            coverage[study] = {**info, "target_reported_genes":len(t["genes"])}
            for gsm, label, pred in zip(t["gsm"], yy, preds):
                predictions.append({"study":study,"gsm":gsm,"method":method,
                    "candidate_state":"DTP" if label else "Control",
                    "prediction":"DTP" if pred else "Control",
                    "correct":int(label == pred)})
        coverage[study]["baseline_vs_anchor_switches"] = int(np.sum(
            p_by_method[METHODS[0]] != p_by_method[METHODS[1]]))
    score_lookup = {(r["study"],r["method"]):r for r in score_records}
    comparison = {}
    for method in METHODS:
        ss = [score_lookup[(s,method)] for s in STUDIES]
        comparison[method] = {
            "study_macro_bacc": float(np.mean([r["balanced_accuracy"] for r in ss])),
            "study_macro_dtp_recall": float(np.mean([r["dtp_recall"] for r in ss])),
            "worst_study_bacc": float(min(r["balanced_accuracy"] for r in ss)),
            "pooled_correct":int(sum(r["correct"] for r in ss)),
            "pooled_n":int(sum(r["n"] for r in ss)),
            "source_loco_mean_bacc":float(np.mean([r["balanced_accuracy"] for r in fold_records if r["method"]==method])),
        }
    return {"metrics":score_records,"predictions":predictions,
            "coverage":coverage,"source_folds":fold_records,"comparison":comparison,
            "source_selected_gene_names":selected_names,"source_anchor_gene_names":anchor_names}



def random_anchor_controls(X, y, genes, targets, stage11, np, rankdata,
                           f_classif, LogisticRegression, StandardScaler,
                           n_panels=32, seed=20261118):
    """Pre-specified source-only random-reference control (NOT a significance test).

    Use exactly the same 100 source-selected genes and eligible anchor candidate
    pool as the original stable-panel method. Random panels are chosen without
    reading target studies. Target labels only score finished predictions.
    This is an exploratory sensitivity check on historically inspected studies.
    """
    if n_panels < 1 or n_panels > 200:
        raise ValueError("Number of control panels must be 1..200")
    selected, stable, ranks = stage11.select_source_features_and_anchors(
        X, y, rankdata, np, f_classif, k=K, anchors=ANCHORS)
    variance = ranks.var(axis=0)
    means = ranks.mean(axis=0)
    expressed = (X > 0).mean(axis=0)
    pool = np.flatnonzero((variance > 1e-12) & (means >= 0.15) &
                          (means <= 0.85) & (expressed >= 0.75))
    pool = np.setdiff1d(pool, selected, assume_unique=False)
    if len(pool) < ANCHORS or not set(stable).issubset(set(pool)):
        raise AssertionError("Source training eligible-anchor pool is inconsistent")
    selected_names = [genes[int(i)] for i in selected]
    rng = np.random.default_rng(seed)
    trial_rows = []
    study_rows = []
    from sklearn.pipeline import make_pipeline
    for trial in range(n_panels):
        panel = np.sort(rng.choice(pool, size=ANCHORS, replace=False))
        ref_names = [genes[int(i)] for i in panel]
        Ztrain = stage11.ecdf_features(X, selected, panel, np)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, max_iter=2000, solver="liblinear", random_state=SEED))
        model.fit(Ztrain, y)
        mean = Ztrain.mean(axis=0)
        trial_scores = []
        for study in STUDIES:
            t = targets[study]
            projected, coverage = target_transform(
                t["matrix"], t["genes"], selected_names, ref_names,
                mean, "source_anchor_ecdf", rankdata, np)
            predicted = model.predict(projected)
            yy = np.array([int(z == "DTP") for z in t["states"]], dtype=int)
            met = confusion_metrics(yy, predicted, np)
            row = {"trial": trial, "study": study,
                   "balanced_accuracy": met["balanced_accuracy"],
                   "control_recall": met["control_recall"],
                   "dtp_recall": met["dtp_recall"],
                   "correct": met["correct"], "n": met["n"],
                   "anchors_present": coverage["anchors_present"],
                   "selected_features_present": coverage["selected_present"]}
            study_rows.append(row)
            trial_scores.append(row)
        trial_rows.append({"trial": trial,
                           "study_macro_bacc": float(np.mean([r["balanced_accuracy"] for r in trial_scores])),
                           "study_macro_dtp_recall": float(np.mean([r["dtp_recall"] for r in trial_scores])),
                           "worst_study_bacc": float(min(r["balanced_accuracy"] for r in trial_scores)),
                           "pooled_correct": int(sum(r["correct"] for r in trial_scores)),
                           "pooled_n": int(sum(r["n"] for r in trial_scores)),
                           "min_anchor_coverage": min(r["anchors_present"] for r in trial_scores),
                           "mean_anchor_coverage": float(np.mean([r["anchors_present"] for r in trial_scores]))})
    # Do not select the best random panel. Report the FULL control distribution.
    return {"n_panels": n_panels, "seed": seed,
            "eligible_source_anchor_genes": len(pool),
            "random_reference_trials": trial_rows,
            "random_reference_study_metrics": study_rows}


def summarize_controls(control, stable_macro):
    scores = [x["study_macro_bacc"] for x in control["random_reference_trials"]]
    if not scores:
        raise ValueError("No random-reference control trials")
    import statistics
    return {"stable_source_anchor_macro_bacc": float(stable_macro),
            "random_control_min": float(min(scores)),
            "random_control_median": float(statistics.median(scores)),
            "random_control_max": float(max(scores)),
            "random_control_fraction_at_least_stable": float(sum(v >= stable_macro - 1e-12 for v in scores) / len(scores)),
            "interpretation": "Exploratory, repeatedly inspected target studies; NOT a p-value or independent model selection."}


def save_csv(path, records):
    if not records: raise ValueError("No rows")
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def run(anchor_controls=32):
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    stage07 = module_from(SCRIPT07, "source_only_stage07_for_stage13")
    stage11 = module_from(SCRIPT11, "source_only_stage11_for_stage13")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("source_study") != "GSE193258" or protocol.get("flags",{}).get("biology_approved_for_training") is not False:
        raise ValueError("Only the existing provisional GSE193258 source protocol is permitted")
    source_file = stage07.safe_local_path(protocol["source_matrix"], "processed")
    genes, gsms, X = stage07.load_matrix(source_file, protocol["source_matrix_sha256"], np)
    source_info = protocol["source_gsm_to_label"]
    if gsms != [z["gsm"] for z in source_info] or len(X) != 24 or len(genes) != 19712:
        raise ValueError("Unexpected source matrix/protocol identity")
    y = np.array([int(z["state"] == "DTP") for z in source_info],dtype=int)
    groups = np.array([z["cell_line"] for z in source_info])
    cleaning = json.loads((METADATA / "cleaning_report.json").read_text(encoding="utf-8"))
    targets = stage07.prepare_targets(np, cleaning,
        stage07.csv_rows(METADATA / "expression_alignment.csv"),
        stage07.csv_rows(METADATA / "cohort_review.csv"), ROOT)
    if set(targets) != set(STUDIES): raise ValueError("Target study set changed")
    outcome = compare(X,y,groups,genes,targets,stage11,np,rankdata,f_classif,
                      LogisticRegression,StandardScaler)
    controls = random_anchor_controls(X, y, genes, targets, stage11, np, rankdata,
        f_classif, LogisticRegression, StandardScaler,
        n_panels=anchor_controls)
    control_summary = summarize_controls(controls,
        outcome["comparison"]["source_anchor_ecdf"]["study_macro_bacc"])
    report = {
        "status":"EXPLORATORY_POST_HOC_PREVIOUSLY_SEEN_STUDIES_NOT_INDEPENDENT_VALIDATION",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "source_matrix_sha256":protocol["source_matrix_sha256"],
        "protocol_fingerprint":protocol.get("fingerprint_sha256"),
        "code_sha256":{"07":sha256(SCRIPT07),"11":sha256(SCRIPT11)},
        "fixed_parameters":{"features":K,"anchors":ANCHORS,"C":C,"seed":SEED,
                            "classifier":"LogisticRegression","solver":"liblinear"},
        "study_names":list(STUDIES),
        "selected_gene_names":outcome["source_selected_gene_names"],
        "anchor_gene_names":outcome["source_anchor_gene_names"],
        "source_folds":outcome["source_folds"],
        "target_coverage":outcome["coverage"],
        "macro_comparison":outcome["comparison"],
        "per_study":outcome["metrics"],
        "random_anchor_control":{**control_summary,
            "n_panels":controls["n_panels"],
            "seed":controls["seed"],
            "eligible_source_anchor_genes":controls["eligible_source_anchor_genes"]},
        "target_labels_used_only_for_metrics":True,
        "source_only_feature_anchor_selection":True,
        "target_gene_names_used_only_for_covered_feature_projection":True,
        "target_expression_used_for_training_or_tuning":False,
        "limitations":[
            "Targets and source folds were already inspected in stages 01-12; no independence claim.",
            "Target GSM labels and biological replicates await manual biological verification.",
            "Source 21d DTP vs Control conflates drug exposure duration/dose and cell state.",
            "Expression measurement units and gene-symbol annotations differ across studies.",
            "GSE303413 has six unresolved dose annotation discrepancies.",
            "All three candidate representations use the same 100 selected genes, but separately fitted models.",
            "Feature-only percentile ranks are a critical no-anchor baseline and are not claimed as novel.",
            "Any gain is post-hoc developmental evidence, not superiority of a new method.",
        ],
    }
    OUT.mkdir(parents=True,exist_ok=True)
    save_csv(OUT/"13_metrics.csv",outcome["metrics"])
    save_csv(OUT/"13_predictions.csv",outcome["predictions"])
    save_csv(OUT/"13_anchor_control_trials.csv",controls["random_reference_trials"])
    save_csv(OUT/"13_anchor_control_by_study.csv",controls["random_reference_study_metrics"])
    (OUT/"13_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Stage 13: source-only gene/anchor selection; real target measurements; NO target tuning")
    for study in STUDIES:
        c = outcome["coverage"][study]
        x = next(z for z in outcome["metrics"] if z["study"]==study and z["method"]==METHODS[0])
        z = next(z for z in outcome["metrics"] if z["study"]==study and z["method"]==METHODS[1])
        f = next(z for z in outcome["metrics"] if z["study"]==study and z["method"]==METHODS[2])
        print(f"{study}: n={x['n']}, baseline bACC={x['balanced_accuracy']:.3f} DTP={x['dtp_recall']:.3f}; "
              f"anchor bACC={z['balanced_accuracy']:.3f} DTP={z['dtp_recall']:.3f}; "
              f"NO-ANCHOR bACC={f['balanced_accuracy']:.3f} DTP={f['dtp_recall']:.3f}; "
              f"features={c['selected_present']}/{K}, anchors={c['anchors_present']}/{ANCHORS}, "
              f"between-method switches={c['baseline_vs_anchor_switches']}")
    for method in METHODS:
        x = outcome["comparison"][method]
        print(f"{method}: source LOCO={x['source_loco_mean_bacc']:.3f}; "
              f"target study-macro bACC={x['study_macro_bacc']:.3f}; "
              f"DTP recall={x['study_macro_dtp_recall']:.3f}; worst-study bACC={x['worst_study_bacc']:.3f}")
    print("New control: selected_feature_percentile uses NO anchor genes; source-only feature selection")
    print(f"Random-anchor control ({controls['n_panels']} source-only panels): "
          f"bACC min={control_summary['random_control_min']:.3f} "
          f"median={control_summary['random_control_median']:.3f} "
          f"max={control_summary['random_control_max']:.3f}; "
          f"fraction >= stable panel={control_summary['random_control_fraction_at_least_stable']:.3f}")
    print("Saved outputs/anchor_transfer/13_summary.json, 13_metrics.csv, "
          "13_predictions.csv, 13_anchor_control_trials.csv, 13_anchor_control_by_study.csv")
    print("EXPLORATORY ONLY — no biological approval or independent external validation")


def self_test():
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import f_classif
    stage11 = module_from(SCRIPT11, "stage11_test_13")
    rng=np.random.default_rng(1024)
    # Source-derived features and anchors; explicitly ensure target headers
    # aren't read before fitting by not making any target objects here.
    n=500
    y=np.array([0]*9+[1]*9)
    X=rng.uniform(0.1,10,size=(18,n))
    X[y==1,0:12]+=15
    feat, anc, ranks=stage11.select_source_features_and_anchors(X,y,rankdata,np,f_classif,k=12,anchors=32)
    assert len(feat)==12 and len(anc)==32 and not set(feat)&set(anc)
    # Stage-13 actual source fitting function is independent of assumed feature count.
    models=fit_source(X,y,feat,anc,ranks,stage11,np,LogisticRegression,StandardScaler)
    assert set(models)==set(METHODS)
    ids=[f"G{i}" for i in range(n)]
    selected=[ids[int(i)] for i in feat]
    anchors=[ids[int(i)] for i in anc]
    for method in METHODS:
        data,info=target_transform(X,ids,selected,anchors,models[method]["train_mean"],
                                    method,rankdata,np)
        direct=(ranks[:,feat] if method=="all_gene_percentile" else
                stage11.ecdf_features(X,feat,anc,np) if method=="source_anchor_ecdf" else
                rankdata(X[:,feat],method="average",axis=1)/len(feat))
        assert np.allclose(data,direct),method
        assert info["selected_missing"]==0 and info["anchors_missing"]==0
    # For baseline, adding irrelevant genes alters ranks; for anchor representation
    # it cannot change output if selected + anchor values stay unchanged.
    v=np.concatenate([X, rng.uniform(0.01,20,size=(18,50))],axis=1)
    ids2=ids+[f"OTHER_{i}" for i in range(50)]
    bas1,_=target_transform(X,ids,selected,anchors,models[METHODS[0]]["train_mean"],
                             METHODS[0],rankdata,np)
    bas2,_=target_transform(v,ids2,selected,anchors,models[METHODS[0]]["train_mean"],
                             METHODS[0],rankdata,np)
    anc1,_=target_transform(X,ids,selected,anchors,models[METHODS[1]]["train_mean"],
                             METHODS[1],rankdata,np)
    anc2,_=target_transform(v,ids2,selected,anchors,models[METHODS[1]]["train_mean"],
                             METHODS[1],rankdata,np)
    assert np.max(np.abs(bas1-bas2))>0.001
    assert np.allclose(anc1,anc2)
    feat1,_=target_transform(X,ids,selected,anchors,
               models["selected_feature_percentile"]["train_mean"],
               "selected_feature_percentile",rankdata,np)
    feat2,_=target_transform(v,ids2,selected,anchors,
               models["selected_feature_percentile"]["train_mean"],
               "selected_feature_percentile",rankdata,np)
    assert np.allclose(feat1,feat2), "Feature-only ranks must ignore background genes"
    # Missing selected genes get the locked TRAIN mean, never target mean.
    gone=set(selected[:3]+anchors[:2])
    keep=[i for i,g in enumerate(ids) if g not in gone]
    reduced=[ids[i] for i in keep]
    for method in METHODS:
        Z,info=target_transform(X[:,keep],reduced,selected,anchors,
                    models[method]["train_mean"],method,rankdata,np)
        assert info["selected_missing"]==3 and info["anchors_missing"]==2
        assert np.allclose(Z[:,:3],np.broadcast_to(models[method]["train_mean"][:3],(len(X),3)))
    # Test random-panel control with the unchanged fitted features on synthetic data.
    global K, ANCHORS
    oldk, olda = K, ANCHORS
    try:
        K, ANCHORS = 12, 32
        gs = np.array(["A"]*9+["B"]*9)
        target_ids = [ids[i] for i in range(n)]
        targets = {study:{"genes":target_ids, "matrix":X[[0,1,9,10]],
                         "states":["Control","Control","DTP","DTP"]}
                   for study in STUDIES}
        controls1 = random_anchor_controls(X,y,ids,targets,stage11,np,rankdata,
                     f_classif,LogisticRegression,StandardScaler,n_panels=3,seed=29)
        controls2 = random_anchor_controls(X,y,ids,targets,stage11,np,rankdata,
                     f_classif,LogisticRegression,StandardScaler,n_panels=3,seed=29)
        assert controls1 == controls2
        assert len(controls1["random_reference_trials"]) == 3
        assert len(controls1["random_reference_study_metrics"]) == 9
    finally:
        K, ANCHORS = oldk, olda
    print("SELF-TEST PASSED (training-only selection; exact-name target projection; "
          "missing-gene imputation; anchor/feature-only invariance; unchanged source transform; "
          "reproducible random-anchor controls)")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--exploratory-only",action="store_true")
    parser.add_argument("--anchor-controls",type=int,default=32,
                        help="Source-only random-reference control panels (1..200)")
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.exploratory_only:
        parser.error("Real datasets require --exploratory-only (these targets are not independent)")
    run(anchor_controls=args.anchor_controls)


if __name__=="__main__":
    main()
