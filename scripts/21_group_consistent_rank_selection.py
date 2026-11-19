"""Stage 21 | Group-consistent gene-rank feature-selection prototype.

Research question (EXPLORATORY): does source-training-cell-line agreement in
class-vs-rest gene effects improve 3-class Leave-One-Cell-Line-Out performance
versus source-only ANOVA selection, with the SAME rank representation, K, C,
and classifier? This is a candidate adaptation of known stability ideas, NOT
proof of algorithmic novelty, independent validation, or DTP-specific biology.

Run from repository root, after Stage16 --three-class-washout:
    python scripts/21_group_consistent_rank_selection.py --self-test
    python scripts/21_group_consistent_rank_selection.py --exploratory-only

No external data/labels are read. Folds fixed to GSE193258 source study only;
Acute 24h versus DTP 21d is confounded by time and treatment conditions.
Artifacts under outputs/group_consistent_rank/ only. No existing output modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts"
OUT = ROOT / "outputs" / "group_consistent_rank"
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
CLASSES = ("Control", "Acute", "DTP")
K = 100
C = 0.01


def import_script(filename: str, label: str):
    path = SCRIPT / filename
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def within_sample_rank(X):
    import numpy as np
    from scipy.stats import rankdata
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 9 or X.shape[1] < K:
        raise ValueError("Invalid expression shape")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite expression")
    return rankdata(X, axis=1, method="average") / X.shape[1]


def select_panel(X, y, groups, *, mode: str, k: int = K):
    """Fit feature selection using ONLY training observations and training groups.

    Proposed mode: ANOVA signal multiplied by the strongest class-vs-rest
    per-line effect whose direction is consistent across training lines.
    No parameter is chosen by observed held-out performance.
    """
    import numpy as np
    from sklearn.feature_selection import f_classif
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    if len(X) != len(y) or len(groups) != len(y) or set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Require aligned 3-class training data")
    lines = sorted(set(groups.tolist()))
    if len(lines) != 3:
        raise ValueError("Require exactly three training cell lines")
    if any(Counter(y[groups == line]) != {0: 3, 1: 3, 2: 3} for line in lines):
        raise ValueError("Each training cell line needs 3 replicates per class")
    ranks = within_sample_rank(X)
    usable = np.flatnonzero(ranks.var(axis=0) > 1e-12)
    if len(usable) < k:
        raise ValueError("Too few variable genes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f_scores, _ = f_classif(ranks[:, usable], y)
    finite_f = np.isfinite(f_scores) & (f_scores >= 0)
    f_scores = np.where(finite_f, f_scores, -np.inf)
    if mode == "anova":
        score = f_scores
    elif mode == "group_consistent":
        effect_scores = []
        for klass in (0, 1, 2):
            effects = []
            for line in lines:
                m = groups == line
                a = ranks[m & (y == klass)][:, usable].mean(axis=0)
                b = ranks[m & (y != klass)][:, usable].mean(axis=0)
                effects.append(a - b)
            D = np.stack(effects)
            magnitude = np.median(np.abs(D), axis=0)
            directional_agreement = np.abs(np.sign(D).sum(axis=0)) / len(lines)
            consistency = np.min(np.abs(D), axis=0) / (np.max(np.abs(D), axis=0) + 1e-12)
            effect_scores.append(magnitude * directional_agreement**2 * (0.5 + 0.5*consistency))
        stable = np.max(np.stack(effect_scores), axis=0)
        score = np.where(finite_f, np.log1p(np.where(finite_f, f_scores, 0)) * stable, -np.inf)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # Lexicographic index is an explicit deterministic tie break.
    ranking = np.lexsort((usable, -score))
    selected = usable[ranking[:k]]
    if not np.isfinite(score[ranking[:k]]).all():
        raise ValueError("Nonfinite feature-selection score")
    return selected


def fit_predict(Xtr, ytr, Xtest, selected):
    """Exact Stage16 selected-feature rank representation and LR settings."""
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr = rankdata(Xtr[:, selected], axis=1, method="average") / len(selected)
    te = rankdata(Xtest[:, selected], axis=1, method="average") / len(selected)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=C, solver="lbfgs", max_iter=3000, random_state=42))
    model.fit(tr, ytr)
    return model.predict(te).astype(int)


def count_metrics(y, preds):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    y, preds = np.asarray(y), np.asarray(preds)
    if len(y) != len(preds) or Counter(y) != {0: len(y)//3, 1:len(y)//3, 2:len(y)//3}:
        raise ValueError("Test set must contain balanced three-class observations")
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    return {"n": int(len(y)), "correct": int((y == preds).sum()),
            "balanced_accuracy": float(balanced_accuracy_score(y,preds)),
            "macro_f1": float(f1_score(y,preds,average="macro",zero_division=0)),
            "recall_control": float(cm[0,0]/cm[0].sum()),
            "recall_acute": float(cm[1,1]/cm[1].sum()),
            "recall_dtp": float(cm[2,2]/cm[2].sum()),
            "confusion_matrix": cm.tolist()}


def load_data():
    import numpy as np
    stage17 = import_script("17_three_class_external_transfer.py", "stage17_gcrs")
    stage16 = import_script("16_acute_specificity_challenge.py", "stage16_gcrs")
    genes, ids, X, y, provenance = stage17.source_data()
    metadata = stage16.csv_rows(ROOT / "data/metadata/geo_samples.csv")
    by_id = {r["gsm"]: r for r in metadata if r.get("gse") == "GSE193258"}
    expected = {0:"DMSO", 1:"osimertinib acute 24 hrs.", 2:"osimertinib DTPs 3w."}
    if len(ids) != 36 or len(set(ids)) != 36 or X.shape != (36, 19712):
        raise ValueError("Unexpected frozen source identities or values")
    groups=[]
    for gsm, klass in zip(ids, y):
        if gsm not in by_id or by_id[gsm]["reported_treatment"] != expected[int(klass)]:
            raise ValueError("Sample state/source metadata mismatch: " + gsm)
        groups.append(by_id[gsm]["reported_cell_line"])
    if set(groups) != set(LINES):
        raise ValueError("Unexpected source cell lines")
    return genes, ids, X, y, np.asarray(groups), provenance


def run():
    import numpy as np
    genes, ids, X, y, groups, provenance = load_data()
    frozen = ROOT / "outputs/acute_challenge/16_three_class_predictions.csv"
    if not frozen.is_file():
        raise FileNotFoundError(f"Frozen Stage16 predictions required: {frozen}")
    with frozen.open(encoding="utf-8-sig",newline="") as f:
        records = list(csv.DictReader(f))
    reference = {(r["held_out_line"],r["gsm"]):r["predicted_state"] for r in records
        if r["method"] == "rank_selected_lr100_3class" and r["true_state"] in CLASSES}
    raw_reference = {(r["held_out_line"],r["gsm"]):r for r in records
        if r["method"] == "raw_log2tpm_lr500_3class" and r["true_state"] in CLASSES}
    if len(reference) != 36 or len(raw_reference) != 36:
        raise ValueError("Frozen baseline/raw comparator must have 36 unique held-out predictions")
    per_sample, per_line, gene_lists = [], [], {}
    for line in LINES:
        tr = np.flatnonzero(groups != line)
        ts = np.flatnonzero(groups == line)
        if len(tr) != 27 or len(ts) != 9 or Counter(y[ts]) != {0:3,1:3,2:3}:
            raise ValueError("Invalid leave-one-cell-line-out fold")
        fold_choices = {}
        for method in ("anova", "group_consistent"):
            chosen = select_panel(X[tr],y[tr],groups[tr],mode=method)
            gene_lists[(line,method)] = set(int(i) for i in chosen)
            fold_choices[method] = chosen
            pred = fit_predict(X[tr], y[tr], X[ts], chosen)
            if method == "anova":
                for i,p in zip(ts,pred):
                    expected = reference.get((line,ids[i]))
                    if expected != CLASSES[int(p)]:
                        raise ValueError(f"STOP: Stage16 baseline replay differs for {line}/{ids[i]}")
            metrics = count_metrics(y[ts],pred)
            per_line.append({"method":method,"held_out_line":line,**metrics})
            for i,p in zip(ts,pred):
                per_sample.append({"method":method,"held_out_line":line,"gsm":ids[i],
                    "true_state":CLASSES[int(y[i])],"predicted_state":CLASSES[int(p)],
                    "correct":int(y[i]==p)})
        if set(fold_choices["anova"])==set(fold_choices["group_consistent"]):
            pass  # Valid possible finding; do not force an artificial improvement.
    pooled = {}
    for method in ("anova", "group_consistent"):
        r = [x for x in per_sample if x["method"]==method]
        truth = np.array([CLASSES.index(x["true_state"]) for x in r])
        predicted = np.array([CLASSES.index(x["predicted_state"]) for x in r])
        pooled[method] = count_metrics(truth,predicted)
    # Preserve the stronger already-frozen raw-expression reference for fair reporting.
    raw_records = []
    for line in LINES:
        idxs = np.flatnonzero(groups == line)
        for i in idxs:
            existing = raw_reference.get((line, ids[i]))
            if existing is None or existing["true_state"] != CLASSES[int(y[i])]:
                raise ValueError("Frozen raw comparator identity/label mismatch")
            raw_records.append(existing)
    raw_truth = np.array([CLASSES.index(x["true_state"]) for x in raw_records])
    raw_preds = np.array([CLASSES.index(x["predicted_state"]) for x in raw_records])
    frozen_raw_summary = count_metrics(raw_truth,raw_preds)
    overlaps = {}
    for line in LINES:
        a,b=gene_lists[(line,"anova")],gene_lists[(line,"group_consistent")]
        overlaps[line] = {"shared_genes":len(a&b),"union_genes":len(a|b),
                           "jaccard":len(a&b)/len(a|b)}
    OUT.mkdir(parents=True,exist_ok=True)
    for name, recs in (("21_predictions.csv",per_sample),
                       ("21_fold_metrics.csv",[{k:v for k,v in r.items() if k!="confusion_matrix"} for r in per_line])):
        with (OUT/name).open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(recs[0]));writer.writeheader();writer.writerows(recs)
    report={"status":"EXPLORATORY_METHOD_PROTOTYPE_NOT_NOVELTY_OR_INDEPENDENT_VALIDATION",
      "created_utc":datetime.now(timezone.utc).isoformat(),"source_study":"GSE193258",
      "source_provenance":provenance,"source_n":36,"cell_lines":list(LINES),
      "evaluation":"4 leave-one-cell-line-out folds, held-out sample never used in feature selection or fitting",
      "model":"selected-gene within-sample ranks + StandardScaler + LogisticRegression(C=0.01)",
      "feature_count":K,"methods":{"anova":"Prior Stage16 selection, reproduced exactly",
        "group_consistent":"Training cell-line class-vs-rest rank-effect consistency weighted by ANOVA"},
      "pooled_metrics":pooled,"frozen_raw_lr500_reference":frozen_raw_summary,
      "fold_metrics":per_line,"feature_overlap":overlaps,
      "source_gene_panel_sha256":{line:{m:hashlib.sha256("\n".join(genes[i] for i in
             sorted(gene_lists[(line,m)])).encode()).hexdigest() for m in ("anova","group_consistent")}
             for line in LINES},
      "limitations":["All source folds inspected during earlier development; new method comparison is post-hoc exploratory.",
       "Acute 24h versus DTP 21d confounded by duration, drug exposure and treatment protocol.",
       "Novelty of group-consistent feature scoring NOT established: related stable-feature methods exist.",
       "No independent held-out external study, clinical outcomes, or biological marker certification.",
       "Small n=36, only four experimental cell lines; no statistical significance or proof of superiority."]}
    (OUT/"21_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Stage 21 — Group-consistent rank feature selection (EXPLORATORY)","",
           "Question: Does adding training-cell-line consistency improve held-out-cell-line classification?", "",
           "Same 100 features, within-panel rank representation, Logistic Regression, and source folds.","",
           "| Method | Correct / 36 | bACC | Macro-F1 | DTP recall |",
           "|---|---:|---:|---:|---:|"]
    for m in ("anova","group_consistent"):
        z=pooled[m]
        lines.append(f"| {m} | {z['correct']}/36 | {z['balanced_accuracy']:.3f} | {z['macro_f1']:.3f} | {z['recall_dtp']:.3f} |")
    z = frozen_raw_summary
    lines.append(f"| frozen raw 500-gene reference (Stage16) | {z['correct']}/36 | {z['balanced_accuracy']:.3f} | {z['macro_f1']:.3f} | {z['recall_dtp']:.3f} |")
    lines += ["", "| Held-out cell line | ANOVA bACC | Group-consistent bACC | Shared selected genes |",
             "|---|---:|---:|---:|"]
    for line in LINES:
        a=next(r for r in per_line if r["method"]=="anova" and r["held_out_line"]==line)
        b=next(r for r in per_line if r["method"]=="group_consistent" and r["held_out_line"]==line)
        lines.append(f"| {line} | {a['balanced_accuracy']:.3f} | {b['balanced_accuracy']:.3f} | {overlaps[line]['shared_genes']}/100 |")
    lines += ["","**Limits:** Not new biological evidence, untouched validation, causal deconfounding, or novelty proof.",
             "All results are retrospective and only test a candidate method."]
    (OUT/"21_comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Stage21 completed: exact Stage16 rank-selected LOCO baseline reproduced")
    for m,z in pooled.items():
        print(f"{m}: {z['correct']}/36, bACC={z['balanced_accuracy']:.3f}, macro-F1={z['macro_f1']:.3f}, DTP recall={z['recall_dtp']:.3f}")
    print("Frozen raw 500-gene reference:",f"{frozen_raw_summary['correct']}/36", "(not refitted)")
    print("Saved:",OUT/"21_comparison.md")
    print("EXPLORATORY ONLY; no independent validation or proven novel algorithm")


def self_test():
    import numpy as np
    rng=np.random.default_rng(210)
    n_genes=700
    groups=np.repeat(["A","B","C"],9)
    y=np.tile(np.repeat([0,1,2],3),3)
    X=rng.normal(1.0,0.2,(27,n_genes))
    X[y==1,10:50] += 0.7
    X[y==2,50:90] += 0.7
    a=select_panel(X,y,groups,mode="anova",k=100)
    b=select_panel(X,y,groups,mode="group_consistent",k=100)
    assert len(a)==len(set(a))==100
    assert len(b)==len(set(b))==100
    assert np.array_equal(b,select_panel(X,y,groups,mode="group_consistent",k=100))
    yp=fit_predict(X,y,X,b)
    stats=count_metrics(y,yp)
    assert stats["n"]==27 and 0<=stats["macro_f1"]<=1
    # Guard invalid line structure and missing classes.
    bad=groups.copy();bad[0]="C"
    try:
        select_panel(X,y,bad,mode="group_consistent")
    except ValueError:pass
    else:raise AssertionError("Invalid group split was accepted")
    try:
        select_panel(X,y[:26],groups,mode="group_consistent")
    except ValueError:pass
    else:raise AssertionError("Misaligned samples were accepted")
    print("SELF-TEST PASSED: source-only selection, reproducibility, 3-class metrics, split guards")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--exploratory-only",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test()
    elif args.exploratory_only:
        run()
    else:
        p.error("Real-data experiment requires --exploratory-only")

if __name__=="__main__":
    main()
