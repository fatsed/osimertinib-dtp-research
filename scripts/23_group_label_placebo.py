"""Stage 23 | Training-group-label placebo for Stage 21 GCR-FS.

Fixed negative control: stratify by outcome class and shuffle only TRAINING
cell-line memberships within each class, preserving 3 observations per class
per pseudo-line. Compare the unchanged Stage21 GCR-FS selector's source LOCO
predictions against real grouping, Stage21 ANOVA, and the frozen raw comparator.

This is a post-hoc exploratory diagnostic, not a permutation significance test,
independent validation, an algorithm novelty proof, or evidence of causal
DTP-specific biology. No model recipe or external dataset is modified.

Usage from repository root:
  python scripts/23_group_label_placebo.py --self-test
  python scripts/23_group_label_placebo.py --exploratory-only
  python scripts/23_group_label_placebo.py --exploratory-only --permutations 25

Outputs: outputs/group_consistent_rank/23_placebo_summary.json,
         23_placebo_trials.csv, 23_placebo_report.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs/group_consistent_rank"
SEED = 230619
METHOD = "group_consistent"


def load_stage21():
    path = SCRIPTS / "21_group_consistent_rank_selection.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("stage21_placebo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import Stage21")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pseudo_groups_within_class(y, original_groups, rng):
    """Shuffle training group identities conditional on y; no test observations."""
    y = np.asarray(y)
    original_groups = np.asarray(original_groups)
    if len(y) != len(original_groups) or set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Invalid training labels/groups")
    lines = sorted(set(original_groups.tolist()))
    if len(lines) != 3 or any(Counter(y[original_groups == line]) != {0:3, 1:3, 2:3} for line in lines):
        raise ValueError("Training fold must be 3 lines x 3 classes x 3 replicates")
    shuffled = original_groups.copy()
    for klass in (0, 1, 2):
        ix = np.flatnonzero(y == klass)
        shuffled[ix] = rng.permutation(original_groups[ix])
    for line in lines:
        if Counter(y[shuffled == line]) != {0:3, 1:3, 2:3}:
            raise AssertionError("Pseudo-group shuffle broke class balance")
    return shuffled


def panel_hash(indices, genes):
    names = "\n".join(genes[int(i)] for i in sorted(set(int(i) for i in indices)))
    return hashlib.sha256(names.encode()).hexdigest()


def mean_jaccard(panels):
    vals = [len(a & b) / len(a | b) for a, b in combinations(panels, 2)]
    if len(vals) != 6:
        raise ValueError("Expected four panels and six pairwise comparisons")
    return float(np.mean(vals))


def metrics_equal(a, b, tol=1e-9):
    for field in ("correct", "n"):
        if int(a[field]) != int(b[field]):
            return False
    for field in ("balanced_accuracy", "macro_f1", "recall_dtp"):
        if abs(float(a[field]) - float(b[field])) > tol:
            return False
    return True


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def self_test():
    rng = np.random.default_rng(23)
    g = np.repeat(["A", "B", "C"], 9)
    y = np.tile(np.repeat([0, 1, 2], 3), 3)
    p = pseudo_groups_within_class(y, g, np.random.default_rng(49))
    q = pseudo_groups_within_class(y, g, np.random.default_rng(49))
    assert np.array_equal(p, q)
    assert Counter(zip(p.tolist(), y.tolist())) == Counter(zip(g.tolist(), y.tolist()))
    assert set(p) == set(g) and len(p) == len(g)
    try:
        pseudo_groups_within_class(y[:-1], g, rng)
    except ValueError:
        pass
    else:
        raise AssertionError("Unaligned groups accepted")
    a = [{1,2,3},{1,2,4},{1,3,4},{2,3,4}]
    assert abs(mean_jaccard(a) - .5) < 1e-12
    good={"correct":3,"n":3,"balanced_accuracy":1.,"macro_f1":1.,"recall_dtp":1.}
    assert metrics_equal(good, dict(good))
    assert not metrics_equal(good, {**good,"correct":2})
    print("SELF-TEST PASSED: stratified training-only pseudo-groups, deterministic shuffling, panel stability and guard checks")


def run(permutations):
    if not 1 <= permutations <= 50:
        raise ValueError("Permutations must be from 1 to 50")
    s21 = load_stage21()
    prior_path = OUT / "21_summary.json"
    prior_predictions = OUT / "21_predictions.csv"
    if not prior_path.is_file() or not prior_predictions.is_file():
        raise FileNotFoundError("First run Stage21 on real source data")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if "EXPLORATORY" not in prior.get("status", ""):
        raise ValueError("Incorrect prior protocol status")
    genes, ids, X, y, groups, provenance = s21.load_data()
    if X.shape != (36,19712) or len(set(ids)) != 36 or set(groups) != set(s21.LINES):
        raise ValueError("Frozen source cohort changed")
    if (prior["source_provenance"] != provenance or
        prior["cell_lines"] != list(s21.LINES) or prior["feature_count"] != 100):
        raise ValueError("Stage21 source provenance or design changed")
    frozen_rows = load_csv(prior_predictions)
    frozen = {}
    for row in frozen_rows:
        key = (row["method"], row["held_out_line"], row["gsm"])
        if key in frozen:
            raise ValueError("Duplicate Stage21 held-out prediction")
        frozen[key] = row
    if len(frozen) != 72:
        raise ValueError("Expected 72 source LOCO predictions: two methods x 36")
    folds = []
    for held in s21.LINES:
        train = np.flatnonzero(groups != held)
        test = np.flatnonzero(groups == held)
        if len(train) != 27 or len(test) != 9 or Counter(y[test]) != {0:3,1:3,2:3}:
            raise ValueError("Invalid held-out source line")
        panels = {}
        for mode in ("anova", METHOD):
            ix = s21.select_panel(X[train], y[train], groups[train], mode=mode)
            if panel_hash(ix, genes) != prior["source_gene_panel_sha256"][held][mode]:
                raise ValueError(f"Stage21 selection mismatch: {held}/{mode}")
            pred = s21.fit_predict(X[train], y[train], X[test], ix)
            for pos, guess in zip(test,pred):
                rec=frozen.get((mode,held,ids[pos]))
                if (rec is None or rec["true_state"] != s21.CLASSES[int(y[pos])] or
                    rec["predicted_state"] != s21.CLASSES[int(guess)]):
                    raise ValueError(f"Stage21 predictions changed: {held}/{mode}/{ids[pos]}")
            panels[mode] = set(ix.tolist())
        folds.append({"line":held,"train":train,"test":test,"panels":panels})
    real_j = mean_jaccard([f["panels"][METHOD] for f in folds])
    true_metric = prior["pooled_metrics"][METHOD]
    anova_metric = prior["pooled_metrics"]["anova"]
    frozen_raw = prior["frozen_raw_lr500_reference"]
    all_trials=[]
    for trial in range(permutations):
        actual=[];predicted=[];pseudo_panels=[]
        for j,fold in enumerate(folds):
            tr, te = fold["train"],fold["test"]
            # Independent deterministic random pseudo-groups per fold and trial.
            prng = np.random.default_rng(SEED+10000*trial+j)
            g_pseudo=pseudo_groups_within_class(y[tr],groups[tr],prng)
            chosen=s21.select_panel(X[tr],y[tr],g_pseudo,mode=METHOD)
            p=s21.fit_predict(X[tr],y[tr],X[te],chosen)
            actual.extend(y[te].tolist());predicted.extend(p.tolist())
            pseudo_panels.append(set(chosen.tolist()))
        score=s21.count_metrics(np.array(actual),np.array(predicted))
        all_trials.append({"trial":trial+1,"correct":score["correct"],
            "balanced_accuracy":score["balanced_accuracy"],
            "macro_f1":score["macro_f1"],"dtp_recall":score["recall_dtp"],
            "mean_panel_jaccard":mean_jaccard(pseudo_panels)})
    def stats(key):
        vals=np.array([float(item[key]) for item in all_trials])
        return {"mean":float(vals.mean()),"median":float(np.median(vals)),
                "min":float(vals.min()),"max":float(vals.max()),
                "fraction_ge_observed":float(np.mean(vals >= (real_j if key=="mean_panel_jaccard"
                                                           else true_metric["balanced_accuracy"] if key=="balanced_accuracy"
                                                           else true_metric["correct"] if key=="correct"
                                                           else true_metric["macro_f1"] if key=="macro_f1"
                                                           else true_metric["recall_dtp"])))}
    summary={"status":"EXPLORATORY_POSTHOC_GROUP_PLACEBO_NOT_INDEPENDENT_OR_INFERENTIAL",
       "purpose":"Training-group label sensitivity for GCR-FS; preserves labels and per-class group counts",
       "n_permutations":permutations,"seed":SEED,"source_study":"GSE193258",
       "source_n":36,"n_cell_lines":4,"training_observations_per_fold":27,
       "source_provenance":provenance,"real_gcr":true_metric,
       "real_anova":anova_metric,"frozen_raw_lr500":frozen_raw,
       "real_gcr_panel_mean_jaccard":real_j,
       "pseudo_group_diagnostics":{k:stats(k) for k in ("correct","balanced_accuracy","macro_f1","dtp_recall","mean_panel_jaccard")},
       "limitations":["Only historically inspected source folds; cannot establish independent algorithm superiority.",
           "Pseudo-cell-line labels are randomized within true classes in TRAIN data only; this is a diagnostic placebo.",
           "Descriptive fractions of placebo trials are NOT p-values or valid post-hoc significance claims.",
           "All four LOCO folds share training observations, so Jaccard values are correlated.",
           "Acute 24h vs DTP 21d confounded by treatment duration and exposure.",
           "The stronger raw 500-gene baseline must remain in any manuscript."]}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"23_placebo_trials.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_trials[0]));w.writeheader();w.writerows(all_trials)
    (OUT/"23_placebo_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    d=summary["pseudo_group_diagnostics"]
    lines=["# Stage23 — Training-group-label placebo (EXPLORATORY)","",
       "This placebo shuffles cell-line labels **within each class on TRAINING folds only**.",
       "Same GCR-FS scoring formula, 100 genes and Logistic Regression classifier.","",
       "| Source LOCO method | Correct/36 | bACC | Macro-F1 | DTP recall |",
       "|---|---:|---:|---:|---:|",
       f"| Stage21 ANOVA | {anova_metric['correct']}/36 | {anova_metric['balanced_accuracy']:.3f} | {anova_metric['macro_f1']:.3f} | {anova_metric['recall_dtp']:.3f} |",
       f"| Stage21 real-cell-line GCR-FS | {true_metric['correct']}/36 | {true_metric['balanced_accuracy']:.3f} | {true_metric['macro_f1']:.3f} | {true_metric['recall_dtp']:.3f} |",
       f"| Frozen raw 500-gene comparator | {frozen_raw['correct']}/36 | {frozen_raw['balanced_accuracy']:.3f} | {frozen_raw['macro_f1']:.3f} | {frozen_raw['recall_dtp']:.3f} |",
       "","## Placebo distribution", "",
       "| Metric | Real GCR-FS | Pseudo-group median | Pseudo-group range | Fraction placebo >= real |",
       "|---|---:|---:|---:|---:|",
       f"| LOCO bACC | {true_metric['balanced_accuracy']:.3f} | {d['balanced_accuracy']['median']:.3f} | {d['balanced_accuracy']['min']:.3f}–{d['balanced_accuracy']['max']:.3f} | {d['balanced_accuracy']['fraction_ge_observed']:.3f} |",
       f"| Panel mean Jaccard | {real_j:.3f} | {d['mean_panel_jaccard']['median']:.3f} | {d['mean_panel_jaccard']['min']:.3f}–{d['mean_panel_jaccard']['max']:.3f} | {d['mean_panel_jaccard']['fraction_ge_observed']:.3f} |",
       f"| DTP recall | {true_metric['recall_dtp']:.3f} | {d['dtp_recall']['median']:.3f} | {d['dtp_recall']['min']:.3f}–{d['dtp_recall']['max']:.3f} | {d['dtp_recall']['fraction_ge_observed']:.3f} |",
       "",f"Fixed seed: {SEED}; pseudo-group assignments tested: {permutations}.",
       "The fraction of placebo trials exceeding the real result is descriptive, not a p-value.",
       "No independent validation, significance, general superiority or biological specificity claim is supported."]
    (OUT/"23_placebo_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Stage23 completed: {permutations} stratified training-group placebo trials")
    print(f"Real GCR-FS: bACC={true_metric['balanced_accuracy']:.3f}, Jaccard={real_j:.3f}")
    print(f"Pseudo-group median: bACC={d['balanced_accuracy']['median']:.3f}, Jaccard={d['mean_panel_jaccard']['median']:.3f}")
    print("Saved outputs/group_consistent_rank/23_placebo_{report.md,summary.json,trials.csv}")
    print("EXPLORATORY: no statistical significance or independent validation claim")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--exploratory-only',action='store_true')
    p.add_argument('--permutations',type=int,default=15)
    args=p.parse_args()
    if args.self_test:
        self_test()
    elif args.exploratory_only:
        run(args.permutations)
    else:
        p.error('Real data require --exploratory-only')


if __name__=='__main__':
    main()
