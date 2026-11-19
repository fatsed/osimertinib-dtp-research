"""Stage 27 | DTP-vs-both-negative-conditions, source-only exploratory benchmark.

Copy this file into scripts/ of fatsed/osimertinib-dtp-research.
Run from the repository root:
  python scripts/27_dtp_specific_contrast_benchmark.py --self-test
  python scripts/27_dtp_specific_contrast_benchmark.py --exploratory-only

Question: Can a TRAINING-cell-line-only, direction-consistent two-contrast
feature panel identify the observed DTP group against BOTH Control and Acute?

Primary outer evaluation: GSE193258, four Leave-One-Cell-Line-Out (LOCO) folds.
Each outer fit uses 27 samples from three lines (9 Control, 9 Acute, 9 DTP).
Each test fold has 9 held-out samples (3 per group).  Controls + Acutes
are the NEGATIVE class during fitting; each is evaluated separately at test.

Three independently TRAINED fair binary comparators:
  1) raw_lr500: raw log2TPM, 500 training-selected ANOVA genes, LR.
  2) rank_anova_lr100: 100 binary ANOVA genes from full-gene training ranks,
     within-panel rank representation, LR.
  3) dual_contrast_rank_lr100: identical rank/LR pipeline, but choose 100
     genes with DTP-Control and DTP-Acute expression-rank contrasts in the
     SAME strict direction in EACH of the 3 training lines. The ranking is
     min directional effect across the six training contrasts * log(1+F).

All LR classifiers use C=0.01 and class_weight='balanced', fixed before run.
No test/washout labels, features, margins or metadata can enter selection,
fit, model choice or decision threshold (zero). There is NO tuning here.

Additional DESCRIPTIVE diagnostics:
  * Historical Stage16 Acute-vs-DTP signed_rank_full, already frozen, scored
    on source Control/Acute/DTP. It has not learned Control as a negative;
    therefore it is NOT a fair trained DTP-vs-rest comparator.
  * Frozen source study's ShortWashout/LongWashout rows, scored by newly
    trained models but NEVER used as DTP/non-DTP labels or in selection.

Washout is a different harvested condition, NOT a verified negative DTP
state and NOT an independent cohort. Acute 24h and DTP 21d are fully
confounded with treatment duration. This is retrospective exploratory work,
not validation of biological DTP specificity or of algorithmic novelty.

Output: outputs/dtp_specificity/27_{report.md,summary.json,predictions.csv,
         washout_predictions.csv,selected_genes.csv}.
Existing reports and data are never changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ACUTE = ROOT / "outputs" / "acute_challenge"
OUT = ROOT / "outputs" / "dtp_specificity"
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
STATES = ("Control", "Acute", "DTP")
METHODS = ("raw_lr500", "rank_anova_lr100", "dual_contrast_rank_lr100")
HISTORICAL = "historical_acute_dtp_signed_rank_full"
RANK_K = 100
RAW_K = 500
C = 0.01
SEED = 42


def module(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def csv_read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen output: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_text(records: list[dict]) -> str:
    if not records:
        raise ValueError("Cannot export empty records")
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(records[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(records)
    return buf.getvalue()


def immutable_write(path: Path, contents: str) -> None:
    encoded = contents.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(
                f"Existing output differs; will NOT overwrite it: {path}. "
                "Preserve the original before running a changed experiment.")
        return
    path.write_bytes(encoded)


def validate_train(y3, groups, *, n_lines: int = 3):
    import numpy as np
    y3 = np.asarray(y3, dtype=int)
    groups = np.asarray(groups)
    if len(y3) != len(groups) or len(y3) != 9 * n_lines:
        raise ValueError("Unbalanced/misaligned training identities")
    lines = sorted(set(groups.tolist()))
    if len(lines) != n_lines:
        raise ValueError("Expected three distinct training lines")
    for line in lines:
        if Counter(y3[groups == line].tolist()) != {0: 3, 1: 3, 2: 3}:
            raise ValueError(f"Broken 3-class replicate counts in {line}")
    return lines


def rank_all(X):
    import numpy as np
    from scipy.stats import rankdata
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all() or X.shape[1] < RAW_K:
        raise ValueError("Invalid expression input")
    return rankdata(X, axis=1, method="average") / X.shape[1]


def select_rank_panel(X_train, y3_train, train_groups, *, mode: str, k: int = RANK_K):
    """Select only from training lines.  No test arguments are accepted.

    For the strict dual contrast, each selected gene must satisfy all six
    DTP-minus-control/acute comparisons with identical nonzero sign.
    Insufficient candidates cause a hard failure, never a silent fallback.
    """
    import numpy as np
    from sklearn.feature_selection import f_classif

    lines = validate_train(y3_train, train_groups)
    y = np.asarray(y3_train, dtype=int)
    g = np.asarray(train_groups)
    ranks = rank_all(X_train)
    if ranks.shape[0] != len(y) or not 1 <= k <= ranks.shape[1]:
        raise ValueError("Misaligned gene ranks or feature count")
    usable = np.flatnonzero(ranks.var(axis=0) > 1e-12)
    if len(usable) < k:
        raise ValueError("Fewer than k nonconstant training genes")
    binary = (y == 2).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f, _ = f_classif(ranks[:, usable], binary)
    good = np.isfinite(f) & (f >= 0)
    f = np.where(good, f, -np.inf)
    directional = None
    strict_count = None
    if mode == "anova":
        score = f
    elif mode == "dual_contrast":
        effects = []
        for line in lines:
            mask = g == line
            dtp = ranks[mask & (y == 2)][:, usable].mean(axis=0)
            ctl = ranks[mask & (y == 0)][:, usable].mean(axis=0)
            acute = ranks[mask & (y == 1)][:, usable].mean(axis=0)
            effects.extend([dtp - ctl, dtp - acute])
        effects = np.stack(effects)  # six TRAINING contrasts x usable genes
        min_up = np.min(effects, axis=0)
        min_down = np.min(-effects, axis=0)
        directional = np.where(min_up > 0, 1, np.where(min_down > 0, -1, 0))
        strict = directional != 0
        strict_count = int(np.count_nonzero(strict & good))
        if strict_count < k:
            raise ValueError(
                f"Only {strict_count} strictly dual-consistent genes, need k={k}. "
                "No fallback to non-specific features is permitted.")
        strength = np.maximum(min_up, min_down)
        score = np.where(strict & good, strength * np.log1p(np.maximum(f, 0)), -np.inf)
    else:
        raise ValueError(f"Unknown selection mode: {mode}")
    order = np.lexsort((usable, -score))
    if not np.isfinite(score[order[:k]]).all():
        raise ValueError("Cannot select k genes with finite positive evidence")
    selected = usable[order[:k]]
    selected_directions = None if directional is None else directional[order[:k]]
    if selected_directions is not None and not np.all(np.abs(selected_directions) == 1):
        raise AssertionError("Inconsistent gene passed strict two-contrast gate")
    return selected, {"strict_candidate_count": strict_count,
                      "selected_up": None if selected_directions is None else int((selected_directions == 1).sum()),
                      "selected_down": None if selected_directions is None else int((selected_directions == -1).sum())}


def fit_predict(X_train, y3_train, groups_train, X_test):
    """All training and feature selection is strictly isolated from X_test."""
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y3_train = np.asarray(y3_train, dtype=int)
    validate_train(y3_train, groups_train)
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Model train/test gene spaces differ")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Model input has invalid expression values")
    binary = (y3_train == 2).astype(int)
    outputs = {}
    metadata = {}
    # Same balanced binary LR recipe across all three comparators.
    raw = make_pipeline(
        VarianceThreshold(), SelectKBest(f_classif, k=RAW_K), StandardScaler(),
        LogisticRegression(C=C, solver="liblinear", class_weight="balanced",
                           max_iter=3000, random_state=SEED))
    raw.fit(X_train, binary)
    vt = raw.named_steps["variancethreshold"]
    kb = raw.named_steps["selectkbest"]
    selected_raw = np.flatnonzero(vt.get_support())[kb.get_support()]
    if len(selected_raw) != RAW_K:
        raise AssertionError("Raw selector returned wrong number of genes")
    outputs["raw_lr500"] = np.asarray(raw.decision_function(X_test), dtype=float).reshape(-1)
    metadata["raw_lr500"] = {"selected": selected_raw, "strict_candidate_count": None,
                             "selected_up": None, "selected_down": None}

    for mode, method in (("anova", "rank_anova_lr100"),
                         ("dual_contrast", "dual_contrast_rank_lr100")):
        selected, info = select_rank_panel(X_train, y3_train, groups_train, mode=mode)
        train_z = rankdata(X_train[:, selected], axis=1, method="average") / len(selected)
        test_z = rankdata(X_test[:, selected], axis=1, method="average") / len(selected)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, solver="liblinear", class_weight="balanced",
            max_iter=3000, random_state=SEED))
        pipe.fit(train_z, binary)
        outputs[method] = np.asarray(pipe.decision_function(test_z), dtype=float).reshape(-1)
        metadata[method] = {"selected": selected, **info}
    if set(outputs) != set(METHODS) or set(metadata) != set(METHODS):
        raise AssertionError("Method mismatch")
    if any(not np.isfinite(scores).all() or len(scores) != len(X_test)
           for scores in outputs.values()):
        raise ValueError("Invalid test scores")
    return outputs, metadata


def state_metrics(y3, predicted):
    """Binary DTP-vs-rest metrics plus *separate* Control and Acute error rates."""
    import numpy as np
    y3 = np.asarray(y3, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    if len(pred) != len(y3) or not set(np.unique(y3)) == {0, 1, 2}:
        raise ValueError("Expected all three source states")
    result = {}
    for state, idx in zip(STATES, (0, 1, 2)):
        mask = y3 == idx
        result[f"n_{state.lower()}"] = int(mask.sum())
        result[f"dtp_call_fraction_{state.lower()}"] = float(np.mean(pred[mask] == 1))
    if len(set((result["n_control"],result["n_acute"],result["n_dtp"]))) != 1:
        raise ValueError("Expected three equally sized source classes")
    negative_specificity = float(np.mean(pred[y3 != 2] == 0))
    dtp_recall = result["dtp_call_fraction_dtp"]
    result.update({"n":len(y3), "correct":int(np.sum(pred == (y3 == 2))),
                   "binary_bacc":float((negative_specificity + dtp_recall)/2),
                   "three_group_macro_recall":float(((1 - result["dtp_call_fraction_control"]) +
                        (1 - result["dtp_call_fraction_acute"]) + dtp_recall)/3),
                   "dtp_recall":dtp_recall,
                   "control_false_dtp_rate":result["dtp_call_fraction_control"],
                   "acute_false_dtp_rate":result["dtp_call_fraction_acute"]})
    return result


def read_historical_diagnostic():
    """Frozen Stage16 full washout score is a diagnostic, not a fair binary fit."""
    records = csv_read(ACUTE / "16_washout_predictions.csv")
    expected = {(line, state) for line in LINES
                for state in ("Control", "Acute", "DTP", "ShortWashout", "LongWashout")}
    if len(records) != 60 or Counter((r["held_out_line"],r["state"]) for r in records) != \
            Counter({key:3 for key in expected}):
        raise ValueError("Stage16 washout reference is incomplete")
    lookup = {}
    for r in records:
        key = (r["held_out_line"], r["gsm"])
        if key in lookup or r["predicted_state"] not in ("Acute", "DTP"):
            raise ValueError("Duplicate or invalid historical washout row")
        if not (-float("inf") < float(r["dtp_decision_margin"]) < float("inf")):
            raise ValueError("Nonfinite historical margin")
        lookup[key] = r
    prev = csv_read(ACUTE / "16_acute_vs_dtp_predictions.csv")
    signed = [r for r in prev if r["method"] == "signed_rank_full"]
    if len(signed) != 24:
        raise ValueError("Missing Stage16 source comparator")
    for r in signed:
        old = lookup.get((r["held_out_line"],r["gsm"]))
        if old is None or old["state"] != r["true_state"] or \
           old["predicted_state"] != r["predicted_state"] or \
           abs(float(old["dtp_decision_margin"]) - float(r["signed_dtp_margin"])) > 1e-7:
            raise ValueError("Frozen Stage16 diagnostic changed")
    return lookup


def load_audited_source_and_washout():
    """Reuse official mappings/hash checks from Stage16/21; no new dataset."""
    import numpy as np
    s21 = module(SCRIPTS / "21_group_consistent_rank_selection.py", "stage21_for27")
    s16 = module(SCRIPTS / "16_acute_specificity_challenge.py", "stage16_for27")
    s7 = module(SCRIPTS / "07_external_stress_test.py", "stage07_for27")
    genes, ids, X, y3, groups, provenance = s21.load_data()
    if len(genes) != 19712 or X.shape != (36,19712) or len(set(ids)) != 36:
        raise ValueError("Frozen three-state source changed")
    if any(Counter(np.asarray(y3)[np.asarray(groups) == line].tolist()) != {0:3,1:3,2:3}
           for line in LINES):
        raise ValueError("Wrong source replicate structure")
    p = json.loads((ROOT / "data/metadata/evaluation_protocol.json").read_text(encoding="utf-8"))
    if p.get("source_study") != "GSE193258" or \
       p.get("flags", {}).get("biology_approved_for_training") is not False:
        raise ValueError("Unexpected source protocol state")
    source_path = s7.safe_local_path(p["source_matrix"], "processed")
    genes_src, ids_src, X_src = s7.load_matrix(source_path,p["source_matrix_sha256"],np)
    if genes_src != genes or X_src.shape != (24,19712):
        raise ValueError("Source/washout gene mapping differs")
    if ids_src != [r["gsm"] for r in p["source_gsm_to_label"]]:
        raise ValueError("Source protocol GSMs changed")
    metadata = s16.csv_rows(s16.META / "geo_samples.csv")
    alignments = s16.csv_rows(s16.META / "expression_alignment.csv")
    wash_records, known = s16.verified_washout_columns(p, metadata, alignments)
    wash = s16.read_verified_washout(
        s16.RAW, s16.META / "expression_manifest.json", wash_records,
        genes, X_src, ids_src, known, np)
    if wash.shape != (24,19712) or len(wash_records) != 24:
        raise ValueError("Unexpected washout matrix shape")
    wash_ids = [gsm for gsm,_ in wash_records]
    if len(set(wash_ids)) != 24 or set(wash_ids) & set(ids):
        raise ValueError("Washout/source sample overlap")
    return genes,ids,X,np.asarray(y3,dtype=int),np.asarray(groups),provenance,wash_records,wash


def run():
    import numpy as np

    historical = read_historical_diagnostic()
    genes, ids, X, y3, groups, provenance, wash_records, wash = load_audited_source_and_washout()
    if any(str(x) not in LINES for x in groups):
        raise ValueError("Unexpected held-out line")
    for outer in LINES:
        validate_train(y3[groups != outer],groups[groups != outer])
    historical_source=[]
    for i,gsm in enumerate(ids):
        r = historical.get((str(groups[i]),gsm))
        if r is None or r["state"] != STATES[int(y3[i])]:
            raise ValueError(f"Historical diagnostic source mismatch: {gsm}")
        historical_source.append(int(r["predicted_state"] == "DTP"))

    predictions=[]
    wash_preds=[]
    panels={m:[] for m in METHODS}
    gene_rows=[]
    panel_info=[]
    for outer in LINES:
        tr=np.flatnonzero(groups != outer)
        te=np.flatnonzero(groups == outer)
        if len(tr)!=27 or len(te)!=9 or Counter(y3[te].tolist()) != {0:3,1:3,2:3}:
            raise ValueError("Broken outer LOCO split")
        wash_idx=[i for i,(gsm,r) in enumerate(wash_records) if r["cell_line"] == outer]
        if len(wash_idx)!=6 or Counter(wash_records[i][1]["state"] for i in wash_idx) != \
                {"ShortWashout":3, "LongWashout":3}:
            raise ValueError(f"Unexpected washout experimental conditions in {outer}")
        # These washout rows are test-only. Their labels never flow to fit_predict.
        test=np.vstack((X[te],wash[wash_idx]))
        outputs, metadata=fit_predict(X[tr],y3[tr],groups[tr],test)
        for method in METHODS:
            margin=np.asarray(outputs[method],dtype=float)
            for j,i in enumerate(te):
                predictions.append({"method":method,"held_out_line":outer,"gsm":ids[i],
                    "true_state":STATES[int(y3[i])],"margin":f"{float(margin[j]):.12g}",
                    "predicted_dtp":int(margin[j]>0),"correct":int((margin[j]>0)==(y3[i]==2))})
            for j,i in enumerate(wash_idx,start=len(te)):
                gsm,r=wash_records[i]
                old=historical.get((outer,gsm))
                if old is None or old["state"]!=r["state"]:
                    raise ValueError("Frozen washout sample mismatch")
                wash_preds.append({"method":method,"held_out_line":outer,"gsm":gsm,
                    "experimental_condition":r["state"],
                    "days_after_drug_removal":r["days_after_drug_removal"],
                    "margin":f"{float(margin[j]):.12g}",
                    "predicted_dtp_call_descriptive_only":int(margin[j]>0)})
            chosen=metadata[method]["selected"]
            panels[method].append(set(chosen.tolist()))
            panel_info.append({"method":method,"held_out_line":outer,
                "selected_n":len(chosen),
                "strict_consistent_candidates_in_train":metadata[method]["strict_candidate_count"],
                "up_genes_selected":metadata[method]["selected_up"],
                "down_genes_selected":metadata[method]["selected_down"]})
            for ix in chosen:
                gene_rows.append({"method":method,"held_out_line":outer,
                                  "gene":genes[int(ix)]})

    if len(predictions)!=len(METHODS)*36 or len(wash_preds)!=len(METHODS)*24:
        raise ValueError("Incomplete source or washout prediction coverage")
    fold_metrics=[]
    pooled={}
    for method in METHODS:
        for outer in (*LINES,"ALL"):
            rows=[r for r in predictions if r["method"]==method and
                  (outer=="ALL" or r["held_out_line"]==outer)]
            yy=np.asarray([STATES.index(r["true_state"]) for r in rows])
            pp=np.asarray([int(r["predicted_dtp"]) for r in rows])
            metrics=state_metrics(yy,pp)
            fold_metrics.append({"method":method,"held_out_line":outer,**metrics})
            if outer=="ALL": pooled[method]=metrics

    historical_fold=[]
    for outer in (*LINES,"ALL"):
        idxs=[i for i in range(len(ids)) if outer=="ALL" or groups[i]==outer]
        historic=state_metrics(y3[idxs],np.asarray(historical_source)[idxs])
        historical_fold.append({"method":HISTORICAL,"held_out_line":outer,**historic})
    pooled[HISTORICAL]=historical_fold[-1] | {}  # diagnostic, NOT matched fit

    washout_descriptive=[]
    for method in METHODS:
        for outer in LINES:
            for state in ("ShortWashout","LongWashout"):
                rows=[r for r in wash_preds if r["method"]==method and
                      r["held_out_line"]==outer and r["experimental_condition"]==state]
                if len(rows)!=3:
                    raise ValueError("Missing washout rows")
                margins=np.asarray([float(r["margin"]) for r in rows])
                washout_descriptive.append({"method":method,"held_out_line":outer,
                    "washout_condition":state,"n":3,
                    "mean_margin":float(np.mean(margins)),
                    "dtp_call_fraction_descriptive_only":float(np.mean(margins>0))})

    stability={}
    for method in METHODS:
        vec=[len(a&b)/len(a|b) for a,b in combinations(panels[method],2)]
        if len(vec)!=6:
            raise AssertionError("Missing panel pairs")
        stability[method]={"mean_pairwise_gene_jaccard":float(np.mean(vec)),
                           "min_pairwise_gene_jaccard":float(min(vec)),
                           "max_pairwise_gene_jaccard":float(max(vec))}

    prior_stage16=json.loads((ACUTE/"16_acute_vs_dtp_summary.json").read_text(encoding="utf-8"))
    report={
        "status":"EXPLORATORY_PREVIOUSLY_INSPECTED_NOT_INDEPENDENT",
        "question":"Can a source-training-line dual-contrast rank panel distinguish observed DTP from both Control and Acute?",
        "study":"GSE193258", "n_source":36, "n_washout_descriptive_only":24,
        "cell_lines":list(LINES),"states":list(STATES),
        "labels":{"Control":"DMSO","Acute":"osimertinib 24h","DTP":"osimertinib 21d"},
        "outer_train_n":27,"outer_test_n":9,"binary_training_positive":"DTP",
        "binary_training_negative":["Control","Acute"],
        "threshold":0.0,"class_weight":"balanced","regularization_C":C,
        "raw_k":RAW_K,"rank_k":RANK_K,"methods":list(METHODS),
        "fit_uses_only_other_3_lines":True,
        "gene_selection_uses_outer_test_data":False,
        "washout_used_for_training_or_selection":False,
        "frozen_source_sha256":prior_stage16["source_matrix_sha256"],
        "frozen_raw_sha256":prior_stage16["raw_source_sha256"],
        "historical_washout_csv_sha256":file_sha256(ACUTE/"16_washout_predictions.csv"),
        "source_provenance":provenance,
        "metrics":fold_metrics,"historical_diagnostic_metrics":historical_fold,
        "pooled_metrics":pooled,"panel_info":panel_info,
        "panel_stability":stability,"washout_descriptive":washout_descriptive,
        "limitations":[
            "Previously inspected four source cell lines; retrospective hypothesis development.",
            "Three biological-condition groups differ in treatment and duration; not a causal DTP-specific marker.",
            "Control and Acute constitute separate named negatives, not proof of DTP biology.",
            "The historical Acute/DTP model did not train on Control; diagnostic not fair head-to-head.",
            "Washout labels are NOT verified DTP-negative labels; no washout accuracy is computed.",
            "Only four held-out cell lines, 3 replicates per class, no independent validation.",
            "The proposed dual contrast is an adapted feature-selection heuristic, not proven novel."
        ]
    }
    md=["# Stage 27 — DTP against two observed negative conditions", "",
        "**Exploratory retrospective benchmark. No independent validation or biological DTP-specificity claim.**", "",
        "All models use outer LOCO. Each fit: 27 samples from 3 training lines;",
        "test: 9 samples from 1 excluded line. The two negatives (Control, Acute)",
        "are learned as negative labels, but are inspected individually at evaluation.", "",
        "## Pooled held-out GSE193258 scores", "",
        "| Model | Correct/36 | Binary bACC | 3-group macro recall | Control false DTP | Acute false DTP | DTP recall |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for method in (*METHODS,HISTORICAL):
        z=pooled[method]
        suffix=" (historical diagnostic only)" if method==HISTORICAL else ""
        md.append(f"| {method}{suffix} | {z['correct']}/36 | {z['binary_bacc']:.3f} | "
                  f"{z['three_group_macro_recall']:.3f} | {z['control_false_dtp_rate']:.3f} | "
                  f"{z['acute_false_dtp_rate']:.3f} | {z['dtp_recall']:.3f} |")
    md += ["", "## Per-held-out cell line: 3-group macro recall", "",
           "| Model | H1975 | HCC2935 | HCC827 | PC9 |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        m={r["held_out_line"]:r for r in fold_metrics if r["method"]==method}
        md.append("| "+method+" | "+" | ".join(f"{m[l]['three_group_macro_recall']:.3f}" for l in LINES)+" |")
    md += ["", "## Selection audit", "",
           "Dual-contrast features must have DTP-vs-Control and DTP-vs-Acute",
           "differences with identical sign in EACH of the 3 training lines.",
           "If fewer than 100 qualify in any fold, run stops with no fallback.", "",
           "| Model | Mean selected-gene Jaccard (4 folds) |", "|---|---:|"]
    for method in METHODS:
        md.append(f"| {method} | {stability[method]['mean_pairwise_gene_jaccard']:.3f} |")
    md += ["", "## Washout descriptive check (not a labeled test)", "",
           "Predicted-DTP fractions and uncalibrated decision margins are in",
           "`27_washout_predictions.csv` and `27_summary.json`. The Short/Long",
           "Washout conditions must NOT be treated as confirmed negatives.", "",
           "## Interpretation limits", "",
           "- The earlier Acute-vs-DTP signed-rank model calls all source Control samples DTP; it never trained on Control.",
           "- Stage27 introduces Control as a training negative and tests separate false-DTP rates.",
           "- Across the source study, duration of drug exposure remains confounded with state labels.",
           "- All four cell lines have previously been inspected; no post-hoc winner may be called independently validated.",
           "- Do not change selection rule, k, C, feature scale or threshold based on these outer-fold scores.", ""]
    OUT.mkdir(parents=True,exist_ok=True)
    files={
        "27_predictions.csv":csv_text(predictions),
        "27_washout_predictions.csv":csv_text(wash_preds),
        "27_selected_genes.csv":csv_text(gene_rows),
        "27_summary.json":json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
        "27_report.md":"\n".join(md)
    }
    # Preflight avoids partial replacement of existing experiments.
    for name,contents in files.items():
        path=OUT/name
        if path.exists() and path.read_bytes()!=contents.encode("utf-8"):
            raise FileExistsError(f"Refusing to replace changed prior result: {path}")
    for name,contents in files.items():
        immutable_write(OUT/name,contents)
    print("Stage27 completed: trained 3 DTP-vs-(Control+Acute) methods with source-only LOCO")
    for method in (*METHODS,HISTORICAL):
        z=pooled[method]
        print(f"{method}: {z['correct']}/36 3-group_macro={z['three_group_macro_recall']:.3f} "
              f"Control_false_DTP={z['control_false_dtp_rate']:.3f} "
              f"Acute_false_DTP={z['acute_false_dtp_rate']:.3f} "
              f"DTP_recall={z['dtp_recall']:.3f}")
    print("Saved outputs/dtp_specificity/27_{report.md,summary.json,predictions.csv,washout_predictions.csv,selected_genes.csv}")
    print("EXPLORATORY: washout is descriptive; duration confounding remains; no novel algorithm or independent validation claim")


def self_test():
    import numpy as np
    rng=np.random.default_rng(27)
    groups=np.repeat(["A","B","C"],9)
    y=np.tile(np.repeat([0,1,2],3),3)
    X=rng.normal(3,.2,size=(27,700))
    X[y==2,0:160] += 6.0
    X[y==2,160:320] -= 2.0
    validate_train(y,groups)
    assert state_metrics([0,0,1,1,2,2],[0,0,0,1,1,1])["three_group_macro_recall"]==5/6
    sel,info=select_rank_panel(X,y,groups,mode="dual_contrast",k=100)
    sel2,info2=select_rank_panel(X,y,groups,mode="dual_contrast",k=100)
    assert np.array_equal(sel,sel2) and info==info2 and info["strict_candidate_count"]>=100
    assert info["selected_up"]+info["selected_down"]==100
    a,_=select_rank_panel(X,y,groups,mode="anova",k=100)
    assert len(set(sel))==len(set(a))==100
    # Both datasets are identical except for test rows: selected features must not change.
    o1,p1=fit_predict(X,y,groups,X[:5])
    o2,p2=fit_predict(X,y,groups,np.flip(X[:5],axis=0))
    assert all(np.array_equal(p1[k]["selected"],p2[k]["selected"]) for k in METHODS)
    assert all(np.allclose(o1[k][::-1],o2[k]) for k in METHODS)
    try:
        select_rank_panel(X,y,np.array(["A"]*27),mode="dual_contrast")
    except ValueError: pass
    else: raise AssertionError("Invalid source cell line structure accepted")
    try:
        select_rank_panel(np.ones_like(X),y,groups,mode="dual_contrast")
    except ValueError: pass
    else: raise AssertionError("No-signal data incorrectly passed strict consistency")
    # A target label does not enter the fitting function signature.
    assert len(o1)==3 and all(np.isfinite(v).all() for v in o1.values())
    print("SELF-TEST PASSED: strict dual contrasts, fold guards, deterministic panels, no test-label access, 3-condition metrics")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--exploratory-only",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test();return
    if not args.exploratory_only:
        p.error("Previously inspected cohort: explicitly specify --exploratory-only")
    run()


if __name__ == "__main__":
    main()
