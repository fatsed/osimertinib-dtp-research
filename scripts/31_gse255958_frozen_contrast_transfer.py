#!/usr/bin/env python3
"""Stage 31: retrospective GSE255958 PC9 cross-study stress test.

Never a confirmatory test: GSE255958 and source PC9 folds were previously
inspected. Source-only, PC9-excluded fits replay frozen Stage-28 predictions.
Only fixed acute-only and dual-contrast 100-gene rank panels are scored.
No target labels, features, or external margins are used to choose a model,
parameter, panel, threshold, or normalization. Zero decision threshold.

Target gene-coverage policy is frozen in this code, BEFORE reading the external
archive: require >=90/100 nonambiguous selected genes in ALL nine target samples.
When any of the 100 features is absent, calculate within-sample ranks among
observed panel features and use TRAINING-source panel-rank means for missing
positions. This is a prespecified missingness *adaptation*, not an exact
100-gene equivalent. Disclose available count and affected predictions.

Outputs are never overwritten if contents have changed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs" / "cross_study_contrast"
STATES = ("Control", "Acute", "DTP")
METHODS = ("acute_only_rank_lr100", "dual_contrast_rank_lr100")
SOURCE_LINES = ("H1975", "HCC2935", "HCC827")
TARGET_GSE = "GSE255958"
SELECTED_K = 100
MIN_COVERAGE = 90
C = 0.01
THRESHOLD = 0.0
SOURCE_ROWS = ROOT / "outputs" / "dtp_specificity" / "28_predictions.csv"
SOURCE_GENES_ACUTE = ROOT / "outputs" / "dtp_specificity" / "28_selected_genes.csv"
SOURCE_GENES_DUAL = ROOT / "outputs" / "dtp_specificity" / "27_selected_genes.csv"
STAGE17_SUMMARY = ROOT / "outputs" / "three_class_external" / "17_pc9_excluded_summary.json"
STAGE17_PREDICTIONS = ROOT / "outputs" / "three_class_external" / "17_pc9_excluded_predictions.csv"


def imp(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing existing script: {path}")
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing existing experiment output: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def dump_csv(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("Cannot export empty prediction records")
    f = io.StringIO(newline="")
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return f.getvalue()


def panels_from_frozen():
    specs = {
        "acute_only_rank_lr100": SOURCE_GENES_ACUTE,
        "dual_contrast_rank_lr100": SOURCE_GENES_DUAL,
    }
    names = {}
    for method, path in specs.items():
        rows = [r["gene"] for r in read_csv(path)
                if r["method"] == method and r["held_out_line"] == "PC9"]
        if len(rows) != SELECTED_K or len(set(rows)) != SELECTED_K:
            raise ValueError(f"Frozen PC9 panel incomplete or duplicated: {method}")
        names[method] = rows
    return names


def source_model_replay(panels):
    """Re-train frozen source-only PC9 fold; compare ALL margins to Stage28."""
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    s21 = imp(SCRIPTS / "21_group_consistent_rank_selection.py", "s21_stage31")
    s27 = imp(SCRIPTS / "27_dtp_specific_contrast_benchmark.py", "s27_stage31")
    s28 = imp(SCRIPTS / "28_dual_contrast_ablation.py", "s28_stage31")
    genes, gsms, X, y, groups, provenance = s21.load_data()
    y, groups = np.asarray(y, dtype=int), np.asarray(groups)
    if (X.shape != (36, 19712) or len(set(gsms)) != 36 or
        Counter(y.tolist()) != {0:12, 1:12, 2:12} or
        set(groups) != {*SOURCE_LINES, "PC9"}):
        raise ValueError("Frozen source matrix/cohort changed")
    tr, te = np.flatnonzero(groups != "PC9"), np.flatnonzero(groups == "PC9")
    if (len(tr) != 27 or len(te) != 9 or
        Counter(y[tr].tolist()) != {0:9, 1:9, 2:9} or
        Counter(y[te].tolist()) != {0:3, 1:3, 2:3}):
        raise ValueError("Source PC9-excluded design changed")
    s27.validate_train(y[tr], groups[tr])

    frozen_rows = read_csv(SOURCE_ROWS)
    ref = {}
    for r in frozen_rows:
        if r["held_out_line"] == "PC9" and r["method"] in METHODS:
            key = (r["method"], r["gsm"])
            if key in ref:
                raise ValueError(f"Duplicate Stage28 reference: {key}")
            ref[key] = r
    if len(ref) != len(METHODS) * 9:
        raise ValueError("Stage28 must have exactly 18 frozen PC9 predictions")
    models = {}
    source_rows = []
    for method in METHODS:
        if method == "acute_only_rank_lr100":
            panel_idx, _ = s28.select_single_contrast(X[tr], y[tr], groups[tr], negative=1)
        else:
            panel_idx, _ = s27.select_rank_panel(X[tr], y[tr], groups[tr], mode="dual_contrast")
        selected = [genes[int(i)] for i in panel_idx]
        if selected != panels[method]:
            raise ValueError(f"Stage28 ordered gene panel changed: {method}")
        ztr = rankdata(X[tr][:, panel_idx], method="average", axis=1) / SELECTED_K
        zte = rankdata(X[te][:, panel_idx], method="average", axis=1) / SELECTED_K
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, solver="liblinear", class_weight="balanced", max_iter=3000,
            random_state=42))
        model.fit(ztr, (y[tr] == 2).astype(int))
        pred_margin = np.asarray(model.decision_function(zte), dtype=float)
        for i, margin in zip(te, pred_margin):
            gsm = gsms[int(i)]
            r = ref.get((method, gsm))
            if r is None:
                raise ValueError(f"Stage28 missing source PC9 record {gsm}")
            if (r["true_state"] != STATES[int(y[i])] or
                int(margin > THRESHOLD) != int(r["predicted_dtp"]) or
                not np.isclose(margin, float(r["margin"]), atol=1e-8, rtol=1e-7)):
                raise ValueError(f"Stage28 source PC9 margin changed: {method}/{gsm}")
            source_rows.append({"method": method, "source_gsm": gsm,
                                "true_state": STATES[int(y[i])],
                                "source_margin": float(margin)})
        models[method] = {"classifier":model, "panel": selected,
                          "train_feature_mean":ztr.mean(axis=0)}
    return models, source_rows, provenance


def validate_prior_target(gsms: list[str], truth: dict[str,str]):
    """Prior GSE255958 GSM/label mapping must agree with the old study review."""
    rows = read_csv(STAGE17_PREDICTIONS)
    prior = {}
    for r in rows:
        if r["scope"] != "external_GSE255958_PC9":
            continue
        if r["gsm"] not in truth:
            raise ValueError(f"Unexpected prior target GSM {r['gsm']}")
        if r["gsm"] in prior and r["true_state"] != prior[r["gsm"]]:
            raise ValueError("Inconsistent Stage17 label mapping")
        prior[r["gsm"]] = r["true_state"]
    if set(prior) != set(gsms) or any(prior[g] != truth[g] for g in gsms):
        raise ValueError("GSE255958 labels differ from audited Stage17 review")


def target_rank_features(targets, order: list[str], model: dict,
                         minimum: int = MIN_COVERAGE):
    """Return source-width target panel-ranks with predeclared missingness policy.

    This function does not accept target labels or target outcome counts.
    No adaptive parameters or target-trained statistics are computed.
    """
    import numpy as np
    from scipy.stats import rankdata
    panel = model["panel"]
    mean = np.asarray(model["train_feature_mean"], dtype=float)
    if len(panel) != SELECTED_K or len(set(panel)) != SELECTED_K or mean.shape != (SELECTED_K,):
        raise ValueError("Malformed source panel")
    if not 1 <= minimum <= SELECTED_K:
        raise ValueError("Invalid coverage policy")
    z = np.repeat(mean[None, :], len(order), axis=0)
    coverage = []
    for i, gsm in enumerate(order):
        if gsm not in targets or not isinstance(targets[gsm], dict):
            raise ValueError(f"Missing target RSEM data for {gsm}")
        measured = targets[gsm]
        observed = [j for j,g in enumerate(panel) if g in measured]
        n = len(observed)
        if n < minimum:
            raise ValueError(f"Target {gsm} covers only {n}/{SELECTED_K} selected genes")
        values = np.asarray([measured[panel[j]] for j in observed], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Nonfinite or negative TPM for {gsm}")
        # Monotone within-sample measurement shift does not change these ranks.
        z[i, observed] = rankdata(values, method="average") / n
        coverage.append({"gsm":gsm,"available":n,"missing":SELECTED_K-n,
                         "missing_gene_ids":";".join(panel[j] for j in range(SELECTED_K) if j not in set(observed))})
    if not np.isfinite(z).all():
        raise AssertionError("Rank representation contains nonfinite values")
    return z, coverage


def target_metrics(labels, scores):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    states=np.asarray(labels)
    scores=np.asarray(scores,dtype=float)
    if Counter(states.tolist()) != {"Control":3,"Acute":3,"DTP":3}:
        raise ValueError("Expected 3 target samples for each experimental state")
    if not np.isfinite(scores).all() or scores.shape != (9,):
        raise ValueError("Wrong/nonfinite target score vector")
    pred=scores>THRESHOLD
    counts={state:int(sum(pred[states==state])) for state in STATES}
    return {
        "n":9,"correct":int(sum((pred==(states=="DTP")))),
        "binary_balanced_accuracy":float((sum(~pred[states!="DTP"])/6 + sum(pred[states=="DTP"])/3)/2),
        "three_condition_macro_recall":float(((1-counts["Control"]/3)+(1-counts["Acute"]/3)+counts["DTP"]/3)/3),
        "control_false_dtp":counts["Control"],"acute_false_dtp":counts["Acute"],
        "dtp_recall":float(counts["DTP"]/3),
        "dtp_correct":counts["DTP"],
        "binary_auc":float(roc_auc_score((states=="DTP").astype(int),scores)),
    }


def write_immutable(records: dict[str,str]):
    OUT.mkdir(parents=True,exist_ok=True)
    for name,body in records.items():
        path=OUT/name
        if path.exists() and path.read_bytes()!=body.encode("utf-8"):
            raise FileExistsError(f"Not overwriting existing result: {path}")
    for name,body in records.items():
        path=OUT/name
        if not path.exists():path.write_text(body,encoding="utf-8",newline="")


def run():
    import numpy as np
    s17 = imp(SCRIPTS / "17_three_class_external_transfer.py", "s17_stage31")
    panels = panels_from_frozen()
    models, source_rows, source_provenance = source_model_replay(panels)
    # The target archive has NOT been read at this point. We already replayed
    # frozen source PC9 predictions, and model coefficients are fixed.
    frozen_summary = json.loads(STAGE17_SUMMARY.read_text(encoding="utf-8"))
    if (frozen_summary.get("source_pc9_excluded") is not True or
        frozen_summary.get("target_tar_sha256") != sha(s17.TARGET)):
        raise ValueError("GSE255958 archive differs from Stage17 audited archive")
    target_ids = [gsm for state in STATES for gsm in s17.GROUPS[state]]
    target_truth = {gsm:state for state in STATES for gsm in s17.GROUPS[state]}
    if len(target_ids)!=9 or len(set(target_ids))!=9:
        raise ValueError("Stage17 fixed target GSM mapping is not 3+3+3")
    resistant=set(s17.GROUPS["Resistant"])
    if resistant & set(target_ids) or len(resistant)!=3:
        raise ValueError("Resistant samples must remain excluded")
    validate_prior_target(target_ids,target_truth)
    raw_target=s17.read_target_tar(s17.TARGET)
    if set(raw_target)!=(set(target_ids)|resistant):
        raise ValueError("Stage17 target archive sample set mismatch")
    targets={gsm:raw_target[gsm]["data"] for gsm in target_ids}
    if any(len(t)<15000 for t in targets.values()):
        raise ValueError("GSE255958 samples have insufficient RSEM genes")

    prediction_rows=[];coverage_rows=[]; results={};shift={}
    for method in METHODS:
        m=models[method]
        Z,cover=target_rank_features(targets,target_ids,m)
        margin=np.asarray(m["classifier"].decision_function(Z),dtype=float)
        labels=[target_truth[gsm] for gsm in target_ids]
        met=target_metrics(labels,margin)
        results[method]=met
        for row in cover:
            coverage_rows.append({"method":method,**row})
        for gsm,label,s,co in zip(target_ids,labels,margin,cover):
            prediction_rows.append({"method":method,"gsm":gsm,"true_state":label,
                                    "source_only_margin":f"{s:.12g}",
                                    "predicted_dtp":int(s>THRESHOLD),
                                    "correct":int((s>THRESHOLD)==(label=="DTP")),
                                    "panel_genes_available":co["available"],
                                    "panel_genes_imputed_source_mean":co["missing"]})
        sr={state:[r["source_margin"] for r in source_rows if
                   r["method"]==method and r["true_state"]==state] for state in STATES}
        tr={state:margin[[i for i,x in enumerate(labels) if x==state]] for state in STATES}
        shift[method]={state:{"source_PC9_median_margin":float(np.median(sr[state])),
                              "external_PC9_median_margin":float(np.median(tr[state])),
                              "external_minus_source_median":float(np.median(tr[state])-np.median(sr[state]))}
                       for state in STATES}
    paired = {}
    a = {r["gsm"]:r for r in prediction_rows if r["method"]==METHODS[0]}
    b = {r["gsm"]:r for r in prediction_rows if r["method"]==METHODS[1]}
    if set(a)!=set(target_ids) or set(b)!=set(target_ids):
        raise ValueError("External prediction identities differ between methods")
    paired = {
        "both_correct":sum(a[g]["correct"] and b[g]["correct"] for g in target_ids),
        "acute_only_correct":sum(a[g]["correct"] and not b[g]["correct"] for g in target_ids),
        "dual_only_correct":sum(not a[g]["correct"] and b[g]["correct"] for g in target_ids),
        "both_wrong":sum(not a[g]["correct"] and not b[g]["correct"] for g in target_ids),
    }
    if sum(paired.values()) != 9:
        raise AssertionError("Paired sample counts inconsistent")
    source_pc9_recall = {}
    for method in METHODS:
        matched = [r for r in source_rows if r["method"] == method]
        if len(matched)!=9:
            raise AssertionError("Missing source-PC9 reference scores")
        source_pc9_recall[method] = {
            "source_pc9_n":9,
            "source_pc9_correct_at_zero":sum((r["source_margin"]>THRESHOLD)==(r["true_state"]=="DTP") for r in matched),
        }
    report={
      "status":"EXPLORATORY_HISTORICALLY_INSPECTED_TARGET_NOT_CONFIRMATORY",
      "source":"GSE193258","target":TARGET_GSE,
      "source_training_lines":list(SOURCE_LINES),"source_train_n":27,
      "source_pc9_heldout_n":9,"target_PC9_n":9,
      "target_state_count":{"Control":3,"Acute":3,"DTP":3},
      "model_task":"binary DTP vs (Control+Acute), NOT three-class classification",
      "methods":list(METHODS),"feature_k":SELECTED_K,
      "C":C,"threshold":THRESHOLD,
      "target_missing_gene_policy":{
          "min_present":MIN_COVERAGE,
          "representation":"rank observed panel genes / observed panel count; absent positions TRAIN source mean rank",
          "warning":"Missing gene adaptation is not exact 100-gene feature equivalence"
      },
      "source_pc9_predictions_exactly_reproduced":True,
      "target_used_for_parameter_or_panel_selection":False,
      "target_used_for_threshold_selection":False,
      "resistant_samples_excluded":True,
      "source_provenance":source_provenance,
      "input_sha256":{
          "target_archive":sha(s17.TARGET),
          "stage17_source_pc9_excluded_summary":sha(STAGE17_SUMMARY),
          "stage28_predictions":sha(SOURCE_ROWS),
          "stage28_selected_genes":sha(SOURCE_GENES_ACUTE),
          "stage27_selected_genes":sha(SOURCE_GENES_DUAL)
      },
      "target_coverage":coverage_rows,"metrics":results,"score_shift_descriptive":shift,
      "paired_accuracy_counts":paired,"source_pc9_replayed_metrics":source_pc9_recall,
      "limitations":[
          "GSE255958 was historically inspected in earlier stages: this is retrospective, not independent held-out validation.",
          "Only nine external PC9 samples (3 per condition), not biological-proof DTP specificity.",
          "Source Acute 24h vs DTP 21d; target targeted-therapy exposure durations and study protocols differ.",
          "Feature rank method cannot deconfound treatment duration or reliably infer clinical response.",
          "Sample-to-sample dependencies and batch effects require source study review.",
          "Missing selected genes, if any, are imputed using frozen source mean ranks with no target tuning.",
          "Do not posthoc select winner, tune threshold, gene mapping, or K based on these results."
      ]}
    md=["# Stage 31 — Frozen DTP-contrast cross-study stress test", "",
        "**Exploratory only: previously inspected GSE255958. No independent validation or novel algorithm claim.**", "",
        "Source fit: GSE193258 H1975/HCC2935/HCC827 (n=27); target: GSE255958 PC9 (n=9).",
        "Binary output: DTP vs two named negative conditions (Control and Acute).", "",
        "| Method | Correct/9 | Binary bACC | 3-condition macro recall | Control false DTP | Acute false DTP | DTP recall | Binary AUC |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        r=results[method]
        md.append(f"| {method} | {r['correct']}/9 | {r['binary_balanced_accuracy']:.3f} | "
                  f"{r['three_condition_macro_recall']:.3f} | {r['control_false_dtp']}/3 | "
                  f"{r['acute_false_dtp']}/3 | {r['dtp_correct']}/3 | {r['binary_auc']:.3f} |")
    md += ["", "## Selected-gene availability", "",
        "Predeclared minimum coverage: >=90 of 100 gene IDs for every sample and method.",
        "Missing IDs (if any) are imputed in rank space using training-source means; not an exact panel transfer.", "",
        "| Method | Minimum coverage | Maximum coverage | Missingness adaptation used |", "|---|---:|---:|---|"]
    for method in METHODS:
        cov=[r["available"] for r in coverage_rows if r["method"]==method]
        md.append(f"| {method} | {min(cov)}/100 | {max(cov)}/100 | {'yes' if min(cov)<100 else 'no'} |")
    md += ["", "## Paired external predictions (same nine GSM samples)", "",
        f"Both correct: {paired['both_correct']}/9; Acute-only correct: {paired['acute_only_correct']}/9; "
        f"Dual-only correct: {paired['dual_only_correct']}/9; both wrong: {paired['both_wrong']}/9.", "",
        "## Held-out source PC9 replay", "",
        "| Method | Source PC9 correct/9 |",
        "|---|---:|"]
    for method in METHODS:
        md.append(f"| {method} | {source_pc9_recall[method]['source_pc9_correct_at_zero']}/9 |")
    md += ["", "## Score-shift diagnostic (medians; not calibration)", "",
           "| Model | Group | Source-PC9 median | External-PC9 median | Difference |",
           "|---|---|---:|---:|---:|"]
    for method in METHODS:
        for state in STATES:
            p=shift[method][state]
            md.append(f"| {method} | {state} | {p['source_PC9_median_margin']:.4f} | "
                      f"{p['external_PC9_median_margin']:.4f} | {p['external_minus_source_median']:+.4f} |")
    md += ["", "## How to interpret", "",
        "- These are *binary DTP-like calls* for two named negatives; not a three-class predictor.",
        "- The target was previously examined. Results are hypothesis-generating only.",
        "- Score shifts use 9 source-PC9 vs 9 external-PC9 samples, both obtained from models trained without source PC9.",
        "- Clinical prediction, causal DTP markers, and duration-independent state inference are **not** supported.",
        "- Compare both models; do not tune or choose winner using this 9-sample target.", ""]
    output={"31_report.md":"\n".join(md),
            "31_summary.json":json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
            "31_predictions.csv":dump_csv(prediction_rows),
            "31_gene_coverage.csv":dump_csv(coverage_rows)}
    write_immutable(output)
    print("Stage31 completed: GSE255958 PC9 source-excluded target scoring; frozen Stage28 source margins VERIFIED")
    for method in METHODS:
        r=results[method]
        nums=[x["available"] for x in coverage_rows if x["method"]==method]
        print(f"{method}: correct={r['correct']}/9 binary_bACC={r['binary_balanced_accuracy']:.3f} "
              f"three_cond_macro={r['three_condition_macro_recall']:.3f} "
              f"Control_false_DTP={r['control_false_dtp']}/3 Acute_false_DTP={r['acute_false_dtp']}/3 "
              f"DTP_recall={r['dtp_correct']}/3 genes={min(nums)}-{max(nums)}/100")
    print("Saved outputs/cross_study_contrast/31_{report.md,summary.json,predictions.csv,gene_coverage.csv}")
    print("EXPLORATORY ONLY: historically inspected target, no algorithm novelty or biological DTP specificity established")


def self_test():
    import numpy as np
    from scipy.stats import rankdata
    # Deterministic, source-only feature transformation; monotonic invariance.
    genes=[f"G{i}" for i in range(100)]
    values=np.linspace(1, 200, 100)
    model={"panel":genes,"train_feature_mean":np.full(100,.51)}
    z,coverage=target_rank_features({"T":dict(zip(genes,values))},["T"],model)
    z2,_=target_rank_features({"T":dict(zip(genes,np.log1p(values)))},["T"],model)
    assert z.shape==(1,100) and np.array_equal(z,z2) and coverage[0]["missing"]==0
    missing={g:float(i) for i,g in enumerate(genes[:91])}
    z,co=target_rank_features({"T":missing},["T"],model)
    assert co[0]["available"]==91 and np.all(z[0,91:]==.51)
    assert np.array_equal(z[0,:91],rankdata(list(missing.values()))/91)
    try: target_rank_features({"T":dict(zip(genes[:89],range(89)))},["T"],model)
    except ValueError: pass
    else: raise AssertionError("Insufficient gene coverage was not rejected")
    try: target_metrics(["Control"]*3+["Acute"]*3+["DTP"]*3,[0.0]*8+[float('nan')])
    except ValueError: pass
    else: raise AssertionError("NaN score was not rejected")
    y=STATES*3
    margins=np.array([-2.,-1.,2.]*3)
    score=target_metrics(y,margins)
    assert score["correct"]==9 and score["binary_balanced_accuracy"]==1
    assert score["three_condition_macro_recall"]==1 and score["binary_auc"]==1
    assert MIN_COVERAGE==90 and C==.01 and THRESHOLD==0 and len(METHODS)==2
    print("SELF-TEST PASSED: rank invariance, source-mean missingness, coverage fail-closed, no target-label feature transform, state metrics")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--exploratory-only",action="store_true")
    a=p.parse_args()
    if a.self_test: self_test()
    elif a.exploratory_only: run()
    else: p.error("Real scoring requires --exploratory-only")

if __name__ == "__main__": main()
