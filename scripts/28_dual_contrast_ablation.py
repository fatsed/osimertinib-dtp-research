"""Stage 28 | Source-only ablation of Stage27's DTP dual-contrast feature gate.

Copy into osimertinib-dtp-research/scripts/ and run from repository root:
  python scripts/28_dual_contrast_ablation.py --self-test
  python scripts/28_dual_contrast_ablation.py --exploratory-only

Question: Does requiring consistent DTP-vs-BOTH-Control-and-Acute effects
improve DTP-vs-rest generalization compared to either single effect alone?

Frozen comparator: Stage27 ANOVA, raw LR500, and dual-contrast rank LR100.
Two new ablations: (1) DTP-versus-Control direction agreement across the
three training cell lines, (2) DTP-versus-Acute direction agreement. All
retain Stage27's binary DTP-vs-rest ANOVA weighting, 100 selected genes,
within-panel ranks, balanced LR(C=0.01), fixed decision threshold 0, and
four outer leave-one-cell-line-out folds. No target/washout labels or scores
enter selection, fitting, or thresholds.

Stage27's existing predictions and gene panels are exactly replayed and
verified BEFORE writing new files; previous outputs are never overwritten.

Retrospective exploratory analysis: same previously inspected 36 source
samples, duration and treatment confounded, no independent validation.
Ablation results do not prove algorithm novelty or biological DTP specificity.
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
OUTPUT = ROOT / "outputs" / "dtp_specificity"
STATES = ("Control", "Acute", "DTP")
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
ABLATIONS = ("control_only_rank_lr100", "acute_only_rank_lr100")
FROZEN = ("raw_lr500", "rank_anova_lr100", "dual_contrast_rank_lr100")
METHODS = FROZEN + ABLATIONS
K = 100
C = 0.01


def import_local(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen Stage27 file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_content(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("Cannot serialize empty results")
    f = io.StringIO(newline="")
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return f.getvalue()


def select_single_contrast(Xtrain, y3, train_groups, *, negative: int, k: int = K):
    """Return strict per-line sign-consistent feature panel, train-only.

    Same global binary DTP-vs-rest ANOVA weighting as Stage27, but a gate
    requiring only one of the two biological-condition contrasts. The
    dual-contrast rule itself is NOT reimplemented: Stage27 replays it.
    """
    import numpy as np
    from sklearn.feature_selection import f_classif

    if negative not in (0, 1):
        raise ValueError("Negative contrast must be Control(0) or Acute(1)")
    s27 = import_local(SCRIPTS / "27_dtp_specific_contrast_benchmark.py", "stage27_selector_for28")
    lines = s27.validate_train(y3, train_groups)
    y = np.asarray(y3, dtype=int)
    groups = np.asarray(train_groups)
    ranks = s27.rank_all(Xtrain)
    if len(ranks) != len(y) or k < 1 or k > ranks.shape[1]:
        raise ValueError("Wrong training matrix or feature count")
    usable = np.flatnonzero(ranks.var(axis=0) > 1e-12)
    if len(usable) < k:
        raise ValueError("Too few variable genes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f, _ = f_classif(ranks[:, usable], (y == 2).astype(int))
    good = np.isfinite(f) & (f >= 0)
    f = np.where(good, f, -np.inf)

    effects = []
    for line in lines:
        within = groups == line
        dtp = ranks[within & (y == 2)][:, usable].mean(axis=0)
        neg = ranks[within & (y == negative)][:, usable].mean(axis=0)
        effects.append(dtp - neg)
    effects = np.stack(effects)
    up = effects.min(axis=0)
    down = (-effects).min(axis=0)
    direction = np.where(up > 0, 1, np.where(down > 0, -1, 0))
    strict = (direction != 0) & good
    available = int(strict.sum())
    if available < k:
        raise ValueError(f"Only {available} strictly consistent genes for contrast {negative}; need {k}")
    strength = np.maximum(up, down)
    score = np.where(strict, strength * np.log1p(np.maximum(f, 0)), -np.inf)
    ordered = np.lexsort((usable, -score))
    if not np.isfinite(score[ordered[:k]]).all():
        raise ValueError("Insufficient finite selection scores")
    indices = usable[ordered[:k]]
    directions = direction[ordered[:k]]
    if not np.all(np.abs(directions) == 1) or len(set(indices)) != k:
        raise AssertionError("Selector returned an invalid panel")
    return indices, {"eligible_training_genes": available,
                     "selected_up": int((directions == 1).sum()),
                     "selected_down": int((directions == -1).sum())}


def apply_selected_panel(Xtrain, y3_train, Xtest, panel):
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train = rankdata(Xtrain[:, panel], axis=1, method="average") / len(panel)
    test = rankdata(Xtest[:, panel], axis=1, method="average") / len(panel)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=C, solver="liblinear", class_weight="balanced", max_iter=3000,
        random_state=42))
    model.fit(train, (np.asarray(y3_train) == 2).astype(int))
    margin = np.asarray(model.decision_function(test), dtype=float).reshape(-1)
    if not np.isfinite(margin).all() or len(margin) != len(Xtest):
        raise ValueError("Invalid margin output")
    return margin


def verify_stage27(st27, genes, ids, y, groups, frozen, gene_rows, report):
    """Validate frozen protocol/identity without running new training."""
    if (report.get("study") != "GSE193258" or report.get("n_source") != 36 or
        report.get("rank_k") != K or report.get("regularization_C") != C or
        report.get("methods") != list(FROZEN) or report.get("threshold") != 0.0 or
        report.get("fit_uses_only_other_3_lines") is not True):
        raise ValueError("Stage27 protocol is not the frozen comparator")
    if len(ids) != 36 or len(set(ids)) != 36 or len(genes) != 19712 or len(set(genes)) != 19712:
        raise ValueError("Unexpected source gene/sample identities")
    keys = {(m, str(groups[ix]), sid) for m in FROZEN for ix, sid in enumerate(ids)}
    by_key = {}
    for row in frozen:
        key = (row["method"],row["held_out_line"],row["gsm"])
        if key in by_key:
            raise ValueError(f"Duplicate frozen prediction {key}")
        by_key[key] = row
    if set(by_key) != keys or len(by_key) != 108:
        raise ValueError("Stage27 predictions missing/extra sample identities")
    ref_panels = {}
    for row in gene_rows:
        key = (row["method"],row["held_out_line"])
        if key[0] not in FROZEN or key[1] not in LINES:
            raise ValueError("Unknown Stage27 selected-gene row")
        ref_panels.setdefault(key,[]).append(row["gene"])
    if set(ref_panels) != {(m, l) for m in FROZEN for l in LINES}:
        raise ValueError("Stage27 gene panels incomplete")
    for (m,l),g in ref_panels.items():
        required = 500 if m == "raw_lr500" else 100
        if len(g) != required or len(set(g)) != required:
            raise ValueError(f"Bad frozen panel {m}/{l}")
    for ix,sid in enumerate(ids):
        line = str(groups[ix]); true=STATES[int(y[ix])]
        for m in FROZEN:
            r = by_key[(m,line,sid)]
            if r["true_state"] != true or int(r["correct"]) != int(int(r["predicted_dtp"]) == (y[ix] == 2)):
                raise ValueError("Frozen sample label/prediction mismatch")
    return by_key, ref_panels


def score_metrics(stage27, rows):
    import numpy as np
    y = np.array([STATES.index(r["true_state"]) for r in rows], dtype=int)
    p = np.array([int(r["predicted_dtp"]) for r in rows], dtype=int)
    return stage27.state_metrics(y,p)


def evaluate():
    import numpy as np
    stage27 = import_local(SCRIPTS / "27_dtp_specific_contrast_benchmark.py", "stage27_for28")
    stage21 = import_local(SCRIPTS / "21_group_consistent_rank_selection.py", "stage21_for28")
    summary_path = OUTPUT / "27_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Run Stage27 first: outputs/dtp_specificity/27_summary.json")
    ref_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ref_rows = read_csv(OUTPUT / "27_predictions.csv")
    ref_genes = read_csv(OUTPUT / "27_selected_genes.csv")
    genes, ids, X, y, groups, provenance = stage21.load_data()
    y=np.asarray(y,dtype=int);groups=np.asarray(groups)
    if X.shape != (36,19712) or set(groups.tolist()) != set(LINES):
        raise ValueError("Wrong source cohort")
    frozen, ref_panels = verify_stage27(stage27, genes, ids, y, groups, ref_rows, ref_genes, ref_summary)
    if ref_summary.get("source_provenance") != provenance:
        raise ValueError("Stage27 vs Stage28 source provenance mismatch")
    prior16 = json.loads((ROOT/"outputs/acute_challenge/16_acute_vs_dtp_summary.json").read_text(encoding="utf-8"))
    if ref_summary["frozen_source_sha256"] != prior16["source_matrix_sha256"] or ref_summary["frozen_raw_sha256"] != prior16["raw_source_sha256"]:
        raise ValueError("Frozen raw study checksums changed")

    predictions=[]; gene_panels={m:[] for m in METHODS}; panel_details=[]; new_selected_genes=[]
    for line in LINES:
        tr=np.flatnonzero(groups != line);te=np.flatnonzero(groups == line)
        stage27.validate_train(y[tr],groups[tr])
        if len(tr)!=27 or len(te)!=9 or Counter(y[te].tolist()) != {0:3,1:3,2:3}:
            raise ValueError("Invalid outer fold")
        # Critical: replay Stage27 on the exact same source test rows and verify
        # BOTH predictions and selected gene lists, fail closed if divergent.
        original_scores, meta = stage27.fit_predict(X[tr],y[tr],groups[tr],X[te])
        for m in FROZEN:
            expected_panel = set(ref_panels[(m,line)])
            actual_panel = {genes[int(i)] for i in meta[m]["selected"]}
            if actual_panel != expected_panel:
                raise ValueError(f"Stage27 panel changed: {m}/{line}")
            gene_panels[m].append(actual_panel)
            for j,idx in enumerate(te):
                r=frozen[(m,line,ids[idx])]
                margin=float(original_scores[m][j])
                if (int(margin>0) != int(r["predicted_dtp"]) or
                    not np.isclose(margin,float(r["margin"]),rtol=1e-7,atol=1e-8)):
                    raise ValueError(f"Stage27 frozen prediction changed: {m}/{line}/{ids[idx]}")
                predictions.append({"method":m,"held_out_line":line,"gsm":ids[idx],
                    "true_state":STATES[int(y[idx])],"margin":f"{margin:.12g}",
                    "predicted_dtp":int(margin>0),"correct":int((margin>0)==(y[idx]==2))})
        for m,negative in zip(ABLATIONS,(0,1)):
            chosen,info=select_single_contrast(X[tr],y[tr],groups[tr],negative=negative)
            margin=apply_selected_panel(X[tr],y[tr],X[te],chosen)
            gene_names=[genes[int(i)] for i in chosen]
            gene_panels[m].append(set(gene_names))
            panel_details.append({"method":m,"held_out_line":line,"strict_eligible_genes":info["eligible_training_genes"],
               "selected_up":info["selected_up"],"selected_down":info["selected_down"],
               "panel_jaccard_with_dual":len(set(gene_names)&gene_panels["dual_contrast_rank_lr100"][-1]) / len(set(gene_names)|gene_panels["dual_contrast_rank_lr100"][-1])})
            for name in gene_names:
                new_selected_genes.append({"method":m,"held_out_line":line,"gene":name})
            for j,idx in enumerate(te):
                predictions.append({"method":m,"held_out_line":line,"gsm":ids[idx],
                    "true_state":STATES[int(y[idx])],"margin":f"{float(margin[j]):.12g}",
                    "predicted_dtp":int(margin[j]>0),"correct":int((margin[j]>0)==(y[idx]==2))})
    if len(predictions)!=len(METHODS)*36 or len(new_selected_genes)!=len(ABLATIONS)*4*K:
        raise ValueError("Incomplete output coverage")

    metrics=[]; pooled={}; panel_stability={}
    for m in METHODS:
        for line in (*LINES,"ALL"):
            subset=[r for r in predictions if r["method"]==m and (line=="ALL" or r["held_out_line"]==line)]
            if len(subset)!=(36 if line=="ALL" else 9):
                raise ValueError(f"Missing metrics group {m}/{line}")
            z=score_metrics(stage27,subset)
            metrics.append({"method":m,"held_out_line":line,**z})
            if line=="ALL":
                pooled[m]=z
                frozen_total=ref_summary["pooled_metrics"].get(m)
                if frozen_total is not None:
                    for k in ("correct","three_group_macro_recall","control_false_dtp_rate", "acute_false_dtp_rate","dtp_recall"):
                        if not np.isclose(z[k],frozen_total[k],atol=1e-12,rtol=0):
                            raise ValueError(f"Stage27 summary changed: {m} metric {k}")
        sets=gene_panels[m]
        if len(sets)!=4:
            raise ValueError("Incomplete selected gene panels")
        j=[len(a&b)/len(a|b) for a,b in combinations(sets,2)]
        panel_stability[m]={"mean_jaccard":float(np.mean(j)),"selected_n_per_fold":len(sets[0])}
        if m in FROZEN and not np.isclose(panel_stability[m]["mean_jaccard"],
            ref_summary["panel_stability"][m]["mean_pairwise_gene_jaccard"],atol=1e-12,rtol=0):
            raise ValueError("Stage27 stability summary changed")

    predictions_by_method={m:{r["gsm"]:int(r["correct"]) for r in predictions if r["method"]==m} for m in METHODS}
    discordant={}
    for m in ABLATIONS:
        a=predictions_by_method["dual_contrast_rank_lr100"]
        b=predictions_by_method[m]
        if set(a)!=set(b):
            raise ValueError("Cannot pair samples for ablation comparison")
        discordant[m]={"dual_only_correct":sum(a[s]==1 and b[s]==0 for s in a),
                       "single_only_correct":sum(a[s]==0 and b[s]==1 for s in a),
                       "both_correct":sum(a[s]==b[s]==1 for s in a),
                       "both_incorrect":sum(a[s]==b[s]==0 for s in a)}

    report={"status":"EXPLORATORY_RETROSPECTIVE_ABLATION_NOT_INDEPENDENT", "study":"GSE193258",
        "question":"Does strict dual DTP-vs-Control AND DTP-vs-Acute selection outperform each single-contrast gate?",
        "n_source":36,"held_out_lines":list(LINES),"train_per_outer":27,"test_per_outer":9,
        "no_test_labels_for_selection":True,"no_test_samples_for_fit":True,"washout_used":False,
        "rank_k":K,"C":C,"decision_threshold":0.0,"primary_metric":"three_group_macro_recall",
        "feature_scoring":"min strict per-training-line contrast effect * log1p(binary DTP-vs-rest ANOVA F)",
        "comparison_policy":"No posthoc hyperparameter adjustments; Stage27 frozen 3 reference methods reproduced exactly",
        "frozen_27_summary_sha256":sha256(summary_path),
        "frozen_27_predictions_sha256":sha256(OUTPUT/"27_predictions.csv"),
        "frozen_27_genes_sha256":sha256(OUTPUT/"27_selected_genes.csv"),
        "source_provenance":provenance,"metrics":metrics,"pooled":pooled,
        "panel_stability":panel_stability,"ablation_panel_details":panel_details,
        "paired_correctness":discordant,
        "limitations":[
            "All four outer test lines previously inspected during method development; no untouched validation.",
            "Control, Acute and DTP differ in treatment and exposure duration; no causal DTP biology conclusion.",
            "The ablation isolates the strict contrast-selection gate, not all possible feature-selection designs.",
            "Washout remains a descriptive challenge only; no negative labels assumed.",
            "With four cell lines and 36 samples, small differences are uncertain; no significance or algorithm novelty claim."]}

    pretty={m: m.replace("_"," ") for m in METHODS}
    md=["# Stage 28 — Dual-contrast ablation (exploratory)","",
        "**Question:** Does selecting genes consistent in BOTH DTP–Control and DTP–Acute effects improve a fixed DTP-vs-rest rank classifier relative to requiring just one?","",
        "All models are trained on exactly 27 samples from three cell lines per fold, use threshold 0, and are evaluated on the nine samples of the fourth cell line.",
        "Stage27 methods were replayed identically, including per-sample margins and gene panels.","",
        "| Method | Correct/36 | 3-group macro | Control false DTP | Acute false DTP | DTP recall | Mean gene Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        z=pooled[m]
        md.append(f"| {m} | {z['correct']}/36 | {z['three_group_macro_recall']:.3f} | "
                  f"{z['control_false_dtp_rate']:.3f} | {z['acute_false_dtp_rate']:.3f} | "
                  f"{z['dtp_recall']:.3f} | {panel_stability[m]['mean_jaccard']:.3f} |")
    md.extend(["","## Per-held-out line: 3-group macro recall","",
               "| Method | H1975 | HCC2935 | HCC827 | PC9 |","|---|---:|---:|---:|---:|"])
    for m in METHODS:
        sub={z["held_out_line"]:z for z in metrics if z["method"]==m}
        md.append("| "+m+" | "+" | ".join(f"{sub[line]['three_group_macro_recall']:.3f}" for line in LINES)+" |")
    md.extend(["","## Paired error changes: dual versus individual gates","",
               "| Single gate | Dual-only correct | Single-only correct | Both correct | Both incorrect |",
               "|---|---:|---:|---:|---:|"])
    for m,d in discordant.items():
        md.append(f"| {m} | {d['dual_only_correct']} | {d['single_only_correct']} | {d['both_correct']} | {d['both_incorrect']} |")
    md.extend(["","**Interpretation:** Report both positive and negative ablations. A higher score on historically inspected folds is exploratory, not evidence of independent superiority.",
               "Do not tune gene count, weighting, or threshold based on these results. Acute (24h) versus DTP (21d) remains duration-confounded.",""])

    OUTPUT.mkdir(parents=True,exist_ok=True)
    files={"28_predictions.csv":csv_content(predictions),
           "28_selected_genes.csv":csv_content(new_selected_genes),
           "28_summary.json":json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+"\n",
           "28_report.md":"\n".join(md)}
    # One preflight for every output prevents overwriting or partially replacing prior experiments.
    for name,content in files.items():
        p=OUTPUT/name
        if p.exists() and p.read_bytes()!=content.encode("utf-8"):
            raise FileExistsError(f"Output exists and differs; refusing overwrite: {p}")
    for name,content in files.items():
        p=OUTPUT/name
        if not p.exists():
            p.write_bytes(content.encode("utf-8"))
    print("Stage28 complete: Stage27 source predictions, gene panels, metrics EXACTLY REPLAYED")
    for m in METHODS:
        z=pooled[m]
        print(f"{m}: {z['correct']}/36 3-group_macro={z['three_group_macro_recall']:.3f} "
              f"Control_false_DTP={z['control_false_dtp_rate']:.3f} "
              f"Acute_false_DTP={z['acute_false_dtp_rate']:.3f} "
              f"DTP_recall={z['dtp_recall']:.3f} Jaccard={panel_stability[m]['mean_jaccard']:.3f}")
    print("Saved outputs/dtp_specificity/28_{report.md,summary.json,predictions.csv,selected_genes.csv}")
    print("EXPLORATORY: no untouched validation, no time-deconfounding, no algorithm novelty claim")


def self_test():
    import numpy as np
    rng=np.random.default_rng(28)
    groups=np.repeat(["A","B","C"],9)
    y=np.tile(np.repeat([0,1,2],3),3)
    X=rng.normal(3,.2,size=(27,700))
    X[y==2,:175] += 5
    X[y==2,175:350] -= 2
    stage27=import_local(SCRIPTS/"27_dtp_specific_contrast_benchmark.py","stage27_selftest28")
    stage27.validate_train(y,groups)
    dual,_=stage27.select_rank_panel(X,y,groups,mode="dual_contrast",k=K)
    selected=[]
    for negative in (0,1):
        s,info=select_single_contrast(X,y,groups,negative=negative)
        s2,_=select_single_contrast(X,y,groups,negative=negative)
        assert np.array_equal(s,s2) and len(set(s))==K
        assert info["selected_up"]+info["selected_down"]==K
        selected.append(s)
    # A strict dual-consistent eligible gene must be eligible for both
    # single gates. Selection of TOP-100 need not itself be a subset.
    rank=stage27.rank_all(X)
    for j in dual:
        for neg in (0,1):
            effects=[rank[(groups==line)&(y==2),j].mean()-rank[(groups==line)&(y==neg),j].mean()
                     for line in ("A","B","C")]
            assert all(v>0 for v in effects) or all(v<0 for v in effects)
    out=apply_selected_panel(X,y,X[:6],selected[0])
    out2=apply_selected_panel(X,y,X[:6][::-1],selected[0])
    assert np.allclose(out, out2[::-1])
    bad=groups.copy();bad[0]="C"
    try:select_single_contrast(X,y,bad,negative=0)
    except ValueError:pass
    else:raise AssertionError("Broken training group structure passed")
    try:select_single_contrast(X,y,groups,negative=3)
    except ValueError:pass
    else:raise AssertionError("Unknown biological condition passed")
    try:select_single_contrast(np.ones_like(X),y,groups,negative=0)
    except ValueError:pass
    else:raise AssertionError("No-signal matrix passed")
    print("SELF-TEST PASSED: two strict single-contrast gates, deterministic panels, "
          "unchanged binary model, fail-closed training groups, no-target-label design")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--exploratory-only",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test();return
    if not args.exploratory_only:
        p.error("Previously inspected source study; specify --exploratory-only")
    evaluate()


if __name__=="__main__":
    main()
