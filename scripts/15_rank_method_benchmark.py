"""15 | Fixed rank-method comparators on source LOCO and already-inspected GEO cohorts.

Scientific status: EXPLORATORY DEVELOPMENT. This code adds two transparent,
training-only rank-method controls; neither is claimed to reproduce the official
Bioconductor singscore or an optimized k-TSP implementation.

Methods:
  selected_feature_percentile: exact stage-13 frozen 100-gene LR comparator.
  signed_rank_full: singscore-*inspired* signed mean-rank signature of 100 source
      selected genes, with one-dimensional source-only LogisticRegression.
  signed_rank_selected: the same score from ranks within the 100-gene panel.
  top_pairs_11: fixed 11 disjoint top-scoring relative-expression pairs, from
      the same 100 source-selected genes; source-trained majority voting.

No selection, fitting, tuning or calibration on external labels or expression.
The target measurements are used ONLY to produce predictions. At analysis time,
labels are read to score those predictions. Source folds and targets were all
historically inspected: no independent validation, no novelty/superiority claim.

Reuses locally available scripts 07, 11, 13, optionally 14, and their existing
files. No new dependencies. Does not modify upstream files or retrain stage 13.

Run from project root:
    python scripts/15_rank_method_benchmark.py --self-test
    python scripts/15_rank_method_benchmark.py --exploratory-only

Outputs (gitignored): outputs/rank_methods/15_summary.json,
    15_metrics.csv, 15_predictions.csv

Reference algorithms:
  Foroutan et al., BMC Bioinformatics 2018, doi:10.1186/s12859-018-2435-4
  Geman et al., Stat Appl Genet Mol Biol 2004, doi:10.2202/1544-6115.1071
  MetaKTSP, Bioinformatics 2016, doi:10.1093/bioinformatics/btw186
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from math import ceil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
META = ROOT / "data/metadata"
OUT = ROOT / "outputs/rank_methods"
PROTOCOL = META / "evaluation_protocol.json"
STAGE13_JSON = ROOT / "outputs/anchor_transfer/13_summary.json"
STAGE13_PRED = ROOT / "outputs/anchor_transfer/13_predictions.csv"
STAGE14_JSON = ROOT / "outputs/gse285632_audit/14_report.json"
STAGE14_PRED = ROOT / "outputs/gse285632_audit/14_predictions.csv"
GSE285632 = ROOT / "data/raw/GSE285632/GSE285632_All.DEG.Expression.xlsx"
TARGETS = ("GSE153183", "GSE303413", "GSE341996")
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
METHODS = (
    "selected_feature_percentile", "signed_rank_full",
    "signed_rank_selected", "top_pairs_11",
)
K = 100
N_PAIRS = 11
C = 0.01
SEED = 42


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    if not records:
        raise ValueError("No report rows")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)



def rank_selected(values, np, rankdata):
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Rank representation needs >=2 measured genes")
    return rankdata(values, axis=1, method="average") / values.shape[1]


def source_selection(X, y, stage11, np, rankdata, f_classif):
    selected, refs, allrank = stage11.select_source_features_and_anchors(
        X, y, rankdata, np, f_classif, k=K, anchors=256)
    return selected, allrank


def signature_direction(selected_ranks, y, np):
    """Learn sign of each selected gene from source TRAIN labels only."""
    delta = selected_ranks[y == 1].mean(0) - selected_ranks[y == 0].mean(0)
    up = delta > 0
    down = delta < 0
    if not up.any() or not down.any():
        raise ValueError("No bidirectional source-only signature")
    # Exclude ties (delta == 0), never resolve with a target label.
    return up, down


def signature_score(feature_ranks, up, down, available, np):
    """Difference of mean directed ranks; not the official singscore package."""
    if feature_ranks.shape[1] != len(up) or len(up) != len(available):
        raise ValueError("Bad feature/rank mask size")
    pos, neg = up & available, down & available
    if not pos.any() or not neg.any():
        raise ValueError("Insufficient observed up/down signature genes")
    return (feature_ranks[:, pos].mean(axis=1) -
            feature_ranks[:, neg].mean(axis=1)).reshape(-1, 1)


def pair_rules(raw100, y, np, n_pairs=N_PAIRS):
    """Greedy fixed-k disjoint TSP-style votes, no target-driven optimization."""
    m = raw100.shape[1]
    if m < 2*n_pairs or set(np.unique(y)) != {0, 1}:
        raise ValueError("Insufficient source genes or missing class")
    # Pair ordering is invariant to any strictly increasing per-sample transform.
    i, j = np.triu_indices(m, k=1)
    b = np.where(raw100[:, i] == raw100[:, j], .5,
                 (raw100[:, i] > raw100[:, j]).astype(float))
    p0 = b[y == 0].mean(axis=0)
    p1 = b[y == 1].mean(axis=0)
    delta = p1 - p0
    # Deterministic tie breaks: index order after score.
    order = np.lexsort((j, i, -np.abs(delta)))
    chosen = []
    used = set()
    for idx in order:
        a, c = int(i[idx]), int(j[idx])
        if a in used or c in used or abs(delta[idx]) < 1e-12:
            continue
        chosen.append((a, c, 1 if delta[idx] > 0 else -1, float(abs(delta[idx]))))
        used.add(a)
        used.add(c)
        if len(chosen) == n_pairs:
            break
    if len(chosen) < n_pairs:
        raise ValueError(f"Only {len(chosen)} discriminating disjoint pairs; require {n_pairs}")
    return chosen


def pair_predict(observed100, available, rules, np):
    """Vote from available source-chosen pairs; zero available pairs stops."""
    if len(available) != observed100.shape[1]:
        raise ValueError("Feature availability mismatch")
    vote = np.zeros(len(observed100), dtype=float)
    included = 0
    for a, b, sign, _ in rules:
        if not (available[a] and available[b]):
            continue
        order = observed100[:, a] > observed100[:, b]
        equal = observed100[:, a] == observed100[:, b]
        # Tied expression contributes a neutral vote, not an invented direction.
        supports_dtp = order if sign > 0 else ~order
        vote += np.where(equal, .5, supports_dtp.astype(float))
        included += 1
    if included == 0:
        raise ValueError("No source-trained TSP pairs measured in target")
    # Conservative deterministic tie break: Control when votes == 1/2.
    return (vote > (included/2)).astype(int), included


def train_models(Xtr, ytr, stage11, np, rankdata, f_classif,
                 LogisticRegression, StandardScaler):
    from sklearn.pipeline import make_pipeline
    selected, allr = source_selection(Xtr, ytr, stage11, np, rankdata, f_classif)
    raw = Xtr[:, selected]
    fullr = allr[:, selected]
    localr = rank_selected(raw, np, rankdata)
    up, down = signature_direction(localr, ytr, np)
    present = np.ones(len(selected), dtype=bool)
    signatures = {
        "signed_rank_full": signature_score(fullr, up, down, present, np),
        "signed_rank_selected": signature_score(localr, up, down, present, np),
    }
    fits = {}
    for name, mat in signatures.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, max_iter=2000, solver="liblinear", random_state=SEED))
        clf.fit(mat, ytr)
        fits[name] = clf
    pairset = pair_rules(raw, ytr, np)
    # Stage-13's trained comparison is created with stage-13 itself, separately.
    return {"selected":selected, "up":up, "down":down,
            "signatures":fits, "pair_rules":pairset}


def target_source_representations(Xtarget, target_genes, selected_names,
                                  selected_up, selected_down, np, rankdata):
    if len(set(target_genes)) != len(target_genes):
        raise ValueError("Target has duplicate gene names")
    if Xtarget.ndim != 2 or Xtarget.shape[1] != len(target_genes):
        raise ValueError("Target shape inconsistent with genes")
    idx = {g:k for k,g in enumerate(target_genes)}
    available = np.array([g in idx for g in selected_names], dtype=bool)
    n = len(selected_names)
    if available.sum() < ceil(0.90*n):
        raise ValueError(f"Only {available.sum()} of {n} source-selected genes available")
    raw = np.zeros((len(Xtarget), n), dtype=float)
    cols = [idx[g] for g in selected_names if g in idx]
    raw[:, available] = Xtarget[:, cols]
    # Full-gene ranks are calculated without using test labels; denominator
    # differs from source, intentionally probing transfer sensitivity.
    fullr = np.zeros_like(raw)
    fullr[:, available] = rank_selected(Xtarget, np, rankdata)[:, cols]
    localr = np.zeros_like(raw)
    localr[:, available] = rank_selected(Xtarget[:, cols], np, rankdata)
    s_full = signature_score(fullr, selected_up, selected_down, available, np)
    s_local = signature_score(localr, selected_up, selected_down, available, np)
    return raw, available, s_full, s_local


def metrics(y, pred, np):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if len(y) != len(pred) or Counter(y) != {0:len(y)//2, 1:len(y)//2}:
        raise ValueError("Evaluation must have equally represented binary groups")
    control = float(np.mean(pred[y==0]==0))
    dtp = float(np.mean(pred[y==1]==1))
    return {"n":len(y),"balanced_accuracy":(control+dtp)/2,
            "control_recall":control,"dtp_recall":dtp,
            "correct":int(np.sum(y==pred))}


def benchmark_group(Xtr, ytr, Xtest, ytest, source_genes, target_genes,
                    target_ids, group_name, stage11, stage13, np, rankdata,
                    f_classif, LogisticRegression, StandardScaler,
                    frozen_expected_selected=None):
    fit = train_models(Xtr,ytr,stage11,np,rankdata,f_classif,
                       LogisticRegression,StandardScaler)
    selected = fit["selected"]
    chosen_names = [source_genes[int(k)] for k in selected]
    if frozen_expected_selected is not None and chosen_names != frozen_expected_selected:
        raise ValueError("Source feature selection changed versus locked stage 13")
    # For consistency with stage 13 selected-only LR, reuse the exact stage-13
    # training and target-transform functions rather than independently re-code it.
    refs = stage11.select_source_features_and_anchors(
        Xtr,ytr,rankdata,np,f_classif,k=K,anchors=256)[1]
    anchors = [source_genes[int(k)] for k in refs]
    allrank = rank_selected(Xtr, np, rankdata)
    source_models = stage13.fit_source(Xtr,ytr,selected,refs,allrank,stage11,np,
                                      LogisticRegression,StandardScaler)
    base = source_models["selected_feature_percentile"]
    z,base_coverage = stage13.target_transform(Xtest,target_genes,chosen_names,
        anchors,base["train_mean"],"selected_feature_percentile",rankdata,np)
    preds = {"selected_feature_percentile":base["model"].predict(z)}
    raw, present, full, local = target_source_representations(
        Xtest,target_genes,chosen_names,fit["up"],fit["down"],np,rankdata)
    preds["signed_rank_full"] = fit["signatures"]["signed_rank_full"].predict(full)
    preds["signed_rank_selected"] = fit["signatures"]["signed_rank_selected"].predict(local)
    preds["top_pairs_11"], measured_pairs = pair_predict(raw,present,fit["pair_rules"],np)
    yy = np.asarray(ytest,dtype=int)
    if len(target_ids) != len(yy):
        raise ValueError("Target IDs/label count mismatch")
    rows=[]; details=[]
    for method in METHODS:
        p = preds[method]
        rows.append({"scope":group_name,"method":method,**metrics(yy,p,np),
                     "features_present":int(present.sum()),
                     "pairs_present":measured_pairs if method=="top_pairs_11" else ""})
        for sid,t,v in zip(target_ids,yy,p):
            details.append({"scope":group_name,"sample_id":sid,"method":method,
                            "candidate_state":"DTP" if t else "Control",
                            "prediction":"DTP" if v else "Control",
                            "correct":int(t==v)})
    return rows,details,chosen_names,fit["pair_rules"]


def check_prior_predictions(rows, predictions, method_name="selected_feature_percentile"):
    actual={(r["scope"],r["sample_id"]):r["prediction"]
            for r in predictions if r["method"]==method_name}
    expected={(r.get("study"),r.get("gsm",r.get("excel_column"))):r["prediction"]
              for r in rows if r["method"]==method_name}
    if actual != expected:
        example=[(k,actual.get(k),expected.get(k)) for k in (set(actual)|set(expected))
                 if actual.get(k)!=expected.get(k)]
        raise ValueError(f"STOP: stage-13/14 frozen predictions changed: {example[:6]}")


def self_test():
    import numpy as np
    from scipy.stats import rankdata
    # Invariance under monotone transform and independence from background genes.
    rng=np.random.default_rng(15)
    X=rng.normal(size=(18,40))
    y=np.array([0]*9+[1]*9)
    X[y==1,1:15] += 4
    rules=pair_rules(X,y,np,n_pairs=5)
    vote,avail=pair_predict(X,np.ones(40,dtype=bool),rules,np)
    vote2,_=pair_predict(np.exp(X),np.ones(40,dtype=bool),rules,np)
    assert np.array_equal(vote,vote2) and avail==5
    up=np.array([True]*20+[False]*20)
    down=~up
    learned_up, learned_down = signature_direction(rank_selected(X, np, rankdata),y,np)
    assert learned_up.any() and learned_down.any()
    assert not np.any(learned_up & learned_down)
    ranks=rank_selected(X,np,rankdata)
    score=signature_score(ranks,up,down,np.ones(40,dtype=bool),np)
    assert np.allclose(score,signature_score(rank_selected(np.exp(X),np,rankdata),up,down,np.ones(40,dtype=bool),np))
    # Missing genes must never be treated as artificial zeros for rank scoring.
    reduced=[f"G{i}" for i in range(40) if i!=5]
    mats,mask,full,local=target_source_representations(
        X[:,[i for i in range(40) if i!=5]],reduced,
        [f"G{i}" for i in range(40)],up,down,np,rankdata)
    assert int(mask.sum())==39 and full.shape==(18,1) and local.shape==(18,1)
    try:
        pair_predict(np.ones((2,40)),np.zeros(40,dtype=bool),rules,np)
        raise AssertionError("Expected no-pair failure")
    except ValueError:
        pass
    print("SELF-TEST PASSED (TSP monotone invariance, signed-rank invariance, source-only direction, missing-gene abstention)")


def run():
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    stage13_path=SCRIPTS/"13_cross_study_anchor_benchmark.py"
    # Stage 13 defines the project's existing, tested dynamic-module loader.
    import importlib.util
    def load(path, name):
        if not path.is_file():
            raise FileNotFoundError(path)
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    stage13=load(stage13_path,"stage13_for_rank15")
    stage11=load(SCRIPTS/"11_source_only_gene_space_benchmark.py","stage11_for_rank15")
    stage07=load(SCRIPTS/"07_external_stress_test.py","stage07_for_rank15")
    if not STAGE13_JSON.is_file() or not STAGE13_PRED.is_file():
        raise FileNotFoundError("Locked stage-13 reports missing; run 13 first")
    previous=json.loads(STAGE13_JSON.read_text(encoding="utf-8"))
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if previous.get("source_matrix_sha256") != protocol.get("source_matrix_sha256"):
        raise ValueError("Source matrix/protocol mismatch against stage 13")
    if previous.get("protocol_fingerprint") != protocol.get("fingerprint_sha256"):
        raise ValueError("Protocol fingerprint changed")
    if (stage13.K,stage13.C,stage13.SEED)!=(K,C,SEED):
        raise ValueError("Frozen stage-13 parameters changed")
    if protocol.get("flags",{}).get("biology_approved_for_training") is not False:
        raise ValueError("This exploratory tool expects pending biological approval")
    source_path=stage07.safe_local_path(protocol["source_matrix"],"processed")
    genes,ids,X=stage07.load_matrix(source_path,protocol["source_matrix_sha256"],np)
    sample_info=protocol["source_gsm_to_label"]
    if ids != [r["gsm"] for r in sample_info] or X.shape!=(24,19712):
        raise ValueError("Source shape and sample identities changed")
    y=np.array([int(r["state"]=="DTP") for r in sample_info],dtype=int)
    groups=np.array([r["cell_line"] for r in sample_info])
    all_rows=[]; all_predictions=[]
    for line in LINES:
        tr,te=groups!=line,groups==line
        rows,details,_,_=benchmark_group(X[tr],y[tr],X[te],y[te],genes,genes,
            np.array(ids)[te].tolist(),"LOCO:"+line,
            stage11,stage13,np,rankdata,f_classif,LogisticRegression,StandardScaler)
        all_rows.extend(rows);all_predictions.extend(details)
    cleaning=json.loads((META/"cleaning_report.json").read_text(encoding="utf-8"))
    targets=stage07.prepare_targets(np,cleaning,
        stage07.csv_rows(META/"expression_alignment.csv"),
        stage07.csv_rows(META/"cohort_review.csv"),ROOT)
    if set(targets)!=set(TARGETS):
        raise ValueError("Unexpected external cohort set")
    external={}
    for s in TARGETS:
        external[s]=(targets[s]["matrix"],targets[s]["genes"],targets[s]["gsm"],
                     np.array([int(t=="DTP") for t in targets[s]["states"]]))
    opt=(GSE285632.exists(),STAGE14_JSON.exists(),STAGE14_PRED.exists())
    if any(opt) and not all(opt):
        raise FileNotFoundError("GSE285632 file/stage14 report incomplete; must verify all")
    if all(opt):
        stage14=load(SCRIPTS/"14_gse285632_locked_audit.py","stage14_for_rank15")
        g285,x285,qc=stage14.parse_workbook(GSE285632)
        old14=json.loads(STAGE14_JSON.read_text(encoding="utf-8"))
        if qc["input_sha256"] != old14.get("excel_file_qc",{}).get("input_sha256"):
            raise ValueError("GSE285632 workbook changed since stage 14")
        external["GSE285632"]=(x285,g285,list(stage14.COLUMNS[1:]),np.array([0]*3+[1]*3))
    all_pair_rules=[]
    for study,(mat,tgenes,tids,yt) in external.items():
        rows,details,selected,pairs=benchmark_group(
            X,y,mat,yt,genes,tgenes,tids,study,
            stage11,stage13,np,rankdata,f_classif,LogisticRegression,StandardScaler,
            frozen_expected_selected=previous["selected_gene_names"])
        all_rows.extend(rows);all_predictions.extend(details)
        if not all_pair_rules:
            all_pair_rules=[{"gene_a":selected[a],"gene_b":selected[b],
                             "dtp_if_a_gt_b":sign>0,"source_pair_score":score}
                            for a,b,sign,score in pairs]
    check_prior_predictions(read_csv(STAGE13_PRED),
        [r for r in all_predictions if r["scope"] in TARGETS])
    if all(opt):
        check_prior_predictions(read_csv(STAGE14_PRED),
            [r for r in all_predictions if r["scope"]=="GSE285632"])
    summary={}
    for method in METHODS:
        # Keep the previously inspected three-study macro separate from the
        # unverified GSE285632 P/D six-column workbook; do NOT pool them.
        bystudy=[r for r in all_rows if r["method"]==method and r["scope"] in TARGETS]
        byfold=[r for r in all_rows if r["method"]==method and r["scope"].startswith("LOCO:")]
        summary[method]={"source_loco_bacc":float(np.mean([r["balanced_accuracy"] for r in byfold])),
            "three_study_macro_bacc":float(np.mean([r["balanced_accuracy"] for r in bystudy])),
            "three_study_macro_dtp_recall":float(np.mean([r["dtp_recall"] for r in bystudy])),
            "worst_three_study_bacc":float(min(r["balanced_accuracy"] for r in bystudy)),
            "three_study_correct":sum(r["correct"] for r in bystudy),
            "three_study_n":sum(r["n"] for r in bystudy)}
        if "GSE285632" in external:
            q=next(r for r in all_rows if r["method"]==method and r["scope"]=="GSE285632")
            summary[method]["gse285632_provisional_bacc"]=q["balanced_accuracy"]
            summary[method]["gse285632_provisional_correct"]=q["correct"]
    report={"status":"EXPLORATORY_HISTORICALLY_INSPECTED_COHORTS_NOT_INDEPENDENT",
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "source_sha256":protocol["source_matrix_sha256"],
            "stage13_report_sha256":digest(STAGE13_JSON),
            "stage13_code_sha256":digest(stage13_path),
            "target_names":list(external),"primary_development_studies":list(TARGETS),"gse285632_is_provisional_separate_study":("GSE285632" in external),"parameters":{"features":K,"pairs":N_PAIRS,"C":C,"seed":SEED},
            "method_descriptions":{
                "selected_feature_percentile":"Exact existing stage-13 100-gene LR model; reproduced and checked per sample",
                "signed_rank_full":"Singscore-inspired signed mean-rank signature of source-selected up/down genes; ranks over full measured gene universe; 1D source-only logistic classifier, NOT official singscore",
                "signed_rank_selected":"Same source-trained signed mean-rank signature using within-panel 100-gene ranks; 1D source-only logistic classifier, NOT official singscore",
                "top_pairs_11":"11 disjoint training-only high-discrimination ordering pairs from same 100 genes; majority vote with abstention on missing pairs; fixed-k TSP-style, NOT an optimized k-TSP package"},
            "summary":summary,"per_study_and_fold":all_rows,"source_selected_pair_rules":all_pair_rules,
            "limitations":["External study labels, source folds and GSE285632 P/D grouping have all been inspected during development; no independent validation claim.",
                           "GSE285632 P/D-to-GSM mapping, expression units and DEG preselection remain unverified.",
                           "Different target gene measurement scales and annotations may change rankings; exact-name symbols alone do not harmonize biology.",
                           "Control/DTP is confounded with drug exposure duration and protocol.",
                           "Method choices were made after inspecting these targets: exploratory method comparison only.",
                           "Signed-rank and TSP methods are simple Python proxies, not reproductions of official packages or proof of new algorithm superiority.",
                           "Methods differ in model capacity (100-dimensional LR vs 1-dimensional signature LR vs 11-pair voting); this benchmarks practical alternatives, not a causal ablation of representation alone."]}
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/"15_metrics.csv",all_rows)
    write_csv(OUT/"15_predictions.csv",all_predictions)
    (OUT/"15_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Stage 15: source-only fixed-method exploratory benchmark; per-sample frozen stage-13/14 predictions REPRODUCED")
    for method in METHODS:
        s=summary[method]
        print(f"{method}: source LOCO={s['source_loco_bacc']:.3f}; "
              f"three-study macro={s['three_study_macro_bacc']:.3f}; "
              f"worst={s['worst_three_study_bacc']:.3f}; "
              f"correct={s['three_study_correct']}/{s['three_study_n']}")
    if "GSE285632" in external:
        print("GSE285632 scores are PROVISIONAL and excluded from three-study macro:")
    for s in external:
        print(s+": "+"; ".join(f"{r['method']}={r['balanced_accuracy']:.3f}"
              for r in all_rows if r['scope']==s))
    print("Saved outputs/rank_methods/15_summary.json, 15_metrics.csv, 15_predictions.csv")
    print("EXPLORATORY ONLY; known methods approximated; no new-method novelty or independent validation")


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
        p.error("Real cohorts require --exploratory-only")


if __name__=="__main__":
    main()
