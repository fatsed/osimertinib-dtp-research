"""Stage 26: nested source-only threshold challenge for Acute-vs-DTP.

Run from repository root:
  python scripts/26_source_only_threshold_challenge.py --self-test
  python scripts/26_source_only_threshold_challenge.py --exploratory-only

For each OUTER leave-one-cell-line-out (LOCO) fold:
  * Retain 3 training cell lines (18 observations; balanced Acute and DTP).
  * Obtain out-of-fold margins by INNER LOCO on these 3 lines (12 fit/6 validate).
  * Select ONE scalar decision threshold: maximize worst inner-line balanced
    accuracy; tie-break by mean inner-line balanced accuracy, then distance
    to zero, then threshold value. Predefined once, never look at outer labels.
  * Fit model once on all 18 outer-training observations, compare fixed zero
    with inner-derived threshold on the same 6 outer-held-out observations.

Two EXACT Stage-16 model recipes (no model-parameter tuning):
  - signed_rank_full: Stage-15 source-only 100-feature signed-rank signature.
  - raw_log2tpm_lr500: source-only f_classif 500-feature raw LR.

IMPORTANT: This is a post-hoc, previously inspected source cohort.  Results
cannot establish new method superiority or duration-independent DTP biology.
Do not tune the choice of threshold rule or select a model using outer results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PREVIOUS = ROOT / "outputs" / "acute_challenge"
OUT = ROOT / "outputs" / "threshold_challenge"
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
METHODS = ("signed_rank_full", "raw_log2tpm_lr500")
SELECTION_RULE = "maximize_min_inner_line_bacc_then_mean_then_closest_to_zero"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import upstream file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def balanced(y, pred) -> dict:
    import numpy as np
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    if len(y) != len(pred) or Counter(y.tolist()) != {0: len(y)//2, 1: len(y)//2}:
        raise ValueError("Expected equally represented binary labels")
    rec_a = float(np.mean(pred[y == 0] == 0))
    rec_d = float(np.mean(pred[y == 1] == 1))
    return {"n": int(len(y)), "correct": int(np.sum(y == pred)),
            "balanced_accuracy": (rec_a + rec_d) / 2,
            "acute_recall": rec_a, "dtp_recall": rec_d,
            "false_dtp_as_acute": int(np.sum((y == 1) & (pred == 0))),
            "false_acute_as_dtp": int(np.sum((y == 0) & (pred == 1)))}


def choose_threshold(margins, labels, held_lines):
    """One fixed deterministic source-only nested-LOCO threshold rule.

    All arguments must come from inner-held-out TRAINING-line predictions.
    This function has no access to outer test observations or labels.
    """
    import numpy as np
    score = np.asarray(margins, dtype=float)
    y = np.asarray(labels, dtype=int)
    line = np.asarray(held_lines)
    if len(score) != 18 or len(y) != 18 or len(line) != 18 or not np.isfinite(score).all():
        raise ValueError("Inner OOF data must contain 18 finite margins")
    names = sorted(set(line.tolist()))
    if len(names) != 3 or any(Counter(y[line == x].tolist()) != {0: 3, 1: 3} for x in names):
        raise ValueError("Inner OOF requires 3 lines x 3 Acute and 3 DTP")

    vals = np.unique(score)
    spread = max(1.0, float(vals[-1] - vals[0]) + 1.0)
    candidates = [0.0, float(vals[0] - spread), float(vals[-1] + spread)]
    candidates += [float((a + b) / 2.0) for a, b in zip(vals[:-1], vals[1:])]
    candidates = sorted(set(candidates))

    records = []
    for t in candidates:
        p = (score > t).astype(int)
        per_line = [balanced(y[line == x], p[line == x])["balanced_accuracy"] for x in names]
        records.append({"threshold": float(t), "worst": float(min(per_line)),
                        "mean": float(sum(per_line)/3), "per_line": dict(zip(names, per_line))})
    best = min(records, key=lambda r: (-r["worst"], -r["mean"],
                                       abs(r["threshold"]), r["threshold"]))
    zero = next(r for r in records if r["threshold"] == 0.0)
    return {"threshold": best["threshold"],
            "inner_worst_bacc": best["worst"], "inner_mean_bacc": best["mean"],
            "inner_line_bacc": best["per_line"],
            "inner_zero_worst_bacc": zero["worst"],
            "inner_zero_mean_bacc": zero["mean"],
            "candidate_count": len(candidates)}


def fit_scores(Xtr, ytr, Xte, genes, upstream):
    """Exactly reuse Stage-15 rank model and Stage-16 raw comparator recipes."""
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    s11, s15 = upstream["s11"], upstream["s15"]
    if (s15.K, s15.C, s15.SEED) != (100, 0.01, 42):
        raise ValueError("Upstream Stage-15 signed-rank recipe has changed")
    if Xtr.shape[1] != 19712 or Xte.shape[1] != 19712 or len(genes) != 19712:
        raise ValueError("Source gene matrix changed")

    fitted = s15.train_models(Xtr, ytr, s11, np, rankdata, f_classif,
                              LogisticRegression, StandardScaler)
    selected = [genes[int(i)] for i in fitted["selected"]]
    _, present, full, _ = s15.target_source_representations(
        Xte, genes, selected, fitted["up"], fitted["down"], np, rankdata)
    if not present.all():
        raise ValueError("Unexpected missing source feature")
    signed = fitted["signatures"]["signed_rank_full"]
    signed_score = np.asarray(signed.decision_function(full), dtype=float).reshape(-1)
    signed_pred = np.asarray(signed.predict(full), dtype=int).reshape(-1)

    raw = make_pipeline(VarianceThreshold(), SelectKBest(f_classif, k=500),
                        StandardScaler(), LogisticRegression(C=.01, solver="liblinear",
                                                              max_iter=2000, random_state=42))
    raw.fit(Xtr, ytr)
    raw_score = np.asarray(raw.decision_function(Xte), dtype=float).reshape(-1)
    raw_pred = np.asarray(raw.predict(Xte), dtype=int).reshape(-1)
    answer = {"signed_rank_full": (signed_score, signed_pred),
              "raw_log2tpm_lr500": (raw_score, raw_pred)}
    for name, (margins, pred) in answer.items():
        if len(margins) != len(Xte) or not np.isfinite(margins).all():
            raise ValueError(f"Nonfinite/unmatched {name} scores")
        if not np.array_equal(pred, (margins > 0).astype(int)):
            raise ValueError(f"Zero threshold does not reproduce {name} model.predict")
    return answer


def load_frozen_source():
    """Reuse Stage-16 audited data mapping; do not rewrite input cohort."""
    import numpy as np

    s16 = load_module(SCRIPTS / "16_acute_specificity_challenge.py", "stage16_for26")
    s7 = load_module(SCRIPTS / "07_external_stress_test.py", "stage07_for26")
    s11 = load_module(SCRIPTS / "11_source_only_gene_space_benchmark.py", "stage11_for26")
    s15 = load_module(SCRIPTS / "15_rank_method_benchmark.py", "stage15_for26")
    p = json.loads((ROOT / "data/metadata/evaluation_protocol.json").read_text(encoding="utf-8"))
    old = json.loads((PREVIOUS / "16_acute_vs_dtp_summary.json").read_text(encoding="utf-8"))
    if (p.get("source_study") != "GSE193258" or old.get("n_samples") != 24 or
        old.get("status") != "NOT_INDEPENDENT_BIOLOGY_PENDING_DURATION_CONFOUNDED" or
        p["flags"].get("biology_approved_for_training") is not False):
        raise ValueError("Unexpected/unapproved source protocol")
    srcpath = s7.safe_local_path(p["source_matrix"], "processed")
    genes, ids, source = s7.load_matrix(srcpath, p["source_matrix_sha256"], np)
    if (ids != [r["gsm"] for r in p["source_gsm_to_label"]] or
        source.shape != (24, 19712) or
        p["source_matrix_sha256"] != old["source_matrix_sha256"]):
        raise ValueError("Frozen source matrix/identity mismatch")
    chosen = s16.verified_acute_columns(p, s16.csv_rows(s16.META / "geo_samples.csv"),
                                        s16.csv_rows(s16.META / "expression_alignment.csv"))
    acute_ids, acute, raw_sha = s16.read_verified_raw(s16.RAW,
        s16.META / "expression_manifest.json", chosen, genes, source, ids, np)
    if raw_sha != old["raw_source_sha256"]:
        raise ValueError("Raw RNA-seq archive hash does not match Stage-16 report")
    src_labels = p["source_gsm_to_label"]
    dtp_ix = [i for i, rec in enumerate(src_labels) if rec["state"] == "DTP"]
    X = np.vstack((acute, source[dtp_ix]))
    y = np.asarray([0] * 12 + [1] * 12, dtype=int)
    gsm = [g for g, _ in acute_ids] + [ids[i] for i in dtp_ix]
    lines = np.asarray([r["cell_line"] for _, r in acute_ids] +
                       [src_labels[i]["cell_line"] for i in dtp_ix])
    if (X.shape != (24, 19712) or len(set(gsm)) != 24 or
        any(Counter(y[lines == l].tolist()) != {0: 3, 1: 3} for l in LINES)):
        raise ValueError("Expected balanced four-line Acute/DTP cohort")
    return genes, gsm, X, y, lines, {"s11": s11, "s15": s15}, old


def verify_frozen(frozen, method, line, gsm, true, scores, predicted):
    for sid, label, score, p in zip(gsm, true, scores, predicted):
        k = (method, line, sid)
        if k not in frozen:
            raise ValueError(f"Missing frozen Stage-16 prediction: {k}")
        rec = frozen[k]
        if rec["true_state"] != ("DTP" if label else "Acute") or \
           rec["predicted_state"] != ("DTP" if p else "Acute") or \
           not math.isclose(float(rec["signed_dtp_margin"]), float(score),
                            rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"Stage-16 model replay mismatch: {k}")


def get_frozen() -> dict:
    records = read_csv(PREVIOUS / "16_acute_vs_dtp_predictions.csv")
    if len(records) != 120:
        raise ValueError("Stage-16 120 frozen predictions are required")
    frozen = {}
    for r in records:
        if r["method"] not in METHODS:
            continue
        key = (r["method"], r["held_out_line"], r["gsm"])
        if key in frozen:
            raise ValueError(f"Duplicate frozen prediction: {key}")
        frozen[key] = r
    if len(frozen) != 48:
        raise ValueError("Expected 2 methods x 24 frozen predictions")
    return frozen


def evaluate(records, method, pred_key, subset=None):
    rows = [r for r in records if r["method"] == method and
            (subset is None or r["held_out_line"] == subset)]
    y = [int(r["true_state"] == "DTP") for r in rows]
    p = [int(r[pred_key] == "DTP") for r in rows]
    return balanced(y, p)


def immutable_write(path: Path, data: str):
    b = data.encode("utf-8")
    if path.exists():
        if path.read_bytes() == b:
            return
        raise FileExistsError(f"Different report already exists; refusing overwrite: {path}")
    path.write_bytes(b)


def run():
    import numpy as np
    from sklearn.metrics import roc_auc_score

    genes, gsm, X, y, groups, upstream, old = load_frozen_source()
    frozen = get_frozen()
    rows = []
    threshold_records = []
    for outer_line in LINES:
        outer_tr = np.flatnonzero(groups != outer_line)
        outer_te = np.flatnonzero(groups == outer_line)
        inner_lines = [l for l in LINES if l != outer_line]
        inner_oof = {name: [] for name in METHODS}
        for inner_line in inner_lines:
            fit = outer_tr[groups[outer_tr] != inner_line]
            val = outer_tr[groups[outer_tr] == inner_line]
            if (len(fit), len(val)) != (12, 6):
                raise ValueError("Nested LOCO split unexpectedly changed")
            predictions = fit_scores(X[fit], y[fit], X[val], genes, upstream)
            for method, (margin, _) in predictions.items():
                for j, ix in enumerate(val):
                    inner_oof[method].append((float(margin[j]), int(y[ix]), str(groups[ix])))
        outer_predictions = fit_scores(X[outer_tr], y[outer_tr], X[outer_te], genes, upstream)
        for method in METHODS:
            inner = inner_oof[method]
            policy = choose_threshold([x[0] for x in inner], [x[1] for x in inner],
                                      [x[2] for x in inner])
            margin, baseline = outer_predictions[method]
            verify_frozen(frozen, method, outer_line,
                          [gsm[i] for i in outer_te], y[outer_te], margin, baseline)
            calibrated = (margin > policy["threshold"]).astype(int)
            threshold_records.append({"method": method, "held_out_line": outer_line, **policy})
            for j, i in enumerate(outer_te):
                rows.append({"method": method, "held_out_line": outer_line,
                             "gsm": gsm[i], "true_state": "DTP" if y[i] else "Acute",
                             "margin": f"{float(margin[j]):.12g}",
                             "zero_threshold_prediction": "DTP" if baseline[j] else "Acute",
                             "inner_selected_threshold": f"{policy['threshold']:.12g}",
                             "nested_threshold_prediction": "DTP" if calibrated[j] else "Acute"})

    if (len(rows) != 48 or len(threshold_records) != 8 or
        len({(r["method"], r["gsm"]) for r in rows}) != 48):
        raise ValueError("Incomplete outer fold predictions")

    results = []
    for method in METHODS:
        for line in (*LINES, "ALL"):
            subset = None if line == "ALL" else line
            r0 = evaluate(rows, method, "zero_threshold_prediction", subset)
            r1 = evaluate(rows, method, "nested_threshold_prediction", subset)
            subset_rows = [r for r in rows if r["method"] == method and
                           (subset is None or r["held_out_line"] == subset)]
            auc = float(roc_auc_score([int(x["true_state"] == "DTP") for x in subset_rows],
                                      [float(x["margin"]) for x in subset_rows]))
            results.append({"method": method, "held_out_line": line,
                            "baseline": r0, "nested_threshold": r1,
                            "roc_auc": auc})

    report = {"status": "EXPLORATORY_PREVIOUSLY_INSPECTED_NO_EXTERNAL_VALIDATION",
              "study": "GSE193258", "classes": {"Acute": "24h", "DTP": "21d"},
              "samples": 24, "held_out_lines": list(LINES),
              "models": list(METHODS), "threshold_selection": SELECTION_RULE,
              "inner_oof_n_per_outer": 18, "inner_training_n_per_fit": 12,
              "outer_training_n": 18, "outer_testing_n": 6,
              "no_outer_labels_used_for_threshold": True,
              "frozen_stage16_csv_sha256": sha256(PREVIOUS / "16_acute_vs_dtp_predictions.csv"),
              "frozen_stage16_source_matrix_sha256": old["source_matrix_sha256"],
              "frozen_stage16_raw_sha256": old["raw_source_sha256"],
              "outer_model_replay_verified": True,
              "thresholds": threshold_records, "metrics": results,
              "limitations": [
                 "Historical source-fold outcomes were already inspected before design of Stage 26.",
                 "No independent dataset; n=6 per held-out line and four cell lines only.",
                 "Acute 24h and DTP 21d are inseparable from exposure duration in this design.",
                 "Inner margin scales arise from different fitted models; pooling may be unstable.",
                 "Threshold selection uses only training-line OOF labels; outer labels evaluate only.",
                 "AUC on the four pooled outer folds is descriptive; each fold uses a different fit.",
                 "No new algorithm, causal, clinical, or cross-study superiority is established."]}
    markdown = ["# Stage 26 — nested source-only threshold challenge", "",
                "**Exploratory only.** Same previously inspected GSE193258; no independent validation.",
                "", "A fixed single threshold is selected on 18 out-of-fold margins from",
                "the three source training cell lines (inner LOCO, two lines used per fit).",
                "Primary rule: maximize worst inner-line balanced accuracy, then mean,",
                "then closest to zero. No outer-test labels enter threshold selection.", "",
                "| Method | Outer line | Threshold | Zero bACC | Nested bACC | AUC |", "|---|---|---:|---:|---:|---:|"]
    for m in METHODS:
        for l in LINES:
            t = next(z for z in threshold_records if z["method"] == m and z["held_out_line"] == l)
            metric = next(z for z in results if z["method"] == m and z["held_out_line"] == l)
            markdown.append(f"| {m} | {l} | {t['threshold']:.6g} | {metric['baseline']['balanced_accuracy']:.3f} | {metric['nested_threshold']['balanced_accuracy']:.3f} | {metric['roc_auc']:.3f} |")
        metric = next(z for z in results if z["method"] == m and z["held_out_line"] == "ALL")
        markdown.append(f"| **{m}** | **ALL** | — | **{metric['baseline']['balanced_accuracy']:.3f}** | **{metric['nested_threshold']['balanced_accuracy']:.3f}** | **{metric['roc_auc']:.3f}** |")
    markdown += ["", "## Interpretation guardrails", "",
                "- Any gain is exploratory, not proof of domain-robust algorithm superiority.",
                "- Thresholds on inner held-out score distributions may fail when test score scale shifts.",
                "- Never choose the final threshold rule after seeing outer-fold results.",
                "- This cohort confounds treatment duration with Acute/DTP state.", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "26_predictions.csv"
    import io
    sio = io.StringIO(newline="")
    fields = list(rows[0])
    w = csv.DictWriter(sio, fieldnames=fields, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    immutable_write(csv_path, sio.getvalue())
    immutable_write(OUT / "26_summary.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    immutable_write(OUT / "26_report.md", "\n".join(markdown))
    print("Stage26 nested source-only threshold challenge complete; Stage16 frozen model predictions verified")
    for m in METHODS:
        overall = next(z for z in results if z["method"] == m and z["held_out_line"] == "ALL")
        print(f"{m}: baseline {overall['baseline']['correct']}/24 bACC={overall['baseline']['balanced_accuracy']:.3f}; "
              f"nested {overall['nested_threshold']['correct']}/24 bACC={overall['nested_threshold']['balanced_accuracy']:.3f}")
    print("Saved outputs/threshold_challenge/26_{report.md,summary.json,predictions.csv}")
    print("EXPLORATORY ONLY; no independent validation or biological DTP-specific claim")


def self_test():
    import numpy as np
    # Three balanced six-sample held-out lines; separated labels with offset.
    vals, labels, lines = [], [], []
    for l, shift in zip(("A", "B", "C"), (0.0, 0.2, 0.6)):
        vals.extend(np.array([-1.0, -0.8, -0.6, 0.5, 0.7, 0.9]) + shift)
        labels.extend([0, 0, 0, 1, 1, 1]); lines.extend([l] * 6)
    x = choose_threshold(vals, labels, lines)
    z = choose_threshold(vals, labels, lines)
    assert x == z and x["inner_worst_bacc"] == 1.0
    assert x["candidate_count"] >= 4 and math.isfinite(x["threshold"])
    assert balanced([0, 0, 1, 1], [0, 1, 0, 1])["balanced_accuracy"] == 0.5
    try:
        choose_threshold(vals[:-1], labels, lines)
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed inner OOF passed")
    try:
        choose_threshold(vals, labels, ["A"] * 18)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing training-group diversity passed")
    # Outer labels are neither an argument to the selector nor to fit_scores.
    # Explicitly verify frozen-model mismatch is detected.
    frozen = {("signed_rank_full", "PC9", "GSM_TEST"):
              {"true_state": "DTP", "predicted_state": "DTP", "signed_dtp_margin": "0.25"}}
    verify_frozen(frozen, "signed_rank_full", "PC9", ["GSM_TEST"], [1], [.25], [1])
    try:
        verify_frozen(frozen, "signed_rank_full", "PC9", ["GSM_TEST"], [1], [-.25], [0])
    except ValueError:
        pass
    else:
        raise AssertionError("Frozen-prediction mismatch went unnoticed")
    print("SELF-TEST PASSED: balanced folds, deterministic minimax threshold, fail-closed guards, frozen replay check")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--exploratory-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not args.exploratory_only:
        parser.error("Stage 26 uses historically inspected data: explicitly pass --exploratory-only")
    run()


if __name__ == "__main__":
    main()
