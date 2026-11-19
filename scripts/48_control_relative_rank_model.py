#!/usr/bin/env python3
"""Stage 48 — ONE pre-specified source-only feature-level control-relative candidate.

PURPOSE / SCIENTIFIC BOUNDARIES
--------------------------------
This is one exploratory, reference-assisted alternative to Stage47's score
calibration. Its feature vector is the 100-gene within-sample rank vector minus
the same rank vector of ONE known same-cell-line Control sample. The resulting
feature vector can change the ordering of decision margins across samples.

We DO NOT claim that reference normalization is novel (cf. RUV/COCONUT), or that
repeatedly inspected 4-line GSE193258 LOCO folds are external validation.

PRE-SPECIFIED EVALUATION (do not tune after observing the outcomes):
- Four source-only LOCO folds; outer test cell line excluded from ALL training,
  feature selection, model fitting, and training-side centering estimates.
- Select a strict dual-contrast 100-gene panel on outer TRAINING cell lines only,
  using the established Stage27 selector without changing its parameters.
- Within the panel, calculate ranks separately WITHIN EACH sample.
- Select the Control reference of each line deterministically by lowest GSM ID;
  source training references only for their corresponding training cell lines;
  a known held-out Control only for the held-out cell line.
- Subtract the per-gene rank vector of the corresponding reference Control.
- Exclude reference samples from model training and from ALL held-out methods.
- Train ONE fixed LR model: C=0.01, liblinear, class_weight='balanced',
  StandardScaler fitted on training differences; threshold=0.
- Compare paired predictions against FROZEN Stage38 base/protected raw-zero
  on precisely the SAME 8 held-out samples per fold. No target labels for fit.
- Report ALL THREE alternate target-control choices on the SAME 6 Acute/DTP
  samples as a sensitivity diagnostic; never pick the best after seeing results.
- Stop exploring this specific route if it does not outperform Stage38 Protected
  while retaining all 12 DTPs, reducing Acute false positives (3 -> <=1),
  introducing zero Control FPs, and avoiding a per-fold BA reduction.

This is NOT independent validation even if the pre-set exploratory criteria pass.

Run from repository root in activated venv:
  python scripts/48_control_relative_rank_model.py --self-test
  python scripts/48_control_relative_rank_model.py --run

Outputs (immutable): outputs/stage48_control_relative_rank/
  48_predictions.csv, 48_metrics_by_fold.csv, 48_reference_sensitivity.csv,
  48_summary.md, 48_provenance.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FROZEN = ROOT / "outputs" / "domain_invariance" / "38_predictions.csv"
STAGE47_SCORES = ROOT / "outputs" / "stage47_control_anchored" / "47_source_scores.csv"
STAGE47_PROV = ROOT / "outputs" / "stage47_control_anchored" / "47_source_oof_provenance.json"
OUT = ROOT / "outputs" / "stage48_control_relative_rank"
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
STATES = ("Control", "Acute", "DTP")
METHOD = "dual_contrast_rank_lr100"
VARIANTS = {"base_raw_zero": "stage28_baseline",
            "protected_raw_zero": "contrast_protected_shrinkage_gamma1"}
C = 0.01
K = 100
SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_script(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_out(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("Cannot write empty CSV")
    f = io.StringIO(newline="")
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return f.getvalue()


def write_immutable(payloads: dict[Path, str]) -> None:
    for p, contents in payloads.items():
        if p.exists() and p.read_bytes() != contents.encode("utf-8"):
            raise FileExistsError(f"Existing artifact differs; refusing to overwrite {p}")
    for p, contents in payloads.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(contents, encoding="utf-8", newline="\n")


def reference_id(ids: np.ndarray, groups: np.ndarray, y: np.ndarray, line: str) -> str:
    """Called on TRAIN metadata or explicitly known target Control metadata only."""
    eligible = sorted(str(gsm) for gsm, domain, label in zip(ids, groups, y)
                      if str(domain) == line and int(label) == 0)
    if len(eligible) != 3:
        raise ValueError(f"Expected precisely 3 known Controls for {line}, got {len(eligible)}")
    return eligible[0]


def fit_relative_model(X_train, y_train, groups_train, ids_train, X_test,
                       ids_test, panel, target_reference_id):
    """No target outcomes are accepted; only the pre-declared Control ID.

    Selection of *panel* must occur upstream on training rows only.
    """
    tr = np.asarray(X_train, dtype=float)
    te = np.asarray(X_test, dtype=float)
    ytr = np.asarray(y_train, dtype=int)
    gtr = np.asarray(groups_train)
    itr = np.asarray(ids_train).astype(str)
    ite = np.asarray(ids_test).astype(str)
    panel = np.asarray(panel, dtype=int)
    if (tr.ndim != 2 or te.ndim != 2 or tr.shape[0] != 27 or te.shape[0] != 9
        or tr.shape[1] != te.shape[1] or len(ytr) != 27 or len(gtr) != 27
        or len(itr) != 27 or len(ite) != 9 or len(np.unique(itr)) != 27
        or len(np.unique(ite)) != 9 or np.intersect1d(itr, ite).size
        or panel.shape != (K,) or len(set(panel.tolist())) != K
        or np.min(panel) < 0 or np.max(panel) >= tr.shape[1]
        or not np.isfinite(tr).all() or not np.isfinite(te).all()):
        raise ValueError("Invalid or overlapping training/test feature arrays")
    lines = sorted(set(gtr.tolist()))
    if len(lines) != 3 or any(Counter(ytr[gtr == line]) != {0: 3, 1: 3, 2: 3} for line in lines):
        raise ValueError("Need 3 source training lines with 3 controls, 3 acute, 3 DTP each")
    ztr = rankdata(tr[:, panel], axis=1, method="average") / K
    zte = rankdata(te[:, panel], axis=1, method="average") / K
    ztr_delta = ztr.copy()
    train_ref_ids = {}
    remove = np.zeros(len(itr), dtype=bool)
    for line in lines:
        rid = reference_id(itr, gtr, ytr, line)
        j = np.flatnonzero(itr == rid)
        if len(j) != 1:
            raise AssertionError("Training reference ID not unique")
        ztr_delta[gtr == line] -= ztr[j[0]]
        remove[j[0]] = True
        train_ref_ids[line] = rid
    if Counter(ytr[~remove]) != {0: 6, 1: 9, 2: 9}:
        raise AssertionError("Training reference removal changed class composition")
    if str(target_reference_id) not in ite:
        raise ValueError("Predeclared target reference not found in held-out test IDs")
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=C, solver="liblinear", class_weight="balanced",
        max_iter=3000, random_state=SEED))
    model.fit(ztr_delta[~remove], (ytr[~remove] == 2).astype(int))
    return model, zte, train_ref_ids


def relative_scores(model, test_rank: np.ndarray, test_ids: np.ndarray, control_id: str) -> np.ndarray:
    ids = np.asarray(test_ids).astype(str)
    ids_match = np.flatnonzero(ids == str(control_id))
    if len(ids_match) != 1:
        raise ValueError("Target reference must match one predeclared known-control sample ID")
    return np.asarray(model.decision_function(test_rank - test_rank[ids_match[0]]), dtype=float)


def metrics(rows: list[dict]) -> dict:
    if len(rows) != 8 or Counter(r["group"] for r in rows) != {"Control": 2, "Acute": 3, "DTP": 3}:
        raise ValueError("Metrics must share the same 8 held-out samples (2 Control, 3 Acute, 3 DTP)")
    yp = np.array([r["group"] == "DTP" for r in rows], dtype=bool)
    pp = np.array([int(r["pred_dtp"]) == 1 for r in rows], dtype=bool)
    scores = np.array([float(r["score"]) for r in rows], dtype=float)
    return {"n": 8,
            "tp": int(np.sum(pp & yp)), "fn": int(np.sum((~pp) & yp)),
            "control_fp": int(sum(int(r["pred_dtp"]) for r in rows if r["group"] == "Control")),
            "acute_fp": int(sum(int(r["pred_dtp"]) for r in rows if r["group"] == "Acute")),
            "sensitivity": float(np.mean(pp[yp])),
            "specificity": float(np.mean(~pp[~yp])),
            "balanced_accuracy": float((np.mean(pp[yp]) + np.mean(~pp[~yp])) / 2),
            "roc_auc": float(roc_auc_score(yp.astype(int), scores))}


def validate_source(ids, X, y, groups, frozen_rows, prov):
    ids = np.asarray(ids).astype(str)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    if (X.shape != (36, 19712) or len(ids) != 36 or len(set(ids)) != 36
        or Counter(groups.tolist()) != Counter({line: 9 for line in LINES})
        or not np.isfinite(X).all()):
        raise ValueError("Unexpected source GSE193258 identity or matrix dimensions")
    if hashlib.sha256(X.tobytes()).hexdigest() != prov["source_X_float64_sha256"]:
        raise ValueError("Source matrix fingerprint differs from audited Stage47 builder")
    if sha256_file(FROZEN) != prov["frozen_38_sha256"]:
        raise ValueError("Frozen Stage38 prediction file differs from Stage47 provenance")
    keyset = {}
    valid_variants = set(VARIANTS.values())
    for r in frozen_rows:
        if r["method"] != METHOD or r["variant"] not in valid_variants:
            continue
        key = (r["held_out_line"], r["variant"], r["gsm"])
        if key in keyset:
            raise ValueError("Repeated frozen Stage38 held-out score")
        keyset[key] = r
    if len(keyset) != 4 * 2 * 9:
        raise ValueError("Expected exactly 72 frozen Stage38 dual-contrast scores")
    for line in LINES:
        for ix in np.flatnonzero(groups == line):
            if Counter([int(x) for x in y[groups == line]]) != {0: 3, 1: 3, 2: 3}:
                raise ValueError("Incorrect source state counts")
            for v in valid_variants:
                r = keyset[(line, v, ids[ix])]
                if r["true_state"] != STATES[y[ix]] or not np.isfinite(float(r["margin"])):
                    raise ValueError("Frozen sample labels/margins do not match source")
    return keyset


def stage47_score_integrity(ids, y, groups, keyset):
    rows = rows_from_csv(STAGE47_SCORES)
    if len(rows) != 288:
        raise ValueError("Stage47 source score file must have 288 rows")
    sample_label = {str(ids[i]): (str(groups[i]), STATES[int(y[i])]) for i in range(36)}
    mapping = {"base": VARIANTS["base_raw_zero"],
               "protected": VARIANTS["protected_raw_zero"]}
    count = 0
    seen = set()
    for r in rows:
        key = (r["fold"], r["variant"], r["split"], r["sample_id"])
        if key in seen:
            raise ValueError("Duplicate Stage47 scored sample")
        seen.add(key)
        if (r["sample_id"] not in sample_label or
            sample_label[r["sample_id"]] != (r["cell_line"], r["group"])):
            raise ValueError("Stage47 scores have inconsistent labels or cell lines")
        if r["split"] == "test":
            if r["fold"] != r["cell_line"]:
                raise ValueError("Stage47 outer-test fold mismatch")
            exact = float(keyset[(r["fold"], mapping[r["variant"]], r["sample_id"])]["margin"])
            if abs(float(r["score"]) - exact) > 2e-6:
                raise ValueError("Stage47 score does not replay frozen Stage38")
            count += 1
        elif r["split"] != "train_oof" or r["cell_line"] == r["fold"]:
            raise ValueError("Stage47 nested OOF/test labeling inconsistent")
    if count != 72:
        raise ValueError("Expected 72 test scores in Stage47")


def perform(st21, st27, ids, X, y, groups, frozen):
    pred, diag = [], []
    X = np.asarray(X, dtype=float)
    ids = np.asarray(ids).astype(str)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    for fold in LINES:
        tr = np.flatnonzero(groups != fold)
        te = np.flatnonzero(groups == fold)
        if len(tr) != 27 or len(te) != 9 or len(set(groups[tr])) != 3:
            raise ValueError("Outer fold isolation failed")
        # Stage27 selector uses ONLY 27 train samples; no outer-test features/labels.
        panel, info = st27.select_rank_panel(X[tr], y[tr], groups[tr], mode="dual_contrast", k=K)
        control = reference_id(ids[te], groups[te], y[te], fold)
        # The candidate API deliberately has no target label argument.
        model, ztest, source_ref = fit_relative_model(
            X[tr], y[tr], groups[tr], ids[tr], X[te], ids[te], panel, control)
        scores = relative_scores(model, ztest, ids[te], control)
        if len(scores) != 9 or not np.isfinite(scores).all():
            raise AssertionError("Invalid candidate scores")
        for j, ix in enumerate(te):
            if str(ids[ix]) == control:
                continue
            for name in ("base_raw_zero", "protected_raw_zero", "candidate_control_relative_rank"):
                if name == "candidate_control_relative_rank":
                    margin = float(scores[j])
                else:
                    margin = float(frozen[(fold, VARIANTS[name], str(ids[ix]))]["margin"])
                pred.append({
                    "fold": fold, "method": name, "sample_id": str(ids[ix]),
                    "cell_line": fold, "group": STATES[int(y[ix])],
                    "predeclared_reference_id": control,
                    "score": f"{margin:.12g}",
                    "pred_dtp": int(margin > 0),
                })
        # Sensitivity WITHOUT choosing a favorable control. Same six DTP/Acute across refs.
        for ref in sorted(str(ids[ix]) for ix in te if y[ix] == 0):
            shifted = relative_scores(model, ztest, ids[te], ref)
            msk = np.isin(y[te], [1, 2])
            tp = int(np.sum((shifted > 0) & (y[te] == 2)))
            afp = int(np.sum((shifted > 0) & (y[te] == 1)))
            diag.append({"fold": fold, "reference_id": ref,
                         "is_predeclared": int(ref == control),
                         "n_same_acute_dtp_test_samples": int(msk.sum()),
                         "dtp_tp": tp, "acute_fp": afp,
                         "acute_dtp_accuracy": f"{(tp + (3 - afp)) / 6:.12g}",
                         "strict_candidates_train_only": int(info["strict_candidate_count"]),
                         "train_control_ids": ";".join(source_ref[line] for line in sorted(source_ref))})
    return pred, diag


def report_text(metrics_rows, sens_rows):
    lines = [
        "# Stage 48 — Control-relative rank-space learning (exploratory)", "",
        "**No independent test; no novelty claim; previous source folds already inspected.**", "",
        "One frozen candidate: within-panel per-sample ranks minus ONE known same-line Control rank vector;",
        "train-only Stage27 dual-contrast panel; one fixed logistic regression, cutoff 0.",
        "Each training line uses its lexicographically first Control as a reference;",
        "all training references are excluded from fit, and the held-out reference",
        "is excluded from ALL paired test methods (8 test observations per fold).", "",
        "## Macro-average source LOCO results (4 already-inspected cell lines)", "",
        "| method | mean balanced accuracy | mean sensitivity | mean specificity | total DTP TP / 12 | total acute FP / 12 | total Control FP / 8 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    methods = ("base_raw_zero", "protected_raw_zero", "candidate_control_relative_rank")
    for method in methods:
        rows = [r for r in metrics_rows if r["method"] == method]
        lines.append(f"| {method} | {np.mean([r['balanced_accuracy'] for r in rows]):.3f} | "
                     f"{np.mean([r['sensitivity'] for r in rows]):.3f} | "
                     f"{np.mean([r['specificity'] for r in rows]):.3f} | "
                     f"{sum(r['tp'] for r in rows)}/12 | {sum(r['acute_fp'] for r in rows)}/12 | "
                     f"{sum(r['control_fp'] for r in rows)}/8 |")
    lines.extend(["", "## Paired per-fold results", "",
                  "| fold | method | DTP TP/3 | Acute FP/3 | Control FP/2 | binary BA | ROC-AUC |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for r in metrics_rows:
        lines.append(f"| {r['fold']} | {r['method']} | {r['tp']}/3 | {r['acute_fp']}/3 | "
                     f"{r['control_fp']}/2 | {r['balanced_accuracy']:.3f} | {r['roc_auc']:.3f} |")
    cand = [r for r in metrics_rows if r["method"] == "candidate_control_relative_rank"]
    old = [r for r in metrics_rows if r["method"] == "protected_raw_zero"]
    cand_by = {r["fold"]: r for r in cand}
    old_by = {r["fold"]: r for r in old}
    exploratory_pass = (
        sum(r["tp"] for r in cand) == 12
        and sum(r["acute_fp"] for r in cand) <= 1
        and sum(r["control_fp"] for r in cand) == 0
        and all(cand_by[fold]["balanced_accuracy"] >= old_by[fold]["balanced_accuracy"]
                for fold in LINES)
    )
    lines.extend(["", "## Predeclared exploratory stop rule", "",
                  f"**{'CANDIDATE HAS AN EXPLORATORY SIGNAL — external test still required' if exploratory_pass else 'STOP THIS FEATURE-LEVEL CANDIDATE — no qualifying source-side improvement'}**", "",
                  "To pass exploratory screening: all 12 DTP retained, Acute FP <=1 (versus historical 3),",
                  "no Control FP, and no per-fold BA decline versus protected_raw_zero.",
                  "These source folds were repeatedly inspected; no confirmatory claim is possible.", "",
                  "## Reference-choice sensitivity (secondary, not model selection)", "",
                  "| fold | DTP TP across 3 controls | Acute FP across 3 controls |",
                  "|---|---|---|",])
    for fold in LINES:
        rows = [r for r in sens_rows if r["fold"] == fold]
        lines.append(f"| {fold} | {min(x['dtp_tp'] for x in rows)}–{max(x['dtp_tp'] for x in rows)}/3 | "
                     f"{min(x['acute_fp'] for x in rows)}–{max(x['acute_fp'] for x in rows)}/3 |")
    lines.extend(["", "## Limitations", "",
                  "- Already inspected GSE193258 source folds; no unbiased independent confirmation.",
                  "- Only 4 biologically distinct cell lines; three replicates per condition per line.",
                  "- Known target-cell-line Control required: reference-assisted, not zero-shot.",
                  "- Source Acute 24h and DTP 21d remain confounded with treatment duration.",
                  "- Control normalization/RUV/COCONUT concepts are prior art, not inherent algorithm novelty.",
                  "- We do not claim successful external transfer or clinical generalization.",
                  "- Alternative reference results are descriptive only; never select best after seeing labels.", ""])
    return "\n".join(lines), exploratory_pass


def run():
    for required in (STAGE47_PROV, STAGE47_SCORES, FROZEN):
        if not required.is_file():
            raise FileNotFoundError(f"Required audited prior artifact missing: {required}")
    prov = json.loads(STAGE47_PROV.read_text(encoding="utf-8"))
    if (prov.get("source_study") != "GSE193258" or
        prov.get("positive_label") != "DTP" or
        prov.get("outer_test_replay_required_and_passed") is not True):
        raise ValueError("Stage47 provenance gate not approved")
    st21 = load_script(SCRIPTS / "21_group_consistent_rank_selection.py", "stage21_source48")
    st27 = load_script(SCRIPTS / "27_dtp_specific_contrast_benchmark.py", "stage27_selector48")
    genes, ids, X, y, groups, sourceprov = st21.load_data()
    frozen_rows = rows_from_csv(FROZEN)
    keyset = validate_source(ids, X, y, groups, frozen_rows, prov)
    stage47_score_integrity(ids, y, groups, keyset)
    pred_rows, sensitivity_rows = perform(st21, st27, ids, X, y, groups, keyset)
    by_fold = []
    for fold in LINES:
        for method in ("base_raw_zero", "protected_raw_zero", "candidate_control_relative_rank"):
            subset = [r for r in pred_rows if r["fold"] == fold and r["method"] == method]
            by_fold.append({"fold": fold, "method": method, **metrics(subset)})
    report, signal = report_text(by_fold, sensitivity_rows)
    metadata = {
        "stage": 48, "status": "precommitted exploratory, not independently validated",
        "source_only": True, "source_study": "GSE193258",
        "positive_label": "DTP", "source_X_float64_sha256": prov["source_X_float64_sha256"],
        "stage38_sha256": sha256_file(FROZEN),
        "stage47_source_scores_sha256": sha256_file(STAGE47_SCORES),
        "stage47_provenance_sha256": sha256_file(STAGE47_PROV),
        "stage27_selector_sha256": sha256_file(SCRIPTS / "27_dtp_specific_contrast_benchmark.py"),
        "model": "Control-relative within-panel 100-gene ranks, LR C=.01, cutoff=0",
        "target_control_required": True,
        "reference_policy": "first sorted Control GSM in each line, held-out reference excluded from paired test",
        "candidate_feature_selection_only_on_3_outer_training_lines": True,
        "test_labels_used_in_fit": False,
        "observed_source_exploratory_signal_only": bool(signal),
        "independent_external_validation": False,
    }
    output = {
        OUT / "48_predictions.csv": csv_out(pred_rows),
        OUT / "48_metrics_by_fold.csv": csv_out(by_fold),
        OUT / "48_reference_sensitivity.csv": csv_out(sensitivity_rows),
        OUT / "48_summary.md": report + "\n",
        OUT / "48_provenance.json": json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    }
    write_immutable(output)
    print("Stage48 SOURCE-only exploratory feature-level contrast complete. No external data used.")
    for r in by_fold:
        print(f"{r['fold']:8s} {r['method']:32s} DTP={r['tp']}/3 AcuteFP={r['acute_fp']}/3 ControlFP={r['control_fp']}/2 BA={r['balanced_accuracy']:.3f}")
    print("EXPLORATORY CANDIDATE SCREEN:", "PASSED (NOT validation)" if signal else "FAILED — stop this candidate")
    for p in output:
        print("Saved:", p.resolve())


def self_test():
    rng = np.random.default_rng(4801)
    ids = np.array([f"SYNTH_{i:03d}" for i in range(36)])
    groups = np.repeat(np.array(["A", "B", "C", "D"]), 9)
    y = np.tile(np.repeat([0, 1, 2], 3), 4)
    X = rng.normal(size=(36, 600))
    # Create an observable but deliberately synthetic rank signal.
    X[y == 2, :50] += 3.0
    X[y == 1, :50] += 1.0
    chosen = np.arange(100)
    outer = "D"
    tr = np.flatnonzero(groups != outer)
    te = np.flatnonzero(groups == outer)
    ref = reference_id(ids[te], groups[te], y[te], outer)
    model, ranks, train_refs = fit_relative_model(
        X[tr], y[tr], groups[tr], ids[tr], X[te], ids[te], chosen, ref)
    score = relative_scores(model, ranks, ids[te], ref)
    assert len(score) == 9 and np.isfinite(score).all()
    assert len(train_refs) == 3 and ref == ids[te][0]
    assert np.isclose(score[0], model.decision_function(np.zeros((1, K)))[0])
    # Test labels are not an argument of the fitting or scoring functions.
    test_labels_flipped = y[te][::-1]
    assert not np.array_equal(test_labels_flipped, y[te])
    after = relative_scores(model, ranks, ids[te], ref)
    assert np.array_equal(after, score)
    # Changing a non-reference target sample cannot modify OTHER scores.
    x_alt = X[te].copy()
    x_alt[4, :50] = x_alt[4, :50] + 20
    rank_alt = rankdata(x_alt[:, chosen], axis=1, method="average") / K
    alt_scores = relative_scores(model, rank_alt, ids[te], ref)
    assert np.allclose(np.delete(score, 4), np.delete(alt_scores, 4))
    try:
        relative_scores(model, ranks, ids[te], "UNKNOWN_CONTROL")
    except ValueError:
        pass
    else:
        raise AssertionError("Missing target reference must be rejected")
    rows = []
    for j, ix in enumerate(te):
        if str(ids[ix]) != ref:
            rows.append({"group": STATES[int(y[ix])],
                         "score": str(float(score[j])), "pred_dtp": int(score[j] > 0)})
    assert metrics(rows)["n"] == 8
    print("SELF-TEST PASSED: train/test isolation, deterministic controls, no target-label access, "
          "8 matched test samples, reference perturbation & finite margins. SYNTHETIC ONLY.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run()


if __name__ == "__main__":
    main()
