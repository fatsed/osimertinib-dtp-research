#!/usr/bin/env python3
"""Stage 38 | Matched source-only domain-generalization baselines.

A *retrospective* source LOCO diagnostic: compare the frozen Stage28 ERM and
Stage35 contrast-protected model with (i) full-batch, linear, entropic
GroupDRO-style logistic optimization over the THREE SOURCE CELL LINES, and
(ii) source-only covariance-discrepancy shrinkage (CORAL-INSPIRED, **NOT**
canonical CORAL, which typically uses unlabeled TARGET covariance).

All models use the SAME training-selected 100-gene rank panel, same source
StandardScaler (fit only on 27 training samples), same binary objective, same
C=.01 and decision threshold zero. No target/test fitting; no target data
read. GroupDRO has a fixed exp-gradient stepsize 0.3 and 40 full-batch
updates, no hyperparameter tuning. CORAL-inspired transform has fixed gamma=1.
These particular implementations are intentionally documented, not claims of
exact reproduction of authors' original large-scale settings.

RUN from repository root:
  python scripts/38_matched_source_dg_benchmark.py --self-test
  python scripts/38_matched_source_dg_benchmark.py --exploratory-only

Outputs (append-only): outputs/domain_invariance/38_{report.md,summary.json,
  predictions.csv,train_diagnostics.csv}

NO DICA result is claimed; no untouched validation or algorithm originality.
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
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs" / "domain_invariance"
FROZEN = ROOT / "outputs" / "dtp_specificity"
STATES = ("Control", "Acute", "DTP")
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
METHODS = ("acute_only_rank_lr100", "dual_contrast_rank_lr100")
VARIANTS = ("stage28_baseline", "contrast_protected_shrinkage_gamma1",
            "linear_groupdro_line", "source_covariance_shrinkage")
C = 0.01
K = 100
GROUPDRO_ETA = 0.3
GROUPDRO_STEPS = 40
COV_GAMMA = 1.0


def imp(path, name):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required code: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources():
    return (
        imp(SCRIPTS / "21_group_consistent_rank_selection.py", "s21_for38"),
        imp(SCRIPTS / "27_dtp_specific_contrast_benchmark.py", "s27_for38"),
        imp(SCRIPTS / "28_dual_contrast_ablation.py", "s28_for38"),
        imp(SCRIPTS / "34_source_only_class_conditional_shrinkage.py", "s34_for38"),
        imp(SCRIPTS / "35_contrast_protected_domain_shrinkage.py", "s35_for38"),
    )


def filehash(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def readcsv(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing frozen prior results: {path}")
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dumpcsv(rows):
    if not rows:
        raise ValueError("Empty output")
    stream = io.StringIO(newline="")
    w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return stream.getvalue()


def save_immutable(files):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        p = OUT / name
        if p.is_file() and p.read_bytes() != data.encode("utf-8"):
            raise FileExistsError(f"Refusing to modify prior results: {p}")
    for name, data in files.items():
        p = OUT / name
        if not p.exists():
            p.write_bytes(data.encode("utf-8"))


def check_training(z, y3, g):
    z = np.asarray(z, float)
    y3 = np.asarray(y3, int)
    g = np.asarray(g)
    if z.shape != (27, K) or len(g) != 27 or not np.isfinite(z).all():
        raise ValueError("Expected 27 x 100 finite TRAIN-only ranks")
    lines = sorted(set(g.tolist()))
    if len(lines) != 3 or any(Counter(y3[g == line].tolist()) != {0: 3, 1: 3, 2: 3} for line in lines):
        raise ValueError("Source-only 3 domain x 3 condition x 3 replicate design broken")
    return lines


def log_losses_and_gradient(theta, a, binary, weights):
    """Weighted logistic objective with liblinear-style penalized intercept.

    sklearn/liblinear minimizes 0.5*||[coef,intercept]||^2 + C*sum_i cw_i*logloss.
    Here weights are cw_i * (3*q_group) for equally sized source domains.
    """
    logits = a @ theta
    losses = np.logaddexp(0, logits) - binary * logits
    probs = expit(logits)
    obj = 0.5 * np.dot(theta, theta) + C * np.dot(weights, losses)
    grad = theta + C * (a.T @ (weights * (probs - binary)))
    return float(obj), grad


def fit_weighted_logistic(a, binary, weights, x0=None):
    a = np.asarray(a, float)
    binary = np.asarray(binary, float)
    weights = np.asarray(weights, float)
    if a.shape != (27, K + 1) or binary.shape != (27,) or weights.shape != (27,) or (weights <= 0).any():
        raise ValueError("Invalid weighted logistic input")
    result = minimize(log_losses_and_gradient,
                      np.zeros(K + 1) if x0 is None else x0,
                      args=(a, binary, weights), jac=True, method="L-BFGS-B",
                      options={"maxiter": 250, "ftol": 1e-13, "gtol": 1e-10})
    if not result.success and np.linalg.norm(result.jac) > 1e-5:
        raise RuntimeError(f"GroupDRO logistic optimizer failed: {result.message}")
    if not np.isfinite(result.x).all():
        raise ValueError("Non-finite GroupDRO coefficients")
    return result.x


def source_groupdro(ztrain_std, ztest_std, y3, groups, eta=GROUPDRO_ETA, steps=GROUPDRO_STEPS):
    """Fixed full-batch GroupDRO-style exponential group reweighting.

    Group=training cell line (3 groups), with class-balanced logistic loss
    inside each group. Each loop: minimize weighted regularized logistic risk,
    then mirror-ascent group weights using measured group risk. Target excluded.
    """
    lines = check_training(ztrain_std, y3, groups)
    zte = np.asarray(ztest_std, float)
    if zte.ndim != 2 or zte.shape[1] != K or not np.isfinite(zte).all():
        raise ValueError("Bad held-out feature matrix")
    if eta <= 0 or steps < 1 or steps > 200 or not np.isfinite(eta):
        raise ValueError("Fixed groupDRO optimizer settings invalid")
    y3 = np.asarray(y3, int)
    g = np.asarray(groups)
    binary = (y3 == 2).astype(float)
    cw = np.where(binary == 1.0, 1.5, 0.75)  # globally balanced, 9 vs 18
    a = np.column_stack((np.asarray(ztrain_std, float), np.ones(27)))
    te = np.column_stack((zte, np.ones(len(zte))))
    group_idx = np.array([lines.index(x) for x in g], dtype=int)
    q = np.ones(3) / 3
    theta = None
    for _ in range(steps):
        weights = cw * (3.0 * q[group_idx])
        theta = fit_weighted_logistic(a, binary, weights, theta)
        margins = a @ theta
        losses = np.logaddexp(0, margins) - binary * margins
        group_risk = np.array([float(np.mean((cw * losses)[group_idx == i])) for i in range(3)])
        logs = np.log(q) + eta * group_risk
        logs -= np.max(logs)
        q = np.exp(logs)
        q /= q.sum()
    # Final weighted solve for final weights, with no model selection.
    theta = fit_weighted_logistic(a, binary, cw * (3.0 * q[group_idx]), theta)
    margin = te @ theta
    return margin, {"train_group_weights": {line: float(q[i]) for i, line in enumerate(lines)},
                    "mean_train_line_weighted_losses": {line: float(group_risk[i]) for i, line in enumerate(lines)},
                    "groupdro_eta": eta, "groupdro_steps": steps}


def source_covariance_shrinkage(ztrain_std, y3, groups, gamma=COV_GAMMA):
    """ONE shared transform based only on training domain covariance mismatch.

    This is NOT classic CORAL: no unlabeled target features/covariance are seen.
    C_l is the covariance of each training cell line's standardized ranks,
    M=(1/3)*sum_l (C_l - Cbar).T@(C_l - Cbar),
    P=(I+gamma*M)^(-1/2).  Same shared P applied to train and test.
    """
    lines = check_training(ztrain_std, y3, groups)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Invalid covariance shrink gamma")
    z = np.asarray(ztrain_std, float)
    gg = np.asarray(groups)
    covs = np.stack([np.cov(z[gg == line], rowvar=False, ddof=1) for line in lines])
    mean_cov = covs.mean(axis=0)
    differences = covs - mean_cov
    M = np.einsum("dki,dkj->ij", differences, differences, optimize=True) / len(lines)
    eigen, vec = np.linalg.eigh((M + M.T) / 2)
    if float(eigen.min()) < -1e-8:
        raise ValueError("Unexpected non-PSD source covariance penalty")
    P = (vec * (1 / np.sqrt(1 + gamma * np.maximum(eigen, 0)))) @ vec.T
    P = (P + P.T) / 2
    if gamma == 0 and not np.allclose(P, np.eye(K), rtol=1e-9, atol=1e-9):
        raise AssertionError("Gamma=0 should be identity")
    if not np.isfinite(P).all():
        raise ValueError("Invalid covariance transform")
    # Covariance mismatch after transformation: diag (same P) computed from TRAIN only.
    post = np.asarray([P.T @ cc @ P for cc in covs])
    original = float(np.linalg.norm(differences))
    after = float(np.linalg.norm(post - post.mean(axis=0)))
    return P, {"cov_discrepancy_fro_before": original,
               "cov_discrepancy_fro_after": after,
               "cov_gamma": gamma}


def fit_cov_model(ztrain_std, ztest_std, y3, groups, st34):
    P, trace = source_covariance_shrinkage(ztrain_std, y3, groups)
    clf = st34.lr_model().fit(ztrain_std @ P, (np.asarray(y3) == 2).astype(int))
    return np.asarray(clf.decision_function(ztest_std @ P), float), trace


def validate_frozen(genes, ids, X, y, groups, st34, st35):
    existing, panels, s34 = st35.source_frozen(st34, genes, ids, y, groups)
    frozen_rows = readcsv(OUT / "35_predictions.csv")
    locked = {}
    for r in frozen_rows:
        key = (r["method"], r["variant"], r["held_out_line"], r["gsm"])
        if key in locked:
            raise ValueError("Repeated Stage35 prediction")
        locked[key] = r
    if len(locked) != 216:
        raise ValueError("Stage35 must have 216 sample-level predictions")
    for key, r in locked.items():
        if r["variant"] == "contrast_protected_shrinkage_gamma1":
            continue
        earlier = existing[key]
        if not np.isclose(float(earlier["margin"]), float(r["margin"]), atol=1e-8, rtol=1e-7):
            raise ValueError("Stages 34 and 35 disagree")
    summary = json.loads((OUT / "35_summary.json").read_text(encoding="utf-8"))
    if summary.get("source_provenance") != s34.get("source_provenance") or summary.get("gamma_fixed") != 1:
        raise ValueError("Source provenance / prototype lock mismatch")
    return locked, panels, summary


def evaluate(genes, ids, X, y, groups, st27, st28, st34, st35, frozen=None, panels=None):
    y = np.asarray(y, int)
    g = np.asarray(groups)
    X = np.asarray(X, float)
    if len(ids) != 36 or len(set(ids)) != 36 or len(genes) != X.shape[1] or set(g) != set(LINES):
        raise ValueError("Unexpected source cohort")
    out = []
    diagnostics = []
    for outer in LINES:
        tr = np.flatnonzero(g != outer)
        te = np.flatnonzero(g == outer)
        check_training(np.zeros((27, K)), y[tr], g[tr])
        if len(te) != 9 or Counter(y[te].tolist()) != {0: 3, 1: 3, 2: 3}:
            raise ValueError("Invalid held-out cohort")
        for method in METHODS:
            if method == "acute_only_rank_lr100":
                panel, _ = st28.select_single_contrast(X[tr], y[tr], g[tr], negative=1)
            else:
                panel, _ = st27.select_rank_panel(X[tr], y[tr], g[tr], mode="dual_contrast")
            if len(panel) != K or len(set(panel.tolist())) != K:
                raise ValueError("Feature panel invalid")
            if panels is not None and set(genes[int(i)] for i in panel) != set(panels[(method, outer)]):
                raise ValueError(f"Stage35 selected panel changed: {method}/{outer}")
            ztr = st34.within_panel_rank(X[tr], panel)
            zte = st34.within_panel_rank(X[te], panel)
            prior, _trace = st35.fit_triplet(ztr, zte, y[tr], g[tr], st34)
            baseline, protected = prior[0], prior[2]
            # Every prior score must exactly replay BEFORE proceeding to new baselines.
            if frozen is not None:
                for j, idx in enumerate(te):
                    for variant, scores in ((VARIANTS[0], baseline), (VARIANTS[1], protected)):
                        rec = frozen.get((method, variant, outer, ids[idx]))
                        if rec is None or rec["true_state"] != STATES[int(y[idx])] or not np.isclose(
                            float(rec["margin"]), float(scores[j]), atol=1e-8, rtol=1e-7
                        ):
                            raise ValueError(f"Stage35 frozen replay mismatch: {method}/{variant}/{outer}/{ids[idx]}")
            scaler = StandardScaler().fit(ztr)
            z0 = scaler.transform(ztr)
            zte0 = scaler.transform(zte)
            groupdro, gd_trace = source_groupdro(z0, zte0, y[tr], g[tr])
            cov, cov_trace = fit_cov_model(z0, zte0, y[tr], g[tr], st34)
            diagnostics.append({"method": method, "held_out_line": outer,
                                "group_weights": gd_trace["train_group_weights"],
                                "group_losses": gd_trace["mean_train_line_weighted_losses"],
                                **cov_trace})
            for j, idx in enumerate(te):
                for variant, score in zip(VARIANTS, (baseline[j], protected[j], groupdro[j], cov[j])):
                    pred = int(float(score) > 0)
                    out.append({"method": method, "variant": variant,
                                "held_out_line": outer, "gsm": ids[idx],
                                "true_state": STATES[int(y[idx])],
                                "margin": f"{float(score):.12g}",
                                "predicted_dtp": pred,
                                "correct": int(pred == (y[idx] == 2))})
    if len(out) != 2 * 4 * 36 or len(diagnostics) != 8:
        raise AssertionError("Incomplete matched benchmark")
    return out, diagnostics


def summarize(rows, st34):
    metrics = {}
    pairing = {}
    for method in METHODS:
        metrics[method] = {}
        for variant in VARIANTS:
            these = [r for r in rows if r["method"] == method and r["variant"] == variant]
            if len(these) != 36 or len({r["gsm"] for r in these}) != 36:
                raise ValueError("Duplicate/incomplete prediction coverage")
            y = [STATES.index(r["true_state"]) for r in these]
            scores = [float(r["margin"]) for r in these]
            folds = {}
            for line in LINES:
                selected = [r for r in these if r["held_out_line"] == line]
                folds[line] = st34.score_metrics([STATES.index(r["true_state"]) for r in selected],
                                                 [float(r["margin"]) for r in selected])
            metrics[method][variant] = {"pooled": st34.score_metrics(y, scores), "per_line": folds,
                                        "worst_line_macro": min(q["three_condition_macro_recall"] for q in folds.values())}
        prot = {r["gsm"]: int(r["correct"]) for r in rows if r["method"] == method and r["variant"] == VARIANTS[1]}
        pairing[method] = {}
        for other in VARIANTS:
            obs = {r["gsm"]: int(r["correct"]) for r in rows if r["method"] == method and r["variant"] == other}
            pairing[method][other] = {"protected_only_correct": sum(prot[s] == 1 and obs[s] == 0 for s in prot),
                                      "other_only_correct": sum(prot[s] == 0 and obs[s] == 1 for s in prot)}
    return metrics, pairing


def report_md(summary):
    out = ["# Stage 38 — Matched source-only domain-generalization benchmark", "",
           "**Retrospective exploratory four-cell-line LOCO. No untouched external study, no novelty or clinical claim.**", "",
           "Both models trained on the SAME 100-gene rank panel selected using training lines only. "
           "Comparators are SOURCE-ONLY analogues, not exact original full-scale GroupDRO/CORAL implementation. CORAL normally uses target covariance; here target is never read.", "",
           "| Selector | Method | Correct/36 | 3-condition macro | Control false DTP | Acute false DTP | DTP recall | Worst line macro |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for variant in VARIANTS:
            r = summary["metrics"][method][variant]
            z = r["pooled"]
            out.append(f"| {method} | {variant} | {z['correct']}/36 | {z['three_condition_macro_recall']:.3f} | "
                       f"{z['control_false_dtp']:.3f} | {z['acute_false_dtp']:.3f} | {z['dtp_recall']:.3f} | "
                       f"{r['worst_line_macro']:.3f} |")
    out.extend(["", "## Paired error differences (protected vs other)", "",
                "| Selector | Other method | Protected only correct | Other only correct |",
                "|---|---|---:|---:|"])
    for method in METHODS:
        for other, cnt in summary["paired"].get(method, {}).items():
            out.append(f"| {method} | {other} | {cnt['protected_only_correct']} | {cnt['other_only_correct']} |")
    out.extend(["", "## Benchmarked algorithms / limitations", "",
        "- `linear_groupdro_line`: full-batch linear logistic with exponentiated reweighting of the THREE TRAINING CELL LINES. Same class-balanced loss, intercept penalization, and C as ERM; eta=.3, 40 updates fixed. Not identical to original deep GroupDRO architecture/optimizer.",
        "- `source_covariance_shrinkage`: one shared, source-fitted covariance-discrepancy shrinkage transform for train/test, gamma=1. CORAL-INSPIRED only; original CORAL requires an unlabeled target domain, which is unavailable under the strict source-only rule.",
        "- No exact DICA or FIXED implementation is claimed; they remain named prior-art comparisons for a broader benchmark.",
        "- All 36 LOCO test rows were previously inspected while developing the protected method; improvements cannot confirm superiority or novelty.",
        "- 3 technical/biological replicates per condition and just 4 cell lines; treatment duration is confounded with observed states.",
        "- This is a method comparability diagnostic, not hyperparameter tuning or preregistration. No foreign study or target label was used.",
        "", "## References", "",
        "- Sagawa et al., Group DRO (ICLR 2020): https://mlanthology.org/iclr/2020/sagawa2020iclr-distributionally/",
        "- Sun, Feng and Saenko, CORAL (AAAI 2016): https://mlanthology.org/aaai/2016/sun2016aaai-return/", ""])
    return "\n".join(out)


def self_test():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    rng = np.random.default_rng(3801)
    z = rng.normal(size=(27, K))
    labels = np.array([s for s in range(3) for _ in range(3)] * 3)
    groups = np.array([str(i) for i in range(3) for _ in range(9)])
    te = rng.normal(size=(9, K))
    # Match liblinear's intercept penalty / balanced loss when q is uniform.
    std = StandardScaler().fit(z)
    z0 = std.transform(z)
    ref = make_pipeline(StandardScaler(), LogisticRegression(C=C, solver="liblinear",
                    class_weight="balanced", random_state=42, max_iter=3000)).fit(z, (labels == 2).astype(int))
    a = np.column_stack((z0, np.ones(27)))
    theta = fit_weighted_logistic(a, (labels == 2).astype(float),
                                  np.where(labels == 2, 1.5, 0.75))
    manual = np.column_stack((std.transform(te), np.ones(9))) @ theta
    if not np.allclose(manual, ref.decision_function(te), atol=2e-3, rtol=3e-3):
        raise AssertionError("GroupDRO logistic objective does not match frozen liblinear baseline")
    gd, trace = source_groupdro(z0, std.transform(te), labels, groups)
    if gd.shape != (9,) or not np.isfinite(gd).all() or abs(sum(trace["train_group_weights"].values()) - 1) > 1e-8:
        raise AssertionError("GroupDRO output invalid")
    P0, _ = source_covariance_shrinkage(z0, labels, groups, gamma=0)
    P, info = source_covariance_shrinkage(z0, labels, groups)
    if not np.allclose(P0, np.eye(K), atol=1e-8) or info["cov_discrepancy_fro_after"] > info["cov_discrepancy_fro_before"] + 1e-7:
        raise AssertionError("Covariance transform incorrect")
    try:
        check_training(z0, labels, np.array(["one"] * 27))
        raise AssertionError("No error for broken source group structure")
    except ValueError:
        pass
    # Inject synthetic data into the actual four-fold evaluator when upstream scripts exist.
    if (SCRIPTS / "35_contrast_protected_domain_shrinkage.py").is_file():
        _, st27, st28, st34, st35 = sources()
        genes = [f"SIM_{i:04d}" for i in range(760)]
        xrows = []
        yr = []
        gr = []
        ids = []
        base = rng.normal(size=760)
        offsets = rng.normal(scale=0.1, size=(4, 760))
        for j, line in enumerate(LINES):
            for state in (0, 1, 2):
                for repeat in range(3):
                    x = base + offsets[j] + rng.normal(scale=0.15, size=760)
                    if state == 2:
                        x[:180] += 2.5
                        x[180:340] -= 2.5
                    elif state == 1:
                        x[340:400] += 0.1
                    xrows.append(x)
                    yr.append(state)
                    gr.append(line)
                    ids.append(f"SIM_{j}_{state}_{repeat}")
        rows, diagnostics = evaluate(genes, ids, np.asarray(xrows), np.asarray(yr),
                                     np.asarray(gr), st27, st28, st34, st35)
        metrics, paired = summarize(rows, st34)
        example = {"metrics": metrics, "paired": paired}
        if len(rows) != 288 or len(diagnostics) != 8 or "source_covariance_shrinkage" not in report_md(example):
            raise AssertionError("Four-fold synthetic integration failed")
    print("SELF-TEST PASSED: liblinear-matched GroupDRO loss, source-only covariance transform, no-test guard, synthetic 4-fold paired integration")


def run():
    s21, s27, s28, st34, st35 = sources()
    genes, ids, X, y, groups, provenance = s21.load_data()
    if X.shape != (36, 19712) or len(set(genes)) != 19712:
        raise ValueError("Unexpected source expression cohort")
    frozen, panels, old = validate_frozen(genes, ids, X, y, groups, st34, st35)
    if old["source_provenance"] != provenance:
        raise ValueError("Frozen data provenance mismatch")
    # Optional local Stage37 method freeze. No dependency on Git-ignored local results.
    local_lock = ROOT / "results" / "37_method_lock.json"
    if local_lock.is_file():
        lock = json.loads(local_lock.read_text(encoding="utf-8"))
        for key, filename in (("scripts/35_contrast_protected_domain_shrinkage.py",
                               SCRIPTS / "35_contrast_protected_domain_shrinkage.py"),
                              ("outputs/domain_invariance/35_predictions.csv",
                               OUT / "35_predictions.csv")):
            if filehash(filename) != lock["source_file_sha256"][key]:
                raise ValueError(f"Stage37 code/data lock mismatch: {key}")
    rows, diags = evaluate(genes, ids, X, y, groups, s27, s28, st34, st35,
                           frozen=frozen, panels=panels)
    metrics, paired = summarize(rows, st34)
    summary = {"status": "EXPLORATORY_HISTORICAL_SOURCE_ONLY_NOT_INDEPENDENT",
               "study": "GSE193258", "n": 36, "held_out_lines": list(LINES),
               "selectors": list(METHODS), "methods": list(VARIANTS),
               "fixed_groupdro_eta": GROUPDRO_ETA, "fixed_groupdro_steps": GROUPDRO_STEPS,
               "fixed_cov_gamma": COV_GAMMA, "C": C, "threshold": 0.0,
               "groupdro_groups": "source_training_cell_lines_only",
               "coral_type": "source-only CORAL-inspired covariance shrinkage, NOT original CORAL",
               "metrics": metrics, "paired": paired, "diagnostics": diags,
               "training_only": True, "no_external_data": True,
               "source_provenance": provenance,
               "input_sha256": {str(path.relative_to(ROOT)): filehash(path) for path in (
                   SCRIPTS / "28_dual_contrast_ablation.py", SCRIPTS / "35_contrast_protected_domain_shrinkage.py",
                   OUT / "35_predictions.csv", OUT / "35_summary.json")},
               "limitations": ["LOCO folds already viewed; cannot confirm method superiority",
                               "No exact DICA or original transductive CORAL run",
                               "GroupDRO uses a fixed full-batch linear adaptation, not original deep implementation",
                               "Method Stage35 was designed after previous source test evaluations",
                               "DTP treatment duration confounded; no phenotype specific or clinical validity"]}
    artifacts = {"38_predictions.csv": dumpcsv(rows),
                 "38_train_diagnostics.csv": dumpcsv([
                     {"method": d["method"], "held_out_line": d["held_out_line"],
                      "group_weights_json": json.dumps(d["group_weights"], sort_keys=True),
                      "train_group_losses_json": json.dumps(d["group_losses"], sort_keys=True),
                      "cov_discrepancy_fro_before": f'{d["cov_discrepancy_fro_before"]:.12g}',
                      "cov_discrepancy_fro_after": f'{d["cov_discrepancy_fro_after"]:.12g}'}
                     for d in diags]),
                 "38_summary.json": json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                 "38_report.md": report_md(summary)}
    save_immutable(artifacts)
    print("Stage38 complete: Stage35 exact source score replay, matched source-only GroupDRO and CORAL-inspired baseline")
    for method in METHODS:
        print(method + ": " + "  ".join(
            f"{variant}={metrics[method][variant]['pooled']['correct']}/36"
            for variant in VARIANTS))
    print("Saved outputs/domain_invariance/38_{report.md,summary.json,predictions.csv,train_diagnostics.csv}")
    print("EXPLORATORY ONLY: historical LOCO; no original CORAL/DICA score, independent validation or novelty claim")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--exploratory-only", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    elif args.exploratory_only:
        run()
    else:
        p.error("Use --self-test or --exploratory-only")


if __name__ == "__main__":
    main()
