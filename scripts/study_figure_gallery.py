#!/usr/bin/env python3
"""Osimertinib-DTP | one-file figure studio (exploratory, November 2026).

This is the ONLY new figure-building script required in scripts/.
It reads immutable results from Stages 21/27/38/39/48/49 and produces
independent PNG + SVG manuscript CANDIDATES with a consistent editorial style.
It NEVER trains models, selects new features, changes research outputs,
uses invented samples, or makes independent-validation/biomarker claims.

Run from repository root (Python 3.12 environment):
    python scripts/study_figure_gallery.py --audit-only
    python scripts/study_figure_gallery.py --build-all
    python scripts/study_figure_gallery.py --only upset jaccard frequency
    python scripts/study_figure_gallery.py --only f5 gene_heatmap
    python scripts/study_figure_gallery.py --list

Default --build-all writes ONLY to figures/candidates/editorial_v1/.
Existing work under figures/candidates/ and all original outputs/ are preserved.
A later --build-all safely SKIPs existing candidates if source hashes are unchanged.
Use --force ONLY to replace files INSIDE editorial_v1 with newly generated art.
To preserve the audit trail, changing input hashes requires an explicit --force.

Limitations: source LOCO folds have already been inspected; GSE255958 is a
previously examined nine-sample external dataset. The three observed source
conditions differ in exposure duration. Figures are exploratory descriptions,
not independent validation or clinical/biological state confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA

# Stage 21 is dynamically imported for the EXACT audited source cohort.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "figures" / "candidates" / "editorial_v1"
STUDY = "GSE193258"
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
STATES = ("Control", "Acute", "DTP")
METHODS_38 = ("acute_only_rank_lr100", "dual_contrast_rank_lr100")
VARIANTS_38 = ("stage28_baseline", "contrast_protected_shrinkage_gamma1",
               "linear_groupdro_line", "source_covariance_shrinkage")
METHODS_27 = ("raw_lr500", "rank_anova_lr100", "dual_contrast_rank_lr100")
METHODS_48 = ("base_raw_zero", "protected_raw_zero", "candidate_control_relative_rank")
METHODS_49 = ("protected_raw_zero", "stage48_zero", "stage48_nested_oof_threshold")
METHOD = "dual_contrast_rank_lr100"
BASE = "stage28_baseline"
PROTECTED = "contrast_protected_shrinkage_gamma1"

# Editorial muted palette; meaning is fixed across source, external and gene plots.
COLORS = {"Control": "#4F6D82", "Acute": "#B28A45", "DTP": "#804C61"}
LINE_COLORS = {"H1975": "#4F6D82", "HCC2935": "#B28A45",
               "HCC827": "#638579", "PC9": "#804C61"}
INK = "#2C3439"
MUTED = "#738087"
NEUTRAL = "#AAB3B5"
GRID = "#E2E5E3"
THRESH = "#677279"
ACCENT = COLORS["DTP"]
BLUE = COLORS["Control"]
MARKERS = {"Control": "o", "Acute": "s", "DTP": "D"}

PATHS = {
    "source_38": ROOT / "outputs/domain_invariance/38_predictions.csv",
    "external_39": ROOT / "outputs/cross_study_contrast/39_predictions.csv",
    "genes_27": ROOT / "outputs/dtp_specificity/27_selected_genes.csv",
    "predictions_27": ROOT / "outputs/dtp_specificity/27_predictions.csv",
    "washout_27": ROOT / "outputs/dtp_specificity/27_washout_predictions.csv",
    "predictions_48": ROOT / "outputs/stage48_control_relative_rank/48_predictions.csv",
    "reference_48": ROOT / "outputs/stage48_control_relative_rank/48_reference_sensitivity.csv",
    "predictions_49": ROOT / "outputs/stage49_final_check/49_predictions.csv",
    "thresholds_49": ROOT / "outputs/stage49_final_check/49_thresholds.csv",
    "inner_oof_49": ROOT / "outputs/stage49_final_check/49_inner_oof_scores.csv",
}
PRIMARY = frozenset(("source_38", "external_39", "genes_27"))

REF_F6_AUDIT = "F6_gene_panel_UpSet_audit.json"
REF_F6_INTER = "F6_gene_panel_UpSet_exact_intersections.csv"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def editorial_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none", "axes.linewidth": 0.7,
        "savefig.dpi": 400,
    })


def clean(ax, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=0.6, length=3)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.65, alpha=0.8)
        ax.set_axisbelow(True)


def save(fig, stem: str, output: Path) -> None:
    path = output / stem
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("SAVED", path.name + ".png / .svg")


def plot_title(ax, name: str) -> None:
    ax.set_title(name, loc="left", pad=10, weight="medium")


def label_footnote(fig, note: str, y: float = 0.015, x: float = 0.09) -> None:
    fig.text(x, y, note, fontsize=7.2, color=MUTED, ha="left", va="bottom")


# ---------- Research CSV integrity audit (read-only) ----------
def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require(df: pd.DataFrame, *columns: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    ensure(not missing, f"Missing columns: {missing}")
    ensure(not df.empty, "File is empty")
    for col in columns:
        ensure(df[col].astype(str).str.strip().ne("").all(), f"Blank values in {col}")


def unique(df: pd.DataFrame, *columns: str) -> None:
    ensure(not df.duplicated(list(columns)).any(), f"Duplicate keys: {columns}")


def numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    require(df, column)
    vals = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    ensure(np.isfinite(vals).all(), f"Non-finite or nonnumeric: {column}")
    return vals


def ints(df: pd.DataFrame, column: str, lo: int, hi: int) -> np.ndarray:
    x = numeric(df, column)
    ensure(np.all(x == np.floor(x)) and np.all((x >= lo) & (x <= hi)),
           f"Invalid integer values in {column} (expected {lo}..{hi})")
    return x.astype(int)


def expected_states(part: pd.DataFrame, column: str = "true_state",
                    counts: tuple[int, int, int] = (3, 3, 3)) -> None:
    ensure(part[column].value_counts().to_dict() == dict(zip(STATES, counts)),
           f"Unexpected {column} counts: {part[column].value_counts().to_dict()}")


def pair_set(df: pd.DataFrame, id_col: str, label_col: str) -> set[tuple[str, str]]:
    return set(zip(df[id_col].astype(str), df[label_col].astype(str)))


def source_38(df: pd.DataFrame) -> None:
    require(df, "method", "variant", "held_out_line", "gsm", "true_state",
            "margin", "predicted_dtp", "correct")
    unique(df, "method", "variant", "held_out_line", "gsm")
    ensure(len(df) == 288, f"Stage 38: expected 288 rows, got {len(df)}")
    ensure(set(df.method) == set(METHODS_38), "Stage 38 method set changed")
    ensure(set(df.variant) == set(VARIANTS_38), "Stage 38 variant set changed")
    ensure(set(df.held_out_line) == set(LINES), "Stage 38 cell lines changed")
    margins = numeric(df, "margin")
    pred = ints(df, "predicted_dtp", 0, 1)
    correct = ints(df, "correct", 0, 1)
    ensure(np.array_equal(pred, (margins > 0).astype(int)),
           "Stage 38 score/prediction inconsistency")
    ensure(np.array_equal(correct, (pred == (df.true_state.to_numpy() == "DTP")).astype(int)),
           "Stage 38 correctness inconsistent with observed-state label")
    ref = None
    for (_, _, fold), part in df.groupby(["method", "variant", "held_out_line"]):
        ensure(len(part) == 9, f"Stage 38 incomplete fold: {fold}")
        expected_states(part)
    for (method, variant), part in df.groupby(["method", "variant"]):
        ensure(part.gsm.nunique() == 36, "Stage 38 duplicated or missing GSM")
        cur = pair_set(part, "gsm", "true_state")
        if ref is None:
            ref = cur
        else:
            ensure(cur == ref, f"Stage 38 sample identities differ: {method}/{variant}")
    ensure(df.groupby("gsm").held_out_line.nunique().eq(1).all(),
           "Stage 38 GSM assigned to multiple cell lines")
    print("PASS  Stage 38: 288 rows, 36 matched GSMs, 4 LOCO folds, 8 model combinations")


def external_39(df: pd.DataFrame) -> None:
    require(df, "method", "variant", "gsm", "true_state", "score", "predicted_dtp",
            "correct", "genes_available", "genes_source_mean_imputed")
    unique(df, "method", "variant", "gsm")
    ensure(len(df) == 36, "Stage 39: expected 36 rows")
    ensure(set(df.method) == set(METHODS_38), "Stage 39 method set changed")
    ensure(set(df.variant) == set(VARIANTS_38[:2]), "Stage 39 variant set changed")
    scores = numeric(df, "score")
    pred = ints(df, "predicted_dtp", 0, 1)
    correct = ints(df, "correct", 0, 1)
    available = ints(df, "genes_available", 0, 100)
    imputed = ints(df, "genes_source_mean_imputed", 0, 100)
    ensure(np.array_equal(pred, (scores > 0).astype(int)), "Stage 39 score/decision mismatch")
    ensure(np.array_equal(correct, (pred == (df.true_state.to_numpy() == "DTP")).astype(int)),
           "Stage 39 correct/label mismatch")
    ensure(np.array_equal(available + imputed, np.full(len(df), 100)),
           "Stage 39 available+imputed must equal 100 genes")
    ref = None
    for (method, variant), part in df.groupby(["method", "variant"]):
        ensure(len(part) == 9, f"Stage 39 incomplete model: {method}/{variant}")
        expected_states(part)
        cur = pair_set(part, "gsm", "true_state")
        if ref is None:
            ref = cur
        else:
            ensure(cur == ref, "Stage 39 external GSM pairs changed across variants")
    print(f"PASS  Stage 39: nine external GSMs, feature coverage {available.min()}..{available.max()}/100")


def genes_27(df: pd.DataFrame) -> None:
    require(df, "method", "held_out_line", "gene")
    unique(df, "method", "held_out_line", "gene")
    ensure(set(df.method) == set(METHODS_27), "Stage 27 gene method set changed")
    ensure(set(df.held_out_line) == set(LINES), "Stage 27 gene fold set changed")
    ensure(len(df) == 2800, f"Stage 27: expected 2800 selected-gene rows, got {len(df)}")
    for (method, fold), part in df.groupby(["method", "held_out_line"]):
        expect_n = 500 if method == "raw_lr500" else 100
        ensure(len(part) == expect_n, f"Selected genes missing: {method}/{fold}")
    panels = {
        fold: set(df.loc[(df.method == "dual_contrast_rank_lr100") &
                         (df.held_out_line == fold), "gene"]) for fold in LINES
    }
    union = set.union(*panels.values())
    counts = Counter(sum(gene in panels[f] for f in LINES) for gene in union)
    ensure(sum(k * n for k, n in counts.items()) == 400,
           "Stage 27 gene panel membership count must be 400")
    jac = [len(panels[a] & panels[b]) / len(panels[a] | panels[b])
           for a, b in combinations(LINES, 2)]
    print("PASS  Stage 27 genes: 4 x 100 dual-contrast selections; "
          f"union={len(union)}; fold frequencies={dict(sorted(counts.items()))}; "
          f"mean Jaccard={np.mean(jac):.6f}")


def predictions_27(df: pd.DataFrame, source: pd.DataFrame | None) -> None:
    require(df, "method", "held_out_line", "gsm", "true_state", "margin",
            "predicted_dtp", "correct")
    unique(df, "method", "held_out_line", "gsm")
    ensure(len(df) == 108, "Stage 27: expected 108 model predictions")
    ensure(set(df.method) == set(METHODS_27), "Stage 27 prediction method set changed")
    score = numeric(df, "margin")
    pred = ints(df, "predicted_dtp", 0, 1)
    correct = ints(df, "correct", 0, 1)
    ensure(np.array_equal(pred, (score > 0).astype(int)), "Stage 27 margin/decision mismatch")
    ensure(np.array_equal(correct, (pred == (df.true_state.to_numpy() == "DTP")).astype(int)),
           "Stage 27 correct/label mismatch")
    for (method, fold), part in df.groupby(["method", "held_out_line"]):
        ensure(len(part) == 9 and fold in LINES, "Stage 27 fold coverage incorrect")
        expected_states(part)
    for method, part in df.groupby("method"):
        ensure(part.gsm.nunique() == 36, f"Stage 27 incomplete method: {method}")
    if source is not None:
        s = source[(source.method == "dual_contrast_rank_lr100") &
                   (source.variant == "stage28_baseline")]
        ensure(pair_set(df[df.method == "dual_contrast_rank_lr100"], "gsm", "true_state") ==
               pair_set(s, "gsm", "true_state"), "Stage 27 and 38 source sample labels differ")
    print("PASS  Stage 27 predictions: 108 held-out predictions")


def washout_27(df: pd.DataFrame) -> None:
    require(df, "method", "held_out_line", "gsm", "experimental_condition",
            "days_after_drug_removal", "margin", "predicted_dtp_call_descriptive_only")
    unique(df, "method", "held_out_line", "gsm")
    ensure(len(df) == 72, "Stage 27 washout expected 72 predictions")
    ensure(set(df.method) == set(METHODS_27), "Stage 27 washout method set changed")
    score = numeric(df, "margin")
    calls = ints(df, "predicted_dtp_call_descriptive_only", 0, 1)
    ensure(np.array_equal(calls, (score > 0).astype(int)), "Washout margin/call mismatch")
    for (method, fold), part in df.groupby(["method", "held_out_line"]):
        ensure(fold in LINES and len(part) == 6,
               f"Washout count incorrect: {method}/{fold}")
        ensure(part.experimental_condition.value_counts().to_dict() ==
               {"ShortWashout": 3, "LongWashout": 3},
               f"Washout condition counts unexpected: {method}/{fold}")
    print("PASS  Stage 27 washout: 72 descriptive calls (NOT validated negative labels)")


def check_eight_sample_folds(df: pd.DataFrame, *, fold_col: str,
                             group_col: str, method_col: str,
                             id_col: str, methods: tuple[str, ...]) -> None:
    ensure(set(df[fold_col]) == set(LINES), "Unexpected fold names")
    ensure(set(df[method_col]) == set(methods), "Unexpected methods")
    for fold in LINES:
        matched = []
        for method in methods:
            part = df[(df[fold_col] == fold) & (df[method_col] == method)]
            ensure(len(part) == 8, f"Expected eight matched held-out samples in {fold}/{method}")
            expected_states(part, group_col, (2, 3, 3))
            matched.append(pair_set(part, id_col, group_col))
        ensure(all(s == matched[0] for s in matched[1:]),
               f"Unequal held-out sample sets among competitors in {fold}")


def predictions_48(df: pd.DataFrame, source: pd.DataFrame | None) -> None:
    require(df, "fold", "method", "sample_id", "cell_line", "group",
            "predeclared_reference_id", "score", "pred_dtp")
    unique(df, "fold", "method", "sample_id")
    ensure(len(df) == 96, "Stage 48 should have 96 predictions")
    check_eight_sample_folds(df, fold_col="fold", group_col="group",
                             method_col="method", id_col="sample_id", methods=METHODS_48)
    score = numeric(df, "score")
    pred = ints(df, "pred_dtp", 0, 1)
    ensure(np.array_equal(pred, (score > 0).astype(int)), "Stage 48 score/decision mismatch")
    ensure((df.fold == df.cell_line).all(), "Stage 48 fold/cell-line mismatch")
    ensure((df.sample_id != df.predeclared_reference_id).all(),
           "Stage 48 reference sample included in evaluation")
    ensure(df.groupby("fold").predeclared_reference_id.nunique().eq(1).all(),
           "Stage 48 must have one predeclared control per outer fold")
    if source is not None:
        source_recs = source[(source.method == "dual_contrast_rank_lr100") &
                             (source.variant == "stage28_baseline")]
        known = {(r.gsm, r.held_out_line): r.true_state
                 for r in source_recs.itertuples()}
        for r in df.itertuples():
            ensure(known.get((r.sample_id, r.fold)) == r.group,
                   f"Stage 48 sample not found in original source: {r.sample_id}")
            ensure(known.get((r.predeclared_reference_id, r.fold)) == "Control",
                   f"Stage 48 predeclared reference not verified Control: {r.fold}")
    print("PASS  Stage 48: paired 8-sample folds, Control reference excluded")


def reference_48(df: pd.DataFrame) -> None:
    require(df, "fold", "reference_id", "is_predeclared", "n_same_acute_dtp_test_samples",
            "dtp_tp", "acute_fp", "acute_dtp_accuracy")
    unique(df, "fold", "reference_id")
    ensure(len(df) == 12 and set(df.fold) == set(LINES),
           "Stage 48 reference sensitivity expected 4x3 entries")
    ints(df, "is_predeclared", 0, 1)
    n = ints(df, "n_same_acute_dtp_test_samples", 0, 6)
    tp = ints(df, "dtp_tp", 0, 3)
    afp = ints(df, "acute_fp", 0, 3)
    accuracy = numeric(df, "acute_dtp_accuracy")
    ensure(np.all(n == 6), "Reference sensitivity should use same 6 Acute/DTP samples")
    ensure(np.allclose(accuracy, (tp + 3 - afp) / 6, rtol=0, atol=1e-10),
           "Reference sensitivity accuracy doesn't match TP and false calls")
    for fold, part in df.groupby("fold"):
        ensure(len(part) == 3 and ints(part, "is_predeclared", 0, 1).sum() == 1,
               f"Reference sensitivity incorrect: {fold}")
    print("PASS  Stage 48 reference sensitivity: three Control choices per fold")


def predictions_49(df: pd.DataFrame, st48: pd.DataFrame | None,
                   thresholds: pd.DataFrame | None) -> None:
    require(df, "fold", "method", "sample_id", "group", "reference_id",
            "score", "threshold", "pred_dtp")
    unique(df, "fold", "method", "sample_id")
    ensure(len(df) == 96, "Stage 49 should have 96 predictions")
    check_eight_sample_folds(df, fold_col="fold", group_col="group",
                             method_col="method", id_col="sample_id", methods=METHODS_49)
    score, threshold = numeric(df, "score"), numeric(df, "threshold")
    pred = ints(df, "pred_dtp", 0, 1)
    ensure(np.array_equal(pred, (score > threshold).astype(int)),
           "Stage 49 score/threshold decision mismatch")
    ensure(df.groupby("fold").reference_id.nunique().eq(1).all(),
           "Stage 49 inconsistent reference across methods")
    ensure(np.all(threshold[df.method != "stage48_nested_oof_threshold"] == 0),
           "Stage 49 zero-cutoff comparator was modified")
    if st48 is not None:
        links = {"protected_raw_zero": "protected_raw_zero",
                 "stage48_zero": "candidate_control_relative_rank",
                 "stage48_nested_oof_threshold": "candidate_control_relative_rank"}
        old = {(r.fold, r.method, r.sample_id): r for r in st48.itertuples()}
        for r in df.itertuples():
            old_r = old.get((r.fold, links[r.method], r.sample_id))
            ensure(old_r is not None and old_r.group == r.group and
                   old_r.predeclared_reference_id == r.reference_id,
                   f"Stage 49 not paired to Stage 48: {r.fold}/{r.method}/{r.sample_id}")
            ensure(np.isclose(float(old_r.score), float(r.score), atol=2e-7, rtol=0),
                   "Stage 49 changed a frozen Stage 48 score")
    if thresholds is not None:
        tmap = {r.outer_fold: float(r.threshold) for r in thresholds.itertuples()}
        for fold in LINES:
            vals = df.loc[(df.fold == fold) &
                          (df.method == "stage48_nested_oof_threshold"), "threshold"].astype(float)
            ensure(np.allclose(vals.to_numpy(), tmap[fold], rtol=0, atol=1e-10),
                   f"Stage 49 OOF threshold mismatch: {fold}")
    print("PASS  Stage 49: matched Stage 48 scores with fixed/nested thresholds")


def thresholds_49(df: pd.DataFrame) -> None:
    require(df, "outer_fold", "threshold", "train_oof_balanced_accuracy", "n_train_oof",
            "nested_models", "n_inner_training_samples", "n_outer_training_samples")
    unique(df, "outer_fold")
    ensure(len(df) == 4 and set(df.outer_fold) == set(LINES),
           "Stage 49: expected four outer-fold thresholds")
    numeric(df, "threshold")
    acc = numeric(df, "train_oof_balanced_accuracy")
    ensure(np.all((acc >= 0) & (acc <= 1)), "Invalid OOF balanced accuracies")
    for col, value in (("n_train_oof", 24), ("nested_models", 3),
                       ("n_inner_training_samples", 18), ("n_outer_training_samples", 27)):
        ensure(np.all(ints(df, col, 0, 100) == value), f"Stage 49 {col} changed")
    print("PASS  Stage 49 thresholds: four train-only OOF cutoffs")


def inner_oof_49(df: pd.DataFrame) -> None:
    require(df, "outer_fold", "inner_held_out_line", "sample_id", "group", "score",
            "inner_reference_id", "n_inner_train", "inner_training_lines", "train_reference_ids")
    unique(df, "outer_fold", "inner_held_out_line", "sample_id")
    ensure(len(df) == 96, "Stage 49 inner OOF expected 96 predictions")
    numeric(df, "score")
    ensure(np.all(ints(df, "n_inner_train", 0, 30) == 18),
           "Stage 49 OOF inner training should be 18")
    for (outer, inner), part in df.groupby(["outer_fold", "inner_held_out_line"]):
        ensure(outer in LINES and inner in LINES and outer != inner,
               "OOF inner holdout must differ from outer holdout")
        ensure(len(part) == 8, "Stage 49 inner OOF missing an eight-sample block")
        expected_states(part, "group", (2, 3, 3))
        ensure((part.sample_id != part.inner_reference_id).all(),
               "Stage 49 inner reference was scored as a held-out sample")
    ensure(df.groupby("outer_fold").size().eq(24).all(),
           "Stage 49 outer fold must have 24 train-only OOF predictions")
    print("PASS  Stage 49 nested OOF: 4 outer folds x 24 train-only scores")



_SOURCE_CACHE = None


def load_source_matrix():
    """Load exactly the audited Stage21 GSE193258 matrix; no fitting."""
    global _SOURCE_CACHE
    if _SOURCE_CACHE is not None:
        return _SOURCE_CACHE
    script = ROOT / "scripts/21_group_consistent_rank_selection.py"
    if not script.is_file():
        raise FileNotFoundError(f"Stage 21 script missing: {script}")
    spec = importlib.util.spec_from_file_location("study_gallery_stage21", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load original Stage21")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    genes, ids, X, y, groups, prov = module.load_data()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=str)
    ids = np.asarray(ids, dtype=str)
    genes = np.asarray(genes, dtype=str)
    ensure(X.shape == (36, 19712) and np.isfinite(X).all(), "Source matrix not 36x19712 finite")
    ensure(len(set(ids)) == 36 and len(set(genes)) == 19712, "Source IDs/genes duplicated")
    ensure(Counter(y.tolist()) == {0:12, 1:12, 2:12}, "Wrong source class frequencies")
    ensure(Counter(groups.tolist()) == {k:9 for k in LINES}, "Wrong source cell-line frequencies")
    _SOURCE_CACHE = (genes, ids, X, y, groups, prov)
    return _SOURCE_CACHE


def audit_all(require_all: bool, load_source: bool) -> dict[str, pd.DataFrame]:
    print("\nFIGURE DATA AUDIT | read-only")
    loaded = {}
    warnings, errors = [], []
    for name, path in PATHS.items():
        if not path.is_file():
            msg = f"Missing {path.relative_to(ROOT)}"
            (errors if require_all or name in PRIMARY else warnings).append(msg)
            print("FAIL" if require_all or name in PRIMARY else "WARN", msg)
            continue
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
            ensure(not df.empty, f"Empty CSV: {path}")
            loaded[name] = df
            print(f"READ {name:16} {len(df):4d} rows SHA256 {digest(path)[:12]}")
        except Exception as e:
            errors.append(f"{name}: {e}")
    validators = [
        ("source_38", source_38, ()),
        ("external_39", external_39, ()),
        ("genes_27", genes_27, ()),
        ("predictions_27", predictions_27, ("source_38",)),
        ("washout_27", washout_27, ()),
        ("predictions_48", predictions_48, ("source_38",)),
        ("reference_48", reference_48, ()),
        ("thresholds_49", thresholds_49, ()),
        ("predictions_49", predictions_49, ("predictions_48", "thresholds_49")),
        ("inner_oof_49", inner_oof_49, ()),
    ]
    for name, validator, dependencies in validators:
        if name in loaded:
            try:
                validator(loaded[name], *(loaded.get(d) for d in dependencies))
            except Exception as e:
                errors.append(f"Invalid {name}: {e}")
    if load_source:
        try:
            genes, ids, X, _, _, _ = load_source_matrix()
            print(f"PASS Stage21 matrix {X.shape[0]} samples x {len(genes)} genes")
        except Exception as e:
            errors.append(f"Source matrix: {e}")
    for warning in warnings:
        print("WARNING", warning)
    for error in errors:
        print("ERROR", error)
    if errors:
        raise RuntimeError(f"Audit failed with {len(errors)} error(s); NOTHING BUILT")
    print("AUDIT PASSED | exploratory consistency only, NOT independent validation")
    return loaded


def checked_source(df: pd.DataFrame, variant: str = PROTECTED) -> pd.DataFrame:
    r = df[(df.method == METHOD) & (df.variant == variant)].copy()
    ensure(len(r) == 36 and r.gsm.nunique() == 36, "Source subset incomplete")
    r["margin"] = r["margin"].astype(float)
    r["predicted_dtp"] = r["predicted_dtp"].astype(int)
    return r


def checked_external(df: pd.DataFrame, variant: str = PROTECTED) -> pd.DataFrame:
    r = df[(df.method == METHOD) & (df.variant == variant)].copy()
    ensure(len(r) == 9 and r.gsm.nunique() == 9, "External subset incomplete")
    r["score"] = r["score"].astype(float)
    r["predicted_dtp"] = r["predicted_dtp"].astype(int)
    return r


def paired_source(df: pd.DataFrame) -> pd.DataFrame:
    base = checked_source(df, BASE)
    protected = checked_source(df, PROTECTED)
    keys = ["gsm", "held_out_line", "true_state"]
    pair = base[keys + ["margin", "predicted_dtp"]].merge(
        protected[keys + ["margin", "predicted_dtp"]], on=keys,
        suffixes=("_base", "_protected"), validate="one_to_one")
    ensure(len(pair) == 36, "Source pairing lost samples")
    return pair


def paired_external(df: pd.DataFrame) -> pd.DataFrame:
    base = checked_external(df, BASE)
    protected = checked_external(df, PROTECTED)
    keys = ["gsm", "true_state"]
    pair = base[keys + ["score", "predicted_dtp"]].merge(
        protected[keys + ["score", "predicted_dtp"]], on=keys,
        suffixes=("_base", "_protected"), validate="one_to_one")
    ensure(len(pair) == 9, "External pairing lost samples")
    return pair

# ---------- Source and external prediction figures ----------


def build_f1(data: dict, out: Path) -> None:
    """Decision transition table, not 36 visually redundant spaghetti lines."""
    p = paired_source(data["source_38"])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    fig.subplots_adjust(left=0.15, right=0.91, top=0.84, bottom=0.20)
    grid_n = np.zeros((2, 2), int)
    for rec in p.itertuples():
        grid_n[rec.predicted_dtp_base, rec.predicted_dtp_protected] += 1
    shades = LinearSegmentedColormap.from_list("neutral_blue", ["#F5F7F7", "#CBD7DC", BLUE])
    vmax = max(int(grid_n.max()), 1)
    ax.imshow(grid_n, cmap=shades, vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")
    for i in range(2):
        for j in range(2):
            cell = p[(p.predicted_dtp_base == i) & (p.predicted_dtp_protected == j)]
            n = len(cell)
            color = "white" if n > 0.65 * vmax else INK
            ax.text(j, i-0.055, str(n), ha="center", va="center", fontsize=21,
                    weight="semibold", color=color)
            by_state = cell.true_state.value_counts()
            summary = "  ·  ".join(f"{s} {int(by_state.get(s,0))}" for s in STATES if by_state.get(s,0))
            if summary:
                ax.text(j, i+0.23, summary, ha="center", va="center", fontsize=7.5,
                        color=color, wrap=True)
    ax.set_xticks([0, 1], ["Non-DTP", "DTP"])
    ax.set_yticks([0, 1], ["Non-DTP", "DTP"])
    ax.set_xlabel("Contrast-protected decision")
    ax.set_ylabel("Baseline decision")
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=4)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    n_change = int((p.predicted_dtp_base != p.predicted_dtp_protected).sum())
    fig.suptitle("Held-out decision transitions", x=0.15, y=0.975, ha="left", fontsize=12)
    fig.text(.15, .88, f"{n_change} of 36 matched decisions changed; source LOCO", color=MUTED, fontsize=8.3)
    label_footnote(fig, "Exploratory source evaluation · fixed score threshold = 0", x=0.15, y=0.04)
    save(fig, "01_source_decision_transitions", out)


def build_f2(data: dict, out: Path) -> None:
    p = paired_source(data["source_38"])
    variants = (("Baseline", "predicted_dtp_base"), ("Contrast-protected", "predicted_dtp_protected"))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), sharey=True)
    fig.subplots_adjust(left=.09, right=.985, top=.81, bottom=.24, wspace=.11)
    for ax, (title, col) in zip(axes, variants):
        plot_title(ax, title)
        for ix, state in enumerate(STATES):
            for jx, line in enumerate(LINES):
                part = p[(p.held_out_line == line) & (p.true_state == state)]
                n = int(part[col].sum())
                ax.scatter(jx, ix, s=(52 + 90 * n), marker="o", facecolor=COLORS[state] if n else "white",
                           edgecolor=COLORS[state], linewidth=1.15, alpha=.90, zorder=3)
                ax.text(jx, ix, str(n), fontsize=9, va="center", ha="center",
                        color="white" if n >= 2 else INK, weight="semibold", zorder=4)
        ax.set_xticks(range(4), LINES)
        ax.set_yticks(range(3), STATES)
        ax.set_xlim(-.55, 3.55)
        ax.set_ylim(2.35, -.35)
        ax.grid(axis="both", color=GRID, linewidth=.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("DTP calls by observed condition and held-out cell line", x=.09, y=.99, ha="left", fontsize=12)
    label_footnote(fig, "Numbers = samples called DTP, out of three per condition and cell line; fixed cutoff = 0", x=.09, y=.055)
    save(fig, "02_source_dtp_calls_bubblemap", out)


def build_f3(data: dict, out: Path) -> None:
    p = paired_source(data["source_38"])
    all_scores = p[["margin_base", "margin_protected"]].to_numpy(dtype=float)
    low = min(0, float(np.min(all_scores)))
    high = max(0, float(np.max(all_scores)))
    pad = max((high - low) * .09, .08)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.90, bottom=.14, hspace=.28, wspace=.14)
    for ax, line in zip(axes.flat, LINES):
        part = p[p.held_out_line == line].sort_values(["true_state", "gsm"])
        for state in STATES:
            rows = part[part.true_state == state].sort_values("gsm")
            for k, r in enumerate(rows.itertuples()):
                x0, x1 = -0.02 + k * .02, .98 + k * .02
                c = COLORS[state]
                ax.plot([x0, x1], [r.margin_base, r.margin_protected], color=c, alpha=.59, linewidth=1.05)
                ax.scatter([x0, x1], [r.margin_base, r.margin_protected], s=25,
                           color=c, marker=MARKERS[state], zorder=5,
                           edgecolor="white", linewidth=.45)
        ax.axhline(0, ls=(0, (4, 3)), lw=.9, color=THRESH)
        ax.set_xticks([0, 1], ["Baseline", "Protected"])
        ax.set_xlim(-.16, 1.16)
        ax.set_ylim(low-pad, high+pad)
        plot_title(ax, line)
        clean(ax)
    axes[0, 0].set_ylabel("Held-out decision margin")
    axes[1, 0].set_ylabel("Held-out decision margin")
    fig.suptitle("Matched score changes under contrast protection", x=.09, y=.99, ha="left", fontsize=12)
    legend = [Line2D([0], [0], lw=0, marker=MARKERS[s], color=COLORS[s], markersize=6, label=s) for s in STATES]
    fig.legend(handles=legend, ncol=3, loc="lower center", bbox_to_anchor=(.5, .045), frameon=False, fontsize=8.5)
    label_footnote(fig, "Each segment links the same GSM sample · source LOCO, not independent validation", x=.09, y=.004)
    save(fig, "03_source_paired_margins", out)


def build_f4(data: dict, out: Path) -> None:
    p = paired_external(data["external_39"])
    p["order"] = p.true_state.map({s:i for i,s in enumerate(STATES)})
    p = p.sort_values(["order", "gsm"])
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    fig.subplots_adjust(left=.10, right=.98, top=.82, bottom=.26)
    for idx, row in enumerate(p.itertuples()):
        color = COLORS[row.true_state]
        ax.plot([idx-.10, idx+.10], [row.score_base, row.score_protected], color=color,
                alpha=.65, lw=1.65, zorder=2)
        ax.scatter(idx-.10, row.score_base, marker="o", s=44, color=color,
                   edgecolors="white", linewidth=.6, zorder=4)
        ax.scatter(idx+.10, row.score_protected, marker="D", s=40, color=color,
                   edgecolors="white", linewidth=.6, zorder=4)
    for k in [2.5, 5.5]:
        ax.axvline(k, color=GRID, linewidth=1)
    ax.axhline(0, ls=(0, (4, 3)), color=THRESH, linewidth=.95)
    ax.set_xticks(range(9), [str(g)[-4:] for g in p.gsm])
    ax.set_xlim(-.65, 8.65)
    ax.set_ylabel("DTP decision margin")
    ax.set_xlabel("GSM ID (last four digits)")
    for i, state in enumerate(STATES):
        ax.text(i*3+1, 1.05, state, color=COLORS[state], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.2, fontweight="medium")
    fig.suptitle("Frozen model scores in GSE255958 (PC9)", x=.10, y=.99, ha="left", fontsize=11.5)
    clean(ax)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, markersize=5, label="Baseline"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=INK, markersize=5, label="Protected"),
    ]
    ax.legend(handles=legend, loc="upper left", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(.01,.99))
    label_footnote(fig, "Previously inspected exploratory cohort (n=9) · NOT independent external validation", x=.10, y=.025)
    save(fig, "04_external_paired_margins", out)


def build_f5(data: dict, out: Path) -> None:
    genes, ids, X, y, line, provenance = load_source_matrix()
    # PCA of source-wide within-sample transcriptomic ranks; no label data used in PCA fitting.
    rank = rankdata(X, axis=1, method="average") / X.shape[1]
    # The percentages are calculated from exactly the same PCA fit as the coordinates.
    pca = PCA(n_components=2, svd_solver="full")
    coords = pca.fit_transform(rank)
    variance = 100.0 * pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.22, wspace=.16)
    for axis, values, palette, title, keys in [
        (axes[0], np.array([STATES[i] for i in y]), COLORS, "Observed condition", STATES),
        (axes[1], line, LINE_COLORS, "Cell line", LINES),
    ]:
        for k in keys:
            chosen = np.flatnonzero(values == k)
            axis.scatter(coords[chosen,0], coords[chosen,1], s=49,
                         color=palette[k], edgecolor="white", linewidth=.65,
                         alpha=.9, label=k)
        axis.set_xlabel(f"PC1 ({variance[0]:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({variance[1]:.1f}% variance)")
        plot_title(axis, title)
        clean(axis, grid=None)
        axis.axhline(0, color=GRID, lw=.7, zorder=-1)
        axis.axvline(0, color=GRID, lw=.7, zorder=-1)
        axis.legend(frameon=False, loc="best", fontsize=7.9)
    fig.suptitle("Within-sample rank PCA | GSE193258 (n=36)", x=.09, y=.98, ha="left", fontsize=12)
    label_footnote(fig, "Exploratory 36 bulk samples · no single-cell trajectory or verified DTP-state claim", x=.09, y=.045)
    save(fig, "05_source_within_sample_rank_pca", out)
    # Saved in the eventual manifest via metadata; not interpreted as confirmatory separation.
    print(f"PCA computed variance: PC1 {variance[0]:.3f}% | PC2 {variance[1]:.3f}%")


def build_roc(data: dict, out: Path) -> None:
    from sklearn.metrics import roc_curve, roc_auc_score
    p = paired_source(data["source_38"])
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.97, top=.91, bottom=.10, wspace=.13, hspace=.29)
    styles = (("margin_base", "Baseline", "--"), ("margin_protected", "Protected", "-"))
    for ax, line in zip(axes.flat, LINES):
        q = p[p.held_out_line == line]
        truth = (q.true_state == "DTP").astype(int)
        for col, label, ls in styles:
            fpr, tpr, _ = roc_curve(truth, q[col])
            auc = roc_auc_score(truth, q[col])
            ax.step(fpr, tpr, where="post", ls=ls, linewidth=1.6 if ls=="-" else 1.2,
                    color=ACCENT if ls=="-" else BLUE, label=f"{label}: AUC {auc:.2f}")
        ax.plot([0,1], [0,1], color=NEUTRAL, linewidth=.85, linestyle=":")
        ax.set(xlim=(-.02,1.02), ylim=(-.02,1.04))
        plot_title(ax, line)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
        clean(ax)
    for ax in axes[1]:
        ax.set_xlabel("False positive rate")
    for ax in axes[:,0]:
        ax.set_ylabel("True positive rate")
    fig.suptitle("DTP vs Control + Acute | source LOCO ROC", x=.10, y=.99, ha="left", fontsize=11.5)
    label_footnote(fig, "Only 3 DTP + 6 non-DTP test samples per fold; curves are exploratory and discrete", x=.10, y=.007)
    save(fig, "06_source_loco_roc", out)


def build_pr(data: dict, out: Path) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    p = paired_source(data["source_38"])
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.97, top=.91, bottom=.10, wspace=.13, hspace=.29)
    styles = (("margin_base", "Baseline", "--"), ("margin_protected", "Protected", "-"))
    for ax, line in zip(axes.flat, LINES):
        q = p[p.held_out_line == line]
        truth = (q.true_state == "DTP").astype(int)
        for col, label, ls in styles:
            precision, recall, _ = precision_recall_curve(truth, q[col])
            ap = average_precision_score(truth, q[col])
            ax.step(recall, precision, where="post", ls=ls, linewidth=1.6 if ls=="-" else 1.2,
                    color=ACCENT if ls=="-" else BLUE, label=f"{label}: AP {ap:.2f}")
        ax.axhline(1/3, color=NEUTRAL, linewidth=.85, linestyle=":")
        ax.set(xlim=(-.02,1.02), ylim=(-.02,1.04))
        plot_title(ax, line)
        ax.legend(loc="lower left", fontsize=7, frameon=False)
        clean(ax)
    for ax in axes[1]:
        ax.set_xlabel("Recall")
    for ax in axes[:,0]:
        ax.set_ylabel("Precision")
    fig.suptitle("DTP vs Control + Acute | source LOCO precision–recall", x=.10, y=.99, ha="left", fontsize=11.2)
    label_footnote(fig, "3/9 positive samples per held-out fold · exploratory average precision, no clinical validity", x=.10, y=.007)
    save(fig, "07_source_loco_precision_recall", out)


def build_thresholds(data: dict, out: Path) -> None:
    d = data["predictions_49"].copy()
    d["pred_dtp"] = d.pred_dtp.astype(int)
    methods = METHODS_49
    group_labels = ("Control FP", "Acute FP", "DTP TP")
    rows = []
    for fold in LINES:
        for name in methods:
            q = d[(d.fold == fold)&(d.method == name)]
            rows.append((fold, name,
                         int(q.loc[q.group == "Control", "pred_dtp"].sum()),
                         int(q.loc[q.group == "Acute", "pred_dtp"].sum()),
                         int(q.loc[q.group == "DTP", "pred_dtp"].sum())))
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.8), sharey=True)
    fig.subplots_adjust(left=.095, right=.98, bottom=.25, top=.80, wspace=.18)
    for ix, ax in enumerate(axes):
        for j, method in enumerate(methods):
            points = [r[ix+2] for r in rows if r[1] == method]
            ax.plot(np.arange(4)+(-.18 + j*.18), points, marker=["o","s","D"][j],
                    color=[NEUTRAL, BLUE, ACCENT][j], linestyle="none", markersize=6,
                    label=["Protected zero", "Relative zero", "Relative OOF threshold"][j])
        ax.set_xticks(range(4), LINES, rotation=25, ha="right")
        ax.set_ylim(-.3, 3.3)
        ax.set_yticks([0,1,2,3])
        plot_title(ax, group_labels[ix])
        clean(ax)
    axes[0].set_ylabel("DTP-positive calls / 3 (Control: / 2)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(.53,.04))
    fig.suptitle("Source-only cutoff comparison | Stage 49", x=.095, y=.98, ha="left", fontsize=11.6)
    label_footnote(fig, "Known Control reference required; same 8 held-out samples per line; no new external validation", x=.095, y=.0)
    save(fig, "08_source_threshold_transfer", out)

# ---------- Training-selected gene panels and descriptive expression ----------


def gene_panels(data: dict) -> dict[str, set[str]]:
    df = data["genes_27"]
    chosen = df[df.method == METHOD]
    return {fold: set(chosen.loc[chosen.held_out_line == fold, "gene"].astype(str)) for fold in LINES}


def gene_analysis(data: dict):
    panels = gene_panels(data)
    membership = {}
    for gene in set.union(*panels.values()):
        membership[gene] = tuple(gene in panels[fold] for fold in LINES)
    exact = defaultdict(list)
    for gene, bitmask in membership.items():
        exact[bitmask].append(gene)
    ordered = sorted(exact, key=lambda m:(-len(exact[m]), "".join(str(int(v)) for v in m)))
    records = []
    for n, bits in enumerate(ordered, 1):
        records.append(dict(
            intersection_index=n,
            folds=";".join(f for f,b in zip(LINES,bits) if b),
            membership_4bit="".join(str(int(b)) for b in bits),
            n_folds=sum(bits), n_genes=len(exact[bits]),
            genes_alphabetical=";".join(sorted(exact[bits]))
        ))
    intersections = pd.DataFrame(records)
    jaccard = pd.DataFrame(index=LINES, columns=LINES, dtype=float)
    for a in LINES:
        for b in LINES:
            jaccard.loc[a, b] = len(panels[a] & panels[b]) / len(panels[a] | panels[b])
    freq = Counter(sum(bits) for bits in membership.values())
    counts = {str(k):int(freq.get(k,0)) for k in range(1,5)}
    assert sum(int(k)*v for k,v in counts.items()) == 400
    stats = {
        "study": STUDY,
        "method": METHOD,
        "selection_protocol": "100 genes, training-only for each LOCO fold",
        "per_fold_selected": {f:len(panels[f]) for f in LINES},
        "n_unique_genes": len(membership),
        "n_exact_nonempty_intersections": len(exact),
        "n_genes_selected_in_k_folds": counts,
        "mean_pairwise_jaccard": float(np.mean([jaccard.loc[a,b] for a,b in combinations(LINES,2)])),
        "pairwise_jaccard": {a:{b:float(jaccard.loc[a,b]) for b in LINES} for a in LINES},
        "data_source": "outputs/dtp_specificity/27_selected_genes.csv",
        "source_csv_sha256": digest(PATHS["genes_27"]),
        "interpretation": "Exploratory source-only feature-selection stability; NOT validated biomarkers",
    }
    return panels, intersections, jaccard, stats


def reference_f6(kind: str) -> Path | None:
    filename = REF_F6_AUDIT if kind == "audit" else REF_F6_INTER
    for folder in [ROOT / "figures/candidates", ROOT / "figures/candidates/gene_panel_stability",
                   ROOT / "outputs/manuscript_figure_candidates"]:
        if (folder / filename).is_file():
            return folder / filename
    return None


def check_gene_history(intersections: pd.DataFrame, jaccard: pd.DataFrame, stats: dict) -> None:
    old_audit, old_csv = reference_f6("audit"), reference_f6("intersections")
    if old_audit:
        old = json.loads(old_audit.read_text(encoding="utf-8-sig"))
        ensure(old.get("csv_sha256") == stats["source_csv_sha256"],
               "F6 historical input SHA mismatch; refuse to silently change panels")
        for k in ("study", "method", "per_fold_selected", "n_unique_genes",
                  "n_exact_nonempty_intersections", "n_genes_selected_in_k_folds"):
            ensure(old[k] == stats[k], f"F6 historic audit mismatch: {k}")
        ensure(np.isclose(float(old["mean_pairwise_jaccard"]), stats["mean_pairwise_jaccard"], atol=1e-12),
               "F6 historic Jaccard mean mismatch")
        for a in LINES:
            for b in LINES:
                ensure(np.isclose(old["pairwise_jaccard"][a][b], stats["pairwise_jaccard"][a][b], atol=1e-12),
                       f"F6 historical Jaccard mismatch {a}/{b}")
        print("MATCH historical F6 audit", old_audit.relative_to(ROOT))
    if old_csv:
        earlier = pd.read_csv(old_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        keys = ("folds","membership_4bit","n_folds","n_genes","genes_alphabetical")
        ensure(set(keys).issubset(earlier.columns), "Previous F6 intersection keys absent")
        a = earlier[list(keys)].sort_values("membership_4bit").reset_index(drop=True)
        b = intersections[list(keys)].astype(str).sort_values("membership_4bit").reset_index(drop=True)
        ensure(a.equals(b), "F6 gene-level exact intersections differ from historical record")
        print("MATCH historical F6 intersections", old_csv.relative_to(ROOT))


def build_upset(data: dict, out: Path) -> None:
    _, rows, jac, stats = gene_analysis(data)
    check_gene_history(rows, jac, stats)
    patterns = rows["membership_4bit"].to_list()
    counts = rows["n_genes"].astype(int).to_numpy()
    allfour = rows["n_folds"].astype(int).eq(4).to_numpy()
    n = len(rows)
    x = np.arange(n)
    fig = plt.figure(figsize=(9.4, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.2], hspace=.08,
                          left=.13, right=.98, top=.82, bottom=.14)
    a = fig.add_subplot(gs[0])
    b = fig.add_subplot(gs[1], sharex=a)
    a.bar(x, counts, width=.7, color=[ACCENT if yes else BLUE for yes in allfour])
    for i,v in enumerate(counts):
        a.text(i, v + 1.2, str(v), ha="center", fontsize=8.5)
    a.set_ylim(0, max(counts)*1.18)
    a.set_ylabel("Genes per exact intersection")
    a.tick_params(axis="x", labelbottom=False, length=0)
    clean(a)
    ys = np.arange(4)[::-1]
    for i, bitmask in enumerate(patterns):
        chosen = [ys[k] for k, bit in enumerate(bitmask) if bit == "1"]
        col = ACCENT if allfour[i] else BLUE
        if len(chosen) > 1:
            b.plot([i,i], [min(chosen), max(chosen)], color=col, lw=1.6)
        for k, bit in enumerate(bitmask):
            b.scatter(i, ys[k], s=38 if bit == "1" else 19,
                      color=col if bit == "1" else "#D2D8DB", zorder=3)
    b.set_yticks(ys, LINES)
    b.set_xticks(x, np.arange(1,n+1))
    b.set_xlim(-.5,n-.5)
    b.set_ylim(-.6,3.6)
    b.set_xlabel("Exact intersection (largest first)")
    b.tick_params(axis="both", length=0)
    for s in b.spines.values():
        s.set_visible(False)
    fig.text(.13,.96,"Overlap of fold-selected gene panels",fontsize=11.5,ha="left")
    fig.text(.13,.91,"Four 100-gene panels | source LOCO | training-only selection",fontsize=8.2,color=MUTED,ha="left")
    label_footnote(fig,"Exploratory selection overlap; not independently validated DTP biomarkers",x=.13,y=.02)
    save(fig,"09_gene_panel_upset",out)


def build_jaccard(data: dict, out: Path) -> None:
    _, inter, jac, stats = gene_analysis(data)
    check_gene_history(inter, jac, stats)
    a = jac.to_numpy(dtype=float)
    diag = np.eye(len(LINES),dtype=bool)
    masked = np.ma.masked_where(diag, a)
    cmap = LinearSegmentedColormap.from_list("blue_stone",["#F4F6F6","#A5B8C0", BLUE])
    cmap.set_bad("#F2F3F2")
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    fig.subplots_adjust(left=.18,right=.87,top=.82,bottom=.19)
    im = ax.imshow(masked, interpolation="nearest", cmap=cmap, vmin=0, vmax=.30)
    ax.set_xticks(range(4),LINES,rotation=26,ha="right")
    ax.set_yticks(range(4),LINES)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,"—" if i==j else f"{a[i,j]:.2f}",ha="center",va="center",
                    color=MUTED if i==j else INK,fontsize=9)
    ax.set_xticks(np.arange(-.5,4,1),minor=True)
    ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which="minor",color="white",lw=2.5)
    ax.tick_params(which="both",length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im,ax=ax,fraction=.047,pad=.06)
    cb.set_label("Jaccard (different folds)",fontsize=8.3)
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(False)
    fig.text(.18,.96,"Pairwise overlap of selected genes",fontsize=11.4)
    fig.text(.18,.907,"Off-diagonal Jaccard; diagonal intentionally omitted",fontsize=8.1,color=MUTED)
    save(fig,"10_gene_panel_jaccard",out)


def build_frequency(data: dict, out: Path) -> None:
    _, inter, jac, stats = gene_analysis(data)
    check_gene_history(inter,jac,stats)
    values = [stats["n_genes_selected_in_k_folds"][str(k)] for k in range(1,5)]
    fig,ax=plt.subplots(figsize=(6.8,3.6))
    fig.subplots_adjust(left=.19,right=.94,top=.79,bottom=.23)
    y = np.arange(4)
    ax.barh(y,values,height=.58,color=[NEUTRAL,BLUE,BLUE,ACCENT])
    ax.set_yticks(y,[f"{k} of 4 folds" for k in range(1,5)])
    ax.invert_yaxis()
    ax.set_xlim(0,max(values)*1.20)
    ax.set_xlabel("Number of unique genes")
    for i,n in enumerate(values):
        ax.text(n+max(values)*.015,i,str(n),va="center",fontsize=8.5)
    ax.tick_params(axis="y",length=0)
    clean(ax,grid="x")
    fig.text(.19,.96,"Gene selection frequency",fontsize=11.4)
    fig.text(.19,.89,"Number of source LOCO panels selecting a gene",fontsize=8.1,color=MUTED)
    save(fig,"11_gene_panel_selection_frequency",out)


def build_gene_heatmap(data: dict, out: Path) -> None:
    """Descriptive visualization of 22 all-fold genes; post hoc, not a test-set feature plot."""
    panels, inter, jac, stats = gene_analysis(data)
    common = sorted(set.intersection(*panels.values()))
    ensure(len(common) == stats["n_genes_selected_in_k_folds"]["4"], "Shared gene set mismatch")
    genes, ids, X, y, groups, _ = load_source_matrix()
    gene_index = {str(g):i for i,g in enumerate(genes)}
    absent = [g for g in common if g not in gene_index]
    ensure(not absent, f"Stage27 gene symbols absent from audited Stage21 matrix: {absent[:8]}")
    # All-gene ranks for each sample (same rank definition as the source exploratory analysis).
    # All 22 genes are chosen from historical *four-fold intersection*, which uses
    # all folds' training labels. This is visualization only, not leakage-free testing.
    col_idxs = np.array([gene_index[g] for g in common])
    sample_rank = rankdata(X, axis=1, method="average")[:,col_idxs] / X.shape[1]
    means = sample_rank.mean(axis=0)
    sds = sample_rank.std(axis=0, ddof=1)
    ensure(np.all(sds > 0), "At least one selected gene has zero cross-sample rank variance")
    z = ((sample_rank - means)/sds).T
    order = sorted(range(len(ids)), key=lambda i:(LINES.index(groups[i]), int(y[i]), ids[i]))
    z = z[:,order]
    sorted_states = [STATES[y[i]] for i in order]
    sorted_lines = [groups[i] for i in order]
    fig = plt.figure(figsize=(10.7,7.0))
    gs = fig.add_gridspec(2,1,height_ratios=[.28,5.4],left=.20,right=.89,top=.88,bottom=.12,hspace=.04)
    ann = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1],sharex=ann)
    from matplotlib.colors import ListedColormap
    codes = np.array([[STATES.index(s) for s in sorted_states]])
    ann.imshow(codes,aspect="auto",cmap=ListedColormap([COLORS[s] for s in STATES]),vmin=-.5,vmax=2.5)
    ann.set_yticks([])
    ann.set_xticks([])
    ann.tick_params(length=0)
    for sp in ann.spines.values():sp.set_visible(False)
    # Diverging color map shows within-gene centered ranks, NOT differential-expression logFC.
    cmap = LinearSegmentedColormap.from_list("centered",["#3C7185", "#F4F5F2", "#9C6759"])
    zlim = max(1.5,min(3.5,float(np.nanpercentile(np.abs(z),99))))
    im = ax.imshow(z,aspect="auto",cmap=cmap,vmin=-zlim,vmax=zlim,interpolation="nearest")
    ax.set_yticks(range(len(common)), common,fontsize=7.6)
    ax.set_xticks([4,13,22,31], LINES, fontsize=8.1)
    ann.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for pos in (8.5,17.5,26.5):
        ax.axvline(pos,color="white",lw=1.7)
        ann.axvline(pos,color="white",lw=1.7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    cb = fig.colorbar(im,ax=ax,fraction=.03,pad=.023)
    cb.set_label("Gene-wise z-score of sample rank",fontsize=8.0)
    cb.ax.tick_params(length=0,labelsize=7)
    cb.outline.set_visible(False)
    fig.text(.20,.975,"Commonly selected genes | source rank profiles",fontsize=11.3)
    fig.text(.20,.937,"22 genes selected in all four folds; samples ordered by line and condition",fontsize=8.2,color=MUTED)
    legend=[Line2D([0],[0],marker="s",lw=0,markersize=6,color=COLORS[s],label=s) for s in STATES]
    fig.legend(handles=legend,ncol=3,frameon=False,fontsize=7.6,loc="lower left",bbox_to_anchor=(.19,.025))
    label_footnote(fig,"Descriptive/post-hoc gene set; NOT unbiased evaluation or evidence of DTP-specific markers",x=.20,y=.005)
    save(fig,"12_source_common_gene_rank_heatmap",out)


def build_boxplots(data: dict, out: Path) -> None:
    """One compact 4-panel plot: raw observations + median and min–max; n=3 per cell."""
    source = checked_source(data["source_38"],PROTECTED)
    allv = source.margin.to_numpy(dtype=float)
    lo,hi=min(0,float(allv.min())),max(0,float(allv.max()))
    pad=max((hi-lo)*.08,.08)
    fig,axes=plt.subplots(2,2,figsize=(9.2,6.2),sharey=True)
    fig.subplots_adjust(left=.105,right=.98,top=.89,bottom=.13,hspace=.32,wspace=.13)
    for ax,line in zip(axes.flat,LINES):
        part = source[source.held_out_line==line]
        for j,state in enumerate(STATES):
            sample = part[part.true_state==state].sort_values("gsm")
            vals = sample.margin.to_numpy(dtype=float)
            med = float(np.median(vals))
            ax.vlines(j,float(np.min(vals)),float(np.max(vals)),color=COLORS[state],lw=2.2,alpha=.60,zorder=1)
            ax.hlines(med,j-.19,j+.19,lw=1.65,color=INK,zorder=2)
            offsets=np.array([-.11,0,.11])
            ax.scatter(j+offsets,vals,s=42,marker=MARKERS[state],color=COLORS[state],
                       edgecolor="white",linewidth=.65,zorder=4)
        ax.axhline(0,color=THRESH,linestyle=(0,(4,3)),linewidth=.9)
        ax.set_xticks(range(3),STATES)
        ax.set_xlim(-.47,2.47)
        ax.set_ylim(lo-pad,hi+pad)
        plot_title(ax,line)
        clean(ax)
    for ax in axes[:,0]:ax.set_ylabel("DTP decision margin")
    fig.suptitle("Individual source LOCO scores | contrast-protected",x=.105,y=.985,ha="left",fontsize=11.5)
    label_footnote(fig,"Dots = observed samples (n=3 per state/line); bar = median; vertical line = min–max; fixed cutoff = 0",x=.105,y=.027)
    save(fig,"13_source_score_individual_points",out)

# ---------- One-file command-line controller ----------

BUILDERS: dict[str, tuple[str, Callable]] = {
    "f1": ("01_source_decision_transitions", build_f1),
    "f2": ("02_source_dtp_calls_bubblemap", build_f2),
    "f3": ("03_source_paired_margins", build_f3),
    "f4": ("04_external_paired_margins", build_f4),
    "f5": ("05_source_within_sample_rank_pca", build_f5),
    "roc": ("06_source_loco_roc", build_roc),
    "pr": ("07_source_loco_precision_recall", build_pr),
    "thresholds": ("08_source_threshold_transfer", build_thresholds),
    "upset": ("09_gene_panel_upset", build_upset),
    "jaccard": ("10_gene_panel_jaccard", build_jaccard),
    "frequency": ("11_gene_panel_selection_frequency", build_frequency),
    "gene_heatmap": ("12_source_common_gene_rank_heatmap", build_gene_heatmap),
    "boxplots": ("13_source_score_individual_points", build_boxplots),
}

DEPENDENCIES = {
    "f1": {"source_38"}, "f2": {"source_38"}, "f3": {"source_38"},
    "f4": {"external_39"}, "f5": {"matrix"}, "roc": {"source_38"},
    "pr": {"source_38"}, "thresholds": {"predictions_49"},
    "upset": {"genes_27"}, "jaccard": {"genes_27"},
    "frequency": {"genes_27"}, "gene_heatmap": {"genes_27", "matrix"},
    "boxplots": {"source_38"},
}


def source_fingerprints(data: dict, chosen: list[str]) -> dict[str, str]:
    requirements = set.union(*(DEPENDENCIES[key] for key in chosen))
    hashes = {}
    script = ROOT / "scripts/study_figure_gallery.py"
    if script.is_file():
        hashes["scripts/study_figure_gallery.py"] = digest(script)
    for key in sorted(requirements):
        if key == "matrix":
            genes, ids, X, y, groups, prov = load_source_matrix()
            # Byte-level fingerprints ensure audit consistency across repeated runs.
            hashes["stage21_matrix_float64"] = hashlib.sha256(
                np.ascontiguousarray(X).tobytes()).hexdigest()
            hashes["stage21_samples"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
            hashes["stage21_loader_py"] = digest(ROOT / "scripts/21_group_consistent_rank_selection.py")
        else:
            hashes[str(PATHS[key].relative_to(ROOT)).replace("\\", "/")] = digest(PATHS[key])
    return hashes


def build(data: dict[str, pd.DataFrame], chosen: list[str], *, force: bool, out: Path) -> None:
    assert all(k in BUILDERS for k in chosen)
    editorial_style()
    fingerprints = source_fingerprints(data, chosen)
    manifest_path = out / "figure_manifest.json"
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("schema") != "osimertinib-figure-studio-v1" and not force:
            raise ValueError("Unknown candidate manifest; use --force only after manual review")
        historic = previous.get("input_sha256", {})
        for name, actual_hash in fingerprints.items():
            if name in historic and historic[name] != actual_hash and not force:
                raise ValueError(
                    f"Source file changed since last figure run: {name}\n"
                    "No figures overwritten. Review inputs, then use --force if intentional."
                )
    elif out.is_dir():
        existing = [x for x in out.iterdir() if x.is_file()]
        if existing and not force:
            raise RuntimeError(
                "Editorial output directory contains files without our manifest. "
                "Inspect the folder before using --force."
            )

    # Audit all required data and source matrix BEFORE output directory is created.
    # Missing inputs fail closed; previous outputs remain untouched.
    if not out.is_dir():
        out.mkdir(parents=True, exist_ok=True)
    manifest = previous if previous else {
        "schema": "osimertinib-figure-studio-v1", "status": "exploratory_candidate_only",
        "palette": COLORS, "figure_outputs": {}, "input_sha256": {},
        "caveats": [
            "Repeatedly inspected source LOCO folds; not confirmatory validation",
            "Previously viewed external GSE255958; not untouched independent testing",
            "Acute (24h) and DTP (21d) labels are confounded by treatment duration",
            "Gene panel stability is not independently confirmed biological specificity",
        ]
    }
    manifest["input_sha256"].update(fingerprints)

    for key in chosen:
        stem, builder = BUILDERS[key]
        png, svg = out / f"{stem}.png", out / f"{stem}.svg"
        if png.is_file() and svg.is_file() and not force:
            # Do not treat an unregistered figure as reproducibly verified.
            if stem not in manifest["figure_outputs"]:
                raise RuntimeError(f"Figure exists but is not in manifest: {stem}; use --force after review")
            print(f"SKIP {key:13} existing PNG/SVG (input fingerprints unchanged)")
            continue
        if (png.exists() or svg.exists()) and not force:
            raise RuntimeError(f"Incomplete figure pair: {stem}; use --force after review")
        print(f"BUILD {key:13} -> {stem}")
        builder(data, out)
        if not png.is_file() or not svg.is_file():
            raise RuntimeError(f"Figure builder produced incomplete outputs: {stem}")
        manifest["figure_outputs"][stem] = {
            "builder": key,
            "png_sha256": digest(png),
            "svg_sha256": digest(svg),
            "depends_on": sorted(DEPENDENCIES[key]),
        }
        # Update after every successful builder; interrupted runs remain resumable.
        (out / "figure_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
    print("\nPIPELINE COMPLETED")
    print("Candidate folder:", out)
    print(f"Registered figures: {len(manifest['figure_outputs'])}")
    print("Original outputs/ and previous figures/candidates files were NOT touched.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true", help="Check all 10 CSVs + Stage21 matrix; no output files")
    mode.add_argument("--build-all", action="store_true", help="Audit and generate all candidate figures")
    mode.add_argument("--only", nargs="+", choices=tuple(BUILDERS), metavar="FIGURE",
                      help="Audit and generate chosen figure types (see --list)")
    mode.add_argument("--list", action="store_true", help="Show all available figure builders")
    p.add_argument("--force", action="store_true", help="Overwrite candidates only inside editorial_v1")
    args = p.parse_args()
    if args.list:
        for key, (stem, _) in BUILDERS.items():
            print(f"{key:14} {stem}")
        return 0
    if args.audit_only and args.force:
        p.error("--force does not apply to read-only auditing")

    chosen = list(BUILDERS) if args.build_all else list(dict.fromkeys(args.only or []))
    need_matrix = args.audit_only or args.build_all or any("matrix" in DEPENDENCIES[key] for key in chosen)

    # Strictly audit every available result; --build-all and --audit-only require all ten.
    data = audit_all(require_all=args.audit_only or args.build_all, load_source=need_matrix)
    if args.audit_only:
        print("AUDIT ONLY: NO FILES WRITTEN")
        return 0
    for key in chosen:
        missing = [n for n in DEPENDENCIES[key] if n != "matrix" and n not in data]
        if missing:
            raise FileNotFoundError(f"Cannot build {key}: missing audited input(s) {missing}")
    build(data, chosen, force=args.force, out=DEST)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
