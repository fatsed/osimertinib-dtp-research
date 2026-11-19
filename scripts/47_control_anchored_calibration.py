#!/usr/bin/env python3
"""Stage 47 — SOURCE-only Control-Anchored Calibration (exploratory).

This stage DOES NOT reuse GSE249721 counts as source data, train models,
validate external cohorts, or automatically load a Stage-46 score file.

Input is a *new*, audited CSV of (a) outer LOCO held-out scores and
(b) nested inner-LOCO training-side OOF scores from the original source cohort.
Scores must use the SAME continuous, signed decision-score definition.

For each outer held-out cell line:
    raw_zero       = raw score > 0
    raw_tuned      = raw score > train-OOF-selected threshold
    anchored_zero  = score - reference_control_score > 0
    anchored_tuned = score - reference_control_score > train-OOF threshold

One known control from each cell line is fixed by sample ID (not outcomes or
scores) and excluded from evaluation and threshold fitting for ALL methods.

Run `python scripts/47_control_anchored_calibration.py --self-test` first.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REQUIRED = ("fold", "variant", "split", "sample_id", "cell_line", "group", "score")
METHODS = ("raw_zero", "raw_tuned", "anchored_zero", "anchored_tuned")
SOURCE_MODELS = {
    "base": "dual_contrast_rank_lr100__stage28_baseline",
    "protected": "dual_contrast_rank_lr100__contrast_protected_shrinkage_gamma1",
}


def _name_as_str(df: pd.DataFrame, columns: tuple[str, ...] = REQUIRED[:-1]) -> pd.DataFrame:
    for name in columns:
        df[name] = df[name].astype("string").str.strip()
        if df[name].isna().any() or df[name].eq("").any():
            raise ValueError(f"Missing/empty values in column {name!r}")
    return df


def _select_variants(df: pd.DataFrame, base: str | None, protected: str | None) -> pd.DataFrame:
    available = set(df["variant"].unique())
    if base is None and protected is None:
        if {"base", "protected"}.issubset(available):
            base, protected = "base", "protected"
        elif set(SOURCE_MODELS.values()).issubset(available):
            base, protected = SOURCE_MODELS["base"], SOURCE_MODELS["protected"]
        else:
            raise ValueError(
                "Cannot identify Base/Protected from variant names. "
                "Pass --base-variant and --protected-variant with exact values. "
                f"Found: {sorted(available)}"
            )
    elif not base or not protected:
        raise ValueError("Specify BOTH --base-variant and --protected-variant")
    if base == protected or base not in available or protected not in available:
        raise ValueError(
            f"Invalid Base/Protected selection: {base!r} / {protected!r}; "
            f"available={sorted(available)}"
        )
    subset = df[df["variant"].isin((base, protected))].copy()
    subset["variant"] = subset["variant"].map({base: "base", protected: "protected"})
    return subset


def load_and_validate(
    path: Path, *, positive: str = "DTC", expected_folds: int = 4,
    base_variant: str | None = None, protected_variant: str | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    if positive not in {"DTP", "DTC"}:
        raise ValueError("Select positive label explicitly as 'DTC' or 'DTP'")
    raw = pd.read_csv(path, dtype="string")
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    df = _name_as_str(raw[list(REQUIRED)].copy())
    df["score"] = pd.to_numeric(df["score"], errors="raise").astype(float)
    if not np.isfinite(df.score.to_numpy()).all():
        raise ValueError("All score values must be finite")
    df = _select_variants(df, base_variant, protected_variant)
    if set(df.split.unique()) != {"train_oof", "test"}:
        raise ValueError("split must contain exactly 'train_oof' and 'test'")
    found_groups = set(df.group.unique())
    if found_groups != {"Control", "Acute", positive}:
        raise ValueError(
            f"Expected exactly Control, Acute, {positive}; found {sorted(found_groups)}. "
            "Do NOT silently treat DTC and DTP as synonyms."
        )
    if df.duplicated(["fold", "variant", "split", "sample_id"]).any():
        raise ValueError("Duplicate (fold, variant, split, sample_id) rows")
    folds = sorted(df.fold.unique())
    if len(folds) != expected_folds:
        raise ValueError(f"Expected {expected_folds} held-out folds, found {folds}")
    if set(folds) != set(df.cell_line.unique()):
        raise ValueError("Every fold must be named for its held-out cell line")

    # Same biological cohort in every fold: no cherry-picked / missing samples.
    cohort_map: dict[str, set[tuple[str, str]]] = {}
    for fold in folds:
        for split in ("train_oof", "test"):
            a = df[(df.fold == fold) & (df.split == split) & (df.variant == "base")]
            b = df[(df.fold == fold) & (df.split == split) & (df.variant == "protected")]
            a_key = set(map(tuple, a[["sample_id", "cell_line", "group"]].itertuples(index=False, name=None)))
            b_key = set(map(tuple, b[["sample_id", "cell_line", "group"]].itertuples(index=False, name=None)))
            if not a_key or a_key != b_key:
                raise ValueError(f"Unequal samples or labels across variants in {fold}/{split}")
            expected_lines = {fold} if split == "test" else set(folds) - {fold}
            if set(a.cell_line.unique()) != expected_lines:
                raise ValueError(f"{fold}/{split}: wrong cell lines; expected {sorted(expected_lines)}")
            for line, block in a.groupby("cell_line", sort=True):
                if set(block.group.unique()) != {"Control", "Acute", positive}:
                    raise ValueError(f"{fold}/{split}/{line}: all 3 groups required")
                if int(block.group.eq("Control").sum()) < 2:
                    raise ValueError(f"{fold}/{split}/{line}: need >=2 Control samples")
                mapping = set(zip(block.sample_id.tolist(), block.group.tolist()))
                if line in cohort_map and mapping != cohort_map[line]:
                    raise ValueError(f"{line}: biological sample set/labels differ across outer folds")
                cohort_map[line] = mapping
        # Defensive check against target sample leakage in threshold training.
        for variant in ("base", "protected"):
            tr = set(df[(df.fold == fold) & (df.variant == variant) &
                        (df.split == "train_oof")].sample_id)
            te = set(df[(df.fold == fold) & (df.variant == variant) &
                        (df.split == "test")].sample_id)
            if tr & te:
                raise ValueError(f"{fold}/{variant}: outer train/test sample ID leakage")
    return df


def add_reference_scores(block: pd.DataFrame) -> pd.DataFrame:
    """Select smallest Control sample_id independently by split and line."""
    parts = []
    for (_, _), subset in block.groupby(["split", "cell_line"], sort=True):
        controls = subset.loc[subset.group == "Control", "sample_id"].sort_values()
        if controls.empty:
            raise ValueError("No reference Control")
        ref_id = str(controls.iloc[0])
        s_ref = float(subset.loc[subset.sample_id == ref_id, "score"].iloc[0])
        part = subset.copy()
        part["reference_id"] = ref_id
        part["reference_score"] = s_ref
        part["is_reference"] = part.sample_id.eq(ref_id)
        part["relative_score"] = part.score - s_ref
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Maximize train-only balanced accuracy; stable deterministic ties."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if len(x) != len(y) or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("Invalid threshold-training input")
    if not y.any() or y.all():
        raise ValueError("Need positive and negative TRAIN-OOF examples")
    vals = np.unique(x)
    mids = vals[:-1] + (vals[1:] - vals[:-1]) / 2
    candidates = np.r_[np.nextafter(vals[0], -np.inf), mids,
                       np.nextafter(vals[-1], np.inf)]
    best: tuple[tuple[float, float, float, float], float, float] | None = None
    for t in candidates:
        pred = x > t
        sensitivity = float(pred[y].mean())
        specificity = float((~pred[~y]).mean())
        ba = (sensitivity + specificity) / 2.0
        key = (round(ba, 12), round(sensitivity, 12), -abs(float(t)), -float(t))
        if best is None or key > best[0]:
            best = (key, float(t), ba)
    assert best is not None
    return best[1], best[2]


def _metrics(p: pd.DataFrame, positive: str) -> dict[str, float | int]:
    y = p.group.eq(positive).to_numpy(dtype=bool)
    hat = p.pred_positive.to_numpy(dtype=bool)
    c = p.group.eq("Control").to_numpy(dtype=bool)
    a = p.group.eq("Acute").to_numpy(dtype=bool)
    if not y.any() or y.all():
        raise ValueError("Each test fold needs positives and negatives")
    tpr = float(hat[y].mean())
    tnr = float((~hat[~y]).mean())
    return {
        "n": len(y), "n_positive": int(y.sum()), "n_negative": int((~y).sum()),
        "n_control": int(c.sum()), "n_acute": int(a.sum()),
        "tp": int((hat & y).sum()), "fn": int(((~hat) & y).sum()),
        "tn": int(((~hat) & (~y)).sum()), "fp": int((hat & (~y)).sum()),
        "control_fp": int((hat & c).sum()), "acute_fp": int((hat & a).sum()),
        "sensitivity": tpr, "specificity": tnr,
        "balanced_accuracy": (tpr + tnr) / 2,
        "roc_auc": float(roc_auc_score(y.astype(int), p.decision_score)),
    }


def evaluate(df: pd.DataFrame, positive: str = "DTC") -> tuple[pd.DataFrame, ...]:
    metrics: list[dict] = []
    predictions: list[pd.DataFrame] = []
    thresholds: list[dict] = []
    references: list[dict] = []
    for fold in sorted(df.fold.unique()):
        for variant in ("base", "protected"):
            block = add_reference_scores(df[(df.fold == fold) & (df.variant == variant)].copy())
            for r in block.loc[block.is_reference].itertuples():
                references.append({
                    "fold": fold, "variant": variant, "split": r.split,
                    "cell_line": r.cell_line, "reference_id": r.reference_id,
                    "reference_score": r.reference_score,
                })
            tr = block[(block.split == "train_oof") & (~block.is_reference)]
            te = block[(block.split == "test") & (~block.is_reference)]
            if tr.empty or te.empty:
                raise ValueError(f"Empty fit/evaluation partition: {fold}/{variant}")
            ytr = tr.group.eq(positive).to_numpy(dtype=bool)
            trained = {"raw_zero": 0., "anchored_zero": 0.}
            for method, col in (("raw_tuned", "score"), ("anchored_tuned", "relative_score")):
                t, ba = select_threshold(tr[col].to_numpy(dtype=float), ytr)
                trained[method] = t
                thresholds.append({
                    "fold": fold, "variant": variant, "method": method,
                    "threshold": t, "train_oof_balanced_accuracy": ba,
                    "n_train_oof_without_reference": len(tr),
                })
            for method in METHODS:
                col = "relative_score" if method.startswith("anchored") else "score"
                p = te.copy()
                p["method"] = method
                p["decision_score"] = p[col]
                p["threshold"] = trained[method]
                p["pred_positive"] = p.decision_score > trained[method]
                metrics.append({
                    "fold": fold, "variant": variant, "method": method,
                    "threshold": trained[method], **_metrics(p, positive),
                })
                predictions.append(p[[
                    "fold", "variant", "method", "sample_id", "cell_line", "group",
                    "score", "reference_id", "reference_score", "relative_score",
                    "decision_score", "threshold", "pred_positive",
                ]])
    m = pd.DataFrame(metrics)
    # A constant within-fold score shift cannot change AUC. Explicit assertion.
    for (fold, variant), b in m.groupby(["fold", "variant"]):
        if b.roc_auc.max() - b.roc_auc.min() > 1e-9:
            raise AssertionError(f"AUC invariance violated for {fold}/{variant}")
    refs = pd.DataFrame(references)
    ar = refs.loc[refs.variant == "base", ["fold", "split", "cell_line", "reference_id"]]
    br = refs.loc[refs.variant == "protected", ["fold", "split", "cell_line", "reference_id"]]
    pd.testing.assert_frame_equal(
        ar.sort_values(["fold", "split", "cell_line"]).reset_index(drop=True),
        br.sort_values(["fold", "split", "cell_line"]).reset_index(drop=True),
    )
    return m, pd.concat(predictions, ignore_index=True), pd.DataFrame(thresholds), refs


def _to_md_table(df: pd.DataFrame) -> str:
    """Pure-Python Markdown table; no extra 'tabulate' dependency."""
    fmt = df.copy()
    for col in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[col]):
            fmt[col] = fmt[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    lines = ["| " + " | ".join(map(str, fmt.columns)) + " |",
             "| " + " | ".join("---" for _ in fmt.columns) + " |"]
    for row in fmt.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_report(metrics: pd.DataFrame, positive: str, source_csv: str) -> str:
    cols = ["sensitivity", "specificity", "balanced_accuracy", "roc_auc"]
    macro = metrics.groupby(["variant", "method"], sort=False)[cols].mean().reset_index()
    lines = [
        "# Stage 47 — Control-Anchored Calibration: source-only LOCO",
        "", "**STATUS: exploratory / NOT external validation / NOT algorithmic novelty proof.**", "",
        f"Input: `{source_csv}`", f"Positive label: `{positive}`", "",
        "The held-out control selected by sample ID is excluded from ALL evaluations.",
        "Tuned thresholds are selected using TRAIN-only nested inner-LOCO OOF scores.",
        "Anchoring is one constant shift per held-out cell line and cannot improve its ROC-AUC.",
        "", "## Macro averages across 4 held-out cell lines", "",
        _to_md_table(macro), "", "## Per-fold paired comparison", "",
    ]
    for variant in ("base", "protected"):
        block = metrics[metrics.variant == variant].set_index(["fold", "method"])
        rows = []
        for fold in sorted(metrics.fold.unique()):
            raw, anchored = block.loc[(fold, "raw_tuned")], block.loc[(fold, "anchored_tuned")]
            rows.append({
                "fold": fold,
                "delta_TP": int(anchored.tp - raw.tp),
                "delta_Control_FP": int(anchored.control_fp - raw.control_fp),
                "delta_Acute_FP": int(anchored.acute_fp - raw.acute_fp),
                "delta_BA": float(anchored.balanced_accuracy - raw.balanced_accuracy),
            })
        lines += [f"### {variant}: anchored_tuned minus raw_tuned", "",
                  _to_md_table(pd.DataFrame(rows)), ""]
    lines += [
        "## Limitations and audit conditions", "",
        "- We cannot verify OOF provenance from CSV alone; the run requires explicit user attestation.",
        "- Train-OOF scores come from inner models trained on fewer cell lines than the final outer model. "
        "The score scales must be checked before interpreting transferred thresholds.",
        "- A known same-line control is required at inference; this is reference-assisted, not zero-shot.",
        "- Small per-line positive sample counts mean single decisions strongly affect sensitivity.",
        "- Source-only LOCO is NOT independent external cohort validation.",
        "- GSE249721 count-space probing in Stage 46 is not used as evidence of external performance here.",
        "- All methods share identical held-out evaluation samples (the anchor is removed everywhere).", "",
    ]
    return "\n".join(lines)


def _toy_csv(path: Path) -> None:
    # SYNTHETIC UNIT TEST ONLY; never used for scientific claims.
    lines = ("PC9", "HCC4006", "TestLine3", "TestLine4")
    offsets = {"PC9": -0.4, "HCC4006": 0.9, "TestLine3": 0.2, "TestLine4": -0.1}
    rows = []
    for fold in lines:
        for variant in ("base", "protected"):
            for line in lines:
                split = "test" if line == fold else "train_oof"
                for group, center in (("Control", -0.6), ("Acute", -0.1), ("DTC", 0.6)):
                    for j in range(3):
                        rows.append([fold, variant, split, f"{line}_{group}_{j}", line, group,
                                     center + offsets[line] + j * 0.02 + (0.01 if variant == "protected" else 0)])
    pd.DataFrame(rows, columns=REQUIRED).to_csv(path, index=False)


class Stage47Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "synthetic_only.csv"
        _toy_csv(self.path)
        self.df = load_and_validate(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reference_is_excluded_and_auc_preserved(self):
        m, pred, thresholds, refs = evaluate(self.df)
        self.assertEqual(len(m), 4 * 2 * 4)
        self.assertEqual(len(pred), 4 * 2 * 4 * 8)
        self.assertFalse((pred.sample_id == pred.reference_id).any())
        self.assertEqual(len(thresholds), 4 * 2 * 2)
        self.assertEqual(len(refs), 4 * 2 * 4)
        for _, part in m.groupby(["fold", "variant"]):
            self.assertLess(part.roc_auc.max() - part.roc_auc.min(), 1e-10)

    def test_test_labels_cannot_change_threshold(self):
        _, _, before, _ = evaluate(self.df)
        changed = self.df.copy()
        mask = (changed.split == "test") & changed.group.isin(["Acute", "DTC"])
        changed.loc[mask, "group"] = changed.loc[mask, "group"].map({"Acute": "DTC", "DTC": "Acute"})
        _, _, after, _ = evaluate(changed)
        pd.testing.assert_frame_equal(before, after)

    def test_mismatched_variant_rejected(self):
        data = self.df.copy()
        mask = (data.fold == "PC9") & (data.variant == "protected") & (data.sample_id == "PC9_DTC_0")
        data.loc[mask, "sample_id"] = "wrong_id"
        data.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "Unequal samples"):
            load_and_validate(self.path)

    def test_missing_oof_rejected(self):
        data = self.df[self.df.split == "test"].copy()
        data.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "train_oof"):
            load_and_validate(self.path)

    def test_reference_does_not_use_scores(self):
        base = add_reference_scores(self.df[(self.df.fold == "PC9") & (self.df.variant == "base")])
        bumped = self.df.copy()
        mask = (bumped.fold == "PC9") & (bumped.variant == "base") & (bumped.group == "Control")
        bumped.loc[mask, "score"] += 1000
        revised = add_reference_scores(bumped[(bumped.fold == "PC9") & (bumped.variant == "base")])
        self.assertEqual(set(base.loc[base.is_reference, "sample_id"]),
                         set(revised.loc[revised.is_reference, "sample_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic integrity tests (no research results)")
    parser.add_argument("--input", type=Path, help="Audited source-only scores CSV; NOT Stage-46 count-space probe")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/stage47_control_anchored"))
    parser.add_argument("--positive-label", choices=("DTC", "DTP"), default="DTC",
                        help="Stage-46 displays DTC; select DTP only if it truly matches the original source")
    parser.add_argument("--base-variant", help="Exact 'variant' value for dual-contrast base")
    parser.add_argument("--protected-variant", help="Exact 'variant' value for dual-contrast protected")
    parser.add_argument("--expected-folds", type=int, default=4)
    parser.add_argument("--confirm-oof-provenance", action="store_true",
                        help="I audited nested inner-LOCO and absence of outer-test data in ALL fitting")
    parser.add_argument("--confirm-signed-scores", action="store_true",
                        help="I audited score direction and that zero is valid as the RAW signed decision cutoff")
    args = parser.parse_args()

    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(Stage47Tests)
        )
        if not result.wasSuccessful():
            parser.exit(1, "SELF-TEST FAILED\n")
        print("SELF-TEST PASSED: synthetic integrity checks only; no scientific evaluation performed")
        return

    if args.input is None:
        parser.error("Required: --input (source-only outer-test + nested train_oof CSV). Use --self-test first.")
    if not args.confirm_oof_provenance:
        parser.error("Required after auditing provenance: --confirm-oof-provenance")
    if not args.confirm_signed_scores:
        parser.error("Required after auditing raw signed score semantics: --confirm-signed-scores")
    if args.expected_folds < 2:
        parser.error("--expected-folds must be >= 2")

    df = load_and_validate(args.input, positive=args.positive_label,
                           expected_folds=args.expected_folds,
                           base_variant=args.base_variant,
                           protected_variant=args.protected_variant)
    m, predictions, thresholds, refs = evaluate(df, args.positive_label)
    args.outdir.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.outdir / "metrics_by_fold.csv", index=False)
    predictions.to_csv(args.outdir / "sample_predictions.csv", index=False)
    thresholds.to_csv(args.outdir / "thresholds_train_only.csv", index=False)
    refs.to_csv(args.outdir / "control_references.csv", index=False)
    cols = ["sensitivity", "specificity", "balanced_accuracy", "roc_auc"]
    m.groupby(["variant", "method"], sort=False)[cols].mean().reset_index().to_csv(
        args.outdir / "macro_summary.csv", index=False
    )
    (args.outdir / "report.md").write_text(
        build_report(m, args.positive_label, str(args.input)), encoding="utf-8"
    )
    (args.outdir / "audit.json").write_text(
        json.dumps({
            "stage": 47, "protocol": "source-only outer LOCO + nested inner-LOCO OOF",
            "raw_score_signed_and_zero_cutoff_user_confirmed": True,
            "inner_oof_provenance_user_confirmed_not_verified_by_script": True,
            "reference_selection": "lexicographically first Control sample_id per cell line",
            "reference_excluded_from_all_eval_and_threshold_training": True,
            "test_labels_not_used_for_threshold": True,
            "source_input": str(args.input), "input_rows_after_variant_selection": len(df),
            "outer_folds": sorted(df.fold.unique().tolist()),
            "positive_group": args.positive_label,
            "GSE249721_external_validation": False,
            "publication_ready_generalization_claim": False,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Stage 47 SOURCE-only exploratory evaluation complete.")
    print(f"Saved 7 outputs to: {args.outdir.resolve()}")
    print(m.groupby(["variant", "method"], sort=False)[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
