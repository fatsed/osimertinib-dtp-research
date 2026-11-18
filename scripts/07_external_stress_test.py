"""07 — Exploratory source-to-study transfer stress test (NOT external validation).

Run from the repository root (Python 3.12; pinned requirements installed):
    python scripts/07_external_stress_test.py --self-test
    python scripts/07_external_stress_test.py --exploratory-only

Inputs are already-generated local files in data/metadata/ and data/processed/.
Never downloads data, changes raw files, harmonizes gene annotations, or claims
biological approval. External labels are read ONLY for the final report, not
for selecting features, fitting transformations, or tuning the classifier.

A single fixed source-trained model is tested separately on three GEO studies.
A sample-wise within-gene-universe percentile-rank transform is a provisional
scale-robust stress test, NOT a validated cross-study RNA-seq normalization.

Generated outputs (gitignored): outputs/external_stress/{transfer_summary.json,
transfer_predictions.csv,transfer_metrics.csv}.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
OUTPUT = ROOT / "outputs" / "external_stress"
TARGETS = ("GSE153183", "GSE303413", "GSE341996")
LABELS = {"Control": 0, "DTP": 1}
# Chosen from the previously executed source-only exploratory benchmark;
# never optimized against results in the target studies.
FIXED_HYPERPARAMETERS = {"model": "LogisticRegression", "k": 100, "C": 0.01}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_local_path(raw: str, directory: str) -> Path:
    value = Path(raw.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts or value.parts[:2] != ("data", directory):
        raise ValueError(f"Unsafe or unexpected data path: {raw!r}")
    return ROOT / value


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty or invalid CSV: {path}")
        return list(reader)


def load_matrix(path: Path, sha256: str, np):
    if not path.is_file():
        raise FileNotFoundError(f"Derived matrix is missing: {path}")
    if digest(path) != sha256:
        raise ValueError(f"SHA-256 mismatch — file may be outdated or edited: {path}")
    genes, values = [], []
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if len(header) < 3 or header[0].casefold().strip() != "gene":
            raise ValueError(f"Unexpected matrix header: {path}")
        if len(set(header[1:])) != len(header) - 1:
            raise ValueError(f"Duplicate sample columns: {path}")
        for line, row in enumerate(reader, 2):
            if len(row) != len(header) or not row[0]:
                raise ValueError(f"Malformed matrix row {line}: {path}")
            try:
                numbers = [float(x) for x in row[1:]]
            except ValueError as exc:
                raise ValueError(f"Non-numeric value at row {line}: {path}") from exc
            if any(not math.isfinite(x) for x in numbers):
                raise ValueError(f"Non-finite expression value at row {line}: {path}")
            genes.append(row[0]); values.append(numbers)
    if not genes or len(genes) != len(set(genes)):
        raise ValueError(f"Empty or repeated gene identifiers: {path}")
    return genes, header[1:], np.asarray(values, dtype=np.float64).T


def rank_samples(matrix, rankdata, np):
    """Within-sample percentile ranks; handles ties (many zero counts) fairly."""
    if matrix.ndim != 2 or matrix.shape[1] < 100:
        raise ValueError("Need a 2D matrix with >=100 common genes")
    return rankdata(matrix, axis=1, method="average") / matrix.shape[1]


def pipeline(ml):
    return ml["Pipeline"]([
        ("variance", ml["VarianceThreshold"]()),
        ("anova", ml["SelectKBest"](score_func=ml["f_classif"], k=FIXED_HYPERPARAMETERS["k"])),
        ("scale", ml["StandardScaler"]()),
        ("classifier", ml["LogisticRegression"](
            C=FIXED_HYPERPARAMETERS["C"], class_weight="balanced", solver="liblinear",
            max_iter=2000, random_state=0)),
    ])


def model_metrics(y, pred, ml) -> dict:
    cm = ml["confusion_matrix"](y, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(y)),
        "balanced_accuracy": float(ml["balanced_accuracy_score"](y, pred)),
        "control_recall": float(tn / (tn + fp)) if tn + fp else None,
        "dtp_recall": float(tp / (tp + fn)) if tp + fn else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def source_loco(X, protocol, ml):
    identity = {r["gsm"]: (r["state"], r["cell_line"]) for r in protocol["source_gsm_to_label"]}
    source_ids = [r["gsm"] for r in protocol["source_gsm_to_label"]]
    if len(source_ids) != 24 or len(set(source_ids)) != 24 or X.shape[0] != 24:
        raise ValueError("The frozen source cohort must contain exactly 24 GSMs")
    indices = {gsm: index for index, gsm in enumerate(source_ids)}
    if len(protocol["outer_folds"]) != 4:
        raise ValueError("Source protocol must contain exactly four LOCO folds")
    records = []
    tested_ids = []
    tested_lines = []
    for fold in protocol["outer_folds"]:
        train, test = fold["train_gsm"], fold["test_gsm"]
        if len(train) != 18 or len(test) != 6 or set(train) & set(test):
            raise ValueError("Source LOCO train/test split is invalid")
        if set(train) | set(test) != set(source_ids):
            raise ValueError("LOCO fold omits or duplicates GSMs")
        line = fold["held_out_cell_line"]
        tested_ids.extend(test)
        tested_lines.append(line)
        if any(identity[s][1] != line for s in test):
            raise ValueError("Held-out cell line does not match sample identity")
        est = pipeline(ml)
        yt = ml["np"].array([LABELS[identity[s][0]] for s in train])
        est.fit(X[[indices[s] for s in train]], yt)
        yp = est.predict(X[[indices[s] for s in test]])
        ytest = ml["np"].array([LABELS[identity[s][0]] for s in test])
        records.append({"held_out_cell_line": line, **model_metrics(ytest, yp, ml)})
    if Counter(tested_ids) != Counter(source_ids) or len(set(tested_lines)) != 4:
        raise ValueError("Each GSM and cell line must be held out exactly once")
    return records


def prepare_targets(np, cleaning, alignment, cohort, root: Path):
    """Align candidate column labels and GSMs, but do not approve biological identity."""
    by_study = {x["gse"]: x for x in cleaning["results"] if x["gse"] in TARGETS}
    if set(by_study) != set(TARGETS):
        raise ValueError("Not all three study-specific screened matrices are available")
    output = {}
    for gse in TARGETS:
        entry = by_study[gse]
        matrix_path = safe_local_path(entry["derived_file"], "processed")
        genes, columns, matrix = load_matrix(matrix_path, entry["derived_sha256"], np)
        # Original source column names must map 1:1 to exactly the prospective Control/DTP GSMs.
        maps = [r for r in alignment if r["gse"] == gse and r["filename"] == entry["source_filename"]]
        maps_by_col = {r["matrix_column"]: r for r in maps}
        if len(maps_by_col) != len(maps) or set(columns) != set(maps_by_col):
            raise ValueError(f"Missing or duplicate alignment column for {gse}")
        assigned = {r["gsm"]: r for r in cohort if r["gse"] == gse
                    and r["candidate_analysis"] == "control_dtp_candidate"}
        selected_gsm = [maps_by_col[c]["gsm"] for c in columns]
        if len(set(selected_gsm)) != len(columns) or set(selected_gsm) != set(assigned):
            raise ValueError(f"Unmapped/duplicated sample GSM or extra columns in {gse}")
        for col in columns:
            r = maps_by_col[col]
            if r["mapping_status"] != "CANDIDATE_REQUIRES_REVIEW":
                raise ValueError(f"Unexpected mapping status {gse}: {col}")
        states = [assigned[gsm]["candidate_state"] for gsm in selected_gsm]
        if Counter(states) != Counter({"Control": len(columns)//2, "DTP": len(columns)//2}):
            raise ValueError(f"Invalid target state distribution: {gse}")
        if any(assigned[s]["approved_for_training"] == "true" for s in selected_gsm):
            raise ValueError(f"Provisional test assumes no sample approval yet: {gse}")
        if (matrix < 0).any():
            raise ValueError(f"Negative expression values in external source: {gse}")
        output[gse] = {
            "genes": genes, "matrix": matrix, "columns": columns,
            "gsm": selected_gsm, "states": states, "sha256": entry["derived_sha256"],
            "flags": sorted({flag for gsm in selected_gsm for flag in assigned[gsm]["review_flags"].split(";") if flag}),
        }
    return output


def run():
    try:
        import numpy as np
        import sklearn
        from scipy.stats import rankdata
        from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("Dependencies missing; run: python -m pip install -r requirements.txt") from exc
    ml = locals()
    protocol_path = META / "evaluation_protocol.json"
    cleaning_path = META / "cleaning_report.json"
    align_path = META / "expression_alignment.csv"
    cohort_path = META / "cohort_review.csv"
    for p in (protocol_path, cleaning_path, align_path, cohort_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("source_study") != "GSE193258" or protocol.get("flags",{}).get("biology_approved_for_training") is not False:
        raise ValueError("This code expects the frozen exploratory GSE193258 protocol")
    source_file = safe_local_path(protocol["source_matrix"], "processed")
    src_genes, src_columns, src_values = load_matrix(source_file, protocol["source_matrix_sha256"], np)
    if src_columns != [r["gsm"] for r in protocol["source_gsm_to_label"]]:
        raise ValueError("Source matrix GSM order differs from evaluation protocol")
    if len(src_genes) != protocol["source_gene_count"]:
        raise ValueError("Source gene count differs from evaluation protocol")
    cleaning = json.loads(cleaning_path.read_text(encoding="utf-8"))
    targets = prepare_targets(np, cleaning, csv_rows(align_path), csv_rows(cohort_path), ROOT)
    # One frozen gene universe and one frozen source model for all three studies.
    common = set(src_genes)
    for study in TARGETS:
        common.intersection_update(targets[study]["genes"])
    kept_genes = [g for g in src_genes if g in common]
    if len(kept_genes) < 1000:
        raise ValueError(f"Shared gene universe too small: {len(kept_genes)}")
    source_idx = {g: i for i, g in enumerate(src_genes)}
    src_X = rank_samples(src_values[:, [source_idx[g] for g in kept_genes]], rankdata, np)
    labels = np.array([LABELS[r["state"]] for r in protocol["source_gsm_to_label"]], dtype=int)
    source_checks = source_loco(src_X, protocol, ml)
    model = pipeline(ml)
    model.fit(src_X, labels)  # Source only: no target expression or labels used here.
    # VarianceThreshold precedes SelectKBest, so selected genes require its mask.
    variance_mask = model.named_steps["variance"].get_support()
    remaining = [g for g, keep in zip(kept_genes, variance_mask) if keep]
    chosen_genes = [g for g, keep in zip(remaining, model.named_steps["anova"].get_support()) if keep]
    rows, summary_rows = [], []
    for gse in TARGETS:
        t = targets[gse]
        idx = {g: i for i, g in enumerate(t["genes"])}
        Xt = rank_samples(t["matrix"][:, [idx[g] for g in kept_genes]], rankdata, np)
        pred = model.predict(Xt)
        y = np.array([LABELS[x] for x in t["states"]], dtype=int)
        metrics = model_metrics(y, pred, ml)
        summary_rows.append({"study":gse, "n_genes_shared":len(kept_genes), **metrics})
        for gsm, state, yhat in zip(t["gsm"], t["states"], pred):
            rows.append({"study":gse, "gsm":gsm, "true_candidate_state":state,
                         "predicted_state":"DTP" if yhat else "Control", "correct":int(LABELS[state]==int(yhat)),
                         "biological_identity":"NOT_APPROVED"})
    report = {
        "status":"EXPLORATORY_POST_HOC_TRANSFER_STRESS_TEST_NOT_INDEPENDENT_VALIDATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_study": "GSE193258", "source_n": 24,
        "target_studies": {gse:{"n":len(targets[gse]["gsm"]), "review_flags":targets[gse]["flags"]} for gse in TARGETS},
        "shared_exact_gene_names":len(kept_genes),
        "gene_universe_uses_target_headers_only": True,
        "gene_symbol_semantics_harmonized": False,
        "source_model_fitted_once": True,
        "target_expression_used_for_parameter_tuning": False,
        "target_labels_used_for_training": False,
        "per_sample_percentile_rank_transform": True,
        "rank_transform_proven_for_cross_study_comparability": False,
        "fixed_hyperparameters": FIXED_HYPERPARAMETERS,
        "source_loco_rank_based_fixed_hyperparameters":source_checks,
        "external_metrics":summary_rows,
        "source_matrix_sha256":protocol["source_matrix_sha256"],
        "external_derived_sha256": {gse:targets[gse]["sha256"] for gse in TARGETS},
        "chosen_training_genes":chosen_genes,
        "limitations":[
            "These studies have been inspected during development and are NOT untouched validation.",
            "Source Control vs DTP labels are confounded with 24h vs 21d exposure, dose and culture.",
            "Each target's GSM mapping is title-based and not biologically approved.",
            "Exact shared gene names do not guarantee the same gene annotation/reference.",
            "Percentile ranks are a pragmatic sensitivity check, not validated batch correction.",
            "Same cell lines appear in source and some external studies; study independence is not cell-line independence.",
            "GSE303413 contains six treated sample labels with dose/protocol discrepancies.",
            "Target studies were not used for model selection; labels used only to score finished predictions.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    def write_csv(path, data):
        with path.open("w", encoding="utf-8", newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    write_csv(OUTPUT/"transfer_predictions.csv", rows)
    write_csv(OUTPUT/"transfer_metrics.csv", summary_rows)
    (OUTPUT/"transfer_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Provisional shared exact-name genes: {len(kept_genes):,}")
    print("Source nested-LOCO (fixed-k/C, rank input): ", [round(z["balanced_accuracy"],3) for z in source_checks])
    for row in summary_rows:
        print(f"{row['study']}: n={row['n']}, bACC={row['balanced_accuracy']:.3f}, DTP recall={row['dtp_recall']:.3f}")
    print("EXPLORATORY — NO BIOLOGICAL APPROVAL OR INDEPENDENT EXTERNAL VALIDATION")
    return report


def self_test():
    import numpy as np
    from scipy.stats import rankdata
    assert (rank_samples(np.array([[1.,1.,2.,3.]+list(range(96))]),rankdata,np).shape == (1,100))
    assert abs(rank_samples(np.array([[0.]*100]),rankdata,np)[0,0] - 0.505) < 1e-10
    assert LABELS["DTP"] == 1 and LABELS["Control"] == 0
    assert FIXED_HYPERPARAMETERS == {"model":"LogisticRegression","k":100,"C":0.01}
    import sklearn
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    ml = locals()
    ids = [f"GSM{i}" for i in range(24)]
    lines = ["A", "B", "C", "D"]
    source_protocol = {
        "source_gsm_to_label": [{"gsm": ids[i], "cell_line":lines[i//6],
                                 "state":"Control" if i%6<3 else "DTP"} for i in range(24)],
        "outer_folds": [],
    }
    for line in lines:
        test = [ids[i] for i in range(24) if lines[i//6] == line]
        train = [ids[i] for i in range(24) if lines[i//6] != line]
        source_protocol["outer_folds"].append({"held_out_cell_line":line,
                                                "train_gsm":train,"test_gsm":test})
    X = np.random.default_rng(42).normal(size=(24,300))
    X[:,0] += np.array([1 if i%6>=3 else 0 for i in range(24)])*4
    folds = source_loco(X, source_protocol, ml)
    assert len(folds) == 4 and all(f["n"] == 6 for f in folds)
    broken = json.loads(json.dumps(source_protocol))
    broken["outer_folds"][0]["test_gsm"][0] = ids[6]
    try:
        source_loco(X, broken, ml)
        raise AssertionError("Invalid LOCO test member was accepted")
    except ValueError:
        pass
    path = Path("data/processed/development/sample.tsv.gz")
    assert not path.is_absolute()  # Source path convention
    try:
        safe_local_path("../data/processed/file.gz", "processed")
        raise AssertionError("Unsafe path was accepted")
    except ValueError:
        pass
    try:
        safe_local_path("data/raw/file.gz", "processed")
        raise AssertionError("Raw matrix was accepted as processed")
    except ValueError:
        pass
    print("SELF-TEST PASSED (tie-aware ranks, fixed source-only model settings, strict paths)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--exploratory-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if not args.exploratory_only:
        parser.error("Requires --exploratory-only; this is NOT independent validation")
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
