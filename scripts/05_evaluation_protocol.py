"""Stage 3: validate the source-only cohort and freeze exploratory LOCO splits.

Use from project root:
    python scripts/evaluation_protocol.py --self-test
    python scripts/evaluation_protocol.py

Reads existing data/metadata reports and the GSE193258 provisional matrix.
Writes ONLY data/metadata/evaluation_protocol.json, loco_folds.csv and
study_gene_overlap.csv. Never modifies source/processed matrices, pools
expression scales, approves biological identity, or trains a model.

The protocol is an EXPLORATORY DRAFT pending independent scientific review.
No third-party dependencies; Python 3.10+.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_GSE = "GSE193258"
SOURCE_MATRIX = Path("data/processed/development/GSE193258_Control_DTP_provisional.tsv.gz")
SOURCE_RAW_FILENAME = "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"
STUDY_GENE_FILES = {
    "GSE153183": Path("data/processed/GSE153183/GSE153183_gene_symbols_screened.tsv.gz"),
    "GSE249721": Path("data/raw/GSE249721/GSE249721_all_samples_processed_data.tsv.gz"),
    "GSE303413": Path("data/processed/GSE303413/GSE303413_gene_symbols_screened.tsv.gz"),
    "GSE341996": Path("data/processed/GSE341996/GSE341996_gene_symbols_screened.tsv.gz"),
}
FOLD_COLUMNS = ["outer_held_out_cell_line", "partition", "gsm", "cell_line", "state"]
OVERLAP_COLUMNS = [
    "gse", "source_file", "status", "gene_rows", "unique_gene_ids",
    "duplicate_gene_rows", "exact_source_overlap", "pct_source_genes_in_study",
    "pct_study_genes_in_source", "note",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Empty header in {path}")
        return list(reader)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(data: dict) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def safe_write(path: Path, contents: bytes) -> None:
    """An existing output is immutable unless byte-for-byte reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError(
                f"Different frozen protocol already exists at {path}. "
                "Preserve the old version and review the changed inputs before replacing it."
            )
        return
    path.write_bytes(contents)


def csv_bytes(rows: list[dict[str, object]], fields: list[str]) -> bytes:
    import io
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def validate_source_rows(rows: list[dict[str, str]], alignment: list[dict[str, str]]) -> list[dict[str, str]]:
    source = [
        r for r in rows
        if r["gse"] == SOURCE_GSE and r["candidate_analysis"] == "control_dtp_candidate"
    ]
    if len(source) != 24:
        raise ValueError(f"Expected 24 provisional source samples; got {len(source)}")
    if len({r["gsm"] for r in source}) != 24:
        raise ValueError("Source sample GSM is not unique")
    expected_lines = {"H1975", "HCC2935", "HCC827", "PC9"}
    group_counts = Counter((r["cell_line_reported"], r["candidate_state"]) for r in source)
    expected_groups = {(line, state): 3 for line in expected_lines for state in ("Control", "DTP")}
    if group_counts != expected_groups:
        raise ValueError(f"Source line x state distribution mismatch: {group_counts}")
    approved = [r["gsm"] for r in source if r["approved_for_training"].lower() == "true"]
    if approved:
        raise ValueError("Some samples are marked approved by an unreviewed upstream table")
    if any(r["source_history"] != "previous_development" for r in source):
        raise ValueError("Historical development status is inconsistent")
    mapped = [
        row for row in alignment
        if row["gse"] == SOURCE_GSE and row["filename"] == SOURCE_RAW_FILENAME
    ]
    matched = Counter(r["gsm"] for r in mapped if r["gsm"] and r["mapping_status"] == "CANDIDATE_REQUIRES_REVIEW")
    for row in source:
        if matched[row["gsm"]] != 1:
            raise ValueError(f"Source mapping is not uniquely supported: {row['gsm']}")
    if len(mapped) != 60 or len({r["gsm"] for r in mapped}) != 60:
        raise ValueError("Source expression header candidate mapping does not cover 60 unique GEO records")
    return sorted(source, key=lambda r: r["gsm"])


def read_source_matrix(root: Path, source: list[dict[str, str]], plan: dict) -> set[str]:
    target = root / SOURCE_MATRIX
    if not target.is_file():
        raise FileNotFoundError(f"Provisional development matrix missing: {target}")
    meta = plan["development_matrix"]
    if meta["path"].replace("\\", "/") != SOURCE_MATRIX.as_posix():
        raise ValueError("Unexpected development matrix path in prior dataset plan")
    if sha256(target) != meta["derived_sha256"]:
        raise ValueError("Development matrix SHA-256 mismatch; aborting")
    if not (meta["source_only"] and not meta["feature_selection_applied"] and not meta["value_transformations_applied"]):
        raise ValueError("Provisional matrix is not untransformed and source-only")
    seen: set[str] = set()
    with gzip.open(target, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        if len(header) != 25 or header[0] != "gene":
            raise ValueError("Expected gene + 24 sample columns in development matrix")
        if len(set(header[1:])) != 24 or set(header[1:]) != {row["gsm"] for row in source}:
            raise ValueError("Development matrix columns do not match source GSM IDs")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header) or not row[0] or row[0] in seen:
                raise ValueError(f"Malformed/duplicate gene row {line_number}")
            seen.add(row[0])
            for raw_value in row[1:]:
                try:
                    val = float(raw_value)
                except ValueError:
                    raise ValueError(f"Non-numeric expression at row {line_number}") from None
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite expression at row {line_number}")
    if len(seen) != 19712 or len(seen) != meta["gene_rows"]:
        raise ValueError(f"Unexpected source-only gene universe: {len(seen)}")
    return seen


def make_loco(source: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict]]:
    lines = sorted({r["cell_line_reported"] for r in source})
    folds: list[dict] = []
    rows: list[dict[str, str]] = []
    for heldout in lines:
        train = [r for r in source if r["cell_line_reported"] != heldout]
        test = [r for r in source if r["cell_line_reported"] == heldout]
        if len(train) != 18 or len(test) != 6 or {r["gsm"] for r in train} & {r["gsm"] for r in test}:
            raise ValueError("LOCO outer fold leakage or unexpected size")
        inner = []
        for validation_line in sorted(set(lines) - {heldout}):
            inner_train = [r["gsm"] for r in train if r["cell_line_reported"] != validation_line]
            inner_val = [r["gsm"] for r in train if r["cell_line_reported"] == validation_line]
            if len(inner_train) != 12 or len(inner_val) != 6 or set(inner_train) & set(inner_val):
                raise ValueError("Nested inner fold overlap")
            inner.append({"held_out_training_cell_line": validation_line,
                          "train_gsm": sorted(inner_train), "validation_gsm": sorted(inner_val)})
        folds.append({"held_out_cell_line": heldout, "train_gsm": sorted(r["gsm"] for r in train),
                      "test_gsm": sorted(r["gsm"] for r in test), "inner_loco": inner})
        for partition, subset in (("train", train), ("test", test)):
            for r in subset:
                rows.append({"outer_held_out_cell_line": heldout, "partition": partition,
                             "gsm": r["gsm"], "cell_line": r["cell_line_reported"],
                             "state": r["candidate_state"]})
    return rows, folds


def read_gene_ids(path: Path) -> tuple[set[str], int]:
    """Gene-ID *diagnostic*, never a normalization or annotation mapping."""
    names: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        first = stream.readline()
        delim = "\t" if first.count("\t") >= first.count(",") else ","
        stream.seek(0)
        reader = csv.reader(stream, delimiter=delim)
        header = next(reader)
        if len(header) < 2:
            raise ValueError(f"Invalid gene matrix header: {path}")
        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(header) or not row[0].strip():
                raise ValueError(f"Invalid gene symbol at row {line_no}: {path}")
            names.append(row[0].strip())
    return set(names), len(names)


def gene_overlap_report(root: Path, source_genes: set[str]) -> list[dict]:
    out = []
    for gse, path in STUDY_GENE_FILES.items():
        target = root / path
        if not target.exists():
            out.append({"gse": gse, "source_file": path.as_posix(), "status": "NOT_FOUND",
                        "gene_rows": "", "unique_gene_ids": "", "duplicate_gene_rows": "",
                        "exact_source_overlap": "", "pct_source_genes_in_study": "",
                        "pct_study_genes_in_source": "", "note": "Run existing data download/QC first"})
            continue
        genes, nrows = read_gene_ids(target)
        intersect = len(genes & source_genes)
        out.append({"gse": gse, "source_file": path.as_posix(),
                    "status": "EXACT_NAME_OVERLAP_DIAGNOSTIC_ONLY", "gene_rows": nrows,
                    "unique_gene_ids": len(genes), "duplicate_gene_rows": nrows - len(genes),
                    "exact_source_overlap": intersect,
                    "pct_source_genes_in_study": round(100 * intersect / len(source_genes), 2),
                    "pct_study_genes_in_source": round(100 * intersect / len(genes), 2) if genes else 0,
                    "note": "No mapping, scaling, batch correction, pooling, or external label use; identifier semantics remain unverified"})
    return out


def build_protocol(root: Path) -> tuple[dict, list[dict], list[dict]]:
    meta = root / "data/metadata"
    required = [meta / name for name in ("cohort_review.csv", "expression_alignment.csv",
                                             "dataset_plan_summary.json", "expression_manifest.json")]
    for file in required:
        if not file.is_file():
            raise FileNotFoundError(file)
    cohort, alignment = read_csv(required[0]), read_csv(required[1])
    plan, manifest = read_json(required[2]), read_json(required[3])
    if plan["candidate_samples"] != 58 or plan["development_candidate_samples"] != 24:
        raise ValueError("Previous plan counts differ; do not silently alter protocol")
    if plan["samples_approved_for_training"] != 0:
        raise ValueError("Unexpected approval flags in plan")
    source = validate_source_rows(cohort, alignment)
    source_path = root / SOURCE_MATRIX
    source_genes = read_source_matrix(root, source, plan)
    if manifest["manifest"][f"{SOURCE_GSE}/{SOURCE_RAW_FILENAME}"]["sha256"] != plan["development_matrix"]["source_sha256"]:
        raise ValueError("Underlying GEO source SHA-256 differs from dataset plan")
    csv_rows, folds = make_loco(source)
    expected_external = {"GSE153183": 4, "GSE249721": 6, "GSE303413": 18, "GSE341996": 6}
    external_counts = Counter(r["gse"] for r in cohort if r["candidate_analysis"] == "control_dtp_candidate" and r["gse"] != SOURCE_GSE)
    if dict(external_counts) != expected_external:
        raise ValueError(f"External study counts changed: {external_counts}")
    gene_overlap = gene_overlap_report(root, source_genes)
    protocol = {
        "protocol_name": "GSE193258_source_only_nested_LOCO_exploratory_v1",
        "protocol_status": "DRAFT_EXPLORATORY_SOURCE_SUPPORTED_NOT_CONFIRMATORY",
        "source_study": SOURCE_GSE,
        "source_gene_count": len(source_genes),
        "source_matrix": SOURCE_MATRIX.as_posix(),
        "source_matrix_sha256": sha256(source_path),
        "input_report_sha256": {name: sha256(path) for name, path in zip(
            ["cohort_review", "expression_alignment", "dataset_plan_summary", "expression_manifest"], required)},
        "source_gsm_to_label": [
            {"gsm": r["gsm"], "state": r["candidate_state"],
             "cell_line": r["cell_line_reported"], "geo_url": r["geo_sample_url"]}
            for r in source
        ],
        "source_evidence": {
            "geo_series": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193258",
            "study_design": "4 cell lines, Control (24h DMSO) vs DTP (21d 500nM osimertinib), 3 reported biological replicates per cell line/arm",
            "source_evidence_scope": "Series overall design; samples mapped by GEO titles; biological identity not independently validated",
        },
        "outer_folds": folds,
        "evaluation_plan": {
            "target": "observed Control vs 21d drug-tolerant state, NOT future clinical response",
            "split_unit": "cell_line",
            "outer": "Leave-One-Cell-Line-Out across 4 source cell lines",
            "inner": "Leave-One-Cell-Line-Out across 3 training cell lines; tuning only inside outer training",
            "fit_within_fold_only": ["feature_selection", "imputation_if_ever_needed", "scaling", "model_tuning", "calibration"],
            "recommended_metrics": ["balanced_accuracy", "DTP_recall", "Control_recall", "per_line_confusion_matrix"],
            "dummy_baseline_required": True,
            "external_expression_used_in_tuning": False,
            "external_studies": {gse: {"candidate_samples": n, "role": "SEPARATE_POST_HOC_TRANSFER_STRESS_TEST_AFTER_REVIEW", "untouched_external_validation": False}
                                 for gse, n in sorted(expected_external.items())},
            "exclude_pdx_from_cell_line_training": True,
            "known_confounders": ["exposure_duration", "drug_dose", "cell_line", "study", "sequencing_processing_pipeline"],
        },
        "flags": {
            "biology_approved_for_training": False,
            "heldout_study_independent_validation_approved": False,
            "gene_symbol_harmonization_complete": False,
            "cross_study_expression_pooled": False,
            "no_models_trained": True,
            "historically_inspected_source_and_GSE255958": True,
            "six_GSE303413_treatment_dose_discrepancies_need_resolution": True,
        },
        "interpretation_limit": (
            "Treatment duration is confounded with class. This protocol estimates source-study "
            "observed-state discrimination, not causal DTP-specific drug adaptation or clinical prediction."
        ),
    }
    protocol["fingerprint_sha256"] = hashlib.sha256(canonical_json(protocol)).hexdigest()
    return protocol, csv_rows, gene_overlap


def run(root: Path) -> dict:
    protocol, folds, overlap = build_protocol(root)
    meta = root / "data/metadata"
    files = [
        (meta / "evaluation_protocol.json", canonical_json(protocol)),
        (meta / "loco_folds.csv", csv_bytes(folds, FOLD_COLUMNS)),
        (meta / "study_gene_overlap.csv", csv_bytes(overlap, OVERLAP_COLUMNS)),
    ]
    # Validate all planned writes before any new file is created.
    for path, content in files:
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Existing output differs: {path}; preserve prior protocol before revision")
    for path, content in files:
        safe_write(path, content)
    return {
        "status": protocol["protocol_status"],
        "outer_folds": len(protocol["outer_folds"]),
        "total_source_gsm": len(protocol["source_gsm_to_label"]),
        "source_gene_count": protocol["source_gene_count"],
        "external_gene_overlap_diagnostics_available": sum(r["status"] != "NOT_FOUND" for r in overlap),
        "external_gene_overlap_diagnostics_missing": [r["gse"] for r in overlap if r["status"] == "NOT_FOUND"],
        "protocol_fingerprint": protocol["fingerprint_sha256"],
        "outputs": [str(p.relative_to(root)) for p, _ in files],
        "modeling_approved": False,
    }


def self_test() -> None:
    rows = []
    for line in ("H1975", "HCC2935", "HCC827", "PC9"):
        for state in ("Control", "DTP"):
            for i in range(3):
                rows.append({"cell_line_reported": line, "candidate_state": state,
                             "gsm": f"{line}-{state}-{i}"})
    fold_rows, folds = make_loco(rows)
    assert len(folds) == 4 and len(fold_rows) == 96
    assert all(len(f["train_gsm"]) == 18 and len(f["test_gsm"]) == 6 and len(f["inner_loco"]) == 3 for f in folds)
    for f in folds:
        assert not set(f["train_gsm"]) & set(f["test_gsm"])
        for j in f["inner_loco"]:
            assert not set(j["train_gsm"]) & set(j["validation_gsm"])
            assert not set(j["validation_gsm"]) & set(f["test_gsm"])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "frozen.csv"
        content = csv_bytes(fold_rows, FOLD_COLUMNS)
        safe_write(p, content)
        safe_write(p, content)
        assert len(read_csv(p)) == 96
        try:
            safe_write(p, b"unapproved modification")
            raise AssertionError("Overwriting frozen folds was not blocked")
        except ValueError:
            pass
        g = Path(d) / "genes.csv.gz"
        with gzip.open(g, "wt", encoding="utf-8") as stream:
            stream.write('gene,Sample1\n"EGFR",2\n"MET",5\n')
        assert read_gene_ids(g) == ({"EGFR", "MET"}, 2)
    print("SELF-TEST PASSED (outer/inner disjoint LOCO splits, immutable outputs, csv gene IDs)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root, for auditing/test runs")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    try:
        result = run(args.root.resolve())
    except (KeyError, ValueError, FileNotFoundError, OSError, csv.Error) as e:
        parser.exit(2, f"PROTOCOL BLOCKED: {e}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
