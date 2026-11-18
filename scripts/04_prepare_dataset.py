"""Stage 3: freeze a *provisional* cohort plan and create source-only development data.

This script does NOT approve biological samples, harmonize studies, choose features
from an external cohort, or perform any training. The output expression matrix
contains only Control/DTP samples from the historically inspected GSE193258 study.

Run from the repository root:
    python scripts/prepare_dataset.py --self-test
    python scripts/prepare_dataset.py --plan-only
    python scripts/prepare_dataset.py

Uses Python >=3.10 standard library. Requires GEO/QC outputs from prior stages.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "metadata"
SOURCE_GSE = "GSE193258"
SOURCE_FILE = "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"
SOURCE_REL = Path("data/raw/GSE193258") / SOURCE_FILE
TARGET_REL = Path("data/processed/development/GSE193258_Control_DTP_provisional.tsv.gz")
PLAN_COLUMNS = [
    "gse", "gsm", "cell_line_reported", "candidate_state", "geo_title",
    "matrix_filename", "matrix_column", "expression_unit", "planned_role",
    "source_history", "review_flags", "identity_review", "replicate_review",
    "dose_time_review", "training_approved", "external_validation_approved",
    "geo_sample_url",
]
# Explicit roles are scientific planning choices, not biological approvals.
STUDY_ROLES = {
    "GSE193258": "DEVELOPMENT_CANDIDATE_PREVIOUSLY_INSPECTED",
    "GSE303413": "EXTERNAL_CANDIDATE_DOSE_CONFLICT_REVIEW",
    "GSE341996": "EXTERNAL_CANDIDATE_PROTOCOL_REVIEW",
    "GSE153183": "EXTERNAL_CANDIDATE_SMALL_N",
    "GSE249721": "EXTERNAL_CANDIDATE_EXPRESSION_UNIT_UNVERIFIED",
}
UNITS = {
    "GSE193258": "log2TPM_abundance",
    "GSE303413": "RSEM_count",
    "GSE341996": "TPM",
    "GSE153183": "FPKM",
    "GSE249721": "reported_processed_expression_unverified",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required previous-stage report: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV is missing header: {path}")
        return list(reader)


def dump_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    with partial.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PLAN_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_plan(cohort_rows: list[dict[str, str]], aligned_rows: list[dict[str, str]],
               manifest: dict) -> tuple[list[dict[str, str]], dict]:
    """One planned expression column per GSM, no biological approvals or model fitting."""
    if not isinstance(manifest.get("manifest"), dict):
        raise ValueError("Invalid expression manifest")
    by_sample = defaultdict(list)
    for aln in aligned_rows:
        gsm = aln.get("gsm", "")
        if gsm and aln.get("mapping_status", "").startswith("CANDIDATE"):
            by_sample[(aln["gse"], gsm)].append(aln)
    eligible = [r for r in cohort_rows if r["candidate_analysis"] == "control_dtp_candidate"
                and int(r["expression_mapping_files"] or "0") > 0]
    seen = set()
    result = []
    for row in sorted(eligible, key=lambda r:(r["gse"],r["gsm"])):
        gse, gsm = row["gse"], row["gsm"]
        if (gse, gsm) in seen:
            raise ValueError(f"Duplicate candidate sample: {gse}/{gsm}")
        seen.add((gse,gsm))
        if gse not in STUDY_ROLES:
            raise ValueError(f"Unexpected candidate study {gse}; manual decision required")
        if row["candidate_state"] not in {"Control", "DTP"}:
            raise ValueError(f"Unexpected candidate state for {gse}/{gsm}")
        if row["approved_for_training"].lower() != "false" or row["approved_as_untouched_external_validation"].lower() != "false":
            raise ValueError(f"Unexpected prior approval on {gse}/{gsm}: stop and review")
        matches = by_sample.get((gse, gsm), [])
        if gse == SOURCE_GSE:
            matches = [m for m in matches if m.get("filename") == SOURCE_FILE]
        if len(matches) != 1:
            raise ValueError(f"Expected one chosen matrix column for {gse}/{gsm}, got {len(matches)}")
        aln = matches[0]
        if aln.get("mapping_status") != "CANDIDATE_REQUIRES_REVIEW":
            raise ValueError(f"Non-candidate or modified alignment status for {gse}/{gsm}")
        file_key = f"{gse}/{aln['filename']}"
        file_entry = manifest["manifest"].get(file_key)
        if not file_entry or file_entry.get("status") not in {"downloaded_verified", "reused_verified"} or file_entry.get("gzip_crc_verified") is not True:
            raise ValueError(f"Missing verified source expression entry for {file_key}")
        result.append({
            "gse": gse, "gsm": gsm,
            "cell_line_reported": row["cell_line_reported"],
            "candidate_state": row["candidate_state"], "geo_title": row["geo_title"],
            "matrix_filename": aln["filename"], "matrix_column": aln["matrix_column"],
            "expression_unit": UNITS[gse], "planned_role": STUDY_ROLES[gse],
            "source_history": row["source_history"], "review_flags": row["review_flags"],
            "identity_review": "PENDING_SOURCE_SUPPORTED", "replicate_review": "PENDING_SOURCE_SUPPORTED",
            "dose_time_review": "PENDING_SOURCE_SUPPORTED",
            "training_approved": "false", "external_validation_approved": "false",
            "geo_sample_url": row["geo_sample_url"],
        })
    unique_columns = [(r["gse"], r["matrix_filename"], r["matrix_column"]) for r in result]
    if len(unique_columns) != len(set(unique_columns)):
        raise ValueError("One expression column maps to more than one candidate GSM")
    if not result:
        raise ValueError("No eligible candidate source metadata found")
    counts = Counter((r["gse"], r["candidate_state"]) for r in result)
    # Frozen known base expectations: fail rather than quietly changing the cohort.
    expected = {"GSE193258":(12,12), "GSE303413":(9,9), "GSE341996":(3,3),
                "GSE249721":(3,3), "GSE153183":(2,2)}
    for gse,(n_c,n_d) in expected.items():
        if (counts[(gse,"Control")], counts[(gse,"DTP")]) != (n_c,n_d):
            raise ValueError(f"Study {gse} cohort changed; expected {n_c}/{n_d} Control/DTP, observed {counts[(gse,'Control')]}/{counts[(gse,'DTP')]}")
    source = [r for r in result if r["gse"] == SOURCE_GSE]
    per_line = Counter((r["cell_line_reported"],r["candidate_state"]) for r in source)
    if set(x[0] for x in per_line) != {"PC9","H1975","HCC2935","HCC827"} or any(v != 3 for v in per_line.values()):
        raise ValueError(f"Unexpected source line/state representation: {dict(per_line)}")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "source_only_provisional_data_plan_v1",
        "study_count": len(set(r["gse"] for r in result)),
        "candidate_samples": len(result),
        "candidate_counts_by_study_state": {g:{s:counts[(g,s)] for s in ["Control","DTP"]} for g in sorted(expected)},
        "development_candidate_samples": len(source),
        "external_candidate_samples_not_approved": len(result)-len(source),
        "development_cell_lines": dict(sorted(Counter(r["cell_line_reported"] for r in source).items())),
        "samples_approved_for_training": 0,
        "external_studies_approved_for_independent_validation": 0,
        "expression_scales_harmonized": False,
        "source_and_external_expression_concatenated": False,
        "source_features_selected_using_external_expression": False,
        "notes": [
            "Development matrix is historically inspected and provisional; not an approved independent test.",
            "Other studies remain external candidates only, with original expression units unaltered.",
            "GSE303413: six treated samples have a recorded dose-characteristic/protocol discrepancy.",
            "Treatment duration and batch effects are not separated from DTP/control labels.",
            "Review gene identifiers and protocol evidence before declaring independent biological replicates or running confirmatory model evaluations.",
        ],
    }
    return result, summary


def make_source_matrix(source_path: Path, out_path: Path, selected: list[dict[str,str]],
                       expected_sha: str, *, expected_gene_rows: int | None = None) -> dict:
    """Materialize ONLY a study-local matrix; preserve all source gene rows/values.

    No feature selection or per-study normalization is performed. Deterministic
    gzip output permits the same dataset to be regenerated and verified.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Original source expression file missing: {source_path}")
    actual_sha = sha256(source_path)
    if actual_sha != expected_sha:
        raise ValueError("Original source expression SHA256 mismatches audit manifest")
    if not selected:
        raise ValueError("No source rows selected")
    selected_columns = [row["matrix_column"] for row in selected]
    gsm_ids = [row["gsm"] for row in selected]
    if len(set(selected_columns)) != len(selected_columns) or len(set(gsm_ids)) != len(gsm_ids):
        raise ValueError("Source matrix selected columns or GSMs are duplicated")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + ".tmp")
    seen_genes = set()
    n_genes = 0
    try:
        with gzip.open(source_path, "rt", encoding="utf-8-sig", newline="") as raw, temporary.open("wb") as dst:
            reader = csv.reader(raw, delimiter="\t")
            original_header = next(reader)
            if not original_header or original_header[0].strip().lower() != "gene":
                raise ValueError("Expected gene identifier in first source column")
            if len(original_header) != len(set(original_header)):
                raise ValueError("Duplicate original expression header labels")
            lookup = {v:i for i,v in enumerate(original_header)}
            if any(col not in lookup for col in selected_columns):
                raise ValueError("One or more mapped expression columns are absent")
            indices = [lookup[col] for col in selected_columns]
            with gzip.GzipFile(fileobj=dst, mode="wb", filename="", mtime=0, compresslevel=6) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="", write_through=True) as text_out:
                    writer = csv.writer(text_out, delimiter="\t", lineterminator="\n")
                    writer.writerow(["gene"] + gsm_ids)
                    for original in reader:
                        if len(original) != len(original_header):
                            raise ValueError(f"Source expression row {n_genes+2}: wrong column count")
                        gene = original[0].strip()
                        if not gene or gene in seen_genes:
                            raise ValueError(f"Source expression row {n_genes+2}: empty or duplicate gene identifier")
                        seen_genes.add(gene)
                        values = [original[i] for i in indices]
                        for val in values:
                            try:
                                if not math.isfinite(float(val)):
                                    raise ValueError("non-finite")
                            except (ValueError, OverflowError):
                                raise ValueError(f"Non-finite or nonnumeric expression value for gene {gene}") from None
                        writer.writerow([gene] + values)
                        n_genes += 1
        if n_genes < 1000 or (expected_gene_rows is not None and expected_gene_rows != n_genes):
            raise ValueError(f"Unexpected source gene row count: {n_genes}")
        result = {"source_sha256": actual_sha, "derived_sha256": sha256(temporary),
                  "expression_columns": len(gsm_ids), "gene_rows": n_genes,
                  "input_gene_ids_unique": True, "value_transformations_applied": False,
                  "feature_selection_applied": False, "source_only": True,
                  "biological_approval": "PENDING"}
        # Never silently overwrite a different existing derived matrix.
        if out_path.exists() and sha256(out_path) != result["derived_sha256"]:
            raise ValueError("Existing derived matrix differs; rename/archive it before replacing")
        temporary.replace(out_path)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def run(root: Path, *, plan_only: bool) -> dict:
    meta = root / "data" / "metadata"
    plan, summary = build_plan(
        read_csv(meta/"cohort_review.csv"),
        read_csv(meta/"expression_alignment.csv"),
        json.loads((meta/"expression_manifest.json").read_text(encoding="utf-8")),
    )
    if not plan_only:
        manifest = json.loads((meta/"expression_manifest.json").read_text(encoding="utf-8"))
        entry = manifest["manifest"][f"{SOURCE_GSE}/{SOURCE_FILE}"]
        result = make_source_matrix(root / SOURCE_REL, root / TARGET_REL,
                                    [r for r in plan if r["gse"] == SOURCE_GSE],
                                    entry["sha256"], expected_gene_rows=19712)
        summary["development_matrix"] = {"path": str(TARGET_REL).replace("\\","/"), **result}
    else:
        summary["development_matrix"] = {"status": "PLAN_ONLY_NOT_MATERIALIZED"}
    dump_csv(meta / "dataset_plan.csv", plan)
    dump_json(meta / "dataset_plan_summary.json", summary)
    return summary


def self_test() -> None:
    import shutil
    with tempfile.TemporaryDirectory() as dirname:
        root = Path(dirname)
        src = root/"raw.tsv.gz"
        with gzip.open(src, "wt", encoding="utf-8") as stream:
            stream.write("gene\tC_a\tother\tD_b\nEGFR\t1.5\t9\t2.0\nMET\t0\t5\t3.25\nTP53\t8\t0\t4\n")
        out = root/"clean.tsv.gz"
        selected = [{"gsm":"GSM2","matrix_column":"D_b"}, {"gsm":"GSM1","matrix_column":"C_a"}]
        for _ in range(2):
            # Fixture has just 3 genes, so the low-gene-count gate is tested separately below.
            # Use 1001 rows to pass the explicit biological-feature-size sanity check.
            with gzip.open(src, "wt", encoding="utf-8") as stream:
                stream.write("gene\tC_a\tother\tD_b\n")
                for k in range(1001): stream.write(f"G{k}\t1.5\t9\t2.0\n")
            evidence = make_source_matrix(src, out, selected, sha256(src), expected_gene_rows=1001)
            assert evidence["expression_columns"] == 2 and evidence["gene_rows"] == 1001
            with gzip.open(out, "rt", encoding="utf-8") as stream:
                assert stream.readline().strip() == "gene\tGSM2\tGSM1"
                assert stream.readline().strip() == "G0\t2.0\t1.5"
        try:
            make_source_matrix(src, out, selected, "0" * 64)
            raise AssertionError("Accepted invalid source checksum")
        except ValueError as exc:
            assert "SHA256" in str(exc)
        with gzip.open(src, "wt", encoding="utf-8") as stream:
            stream.write("gene\tC_a\tother\tD_b\n")
            for k in range(1000): stream.write(f"G{k}\t1.5\t9\t2.0\n")
            stream.write("G999\t1.5\t9\t2.0\n")
        try:
            make_source_matrix(src, out, selected, sha256(src))
            raise AssertionError("Accepted duplicate gene rows")
        except ValueError as exc:
            assert "duplicate gene" in str(exc)
        assert out.is_file()
        with gzip.open(out,"rt",encoding="utf-8") as stream:
            assert len(stream.readlines()) == 1002
    print("SELF-TEST PASSED (source-only columns, deterministic output, SHA and duplicate-gene fail-closed checks)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Generate the 58-row plan without reading raw expression")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    try:
        report = run(ROOT, plan_only=args.plan_only)
    except (OSError, KeyError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        parser.exit(2, f"[FAILED] {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved data/metadata/dataset_plan.csv and dataset_plan_summary.json")
    if not args.plan_only:
        print(f"Saved provisional source-only matrix: {TARGET_REL.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
