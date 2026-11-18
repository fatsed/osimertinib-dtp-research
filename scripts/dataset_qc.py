"""Audit local GEO expression matrices against sample-level GEO metadata.

This script NEVER joins biological cohorts into a training matrix and NEVER
approves the biological identity of a replicate. It produces auditable mapping
candidates, unmatched columns and conservative per-matrix data-quality checks.

Run from the repository root:
    python scripts/dataset_qc.py --self-test
    python scripts/dataset_qc.py

Requires Python 3.10+ standard library. No network and no manual CSV downloads.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT / "data" / "metadata" / "geo_samples.csv"
MANIFEST_FILE = ROOT / "data" / "metadata" / "expression_manifest.json"
OUT = ROOT / "data" / "metadata"
RAW = ROOT / "data" / "raw"

ALIGNMENT_FIELDS = [
    "gse", "filename", "matrix_column", "gsm", "geo_title", "cell_line_raw",
    "treatment_raw", "match_rule", "mapping_status", "biological_review",
]
QC_FIELDS = [
    "gse", "filename", "expression_unit", "file_status", "checksum_matches_manifest",
    "gene_rows", "expression_columns", "matched_columns", "unmatched_columns",
    "ambiguous_columns", "duplicate_gene_ids", "duplicate_column_names",
    "non_numeric_cells", "nonfinite_cells", "negative_cells", "blank_cells",
    "zero_cells", "qc_status", "note", "qc_issue_codes",
    "duplicate_gene_examples", "non_numeric_examples",
    "distinct_duplicated_gene_ids", "duplicate_gene_frequencies",
    "missing_na_cells", "rows_containing_na", "missing_gene_examples",
    "missing_by_sample", "unexpected_nonnumeric_cells",
]


def key(s: str) -> str:
    """Unify typography only. Do not delete biologically meaningful tokens."""
    return re.sub(r"[\s_.-]+", "", s.strip().casefold().strip('"'))


def aliases(gse: str, row: dict[str, str]) -> list[tuple[str, str]]:
    """Candidate aliases rooted in the supplied GEO sample title, never labels inferred from data."""
    title = row["title"].strip()
    candidates: list[tuple[str, str]] = [(title, "exact_geo_title")]
    if gse == "GSE193258":
        bare = re.sub(r"\s*\[RNA-seq\]\s*$", "", title, flags=re.I)
        candidates.append((bare, "geo_title_without_assay_tag"))
        # A syntactic alias for the published source's abbreviated washout headers.
        # These mappings remain candidates, never biologically certified.
        abbreviated = re.sub(r"_(long|short)_washout(?:_\d+[dh])?_(\d+)$", r"_\1_wash_\2", bare, flags=re.I)
        if abbreviated != bare:
            candidates.append((abbreviated, "source_washout_abbreviation"))
    elif gse == "GSE341996":
        mat = re.fullmatch(r"(Par|DTP)\s+rep\s*(\d+)\s*,?\s*RNA-seq", title, flags=re.I)
        if mat and row.get("reported_cell_line", "").strip().upper().replace("-", "") == "PC9":
            candidates.append((f"PC9-{mat.group(1)}-rep{mat.group(2)}", "study_specific_title_alias"))
    elif gse == "GSE249721":
        candidates.append((title + ".genes.results", "study_specific_results_suffix"))
    elif gse == "GSE153183":
        mat = re.fullmatch(r"PC9\s+(Parental|DTP)\s+rep(\d+)", title, flags=re.I)
        if mat:
            state = "C-C" if mat.group(1).lower() == "parental" else "D-D"
            candidates.append((f"PC9-{state}{mat.group(2)}", "study_specific_title_alias"))
    # GSE243564 PDX sample numbers cannot be safely derived from GEO titles.
    return candidates


def map_columns(gse: str, columns: list[str], sample_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Fail closed on ambiguous mappings and double assignments."""
    by_alias: dict[str, list[tuple[dict[str, str], str]]] = defaultdict(list)
    for row in sample_rows:
        for name, rule in aliases(gse, row):
            if key(name):
                by_alias[key(name)].append((row, rule))
    rows: list[dict[str, str]] = []
    for column in columns:
        matches = by_alias.get(key(column), [])
        unique: dict[str, tuple[dict[str, str], str]] = {}
        for row, rule in matches:
            gsm = row["gsm"]
            if gsm not in unique or rule == "exact_geo_title":
                unique[gsm] = (row, rule)
        base = {"gse": gse, "matrix_column": column,
                "gsm": "", "geo_title": "", "cell_line_raw": "",
                "treatment_raw": "", "match_rule": "", "mapping_status": "UNMATCHED",
                "biological_review": "PENDING"}
        if len(unique) == 1:
            row, rule = next(iter(unique.values()))
            base.update({"gsm": row["gsm"], "geo_title": row["title"],
                         "cell_line_raw": row.get("reported_cell_line", ""),
                         "treatment_raw": row.get("reported_treatment", ""),
                         "match_rule": rule, "mapping_status": "CANDIDATE_REQUIRES_REVIEW"})
        elif len(unique) > 1:
            base["mapping_status"] = "AMBIGUOUS"
            base["match_rule"] = "multiple_sample_candidates"
        rows.append(base)
    counts = Counter(r["gsm"] for r in rows if r["gsm"])
    for row in rows:
        if row["gsm"] and counts[row["gsm"]] > 1:
            row["mapping_status"] = "AMBIGUOUS_DUPLICATE_GSM"
            row["match_rule"] += ";GSM_REUSED"
    return rows


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def detect_delimiter(first_line: str) -> str:
    # All currently selected GEO tables are CSV or TSV. No format guessing beyond that.
    if "\t" not in first_line and "," not in first_line:
        raise ValueError("No TSV/CSV separator in first line")
    return "\t" if first_line.count("\t") >= first_line.count(",") else ","


def scan_matrix(path: Path, entry: dict) -> tuple[list[str], dict]:
    """Stream the WHOLE gzip payload; reject wrong-size rows and invalid numeric values.

    CRC is verified when the stream is consumed. This scans small aggregate stats
    and never loads the gene-by-sample matrix into memory.
    """
    encoding = entry.get("header_encoding", "utf-8-sig")
    duplicate_genes = 0
    gene_counts: Counter[str] = Counter()
    nonnumeric = nonfinite = negative = blank = zero = 0
    missing_na = rows_with_na = unexpected_nonnumeric = 0
    missing_by_column: Counter[str] = Counter()
    missing_gene_examples: list[str] = []
    gene_rows = 0
    duplicate_gene_examples: list[str] = []
    non_numeric_examples: list[str] = []
    with gzip.open(path, "rt", encoding=encoding, errors="strict", newline="") as stream:
        first = stream.readline(2 * 1024 * 1024)
        if len(first) >= 2 * 1024 * 1024 or not first:
            raise ValueError("Missing or excessively long header")
        delim = detect_delimiter(first)
        header = next(csv.reader([first], delimiter=delim))
        header = [part.strip().lstrip("\ufeff") for part in header]
        if len(header) < 2:
            raise ValueError("No expression columns")
        if len(header) != int(entry.get("header_column_count", len(header))):
            raise ValueError("Header width differs from previously verified manifest")
        col_names = header[1:]
        duplicate_cols = len(col_names) - len(set(col_names))
        reader = csv.reader(stream, delimiter=delim)
        for line_idx, cells in enumerate(reader, start=2):
            if not cells or all(not s.strip() for s in cells):
                continue
            if len(cells) != len(header):
                raise ValueError(f"Malformed row {line_idx}: {len(cells)} fields; expected {len(header)}")
            gene = cells[0].strip()
            if not gene:
                raise ValueError(f"Empty gene identifier at row {line_idx}")
            gene_counts[gene] += 1
            if gene_counts[gene] > 1:
                duplicate_genes += 1
                if len(duplicate_gene_examples) < 8:
                    duplicate_gene_examples.append(f"row {line_idx}: {gene[:75]}")
            gene_rows += 1
            row_has_na = False
            for column_index, value in enumerate(cells[1:]):
                value = value.strip()
                if not value:
                    blank += 1
                    continue
                try:
                    v = float(value)
                except ValueError:
                    nonnumeric += 1
                    if value.upper() == "NA":
                        missing_na += 1
                        row_has_na = True
                        missing_by_column[col_names[column_index]] += 1
                        if len(missing_gene_examples) < 8 and gene not in missing_gene_examples:
                            missing_gene_examples.append(gene)
                    else:
                        unexpected_nonnumeric += 1
                    if len(non_numeric_examples) < 8:
                        non_numeric_examples.append(
                            f"row {line_idx}; gene={gene[:45]}; "
                            f"column={col_names[column_index][:55]}; value={value[:65]!r}"
                        )
                    continue
                if not math.isfinite(v):
                    nonfinite += 1
                elif v < 0:
                    negative += 1
                elif v == 0:
                    zero += 1
            if row_has_na:
                rows_with_na += 1
    repeated = [(gene, count) for gene, count in gene_counts.items() if count > 1]
    repeated.sort(key=lambda item: (-item[1], item[0]))
    return col_names, {
        "gene_rows": gene_rows,
        "expression_columns": len(col_names),
        "duplicate_gene_ids": duplicate_genes,
        "duplicate_column_names": duplicate_cols,
        "non_numeric_cells": nonnumeric,
        "nonfinite_cells": nonfinite,
        "negative_cells": negative,
        "blank_cells": blank,
        "zero_cells": zero,
        "duplicate_gene_examples": " | ".join(duplicate_gene_examples),
        "non_numeric_examples": " | ".join(non_numeric_examples),
        "distinct_duplicated_gene_ids": len(repeated),
        "duplicate_gene_frequencies": " | ".join(f"{gene[:60]} ({n} rows)" for gene, n in repeated[:12]),
        "missing_na_cells": missing_na,
        "rows_containing_na": rows_with_na,
        "missing_gene_examples": " | ".join(missing_gene_examples),
        "missing_by_sample": " | ".join(f"{col[:55]}={n}" for col, n in sorted(missing_by_column.items())),
        "unexpected_nonnumeric_cells": unexpected_nonnumeric,
    }


def load_metadata(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"gse", "gsm", "title", "reported_cell_line", "reported_treatment"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("geo_samples.csv is missing required fields")
        by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for row in reader:
            pair = (row["gse"], row["gsm"])
            if pair in seen:
                raise ValueError(f"Duplicate metadata GSM: {pair}")
            seen.add(pair)
            by_study[row["gse"]].append(row)
        return by_study


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def run(metadata_file: Path, manifest_file: Path, raw: Path, out: Path) -> dict:
    samples = load_metadata(metadata_file)
    report = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest = report.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Missing expression_manifest.json manifest dictionary")
    alignments: list[dict] = []
    qc: list[dict] = []
    for label, entry in sorted(manifest.items()):
        gse, name = entry.get("gse", ""), entry.get("filename", "")
        if not re.fullmatch(r"GSE\d+", gse) or Path(name).name != name or not name.endswith(".gz"):
            raise ValueError(f"Unsafe or malformed manifest entry: {label}")
        if entry.get("status") not in {"downloaded_verified", "reused_verified"}:
            continue
        path = raw / gse / name
        data = {"gse": gse, "filename": name,
                "expression_unit": entry.get("reported_expression_unit", ""),
                "file_status": "MISSING", "checksum_matches_manifest": False,
                "gene_rows": 0, "expression_columns": 0,
                "matched_columns": 0, "unmatched_columns": 0,
                "ambiguous_columns": 0,
                "duplicate_gene_ids": 0, "duplicate_column_names": 0,
                "non_numeric_cells": 0, "nonfinite_cells": 0,
                "negative_cells": 0, "blank_cells": 0, "zero_cells": 0,
                "qc_status": "BLOCKED", "note": "", "qc_issue_codes": "",
                "duplicate_gene_examples": "", "non_numeric_examples": "",
                "distinct_duplicated_gene_ids": 0, "duplicate_gene_frequencies": "",
                "missing_na_cells": 0, "rows_containing_na": 0,
                "missing_gene_examples": "", "missing_by_sample": "",
                "unexpected_nonnumeric_cells": 0}
        if not path.is_file():
            data["note"] = "Missing local expression matrix"
        else:
            try:
                if digest_file(path) != entry.get("sha256"):
                    raise ValueError("SHA256 differs from download manifest")
                data["checksum_matches_manifest"] = True
                names, stats = scan_matrix(path, entry)
                data.update(stats)
                matches = map_columns(gse, names, samples.get(gse, []))
                for match in matches:
                    match["filename"] = name
                alignments.extend(matches)
                data["matched_columns"] = sum(x["mapping_status"] == "CANDIDATE_REQUIRES_REVIEW" for x in matches)
                data["unmatched_columns"] = sum(x["mapping_status"] == "UNMATCHED" for x in matches)
                data["ambiguous_columns"] = len(matches) - data["matched_columns"] - data["unmatched_columns"]
                # Negative values are expected in some log-transformed expression
                # scales, but never in raw counts, TPM, FPKM or RPKM.
                is_log_scale = "log2" in str(entry.get("reported_expression_unit", "")).lower()
                content_flags = ("duplicate_gene_ids", "duplicate_column_names", "non_numeric_cells", "nonfinite_cells", "blank_cells")
                bad_content = any(stats[k] for k in content_flags) or (stats["negative_cells"] > 0 and not is_log_scale)
                reasons = [field.upper() for field in content_flags if stats[field] > 0]
                if stats["negative_cells"] > 0 and not is_log_scale:
                    reasons.append("NEGATIVE_IN_NONLOG_SCALE")
                if stats["gene_rows"] == 0:
                    reasons.append("EMPTY_GENE_MATRIX")
                data["qc_issue_codes"] = ";".join(reasons)
                if bad_content or stats["gene_rows"] == 0:
                    data["qc_status"] = "BLOCKED_QC"
                    data["note"] = "Expression matrix has flagged content; inspect before analysis"
                else:
                    data["qc_status"] = "TECHNICAL_SCAN_PASSED_NOT_BIOLOGICALLY_APPROVED"
                    data["note"] = "Do not pool expression units or use provisional mappings as ground truth"
                data["file_status"] = "VERIFIED_SHA256_AND_PARSED"
            except (OSError, ValueError, UnicodeError, csv.Error, EOFError) as exc:
                data["file_status"] = "ERROR"
                data["note"] = f"{type(exc).__name__}: {str(exc)[:250]}"
        qc.append(data)
        print(f"[{gse}] {name}: {data['file_status']} | {data['qc_status']} | {data['matched_columns']}/{data['expression_columns']} mapping candidates", flush=True)
    write_csv(out / "expression_alignment.csv", alignments, ALIGNMENT_FIELDS)
    write_csv(out / "matrix_qc.csv", qc, QC_FIELDS)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_files_scanned": len(qc),
        "matrices_technical_scan_passed": sum(x["qc_status"] == "TECHNICAL_SCAN_PASSED_NOT_BIOLOGICALLY_APPROVED" for x in qc),
        "total_expression_columns": sum(x["expression_columns"] for x in qc),
        "mapping_candidates_across_files_not_unique_samples": sum(x["matched_columns"] for x in qc),
        "unique_study_gsm_mapping_candidates": len({(x["gse"], x["gsm"]) for x in alignments
            if x["mapping_status"] == "CANDIDATE_REQUIRES_REVIEW"}),
        "unmatched_expression_columns": sum(x["unmatched_columns"] for x in qc),
        "ambiguous_expression_columns": sum(x["ambiguous_columns"] for x in qc),
        "biologically_approved_samples": 0,
        "study_independent_external_validation_established": False,
        "na_missing_token_cells": sum(x["missing_na_cells"] for x in qc),
        "unexpected_nonnumeric_cells": sum(x["unexpected_nonnumeric_cells"] for x in qc),
        "duplicate_gene_rows_across_matrices": sum(x["duplicate_gene_ids"] for x in qc),
        "notes": [
            "Mappings are title-based candidates, NOT verification of biological replicates.",
            "GSE193258 has two expression scales for the SAME samples; do not count twice.",
            "GSE243564 PDX data are a separate biological domain and must not be blindly pooled.",
            "Counts, RSEM counts, TPM, FPKM, RPKM and log2TPM must not be concatenated untransformed.",
            "Training/test cohorts must be frozen only AFTER manual source-supported review.",
        ],
    }
    temp = out / "dataset_qc_summary.json.tmp"
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(out / "dataset_qc_summary.json")
    return summary


# Conservative, non-destructive cleanup of the THREE blocked cell-line matrices.
# No model fitting, gene-ID renaming, label approval, or cross-study pooling here.
CLEAN_TARGETS = {
    "GSE153183": "GSE153183_FPKM_matrix.txt.gz",
    "GSE303413": "GSE303413_MN.P.rsem.count.tsv.gz",
    "GSE341996": "GSE341996_TPM_allSamples.csv.gz",
}


def clean_one_matrix(
    source: Path, target: Path, entry: dict, meta_rows: list[dict[str, str]]
) -> dict:
    """Write a loss-audited gene-symbol feature matrix, never altering source.

    Exclude ALL occurrences of non-unique symbols; their genomic identity is
    unresolved, so keeping first or summing would be arbitrary. Exclude rows
    wholly NA (no observed information). For partly missing or otherwise invalid
    rows, FAIL rather than silently imputing or dropping samples.
    """
    if entry.get("status") not in {"downloaded_verified", "reused_verified"}:
        raise ValueError("Input matrix was not downloaded and verified")
    if digest_file(source) != entry.get("sha256"):
        raise ValueError("Raw source checksum differs from verified manifest")
    names, stats = scan_matrix(source, entry)
    mapped = map_columns(entry["gse"], names, meta_rows)
    if len(mapped) != len(names) or any(
        m["mapping_status"] != "CANDIDATE_REQUIRES_REVIEW" for m in mapped
    ):
        raise ValueError("Expression columns lack unique sample-title mapping candidates")
    if stats["duplicate_column_names"] or stats["blank_cells"] or stats["nonfinite_cells"]:
        raise ValueError("Unexpected column, blank-cell or nonfinite-number problem")
    if stats["unexpected_nonnumeric_cells"]:
        raise ValueError("Unexpected nonnumeric values cannot be cleaned automatically")
    if stats["negative_cells"] and "log2" not in entry["reported_expression_unit"].lower():
        raise ValueError("Negative expression values in a nonlogarithmic matrix")

    encoding = entry.get("header_encoding", "utf-8-sig")
    with gzip.open(source, "rt", encoding=encoding, errors="strict", newline="") as stream:
        first = stream.readline(2 * 1024 * 1024)
        sep = detect_delimiter(first)
        header = next(csv.reader([first], delimiter=sep))
        if len(header) != len(names) + 1:
            raise ValueError("Header changed since QC scan")
        counts: Counter[str] = Counter()
        for row in csv.reader(stream, delimiter=sep):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header) or not row[0].strip():
                raise ValueError("Malformed source matrix row")
            counts[row[0].strip()] += 1

    duplicated = {gene for gene, n in counts.items() if n > 1}
    affected_duplicate_rows = sum(counts[gene] for gene in duplicated)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    kept = discarded_all_na = discarded_duplicate = 0
    try:
        with gzip.open(source, "rt", encoding=encoding, errors="strict", newline="") as stream:
            first = stream.readline(2 * 1024 * 1024)
            header = next(csv.reader([first], delimiter=sep))
            # filename='' and mtime=0 make derived gzip bytes reproducible.
            with tmp.open("wb") as raw_out:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, mtime=0) as zipped:
                    with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as txt:
                        writer = csv.writer(txt, delimiter="\t", lineterminator="\n")
                        writer.writerow(["gene"] + names)
                        for row in csv.reader(stream, delimiter=sep):
                            if not row or all(not cell.strip() for cell in row):
                                continue
                            if len(row) != len(header):
                                raise ValueError("Inconsistent source row width")
                            gene = row[0].strip()
                            if gene in duplicated:
                                discarded_duplicate += 1
                                continue
                            values = [s.strip() for s in row[1:]]
                            missing = [s.upper() == "NA" for s in values]
                            if all(missing):
                                discarded_all_na += 1
                                continue
                            if any(missing):
                                raise ValueError(f"Partially missing gene {gene}: review required")
                            try:
                                numerical = [float(s) for s in values]
                            except ValueError as exc:
                                raise ValueError(f"Unrecognized nonnumeric value for {gene}") from exc
                            if not all(math.isfinite(v) for v in numerical):
                                raise ValueError(f"Nonfinite value for {gene}")
                            if any(v < 0 for v in numerical) and "log2" not in entry["reported_expression_unit"].lower():
                                raise ValueError(f"Negative nonlog expression value for {gene}")
                            writer.writerow([gene] + values)
                            kept += 1
        if kept + discarded_all_na + discarded_duplicate != stats["gene_rows"]:
            raise ValueError("Cleanup row accounting mismatch")
        if discarded_duplicate != affected_duplicate_rows:
            raise ValueError("Duplicate gene accounting mismatch")
        _, verified = scan_matrix(tmp, {"header_encoding": "utf-8", "header_column_count": len(names) + 1})
        if (verified["gene_rows"] != kept or
                any(verified[k] for k in ("duplicate_gene_ids", "duplicate_column_names",
                                          "non_numeric_cells", "blank_cells", "nonfinite_cells")) or
                (verified["negative_cells"] and "log2" not in entry["reported_expression_unit"].lower())):
            raise ValueError("Post-write matrix verification failed")
        target_path_hash = digest_file(tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)

    return {
        "gse": entry["gse"], "source_filename": source.name,
        "source_sha256": entry["sha256"], "derived_file": str(target),
        "derived_sha256": target_path_hash,
        "input_gene_rows": stats["gene_rows"],
        "output_gene_rows": kept,
        "excluded_all_na_gene_rows": discarded_all_na,
        "excluded_ambiguous_duplicate_gene_rows": discarded_duplicate,
        "distinct_duplicate_gene_symbols": len(duplicated),
        "expression_columns": len(names),
        "all_columns_have_mapping_candidates": True,
        "biological_sample_identity_approved": False,
        "approved_for_training": False,
        "rule": ("Exclude ALL repeated gene-symbol occurrences, exclude wholly-NA "
                 "genes, block partial missing and invalid values; no imputation or renaming"),
        "note": ("This is a gene-symbol-level technical screening matrix. Source is "
                 "unchanged. Further provenance/gene annotation review is required."),
    }


def prepare_cleaned(
    metadata_file: Path, manifest_file: Path, raw: Path, processed: Path, out: Path
) -> dict:
    """Build auditable, locally derived versions of ONLY three QC-blocked matrices."""
    samples = load_metadata(metadata_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")).get("manifest", {})
    found = {}
    for entry in manifest.values():
        gse = entry.get("gse")
        if gse in CLEAN_TARGETS and entry.get("filename") == CLEAN_TARGETS[gse]:
            if gse in found:
                raise ValueError(f"Duplicate source for {gse}")
            found[gse] = entry
    if set(found) != set(CLEAN_TARGETS):
        raise ValueError(f"Missing requested source matrices: {set(CLEAN_TARGETS) - set(found)}")
    results = []
    for gse in sorted(CLEAN_TARGETS):
        entry = found[gse]
        source = raw / gse / entry["filename"]
        dest = processed / gse / f"{gse}_gene_symbols_screened.tsv.gz"
        record = clean_one_matrix(source, dest, entry, samples.get(gse, []))
        record["derived_file"] = str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest)
        results.append(record)
        print(f"[{gse}] kept={record['output_gene_rows']}; excluded duplicates={record['excluded_ambiguous_duplicate_gene_rows']}; excluded all-NA={record['excluded_all_na_gene_rows']}", flush=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "conservative_technical_gene_screen_only",
        "raw_files_changed": False,
        "harmonized_expression_scales": False,
        "biological_sample_review_complete": False,
        "ready_for_cross_study_training": False,
        "results": results,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cleaning_report.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return report


def self_test() -> None:
    """Exercise real file reading and mappings using small synthetic GZIP fixtures."""
    samples = [
        {"gse": "GSE193258", "gsm": "GSM1", "title": "PC9_osi_DTP_1 [RNA-seq]", "reported_cell_line": "PC9", "reported_treatment": "osimertinib DTPs 3w."},
        {"gse": "GSE341996", "gsm": "GSM2", "title": "Par rep1, RNA-seq", "reported_cell_line": "PC-9", "reported_treatment": "no treatment"},
        {"gse": "GSE303413", "gsm": "GSM3", "title": "MN.P.23.1", "reported_cell_line": "HCC827", "reported_treatment": "Control"},
        {"gse": "GSE249721", "gsm": "GSM4", "title": "HCC827Untreated1", "reported_cell_line": "HCC827", "reported_treatment": "DMSO"},
        {"gse": "GSE153183", "gsm": "GSM5", "title": "PC9 DTP rep1", "reported_cell_line": "", "reported_treatment": "2uM Osimertinib 9d"},
    ]
    cases = [
        ("GSE193258", "PC9_osi_DTP_1", "GSM1"),
        ("GSE341996", "PC9-Par-rep1", "GSM2"),
        ("GSE303413", "MN.P.23.1", "GSM3"),
        ("GSE249721", "HCC827Untreated1.genes.results", "GSM4"),
        ("GSE153183", "PC9-D-D1", "GSM5"),
    ]
    for gse, column, expected in cases:
        result = map_columns(gse, [column], [s for s in samples if s["gse"] == gse])
        assert result[0]["gsm"] == expected, (gse, result)
        assert result[0]["mapping_status"] == "CANDIDATE_REQUIRES_REVIEW"
    assert map_columns("GSE243564", ["Y5748-RNA"], [])[0]["mapping_status"] == "UNMATCHED"
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        (base / "data" / "raw" / "GSE153183").mkdir(parents=True)
        raw = base / "data" / "raw"
        out = base / "data" / "metadata"
        out.mkdir(parents=True)
        matrix = raw / "GSE153183" / "GSE153183_FPKM_matrix.txt.gz"
        with gzip.open(matrix, "wb") as stream:
            stream.write("Gene\tPC9-D-D1\nEGFR\t2.1\nMET\t0.0\n".encode("utf-16"))
        sample_file = out / "geo_samples.csv"
        write_csv(sample_file, [samples[-1]], list(samples[-1]))
        manifest = {"manifest": {"sample": {
            "gse": "GSE153183", "filename": matrix.name,
            "reported_expression_unit": "FPKM", "status": "downloaded_verified",
            "sha256": digest_file(matrix), "header_encoding": "utf-16", "header_column_count": 2,
        }}}
        manifest_file = out / "expression_manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        report = run(sample_file, manifest_file, raw, out)
        assert report["matrices_technical_scan_passed"] == 1
        assert report["unique_study_gsm_mapping_candidates"] == 1
        assert report["biologically_approved_samples"] == 0
        # A log2-transformed matrix may legitimately contain negative numbers.
        with gzip.open(matrix, "wb") as stream:
            stream.write("Gene\tPC9-D-D1\nEGFR\t-1.2\nMET\t0.0\n".encode("utf-16"))
        manifest["manifest"]["sample"]["sha256"] = digest_file(matrix)
        manifest["manifest"]["sample"]["reported_expression_unit"] = "log2TPM_abundance"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        log_report = run(sample_file, manifest_file, raw, out)
        assert log_report["matrices_technical_scan_passed"] == 1
        # The same negative observation in counts must block this matrix.
        manifest["manifest"]["sample"]["reported_expression_unit"] = "estimated_counts"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        count_report = run(sample_file, manifest_file, raw, out)
        assert count_report["matrices_technical_scan_passed"] == 0
        # Duplicated gene IDs and non-numeric tokens are separately diagnosed,
        # but neither is cleaned or accepted automatically.
        with gzip.open(matrix, "wb") as stream:
            stream.write("Gene\tPC9-D-D1\nEGFR\t2.1\nEGFR\tNA\n".encode("utf-16"))
        manifest["manifest"]["sample"]["sha256"] = digest_file(matrix)
        manifest["manifest"]["sample"]["reported_expression_unit"] = "FPKM"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        bad_content_report = run(sample_file, manifest_file, raw, out)
        assert bad_content_report["matrices_technical_scan_passed"] == 0
        with (out / "matrix_qc.csv").open(encoding="utf-8-sig", newline="") as stream:
            diagnostic = next(csv.DictReader(stream))
        assert "DUPLICATE_GENE_IDS" in diagnostic["qc_issue_codes"]
        assert "NON_NUMERIC_CELLS" in diagnostic["qc_issue_codes"]
        assert "row 3: EGFR" in diagnostic["duplicate_gene_examples"]
        assert "value='NA'" in diagnostic["non_numeric_examples"]
        assert diagnostic["missing_na_cells"] == "1"
        assert diagnostic["unexpected_nonnumeric_cells"] == "0"
        assert diagnostic["rows_containing_na"] == "1"
        assert diagnostic["distinct_duplicated_gene_ids"] == "1"
        assert "EGFR (2 rows)" in diagnostic["duplicate_gene_frequencies"]
        # Unrecognized text is NOT silently classified as a missing token.
        with gzip.open(matrix, "wb") as stream:
            stream.write("Gene\tPC9-D-D1\nEGFR\tbogus\nMET\tNA\n".encode("utf-16"))
        manifest["manifest"]["sample"]["sha256"] = digest_file(matrix)
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        unrecognized_report = run(sample_file, manifest_file, raw, out)
        assert unrecognized_report["unexpected_nonnumeric_cells"] == 1
        assert unrecognized_report["na_missing_token_cells"] == 1
        # An altered file is rejected even if it remains parseable.
        matrix.write_bytes(matrix.read_bytes() + b"tampered")
        bad = run(sample_file, manifest_file, raw, out)
        assert bad["matrices_technical_scan_passed"] == 0
    # Verify loss accounting and no silent imputation in the conservative cleaner.
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        raw = base / "input.tsv.gz"
        derived = base / "derived.tsv.gz"
        header = "Gene\tPC9-D-D1\n"
        values = "EGFR\t3.5\nMET\t4.1\nMET\t5.2\nBAGE2\tNA\nTP53\t0\n"
        with gzip.open(raw, "wb") as fp:
            fp.write((header + values).encode("utf-16"))
        entry = {
            "gse": "GSE153183", "status": "downloaded_verified",
            "sha256": digest_file(raw), "header_encoding": "utf-16",
            "header_column_count": 2, "reported_expression_unit": "FPKM",
        }
        result = clean_one_matrix(raw, derived, entry, [samples[-1]])
        assert result["input_gene_rows"] == 5
        assert result["output_gene_rows"] == 2
        assert result["excluded_all_na_gene_rows"] == 1
        assert result["excluded_ambiguous_duplicate_gene_rows"] == 2
        with gzip.open(derived, "rt", encoding="utf-8") as fp:
            assert fp.read().splitlines() == ["gene\tPC9-D-D1", "EGFR\t3.5", "TP53\t0"]
        before = digest_file(derived)
        # Partly missing genes must block; prior output must remain unchanged.
        with gzip.open(raw, "wb") as fp:
            fp.write("Gene\tPC9-D-D1\tPC9-C-C2\nEGFR\t1\tNA\n".encode("utf-16"))
        entry["sha256"] = digest_file(raw)
        entry["header_column_count"] = 3
        try:
            clean_one_matrix(raw, derived, entry, [samples[-1], {
                **samples[-1], "gsm": "GSM6", "title": "PC9 Parental rep2",
            }])
            raise AssertionError("Partial NA cleaning was incorrectly allowed")
        except ValueError as exc:
            assert "Partially missing" in str(exc), str(exc)
        assert digest_file(derived) == before
        # Restore 1-column fixture for repeatability check.
        with gzip.open(raw, "wb") as fp:
            fp.write((header + values).encode("utf-16"))
        entry["header_column_count"] = 2
        entry["sha256"] = digest_file(raw)
        # Repeat output is byte-stable for reproducibility.
        clean_one_matrix(raw, derived, entry, [samples[-1]])
        assert digest_file(derived) == before
        with gzip.open(raw, "wb") as fp:
            fp.write((header + "EGFR\t1\nMET\tNA\nTP53\t2\n").encode("utf-16"))
        entry["sha256"] = digest_file(raw)
        clean_one_matrix(raw, derived, entry, [samples[-1]])
        with gzip.open(raw, "wb") as fp:
            fp.write((header + "EGFR\t1\nTP53\tNA\n").encode("utf-16"))
        # Existing output cannot be mistaken for new raw results if source hash differs.
        try:
            clean_one_matrix(raw, derived, entry, [samples[-1]])
            raise AssertionError("Raw integrity check did not fail")
        except ValueError as exc:
            assert "checksum" in str(exc)
    print("SELF-TEST PASSED (UTF-16, NA-vs-invalid diagnostics, duplicates, aliases, checksum tampering, non-destructive cleanup)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--prepare-clean", action="store_true", help="Create non-destructive gene-symbol screened matrices for 3 blocked studies")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    for file in (SAMPLE_FILE, MANIFEST_FILE):
        if not file.is_file():
            parser.error(f"Missing prerequisite: {file}")
    if args.prepare_clean:
        result = prepare_cleaned(SAMPLE_FILE, MANIFEST_FILE, RAW, ROOT / "data" / "processed", OUT)
        print("Saved: data/metadata/cleaning_report.json")
        return 0
    result = run(SAMPLE_FILE, MANIFEST_FILE, RAW, OUT)
    print("\nSaved: data/metadata/expression_alignment.csv, matrix_qc.csv, dataset_qc_summary.json")
    print(json.dumps(result, indent=2))
    return 0 if result["matrices_technical_scan_passed"] == result["matrix_files_scanned"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
