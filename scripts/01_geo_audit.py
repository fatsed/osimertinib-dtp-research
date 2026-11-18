"""Download GEO series family metadata and build conservative sample-level audit tables.\n\nProduces geo_groups.csv and explicit dose conflicts within review_queue.csv.
Reports are written to data/metadata; raw GEO files remain in data/raw.

Run from the repository root:
    python scripts/geo_audit.py --self-test
    python scripts/geo_audit.py
    python scripts/geo_audit.py --offline
    python scripts/geo_audit.py --download-expression

Requires only Python 3.10+ standard library. No labels are inferred or approved.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import hashlib
from http.client import HTTPException
import json
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
OUT_ROOT = REPO_ROOT / "data" / "metadata"
ACCESSION = re.compile(r"^GSE\d+$")
GSM = re.compile(r"^GSM\d+$")
USER_AGENT = "osimertinib-dtp-research/0.1 (academic GEO metadata audit)"
MAX_COMPRESSED_BYTES = 100 * 1024 * 1024
MAX_EXPRESSION_BYTES = 250 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_HEADER_BYTES = 2 * 1024 * 1024
MAX_EXPRESSION_ATTEMPTS = 4

# Explicit accession/file eligibility for the first feasibility audit, NOT model approval.
# Only study-level GEO supplementary files listed in its original SOFT are permitted.
EXPRESSION_PLAN = {
    "GSE193258": {
        "GSE193258_RNAseq_estimated_counts.tsv.gz": "estimated_counts",
        "GSE193258_RNAseq_log2TPM_abundance.tsv.gz": "log2TPM_abundance",
    },
    "GSE341996": {"GSE341996_TPM_allSamples.csv.gz": "TPM"},
    "GSE303413": {"GSE303413_MN.P.rsem.count.tsv.gz": "RSEM_count"},
    # New, source-verified studies. File inclusion does NOT imply eligibility
    # for common labels, pooling or independent held-out validation.
    "GSE249721": {"GSE249721_all_samples_processed_data.tsv.gz": "reported_processed_expression_unverified"},
    "GSE153183": {"GSE153183_FPKM_matrix.txt.gz": "FPKM"},
    # Patient-derived xenografts are a SEPARATE experimental domain: never pool
    # with cell-line replicates or call them independent patients.
    "GSE243564": {
        "GSE243564_geneRpkm_YU-003.csv.gz": "RPKM_PDX",
        "GSE243564_geneRpkm_YU-006.csv.gz": "RPKM_PDX",
        "GSE243564_geneRpkm_YU-030.csv.gz": "RPKM_PDX",
    },
    # GSE285632 is metadata-only for now: supplementary data are XLSX and
    # not supported by the gzip TSV/CSV integrity scanner.
    # GSE243565 is also metadata-only here: special ASCL1-perturbed cohort,
    # with an XLS.GZ whose actual table format needs separate verification.
    # GSE255958 is a mixed-condition RAW.tar, deliberately not fetched here.
}


# Prior exposure is a RESEARCH-PROTOCOL decision, not an assertion about labels.
DEFAULT_COHORTS = {
    "GSE193258": "prior_development",
    "GSE255958": "previously_inspected_exploratory",
    "GSE341996": "new_candidate_unreviewed",
    "GSE303413": "new_candidate_unreviewed",
}

SAMPLE_COLUMNS = [
    "gse", "gsm", "title", "organism", "source_name", "library_strategy",
    "reported_cell_line", "reported_treatment", "reported_duration",
    "reported_condition", "characteristics_json", "treatment_protocol",
    "description", "raw_metadata_role", "scientific_review_status",
    "geo_sample_url",
]
SERIES_COLUMNS = [
    "gse", "title", "series_type", "reported_sample_records", "unique_gsm",
    "raw_metadata_role", "scientific_review_status", "metadata_status",
    "supplementary_files_json", "geo_series_url", "error",
]
REVIEW_COLUMNS = [
    "gse", "gsm", "issue", "details", "review_status",
]
GROUP_COLUMNS = [
    "gse", "reported_cell_line", "reported_treatment", "gsm_count",
    "example_gsm", "metadata_role", "scientific_review_status",
]
# Protocol doses are source statements; they MUST NOT silently replace sample labels.
CELL_LINE_DOSE_RE = re.compile(
    r"\b([A-Z][A-Z0-9-]+)\s+cells\s+were\s+treated\s+with\s+"
    r"(\d+(?:\.\d+)?)\s*nM\s+osimertinib\b", re.IGNORECASE
)
SAMPLE_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*nM(?=\b|_)", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def geo_soft_url(accession: str) -> str:
    if not ACCESSION.fullmatch(accession):
        raise ValueError(f"Not a GEO Series accession: {accession!r}")
    bucket = re.sub(r"\d{3}$", "nnn", accession)
    return (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/{accession}/soft/"
        f"{accession}_family.soft.gz"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def fetch_soft(accession: str, path: Path, refresh: bool = False) -> str:
    """Download bounded GEO SOFT. Existing files are reused; no silent overwrite."""
    if path.is_file() and not refresh:
        return "reused_local"
    url = geo_soft_url(accession)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temp_path = path.with_suffix(path.suffix + ".part")
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
            with urlopen(req, timeout=45) as response, temp_path.open("wb") as output:
                if response.status != 200:
                    raise OSError(f"Unexpected HTTP status: {response.status}")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > MAX_COMPRESSED_BYTES:
                    raise ValueError("Compressed metadata exceeds configured safety limit")
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > MAX_COMPRESSED_BYTES:
                        raise ValueError("Compressed metadata exceeded safety limit")
                    output.write(block)
                if size == 0:
                    raise ValueError("Empty GEO metadata response")
                if content_length is not None and size != int(content_length):
                    raise OSError("Incomplete GEO metadata download")
            with gzip.open(temp_path, "rb") as stream:
                if not stream.read(2):
                    raise ValueError("Gzip archive is empty")
            temp_path.replace(path)
            return "downloaded"
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            temp_path.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(f"Unable to download {accession}: {error}") from error
            time.sleep(attempt + 1)
    raise AssertionError("Unreachable")


def _single(values: list[str]) -> str:
    options = {v.strip() for v in values if v.strip()}
    return next(iter(options)) if len(options) == 1 else ""


def _field(fields: dict[str, list[str]], name: str) -> str:
    return _single(fields.get(name, []))


def _explicit_value(characteristics: list[str], keys: set[str]) -> str:
    results: list[str] = []
    for raw in characteristics:
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        normalized = re.sub(r"[\s_-]+", " ", key.strip().lower())
        if normalized in keys and value.strip():
            results.append(value.strip())
    return _single(results)


def parse_soft(path: Path, accession: str) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Parse actual submitter fields. SOFT expression tables are skipped.

    Keep original treatment and characteristics; do NOT map them to DTP/Acute.
    """
    series: dict[str, list[str]] = defaultdict(list)
    samples: list[dict[str, str]] = []
    current: dict[str, list[str]] | None = None
    current_gsm = ""
    within_table = False
    record_kind = ""

    def finish_sample() -> None:
        if current is None or not GSM.fullmatch(current_gsm):
            return
        chars = current.get("characteristics_ch1", [])
        sample = {
            "gse": accession,
            "gsm": current_gsm,
            "title": _field(current, "title"),
            "organism": _field(current, "organism_ch1"),
            "source_name": _field(current, "source_name_ch1"),
            "library_strategy": _field(current, "library_strategy"),
            "reported_cell_line": _explicit_value(chars, {"cell line", "cellline"}),
            "reported_treatment": _explicit_value(chars, {"treatment", "drug", "drug treatment", "treatment condition"}),
            "reported_duration": _explicit_value(chars, {"duration", "treatment duration", "exposure time", "treatment time", "timepoint", "time point"}),
            "reported_condition": _explicit_value(chars, {"condition", "state", "cell state", "phenotype", "treatment state"}),
            "characteristics_json": json.dumps(chars, ensure_ascii=False),
            "treatment_protocol": " | ".join(current.get("treatment_protocol_ch1", [])),
            "description": " | ".join(current.get("description", [])),
            "raw_metadata_role": DEFAULT_COHORTS.get(accession, "unspecified_candidate"),
            "scientific_review_status": "PENDING_REVIEW",
            "geo_sample_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={current_gsm}",
        }
        samples.append(sample)

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if line.startswith("^"):
                finish_sample()
                current, current_gsm = None, ""
                within_table = False
                if "=" not in line:
                    record_kind = ""
                    continue
                tag, identity = (part.strip() for part in line[1:].split("=", 1))
                record_kind = tag.upper()
                if record_kind == "SAMPLE" and GSM.fullmatch(identity):
                    current_gsm = identity
                    current = defaultdict(list)
                continue
            lowered = line.lower().strip()
            if lowered.endswith("_table_begin") and lowered.startswith("!"):
                within_table = True
                continue
            if lowered.endswith("_table_end") and lowered.startswith("!"):
                within_table = False
                continue
            if within_table or not line.startswith("!") or "=" not in line:
                continue
            key, value = (part.strip() for part in line[1:].split("=", 1))
            if record_kind == "SERIES" and key.startswith("Series_"):
                series[key[len("Series_"):].lower()].append(value)
            elif current is not None and record_kind == "SAMPLE" and key.startswith("Sample_"):
                current[key[len("Sample_"):].lower()].append(value)
    finish_sample()
    return {
        "title": _field(series, "title"),
        "series_type": " | ".join(series.get("type", [])),
        "supplementary_files": series.get("supplementary_file", []),
    }, samples


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def summarize_observed_groups(samples: list[dict[str, str]]) -> list[dict[str, object]]:
    """Group the unmodified GEO metadata; never infer ground-truth states."""
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        key = (row["gse"], row["reported_cell_line"], row["reported_treatment"])
        groups[key].append(row)
    return [
        {
            "gse": gse,
            "reported_cell_line": line,
            "reported_treatment": treatment,
            "gsm_count": len(rows),
            "example_gsm": rows[0]["gsm"],
            "metadata_role": rows[0]["raw_metadata_role"],
            "scientific_review_status": "PENDING_REVIEW",
        }
        for (gse, line, treatment), rows in sorted(groups.items())
    ]


def find_dose_discrepancies(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Flag an explicit dose conflict between GEO characteristics and treatment protocol.

    This is a narrow, fail-closed check: where there is no unambiguous
    cell-line-specific protocol dose, no matching or correction is attempted.
    """
    warnings: list[dict[str, str]] = []
    for row in samples:
        treatment = row["reported_treatment"]
        cell_line = row["reported_cell_line"]
        sample_doses = {float(value) for value in SAMPLE_DOSE_RE.findall(treatment)}
        protocol_doses: dict[str, set[float]] = defaultdict(set)
        for line, value in CELL_LINE_DOSE_RE.findall(row["treatment_protocol"]):
            protocol_doses[line.upper().replace("-", "")].add(float(value))
        expected = protocol_doses.get(cell_line.upper().replace("-", ""), set())
        if len(sample_doses) != 1 or len(expected) != 1:
            continue
        sample_dose, protocol_dose = next(iter(sample_doses)), next(iter(expected))
        if sample_dose != protocol_dose:
            warnings.append({
                "gse": row["gse"], "gsm": row["gsm"],
                "issue": "dose_disagreement_characteristics_vs_protocol",
                "details": (
                    f"Cell line {cell_line}: treatment characteristic {sample_dose:g} nM; "
                    f"protocol {protocol_dose:g} nM. MANUAL REVIEW REQUIRED; "
                    "no dose value was overwritten."
                ),
                "review_status": "PENDING_REVIEW",
            })
    return warnings


def audit(accessions: list[str], raw_root: Path, out_root: Path, offline: bool = False, refresh: bool = False) -> dict[str, object]:
    all_samples: list[dict[str, str]] = []
    series_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, str]] = []
    provenance: dict[str, object] = {}
    for accession in accessions:
        url = geo_soft_url(accession)
        local = raw_root / accession / f"{accession}_family.soft.gz"
        status, error = "", ""
        metadata: dict[str, object] = {}
        rows: list[dict[str, str]] = []
        try:
            if offline:
                if not local.is_file():
                    raise FileNotFoundError(f"Missing local metadata file: {local}")
                status = "reused_local_offline"
            else:
                status = fetch_soft(accession, local, refresh=refresh)
            metadata, rows = parse_soft(local, accession)
            ids = [row["gsm"] for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate GSM sample records")
            if not ids:
                raise ValueError("No GSM sample records in family SOFT")
            provenance[accession] = {
                "soft_url": url, "local_file": str(local), "sha256": sha256(local),
                "download_status": status, "checked_at_utc": utc_now(),
            }
            all_samples.extend(rows)
            for row in rows:
                missing = [k for k in ("reported_cell_line", "reported_treatment", "reported_duration") if not row[k]]
                if missing:
                    review_rows.append({"gse": accession, "gsm": row["gsm"],
                                        "issue": "missing_explicit_characteristics",
                                        "details": ", ".join(missing), "review_status": "PENDING_REVIEW"})
        except (OSError, ValueError, EOFError, RuntimeError) as exc:
            status = "ERROR"
            error = str(exc)
            provenance[accession] = {"soft_url": url, "download_status": status,
                                     "error": error, "checked_at_utc": utc_now()}
            review_rows.append({"gse": accession, "gsm": "", "issue": "metadata_unavailable",
                                "details": error, "review_status": "PENDING_REVIEW"})
        series_rows.append({
            "gse": accession,
            "title": metadata.get("title", ""),
            "series_type": metadata.get("series_type", ""),
            "reported_sample_records": len(rows),
            "unique_gsm": len({r["gsm"] for r in rows}),
            "raw_metadata_role": DEFAULT_COHORTS.get(accession, "unspecified_candidate"),
            "scientific_review_status": "PENDING_REVIEW",
            "metadata_status": status,
            "supplementary_files_json": json.dumps(metadata.get("supplementary_files", [])),
            "geo_series_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
            "error": error,
        })
        print(f"[{accession}] {status}, {len(rows)} GSM records" + (f" | {error}" if error else ""), flush=True)

    observed_groups = summarize_observed_groups(all_samples)
    dose_discrepancies = find_dose_discrepancies(all_samples)
    review_rows.extend(dose_discrepancies)
    write_csv(out_root / "geo_series.csv", series_rows, SERIES_COLUMNS)
    write_csv(out_root / "geo_samples.csv", all_samples, SAMPLE_COLUMNS)
    write_csv(out_root / "geo_groups.csv", observed_groups, GROUP_COLUMNS)
    write_csv(out_root / "review_queue.csv", review_rows, REVIEW_COLUMNS)
    summary: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "accessions_requested": accessions,
        "series_successful": sum(row["metadata_status"] != "ERROR" for row in series_rows),
        "series_failed": sum(row["metadata_status"] == "ERROR" for row in series_rows),
        "geo_gsm_records": len(all_samples),
        "sample_rows_requiring_scientific_review": len(all_samples),
        "review_queue_items": len(review_rows),
        "observed_metadata_groups": len(observed_groups),
        "explicit_dose_discrepancies": len(dose_discrepancies),
        "biologically_independent_replicates_verified": False,
        "external_validation_approved": False,
        "expression_data_downloaded": False,
        "limitations": [
            "GSM records are not automatically independent biological replicates.",
            "Submitter fields are preserved, not harmonized or approved as ground-truth labels.",
            "Different treatment durations and protocols may confound DTP/Acute comparisons.",
            "No expression matrices are downloaded in this metadata-first audit.",
            "Previously inspected cohorts are not untouched external validation.",
        ],
        "source": "NCBI GEO SOFT family metadata",
    }
    write_json(out_root / "provenance.json", provenance)
    write_json(out_root / "summary.json", summary)
    print(f"\nSaved reports to: {out_root}")
    return summary



def _approved_expression_urls(series: list[dict[str, str]], accessions: list[str]) -> list[dict[str, str]]:
    """Derive file URLs only from GEO SOFT, requiring an explicit allowlist."""
    by_acc = {row["gse"]: row for row in series}
    plan: list[dict[str, str]] = []
    for accession in accessions:
        allowed = EXPRESSION_PLAN.get(accession, {})
        if not allowed:
            print(f"[METADATA ONLY] {accession}: no eligible expression file on the validated download list")
            continue
        row = by_acc.get(accession)
        if row is None or row["metadata_status"] == "ERROR":
            raise ValueError(f"Cannot download {accession}: missing valid series metadata")
        supplementary = json.loads(row["supplementary_files_json"])
        if not isinstance(supplementary, list):
            raise ValueError(f"Invalid GEO supplementary file list for {accession}")
        seen: set[str] = set()
        for original in supplementary:
            parsed = urlparse(original)
            filename = parsed.path.rsplit("/", 1)[-1]
            if filename not in allowed:
                continue
            if filename in seen:
                raise ValueError(f"Duplicate GEO supplementary file: {accession}/{filename}")
            seen.add(filename)
            expected_dir = f"/geo/series/{accession[:-3]}nnn/{accession}/suppl/"
            if (parsed.scheme not in {"ftp", "https"} or parsed.netloc != "ftp.ncbi.nlm.nih.gov"
                    or parsed.path != expected_dir + filename or parsed.query or parsed.fragment):
                raise ValueError(f"Untrusted GEO supplementary location for {accession}/{filename}")
            plan.append({"gse": accession, "filename": filename,
                         "url": "https://ftp.ncbi.nlm.nih.gov" + parsed.path,
                         "reported_expression_unit": allowed[filename]})
        missing = set(allowed) - seen
        if missing:
            raise ValueError(f"Expected GEO supplementary file missing from {accession}: {sorted(missing)}")
    return plan


def _detect_expression_encoding(prefix: bytes) -> str:
    """Recognize text encodings, without treating arbitrary binary as valid expression."""
    if prefix.startswith(b"\xff\xfe\x00\x00"):
        return "utf-32"
    if prefix.startswith(b"\x00\x00\xfe\xff"):
        return "utf-32"
    if prefix.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if b"\x00" not in prefix:
        return "utf-8-sig"
    # Some submitters export UTF-16 without a BOM. Require a strong ASCII-text
    # signature; random binary must never be reinterpreted as a gene matrix.
    head = prefix[:min(len(prefix) // 2 * 2, 512)]
    if len(head) >= 32:
        even_zeros = head[::2].count(0)
        odd_zeros = head[1::2].count(0)
        half = len(head) // 2
        if odd_zeros >= 0.70 * half and even_zeros <= 0.05 * half:
            return "utf-16-le"
        if even_zeros >= 0.70 * half and odd_zeros <= 0.05 * half:
            return "utf-16-be"
    raise ValueError(
        "Gzip payload contains NUL bytes but is not recognized text "
        f"(first 16 decompressed bytes: {prefix[:16].hex()}); "
        "not approved as a tabular expression matrix"
    )


def _inspect_expression_gzip(path: Path) -> dict[str, object]:
    """Verify CRC, then inspect text encoding and delimited header safely.

    GEO supplementary text may be UTF-8 or UTF-16, with different newline
    conventions. No expression values are silently transformed or relabeled.
    """
    if path.stat().st_size > MAX_EXPRESSION_BYTES:
        raise ValueError(f"Expression file exceeds compressed safety limit: {path.name}")
    # Decode only the *header*, even for wide matrices. gzip CRC verification is
    # a separate streaming pass and must reach EOF.
    with gzip.open(path, "rb") as raw:
        prefix = raw.read(4096)
    if not prefix:
        raise ValueError("Empty expression payload after gzip decompression")
    encoding = _detect_expression_encoding(prefix)
    try:
        with gzip.open(path, "rb") as raw:
            with io.TextIOWrapper(raw, encoding=encoding, errors="strict", newline=None) as stream:
                header = stream.readline(MAX_HEADER_BYTES + 1)
                if not header or len(header) > MAX_HEADER_BYTES:
                    raise ValueError(f"Missing or too-long text header (encoding={encoding})")
    except UnicodeError as exc:
        raise ValueError(
            f"Expression payload cannot be decoded as {encoding}; "
            f"first 16 decompressed bytes={prefix[:16].hex()}"
        ) from exc
    header = header.lstrip("\ufeff").rstrip("\r\n")
    if "\x00" in header:
        raise ValueError(f"NUL found in decoded expression header (encoding={encoding})")
    delimiter = "\t" if header.count("\t") >= header.count(",") else ","
    if delimiter not in header:
        raise ValueError(
            f"Expression header not recognized as TSV/CSV "
            f"(encoding={encoding}, first 100 chars={header[:100]!r})"
        )
    columns = next(csv.reader([header], delimiter=delimiter))
    if len(columns) < 2 or any(not c.strip() for c in columns[1:]):
        raise ValueError(f"Invalid expression columns in header (encoding={encoding})")
    size = 0
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Decompressed expression exceeds safety limit")
    return {"compressed_bytes": path.stat().st_size,
            "uncompressed_bytes": size,
            "header_column_count": len(columns),
            "header_first_columns": columns[:8],
            "header_encoding": encoding,
            "gzip_crc_verified": True,
            "gsm_header_matches": sum(bool(GSM.fullmatch(x)) for x in columns),
            "sha256": sha256(path)}


class IncompleteExpressionTransfer(OSError):
    """A response or gzip stream ended before the verified file was complete."""


def _fetch_expression_verified(url: str, temp: Path, max_attempts: int = MAX_EXPRESSION_ATTEMPTS) -> dict[str, object]:
    """Stream, safely resume when supported, and verify the entire gzip before promotion.

    A partial `.part` is never a usable data input. On an HTTP 200 response to a
    Range request, restart at byte zero; on HTTP 206, verify Content-Range.
    This function uses source-listed HTTPS URLs only (enforced by caller).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    temp.parent.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, object]] = []
    for attempt in range(1, max_attempts + 1):
        have = temp.stat().st_size if temp.exists() else 0
        if have > MAX_EXPRESSION_BYTES:
            temp.unlink()
            have = 0
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if have:
            headers["Range"] = f"bytes={have}-"
        row: dict[str, object] = {"attempt": attempt, "partial_bytes_before": have}
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                status = response.status
                row["http_status"] = status
                if status not in (200, 206):
                    raise OSError(f"Unexpected HTTP status {status}")
                if response.headers.get("Content-Encoding", "identity").lower() not in ("identity", ""):
                    raise ValueError("Unexpected transport Content-Encoding; refusing ambiguous gzip")
                content_length = response.headers.get("Content-Length")
                expected_response_bytes = int(content_length) if content_length is not None else None
                if expected_response_bytes is not None and expected_response_bytes < 0:
                    raise ValueError("Invalid negative Content-Length")
                row["content_length"] = expected_response_bytes
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range.strip())
                    if not have or match is None or int(match.group(1)) != have:
                        raise ValueError("Unsafe HTTP 206 response: invalid or unexpected Content-Range")
                    end_byte = int(match.group(2))
                    if end_byte < have:
                        raise ValueError("Invalid HTTP Content-Range end")
                    if expected_response_bytes is not None and expected_response_bytes != end_byte - have + 1:
                        raise ValueError("HTTP Content-Range conflicts with Content-Length")
                    declared_total = None if match.group(3) == "*" else int(match.group(3))
                    if declared_total is not None and (declared_total <= end_byte or declared_total > MAX_EXPRESSION_BYTES):
                        raise ValueError("Invalid or excessive HTTP Content-Range total")
                    write_mode = "ab"
                    initial = have
                else:
                    # A server may ignore Range. Never append a full 200 response.
                    declared_total = expected_response_bytes
                    write_mode = "wb"
                    initial = 0
                if declared_total is not None and declared_total > MAX_EXPRESSION_BYTES:
                    raise ValueError("Expression file exceeds compressed safety limit")
                received = 0
                with temp.open(write_mode) as output:
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if initial + received > MAX_EXPRESSION_BYTES:
                            raise ValueError("Expression transfer exceeded compressed safety limit")
                        output.write(chunk)
                row["response_bytes_received"] = received
                row["partial_bytes_after"] = temp.stat().st_size
                if received == 0:
                    raise IncompleteExpressionTransfer("HTTP response contained zero file bytes")
                if expected_response_bytes is not None and received != expected_response_bytes:
                    raise IncompleteExpressionTransfer(
                        f"HTTP body truncated: received {received:,} of {expected_response_bytes:,} bytes"
                    )
                if declared_total is not None and temp.stat().st_size != declared_total:
                    raise IncompleteExpressionTransfer(
                        f"File incomplete: have {temp.stat().st_size:,} of {declared_total:,} bytes"
                    )
            # CRC and full decompression confirm the file is actually complete.
            inspected = _inspect_expression_gzip(temp)
            row["verified_crc"] = True
            diagnostics.append(row)
            return {"attempts": attempt, "transfer_attempts": diagnostics, **inspected}
        except (HTTPError, URLError, HTTPException, TimeoutError, OSError, ValueError, EOFError) as exc:
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            row["partial_bytes_after"] = temp.stat().st_size if temp.exists() else 0
            diagnostics.append(row)
            # An unsuccessful request may leave a valid prefix; keep it for a safe Range retry.
            # A malformed gzip or invalid range could be poison; restart fresh next attempt.
            if isinstance(exc, (ValueError, gzip.BadGzipFile)) and not isinstance(exc, IncompleteExpressionTransfer):
                temp.unlink(missing_ok=True)
            elif isinstance(exc, OSError) and not isinstance(exc, IncompleteExpressionTransfer):
                # Stream/socket errors can leave a valid prefix; don't discard.
                pass
            if attempt == max_attempts:
                attempts_summary = "; ".join(
                    f"try {d['attempt']} HTTP {d.get('http_status', '?')}: "
                    f"{d.get('error_type', 'unknown')} ({d.get('partial_bytes_after', 0)} bytes)"
                    for d in diagnostics
                )
                raise IncompleteExpressionTransfer(
                    f"Download failed after {attempt} attempts. Last error: {type(exc).__name__}: {exc}. "
                    f"Partial bytes: {row['partial_bytes_after']:,}. Attempts: {attempts_summary}"
                ) from exc
            time.sleep(min(2 ** (attempt - 1), 4))
    raise AssertionError("Unreachable: transfer loop exhausted")


def download_expression_files(series_csv: Path, accessions: list[str], raw_root: Path,
                              out_root: Path) -> dict[str, object]:
    """Fetch exact GEO-listed matrices; only complete, CRC-checked data are accepted."""
    with series_csv.open(newline="", encoding="utf-8-sig") as stream:
        series_rows = list(csv.DictReader(stream))
    plan = _approved_expression_urls(series_rows, accessions)
    manifest: dict[str, object] = {}
    failures: list[str] = []
    for entry in plan:
        accession, filename = entry["gse"], entry["filename"]
        dest = raw_root / accession / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(filename + ".part")
        try:
            if dest.is_file():
                # Existing raw data is immutable. If invalid, fail rather than overwrite.
                observed = _inspect_expression_gzip(dest)
                status = "reused_verified"
                transfer_info: dict[str, object] = {"attempts": 0}
            else:
                transfer_info = _fetch_expression_verified(entry["url"], temp)
                # Atomic promote only after CRC and header verification.
                temp.replace(dest)
                observed = _inspect_expression_gzip(dest)
                status = "downloaded_verified"
            manifest[f"{accession}/{filename}"] = {
                **entry, **observed, "status": status,
                "local_file": str(dest.relative_to(REPO_ROOT)),
                "attempts": transfer_info["attempts"],
                "transfer_attempts": transfer_info.get("transfer_attempts", []),
            }
            print(f"[EXPRESSION] {accession}/{filename}: {status}, "
                  f"{observed['compressed_bytes']:,} bytes, CRC OK")
        except (HTTPError, URLError, HTTPException, TimeoutError, OSError, ValueError, EOFError, RuntimeError) as exc:
            failures.append(f"{accession}/{filename}")
            # Keep unverified .part for a later safe resume. The model never reads it.
            manifest[f"{accession}/{filename}"] = {
                **entry, "status": "ERROR", "error": str(exc),
                "partial_bytes": temp.stat().st_size if temp.exists() else 0,
                "partial_file_unverified": bool(temp.exists()),
            }
            print(f"[EXPRESSION FAILED] {accession}/{filename}: {exc}", file=sys.stderr)
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"generated_at_utc": utc_now(), "manifest": manifest, "failed": failures,
              "expression_files_verified": sum(x.get("gzip_crc_verified", False) for x in manifest.values()),
              "independent_validation_approved": False,
              "sample_column_mapping_verified": False,
              "data_harmonization_performed": False,
              "warning": "Counts, TPM and log2TPM MUST NOT be concatenated as-is."}
    write_json(out_root / "expression_manifest.json", report)
    return report


def self_test() -> None:
    # Exercises parser and report generation using a small local SOFT fixture.
    text = """^SERIES = GSE193258
!Series_title = Synthetic series
!Series_type = Expression profiling by high throughput sequencing
!Series_supplementary_file = ftp://example.org/counts.tsv.gz
^SAMPLE = GSM123456
!Sample_title = Test replicate one
!Sample_characteristics_ch1 = cell line: PC9
!Sample_characteristics_ch1 = treatment: osimertinib
!Sample_characteristics_ch1 = treatment duration: 24 h
!Sample_characteristics_ch1 = condition: treated
!sample_table_begin
ID_REF VALUE
X 5
!sample_table_end
^SAMPLE = GSM123457
!Sample_title = Test replicate two
!Sample_characteristics_ch1 = cell line: PC9
!Sample_characteristics_ch1 = treatment: osimertinib
^PLATFORM = GPL123
"""
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        raw = root / "raw" / "GSE193258" / "GSE193258_family.soft.gz"
        raw.parent.mkdir(parents=True)
        with gzip.open(raw, "wt", encoding="utf-8") as stream:
            stream.write(text)
        parsed, rows = parse_soft(raw, "GSE193258")
        assert len(rows) == 2 and parsed["title"] == "Synthetic series"
        assert rows[0]["reported_duration"] == "24 h"
        assert rows[0]["reported_cell_line"] == "PC9"
        assert not rows[1]["reported_duration"]
        result = audit(["GSE193258"], root / "raw", root / "out", offline=True)
        assert result["geo_gsm_records"] == 2
        assert result["observed_metadata_groups"] == 1
        assert result["external_validation_approved"] is False
        assert len(list(csv.DictReader((root / "out" / "geo_groups.csv").open(encoding="utf-8-sig")))) == 1
        # A genuine conflict must be reported and never auto-corrected.
        fake = {
            "gse": "GSE303413", "gsm": "GSM9125502",
            "reported_cell_line": "HCC4006",
            "reported_treatment": "Osi_100nM_DTPC_14d",
            "treatment_protocol": (
                "HCC827 cells were treated with 100 nM osimertinib, "
                "HCC4006 cells were treated with 200 nM osimertinib, "
                "and H1975 cells were treated with 500 nM osimertinib."
            ),
        }
        discrepancies = find_dose_discrepancies([fake])
        assert len(discrepancies) == 1
        assert "200 nM" in discrepancies[0]["details"]
        assert fake["reported_treatment"] == "Osi_100nM_DTPC_14d"
        assert len(list(csv.DictReader((root / "out" / "geo_samples.csv").open(encoding="utf-8-sig")))) == 2
        assert geo_soft_url("GSE193258").endswith("/GSE193258_family.soft.gz")
        # Exact GEO file names are discovered from SOFT and restricted by a source allowlist.
        fake_series = [{"gse": "GSE341996", "metadata_status": "reused_local_offline",
                        "supplementary_files_json": json.dumps([
                            "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE341nnn/GSE341996/suppl/"
                            "GSE341996_TPM_allSamples.csv.gz"])}]
        approved = _approved_expression_urls(fake_series, ["GSE341996"])
        assert len(approved) == 1 and approved[0]["url"].startswith("https://")
        for accession, name, unit in (
            ("GSE249721", "GSE249721_all_samples_processed_data.tsv.gz", "reported_processed_expression_unverified"),
            ("GSE153183", "GSE153183_FPKM_matrix.txt.gz", "FPKM"),
            ("GSE243564", "GSE243564_geneRpkm_YU-003.csv.gz", "RPKM_PDX"),
        ):
            url = f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:-3]}nnn/{accession}/suppl/{name}"
            item = [{"gse": accession, "metadata_status": "reused_local_offline",
                     "supplementary_files_json": json.dumps([url])}]
            if accession == "GSE243564":
                # For this cohort all three PDX expression files must be present.
                item[0]["supplementary_files_json"] = json.dumps([
                    f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243564/suppl/{n}"
                    for n in EXPRESSION_PLAN["GSE243564"]
                ])
            out = _approved_expression_urls(item, [accession])
            assert any(x["filename"] == name and x["reported_expression_unit"] == unit for x in out)
        assert _approved_expression_urls([], ["GSE285632"]) == []
        try:
            bad = dict(fake_series[0], supplementary_files_json=json.dumps([
                "ftp://other-domain.example/geo/series/GSE341nnn/GSE341996/suppl/"
                "GSE341996_TPM_allSamples.csv.gz"]))
            _approved_expression_urls([bad], ["GSE341996"])
            raise AssertionError("Unsafe URL was accepted")
        except ValueError:
            pass
        fixture_expression = root / "fixture.tsv.gz"
        with gzip.open(fixture_expression, "wt", encoding="utf-8") as stream:
            stream.write("Gene\tGSM123456\nEGFR\t1\n")
        check = _inspect_expression_gzip(fixture_expression)
        assert check["gzip_crc_verified"] and check["gsm_header_matches"] == 1
        for encoding in ("utf-16", "utf-16-le", "utf-8-sig"):
            fixture_unicode = root / f"fixture_{encoding}.tsv.gz"
            with gzip.open(fixture_unicode, "wb") as stream:
                stream.write("gene\tPC9-Par-rep1\tPC9-DTP-rep1\nEGFR\t1\t2\n".encode(encoding))
            audited = _inspect_expression_gzip(fixture_unicode)
            assert audited["gzip_crc_verified"]
            assert audited["header_column_count"] == 3, (encoding, audited)
            assert audited["header_encoding"] == encoding, (encoding, audited)
        with gzip.open(root / "not_text.tsv.gz", "wb") as stream:
            stream.write(b"PK\x03\x04\x00\x00\x00\x00\x01\x00")
        try:
            _inspect_expression_gzip(root / "not_text.tsv.gz")
            raise AssertionError("Non-text binary expression was incorrectly accepted")
        except ValueError as exc:
            assert "NUL" in str(exc)
        # Simulate a truncated HTTP response, then a valid 206 Range continuation.
        import io
        from unittest.mock import patch
        payload = fixture_expression.read_bytes()
        first_part = payload[:max(10, len(payload) // 3)]
        second_part = payload[len(first_part):]

        class _Response(io.BytesIO):
            def __init__(self, body: bytes, status: int, headers: dict[str, str]):
                super().__init__(body)
                self.status = status
                self.headers = headers

        calls = []

        def _fake_open(request, timeout=90):
            calls.append(request.get_header("Range"))
            if len(calls) == 1:
                return _Response(first_part, 200, {"Content-Length": str(len(payload))})
            assert request.get_header("Range") == f"bytes={len(first_part)}-"
            return _Response(second_part, 206, {
                "Content-Length": str(len(second_part)),
                "Content-Range": f"bytes {len(first_part)}-{len(payload)-1}/{len(payload)}",
            })

        with patch(__name__ + ".urlopen", side_effect=_fake_open), patch(__name__ + ".time.sleep"):
            download_result = _fetch_expression_verified(
                "https://ftp.ncbi.nlm.nih.gov/test-fixture.tsv.gz", root / "download.part", 2
            )
        assert download_result["attempts"] == 2
        assert calls == [None, f"bytes={len(first_part)}-"]
        assert (root / "download.part").read_bytes() == payload
        try:
            geo_soft_url("GSE193258/../../tmp")
            raise AssertionError("Invalid accession was not rejected")
        except ValueError:
            pass
    print("SELF-TEST PASSED (offline fixture; no network used)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accessions", nargs="+", default=list(DEFAULT_COHORTS),
                        help="GEO Series IDs, e.g. GSE193258 GSE255958")
    parser.add_argument("--offline", action="store_true", help="Use downloaded SOFT files without network")
    parser.add_argument("--refresh", action="store_true", help="Redownload files even if already present")
    parser.add_argument("--self-test", action="store_true", help="Run local test and exit")
    parser.add_argument("--download-expression", action="store_true",
                        help="Download source-listed, allowlisted expression matrices after metadata audit")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.offline and args.refresh:
        parser.error("--offline and --refresh cannot be used together")
    if args.offline and args.download_expression:
        parser.error("--download-expression requires online mode")
    accessions = list(dict.fromkeys(x.strip().upper() for x in args.accessions))
    for acc in accessions:
        if not ACCESSION.fullmatch(acc):
            parser.error(f"Invalid GEO accession: {acc}")
    summary = audit(accessions, RAW_ROOT, OUT_ROOT, offline=args.offline, refresh=args.refresh)
    if summary["series_failed"]:
        print("WARNING: Some accessions failed; see geo_series.csv and review_queue.csv", file=sys.stderr)
        return 1
    if args.download_expression:
        report = download_expression_files(OUT_ROOT / "geo_series.csv", accessions, RAW_ROOT, OUT_ROOT)
        summary["expression_data_downloaded"] = report["expression_files_verified"] > 0
        summary["expression_files_verified"] = report["expression_files_verified"]
        summary["expression_download_failures"] = report["failed"]
        write_json(OUT_ROOT / "summary.json", summary)
        if report["failed"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
