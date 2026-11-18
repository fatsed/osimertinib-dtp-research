"""Build an auditable, unapproved sample-cohort shortlist from existing GEO metadata.

Run from the repository root:
    python scripts/cohort_review.py --self-test
    python scripts/cohort_review.py

Reads geo_samples.csv, expression_alignment.csv, expression_manifest.json,
and cleaning_report.json. Does not download data, process expression matrices,
change source values, approve biological replicates, or select a trained model.
Python 3.10+ standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "metadata"
FIELDS = [
    "gse", "gsm", "geo_title", "cell_line_reported", "treatment_reported",
    "candidate_state", "candidate_analysis", "expression_mapping_files",
    "expression_filenames", "expression_mapping_status", "gene_screen_status",
    "source_history", "review_flags", "scientific_review_status",
    "approved_for_training", "approved_as_untouched_external_validation",
    "geo_sample_url",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header: {path}")
        return list(reader)


def classify(gse: str, line: str, condition: str, title: str) -> tuple[str, str]:
    """Pre-registered provisional rules; no fuzzy relabeling of unrelated drugs."""
    if gse == "GSE193258":
        if condition == "DMSO":
            return "Control", "control_dtp_candidate"
        if condition == "osimertinib DTPs 3w.":
            return "DTP", "control_dtp_candidate"
        if condition == "osimertinib acute 24 hrs.":
            return "Acute", "acute_dtp_development_only"
        return "OTHER", "secondary_timecourse_only"
    if gse == "GSE249721":
        if line == "HCC827" and condition == "DMSO":
            return "Control", "control_dtp_candidate"
        if line == "HCC827" and condition == "osimertinib 1 uM, 11 days":
            return "DTP", "control_dtp_candidate"
        return "OTHER", "other_drug_or_condition"
    if gse == "GSE303413":
        if condition == "Control":
            return "Control", "control_dtp_candidate"
        if condition == "Osi_100nM_DTPC_14d" and line in {"H1975", "HCC4006", "HCC827"}:
            return "DTP", "control_dtp_candidate"
        return "OTHER", "unspecified"
    if gse == "GSE341996":
        if line == "PC-9" and condition == "no treatment" and title.startswith("Par rep"):
            return "Control", "control_dtp_candidate"
        if line == "PC-9" and condition == "osimertinib treatment" and title.startswith("DTP rep"):
            return "DTP", "control_dtp_candidate"
        return "OTHER", "unspecified"
    if gse == "GSE153183":
        if condition == "DMSO 9d" and title.startswith("PC9 Parental rep"):
            return "Control", "control_dtp_candidate"
        if condition == "2μM Osimertinib 9d" and title.startswith("PC9 DTP rep"):
            return "DTP", "control_dtp_candidate"
        return "OTHER", "unspecified"
    if gse == "GSE243564":
        return "PDX", "separate_pdx_study"
    if gse == "GSE255958":
        return "UNASSESSED", "historically_inspected_no_matrix"
    return "UNASSESSED", "metadata_only"


def build_rows(samples: list[dict[str, str]], alignment: list[dict[str, str]],
               manifest: dict, cleaning: dict) -> tuple[list[dict[str, str]], dict]:
    required_samples = {"gse", "gsm", "title", "reported_cell_line", "reported_treatment", "geo_sample_url"}
    required_alignment = {"gse", "gsm", "filename", "mapping_status"}
    if not samples or not required_samples.issubset(samples[0]):
        raise ValueError("geo_samples.csv missing expected columns")
    if not alignment or not required_alignment.issubset(alignment[0]):
        raise ValueError("expression_alignment.csv missing expected columns")
    if not isinstance(manifest.get("manifest"), dict):
        raise ValueError("expression_manifest.json is missing the manifest dictionary")
    if cleaning.get("analysis") != "conservative_technical_gene_screen_only" or cleaning.get("raw_files_changed") is not False:
        raise ValueError("Unexpected cleaning report or raw-file modification")

    verified_files = {
        (item.get("gse"), item.get("filename"))
        for item in manifest["manifest"].values()
        if item.get("status") in {"downloaded_verified", "reused_verified"}
        and item.get("gzip_crc_verified") is True and item.get("sha256")
    }
    cleaned = {
        (r["gse"], r["source_filename"])
        for r in cleaning.get("results", [])
        if r.get("derived_sha256") and r.get("approved_for_training") is False
    }
    matches: dict[tuple[str, str], set[str]] = defaultdict(set)
    for a in alignment:
        if a.get("gsm") and a.get("mapping_status") == "CANDIDATE_REQUIRES_REVIEW":
            matches[(a["gse"], a["gsm"])].add(a["filename"])
    ids = [(s["gse"], s["gsm"]) for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicated study/GSM in metadata; refusing to classify")
    sample_id_set = set(ids)
    if set(matches) - sample_id_set:
        raise ValueError("Expression alignment has GSMs missing from geo_samples.csv")

    outputs: list[dict[str, str]] = []
    for s in sorted(samples, key=lambda z: (z["gse"], z["gsm"])):
        gse, gsm = s["gse"], s["gsm"]
        state, analysis = classify(gse, s["reported_cell_line"], s["reported_treatment"], s["title"])
        file_names = sorted(matches.get((gse, gsm), set()))
        supported = all((gse, name) in verified_files for name in file_names)
        flags = ["BIOLOGICAL_REPLICATE_NOT_VERIFIED"]
        if not file_names:
            flags.append("NO_APPROVED_SAMPLE_COLUMN_MAPPING")
        elif not supported:
            flags.append("EXPRESSION_FILE_UNVERIFIED")
        if len(file_names) > 1:
            if gse == "GSE193258" and len(file_names) == 2:
                flags.append("TWO_EXPRESSION_SCALES_SAME_GSM_NOT_TWO_SAMPLES")
            else:
                flags.append("MULTIPLE_MATRIX_FILES_REVIEW")
        if state == "OTHER":
            flags.append("NOT_MAIN_CONTROL_DTP_CONTRAST")
        if gse == "GSE243564":
            flags.append("PDX_SEPARATE_DOMAIN")
        if gse == "GSE303413" and state == "DTP" and s["reported_cell_line"] in {"H1975", "HCC4006"}:
            flags.append("DOSE_LABEL_PROTOCOL_CONFLICT")
        if gse == "GSE193258":
            flags.append("PREVIOUSLY_INSPECTED_DEVELOPMENT")
        if gse == "GSE255958":
            flags.append("HISTORICALLY_INSPECTED_NOT_UNTOUCHED")
        if gse in {"GSE153183", "GSE303413", "GSE341996", "GSE249721"} and state == "DTP":
            flags.append("EXPOSURE_TIME_DIFFERENCE_MAY_CONFOUND_STATE")
        if any((gse, name) in cleaned for name in file_names):
            screen = "DERIVED_GENE_SCREEN_EXISTS_NOT_BIOLOGICALLY_APPROVED"
        elif file_names and supported:
            screen = "TECHNICAL_VERIFICATION_ONLY"
        else:
            screen = "NOT_AVAILABLE_OR_UNVERIFIED"
        outputs.append({
            "gse": gse, "gsm": gsm, "geo_title": s["title"],
            "cell_line_reported": s["reported_cell_line"],
            "treatment_reported": s["reported_treatment"],
            "candidate_state": state, "candidate_analysis": analysis,
            "expression_mapping_files": str(len(file_names)),
            "expression_filenames": " | ".join(file_names),
            "expression_mapping_status": "SOURCE_TITLE_CANDIDATE_ONLY" if file_names else "MISSING",
            "gene_screen_status": screen,
            "source_history": "previous_development" if gse == "GSE193258" else "historically_inspected" if gse == "GSE255958" else "review_candidate",
            "review_flags": ";".join(flags),
            "scientific_review_status": "PENDING_SOURCE_SUPPORTED_REVIEW",
            "approved_for_training": "false",
            "approved_as_untouched_external_validation": "false",
            "geo_sample_url": s["geo_sample_url"],
        })
    prelim = [r for r in outputs if r["candidate_analysis"] == "control_dtp_candidate" and r["expression_mapping_files"] != "0"]
    counts = dict(sorted(Counter(r["gse"] for r in prelim).items()))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_gsm_rows": len(outputs),
        "unique_study_gsm_with_source_title_mapping": sum(r["expression_mapping_files"] != "0" for r in outputs),
        "provisional_control_dtp_samples": len(prelim),
        "provisional_control_dtp_by_study": counts,
        "dose_conflict_sample_rows": sum("DOSE_LABEL_PROTOCOL_CONFLICT" in r["review_flags"] for r in outputs),
        "biology_approved_samples": 0,
        "untouched_external_validation_approved": False,
        "data_harmonization_performed": False,
        "notes": [
            "Counts are provisional mapping candidates, not approved independent biological replicates.",
            "GSE193258 is previously inspected development data, not untouched validation.",
            "No model should use a sample until GEO source/replicates and matrix mapping are independently reviewed.",
            "Different expression units and exposure durations require explicit experimental design.",
            "Gene symbol filtering alone does not establish cross-study gene identifier compatibility.",
        ],
    }
    return outputs, report


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    assert classify("GSE193258", "PC9", "osimertinib DTPs 3w.", "PC9_osi_DTP_1 [RNA-seq]") == ("DTP", "control_dtp_candidate")
    assert classify("GSE249721", "PC9", "erlotinib 1 uM, 21 days", "x")[1] == "other_drug_or_condition"
    assert classify("GSE249721", "HCC827", "osimertinib 1 uM, 11 days", "x")[1] == "control_dtp_candidate"
    assert classify("GSE341996", "PC-9", "osimertinib treatment", "Unknown title")[0] == "OTHER"
    sample = [{"gse":"GSE303413", "gsm":"GSM1", "title":"MN.P.23.1",
               "reported_cell_line":"H1975", "reported_treatment":"Osi_100nM_DTPC_14d", "geo_sample_url":"https://www.ncbi.nlm.nih.gov"}]
    alignment = [{"gse":"GSE303413", "gsm":"GSM1", "filename":"x.tsv.gz", "mapping_status":"CANDIDATE_REQUIRES_REVIEW"}]
    manifest = {"manifest":{"x":{"gse":"GSE303413", "filename":"x.tsv.gz", "status":"reused_verified", "gzip_crc_verified":True, "sha256":"abc"}}}
    cleaning = {"analysis":"conservative_technical_gene_screen_only", "raw_files_changed":False, "results":[]}
    rows, report = build_rows(sample, alignment, manifest, cleaning)
    assert report["provisional_control_dtp_samples"] == 1
    assert report["dose_conflict_sample_rows"] == 1
    assert rows[0]["approved_for_training"] == "false"
    assert "DOSE_LABEL_PROTOCOL_CONFLICT" in rows[0]["review_flags"]
    try:
        build_rows(sample + sample, alignment, manifest, cleaning)
        raise AssertionError("Duplicate GSM was not blocked")
    except ValueError as e:
        assert "Duplicated" in str(e)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)/"out.csv"
        write_csv(p, rows)
        assert len(read_csv(p)) == 1
    print("SELF-TEST PASSED (provisional labels, no auto-approval, duplicate GSM and dose-conflict guards)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    paths = {
        "samples":META_DIR/"geo_samples.csv",
        "alignment":META_DIR/"expression_alignment.csv",
        "manifest":META_DIR/"expression_manifest.json",
        "cleaning":META_DIR/"cleaning_report.json",
    }
    for p in paths.values():
        if not p.is_file():
            parser.error(f"Required existing audit report missing: {p}")
    rows, summary = build_rows(
        read_csv(paths["samples"]), read_csv(paths["alignment"]),
        json.loads(paths["manifest"].read_text(encoding="utf-8")),
        json.loads(paths["cleaning"].read_text(encoding="utf-8")),
    )
    write_csv(META_DIR/"cohort_review.csv", rows)
    (META_DIR/"cohort_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"
    )
    print("Saved data/metadata/cohort_review.csv and cohort_review_summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
