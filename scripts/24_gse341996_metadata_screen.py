"""Stage 24: metadata-only, non-blinded eligibility screen for GSE341996.

From repo root:
    python scripts/24_gse341996_metadata_screen.py --self-test
    python scripts/24_gse341996_metadata_screen.py --download

For a separately downloaded official GEO family SOFT file:
    python scripts/24_gse341996_metadata_screen.py --soft <path-to-official-soft-or-soft.gz>

No expression matrices, predictors, or external class labels are evaluated.
This tool NEVER certifies biological state validity or an untouched holdout.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ACCESSION = "GSE341996"
SOURCE_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE341nnn/"
              "GSE341996/soft/GSE341996_family.soft.gz")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / ACCESSION / (ACCESSION + "_family.soft.gz")
OUT = ROOT / "outputs" / "dataset_screening" / ACCESSION
MAX_DOWNLOAD = 80 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def get_meta_string(fields, name):
    return " | ".join(fields.get(name, []))


def parse_soft(path):
    """Stream GEO SOFT; ignore tables and anything outside sample/series metadata."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    opened = gzip.open if p.suffix.lower() == ".gz" else open
    study = {"accession": "", "title": "", "summary": [], "overall_design": [], "sample_ids": []}
    samples = []
    section = None
    current = None
    in_table = False
    with opened(p, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if line.startswith("^"):
                in_table = False
                key, _, val = line[1:].partition(" = ")
                if key == "SERIES":
                    section = "series"
                    study["accession"] = val.strip()
                elif key == "SAMPLE":
                    section = "sample"
                    current = {"gsm": val.strip(), "fields": {}}
                    samples.append(current)
                else:
                    section = "skip"
                    current = None
                continue
            if line in ("!sample_table_begin", "!platform_table_begin"):
                in_table = True
            if line in ("!sample_table_end", "!platform_table_end"):
                in_table = False
            if in_table or not line.startswith("!"):
                continue
            field, sep, val = line[1:].partition(" = ")
            if not sep:
                continue
            field = field.strip().lower()
            val = val.strip()
            if section == "series":
                if field == "series_title":
                    study["title"] = val
                elif field == "series_summary":
                    study["summary"].append(val)
                elif field == "series_overall_design":
                    study["overall_design"].append(val)
                elif field == "series_sample_id":
                    study["sample_ids"].append(val)
            elif section == "sample" and current is not None:
                current["fields"].setdefault(field, []).append(val)
    if study["accession"] != ACCESSION:
        raise ValueError("GEO accession mismatch (expected GSE341996, got %r)" % study["accession"])
    if not samples:
        raise ValueError("No ^SAMPLE sections detected; do not infer sample count from matrix headers")
    ids = [s["gsm"] for s in samples]
    if any(not re.fullmatch(r"GSM\d+", z) for z in ids) or len(set(ids)) != len(ids):
        raise ValueError("Malformed or duplicate GSM IDs")
    series_ids = study["sample_ids"]
    if series_ids and set(ids) != set(series_ids):
        raise ValueError("Mismatch between ^SAMPLE and !Series_sample_id listings")
    return study, samples


def candidate_state(s):
    """Weak hints only from title/characteristics; never an approved biological label."""
    f = s["fields"]
    clues = " | ".join(
        get_meta_string(f, field) for field in (
            "sample_title", "sample_source_name_ch1", "sample_characteristics_ch1", "sample_description"))
    clues = clues.lower()
    patterns = {
        "POSSIBLE_CONTROL": r"\b(control|ctrl|dmso|vehicle|untreated|parental)\b",
        "POSSIBLE_ACUTE": r"\b(acute|short[- ]term[ -]treatment)\b",
        "POSSIBLE_DTP": r"\b(dtp|persister|drug[- ]tolerant)\b",
        "POSSIBLE_RESISTANT": r"\b(resistant|resistance|acquired[- ]resistance)\b",
    }
    hits = [k for k, expr in patterns.items() if re.search(expr, clues)]
    if len(hits) != 1:
        return "UNRESOLVED" if not hits else "AMBIGUOUS:" + "+".join(hits)
    return hits[0]


def rows_from(samples):
    rows = []
    for sample in samples:
        f = sample["fields"]
        rows.append({
            "gsm": sample["gsm"],
            "title": get_meta_string(f, "sample_title"),
            "organism": get_meta_string(f, "sample_organism_ch1"),
            "source_name": get_meta_string(f, "sample_source_name_ch1"),
            "characteristics": get_meta_string(f, "sample_characteristics_ch1"),
            "sample_treatment_protocol": get_meta_string(f, "sample_treatment_protocol_ch1"),
            "sample_growth_protocol": get_meta_string(f, "sample_growth_protocol_ch1"),
            "candidate_state_UNVERIFIED": candidate_state(sample),
            "biology_approved": "NO",
        })
    return rows


def report_study(study, rows, source_path):
    counts = dict(sorted(Counter(r["candidate_state_UNVERIFIED"] for r in rows).items()))
    title = study["title"] or "(not provided)"
    summary = {
        "accession": ACCESSION,
        "status": "METADATA_SCREENED_MANUAL_ADJUDICATION_REQUIRED",
        "source": SOURCE_URL,
        "metadata_file_sha256": sha256(source_path),
        "sample_count": len(rows),
        "candidate_state_counts_UNVERIFIED": counts,
        "title": title,
        "series_summary": study["summary"],
        "series_overall_design": study["overall_design"],
        "touched_expression": False,
        "ran_classifier": False,
        "untouched_for_model_scoring": True,
        "labels_approved": False,
        "three_class_eligible": None,
        "questions_before_approval": [
            "Verify every GSM maps to one independent biological replicate and cell line.",
            "Verify Control, Acute and DTP are distinct *source-supported* labels.",
            "Verify doses, exposure durations, interventions, and assay processing.",
            "Determine whether this is bulk RNA-seq and whether all three classes exist.",
            "Identify overlapping experiments or prior exploratory inspection.",
            "Freeze gene mapping and model protocol before inspecting expression.",
        ],
    }
    lines = [f"# {ACCESSION} — Official GEO metadata eligibility screen",
             "", "**Status: MANUAL BIOLOGICAL REVIEW REQUIRED. No expression scoring performed.**",
             "", f"- Source: {SOURCE_URL}",
             f"- Title: {title}", f"- GEO sample sections: **{len(rows)}**",
             f"- Official SOFT SHA-256: `{summary['metadata_file_sha256']}`", "",
             "## Series-level experimental design", ""]
    lines += (study["overall_design"] or ["Not supplied in SOFT file"])
    lines += ["", "## Weak sample-title hints (NOT verified labels)", "",
              "| Hint | n |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in counts.items()]
    lines += ["", "See `24_gsm_metadata.csv` for every GSM and its original annotations.",
              "", "## Approval gate — every item requires manual source review", ""]
    lines += [f"- [ ] {q}" for q in summary["questions_before_approval"]]
    lines += ["", "**No classifier fitting, feature selection, label adjudication, or validation claim is authorized by this report.**", ""]
    return summary, "\n".join(lines)


def write_report(study, rows, source_path, out):
    out.mkdir(parents=True, exist_ok=True)
    summary, markdown = report_study(study, rows, source_path)
    with (out / "24_gsm_metadata.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "24_eligibility.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "24_eligibility.md").write_text(markdown, encoding="utf-8")
    return summary


def download_official(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "DTPResearchEligibilityScreen/1.0 (metadata-only)"})
    temp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(req, timeout=30) as response, temp.open("wb") as f:
            total = 0
            while True:
                part = response.read(1 << 20)
                if not part:
                    break
                total += len(part)
                if total > MAX_DOWNLOAD:
                    raise ValueError("GEO SOFT archive exceeds size limit")
                f.write(part)
        if total == 0:
            raise ValueError("Official GEO SOFT download is empty")
        temp.replace(dest)
        return dest
    finally:
        if temp.exists():
            temp.unlink()


def self_test():
    fake = """^SERIES = GSE341996
!Series_title = Test GEO series
!Series_overall_design = Three examples; not real patient or biological data
!Series_sample_id = GSM9100001
!Series_sample_id = GSM9100002
!Series_sample_id = GSM9100003
^SAMPLE = GSM9100001
!Sample_title = PC9 DMSO control rep1
!Sample_characteristics_ch1 = cell line: PC9
^SAMPLE = GSM9100002
!Sample_title = PC9 osimertinib acute rep1
^SAMPLE = GSM9100003
!Sample_title = PC9 osimertinib DTP rep1
"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mock.soft.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.write(fake)
        study, samples = parse_soft(p)
        rows = rows_from(samples)
        assert study["accession"] == ACCESSION
        assert [r["candidate_state_UNVERIFIED"] for r in rows] == [
            "POSSIBLE_CONTROL", "POSSIBLE_ACUTE", "POSSIBLE_DTP"]
        report = write_report(study, rows, p, Path(d) / "output")
        assert report["three_class_eligible"] is None and not report["labels_approved"]
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.write(fake.replace("^SERIES = GSE341996", "^SERIES = GSE000000"))
        try:
            parse_soft(p)
        except ValueError as exc:
            assert "accession mismatch" in str(exc)
        else:
            raise AssertionError("Mismatch guard did not activate")
    print("SELF-TEST PASSED: accession guards, GEO sample parsing, weak label hints, no approval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--download", action="store_true", help="Fetch official GEO family SOFT only; no expression matrix")
    group.add_argument("--soft", type=Path, help="Existing official GEO family SOFT or .soft.gz")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    path = None
    try:
        path = download_official(RAW) if args.download else args.soft
        study, samples = parse_soft(path)
        rows = rows_from(samples)
        report = write_report(study, rows, path, args.out)
        print(f"GEO metadata parsed: {report['sample_count']} unique GSM samples")
        print("Candidate title hints (NOT labels):", report["candidate_state_counts_UNVERIFIED"])
        print("Saved:", args.out / "24_eligibility.md")
        print("NO MODEL RUN; eligibility still requires biological manual review")
        return 0
    except (OSError, ValueError, EOFError, urllib.error.URLError) as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        error = {"accession": ACCESSION, "status": "OFFICIAL_GEO_METADATA_NOT_VERIFIED",
                 "official_url": SOURCE_URL, "error": f"{type(exc).__name__}: {exc}",
                 "model_run": False, "biology_approved": False}
        (args.out / "24_eligibility.json").write_text(json.dumps(error, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.out / "24_eligibility.md").write_text(
            f"# {ACCESSION} — metadata screen blocked\n\nOfficial GEO metadata could not be verified.\n\n````\n{error['error']}\n````\n\nNo model was run.\n",
            encoding="utf-8")
        print("SCREEN BLOCKED:", error["error"], file=sys.stderr)
        print("Saved diagnostic:", args.out / "24_eligibility.md", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
