"""Stage 25 | Metadata-only independence gate for GSE341996 vs GSE153183.

Run from the osimertinib-dtp-research repository root:
    python scripts/25_gse341996_independence_gate.py --self-test
    python scripts/25_gse341996_independence_gate.py --download-reference

Reads only official GEO *family SOFT* metadata files. It NEVER downloads or
reads expression matrices, accesses model predictions, fits a classifier,
or declares an independent validation cohort. Detection of similarities is
an audit warning, not proof that sample libraries were reused.

Required existing Stage24 inputs:
    data/raw/GSE341996/GSE341996_family.soft.gz
    outputs/dataset_screening/GSE341996/24_eligibility.json

The --download-reference option fetches ONLY GSE153183 family SOFT metadata
from NCBI to data/raw/GSE153183/GSE153183_family.soft.gz.

Generated (git-ignored) outputs:
    outputs/dataset_screening/GSE341996/25_independence_gate.json
    outputs/dataset_screening/GSE341996/25_independence_gate.md
    outputs/dataset_screening/GSE341996/25_condition_rep_matches.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "GSE341996"
REFERENCE = "GSE153183"
CAND_SOFT = ROOT / "data/raw/GSE341996/GSE341996_family.soft.gz"
REF_SOFT = ROOT / "data/raw/GSE153183/GSE153183_family.soft.gz"
STAGE24 = ROOT / "outputs/dataset_screening/GSE341996/24_eligibility.json"
OUT = ROOT / "outputs/dataset_screening/GSE341996"
REF_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/"
           "GSE153183/soft/GSE153183_family.soft.gz")
IDS = re.compile(r"\b(?:SAMN|SAMEA|SAMD|SRX|SRR|ERX|ERR|DRX|DRR|PRJNA|PRJEB|PRJDB)\d+\b", re.I)
SAMPLE_IDS = re.compile(r"^(?:SAMN|SAMEA|SAMD|SRX|SRR|ERX|ERR|DRX|DRR)\d+$")
MAX_BYTES = 80 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_soft(path: Path, accession: str) -> dict:
    """Parse series/sample metadata only. Always skip sample/platform tables."""
    if not path.is_file():
        raise FileNotFoundError(f"Official SOFT metadata not found: {path}")
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    series, samples = {}, []
    mode, current, in_table, observed_accession = None, None, False, None
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if line.startswith("^"):
                in_table = False
                section, _, value = line[1:].partition(" = ")
                if section == "SERIES":
                    mode, observed_accession = "series", value.strip()
                elif section == "SAMPLE":
                    mode = "sample"
                    current = {"gsm": value.strip(), "fields": {}}
                    samples.append(current)
                else:
                    mode, current = None, None
                continue
            if line in ("!sample_table_begin", "!platform_table_begin"):
                in_table = True
            if line in ("!sample_table_end", "!platform_table_end"):
                in_table = False
                continue
            if in_table or not line.startswith("!"):
                continue
            key, sep, value = line[1:].partition(" = ")
            if not sep:
                continue
            key, value = key.strip().lower(), value.strip()
            if mode == "series":
                series.setdefault(key, []).append(value)
            elif mode == "sample" and current is not None:
                current["fields"].setdefault(key, []).append(value)
    if observed_accession != accession:
        raise ValueError(f"Expected {accession}, got {observed_accession!r}: {path}")
    gsm = [s["gsm"] for s in samples]
    if not gsm or len(set(gsm)) != len(gsm) or any(not re.fullmatch(r"GSM\d+", g) for g in gsm):
        raise ValueError("Missing, repeated, or invalid GSM IDs")
    declared = set(series.get("series_sample_id", []))
    if declared and declared != set(gsm):
        raise ValueError(f"{accession}: declared GSM IDs do not match sample records")
    return {"accession": accession, "series": series, "samples": samples,
            "soft_sha256": sha256(path)}


def field(fields: dict, name: str) -> str:
    return " | ".join(fields.get(name, []))


def sample_condition_rep(sample: dict) -> tuple[str, str] | None:
    """Find name-only condition/replicate coincidences; never approve labels."""
    title = field(sample["fields"], "sample_title").lower()
    m = re.search(r"\brep(?:licate)?[\s_\-#:]*(\d+)\b", title)
    if not m:
        return None
    if re.search(r"\bdtp\b|\bpersister\b", title):
        group = "DTP"
    elif re.search(r"\bpar(?:ental)?\b", title):
        group = "PARENTAL"
    else:
        return None
    return group, m.group(1)


def ids_by_sample(sample: dict) -> set[str]:
    """Read deposited experimental identifiers in relation/accession metadata."""
    f = sample["fields"]
    rel_text = " | ".join(v for k, a in f.items()
                          if "relation" in k or "accession" in k
                          for v in a)
    return {v.upper() for v in IDS.findall(rel_text) if SAMPLE_IDS.fullmatch(v.upper())}


def all_series_text(dataset: dict) -> str:
    wanted = ("series_title", "series_summary", "series_overall_design",
              "series_type", "series_sample_id")
    s = dataset["series"]
    return " | ".join(v for key in wanted for v in s.get(key, [])).lower()


def signature(dataset: dict) -> dict[str, bool]:
    t = all_series_text(dataset)
    t = t.replace("μ", "u").replace("µ", "u")
    return {
        "pc9": bool(re.search(r"\bpc[- ]?9\b", t)),
        "osimertinib": "osimertinib" in t,
        "2uM": bool(re.search(r"\b2\s*(?:u\s*m|micro\s*molar|micromolar)\b", t)),
        "nine_days": bool(re.search(r"\b9\s*(?:days?|d)\b|\bnine\s+days?\b", t)),
        "parental": bool(re.search(r"\bparental\b|\bparent\b", t)),
        "dtp": bool(re.search(r"\bdtp\b|\bpersister", t)),
    }


def compare(a: dict, b: dict) -> dict:
    cand, ref = a["samples"], b["samples"]
    matches = []
    for ca in cand:
        ac = sample_condition_rep(ca)
        if ac is None:
            continue
        for rb in ref:
            if ac == sample_condition_rep(rb):
                shared = sorted(ids_by_sample(ca) & ids_by_sample(rb))
                matches.append({
                    "condition_hint_NOT_APPROVED": ac[0],
                    "replicate_label": ac[1],
                    "candidate_gsm": ca["gsm"],
                    "reference_gsm": rb["gsm"],
                    "candidate_title": field(ca["fields"], "sample_title"),
                    "reference_title": field(rb["fields"], "sample_title"),
                    "shared_biosample_or_sra_identifiers": ";".join(shared),
                })
    cand_ids = {k for s in cand for k in ids_by_sample(s)}
    ref_ids = {k for s in ref for k in ids_by_sample(s)}
    shared_ids = sorted(cand_ids & ref_ids)
    sig_a, sig_b = signature(a), signature(b)
    common_protocol = all(sig_a[k] and sig_b[k] for k in ("pc9", "osimertinib", "2uM", "nine_days"))
    if shared_ids:
        verdict = "DIRECT_SHARED_SAMPLE_ACCESSION_IDS_REVIEW_REQUIRED"
    elif len(matches) >= 3 and common_protocol:
        verdict = "HIGH_RISK_SAME_PROTOCOL_AND_REPLICATE_NAMES_UNRESOLVED"
    else:
        verdict = "INDEPENDENCE_UNPROVEN_REVIEW_REQUIRED"
    return {
        "verdict": verdict,
        "independent_validation_approved": False,
        "method": "GEO_SOFT_METADATA_ONLY_NO_EXPRESSION",
        "candidate": a["accession"], "reference": b["accession"],
        "candidate_soft_sha256": a["soft_sha256"],
        "reference_soft_sha256": b["soft_sha256"],
        "candidate_n_gsm": len(cand), "reference_n_gsm": len(ref),
        "candidate_gsms": [s["gsm"] for s in cand],
        "reference_gsms": [s["gsm"] for s in ref],
        "candidate_study_signature": sig_a,
        "reference_study_signature": sig_b,
        "same_pc9_drug_dose_day_design": common_protocol,
        "same_condition_replicate_label_matches": matches,
        "shared_sra_or_biosample_ids": shared_ids,
        "n_candidate_sample_ids_listed": len(cand_ids),
        "n_reference_sample_ids_listed": len(ref_ids),
        "limitations": [
            "Same replicate names and protocols do not prove reused sequencing libraries.",
            "Distinct GSM, BioSample or SRX accessions do not prove independent biological experiments.",
            "GSE341996 has no Acute group in its declared two-arm study design.",
            "Replicate labels do not certify independent biological cultures.",
            "Genotype WT on GSE341996 sample records needs clarification; do not infer EGFR wild-type from this field.",
            "The DTP-versus-parental contrast confounds drug exposure and duration.",
            "No classifier trained, scored or selected; GSE341996 expression data remain uninspected.",
        ],
    }


def write_report(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "25_independence_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    matches = report["same_condition_replicate_label_matches"]
    with (out / "25_condition_rep_matches.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["condition_hint_NOT_APPROVED", "replicate_label", "candidate_gsm",
                   "reference_gsm", "candidate_title", "reference_title",
                   "shared_biosample_or_sra_identifiers"]
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(matches)
    lines = [
        "# Stage 25 — GSE341996 independence/provenance gate", "",
        f"**Verdict: {report['verdict']}**", "",
        "**Independent external validation NOT approved.** Only GEO SOFT metadata was read; no expression matrices or models were accessed.", "",
        "## Comparison of study metadata", "",
        f"- Candidate {report['candidate']}: {report['candidate_n_gsm']} GSM records",
        f"- Historical {report['reference']}: {report['reference_n_gsm']} GSM records",
        f"- Shared PC9/osimertinib/2 µM/9-day design: **{report['same_pc9_drug_dose_day_design']}**",
        f"- Matching condition+replicate TITLE labels: **{len(matches)}** (not evidence of duplicated experiments by itself)",
        f"- Identical deposited sample SRA/BioSample IDs: **{len(report['shared_sra_or_biosample_ids'])}**",
        "", "## Matched sample-title labels (NOT confirmed sample identity)", "",
        "| Condition hint | Replicate | GSE341996 GSM | GSE153183 GSM |",
        "|---|---|---|---|",
    ]
    lines += [f"| {m['condition_hint_NOT_APPROVED']} | {m['replicate_label']} | {m['candidate_gsm']} | {m['reference_gsm']} |"
              for m in matches]
    if not matches:
        lines += ["| None found | — | — | — |"]
    lines += ["", "## What this audit cannot resolve", ""]
    lines += [f"- {s}" for s in report["limitations"]]
    lines += ["", "## Next eligibility gate", "",
              "1. Check submitter(s), publication, original study/sample provenance and whether libraries were re-deposited.",
              "2. Check GEO and SRA/BioSample links for run-level and culture-level independence; do not equate distinct accession IDs with independence.",
              "3. Request clarification or find a different study if overlap cannot be ruled out.",
              "4. If independently eligible, preregister a source-trained **binary Control-vs-DTP** experiment; never use a 3-class score without Acute samples.",
              "5. Do not inspect GSE341996 expression or fit/tune a model on it until the protocol is frozen.",
              "", "## Official records", "",
              "- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE341996",
              "- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153183", ""]
    (out / "25_independence_gate.md").write_text("\n".join(lines), encoding="utf-8")


def download_reference(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size:
        return dest
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(REF_URL, headers={"User-Agent": "DTPResearchMetadataOverlap/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r, tmp.open("wb") as w:
            downloaded = 0
            while True:
                data = r.read(1 << 20)
                if not data:
                    break
                downloaded += len(data)
                if downloaded > MAX_BYTES:
                    raise ValueError("Reference GEO metadata file exceeds size limit")
                w.write(data)
        if not downloaded:
            raise ValueError("Empty metadata response")
        tmp.replace(dest)
        return dest
    finally:
        if tmp.exists():
            tmp.unlink()


def self_test() -> None:
    def fake_series(acc, samples, same_ids=False):
        lines = [f"^SERIES = {acc}",
                 "!Series_title = PC9 osimertinib persister RNA-seq",
                 "!Series_summary = PC9 parental versus DTP; nine days of 2 uM osimertinib",
                 "!Series_sample_id = " + "\n!Series_sample_id = ".join(s[0] for s in samples)]
        for gsm, title, ids in samples:
            lines += [f"^SAMPLE = {gsm}", f"!Sample_title = {title}",
                      f"!Sample_relation = BioSample: https://www.ncbi.nlm.nih.gov/biosample/{ids}",
                      "!sample_table_begin", "NOT_METADATA\t10", "!sample_table_end"]
        return "\n".join(lines) + "\n"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        cs = [("GSM9921464", "Par rep1, RNA-seq", "SAMN111"),
              ("GSM9921465", "Par rep2, RNA-seq", "SAMN112"),
              ("GSM9921467", "DTP rep1, RNA-seq", "SAMN113"),
              ("GSM9921469", "DTP rep3, RNA-seq", "SAMN114")]
        rs = [("GSM4635290", "PC9 Parental rep2", "SAMN211"),
              ("GSM4635292", "PC9 DTP rep1", "SAMN212"),
              ("GSM4635293", "PC9 DTP rep3", "SAMN213")]
        a = p / "a.soft.gz"
        b = p / "b.soft.gz"
        for path, acc, samples in [(a, CANDIDATE, cs), (b, REFERENCE, rs)]:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(fake_series(acc, samples))
        x, y = read_soft(a, CANDIDATE), read_soft(b, REFERENCE)
        z = compare(x, y)
        assert z["verdict"] == "HIGH_RISK_SAME_PROTOCOL_AND_REPLICATE_NAMES_UNRESOLVED"
        assert len(z["same_condition_replicate_label_matches"]) == 3
        assert not z["shared_sra_or_biosample_ids"] and not z["independent_validation_approved"]
        write_report(z, p / "out")
        assert (p / "out/25_independence_gate.md").is_file()
        # Direct shared biological sample ID must raise a stronger warning.
        rs[0] = ("GSM4635290", "PC9 Parental rep2", "SAMN112")
        with gzip.open(b, "wt", encoding="utf-8") as f:
            f.write(fake_series(REFERENCE, rs))
        q = compare(x, read_soft(b, REFERENCE))
        assert q["verdict"] == "DIRECT_SHARED_SAMPLE_ACCESSION_IDS_REVIEW_REQUIRED"
        assert q["shared_sra_or_biosample_ids"] == ["SAMN112"]
        # Accession mismatch is a fail-closed error.
        try:
            read_soft(b, CANDIDATE)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing series accession guard")
    print("SELF-TEST PASSED: study parsing, matched replicate titles, shared IDs, fail-closed guards")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--download-reference", action="store_true",
                   help="Fetch GSE153183 SOFT metadata only, then compare against Stage24 GSE341996")
    g.add_argument("--compare-local", action="store_true",
                   help="Both official SOFT files must already exist locally")
    p.add_argument("--candidate-soft", type=Path, default=CAND_SOFT)
    p.add_argument("--reference-soft", type=Path, default=REF_SOFT)
    p.add_argument("--stage24-json", type=Path, default=STAGE24)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.stage24_json.is_file():
        raise FileNotFoundError("Run Stage24 first; Stage24 eligibility report is required")
    stage24 = json.loads(args.stage24_json.read_text(encoding="utf-8"))
    if stage24.get("accession") != CANDIDATE or stage24.get("ran_classifier") is not False:
        raise ValueError("Unexpected or unsafe Stage24 metadata report")
    a = read_soft(args.candidate_soft, CANDIDATE)
    if a["soft_sha256"] != stage24.get("metadata_file_sha256"):
        raise ValueError("GSE341996 GEO SOFT metadata SHA does not match Stage24 report")
    if len(a["samples"]) != stage24.get("sample_count"):
        raise ValueError("GSE341996 sample count differs from Stage24 report")
    if args.download_reference:
        download_reference(args.reference_soft)
    b = read_soft(args.reference_soft, REFERENCE)
    report = compare(a, b)
    write_report(report, args.out)
    print("Stage25 metadata-only overlap audit:", report["verdict"])
    print("GSMs: candidate=%d reference=%d | matching condition+replicate labels=%d | shared experimental accessions=%d" %
          (len(a["samples"]), len(b["samples"]),
           len(report["same_condition_replicate_label_matches"]),
           len(report["shared_sra_or_biosample_ids"])))
    print("Saved:", args.out / "25_independence_gate.md")
    print("NO EXPRESSION OR MODEL READ; GSE341996 independent validation NOT approved")


if __name__ == "__main__":
    main()
