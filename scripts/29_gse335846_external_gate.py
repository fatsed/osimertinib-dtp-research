"""Stage 29 — freeze external evaluation design and inspect GSE335846 metadata.

Copy to scripts/29_gse335846_external_gate.py in osimertinib-dtp-research.

  python scripts/29_gse335846_external_gate.py --self-test
  python scripts/29_gse335846_external_gate.py --download

This is a NO-SCORING gate: never trains a model or reads any expression VALUES.
Only inspects official GEO SOFT metadata and the supplementary matrix HEADER.
The Stage 28 candidate recipes and source-study fingerprints are frozen before
new external files are downloaded; saved protocol may not silently change.

New cohort contains 5 Day-0 PC9 and 4 Day-28 PC9 FUCCI-sorted samples.
It does NOT contain an Acute condition. Day-28 is a study condition, NOT an
individually verified DTP label. It is not an independent cell-line holdout
and the number of independent biological replicates is not established.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCESSION = "GSE335846"
RAW = ROOT / "data" / "raw" / ACCESSION
OUT = ROOT / "outputs" / "external_eligibility" / ACCESSION
SERIES = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE335nnn/{ACCESSION}"
SOFT_URL = f"{SERIES}/soft/{ACCESSION}_family.soft.gz"
MATRIX_URL = f"{SERIES}/suppl/{ACCESSION}_RNA-seq_for_GEO.txt.gz"
SOFT_PATH = RAW / f"{ACCESSION}_family.soft.gz"
MATRIX_PATH = RAW / f"{ACCESSION}_RNA-seq_for_GEO.txt.gz"
METHODS = ("acute_only_rank_lr100", "dual_contrast_rank_lr100")
EXPECTED = {
    "GSM9822015": ("Day 0 G1 clone 1", "Day0", "G1", "1"),
    "GSM9822016": ("Day 0 G1 clone 7", "Day0", "G1", "7"),
    "GSM9822017": ("Day 0 S clone 1", "Day0", "S", "1"),
    "GSM9822018": ("Day 0 S clone 7", "Day0", "S", "7"),
    "GSM9822019": ("Day 0 G2 clone 1", "Day0", "G2", "1"),
    "GSM9822020": ("Day 28 G1 clone 1", "Day28", "G1", "1"),
    "GSM9822021": ("Day 28 G1 clone 7", "Day28", "G1", "7"),
    "GSM9822022": ("Day 28 S clone 1", "Day28", "S", "1"),
    "GSM9822023": ("Day 28 G2 clone 1", "Day28", "G2", "1"),
}
# Fix recipes before inspecting target expression. Only 3 lines WITHOUT PC9.
PROTOCOL = {
    "study": ACCESSION,
    "status": "FROZEN_DESIGN_PENDING_GEO_FORMAT_AND_BIOLOGICAL_REVIEW",
    "purpose": "exploratory external study/day contrast, not confirmatory DTP specificity",
    "development_study": "GSE193258",
    "training_cell_lines": ["H1975", "HCC2935", "HCC827"],
    "excluded_training_cell_line": "PC9",
    "source_training_count": 27,
    "source_training_class_counts": {"Control": 9, "Acute": 9, "DTP": 9},
    "target_sample_count": 9,
    "target_conditions": {"Day0": 5, "Day28": 4},
    "target_label_semantics": "Day28 is a treatment-duration condition; DTP is NOT verified per cell",
    "methods": list(METHODS),
    "training_objective": "DTP vs Control+Acute, same as stages 27-28",
    "selector": "Stage 28 source-training only, k=100, unchanged",
    "source_representation": "within-sample rank across selected genes",
    "model": "StandardScaler then LogisticRegression(C=0.01, solver=liblinear, class_weight=balanced, random_state=42)",
    "decision_threshold": 0.0,
    "external_feature_coverage": "require exact one-to-one gene identifiers for all selected 100 genes; otherwise stop and review",
    "forbidden": ["target-label-derived training", "target-based threshold tuning", "posthoc winner selection", "silently pooling TPM and raw counts"],
    "evaluation_after_approval": ["confusion matrix Day0 vs Day28", "balanced accuracy", "false positive Day0", "Day28 recall", "scores by cell-cycle phase", "gene coverage"],
    "no_acute_group": True,
    "previously_inspected_development_folds": True,
    "sample_names": {k: v[0] for k, v in EXPECTED.items()},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(data: dict) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def immutable_write(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing frozen file differs, refusing overwrite: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def source_fingerprints(root: Path) -> dict:
    files = [
        "scripts/27_dtp_specific_contrast_benchmark.py",
        "scripts/28_dual_contrast_ablation.py",
        "outputs/dtp_specificity/27_summary.json",
        "outputs/dtp_specificity/28_report.md",
        "outputs/dtp_specificity/28_predictions.csv",
        "outputs/dtp_specificity/28_selected_genes.csv",
        "data/metadata/evaluation_protocol.json",
    ]
    missing = [rel for rel in files if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError("Stage27/28 source files missing: " + ", ".join(missing))
    report = json.loads((root / "outputs/dtp_specificity/27_summary.json").read_text(encoding="utf-8"))
    if report.get("methods") != ["raw_lr500", "rank_anova_lr100", "dual_contrast_rank_lr100"] or report.get("n_source") != 36:
        raise ValueError("Frozen Stage27 protocol does not match expected cohort")
    ablation = (root / "outputs/dtp_specificity/28_predictions.csv")
    with ablation.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if Counter(row.get("method") for row in rows) != Counter({
        "raw_lr500": 36, "rank_anova_lr100": 36, "dual_contrast_rank_lr100": 36,
        "control_only_rank_lr100": 36, "acute_only_rank_lr100": 36,
    }):
        raise ValueError("Stage28 predictions do not have the five frozen methods")
    return {rel: sha256(root / rel) for rel in files}


def freeze(root: Path, out: Path) -> dict:
    contract = dict(PROTOCOL)
    contract["source_fingerprints_sha256"] = source_fingerprints(root)
    immutable_write(out / "29_frozen_protocol.json", canonical(contract))
    return contract


def download_limited(url: str, target: Path, max_bytes: int) -> None:
    if target.is_file():
        if target.stat().st_size == 0:
            raise ValueError(f"Existing file is empty: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchMetadataAudit/1.0 (NCBI GEO public data)"})
    temp = target.with_name(target.name + ".part")
    count = 0
    try:
        with urllib.request.urlopen(req, timeout=75) as response, temp.open("wb") as f:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > max_bytes:
                    raise ValueError("Unexpectedly large download, refusing")
                f.write(block)
        if count == 0:
            raise ValueError("Downloaded empty file")
        with gzip.open(temp, "rb") as f:
            f.read(1)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def parse_soft(path: Path) -> dict:
    samples = {}
    series = {}
    gsm = None
    with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("^SAMPLE = "):
                gsm = line.split("=", 1)[1].strip()
                if gsm in samples:
                    raise ValueError("Duplicate GSM in GEO SOFT")
                samples[gsm] = {}
            elif line.startswith("^SERIES = "):
                gsm = None
                series["accession"] = line.split("=", 1)[1].strip()
            elif line.startswith("^"):
                gsm = None
            elif line.startswith("!Sample_") and gsm in samples:
                key, _, value = line[1:].partition(" = ")
                if key in {"Sample_title", "Sample_source_name_ch1", "Sample_characteristics_ch1",
                           "Sample_data_processing", "Sample_treatment_protocol_ch1", "Sample_extract_protocol_ch1"}:
                    samples[gsm].setdefault(key, []).append(value[:2000])
            elif line.startswith("!Series_"):
                key, _, value = line[1:].partition(" = ")
                if key in {"Series_title", "Series_overall_design", "Series_summary", "Series_supplementary_file"}:
                    series.setdefault(key, []).append(value[:3000])
    if series.get("accession") != ACCESSION:
        raise ValueError("Unexpected GEO accession")
    if set(samples) != set(EXPECTED):
        raise ValueError(f"Unexpected GEO GSM IDs; expected {len(EXPECTED)}, saw {len(samples)}")
    out = []
    for gsm, expected in EXPECTED.items():
        title = samples[gsm].get("Sample_title", [""])[0]
        if title.casefold().strip() != expected[0].casefold():
            raise ValueError(f"Official title differs for {gsm}: {title!r}")
        out.append({"gsm": gsm, "title": title, "day_group": expected[1],
                    "cell_cycle_phase": expected[2], "clone": expected[3],
                    "official_remarks": " | ".join(samples[gsm].get("Sample_characteristics_ch1", []))[:750]})
    return {"series": series, "sample_rows": out,
            "processing_descriptions": sorted(set(x for v in samples.values() for x in v.get("Sample_data_processing", [])))[:20]}


def inspect_matrix_header(path: Path) -> dict:
    # IMPORTANT: do NOT inspect expression numbers or outcomes. Header only.
    with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace") as f:
        first = []
        for line in f:
            if not line.strip():
                continue
            first.append(line.strip("\r\n"))
            if len(first) == 3:
                break
    if not first:
        raise ValueError("Empty expression matrix")
    header = first[0]
    delimiter = "\t" if header.count("\t") >= header.count(",") else ","
    names = next(csv.reader([header], delimiter=delimiter))
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("Matrix header is not clearly sample-by-gene; manual inspection required")
    return {"delimiter": "tab" if delimiter == "\t" else "comma", "header_columns": names[:70],
            "header_column_count": len(names), "note": "No expression VALUES inspected. Column names require manual mapping to GSMs."}


def write_report(out: Path, result: dict) -> None:
    immutable_write(out / "29_screen.json", canonical(result))
    a = result["parsed"]
    expected = Counter(r["day_group"] for r in a["sample_rows"])
    lines = ["# Stage 29 — independent external cohort format and eligibility gate", "",
             "**Status: SCREENED, NO MODEL SCORING. Independent external biological validity not yet approved.**", "",
             f"- Study: {ACCESSION}", "- PC9, day zero vs day 28 osimertinib, FUCCI sorted by cell-cycle phase",
             f"- Samples: {len(a['sample_rows'])}; day counts: {dict(expected)}", "- No Acute arm; day 28 is not a confirmed individual DTP label",
             "- Evaluation pre-frozen: source H1975/HCC2935/HCC827 only, 27 samples",
             f"- Pre-frozen methods: {', '.join(METHODS)}", "- Decision threshold fixed at zero", "",
             "## Official sample IDs", "", "| GSM | Group | Phase | Clone | Title |", "|---|---|---|---|---|"]
    lines.extend(f"| {r['gsm']} | {r['day_group']} | {r['cell_cycle_phase']} | {r['clone']} | {r['title']} |"
                 for r in a["sample_rows"])
    lines.extend(["", "## Source processing and matrix header", "",
                  f"- Header columns ({result['matrix_header']['header_column_count']}): ",
                  "```", " | ".join(result["matrix_header"]["header_columns"]), "```", "",
                  "- Explicit units must be checked against official sample processing metadata. Do not assume TPM.", "- Gene identifiers must map one-to-one to 100 source-selected genes for both models.",
                  "- Check biological independence, reporting batch and dependence of repeat clones.",
                  "- The outcome labels have not been used to choose a model or tune threshold.",
                  "", "## Recorded source processing descriptions", ""])
    lines.extend("- " + s for s in a["processing_descriptions"][:12])
    lines.extend(["", "**Gate outcome: NEEDS_EXPRESSION_FORMAT_AND_PROVENANCE_REVIEW before any fit or scoring.**", ""])
    immutable_write(out / "29_report.md", "\n".join(lines).encode("utf-8"))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        test = root / "test.soft.gz"
        soft = [f"^SERIES = {ACCESSION}", "!Series_title = test", "!Series_overall_design = synthetic"]
        for gsm, e in EXPECTED.items():
            soft.extend([f"^SAMPLE = {gsm}", f"!Sample_title = {e[0]}", "!Sample_data_processing = synthetic"])
        with gzip.open(test, "wt", encoding="utf-8") as f:
            f.write("\n".join(soft) + "\n")
        parsed = parse_soft(test)
        assert Counter(r["day_group"] for r in parsed["sample_rows"]) == {"Day0": 5, "Day28": 4}
        with gzip.open(root / "tiny.txt.gz", "wt") as f:
            f.write("gene\tGSM9822015\tGSM9822020\nMYGENE\t12\t3\n")
        m = inspect_matrix_header(root / "tiny.txt.gz")
        assert m["header_column_count"] == 3 and m["header_columns"][1] == "GSM9822015"
        p = root / "frozen.json"
        immutable_write(p, b"original")
        try:
            immutable_write(p, b"changed")
            raise AssertionError("No mutation guard")
        except RuntimeError:
            pass
        try:
            bad = root / "bad.soft.gz"
            with gzip.open(bad, "wt") as f:
                f.write("^SERIES = GSE000000\n^SAMPLE = GSM0\n")
            parse_soft(bad)
            raise AssertionError("Bad provenance accepted")
        except ValueError:
            pass
    print("SELF-TEST PASSED: nine GEO sample identities, frozen-file guard, matrix header only, no expression values")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.download:
        p.error("Run --self-test or --download")
    contract = freeze(ROOT, OUT)  # always BEFORE target download or inspection
    print("Frozen source candidate recipes/provenance before external data download")
    download_limited(SOFT_URL, SOFT_PATH, 80 * 1024 * 1024)
    download_limited(MATRIX_URL, MATRIX_PATH, 100 * 1024 * 1024)
    parsed = parse_soft(SOFT_PATH)
    header = inspect_matrix_header(MATRIX_PATH)
    result = {"study": ACCESSION, "status": "NEEDS_EXPRESSION_FORMAT_AND_PROVENANCE_REVIEW",
              "no_classifier_fitted": True, "no_expression_values_inspected": True,
              "protocol_sha256": sha256(OUT / "29_frozen_protocol.json"),
              "official_soft_sha256": sha256(SOFT_PATH), "supplementary_matrix_sha256": sha256(MATRIX_PATH),
              "parsed": parsed, "matrix_header": header}
    write_report(OUT, result)
    print(f"Stage29: {len(parsed['sample_rows'])} GEO samples screened, {header['header_column_count']} matrix header fields")
    print("NO MODEL RUN; no expression values read; target cohort not approved for scoring")
    print(f"Saved {OUT / '29_report.md'}")


if __name__ == "__main__":
    main()
