"""Stage 18 | Read-only protocol-integrity and confounding audit.

Run from the repository root:
    python scripts/18_protocol_integrity_audit.py --self-test
    python scripts/18_protocol_integrity_audit.py
    python scripts/18_protocol_integrity_audit.py --full-hashes --run-upstream-self-tests

This is a provenance/integrity audit, NOT a rerun of model evaluation, an
independent validation, or a biological certification. It never modifies
Stage 01--17 inputs or historical results. Outputs only:
    outputs/protocol_audit/18_audit.json
    outputs/protocol_audit/18_audit.md

PASS means that a *specific* mechanically checked condition holds; it NEVER
means the entire ML pipeline is free of leakage or that class biology is valid.
The user must inspect all WARN/FAIL findings before making scientific claims.

Standard-library-only; Python 3.10+.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_GSE = "GSE193258"
TARGET_GSE = "GSE255958"
LINES = {"H1975", "HCC2935", "HCC827", "PC9"}
SOURCE_STATES = {"Control", "DTP"}
THREE_STATES = {"Control", "Acute", "DTP"}
SOURCE_RAW_NAME = "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"
TARGET_GROUPS = {
    "Control": [f"GSM808{x}" for x in (3374, 3375, 3376)],
    "Acute": [f"GSM808{x}" for x in (3377, 3378, 3379)],
    "DTP": [f"GSM808{x}" for x in (3380, 3381, 3382)],
    "Resistant": [f"GSM808{x}" for x in (3383, 3384, 3385)],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_source_rows(rows):
    """Return (bool, detail). Only validates identifiers/counts/fold disjointness."""
    if len(rows) != 24 or not all(isinstance(r, dict) for r in rows):
        return False, "Expected 24 Control/DTP protocol rows"
    ids = [r.get("gsm") for r in rows]
    if any(not x for x in ids) or len(set(ids)) != 24:
        return False, "Source GSM identifiers are missing or duplicated"
    c = Counter((r.get("cell_line"), r.get("state")) for r in rows)
    expected = Counter({(line, state): 3 for line in LINES for state in SOURCE_STATES})
    if c != expected:
        return False, f"Source line/state counts differ: {dict(c)}"
    for line in sorted(LINES):
        train = {r["gsm"] for r in rows if r["cell_line"] != line}
        test = {r["gsm"] for r in rows if r["cell_line"] == line}
        if len(train) != 18 or len(test) != 6 or train & test:
            return False, f"Invalid leave-one-cell-line-out split: {line}"
        if train | test != set(ids):
            return False, f"Incomplete LOCO split: {line}"
    return True, "24 unique Control/DTP GSMs; 4 cell lines x 3 reps x 2 states; disjoint 18/6 LOCO splits"


def validate_acute_metadata(rows, source_rows):
    """Check independent GEO-label table against the locked source cohort."""
    candidates = [r for r in rows if r.get("gse") == SOURCE_GSE]
    ids = [r.get("gsm") for r in candidates]
    if len(candidates) != 60 or len(set(ids)) != 60 or any(not x for x in ids):
        return False, f"Expected 60 unique GEO sample records, got {len(candidates)}"
    acute = [r for r in candidates if r.get("reported_treatment") == "osimertinib acute 24 hrs."]
    c = Counter(r.get("reported_cell_line") for r in acute)
    if c != Counter({line: 3 for line in LINES}):
        return False, f"Acute per-cell-line counts invalid: {dict(c)}"
    by_id = {r["gsm"]: r for r in candidates}
    if set(r["gsm"] for r in source_rows) & set(r["gsm"] for r in acute):
        return False, "Acute and locked Control/DTP GSM overlap"
    for s in source_rows:
        sample = by_id.get(s["gsm"])
        expected = "DMSO" if s["state"] == "Control" else "osimertinib DTPs 3w."
        if sample is None or sample.get("reported_cell_line") != s["cell_line"] or sample.get("reported_treatment") != expected:
            return False, f"GEO cell-line/treatment mismatch: {s['gsm']}"
    return True, "60 unique GEO rows; 12 distinct Acute samples; treatments match all 24 locked source GSMs"


def validate_alignment(rows, source_rows, geo_rows):
    """Metadata alignment integrity, NOT experimental biological identity."""
    seen = {r["gsm"] for r in source_rows}
    seen.update(r["gsm"] for r in geo_rows if r.get("gse") == SOURCE_GSE and
                r.get("reported_treatment") == "osimertinib acute 24 hrs.")
    aligned = [r for r in rows if r.get("gse") == SOURCE_GSE and r.get("filename") == SOURCE_RAW_NAME]
    matches = [r for r in aligned if r.get("gsm") in seen]
    counts = Counter(r.get("gsm") for r in matches)
    if len(matches) != 36 or counts != Counter({g: 1 for g in seen}):
        return False, "Not all 36 source/Acute GSMs have exactly one expression alignment"
    cols = [r.get("matrix_column") for r in matches]
    if any(not c for c in cols) or len(set(cols)) != len(cols):
        return False, "Expression matrix columns are empty or duplicated"
    if any(r.get("mapping_status") != "CANDIDATE_REQUIRES_REVIEW" for r in matches):
        return False, "Mapping-status field differs from the frozen candidate convention"
    return True, "36 GSM-to-expression column mappings are unique; mapping status remains candidate/review required"


def target_tar_members(path):
    """Inspect TAR member identifiers without extraction or decompression."""
    expected = {g for v in TARGET_GROUPS.values() for g in v}
    found = []
    with tarfile.open(path, "r") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".genes.results.gz"):
                continue
            gsms = re.findall(r"GSM\d+", Path(member.name).name)
            if len(gsms) == 1 and gsms[0] in expected:
                found.append(gsms[0])
    return Counter(found) == Counter({gsm: 1 for gsm in expected}), found


def audit(root: Path, *, full_hashes: bool = False, run_upstream_self_tests: bool = False):
    root = root.resolve()
    findings = []

    def put(check, status, evidence, scope):
        assert status in {"PASS", "WARN", "FAIL"}
        findings.append({"check": check, "status": status, "evidence": str(evidence), "scope": scope})

    # Stage code existence is only provenance, not an implementation correctness guarantee.
    absent = [f"scripts/{i:02d}_*.py" for i in range(1, 18)
              if len(list((root / "scripts").glob(f"{i:02d}_*.py"))) != 1]
    put("01_scripts_present", "FAIL" if absent else "PASS",
        f"Missing/duplicate numbered stage scripts: {absent}" if absent else
        "Exactly one Python script found for every stage 01--17", "repository structure")
    scripts = sorted((root / "scripts").glob("[0-1][0-9]_*.py")) if (root / "scripts").is_dir() else []
    script_hashes = {p.relative_to(root).as_posix(): sha256(p) for p in scripts}

    if (root / "README.md").is_file():
        put("02_readme", "PASS", "README.md exists (contents not assessed)", "documentation")
    else:
        put("02_readme", "WARN", "README.md missing; reproduction instructions need documenting", "documentation")
    ignore = root / ".gitignore"
    if ignore.is_file():
        s = ignore.read_text(encoding="utf-8-sig")
        required = ("/data/raw/", "/data/processed/", "/data/metadata/", "/outputs/", ".venv")
        missing = [token for token in required if token not in s]
        put("03_gitignore", "PASS" if not missing else "WARN",
            "Local data/outputs/venv ignore patterns found" if not missing else f"Missing patterns: {missing}",
            "publication hygiene; cannot prove no sensitive data was previously committed")
    else:
        put("03_gitignore", "WARN", "No .gitignore present", "publication hygiene")

    protocol_path = root / "data/metadata/evaluation_protocol.json"
    if not protocol_path.is_file():
        put("04_frozen_protocol", "FAIL", f"Missing {protocol_path.relative_to(root)}", "local metadata")
        protocol = None
    else:
        try:
            protocol = read_json(protocol_path)
            if protocol.get("source_study") != SOURCE_GSE:
                raise ValueError("Source accession unexpectedly changed")
            put("04_frozen_protocol", "PASS", "GSE193258 evaluation protocol exists and parses", "local metadata")
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            protocol = None
            put("04_frozen_protocol", "FAIL", f"Invalid evaluation protocol: {exc}", "local metadata")

    source = None
    if protocol is not None:
        source = protocol.get("source_gsm_to_label", [])
        ok, info = validate_source_rows(source)
        put("05_source_loco", "PASS" if ok else "FAIL", info, "protocol GSM and fold identities")
        outer = protocol.get("outer_folds")
        if isinstance(outer, list) and len(outer) == 4 and ok:
            expected_by_line = {line: (
                {r["gsm"] for r in source if r["cell_line"] != line},
                {r["gsm"] for r in source if r["cell_line"] == line},
            ) for line in LINES}
            actual_lines = [f.get("held_out_cell_line") for f in outer]
            valid = Counter(actual_lines) == Counter({x: 1 for x in LINES})
            for f in outer:
                line = f.get("held_out_cell_line")
                if line not in expected_by_line:
                    valid = False
                    continue
                tr, te = expected_by_line[line]
                if set(f.get("train_gsm", [])) != tr or set(f.get("test_gsm", [])) != te:
                    valid = False
            put("06_frozen_fold_manifest", "PASS" if valid else "FAIL",
                "Frozen outer-fold train/test GSM sets agree with source labels" if valid else
                "Frozen LOCO manifest differs from derived disjoint sets", "outer fold metadata")
        else:
            put("06_frozen_fold_manifest", "FAIL", "Invalid or missing outer_folds", "outer fold metadata")

        flags = protocol.get("flags", {})
        bio = flags.get("biology_approved_for_training") is True
        put("07_biological_approval", "PASS" if bio else "WARN",
            "Biology approved in protocol (approval evidence still requires review)" if bio else
            "Protocol has NOT approved biological training labels; do not claim biological certification",
            "scientific interpretation")
        if flags.get("historically_inspected_source_and_GSE255958") is True:
            put("08_external_independence", "WARN",
                "GSE255958 was previously inspected. Stage 17 is exploratory, NOT untouched validation",
                "external-validation status")
        else:
            put("08_external_independence", "WARN",
                "Historical inspection flag absent/different; cannot establish untouched validation",
                "external-validation status")
        if flags.get("gene_symbol_harmonization_complete") is not True:
            put("09_gene_harmonization", "WARN", "Cross-study gene symbol harmonization is not approved as complete",
                "cross-study compatibility")
        else:
            put("09_gene_harmonization", "PASS", "Protocol flags harmonization complete; evidence not independently checked",
                "cross-study compatibility")
        if flags.get("six_GSE303413_treatment_dose_discrepancies_need_resolution") is True:
            put("10_GSE303413_dose", "WARN", "Six treatment-dose discrepancies are explicitly unresolved",
                "target-label quality")
        else:
            put("10_GSE303413_dose", "WARN", "GSE303413 dose discrepancies require separate manual confirmation",
                "target-label quality")

        source_file = protocol.get("source_matrix")
        expected_sha = protocol.get("source_matrix_sha256")
        if not source_file or not expected_sha or len(expected_sha) != 64 or Path(source_file).is_absolute() or ".." in Path(source_file).parts:
            put("11_source_matrix", "FAIL", "Unsafe/missing source matrix relative path or checksum", "derived source")
        else:
            location = root / source_file
            if not location.is_file():
                put("11_source_matrix", "FAIL", f"Source matrix unavailable: {source_file}", "derived source")
            elif full_hashes:
                got = sha256(location)
                put("11_source_matrix", "PASS" if got == expected_sha else "FAIL",
                    "Derived source matrix SHA-256 matches protocol" if got == expected_sha else
                    f"Source matrix hash mismatch {got} != {expected_sha}", "derived source")
            else:
                put("11_source_matrix", "WARN",
                    "Source matrix exists but bytes were not hashed; rerun with --full-hashes", "derived source")

    geo = root / "data/metadata/geo_samples.csv"
    align = root / "data/metadata/expression_alignment.csv"
    if source is not None and validate_source_rows(source)[0]:
        if geo.is_file():
            try:
                g = read_csv(geo)
                ok, info = validate_acute_metadata(g, source)
                put("12_geo_acute_and_source", "PASS" if ok else "FAIL", info, "metadata/GSM")
            except (OSError, ValueError, KeyError) as exc:
                g = None
                put("12_geo_acute_and_source", "FAIL", f"Could not read GEO metadata: {exc}", "metadata/GSM")
        else:
            g = None
            put("12_geo_acute_and_source", "FAIL", "geo_samples.csv missing", "metadata/GSM")
        if align.is_file() and g is not None and validate_acute_metadata(g, source)[0]:
            try:
                ok, info = validate_alignment(read_csv(align), source, g)
                put("13_expression_mapping", "PASS" if ok else "FAIL", info, "metadata/expression")
            except (OSError, ValueError, KeyError) as exc:
                put("13_expression_mapping", "FAIL", f"Could not read expression alignments: {exc}", "metadata/expression")
        else:
            put("13_expression_mapping", "FAIL", "No validated GEO/source mapping input", "metadata/expression")
    else:
        put("12_geo_acute_and_source", "FAIL", "Protocol source rows unavailable/invalid", "metadata/GSM")
        put("13_expression_mapping", "FAIL", "Protocol source rows unavailable/invalid", "metadata/expression")

    # Downstream outputs are not required for an audit of source metadata alone;
    # absent outputs are reported explicitly (not treated as evidence of success).
    stage17_report = root / "outputs/three_class_external/17_summary.json"
    stage17_data = None
    if stage17_report.is_file():
        try:
            rep = read_json(stage17_report)
            stage17_data = rep
            ok = (rep.get("status") == "EXPLORATORY_NOT_INDEPENDENT" and
                  rep.get("source_study") == SOURCE_GSE and
                  rep.get("target_study") == TARGET_GSE and
                  rep.get("source_n") == 36 and rep.get("target_n") == 9 and
                  set(rep.get("excluded_resistant_gsms", [])) == set(TARGET_GROUPS["Resistant"]))
            put("14_stage17_report", "PASS" if ok else "FAIL",
                "Stage17 summary explicitly labels 36-to-9 Control/Acute/DTP exploratory transfer and Resistant exclusion"
                if ok else "Stage17 summary identifiers/status/classes/counts unexpected", "stored Stage17 output")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            put("14_stage17_report", "FAIL", f"Stage17 summary invalid: {exc}", "stored Stage17 output")
    else:
        put("14_stage17_report", "WARN", "Stage17 summary absent; cannot verify stored external scoring labels", "stored Stage17 output")

    excluded_report = root / "outputs/three_class_external/17_pc9_excluded_summary.json"
    if excluded_report.is_file():
        try:
            rep = read_json(excluded_report)
            ok = (rep.get("status") == "EXPLORATORY_NOT_INDEPENDENT" and
                  rep.get("source_pc9_excluded") is True and
                  rep.get("stage16_source_PC9_predictions_reproduced") is True and
                  rep.get("n_source_training") == 27 and rep.get("external_GSE255958_PC9_n") == 9)
            put("15_stage17_pc9_exclusion", "PASS" if ok else "FAIL",
                "Stored 27-source/9-target PC9-exclusion report has expected guard assertions"
                if ok else "Stored PC9-excluded summary missing required checks", "stored Stage17 sensitivity")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            put("15_stage17_pc9_exclusion", "FAIL", f"PC9 sensitivity report invalid: {exc}", "stored Stage17 sensitivity")
    else:
        put("15_stage17_pc9_exclusion", "WARN", "Stage17 PC9-exclusion summary absent", "stored Stage17 sensitivity")

    target_file = root / "data/raw/GSE255958/GSE255958_RAW.tar"
    if target_file.is_file():
        try:
            good, found = target_tar_members(target_file)
            put("16_target_archive_mapping", "PASS" if good else "FAIL",
                "12 expected unique target GSMs present as RSEM TAR members" if good else
                f"Target RSEM TAR members differ from 12 expected GSMs: {found}", "external archive names only")
        except (OSError, tarfile.TarError) as exc:
            put("16_target_archive_mapping", "FAIL", f"Unreadable target archive: {exc}", "external archive names only")
    else:
        put("16_target_archive_mapping", "WARN", "GSE255958 raw TAR absent; cannot verify external sample file names", "external archive")

    if full_hashes:
        raw_source = root / "data/raw/GSE193258" / SOURCE_RAW_NAME
        manifest_file = root / "data/metadata/expression_manifest.json"
        if not raw_source.is_file() or not manifest_file.is_file():
            put("21_source_raw_sha256", "FAIL",
                "Source raw gzip or expression_manifest.json absent", "source raw-data provenance")
        else:
            try:
                manifest = read_json(manifest_file)
                entry = manifest["manifest"][SOURCE_GSE + "/" + SOURCE_RAW_NAME]
                expected = entry["sha256"]
                got = sha256(raw_source)
                good = (len(expected) == 64 and got == expected and
                        entry.get("gzip_crc_verified") is True and
                        entry.get("reported_expression_unit") == "log2TPM_abundance")
                put("21_source_raw_sha256", "PASS" if good else "FAIL",
                    "Raw source gzip SHA-256/manifest check succeeded (GZIP CRC was verified at download time)"
                    if good else "Raw gzip digest or manifest metadata inconsistent",
                    "source raw-data provenance")
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                put("21_source_raw_sha256", "FAIL", str(exc), "source raw-data provenance")
        if target_file.is_file() and stage17_data is not None:
            expected = stage17_data.get("target_tar_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                put("22_external_tar_sha256", "FAIL", "Missing expected target TAR digest in Stage17 summary",
                    "external archive provenance")
            else:
                got = sha256(target_file)
                put("22_external_tar_sha256", "PASS" if got == expected else "FAIL",
                    "GSE255958 TAR SHA-256 matches stored Stage17 summary"
                    if got == expected else "Target TAR differs from stored Stage17 summary digest",
                    "external archive provenance")
        else:
            put("22_external_tar_sha256", "WARN", "Target TAR or Stage17 summary absent; cannot cross-check hash",
                "external archive provenance")

    put("17_duration_confounding", "WARN",
        "GSE193258: Acute 24h vs DTP 21d; treatment duration and state cannot be disentangled by this label contrast",
        "study design; not a code bug")
    put("18_processing_domain_shift", "WARN",
        "External raw expression uses target RSEM TPM log2(TPM+0.01); source matrix has a different processing provenance."
        " Rank denominators can differ by target gene universe.", "expression transfer; manual review required")
    if run_upstream_self_tests:
        for stage in range(13, 18):
            candidates = list((root / "scripts").glob(f"{stage:02d}_*.py"))
            if len(candidates) != 1:
                put(f"upstream_{stage:02d}_self_test", "FAIL",
                    "Expected exactly one stage script", "subprocess self-test")
                continue
            try:
                completed = subprocess.run([sys.executable, str(candidates[0]), "--self-test"],
                                           cwd=root, capture_output=True, text=True, timeout=150,
                                           check=False)
                transcript = (completed.stdout + "\n" + completed.stderr).strip()[-1000:]
                good = completed.returncode == 0 and "SELF-TEST PASSED" in completed.stdout
                put(f"upstream_{stage:02d}_self_test", "PASS" if good else "FAIL",
                    transcript if transcript else f"Exit code: {completed.returncode}; no output",
                    "synthetic upstream self-test, not a real-data rerun")
            except (OSError, subprocess.TimeoutExpired) as exc:
                put(f"upstream_{stage:02d}_self_test", "FAIL", str(exc),
                    "synthetic upstream self-test")
    else:
        put("20_upstream_self_tests", "WARN",
            "Upstream Stage13--17 --self-test not executed; use --run-upstream-self-tests",
            "runtime checks not performed")

    put("19_not_dynamic_leakage_proof", "WARN",
        "This read-only audit does not execute fit_transfer, replay model fitting, or prove absence of every leakage path."
        " Compare code + run self-tests separately.", "static audit limitations")

    commit = None
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=5, check=False)
        if proc.returncode == 0:
            commit = proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "stage": 18,
        "audit_name": "protocol_integrity_confounding_read_only",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": commit,
        "file_hashes": script_hashes,
        "full_hashes_checked": full_hashes,
        "scope": "Mechanical source/metadata/output/provenance checks; not independent validation",
        "status_counts": dict(Counter(r["status"] for r in findings)),
        "findings": findings,
    }


def render_md(report):
    lines = ["# Stage 18: Protocol Integrity & Confounding Audit", "",
             f"Generated UTC: {report['generated_utc']}", "",
             "**Scope:** read-only mechanical checks; not independent validation or biological approval.", "",
             "| Check | Status | Evidence |", "|---|---|---|"]
    for r in report["findings"]:
        detail = r["evidence"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{r['check']}` | **{r['status']}** | {detail} |")
    lines += ["", "## Interpretation", "",
              "- PASS certifies only the narrow mechanical condition named in that row.",
              "- WARN means a material limitation, not necessarily a code defect.",
              "- FAIL means an integrity check or required local input failed.",
              "- This report cannot prove a biological DTP-specific marker or independent generalization.",
              "- Do not select a winner based on historically inspected external targets.", ""]
    return "\n".join(lines)


def self_test():
    rows = [{"gsm": f"GSM_{line}_{state}_{i}", "cell_line": line, "state": state}
            for line in sorted(LINES) for state in sorted(SOURCE_STATES) for i in range(3)]
    ok, _ = validate_source_rows(rows)
    assert ok
    modified = [dict(r) for r in rows]
    modified[-1]["gsm"] = modified[0]["gsm"]
    assert not validate_source_rows(modified)[0]
    geo = []
    for r in rows:
        geo.append({"gse": SOURCE_GSE, "gsm": r["gsm"],
                    "reported_cell_line": r["cell_line"],
                    "reported_treatment": "DMSO" if r["state"] == "Control" else "osimertinib DTPs 3w."})
    for line in sorted(LINES):
        for i in range(3):
            geo.append({"gse": SOURCE_GSE, "gsm": f"GSM_{line}_Acute_{i}",
                        "reported_cell_line": line, "reported_treatment": "osimertinib acute 24 hrs."})
    for i in range(24):
        geo.append({"gse": SOURCE_GSE, "gsm": f"GSM_washout_{i}",
                    "reported_cell_line": "PC9", "reported_treatment": "washout"})
    assert validate_acute_metadata(geo, rows)[0]
    wrong = [dict(r) for r in geo]
    wrong[0]["reported_treatment"] = "osimertinib DTPs 3w."
    assert not validate_acute_metadata(wrong, rows)[0]
    all36 = [r for r in geo if r["gsm"].startswith("GSM_") and "washout" not in r["gsm"]]
    alignment = [{"gse": SOURCE_GSE, "filename": SOURCE_RAW_NAME,
                  "gsm": r["gsm"], "matrix_column": r["gsm"],
                  "mapping_status": "CANDIDATE_REQUIRES_REVIEW"} for r in all36]
    assert validate_alignment(alignment, rows, geo)[0]
    alignment[0]["matrix_column"] = alignment[1]["matrix_column"]
    assert not validate_alignment(alignment, rows, geo)[0]
    toy = {"generated_utc": "2026-11-18T00:00:00+00:00", "findings": [
        {"check": "smoke", "status": "WARN", "evidence": "Not independent", "scope": "test"}]}
    assert "Not independent" in render_md(toy)
    print("SELF-TEST PASSED: source LOCO, metadata labels, expression alignment, and Markdown report")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true", help="Run synthetic unit tests, no local data required")
    p.add_argument("--full-hashes", action="store_true", help="SHA-256 verify source derived/raw matrices and external TAR (slower)")
    p.add_argument("--run-upstream-self-tests", action="store_true",
                   help="Run existing Stage13--17 synthetic self-tests without redoing actual model scoring")
    p.add_argument("--root", type=Path, default=ROOT, help="Repository root; defaults to the parent of scripts/")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return 0
    report = audit(args.root, full_hashes=args.full_hashes,
                   run_upstream_self_tests=args.run_upstream_self_tests)
    out = args.root / "outputs/protocol_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "18_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "18_audit.md").write_text(render_md(report), encoding="utf-8")
    for item in report["findings"]:
        print(f"[{item['status']}] {item['check']}: {item['evidence']}")
    print(f"\nCounts: {report['status_counts']}")
    print("Saved: outputs/protocol_audit/18_audit.json and 18_audit.md")
    print("Interpretation: PASS is a narrow check, never independent or biological validation.")
    return 1 if any(x["status"] == "FAIL" for x in report["findings"]) else 0


if __name__ == "__main__":
    sys.exit(main())
