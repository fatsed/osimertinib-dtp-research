#!/usr/bin/env python3
"""Stage 43 | GSE249721 non-predictive gene-space & matrix-format audit.

Run from the osimertinib-dtp-research repository root:
  python scripts/43_gse249721_feature_eligibility_audit.py --self-test
  python scripts/43_gse249721_feature_eligibility_audit.py --audit-only

Requires existing local data/raw/GSE249721/* and frozen Stage40 package,
plus scripts/42_gse249721_eligibility_gate.py. NO download, NO training,
NO DTC predictions, NO use of target labels for a model or hyperparameter.

This reads the full matrix for **unlabelled format & numeric QC** only.
It records exact model-feature coverage, but NEVER modifies or imputes gene
identifiers, normalizes expression, fits a model or computes a class outcome.
Results stay in outputs/external_eligibility/GSE249721/ (git-ignored).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'data/raw/GSE249721/GSE249721_all_samples_processed_data.tsv.gz'
SOFT = ROOT / 'data/raw/GSE249721/GSE249721_family.soft.gz'
PACKAGE = ROOT / 'outputs/research_model_package/40_frozen_research_models.json'
AUDIT = ROOT / 'outputs/research_model_package/40_replay_audit.json'
ST42 = ROOT / 'scripts/42_gse249721_eligibility_gate.py'
OUT = ROOT / 'outputs/external_eligibility/GSE249721'
MODEL_NAMES = {
    f'{method}__{variant}'
    for method in ('acute_only_rank_lr100', 'dual_contrast_rank_lr100')
    for variant in ('stage28_baseline', 'contrast_protected_shrinkage_gamma1')
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage42(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Need existing frozen stage 42 script: {path}')
    spec = importlib.util.spec_from_file_location('stage42_frozen_metadata', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load stage 42: {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_models(path: Path, audit_path: Path) -> dict:
    if not path.is_file() or not audit_path.is_file():
        raise FileNotFoundError('Stage40 exported package and replay audit are required in outputs/research_model_package/')
    payload_bytes = path.read_bytes()
    pkg = json.loads(payload_bytes)
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if (audit.get('status') != 'REPLAY_VERIFIED_TECHNICAL_EXPORT'
        or hashlib.sha256(payload_bytes).hexdigest() != audit.get('package_payload_sha256')):
        raise ValueError('Stage40 package SHA-256 does NOT match the frozen replay audit; no audit will be produced')
    if (pkg.get('version') != 1
        or pkg.get('status') != 'RESEARCH_ONLY_NOT_INDEPENDENTLY_VALIDATED'
        or pkg.get('model_parameters',{}).get('threshold') != 0.0
        or set(pkg.get('models',{})) != MODEL_NAMES):
        raise ValueError('Unexpected frozen model package variant or decision threshold')
    for name, model in pkg['models'].items():
        genes = model.get('panel_genes', [])
        if (len(genes) != 100 or len(set(genes)) != 100
            or any(not isinstance(g,str) or not g for g in genes)
            or model.get('minimum_gene_coverage') != 90
            or model.get('threshold') != 0.0
            or len(model.get('rank_linear_weights',[])) != 100
            or len(model.get('source_mean_panel_ranks',[])) != 100):
            raise ValueError(f'Unexpected panel or policy in {name}')
    return pkg


def read_soft_titles(path: Path) -> dict:
    """Read sample titles from locally downloaded GEO family SOFT, not class labels."""
    if not path.is_file():
        raise FileNotFoundError(path)
    opener = gzip.open if path.suffix == '.gz' else open
    records = {}
    gsm = None
    with opener(path, 'rt', encoding='utf-8-sig', errors='replace') as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith('^SAMPLE') and '=' in line:
                gsm = line.partition('=')[2].strip()
                if gsm in records:
                    raise ValueError(f'Duplicate SOFT GSM section: {gsm}')
                if not re.fullmatch(r'GSM\d+', gsm):
                    raise ValueError(f'Invalid SOFT sample key: {gsm}')
                records[gsm] = None
            elif gsm and line.lower().startswith('!sample_title') and '=' in line:
                value = line.partition('=')[2].strip()
                if records[gsm] is not None:
                    raise ValueError(f'Duplicate SOFT title: {gsm}')
                records[gsm] = value
    return records


def classify_ids(ids: list[str]) -> dict:
    ens = sum(bool(re.fullmatch(r'ENSG\d{10,}(?:\.\d+)?', x)) for x in ids)
    numeric = sum(x.isdigit() for x in ids)
    return {
        'ensembl_like': ens,
        'entrez_numeric_like': numeric,
        'other_or_symbol_like': len(ids)-ens-numeric,
        'examples_first_8': ids[:8],
        'note': 'Type guess from row text only; no automatic Ensembl-to-symbol mapping or symbol normalization.'
    }


def scan_matrix(path: Path, expected_titles: set[str]) -> dict:
    """Full integrity + *unlabelled* numeric QC. Never computes class statistics.

    Streaming one row at a time; gene IDs retained, expression values discarded
    after shape/finite/nonnegative/unlabelled range checks. No model scoring.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError('Empty matrix')
        header = next(csv.reader([header_line], delimiter='\t'))
        if len(header) != 45:
            raise ValueError(f'Expected 1 ID + 44 samples, found {len(header)} columns')
        sample_cols = header[1:]
        if len(set(sample_cols)) != 44:
            raise ValueError('Duplicate matrix sample column labels')
        expected = {x+'.genes.results' for x in expected_titles}
        extras = sorted(set(sample_cols)-expected)
        absent = sorted(expected-set(sample_cols))
        # Audit checks all 44 expected titles, but never uses cohort labels to fit anything.
        genes = []
        duplicates = []
        seen = set()
        n_rows = 0
        invalid_values = 0
        negative_values = 0
        nonfinite_values = 0
        zeros = 0
        numeric_cells = 0
        integer_like = 0
        positive_min = float('inf')
        positive_max = 0.0
        total = 0.0
        line_width_errors = []
        for line_no, line in enumerate(f, start=2):
            if not line.strip():
                line_width_errors.append(line_no)
                continue
            cells = line.rstrip('\r\n').split('\t')
            if len(cells) != 45:
                line_width_errors.append(line_no)
                if len(line_width_errors) > 10:
                    raise ValueError(f'Too many malformed matrix rows: {line_width_errors[:10]}')
                continue
            gene_id = cells[0].strip()
            n_rows += 1
            if not gene_id:
                raise ValueError(f'Blank gene identifier at data row {line_no}')
            if gene_id in seen and len(duplicates) < 12:
                duplicates.append(gene_id)
            seen.add(gene_id)
            genes.append(gene_id)
            for val in cells[1:]:
                if val.strip() in ('', 'NA', 'NaN', 'nan', 'null', 'None'):
                    invalid_values += 1
                    continue
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    invalid_values += 1
                    continue
                if not math.isfinite(v):
                    nonfinite_values += 1
                    continue
                if v < 0:
                    negative_values += 1
                if v == 0:
                    zeros += 1
                elif v > 0:
                    positive_min = min(positive_min, v)
                    positive_max = max(positive_max, v)
                if abs(v-round(v)) <= 1e-6 * max(1.0, abs(v)):
                    integer_like += 1
                total += v
                numeric_cells += 1
    nvalues = n_rows * 44
    return {
        'matrix_first_header_field': header[0],
        'n_header_columns': len(header),
        'n_sample_columns': len(sample_cols),
        'sample_columns': sample_cols,
        'all_columns_match_geo_title_plus_genes_results': not extras and not absent,
        'unrecognized_sample_headers': extras[:12],
        'missing_expected_sample_headers': absent[:12],
        'gene_id_row_count': n_rows,
        'unique_gene_id_count': len(seen),
        'duplicate_gene_id_count': n_rows-len(seen),
        'duplicate_gene_ids_preview': duplicates,
        'gene_identifier_type': classify_ids(genes),
        'gene_ids': seen,
        'numeric_integrity': {
            'expected_expression_cells': nvalues,
            'valid_numeric_cells': numeric_cells,
            'missing_or_unparseable_values': invalid_values,
            'nonfinite_values': nonfinite_values,
            'negative_values': negative_values,
            'fraction_zeros_among_valid': (zeros/numeric_cells if numeric_cells else None),
            'fraction_integer_like_among_valid': (integer_like/numeric_cells if numeric_cells else None),
            'min_positive': (positive_min if positive_min != float('inf') else None),
            'max_positive': positive_max,
            'sum_all_44_samples': total,
            'warning': 'These global unlabeled summaries do NOT establish whether entries are RSEM expected_count, TPM or FPKM.'
        },
        'malformed_line_numbers_preview': line_width_errors[:10],
        'matrix_sha256': sha256_file(path),
    }


def compile_report(stage42, pkg: dict, matrix: dict, soft_titles: dict, soft_sha: str) -> tuple[dict, list[dict], list[dict]]:
    expected = stage42.rows()
    from_geo = {r['gsm']: r['geo_title'] for r in expected}
    soft_equal = len(soft_titles) == 44 and soft_titles == from_geo
    sample_set = set(matrix['sample_columns'])
    manifest = [dict(r, matrix_header=r['geo_title']+'.genes.results',
                matrix_found=(r['geo_title']+'.genes.results' in sample_set),
                soft_title_match=(soft_titles.get(r['gsm']) == r['geo_title'])) for r in expected]

    gene_ids = matrix['gene_ids']
    coverage = []
    for name,model in sorted(pkg['models'].items()):
        p = model['panel_genes']
        available = [g for g in p if g in gene_ids]
        missing = [g for g in p if g not in gene_ids]
        coverage.append({
            'model':name,
            'panel_n':len(p),
            'direct_gene_id_matches':len(available),
            'missing_gene_ids':';'.join(missing),
            'required_minimum':model['minimum_gene_coverage'],
            'passes_direct_coverage':len(available)>=model['minimum_gene_coverage']
        })

    reasons=[]
    if not matrix['all_columns_match_geo_title_plus_genes_results']:
        reasons.append('MATRIX_SAMPLE_HEADERS_NOT_EXACTLY_GEO_TITLES')
    if not soft_equal:
        reasons.append('GEO_SOFT_GSM_TITLE_MAPPING_NOT_CONFIRMED')
    if matrix['gene_id_row_count'] == 0:
        reasons.append('NO_FEATURE_ROWS')
    if matrix['duplicate_gene_id_count']:
        reasons.append('DUPLICATE_GENE_IDENTIFIERS')
    if matrix['malformed_line_numbers_preview']:
        reasons.append('MALFORMED_MATRIX_ROW_SHAPES')
    qc=matrix['numeric_integrity']
    if qc['missing_or_unparseable_values'] or qc['nonfinite_values'] or qc['negative_values'] or not qc['valid_numeric_cells']:
        reasons.append('INVALID_EXPRESSION_VALUES')
    if any(not r['passes_direct_coverage'] for r in coverage):
        reasons.append('ONE_OR_MORE_FROZEN_MODELS_BELOW_90_DIRECT_GENE_MATCHES')
    # Even if every technical feature check passes, expression UNIT isn't
    # guaranteed by the matrix header or RSEM naming conventions.
    unit_status = 'UNCONFIRMED_BY_GEO_AND_PAPER: RSEM mapped/counted; exact deposited columns unlabeled'
    if reasons:
        status = 'BLOCKED_TECHNICAL_ELIGIBILITY'
    else:
        status = 'TECHNICAL_ELIGIBILITY_PASSED_BUT_UNIT_AND_BIOLOGICAL_GATE_PENDING'

    output = {
        'gse':'GSE249721',
        'status':status,
        'approved_to_score':False,
        'reasons':reasons,
        'additional_pre_scoring_checks':['CONFIRM_EXACT_EXPRESSION_UNIT_AND_TRANSFORM',
            'INDEPENDENT_BIOLOGICAL_REPLICATE_AND_DTC_ENDPOINT_REVIEW',
            'PREDECLARED_ALL_FOUR_MODEL_STRESS_TEST_NO_SELECTION'],
        'raw_matrix_sha256':matrix['matrix_sha256'],
        'soft_sha256':soft_sha,
        'frozen_model_package_sha256':hashlib.sha256(json.dumps(pkg, ensure_ascii=False, sort_keys=True, indent=2,allow_nan=False).encode('utf-8').strip()+b'\n').hexdigest(),
        'n_samples':matrix['n_sample_columns'],
        'n_gene_rows':matrix['gene_id_row_count'],
        'n_unique_gene_ids':matrix['unique_gene_id_count'],
        'n_duplicate_gene_ids':matrix['duplicate_gene_id_count'],
        'sample_header_match':matrix['all_columns_match_geo_title_plus_genes_results'],
        'soft_titles_match_geo':soft_equal,
        'soft_records_found':len(soft_titles),
        'sample_header_errors':{
            'unrecognized':matrix['unrecognized_sample_headers'],
            'missing':matrix['missing_expected_sample_headers']},
        'gene_id_format':matrix['gene_identifier_type'],
        'numeric_integrity':matrix['numeric_integrity'],
        'expression_unit_status':unit_status,
        'model_coverage':coverage,
        'cohort_policy':{'primary_erlotinib_three_condition':18,
            'secondary_erlotinib_unmatched_control':8,
            'secondary_osimertinib_two_condition':6,
            'excluded_noncomparable':12},
        'data_use':'UNLABELLED_MATRIX_QC_ONLY; no training, no predictions, no model or threshold tuning',
        'historical_status':'GSE249721 previously metadata-screened; not an untouched independent validation claim'
    }
    return output, manifest, coverage


def csv_body(rows:list[dict])->str:
    if not rows:
        raise ValueError('No rows')
    out=io.StringIO(newline='')
    writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return out.getvalue()


def report_text(audit:dict)->str:
    c=audit['model_coverage']
    lines=[
        '# GSE249721 — Stage 43 feature and matrix eligibility audit',
        '',
        f"**Status:** `{audit['status']}` — **NO model scored; not approved for scoring**.",
        '',
        '## Matrix and metadata',
        '',
        f"- Sample columns: {audit['n_samples']}; exact GEO-title mapping: {audit['sample_header_match']}.",
        f"- SOFT sample titles: {audit['soft_records_found']}; exact GSM/title match: {audit['soft_titles_match_geo']}.",
        f"- Gene-ID rows: {audit['n_gene_rows']}; unique: {audit['n_unique_gene_ids']}; duplicated: {audit['n_duplicate_gene_ids']}.",
        f"- Gene ID examples: {audit['gene_id_format']['examples_first_8']!r}.",
        f"- Expression unit: {audit['expression_unit_status']}.",
        '',
        '## Frozen model feature ID matches',
        '',
        '| Model | Direct exact gene-ID matches | Minimum | Pass |',
        '|---|---:|---:|---|',
    ]
    for r in c:
        lines.append(f"| `{r['model']}` | {r['direct_gene_id_matches']}/100 | {r['required_minimum']} | {r['passes_direct_coverage']} |")
    lines.extend([
        '',
        '## Non-label-dependent numeric QC',
        '',
        f"- Missing/unparseable cells: {audit['numeric_integrity']['missing_or_unparseable_values']}; nonfinite: {audit['numeric_integrity']['nonfinite_values']}; negative: {audit['numeric_integrity']['negative_values']}.",
        f"- Fraction zeros: {audit['numeric_integrity']['fraction_zeros_among_valid']}; fraction integer-like: {audit['numeric_integrity']['fraction_integer_like_among_valid']}.",
        '',
        '## Blocking reasons',
        '',
    ])
    lines += ['- '+x for x in audit['reasons']] if audit['reasons'] else ['- All checked *technical* eligibility conditions passed; expression unit and biological gate still open.']
    lines.extend([
        '',
        '## Next research decision',
        '',
        'Do not train, score, fit, calibrate or select a method based on GSE249721 at this stage. '
        'If gene IDs need annotation, require a source-locked one-to-one mapping with duplicate/ambiguity guards. '
        'Resolve the exact processed expression measurement and biological replicate eligibility before a single predeclared 18-sample cross-drug stress test.',
        '',
        'Source: NCBI GSE249721 and Figarol et al. Nature Communications (2024), doi:10.1038/s41467-024-49360-4.',
        '',
        'Technical output is not validation and cannot establish biological or algorithmic novelty.',
        ''
    ])
    return '\n'.join(lines)


def immutable_outputs(mapping: dict[Path,str]):
    # Do not create some artifacts if another previously exists with conflicting bytes.
    for p, text in mapping.items():
        if p.exists() and p.read_bytes() != text.encode('utf-8'):
            raise FileExistsError(f'Refusing to overwrite changed eligibility result: {p}')
    for p, text in mapping.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(text.encode('utf-8'))


def audit(matrix_path=MATRIX, soft_path=SOFT, package_path=PACKAGE,
          package_audit_path=AUDIT, stage42_path=ST42, out=OUT):
    mod = load_stage42(Path(stage42_path))
    frozen = mod.rows()
    pkg = load_models(Path(package_path), Path(package_audit_path))
    matrix = scan_matrix(Path(matrix_path), set(r['geo_title'] for r in frozen))
    soft = read_soft_titles(Path(soft_path))
    summary, manifest, coverage = compile_report(mod, pkg, matrix, soft, sha256_file(Path(soft_path)))
    results = {
        Path(out)/'43_summary.json':json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',
        Path(out)/'43_sample_mapping.csv':csv_body(manifest),
        Path(out)/'43_feature_coverage.csv':csv_body(coverage),
        Path(out)/'43_report.md':report_text(summary),
    }
    immutable_outputs(results)
    print(f"Stage43 complete: rows={summary['n_gene_rows']} unique_genes={summary['n_unique_gene_ids']} samples={summary['n_samples']}; sample/soft match={summary['sample_header_match']}/{summary['soft_titles_match_geo']}")
    for r in coverage:
        print(f"  {r['model']}: exact_coverage={r['direct_gene_id_matches']}/100 passes_minimum={r['passes_direct_coverage']}")
    print(f"GATE: {summary['status']}; blockers: {summary['reasons'] or 'measurement unit and biological review still pending'}")
    print(f'Saved {Path(out)/"43_report.md"}')
    print('NO MODEL TRAINING, SCORING, PREDICTION, TUNING OR LABEL-DEPENDENT EXPRESSION REVIEW')
    return summary


def self_test():
    stage42 = load_stage42(ST42)
    source = stage42.rows()
    real_pkg = json.loads(PACKAGE.read_text(encoding='utf-8')) if PACKAGE.is_file() else None
    if real_pkg is not None:
        assert len(real_pkg['models']) == 4
    with tempfile.TemporaryDirectory(prefix='gse249721_43_') as td:
        root = Path(td)
        labels = [r['geo_title'] for r in source]
        fake_matrix = root/'matrix.tsv.gz'
        fake_soft = root/'family.soft.gz'
        with gzip.open(fake_soft,'wt',encoding='utf-8',newline='\n') as f:
            for r in source:
                f.write(f"^SAMPLE = {r['gsm']}\n!Sample_title = {r['geo_title']}\n")
        with gzip.open(fake_matrix,'wt',encoding='utf-8',newline='\n') as f:
            f.write('\t'+'\t'.join(t+'.genes.results' for t in labels)+'\n')
            for gene in ['CCL2','GSTM3','ENSG00000123456.3']:
                f.write(gene+'\t'+'\t'.join('2.5' for _ in range(44))+'\n')
        d = scan_matrix(fake_matrix,set(labels))
        assert d['n_sample_columns']==44 and d['gene_id_row_count']==3
        assert d['duplicate_gene_id_count']==0 and d['all_columns_match_geo_title_plus_genes_results']
        assert read_soft_titles(fake_soft)=={r['gsm']:r['geo_title'] for r in source}
        assert d['numeric_integrity']['missing_or_unparseable_values']==0
        assert d['gene_identifier_type']['ensembl_like']==1
        # Require fully frozen panel list, but synthetic matrix is intentionally below coverage.
        if real_pkg:
            summary, samples, cov = compile_report(stage42,real_pkg,d,read_soft_titles(fake_soft),sha256_file(fake_soft))
            assert not summary['approved_to_score'] and len(cov)==4
            assert 'ONE_OR_MORE_FROZEN_MODELS_BELOW_90_DIRECT_GENE_MATCHES' in summary['reasons']
            assert len(samples)==44 and sum(r['matrix_found'] for r in samples)==44
        # Duplicate identifiers must be counted.
        with gzip.open(fake_matrix,'at',encoding='utf-8',newline='\n') as f:
            f.write('CCL2\t'+'\t'.join('2' for _ in range(44))+'\n')
        d2=scan_matrix(fake_matrix,set(labels))
        assert d2['duplicate_gene_id_count']==1
        # Bad header must be caught.
        with gzip.open(fake_matrix,'wt',encoding='utf-8',newline='\n') as f:
            f.write('feature\t'+'\t'.join(labels[:43])+'\n')
        try:
            scan_matrix(fake_matrix,set(labels))
        except ValueError as ex:
            assert 'Expected 1 ID + 44 samples' in str(ex)
        else:
            raise AssertionError('Malformed matrix header was not rejected')
    print('SELF-TEST PASSED: 44-sample GEO header mapping, GSM/SOFT match, gene IDs, unlabeled numeric QC, duplicate and coverage guards')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-test',action='store_true')
    g.add_argument('--audit-only',action='store_true')
    x=p.parse_args()
    if x.self_test:
        self_test()
    else:
        audit()


if __name__=='__main__':
    main()
