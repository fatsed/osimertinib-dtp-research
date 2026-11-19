#!/usr/bin/env python3
"""Stage 44 | Source-locked GSE249721 ID reconciliation (NO MODEL SCORING).

From osimertinib-dtp-research root:
  python scripts/44_gse249721_one_to_one_reconciliation.py --self-test
  python scripts/44_gse249721_one_to_one_reconciliation.py --audit-only

Reconciles only two preobserved formatting mismatches from Stage43:
  gene row '"ENSG00000000003_TSPAN6"' -> uniquely linked symbol 'TSPAN6'
  six matrix headers 'PC9-3_...' -> matching GEO titles 'PC9_3_...'

Strict one-to-one, duplicate/ambiguity guard. All Stage42, Stage43 and Stage40
source hashes/identities are verified first. Reads gene IDs and sample headers,
NOT expression values, target outcomes or model scores. No dataset download,
training, feature substitution, imputation, calibration or threshold choice.
Never overwrites Stage43 or any existing different Stage44 report. This does
NOT approve scoring: expression measurement unit and biological cohort gate
remain unresolved. Historical target inspection also limits independence.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'data/raw/GSE249721/GSE249721_all_samples_processed_data.tsv.gz'
SOFT = ROOT / 'data/raw/GSE249721/GSE249721_family.soft.gz'
PACKAGE = ROOT / 'outputs/research_model_package/40_frozen_research_models.json'
PKG_AUDIT = ROOT / 'outputs/research_model_package/40_replay_audit.json'
ST42 = ROOT / 'scripts/42_gse249721_eligibility_gate.py'
ST43 = ROOT / 'scripts/43_gse249721_feature_eligibility_audit.py'
SUMMARY43 = ROOT / 'outputs/external_eligibility/GSE249721/43_summary.json'
OUT = ROOT / 'outputs/external_eligibility/GSE249721'
ENSG_SYMBOL = re.compile(r'^(ENSG\d{11}(?:\.\d+)?)_([A-Za-z0-9._-]+)$')


def import_mod(path, name):
    if not path.is_file():
        raise FileNotFoundError(f'Expected prior script missing: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def parse_gene(raw: str):
    """Remove only paired exterior quotes; never guess aliases or versions."""
    x = raw.strip()
    if len(x) >= 2 and x[0] == '"' and x[-1] == '"':
        x = x[1:-1]
    match = ENSG_SYMBOL.fullmatch(x)
    if not match:
        return None
    return match.group(1), match.group(2)


def sample_mappings(headers: list[str], frozen: list[dict]):
    """Six exact, predeclared PC9-3 -> PC9_3 title aliases; no general cleanup."""
    titles = {r['geo_title'] + '.genes.results': r for r in frozen}
    if len(titles) != 44 or len(headers) != 44 or len(set(headers)) != 44:
        raise ValueError('Expected exactly 44 unique GEO records and matrix headers')
    out = []
    matched = set()
    aliases = 0
    for header in headers:
        candidate = header
        fix = 'EXACT'
        if candidate not in titles:
            if not candidate.startswith('PC9-3_'):
                raise ValueError(f'Unexpected/non-predeclared sample header: {candidate}')
            candidate = 'PC9_3_' + candidate[len('PC9-3_'):]
            fix = 'PC9_HYPHEN_TO_UNDERSCORE'
            aliases += 1
            if candidate not in titles:
                raise ValueError(f'Unrecognized PC9-3 sample alias: {header}')
        if candidate in matched:
            raise ValueError(f'Non-injective matrix-to-GEO sample mapping: {candidate}')
        matched.add(candidate)
        sample = titles[candidate]
        out.append(dict(matrix_header=header, geo_title=sample['geo_title'],
            gsm=sample['gsm'], cell_line=sample['cell_line'],
            mapping_policy=fix, cohort=sample['proposed_cohort']))
    if matched != set(titles):
        raise ValueError('Not every frozen GEO sample matched uniquely')
    if aliases != 6:
        raise ValueError(f'Expected exactly six previously identified PC9-3 aliases, found {aliases}')
    return out


def matrix_header_and_ids(path: Path):
    """Streaming audit of row identifiers only; all 44 expression columns skipped."""
    if not path.is_file():
        raise FileNotFoundError(path)
    h = gzip.open if path.suffix.lower() == '.gz' else open
    records=[]
    with h(path, 'rt', encoding='utf-8-sig', newline='') as f:
        header = next(csv.reader([f.readline()], delimiter='\t'))
        if len(header) != 45 or len(set(header[1:])) != 44:
            raise ValueError('Matrix header is not 1 identifier plus 44 unique samples')
        for line_no, line in enumerate(f, 2):
            if not line.strip():
                raise ValueError(f'Blank row at line {line_no}')
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) != 45:
                raise ValueError(f'Wrong number of columns at line {line_no}')
            records.append(fields[0].strip())
    return header[1:], records


def gene_mapping(ids: list[str], model_panels: dict[str,list[str]]):
    ensembl_seen=set()
    by_symbol=defaultdict(list)
    malformed=[]
    for raw in ids:
        parsed=parse_gene(raw)
        if not parsed:
            malformed.append(raw)
            continue
        ens,symbol=parsed
        if ens in ensembl_seen:
            raise ValueError(f'Duplicate Ensembl ID after parsing: {ens}')
        ensembl_seen.add(ens)
        by_symbol[symbol].append(ens)
    coverage=[]
    needed=sorted({gene for panel in model_panels.values() for gene in panel})
    gene_rows=[]
    for gene in needed:
        ensembl=sorted(by_symbol.get(gene,[]))
        status='UNIQUE' if len(ensembl)==1 else ('MISSING' if not ensembl else 'AMBIGUOUS')
        gene_rows.append(dict(symbol=gene, mapping_status=status,
            n_ensembl_matches=len(ensembl), ensembl_ids=';'.join(ensembl)))
    by_genename={x['symbol']:x for x in gene_rows}
    for name, panel in sorted(model_panels.items()):
        if len(panel)!=100 or len(set(panel))!=100:
            raise ValueError(f'Model panel not 100 unique genes: {name}')
        unique=[g for g in panel if by_genename[g]['mapping_status']=='UNIQUE']
        missing=[g for g in panel if by_genename[g]['mapping_status']=='MISSING']
        ambiguous=[g for g in panel if by_genename[g]['mapping_status']=='AMBIGUOUS']
        coverage.append(dict(model=name, unique_genes=len(unique),
            missing_genes=len(missing), ambiguous_genes=len(ambiguous),
            minimum_required=90, passes_threshold=len(unique)>=90,
            missing_symbols=';'.join(missing), ambiguous_symbols=';'.join(ambiguous)))
    return dict(n_rows=len(ids),parsed=len(ids)-len(malformed),
        unparsed=len(malformed), unparsed_examples=malformed[:12],
        unique_ensembl=len(ensembl_seen), unique_symbols=len(by_symbol),
        duplicate_symbol_count=sum(len(v)>1 for v in by_symbol.values()),
        coverage=coverage, gene_rows=gene_rows)


def immutable(output: dict[Path, str]):
    for path, body in output.items():
        if path.exists() and path.read_bytes()!=body.encode('utf-8'):
            raise FileExistsError(f'Existing different Stage44 output will not be overwritten: {path}')
    for path, body in output.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            path.write_bytes(body.encode('utf-8'))


def csv_text(rows):
    if not rows:
        raise ValueError('No data rows for CSV')
    io_buf=io.StringIO(newline='')
    w=csv.DictWriter(io_buf,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows)
    return io_buf.getvalue()


def report_md(summary):
    lines=['# Stage 44 — GSE249721 one-to-one identifier reconciliation', '',
      f"**Status:** `{summary['status']}`. **Scoring approved: NO.**",'',
      '## Sample identities', '',
      f"- Matched 44/44 GEO samples uniquely; exactly {summary['sample_alias_count']} explicitly predeclared PC9-3/PC9_3 alias corrections.",
      '- Matched against frozen Stage42 GEO identities and local SOFT titles; expression values never accessed here.','',
      '## Gene symbols from composite row IDs','',
      f"- Row IDs parsed: {summary['n_parsed_gene_rows']}/{summary['n_gene_rows']}.",
      f"- Distinct symbols: {summary['unique_gene_symbols']}; symbol collisions: {summary['duplicate_symbols_in_matrix']}.",
      '- Parsed format: quoted or unquoted ENSG + underscore + suffix. Suffix used as deposited symbol; no external annotation map or heuristic synonym correction.','',
      '| Frozen model | Unique panel genes | Missing | Ambiguous | Coverage >=90 |',
      '|---|---:|---:|---:|---|']
    for x in summary['model_coverage']:
        lines.append(f"| `{x['model']}` | {x['unique_genes']}/100 | {x['missing_genes']} | {x['ambiguous_genes']} | {x['passes_threshold']} |")
    lines.extend(['', '## Scoring eligibility NOT granted', '',
      '- The exact deposited expression unit/transform remains unconfirmed (counts vs TPM vs log-transformed values).',
      '- Historical inspection/biological independence and DTC endpoint differences still limit interpretation.',
      '- No target score, class-wise expression comparison, fitting, calibration or method selection performed.',
      '- Use only the frozen Stage40 four-model policy if a later independent, predeclared gate is approved.', '',
      '## Additional blockers',''])
    lines += ['- '+x for x in summary['remaining_technical_blockers']] if summary['remaining_technical_blockers'] else ['- ID-level technical checks passed; measurement and biology remain pending.']
    lines.append('')
    return '\n'.join(lines)


def run():
    s42=import_mod(ST42,'stage42_readonly')
    s43=import_mod(ST43,'stage43_readonly')
    frozen=s42.rows()
    pkg=s43.load_models(PACKAGE,PKG_AUDIT)
    previous=json.loads(SUMMARY43.read_text(encoding='utf-8'))
    if (previous.get('gse')!='GSE249721' or previous.get('status')!='BLOCKED_TECHNICAL_ELIGIBILITY'
        or previous.get('n_samples')!=44 or previous.get('n_gene_rows')!=34194
        or previous.get('approved_to_score') is not False):
        raise ValueError('Stage43 frozen summary does not match expected, abort')
    if sha(MATRIX)!=previous.get('raw_matrix_sha256') or sha(SOFT)!=previous.get('soft_sha256'):
        raise ValueError('Raw/metadata files have changed since Stage43; do not continue')
    soft=s43.read_soft_titles(SOFT)
    if soft != {r['gsm']:r['geo_title'] for r in frozen}:
        raise ValueError('SOFT titles do not match frozen Stage42 identities')
    if (previous.get('numeric_integrity',{}).get('missing_or_unparseable_values')!=0
        or previous['numeric_integrity'].get('nonfinite_values')!=0
        or previous['numeric_integrity'].get('negative_values')!=0):
        raise ValueError('Stage43 numeric QC not passed')
    headers, ids=matrix_header_and_ids(MATRIX)
    samples=sample_mappings(headers,frozen)
    if len(ids)!=previous['n_gene_rows']:
        raise ValueError('Stage43 gene row count mismatch')
    models={name:obj['panel_genes'] for name,obj in pkg['models'].items()}
    gm=gene_mapping(ids,models)
    blockers=[]
    if gm['unparsed']:
        blockers.append('UNRECOGNIZED_GENE_ROW_IDENTIFIERS')
    if any(x['unique_genes']<90 for x in gm['coverage']):
        blockers.append('FROZEN_PANEL_COVERAGE_BELOW_90_AFTER_UNIQUE_SYMBOL_MAPPING')
    # Any duplicates of selected symbols are excluded rather than combined; a
    # model can still pass if >=90 remaining unique features, consistent with Stage40.
    status=('IDENTITY_TECHNICAL_GATE_PASSED_UNIT_AND_BIOLOGY_PENDING' if not blockers
            else 'BLOCKED_TECHNICAL_IDENTITY_GATE')
    result=dict(stage=44, gse='GSE249721', status=status,
        approved_to_score=False, previous_stage43_status=previous['status'],
        raw_matrix_sha256=previous['raw_matrix_sha256'],
        soft_sha256=previous['soft_sha256'], stage43_summary_sha256=sha(SUMMARY43),
        frozen_model_package_sha256=sha(PACKAGE),
        sample_count=len(samples), sample_alias_count=sum(x['mapping_policy']!='EXACT' for x in samples),
        n_gene_rows=len(ids), n_parsed_gene_rows=gm['parsed'],
        n_unparsed_gene_rows=gm['unparsed'], unparsed_examples=gm['unparsed_examples'],
        unique_gene_symbols=gm['unique_symbols'], duplicate_symbols_in_matrix=gm['duplicate_symbol_count'],
        model_coverage=gm['coverage'], remaining_technical_blockers=blockers,
        measurement_unit='UNCONFIRMED',
        historical_status='Previously metadata-inspected; not an untouched validation cohort',
        next_gate='Verify exact expression unit/transform and biological eligibility before any prediction.',
        no_scoring_no_training_no_tuning=True)
    result_files={
        OUT/'44_summary.json':json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',
        OUT/'44_sample_mapping.csv':csv_text(samples),
        OUT/'44_selected_gene_mapping.csv':csv_text(gm['gene_rows']),
        OUT/'44_report.md':report_md(result),
    }
    immutable(result_files)
    print(f"Stage44: sample mapping={len(samples)}/44 (exact PC9 aliases={result['sample_alias_count']}); ENSG_symbol rows={gm['parsed']}/{len(ids)}")
    for x in gm['coverage']:
        print(f"  {x['model']}: unique={x['unique_genes']}/100 missing={x['missing_genes']} ambiguous={x['ambiguous_genes']} pass={x['passes_threshold']}")
    print(f"GATE: {status}; scoring NOT approved (expression unit and biological review pending)")
    print(f"Saved {OUT/'44_report.md'}")
    return result


def self_test():
    assert parse_gene('"ENSG00000000003_TSPAN6"')==('ENSG00000000003','TSPAN6')
    assert parse_gene('ENSG00000000005.7_TNMD')==('ENSG00000000005.7','TNMD')
    assert parse_gene('CCL2') is None
    assert parse_gene('"ENSG123_CCL2"') is None
    s42=import_mod(ST42,'stage42_test_only')
    records=s42.rows()
    hdr=[r['geo_title']+'.genes.results' for r in records]
    alias_indices=[i for i,h in enumerate(hdr) if h.startswith('PC9_3_')]
    assert len(alias_indices)==6
    for i in alias_indices: hdr[i]=hdr[i].replace('PC9_3_','PC9-3_',1)
    x=sample_mappings(hdr,records)
    assert len(x)==44 and sum(z['mapping_policy']!='EXACT' for z in x)==6
    try:
        bad=list(hdr);bad[0]=bad[1]
        sample_mappings(bad,records)
        raise AssertionError('Duplicate sample header accepted')
    except ValueError: pass
    try:
        bad=list(hdr);bad[0]='UNKNOWN_SAMP.genes.results'
        sample_mappings(bad,records)
        raise AssertionError('Unknown title accepted')
    except ValueError: pass
    genes=['CCL2','IGFBP5','NPR3']
    ids=[f'"ENSG{i:011d}_{g}"' for i,g in enumerate(genes,1)]
    m=gene_mapping(ids, {'small':genes}) if False else None
    # Exercise full 100-gene panel with one-to-one identifiers and a duplicate.
    panel=[f'GENE{i}' for i in range(100)]
    ids=[f'"ENSG{i:011d}_{gene}"' for i,gene in enumerate(panel,1)]
    one=gene_mapping(ids,{'fake':panel})
    assert one['coverage'][0]['unique_genes']==100 and one['coverage'][0]['passes_threshold']
    ids.append('"ENSG99999999999_GENE4"')
    d=gene_mapping(ids,{'fake':panel})
    assert d['coverage'][0]['unique_genes']==99 and d['coverage'][0]['ambiguous_genes']==1
    with tempfile.TemporaryDirectory(prefix='stage44_reconciliation_') as tmp:
        test=Path(tmp)/'matrix.tsv.gz'
        with gzip.open(test,'wt',encoding='utf-8',newline='\n') as f:
            f.write('gene_id\t'+'\t'.join(hdr)+'\n')
            for raw in ids:
                f.write(raw+'\t'+'\t'.join('0.0' for _ in range(44))+'\n')
        hm,ri=matrix_header_and_ids(test)
        assert hm==hdr and ri==ids
    print('SELF-TEST PASSED: 6 exact PC9 aliases, strict unique ENSG_symbol mapping, collision guards, frozen panels, header-only scan')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mx=parser.add_mutually_exclusive_group(required=True)
    mx.add_argument('--self-test',action='store_true')
    mx.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:run()


if __name__=='__main__':
    main()
