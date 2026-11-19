#!/usr/bin/env python3
"""Stage 46 | Strictly exploratory, FROZEN-model count-space stress probe.

IMPORTANT: GSE249721 deposited values are likely transformed RSEM counts,
not confirmed TPM. The four source models used log2TPM and were NOT validated
on this measurement space. Thus scores from this script are a deliberately
mismatched-measurement sensitivity analysis, NOT external validation, not a
publication-ready generalization benchmark, and not permission to tune a model.

Previously viewed target metadata and outcomes further preclude untouched
validation. All four Stage40 models and all 18 predeclared primary samples are
scored together, without selection, refitting, calibration or cutoff changes.

Run from repository root, Windows PowerShell:
    python scripts/46_gse249721_frozen_count_space_probe.py --self-test
    python scripts/46_gse249721_frozen_count_space_probe.py --exploratory-count-space-probe

Needs local Stage42-45 files/data exactly as used in prior audit. Never downloads
anything; never changes raw files or prior reports; output is write-once.
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
PACKAGE = ROOT / 'outputs/research_model_package/40_frozen_research_models.json'
OUT = ROOT / 'outputs/external_eligibility/GSE249721'
STAGE40 = ROOT / 'scripts/40_frozen_research_model_package.py'
STAGE42 = ROOT / 'scripts/42_gse249721_eligibility_gate.py'
ENSEMBL_ID = re.compile(r'^ENSG\d{11}(?:\.\d+)?_([A-Za-z0-9._-]+)$')
SAMPLE_GROUP = 'primary_erlotinib_three_condition'
METHODS = ('acute_only_rank_lr100','dual_contrast_rank_lr100')
VARIANTS = ('stage28_baseline','contrast_protected_shrinkage_gamma1')
MODEL_NAMES = tuple(m+'__'+v for m in METHODS for v in VARIANTS)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path: Path) -> list[dict]:
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def import_py(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_once(outputs: dict[Path,str]):
    for path,body in outputs.items():
        if path.exists() and path.read_bytes() != body.encode('utf-8'):
            raise FileExistsError(f'Refusing to overwrite different Stage46 artifact: {path}')
    for path,body in outputs.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            path.write_bytes(body.encode('utf-8'))


def to_csv(rows: list[dict]) -> str:
    if not rows:
        raise ValueError('No rows to write')
    buf=io.StringIO(newline='')
    w=csv.DictWriter(buf,fieldnames=list(rows[0].keys()),lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def safe_symbol(raw: str) -> str | None:
    raw = raw.strip()
    if len(raw)>=2 and raw[0]=='"' and raw[-1]=='"':
        raw = raw[1:-1]
    match=ENSEMBL_ID.fullmatch(raw)
    return match.group(1) if match else None


def read_selected_matrix(matrix: Path, approved_gene_ids: dict[str,str],
                         selected_columns: list[str], expected_column_count: int=44,
                         expected_rows: int=34194) -> dict[str,dict[str,float]]:
    """Read only frozen-panel rows for selected samples. Gene mapping is fixed.

    approved_gene_ids maps unique deposited symbols to expected full Ensembl_ID.
    It is assembled only from the Stage44 one-to-one locked mapping CSV.
    """
    values={name:{} for name in selected_columns}
    if len(values)!=len(selected_columns) or not selected_columns:
        raise ValueError('Repeated or empty selected sample names')
    seen=set()
    with gzip.open(matrix,'rt',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f,delimiter='\t')
        header=next(reader)
        if (len(header)!=expected_column_count+1 or len(set(header[1:]))!=expected_column_count
            or any(c not in header[1:] for c in selected_columns)):
            raise ValueError('Unexpected matrix headers; sample order/identity changed')
        inds={name:header.index(name) for name in selected_columns}
        row_count=0
        for row_count, row in enumerate(reader,1):
            if len(row)!=len(header):
                raise ValueError(f'Malformed matrix row {row_count+1}')
            symbol=safe_symbol(row[0])
            if symbol not in approved_gene_ids:
                continue
            raw = row[0].strip().strip('"')
            if raw!=approved_gene_ids[symbol]+'_'+symbol:
                raise ValueError(f'Gene ID changed since locked Stage44: {symbol}')
            if symbol in seen:
                raise ValueError(f'Duplicate frozen-panel symbol: {symbol}')
            seen.add(symbol)
            for sample,col in inds.items():
                val=float(row[col])
                if not math.isfinite(val) or val<0:
                    raise ValueError(f'Invalid expression value for {symbol}/{sample}')
                values[sample][symbol]=val
    if row_count != expected_rows:
        raise ValueError(f'Matrix row count differs from Stage45: {row_count}')
    if seen!=set(approved_gene_ids):
        raise ValueError('Some Stage44 uniquely mapped panel genes disappeared')
    return values


def fixed_metrics(rows: list[dict]) -> list[dict]:
    res=[]
    for model in MODEL_NAMES:
        for cell in ('PC9','HCC4006','POOLED'):
            group=[r for r in rows if r['model']==model and (cell=='POOLED' or r['cell_line']==cell)]
            n_expected=18 if cell=='POOLED' else 9
            if len(group)!=n_expected:
                raise ValueError(f'Wrong number of primary samples in {model}/{cell}')
            c=Counter(r['true_state'] for r in group)
            expected=6 if cell=='POOLED' else 3
            if c!={'Control':expected,'Acute':expected,'DTC':expected}:
                raise ValueError(f'Unexpected group balance {model}/{cell}: {c}')
            right=sum(r['correct']==1 for r in group)
            dtp=sum(r['predicted_dtp']==1 for r in group if r['true_state']=='DTC')
            acute_fp=sum(r['predicted_dtp']==1 for r in group if r['true_state']=='Acute')
            control_fp=sum(r['predicted_dtp']==1 for r in group if r['true_state']=='Control')
            res.append({'model':model,'cell_line':cell,'n':n_expected,
                'correct':right,'accuracy':f'{right/n_expected:.6f}',
                'dtc_detected':dtp,'dtc_total':expected,
                'acute_false_dtc':acute_fp,'acute_total':expected,
                'control_false_dtc':control_fp,'control_total':expected,
                'interpretation':'EXPLORATORY_MISMATCHED_COUNT_SPACE_ONLY'})
    return res


def verify_prior_and_load():
    stage45=load_json(OUT/'45_summary.json')
    stage44path=OUT/'44_summary.json'
    stage44=load_json(stage44path)
    if (stage45.get('stage')!=45 or stage45.get('gse')!='GSE249721'
        or stage45.get('approved_to_score') is not False
        or stage45.get('measurement_unit_proven') is not False
        or stage45.get('samples_inverse_sum_within_5pct_of_1million')!=0
        or stage45.get('primary_cohort_n')!=18):
        raise ValueError('Stage45 unit-uncertain gate not as predeclared')
    if digest(MATRIX)!=stage45['raw_matrix_sha256'] or digest(PACKAGE)!=stage45['frozen_model_package_sha256']:
        raise ValueError('Raw matrix or frozen package has changed')
    if digest(stage44path)!=stage45['stage44_sha256']:
        raise ValueError('Stage44 gate has changed since Stage45')
    if (stage44.get('status')!='IDENTITY_TECHNICAL_GATE_PASSED_UNIT_AND_BIOLOGY_PENDING'
        or stage44.get('sample_count')!=44 or stage44.get('sample_alias_count')!=6
        or stage44.get('raw_matrix_sha256')!=stage45['raw_matrix_sha256']
        or len(stage44.get('model_coverage',[]))!=4
        or not all(x['passes_threshold'] and x['ambiguous_genes']==0 for x in stage44['model_coverage'])):
        raise ValueError('Stage44 identifier gate has changed')
    stage40=import_py(STAGE40,'stage40_read_only_probe')
    pkg=load_json(PACKAGE)
    stage40.validate_package(pkg)
    if set(pkg['models'])!=set(MODEL_NAMES):
        raise ValueError('Four frozen model policy has changed')
    source=import_py(STAGE42,'stage42_read_only_probe')
    records=source.rows()
    if len(records)!=44:
        raise ValueError('GEO sample policy no longer contains 44 samples')
    mapped=csv_rows(OUT/'44_sample_mapping.csv')
    canonical={r['gsm']:r for r in records}
    if len(mapped)!=44 or {r['gsm'] for r in mapped}!=set(canonical):
        raise ValueError('Incomplete sample identity mapping')
    for r in mapped:
        src=canonical[r['gsm']]
        if (r['geo_title']!=src['geo_title'] or r['cell_line']!=src['cell_line']
            or r['cohort']!=src['proposed_cohort']):
            raise ValueError(f'Incorrect mapped sample {r}')
    if sum(r['mapping_policy']!='EXACT' for r in mapped)!=6:
        raise ValueError('PC9 sample aliases differ from Stage44')
    primary=[r for r in mapped if r['cohort']==SAMPLE_GROUP]
    if Counter(r['cell_line'] for r in primary)!={'PC9':9,'HCC4006':9}:
        raise ValueError('Incorrect primary target cohort')
    # Symbol -> exact ENSEMBL, only for Stage44 UNIQUE genes in FROZEN PANELS.
    statuses=csv_rows(OUT/'44_selected_gene_mapping.csv')
    idx={r['symbol']:r for r in statuses}
    needed={gene for model in pkg['models'].values() for gene in model['panel_genes']}
    if not needed.issubset(idx):
        raise ValueError('Stage44 selected gene IDs missing')
    genes={}
    for symbol in needed:
        rec=idx[symbol]
        if rec['mapping_status']=='UNIQUE':
            if rec['n_ensembl_matches']!='1' or not re.fullmatch(r'ENSG\d{11}(?:\.\d+)?',rec['ensembl_ids']):
                raise ValueError(f'Malformed unique Ensembl mapping: {symbol}')
            genes[symbol]=rec['ensembl_ids']
        elif rec['mapping_status']!='MISSING' or rec['n_ensembl_matches']!='0':
            raise ValueError(f'Ambiguous/changed selected gene {symbol}')
    if len(set(genes.values()))!=len(genes):
        raise ValueError('Different selected symbols mapped to same Ensembl ID')
    for model in pkg['models'].values():
        if sum(gene in genes for gene in model['panel_genes'])<90:
            raise ValueError('Frozen panel gene coverage below 90')
    return stage40,pkg,stage45,primary,canonical,genes


def run():
    stage40,pkg,stage45,primary,canonical,genes=verify_prior_and_load()
    selected_cols=[r['matrix_header'] for r in primary]
    expression=read_selected_matrix(MATRIX,genes,selected_cols)
    predictions=[]
    for model_name in MODEL_NAMES:
        model=pkg['models'][model_name]
        for rec in primary:
            src=canonical[rec['gsm']]
            if src['observed_label'] not in ('Control','Acute','DTC'):
                raise ValueError('Nonprimary condition slipped into analysis')
            score,cov=stage40.payload_score(model,expression[rec['matrix_header']])
            binary=int(score>0)
            correct=int(binary==int(src['observed_label']=='DTC'))
            predictions.append({'model':model_name,'gsm':rec['gsm'],
                'matrix_header':rec['matrix_header'],'cell_line':rec['cell_line'],
                'drug':'erlotinib','true_state':src['observed_label'],
                'decision_margin':f'{score:.12g}','predicted_dtp':binary,
                'correct':correct,'genes_available':cov,'genes_missing':100-cov,
                'threshold':0,'measurement':'UNVERIFIED_PROBABLE_LOG2_RSEM_COUNTS',
                'study_role':'EXPLORATORY_COUNT_SPACE_SENSITIVITY_NOT_VALIDATION'})
    if len(predictions)!=72:
        raise ValueError('Four models x 18 samples expected')
    metrics=fixed_metrics(predictions)
    report=['# Stage 46 — Frozen-model count-space sensitivity (NOT VALIDATION)','',
        '**PROVISIONAL EXPLORATORY RESULT ONLY.** No claim of source-to-target predictive generalization,',
        'independent validation, model improvement, causal DTP identity, or clinical use is allowed.','',
        '## Why this is not independent validation','',
        '- GSE249721 processed expression is most consistent with log2(RSEM counts + 1), but precise deposited unit remains unconfirmed.',
        '- Stage40 frozen models were trained on a log2TPM representation; within-gene TPM/count rank ordering may differ because of gene length.',
        '- The target is erlotinib DTC rather than source osimertinib DTP; time-of-treatment differences remain a confound.',
        '- This target was metadata-inspected in earlier work; no claim of untouched fully independent validation.',
        '- Only 18 predeclared samples (PC9 and HCC4006; 6 Control, 6 Acute, 6 DTC). All other 26 samples excluded from this probe.',
        '- All four frozen models evaluated at threshold zero, with missing-panel source-only mean ranks; no fit, threshold tuning or feature selection.','',
        '## Descriptive count-space outcomes (DO NOT use to select a model)','',
        '| Frozen model | PC9 correct/9 | HCC4006 correct/9 | Pooled correct/18 | DTC detected/6 | Acute false DTC/6 | Control false DTC/6 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    by={(r['model'],r['cell_line']):r for r in metrics}
    for name in MODEL_NAMES:
        p=by[(name,'PC9')];h=by[(name,'HCC4006')];q=by[(name,'POOLED')]
        report.append(f"| `{name}` | {p['correct']}/9 | {h['correct']}/9 | {q['correct']}/18 | {q['dtc_detected']}/6 | {q['acute_false_dtc']}/6 | {q['control_false_dtc']}/6 |")
    report+=['','## Interpretation','',
        '- These margins measure behavior under a deliberately mismatched, unverified target measurement scale.',
        '- A high accuracy is NOT evidence of reliable cross-drug validation; a low accuracy does NOT isolate biological failure from unit mismatch.',
        '- Never adjust the models in response to this target result; future independently validated work must use defined compatible quantification.',
        '- Preserve original Stage44/45 **scoring_approved=False** flags; Stage46 is a separately labeled sensitivity probe, not an override.','']
    manifest={'stage':46,'gse':'GSE249721',
        'status':'EXPLORATORY_MISMATCHED_MEASUREMENT_COUNT_SPACE_PROBE_ONLY',
        'prior_scoring_approval_overridden':False,'independent_validation':False,
        'training_or_tuning_performed':False,'all_four_models':list(MODEL_NAMES),
        'cohort':'18 predeclared PC9/HCC4006 erlotinib Control/Acute/DTC',
        'n_samples':18,'n_frozen_scores':72,'threshold':0.0,
        'source_study':'GSE193258','source_expression':'log2TPM',
        'target_expression':'UNIT_UNCONFIRMED_PROBABLE_LOG2_EXPECTED_COUNTS',
        'matrix_sha256':stage45['raw_matrix_sha256'],
        'model_package_sha256':stage45['frozen_model_package_sha256'],
        'stage45_summary_sha256':digest(OUT/'45_summary.json'),
        'sample_mapping_sha256':digest(OUT/'44_sample_mapping.csv'),
        'gene_mapping_sha256':digest(OUT/'44_selected_gene_mapping.csv')}
    files={OUT/'46_count_space_probe_predictions.csv':to_csv(predictions),
        OUT/'46_count_space_probe_metrics.csv':to_csv(metrics),
        OUT/'46_count_space_probe_report.md':'\n'.join(report)+'\n',
        OUT/'46_count_space_probe_manifest.json':json.dumps(manifest,indent=2,ensure_ascii=False,sort_keys=True)+'\n'}
    write_once(files)
    print('Stage46 PROVISIONAL COUNT-SPACE PROBE: 4 models x 18 samples; no fitting or tuning')
    for model in MODEL_NAMES:
        m=by[(model,'POOLED')]
        print(f"  {model}: correct={m['correct']}/18, DTC detected={m['dtc_detected']}/6; "
              f"acute false DTC={m['acute_false_dtc']}/6")
    print('IMPORTANT: UNVERIFIED MEASUREMENT SPACE. NOT INDEPENDENT VALIDATION.')
    print('Saved '+str(OUT/'46_count_space_probe_report.md'))


def self_test():
    assert safe_symbol('"ENSG00000000003_TSPAN6"')=='TSPAN6'
    assert safe_symbol('ENSG00000000003_TSPAN6')=='TSPAN6'
    assert safe_symbol('ENSG00000000003') is None
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'mini.tsv.gz'
        with gzip.open(p,'wt',encoding='utf-8',newline='') as f:
            f.write('gene\tA\tB\n')
            for i in range(100):
                f.write('"ENSG'+str(i).zfill(11)+'_GENE'+str(i)+'"\t'+str(i+1)+'\t'+str(100-i)+'\n')
        genes={f'GENE{i}':'ENSG'+str(i).zfill(11) for i in range(92)}
        observed=read_selected_matrix(p,genes,['A','B'],expected_column_count=2,expected_rows=100)
        assert len(observed['A'])==92 and observed['A']['GENE12']==13.0
        assert observed['B']['GENE12']==88.0
        fake=[]
        for m in MODEL_NAMES:
            for cell in ('PC9','HCC4006'):
                for state in ('Control','Acute','DTC'):
                    for j in range(3):
                        truth=int(state=='DTC')
                        fake.append({'model':m,'cell_line':cell,'true_state':state,
                                     'predicted_dtp':truth,'correct':1})
        metrics=fixed_metrics(fake)
        assert len(metrics)==12 and all(int(row['correct'])==int(row['n']) for row in metrics)
        dest=Path(tmp)/'test.txt'
        write_once({dest:'frozen'})
        write_once({dest:'frozen'})
        try:
            write_once({dest:'modified'})
            raise AssertionError('Write-once guard failed')
        except FileExistsError:
            pass
    print('SELF-TEST PASSED: strict ENSG symbols, selected-row gzip reader, 18 x 4 cohort accounting, immutable artifacts')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    args=p.add_mutually_exclusive_group(required=True)
    args.add_argument('--self-test',action='store_true')
    args.add_argument('--exploratory-count-space-probe',action='store_true',
            help='NOT VALIDATION: probe all 4 frozen models on unverified count-space erlotinib target')
    opt=p.parse_args()
    if opt.self_test:
        self_test()
    else:
        run()
