#!/usr/bin/env python3
"""Stage 45: GSE249721 UNLABELLED measurement fingerprint and cohort protocol audit.

Run from repository root:
  python scripts/45_gse249721_measurement_biology_audit.py --self-test
  python scripts/45_gse249721_measurement_biology_audit.py --audit-only

Uses ONLY local raw matrix, Stage44 ID reconciliation, and source model hash.
Does not fit, predict, select methods, or compare expression by condition.
Never concludes biological identity or measurement unit solely from statistics.

Fingerprint test: if deposited values are log2(TPM + 1) and encompass the
complete expressed-gene universe, sum(2**value - 1) should be ~1,000,000 TPM
per sample. Similarly processed log2(expected_count + 1) could show much higher
and variable sums. This is an empirical diagnostic, not a proof of provenance.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/raw/GSE249721/GSE249721_all_samples_processed_data.tsv.gz'
SOFT = ROOT/'data/raw/GSE249721/GSE249721_family.soft.gz'
PACKAGE = ROOT/'outputs/research_model_package/40_frozen_research_models.json'
OUT = ROOT/'outputs/external_eligibility/GSE249721'
PRIOR = OUT/'44_summary.json'
MAPPING = OUT/'44_sample_mapping.csv'
EXPECTED_COHORTS = {
    'primary_erlotinib_three_condition':18,
    'secondary_unmatched_control':8,
    'secondary_osimertinib_two_condition':6,
    'excluded_noncomparable':12,
}


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def read_mapping(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rr=list(csv.DictReader(f))
    keys={'matrix_header','geo_title','gsm','cell_line','mapping_policy','cohort'}
    if len(rr)!=44 or any(set(r)!=keys for r in rr):
        raise ValueError('Stage44 sample map invalid')
    for k in ('matrix_header','geo_title','gsm'):
        if len({r[k] for r in rr})!=44 or any(not r[k] for r in rr):
            raise ValueError(f'Stage44 sample map not bijective for {k}')
    ct=Counter(r['cohort'] for r in rr)
    if dict(ct)!=EXPECTED_COHORTS:
        raise ValueError(f'Unexpected cohort split: {ct}')
    if sum(r['mapping_policy']!='EXACT' for r in rr)!=6:
        raise ValueError('Six PC9 explicit aliases expected')
    primary=[r for r in rr if r['cohort']=='primary_erlotinib_three_condition']
    if Counter(r['cell_line'] for r in primary)!=Counter({'PC9':9,'HCC4006':9}):
        raise ValueError('Primary cohort must contain PC9 and HCC4006 x 9')
    for cell in ('PC9','HCC4006'):
        subs=[r for r in primary if r['cell_line']==cell]
        c=Counter('control' if 'CT_24h' in r['geo_title'] else
                  'acute' if 'Erlo_24h' in r['geo_title'] else
                  'dtc' if 'Erlo_DTC' in r['geo_title'] else 'unknown'
                  for r in subs)
        if c!=Counter({'control':3,'acute':3,'dtc':3}):
            raise ValueError(f'Unexpected 3-condition primary subgroup {cell}: {c}')
    return rr


def fingerprint(path,mapping):
    """Parse expression values without using labels. Return per-sample scale-only summaries."""
    headers_expected={r['matrix_header'] for r in mapping}
    with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f,delimiter='\t')
        h=next(reader)
        if len(h)!=45 or set(h[1:])!=headers_expected or len(set(h[1:]))!=44:
            raise ValueError('Matrix sample headers changed from Stage44')
        names=h[1:]
        sums=[0.0]*44
        zeros=[0]*44
        quant=[0]*44
        positive=[0]*44
        maximum=[0.0]*44
        rows=0
        seen_ids=set()
        min_pos=float('inf')
        for lineno,rec in enumerate(reader,2):
            if len(rec)!=45:
                raise ValueError(f'Invalid matrix width line {lineno}')
            gid=rec[0]
            if gid in seen_ids:
                raise ValueError(f'Duplicate gene row at line {lineno}')
            seen_ids.add(gid)
            rows+=1
            for j,raw in enumerate(rec[1:]):
                try: x=float(raw)
                except (ValueError,OverflowError):
                    raise ValueError(f'Invalid numeric value line {lineno} sample {j}')
                if not math.isfinite(x) or x<0 or x>50:
                    raise ValueError(f'Nonfinite, negative or implausibly high input at line {lineno}')
                if x==0.0:
                    zeros[j]+=1
                    continue
                positive[j]+=1
                maximum[j]=max(maximum[j],x)
                min_pos=min(min_pos,x)
                y=math.expm1(math.log(2.0)*x)  # 2**x -1, stable for low x
                if not math.isfinite(y):
                    raise ValueError(f'Overflow in inverse transform at line {lineno}')
                sums[j]+=y
                # RSEM expression values are often reported to 0.01; tests
                # numeric consistency of that hypothesis, not data provenance.
                if abs(y*100.0-round(y*100.0))<=1e-4:
                    quant[j]+=1
        if rows!=34194:
            raise ValueError(f'Expected 34194 gene rows, got {rows}')
    by_header={r['matrix_header']:r for r in mapping}
    records=[]
    for j,name in enumerate(names):
        records.append({
            'gsm':by_header[name]['gsm'],
            'matrix_header':name,
            'n_gene_rows':rows,
            'zero_fraction':round(zeros[j]/rows,6),
            'inverse_log2p1_sum':round(sums[j],4),
            'inverse_to_one_million_ratio':round(sums[j]/1_000_000,6),
            'fraction_nonzero_inverse_on_0p01_grid':round(quant[j]/positive[j],6) if positive[j] else None,
            'max_deposited_value':round(maximum[j],6),
        })
    return records,min_pos


def summary_for(records,min_positive,prior):
    totals=[r['inverse_log2p1_sum'] for r in records]
    q=[r['fraction_nonzero_inverse_on_0p01_grid'] or 0 for r in records]
    n_close=sum(abs(v-1_000_000)/1_000_000<=0.05 for v in totals)
    qmed=statistics.median(q)
    total_median=statistics.median(totals)
    # A strong pattern supports but NEVER establishes a deposited unit.
    signature=('CONSISTENT_WITH_LOG2_TPM_PLUS_ONE' if n_close>=40 and qmed>=0.95
               else 'NOT_CONCLUSIVE_FOR_LOG2_TPM_PLUS_ONE')
    return {
        'stage':45,'gse':'GSE249721',
        'stage44_sha256':prior.get('_stage44_sha256','SELF_TEST_ONLY'),
        'raw_matrix_sha256':prior['raw_matrix_sha256'],
        'frozen_model_package_sha256':prior['frozen_model_package_sha256'],
        'sample_count':len(records),'gene_rows':34194,
        'primary_cohort_n':18,'secondary_unmatched_control_n':8,
        'secondary_osimertinib_two_condition_n':6,'excluded_n':12,
        'diagnostic_hypothesis':'deposited x = log2(original_RSEM_measure + 1)',
        'minimum_positive_value':min_positive,
        'minimum_positive_inverse_log2p1':math.expm1(math.log(2.0)*min_positive),
        'median_inverse_log2p1_sample_sum':total_median,
        'min_inverse_log2p1_sample_sum':min(totals),
        'max_inverse_log2p1_sample_sum':max(totals),
        'samples_inverse_sum_within_5pct_of_1million':n_close,
        'median_fraction_on_0p01_grid':qmed,
        'fingerprint':signature,
        'measurement_unit_proven':False,
        'biological_independence_proven':False,
        'historical_status':'Previously metadata inspected; not untouched external validation',
        'candidate_test':'18 sample PC9/HCC4006 erlotinib Control/24h/DTC cross-drug exploratory stress test only',
        'approved_to_score':False,
        'next_decision':'Corroborate deposited expression measure with authors/GEO methods or original RSEM results. Predeclare cross-drug exploratory interpretation and four frozen-model scoring; do not use target outcomes for model selection.',
        'provenance_note':'Numeric fingerprints cannot uniquely distinguish other transformations or independently verify biological replicates.',
    }


def render_report(s):
    return '\n'.join([
        '# Stage 45 — GSE249721 measurement and biology gate','',
        '**No model scoring performed; approval remains false.**','',
        '## Frozen technical eligibility','',
        '- Stage44: 44/44 samples mapped; 92/100 acute-only and 93/100 dual-contrast panel genes unique.',
        '- No ambiguous genes in the four frozen panels. Seven other matrix symbols are duplicated.','',
        '## Empirical measurement fingerprint (unlabelled)','',
        '- Candidate transformation tested: `x = log2(RSEM_measure + 1)`.',
        f"- Minimum deposited positive value: `{s['minimum_positive_value']:.12g}`; inverse transform: `{s['minimum_positive_inverse_log2p1']:.12g}`.",
        f"- Median per-sample sum of `2**x - 1`: `{s['median_inverse_log2p1_sample_sum']:.1f}`.",
        f"- Range across 44 samples: `{s['min_inverse_log2p1_sample_sum']:.1f}` – `{s['max_inverse_log2p1_sample_sum']:.1f}`.",
        f"- Samples close to 1 million after inversion (±5%): `{s['samples_inverse_sum_within_5pct_of_1million']}/44`.",
        f"- Median fraction of nonzero inverted values on a 0.01 grid: `{s['median_fraction_on_0p01_grid']:.5f}`.",
        f"- Numerical fingerprint: `{s['fingerprint']}`.",
        '- IMPORTANT: consistency with a formula is NOT a proof of TPM versus expected counts; exact deposited unit remains unconfirmed.','',
        '## Fixed biological cohort interpretation','',
        '- Primary candidate: PC9 and HCC4006, each with three Control, three erlotinib 24h, three erlotinib DTC = **18 samples**.',
        '- Secondary: H3255 8 samples (unmatched controls, only two acute replicates); HCC827 6 samples (osimertinib, no acute).',
        '- Exclude 12 RPC/DTEC or otherwise noncomparable samples from the primary endpoint.',
        '- Primary assay is **cross-drug (osimertinib -> erlotinib)** and **cross-study**, not an independent osimertinib clinical validation.',
        '- The source Control/Acute/DTP endpoint is not identical to the target Control/24h/DTC endpoint; duration is confounded.',
        '- Previously metadata-inspected target cannot be called untouched/fully independent validation.','',
        '## Decision','',
        '**Scoring approved: NO.** Confirm deposited expression unit/transform and biologic eligibility / replicate provenance before any frozen scoring.',
        'Do not change any of the four frozen models, chosen genes, missing-gene policy, or threshold using target data.',
        '',
    ])


def write_immutable(targets):
    for p,content in targets.items():
        b=content.encode('utf-8')
        if p.exists() and p.read_bytes()!=b:
            raise FileExistsError(f'Existing different output will not be overwritten: {p}')
    OUT.mkdir(parents=True,exist_ok=True)
    for p,content in targets.items():
        if not p.exists():p.write_text(content,encoding='utf-8',newline='\n')


def csv_payload(records):
    dst=io.StringIO(newline='')
    w=csv.DictWriter(dst,fieldnames=list(records[0]),lineterminator='\n')
    w.writeheader();w.writerows(records)
    return dst.getvalue()


def audit():
    for p in [DATA,SOFT,PACKAGE,PRIOR,MAPPING]:
        if not p.is_file():raise FileNotFoundError(f'Missing prior stage/local input: {p}')
    p=json.loads(PRIOR.read_text(encoding='utf-8'))
    if (p.get('status')!='IDENTITY_TECHNICAL_GATE_PASSED_UNIT_AND_BIOLOGY_PENDING'
        or p.get('approved_to_score') is not False or p.get('sample_count')!=44
        or p.get('n_gene_rows')!=34194 or
        any(not x.get('passes_threshold') for x in p.get('model_coverage',[])) or
        len(p.get('model_coverage',[]))!=4):
        raise ValueError('Stage44 not passed or invalid; do not continue')
    if (checksum(DATA)!=p['raw_matrix_sha256'] or
        checksum(SOFT)!=p['soft_sha256'] or
        checksum(PACKAGE)!=p['frozen_model_package_sha256']):
        raise ValueError('Source inputs differ from audited frozen Stage44 hashes')
    mappings=read_mapping(MAPPING)
    rec,min_pos=fingerprint(DATA,mappings)
    s=summary_for(rec,min_pos,{**p,'_stage44_sha256':checksum(PRIOR)})
    targets={OUT/'45_summary.json':json.dumps(s,indent=2,ensure_ascii=False)+'\n',
             OUT/'45_measurement_report.md':render_report(s),
             OUT/'45_sample_fingerprints.csv':csv_payload(rec)}
    write_immutable(targets)
    print('Stage45 measurement audit: 44 samples; source files unchanged')
    print('  minimum positive x =',format(min_pos,'.12g'),'; 2**x - 1 =',format(s['minimum_positive_inverse_log2p1'],'.12g'))
    print('  median inverse sum per sample =',format(s['median_inverse_log2p1_sample_sum'],'.3f'))
    print('  samples inverse total near 1e6 (+/-5%) =',s['samples_inverse_sum_within_5pct_of_1million'],'/44')
    print('  median inverted cent-grid agreement =',format(s['median_fraction_on_0p01_grid'],'.5f'))
    print('  signature:',s['fingerprint'])
    print('  GATE: MEASUREMENT_UNIT_NOT_PROVEN; scoring NOT approved')
    print('Saved',OUT/'45_measurement_report.md')


def self_test():
    for original in (0,0.01,0.13,1,7.18,997.38):
        x=math.log2(original+1)
        restored=math.expm1(math.log(2)*x)
        assert math.isclose(restored,original,rel_tol=1e-10,abs_tol=1e-10)
    assert math.isclose(math.expm1(math.log(2)*0.0143552929770701),0.01,abs_tol=1e-12)
    fake=[{'inverse_log2p1_sum':1e6,'fraction_nonzero_inverse_on_0p01_grid':1.0}]*44
    s=summary_for(fake,math.log2(1.01),{'raw_matrix_sha256':'test','frozen_model_package_sha256':'test'})
    assert s['fingerprint']=='CONSISTENT_WITH_LOG2_TPM_PLUS_ONE'
    assert s['approved_to_score'] is False and s['measurement_unit_proven'] is False
    fake=[{'inverse_log2p1_sum':4e7,'fraction_nonzero_inverse_on_0p01_grid':1.0}]*44
    assert summary_for(fake,math.log2(1.01),{'raw_matrix_sha256':'test','frozen_model_package_sha256':'test'})['fingerprint']=='NOT_CONCLUSIVE_FOR_LOG2_TPM_PLUS_ONE'
    print('SELF-TEST PASSED: inverse log2p1 fingerprint, TPM numerical hypothesis and fail-closed research gate')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-test',action='store_true')
    g.add_argument('--audit-only',action='store_true')
    args=ap.parse_args()
    self_test() if args.self_test else audit()
