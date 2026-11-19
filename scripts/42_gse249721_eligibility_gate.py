#!/usr/bin/env python3
"""GSE249721 metadata-only external eligibility gate (no expression scoring).

Place in scripts/ of osimertinib-dtp-research. From repository root:
  python scripts/42_gse249721_eligibility_gate.py --self-test
  python scripts/42_gse249721_eligibility_gate.py --audit-only
  python scripts/42_gse249721_eligibility_gate.py --inventory-local
  python scripts/42_gse249721_eligibility_gate.py --header-only path/to/your_downloaded_matrix.tsv.gz

Metadata frozen from GEO series GSE249721, inspected Nov 19 2026:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE249721
Paper: https://doi.org/10.1038/s41467-024-49360-4
The header-only option reads only ONE first row of a local matrix; no expression values.
NEVER automatically train, choose a model, tune or score with these outputs.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'external_eligibility' / 'GSE249721'
GEO = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE249721'
PAPER = 'https://doi.org/10.1038/s41467-024-49360-4'

# Exact GSM/titles listed in the official GEO Series page; 44 records.
SOURCE = [
    ('GSM7961962','H3255_Erlo_21j_1'),
    ('GSM7961963','H3255_Erlo_21j_2'),
    ('GSM7961964','H3255_Erlo_21j_3'),
    ('GSM7961965','H3255_Erlo_24h_1'),
    ('GSM7961966','H3255_Erlo_24h_2'),
    ('GSM7961967','H3255_NT1'),
    ('GSM7961968','H3255_NT2'),
    ('GSM7961969','H3255_NT3'),
    ('GSM7961970','H4006_CT_24h_1'),
    ('GSM7961971','H4006_CT_24h_2'),
    ('GSM7961972','H4006_CT_24h_3'),
    ('GSM7961973','H4006_Erlo_24h_1'),
    ('GSM7961974','H4006_Erlo_24h_2'),
    ('GSM7961975','H4006_Erlo_24h_3'),
    ('GSM7961976','H4006_Erlo_DTC_1'),
    ('GSM7961977','H4006_Erlo_DTC_2'),
    ('GSM7961978','H4006_Erlo_DTC_3'),
    ('GSM7961979','H4006_Erlo_DTEC_1'),
    ('GSM7961980','H4006_Erlo_DTEC_2'),
    ('GSM7961981','H4006_Erlo_DTEC_3'),
    ('GSM7961982','HCC827Osimertinib11d1'),
    ('GSM7961983','HCC827Osimertinib11d2'),
    ('GSM7961984','HCC827Osimertinib11d3'),
    ('GSM7961985','HCC827Untreated1'),
    ('GSM7961986','HCC827Untreated2'),
    ('GSM7961987','HCC827Untreated3'),
    ('GSM7961988','PC9_3_DTEC_A_75_jours'),
    ('GSM7961989','PC9_3_DTEC_B_75_jours'),
    ('GSM7961990','PC9_3_DTEC_D_75_jours'),
    ('GSM7961991','PC9_3_DTEC_E_75_jours'),
    ('GSM7961992','PC9_3_NT1_1'),
    ('GSM7961993','PC9_3_NT1_2'),
    ('GSM7961994','PC9_CT_24h_1'),
    ('GSM7961995','PC9_CT_24h_2'),
    ('GSM7961996','PC9_CT_24h_3'),
    ('GSM7961997','PC9_Erlo_24h_1'),
    ('GSM7961998','PC9_Erlo_24h_2'),
    ('GSM7961999','PC9_Erlo_24h_3'),
    ('GSM7962000','PC9_Erlo_DTC_1'),
    ('GSM7962001','PC9_Erlo_DTC_2'),
    ('GSM7962002','PC9_Erlo_DTC_3'),
    ('GSM7962003','PC9_Erlo_DTEC_1'),
    ('GSM7962004','PC9_Erlo_DTEC_2'),
    ('GSM7962005','PC9_Erlo_DTEC_3'),
]


def label_record(gsm: str, title: str) -> dict[str,str]:
    """Return a PREDECLARED eligibility label; never train from title patterns."""
    if title.startswith('H3255_'):
        line, drug, cohort = 'H3255', 'erlotinib', 'secondary_unmatched_control'
        if '_Erlo_21j_' in title: state, condition = 'DTC', '21 days, 1 uM erlotinib'
        elif '_Erlo_24h_' in title: state, condition = 'Acute', '24h, 1 uM erlotinib'
        elif '_NT' in title: state, condition = 'Control', 'untreated; not 24h time-matched'
        else: raise ValueError(title)
    elif title.startswith('H4006_'):
        line, drug = 'HCC4006', 'erlotinib'
        if '_CT_24h_' in title: state, condition, cohort = 'Control', '24h vehicle control', 'primary_erlotinib_three_condition'
        elif '_Erlo_24h_' in title: state, condition, cohort = 'Acute', '24h, 1 uM erlotinib', 'primary_erlotinib_three_condition'
        elif '_Erlo_DTC_' in title: state, condition, cohort = 'DTC', 'drug tolerant, paper: 21 days', 'primary_erlotinib_three_condition'
        elif '_Erlo_DTEC_' in title: state, condition, cohort = 'DTEC', 'drug-tolerant expanded (NOT DTC)', 'excluded_noncomparable'
        else: raise ValueError(title)
    elif title.startswith('HCC827'):
        line, drug, cohort = 'HCC827', 'osimertinib', 'secondary_osimertinib_two_condition'
        if 'Osimertinib11d' in title: state, condition = 'DTC', '11 days, 1 uM osimertinib'
        elif 'Untreated' in title: state, condition = 'Control', 'untreated'
        else: raise ValueError(title)
    elif title.startswith('PC9_'):
        line = 'PC9'
        if title.startswith('PC9_3_DTEC_'):
            drug, state, condition, cohort = 'erlotinib', 'DTEC', 'expanded 75-day culture (NOT DTC)', 'excluded_noncomparable'
        elif title.startswith('PC9_3_NT'):
            drug, state, condition, cohort = 'erlotinib', 'Control', 'separate untreated subclone (NOT 24h time-matched)', 'excluded_noncomparable'
        elif '_CT_24h_' in title:
            drug, state, condition, cohort = 'erlotinib', 'Control', '24h vehicle control', 'primary_erlotinib_three_condition'
        elif '_Erlo_24h_' in title:
            drug, state, condition, cohort = 'erlotinib', 'Acute', '24h, 1 uM erlotinib', 'primary_erlotinib_three_condition'
        elif '_Erlo_DTC_' in title:
            drug, state, condition, cohort = 'erlotinib', 'DTC', 'drug tolerant, paper: 21 days', 'primary_erlotinib_three_condition'
        elif '_Erlo_DTEC_' in title:
            drug, state, condition, cohort = 'erlotinib', 'DTEC', 'drug-tolerant expanded (NOT DTC)', 'excluded_noncomparable'
        else: raise ValueError(title)
    else: raise ValueError(f'Unexpected accession/title: {gsm}/{title}')
    return dict(gsm=gsm, geo_title=title, cell_line=line, arm_drug=drug, condition=condition,
                observed_label=state, proposed_cohort=cohort)


def rows():
    if len(SOURCE) != 44 or len({x for x,y in SOURCE}) != 44 or len({y for x,y in SOURCE}) != 44:
        raise AssertionError('Frozen GEO sample identities are not exactly 44 unique records')
    r=[label_record(g,t) for g,t in SOURCE]
    from collections import Counter
    assert Counter(z['proposed_cohort'] for z in r) == {
        'primary_erlotinib_three_condition':18,
        'secondary_unmatched_control':8,
        'secondary_osimertinib_two_condition':6,
        'excluded_noncomparable':12,
    }, 'Unexpected eligible cohort composition'
    assert Counter(z['observed_label'] for z in r) == {'Control':14, 'Acute':8, 'DTC':12, 'DTEC':10}
    # First primary 18 = PC9/HCC4006 x Control/Acute/DTC x triplicates.
    p=[z for z in r if z['proposed_cohort']=='primary_erlotinib_three_condition']
    assert Counter((z['cell_line'],z['observed_label']) for z in p) == {
        (line,state):3 for line in ('PC9','HCC4006') for state in ('Control','Acute','DTC')}
    return r


def immutable(path:Path, contents:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=contents.encode('utf-8'):
            raise FileExistsError(f'Existing audit differs; refusing overwrite: {path}')
    else: path.write_text(contents,encoding='utf-8',newline='\n')


def csv_text(d):
    s=io.StringIO(newline='')
    w=csv.DictWriter(s,fieldnames=list(d[0]),lineterminator='\n')
    w.writeheader();w.writerows(d);return s.getvalue()


def report():
    return f'''# GSE249721 external eligibility pre-screen (metadata only)

**Status: CONDITIONAL CANDIDATE, NOT APPROVED FOR SCORING.**
No expression values, target labels as outcomes, model margins, target performance, or feature-coverage results have been read or used to select an algorithm.

Sources: [NCBI GEO]({GEO}); [Figarol et al., Nature Communications 2024]({PAPER}).

## Frozen proposed cohorts (not a biological adjudication)

- Primary candidate: **18** samples = PC9 (3 vehicle, 3 erlotinib 24h, 3 erlotinib DTC) + HCC4006 (same groups). **Drug-transfer stress test from osimertinib-trained model to erlotinib**, not direct validation of osimertinib.
- Secondary erlotinib candidate: **8** H3255 = 3 untreated, 2 acute 24h, 3 DTC. Controls are not same 24h DMSO-labelled design and n=2 acute. Analyze separately only.
- Secondary same-drug, two-condition candidate: **6** HCC827 = 3 untreated, 3 osimertinib 11 days. **No Acute group; never report a three-group specificity score.** HCC827 is a source training cell line for frozen Stage40 package, but different study.
- Excluded from DTC score: **12** non-comparable PC9/HCC4006 samples: DTEC (10, including 75-day cultures) and extra untreated PC9 subclones (2). Do not relabel DTEC, resistant or long-term expansion as DTC.

## Pre-scoring blockers (must be resolved without using target labels for tuning)

1. Confirm **exact matrix file** `GSE249721_all_samples_processed_data.tsv.gz`, dimensions, orientation (GEO says sample x gene), GSM-to-row identity, uniqueness of sample and gene IDs, and exact expression measurement unit; original RSEM expected counts versus TPM matters for rank-in-panel features.
2. Map frozen model **100 selected genes** with one-to-one IDs, no silent duplicate aggregation, and meet >=90/100 gene-coverage policy per sample/model. No label-driven gene mapping or imputation.
3. Review culture/biological replicate independence, NT vs CT_24h comparability, and DTC state definition with original paper / individual GSM records.
4. Before reading expression values or scoring targets, lock which **Stage40 frozen models** are scored, endpoint (DTC vs observed Control + Acute), threshold 0, reporting of all candidates, and what counts as ineligible/no-score.
5. Recheck prior use of this dataset in every local/source study; this screen alone does NOT certify untouched validation. The source paper itself analyzed public GSE193259/Criscione alongside its new cohorts; do not confuse external study wet-lab samples with previously reused public data.

## Scientific interpretation boundaries

Even a successful DTC-vs-Control+Acute prediction here would be **cross-drug and cross-study stress testing**, not proof of persistence-specific causal biology or clinical Osimertinib response prediction. DTC versus Acute still carries treatment-duration confounding. Bulk RNA-seq has 44 rows, not thousands of independent single cells for a UMAP/trajectory figure.

This report is metadata evidence only; it **must not** be treated as model results.
'''


def header_only(path:Path):
    if not path.is_file(): raise FileNotFoundError(path)
    opener=gzip.open if path.suffix.lower()=='.gz' else open
    # Read exactly the first line; no expression values accessed or evaluated.
    with opener(path,'rt',encoding='utf-8-sig',errors='replace',newline='') as f:
        head=next(csv.reader([f.readline()],delimiter='\t'))
    if len(head)<2: raise ValueError('Header has fewer than two TSV columns')
    digest=hashlib.sha256()
    with open(path,'rb') as raw:
        for block in iter(lambda: raw.read(1024*1024), b''):
            digest.update(block)
    return {'path':str(path), 'file_sha256':digest.hexdigest(),
            'n_header_columns':len(head), 'first_10_header_fields':head[:10],
            'all_header_fields_are_unique':len(set(head))==len(head),
            'header_matches_any_GSM':len(set(head)&set(g for g,t in SOURCE)),
            'header_matches_any_GEO_title':len(set(head)&set(t for g,t in SOURCE)),
            'warning':'Header-only orientation/format check: expression scale and gene ID uniqueness NOT validated.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    x=p.add_mutually_exclusive_group(required=True)
    x.add_argument('--self-test', action='store_true')
    x.add_argument('--audit-only', action='store_true')
    x.add_argument('--header-only',type=Path,metavar='LOCAL_GEO_MATRIX_TSV_GZ')
    x.add_argument('--inventory-local',action='store_true',help='List already-downloaded files under data/raw/GSE249721 without reading them')
    a=p.parse_args()
    r=rows()
    if a.self_test:
        assert len(r)==44
        assert [t for g,t in SOURCE][:3]==['H3255_Erlo_21j_1','H3255_Erlo_21j_2','H3255_Erlo_21j_3']
        assert sum(z['cell_line']=='HCC827' and z['observed_label']=='Acute' for z in r)==0
        assert sum(z['proposed_cohort']=='primary_erlotinib_three_condition' for z in r)==18
        print('SELF-TEST PASSED: 44 GEO identifiers, frozen 18/8/6/12 cohorts, no expression or scoring')
        return
    if a.inventory_local:
        folder=ROOT/'data'/'raw'/'GSE249721'
        if not folder.is_dir():
            print(f'LOCAL DATA FOLDER NOT FOUND: {folder}')
            print('No download performed; check your local path')
            return
        found=sorted(p for p in folder.rglob('*') if p.is_file())
        print(f'LOCAL INVENTORY: {len(found)} file(s) under {folder}; no file contents read')
        for f in found:
            print(f'  {f.relative_to(ROOT)}  ({f.stat().st_size:,} bytes)')
        if not found:
            print('Folder exists but contains no files; screenshot confirms directory only, not matrix')
        return
    if a.header_only:
        print(json.dumps(header_only(a.header_only),indent=2,ensure_ascii=False));return
    manifest=csv_text(r)
    immutable(OUT/'sample_manifest.csv',manifest)
    immutable(OUT/'eligibility_report.md',report())
    summary=dict(gse='GSE249721',status='CONDITIONAL_METADATA_ONLY_NOT_APPROVED_TO_SCORE',
      total_samples=44,primary_erlotinib_three_condition=18,
      secondary_erlotinib_unmatched_control=8, secondary_osimertinib_two_condition=6,
      excluded_noncomparable=12,expression_read=False,model_scored=False,
      manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),geo=GEO,paper=PAPER)
    immutable(OUT/'audit_summary.json',json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print('GSE249721 METADATA GATE COMPLETE: 44 samples; primary=18 secondary_erlotinib=8 secondary_osimertinib=6 excluded=12')
    print('NO EXPRESSION READ; NO MODEL RUN; NOT APPROVED FOR SCORING')
    print(f'Saved {OUT / "eligibility_report.md"}')

if __name__=='__main__':main()
