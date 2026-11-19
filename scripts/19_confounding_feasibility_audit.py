"""Stage 19A | Metadata-only confounding identifiability audit.

Run from the root of the existing osimertinib-dtp-research repository:
    python scripts/19_confounding_feasibility_audit.py --self-test
    python scripts/19_confounding_feasibility_audit.py

Reads local frozen protocol / GEO metadata / existing stage-18 audit.
Does NOT read expression matrices, train/tune models, download data,
or approve clinical/biological claims. It does not change earlier outputs.

Outputs to outputs/confounding_feasibility/:
    19_sample_design.csv
    19_contrast_feasibility.csv
    19_design_audit.json
    19_design_audit.md

Python standard library only (3.10+).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'confounding_feasibility'
META = ROOT / 'data' / 'metadata'
SOURCE = 'GSE193258'
TARGET = 'GSE255958'
THREE_STATES = ('Control', 'Acute', 'DTP')
SOURCE_LINES = ('H1975', 'HCC2935', 'HCC827', 'PC9')
TARGET_GROUPS = {
    'Control': (3374, 3375, 3376),
    'Acute': (3377, 3378, 3379),
    'DTP': (3380, 3381, 3382),
    'Resistant': (3383, 3384, 3385),
}
# These labels indicate historical study exposure; never assert an untouched test.
PRIOR_STUDIES = {'GSE193258', 'GSE255958', 'GSE153183', 'GSE249721',
                 'GSE303413', 'GSE341996', 'GSE285632', 'GSE243564'}
DURATION = re.compile(r'(?<!\d)(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|days?|d|weeks?|w)\b', re.I)
DOSE = re.compile(r'(?<![A-Za-z\d])(\d+(?:\.\d+)?)\s*(nM|uM|µM|μM)\b', re.I)


def read_csv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'Empty CSV header: {path}')
        return list(reader)


def read_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def hours_from_text(*fields: str) -> float | None:
    """Parse only an explicit duration; never infer drug exposure from class."""
    for text in fields:
        m = DURATION.search(text or '')
        if m:
            x = float(m.group(1))
            u = m.group(2).lower()
            return x * (1 if u.startswith('h') else 24 if u.startswith('d') else 168)
    return None


def dose_from_text(*fields: str) -> str:
    for text in fields:
        m = DOSE.search(text or '')
        if m:
            return f'{m.group(1)} {m.group(2)}'
    return ''


def build_sample_design(protocol: dict, geo: list[dict], cohort: list[dict]) -> list[dict]:
    """Exact source labels, review-candidate external contrasts, and frozen stage17 grouping."""
    by_geo = {(r.get('gse'), r.get('gsm')): r for r in geo}
    if len(by_geo) != len(geo):
        raise ValueError('Duplicate GEO study/GSM rows')
    source = protocol.get('source_gsm_to_label', [])
    if protocol.get('source_study') != SOURCE or len(source) != 24:
        raise ValueError('Expected existing 24-sample frozen source protocol')
    if Counter((r['cell_line'], r['state']) for r in source) != Counter({
        (line, state): 3 for line in SOURCE_LINES for state in ('Control', 'DTP')
    }):
        raise ValueError('Frozen source balance changed')

    result = []
    def append(gse: str, gsm: str, cell: str, state: str, treat: str, duration: str,
               provenance: str, review: str, title: str = '', extra_flag: str = '') -> None:
        # For GSE193258 only, the original study explicitly describes the 24-hour
        # DMSO control and 24-hour acute treatment versus 21-day DTP. These are
        # nominal study-design times, not newly inferred per-sample measurements.
        source_hours = {'Control': 24., 'Acute': 24., 'DTP': 21 * 24.}
        h = source_hours[state] if gse == SOURCE and state in source_hours else hours_from_text(duration, treat, title)
        exposure = ('osimertinib' if 'osimertinib' in (treat + ' ' + title).lower() or
                    'osi_' in (treat + ' ' + title).lower() else
                    'vehicle/control' if state == 'Control' else 'not_confirmed')
        result.append({
            'gse': gse, 'gsm': gsm, 'cell_line': cell,
            'candidate_state': state, 'treatment_reported': treat,
            'duration_reported': duration, 'duration_hours': '' if h is None else f'{h:g}',
            'dose_reported': dose_from_text(treat, duration),
            'exposure_flag': exposure, 'label_provenance': provenance,
            'study_role': review, 'title': title, 'review_flags': extra_flag,
        })

    source_ids = set()
    for r in source:
        g = by_geo.get((SOURCE, r['gsm']))
        if not g:
            raise ValueError('Source GSM missing from GEO metadata: ' + r['gsm'])
        state = r['state']; treat = g.get('reported_treatment', '')
        want = 'DMSO' if state == 'Control' else 'osimertinib DTPs 3w.'
        if g.get('reported_cell_line') != r['cell_line'] or treat != want:
            raise ValueError('Source GSM treatment/cell-line drift: ' + r['gsm'])
        source_ids.add(r['gsm'])
        append(SOURCE, r['gsm'], r['cell_line'], state, treat,
               g.get('reported_duration', ''), 'FROZEN_PROTOCOL_PLUS_GEO',
               'DEVELOPMENT_ONLY', g.get('title', ''))
    acute = [r for r in geo if r.get('gse') == SOURCE and
             r.get('reported_treatment') == 'osimertinib acute 24 hrs.']
    if Counter(r.get('reported_cell_line') for r in acute) != Counter({x:3 for x in SOURCE_LINES}):
        raise ValueError('GSE193258: expected 3 Acute samples per cell line')
    for g in acute:
        if g['gsm'] in source_ids:
            raise ValueError('Acute GSM overlaps source Control/DTP')
        append(SOURCE, g['gsm'], g['reported_cell_line'], 'Acute',
               g.get('reported_treatment',''),g.get('reported_duration',''),
               'EXACT_GEO_ACUTE_TREATMENT', 'DEVELOPMENT_ONLY',g.get('title',''))
    if len({r['gsm'] for r in result}) != 36:
        raise ValueError('Source has incorrect unique 36 GSM identities')

    # Existing candidate labels are recorded for *design exploration only*.
    # These are not independent clinical ground truth and cannot be used to tune.
    for r in cohort:
        if (r.get('gse') == SOURCE or r.get('gse') == TARGET or
            r.get('candidate_analysis') != 'control_dtp_candidate' or
            r.get('candidate_state') not in ('Control','DTP')):
            continue
        gse, gsm = r['gse'], r['gsm']
        g = by_geo.get((gse,gsm), {})
        if g and g.get('reported_cell_line') and g['reported_cell_line'] != r.get('cell_line_reported'):
            raise ValueError('Cohort/GEO cell-line mismatch: ' + gsm)
        append(gse, gsm, r.get('cell_line_reported',''), r['candidate_state'],
               g.get('reported_treatment',r.get('treatment_reported','')),
               g.get('reported_duration',''), 'PROVISIONAL_COHORT_REVIEW',
               'PREVIOUSLY_INSPECTED_NO_INDEPENDENCE', g.get('title',r.get('geo_title','')),
               r.get('review_flags',''))

    # Stage17 mapping is explicit in scripts/17_three_class_external_transfer.py.
    # This is *not* a new label adjudication. Resistant stays separate.
    for state, idxs in TARGET_GROUPS.items():
        for i in idxs:
            gsm = f'GSM808{i}'
            g = by_geo.get((TARGET,gsm), {})
            append(TARGET,gsm,g.get('reported_cell_line','PC9'),state,
                   g.get('reported_treatment',''),g.get('reported_duration',''),
                   'FROZEN_STAGE17_GROUP_MAP','PREVIOUSLY_INSPECTED_EXPLORATORY',
                   g.get('title',''), 'NOT_A_NEW_VALIDATION_TARGET')
    ids = [(r['gse'],r['gsm']) for r in result]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate study/sample in combined design')
    return sorted(result, key=lambda r: (r['gse'],r['cell_line'],r['candidate_state'],r['gsm']))


def contrast_rows(samples: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in samples:
        grouped[(row['gse'],row['cell_line'])].append(row)
    rows = []
    for (gse,line), entries in sorted(grouped.items()):
        for left,right in (('Control','Acute'),('Control','DTP'),('Acute','DTP'),
                           ('DTP','Resistant')):
            a = [x for x in entries if x['candidate_state']==left]
            b = [x for x in entries if x['candidate_state']==right]
            if not a or not b:
                continue
            ah = {float(x['duration_hours']) for x in a if x['duration_hours']}
            bh = {float(x['duration_hours']) for x in b if x['duration_hours']}
            complete = len(ah)>0 and len(bh)>0 and all(x['duration_hours'] for x in a+b)
            if not complete:
                verdict = 'DURATION_METADATA_INCOMPLETE'
            elif ah.isdisjoint(bh):
                verdict = 'NO_MATCHED_DURATION'
            else:
                verdict = 'NOMINAL_DURATION_OVERLAP_CHECK_OTHER_CONFOUNDERS'
            rows.append({
                'gse':gse,'cell_line':line,'contrast':f'{left}_vs_{right}',
                'n_left':len(a),'n_right':len(b),
                'left_hours':';'.join(f'{v:g}' for v in sorted(ah)),
                'right_hours':';'.join(f'{v:g}' for v in sorted(bh)),
                'duration_identifiability':verdict,
                'study_role':entries[0]['study_role'],
                'interpretation':'Metadata-only; matching exposure times does not establish causal identification',
            })
    return rows


def assess(root: Path) -> dict:
    m = root / 'data' / 'metadata'
    audit_path = root / 'outputs' / 'protocol_audit' / '18_audit.json'
    audit = read_json(audit_path)
    if audit.get('stage') != 18 or any(f.get('status')=='FAIL' for f in audit.get('findings', [])):
        raise ValueError('Stage 18 audit missing, invalid, or has FAIL; stop before design audit')
    protocol = read_json(m/'evaluation_protocol.json')
    geo = read_csv(m/'geo_samples.csv')
    cohort = read_csv(m/'cohort_review.csv')
    samples = build_sample_design(protocol,geo,cohort)
    contrasts = contrast_rows(samples)
    if len([r for r in samples if r['gse']==SOURCE])!=36:
        raise ValueError('Source three-class count changed')
    if len([r for r in samples if r['gse']==TARGET])!=12:
        raise ValueError('Stage17 target group count changed')
    statuses = Counter(r['duration_identifiability'] for r in contrasts)
    return {'stage':19,'analysis':'metadata_only_confounding_feasibility',
            'generated_utc':datetime.now(timezone.utc).isoformat(),
            'stage18_commit':audit.get('repository_commit'),
            'scope':'Study-design and label association, not expression or causal inference',
            'source_three_class_n':36,'stage17_target_n':12,
            'sources_seen_before':sorted(PRIOR_STUDIES),
            'sample_count_by_study':dict(sorted(Counter(r['gse'] for r in samples).items())),
            'contrast_status_counts':dict(statuses),
            'source_acute_dtp_duration_separable':False,
            'warnings':['Candidate labels are not biology-approved.',
                        'Study-level confounding may persist with matched duration.',
                        'Exposure duration and class are confounded in source Acute-vs-DTP.',
                        'GSE255958 is historically inspected.',
                        'Rows with no explicit duration are unresolved, never set to zero.',
                        'Do not treat samples or cells within one study as independent studies.',
                        'No gene-expression data were read and no model was fitted.'],
            'samples':samples,'contrasts':contrasts}


def write_outputs(report: dict, out: Path) -> None:
    out.mkdir(parents=True,exist_ok=True)
    def write_csv(name, rows):
        if not rows:
            raise ValueError('No CSV rows: '+name)
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    write_csv('19_sample_design.csv',report['samples'])
    write_csv('19_contrast_feasibility.csv',report['contrasts'])
    (out/'19_design_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    notes = [
        '# Stage 19A — Metadata-only confounding feasibility',
        '',f"Stage 18 provenance commit: `{report['stage18_commit']}`",
        '', '**This is NOT a biological validation, statistical model, or causal effect estimate.**','',
        '## Sample counts by study', '', '| Study | Samples included |','|---|---:|',
    ]
    notes += [f'| {s} | {v} |' for s,v in report['sample_count_by_study'].items()]
    notes += ['', '## Nominal exposure-time identifiability', '',
              '| Study | Cell line | Contrast | Left (hours) | Right (hours) | Verdict |',
              '|---|---|---|---|---|---|']
    notes += [f"| {r['gse']} | {r['cell_line']} | {r['contrast']} | {r['left_hours'] or 'unknown'} | {r['right_hours'] or 'unknown'} | {r['duration_identifiability']} |" for r in report['contrasts']]
    notes += ['', '## Interpretation', '',
        '- GSE193258 Acute (24h) vs DTP (21d) does not identify duration-independent DTP effects.',
        '- A same-duration pair may still differ by drug exposure, induction, clone, batch, processing, or biology.',
        '- GSE255958 was already inspected; this cannot be an untouched confirmation.',
        '- GSE303413 treatment-dose conflicts must be adjudicated from original sample records.',
        '- Candidate labels require independent source-supported biological review.',
        '- Do not use any new study for model selection before locking a prospective evaluation recipe.',
        '', 'See `19_sample_design.csv` and `19_contrast_feasibility.csv` for sample-level audit data.', '']
    (out/'19_design_audit.md').write_text('\n'.join(notes),encoding='utf-8')


def self_test() -> None:
    assert hours_from_text('9d')==216
    assert hours_from_text('24 hrs.')==24
    assert hours_from_text('3w.')==504
    assert hours_from_text('unknown') is None
    assert dose_from_text('osimertinib 500nM 24h')=='500 nM'
    source=[];geo=[]
    for line in SOURCE_LINES:
        for state in ('Control','DTP'):
            for k in range(3):
                gsm=f'{line}_{state}_{k}'
                source.append({'gsm':gsm,'cell_line':line,'state':state})
                geo.append({'gse':SOURCE,'gsm':gsm,'reported_cell_line':line,
                    'reported_treatment':'DMSO' if state=='Control' else 'osimertinib DTPs 3w.',
                    'reported_duration':'','title':''})
        for k in range(3):
            geo.append({'gse':SOURCE,'gsm':f'{line}_Acute_{k}',
                'reported_cell_line':line,'reported_treatment':'osimertinib acute 24 hrs.',
                'reported_duration':'','title':''})
    table=build_sample_design({'source_study':SOURCE,'source_gsm_to_label':source},geo,[])
    assert len([s for s in table if s['gse']==SOURCE])==36
    src=[c for c in contrast_rows(table) if c['gse']==SOURCE]
    assert len(src)==12
    assert all(c['duration_identifiability']=='NO_MATCHED_DURATION' for c in src if c['contrast'].endswith('_vs_DTP'))
    assert all(c['duration_identifiability']=='NOMINAL_DURATION_OVERLAP_CHECK_OTHER_CONFOUNDERS' for c in src if c['contrast']=='Control_vs_Acute')
    assert len([x for x in table if x['candidate_state']=='Resistant'])==3
    print('SELF-TEST PASSED: exact source class mapping, three-class counts, duration parsing, non-overlap, resistant exclusion')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test();return
    report=assess(ROOT)
    write_outputs(report,OUT)
    print('Stage19A metadata-only confounding feasibility — report written:')
    print(OUT/'19_design_audit.md')
    print('Studies:',report['sample_count_by_study'])
    print('Duration contrast results:',report['contrast_status_counts'])
    print('NO expression matrices read; NO model fit; NO independent validation claim.')

if __name__=='__main__':
    main()
