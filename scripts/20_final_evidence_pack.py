"""Stage 20 | Freeze an auditable, exploratory evidence pack for the manuscript.

Use from the ROOT of osimertinib-dtp-research (Python >= 3.10):

  python scripts/20_final_evidence_pack.py --self-test
  python scripts/20_final_evidence_pack.py

Reads only existing outputs of stages 15,16,17,18,19. Does not read expression
matrices, re-fit models, choose a winner, use the target for tuning, or change
any prior results. It FAILS CLOSED on missing, inconsistent or unexpected rows.

Outputs: outputs/final_evidence/
  20_paper_ready_tables.md
  20_three_class_summary.csv
  20_three_class_confusion_counts.csv
  20_source_loco_by_cell_line.csv
  20_pc9_sensitivity.csv
  20_binary_historical_benchmarks.csv
  20_confounding_contrasts.csv
  20_unresolved_warnings.csv
  20_evidence_manifest.json

This pack is descriptive and NON-CONFIRMATORY. Neither test-set replicates nor
folds establish statistical independence or a causal DTP-specific signature.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'final_evidence'
STATES = ('Control', 'Acute', 'DTP')
LINES = ('H1975', 'HCC2935', 'HCC827', 'PC9')
METHODS = (
    'rank_full_lr100_3class',
    'rank_selected_lr100_3class',
    'signed_rank_ovr_full_3class',
    'raw_log2tpm_lr500_3class',
)
TARGET = {
    'Control': tuple(f'GSM808{j}' for j in (3374,3375,3376)),
    'Acute': tuple(f'GSM808{j}' for j in (3377,3378,3379)),
    'DTP': tuple(f'GSM808{j}' for j in (3380,3381,3382)),
}
FILES = {
    'stage15': 'outputs/rank_methods/15_summary.json',
    'stage16_pred': 'outputs/acute_challenge/16_three_class_predictions.csv',
    'stage16_report': 'outputs/acute_challenge/16_three_class_summary.json',
    'stage17_pred': 'outputs/three_class_external/17_predictions.csv',
    'stage17_report': 'outputs/three_class_external/17_summary.json',
    'stage17_excl_pred': 'outputs/three_class_external/17_pc9_excluded_predictions.csv',
    'stage17_excl_report': 'outputs/three_class_external/17_pc9_excluded_summary.json',
    'stage18': 'outputs/protocol_audit/18_audit.json',
    'stage19': 'outputs/confounding_feasibility/19_design_audit.json',
}


def require(test: bool, msg: str):
    if not test:
        raise ValueError('INTEGRITY STOP: ' + msg)


def load_json(path: Path) -> dict:
    with path.open(encoding='utf-8-sig') as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f'Expected JSON object: {path}')
    return obj


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(bool(reader.fieldnames), f'Empty CSV: {path}')
        data = list(reader)
    require(bool(data), f'No rows: {path}')
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compute_stats(rows: list[dict]) -> dict:
    require(bool(rows), 'Cannot compute metrics on empty rows')
    cm = [[0]*3 for _ in STATES]
    for x in rows:
        require(x['true_state'] in STATES and x['predicted_state'] in STATES,
                'Invalid three-class state')
        cm[STATES.index(x['true_state'])][STATES.index(x['predicted_state'])] += 1
    n = sum(map(sum, cm))
    recall, f1 = [], []
    for i in range(3):
        support = sum(cm[i]); predicted = sum(cm[r][i] for r in range(3))
        require(support > 0, f'Class absent from scored cohort: {STATES[i]}')
        r = cm[i][i]/support
        p = cm[i][i]/predicted if predicted else 0.
        recall.append(r)
        f1.append(2*p*r/(p+r) if p+r else 0.)
    return {'n':n, 'correct':sum(cm[i][i] for i in range(3)),
            'balanced_accuracy':sum(recall)/3, 'macro_f1':sum(f1)/3,
            **{f'recall_{s.lower()}':recall[i] for i,s in enumerate(STATES)},
            'confusion_matrix':cm}


def approx_equal(a, b, what, atol=1e-8):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            abs(float(a)-float(b))<=atol, what)


def indexed(rows: list[dict], keys: tuple[str, ...], name: str) -> dict:
    out = {}
    for r in rows:
        k = tuple(r[field] for field in keys)
        require(k not in out, f'{name} has duplicate {k}')
        out[k] = r
    return out


def check_source(rows: list[dict], report: dict) -> tuple[list[dict],dict]:
    require('EXPLORATORY' in report.get('status',''), 'Stage16 report is not exploratory')
    require(report.get('trained_states')==list(STATES), 'Stage16 class definition changed')
    require(report.get('independent_validation') is False, 'Stage16 independence flag changed')
    require(len(rows)==len(METHODS)*60, 'Stage16 must have 60 rows per method (36 labeled + 24 washout)')
    bykey = indexed(rows,('method','gsm'),'Stage16')
    require({r['method'] for r in rows}==set(METHODS),'Stage16 method set changed')
    summary=[]; byline=[]
    for method in METHODS:
        rr = [r for r in rows if r['method']==method]
        eligible = [r for r in rr if r['true_state'] in STATES]
        wash = [r for r in rr if r['true_state'] not in STATES]
        require(len(eligible)==36 and len(wash)==24, f'Stage16 counts: {method}')
        require(all(r.get('is_labeled_three_class_holdout')=='true' for r in eligible),
                f'Stage16 holdout flags incorrect: {method}')
        require(Counter(r['true_state'] for r in eligible)==Counter({s:12 for s in STATES}),
                f'Stage16 global class imbalance: {method}')
        require(Counter(r['held_out_line'] for r in eligible)==Counter({line:9 for line in LINES}),
                f'Stage16 fold count error: {method}')
        require(Counter((r['held_out_line'],r['true_state']) for r in eligible)==
                Counter({(line,s):3 for line in LINES for s in STATES}),
                f'Stage16 per-line/class imbalance: {method}')
        require(Counter((r['held_out_line'],r['true_state']) for r in wash)==
                Counter({(line,s):3 for line in LINES for s in ('ShortWashout','LongWashout')}),
                f'Stage16 washout row imbalance: {method}')
        stats=compute_stats(eligible)
        summary.append({'scope':'source_LOCO_pooled','method':method,'training_n_per_fit':27,
                        'evaluation_type':'historically_inspected_within_study',**stats})
        for line in LINES:
            sub=[r for r in eligible if r['held_out_line']==line]
            byline.append({'scope':'source_LOCO','held_out_line':line,'method':method,
                           'training_n':27, **compute_stats(sub)})
    stored=report.get('three_class_holdout_metrics',[])
    require(isinstance(stored,list) and len(stored)==len(METHODS)*5,
            'Stage16 stored three-class metrics count unexpected')
    saved=indexed(stored,('method','held_out_line'),'Stage16 stored metrics')
    for r in summary:
        z=saved[(r['method'],'ALL')]
        require(int(z['n'])==r['n'] and int(z['correct'])==r['correct'],
                'Stage16 pooled counts do not match stored metrics')
        approx_equal(z['balanced_accuracy'],r['balanced_accuracy'],'Stage16 pooled balanced accuracy drift')
        approx_equal(z['macro_f1'],r['macro_f1'],'Stage16 pooled macro-F1 drift')
    for r in byline:
        z=saved[(r['method'],r['held_out_line'])]
        require(int(z['n'])==r['n'] and int(z['correct'])==r['correct'],
                'Stage16 per-line counts drift')
        approx_equal(z['balanced_accuracy'],r['balanced_accuracy'],'Stage16 per-line balanced accuracy drift')
    return summary, byline, bykey


def check_external(rows: list[dict], report: dict, expected_n: int) -> tuple[list[dict],dict]:
    require(report.get('status')=='EXPLORATORY_NOT_INDEPENDENT',
            'Stage17 exploratory status changed')
    require(len(rows)==len(METHODS)*expected_n, f'Stage17 expected {expected_n} rows per method')
    indexed_rows=indexed(rows,('method','gsm'),'Stage17 external predictions')
    result=[]
    expected_ids={gsm:state for state,gsms in TARGET.items() for gsm in gsms}
    require(expected_n==len(expected_ids),'Only 9-sample external cohort supported')
    for method in METHODS:
        rr=[r for r in rows if r['method']==method]
        require({r['gsm'] for r in rr}==set(expected_ids), 'Stage17 external GSMs changed')
        for r in rr:
            require(r['true_state']==expected_ids[r['gsm']],f'External label mismatch: {r["gsm"]}')
            require(str(r.get('correct',''))==str(int(r['true_state']==r['predicted_state'])),
                    'Stage17 correct flag inconsistent')
        result.append({'scope':'external_full_source','method':method,
                       'training_n_per_fit':36,'evaluation_type':'previously_inspected_cross_study',
                       **compute_stats(rr)})
    metrics=indexed(report.get('metrics',[]),('method',),'Stage17 stored metrics')
    require(set(metrics)=={(m,) for m in METHODS}, 'Stage17 stored method metrics missing')
    for row in result:
        z=metrics[(row['method'],)]
        require(int(z['n'])==row['n'] and int(z['correct'])==row['correct'],
                'Stage17 stored metrics counts mismatch')
        approx_equal(z['balanced_accuracy'],row['balanced_accuracy'],'Stage17 bACC mismatch')
        approx_equal(z['macro_f1'],row['macro_f1'],'Stage17 macro-F1 mismatch')
    require(set(report.get('excluded_resistant_gsms',[])) ==
            {f'GSM808{i}' for i in (3383,3384,3385)},
            'Stage17 resistant exclusion changed')
    return result,indexed_rows


def check_pc9_excluded(rows: list[dict], report: dict, src_index:dict, ext_index:dict):
    require(report.get('status')=='EXPLORATORY_NOT_INDEPENDENT',
            'PC9-excluded report independence status changed')
    require(report.get('source_pc9_excluded') is True,'PC9-excluded source guard missing')
    require(report.get('stage16_source_PC9_predictions_reproduced') is True,
            'Stage16 PC9 replay guard missing')
    require(report.get('external_previous_full_source_predictions_checked') is True,
            'Stage17 full-source reconciliation guard missing')
    require(len(rows)==len(METHODS)*18,'Expected 18 records/method (source PC9 + external PC9)')
    ix=indexed(rows,('scope','method','gsm'),'Stage17 PC9-exclusion predictions')
    result=[]; switches=[]
    for method in METHODS:
        for scope in ('source_PC9_LOCO','external_GSE255958_PC9'):
            rr=[r for r in rows if r['scope']==scope and r['method']==method]
            require(len(rr)==9,'Unexpected PC9 group size')
            if scope=='external_GSE255958_PC9':
                expected={gsm:state for state,gsms in TARGET.items() for gsm in gsms}
                require({r['gsm'] for r in rr}==set(expected), 'PC9-excluded external GSM map changed')
                for r in rr:
                    require(r['true_state']==expected[r['gsm']], 'PC9-excluded external true label drift')
                    old=ext_index[(method,r['gsm'])]['predicted_state']
                    require(r['full_source_prediction']==old,'PC9 full-source comparison incorrect')
                    require(str(int(old!=r['predicted_state']))==str(r['changed_from_full_source']),
                            'PC9 changed prediction flag inconsistent')
                switches.append({'method':method,'external_n':9,
                  'correct_full_source':sum(ext_index[(method,r['gsm'])]['predicted_state']==r['true_state'] for r in rr),
                  'correct_PC9_excluded':sum(r['predicted_state']==r['true_state'] for r in rr),
                  'prediction_switches':sum(ext_index[(method,r['gsm'])]['predicted_state']!=r['predicted_state'] for r in rr),
                  'interpretation':'descriptive training-composition sensitivity, NOT independent test'})
                new_scope='external_without_source_PC9'
            else:
                source_ids={gsm for (m,gsm),r in src_index.items() if m==method and
                            r['held_out_line']=='PC9' and r['true_state'] in STATES}
                require({r['gsm'] for r in rr}==source_ids, 'Stage17 source PC9 LOCO GSMs mismatch Stage16')
                for r in rr:
                    old=src_index[(method,r['gsm'])]
                    require(old['predicted_state']==r['predicted_state'] and
                            old['true_state']==r['true_state'], 'Stage16 PC9 replay predictions drift')
                new_scope='source_PC9_holdout'
            require(Counter(r['true_state'] for r in rr)==Counter({s:3 for s in STATES}),
                    'Stage17 PC9 per-class counts changed')
            result.append({'scope':new_scope,'method':method,'training_n_per_fit':27,
               'evaluation_type':'historically_inspected_Pc9_sensitivity',**compute_stats(rr)})
    saved=indexed(report.get('metrics',[]),('scope','method'),'PC9-excluded stored metrics')
    require(len(saved)==len(METHODS)*2,'PC9-excluded stored metric size unexpected')
    for r in result:
        original_scope=('source_PC9_LOCO' if r['scope']=='source_PC9_holdout'
                        else 'external_GSE255958_PC9')
        z=saved[(original_scope,r['method'])]
        require(int(z['correct'])==r['correct'] and int(z['n'])==r['n'],
                'PC9-excluded summary correct/n drift')
        approx_equal(z['balanced_accuracy'],r['balanced_accuracy'],'PC9-excluded bACC drift')
        approx_equal(z['macro_f1'],r['macro_f1'],'PC9-excluded macro-F1 drift')
    return result,switches


def make_binary_table(stage15:dict) -> list[dict]:
    require('EXPLORATORY' in stage15.get('status',''), 'Stage15 status not exploratory')
    summary=stage15.get('summary',{})
    require(isinstance(summary,dict) and bool(summary),'Stage15 summary missing')
    out=[]
    for method,info in sorted(summary.items()):
        out.append({'method':method,
            'source_LOCO_bacc':info['source_loco_bacc'],
            'historical_three_study_macro_bacc':info['three_study_macro_bacc'],
            'historical_worst_study_bacc':info['worst_three_study_bacc'],
            'three_study_correct':info['three_study_correct'],
            'three_study_n':info['three_study_n'],
            'interpretation':'development-only benchmark; not independent validation'})
    return out


def audit_warnings(stage18:dict,stage19:dict) -> tuple[list[dict],list[dict]]:
    require(stage18.get('stage')==18 and stage19.get('stage')==19,
            'Stage18/19 reports absent or incorrect')
    findings=stage18.get('findings',[])
    require(findings and not any(r['status']=='FAIL' for r in findings),'Stage18 FAIL or no findings')
    require(stage19.get('source_acute_dtp_duration_separable') is False,
            'Stage19 confounding declaration changed')
    require(stage19.get('source_three_class_n')==36 and stage19.get('stage17_target_n')==12,
            'Stage19 cohort provenance mismatch')
    stage18_commit=stage18.get('repository_commit')
    require(isinstance(stage18_commit,str) and len(stage18_commit)>=7,'Missing Stage18 repository commit')
    require(stage19.get('stage18_commit')==stage18_commit,'Stage18 vs Stage19 commit mismatch')
    require(stage19.get('sample_count_by_study',{}).get('GSE193258')==36,
            'Stage19 source count mismatch')
    require(stage19.get('sample_count_by_study',{}).get('GSE255958')==12,
            'Stage19 GSE255958 sample count mismatch')
    warn=[]
    for row in findings:
        if row['status']=='WARN':
            warn.append({'check':row['check'],'finding':row['evidence'],
                         'status':'UNRESOLVED_AT_STAGE20'})
    conf=stage19.get('contrast_status_counts',{})
    require(sum(conf.values())>0,'Stage19 confounding comparison summary absent')
    contrast_rows=[{'verdict':key,'number_of_design_contrasts':value}
                   for key,value in sorted(conf.items())]
    return warn,contrast_rows


def confusion_long(all_rows):
    out=[]
    for row in all_rows:
        cm=row['confusion_matrix']
        require(len(cm)==3 and all(len(x)==3 for x in cm),'Invalid confusion matrix shape')
        for i,actual in enumerate(STATES):
            for j,pred in enumerate(STATES):
                out.append({'scope':row['scope'],'method':row['method'],
                            'true_state':actual,'predicted_state':pred,'count':cm[i][j]})
    return out


def fmt(x) -> str:
    return f'{float(x):.3f}'


def markdown_table(header:list[str], data:list[list[str]]) -> list[str]:
    return (['| ' + ' | '.join(header) + ' |',
             '|'+'|'.join(['---']*len(header))+'|'] +
            ['| ' + ' | '.join(str(v).replace('|','/') for v in row) + ' |' for row in data])


def render_md(all_rows, byline, switches, binary, warnings, conf, manifest):
    text=[
        '# Stage 20 — Exploratory evidence tables for professor review','',
        '**Do not use these tables to claim an independently validated classifier.**',
        'Every metric below is descriptive, recomputed from previously generated per-sample predictions.',
        'No model fitting, threshold optimization, statistical significance test or new validation was performed.',
        '', '## Table 1 — Control / Acute / DTP classification', '',
    ]
    order={'source_LOCO_pooled':0, 'external_full_source':1,
           'external_without_source_PC9':2, 'source_PC9_holdout':3}
    shown=sorted(all_rows,key=lambda r:(order[r['scope']],METHODS.index(r['method'])))
    text += markdown_table(
        ['Evaluation','Method','Train n','Correct / n','bACC','Macro-F1','Control recall','Acute recall','DTP recall'],
        [[r['scope'],r['method'],str(r['training_n_per_fit']),f"{r['correct']}/{r['n']}",
          fmt(r['balanced_accuracy']),fmt(r['macro_f1']),fmt(r['recall_control']),
          fmt(r['recall_acute']),fmt(r['recall_dtp'])] for r in shown])
    text += [
        '', 'Source LOCO pooled = the 36 held-out predictions across four cell lines, **not** a single fit.',
        'External 9-sample scores are from GSE255958, already inspected; 3 Control, 3 Acute and 3 DTP.',
        'The three Resistant samples were excluded by original design and were **not** assigned a DTP label.',
        'Training-source composition and study/protocol/processing differences prevent a confirmatory claim.',
        '', '## Table 2 — Training-source PC9 exclusion sensitivity', ''
    ]
    text += markdown_table(
        ['Method','Full source correct','Without source PC9 correct','Prediction switches (of 9)'],
        [[r['method'],f"{r['correct_full_source']}/9",f"{r['correct_PC9_excluded']}/9",
          str(r['prediction_switches'])] for r in switches])
    text += ['', 'These are **paired descriptive differences on the same previously inspected external nine samples**.',
             'They do not define a post-hoc winning method or prove cell-line independence.',
             '', '## Table 3 — Source leave-one-cell-line-out performance', '']
    text += markdown_table(['Method','Held-out cell line','Correct / 9','bACC','Macro-F1'],
        [[r['method'],r['held_out_line'],f"{r['correct']}/{r['n']}",
          fmt(r['balanced_accuracy']),fmt(r['macro_f1'])] for r in byline])
    text += ['', '## Table 4 — Historical binary benchmarks (Control vs DTP)', '']
    text += markdown_table(['Method','Source LOCO bACC','3-study macro bACC','Worst-study bACC','Correct / n'],
        [[r['method'],fmt(r['source_LOCO_bacc']),fmt(r['historical_three_study_macro_bacc']),
          fmt(r['historical_worst_study_bacc']),f"{r['three_study_correct']}/{r['three_study_n']}"]
          for r in binary])
    text += ['', 'Binary and three-class results have different tasks, sample counts and model definitions.',
             'Do not compare their bACC as if they were one common experiment.',
             '', '## Stage 19 design feasibility (metadata only)', '']
    text += markdown_table(['Duration-comparison status','Number of within-study/cell-line contrasts'],
                           [[r['verdict'],str(r['number_of_design_contrasts'])] for r in conf])
    text += ['', 'Acute 24 h versus DTP 21 d in GSE193258 is duration-confounded. Metadata contrast counts are not independent samples.',
             '', '## Stage 18 unresolved flags', '']
    text += markdown_table(['Check','Unresolved finding'],
                           [[r['check'],r['finding']] for r in warnings])
    text += ['', '## Rules for writing the manuscript', '',
             '1. Use the phrase **exploratory observed-state classification / cross-study stress test**.',
             '2. Do not claim independent external validation, causal DTP signatures, clinical treatment response prediction, or novel algorithm superiority.',
             '3. Treat unverified GEO sample-to-expression and biology labels as pending adjudication; disclose the six GSE303413 dose discrepancies.',
             '4. Report per-class recall, macro-F1 and confusion counts; do not interpret 9/9 as evidence of generalization.',
             '5. Explicitly distinguish raw log2(TPM+0.01) target processing and source expression provenance.',
             '6. Use these results to complete a **supervisor-review draft**; a publishable confirmatory claim needs independently reviewed data.',
             '', '## Reproducibility', '',
             f"- Original Stage18 code commit: `{manifest['stage18_code_commit']}`",
             f"- Input reports and per-sample files: `{len(manifest['input_sha256'])}`; SHA-256 saved in `20_evidence_manifest.json`.",
             '- Source scripts and prediction files were not overwritten.',
             '']
    return '\n'.join(text)


def save_csv(path:Path, rows:list[dict]):
    require(bool(rows),f'Empty output: {path.name}')
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(rows[0])
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        for row in rows:
            require(set(row)==set(keys),f'Unequal CSV fields in {path.name}')
            w.writerow(row)


def run(root:Path=ROOT):
    paths={name:root/path for name,path in FILES.items()}
    for name,path in paths.items():
        require(path.is_file(),f'Missing {name}: {path}')
    data={key:(load_csv(path) if path.suffix=='.csv' else load_json(path)) for key,path in paths.items()}
    src,byline,srcidx=check_source(data['stage16_pred'],data['stage16_report'])
    ext,extidx=check_external(data['stage17_pred'],data['stage17_report'],9)
    pc9,switches=check_pc9_excluded(data['stage17_excl_pred'],data['stage17_excl_report'],srcidx,extidx)
    binary=make_binary_table(data['stage15'])
    warnings,contrasts=audit_warnings(data['stage18'],data['stage19'])
    report={'stage':20,'status':'DESCRIPTIVE_EXPLORATORY_NOT_INDEPENDENT',
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'stage18_code_commit':data['stage18']['repository_commit'],
      'input_sha256':{key:{'path':str(path.relative_to(root)).replace('\\','/'), 'sha256':sha256(path)}
                      for key,path in paths.items()},
      'assumptions':{'three_classes':list(STATES),'stage17_resistant_in_main_evaluation':False,
                     'source_loco_is_four_separate_fits':True,
                     'GSE255958_previously_inspected':True,
                     'confidence_intervals_not_computed_due_to_biological_dependence':True,
                     'not_confirmatory':True},
      'source_loco_n':36,'external_3class_n':9,
      'unresolved_stage18_warnings':len(warnings),
      'stage19_design_contrasts':sum(r['number_of_design_contrasts'] for r in contrasts),
      'notes':['Comparisons are descriptive and not a model-selection opportunity.',
               'Stage20 does not prove absence of data leakage or grant biological approval.',
               'Do not count biological replicates as independent studies.']}
    out=root/'outputs'/'final_evidence'
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'20_three_class_summary.csv',src+ext+pc9)
    save_csv(out/'20_three_class_confusion_counts.csv',confusion_long(src+ext+pc9))
    save_csv(out/'20_source_loco_by_cell_line.csv',byline)
    save_csv(out/'20_pc9_sensitivity.csv',switches)
    save_csv(out/'20_binary_historical_benchmarks.csv',binary)
    save_csv(out/'20_confounding_contrasts.csv',contrasts)
    save_csv(out/'20_unresolved_warnings.csv',warnings)
    (out/'20_evidence_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (out/'20_paper_ready_tables.md').write_text(render_md(src+ext+pc9,byline,switches,binary,warnings,contrasts,report),encoding='utf-8')
    print('STAGE20 PASSED: matched source LOCO / external / PC9-excluded prediction identities and metrics')
    print('Stage20 evidence pack:',out)
    print('Three-class summary rows:',len(src+ext+pc9),'| Source folds:',len(byline),'| Stage18 warnings:',len(warnings))
    print('No model fitted; no independent validation asserted.')
    return report


def self_test():
    require(compute_stats([{'true_state':s,'predicted_state':s} for s in STATES])['macro_f1']==1,
            'Perfect multiclass F1 arithmetic')
    require(abs(compute_stats([{'true_state':s,'predicted_state':'Control'} for s in STATES])['balanced_accuracy']-1/3)<1e-9,
            'Constant prediction bACC arithmetic')
    methods=METHODS
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        def put(name,obj):
            path=root/FILES[name];path.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix=='.json':
                path.write_text(json.dumps(obj),encoding='utf-8')
            else:
                save_csv(path,obj)
        src=[]
        for method in methods:
            for line in LINES:
                for state in STATES:
                    for rep in range(3):
                        src.append({'method':method,'held_out_line':line,'gsm':f'{line}_{state}_{rep}',
                          'true_state':state,'predicted_state':state,
                          'is_labeled_three_class_holdout':'true'})
                for state in ('ShortWashout','LongWashout'):
                    for rep in range(3):
                        src.append({'method':method,'held_out_line':line,'gsm':f'{line}_{state}_{rep}',
                          'true_state':state,'predicted_state':'Control',
                          'is_labeled_three_class_holdout':'false'})
        metrics=[]
        for method in methods:
            for line in list(LINES)+['ALL']:
                rows=[r for r in src if r['method']==method and r['true_state'] in STATES and
                      (line=='ALL' or r['held_out_line']==line)]
                metrics.append({'method':method,'held_out_line':line,**compute_stats(rows)})
        put('stage16_pred',src)
        put('stage16_report',{'status':'EXPLORATORY_NOT_INDEPENDENT','trained_states':list(STATES),
              'independent_validation':False,'three_class_holdout_metrics':metrics})
        external=[]; exc=[]; main_metrics=[]; exc_metrics=[]
        for method in methods:
            for state, gsms in TARGET.items():
                for gsm in gsms:
                    external.append({'method':method,'gsm':gsm,'true_state':state,
                      'predicted_state':state,'correct':'1'})
            main_metrics.append({'method':method,**compute_stats([r for r in external if r['method']==method])})
            for scope in ('source_PC9_LOCO','external_GSE255958_PC9'):
                for state in STATES:
                    for k in range(3):
                        gsm=(f'PC9_{state}_{k}' if scope=='source_PC9_LOCO' else TARGET[state][k])
                        exc.append({'scope':scope,'method':method,'gsm':gsm,'true_state':state,
                            'predicted_state':state,'full_source_prediction':state if scope!='source_PC9_LOCO' else '',
                            'changed_from_full_source':'0' if scope!='source_PC9_LOCO' else ''})
                exc_metrics.append({'scope':scope,'method':method,**compute_stats([r for r in exc if r['scope']==scope and r['method']==method])})
        put('stage17_pred',external)
        put('stage17_report',{'status':'EXPLORATORY_NOT_INDEPENDENT','metrics':main_metrics,
            'excluded_resistant_gsms':[f'GSM808{i}' for i in (3383,3384,3385)]})
        put('stage17_excl_pred',exc)
        put('stage17_excl_report',{'status':'EXPLORATORY_NOT_INDEPENDENT',
            'source_pc9_excluded':True,'stage16_source_PC9_predictions_reproduced':True,
            'external_previous_full_source_predictions_checked':True,'metrics':exc_metrics})
        put('stage15',{'status':'EXPLORATORY','summary':{
            'method_a':{'source_loco_bacc':.5,'three_study_macro_bacc':.55,
                        'worst_three_study_bacc':.3,'three_study_correct':9,'three_study_n':18}}})
        commit='abcdef0123456789'
        put('stage18',{'stage':18,'repository_commit':commit,'findings':[
            {'check':'example_warning','status':'WARN','evidence':'Example scientific limitation'}]})
        put('stage19',{'stage':19,'stage18_commit':commit,
            'source_three_class_n':36,'stage17_target_n':12,
            'source_acute_dtp_duration_separable':False,
            'sample_count_by_study':{'GSE193258':36,'GSE255958':12},
            'contrast_status_counts':{'NO_MATCHED_DURATION':2}})
        report=run(root)
        require(report['source_loco_n']==36, 'Test end-to-end report')
        report_path=root/'outputs/final_evidence/20_paper_ready_tables.md'
        require(report_path.is_file() and '9' in report_path.read_text(), 'Missing paper report')
        # Negative control: sample-label drift MUST abort before writing new results.
        broken=list(external)
        broken[0]=dict(broken[0],true_state='DTP')
        put('stage17_pred',broken)
        try:
            run(root)
        except ValueError as e:
            require('label mismatch' in str(e),'Expected label mismatch on bad external sample')
        else:
            raise AssertionError('Failed to stop mislabeled external sample')
    print('SELF-TEST PASSED: metrics, source/target identity, report generation, and negative mislabel guard')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        run()

if __name__=='__main__':
    main()
