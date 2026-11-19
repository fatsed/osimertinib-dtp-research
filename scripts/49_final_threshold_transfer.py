#!/usr/bin/env python3
"""Stage 49 — FINAL, single pre-specified source-only threshold-transfer check.

This DOES NOT introduce a new classifier or new feature set. Stage 48's frozen
control-relative-rank learner had ROC-AUC=1 per held-out line, but its fixed
threshold 0 missed all DTPs in two lines. Test ONE alternative: choose a
threshold by maximizing balanced accuracy on nested, training-side *out-of-fold*
Stage-48-recipe scores, then transfer that threshold to the UNCHANGED Stage-48
outer held-out scores. Do not tune thresholds using outer-fold outcomes.

SOURCE ONLY: GSE193258, four repeatedly inspected cell lines. A known Control
from the new cell line is required. The 18-sample inner models and 27-sample
outer models may have different score scales. No independent validation,
novelty claim, or clinical/biological prediction is established.

Run from repo root:
    python scripts/49_final_threshold_transfer.py --self-test
    python scripts/49_final_threshold_transfer.py --run

Inputs must already exist locally from Stages 38, 47 and 48. This script will
replay Stage-48 outer scores, rebuild only the 12 nested fit/held-out splits,
compare to Stage-48 zero cutoff and frozen Stage-38 protected raw-zero on the
SAME eight held-out samples per fold. If there is no precommitted exploratory
improvement, STOP algorithmic search on these source folds.

Saved immutably to outputs/stage49_final_check/ (never overwrites prior work).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
OUT = ROOT / 'outputs' / 'stage49_final_check'
STAGE48_OUT = ROOT / 'outputs' / 'stage48_control_relative_rank'
STAGE47_OUT = ROOT / 'outputs' / 'stage47_control_anchored'
FROZEN38 = ROOT / 'outputs' / 'domain_invariance' / '38_predictions.csv'
LINES = ('H1975', 'HCC2935', 'HCC827', 'PC9')
STATES = ('Control', 'Acute', 'DTP')
K, C = 100, 0.01
METHODS = ('protected_raw_zero', 'stage48_zero', 'stage48_nested_oof_threshold')


def imp(file, name):
    path = SCRIPTS / file
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for buf in iter(lambda: file.read(1 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_text(rows):
    if not rows:
        raise ValueError('No rows to export')
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def immutable_save(files):
    # Check the ENTIRE batch before writing any file.
    for path, text in files.items():
        if path.exists() and path.read_bytes() != text.encode('utf-8'):
            raise FileExistsError(f'Immutable Stage49 output differs; refusing overwrite: {path}')
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(text, encoding='utf-8', newline='\n')


def train_inner_relative(Xtr, ytr, gtr, idtr, Xtest, idtest, panel, control_ref,
                         *, model_factory=None):
    """Fit a 2-line generalization of the EXACT Stage-48 LR recipe.

    Target outcomes and outer-test samples are NOT function parameters.
    The panel is selected independently on these 2 training lines.
    """
    Xtr, Xtest = np.asarray(Xtr, float), np.asarray(Xtest, float)
    ytr, gtr = np.asarray(ytr, int), np.asarray(gtr).astype(str)
    idtr, idtest = np.asarray(idtr).astype(str), np.asarray(idtest).astype(str)
    panel = np.asarray(panel, int)
    if (Xtr.ndim != 2 or Xtest.ndim != 2 or Xtr.shape[0] != 18
        or Xtest.shape[0] != 9 or Xtr.shape[1] != Xtest.shape[1]
        or len(ytr) != 18 or len(gtr) != 18 or len(idtr) != 18
        or len(idtest) != 9 or len(set(idtr)) != 18 or len(set(idtest)) != 9
        or set(idtr) & set(idtest) or panel.shape != (K,)
        or len(set(panel.tolist())) != K or min(panel) < 0 or max(panel) >= Xtr.shape[1]
        or not np.isfinite(Xtr).all() or not np.isfinite(Xtest).all()):
        raise ValueError('Inner training/held-out separation or feature design invalid')
    lines = sorted(set(gtr))
    if len(lines) != 2 or any(Counter(ytr[gtr == line]) != {0: 3, 1: 3, 2: 3} for line in lines):
        raise ValueError('Inner fit must comprise two complete source cell lines')
    if control_ref not in set(idtest):
        raise ValueError('Inner reference is not in inner held-out cell line')
    tr_ranks = rankdata(Xtr[:, panel], axis=1, method='average') / K
    te_ranks = rankdata(Xtest[:, panel], axis=1, method='average') / K
    delta = tr_ranks.copy()
    excluded = np.zeros(len(ytr), bool)
    refs = []
    for line in lines:
        refs_in_line = sorted(idtr[(gtr == line) & (ytr == 0)].tolist())
        if len(refs_in_line) != 3:
            raise ValueError('Each inner training line must contain 3 Controls')
        idx = np.flatnonzero(idtr == refs_in_line[0])
        if len(idx) != 1:
            raise AssertionError('Reference ID mismatch')
        delta[gtr == line] -= tr_ranks[idx[0]]
        excluded[idx[0]] = True
        refs.append(refs_in_line[0])
    if Counter(ytr[~excluded]) != {0: 4, 1: 6, 2: 6}:
        raise AssertionError('Training reference exclusion altered class counts')
    if model_factory is None:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, solver='liblinear', class_weight='balanced', max_iter=3000,
            random_state=42))
    else:
        model = model_factory()
    model.fit(delta[~excluded], (ytr[~excluded] == 2).astype(int))
    j = np.flatnonzero(idtest == control_ref)
    if len(j) != 1:
        raise AssertionError('Invalid held-out reference ID')
    scores = np.asarray(model.decision_function(te_ranks - te_ranks[j[0]]), float).reshape(-1)
    if scores.shape != (9,) or not np.isfinite(scores).all():
        raise ValueError('Non-finite inner held-out margins')
    return scores, refs


def inner_oof_for_outer(outer, ids, X, y, groups, builder):
    """24 truly nested training-side scores; never includes the outer held-out line."""
    rows = []
    for inner in LINES:
        if inner == outer:
            continue
        tr_idx = np.flatnonzero((groups != outer) & (groups != inner))
        va_idx = np.flatnonzero(groups == inner)
        if len(tr_idx) != 18 or len(va_idx) != 9 or outer in set(groups[tr_idx]) or inner in set(groups[tr_idx]):
            raise AssertionError('Nested split includes forbidden line')
        panel, _ = builder.select_dual_panel(X[tr_idx], y[tr_idx], groups[tr_idx], 2)
        refs = sorted(ids[va_idx][y[va_idx] == 0].tolist())
        if len(refs) != 3:
            raise ValueError('Need exactly three known inner validation Control samples')
        inner_control = refs[0]
        scored, train_refs = train_inner_relative(X[tr_idx], y[tr_idx], groups[tr_idx], ids[tr_idx],
                                                  X[va_idx], ids[va_idx], panel, inner_control)
        for j, ix in enumerate(va_idx):
            if ids[ix] == inner_control:
                continue
            rows.append({'outer_fold': outer, 'inner_held_out_line': inner,
                         'sample_id': ids[ix], 'group': STATES[int(y[ix])],
                         'score': format(float(scored[j]), '.15g'),
                         'inner_reference_id': inner_control,
                         'n_inner_train': 18,
                         'inner_training_lines': ';'.join(sorted(set(groups[tr_idx]))),
                         'train_reference_ids': ';'.join(train_refs)})
    if len(rows) != 24 or Counter(r['group'] for r in rows) != {'Control': 6, 'Acute': 9, 'DTP': 9}:
        raise AssertionError('Wrong inner OOF cohort: expected 6 Control + 9 Acute + 9 DTP')
    if len({r['sample_id'] for r in rows}) != 24 or any(r['inner_held_out_line'] == outer for r in rows):
        raise AssertionError('Outer-test contamination or duplicate inner sample')
    return rows


def threshold_from_oof(rows, stage47):
    scores = np.asarray([float(row['score']) for row in rows], float)
    labels = np.asarray([row['group'] == 'DTP' for row in rows], bool)
    return stage47.select_threshold(scores, labels)


def sample_metrics(rows, method):
    a = [r for r in rows if r['method'] == method]
    if len(a) != 8 or Counter(r['group'] for r in a) != {'Control': 2, 'Acute': 3, 'DTP': 3}:
        raise ValueError('Paired held-out cohort is not 2 Control + 3 Acute + 3 DTP')
    label = np.array([r['group'] == 'DTP' for r in a], bool)
    pred = np.array([int(r['pred_dtp']) == 1 for r in a], bool)
    scores = np.array([float(r['score']) for r in a], float)
    return dict(n=len(a), tp=int((pred & label).sum()), fn=int(((~pred) & label).sum()),
                control_fp=int(sum(pred[i] for i,r in enumerate(a) if r['group']=='Control')),
                acute_fp=int(sum(pred[i] for i,r in enumerate(a) if r['group']=='Acute')),
                sensitivity=float(pred[label].mean()),
                specificity=float((~pred[~label]).mean()),
                balanced_accuracy=float((pred[label].mean() + (~pred[~label]).mean())/2),
                roc_auc=float(roc_auc_score(label.astype(int),scores)))


def report(rows_by_fold, thresholds, audit):
    lines = ['# Stage 49 — Final source-only nested threshold transfer test', '',
             '**Exploratory only. This is NOT a novel algorithm or independent validation.**', '',
             'Hypothesis: a train-only nested OOF threshold can repair Stage48\'s fixed-zero cutoff failures.',
             'Stage48 classifier, target reference, selected genes, and outer held-out scores are FROZEN.',
             'The only change is a threshold obtained from 24 training-side OOF predictions per outer fold.',
             'Nested fit: 18 samples from two lines; outer Stage48 fit: 27 samples from three lines.',
             '', '## Paired performance', '',
             '| fold | method | DTP TP/3 | Acute FP/3 | Control FP/2 | BA | ROC-AUC |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows_by_fold:
        lines.append(f"| {row['fold']} | {row['method']} | {row['tp']}/3 | {row['acute_fp']}/3 | "
                     f"{row['control_fp']}/2 | {row['balanced_accuracy']:.3f} | {row['roc_auc']:.3f} |")
    lines.extend(['', '## Macro means (four lines)', '',
                  '| method | mean BA | sensitivity | specificity | total DTP | Acute FP | Control FP |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for method in METHODS:
        these = [r for r in rows_by_fold if r['method']==method]
        lines.append(f"| {method} | {np.mean([r['balanced_accuracy'] for r in these]):.3f} | "
                     f"{np.mean([r['sensitivity'] for r in these]):.3f} | "
                     f"{np.mean([r['specificity'] for r in these]):.3f} | "
                     f"{sum(r['tp'] for r in these)}/12 | {sum(r['acute_fp'] for r in these)}/12 | "
                     f"{sum(r['control_fp'] for r in these)}/8 |")
    a = {r['fold']:r for r in rows_by_fold if r['method']=='stage48_nested_oof_threshold'}
    b = {r['fold']:r for r in rows_by_fold if r['method']=='protected_raw_zero'}
    success = (sum(a[fold]['tp'] for fold in LINES)==12
               and sum(a[fold]['acute_fp'] for fold in LINES)<=1
               and sum(a[fold]['control_fp'] for fold in LINES)==0
               and all(a[fold]['balanced_accuracy']>=b[fold]['balanced_accuracy'] for fold in LINES))
    lines.extend(['', '## Pre-committed STOP/PASS decision', '',
                  '**'+('EXPLORATORY PASS: independent validation STILL required' if success else
                         'STOP: no qualifying improvement against Stage38 protected raw-zero')+'**', '',
                  'To qualify: 12/12 DTP TP, <=1 Acute FP, 0 Control FP, and no held-out-line BA decline.',
                  '', '## Source-only OOF thresholds', '',
                  '| outer fold | threshold from 24 train OOF | OOF BA |',
                  '|---|---:|---:|'])
    for row in thresholds:
        lines.append(f"| {row['outer_fold']} | {row['threshold']:.6g} | {row['train_oof_balanced_accuracy']:.3f} |")
    lines.extend(['', '## Limitations', '',
                  '- All four source LOCO folds and baseline behavior have previously been inspected repeatedly.',
                  '- No external cohort used; this is not independent evidence of novelty.',
                  '- Inner training has 18 samples while outer model has 27; absolute margin scale may not transfer.',
                  '- The Stage48 model uses a known same-cell-line Control and omits it from every paired evaluation.',
                  '- Inner panel and reference are selected entirely within each permitted inner training/validation split.',
                  '- The target Control identity is required to compute relative ranks; not a zero-shot predictor.',
                  '- Source Acute and DTP differ in treatment duration, and biology cannot be disentangled here.',
                  '- If this candidate fails the precommitted gate, do not keep testing further cutoffs on these four lines.',
                  '', '## Audit summary', '', f"- Provenance-checked outer replay rows: {audit['outer_replayed_rows']}",
                  f"- Nested source-only models: {audit['nested_models']}",
                  '- No outer outcomes enter inner threshold fitting.', ''])
    return '\n'.join(lines), success


def run():
    stage48 = imp('48_control_relative_rank_model.py', 'stage48_frozen_final49')
    stage47 = imp('47_control_anchored_calibration.py', 'stage47_threshold_final49')
    builder = imp('stage47_build_source_oof.py', 'builder47_selector_final49')
    st21 = imp('21_group_consistent_rank_selection.py', 'st21_source_final49')
    st27 = imp('27_dtp_specific_contrast_benchmark.py', 'st27_selector_final49')
    required = [STAGE48_OUT/'48_predictions.csv', STAGE48_OUT/'48_provenance.json',
                STAGE47_OUT/'47_source_scores.csv', STAGE47_OUT/'47_source_oof_provenance.json', FROZEN38]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError('Run audited Stages 38, 47 and 48 first; missing '+str([str(p) for p in required if not p.is_file()]))
    previous = json.loads((STAGE48_OUT/'48_provenance.json').read_text(encoding='utf-8'))
    prov47 = json.loads((STAGE47_OUT/'47_source_oof_provenance.json').read_text(encoding='utf-8'))
    if (previous.get('source_study')!='GSE193258' or previous.get('positive_label')!='DTP'
        or previous.get('independent_external_validation') is not False
        or previous.get('stage38_sha256')!=sha(FROZEN38)
        or previous.get('stage47_source_scores_sha256')!=sha(STAGE47_OUT/'47_source_scores.csv')
        or previous.get('stage47_provenance_sha256')!=sha(STAGE47_OUT/'47_source_oof_provenance.json')
        or prov47.get('source_study')!='GSE193258'
        or prov47.get('outer_test_replay_required_and_passed') is not True):
        raise ValueError('Prior stage provenance mismatch; refuse final test')
    genes, ids, X, y, groups, _ = st21.load_data()
    ids, X, y, groups = np.asarray(ids).astype(str), np.asarray(X,float), np.asarray(y,int), np.asarray(groups).astype(str)
    frozen = stage48.validate_source(ids,X,y,groups,read_csv(FROZEN38),prov47)
    stage48.stage47_score_integrity(ids,y,groups,frozen)
    replay,_ = stage48.perform(st21,st27,ids,X,y,groups,frozen)
    existing = read_csv(STAGE48_OUT/'48_predictions.csv')
    replay_map = {(r['fold'],r['method'],r['sample_id']):r for r in replay}
    existing_map = {(r['fold'],r['method'],r['sample_id']):r for r in existing}
    if len(replay_map)!=96 or len(existing_map)!=96 or replay_map.keys()!=existing_map.keys():
        raise ValueError('Stage48 immutable score rows missing/duplicated')
    for key,r in existing_map.items():
        original=replay_map[key]
        if (r['group']!=original['group'] or r['predeclared_reference_id']!=original['predeclared_reference_id']
            or abs(float(r['score'])-float(original['score']))>2e-7):
            raise ValueError('Stage48 stored scores do not match current source fit '+str(key))
    # Stage48 output scores remain unchanged; ONLY threshold changes.
    all_predictions, all_oof, threshold_rows, metrics_rows=[], [], [], []
    for fold in LINES:
        oof=inner_oof_for_outer(fold,ids,X,y,groups,builder)
        cut,train_ba=threshold_from_oof(oof,stage47)
        all_oof.extend(oof)
        threshold_rows.append({'outer_fold':fold,'threshold':float(cut),
                               'train_oof_balanced_accuracy':float(train_ba),
                               'n_train_oof':len(oof),'nested_models':3,
                               'n_inner_training_samples':18,
                               'n_outer_training_samples':27})
        for name in METHODS:
            frozen_method = ('protected_raw_zero' if name=='protected_raw_zero' else 'candidate_control_relative_rank')
            candidates=[r for r in existing if r['fold']==fold and r['method']==frozen_method]
            if len(candidates)!=8:
                raise ValueError('Frozen Stage48 paired evaluation cohort missing')
            for r in candidates:
                margin = float(r['score'])
                thresh = float(cut if name=='stage48_nested_oof_threshold' else 0.)
                pred = int(margin>thresh)
                # Guard: Stage48 original decisions are preserved exactly.
                if name in ('stage48_zero','protected_raw_zero') and pred!=int(r['pred_dtp']):
                    raise AssertionError('Frozen Stage48 decisions mismatch')
                all_predictions.append({'fold':fold,'method':name,'sample_id':r['sample_id'],
                                        'group':r['group'],'reference_id':r['predeclared_reference_id'],
                                        'score':format(margin,'.15g'),
                                        'threshold':format(thresh,'.15g'),
                                        'pred_dtp':pred})
            metrics_rows.append({'fold':fold,'method':name,**sample_metrics(
                [r for r in all_predictions if r['fold']==fold],name)})
    if len(all_predictions)!=96 or len(all_oof)!=96 or len(metrics_rows)!=12 or len(threshold_rows)!=4:
        raise AssertionError('Unexpected run accounting')
    # EXACT matched held-out samples for all competitors.
    for fold in LINES:
        compare=[{(r['sample_id'],r['group']) for r in all_predictions
                  if r['fold']==fold and r['method']==name} for name in METHODS]
        if not (compare[0]==compare[1]==compare[2]):
            raise AssertionError('Unpaired comparator subsets')
    audit={'stage':49, 'status':'one final exploratory attempt, not independent validation',
           'source_study':'GSE193258', 'protocol':'Stage48 fixed model + nested OOF threshold only',
           'n_nested_models':12, 'nested_models':12, 'outer_replayed_rows':len(replay_map),
           'positive':'DTP', 'target_control_required':True,
           'outer_fit_lines':3,'inner_fit_lines':2,
           'stage48_score_sha256':sha(STAGE48_OUT/'48_predictions.csv'),
           'stage48_provenance_sha256':sha(STAGE48_OUT/'48_provenance.json'),
           'stage47_source_sha256':sha(STAGE47_OUT/'47_source_scores.csv'),
           'frozen38_sha256':sha(FROZEN38),
           'algorithm_novelty_proven':False,'independent_external_validation':False,
           'precommitted_gate':'12/12 DTP, <=1 Acute false DTP, zero Control false DTP, no per-fold BA decrease versus Protected raw zero'}
    md,passed=report(metrics_rows,threshold_rows,{'outer_replayed_rows':len(replay_map),'nested_models':12})
    audit['precommitted_exploratory_gate_passed']=passed
    files={OUT/'49_inner_oof_scores.csv':csv_text(all_oof),
           OUT/'49_thresholds.csv':csv_text(threshold_rows),
           OUT/'49_predictions.csv':csv_text(all_predictions),
           OUT/'49_metrics_by_fold.csv':csv_text(metrics_rows),
           OUT/'49_report.md':md+'\n',
           OUT/'49_provenance.json':json.dumps(audit,indent=2,ensure_ascii=False)+'\n'}
    immutable_save(files)
    print('STAGE49 COMPLETE: source-only final nested OOF threshold check')
    for method in METHODS:
        subset=[r for r in metrics_rows if r['method']==method]
        print(f"{method:30s} BA={np.mean([r['balanced_accuracy'] for r in subset]):.3f} "
              f"DTP={sum(r['tp'] for r in subset)}/12 "
              f"AcuteFP={sum(r['acute_fp'] for r in subset)}/12 "
              f"ControlFP={sum(r['control_fp'] for r in subset)}/8")
    print('PRECOMMITTED GATE:', 'EXPLORATORY PASS (external validation still required)' if passed else 'FAILED — STOP further source tuning')
    for p in files:print('Saved:',p.resolve())


def self_test():
    rng=np.random.default_rng(4901)
    groups=np.repeat(np.array(['A','B','C','D']),9)
    labels=np.tile(np.repeat([0,1,2],3),4)
    ids=np.array([f'SYNTH_{i:03d}' for i in range(36)])
    X=rng.normal(0,0.3,(36,520))
    X+=np.linspace(-1,1,520)
    X[labels==2,:200]+=np.linspace(20,35,200)
    # One synthetic nested 2-line fit, using fixed panel chosen without held-out line.
    train=np.flatnonzero(np.isin(groups,['A','B']))
    test=np.flatnonzero(groups=='C')
    builder=imp('stage47_build_source_oof.py','stage47_builder_test49')
    stage47=imp('47_control_anchored_calibration.py','stage47_threshold_test49')
    panel,_=builder.select_dual_panel(X[train],labels[train],groups[train],2)
    ref=sorted(ids[test][labels[test]==0].tolist())[0]
    scores,training_refs=train_inner_relative(X[train],labels[train],groups[train],ids[train],
                                              X[test],ids[test],panel,ref)
    assert len(training_refs)==2 and scores.shape==(9,) and np.isfinite(scores).all()
    assert len(set(ids[train])&set(ids[test]))==0
    rows=[{'group':STATES[int(labels[j])],'score':format(float(scores[k]),'.15g')}
          for k,j in enumerate(test) if ids[j]!=ref]
    assert Counter(r['group'] for r in rows)=={'Control':2,'Acute':3,'DTP':3}
    # Test outcomes do not flow to training-side OOF threshold fitting.
    original=[{'score':str(x),'group':'DTP' if y else 'Acute'} for x,y in
              zip(np.arange(8,dtype=float),[0,0,0,0,1,1,1,1])]
    cut,ba=threshold_from_oof(original,stage47)
    assert np.isfinite(cut) and 0<=ba<=1
    fake_test_labels=np.array([0,1,0,1,1,0,1,0])
    flipped_test_labels=1-fake_test_labels
    assert not np.array_equal(fake_test_labels,flipped_test_labels)
    cut2,_=threshold_from_oof(original,stage47)
    assert cut2==cut
    print('SELF-TEST PASSED: nested 2-line Control-relative fit, isolated labels, fixed panel, finite scores, deterministic threshold. SYNTHETIC ONLY.')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--run',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test()
    elif args.run:run()


if __name__=='__main__':main()
