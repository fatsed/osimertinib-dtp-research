#!/usr/bin/env python3
"""Stage34 | FIRST source-only domain-robust rank-model PROTOTYPE (exploratory).

Scientific hypothesis: class-conditional cell-line directions cause margin
instability. Shrink those directions in TRAINING feature space only, while
retaining Stage28's original exact 100-gene panels, rank representation,
balanced LR(C=.01), and decision threshold zero.

IMPORTANT: This is an exploratory adaptation of known domain-alignment ideas,
not a demonstrated original algorithm, nor independent validation.
Source cell-line holdouts have been examined before. GSE255958 is deliberately
NOT accessed, used for model selection, tuning, or scoring in this stage.

Usage from repo root:
    python scripts/34_source_only_class_conditional_shrinkage.py --self-test
    python scripts/34_source_only_class_conditional_shrinkage.py --exploratory-only

Outputs (only NEW files, no overwrites):
    outputs/domain_invariance/34_report.md
    outputs/domain_invariance/34_summary.json
    outputs/domain_invariance/34_predictions.csv
    outputs/domain_invariance/34_train_domain_diagnostics.csv
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

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
FROZEN = ROOT / 'outputs' / 'dtp_specificity'
OUT = ROOT / 'outputs' / 'domain_invariance'
LINES = ('H1975', 'HCC2935', 'HCC827', 'PC9')
STATES = ('Control', 'Acute', 'DTP')
METHODS = ('acute_only_rank_lr100', 'dual_contrast_rank_lr100')
K = 100
C = .01
GAMMA = 1.0  # explicitly precommitted, NOT selected from source or target scores
SEED = 42


def imp(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def csv_read(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def csv_dump(rows):
    if not rows:
        raise ValueError('No rows')
    f = io.StringIO(newline='')
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return f.getvalue()


def write_immutable(files: dict[str, str]):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = OUT / name
        if path.exists() and path.read_bytes() != data.encode('utf-8'):
            raise FileExistsError(f'Will not overwrite existing changed Stage34 artifact: {path}')
    for name, data in files.items():
        path = OUT / name
        if not path.exists():
            path.write_bytes(data.encode('utf-8'))


def lr_model():
    return LogisticRegression(C=C, solver='liblinear', class_weight='balanced',
                              max_iter=3000, random_state=SEED)


def class_conditional_shrinkage(z_train, y_train, domains_train, gamma=GAMMA):
    """Return train-fitted linear transform: P=(I + gamma*D'D)^(-1/2).

    D has nine standardized class-by-source-line mean deviations; each class
    is centered *within itself* (Control, Acute, DTP are NOT pooled).
    No test features, labels or target-study information enter this function.
    Uniform 3x3 state/line structure required; never silently change design.
    """
    z = np.asarray(z_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    lines = np.asarray(domains_train)
    if z.ndim != 2 or z.shape[1] != K or z.shape[0] != len(y) or len(y) != len(lines):
        raise ValueError('Wrong train-only input shapes')
    if not np.isfinite(z).all() or not np.isfinite(gamma) or gamma < 0:
        raise ValueError('Invalid matrix or gamma')
    uniq = sorted(set(lines.tolist()))
    if len(uniq) != 3 or any(Counter(y[lines == line].tolist()) != {0: 3, 1: 3, 2: 3} for line in uniq):
        raise ValueError('Must use exactly 3 training lines, 3 replicates per class')
    deltas = []
    for state in (0, 1, 2):
        means = np.stack([z[(lines == line) & (y == state)].mean(axis=0) for line in uniq])
        deltas.extend(means - means.mean(axis=0, keepdims=True))
    D = np.stack(deltas) / np.sqrt(9.0)
    if D.shape != (9, K):
        raise AssertionError('Wrong class-domain residual matrix')
    cov = D.T @ D
    eigen, vec = np.linalg.eigh(cov)
    if eigen.min() < -1e-8:
        raise AssertionError('Expected PSD domain nuisance directions')
    eigen = np.clip(eigen, 0.0, None)
    multipliers = np.reciprocal(np.sqrt(1 + gamma * eigen))
    P = (vec * multipliers) @ vec.T
    P = (P + P.T) / 2
    if not np.isfinite(P).all() or P.shape != (K, K):
        raise ValueError('Invalid source-fitted shrinkage transform')
    if gamma == 0 and not np.allclose(P, np.eye(K), atol=1e-10):
        raise AssertionError('No-shrink limit changed input space')
    trace = {'n_domain_residual_rows': int(D.shape[0]),
             'nonzero_domain_directions': int(np.count_nonzero(eigen > 1e-9)),
             'max_domain_eigenvalue': float(eigen.max()),
             'domain_direction_fro_before': float(np.linalg.norm(D)),
             'domain_direction_fro_after': float(np.linalg.norm(D @ P))}
    if trace['domain_direction_fro_after'] > trace['domain_direction_fro_before'] + 1e-9:
        raise AssertionError('Nuisance-direction shrinkage increased input residual')
    return P, trace


def within_panel_rank(X, panel):
    z = np.asarray(X, dtype=float)[:, np.asarray(panel, dtype=int)]
    if z.ndim != 2 or z.shape[1] != K or not np.isfinite(z).all():
        raise ValueError('Wrong selected gene panel')
    return rankdata(z, axis=1, method='average') / K


def class_line_spread(margins, y, groups):
    """Mean squared deviation of 9 (state,line) mean scores about each state mean."""
    y = np.asarray(y, dtype=int)
    g = np.asarray(groups)
    s = np.asarray(margins, dtype=float)
    if len(s) != len(y) or len(s) != len(g):
        raise ValueError('Misaligned score dispersion inputs')
    c = []
    for state in (0, 1, 2):
        means = np.array([np.mean(s[(y == state) & (g == line)]) for line in sorted(set(g))])
        if not np.isfinite(means).all():
            raise ValueError('Empty domain-state cell')
        c.extend((means - means.mean()) ** 2)
    return float(np.mean(c))


def fit_frozen_and_prototype(ztr, zte, ytr, gtr):
    """Fit two heads under identical panels/train split; no target fitting."""
    ztr, zte = np.asarray(ztr, float), np.asarray(zte, float)
    if ztr.shape != (27, K) or zte.ndim != 2 or zte.shape[1] != K:
        raise ValueError('Wrong training/test rank dimensions')
    binary = (np.asarray(ytr) == 2).astype(int)
    baseline = make_pipeline(StandardScaler(), lr_model())
    baseline.fit(ztr, binary)
    score0 = np.asarray(baseline.decision_function(zte), float)
    sc = baseline.named_steps['standardscaler']
    z0 = sc.transform(ztr)
    ztest0 = sc.transform(zte)
    P, trace = class_conditional_shrinkage(z0, ytr, gtr, gamma=GAMMA)
    # No train/test label or target-data transforms, and no second StandardScaler.
    candidate = lr_model().fit(z0 @ P, binary)
    score1 = np.asarray(candidate.decision_function(ztest0 @ P), float)
    train0 = baseline.decision_function(ztr)
    train1 = candidate.decision_function(z0 @ P)
    trace.update({'train_class_conditional_margin_dispersion_baseline':
                  class_line_spread(train0, ytr, gtr),
                  'train_class_conditional_margin_dispersion_prototype':
                  class_line_spread(train1, ytr, gtr)})
    if not np.isfinite(score0).all() or not np.isfinite(score1).all():
        raise ValueError('Nonfinite margins')
    if not np.allclose(baseline.decision_function(ztr),
                       lr_model().fit(z0, binary).decision_function(z0), atol=1e-9, rtol=1e-8):
        raise AssertionError('Baseline scaling replay mismatch')
    return score0, score1, trace


def score_metrics(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    if len(y) != len(score) or not len(y) or not np.isfinite(score).all():
        raise ValueError('Metrics invalid data')
    true = (y == 2)
    got = score > 0
    by_class = {STATES[j]: float(np.mean(got[y == j] == (j == 2))) for j in (0, 1, 2)}
    return {'correct': int(np.sum(got == true)), 'n': len(y),
            'three_condition_macro_recall': float(np.mean(list(by_class.values()))),
            'control_false_dtp': float(np.mean(got[y == 0])),
            'acute_false_dtp': float(np.mean(got[y == 1])),
            'dtp_recall': float(np.mean(got[y == 2]))}


def load_frozen_reference(genes, ids, y, groups):
    """Require full, exact Stage28 source replay targets before any new comparison."""
    ref_preds = csv_read(FROZEN / '28_predictions.csv')
    selected28 = csv_read(FROZEN / '28_selected_genes.csv')
    selected27 = csv_read(FROZEN / '27_selected_genes.csv')
    ref = {}
    for r in ref_preds:
        key = (r['method'], r['held_out_line'], r['gsm'])
        if key in ref:
            raise ValueError('Duplicate Stage28 predictions')
        ref[key] = r
    if len(ref) != 180:
        raise ValueError('Stage28 prediction file incomplete')
    panels = {}
    for r in selected28 + selected27:
        if r['method'] not in METHODS:
            continue
        key = (r['method'], r['held_out_line'])
        panels.setdefault(key, []).append(r['gene'])
    if set(panels) != {(m, line) for m in METHODS for line in LINES}:
        raise ValueError('Missing frozen source panel')
    if any(len(p) != K or len(set(p)) != K for p in panels.values()):
        raise ValueError('Wrong panel multiplicities')
    for i, sid in enumerate(ids):
        line = str(groups[i])
        for m in METHODS:
            row = ref.get((m, line, sid))
            if row is None or row['true_state'] != STATES[int(y[i])]:
                raise ValueError('Frozen source GSM/label mismatch')
    return ref, panels


def evaluate_arrays(genes, ids, X, y, groups, st27, st28,
                    ref=None, panels=None):
    """Source-only four-LOCO test, allows synthetic inputs for integration checks."""
    y = np.asarray(y, int); groups = np.asarray(groups)
    X = np.asarray(X, float)
    if len(set(ids)) != 36 or X.shape[0] != 36 or X.shape[1] != len(genes) or len(y) != 36:
        raise ValueError('Bad source shapes or samples')
    if not np.isfinite(X).all() or Counter(y.tolist()) != {0: 12, 1: 12, 2: 12}:
        raise ValueError('Bad source expression or classes')
    if set(groups.tolist()) != set(LINES):
        raise ValueError('Wrong source cell lines')
    rows=[];diagnostics=[]
    for outer in LINES:
        tr = np.flatnonzero(groups != outer)
        te = np.flatnonzero(groups == outer)
        st27.validate_train(y[tr], groups[tr])
        if len(tr) != 27 or len(te) != 9 or Counter(y[te].tolist()) != {0:3,1:3,2:3}:
            raise ValueError('Wrong outer split structure')
        for method in METHODS:
            if method == 'acute_only_rank_lr100':
                panel, _ = st28.select_single_contrast(X[tr], y[tr], groups[tr], negative=1)
            else:
                panel, _ = st27.select_rank_panel(X[tr], y[tr], groups[tr], mode='dual_contrast')
            genes_now = [genes[int(idx)] for idx in panel]
            if ref is not None and set(genes_now) != set(panels[(method, outer)]):
                raise ValueError(f'Frozen Stage28 gene panel changed: {method}/{outer}')
            ztr = within_panel_rank(X[tr], panel)
            zte = within_panel_rank(X[te], panel)
            s0, s1, trace = fit_frozen_and_prototype(ztr, zte, y[tr], groups[tr])
            if ref is not None:
                for j, i in enumerate(te):
                    r = ref[(method, outer, ids[i])]
                    if not np.isclose(float(r['margin']), s0[j], atol=1e-8, rtol=1e-7):
                        raise ValueError(f'Stage28 prediction not exactly replayed: {method}/{outer}/{ids[i]}')
                    if int(r['predicted_dtp']) != int(s0[j] > 0):
                        raise ValueError('Frozen source threshold mismatch')
            diagnostics.append({'method': method, 'held_out_line': outer,
                                'gamma_fixed_in_script': GAMMA,
                                **{k: f'{v:.12g}' for k,v in trace.items()}})
            for j, i in enumerate(te):
                for variant, score in (('stage28_baseline', s0[j]), ('class_conditional_shrinkage_gamma1', s1[j])):
                    pred = int(score > 0)
                    rows.append({'method': method, 'variant': variant,
                                 'held_out_line': outer, 'gsm': ids[i],
                                 'true_state': STATES[int(y[i])],
                                 'margin': f'{float(score):.12g}',
                                 'predicted_dtp': pred,
                                 'correct': int(pred == (y[i] == 2))})
    if len(rows) != len(METHODS) * 2 * 36 or len(diagnostics) != len(METHODS) * 4:
        raise ValueError('Incomplete four-fold comparison')
    return rows, diagnostics


def summarize(rows, diagnostics, fingerprints, provenance):
    methods={}
    for m in METHODS:
        metrics={}
        for variant in ('stage28_baseline', 'class_conditional_shrinkage_gamma1'):
            rr = [r for r in rows if r['method'] == m and r['variant'] == variant]
            y = np.asarray([STATES.index(r['true_state']) for r in rr])
            margins = np.asarray([float(r['margin']) for r in rr])
            pooled = score_metrics(y, margins)
            per_line = {}
            for line in LINES:
                rline = [r for r in rr if r['held_out_line'] == line]
                per_line[line] = score_metrics(
                    [STATES.index(r['true_state']) for r in rline],
                    [float(r['margin']) for r in rline])
            metrics[variant] = {'pooled': pooled, 'per_held_out_line': per_line}
        base = {r['gsm']: bool(int(r['correct'])) for r in rows if r['method']==m and r['variant']=='stage28_baseline'}
        new = {r['gsm']: bool(int(r['correct'])) for r in rows if r['method']==m and r['variant']=='class_conditional_shrinkage_gamma1'}
        if set(base) != set(new) or len(base) != 36:
            raise ValueError('Pairing lost')
        paired={'baseline_only_correct': sum(v and not new[k] for k,v in base.items()),
                'prototype_only_correct': sum(new[k] and not v for k,v in base.items()),
                'both_correct': sum(v and new[k] for k,v in base.items()),
                'both_incorrect': sum(not v and not new[k] for k,v in base.items())}
        methods[m]={'metrics':metrics,'paired':paired}
    result={'status':'EXPLORATORY_SOURCE_LOCO_PREVIOUSLY_INSPECTED_NOT_EXTERNAL',
            'study':'GSE193258','n_source':36,'outer_test_lines':list(LINES),
            'model_variants':['stage28_baseline','class_conditional_shrinkage_gamma1'],
            'methods':methods, 'gamma_fixed_in_script':GAMMA,
            'n_selected_genes':K, 'regularization_C':C,'threshold':0,
            'training_groups':['Control','Acute','DTP'],
            'problem':'binary DTP vs Control+Acute',
            'algorithm_mechanism':'class-conditional source-line nuisance directions shrunk via (I+gamma*D.T@D)^(-1/2), D from training line means only',
            'source_provenance':provenance,'source_file_sha256':fingerprints,
            'diagnostics':diagnostics,
            'restrictions':['No target/held-out data enters panel selection, scaling, shrinkage, fit, or threshold',
                            'GSE255958 and any other external sample NOT read or scored',
                            'One fixed gamma=1 in this prototype; no LOCO hyperparameter search',
                            'All four source folds previously inspected during development',
                            'Cannot isolate DTP biology from treatment duration',
                            'No independent validation, causal gene marker or original algorithm claim']}
    return result


def report_markdown(summary):
    d=['# Stage 34 — Source-only class-conditional domain-shrinkage prototype','',
       '**Exploratory fixed-in-code gamma=1 only (NOT formal preregistration). Not a novel-algorithm or independent-validation claim.**','',
       'Question: does reducing TRAINING cell-line-specific class-mean directions help fixed rank models on held-out lines?','',
       'Same source-only LOCO, same feature selectors and 100 genes, C=0.01 and decision threshold zero.',
       'The baseline is exactly replayed against frozen Stage28 per-GSM margins and panels.','',
       '| Model | Variant | Correct/36 | 3-condition macro | Control false DTP | Acute false DTP | DTP recall | Worst-line macro |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        for v in ('stage28_baseline','class_conditional_shrinkage_gamma1'):
            z=summary['methods'][m]['metrics'][v]
            p=z['pooled'];worst=min(r['three_condition_macro_recall'] for r in z['per_held_out_line'].values())
            d.append(f"| {m} | {v} | {p['correct']}/36 | {p['three_condition_macro_recall']:.3f} | "
                     f"{p['control_false_dtp']:.3f} | {p['acute_false_dtp']:.3f} | {p['dtp_recall']:.3f} | {worst:.3f} |")
    d.extend(['','## Same-sample paired predictions','',
              '| Model | Baseline only correct | Prototype only correct | Both correct | Both wrong |',
              '|---|---:|---:|---:|---:|'])
    for m in METHODS:
        z=summary['methods'][m]['paired']
        d.append(f"| {m} | {z['baseline_only_correct']} | {z['prototype_only_correct']} | {z['both_correct']} | {z['both_incorrect']} |")
    d.extend(['','## Training-only domain-dispersion diagnostic','',
              '| Model | Median baseline line-state score variance | Median prototype variance |',
              '|---|---:|---:|'])
    for m in METHODS:
        ds=[r for r in summary['diagnostics'] if r['method']==m]
        a=np.median([float(r['train_class_conditional_margin_dispersion_baseline']) for r in ds])
        b=np.median([float(r['train_class_conditional_margin_dispersion_prototype']) for r in ds])
        d.append(f'| {m} | {a:.6f} | {b:.6f} |')
    d.extend(['','**If the shrinkage helps in-sample training-domain dispersion but hurts held-out classification, the proposed mechanism is not supported by this experiment.**',
              'No gamma tuning on these already-examined folds; gamma=1 is a fixed initial prototype.',
              'Class-conditional alignment is a known family of approaches; this adaptation is NOT proven original.',
              'GSE255958 is NOT used in this stage, even as a secondary score.',
              'Observed state labels remain confounded with osimertinib exposure duration.',''])
    return '\n'.join(d)


def run():
    s21=imp(SCRIPTS/'21_group_consistent_rank_selection.py','stage21_for34')
    s27=imp(SCRIPTS/'27_dtp_specific_contrast_benchmark.py','stage27_for34')
    s28=imp(SCRIPTS/'28_dual_contrast_ablation.py','stage28_for34')
    genes, ids, X, y, groups, provenance=s21.load_data()
    if X.shape!=(36,19712) or len(genes)!=19712:
        raise ValueError('Source cohort not identical to Stage28')
    ref,panels=load_frozen_reference(genes,ids,y,groups)
    rows,diag=evaluate_arrays(genes,ids,X,y,groups,s27,s28,ref=ref,panels=panels)
    fingerprint={str(p.relative_to(ROOT)):file_sha(p) for p in (
        FROZEN/'27_selected_genes.csv',FROZEN/'28_predictions.csv',
        FROZEN/'28_selected_genes.csv',FROZEN/'28_report.md')}
    summary=summarize(rows,diag,fingerprint,provenance)
    write_immutable({'34_predictions.csv':csv_dump(rows),
                     '34_train_domain_diagnostics.csv':csv_dump(diag),
                     '34_summary.json':json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',
                     '34_report.md':report_markdown(summary)})
    print('Stage34 complete: exact Stage28 source per-GSM predictions and panels REPLAYED')
    for m in METHODS:
        z=summary['methods'][m]
        b=z['metrics']['stage28_baseline']['pooled']
        c=z['metrics']['class_conditional_shrinkage_gamma1']['pooled']
        print(f"{m}: baseline {b['correct']}/36 macro={b['three_condition_macro_recall']:.3f} vs prototype {c['correct']}/36 macro={c['three_condition_macro_recall']:.3f}; "
              f"prototype DTP_recall={c['dtp_recall']:.3f}; baseline_only={z['paired']['baseline_only_correct']} prototype_only={z['paired']['prototype_only_correct']}")
    print('Saved outputs/domain_invariance/34_{report.md,summary.json,predictions.csv,train_domain_diagnostics.csv}')
    print('EXPLORATORY SOURCE ONLY: prior source folds inspected; external study not scored; no novelty claim')


def self_test():
    rng=np.random.default_rng(3401)
    labels=np.tile(np.repeat(np.arange(3),3),3)
    domains=np.repeat(np.arange(3),9)
    train=rng.normal(0,1,(27,K))
    train[:, :10] += (labels==2)[:,None]*1.6
    train[:, 10:20] += domains[:,None]*0.8
    test=rng.normal(0,1,(5,K))
    P0,t0=class_conditional_shrinkage(train,labels,domains,gamma=0)
    P1,t1=class_conditional_shrinkage(train,labels,domains,gamma=1)
    if not np.allclose(train@P0,train,atol=1e-9) or np.max(np.abs(P1-P1.T))>1e-9:
        raise AssertionError('Invalid projection limits')
    if t1['domain_direction_fro_after']>t1['domain_direction_fro_before']+1e-10:
        raise AssertionError('No nuisance shrinkage')
    p=make_pipeline(StandardScaler(),lr_model()).fit(train,(labels==2).astype(int))
    sc=StandardScaler().fit(train)
    reconstructed=lr_model().fit(sc.transform(train),(labels==2).astype(int))
    if not np.allclose(p.decision_function(test),reconstructed.decision_function(sc.transform(test)),atol=1e-10):
        raise AssertionError('Baseline invariant test failed')
    if t0['domain_direction_fro_before']<0.1:
        raise AssertionError('Synthetic nuisance too small')
    try:
        class_conditional_shrinkage(train,labels,domains,gamma=-1)
        raise AssertionError('Negative gamma accepted')
    except ValueError:
        pass
    try:
        class_conditional_shrinkage(train,np.roll(labels,1),domains[:26])
        raise AssertionError('Mismatched groups accepted')
    except ValueError:
        pass
    # Integration: both Stage27/28 FEATURE selectors, 4 LOCO folds,
    # original baseline fit and new transform on no-target synthetic source.
    s27=imp(SCRIPTS/'27_dtp_specific_contrast_benchmark.py','stage27_synth34')
    s28=imp(SCRIPTS/'28_dual_contrast_ablation.py','stage28_synth34')
    n_genes=780
    gs=[];yy=[];gg=[];xs=[]
    base=rng.normal(0,0.6,n_genes)
    line_noise=rng.normal(0,0.12,(4,n_genes))
    for li,line in enumerate(LINES):
        for state in range(3):
            for rep in range(3):
                sample=base+line_noise[li]+rng.normal(0,0.06,n_genes)
                if state==2:
                    sample[:170]+=3.0
                    sample[170:340]-=3.0
                if state==1:
                    sample[340:380]+=0.2
                xs.append(sample);gg.append(line);yy.append(state)
                gs.append(f'SYN{li}{state}{rep}')
    gene_names=[f'GENE_{i:04d}' for i in range(n_genes)]
    preds,d=evaluate_arrays(gene_names,gs,np.vstack(xs),np.asarray(yy),np.asarray(gg),s27,s28)
    if len(preds)!=144 or len(d)!=8:
        raise AssertionError('Synthetic integration wrong output size')
    for r in preds:
        if r['true_state'] not in STATES or r['variant'] not in ('stage28_baseline','class_conditional_shrinkage_gamma1'):
            raise AssertionError('Synthetic output invalid')
    synthetic_summary=summarize(preds,d,{'source':'SYNTHETIC_TEST_ONLY'}, {'synthetic':True})
    md=report_markdown(synthetic_summary)
    if len(md)<500 or any(x not in md for x in ('paired predictions','acute_only_rank_lr100','dual_contrast_rank_lr100')):
        raise AssertionError('Synthetic summary/report integration failed')
    json.dumps(synthetic_summary,allow_nan=False)
    print('SELF-TEST PASSED: exact baseline at gamma=0, PSD shrinkage, train-only means, Stage27/28 selectors, four-fold synthetic integration and reporting')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--exploratory-only',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test()
    elif a.exploratory_only:
        run()
    else:
        p.error('Real source-only comparison requires --exploratory-only')


if __name__=='__main__':
    main()
