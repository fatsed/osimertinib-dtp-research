#!/usr/bin/env python3
"""Stage-47 auxiliary, PRE-SPECIFIED exploratory score-scale transport probe.

Question: can a TRAINING-only affine mapping of nested inner-LOCO margins
(2-line fit, 18 samples) to outer-model margins (3-line fit, 27 samples)
make tuned decision thresholds transfer more reliably to a held-out line?

This is NOT causal identification of training-size effects: the inner models
also see different biological training lines. All 4 source folds were already
inspected during method development; results are exploratory, not validation.

The full outer model's 27 source-training rows are scored IN SAMPLE only to
estimate an affine map, not to evaluate performance. The estimated map is
fixed before using held-out test scores. The procedure is intentionally a
single fixed candidate, not a hyperparameter search or an algorithmic novelty
claim. It needs no external GSE dataset and never re-fits Stage-38 predictions.

Usage (from repository root):
    python scripts/stage47_scale_transfer_probe.py --self-test
    python scripts/stage47_scale_transfer_probe.py --run

Writes distinct 47_scale_transfer_*.csv/.md under outputs/stage47_control_anchored;
never overwrites a differing existing artifact. Does not modify stages 1-47.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
OUT = ROOT / 'outputs' / 'stage47_control_anchored'
INPUT = OUT / '47_source_scores.csv'
PROVENANCE = OUT / '47_source_oof_provenance.json'


def import_script(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def affine_train_only(inner_oof: np.ndarray, outer_train_in_sample: np.ndarray):
    """Single fixed robust *distributional* mapping, without target input.

    s_outer ~ a + b*s_inner, using medians and IQRs on matched train IDs.
    In-sample outer scores have optimistic separation: the mapping is a
    diagnostic and may NOT transfer to held-out data.
    """
    x = np.asarray(inner_oof, float)
    y = np.asarray(outer_train_in_sample, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) != 27:
        raise ValueError('Expected 27 aligned TRAINING scores from each fitting regime')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid training scores')
    ix = float(np.quantile(x, .75) - np.quantile(x, .25))
    iy = float(np.quantile(y, .75) - np.quantile(y, .25))
    if ix <= 1e-12 or iy <= 1e-12:
        raise ValueError('Degenerate training score IQR; do not repair using test data')
    b = iy / ix
    a = float(np.median(y) - b * np.median(x))
    if not np.isfinite([a, b]).all() or b <= 0:
        raise ValueError('Invalid affine mapping')
    return a, b, ix, iy


def mapped_thresholds(raw_thr: float, anchored_thr: float, a: float, b: float):
    """Raw has shift and scale; target-referenced margins cancel shift."""
    return a + b * raw_thr, b * anchored_thr


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without the optional tabulate package."""
    f = frame.copy()
    for col in f:
        if pd.api.types.is_float_dtype(f[col]):
            f[col] = f[col].map(lambda x: f'{x:.4f}' if pd.notna(x) else 'NA')
    head = '| ' + ' | '.join(str(c) for c in f.columns) + ' |'
    rule = '| ' + ' | '.join('---' for _ in f.columns) + ' |'
    rows = ['| ' + ' | '.join(str(v) for v in row) + ' |'
            for row in f.itertuples(index=False, name=None)]
    return '\n'.join([head, rule] + rows)


def csv_text(rows):
    if not rows:
        raise ValueError('Empty output')
    frame = pd.DataFrame(rows)
    return frame.to_csv(index=False, lineterminator='\n')


def immutable_write_many(files: dict[Path, str]):
    for path, text in files.items():
        if path.exists() and path.read_bytes() != text.encode('utf-8'):
            raise FileExistsError(f'Existing Stage-47 output differs; refusing overwrite: {path}')
    OUT.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        if not path.exists():
            path.write_text(content, encoding='utf-8', newline='\n')


def self_test():
    rng = np.random.default_rng(47)
    # Non-degenerate scores; affine transport exactly reproduces known a,b.
    x = np.linspace(-2, 2, 27)
    b0, a0 = 1.7, -0.31
    a, b, q1, q2 = affine_train_only(x, a0 + b0 * x)
    assert np.isclose(a, a0) and np.isclose(b, b0)
    raw, anchored = mapped_thresholds(.4, .6, a, b)
    assert np.isclose(raw, a0+b0*.4)
    assert np.isclose(anchored, b0*.6)
    # Prediction transport identity with any test score; no test labels available.
    test = rng.normal(size=100)
    assert np.array_equal(test > .4, a0+b0*test > raw)
    ref, delta = -.1, rng.normal(size=100)
    assert np.array_equal(delta > .6, b0*delta > anchored)
    for broken in (np.zeros(27), np.full(27, np.nan)):
        try:
            affine_train_only(broken, x)
        except ValueError:
            pass
        else:
            raise AssertionError('Degenerate score mapping not rejected')
    print('SELF-TEST PASSED: affine transport, positivity, cutoff algebra, rejection of degenerate training scores. SYNTHETIC ONLY.')


def run():
    builder = import_script(SCRIPTS / 'stage47_build_source_oof.py', 'stage47_oof_for_scale_probe')
    stage47 = import_script(SCRIPTS / '47_control_anchored_calibration.py', 'stage47_evaluator_for_scale_probe')
    stage21 = builder.import_py(SCRIPTS / '21_group_consistent_rank_selection.py', 'stage21_scale_probe')
    stage34 = builder.import_py(SCRIPTS / '34_source_only_class_conditional_shrinkage.py', 'stage34_scale_probe')

    for path in (INPUT, PROVENANCE, builder.FROZEN):
        if not path.is_file():
            raise FileNotFoundError(path)
    prov = json.loads(PROVENANCE.read_text(encoding='utf-8'))
    if (prov.get('source_study') != 'GSE193258' or prov.get('positive_label') != 'DTP'
        or prov.get('inner_oof_model_fit_lines') != 2 or prov.get('outer_model_fit_lines') != 3
        or not prov.get('outer_test_replay_required_and_passed', False)):
        raise ValueError('Unexpected source provenance or inner/outer model design')
    if sha256(builder.FROZEN) != prov.get('frozen_38_sha256'):
        raise ValueError('Stage38 frozen output has changed')
    if sha256(SCRIPTS / 'stage47_build_source_oof.py') != prov.get('builder_sha256'):
        raise ValueError('Stage47 builder changed since scores export; regenerate and re-audit')
    for name, old_sha in prov.get('upstream_script_sha256', {}).items():
        if sha256(SCRIPTS / name) != old_sha:
            raise ValueError(f'Upstream script changed since source score generation: {name}')

    df = stage47.load_and_validate(INPUT, positive='DTP', expected_folds=4)
    metrics_old, _, _, _ = stage47.evaluate(df, 'DTP')
    genes, ids, X, y, groups, _ = stage21.load_data()
    ids = np.asarray(ids, str)
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups, str)
    if len(genes) != 19712 or X.shape != (36, 19712) or len(set(ids)) != 36:
        raise ValueError('Source cohort unexpected')
    if hashlib.sha256(X.tobytes()).hexdigest() != prov.get('source_X_float64_sha256'):
        raise ValueError('Source expression fingerprint changed')
    if set(groups) != set(df.fold.unique()):
        raise ValueError('Source cell lines mismatch')

    diagnostics = []
    computed = []
    decisions = []
    for fold in sorted(df.fold.unique()):
        ti = np.flatnonzero(groups != fold)
        te = np.flatnonzero(groups == fold)
        if len(ti) != 27 or len(te) != 9:
            raise AssertionError('Broken source LOCO split')
        # EXACT original Stage38 recipe. Score the outer model's own TRAIN
        # samples only for scale estimation; outer TEST scores are verified
        # against frozen Stage38 and then evaluated without any tuning.
        fit_scores, _ = builder.fit_dual_scores(
            X[ti], y[ti], groups[ti], np.concatenate((X[ti], X[te]), axis=0),
            stage34, 3,
        )
        for variant in ('base', 'protected'):
            block = stage47.add_reference_scores(
                df[(df.fold == fold) & (df.variant == variant)].copy()
            )
            actual = block[block.split == 'test'].set_index('sample_id')
            # Frozen replay verifies we did not use a changed model recipe.
            for j, idx in enumerate(te):
                sample = str(ids[idx])
                saved = float(actual.loc[sample, 'score'])
                if abs(float(fit_scores[variant][len(ti) + j]) - saved) > 2e-6:
                    raise AssertionError(f'Stage38 replay mismatch: {fold}/{variant}/{sample}')

            inner = block[block.split == 'train_oof'].set_index('sample_id')
            wanted_ids = [str(ids[idx]) for idx in ti]
            if set(inner.index) != set(wanted_ids) or len(inner) != 27:
                raise ValueError('Nested OOF scores are not aligned with outer train samples')
            oof = inner.loc[wanted_ids, 'score'].to_numpy(float)
            in_sample = np.asarray(fit_scores[variant][:27], float)
            a, b, iqr_oof, iqr_outer = affine_train_only(oof, in_sample)

            train = block[(block.split == 'train_oof') & (~block.is_reference)]
            y_train = train.group.eq('DTP').to_numpy(bool)
            raw_thr, _ = stage47.select_threshold(train.score.to_numpy(float), y_train)
            anchor_thr, _ = stage47.select_threshold(train.relative_score.to_numpy(float), y_train)
            mapped_raw, mapped_anchor = mapped_thresholds(raw_thr, anchor_thr, a, b)
            first_reference = block[(block.split == 'test') & (block.is_reference)]
            if len(first_reference) != 1:
                raise ValueError('Expected exactly one reference Control in test fold')
            ref_score = float(first_reference.iloc[0]['score'])
            diagnostics.append({
                'fold': fold, 'variant': variant, 'train_n': 27,
                'inner_fit_n': 18, 'outer_fit_n': 27,
                'inner_oof_iqr': iqr_oof, 'outer_train_insample_iqr': iqr_outer,
                'estimated_scale_b': b, 'estimated_shift_a': a,
                'raw_threshold_oof': raw_thr, 'raw_threshold_mapped': mapped_raw,
                'anchored_threshold_oof': anchor_thr,
                'anchored_threshold_mapped': mapped_anchor,
                'heldout_reference_score': ref_score,
                'anchored_equiv_raw_cut_original': ref_score + anchor_thr,
                'anchored_equiv_raw_cut_mapped': ref_score + mapped_anchor,
            })

            paired_test = block[(block.split == 'test') & (~block.is_reference)].copy()
            candidate = (
                ('raw_scale_transport', 'score', mapped_raw),
                ('anchored_scale_transport', 'relative_score', mapped_anchor),
            )
            for method, col, thr in candidate:
                frame = paired_test.copy()
                frame['decision_score'] = frame[col]
                frame['pred_positive'] = frame[col].to_numpy(float) > thr
                mm = stage47._metrics(frame, 'DTP')
                computed.append({'fold':fold,'variant':variant,'method':method,
                                 'threshold':float(thr), **mm})
                for row in frame.itertuples():
                    decisions.append({'fold':fold,'variant':variant,'method':method,
                                      'sample_id':str(row.sample_id),
                                      'cell_line':str(row.cell_line), 'group':str(row.group),
                                      'score':float(row.score),
                                      'reference_score':float(row.reference_score),
                                      'decision_score':float(row.decision_score),
                                      'threshold':float(thr),
                                      'pred_positive':bool(row.pred_positive)})
    new_metrics = pd.concat([metrics_old, pd.DataFrame(computed)], ignore_index=True)
    expected_methods = set(stage47.METHODS) | {'raw_scale_transport', 'anchored_scale_transport'}
    assert set(new_metrics.method) == expected_methods and len(new_metrics) == 4*2*6
    if not (new_metrics.groupby(['fold','variant']).size() == 6).all():
        raise AssertionError('Incomplete comparison')
    if (new_metrics.groupby(['fold','variant']).roc_auc.agg(lambda s:s.max()-s.min())>1e-9).any():
        raise AssertionError('ROC-AUC should be invariant to affine monotonic score calibration')

    methods = ['raw_zero', 'raw_tuned', 'anchored_tuned', 'raw_scale_transport', 'anchored_scale_transport']
    macro = (new_metrics[new_metrics.method.isin(methods)].groupby(['variant','method'])[
        ['sensitivity','specificity','balanced_accuracy','roc_auc','tp','fn','control_fp','acute_fp']
    ].mean().reset_index())
    sep = new_metrics[new_metrics.method.isin(methods)].copy()
    pivot = sep.pivot_table(index=['fold','variant'],columns='method',values='balanced_accuracy')
    for v in ['base','protected']:
        z=pivot.xs(v,level='variant')
        print(f'\n{v} — held-out paired BA by fold:')
        print(z.round(3).to_string())
    print('\nSource-only macro-averaged BA:')
    print(macro[['variant','method','balanced_accuracy','sensitivity','specificity']].round(3).to_string(index=False))

    lines = [
        '# Stage 47 — TRAIN-only score-scale transport diagnostic', '',
        '**STATUS: POST-HOC exploratory source-only probe; NOT independent validation or proof of novelty.**',
        '', 'One fixed candidate: estimate a positive affine mapping between nested '
        'inner-LOCO train-OOF margins and the full outer model’s in-sample '
        'margins on the SAME 27 source-training samples.',
        '', 'Scale `b = IQR(outer-training in-sample scores) / IQR(inner-training OOF scores)`;',
        '`a = median(outer-training scores) - b*median(inner-OOF scores)`.',
        '', 'Raw tuned cutoff maps to `a+b*t_raw`; reference-anchored tuned '
        'cutoff maps to `b*t_anchored` (the translation cancels).',
        '', 'Source-only held-out lines are never used in threshold fitting or scale estimation.',
        'Outer held-out predictions are replay-checked against the frozen Stage38 file.',
        '', '## Macro metrics (averages over the 4 source cell lines)', '',
        markdown_table(macro),
        '', '## Per-fold matched evaluation', '',
        markdown_table(sep[['fold','variant','method','tp','fn','control_fp','acute_fp','balanced_accuracy']]),
        '', '## Important limitations', '',
        '- The affine mapping is derived using IN-SAMPLE outer-model training scores; '
        'this may substantially overestimate separation and is NOT a valid estimate '
        'of out-of-sample scale stability.',
        '- Inner fitting changes BOTH training sample count (18 versus 27) and '
        'which cell lines are used. This experiment cannot isolate sample-size effects.',
        '- All four source folds and reference choices have been inspected repeatedly.',
        '- n=3 DTP per line; no meaningful confirmatory statistical evidence.',
        '- GSE249721 and GSE255958 are not used here; no independent external validation.',
        '- If transport improves BA post-hoc, this alone does not establish an '
        'algorithmic novelty claim; if it fails, stop this particular calibration route.',
        '- Raw-zero must remain a key competitor; a stronger comparator cannot be omitted.',
        '', 'Artifacts are immutable and separately named; no previous scripts or outputs were modified.',
    ]
    files = {
        OUT/'47_scale_transfer_diagnostics.csv':csv_text(diagnostics),
        OUT/'47_scale_transfer_metrics.csv':new_metrics.to_csv(index=False,lineterminator='\n'),
        OUT/'47_scale_transfer_predictions.csv':csv_text(decisions),
        OUT/'47_scale_transfer_report.md':'\n'.join(lines)+'\n',
    }
    # Avoid requiring external Markdown formatting dependency `tabulate` in
    # environments that only have the scientific requirements installed.
    immutable_write_many(files)
    print('\nSCALE PROBE COMPLETED. This is exploratory only.')
    for p in files: print('Saved:',p)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-test',action='store_true')
    g.add_argument('--run',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    else:run()


if __name__ == '__main__':
    main()
