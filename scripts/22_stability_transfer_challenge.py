"""Stage 22 | One fixed-method stability and PC9 transfer challenge.

Uses the existing Stage-21 ANOVA and group-consistent selectors as-is.
No tuning, model selection or new algorithm changes. Source GSE193258
(36 samples) is used only within LOCO training folds. External GSE255958
(9 previously inspected PC9 samples: Control/Acute/DTP) is an EXPLORATORY
stress test, NOT independent validation. Resistant is excluded.

Run at repository root, after Stages 17 and 21:
  python scripts/22_stability_transfer_challenge.py --self-test
  python scripts/22_stability_transfer_challenge.py --exploratory-only

Outputs: outputs/group_consistent_rank/22_comparison.md,
22_predictions.csv, 22_summary.json. Existing artifacts are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
OUT = ROOT / 'outputs/group_consistent_rank'
STATES = ('Control', 'Acute', 'DTP')
METHODS = ('anova', 'group_consistent')


def load_module(path, label):
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load module: {path}')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fit_source_model(Xtrain, ytrain, selected):
    features = rankdata(Xtrain[:, selected], axis=1, method='average') / len(selected)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.01, solver='lbfgs', max_iter=3000, random_state=42))
    model.fit(features, ytrain)
    return model, features.mean(axis=0)


def project_external(targets, target_ids, selected_names, source_means, all_source_genes):
    """Stage17's within-selected-panel ranks and TRAIN-mean missing imputation.

    No labels, target-set statistics, or target-based feature selection.
    Returns per-sample coverage to expose differing gene universes.
    """
    rows = np.broadcast_to(source_means, (len(target_ids),len(selected_names))).copy()
    cover = []
    for i, gsm in enumerate(target_ids):
        data = targets[gsm]
        if len(data) < 1000:
            raise ValueError(f'Inadequate external gene universe: {gsm}')
        present = [j for j, gene in enumerate(selected_names) if gene in data]
        if len(present) < 90:
            raise ValueError(f'Selected panel coverage <90/100 for {gsm}')
        n_overlap = len(all_source_genes.intersection(data))
        if n_overlap < min(14000, int(np.ceil(.70*len(all_source_genes)))):
            raise ValueError(f'Insufficient source/target gene overlap: {gsm}')
        values = np.array([data[selected_names[j]] for j in present], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f'Invalid external TPM values: {gsm}')
        rows[i, present] = rankdata(values, method='average') / len(present)
        cover.append({'gsm':gsm,'selected_available':len(present),'source_genes_available':n_overlap})
    return rows, cover


def jaccard(a,b):
    if not a and not b:
        raise ValueError('Empty panels')
    return len(a & b) / len(a | b)


def check_frozen_predictions(records, method, scope, ids, names, predicted, key_fields,
                             label_field='predicted_state'):
    expected = {(gsm,) if len(key_fields)==1 else (names[i],gsm) for i,gsm in enumerate(ids)}
    lookup = {}
    for row in records:
        if row.get('method')!=method:
            continue
        if scope is not None and row.get('scope')!=scope:
            continue
        key = tuple(row.get(k) for k in key_fields)
        # The Stage21 file also contains other held-out cell lines.
        if key not in expected:
            continue
        if key in lookup:
            raise ValueError(f'Duplicate frozen prediction {key}')
        lookup[key] = row
    if set(lookup)!=expected:
        raise ValueError(f'Frozen prediction identities mismatch ({method}/{scope})')
    for i,gsm in enumerate(ids):
        key = (gsm,) if len(key_fields)==1 else (names[i],gsm)
        if lookup[key][label_field] != STATES[int(predicted[i])]:
            raise ValueError(f'Frozen prediction changed ({method}/{scope}/{gsm})')


def self_test():
    rng=np.random.default_rng(222)
    X=rng.normal(2,.5,size=(27,700))
    y=np.repeat([0,1,2],9)
    X[y==1,30:50]+=1
    X[y==2,50:70]+=1
    selected=np.arange(100)
    model, mean=fit_source_model(X,y,selected)
    direct=rankdata(X[:3][:,selected],axis=1,method='average')/100
    assert np.array_equal(model.predict(direct),model.predict(direct.copy()))
    genes=[f'G{i}' for i in range(700)]
    targets={'T0':dict(zip(genes,X[0]-X[0].min()+.1)),
             'T1':dict(zip(genes,X[1]-X[1].min()+.1))}
    sel_names=genes[:100]
    # Minimal fixture has 700 genes, not the 14k genes required in real data.
    # Independently exercise the exact panel rank formula.
    available=np.array([targets['T0'][g] for g in sel_names])
    assert np.array_equal(rankdata(available)/100,
                          rankdata(np.array(list(targets['T0'].values()))[:100])/100)
    # Projection checks: one missing gene uses the source training mean.
    large={f'G{i}':float((i*7)%31)/10 for i in range(16000) if i != 0}
    z, coverage=project_external({'T':large},['T'],[f'G{i}' for i in range(100)],
                                 mean,set(f'G{i}' for i in range(16000)))
    assert coverage[0]['selected_available']==99
    assert z[0,0]==mean[0]
    assert np.allclose(z[0,1:],rankdata([large[f'G{i}'] for i in range(1,100)])/99)
    assert jaccard({1,2,3},{2,3,4})==.5
    # Extra frozen rows from other folds should not cause false mismatches.
    check_frozen_predictions([
         {'method':'anova','gsm':'T0','held_out_line':'PC9','predicted_state':'Control'},
         {'method':'anova','gsm':'OTHER','held_out_line':'H1975','predicted_state':'DTP'}],
         'anova',None,['T0'],['PC9'],np.array([0]),['held_out_line','gsm'])
    try:
        check_frozen_predictions([{'method':'anova','gsm':'T0','predicted_state':'Control'}],
             'anova',None,['T0'],[''],np.array([1]),['gsm'])
    except ValueError: pass
    else: raise AssertionError('Frozen prediction mislabel went undetected')
    assert mean.shape==(100,)
    print('SELF-TEST PASSED: fit/panel ranking, Jaccard, frozen-prediction mismatch guard')


def run():
    s21 = load_module(SCRIPTS/'21_group_consistent_rank_selection.py','stage21_for22')
    s17 = load_module(SCRIPTS/'17_three_class_external_transfer.py','stage17_for22')
    prior21 = read_json(OUT/'21_summary.json')
    prior17 = read_json(ROOT/'outputs/three_class_external/17_pc9_excluded_summary.json')
    prior17_rows = load_csv(ROOT/'outputs/three_class_external/17_pc9_excluded_predictions.csv')
    prior21_rows = load_csv(OUT/'21_predictions.csv')
    if ('EXPLORATORY' not in prior21['status'] or 'EXPLORATORY' not in prior17['status'] or
        not prior17.get('source_pc9_excluded')):
        raise ValueError('Prior reports are not the expected exploratory PC9-excluded design')
    genes, ids, X, y, groups, provenance = s21.load_data()
    if len(genes)!=19712 or len(ids)!=36 or X.shape!=(36,19712):
        raise ValueError('Unexpected frozen source matrix')
    if list(s21.LINES)!=['H1975','HCC2935','HCC827','PC9']:
        raise ValueError('Source cell-line definitions changed')
    if set(groups)!=set(s21.LINES):
        raise ValueError('Source lines changed')
    if not s17.TARGET.is_file():
        raise FileNotFoundError(f'Missing prior target archive: {s17.TARGET}')
    if sha256_file(s17.TARGET)!=prior17['target_tar_sha256']:
        raise ValueError('External archive does not match audited Stage17 input')
    archive=s17.read_target_tar(s17.TARGET)
    external_ids=[gsm for state in STATES for gsm in s17.GROUPS[state]]
    if len(external_ids)!=9 or len(set(external_ids))!=9:
        raise ValueError('External GSM set changed')
    targets={gsm:archive[gsm]['data'] for gsm in external_ids}
    external_truth=np.array([STATES.index(state) for state in STATES for _ in s17.GROUPS[state]])
    source_gene_set=set(genes)
    predictions=[]; metrics={}; coverage={}; panels={}; jaccards={}
    for line in s21.LINES:
        tr=np.flatnonzero(groups!=line)
        if len(tr)!=27 or Counter(y[tr])!={0:9,1:9,2:9}:
            raise ValueError('Invalid training fold')
        for method in METHODS:
            chosen=s21.select_panel(X[tr], y[tr], groups[tr],mode=method)
            panel=set(int(x) for x in chosen)
            if len(panel)!=100:
                raise ValueError('Incorrect panel size')
            fingerprint=hashlib.sha256('\n'.join(genes[i] for i in sorted(panel)).encode()).hexdigest()
            if prior21['source_gene_panel_sha256'][line][method] != fingerprint:
                raise ValueError(f'Stage21 panel changed: {line}/{method}')
            panels[(line,method)]=panel
    for method in METHODS:
        values=[jaccard(panels[(a,method)],panels[(b,method)])
                for a,b in combinations(s21.LINES,2)]
        jaccards[method]={'mean':float(np.mean(values)), 'min':float(min(values)),
                          'max':float(max(values)), 'pairwise':values}
    tr=np.flatnonzero(groups!='PC9'); te=np.flatnonzero(groups=='PC9')
    if len(tr)!=27 or len(te)!=9 or Counter(y[te])!={0:3,1:3,2:3}:
        raise ValueError('PC9 split invalid')
    for method in METHODS:
        # Preserve selector score-order to reproduce frozen predictions exactly.
        chosen=s21.select_panel(X[tr],y[tr],groups[tr],mode=method)
        model,train_mean=fit_source_model(X[tr],y[tr],chosen)
        source_z=rankdata(X[te][:,chosen],axis=1,method='average')/100
        source_pred=model.predict(source_z)
        source_ids=[ids[i] for i in te]
        check_frozen_predictions(prior21_rows,method,None,source_ids,
             ['PC9']*len(te),source_pred,['held_out_line','gsm'])
        external_z,cover=project_external(targets,external_ids,
            [genes[int(i)] for i in chosen],train_mean,source_gene_set)
        external_pred=model.predict(external_z)
        if method=='anova':
            check_frozen_predictions(prior17_rows,'rank_selected_lr100_3class',
                'external_GSE255958_PC9',external_ids,['']*9,external_pred,['gsm'])
        coverage[method]=cover
        metrics[method]={'source_PC9':s21.count_metrics(y[te],source_pred),
                         'external_PC9':s21.count_metrics(external_truth,external_pred)}
        for scope,gsms,truth,pred in (
            ('source_PC9',source_ids,y[te],source_pred),
            ('external_GSE255958_PC9',external_ids,external_truth,external_pred)):
            for gsm,label,p in zip(gsms,truth,pred):
                predictions.append({'scope':scope,'method':method,'gsm':gsm,
                  'true_state':STATES[int(label)],'predicted_state':STATES[int(p)],
                  'correct':int(label==p)})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'22_predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    result={'status':'EXPLORATORY_HISTORICALLY_INSPECTED_TARGET_NOT_INDEPENDENT',
        'experiment':'fixed Stage21 methods, source-PC9-excluded transfer to previously inspected GSE255958 PC9',
        'source_study':'GSE193258','target_study':'GSE255958',
        'training_n':27,'source_pc9_n':9,'external_n':9,'states':list(STATES),
        'source_provenance':provenance,'target_tar_sha256':sha256_file(s17.TARGET),
        'source_only_select_and_fit':True,'target_data_for_fitting_or_feature_selection':False,
        'source_PC9_predictions_reproduced_from_Stage21':True,
        'external_ANOVA_predictions_reproduced_from_Stage17':True,
        'feature_set_stability_pairwise_Jaccard':jaccards,
        'performance':metrics,'selected_feature_coverage':coverage,
        'limitations':['Target study inspected before method proposal, so transfer result is descriptive only.',
          '3 external replicates per class from one PC9 study; not a replicate of independent studies.',
          'Treatment duration is confounded with states; processing differs across studies.',
          'Feature stability across four overlapping source folds cannot prove population-level stability.',
          'GCR-FS novelty, significance and superiority are not demonstrated.']}
    (OUT/'22_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    md=['# Stage 22 — Group-consistent rank stability and PC9 transfer (exploratory)','',
        '**Previously inspected external data; NOT independent validation.**','',
        '| Method | Source PC9 correct / 9 | External PC9 correct / 9 | External bACC | Across-fold gene-panel mean Jaccard |',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        a=metrics[method]['source_PC9'];b=metrics[method]['external_PC9']
        md.append(f"| {method} | {a['correct']}/9 | {b['correct']}/9 | {b['balanced_accuracy']:.3f} | {jaccards[method]['mean']:.3f} |")
    md += ['', '| Method | Control recall | Acute recall | DTP recall | Macro-F1 | Min selected genes available |',
           '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        m=metrics[method]['external_PC9']
        n=min(c['selected_available'] for c in coverage[method])
        md.append(f"| {method} | {m['recall_control']:.3f} | {m['recall_acute']:.3f} | {m['recall_dtp']:.3f} | {m['macro_f1']:.3f} | {n}/100 |")
    md += ['', 'The Stage21 source PC9 predictions and Stage17 external ANOVA predictions were reproduced by GSM.',
       'Same three-class model family, 100 genes, source-only training, and 27 training observations.',
       'External target labels were used solely for evaluation. Study and treatment confounders remain.',
       'This is a development-stage, retrospectively inspected comparison: do not claim independent improvement or proven novel method.']
    (OUT/'22_comparison.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('Stage22 PASSED: Stage21 source-PC9 and Stage17 external ANOVA predictions reproduced')
    for method in METHODS:
        a=metrics[method]['source_PC9'];b=metrics[method]['external_PC9']
        print(f"{method}: source PC9 {a['correct']}/9; external PC9 {b['correct']}/9 "
              f"bACC={b['balanced_accuracy']:.3f}; panel Jaccard={jaccards[method]['mean']:.3f}")
    print('Saved outputs/group_consistent_rank/22_{comparison.md,summary.json,predictions.csv}')
    print('EXPLORATORY: no independence, no method tuning or biology certification')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--exploratory-only', action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test()
    elif args.exploratory_only:
        run()
    else:
        p.error('Real-data experiment requires --exploratory-only')


if __name__=='__main__':
    main()
