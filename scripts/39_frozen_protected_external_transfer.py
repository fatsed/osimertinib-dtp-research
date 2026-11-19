#!/usr/bin/env python3
"""Stage 39 | Historical cross-study transfer of the FROZEN Stage35 protected model.

Purpose: complete technical external-scoring coverage, NOT validate/choose/tune the
algorithm. GSE255958 was already inspected in stages 17/31/33. The two selectors,
model coefficients, thresholds and missingness policy come exclusively from the
27 source training samples (H1975/HCC2935/HCC827). PC9 source and GSE255958
samples are test-only. Stages 31 and 35 must reproduce EXACTLY before scoring.

Run from the repo root:
    python scripts/39_frozen_protected_external_transfer.py --self-test
    python scripts/39_frozen_protected_external_transfer.py --exploratory-only

Original historical output files are never modified. New immutable files:
    outputs/cross_study_contrast/39_report.md
    outputs/cross_study_contrast/39_summary.json
    outputs/cross_study_contrast/39_predictions.csv

NOT an independent external validation; NOT a novel-algorithm claim.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
OUT = ROOT / 'outputs' / 'cross_study_contrast'
SRC = ROOT / 'outputs' / 'dtp_specificity'
DI = ROOT / 'outputs' / 'domain_invariance'
STATES = ('Control','Acute','DTP')
METHODS = ('acute_only_rank_lr100','dual_contrast_rank_lr100')
VARIANTS = ('stage28_baseline','contrast_protected_shrinkage_gamma1')
PREVIOUS_SHA = {
    'scripts/35_contrast_protected_domain_shrinkage.py': '52d330f9fc26ee15c3fb08751ca509653a2e42b567ba16dadd1009b16953642b',
    'scripts/31_gse255958_frozen_contrast_transfer.py': '4c0ba38547f14aa843edfd1c51f63903eb213c993e0f359b9438a2c378e02460',
    'outputs/domain_invariance/35_predictions.csv': '64e753b1c301ca06ebe991dd1f10badbc59537be705296d9bf4834e3077b229d',
    'outputs/domain_invariance/35_summary.json': 'd69a78e9ea17c5eda5209ae7c4056aa9ac95c7908f29eb28fd36d167ee62bd10',
    'outputs/cross_study_contrast/31_predictions.csv': 'f2a27e0fcb479d2cd5b11a26cf46fc05e72901faf53790a6763510f6530948fa',
    'outputs/cross_study_contrast/31_summary.json': 'f6a7f2ef4b51cc563ad845ca31958724c78bad454b33e01d77971452542d12f9',
}


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):
            h.update(c)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def verify_prior_files():
    """Fail closed on any mutation of the experimental code and frozen outputs."""
    for p,expected in PREVIOUS_SHA.items():
        full=ROOT/p
        if not full.is_file():
            raise FileNotFoundError(f'Required frozen file is missing: {full}')
        got=sha(full)
        if got!=expected:
            raise ValueError(f'Frozen input changed: {p}\nexpected {expected}\nobserved {got}. Do not overwrite historical runs.')
    return dict(PREVIOUS_SHA)


def import_source_scripts(required=None):
    import importlib.util
    def mod(fname,label):
        path=SCRIPTS/fname
        if not path.is_file(): raise FileNotFoundError(path)
        spec=importlib.util.spec_from_file_location(label,path)
        if spec is None or spec.loader is None: raise RuntimeError(f'Cannot import {path}')
        obj=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        return obj
    spec={
        's17':('17_three_class_external_transfer.py','s17_for39'),
        's21':('21_group_consistent_rank_selection.py','s21_for39'),
        's27':('27_dtp_specific_contrast_benchmark.py','s27_for39'),
        's28':('28_dual_contrast_ablation.py','s28_for39'),
        's31':('31_gse255958_frozen_contrast_transfer.py','s31_for39'),
        's34':('34_source_only_class_conditional_shrinkage.py','s34_for39'),
        's35':('35_contrast_protected_domain_shrinkage.py','s35_for39'),
    }
    if required is None: required=tuple(spec)
    return {k:mod(*spec[k]) for k in required}


def source_models(mods, frozen_panels, baseline_models, baseline_source_rows):
    """Source-fit protected model; replay all nine source-PC9 Stage35 margins."""
    from scipy.stats import rankdata
    s21,s27,s28,s34,s35=(mods[k] for k in ('s21','s27','s28','s34','s35'))
    genes,ids,X,y,groups,provenance=s21.load_data()
    y=np.asarray(y,dtype=int)
    groups=np.asarray(groups)
    if (X.shape!=(36,19712) or len(set(genes))!=19712 or len(set(ids))!=36 or
        Counter(y.tolist())!={0:12,1:12,2:12} or set(groups)!=set(s35.LINES)):
        raise ValueError('Source cohort differs from Stage35')
    tr=np.flatnonzero(groups!='PC9')
    te=np.flatnonzero(groups=='PC9')
    if (len(tr)!=27 or len(te)!=9 or
        Counter(y[tr].tolist())!={0:9,1:9,2:9} or
        Counter(y[te].tolist())!={0:3,1:3,2:3}):
        raise ValueError('Unexpected source PC9 split')
    s27.validate_train(y[tr],groups[tr])
    if not np.isclose(s35.GAMMA,1.0) or s35.K!=100:
        raise ValueError('Stage35 settings not frozen')
    prev=read_csv(DI/'35_predictions.csv')
    reference={}
    for r in prev:
        if r['held_out_line']=='PC9' and r['method'] in METHODS and r['variant'] in VARIANTS:
            key=(r['method'],r['variant'],r['gsm'])
            if key in reference: raise ValueError(f'Duplicate Stage35 PC9 reference: {key}')
            reference[key]=r
    if len(reference)!=len(METHODS)*len(VARIANTS)*9:
        raise ValueError('Stage35 PC9 reference must contain 36 records')
    source_baseline={(r['method'],r['source_gsm']):r for r in baseline_source_rows}
    if len(source_baseline)!=18: raise ValueError('Incomplete Stage31 source PC9 replay')
    fitted={}; source_rows=[]
    for method in METHODS:
        if method=='acute_only_rank_lr100':
            index,_=s28.select_single_contrast(X[tr],y[tr],groups[tr],negative=1)
        else:
            index,_=s27.select_rank_panel(X[tr],y[tr],groups[tr],mode='dual_contrast')
        ordered=[genes[int(i)] for i in index]
        if ordered!=frozen_panels[method]:
            raise ValueError(f'Stage31 frozen ordered feature panel mismatch: {method}')
        if len(index)!=100 or len(set(index.tolist()))!=100:
            raise ValueError('Invalid frozen selected genes')
        ztr=s34.within_panel_rank(X[tr],index)
        zte=s34.within_panel_rank(X[te],index)
        # Stage31's exact baseline coefficients, feature rank and score replay.
        base=np.asarray(baseline_models[method]['classifier'].decision_function(zte),float)
        scaler=StandardScaler().fit(ztr)
        standardized_tr=scaler.transform(ztr)
        standardized_te=scaler.transform(zte)
        P,diagnostic=s35.protected_shrinkage(standardized_tr,y[tr],groups[tr],gamma=1.0)
        clf=s34.lr_model().fit(standardized_tr@P,(y[tr]==2).astype(int))
        prot=np.asarray(clf.decision_function(standardized_te@P),float)
        if not np.isfinite(prot).all() or len(prot)!=9:
            raise ValueError('Invalid protected PC9 source score')
        for j,i in enumerate(te):
            gsm=ids[int(i)]
            for variant,score in zip(VARIANTS,(base[j],prot[j])):
                key=(method,variant,gsm)
                ref=reference.get(key)
                v=float(score)
                if (ref is None or ref['true_state']!=STATES[int(y[i])] or
                    int(ref['predicted_dtp'])!=int(v>0) or
                    not np.isclose(v,float(ref['margin']),atol=1e-8,rtol=1e-7)):
                    raise ValueError(f'Frozen Stage35 PC9 score mismatch: {key}')
                if variant=='stage28_baseline':
                    earlier=source_baseline.get((method,gsm))
                    if earlier is None or earlier['true_state']!=STATES[int(y[i])] or \
                       not np.isclose(v,float(earlier['source_margin']),rtol=1e-7,atol=1e-8):
                        raise ValueError(f'Frozen Stage31 source-PC9 score mismatch: {method}/{gsm}')
                source_rows.append({'method':method,'variant':variant,'gsm':gsm,
                                    'true_state':STATES[int(y[i])],'margin':v})
        fitted[method]={
            'baseline':baseline_models[method],
            'protected':{'scaler':scaler,'P':P,'clf':clf,'panel':ordered,
                         'train_feature_mean':ztr.mean(axis=0)},
            'diagnostic':diagnostic,
        }
    return fitted,source_rows,provenance


def target_archive(mods, s31):
    """Verify study provenance and sample identities, exclude Resistant; then read RNA-seq."""
    s17=mods['s17']
    st31=json.loads((OUT/'31_summary.json').read_text(encoding='utf-8'))
    st17=json.loads(s31.STAGE17_SUMMARY.read_text(encoding='utf-8'))
    if (st31.get('source')!='GSE193258' or st31.get('target')!='GSE255958' or
        st31.get('source_training_lines')!=list(s31.SOURCE_LINES) or
        st31.get('feature_k')!=100 or st31.get('C')!=.01 or st31.get('threshold')!=0):
        raise ValueError('Stage31 source/external protocol differs')
    if (not st17.get('source_pc9_excluded') or
        sha(s17.TARGET)!=st17.get('target_tar_sha256') or
        sha(s17.TARGET)!=st31['input_sha256']['target_archive']):
        raise ValueError('Audited external archive changed')
    gsms=[g for state in STATES for g in s17.GROUPS[state]]
    truth={g:state for state in STATES for g in s17.GROUPS[state]}
    resistant=set(s17.GROUPS['Resistant'])
    if len(gsms)!=9 or len(set(gsms))!=9 or len(resistant)!=3 or set(gsms)&resistant:
        raise ValueError('Expected 9 study samples and 3 excluded Resistant samples')
    s31.validate_prior_target(gsms,truth)
    archived=s17.read_target_tar(s17.TARGET)
    if set(archived)!=set(gsms)|resistant:
        raise ValueError('GSE255958 archive sample identities do not match prior analysis')
    data={gsm:archived[gsm]['data'] for gsm in gsms}
    if any(len(v)<15000 for v in data.values()):
        raise ValueError('Insufficient RSEM genes in an external sample')
    return gsms,truth,data


def score_external(fitted, gsms,truth,targets,source_rows, s31):
    """Target expression is used ONLY in fixed feature transform + prediction."""
    previous=read_csv(OUT/'31_predictions.csv')
    prev={}
    for r in previous:
        if r['method'] in METHODS:
            key=(r['method'],r['gsm'])
            if key in prev:raise ValueError('Duplicated Stage31 external prediction')
            prev[key]=r
    if len(prev)!=18:raise ValueError('Stage31 external baseline incomplete')
    records=[];metrics={};coverage={}
    labels=[truth[g] for g in gsms]
    for method in METHODS:
        f=fitted[method]
        baseline=f['baseline']
        protected=f['protected']
        if baseline['panel']!=protected['panel']:
            raise ValueError('Source gene panels differ between variants')
        # Exact same Stage31 missing-gene handling for both model variants.
        rawZ,cov=s31.target_rank_features(targets,gsms,baseline)
        otherZ,otherCov=s31.target_rank_features(targets,gsms,protected)
        if not np.array_equal(rawZ,otherZ) or cov!=otherCov:
            raise ValueError('Baseline and protected methods did not receive the identical target representation')
        assert rawZ.shape==(9,100)
        cover=[int(x['available']) for x in cov]
        coverage[method]={'min_available':min(cover),'max_available':max(cover),
                          'source_rank_imputation':min(cover)<100}
        baseScores=np.asarray(baseline['classifier'].decision_function(rawZ),float)
        protScores=np.asarray(protected['clf'].decision_function(
                           protected['scaler'].transform(rawZ)@protected['P']),float)
        if not np.isfinite(protScores).all(): raise ValueError('Non-finite protected external scores')
        for j,gsm in enumerate(gsms):
            old=prev.get((method,gsm))
            if old is None or old['true_state']!=truth[gsm] or \
               not np.isclose(baseScores[j],float(old['source_only_margin']),atol=1e-8,rtol=1e-7) or \
               int(baseScores[j]>0)!=int(old['predicted_dtp']) or \
               int(cov[j]['available'])!=int(old['panel_genes_available']):
                raise ValueError(f'Frozen Stage31 target prediction mismatch: {method}/{gsm}')
        for variant,scores in zip(VARIANTS,(baseScores,protScores)):
            metrics.setdefault(method,{})[variant]=s31.target_metrics(labels,scores)
            for j,gsm in enumerate(gsms):
                records.append({'method':method,'variant':variant,'gsm':gsm,
                    'true_state':truth[gsm], 'score':f'{float(scores[j]):.12g}',
                    'predicted_dtp':int(scores[j]>0),
                    'correct':int((scores[j]>0)==(truth[gsm]=='DTP')),
                    'genes_available':int(cov[j]['available']),
                    'genes_source_mean_imputed':int(cov[j]['missing'])})
    if len(records)!=36: raise AssertionError('Expected 2 methods x 2 variants x 9 external predictions')
    paired={}
    for method in METHODS:
        base={r['gsm']:r for r in records if r['method']==method and r['variant']==VARIANTS[0]}
        prot={r['gsm']:r for r in records if r['method']==method and r['variant']==VARIANTS[1]}
        if set(base)!=set(gsms) or set(prot)!=set(gsms): raise AssertionError('Incorrect paired identity coverage')
        paired[method]={
            'protected_only_correct':sum(int(prot[g]['correct'])==1 and int(base[g]['correct'])==0 for g in gsms),
            'baseline_only_correct':sum(int(prot[g]['correct'])==0 and int(base[g]['correct'])==1 for g in gsms),
            'both_correct':sum(int(prot[g]['correct'])==1 and int(base[g]['correct'])==1 for g in gsms),
            'both_wrong':sum(int(prot[g]['correct'])==0 and int(base[g]['correct'])==0 for g in gsms),
        }
        if sum(paired[method].values())!=9: raise AssertionError('Paired results invalid')
    shifts={}
    for method in METHODS:
        shifts[method]={}
        for variant in VARIANTS:
            shift={}
            for state in STATES:
                ss=[r['margin'] for r in source_rows if r['method']==method and r['variant']==variant and r['true_state']==state]
                es=[float(r['score']) for r in records if r['method']==method and r['variant']==variant and r['true_state']==state]
                if len(ss)!=3 or len(es)!=3: raise ValueError('Missing source/external score group')
                shift[state]={'source_PC9_median':float(np.median(ss)),
                              'external_PC9_median':float(np.median(es)),
                              'external_minus_source_median':float(np.median(es)-np.median(ss))}
            shifts[method][variant]=shift
    return records,metrics,coverage,paired,shifts


def csv_string(rows):
    import io
    if not rows:raise ValueError('No rows')
    o=io.StringIO(newline='')
    w=csv.DictWriter(o,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows)
    return o.getvalue()


def immutable_outputs(files):
    OUT.mkdir(parents=True,exist_ok=True)
    for filename,data in files.items():
        p=OUT/filename
        if p.exists() and p.read_bytes()!=data.encode('utf-8'):
            raise FileExistsError(f'Output already exists and differs; refusing overwrite: {p}')
    for filename,data in files.items():
        p=OUT/filename
        if not p.exists():p.write_bytes(data.encode('utf-8'))


def run():
    inputs=verify_prior_files()
    mods=import_source_scripts()
    s31=mods['s31']
    panels=s31.panels_from_frozen()
    baseline,baseline_rows,_=s31.source_model_replay(panels)
    fitted,source_rows,source_provenance=source_models(mods,panels,baseline,baseline_rows)
    print('Stage39: Stage31 + Stage35 source PC9 panels and margins EXACTLY REPLAYED')
    gsms,truth,targets=target_archive(mods,s31)
    records,metrics,coverage,paired,shifts=score_external(fitted,gsms,truth,targets,source_rows,s31)
    summary={
        'status':'EXPLORATORY_PREVIOUSLY_INSPECTED_EXTERNAL_STUDY_NOT_CONFIRMATORY',
        'source':'GSE193258', 'target':'GSE255958',
        'training_cell_lines':['H1975','HCC2935','HCC827'],
        'training_n':27,'source_PC9_test_n':9,'external_PC9_test_n':9,
        'problem':'binary DTP vs Control+Acute observed study labels',
        'methods':list(METHODS),'variants':list(VARIANTS),
        'source_gamma':1.0,'k':100,'C':.01,'threshold':0.0,
        'target_feature_policy':'Stage31 fixed rank-of-observed-panel + source training mean rank for missing; min 90/100',
        'external_target_used_for_fitting_or_tuning':False,
        'frozen_source_margin_and_panel_replays':True,
        'frozen_stage31_external_baseline_replayed':True,
        'stage35_training_line_diagnostic':{m:fitted[m]['diagnostic'] for m in METHODS},
        'coverage':coverage,'metrics':metrics,'paired':paired,
        'nonpaired_source_vs_external_median_score_shift':shifts,
        'input_sha256':inputs,'external_archive_sha256':sha(mods['s17'].TARGET),
        'source_provenance':source_provenance,
        'limitations':[
            'GSE255958 was already examined in stages 17, 31, 33; external scores cannot validate a new method independently.',
            'Stage35 was developed after inspecting source LOCO and old external results; no confirmatory model selection.',
            'Exactly 9 external PC9 samples; each condition is n=3.',
            'Acute/DTP labels differ in exposure duration; this cannot isolate a persistence-specific state or causal biomarker.',
            'Source-mean missing-gene rank adaptation changes representation; availability is below 100/100.',
            'No calibration or threshold changes, hyperparameter search or target-driven selection.',
            'No clinically useful prediction or algorithm novelty established.'
        ]
    }
    md=['# Stage 39 — Frozen Stage35 protected cross-study technical transfer','',
        '**Exploratory ONLY: previously inspected GSE255958, NOT independent validation.**','',
        'Training: 27 GSE193258 samples excluding PC9. Target: 9 GSE255958 PC9 samples.',
        'Exactly same frozen source panels, StandardScaler, contrast-protection gamma=1, LR C=0.01 and threshold 0.','',
        '## External PC9 results — binary DTP vs Control+Acute','',
        '| Selector | Variant | Correct/9 | Binary bACC | 3-condition macro | Control false DTP | Acute false DTP | DTP recall | AUC |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for variant in VARIANTS:
            d=metrics[method][variant]
            md.append(f"| {method} | {variant} | {d['correct']}/9 | {d['binary_balanced_accuracy']:.3f} | "
                      f"{d['three_condition_macro_recall']:.3f} | {d['control_false_dtp']}/3 | "
                      f"{d['acute_false_dtp']}/3 | {d['dtp_correct']}/3 | {d['binary_auc']:.3f} |")
    md+=['','## Target gene coverage & paired changes','',
         'Coverage: source-locked Stage31 policy; no missing feature selected using target outcome labels.','']
    for method in METHODS:
        cov=coverage[method];p=paired[method]
        md.append(f"- {method}: genes {cov['min_available']}–{cov['max_available']}/100; "
                  f"protected-only correct={p['protected_only_correct']}, "
                  f"baseline-only correct={p['baseline_only_correct']}, both correct={p['both_correct']}, both wrong={p['both_wrong']}.")
    md+=['','## Median source-PC9 to external-PC9 score changes (descriptive, not calibration)','',
         '| Selector | Variant | Condition | Source median | External median | Shift |',
         '|---|---|---|---:|---:|---:|']
    for method in METHODS:
        for variant in VARIANTS:
            for state in STATES:
                d=shifts[method][variant][state]
                md.append(f"| {method} | {variant} | {state} | {d['source_PC9_median']:+.4f} | "
                          f"{d['external_PC9_median']:+.4f} | {d['external_minus_source_median']:+.4f} |")
    md+=['','## Interpretation constraints','',
         '- Source-PC9 and Stage31 GSE255958 baseline margins were exactly replayed before new scoring.',
         '- GSE255958 had been examined previously: no new independent validation has occurred.',
         '- AUC on only 3 DTP and 6 non-DTP samples is highly uncertain; it does not establish generalization.',
         '- This is a technical completeness test of the protected model, not model selection or evidence of novelty.',
         '- Do not tune or revise the fixed method based on these nine target samples.',
         '- Time on treatment differs across Control/Acute/DTP; no DTP-specific molecular mechanism is proven.','']
    result={'39_report.md':'\n'.join(md),
            '39_summary.json':json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',
            '39_predictions.csv':csv_string(records)}
    immutable_outputs(result)
    print('Stage39 complete: frozen Stage31 external baseline REPLAYED; frozen Stage35 protected scores reported')
    for method in METHODS:
        x=metrics[method]
        print(f"{method}: baseline={x[VARIANTS[0]]['correct']}/9, protected={x[VARIANTS[1]]['correct']}/9; "
              f"baseline DTP={x[VARIANTS[0]]['dtp_correct']}/3, "
              f"protected DTP={x[VARIANTS[1]]['dtp_correct']}/3, "
              f"coverage={coverage[method]['min_available']}/100")
    print('Saved outputs/cross_study_contrast/39_{report.md,summary.json,predictions.csv}')
    print('RETROSPECTIVE ONLY; target previously inspected; no independent validity or algorithm novelty claim.')


def self_test():
    """Tests actual Stage35 protected fitting and the full target rank policy without source files."""
    from scipy.stats import rankdata
    mods=import_source_scripts(required=('s31','s34','s35'))
    s35,s34,s31=mods['s35'],mods['s34'],mods['s31']
    rng=np.random.default_rng(3901)
    y=np.tile(np.repeat(np.arange(3),3),3)
    groups=np.repeat(['A','B','C'],9)
    z=rng.normal(size=(27,100))
    z[:, :20]+=(y==2)[:,None]*1.5
    z[:, 20:40]+=np.repeat([-0.7,0,0.7],9)[:,None]
    z=rankdata(z,axis=1,method='average')/100
    scaler=StandardScaler().fit(z)
    P,info=s35.protected_shrinkage(scaler.transform(z),y,groups,gamma=1)
    model=s34.lr_model().fit(scaler.transform(z)@P,(y==2).astype(int))
    te=z[:9,:]
    sc=model.decision_function(scaler.transform(te)@P)
    if not np.isfinite(sc).all() or len(sc)!=9 or \
       not np.allclose(P@s35.contrast_vectors(scaler.transform(z),y),
                       s35.contrast_vectors(scaler.transform(z),y),atol=1e-8):
        raise AssertionError('Protected model transformation incorrect')
    ids=[f'G{i}' for i in range(100)]
    observed={g:float(i) for i,g in enumerate(ids) if i>=8}
    fake={'panel':ids,'train_feature_mean':z.mean(axis=0)}
    a,coverage=s31.target_rank_features({'t':observed},['t'],fake)
    expected=np.array([observed[ids[i]] for i in range(8,100)],float)
    if a.shape!=(1,100) or coverage[0]['available']!=92 or \
       not np.allclose(a[0,:8],fake['train_feature_mean'][:8]) or \
       not np.allclose(a[0,8:],rankdata(expected)/92):
        raise AssertionError('Stage31 missing-gene transfer did not match')
    try:
        s31.target_rank_features({'t':dict(list(observed.items())[6:])},['t'],fake)
        raise AssertionError('Too-low target coverage accepted')
    except ValueError:pass
    # No target labels are accepted by the feature transform.
    positive=s31.target_metrics(['Control']*3+['Acute']*3+['DTP']*3,
                                 np.array([-3,-2,-1,-2,-1,-3,1,2,3],float))
    if positive['correct']!=9: raise AssertionError('Target metric accounting failed')
    if set(METHODS)!=set(s31.METHODS) or tuple(VARIANTS)!=('stage28_baseline','contrast_protected_shrinkage_gamma1'):
        raise AssertionError('Unexpected frozen variants')
    print('SELF-TEST PASSED: protected feature subspace replay, fixed source-only transfer, target coverage guard, binary metrics')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--exploratory-only',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.exploratory_only:
        parser.error('Real historical external scoring requires --exploratory-only')
    run()

if __name__=='__main__':
    main()
