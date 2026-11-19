#!/usr/bin/env python3
"""Stage 35 | Source-only contrast-protected nuisance-shrinkage hypothesis test.

Motivation: Stage 34 compressed class-conditional source-domain directions but
reduced held-out-line accuracy. Hypothesis (NOT a proven mechanism): domain
shrinkage also suppresses training DTP-vs-Control/Acute discrimination.

New fixed, source-only candidate: protect the 2-D TRAINING class-contrast span
before shrinking domain residual directions in its orthogonal complement:
    B = orth([mean(DTP)-mean(Control), mean(DTP)-mean(Acute)])
    Dperp = D @ (I - B B.T)
    P = (I + Dperp.T @ Dperp)^(-1/2)   [gamma=1 fixed]
The feature contrast vectors are preserved exactly: P @ B == B.
No held-out-line or external study observations enter feature selection,
class contrasts, domain transform, model fitting, or threshold choice.

This is retrospective exploratory mechanism testing, NOT formal preregistration,
independent validation, novelty, biological specificity, or evidence of Q1.
Do NOT choose gamma from results; do NOT use historically inspected GSE255958
for method selection. This is ONE diagnostic candidate, not a grid search.

Usage from repo root:
    python scripts/35_contrast_protected_domain_shrinkage.py --self-test
    python scripts/35_contrast_protected_domain_shrinkage.py --exploratory-only

Outputs (new, immutable): outputs/domain_invariance/35_{report.md,summary.json,
    predictions.csv,train_diagnostics.csv}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
OUT = ROOT / 'outputs' / 'domain_invariance'
FROZEN = ROOT / 'outputs' / 'dtp_specificity'
LINES = ('H1975', 'HCC2935', 'HCC827', 'PC9')
METHODS = ('acute_only_rank_lr100', 'dual_contrast_rank_lr100')
STATES = ('Control','Acute','DTP')
VARIANTS = ('stage28_baseline', 'class_conditional_shrinkage_gamma1',
            'contrast_protected_shrinkage_gamma1')
GAMMA = 1.0
K = 100


def components():
    import importlib.util
    def imp(filename, name):
        p=SCRIPTS / filename
        if not p.is_file():
            raise FileNotFoundError(f'Missing upstream script: {p}')
        spec=importlib.util.spec_from_file_location(name,p)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return (imp('21_group_consistent_rank_selection.py', 'stage21_s35'),
            imp('27_dtp_specific_contrast_benchmark.py', 'stage27_s35'),
            imp('28_dual_contrast_ablation.py', 'stage28_s35'),
            imp('34_source_only_class_conditional_shrinkage.py', 'stage34_s35'))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1<<20),b''):
            h.update(part)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def dump_csv(rows):
    if not rows: raise ValueError('No records')
    f=io.StringIO(newline='')
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows)
    return f.getvalue()


def immutable_write(files):
    OUT.mkdir(parents=True,exist_ok=True)
    for filename,s in files.items():
        p=OUT/filename
        if p.exists() and p.read_bytes()!=s.encode('utf-8'):
            raise FileExistsError(f'Refusing to overwrite prior artifact: {p}')
    for filename,s in files.items():
        p=OUT/filename
        if not p.exists():p.write_bytes(s.encode('utf-8'))


def nuisance_deviations(z, y, groups):
    z=np.asarray(z,float);y=np.asarray(y,int);g=np.asarray(groups)
    lines=sorted(set(g.tolist()))
    if z.shape!=(27,K) or len(lines)!=3 or not np.isfinite(z).all():
        raise ValueError('Wrong source training design or nonfinite features')
    for line in lines:
        if Counter(y[g==line].tolist()) != {0:3,1:3,2:3}:
            raise ValueError('Wrong class composition in a TRAINING cell line')
    rows=[]
    for state in (0,1,2):
        means=np.stack([z[(g==line)&(y==state)].mean(axis=0) for line in lines])
        rows.extend(means-means.mean(axis=0,keepdims=True))
    D=np.stack(rows)/np.sqrt(9)
    if D.shape!=(9,K):raise AssertionError('Domain contrast matrix wrong shape')
    return D


def contrast_vectors(z, y):
    z=np.asarray(z,float);y=np.asarray(y,int)
    if z.shape!=(27,K) or Counter(y.tolist())!={0:9,1:9,2:9}:
        raise ValueError('Expected 27 TRAINING samples and all three states')
    means=[z[y==j].mean(axis=0) for j in (0,1,2)]
    V=np.stack((means[2]-means[0],means[2]-means[1]),axis=1)
    if not np.isfinite(V).all() or np.linalg.norm(V[:,0])<1e-10 or np.linalg.norm(V[:,1])<1e-10:
        raise ValueError('Degenerate TRAINING DTP state contrasts')
    return V


def spectral_shrink(D, gamma=GAMMA):
    if D.shape!=(9,K) or not np.isfinite(D).all() or not np.isfinite(gamma) or gamma<0:
        raise ValueError('Invalid nuisance matrix or gamma')
    eig,vec=np.linalg.eigh(D.T@D)
    if eig.min() < -1e-8:raise AssertionError('Negative covariance eigenvalue')
    mul=1/np.sqrt(1+gamma*np.maximum(eig,0))
    P=(vec*mul)@vec.T
    return (P+P.T)/2


def protected_shrinkage(z, y, groups, gamma=GAMMA):
    """Fit only on training standardized ranks; preserve BOTH contrast vectors."""
    D=nuisance_deviations(z,y,groups)
    V=contrast_vectors(z,y)
    U,s,_=np.linalg.svd(V,full_matrices=False)
    rank=int(np.sum(s > max(s[0]*1e-9,1e-12)))
    if rank < 1:raise ValueError('No protected contrast direction')
    B=U[:,:rank]
    Dperp=D-(D@B)@B.T
    P=spectral_shrink(Dperp,gamma)
    if not np.allclose(P@B,B,atol=2e-8,rtol=2e-8):
        raise AssertionError('DTP contrasts were not preserved')
    before=np.linalg.norm(D)
    after=np.linalg.norm(D@P)
    v_after=np.linalg.norm(P@V,axis=0)/np.linalg.norm(V,axis=0)
    if not np.allclose(v_after,1,atol=2e-8):raise AssertionError('Class contrast norm lost')
    if after > before+1e-9:raise AssertionError('Domain shrink increased dispersion')
    return P, {'protected_span_dimension':rank,
               'selected_feature_contrast_retention_min':float(v_after.min()),
               'domain_residual_fro_before':float(before),
               'domain_residual_fro_after':float(after),
               'domain_residual_fro_ratio':float(after/before) if before>0 else 1.0}


def fit_triplet(ztr,zte,ytr,gtr, st34):
    ztr=np.asarray(ztr,float);zte=np.asarray(zte,float)
    ytr=np.asarray(ytr,int);gtr=np.asarray(gtr)
    if ztr.shape!=(27,K) or zte.ndim!=2 or zte.shape[1]!=K or not np.isfinite(zte).all():
        raise ValueError('Wrong rank panel shape')
    base=make_pipeline(StandardScaler(),st34.lr_model()).fit(ztr,(ytr==2).astype(int))
    bscore=np.asarray(base.decision_function(zte),float)
    z0=base.named_steps['standardscaler'].transform(ztr)
    zz=base.named_steps['standardscaler'].transform(zte)
    P_old,trace_old=st34.class_conditional_shrinkage(z0,ytr,gtr,gamma=GAMMA)
    old=st34.lr_model().fit(z0@P_old,(ytr==2).astype(int))
    oscore=np.asarray(old.decision_function(zz@P_old),float)
    P_new,trace_new=protected_shrinkage(z0,ytr,gtr,gamma=GAMMA)
    new=st34.lr_model().fit(z0@P_new,(ytr==2).astype(int))
    nscore=np.asarray(new.decision_function(zz@P_new),float)
    V=contrast_vectors(z0,ytr)
    older=np.linalg.norm(P_old@V,axis=0)/np.linalg.norm(V,axis=0)
    sb=base.decision_function(ztr)
    so=old.decision_function(z0@P_old)
    sn=new.decision_function(z0@P_new)
    info={**trace_new,
          'unprotected_contrast_retention_min':float(older.min()),
          'train_score_dispersion_baseline':st34.class_line_spread(sb,ytr,gtr),
          'train_score_dispersion_unprotected':st34.class_line_spread(so,ytr,gtr),
          'train_score_dispersion_protected':st34.class_line_spread(sn,ytr,gtr),
          'train_gap_baseline':float(sb[ytr==2].mean()-sb[ytr!=2].mean()),
          'train_gap_unprotected':float(so[ytr==2].mean()-so[ytr!=2].mean()),
          'train_gap_protected':float(sn[ytr==2].mean()-sn[ytr!=2].mean()),
          'old_domain_fro_after':trace_old['domain_direction_fro_after']}
    if not all(np.isfinite(x).all() for x in (bscore,oscore,nscore)):
        raise ValueError('Non-finite held-out scores')
    return (bscore,oscore,nscore),info


def evaluate(genes, ids, X, y, groups, st27, st28, st34, frozen=None, panels=None):
    y=np.asarray(y,int);groups=np.asarray(groups);X=np.asarray(X,float)
    if len(genes)!=X.shape[1] or X.shape[0]!=36 or len(set(ids))!=36 or set(groups)!=set(LINES):
        raise ValueError('Invalid experimental sample layout')
    records=[];diagnostics=[]
    for outer in LINES:
        tr=np.flatnonzero(groups!=outer);te=np.flatnonzero(groups==outer)
        st27.validate_train(y[tr],groups[tr])
        if len(tr)!=27 or len(te)!=9 or Counter(y[te].tolist())!={0:3,1:3,2:3}:
            raise ValueError('Invalid held-out cohort')
        for method in METHODS:
            if method==METHODS[0]:
                ix,_=st28.select_single_contrast(X[tr],y[tr],groups[tr],negative=1)
            else:
                ix,_=st27.select_rank_panel(X[tr],y[tr],groups[tr],mode='dual_contrast')
            if len(ix)!=K or len(set(ix.tolist()))!=K:raise ValueError('Panel invalid')
            if panels is not None and set(genes[int(i)] for i in ix)!=set(panels[(method,outer)]):
                raise ValueError(f'Frozen panel mismatch {method}/{outer}')
            ztr=st34.within_panel_rank(X[tr],ix)
            zte=st34.within_panel_rank(X[te],ix)
            scores,info=fit_triplet(ztr,zte,y[tr],groups[tr],st34)
            diagnostics.append({'method':method,'held_out_line':outer,**{k:float(v) for k,v in info.items()}})
            for j,idx in enumerate(te):
                for variant,score in zip(VARIANTS,scores):
                    value=float(score[j])
                    pred=int(value>0)
                    row={'method':method,'variant':variant,'held_out_line':outer,
                         'gsm':ids[idx], 'true_state':STATES[int(y[idx])],
                         'margin':f'{value:.12g}', 'predicted_dtp':pred,
                         'correct':int(pred==(y[idx]==2))}
                    if frozen is not None and variant!=VARIANTS[-1]:
                        expect=frozen.get((method,variant,outer,ids[idx]))
                        if expect is None or expect['true_state']!=row['true_state'] or \
                           not np.isclose(float(expect['margin']),value,atol=1e-8,rtol=1e-7) or \
                           int(expect['predicted_dtp'])!=pred:
                            raise ValueError(f'Frozen Stage34 replay mismatch: {method}/{variant}/{outer}/{ids[idx]}')
                    records.append(row)
    if len(records)!=len(METHODS)*len(VARIANTS)*36 or len(diagnostics)!=len(METHODS)*4:
        raise AssertionError('Incomplete stage35')
    return records,diagnostics


def source_frozen(st34,genes,ids,y,groups):
    ref,panels=st34.load_frozen_reference(genes,ids,y,groups)
    prev=csv_rows(OUT/'34_predictions.csv')
    old={}
    for r in prev:
        key=(r['method'],r['variant'],r['held_out_line'],r['gsm'])
        if key in old:raise ValueError('Duplicate Stage34 row')
        old[key]=r
    if len(old)!=144:raise ValueError('Wrong prior stage34 sample coverage')
    summary=json.loads((OUT/'34_summary.json').read_text(encoding='utf-8'))
    if summary.get('study')!='GSE193258' or summary.get('gamma_fixed_in_script')!=GAMMA or summary.get('n_source')!=36:
        raise ValueError('Prior Stage34 not the fixed candidate')
    if summary.get('model_variants')!=list(VARIANTS[:2]):raise ValueError('Prior variants changed')
    # The Stage28 values were independently replayed by Stage34. Verify the exact
    # source score list against Stage28 again to avoid trusting stale metadata.
    for r in prev:
        if r['variant']!='stage28_baseline':continue
        rr=ref[(r['method'],r['held_out_line'],r['gsm'])]
        if r['true_state']!=rr['true_state'] or not np.isclose(float(r['margin']),float(rr['margin']),atol=1e-8,rtol=1e-7):
            raise ValueError('Stage28/34 frozen baseline disagreement')
    return old,panels,summary


def summarize(rows,diagnostics):
    st34=components()[3]
    details={}
    for m in METHODS:
        metric={}
        for variant in VARIANTS:
            subset=[r for r in rows if r['method']==m and r['variant']==variant]
            y=[STATES.index(r['true_state']) for r in subset]
            s=[float(r['margin']) for r in subset]
            folds={line:st34.score_metrics(
                [STATES.index(r['true_state']) for r in subset if r['held_out_line']==line],
                [float(r['margin']) for r in subset if r['held_out_line']==line]) for line in LINES}
            metric[variant]={'pooled':st34.score_metrics(y,s),'per_line':folds}
        base={r['gsm']:int(r['correct']) for r in rows if r['method']==m and r['variant']==VARIANTS[0]}
        cand={r['gsm']:int(r['correct']) for r in rows if r['method']==m and r['variant']==VARIANTS[-1]}
        details[m]={'metrics':metric,'paired_protected_vs_baseline':{
            'baseline_only_correct':sum(base[k] and not cand[k] for k in base),
            'protected_only_correct':sum(cand[k] and not base[k] for k in base),
            'both_correct':sum(base[k] and cand[k] for k in base),
            'both_wrong':sum(not base[k] and not cand[k] for k in base)}}
    return details


def markdown(summary):
    lines=['# Stage 35 — Protecting training DTP contrasts during source-domain shrinkage','',
           '**Exploratory hypothesis test only. Four historical source LOCO folds, no independent validation.**','',
           'Training-only class contrasts DTP−Control and DTP−Acute are protected by a 1–2D orthogonal subspace. Only class-conditional cell-line residuals perpendicular to that subspace are shrunk. Fixed gamma=1, k=100, C=0.01, threshold=0, no test/target fitting or calibration.','',
           '## Held-out source results','',
           '| Method | Variant | Correct/36 | 3-condition macro | Control false DTP | Acute false DTP | DTP recall |',
           '|---|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        for v in VARIANTS:
            z=summary['methods'][m]['metrics'][v]['pooled']
            lines.append(f"| {m} | {v} | {z['correct']}/36 | {z['three_condition_macro_recall']:.3f} | {z['control_false_dtp']:.3f} | {z['acute_false_dtp']:.3f} | {z['dtp_recall']:.3f} |")
    lines+=['','## Paired against Stage28 baseline','',
            '| Method | Baseline-only correct | Protected-only correct | Both correct | Both wrong |',
            '|---|---:|---:|---:|---:|']
    for m in METHODS:
        p=summary['methods'][m]['paired_protected_vs_baseline']
        lines.append(f"| {m} | {p['baseline_only_correct']} | {p['protected_only_correct']} | {p['both_correct']} | {p['both_wrong']} |")
    lines+=['','## Why this is not automatically a better model','',
            'Each training contrast vector is preserved in standardized-rank feature space. This does **not** guarantee test performance, calibration, model margin equality, or biological state specificity. Residual nuisance directions and class contrasts may still be entangled by experimental design.',
            'Compare `35_train_diagnostics.csv` for unprotected/protected feature contrast retention, class-conditional training-score dispersion and class margin gap. Do not pick a winning gamma or add post-hoc hyperparameters from historical folds.','',
            '**Interpretation:** An improved historical score is only hypothesis generation, and a negative result should be retained. No external target samples are scored. No novelty or confirmatory claim.','']
    return '\n'.join(lines)


def self_test():
    rng=np.random.default_rng(3501)
    yy=np.tile(np.repeat(np.arange(3),3),3)
    gg=np.repeat(['A','B','C'],9)
    Z=rng.normal(size=(27,K))
    Z[:, :12]+=(yy==2)[:,None]*1.2
    Z[:,12:20]+=(yy==0)[:,None]*0.4
    Z[:,30:45]+=np.repeat([-1,0,1],9)[:,None]*1.1
    P0,d0=protected_shrinkage(Z,yy,gg,gamma=0)
    P,d=protected_shrinkage(Z,yy,gg,gamma=1)
    if not np.allclose(P0,np.eye(K),atol=1e-9) or \
       not np.isclose(d['selected_feature_contrast_retention_min'],1,atol=1e-8) or \
       d['domain_residual_fro_after']>d['domain_residual_fro_before']:
        raise AssertionError('Contrast preservation or PSD shrink failed')
    # Known nuisance and signal conflict: ordinary shrink affects signal, protection does not.
    _,_,_,st34=components()
    regular,_=st34.class_conditional_shrinkage(Z,yy,gg,gamma=1)
    v=contrast_vectors(Z,yy)
    r_old=np.linalg.norm(regular@v,axis=0)/np.linalg.norm(v,axis=0)
    r_new=np.linalg.norm(P@v,axis=0)/np.linalg.norm(v,axis=0)
    if not np.allclose(r_new,[1,1],atol=1e-8) or (r_old>1+1e-8).any():
        raise AssertionError('Contrast retention computation incorrect')
    try:
        protected_shrinkage(Z,yy,gg,gamma=-1)
        raise AssertionError('Negative gamma accepted')
    except ValueError:pass
    # Full integrated synthetic four-fold test, no real source data needed.
    _,s27,s28,st34=components()
    ngenes=760
    gs=[]; ys=[]; groups=[];xs=[]
    base=rng.normal(0,0.7,ngenes)
    ln=rng.normal(0,0.12,(4,ngenes))
    for i,line in enumerate(LINES):
        for state in range(3):
            for rep in range(3):
                row=base+ln[i]+rng.normal(0,0.08,ngenes)
                if state==2:
                    row[:180]+=3.0
                    row[180:350]-=3.0
                elif state==1:row[350:390]+=0.15
                xs.append(row);ys.append(state);groups.append(line);gs.append(f'SYN_{i}_{state}_{rep}')
    genes=[f'SYNGENE_{i:04d}' for i in range(ngenes)]
    rows,diags=evaluate(genes,gs,np.asarray(xs),np.asarray(ys),np.asarray(groups),s27,s28,st34)
    if len(rows)!=216 or len(diags)!=8:
        raise AssertionError('Wrong integration count')
    details=summarize(rows,diags)
    summary={'methods':details}
    if 'contrast_protected_shrinkage_gamma1' not in markdown(summary):
        raise AssertionError('Reporting error')
    print('SELF-TEST PASSED: exact gamma=0 identity, protected DTP-vs-Control/Acute subspace, source-only nuisance shrinkage, paired four-fold synthetic benchmark')


def run():
    s21,s27,s28,st34=components()
    genes,ids,X,y,groups,provenance=s21.load_data()
    if X.shape!=(36,19712) or len(genes)!=19712 or len(set(genes))!=19712:
        raise ValueError('Unexpected source matrix/genes')
    frozen,panels,old_summary=source_frozen(st34,genes,ids,y,groups)
    if old_summary.get('source_provenance')!=provenance:
        raise ValueError('Source provenance no longer matches frozen Stage34')
    rows,diag=evaluate(genes,ids,X,y,groups,s27,s28,st34,frozen=frozen,panels=panels)
    methods=summarize(rows,diag)
    summary={'status':'EXPLORATORY_MECHANISM_TEST_SOURCE_LOCO_PREVIOUSLY_INSPECTED',
             'source_study':'GSE193258','n_source':36,'outer_lines':list(LINES),
             'methods':methods,'variants':list(VARIANTS),'gamma_fixed':GAMMA,'k':K,'C':0.01,
             'threshold':0,'training_objective':'DTP vs Control+Acute',
             'mechanism':'Protect 2 train-only DTP class-contrast vectors; shrink orthogonal class-conditional cell-line residuals',
             'training_only':True,'no_external_scoring':True,
             'source_provenance':provenance,
             'frozen_inputs_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in
                [FROZEN/'27_selected_genes.csv',FROZEN/'28_selected_genes.csv',FROZEN/'28_predictions.csv',
                 OUT/'34_predictions.csv',OUT/'34_summary.json']},
             'diagnostics':diag,
             'limitations':['Post-hoc algorithm proposal after viewing Stage34 source holdouts',
                            'Previously examined four source folds, not independent validation',
                            'No evidence of original algorithm: prior art comparison not performed',
                            'Acute and DTP treatment-duration confounded',
                            'No hyperparameter tuning, target calibration, or external scoring']}
    artifacts={'35_predictions.csv':dump_csv(rows),
               '35_train_diagnostics.csv':dump_csv(diag),
               '35_summary.json':json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',
               '35_report.md':markdown(summary)}
    immutable_write(artifacts)
    print('Stage35 complete: Stage28 baseline and Stage34 unprotected prototype EXACTLY REPLAYED')
    for m in METHODS:
        d=methods[m]
        b=d['metrics'][VARIANTS[0]]['pooled'];u=d['metrics'][VARIANTS[1]]['pooled'];p=d['metrics'][VARIANTS[2]]['pooled']
        pair=d['paired_protected_vs_baseline']
        print(f"{m}: baseline={b['correct']}/36 unprotected={u['correct']}/36 protected={p['correct']}/36 "
              f"macro={p['three_condition_macro_recall']:.3f} DTP_recall={p['dtp_recall']:.3f} "
              f"baseline_only={pair['baseline_only_correct']} protected_only={pair['protected_only_correct']}")
    print('Saved outputs/domain_invariance/35_{report.md,summary.json,predictions.csv,train_diagnostics.csv}')
    print('EXPLORATORY: historical LOCO, no external validation, no novel-algorithm claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--exploratory-only',action='store_true')
    args=p.parse_args()
    if args.self_test: self_test()
    elif args.exploratory_only:run()
    else:p.error('Use --self-test or --exploratory-only; this is exploratory source work')


if __name__=='__main__':main()
