#!/usr/bin/env python3
"""Stage-47 auxiliary builder: audited source-only nested OOF scores.

This is NOT a new numbered research experiment. It prepares the input for
scripts/47_control_anchored_calibration.py without using external cohorts.

Run from repository root:
  python scripts/stage47_build_source_oof.py --self-test
  python scripts/stage47_build_source_oof.py --build

Builder guards:
  * Source expression and audited labels are loaded via Stage 21.
  * The exact Stage-38 outer held-out scores are replayed FIRST with
    3-cell-line source models (feature selection, LR, protected shrinkage).
  * Each nested training OOF score comes from 2 of the 3 OTHER cell lines.
  * No outer-test cell line enters any fit/selection/normalization/threshold.
  * Frozen outer test margins are copied from Stage 38 after replay.
  * The original source states are Control, Acute, DTP, NOT external DTC.
  * Exports and audit are immutable; cannot overwrite differing artifacts.

The two-line implementation is a necessary inner-fold generalization of the
fixed Stage-27/35 recipe, NOT a second model/parameter search. Threshold
transfer from 18-sample inner fits to 27-sample outer fits still carries
potential score-scale mismatch and is exploratory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_selection import f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
FROZEN = ROOT / 'outputs' / 'domain_invariance' / '38_predictions.csv'
OUT = ROOT / 'outputs' / 'stage47_control_anchored'
SOURCE_LINES = ('H1975', 'HCC2935', 'HCC827', 'PC9')
STATES = ('Control', 'Acute', 'DTP')
VARIANTS = {'base': 'stage28_baseline',
            'protected': 'contrast_protected_shrinkage_gamma1'}
METHOD = 'dual_contrast_rank_lr100'
K = 100
GAMMA = 1.0
REPLAY_TOL = 2e-6
REQUIRED_SCRIPTS = (
    '21_group_consistent_rank_selection.py',
    '27_dtp_specific_contrast_benchmark.py',
    '28_dual_contrast_ablation.py',
    '34_source_only_class_conditional_shrinkage.py',
    '35_contrast_protected_domain_shrinkage.py',
    '38_matched_source_dg_benchmark.py',
)
FIELDS = ('fold','variant','split','sample_id','cell_line','group','score')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def import_py(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_group_design(y,groups,nlines):
    y=np.asarray(y,dtype=int);g=np.asarray(groups)
    unique=sorted(set(g.tolist()))
    if len(unique)!=nlines or len(y)!=9*nlines:
        raise ValueError(f'Expected {nlines} training lines / {9*nlines} samples')
    if any(Counter(y[g==line].tolist())!={0:3,1:3,2:3} for line in unique):
        raise ValueError('Incorrect three-state/three-replicate training design')
    return unique


def select_dual_panel(X,y,groups,nlines):
    """Exact Stage27 dual-contrast gene scoring with 2 OR 3 training lines."""
    lines=validate_group_design(y,groups,nlines)
    X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=int);g=np.asarray(groups)
    if X.shape[0]!=9*nlines or X.ndim!=2 or X.shape[1]<500 or not np.isfinite(X).all():
        raise ValueError('Invalid source-only expression matrix')
    all_ranks=rankdata(X,axis=1,method='average') / X.shape[1]
    usable=np.flatnonzero(all_ranks.var(axis=0)>1e-12)
    if len(usable)<K:raise ValueError('Not enough nonconstant genes')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        f,_=f_classif(all_ranks[:,usable],(y==2).astype(int))
    good=np.isfinite(f)&(f>=0)
    f=np.where(good,f,-np.inf)
    effects=[]
    for line in lines:
        mask=g==line
        dtp=all_ranks[mask&(y==2)][:,usable].mean(axis=0)
        ctl=all_ranks[mask&(y==0)][:,usable].mean(axis=0)
        acute=all_ranks[mask&(y==1)][:,usable].mean(axis=0)
        effects.extend([dtp-ctl,dtp-acute])
    effects=np.stack(effects)
    min_up=effects.min(axis=0)
    min_down=(-effects).min(axis=0)
    direction=np.where(min_up>0,1,np.where(min_down>0,-1,0))
    strict=(direction!=0)&good
    eligible=int(strict.sum())
    if eligible<K:
        raise ValueError(f'Only {eligible} strict two-contrast genes for {nlines} lines; need 100; DO NOT silently lower K')
    strength=np.maximum(min_up,min_down)
    score=np.where(strict,strength*np.log1p(np.maximum(f,0)),-np.inf)
    order=np.lexsort((usable,-score))
    if not np.isfinite(score[order[:K]]).all():
        raise AssertionError('Nonfinite selector evidence')
    chosen=usable[order[:K]]
    return chosen,eligible


def nuisance_deviations(z,y,groups,nlines):
    """Stage35 residual convention; /sqrt(3*nlines), == /sqrt(9) for outer."""
    lines=validate_group_design(y,groups,nlines)
    z=np.asarray(z,dtype=float);y=np.asarray(y,dtype=int);g=np.asarray(groups)
    if z.shape!=(9*nlines,K):raise ValueError('Wrong standardized feature shape')
    rows=[]
    for state in (0,1,2):
        means=np.stack([z[(g==line)&(y==state)].mean(axis=0) for line in lines])
        rows.extend(means-means.mean(axis=0,keepdims=True))
    return np.stack(rows)/np.sqrt(3*nlines)


def protected_transform(z,y,groups,nlines):
    z=np.asarray(z,float);y=np.asarray(y,int);g=np.asarray(groups)
    D=nuisance_deviations(z,y,g,nlines)
    means=[z[y==state].mean(axis=0) for state in (0,1,2)]
    V=np.stack([means[2]-means[0],means[2]-means[1]],axis=1)
    if np.linalg.norm(V[:,0])<1e-10 or np.linalg.norm(V[:,1])<1e-10:
        raise ValueError('Degenerate DTP class contrast')
    U,s,_=np.linalg.svd(V,full_matrices=False)
    rank=int(np.sum(s>max(s[0]*1e-9,1e-12)))
    if rank<1:raise ValueError('No protected contrast direction')
    B=U[:,:rank]
    Dperp=D-(D@B)@B.T
    eigen,vec=np.linalg.eigh(Dperp.T@Dperp)
    if eigen.min() < -1e-8: raise ValueError('Non-PSD shrinkage matrix')
    P=(vec*(1/np.sqrt(1+GAMMA*np.maximum(eigen,0))))@vec.T
    P=(P+P.T)/2
    if not np.allclose(P@B,B,atol=2e-8,rtol=2e-8):
        raise AssertionError('Protected DTP subspace changed')
    return P


def fit_dual_scores(Xtr,ytr,gtr,Xte,st34,nlines):
    validate_group_design(ytr,gtr,nlines)
    Xtr=np.asarray(Xtr,float);Xte=np.asarray(Xte,float)
    if Xtr.shape[1]!=Xte.shape[1] or not np.isfinite(Xte).all():
        raise ValueError('Train/test feature mismatch')
    panel,n_eligible=select_dual_panel(Xtr,ytr,gtr,nlines)
    ztr=rankdata(Xtr[:,panel],axis=1,method='average')/K
    zte=rankdata(Xte[:,panel],axis=1,method='average')/K
    base=make_pipeline(StandardScaler(),st34.lr_model()).fit(ztr,(np.asarray(ytr)==2).astype(int))
    sb=np.asarray(base.decision_function(zte),float).reshape(-1)
    scaler=base.named_steps['standardscaler']
    z0=scaler.transform(ztr);zt=scaler.transform(zte)
    P=protected_transform(z0,ytr,gtr,nlines)
    prot=st34.lr_model().fit(z0@P,(np.asarray(ytr)==2).astype(int))
    sp=np.asarray(prot.decision_function(zt@P),float).reshape(-1)
    if not all(np.isfinite(v).all() for v in (sb,sp)):
        raise ValueError('Nonfinite margin')
    return {'base':sb,'protected':sp}, {'selected_genes':panel.tolist(), 'strict_eligible':n_eligible}


def read_frozen(path):
    if not path.is_file():raise FileNotFoundError(path)
    out={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['method']!=METHOD or r['variant'] not in VARIANTS.values():continue
            key=(r['variant'],r['held_out_line'],r['gsm'])
            if key in out:raise ValueError('Duplicate frozen Stage38 margin')
            if r['true_state'] not in STATES or r['held_out_line'] not in SOURCE_LINES:
                raise ValueError('Wrong Stage38 label or line')
            score=float(r['margin'])
            if not math.isfinite(score) or int(score>0)!=int(r['predicted_dtp']) or int((score>0)==(r['true_state']=='DTP'))!=int(r['correct']):
                raise ValueError('Frozen prediction row has inconsistent scoring semantics')
            out[key]=score
    if len(out)!=72:raise ValueError(f'Expected exactly 72 frozen Dual-Contrast margins, got {len(out)}')
    return out


def assert_frozen_cohort(source_ids,y,groups,frozen):
    for variant in VARIANTS.values():
        expected={(variant,str(g),str(i)) for i,g in zip(source_ids,groups)}
        got={key for key in frozen if key[0]==variant}
        if got!=expected:raise ValueError('Stage38 is NOT the same source cohort as Stage21')


def verify_frozen_labels_against_source(path, source_ids, y, groups):
    expected={(str(sample_id),str(line)):STATES[int(state)]
              for sample_id,line,state in zip(source_ids,groups,y)}
    checked=set()
    with path.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['method']!=METHOD or r['variant'] not in VARIANTS.values():
                continue
            key=(str(r['gsm']),str(r['held_out_line']))
            if key not in expected or r['true_state']!=expected[key]:
                raise ValueError('Stage38 true-state metadata disagrees with audited Stage21 source labels: '+str(key))
            checked.add(key)
    if len(checked)!=36:
        raise ValueError('Incomplete Stage38 true-state audit vs source')


def make_rows(source_ids,X,y,groups,st34,frozen):
    rows=[]; audit=[]
    ids=np.asarray(source_ids,dtype=object);groups=np.asarray(groups)
    for outer in SOURCE_LINES:
        outer_train=np.flatnonzero(groups!=outer)
        outer_test=np.flatnonzero(groups==outer)
        scores,meta=fit_dual_scores(X[outer_train],y[outer_train],groups[outer_train],X[outer_test],st34,3)
        for variant,score in scores.items():
            label=VARIANTS[variant]
            errors=[(str(ids[j]),abs(float(score[k])-frozen[(label,outer,str(ids[j]))])) for k,j in enumerate(outer_test)]
            largest=max(err for _,err in errors)
            if largest>REPLAY_TOL:
                raise AssertionError(f'Original Stage38 outer margin replay failed for {outer}/{variant}, max_err={largest:.3g}; ABORT')
            audit.append({'outer':outer,'inner':'OUTER_REPLAY','variant':variant,
                          'training_lines':','.join(sorted(set(groups[outer_train]))),
                          'n_train':len(outer_train),'n_test':len(outer_test),
                          'strict_eligible':meta['strict_eligible'],
                          'max_replay_abs_error':largest})
            for j in outer_test:
                rows.append({'fold':outer,'variant':variant,'split':'test',
                             'sample_id':str(ids[j]),'cell_line':outer,
                             'group':STATES[int(y[j])],
                             'score':format(frozen[(label,outer,str(ids[j]))],'.12g')})
        for inner in SOURCE_LINES:
            if inner==outer:continue
            # The critical nested isolation: inner trains on ONLY two remaining lines.
            inner_fit=np.flatnonzero((groups!=outer)&(groups!=inner))
            inner_test=np.flatnonzero(groups==inner)
            if outer in set(groups[inner_fit]) or inner in set(groups[inner_fit]) or len(inner_fit)!=18:
                raise AssertionError('Nested OOF train leakage')
            scores,meta=fit_dual_scores(X[inner_fit],y[inner_fit],groups[inner_fit],X[inner_test],st34,2)
            for variant,score in scores.items():
                audit.append({'outer':outer,'inner':inner,'variant':variant,
                              'training_lines':','.join(sorted(set(groups[inner_fit]))),
                              'n_train':len(inner_fit),'n_test':len(inner_test),
                              'strict_eligible':meta['strict_eligible'],
                              'max_replay_abs_error':None})
                for k,j in enumerate(inner_test):
                    rows.append({'fold':outer,'variant':variant,'split':'train_oof',
                                 'sample_id':str(ids[j]),'cell_line':inner,
                                 'group':STATES[int(y[j])],
                                 'score':format(float(score[k]),'.12g')})
    if len(rows)!=4*2*36 or len(audit)!=4*2*4:
        raise AssertionError('Incomplete expected OOF / replay coverage')
    return rows,audit


def csv_to_str(rows,fields):
    buf=io.StringIO(newline='')
    w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    return buf.getvalue()


def immutable_files(files):
    for path,content in files.items():
        if path.exists() and path.read_bytes()!=content.encode('utf-8'):
            raise FileExistsError(f'Refusing to overwrite differing Stage47 file: {path}')
    for path,content in files.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_text(content,encoding='utf-8',newline='\n')


def run_build():
    for name in REQUIRED_SCRIPTS:
        if not (SCRIPTS/name).is_file():raise FileNotFoundError(f'Missing original local script: {SCRIPTS/name}')
    s21=import_py(SCRIPTS/'21_group_consistent_rank_selection.py','stage21_oofbuilder')
    st34=import_py(SCRIPTS/'34_source_only_class_conditional_shrinkage.py','stage34_oofbuilder')
    genes,ids,X,y,groups,prov=s21.load_data()
    X=np.asarray(X,float);y=np.asarray(y,int);groups=np.asarray(groups)
    if X.shape!=(36,19712) or len(genes)!=19712 or len(set(ids))!=36 or set(groups.tolist())!=set(SOURCE_LINES):
        raise ValueError('GSE193258 source cohort mismatch. Do not substitute external GSE249721.')
    for line in SOURCE_LINES:
        validate_group_design(y[groups==line],groups[groups==line],1)
    frozen=read_frozen(FROZEN)
    assert_frozen_cohort(ids,y,groups,frozen)
    verify_frozen_labels_against_source(FROZEN,ids,y,groups)
    rows,audit=make_rows(ids,X,y,groups,st34,frozen)
    trace={
        'stage':47, 'status':'exploratory input builder, NOT scientific outcome',
        'source_study':'GSE193258', 'source_only':True,
        'positive_label':'DTP', 'target_GSE249721_used':False,
        'outer_test_replay_required_and_passed':True,
        'outer_test_scores_from_immutable_stage38':True,
        'inner_oof_model_fit_lines':2, 'outer_model_fit_lines':3,
        'inner_oof_models_do_not_fit_outer_test_line':True,
        'internal_model_recipe':'Stage27 rank-based dual contrast + Stage35 gamma=1 protected shrinkage',
        'thresholds_not_fitted_here':True,
        'source_provenance':prov,
        'source_X_float64_sha256':hashlib.sha256(X.tobytes()).hexdigest(),
        'frozen_38_sha256':digest(FROZEN),
        'upstream_script_sha256':{name:digest(SCRIPTS/name) for name in REQUIRED_SCRIPTS},
        'builder_sha256':digest(Path(__file__)),
        'n_output_rows':len(rows),
        'n_outer_replays':8, 'n_nested_fit_scores':24,
        'limitations':['Historical source folds were inspected repeatedly.',
                       'Inner models use 18 rather than 27 samples; margin scale may differ.',
                       'No independent external validation and no claim of algorithm novelty.'],
    }
    output={
        OUT/'47_source_scores.csv':csv_to_str(rows,FIELDS),
        OUT/'47_source_oof_audit.csv':csv_to_str(audit,('outer','inner','variant','training_lines','n_train','n_test','strict_eligible','max_replay_abs_error')),
        OUT/'47_source_oof_provenance.json':json.dumps(trace,indent=2,ensure_ascii=False)+'\n',
    }
    immutable_files(output)
    print('NESTED OOF BUILT: 4 outer x 2 variants x (9 frozen test + 27 TRAIN-only OOF).')
    print('FROZEN REPLAY PASSED: 8 outer model/fold checks against Stage38.')
    print('SOURCE DTP LABEL CONFIRMED; GSE249721 / GSE255958 not used.')
    for path in output:print('Saved:',path)
    print('NEXT: audit provenance before running Stage47 calibrated comparison.')


def self_test():
    # Designed to test code mechanics, not biological performance or external validity.
    rng=np.random.default_rng(47)
    groups=np.repeat(['A','B','C','D'],9)
    y=np.tile(np.repeat([0,1,2],3),4)
    X=rng.normal(0,0.2,(36,520))
    X+=np.linspace(-2,2,520)[None,:]
    signal=np.r_[np.linspace(20,35,200),np.zeros(320)]
    X[y==2]+=signal
    for outer in ['A','B','C','D']:
        tr=np.flatnonzero(groups!=outer)
        va=np.flatnonzero(groups==outer)
        panel,n=select_dual_panel(X[tr],y[tr],groups[tr],3)
        assert len(panel)==K and n>=K
        for inner in set(groups[tr]):
            fit=np.flatnonzero((groups!=outer)&(groups!=inner))
            panel2,n2=select_dual_panel(X[fit],y[fit],groups[fit],2)
            assert len(panel2)==K and n2>=K and outer not in set(groups[fit])
    class Model:
        @staticmethod
        def lr_model():
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(C=0.01,solver='liblinear',class_weight='balanced',max_iter=3000,random_state=42)
    scores,_=fit_dual_scores(X[:18],y[:18],groups[:18],X[18:27],Model,2)
    assert len(scores['base'])==9 and len(scores['protected'])==9
    assert all(np.isfinite(v).all() for v in scores.values())
    # End-to-end synthetic producer/consumer contract: all 288 required rows,
    # same per-line cohort in every outer fold, then Stage47 validation.
    ids=np.array([f'SYNTH_{i:03d}' for i in range(36)])
    fake={}
    for line in ['A','B','C','D']:
        tr=np.flatnonzero(groups!=line);te=np.flatnonzero(groups==line)
        val,_=fit_dual_scores(X[tr],y[tr],groups[tr],X[te],Model,3)
        for variant,pred in val.items():
            for j,ix in enumerate(te):
                fake[(VARIANTS[variant],line,str(ids[ix]))]=float(pred[j])
    # Temporarily substitute synthetic line universe for this test alone.
    global SOURCE_LINES
    original=SOURCE_LINES
    try:
        SOURCE_LINES=('A','B','C','D')
        rows,audit=make_rows(ids,X,y,groups,Model,fake)
    finally:
        SOURCE_LINES=original
    assert len(rows)==288 and len(audit)==32
    expected={('test',9),('train_oof',27)}
    for line in ['A','B','C','D']:
        for variant in VARIANTS:
            split_n={(sp,sum(1 for x in rows if x['fold']==line and x['variant']==variant and x['split']==sp))
                     for sp in ('test','train_oof')}
            assert split_n==expected
    print('SELF-TEST PASSED: two/three-line selectors, nested isolation, finite margins, '
          '288-row synthetic OOF export with 8 exact replay matches. NO real-data evaluation.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--self-test',action='store_true')
    m.add_argument('--build',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    elif args.build:run_build()


if __name__=='__main__':main()
