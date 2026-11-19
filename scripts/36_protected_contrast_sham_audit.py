#!/usr/bin/env python3
"""Stage 36: source-only falsification audit of Stage35 contrast-protected shrinkage.

Purpose: compare the fixed Stage35 real training-contrast protection against
matched two-dimensional *sham* protected subspaces. Tests whether Stage35's
historical source-LOCO benefit is specific to actual TRAINING DTP contrasts,
not merely to preserving any two directions. No target/external data are read.

Controls (both matched to rank=2, same nuisance D, gamma=1, classifier,
selector, thresholds, folds, with NO held-out-derived model choices):
  - shuffled: shuffle class labels WITHIN EACH TRAINING cell line ONLY to form
    two sham contrast directions; true labels still define nuisance D and fit.
  - random: two iid Gaussian directions in the 100-d feature space.

Exactly 64 deterministic draws per sham family across four folds. Synthetic
controls do not create new independent biological samples or null p-values.
There is no tuning or selection of favorable runs.

Usage:
 python scripts/36_protected_contrast_sham_audit.py --self-test
 python scripts/36_protected_contrast_sham_audit.py --exploratory-only

Outputs (immutable): outputs/domain_invariance/36_{report.md,summary.json,
 sham_draws.csv,paired_samples.csv}
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,importlib.util
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'scripts'
OUT=ROOT/'outputs'/'domain_invariance'
STAGE35=OUT/'35_predictions.csv'
STAGE35_JSON=OUT/'35_summary.json'
STAGE35_DIAG=OUT/'35_train_diagnostics.csv'
STAGE28=ROOT/'outputs'/'dtp_specificity'/'28_predictions.csv'
METHODS=('acute_only_rank_lr100','dual_contrast_rank_lr100')
LINES=('H1975','HCC2935','HCC827','PC9')
STATES=('Control','Acute','DTP')
N_DRAWS=64
SEED=3601
K=100
GAMMA=1.0
VARIANTS=('stage28_baseline','class_conditional_shrinkage_gamma1','contrast_protected_shrinkage_gamma1')


def imp(path,name):
    if not path.is_file(): raise FileNotFoundError(f'Required source script: {path}')
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    if not path.is_file():raise FileNotFoundError(f'Frozen input missing: {path}')
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def serialize(rows):
    if not rows:raise ValueError('Cannot write empty rows')
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue()


def immutable_write(files):
    OUT.mkdir(exist_ok=True,parents=True)
    for fn,s in files.items():
        p=OUT/fn
        if p.is_file() and p.read_bytes()!=s.encode('utf-8'):
            raise FileExistsError(f'Existing artifact differs (refusing overwrite): {p}')
    for fn,s in files.items():
        p=OUT/fn
        if not p.exists():p.write_bytes(s.encode('utf-8'))


def sham_contrasts(z,y,g,kind,rng):
    """Construct 2 directions only from training data. Never access test values."""
    z=np.asarray(z,float);y=np.asarray(y,int);g=np.asarray(g)
    if z.shape!=(27,K) or Counter(y.tolist())!={0:9,1:9,2:9} or len(set(g))!=3:
        raise ValueError('Training design mismatch')
    if kind=='random':
        return rng.normal(size=(K,2))
    if kind!='shuffled':raise ValueError('Unsupported sham kind')
    yp=y.copy()
    for line in sorted(set(g)):
        ix=np.flatnonzero(g==line)
        if Counter(y[ix].tolist())!={0:3,1:3,2:3}:raise ValueError('Line unbalanced')
        yp[ix]=rng.permutation(y[ix])
    return np.column_stack((z[yp==2].mean(axis=0)-z[yp==0].mean(axis=0),
                            z[yp==2].mean(axis=0)-z[yp==1].mean(axis=0)))


def shrink_sham(D,V,gamma=GAMMA):
    """Thin-SVD form of Stage35 PSD shrink; protects span(V)."""
    if D.shape!=(9,K) or V.shape!=(K,2) or not all(np.isfinite(x).all() for x in (D,V)):
        raise ValueError('Invalid nuisance/contrast shape')
    if gamma<0 or not np.isfinite(gamma):raise ValueError('Invalid gamma')
    u,s,_=np.linalg.svd(V,full_matrices=False)
    tol=max(s[0]*1e-9,1e-12)
    r=int(np.count_nonzero(s>tol))
    if r!=2:raise ValueError('Sham protection must be exactly two-dimensional')
    B=u[:,:r]
    dp=D-(D@B)@B.T
    # D is 9x100: only 6–9 nonzero singular directions, efficient no 100x100 eig
    _,sing,vt=np.linalg.svd(dp,full_matrices=False)
    factors=1.0-1.0/np.sqrt(1.0+gamma*sing**2)
    P=np.eye(K)-(vt.T*factors)@vt
    if not np.allclose(P@B,B,atol=2e-8,rtol=2e-8):
        raise AssertionError('Protected directions moved')
    if np.linalg.norm(D@P)>np.linalg.norm(D)+1e-8:
        raise AssertionError('Nuisance matrix norm increased')
    return P


def comparison_metrics(truth, scores):
    y=np.asarray(truth,int);s=np.asarray(scores,float)
    if y.shape!=(36,) or s.shape!=(36,) or Counter(y.tolist())!={0:12,1:12,2:12}:
        raise ValueError('Expected complete 36 source held-outs')
    if not np.isfinite(s).all():raise ValueError('Nonfinite scores')
    pred=(s>0).astype(int)
    correct=int(np.count_nonzero(pred==(y==2)))
    ctrl=int(pred[y==0].sum());acute=int(pred[y==1].sum());dtp=int(pred[y==2].sum())
    return {'correct':correct,'three_condition_macro_recall':float(((12-ctrl)+(12-acute)+dtp)/36),
            'control_false_dtp':ctrl,'acute_false_dtp':acute,'dtp_recall':dtp}


def replay_frozen(rows,method,outer,ids,y3,bscore,uscore,pscore):
    """Exact per-sample replay of all Stage35 variants, rather than trusting labels."""
    for variant,scores in zip(VARIANTS,(bscore,uscore,pscore)):
        for gsm,truth,score in zip(ids,y3,scores):
            key=(method,variant,outer,gsm)
            ref=rows.get(key)
            if ref is None:raise ValueError(f'Stage35 frozen sample missing: {key}')
            if ref['true_state']!=STATES[int(truth)] or \
                not np.isclose(float(score),float(ref['margin']),atol=1e-8,rtol=1e-7) or \
                int(score>0)!=int(ref['predicted_dtp']):
                raise ValueError(f'Stage35 historical replay mismatch: {key}')


def research_run():
    s21=imp(SCRIPTS/'21_group_consistent_rank_selection.py','s21_36')
    s27=imp(SCRIPTS/'27_dtp_specific_contrast_benchmark.py','s27_36')
    s28=imp(SCRIPTS/'28_dual_contrast_ablation.py','s28_36')
    s34=imp(SCRIPTS/'34_source_only_class_conditional_shrinkage.py','s34_36')
    s35=imp(SCRIPTS/'35_contrast_protected_domain_shrinkage.py','s35_36')
    old=read_csv(STAGE35)
    oldref={}
    for row in old:
        key=(row['method'],row['variant'],row['held_out_line'],row['gsm'])
        if key in oldref:raise ValueError('Duplicate Stage35 identity')
        oldref[key]=row
    if len(oldref)!=len(METHODS)*len(VARIANTS)*36:raise ValueError('Incomplete Stage35 frozen outputs')
    old_summary=json.loads(STAGE35_JSON.read_text(encoding='utf-8'))
    if old_summary.get('source_study')!='GSE193258' or \
        old_summary.get('gamma_fixed')!=GAMMA or old_summary.get('k')!=K or \
        old_summary.get('training_only') is not True or \
        old_summary.get('no_external_scoring') is not True:
        raise ValueError('Stage35 source-only protocol mismatch')
    genes,ids,X,y,groups,provenance=s21.load_data()
    X=np.asarray(X,float);y=np.asarray(y,int);groups=np.asarray(groups)
    if X.shape!=(36,19712) or len(set(genes))!=19712 or len(set(ids))!=36 or \
       Counter(y.tolist())!={0:12,1:12,2:12} or set(groups)!=set(LINES):
        raise ValueError('Unexpected GSE193258 source matrix')
    if provenance!=old_summary['source_provenance']:
        raise ValueError('Source provenance changed from Stage35')
    # This checks Stage35 panel and all Stage28 predictions before new computation.
    frozen,panels,_=s35.source_frozen(s34,genes,ids,y,groups)
    if not all(k in frozen for k in oldref if k[1]!=VARIANTS[-1]):
        raise ValueError('Prior source replay inputs have changed')

    lookup={}
    for method in METHODS:
        for v in VARIANTS:
            lookup[(method,v)]={r['gsm']:r for r in old if r['method']==method and r['variant']==v}
    records=[];shams={m:{kind:{draw:np.full(36,np.nan) for draw in range(N_DRAWS)}
                          for kind in ('shuffled','random')} for m in METHODS}
    # Randomness independent by method and outer line; fixed prior to labels/scores.
    for outer_i,outer in enumerate(LINES):
        tr=np.flatnonzero(groups!=outer);te=np.flatnonzero(groups==outer)
        if len(tr)!=27 or len(te)!=9:raise ValueError('Invalid LOCO split')
        s27.validate_train(y[tr],groups[tr])
        for method_i,method in enumerate(METHODS):
            if method==METHODS[0]:
                panel,_=s28.select_single_contrast(X[tr],y[tr],groups[tr],negative=1)
            else:panel,_=s27.select_rank_panel(X[tr],y[tr],groups[tr],mode='dual_contrast')
            if {genes[int(i)] for i in panel}!=set(panels[(method,outer)]):
                raise ValueError('Frozen Stage28 selected genes differ')
            ztr=s34.within_panel_rank(X[tr],panel)
            zte=s34.within_panel_rank(X[te],panel)
            (bscore,uscore,pscore),_trace=s35.fit_triplet(ztr,zte,y[tr],groups[tr],s34)
            replay_frozen(oldref,method,outer,[ids[i] for i in te],y[te],bscore,uscore,pscore)
            base=make_pipeline(StandardScaler(),s34.lr_model()).fit(ztr,(y[tr]==2).astype(int))
            z0=base.named_steps['standardscaler'].transform(ztr)
            z1=base.named_steps['standardscaler'].transform(zte)
            D=s35.nuisance_deviations(z0,y[tr],groups[tr])
            Vtrue=s35.contrast_vectors(z0,y[tr])
            P_fast=shrink_sham(D,Vtrue)
            P_exact,_=s35.protected_shrinkage(z0,y[tr],groups[tr],gamma=GAMMA)
            if not np.allclose(P_fast,P_exact,atol=3e-8,rtol=3e-8):
                raise AssertionError('Low-rank shrink differs from frozen Stage35')
            for kind_i,kind in enumerate(('shuffled','random')):
                for draw in range(N_DRAWS):
                    rng=np.random.default_rng(SEED+method_i*1000000+outer_i*10000+kind_i*1000+draw)
                    sham=sham_contrasts(z0,y[tr],groups[tr],kind,rng)
                    P=shrink_sham(D,sham)
                    clf=s34.lr_model().fit(z0@P,(y[tr]==2).astype(int))
                    got=clf.decision_function(z1@P)
                    for pos,idx in enumerate(te):shams[method][kind][draw][idx]=float(got[pos])
    for method in METHODS:
        for kind in ('shuffled','random'):
            for draw in range(N_DRAWS):
                score=shams[method][kind][draw]
                metric=comparison_metrics(y,score)
                records.append({'method':method,'control':kind,'draw':draw,
                                'correct':metric['correct'],'macro':f"{metric['three_condition_macro_recall']:.12g}",
                                'control_false_dtp':metric['control_false_dtp'],
                                'acute_false_dtp':metric['acute_false_dtp'],'dtp_recall':metric['dtp_recall']})
    pair=[];summary={}
    for method in METHODS:
        true_rows=lookup[(method,VARIANTS[-1])]
        base_rows=lookup[(method,VARIANTS[0])]
        by_index={idx:ids[idx] for idx in range(36)}
        sorted_id=sorted(range(36),key=lambda i:ids[i])
        true_scores=np.asarray([float(true_rows[ids[i]]['margin']) for i in range(36)])
        base_scores=np.asarray([float(base_rows[ids[i]]['margin']) for i in range(36)])
        observed=comparison_metrics(y,true_scores)
        base=comparison_metrics(y,base_scores)
        distributions={}
        for kind in ('shuffled','random'):
            subset=[r for r in records if r['method']==method and r['control']==kind]
            counts=np.asarray([r['correct'] for r in subset],int)
            distributions[kind]={
                'draws':len(counts),'min_correct':int(counts.min()),
                'median_correct':float(np.median(counts)),'max_correct':int(counts.max()),
                'mean_correct':float(counts.mean()),
                'n_at_least_real_correct':int(np.count_nonzero(counts>=observed['correct'])),
                'n_strictly_better_than_real':int(np.count_nonzero(counts>observed['correct']))}
        summary[method]={'baseline':base,'real_protected':observed,'sham_controls':distributions}
        for idx in sorted_id:
            gsm=ids[idx]
            src=int(base_scores[idx]>0)==int(y[idx]==2)
            real=int(true_scores[idx]>0)==int(y[idx]==2)
            if src==real:continue
            row={'method':method,'held_out_line':str(groups[idx]),'gsm':gsm,'state':STATES[int(y[idx])],
                 'baseline_correct':int(src),'real_protected_correct':int(real),
                 'baseline_margin':f'{base_scores[idx]:.12g}',
                 'real_protected_margin':f'{true_scores[idx]:.12g}'}
            for kind in ('shuffled','random'):
                hits=sum(int((shams[method][kind][j][idx]>0)==(y[idx]==2)) for j in range(N_DRAWS))
                row[kind+'_sham_correct_draws']=hits
            pair.append(row)
    if not pair:
        pair=[{'method':'none','held_out_line':'','gsm':'','state':'','baseline_correct':'',
               'real_protected_correct':'','baseline_margin':'','real_protected_margin':'',
               'shuffled_sham_correct_draws':'','random_sham_correct_draws':''}]
    report={'status':'EXPLORATORY_HISTORICAL_SOURCE_LOCO_POSTHOC_NULL_DIAGNOSTIC',
            'source_study':'GSE193258','n_source':36,'methods':summary,
            'fixed_gamma':GAMMA,'selected_k':K,'n_sham_draws_per_family':N_DRAWS,
            'seed':SEED,'control_protocol':{
                'shuffled':'Reassign three labels of each condition within each of the three training lines only; build sham 2D DTP-vs-Control/Acute span; true training labels used for D and LR.',
                'random':'Two independent random feature directions; same dimension, D, gamma, and LR fitting.',
                'no_target_or_outer_holdout_in_training':True,'no_external_data':True,
                'no_p_value':'Empirical distribution is NOT inferential p-value or biological replication.'},
            'frozen_source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in
                (STAGE35,STAGE35_JSON,STAGE35_DIAG,STAGE28)},
            'caveats':['Source LOCO folds already inspected repeatedly, so observed historical gain is post-hoc.',
                       'All null draws share the same 36 samples; distribution is a mechanism stress test, NOT sampling uncertainty.',
                       'Sham contrasts do not fully match geometry/alignment or strength of real class contrasts.',
                       'Train-score dispersion, cross-study calibration, biology-specific DTP identity remain unresolved.',
                       'Do not tune gamma, k, C, threshold or report best sham draw as new model.',
                       'No novelty or independent validation demonstrated.']}
    md=['# Stage 36 — Contrast-protection sham falsification audit','',
        '**Exploratory post-hoc test on four previously viewed source LOCO folds. Not a test of generalization.**','',
        f'Compare Stage35 true training DTP contrast protection with {N_DRAWS} shuffled-label and {N_DRAWS} random two-direction shams. Same feature selectors, source folds, gamma=1, C=0.01 and threshold=0. All baselines and Stage35 margins are exactly replayed. No external study is read.','',
        '| Method | Baseline correct/36 | Real-protected correct/36 | Shuffled sham min / median / max | Random sham min / median / max |',
        '|---|---:|---:|---:|---:|']
    for m in METHODS:
        r=summary[m]
        a=r['sham_controls']['shuffled'];b=r['sham_controls']['random']
        md.append(f"| {m} | {r['baseline']['correct']}/36 | {r['real_protected']['correct']}/36 | {a['min_correct']} / {a['median_correct']:.1f} / {a['max_correct']} | {b['min_correct']} / {b['median_correct']:.1f} / {b['max_correct']} |")
    md+=['','## How often do shams match/exceed the true contrast protection?','','| Method | Shuffled >= real | Random >= real |','|---|---:|---:|']
    for m in METHODS:
        r=summary[m]['sham_controls'];md.append(f"| {m} | {r['shuffled']['n_at_least_real_correct']}/{N_DRAWS} | {r['random']['n_at_least_real_correct']}/{N_DRAWS} |")
    md+=['','## Cautions','',
         '- Sham distribution is not a statistical null distribution suitable for a p-value or confidence interval.',
         '- Same four source LOCO folds were viewed while designing the algorithm; source accuracy cannot confirm generalization.',
         '- If random protections frequently match the real protection, the class-contrast-specific mechanism is weakened.',
         '- If real contrast protection exceeds most shams, it supports investigating the mechanism, not claiming originality or validated superiority.',
         '- Stage35 training-score dispersion increased in spite of reduced feature-space nuisance norm; that remains a limitation.',
         '- Stage31/33 external results were previously examined and must not be used for hyperparameter selection.',
         '- A truly new experimental source study, or appropriately independent domains, is needed for confirmatory testing.','']
    files={'36_report.md':'\n'.join(md),
           '36_summary.json':json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',
           '36_sham_draws.csv':serialize(records),
           '36_paired_samples.csv':serialize(pair)}
    immutable_write(files)
    print('Stage36 complete: Stage35 baselines and margins EXACTLY REPLAYED; source-only 2D sham controls')
    for m in METHODS:
        r=summary[m]
        print(f"{m}: baseline={r['baseline']['correct']}/36 real_protected={r['real_protected']['correct']}/36; "
              f"shuffled median={r['sham_controls']['shuffled']['median_correct']:.1f}/36 "
              f"random median={r['sham_controls']['random']['median_correct']:.1f}/36; "
              f"shams >= real (shuffled/random)={r['sham_controls']['shuffled']['n_at_least_real_correct']}/{N_DRAWS},"
              f"{r['sham_controls']['random']['n_at_least_real_correct']}/{N_DRAWS}")
    print('Saved outputs/domain_invariance/36_{report.md,summary.json,sham_draws.csv,paired_samples.csv}')
    print('POSTHOC MECHANISTIC DIAGNOSTIC: no independent validation, no novelty claim, no gamma tuning')


def self_test():
    rng=np.random.default_rng(SEED)
    yy=np.tile(np.repeat(np.arange(3),3),3)
    g=np.repeat(['A','B','C'],9)
    z=rng.normal(size=(27,K))
    z[yy==2,:20]+=1.5
    D=np.stack([z[(g==line)&(yy==j)].mean(0) for j in range(3) for line in ('A','B','C')])
    D=(D-D.reshape(3,3,K).mean(1,keepdims=True).repeat(3,axis=1).reshape(9,K))/3
    Vtrue=np.column_stack((z[yy==2].mean(0)-z[yy==0].mean(0),
                            z[yy==2].mean(0)-z[yy==1].mean(0)))
    P=shrink_sham(D,Vtrue,1)
    U,s,_=np.linalg.svd(Vtrue,full_matrices=False)
    if not np.allclose(P@U[:,:2],U[:,:2],atol=2e-8):raise AssertionError('Class span not preserved')
    if not np.allclose(shrink_sham(D,Vtrue,0),np.eye(K),atol=2e-8):
        raise AssertionError('Gamma zero not identity')
    if np.linalg.norm(D@P)>np.linalg.norm(D)+1e-9:raise AssertionError('Domain did not shrink')
    Vshuffle=sham_contrasts(z,yy,g,'shuffled',rng)
    Vrandom=sham_contrasts(z,yy,g,'random',rng)
    for v in (Vshuffle,Vrandom):
        if v.shape!=(K,2) or not np.isfinite(shrink_sham(D,v)).all():
            raise AssertionError('Sham construction failed')
    try:shrink_sham(D,np.full((K,2),np.nan))
    except ValueError:pass
    else:raise AssertionError('Nonfinite protected span not caught')
    metrics=comparison_metrics(np.tile(np.repeat(np.arange(3),3),4),np.tile([-1]*6+[1]*3,4))
    if metrics['correct']!=36 or metrics['dtp_recall']!=12 or metrics['acute_false_dtp']!=0:
        raise AssertionError('Scoring protocol wrong')
    print('SELF-TEST PASSED: dimension-matched shuffled/random sham controls, low-rank PSD shrink, gamma-zero identity, source-only metric accounting')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--exploratory-only',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.exploratory_only:research_run()
    else:parser.error('Choose --self-test or --exploratory-only; source-only exploratory work')

if __name__=='__main__':main()
