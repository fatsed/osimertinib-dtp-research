#!/usr/bin/env python3
"""Stage33 | Frozen cross-study margin decomposition, diagnostic only.

Question: After Stage32 ruled out exact missing-gene coverage as the full
explanation for the transfer drop, how much of a class-mean score shift is
attributable to the missingness transform, and which observed panel-rank
changes drive the remaining shift? Decompose the exact frozen linear margins.

No parameter fitting on target, threshold movement, feature selection or model
selection; external class labels only stratify *post hoc descriptive* means.
No independent validation, no causal gene attribution or algorithm novelty.
All 18 Stage31 target margins and 18 Stage32 source counterfactual margins
must replay before any interpretation is saved.
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

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
OUT = ROOT / 'outputs' / 'cross_study_contrast'
METHODS = ('acute_only_rank_lr100', 'dual_contrast_rank_lr100')
STATES = ('Control','Acute','DTP')


def imp(path, name):
    if not path.is_file(): raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def read_csv(path):
    if not path.is_file():raise FileNotFoundError(path)
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()


def as_csv(rows):
    if not rows:raise ValueError('Empty rows')
    f=io.StringIO(newline='')
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows)
    return f.getvalue()


def immutable(files):
    OUT.mkdir(parents=True,exist_ok=True)
    for name,body in files.items():
        p=OUT/name
        if p.exists() and p.read_bytes()!=body.encode('utf-8'):
            raise FileExistsError(f'Existing result differs; not replacing {p}')
    for name,body in files.items():
        p=OUT/name
        if not p.exists():p.write_bytes(body.encode('utf-8'))


def linear_margin_parameters(classifier):
    """Exact affine map of frozen StandardScaler+LogisticRegression pipeline."""
    if list(classifier.named_steps)!=['standardscaler','logisticregression']:
        raise ValueError('Changed fitted classifier pipeline')
    sc=classifier.named_steps['standardscaler']
    lr=classifier.named_steps['logisticregression']
    if (lr.coef_.shape!=(1,100) or sc.mean_.shape!=(100,) or
        sc.scale_.shape!=(100,) or (sc.scale_<=0).any()):
        raise ValueError('Changed linear model dimensionality')
    weights=lr.coef_[0]/sc.scale_
    offset=float(lr.intercept_[0] - np.dot(weights,sc.mean_))
    if not np.isfinite(weights).all() or not np.isfinite(offset):
        raise ValueError('Nonfinite linear model')
    return weights,offset


def decompose_means(weights,source_full,source_masked,target):
    """Class-mean additive decomposition (NOT paired sample effects)."""
    w=np.asarray(weights,dtype=float)
    a,b,c=[np.asarray(x,dtype=float) for x in (source_full,source_masked,target)]
    if w.shape!=(100,) or any(x.shape!=(100,) for x in (a,b,c)):
        raise ValueError('Expected 100-feature class means')
    missing=w*(b-a)
    residual=w*(c-b)
    total=w*(c-a)
    if not np.allclose(missing+residual,total,atol=1e-12):
        raise AssertionError('Affine decomposition not additive')
    return missing,residual,total


def ensure_replay(model,z,reference_by_gsm,ids,key,method):
    s=np.asarray(model.decision_function(z),dtype=float).reshape(-1)
    if len(s)!=len(ids) or len(set(ids))!=len(ids):raise ValueError('Wrong ID count')
    for g,x in zip(ids,s):
        r=reference_by_gsm.get((method,g))
        if r is None or not np.isclose(x,float(r[key]),rtol=1e-7,atol=1e-8):
            raise ValueError(f'Frozen margin mismatch {method}/{g} ({key})')
    return s


def keyed_unique(rows,method_key,id_key,expected_count):
    idx={}
    for r in rows:
        k=(r[method_key],r[id_key])
        if k in idx:raise ValueError(f'Duplicate reference {k}')
        idx[k]=r
    if len(idx)!=expected_count:raise ValueError(f'Wrong reference count: {len(idx)}')
    return idx


def run():
    s31=imp(SCRIPTS/'31_gse255958_frozen_contrast_transfer.py','s31_for33')
    s32=imp(SCRIPTS/'32_missing_gene_counterfactual.py','s32_for33')
    s21=imp(SCRIPTS/'21_group_consistent_rank_selection.py','s21_for33')
    s17=imp(SCRIPTS/'17_three_class_external_transfer.py','s17_for33')
    s31p=OUT/'31_summary.json';s32p=OUT/'32_summary.json'
    s31r=json.loads(s31p.read_text(encoding='utf-8'))
    s32r=json.loads(s32p.read_text(encoding='utf-8'))
    coverage=read_csv(OUT/'31_gene_coverage.csv')
    target_rows=read_csv(OUT/'31_predictions.csv')
    source32=read_csv(OUT/'32_source_pc9_mask_simulation.csv')
    s32.validate_31_frozen(s31r,target_rows)
    if (s32r.get('status')!='EXPLORATORY_COUNTERFACTUAL_NO_NEW_EXTERNAL_VALIDATION' or
        s32r.get('number_of_random_masks')!=200 or
        s32r.get('frozen_models_and_threshold_unchanged') is not True):
        raise ValueError('Unexpected Stage32 output/protocol')
    for name,path in (('stage31_summary',s31p),('stage31_predictions',OUT/'31_predictions.csv'),
                      ('stage31_gene_coverage',OUT/'31_gene_coverage.csv')):
        if s32r['input_sha256'].get(name)!=digest(path):
            raise ValueError(f'Stage32 input {name} changed')
    if s31r['input_sha256'].get('target_archive')!=digest(s17.TARGET):
        raise ValueError('GSE255958 official archive modified')

    panels=s31.panels_from_frozen()
    masks=s32.masks_from_31(panels,coverage,target_rows)
    models,source_ref,provenance=s31.source_model_replay(panels)
    genes,gsms,X,y,groups,curpro=s21.load_data()
    y=np.asarray(y,dtype=int);groups=np.asarray(groups)
    pc9idx=np.flatnonzero(groups=='PC9')
    if provenance!=curpro or X.shape!=(36,19712) or len(pc9idx)!=9:
        raise ValueError('Source replay provenance changed')
    if Counter(y[pc9idx].tolist())!={0:3,1:3,2:3}:
        raise ValueError('Missing source PC9 groups')
    src_ids=[gsms[int(i)] for i in pc9idx]
    src_states=np.asarray([STATES[int(y[i])] for i in pc9idx])
    source32_ref=keyed_unique(source32,'method','gsm',18)
    target_ref=keyed_unique(target_rows,'method','gsm',18)
    source_ref_by_id=keyed_unique(source_ref,'method','source_gsm',18)

    target_ids=[gsm for state in STATES for gsm in s17.GROUPS[state]]
    target_labels={gsm:state for state in STATES for gsm in s17.GROUPS[state]}
    if len(target_ids)!=9 or len(set(target_ids))!=9 or \
       any(target_ref[(m,g)]['true_state']!=target_labels[g] for m in METHODS for g in target_ids):
        raise ValueError('Target sample mapping changed')
    s31.validate_prior_target(target_ids,target_labels)
    raw=s17.read_target_tar(s17.TARGET)
    if set(raw)!=(set(target_ids)|set(s17.GROUPS['Resistant'])):
        raise ValueError('Target archive rows changed')
    target={g:raw[g]['data'] for g in target_ids}
    tgt_states=np.asarray([target_labels[g] for g in target_ids])
    gene_index={g:i for i,g in enumerate(genes)}
    if len(gene_index)!=19712:raise ValueError('Source duplicate genes')

    detail=[];summaries={};rankings={}
    for method in METHODS:
        model=models[method]['classifier'];panel=models[method]['panel']
        x_source=X[pc9idx][:,[gene_index[g] for g in panel]]
        z_full=s32.mask_rank_matrix(x_source,set(),models[method]['train_feature_mean'])
        z_mask=s32.mask_rank_matrix(x_source,masks[method],models[method]['train_feature_mean'])
        z_tgt,cov=s31.target_rank_features(target,target_ids,models[method])
        w,bias=linear_margin_parameters(model)
        for z in (z_full,z_mask,z_tgt):
            if not np.allclose(model.decision_function(z),z@w+bias,atol=1e-9):
                raise AssertionError('Frozen LR margin cannot be decomposed exactly')
        full_scores=ensure_replay(model,z_full,source_ref_by_id,src_ids,'source_margin',method)
        masked_scores=ensure_replay(model,z_mask,source32_ref,src_ids,
                          'source_pc9_with_external_missing_genes_margin',method)
        tgt_scores=ensure_replay(model,z_tgt,target_ref,target_ids,'source_only_margin',method)
        cov_map={r['gsm']:r for r in cov}
        if any(int(target_ref[(method,g)]['panel_genes_available'])!=cov_map[g]['available']
               for g in target_ids):
            raise ValueError('Coverage changed from Stage31')
        if any(target_ref[(method,g)]['true_state']!=target_labels[g] for g in target_ids):
            raise ValueError('Target annotations disagree with Stage31')
        present={g: (j not in masks[method]) for j,g in enumerate(panel)}
        summaries[method]={}
        for state in STATES:
            ix_a=src_states==state;ix_t=tgt_states==state
            if ix_a.sum()!=3 or ix_t.sum()!=3:raise ValueError('Wrong class sample sizes')
            mean_a=z_full[ix_a].mean(axis=0)
            mean_b=z_mask[ix_a].mean(axis=0)
            mean_c=z_tgt[ix_t].mean(axis=0)
            missing,residual,total=decompose_means(w,mean_a,mean_b,mean_c)
            shift=float(tgt_scores[ix_t].mean()-full_scores[ix_a].mean())
            if not np.isclose(shift,total.sum(),atol=1e-9):
                raise AssertionError('Study margin shift not exactly explained by linear features')
            source_to_mask=float(masked_scores[ix_a].mean()-full_scores[ix_a].mean())
            mask_to_external=float(tgt_scores[ix_t].mean()-masked_scores[ix_a].mean())
            if not np.isclose(missing.sum(),source_to_mask,atol=1e-9) or \
               not np.isclose(residual.sum(),mask_to_external,atol=1e-9):
                raise AssertionError('Missingness/study decomposition mismatch')
            summaries[method][state]={
                'source_full_mean_margin':float(full_scores[ix_a].mean()),
                'source_masked_mean_margin':float(masked_scores[ix_a].mean()),
                'target_mean_margin':float(tgt_scores[ix_t].mean()),
                'mask_effect':source_to_mask,
                'residual_cross_study_shift_after_mask':mask_to_external,
                'total_shift':shift,
                'rank_feature_linear_decomposition_exact':True
            }
            for j,g in enumerate(panel):
                detail.append({'method':method,'state':state,'gene':g,
                   'available_in_external':int(present[g]),
                   'effective_linear_weight':f'{float(w[j]):.12g}',
                   'source_pc9_full_mean_rank':f'{float(mean_a[j]):.12g}',
                   'source_pc9_masked_mean_rank':f'{float(mean_b[j]):.12g}',
                   'external_pc9_mean_rank':f'{float(mean_c[j]):.12g}',
                   'mask_contribution_to_mean_margin_shift':f'{float(missing[j]):.12g}',
                   'residual_contribution_to_mean_margin_shift':f'{float(residual[j]):.12g}',
                   'total_contribution_to_mean_margin_shift':f'{float(total[j]):.12g}'})
        neg_src=float(np.mean([summaries[method][s]['source_masked_mean_margin'] for s in STATES[:2]]))
        neg_tgt=float(np.mean([summaries[method][s]['target_mean_margin'] for s in STATES[:2]]))
        gap_src=summaries[method]['DTP']['source_masked_mean_margin']-neg_src
        gap_tgt=summaries[method]['DTP']['target_mean_margin']-neg_tgt
        rankings[method]={'source_masked_DTP_minus_negative_mean_margin_gap':gap_src,
                          'external_DTP_minus_negative_mean_margin_gap':gap_tgt,
                          'gap_change':gap_tgt-gap_src}

    if len(detail)!=2*3*100:raise AssertionError('Incomplete per-gene decompositions')
    md=['# Stage 33 — Frozen score-shift decomposition','',
        '**Exploratory diagnostic of the PREVIOUSLY INSPECTED GSE255958: not independent validation.**','',
        'Source training: 27 non-PC9 samples. Compare separate groups of 3 source PC9 and 3 external PC9 samples.',
        'All genes, models, coefficients, ranks, missingness policy, thresholds and historical margins are replayed exactly.',
        'Group MEAN margins are used because linear additive decomposition is exact for means, not medians.','',
        '## Additive group-mean margin shifts','',
        '| Model | Group | Source full | Source masked | External | Missing-gene effect | Residual cross-study shift |',
        '|---|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for st in STATES:
            r=summaries[method][st]
            md.append(f'| {method} | {st} | {r["source_full_mean_margin"]:+.4f} | '
                      f'{r["source_masked_mean_margin"]:+.4f} | {r["target_mean_margin"]:+.4f} | '
                      f'{r["mask_effect"]:+.4f} | {r["residual_cross_study_shift_after_mask"]:+.4f} |')
    md+=['','## DTP vs negative (Control+Acute) mean-margin gap','',
          '| Model | Source masked gap | External gap | Change |','|---|---:|---:|---:|']
    for method in METHODS:
        r=rankings[method]
        md.append(f'| {method} | {r["source_masked_DTP_minus_negative_mean_margin_gap"]:+.4f} | '
                  f'{r["external_DTP_minus_negative_mean_margin_gap"]:+.4f} | {r["gap_change"]:+.4f} |')
    md+=['','## Largest descriptive DTP residual contributions (NOT causal biomarkers)','',
         'Terms are weight × external-minus-masked-source group MEAN panel rank. Positive and negative terms cancel.',
         'These compare NONPAIRED biological/experimental cohorts; processing, duration and cell conditions differ.','']
    for method in METHODS:
        md.append(f'**{method}**')
        dtp=[r for r in detail if r['method']==method and r['state']=='DTP']
        dtp=sorted(dtp,key=lambda r:abs(float(r['residual_contribution_to_mean_margin_shift'])),reverse=True)
        md+=['','| Gene | On target | Signed contribution to DTP shift |','|---|---|---:|']
        for r in dtp[:10]:
            md.append(f'| {r["gene"]} | {"yes" if int(r["available_in_external"]) else "no"} | '
                      f'{float(r["residual_contribution_to_mean_margin_shift"]):+.4f} |')
        md+=['']
    md+=['## Interpretation and non-claims','',
         '- This is linear score accounting, **not** causal attribution or isolation of batch effect.',
         '- A change in class-mean gap is distinct from a constant shift shared by every class.',
         '- Do not recalibrate the threshold, modify the gene panel or choose a winner based on this study.',
         '- Different observed conditions and treatment durations may cause biological differences independent of technical batch effects.',
         '- The nine previously inspected external rows do not establish DTP specificity, algorithm novelty or independent generalization.',
         '- Candidate future algorithm mechanisms should be derived and tested on source domains without consulting target labels.','']
    report={
       'status':'EXPLORATORY_SCORE_ACCOUNTING_NOT_CALIBRATION',
       'target':'GSE255958','source':'GSE193258','methods':list(METHODS),
       'training_without_PC9':True,'frozen_threshold':0.0,'source_n':9,'external_n':9,
       'no_retraining_or_target_adaptation':True,'target_class_labels_only_for_posthoc_group_summaries':True,
       'all_stage31_and_stage32_margins_replayed':True,
       'input_sha256':{n:digest(OUT/n) for n in (
        '31_summary.json','31_predictions.csv','31_gene_coverage.csv',
        '32_summary.json','32_source_pc9_mask_simulation.csv')},
       'target_archive_sha256':digest(s17.TARGET),
       'source_provenance':curpro,
       'means':summaries,'class_separation':rankings,
       'limitations':md[md.index('## Interpretation and non-claims')+2:-1]
    }
    immutable({'33_report.md':'\n'.join(md),
               '33_summary.json':json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',
               '33_gene_contributions.csv':as_csv(detail)})
    print('Stage33 complete: exact Stage31/32 source and external margin replay; additive mean-score decomposition')
    for method in METHODS:
        r=summaries[method]['DTP'];g=rankings[method]
        print(f'{method}: DTP mean shift from missingness={r["mask_effect"]:+.4f}; '
              f'residual study shift={r["residual_cross_study_shift_after_mask"]:+.4f}; '
              f'DTP-minus-negative gap source masked={g["source_masked_DTP_minus_negative_mean_margin_gap"]:+.4f} '
              f'external={g["external_DTP_minus_negative_mean_margin_gap"]:+.4f}')
    print('Saved outputs/cross_study_contrast/33_{report.md,summary.json,gene_contributions.csv}')
    print('EXPLORATORY: no threshold change, model selection, target calibration or novelty claim')


def self_test():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    rng=np.random.default_rng(330)
    x=rng.normal(size=(36,100))
    labels=np.array([0]*24+[1]*12)
    x[labels==1,0]+=2
    pipe=make_pipeline(StandardScaler(),LogisticRegression(C=.01,solver='liblinear',class_weight='balanced',random_state=42))
    pipe.fit(x,labels)
    w,c=linear_margin_parameters(pipe)
    z=rng.normal(size=(9,100))
    assert np.allclose(pipe.decision_function(z),z@w+c,atol=1e-10)
    a,b,tar=[rng.normal(size=(100,)) for _ in range(3)]
    missing,resid,total=decompose_means(w,a,b,tar)
    assert np.isclose((missing+resid).sum(),w@(tar-a),atol=1e-12)
    assert np.isclose(total.sum(),(tar@w+c)-(a@w+c),atol=1e-12)
    try:decompose_means(w,np.zeros(99),b,tar)
    except ValueError:pass
    else:raise AssertionError('Shape guard failed')
    print('SELF-TEST PASSED: exact affine margin decomposition, additive class-mean shift, invalid-shape guard')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--exploratory-only',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test()
    elif a.exploratory_only:run()
    else:p.error('Source/target diagnostic requires --exploratory-only')

if __name__=='__main__':main()
