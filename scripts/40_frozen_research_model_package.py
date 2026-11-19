#!/usr/bin/env python3
"""Stage 40: export and independently score a FROZEN research-only rank classifier.

This is *technical packaging*, not a new algorithm, model choice, independent
validation, or clinical predictor. Export source-fitted PC9-excluded models for
both selectors and BOTH original/protected variants. Before any package is
written, reproduce frozen Stage31/35 SOURCE PC9 margins and Stage39 previously
inspected external margins exactly. The external study is REPLAYED ONLY, and
NEVER used for fitting, model selection, calibration or thresholds.

Run from the repository root:
    python scripts/40_frozen_research_model_package.py --self-test
    python scripts/40_frozen_research_model_package.py --export

Optional inference on new preprocessed gene-level expression matrix (gene and
sample columns, TSV/CSV; one gene symbol per row, one sample per column):
    python scripts/40_frozen_research_model_package.py --predict path/to/matrix.tsv

The package contains EXACT score-equivalent linear predictors, ordered panels,
source-only missing-gene rank means, provenance and restrictions. No pickle.
A model output is only 'DTP-like vs non-DTP-like' under observed conditions.
Use the full scientific reports to interpret it; DO NOT use clinically.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import platform
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'research_model_package'
SCRIPT_39 = ROOT / 'scripts' / '39_frozen_protected_external_transfer.py'
METHODS = ('acute_only_rank_lr100', 'dual_contrast_rank_lr100')
VARIANTS = ('stage28_baseline', 'contrast_protected_shrinkage_gamma1')
K = 100
MIN_COVERAGE = 90
THRESHOLD = 0.0
PACKAGE_FILE = '40_frozen_research_models.json'
AUDIT_FILE = '40_replay_audit.json'


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def module(path: Path, label: str):
    if not path.is_file():raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location(label,path)
    if spec is None or spec.loader is None:raise ImportError(path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def frozen_linear_model(panel, source_means, scaler, projection, classifier):
    """Collapse StandardScaler + fixed linear transform + LR to rank-space weights."""
    panel=list(panel);mean=np.asarray(source_means,dtype=float)
    mu=np.asarray(scaler.mean_,dtype=float)
    scale=np.asarray(scaler.scale_,dtype=float)
    P=np.asarray(projection,dtype=float)
    coef=np.asarray(classifier.coef_,dtype=float)
    intercept=np.asarray(classifier.intercept_,dtype=float)
    if (len(panel)!=K or len(set(panel))!=K or mean.shape!=(K,) or mu.shape!=(K,)
        or scale.shape!=(K,) or P.shape!=(K,K) or coef.shape!=(1,K)
        or intercept.shape!=(1,) or not all(isinstance(g,str) and g for g in panel)
        or not np.isfinite(mean).all() or not np.isfinite(mu).all()
        or not np.isfinite(P).all() or not np.isfinite(coef).all()
        or not np.isfinite(scale).all() or np.any(scale<=0)):
        raise ValueError('Malformed frozen linear model input')
    weights=(P@coef[0])/scale
    bias=float(intercept[0]-np.dot(mu,weights))
    if not np.isfinite(weights).all() or not math.isfinite(bias):
        raise ValueError('Nonfinite exported model')
    return {'panel_genes':panel, 'source_mean_panel_ranks':mean.tolist(),
            'rank_linear_weights':weights.tolist(), 'rank_linear_intercept':bias,
            'n_selected_genes':K, 'minimum_gene_coverage':MIN_COVERAGE,
            'threshold':THRESHOLD}


def score_feature_matrix(model, z):
    z=np.asarray(z,dtype=float)
    w=np.asarray(model['rank_linear_weights'],dtype=float)
    if z.ndim!=2 or z.shape[1]!=K or w.shape!=(K,) or not np.isfinite(z).all():
        raise ValueError('Invalid rank score features')
    return z@w + float(model['rank_linear_intercept'])


def panel_rank_for_sample(model, measured):
    """Exactly Stage31 missing-gene policy: rank observed panel only, impute source means."""
    genes=model['panel_genes']
    z=np.asarray(model['source_mean_panel_ranks'],dtype=float).copy()
    if len(genes)!=K or len(set(genes))!=K or z.shape!=(K,) or not np.isfinite(z).all():
        raise ValueError('Invalid model feature identifiers')
    present=[i for i,g in enumerate(genes) if g in measured]
    if len(present)<int(model['minimum_gene_coverage']):
        raise ValueError(f"Only {len(present)}/{K} required genes available, minimum {model['minimum_gene_coverage']}")
    vals=np.asarray([measured[genes[i]] for i in present],dtype=float)
    if not np.isfinite(vals).all() or np.any(vals<0):
        raise ValueError('Nonfinite/negative gene expression is not accepted')
    z[present]=rankdata(vals,method='average')/len(present)
    return z, len(present)


def payload_score(model, measured):
    z,coverage=panel_rank_for_sample(model,measured)
    s=float(score_feature_matrix(model,z[None,:])[0])
    return s,coverage


def immutable_write(path: Path, body: str):
    encoded=body.encode('utf-8')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        if path.read_bytes()!=encoded:
            raise FileExistsError(f'Existing artifact differs; not overwriting: {path}')
        return
    path.write_bytes(encoded)


def export():
    import sklearn
    s39=module(SCRIPT_39,'stage39_for40')
    locked=s39.verify_prior_files()  # Reference code/output fingerprints from Stage39.
    mods=s39.import_source_scripts()
    s31=mods['s31']
    panels=s31.panels_from_frozen()
    baseline,baseline_rows,provenance=s31.source_model_replay(panels)
    fitted,source_replay_rows,source_provenance=s39.source_models(mods,panels,baseline,baseline_rows)
    if provenance!=source_provenance:raise ValueError('Source provenance changed')
    existing=read_csv(s39.OUT/'39_predictions.csv')
    ref={}
    for r in existing:
        key=(r['method'],r['variant'],r['gsm'])
        if key in ref:raise ValueError('Duplicate Stage39 prediction row')
        ref[key]=r
    if len(ref)!=len(METHODS)*len(VARIANTS)*9:
        raise ValueError('Stage39 must contain 36 frozen external scores')
    target_ids,truth,target_data=s39.target_archive(mods,s31)
    if len(target_ids)!=9:raise ValueError('Frozen Stage39 target changed')
    models={};audits=[]
    for method in METHODS:
        group=fitted[method]
        base=group['baseline']
        protected=group['protected']
        ordered=base['panel']
        if ordered!=protected['panel']:raise ValueError('Baseline/protected panel changed')
        pipeline=base['classifier']
        bs=pipeline.named_steps['standardscaler']
        bc=pipeline.named_steps['logisticregression']
        for variant in VARIANTS:
            if variant==VARIANTS[0]:
                ex=frozen_linear_model(ordered,base['train_feature_mean'],bs,np.eye(K),bc)
            else:
                ex=frozen_linear_model(ordered,protected['train_feature_mean'],
                                       protected['scaler'],protected['P'],protected['clf'])
            model_name=method+'__'+variant
            # Verify exact SOURCE-PC9 replay; never train on the nine test samples.
            src_rows=[r for r in source_replay_rows if r['method']==method and r['variant']==variant]
            if len(src_rows)!=9:raise ValueError('Source replay row count changed')
            # held-out source ranks are computed independently from the source matrix.
            genes,source_ids,X,y,groups,_=mods['s21'].load_data()
            te=np.flatnonzero(np.asarray(groups)=='PC9')
            indices=np.asarray([genes.index(g) for g in ordered],dtype=int)
            from_source=rankdata(np.asarray(X)[te][:,indices],axis=1,method='average')/K
            margins=score_feature_matrix(ex,from_source)
            by_gsm={r['gsm']:r for r in src_rows}
            if len(by_gsm)!=9:raise ValueError('Duplicate source PC9 replay')
            for idx,v in zip(te,margins):
                sid=source_ids[int(idx)]
                if sid not in by_gsm or not np.isclose(v,by_gsm[sid]['margin'],rtol=1e-7,atol=1e-8):
                    raise AssertionError(f'Packaged source margin differs {model_name}/{sid}')
            # Verify exact historic Stage39 EXTERNAL predictions, but never use
            # these already-viewed target scores to fit or choose a winner.
            for sid in target_ids:
                value,cov=payload_score(ex,target_data[sid])
                old=ref.get((method,variant,sid))
                if (old is None or old['true_state']!=truth[sid] or
                    not np.isclose(value,float(old['score']),rtol=1e-7,atol=1e-8) or
                    int(value>0)!=int(old['predicted_dtp']) or
                    cov!=int(old['genes_available'])):
                    raise AssertionError(f'Packaged external margin differs {model_name}/{sid}')
                audits.append({'model':model_name,'gsm':sid,'source_coverage':cov,
                               'replayed_external_score':float(value)})
            models[model_name]=ex
    model_payload={
        'version':1,'status':'RESEARCH_ONLY_NOT_INDEPENDENTLY_VALIDATED',
        'training_study':'GSE193258','training_cell_lines':['H1975','HCC2935','HCC827'],
        'n_train':27, 'source_heldout_line':'PC9',
        'target_study_replay_only':'GSE255958',
        'training_target':'binary observed DTP vs Control and Acute',
        'observed_state_duration_confounded':True,
        'candidate_policy':'FOUR RESEARCH MODELS; no posthoc winner selection',
        'methods':list(METHODS),'variants':list(VARIANTS),
        'model_parameters':{'k':K,'C':0.01,'gamma_protected':1.0,'threshold':0.0,'min_gene_coverage':MIN_COVERAGE},
        'rank_policy':'rank measured panel values within sample divided by available gene count; fill missing slots with source TRAINING rank means',
        'models':models,
        'code_fingerprints_sha256':{**locked,
          'scripts/39_frozen_protected_external_transfer.py':sha(SCRIPT_39),
          'outputs/cross_study_contrast/39_predictions.csv':sha(s39.OUT/'39_predictions.csv'),
          'outputs/cross_study_contrast/39_summary.json':sha(s39.OUT/'39_summary.json')},
        'source_provenance':provenance,
        'python_version':platform.python_version(),
        'numpy_version':np.__version__,'scikit_learn_version':sklearn.__version__,
        'limitations':[
            'No untouched independent source/target validation; do not infer generalization.',
            'Source PC9 and GSE255958 have previously been inspected.',
            'Protected model failed to improve fixed-threshold external DTP recall.',
            'Target labels do not enter any fitting, feature choice or calibration.',
            'No clinical intended use, no causal DTP biomarker, no novelty claim.'
        ]
    }
    package=json.dumps(model_payload,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    audit=json.dumps({'status':'REPLAY_VERIFIED_TECHNICAL_EXPORT','source_PC9_scores_replayed':36,
                      'external_GSE255958_scores_replayed':36,
                      'package_payload_sha256':hashlib.sha256(package.encode('utf-8')).hexdigest(),
                      'n_models':4,'n_external_prediction_checks':len(audits),
                      'source_only_training':True,'external_used_only_to_verify_score_replay':True,
                      'examples':audits[:4]},ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    # Preflight BOTH artifacts before writing either one.
    for name,body in ((PACKAGE_FILE,package),(AUDIT_FILE,audit)):
        p=OUT/name
        if p.exists() and p.read_bytes()!=body.encode('utf-8'):
            raise FileExistsError(f'Refusing to alter previously exported package: {p}')
    immutable_write(OUT/PACKAGE_FILE,package)
    immutable_write(OUT/AUDIT_FILE,audit)
    print('Stage40 EXPORT COMPLETE: 4 research-only linear rank models, source PC9 + external frozen margins verified')
    print(f'Saved {OUT/PACKAGE_FILE} and {OUT/AUDIT_FILE}')
    print('NO model selection, target fitting, new validation or clinical predictor claim')


def read_wide_matrix(path: Path):
    """Parse a gene-by-sample TSV/CSV, no labels or patient/clinical data needed."""
    if not path.is_file():raise FileNotFoundError(path)
    delimiter=',' if path.suffix.lower()=='.csv' else '\t'
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f,delimiter=delimiter)
        h=next(reader,None)
        if not h or len(h)<2 or h[0].strip() not in ('gene','gene_id','gene_symbol'):
            raise ValueError('Input format: first column gene / gene_id / gene_symbol; remaining columns sample IDs')
        samples=[x.strip() for x in h[1:]]
        if any(not x for x in samples) or len(set(samples))!=len(samples):
            raise ValueError('Duplicate/empty sample IDs')
        data={sample:{} for sample in samples}
        seen=set()
        for ri,row in enumerate(reader,start=2):
            if len(row)!=len(h):raise ValueError(f'Row {ri} has incorrect number of columns')
            gene=row[0].strip()
            if not gene or gene in seen:raise ValueError(f'Duplicate/empty gene in input row {ri}: {gene}')
            seen.add(gene)
            for sample,value in zip(samples,row[1:]):
                try: x=float(value)
                except ValueError:raise ValueError(f'Nonnumeric expression {gene}/{sample}')
                if not math.isfinite(x) or x<0:raise ValueError(f'Invalid expression {gene}/{sample}')
                data[sample][gene]=x
        if len(seen)<MIN_COVERAGE:raise ValueError('Input expression matrix has too few genes')
    return samples,data


def validate_package(pkg):
    if (pkg.get('version')!=1 or pkg.get('status')!='RESEARCH_ONLY_NOT_INDEPENDENTLY_VALIDATED'
        or pkg.get('n_train')!=27 or pkg.get('model_parameters',{}).get('threshold')!=0.0
        or set(pkg.get('models',{}))!={m+'__'+v for m in METHODS for v in VARIANTS}):
        raise ValueError('Unrecognized package; refusing to score')
    for key,m in pkg['models'].items():
        p=m.get('panel_genes',[]);w=np.asarray(m.get('rank_linear_weights',[]),dtype=float)
        means=np.asarray(m.get('source_mean_panel_ranks',[]),dtype=float)
        if (len(p)!=K or len(set(p))!=K or w.shape!=(K,) or means.shape!=(K,)
            or not np.isfinite(w).all() or not np.isfinite(means).all()
            or not math.isfinite(float(m.get('rank_linear_intercept',float('nan'))))
            or m.get('minimum_gene_coverage')!=MIN_COVERAGE or m.get('threshold')!=0.0):
            raise ValueError(f'Corrupt model record {key}')


def predict(matrix_path: Path, model_path: Path, output: Path):
    pkg=json.loads(model_path.read_text(encoding='utf-8'))
    validate_package(pkg)
    sample_ids,samples=read_wide_matrix(matrix_path)
    rows=[]
    for model_name,model in sorted(pkg['models'].items()):
        for sid in sample_ids:
            score,available=payload_score(model,samples[sid])
            rows.append({'model':model_name,'sample_id':sid,
                         'DTP_like_margin':f'{score:.12g}',
                         'DTP_like_call':int(score>THRESHOLD),
                         'genes_available':available,'genes_missing':K-available,
                         'status':'RESEARCH_ONLY_UNVALIDATED'})
    out=io.StringIO(newline='')
    w=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows)
    immutable_write(output,out.getvalue())
    print(f'RESEARCH-ONLY scores written: {len(rows)} predictions -> {output}')
    print('No predicted probability, clinical response or independent validation implied.')


def self_test():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng=np.random.default_rng(4001)
    y=np.tile(np.repeat(np.arange(3),3),3)
    x=rng.normal(size=(27,K))+(y==2)[:,None]*rng.normal(0,.6,size=(1,K))
    z=rankdata(x,axis=1,method='average')/K
    scaler=StandardScaler().fit(z)
    for kind in ('base','protected'):
        A=rng.normal(size=(K,4));P=np.eye(K) if kind=='base' else np.eye(K)-.005*(A@A.T)
        clf=LogisticRegression(C=.01,solver='liblinear',class_weight='balanced',random_state=42).fit(scaler.transform(z)@P,y==2)
        m=frozen_linear_model([f'GENE{i}' for i in range(K)],z.mean(axis=0),scaler,P,clf)
        test=rng.uniform(0,1,size=(5,K))
        direct=clf.decision_function(scaler.transform(test)@P)
        collapsed=score_feature_matrix(m,test)
        if not np.allclose(direct,collapsed,rtol=1e-9,atol=1e-9):
            raise AssertionError('Linear collapse changes scores')
        present={f'GENE{i}':float(i) for i in range(8,K)}
        zz,cov=panel_rank_for_sample(m,present)
        if cov!=92 or not np.allclose(zz[8:],rankdata(range(8,100))/92):
            raise AssertionError('Target rank handling differs')
        if not np.allclose(zz[:8],z.mean(axis=0)[:8]):
            raise AssertionError('Missing genes not imputed from source')
        try:
            panel_rank_for_sample(m,{f'GENE{i}':float(i) for i in range(11,K)})
            raise AssertionError('Coverage failure did not stop')
        except ValueError:pass
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        matrix=root/'input.tsv'
        matrix.write_text('gene\tS1\tS2\n'+''.join(f'GENE{i}\t{i+1}\t{i+2}\n' for i in range(100)),encoding='utf-8')
        sm,d=read_wide_matrix(matrix)
        if sm!=['S1','S2'] or len(d['S1'])!=100:raise AssertionError('Wrong input reading')
        for part in ('gene\tS1\nGENE1\t1\nGENE1\t2\n', 'gene\tS1\nGENE1\t-1\n'):
            t=root/'invalid.tsv';t.write_text(part)
            try:read_wide_matrix(t);raise AssertionError('Bad input not rejected')
            except ValueError:pass
        f=root/'freeze.txt';immutable_write(f,'hello');immutable_write(f,'hello')
        try:immutable_write(f,'goodbye');raise AssertionError('Overwrite guard failed')
        except FileExistsError:pass
    print('SELF-TEST PASSED: exact linear margin collapse, 92/100 panel rank, source-only imputation, coverage stop, wide matrix parser, immutable output')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--export',action='store_true')
    mode.add_argument('--predict',type=Path,metavar='GENE_MATRIX_TSV_OR_CSV')
    p.add_argument('--model-path',type=Path,default=OUT/PACKAGE_FILE)
    p.add_argument('--output',type=Path,default=OUT/'40_new_sample_scores.csv')
    args=p.parse_args()
    if args.self_test:self_test()
    elif args.export:export()
    else:predict(args.predict,args.model_path,args.output)

if __name__=='__main__':main()
