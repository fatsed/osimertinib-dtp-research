"""Stage 30 | Guarded external Day0-vs-Day28 rank-only stress test (GSE335846).

Place next to Stage29 under scripts/ and run:
  python scripts/30_gse335846_frozen_rank_transfer.py --self-test
  python scripts/30_gse335846_frozen_rank_transfer.py --exploratory-only

First audits gene annotations WITHOUT accessing expression values. Only if the
EXACT source-selected 100 genes of EACH frozen model match unique annotation
rows can it score. Otherwise it writes a report and stops without scoring.
Models/threshold source-locked from Stage29; no Day0/Day28 values or labels
are used for feature selection, model training, calibration or winner selection.

This is NOT validation of biological DTP specificity: GSE335846 has 5 Day0
and 4 Day28 FUCCI cell-cycle-sorted PC9 samples, no Acute, and possible
clone/phase dependence. It reports an exploratory external STUDY/DAY contrast.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'scripts'
OUT=ROOT/'outputs'/'external_eligibility'/'GSE335846'
RAW=ROOT/'data'/'raw'/'GSE335846'/'GSE335846_RNA-seq_for_GEO.txt.gz'
METHODS=('acute_only_rank_lr100','dual_contrast_rank_lr100')
CANDIDATE_COLS=('Feature','ID','Description','Probe')
META_COLS=('Probe','Chromosome','Start','End','Probe Strand','Feature','ID','Description',
           'Feature Strand','Type','Feature Orientation','Distance')
EXPECTED=[('GSM9822015','Day 0 G1 clone 1','Day0','G1','1'),
          ('GSM9822016','Day 0 G1 clone 7','Day0','G1','7'),
          ('GSM9822017','Day 0 S clone 1','Day0','S','1'),
          ('GSM9822018','Day 0 S clone 7','Day0','S','7'),
          ('GSM9822019','Day 0 G2 clone 1','Day0','G2','1'),
          ('GSM9822020','Day 28 G1 clone 1','Day28','G1','1'),
          ('GSM9822021','Day 28 G1 clone 7','Day28','G1','7'),
          ('GSM9822022','Day 28 S clone 1','Day28','S','1'),
          ('GSM9822023','Day 28 G2 clone 1','Day28','G2','1')]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def import_local(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(str(path))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def canonical(obj):return json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n'


def save_new(name,content):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/name;raw=content.encode('utf-8')
    if p.exists() and p.read_bytes()!=raw:
        raise FileExistsError(f'Existing report differs; refusing overwrite: {p}')
    if not p.exists():p.write_bytes(raw)


def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def csvtext(rows):
    b=io.StringIO(newline='')
    w=csv.DictWriter(b,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows);return b.getvalue()


def ensure_protocol():
    p=OUT/'29_frozen_protocol.json';q=OUT/'29_screen.json'
    if not p.exists() or not q.exists():raise FileNotFoundError('Run Stage29 --download first')
    a,b=load_json(p),load_json(q)
    if (a.get('methods')!=list(METHODS) or a.get('decision_threshold')!=0.0 or
        a.get('training_cell_lines')!=['H1975','HCC2935','HCC827'] or
        a.get('source_training_count')!=27 or a.get('target_conditions')!={'Day0':5,'Day28':4} or
        b.get('study')!='GSE335846' or b.get('protocol_sha256')!=sha(p) or
        b.get('no_expression_values_inspected') is not True):
        raise ValueError('Frozen Stage29 protocol/screen mismatch')
    if not RAW.is_file() or sha(RAW)!=b.get('supplementary_matrix_sha256'):
        raise ValueError('External matrix SHA mismatch with Stage29')
    for rel,digest in a['source_fingerprints_sha256'].items():
        path=ROOT/rel
        if not path.is_file() or sha(path)!=digest:
            raise ValueError(f'Frozen source file changed: {rel}')
    if [r['title'] for r in b['parsed']['sample_rows']] != [x[1] for x in EXPECTED]:
        raise ValueError('GEO sample order/identity changed')
    return a,b


def frozen_panels():
    expected={'acute_only_rank_lr100': ('28_selected_genes.csv',100),
              'dual_contrast_rank_lr100': ('27_selected_genes.csv',100)}
    out={}
    for method,(file,k) in expected.items():
        data=read_csv(ROOT/'outputs'/'dtp_specificity'/file)
        panel=[r['gene'] for r in data if r['method']==method and r['held_out_line']=='PC9']
        if len(panel)!=k or len(set(panel))!=k or any(not g.strip() for g in panel):
            raise ValueError(f'Incomplete frozen panel {method}')
        out[method]=panel
    return out


def annotation_audit(path,panels):
    """Read only first 12 metadata fields per row; do not inspect expression."""
    titles=[x[1] for x in EXPECTED]
    selected=set().union(*(set(p) for p in panels.values()))
    positions={col:defaultdict(list) for col in CANDIDATE_COLS}
    examples=[]
    with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f,delimiter='\t')
        header=next(reader)
        if header!=list(META_COLS)+titles:
            raise ValueError('Matrix annotations/sample column order not exact Stage29 header')
        for rindex,row in enumerate(reader):
            if len(row)!=len(header):raise ValueError(f'Wrong field count at data row {rindex+2}')
            attrs=dict(zip(META_COLS,row[:len(META_COLS)]))
            if rindex<4:examples.append({c:attrs[c][:100] for c in CANDIDATE_COLS})
            for col in CANDIDATE_COLS:
                value=attrs[col].strip()
                if value in selected:positions[col][value].append(rindex)
        nrows=rindex+1 if 'rindex' in locals() else 0
    if nrows<1000:raise ValueError(f'Unexpectedly small RNAseq matrix {nrows} rows')
    audit={'rows':nrows,'distinct_selected_genes_union':len(selected),
           'annotation_examples':examples,'candidate_columns':{},'approved_annotation_column':None}
    qualifying=[]
    for col,ix in positions.items():
        duplicate={g:len(k) for g,k in ix.items() if len(k)>1}
        coverage={m:sum(len(ix.get(g,[]))==1 for g in names) for m,names in panels.items()}
        audit['candidate_columns'][col]={'unique_match_count_by_model':coverage,
                                        'selected_gene_duplicate_rows':len(duplicate),
                                        'duplicate_example_genes':dict(list(duplicate.items())[:5]),
                                        'union_exact_unique_matches':sum(len(ix.get(g,[]))==1 for g in selected)}
        if not duplicate and all(coverage[m]==len(panel) for m,panel in panels.items()):
            qualifying.append(col)
    if qualifying:
        prefer=[c for c in ('ID','Feature','Description','Probe') if c in qualifying]
        primary=prefer[0]
        # If two columns claim exact coverage, they must point to exactly the same rows.
        for other in qualifying:
            if other == primary: continue
            if any(positions[other][g][0]!=positions[primary][g][0] for g in selected):
                audit['blocked_reason']='Two complete annotation columns map genes to different rows'
                return audit,None
        audit['approved_annotation_column']=primary
        return audit,{g:positions[primary][g][0] for g in selected}
    audit['blocked_reason']='No annotation column contains each of both exact 100-gene panels exactly once'
    return audit,None


def selected_values(path,selected_to_row):
    """Read expression values only AFTER annotation panel and provenance gate passes."""
    import numpy as np
    matrix={}
    names=[x[1] for x in EXPECTED]
    reverse={i:g for g,i in selected_to_row.items()}
    with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f,delimiter='\t')
        head=next(reader)
        if head!=list(META_COLS)+names:raise ValueError('Header changed while reading')
        for i,row in enumerate(reader):
            if i not in reverse: continue
            if len(row)!=len(head):raise ValueError('Bad selected row width')
            values=np.array([float(v) for v in row[len(META_COLS):]],dtype=float)
            if not np.isfinite(values).all() or (values<0).any():
                raise ValueError('External normalized counts have nonfinite or negative values')
            matrix[reverse[i]]=values
    if set(matrix)!=set(selected_to_row):raise ValueError('Gene rows changed / values missing')
    return matrix


def source_replay_and_fit(panels):
    """Recreate frozen PC9 fold using source-only method and exact source matrices."""
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    s21=import_local(SCRIPTS/'21_group_consistent_rank_selection.py','s21_for30')
    s27=import_local(SCRIPTS/'27_dtp_specific_contrast_benchmark.py','s27_for30')
    s28=import_local(SCRIPTS/'28_dual_contrast_ablation.py','s28_for30')
    genes,gsms,X,y,groups,provenance=s21.load_data()
    y=np.asarray(y,dtype=int);groups=np.asarray(groups)
    if X.shape!=(36,19712) or len(genes)!=19712 or Counter(y.tolist())!={0:12,1:12,2:12}:
        raise ValueError('Source raw cohort changed')
    tr=np.flatnonzero(groups!='PC9');te=np.flatnonzero(groups=='PC9')
    if len(tr)!=27 or len(te)!=9 or Counter(y[tr].tolist())!={0:9,1:9,2:9}:
        raise ValueError('Source PC9 fold mismatch')
    expected={}
    for r in read_csv(ROOT/'outputs'/'dtp_specificity'/'28_predictions.csv'):
        if r['held_out_line']=='PC9' and r['method'] in METHODS:
            key=(r['method'],r['gsm'])
            if key in expected:raise ValueError('Duplicated frozen row')
            expected[key]=r
    if len(expected)!=18:raise ValueError('Missing frozen PC9 predictions')
    models={}
    for method in METHODS:
        if method=='acute_only_rank_lr100':
            panel,info=s28.select_single_contrast(X[tr],y[tr],groups[tr],negative=1)
        else:
            panel,info=s27.select_rank_panel(X[tr],y[tr],groups[tr],mode='dual_contrast')
        selected=[genes[int(i)] for i in panel]
        if len(selected)!=100 or set(selected)!=set(panels[method]):
            raise ValueError(f'Source selection differs from frozen Stage28 panel: {method}')
        ztr=rankdata(X[tr][:,panel],axis=1,method='average')/len(panel)
        zte=rankdata(X[te][:,panel],axis=1,method='average')/len(panel)
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=.01,solver='liblinear',
                   class_weight='balanced',max_iter=3000,random_state=42))
        clf.fit(ztr,(y[tr]==2).astype(int))
        s=clf.decision_function(zte)
        for sid,truth,score in zip((gsms[i] for i in te),y[te],s):
            ref=expected[(method,sid)]
            if (ref['true_state']!=('Control','Acute','DTP')[int(truth)] or
                int(float(score)>0)!=int(ref['predicted_dtp']) or
                not np.isclose(float(score),float(ref['margin']),atol=1e-8,rtol=1e-7)):
                raise ValueError(f'Frozen Stage28 PC9 predictions not reproduced: {method}/{sid}')
        models[method]={'selected':selected,'clf':clf}
    return models


def score(models,matrix):
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score
    rows=[];summary={}
    for method in METHODS:
        selected=models[method]['selected']
        # All selected features in exact frozen/source order. Target is NOT fit/scaled.
        raw=np.asarray([matrix[g] for g in selected],dtype=float).T
        if raw.shape!=(9,100):raise ValueError('Incomplete target data')
        z=rankdata(raw,axis=1,method='average')/len(selected)
        margin=models[method]['clf'].decision_function(z)
        truth=np.asarray([int(x[2]=='Day28') for x in EXPECTED]);pred=(margin>0).astype(int)
        summary[method]={'n_day0':5,'n_day28':4,
            'day0_called_dtp':int(pred[truth==0].sum()),
            'day28_called_dtp':int(pred[truth==1].sum()),
            'correct_day_contrast':int((pred==truth).sum()),
            'balanced_accuracy_DAY_CONTRAST':float(balanced_accuracy_score(truth,pred)),
            'roc_auc_DAY_CONTRAST':float(roc_auc_score(truth,margin))}
        for i,(gsm,title,day,phase,clone) in enumerate(EXPECTED):
            rows.append({'method':method,'gsm':gsm,'sample':title,'day_group':day,
                         'phase':phase,'clone':clone,'source_only_margin':f'{float(margin[i]):.12g}',
                         'called_DTP_like':int(pred[i]),
                         'matched_day_group':int(pred[i]==truth[i])})
    return rows,summary


def main_run():
    protocol,screen=ensure_protocol()
    panels=frozen_panels()
    audit,mapping=annotation_audit(RAW,panels)
    audit.update({'status':'GENE_MAP_APPROVED_EXPLORATORY_ONLY' if mapping else 'BLOCKED_NO_SCORING',
                  'source_protocol_sha256':sha(OUT/'29_frozen_protocol.json'),
                  'target_matrix_sha256':sha(RAW),
                  'note':'Annotations only examined until gene map approval; no target label-based tuning.'})
    save_new('30_annotation_gate.json',canonical(audit))
    if mapping is None:
        report=['# Stage 30 — Gene identifier audit','',
                '**STOPPED: target was NOT scored.**','',
                f'- Matrix rows: {audit["rows"]}',
                '- No annotation column maps both selected 100-gene panels one-to-one.',
                '', '| Column | Acute-only exact | Dual exact | Duplicate selected genes |',
                '|---|---:|---:|---:|']
        for col,info in audit['candidate_columns'].items():
            a=info['unique_match_count_by_model']
            report.append(f'| {col} | {a["acute_only_rank_lr100"]}/100 | {a["dual_contrast_rank_lr100"]}/100 | {info["selected_gene_duplicate_rows"]} |')
        report+=['','**Next:** investigate annotation semantics and mapping provenance; do not impute missing genes, retune or score.','']
        save_new('30_report.md','\n'.join(report))
        print('Stage30 BLOCKED: gene ID coverage not exactly one-to-one; no target expression values or classifier scores read')
        for col,info in audit['candidate_columns'].items():
            print(f'{col}: {info["unique_match_count_by_model"]} duplicate_gene_rows={info["selected_gene_duplicate_rows"]}')
        print(f'Saved {OUT/"30_report.md"}')
        return
    # The processing metadata says size-factor-normalized gene-level counts, NOT TPM.
    # Within-panel ranks are a monotone-in-sample transform, but cross-pipeline
    # gene-length/annotation biases and clone/phase dependence remain.
    models=source_replay_and_fit(panels)
    values=selected_values(RAW,mapping)
    rows,results=score(models,values)
    payload={'status':'EXPLORATORY_EXTERNAL_DAY_CONTRAST_NOT_DTP_SPECIFICITY_VALIDATION',
        'study':'GSE335846','source_study':'GSE193258','excluded_source_line':'PC9',
        'n_target':9,'days':{'Day0':5,'Day28':4},'no_acute_arm':True,
        'gene_annotation_column':audit['approved_annotation_column'],
        'source_models_replayed_on_frozen_PC9':True,'decision_threshold':0.0,
        'source_training_n':27,'gene_coverage_both_panels':100,
        'target_processing':'SeqMonk gene-level counts size-factor normalized (not TPM)',
        'interpretation':'Study/day condition transfer ONLY; day28 != individual DTP label',
        'results':results,
        'limitations':['9 observations from 2 clonally labeled replicates and 3 cell-cycle phases; not independent n=9',
          'FUCCI sorting, dosage, duration, sequencing pipeline and gene-count vs TPM differ from source',
          'No Acute condition. Day28 is not certified DTP per sample.',
          'Scoring is exploratory only, not biological or clinical validation.',
          'Gene-rank invariance to size scaling does not remove gene-length and assay processing differences.',
          'No target-adaptive threshold, feature selection, winner selection or fitting.']}
    report=['# Stage 30 — GSE335846 frozen external Day0/Day28 exploratory transfer','',
      '**Not an independent DTP-specificity validation.** Day28 is a treatment/phase-sorted condition, not a cell-level DTP label.','',
      f'- Exact annotation mapping: **{audit["approved_annotation_column"]}**, 100/100 genes in BOTH models',
      '- Source models trained on H1975/HCC2935/HCC827 only; frozen PC9 source predictions replayed exactly.',
      '- Threshold fixed at zero; no target tuning; transfer uses within-panel gene ranks.',
      '- Target expression consists of size-factor-normalized counts, not source log2TPM.',
      '', '| Frozen model | Day0 called DTP-like /5 | Day28 called DTP-like /4 | Day-contrast bACC | Day-contrast AUC |',
      '|---|---:|---:|---:|---:|']
    for method in METHODS:
        r=results[method]
        report.append(f'| {method} | {r["day0_called_dtp"]}/5 | {r["day28_called_dtp"]}/4 | {r["balanced_accuracy_DAY_CONTRAST"]:.3f} | {r["roc_auc_DAY_CONTRAST"]:.3f} |')
    report+=['','## Important caveats','',
      '- Day 28 ≠ confirmed individual DTP status. No Acute samples exist.',
      '- Same clone/phase structures and FUCCI sorting create dependence; do not treat 9 as independent biological replicates.',
      '- Different expression units and pipelines mean this is a transfer stress test, not a calibrated clinical biomarker.',
      '- The two models were locked BEFORE target values were inspected; negative results must be reported too.','']
    save_new('30_predictions.csv',csvtext(rows))
    save_new('30_summary.json',canonical(payload))
    save_new('30_report.md','\n'.join(report))
    print('Stage30: source models replayed, full gene coverage, target Day0/Day28 scored (EXPLORATORY ONLY)')
    for method,r in results.items():
        print(f'{method}: Day0 called DTP-like {r["day0_called_dtp"]}/5; Day28 called DTP-like {r["day28_called_dtp"]}/4; day bACC={r["balanced_accuracy_DAY_CONTRAST"]:.3f}; day AUC={r["roc_auc_DAY_CONTRAST"]:.3f}')
    print(f'Saved {OUT/"30_report.md"}')


def self_test():
    import numpy as np
    from scipy.stats import rankdata
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/'toy.gz'
        header=list(META_COLS)+[r[1] for r in EXPECTED]
        records=[]
        for i in range(1001):
            gene='TEST'+str(i)
            row=[f'P{i}','1','1','2','+','ENSG'+str(i),gene,'Other','+','exon','+','0']+[str(i+j+1) for j in range(9)]
            records.append(row)
        with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(header);w.writerows(records)
        panels={METHODS[0]:['TEST'+str(i) for i in range(100)],
                METHODS[1]:['TEST'+str(i) for i in range(50,150)]}
        audit,mapping=annotation_audit(path,panels)
        assert audit['approved_annotation_column']=='ID' and len(mapping)==150
        X=selected_values(path,mapping)
        assert len(X)==150 and np.array_equal(X['TEST0'],np.arange(1,10))
        z=np.asarray([X[g] for g in panels[METHODS[0]]]).T
        assert np.array_equal(rankdata(z,axis=1)/100,rankdata(z*8,axis=1)/100)
        modified=[r.copy() for r in records]
        modified[200][6]='TEST0'  # duplicates selected ID -> blocked
        with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(header);w.writerows(modified)
        bad,m=annotation_audit(path,panels)
        assert m is None and bad['approved_annotation_column'] is None
    print('SELF-TEST PASSED: exact 9 samples, one-to-one source gene mapping, duplicate guard, rank invariance, no implicit imputation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--exploratory-only',action='store_true')
    a=p.parse_args()
    if a.self_test:return self_test()
    if not a.exploratory_only:p.error('This is an exploratory external study/day stress test. Specify --exploratory-only')
    main_run()


if __name__=='__main__':main()
