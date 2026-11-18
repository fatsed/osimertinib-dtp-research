"""Stage 17 | Source-trained 3-class transfer to GSE255958 (PC9).

A non-confirmatory, previously inspected external-study stress test. Fit the
unchanged Stage-16 3-class recipes on all 36 source (Control/Acute/DTP)
GSE193258 observations. Evaluate 9 PC9 observations (3 per class) from the
public GSE255958 RSEM TPM archive. The 3 Resistant samples are checked for
exclusion, never mislabeled as DTP.

No target genes/labels are used for source feature selection, fitting, tuning
or scaling. Source gene universe is the frozen GSE193258 19,712 symbols.
Target RSEM TPM is log2(TPM+0.01) ONLY for the raw-expression comparator;
rank methods use monotone ranks, with explicit missing-gene handling.

NOT independent validation, biological certification or a new algorithm.
GSE255958 samples were examined previously; durations/doses, cell-line and
study effects differ. The source classes themselves confound duration.

From repository root:
  python scripts/17_three_class_external_transfer.py --self-test
  python scripts/17_three_class_external_transfer.py --exploratory-only
  python scripts/17_three_class_external_transfer.py --download --exploratory-only
  python scripts/17_three_class_external_transfer.py --exploratory-only --pc9-excluded

Input: data/raw/GSE255958/GSE255958_RAW.tar (public GEO, ~50 MB)
Outputs under outputs/three_class_external/; does not alter old results.
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
import re
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TARGET = ROOT / "data/raw/GSE255958/GSE255958_RAW.tar"
OUT = ROOT / "outputs/three_class_external"
URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE255nnn/GSE255958/suppl/GSE255958_RAW.tar"
GROUPS = {
    "Control": [f"GSM808{n}" for n in (3374,3375,3376)],
    "Acute": [f"GSM808{n}" for n in (3377,3378,3379)],
    "DTP": [f"GSM808{n}" for n in (3380,3381,3382)],
    "Resistant": [f"GSM808{n}" for n in (3383,3384,3385)],
}
STATES = ("Control", "Acute", "DTP")
METHODS = (
    "rank_full_lr100_3class", "rank_selected_lr100_3class",
    "signed_rank_ovr_full_3class", "raw_log2tpm_lr500_3class",
)
REPEAT_ID = re.compile(r"^(.+)_\1$")
GSM_RE = re.compile(r"GSM\d+")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_py(path, name):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_name(gene):
    gene = gene.strip()
    match = REPEAT_ID.fullmatch(gene)
    return match.group(1) if match else gene


def parse_rsem(gzip_stream):
    """Read one GSM's full processed RSEM results, excluding ambiguous IDs."""
    with gzip.GzipFile(fileobj=gzip_stream, mode="rb") as unzipped:
        with io.TextIOWrapper(unzipped, encoding="utf-8-sig", newline="") as fh:
            rows = csv.DictReader(fh, delimiter="\t")
            if not rows.fieldnames or not {"gene_id", "TPM"}.issubset(rows.fieldnames):
                raise ValueError("GSE255958 RSEM sample lacks gene_id/TPM columns")
            data = {}
            ambiguous = set()
            originals = set()
            for rec in rows:
                raw = rec["gene_id"].strip()
                gene = normalize_name(raw)
                if not gene or raw in originals:
                    raise ValueError("Empty/repeated original RSEM gene identifier")
                originals.add(raw)
                try:
                    tpm = float(rec["TPM"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Nonnumeric TPM for {raw}") from exc
                if not math.isfinite(tpm) or tpm < 0:
                    raise ValueError(f"Negative/nonfinite TPM for {raw}")
                if gene in data or gene in ambiguous:
                    ambiguous.add(gene)
                    data.pop(gene, None)
                else:
                    data[gene] = tpm
    if len(data) < 15000:
        raise ValueError(f"Too few valid RSEM genes: {len(data)}")
    return data, len(ambiguous)


def read_target_tar(path):
    """No tar extraction. Require exact 12 GSM-to-RSEM file matches."""
    if not path.is_file() or path.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Missing/incomplete GEO archive: {path}\nDownload: {URL}")
    expected = {gsm for g in GROUPS.values() for gsm in g}
    observed = {}
    with tarfile.open(path, "r") as tf:
        members = {}
        for m in tf:
            if not m.isfile():
                continue
            name = Path(m.name).name
            if not name.endswith(".genes.results.gz"):
                continue
            gsms = GSM_RE.findall(name)
            if len(gsms) != 1 or gsms[0] not in expected:
                continue
            gsm = gsms[0]
            if gsm in members:
                raise ValueError(f"Duplicate RSEM TAR member for {gsm}")
            if m.size > 50_000_000:
                raise ValueError("Suspiciously large RSEM member")
            members[gsm] = m
        missing = expected - set(members)
        if missing:
            examples = [Path(m.name).name for m in tf.getmembers() if m.isfile()][:5]
            raise ValueError(f"Missing 12-sample TAR members: {sorted(missing)}. Examples: {examples}")
        for gsm in sorted(expected):
            stream = tf.extractfile(members[gsm])
            if stream is None:
                raise ValueError(f"Cannot read TAR member for {gsm}")
            with stream:
                data, conflicts = parse_rsem(stream)
            observed[gsm] = {"data": data, "collisions": conflicts,
                             "member": members[gsm].name}
    return observed


def source_data():
    """Same strict source GSM/raw SHA checks as Stage 16; no washout dependency."""
    import numpy as np
    stage16 = load_py(SCRIPTS/"16_acute_specificity_challenge.py", "stage16_external_source")
    stage07 = load_py(SCRIPTS/"07_external_stress_test.py", "stage07_external_source")
    protocol = json.loads((ROOT/"data/metadata/evaluation_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("source_study") != "GSE193258" or protocol["flags"].get("biology_approved_for_training") is not False:
        raise ValueError("Source protocol is not the previously audited provisional cohort")
    genes, ids, source = stage07.load_matrix(
        stage07.safe_local_path(protocol["source_matrix"],"processed"),
        protocol["source_matrix_sha256"], np)
    meta = stage16.csv_rows(ROOT/"data/metadata/geo_samples.csv")
    mapping = stage16.csv_rows(ROOT/"data/metadata/expression_alignment.csv")
    known = stage16.verified_acute_columns(protocol, meta, mapping)
    acute_records, acute, raw_sha = stage16.read_verified_raw(
        stage16.RAW, ROOT/"data/metadata/expression_manifest.json",
        known, genes, source, ids, np)
    train = np.vstack((source, acute))
    y = np.array([STATES.index(row["state"]) for row in protocol["source_gsm_to_label"]] + [1]*12)
    train_ids = ids + [gsm for gsm,_ in acute_records]
    if train.shape != (36, 19712) or len(set(train_ids)) != 36 or Counter(y) != {0:12, 1:12, 2:12}:
        raise ValueError("Source matrix is not 36x19712, 12 samples per class")
    return genes, train_ids, train, y, {"source_sha256":protocol["source_matrix_sha256"],"raw_sha256":raw_sha}


def fit_transfer(X, y, genes, targets, target_is_tpm=True):
    """Same Stage 16 model recipes with target-only transforms and fixed source features.

    targets: mapping GSM -> dict[gene symbol, TPM], or on self-test the same
    source expression scale when target_is_tpm=False.
    """
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif, VarianceThreshold, SelectKBest
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if X.ndim != 2 or len(genes) != X.shape[1] or len(set(genes)) != len(genes):
        raise ValueError("Source matrix / genes mismatch")
    if Counter(y) not in (Counter({0:9,1:9,2:9}), Counter({0:12,1:12,2:12})):
        raise ValueError("Unexpected three-class source training balance")
    if not targets:
        raise ValueError("No external samples")
    index = {gene:i for i,gene in enumerate(genes)}
    raw_ranks = rankdata(X, axis=1, method="average") / X.shape[1]
    valid = np.flatnonzero(raw_ranks.var(0) > 1e-12)
    if len(valid) < 600:
        raise ValueError("Too few variable source genes")
    f,_ = f_classif(raw_ranks[:,valid], y)
    f = np.where(np.isfinite(f),f,-np.inf)
    order = np.lexsort((valid,-f))
    selected = valid[order[:100]]
    if not np.isfinite(f[order[:100]]).all():
        raise ValueError("Invalid source-selected genes")
    sel_names = [genes[i] for i in selected]
    full_train = raw_ranks[:,selected]
    local_train = rankdata(X[:,selected], axis=1, method="average") / 100
    target_ids = list(targets)
    full_test = np.broadcast_to(full_train.mean(0), (len(target_ids),100)).copy()
    local_test = np.broadcast_to(local_train.mean(0), (len(target_ids),100)).copy()
    raw_test = np.broadcast_to(X.mean(0), (len(target_ids),len(genes))).copy()
    cover=[]
    for row_idx,gsm in enumerate(target_ids):
        data = targets[gsm]
        if len(data) != len(set(data)) or len(data) < 1000:
            raise ValueError(f"Invalid target gene IDs: {gsm}")
        names = list(data)
        values = np.asarray(list(data.values()),dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite external values: {gsm}")
        if target_is_tpm and np.any(values < 0):
            raise ValueError(f"Negative RSEM TPM: {gsm}")
        ranks = rankdata(values, method="average") / len(values)
        lookup = {g:i for i,g in enumerate(names)}
        present = [k for k,g in enumerate(sel_names) if g in lookup]
        if len(present) < 90:
            raise ValueError(f"Only {len(present)} of 100 selected genes in {gsm}; no valid transfer score")
        cols = [lookup[sel_names[k]] for k in present]
        full_test[row_idx,present] = ranks[cols]
        local_test[row_idx,present] = rankdata(values[cols],method="average") / len(cols)
        nraw=0
        for name, p in index.items():
            if name in data:
                raw_test[row_idx,p] = math.log2(data[name]+.01) if target_is_tpm else data[name]
                nraw+=1
        if nraw < min(14000, math.ceil(0.70*len(genes))):
            raise ValueError(f"Only {nraw} source genes overlap with external {gsm}")
        cover.append({"gsm":gsm,"measured_gene_count":len(data),
                      "selected_available":len(present),"source_genes_available":nraw})
    source16 = load_py(SCRIPTS/"16_acute_specificity_challenge.py", "stage16_signature_transfer")
    _, signed_train, signed_apply = source16.three_class_signatures(full_train,local_train,y,np)
    signed_test = signed_apply(full_test)
    recipe = {
        "rank_full_lr100_3class":(full_train,full_test),
        "rank_selected_lr100_3class":(local_train,local_test),
        "signed_rank_ovr_full_3class":(signed_train,signed_test),
    }
    predictions={}
    for method,(train_feat,test_feat) in recipe.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,solver="lbfgs",max_iter=3000,random_state=42))
        model.fit(train_feat,y)
        predictions[method]={"labels":model.predict(test_feat),"margins":model.decision_function(test_feat)}
    rawpipe=make_pipeline(VarianceThreshold(),SelectKBest(f_classif,k=500),
        StandardScaler(),LogisticRegression(C=.01,solver="lbfgs",max_iter=3000,random_state=42))
    rawpipe.fit(X,y)
    predictions["raw_log2tpm_lr500_3class"]={"labels":rawpipe.predict(raw_test),"margins":rawpipe.decision_function(raw_test)}
    if any(v["margins"].shape!=(len(target_ids),3) for v in predictions.values()):
        raise ValueError("Unexpected multiclass margin shape")
    return predictions,cover,sel_names


def compare_stage16_replay():
    """Synthetic same-gene-space reproducibility against the existing Stage 16 implementation."""
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif,SelectKBest,VarianceThreshold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng=np.random.default_rng(2026)
    genes=[f"G{i}" for i in range(1200)]
    X=rng.lognormal(size=(27,1200))
    y=np.repeat(np.arange(3),9)
    X[y==1,20:60]*=3
    X[y==2,20:60]*=5
    Xt=rng.lognormal(size=(9,1200))
    stage16=load_py(SCRIPTS/"16_acute_specificity_challenge.py", "stage16_reference_replay")
    reference,sel=stage16.three_class_predict(X,y,Xt,np,rankdata,f_classif,
                    StandardScaler,LogisticRegression,SelectKBest,VarianceThreshold)
    projected={f"T{k}":dict(zip(genes,Xt[k])) for k in range(len(Xt))}
    actual,cover,names=fit_transfer(X,y,genes,projected,target_is_tpm=False)
    if names!=[genes[int(i)] for i in sel]:
        raise AssertionError("Source-selected features differ from Stage16")
    for method in METHODS:
        pred0,margin0=reference[method]
        if not np.array_equal(pred0,actual[method]["labels"]):
            raise AssertionError(f"Stage16 prediction mismatch: {method}")
        if not np.allclose(margin0,actual[method]["margins"],atol=1e-9):
            raise AssertionError(f"Stage16 margin mismatch: {method}")
    return True


def self_test():
    import numpy as np
    assert normalize_name("TP53_TP53")=="TP53"
    assert normalize_name("TP53_SOMETHING")=="TP53_SOMETHING"
    if set(g for v in GROUPS.values() for g in v)!=set(f"GSM808{n}" for n in range(3374,3386)):
        raise AssertionError("Incorrect target GSM map")
    assert compare_stage16_replay()
    # PC9 must be held out across all three classes; treatments must match.
    import numpy as np
    lines=("H1975","HCC2935","HCC827","PC9")
    toy_rows=[];toy_ids=[];toy_y=[]
    conditions={0:"DMSO",1:"osimertinib acute 24 hrs.",2:"osimertinib DTPs 3w."}
    for line in lines:
        for state in range(3):
            for rep in range(3):
                gsm=f"GSM_{line}_{state}_{rep}"
                toy_rows.append({"gse":"GSE193258","gsm":gsm,"reported_cell_line":line,
                                "reported_treatment":conditions[state]})
                toy_ids.append(gsm);toy_y.append(state)
    tr,te=verified_source_cellline_split(toy_ids,np.asarray(toy_y),toy_rows)
    assert len(tr)==27 and len(te)==9 and all("PC9" in toy_ids[i] for i in te)
    broken=[dict(r) for r in toy_rows]
    broken[-1]["reported_treatment"]="DMSO"
    try:
        verified_source_cellline_split(toy_ids,np.asarray(toy_y),broken)
        raise AssertionError("PC9 split allowed mislabeled DTP")
    except ValueError:
        pass
    print("SELF-TEST PASSED (GSM map; Stage16 replay; source-only selection; PC9 exclusion and treatment checks)")


def download():
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    if TARGET.is_file() and TARGET.stat().st_size>1_000_000:
        print("GSE255958 archive already exists; not overwriting:",TARGET)
        return
    tmp=TARGET.with_name(TARGET.name+".part")
    print("Downloading public GEO 50-MB processed archive:",URL,flush=True)
    req=urllib.request.Request(URL,headers={"User-Agent":"ResearchReproducibility/1.0"})
    try:
        with urllib.request.urlopen(req,timeout=90) as src, tmp.open("wb") as dst:
            while True:
                b=src.read(1<<20)
                if not b: break
                dst.write(b)
        if tmp.stat().st_size < 1_000_000:
            raise ValueError("Downloaded archive is too small")
        with tarfile.open(tmp,"r") as f:
            if not f.getmembers():
                raise ValueError("Downloaded TAR has no members")
        tmp.replace(TARGET)
    finally:
        tmp.unlink(missing_ok=True)


def run():
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    genes, train_ids,X,y,source_provenance=source_data()
    tar_data=read_target_tar(TARGET)
    expected=[g for state in STATES for g in GROUPS[state]]
    external={g:tar_data[g]["data"] for g in expected}
    pred,cover,selection=fit_transfer(X,y,genes,external,target_is_tpm=True)
    truth=np.array([STATES.index(s) for s in STATES for _ in GROUPS[s]],dtype=int)
    rows=[]; metrics=[]
    for method in METHODS:
        got=pred[method]["labels"]
        margin=pred[method]["margins"]
        mat=confusion_matrix(truth,got,labels=[0,1,2]).tolist()
        result={"method":method,"n":9,"correct":int((truth==got).sum()),
                "balanced_accuracy":float(balanced_accuracy_score(truth,got)),
                "macro_f1":float(f1_score(truth,got,average="macro",zero_division=0)),
                "control_recall":float(mat[0][0]/3),"acute_recall":float(mat[1][1]/3),
                "dtp_recall":float(mat[2][2]/3),"confusion_matrix":mat}
        metrics.append(result)
        for j,gsm in enumerate(expected):
            rows.append({"study":"GSE255958","gsm":gsm,
                "method":method,"true_state":STATES[int(truth[j])],
                "predicted_state":STATES[int(got[j])],
                "correct":int(truth[j]==got[j]),
                **{f"decision_score_{st.lower()}":f"{margin[j,k]:.12g}" for k,st in enumerate(STATES)},
                "selected_available":cover[j]["selected_available"]})
    OUT.mkdir(parents=True,exist_ok=True)
    def write_csv(name,values):
        with (OUT/name).open("w",encoding="utf-8",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    write_csv("17_predictions.csv",rows)
    write_csv("17_metrics.csv",[{k:v for k,v in x.items() if k!="confusion_matrix"} for x in metrics])
    report={"analysis":"historically_inspected_GSE255958_three_class_source_only_transfer",
         "created_utc":datetime.now(timezone.utc).isoformat(),"status":"EXPLORATORY_NOT_INDEPENDENT",
         "source_n":36,"source_study":"GSE193258","target_n":9,
         "target_study":"GSE255958","excluded_resistant_gsms":GROUPS["Resistant"],
         "source_provenance":source_provenance,"target_tar_sha256":sha256(TARGET),
         "method_recipes":"Stage16 three_class_predict equivalent on complete matched genes",
         "source_selected_100_genes_sha256":hashlib.sha256("\n".join(selection).encode()).hexdigest(),
         "target_coverage":cover,"metrics":metrics,
         "limitations":["Same external study previously inspected; no independent validation.",
                        "External only PC9, three samples per state.",
                        "Source 24h Acute vs 21d DTP; target 48h Acute vs 9d DTP: durations confounded.",
                        "Source hg38 and target hg19 processing / annotation not fully harmonized.",
                        "Raw LR uses log2(TPM+0.01), but study/platform shifts persist.",
                        "Full-gene ranks use each target sample's reported gene universe and thus differ in reference size.",
                        "Unmeasured genes are not used to choose features; missing selected rank features are imputed from source training means."]}
    (OUT/"17_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Stage17 external 3-class: source=36, GSE255958 PC9 target=9 (3 per state), resistant=3 excluded")
    print(f"Source-only selected features: min coverage={min(r['selected_available'] for r in cover)}/100")
    for s in metrics:
        print(f"{s['method']}: bACC={s['balanced_accuracy']:.3f}, macroF1={s['macro_f1']:.3f}, "
              f"Control={s['control_recall']:.3f}, Acute={s['acute_recall']:.3f}, "
              f"DTP={s['dtp_recall']:.3f}, correct={s['correct']}/9")
    print("Saved outputs/three_class_external/{17_summary.json,17_metrics.csv,17_predictions.csv}")
    print("EXPLORATORY ONLY — historically inspected target, not independent validation")



def verified_source_cellline_split(train_ids, y, geo_rows):
    """Strict source GSM/condition mapping; do not infer lines from GSM number."""
    import numpy as np
    by_id={r["gsm"]:r for r in geo_rows if r.get("gse")=="GSE193258"}
    if len(train_ids)!=36 or len(set(train_ids))!=36 or len(y)!=36:
        raise ValueError("Expected 36 unique source GSMs")
    names={0:"DMSO",1:"osimertinib acute 24 hrs.",2:"osimertinib DTPs 3w."}
    groups=[]
    for gsm,code in zip(train_ids,y):
        if gsm not in by_id or int(code) not in names:
            raise ValueError("Unmapped source GSM/class")
        r=by_id[gsm]
        if r["reported_treatment"]!=names[int(code)]:
            raise ValueError(f"Source treatment/class mismatch: {gsm}")
        line=r["reported_cell_line"]
        if line not in {"H1975","HCC2935","HCC827","PC9"}:
            raise ValueError(f"Unexpected source line {line}")
        groups.append(line)
    groups=np.asarray(groups)
    hold=np.flatnonzero(groups=="PC9")
    train=np.flatnonzero(groups!="PC9")
    if (len(hold)!=9 or len(train)!=27 or
        Counter(y[train])!={0:9,1:9,2:9} or
        Counter(y[hold])!={0:3,1:3,2:3}):
        raise ValueError("Invalid PC9-excluded three-class split")
    return train,hold


def pc9_replay_stage16(predictions, train_ids, hold, reference_csv):
    """Compare 27-source-trained predictions on source PC9 to Stage16 LOCO."""
    import numpy as np
    if not reference_csv.is_file():
        raise FileNotFoundError(f"Missing Stage16 frozen predictions: {reference_csv}")
    with reference_csv.open(encoding="utf-8-sig",newline="") as f:
        reference=list(csv.DictReader(f))
    lookup={}
    for r in reference:
        if r.get("held_out_line")=="PC9" and r.get("true_state") in STATES:
            key=(r["method"],r["gsm"])
            if key in lookup:
                raise ValueError("Duplicate source PC9 reference prediction")
            lookup[key]=r
    if len(lookup)!=9*len(METHODS):
        raise ValueError("Incomplete Stage16 PC9 reference")
    for method in METHODS:
        preds=predictions[method]
        for j,i in enumerate(hold):
            r=lookup[(method,train_ids[i])]
            if r["predicted_state"]!=STATES[int(preds["labels"][j])]:
                raise ValueError(f"Stage16 PC9 prediction changed: {method}/{train_ids[i]}")
            for k,state in enumerate(STATES):
                value=float(r["decision_score_"+state.lower()])
                if not np.isclose(value,float(preds["margins"][j,k]),rtol=0,atol=1e-8):
                    raise ValueError(f"Stage16 PC9 margin changed: {method}/{train_ids[i]}")
    return True


def pc9_existing_full_source(reference_csv):
    if not reference_csv.is_file():
        raise FileNotFoundError(f"Missing Stage17 full-source predictions: {reference_csv}")
    with reference_csv.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    expected={g for state in STATES for g in GROUPS[state]}
    index={}
    for row in rows:
        key=(row["method"],row["gsm"])
        if key in index:
            raise ValueError("Duplicate Stage17 full-source prediction")
        index[key]=row
    if len(index)!=len(METHODS)*9:
        raise ValueError("Wrong Stage17 full-source prediction count")
    for method in METHODS:
        if {g for (m,g) in index if m==method}!=expected:
            raise ValueError(f"Wrong full-source target GSMs for {method}")
    return index


def run_pc9_excluded():
    """PC9 is absent from source fitting AND the target is a different study PC9.

    Already-inspected target, so this is a mechanistic sensitivity analysis,
    not an independent validation or a model-selection opportunity.
    """
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    genes,ids,X,y,provenance=source_data()
    source16=load_py(SCRIPTS/"16_acute_specificity_challenge.py","stage16_pc9_guard")
    meta=source16.csv_rows(ROOT/"data/metadata/geo_samples.csv")
    train,hold=verified_source_cellline_split(ids,y,meta)
    internal={ids[i]:dict(zip(genes,X[i])) for i in hold}
    local_preds,local_coverage,sel_local=fit_transfer(X[train],y[train],genes,internal,target_is_tpm=False)
    pc9_replay_stage16(local_preds,ids,hold,ROOT/"outputs/acute_challenge/16_three_class_predictions.csv")

    archive=read_target_tar(TARGET)
    target_ids=[gsm for state in STATES for gsm in GROUPS[state]]
    external={gsm:archive[gsm]["data"] for gsm in target_ids}
    preds,target_coverage,sel_external=fit_transfer(X[train],y[train],genes,external,target_is_tpm=True)
    if sel_external!=sel_local:
        raise ValueError("Source selection differed across target datasets")
    full=pc9_existing_full_source(OUT/"17_predictions.csv")
    truth=np.array([STATES.index(s) for s in STATES for _ in GROUPS[s]],dtype=int)
    metrics=[]; rows=[]
    for scope,values,gsms,labels,coverage in (
        ("source_PC9_LOCO",local_preds,[ids[i] for i in hold],y[hold],local_coverage),
        ("external_GSE255958_PC9",preds,target_ids,truth,target_coverage),
    ):
        for method in METHODS:
            got=values[method]["labels"]
            m=values[method]["margins"]
            cm=confusion_matrix(labels,got,labels=[0,1,2])
            metrics.append({"scope":scope,"method":method,"n":len(gsms),
                "correct":int(np.sum(labels==got)),
                "balanced_accuracy":float(balanced_accuracy_score(labels,got)),
                "macro_f1":float(f1_score(labels,got,average="macro",zero_division=0)),
                "control_recall":float(cm[0,0]/3),
                "acute_recall":float(cm[1,1]/3),
                "dtp_recall":float(cm[2,2]/3)})
            for j,gsm in enumerate(gsms):
                record={"scope":scope,"method":method,"gsm":gsm,
                    "true_state":STATES[int(labels[j])],"predicted_state":STATES[int(got[j])],
                    "correct":int(labels[j]==got[j]),
                    "selected_available":coverage[j]["selected_available"],
                    **{f"decision_score_{state.lower()}":f"{m[j,k]:.12g}" for k,state in enumerate(STATES)}}
                if scope=="external_GSE255958_PC9":
                    previous=full[(method,gsm)]
                    expected_state=previous["true_state"]
                    if record["true_state"]!=expected_state:
                        raise ValueError(f"Stage17 target label mismatch: {gsm}")
                    record["full_source_prediction"]=previous["predicted_state"]
                    record["changed_from_full_source"]=int(record["predicted_state"]!=previous["predicted_state"])
                else:
                    record["full_source_prediction"]=""
                    record["changed_from_full_source"]=""
                rows.append(record)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"17_pc9_excluded_metrics.csv").open("w",encoding="utf-8",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    with (OUT/"17_pc9_excluded_predictions.csv").open("w",encoding="utf-8",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={"analysis":"historically_inspected_PC9_excluded_source_training_external_stress",
        "status":"EXPLORATORY_NOT_INDEPENDENT", "n_source_training":27,
        "source_PC9_heldout_n":9,"external_GSE255958_PC9_n":9,
        "source_pc9_excluded":True,"no_target_fitting_or_tuning":True,
        "source_selected_genes_sha256":hashlib.sha256("\n".join(sel_local).encode()).hexdigest(),
        "source_provenance":provenance,"target_tar_sha256":sha256(TARGET),
        "stage16_source_PC9_predictions_reproduced":True,
        "external_previous_full_source_predictions_checked":True,
        "metrics":metrics,
        "limitations":["The target was inspected before this comparison; results cannot validate a new method or retrospectively select the winner.",
            "Only one external PC9 study with three replicates per class.",
            "Source and target treatment dose/duration and expression processing differ.",
            "Removing source PC9 probes model dependence on within-study PC9 exposure; not an independent heldout study."]}
    (OUT/"17_pc9_excluded_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Stage17 PC9-excluded: 27 source training (NO PC9); source PC9=9; external GSE255958 PC9=9")
    print("Stage16 source-PC9 LOCO predictions and margins REPRODUCED")
    print("Existing Stage17 full-source target predictions matched by GSM/label")
    for method in METHODS:
        a=next(x for x in metrics if x["scope"]=="source_PC9_LOCO" and x["method"]==method)
        b=next(x for x in metrics if x["scope"]=="external_GSE255958_PC9" and x["method"]==method)
        old=sum(full[(method,g)]["correct"]=="1" for g in target_ids)
        print(f"{method}: source PC9 {a['correct']}/9 bACC={a['balanced_accuracy']:.3f}; "
              f"external PC9 {b['correct']}/9 bACC={b['balanced_accuracy']:.3f}; "
              f"previous full-source external={old}/9")
    print("Saved outputs/three_class_external/17_pc9_excluded_{summary.json,metrics.csv,predictions.csv}")
    print("EXPLORATORY: external study was already inspected; do not tune based on this result")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--exploratory-only",action="store_true")
    p.add_argument("--download",action="store_true",help="Explicitly download the official GEO processed TAR (~50MB)")
    p.add_argument("--pc9-excluded",action="store_true",help="Train without any source PC9 and compare source/target PC9; no model tuning")
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.exploratory_only:
        p.error("Real external scoring requires --exploratory-only")
    if args.download:
        download()
    if args.pc9_excluded:
        run_pc9_excluded()
    else:
        run()

if __name__=="__main__":
    main()
