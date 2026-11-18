"""Stage 16: frozen Control/DTP models challenged with unseen Acute samples.

Source GSE193258 has four cell lines, each with Control (DMSO), Acute (24h
osimertinib), DTP (21d osimertinib). For each held-out cell line:
  - Fit ONLY 18 Control/DTP samples from the other three lines.
  - Predict the held-out Control, DTP, and never-trained Acute samples.
  - Compare the exact Stage-15 methods, no new hyperparameters or target tuning.

Acute is NOT a binary training class. Its predicted-DTP rate is a challenge
metric, not a validated false-positive rate or evidence of biological truth.
Acute treatment duration differs from 21d DTP; biological confounding remains.
This is a retrospective, within-study, previously inspected challenge.

CRITICAL safety checks: original raw log2TPM matrix SHA-256 from downloaded
manifest; exact metadata/GSM mapping; all 24 source column values match the
frozen source matrix; per-GSM Control/DTP predictions reproduce Stage 15.

Run from repository root:
  python scripts/16_acute_specificity_challenge.py --self-test
  python scripts/16_acute_specificity_challenge.py --exploratory-only
  python scripts/16_acute_specificity_challenge.py --exploratory-only --acute-vs-dtp
  python scripts/16_acute_specificity_challenge.py --exploratory-only --washout
  python scripts/16_acute_specificity_challenge.py --exploratory-only --three-class-washout

Outputs: outputs/acute_challenge/16_predictions.csv, 16_fold_metrics.csv,
  16_summary.json, 16_acute_specificity.svg (vector figure, no matplotlib).
New --washout outputs: outputs/acute_challenge/16_washout_summary.json,
  16_washout_predictions.csv, 16_washout_group_summary.csv, 16_washout.svg.
Never modifies upstream scripts, source data, or historical output files.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
META = ROOT / "data" / "metadata"
OUTPUT = ROOT / "outputs" / "acute_challenge"
RAW_NAME = "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"
RAW = ROOT / "data" / "raw" / "GSE193258" / RAW_NAME
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
STATES = ("Control", "Acute", "DTP")
METHODS = ("selected_feature_percentile", "signed_rank_full",
           "signed_rank_selected", "top_pairs_11")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_module(path, name):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Module load failed: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_acute_columns(protocol, metadata, alignments, raw_name=RAW_NAME):
    """Require exact GEO mapped identities and the expected 12 Acute GSMs."""
    frozen = {r["gsm"]:r for r in protocol["source_gsm_to_label"]}
    if len(frozen) != 24 or len(set(frozen)) != 24:
        raise ValueError("Expected exactly 24 frozen source GSMs")
    expected_controls = Counter((r["cell_line"],r["state"]) for r in frozen.values())
    if len(expected_controls) != 8 or set(expected_controls.values()) != {3}:
        raise ValueError("Frozen source Control/DTP counts changed")
    by_gsm = {r["gsm"]:r for r in metadata if r["gse"]=="GSE193258"}
    if len(by_gsm) != 60:
        raise ValueError(f"Expected 60 distinct GSE193258 GEO samples; found {len(by_gsm)}")
    by_alignment = {}
    for r in alignments:
        if r["gse"]!="GSE193258" or r["filename"]!=raw_name:
            continue
        gsm = r["gsm"]
        if gsm in by_alignment or r["mapping_status"]!="CANDIDATE_REQUIRES_REVIEW":
            raise ValueError(f"Duplicate or unreviewed expression column: {gsm}")
        by_alignment[gsm] = r
    selected = {}
    for gsm, meta in by_gsm.items():
        if meta["reported_treatment"] != "osimertinib acute 24 hrs.":
            continue
        line = meta["reported_cell_line"]
        if line not in LINES:
            raise ValueError(f"Unexpected acute cell line {line}")
        if gsm in frozen:
            raise ValueError(f"Acute GSM {gsm} overlaps training Control/DTP")
        if gsm not in by_alignment:
            raise ValueError(f"No exact expression mapping for Acute GSM {gsm}")
        a = by_alignment[gsm]
        if (a["treatment_raw"]!=meta["reported_treatment"] or
            a["cell_line_raw"]!=line or a["geo_title"]!=meta["title"]):
            raise ValueError(f"Acute metadata mismatch for {gsm}")
        selected[gsm] = {"cell_line":line, "column":a["matrix_column"], "state":"Acute"}
    if len(selected) != 12 or Counter(r["cell_line"] for r in selected.values()) != Counter({x:3 for x in LINES}):
        raise ValueError(f"Expected 3 Acute per line (12 total), got {len(selected)}")
    for gsm, r in frozen.items():
        if gsm not in by_alignment or gsm not in by_gsm:
            raise ValueError(f"Frozen GSM {gsm} not mapped in full raw file")
        a = by_alignment[gsm]
        m = by_gsm[gsm]
        expected_treatment = "DMSO" if r["state"]=="Control" else "osimertinib DTPs 3w."
        if (a["treatment_raw"]!=expected_treatment or m["reported_treatment"]!=expected_treatment or
            a["cell_line_raw"] !=r["cell_line"] or m["reported_cell_line"]!=r["cell_line"]):
            raise ValueError(f"Frozen GSM {gsm}: treatment or cell-line mismatch")
        selected[gsm]={"cell_line":r["cell_line"],"column":a["matrix_column"],"state":r["state"]}
    if len({r["column"] for r in selected.values()}) != len(selected):
        raise ValueError("The raw matrix reuses a column for multiple GSM samples")
    return selected


def read_verified_raw(raw_path, manifest_path, selected, expected_genes, Xsource, source_ids, np):
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    item=manifest["manifest"].get("GSE193258/"+raw_path.name)
    if item is None or item.get("reported_expression_unit")!="log2TPM_abundance":
        raise ValueError("Raw abundance unit missing/unverified in download manifest")
    if item.get("gzip_crc_verified") is not True:
        raise ValueError("Download manifest has no GZIP CRC check")
    if sha256(raw_path)!=item["sha256"]:
        raise ValueError("Original source data SHA-256 does not match audited manifest")
    acutes=sorted((gsm,r) for gsm,r in selected.items() if r["state"]=="Acute")
    selected_columns = [selected[gsm]["column"] for gsm in source_ids] + [r["column"] for _,r in acutes]
    with gzip.open(raw_path,"rt",encoding="utf-8-sig",newline="") as stream:
        reader=csv.reader(stream,delimiter="\t")
        header=next(reader)
        if header[0].strip().lower()!="gene" or len(header)!=61 or len(header)!=len(set(header)):
            raise ValueError("Expected unique full-study 60-sample raw log2TPM header")
        positions={col:i for i,col in enumerate(header)}
        if any(s not in positions for s in selected_columns):
            raise ValueError("Mapped source/Acute column missing from raw expression file")
        chosen=[positions[s] for s in selected_columns]
        rawgenes=[]; vals=[]
        for rn,row in enumerate(reader,start=2):
            if len(row)!=len(header) or not row[0].strip():
                raise ValueError(f"Malformed full expression matrix at row {rn}")
            try:
                numbers=[float(row[k]) for k in chosen]
            except ValueError as ex:
                raise ValueError(f"Nonnumeric values at raw row {rn}") from ex
            if not all(math.isfinite(v) for v in numbers):
                raise ValueError(f"NaN/Inf in raw row {rn}")
            rawgenes.append(row[0].strip());vals.append(numbers)
    if len(rawgenes)!=len(expected_genes) or rawgenes!=list(expected_genes):
        raise ValueError("Full raw/source gene order or identity differs; stop")
    if len(set(rawgenes))!=len(rawgenes):
        raise ValueError("Duplicate gene IDs in raw expression matrix")
    data=np.asarray(vals,dtype=float).T
    if data.shape!=(36,len(expected_genes)) or Xsource.shape!=(24,len(expected_genes)):
        raise ValueError("Unexpected raw/source sample-gene shapes")
    if not np.array_equal(data[:24],Xsource):
        delta=np.max(np.abs(data[:24]-Xsource))
        raise ValueError(f"Raw source columns do not exactly reproduce development matrix; max delta {delta}")
    return acutes,data[24:],item["sha256"]


def model_predictions(Xtrain, ytrain, Xt, genes, stage11, stage13, stage15, np,
                      rankdata, f_classif, LogisticRegression, StandardScaler):
    """Freeze stage-15 training recipes; outputs label + signed decision margin."""
    fitted15=stage15.train_models(Xtrain,ytrain,stage11,np,rankdata,f_classif,
                                  LogisticRegression,StandardScaler)
    sel=fitted15["selected"]
    selected_names=[genes[int(i)] for i in sel]
    refs=stage11.select_source_features_and_anchors(Xtrain,ytrain,rankdata,np,
                                                     f_classif,k=stage15.K,anchors=256)[1]
    anchor_names=[genes[int(i)] for i in refs]
    allr=stage11.rank_all(Xtrain,np,rankdata)
    base=stage13.fit_source(Xtrain,ytrain,sel,refs,allr,stage11,np,
                            LogisticRegression,StandardScaler)["selected_feature_percentile"]
    mat,_=stage13.target_transform(Xt,genes,selected_names,anchor_names,
                          base["train_mean"],"selected_feature_percentile",rankdata,np)
    p1=base["model"].predict(mat).astype(int)
    d1=base["model"].decision_function(mat)
    raw,avail,s_full,s_local=stage15.target_source_representations(
                      Xt,genes,selected_names,fitted15["up"],fitted15["down"],np,rankdata)
    preds={"selected_feature_percentile":(p1,d1)}
    for method,score in (("signed_rank_full",s_full),("signed_rank_selected",s_local)):
        clf=fitted15["signatures"][method]
        preds[method]=(clf.predict(score).astype(int),clf.decision_function(score))
    pairpred,npairs=stage15.pair_predict(raw,avail,fitted15["pair_rules"],np)
    # Top-pairs returns binary vote only; no calibrated decision margin provided.
    preds["top_pairs_11"]=(pairpred,None)
    if set(preds)!=set(METHODS) or npairs!=11:
        raise ValueError("Stage 15 method set or available pairs changed")
    return preds


def metrics_by_group(records, np):
    results=[]
    for method in METHODS:
        mr=[r for r in records if r["method"]==method]
        if Counter(r["true_state"] for r in mr)!=Counter({"Control":12,"Acute":12,"DTP":12}):
            raise ValueError(f"Not exactly 12 of each source challenge state: {method}")
        for line in list(LINES)+["ALL"]:
            sub=mr if line=="ALL" else [r for r in mr if r["cell_line"]==line]
            by={s:[r for r in sub if r["true_state"]==s] for s in STATES}
            if not all(len(by[s])==(12 if line=="ALL" else 3) for s in STATES):
                raise ValueError("Unbalanced heldout group count")
            c=sum(r["predicted_state"]=="Control" for r in by["Control"])/len(by["Control"])
            d=sum(r["predicted_state"]=="DTP" for r in by["DTP"])/len(by["DTP"])
            a=sum(r["predicted_state"]=="DTP" for r in by["Acute"])/len(by["Acute"])
            results.append({"method":method,"held_out_line":line,"n_control":len(by["Control"]),
               "n_acute":len(by["Acute"]),"n_dtp":len(by["DTP"]),
               "control_recall":c,"dtp_recall":d,"binary_bacc":(c+d)/2,
               "acute_dtp_call_fraction":a,"acute_control_call_fraction":1-a})
    return results


def plot_svg(rows):
    """Single vector multi-panel plot with 3 state groups x 4 cell lines.

    Avoids matplotlib dependency. Illustrates predictions, not inferred
    biological state.  A full version may later be built from 16_predictions.csv.
    """
    colors={"selected_feature_percentile":"#277DA1","signed_rank_selected":"#B45309"}
    shown=list(colors)
    width,height=1060,520
    a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
       '<rect x="0" y="0" width="1060" height="520" fill="white"/>',
       '<text x="42" y="38" font-family="Arial" font-size="21" font-weight="bold" fill="#182532">Acute challenge of Control/DTP-trained classifiers</text>',
       '<text x="42" y="61" font-family="Arial" font-size="13" fill="#435365">GSE193258 | held-out cell-line folds | Acute never used to fit or choose features</text>']
    for method, x, color in (("selected_feature_percentile",42,colors[shown[0]]),
                              ("signed_rank_selected",552,colors[shown[1]])):
        label="Selected-feature percentile" if method=="selected_feature_percentile" else "Selected signed-rank signature"
        a.append(f'<text x="{x}" y="102" font-family="Arial" font-size="17" font-weight="bold" fill="{color}">{escape(label)}</text>')
        a.append(f'<text x="{x}" y="125" font-family="Arial" font-size="12" fill="#435365">Fraction predicted DTP  (0% to 100%)</text>')
        for idx,line in enumerate(LINES):
            y=160+idx*80
            a.append(f'<text x="{x}" y="{y+13}" font-family="Arial" font-size="13" fill="#25384A">{escape(line)}</text>')
            for s_i,state in enumerate(STATES):
                rr=next(r for r in rows if r["held_out_line"]==line and r["method"]==method)
                value=(1-rr["control_recall"] if state=="Control" else
                       rr["acute_dtp_call_fraction"] if state=="Acute" else rr["dtp_recall"])
                box_x=x+106+s_i*122
                a.append(f'<rect x="{box_x}" y="{y-4}" width="91" height="18" fill="#EDF1F4" rx="2"/>')
                a.append(f'<rect x="{box_x}" y="{y-4}" width="{91*value:.2f}" height="18" fill="{color}" rx="2"/>')
                a.append(f'<text x="{box_x+46}" y="{y+9}" font-family="Arial" font-size="11" text-anchor="middle" fill="#11202A">{100*value:.0f}%</text>')
        for i,s in enumerate(STATES):
            a.append(f'<text x="{x+106+i*122+45}" y="480" font-family="Arial" font-size="12" text-anchor="middle" fill="#435365">{s}</text>')
    a.extend(['<text x="42" y="505" font-family="Arial" font-size="11" fill="#65717F">Interpretation: Acute DTP-call fraction is a challenge metric, not a validated false-positive rate.</text>',
              '</svg>'])
    return '\n'.join(a)+'\n'


def run(root=ROOT):
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scriptdir=root/"scripts";meta=root/"data/metadata"; out=root/"outputs/acute_challenge"
    stage07=load_module(scriptdir/"07_external_stress_test.py","m7_ac16")
    stage11=load_module(scriptdir/"11_source_only_gene_space_benchmark.py","m11_ac16")
    stage13=load_module(scriptdir/"13_cross_study_anchor_benchmark.py","m13_ac16")
    stage15=load_module(scriptdir/"15_rank_method_benchmark.py","m15_ac16")
    if (stage15.METHODS!=METHODS or stage15.K!=100 or stage15.C!=0.01 or
        stage15.SEED!=42 or stage13.K!=100 or stage13.C!=0.01):
        raise ValueError("Frozen Stage 15 method family/hyperparameters changed")
    previous=root/"outputs/rank_methods/15_predictions.csv"
    prev_summary=root/"outputs/rank_methods/15_summary.json"
    if not previous.is_file() or not prev_summary.is_file():
        raise FileNotFoundError("Run frozen Stage 15 first to establish source LOCO predictions")
    protocol=json.loads((meta/"evaluation_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("source_study")!="GSE193258" or protocol["flags"]["biology_approved_for_training"] is not False:
        raise ValueError("Not the expected provisional GSE193258 source")
    saved=json.loads(prev_summary.read_text(encoding="utf-8"))
    if saved["source_sha256"]!=protocol["source_matrix_sha256"]:
        raise ValueError("Stage 15 uses a different source matrix")
    source_path=stage07.safe_local_path(protocol["source_matrix"],"processed")
    # stage07 ROOT is the module's own path. Running in repo root matches root.
    if source_path.resolve()!=(root/protocol["source_matrix"]).resolve():
        raise ValueError("Unexpected source matrix path")
    genes,source_ids,X=stage07.load_matrix(source_path,protocol["source_matrix_sha256"],np)
    proto=protocol["source_gsm_to_label"]
    if source_ids!=[r["gsm"] for r in proto] or X.shape!=(24,19712):
        raise ValueError("Source GSM order or sample count changed")
    source_labels=np.array([int(r["state"]=="DTP") for r in proto],dtype=int)
    groups=np.array([r["cell_line"] for r in proto])
    by_gsm=verified_acute_columns(protocol,csv_rows(meta/"geo_samples.csv"),
                                 csv_rows(meta/"expression_alignment.csv"))
    raw=root/"data/raw/GSE193258"/RAW_NAME
    acute_ids,Xacute,raw_sha=read_verified_raw(raw,meta/"expression_manifest.json",by_gsm,
                                               genes,X,source_ids,np)
    if set(g for g,_ in acute_ids)&set(source_ids):
        raise AssertionError("Acute/source sample overlap")
    old=[r for r in csv_rows(previous) if r["scope"].startswith("LOCO:")]
    old_predictions={(r["method"],r["sample_id"]):r["prediction"] for r in old}
    if len(old_predictions)!=24*len(METHODS):
        raise ValueError("Stage 15 source LOCO per-sample predictions incomplete/duplicate")
    records=[]
    for line in LINES:
        idxtrain=np.flatnonzero(groups!=line)
        idxhold=np.flatnonzero(groups==line)
        acute_idx=[i for i,(gsm,r) in enumerate(acute_ids) if r["cell_line"]==line]
        if len(idxtrain)!=18 or len(idxhold)!=6 or len(acute_idx)!=3 or Counter(source_labels[idxtrain])!={0:9,1:9}:
            raise ValueError(f"Invalid LOCO structure for {line}")
        # Evaluate the 6 binary holdouts and 3 entirely withheld Acute samples.
        Xt=np.concatenate((X[idxhold],Xacute[acute_idx]),axis=0)
        target_ids=[source_ids[i] for i in idxhold]+[acute_ids[i][0] for i in acute_idx]
        states=[proto[i]["state"] for i in idxhold]+["Acute"]*3
        results=model_predictions(X[idxtrain],source_labels[idxtrain],Xt,genes,
                      stage11,stage13,stage15,np,rankdata,f_classif,
                      LogisticRegression,StandardScaler)
        for method in METHODS:
            pred,margin=results[method]
            for i,(gsm,state) in enumerate(zip(target_ids,states)):
                name="DTP" if int(pred[i]) else "Control"
                if state in {"Control","DTP"} and name!=old_predictions.get((method,gsm)):
                    raise ValueError(f"STOP: Stage15 frozen LOCO prediction changed: {method}/{gsm}")
                records.append({"cell_line":line,"gsm":gsm,"true_state":state,
                  "method":method,"predicted_state":name,
                  "dtp_decision_margin": "" if margin is None else f"{float(margin[i]):.12g}",
                  "is_acute_challenge":int(state=="Acute"),
                  "correct_binary": "" if state=="Acute" else int(name==state)})
    if len(records)!=36*len(METHODS):
        raise ValueError("Unexpected number of audited predictions")
    summary=metrics_by_group(records,np)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("16_predictions.csv",records),("16_fold_metrics.csv",summary)):
        with (out/name).open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    report={"status":"EXPLORATORY_ACUTE_CHALLENGE_NOT_INDEPENDENT_OR_BIOLOGICALLY_APPROVED",
      "timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "training":"18 source Control/DTP per fold; acute never used for fitting/feature selection",
      "frozen_stage15_source_predictions_reproduced":True,
      "source_matrix_sha256":protocol["source_matrix_sha256"],"original_raw_sha256":raw_sha,
      "stage15_report_sha256":sha256(prev_summary),"stage15_predictions_sha256":sha256(previous),
      "study":"GSE193258","heldout_lines":list(LINES),
      "sample_counts":{"control":12,"acute":12,"dtp":12},
      "model_metrics":summary,
      "limitations":["The source study and folds were inspected previously; no independent validation.",
        "Control/DTP classifier has NO Acute class: acute DTP-call rate is a stress/specificity challenge, not formal Acute classification accuracy.",
        "Acute at 24h and DTP at about 21 days differ in exposure duration; duration/biology confounding unresolved.",
        "Exact GEO/sample mapping is supported by titles and file headers but biological sample approval is still pending.",
        "The four compared implementations have different model capacities and are not exact official singscore/kTSP implementations."]}
    (out/"16_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (out/"16_acute_specificity.svg").write_text(plot_svg(summary),encoding="utf-8")
    print("Stage 16: frozen Stage 15 Control/DTP source predictions REPRODUCED")
    for method in METHODS:
        x=next(r for r in summary if r["method"]==method and r["held_out_line"]=="ALL")
        print(f"{method}: heldout binary bACC={x['binary_bacc']:.3f}; DTP recall={x['dtp_recall']:.3f}; Acute called DTP={x['acute_dtp_call_fraction']:.3f} ({round(12*x['acute_dtp_call_fraction'])}/12)")
    print("Saved outputs/acute_challenge/16_summary.json, 16_predictions.csv, 16_fold_metrics.csv, 16_acute_specificity.svg")
    print("EXPLORATORY ONLY — Acute not a training class; not independent biological validation")
    return report


def self_test():
    import numpy as np
    meta=[]; align=[]
    for i,line in enumerate(LINES):
        for j in range(3):
            gsm=f"GSM{1000+i*10+j}"
            meta.append({"gse":"GSE193258","gsm":gsm,"reported_treatment":"osimertinib acute 24 hrs.","reported_cell_line":line,"title":f"{line}_acute_{j+1} [RNA-seq]"})
            align.append({"gse":"GSE193258","gsm":gsm,"filename":RAW_NAME,"matrix_column":f"{line}_acute_{j+1}","cell_line_raw":line,"treatment_raw":"osimertinib acute 24 hrs.","geo_title":meta[-1]["title"],"mapping_status":"CANDIDATE_REQUIRES_REVIEW"})
        for state,treat in (("Control","DMSO"),("DTP","osimertinib DTPs 3w.")):
            for j in range(3):
                gsm=f"GSM{2000+i*100+(0 if state=='Control' else 10)+j}"
                meta.append({"gse":"GSE193258","gsm":gsm,"reported_treatment":treat,"reported_cell_line":line,"title":f"{line}_{state}_{j+1} [RNA-seq]"})
                align.append({"gse":"GSE193258","gsm":gsm,"filename":RAW_NAME,"matrix_column":f"{line}_{state}_{j+1}","cell_line_raw":line,"treatment_raw":treat,"geo_title":meta[-1]["title"],"mapping_status":"CANDIDATE_REQUIRES_REVIEW"})
    # 60 GEO records: 36 selected + 24 washouts (not used for this challenge).
    for i in range(24):
        meta.append({"gse":"GSE193258","gsm":f"GSM{9000+i}","reported_treatment":"washout","reported_cell_line":LINES[i//6],"title":"unused"})
    source=[{"gsm":r["gsm"],"cell_line":r["cell_line_raw"],"state":"Control" if r["treatment_raw"]=="DMSO" else "DTP"} for r in align if r["treatment_raw"]!="osimertinib acute 24 hrs."]
    got=verified_acute_columns({"source_gsm_to_label":source},meta,align)
    assert len(got)==36 and sum(x["state"]=="Acute" for x in got.values())==12
    broken=[dict(x) for x in align]
    broken[0]["treatment_raw"]="DMSO"
    try:
        verified_acute_columns({"source_gsm_to_label":source},meta,broken)
        raise AssertionError("Failed to detect wrong Acute mapping")
    except ValueError:
        pass
    toy=[]
    for m in METHODS:
        for line in LINES:
            for s in STATES:
                for i in range(3):
                    toy.append({"method":m,"cell_line":line,"true_state":s,
                                "predicted_state":"DTP" if s=="DTP" else "Control"})
    out=metrics_by_group(toy,np)
    assert len(out)==len(METHODS)*5
    assert all(x["acute_dtp_call_fraction"]==0 for x in out)
    svg=plot_svg(out)
    assert svg.startswith("<svg") and "Acute" in svg and "</svg>" in svg
    toy_binary=[]
    for m in list(METHODS)+["raw_log2tpm_lr500"]:
        for line in LINES:
            for state in ("Acute","DTP"):
                for k in range(3):
                    toy_binary.append({"method":m,"held_out_line":line,"true_state":state,
                        "predicted_state":state,"gsm":f"{m}_{line}_{state}_{k}"})
    stats=_acute_dtp_metrics(toy_binary)
    assert len(stats)==5*5
    assert all(r["balanced_accuracy"]==1 and r["n_correct"]==r["n"] for r in stats)
    assert _acute_dtp_svg(stats).endswith("</svg>\n")
    toy_records=[]
    for line in LINES:
        for category in WASHOUT_GROUPS:
            for k in range(3):
                toy_records.append({"held_out_line":line,"state":category,"gsm":f"{line}_{category}_{k}",
                                    "days_after_drug_removal":1 if category=="ShortWashout" else 3 if category=="LongWashout" else "",
                                    "dtp_decision_margin":str(-1 + k + WASHOUT_GROUPS.index(category)*.1),
                                    "predicted_state":"Acute" if k==0 else "DTP"})
    assert len(washout_descriptives(toy_records))==20
    svg=washout_svg(toy_records)
    assert svg.startswith("<svg") and svg.count("<circle")==60 and svg.endswith("</svg>\n")
    assert set(WASHOUT_PROTOCOL)==set(LINES)
    # Independent, lightweight test for the new 3-class score and SVG logic.
    f=np.array([[.8,.2,.7,.1], [.9,.1,.8,.2], [.1,.8,.2,.7],
                [.2,.9,.1,.8], [.4,.5,.6,.7], [.5,.6,.7,.8]])
    z=np.array([0,0,1,1,2,2])
    directions, scores, func=three_class_signatures(f,f,z,np)
    assert len(directions)==3 and scores.shape==(6,3)
    assert np.allclose(scores,func(f))
    tiny=[]
    for method in THREE_CLASS_METHODS:
        for line in LINES:
            for state in THREE_STATES:
                for rep in range(3):
                    tiny.append({"method":method,"held_out_line":line,"gsm":f"{line}_{state}_{rep}",
                          "true_state":state,"predicted_state":state})
            for state in ("ShortWashout","LongWashout"):
                for rep in range(3):
                    tiny.append({"method":method,"held_out_line":line,"gsm":f"{line}_{state}_{rep}",
                          "true_state":state,"predicted_state":"Control"})
    ts=three_class_statistics(tiny)
    tw=three_class_washout_summary(tiny)
    assert len(ts)==len(THREE_CLASS_METHODS)*5 and len(tw)==len(THREE_CLASS_METHODS)*8
    assert all(r["balanced_accuracy"]==1 for r in ts)
    assert "</svg>" in three_class_svg(ts,tw)
    print("SELF-TEST PASSED (GSM/treatment mapping, Stage16 checks, Washout groups, 4-panel SVG)")



def _acute_dtp_metrics(records):
    """Fold-aware metrics; acute is an explicit class in this experiment."""
    out=[]
    for method in list(METHODS)+["raw_log2tpm_lr500"]:
        mr=[r for r in records if r["method"]==method]
        if Counter(r["true_state"] for r in mr)!=Counter({"Acute":12,"DTP":12}):
            raise ValueError(f"Incomplete balanced Acute/DTP holdouts for {method}")
        for line in list(LINES)+["ALL"]:
            sub=mr if line=="ALL" else [r for r in mr if r["held_out_line"]==line]
            a=[r for r in sub if r["true_state"]=="Acute"]
            d=[r for r in sub if r["true_state"]=="DTP"]
            if len(a)!= (12 if line=="ALL" else 3) or len(d)!=len(a):
                raise ValueError(f"Invalid class counts for {method}/{line}")
            acute_rec=sum(r["predicted_state"]=="Acute" for r in a)/len(a)
            dtp_rec=sum(r["predicted_state"]=="DTP" for r in d)/len(d)
            out.append({"method":method,"held_out_line":line,"n":len(sub),
                 "balanced_accuracy":(acute_rec+dtp_rec)/2,
                 "acute_recall":acute_rec,"dtp_recall":dtp_rec,
                 "n_correct":sum(r["predicted_state"]==r["true_state"] for r in sub),
                 "false_dtp_as_acute":sum(r["predicted_state"]=="Acute" for r in d),
                 "false_acute_as_dtp":sum(r["predicted_state"]=="DTP" for r in a)})
    return out


def _acute_dtp_svg(stats):
    """Vector two-panel performance graphic; source-only exploratory evaluation."""
    from xml.sax.saxutils import escape as esc
    methods=list(METHODS)+["raw_log2tpm_lr500"]
    colors=["#264653","#2A9D8F","#E76F51","#6A5ACD","#8B6577"]
    width,height=1040,450
    b=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       '<rect width="1040" height="450" fill="white"/>',
       '<text x="32" y="36" font-size="20" font-weight="bold" font-family="Arial" fill="#172B38">Acute vs DTP: leave-one-cell-line-out challenge</text>',
       '<text x="32" y="58" font-size="12" font-family="Arial" fill="#526577">Exploratory | GSE193258 | Acute 24h vs DTP 21d (duration confounded) | no held-out line in training</text>',
       '<text x="32" y="94" font-size="15" font-family="Arial" font-weight="bold">A. Held-out pooled metrics</text>',
       '<text x="575" y="94" font-size="15" font-family="Arial" font-weight="bold">B. Balanced accuracy by held-out line</text>']
    for i,method in enumerate(methods):
        rr=next(x for x in stats if x["method"]==method and x["held_out_line"]=="ALL")
        y=126+i*56
        b.append(f'<text x="32" y="{y+12}" font-size="12" font-family="Arial">{esc(method.replace("_"," "))}</text>')
        b.append(f'<rect x="246" y="{y-2}" width="{260*rr["balanced_accuracy"]:.1f}" height="15" fill="{colors[i]}" rx="2"/>')
        b.append(f'<text x="{258+260*rr["balanced_accuracy"]:.1f}" y="{y+11}" font-size="12" font-family="Arial">{rr["balanced_accuracy"]:.3f}</text>')
        for k,line in enumerate(LINES):
            item=next(x for x in stats if x["method"]==method and x["held_out_line"]==line)
            x=590+k*108
            val=item["balanced_accuracy"]
            shade=int(245-130*val)
            b.append(f'<rect x="{x}" y="{y-5}" width="97" height="24" rx="3" fill="rgb({shade},{240-int(65*val)},{243-int(83*val)})"/>')
            b.append(f'<text x="{x+48}" y="{y+11}" font-size="12" text-anchor="middle" font-family="Arial">{val:.3f}</text>')
    for i,line in enumerate(LINES):
        b.append(f'<text x="{638+i*108}" y="115" font-size="12" text-anchor="middle" font-family="Arial">{esc(line)}</text>')
    b.append('<text x="32" y="430" font-size="12" font-family="Arial" fill="#526577">Balanced accuracy is computed per held-out cell line; 3 Acute and 3 DTP samples per fold. No independent validation.</text>')
    b.append('</svg>')
    return '\n'.join(b)+'\n'


def run_acute_vs_dtp():
    """New explicit Acute-vs-DTP LOCO contrast; do not alter stage-16 outputs.

    The selection rule, K=100, C=.01 and four published rank comparators remain
    fixed, but fitting is NEW and uses Acute/DTP labels from TRAIN lines only.
    A raw-log2TPM LR with 500 training-selected genes is a prespecified baseline.
    No external GSE targets, test labels, or source holdout features used in fits.
    """
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    s7=load_module(SCRIPTS/"07_external_stress_test.py","m07_acute_dtp")
    s11=load_module(SCRIPTS/"11_source_only_gene_space_benchmark.py","m11_acute_dtp")
    s13=load_module(SCRIPTS/"13_cross_study_anchor_benchmark.py","m13_acute_dtp")
    s15=load_module(SCRIPTS/"15_rank_method_benchmark.py","m15_acute_dtp")
    if tuple(s15.METHODS)!=METHODS or (s15.K,s15.C,s15.SEED)!=(100,.01,42):
        raise ValueError("Expected existing Stage-15 fixed model recipes")
    protocol=json.loads((META/"evaluation_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("source_study")!="GSE193258" or protocol["flags"].get("biology_approved_for_training") is not False:
        raise ValueError("Unexpected source evaluation protocol")
    srcpath=s7.safe_local_path(protocol["source_matrix"],"processed")
    genes, ids, source=s7.load_matrix(srcpath,protocol["source_matrix_sha256"],np)
    source_labels=protocol["source_gsm_to_label"]
    if ids!=[r["gsm"] for r in source_labels] or source.shape!=(24,19712):
        raise ValueError("Source identities/matrix changed")
    chosen=verified_acute_columns(protocol,csv_rows(META/"geo_samples.csv"),
                                    csv_rows(META/"expression_alignment.csv"))
    acute_ids,acute,rawsha=read_verified_raw(RAW,META/"expression_manifest.json",chosen,
                                              genes,source,ids,np)
    acute_lines=np.array([r["cell_line"] for _,r in acute_ids])
    dtp_idx=[i for i,r in enumerate(source_labels) if r["state"]=="DTP"]
    if len(dtp_idx)!=12:
        raise ValueError("Expected exactly 12 source DTP samples")
    X=np.vstack((acute,source[dtp_idx]))
    ids_all=[gsm for gsm,_ in acute_ids]+[ids[i] for i in dtp_idx]
    groups=np.concatenate((acute_lines,np.array([source_labels[i]["cell_line"] for i in dtp_idx])))
    labels=np.array([0]*12+[1]*12,dtype=int)
    if X.shape!=(24,19712) or len(set(ids_all))!=24:
        raise ValueError("Invalid paired Acute/DTP study cohort")
    for line in LINES:
        if Counter(labels[groups==line])!={0:3,1:3}:
            raise ValueError(f"Unexpected held-out group {line}")

    records=[]
    selected_hashes={}
    for line in LINES:
        train=np.flatnonzero(groups!=line); test=np.flatnonzero(groups==line)
        if len(train)!=18 or len(test)!=6 or Counter(labels[train])!={0:9,1:9}:
            raise ValueError(f"Invalid Acute/DTP LOCO fold {line}")
        # Reuse the Stage-16 method implementation with newly trained Acute-vs-DTP models.
        preds=model_predictions(X[train],labels[train],X[test],genes,
             s11,s13,s15,np,rankdata,f_classif,LogisticRegression,StandardScaler)
        # Source-only raw expression comparator; fixed from stage-35-style baseline.
        rawpipe=make_pipeline(VarianceThreshold(),SelectKBest(f_classif,k=500),
                   StandardScaler(),LogisticRegression(C=.01,solver="liblinear",
                                                      max_iter=2000,random_state=42))
        rawpipe.fit(X[train],labels[train])
        preds["raw_log2tpm_lr500"]=(rawpipe.predict(X[test]).astype(int),
                                   rawpipe.decision_function(X[test]))
        # Save source-only selected feature identities for audit, never held-out selection.
        ix=s11.select_source_features_and_anchors(X[train],labels[train],rankdata,np,
                                                     f_classif,k=100,anchors=256)[0]
        selected_hashes[line]=hashlib.sha256(np.asarray(ix,dtype=np.int64).tobytes()).hexdigest()
        for method,(pred,margin) in preds.items():
            if len(pred)!=6:
                raise ValueError("Invalid per-fold predictions")
            for j,i in enumerate(test):
                records.append({"held_out_line":line,"gsm":ids_all[i],
                      "true_state":"DTP" if labels[i] else "Acute",
                      "method":method,"predicted_state":"DTP" if pred[j] else "Acute",
                      "signed_dtp_margin":"" if margin is None else f"{float(margin[j]):.12g}"})
    if len(records)!=24*5 or len({(r["method"],r["gsm"]) for r in records})!=24*5:
        raise ValueError("Incomplete source-only LOCO predictions")
    stats=_acute_dtp_metrics(records)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    def save_rows(name,rows):
        with (OUTPUT/name).open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_rows("16_acute_vs_dtp_predictions.csv",records)
    save_rows("16_acute_vs_dtp_metrics.csv",stats)
    (OUTPUT/"16_acute_vs_dtp.svg").write_text(_acute_dtp_svg(stats),encoding="utf-8")
    report={"analysis":"source_only_acute_vs_dtp_LOCO_exploratory",
      "status":"NOT_INDEPENDENT_BIOLOGY_PENDING_DURATION_CONFOUNDED",
      "source_study":"GSE193258","n_samples":24,"held_out_lines":list(LINES),
      "labels":{"Acute":"24h treatment","DTP":"21d treatment"},
      "feature_selection":"training-lines-only", "fixed_K":100,"fixed_C":.01,
      "raw_comparator_K":500,"raw_source_sha256":rawsha,
      "source_matrix_sha256":protocol["source_matrix_sha256"],
      "selected_index_sha256_by_line":selected_hashes,
      "biology_approved":False,"independent_validation":False,
      "acute_training_included":True,"dtp_training_included":True,
      "limitations":["Four previously inspected cell lines with three replicates per class/line.",
        "Acute 24h versus DTP 21d is completely confounded with treatment duration.",
        "Binary Acute/DTP separation is not a proof of an intrinsic DTP marker.",
        "Other experiments in the repository have already influenced model development.",
        "Comparator implementations are custom approximations, not official singscore/kTSP implementations."],
      "results":stats}
    (OUTPUT/"16_acute_vs_dtp_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Acute-vs-DTP explicit training: 24 samples; 4 held-out cell lines; source-only feature selection")
    for name in list(METHODS)+["raw_log2tpm_lr500"]:
        z=next(r for r in stats if r["method"]==name and r["held_out_line"]=="ALL")
        folds=", ".join(f"{line}={next(r for r in stats if r['method']==name and r['held_out_line']==line)['balanced_accuracy']:.3f}" for line in LINES)
        print(f"{name}: bACC={z['balanced_accuracy']:.3f} Acute={z['acute_recall']:.3f} DTP={z['dtp_recall']:.3f} correct={z['n_correct']}/24; {folds}")
    print("Saved outputs/acute_challenge/16_acute_vs_dtp_{summary.json,metrics.csv,predictions.csv} and 16_acute_vs_dtp.svg")
    print("EXPLORATORY ONLY: different treatment durations and historically inspected folds")
    return report


# The long drug-free washout endpoint is cell-line dependent. These strings
# are exact GEO treatment metadata; no numeric time is inferred from GSM order.
WASHOUT_PROTOCOL = {
    "H1975": ("osimertinib DTPs short washout 24 hrs.", "osimertinib DTPs long washout 72 hrs.", 3),
    "HCC2935": ("osimertinib DTPs short washout 24 hrs.", "osimertinib DTPs long washout 10 days", 10),
    "HCC827": ("osimertinib DTPs short washout 24 hrs.", "osimertinib DTPs long washout 96 hrs.", 4),
    "PC9": ("osimertinib DTPs short washout 24 hrs.", "osimertinib DTPs long washout 7 days", 7),
}
WASHOUT_GROUPS = ("Control", "Acute", "DTP", "ShortWashout", "LongWashout")


def verified_washout_columns(protocol, metadata, alignments):
    """Exact 24 Washout GSMs, original matrix column IDs and treatment labels.

    No guessing based on GSM proximity, spreadsheet order, or fuzzy titles.
    """
    known = verified_acute_columns(protocol, metadata, alignments)
    samples = [r for r in metadata if r["gse"] == "GSE193258"]
    amap = {}
    for row in alignments:
        if row["gse"] == "GSE193258" and row["filename"] == RAW_NAME:
            if row["gsm"] in amap:
                raise ValueError("Duplicate raw mapping in GSE193258")
            amap[row["gsm"]] = row
    expected = {}
    for line, (short_treat, long_treat, long_days) in WASHOUT_PROTOCOL.items():
        expected[(line,short_treat)] = ("ShortWashout",1)
        expected[(line,long_treat)] = ("LongWashout",long_days)
    out = {}
    for meta in samples:
        key = (meta["reported_cell_line"],meta["reported_treatment"])
        if key not in expected:
            continue
        gsm = meta["gsm"]
        if gsm in known or gsm not in amap or gsm in out:
            raise ValueError("Washout identity duplicated or missing from raw mapping: " + gsm)
        a = amap[gsm]
        if (a["mapping_status"] != "CANDIDATE_REQUIRES_REVIEW" or
            a["geo_title"] != meta["title"] or
            a["cell_line_raw"] != meta["reported_cell_line"] or
            a["treatment_raw"] != meta["reported_treatment"]):
            raise ValueError("Washout metadata/matrix disagreement: " + gsm)
        state,days = expected[key]
        out[gsm] = {"cell_line":key[0],"state":state,"days_after_drug_removal":days,
                    "treatment":key[1],"column":a["matrix_column"]}
    counts = Counter((r["cell_line"],r["state"]) for r in out.values())
    if (len(out) != 24 or len({r["column"] for r in out.values()}) != 24 or
        counts != Counter({(line,state):3 for line in LINES for state in ("ShortWashout","LongWashout")}) or
        len(samples) != 60 or set(out) & set(known)):
        raise ValueError("Expected exactly 24 nonoverlapping washout samples (3 each per line/state)")
    return sorted(out.items()), known


def read_verified_washout(raw_path, manifest_path, records, expected_genes, Xsource, source_ids, known, np):
    """Read washout columns and re-check unmodified source values against prior analysis."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest"].get("GSE193258/" + raw_path.name)
    if not manifest or manifest.get("gzip_crc_verified") is not True or manifest.get("reported_expression_unit") != "log2TPM_abundance":
        raise ValueError("Verified original log2TPM manifest required")
    if sha256(raw_path) != manifest["sha256"]:
        raise ValueError("Washout raw matrix checksum mismatch")
    columns = [known[gsm]["column"] for gsm in source_ids] + [r["column"] for _,r in records]
    with gzip.open(raw_path, "rt", encoding="utf-8-sig", newline="") as stream:
        rows=csv.reader(stream,delimiter="\t")
        header=next(rows)
        if len(header) != 61 or len(set(header)) != 61 or header[0].casefold().strip() != "gene":
            raise ValueError("Unexpected 60-sample original raw header")
        index={h:i for i,h in enumerate(header)}
        if any(x not in index for x in columns):
            raise ValueError("Missing source/washout column in raw matrix")
        positions=[index[c] for c in columns]
        vals=[]
        for i,row in enumerate(rows):
            if len(row)!=61 or i >= len(expected_genes) or row[0]!=expected_genes[i]:
                raise ValueError("Raw gene order/identity differs from source")
            try:
                numbers=[float(row[p]) for p in positions]
            except ValueError as exc:
                raise ValueError("Nonnumeric raw expression row") from exc
            if not all(math.isfinite(v) for v in numbers):
                raise ValueError("Nonfinite washout raw values")
            vals.append(numbers)
    if len(vals)!=len(expected_genes):
        raise ValueError("Unexpected raw gene count")
    mat=np.asarray(vals,dtype=float).T
    if mat.shape != (48,len(expected_genes)) or not np.array_equal(mat[:24],Xsource):
        raise ValueError("Previous 24-source measurements not reproduced exactly")
    return mat[24:]


def washout_descriptives(rows):
    import numpy as np
    out=[]
    for line in LINES:
        for group in WASHOUT_GROUPS:
            rr=[r for r in rows if r["held_out_line"]==line and r["state"]==group]
            if len(rr)!=3:
                raise ValueError(f"Incomplete washout study group {line}/{group}")
            margins=np.array([float(r["dtp_decision_margin"]) for r in rr])
            out.append({"held_out_line":line,"group":group,"n":3,
                        "days_after_drug_removal":rr[0]["days_after_drug_removal"],
                        "mean_margin":float(margins.mean()),"median_margin":float(np.median(margins)),
                        "min_margin":float(margins.min()),"max_margin":float(margins.max()),
                        "fraction_predicted_dtp":sum(r["predicted_state"]=="DTP" for r in rr)/3})
    return out


def washout_svg(rows):
    """Four SVG panels. Axis independently scaled by fold; points are biological samples.

    The x order is experimental condition, NOT a repeated longitudinal trajectory.
    """
    width,height=1130,830
    categories=WASHOUT_GROUPS
    colors={"Control":"#64748B", "Acute":"#D97706", "DTP":"#2563EB",
            "ShortWashout":"#0F766E", "LongWashout":"#9333EA"}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1130" height="830" fill="white"/>',
        '<text x="30" y="38" font-family="Arial,sans-serif" font-size="20" font-weight="bold" fill="#15253B">Frozen Acute-vs-DTP rank score across washout conditions</text>',
        '<text x="30" y="60" font-family="Arial,sans-serif" font-size="12" fill="#526577">GSE193258 | held-out line excluded from training | raw uncalibrated decision margins, NOT DTP probabilities</text>']
    for panel,line in enumerate(LINES):
        px=32+(panel%2)*554; py=98+(panel//2)*352
        rr=[r for r in rows if r["held_out_line"]==line]
        if len(rr)!=15:
            raise ValueError("Expected 15 held-out samples per cell line")
        margins=[float(r["dtp_decision_margin"]) for r in rr]
        low=min(min(margins),0);high=max(max(margins),0)
        spread=max(high-low,0.5)
        low-=0.1*spread;high+=0.1*spread
        plot_top=py+44;plot_bot=py+249
        to_y=lambda m:plot_bot-(float(m)-low)/(high-low)*(plot_bot-plot_top)
        zero=to_y(0)
        parts += [f'<rect x="{px}" y="{py}" width="528" height="326" rx="7" fill="#FBFCFE" stroke="#DCE3EB"/>',
                  f'<text x="{px+14}" y="{py+25}" font-family="Arial,sans-serif" font-size="16" font-weight="bold" fill="#243746">{escape(line)}</text>',
                  f'<text x="{px+14}" y="{py+42}" font-family="Arial,sans-serif" font-size="11" fill="#53657A">Long washout: {WASHOUT_PROTOCOL[line][2]} days; short: 24 hours</text>',
                  f'<line x1="{px+49}" y1="{zero:.1f}" x2="{px+509}" y2="{zero:.1f}" stroke="#D64A4A" stroke-width="1.4" stroke-dasharray="5 4"/>',
                  f'<text x="{px+7}" y="{zero+4:.1f}" font-family="Arial,sans-serif" font-size="10" fill="#9D2E2E">0</text>',
                  f'<text x="{px+4}" y="{plot_top+4}" font-family="Arial,sans-serif" font-size="10" fill="#586A7B">{high:.2f}</text>',
                  f'<text x="{px+4}" y="{plot_bot+4}" font-family="Arial,sans-serif" font-size="10" fill="#586A7B">{low:.2f}</text>']
        for gi,group in enumerate(categories):
            group_rows=sorted((r for r in rr if r["state"]==group),key=lambda x:x["gsm"])
            x=px+94+gi*94
            for j,r in enumerate(group_rows):
                xx=x+(-9,0,9)[j]
                yy=to_y(r["dtp_decision_margin"])
                parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5.3" fill="{colors[group]}" stroke="white" stroke-width="1"/>')
            label={"Control":"Control", "Acute":"Acute", "DTP":"DTP",
                   "ShortWashout":"Wash 24h", "LongWashout":"Wash long"}[group]
            parts.append(f'<text x="{x}" y="{py+275}" font-family="Arial,sans-serif" font-size="10" fill="#34465C" text-anchor="middle">{escape(label)}</text>')
        parts.append(f'<text x="{px+12}" y="{py+309}" font-family="Arial,sans-serif" font-size="10" fill="#64748B">Dashed red = fitted Acute/DTP threshold; 3 distinct replicates per condition</text>')
    parts.append('<text x="32" y="819" font-family="Arial,sans-serif" font-size="11" fill="#586A7B">Exploratory: washout samples were not used in fitting. Long-washout durations differ by cell line. No longitudinal cell tracking.</text>')
    parts.append('</svg>')
    return "\n".join(parts)+"\n"


def run_washout():
    """Score Control / Acute / DTP / washout with the *frozen* Stage-16
    Acute-vs-DTP signed_rank_full model, trained on 3 other cell lines.

    Requires the exact Stage-16 per-sample predictions as an immutable check.
    Writes only new washout-named reports in outputs/acute_challenge.
    """
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    s7=load_module(SCRIPTS/"07_external_stress_test.py","m07_washout")
    s11=load_module(SCRIPTS/"11_source_only_gene_space_benchmark.py","m11_washout")
    s13=load_module(SCRIPTS/"13_cross_study_anchor_benchmark.py","m13_washout")
    s15=load_module(SCRIPTS/"15_rank_method_benchmark.py","m15_washout")
    if tuple(s15.METHODS)!=METHODS or (s15.K,s15.C,s15.SEED)!=(100,.01,42):
        raise ValueError("Existing Acute/DTP Stage-16 frozen methods changed")
    previous=OUTPUT/"16_acute_vs_dtp_predictions.csv"
    previous_summary=OUTPUT/"16_acute_vs_dtp_summary.json"
    if not previous.is_file() or not previous_summary.is_file():
        raise FileNotFoundError("First run stage 16 with --exploratory-only --acute-vs-dtp")
    frozen_summary=json.loads(previous_summary.read_text(encoding="utf-8"))
    protocol=json.loads((META/"evaluation_protocol.json").read_text(encoding="utf-8"))
    if (protocol.get("source_study")!="GSE193258" or
        protocol["flags"].get("biology_approved_for_training") is not False or
        frozen_summary.get("source_matrix_sha256") != protocol["source_matrix_sha256"]):
        raise ValueError("Source/frozen Stage 16 provenance mismatch")
    genepath=s7.safe_local_path(protocol["source_matrix"],"processed")
    genes,ids,source=s7.load_matrix(genepath,protocol["source_matrix_sha256"],np)
    metadata=csv_rows(META/"geo_samples.csv")
    mappings=csv_rows(META/"expression_alignment.csv")
    wash_records,known=verified_washout_columns(protocol,metadata,mappings)
    acute_records,acute,raw_sha=read_verified_raw(RAW,META/"expression_manifest.json",known,
                                                    genes,source,ids,np)
    if frozen_summary.get("raw_source_sha256") != raw_sha:
        raise ValueError("Raw matrix differs from frozen Stage-16 Acute/DTP run")
    wash=read_verified_washout(RAW,META/"expression_manifest.json",wash_records,
                               genes,source,ids,known,np)
    source_labels=protocol["source_gsm_to_label"]
    dtp_idx=[i for i,r in enumerate(source_labels) if r["state"]=="DTP"]
    Xtrain_all=np.vstack((acute,source[dtp_idx]))
    training_ids=[gsm for gsm,_ in acute_records]+[ids[i] for i in dtp_idx]
    training_groups=np.array([r["cell_line"] for _,r in acute_records]+[source_labels[i]["cell_line"] for i in dtp_idx])
    training_labels=np.array([0]*12+[1]*12,dtype=int)
    if Xtrain_all.shape!=(24,19712) or len(set(training_ids))!=24:
        raise ValueError("Unexpected Acute/DTP training cohort")
    previous_rows=[r for r in csv_rows(previous) if r["method"]=="signed_rank_full"]
    previous_by_id={(r["held_out_line"],r["gsm"]):r for r in previous_rows}
    if len(previous_by_id)!=24 or len(previous_rows)!=24:
        raise ValueError("Missing or duplicated signed_rank_full frozen predictions")
    records=[]
    for line in LINES:
        train=np.flatnonzero(training_groups!=line)
        if len(train)!=18 or Counter(training_labels[train])!={0:9,1:9}:
            raise ValueError("Invalid leave-one-cell-line-out training fold")
        parts=[]
        for i,r in enumerate(source_labels):
            if r["cell_line"]==line and r["state"]=="Control":
                parts.append((r["gsm"],"Control",0,source[i]))
        for i,(gsm,r) in enumerate(acute_records):
            if r["cell_line"]==line:
                parts.append((gsm,"Acute",0,acute[i]))
        for i,r in enumerate(source_labels):
            if r["cell_line"]==line and r["state"]=="DTP":
                parts.append((r["gsm"],"DTP",0,source[i]))
        for i,(gsm,r) in enumerate(wash_records):
            if r["cell_line"]==line:
                parts.append((gsm,r["state"],r["days_after_drug_removal"],wash[i]))
        if len(parts)!=15 or Counter(x[1] for x in parts)!=Counter({k:3 for k in WASHOUT_GROUPS}):
            raise ValueError("Missing test samples from held-out line " + line)
        target=np.asarray([x[3] for x in parts])
        model=model_predictions(Xtrain_all[train],training_labels[train],target,genes,
              s11,s13,s15,np,rankdata,f_classif,LogisticRegression,StandardScaler)
        pred,margin=model["signed_rank_full"]
        if margin is None or len(margin)!=15 or not np.all(np.isfinite(margin)):
            raise ValueError("Signed-rank full must produce 15 finite margins")
        for k,(gsm,group,days,_) in enumerate(parts):
            actual="DTP" if int(pred[k]) else "Acute"
            if group in ("Acute","DTP"):
                old=previous_by_id.get((line,gsm))
                if old is None or actual!=old["predicted_state"] or not np.isclose(float(margin[k]),float(old["signed_dtp_margin"]),rtol=1e-7,atol=1e-8):
                    raise ValueError("Frozen Stage 16 sample prediction/margin changed for " + gsm)
            records.append({"held_out_line":line,"gsm":gsm,"state":group,
                  "days_after_drug_removal":days if "Washout" in group else "",
                  "model":"signed_rank_full", "predicted_state":actual,
                  "dtp_decision_margin":f"{float(margin[k]):.12g}",
                  "is_binary_holdout":str(group in ("Acute","DTP")).lower(),
                  "true_binary_label":group if group in ("Acute","DTP") else "NOT_DEFINED"})
    if len(records)!=60 or len({r["gsm"] for r in records})!=60:
        raise ValueError("Expected 60 unique measured samples (15 per held-out line)")
    summary_rows=washout_descriptives(records)
    report={"analysis":"frozen_acute_dtp_rank_model_washout_challenge_v1",
       "status":"EXPLORATORY_WITHIN_STUDY_NOT_INDEPENDENT_VALIDATION",
       "study":"GSE193258","source_matrix_sha256":protocol["source_matrix_sha256"],
       "raw_sha256":raw_sha,"prior_stage16_predictions_sha256":sha256(previous),
       "model":"signed_rank_full","trained_on_states":["Acute","DTP"],
       "trained_on_samples_per_fold":18,"tested_on_samples_per_fold":15,
       "held_out_cell_lines":list(LINES),"washout_duration_days_by_line":{k:v[2] for k,v in WASHOUT_PROTOCOL.items()},
       "n_predictions":len(records),"n_washout_samples":24,
       "models_retrained_on_washout":False,"washout_labels_used_for_fitting":False,
       "previous_stage16_binary_predictions_and_margins_reproduced":True,
       "margin_type":"uncalibrated logit-like signed decision margin; 0=model decision threshold",
       "limitations":["Same study and historically inspected washout experiments; no independent validation.",
           "Acute=24h vs DTP=21d training labels are confounded with exposure duration.",
           "Long washout sampling time differs by cell line (3, 4, 7, 10 days).",
           "Replicates at each state are distinct harvested samples; not repeated-cell longitudinal trajectories.",
           "A signed margin is not a calibrated DTP probability or established biological state measure.",
           "Control/washout states are out-of-training-class challenge conditions."],
       "by_line_and_state":summary_rows}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    p1=OUTPUT/"16_washout_predictions.csv"
    p2=OUTPUT/"16_washout_group_summary.csv"
    p3=OUTPUT/"16_washout_summary.json"
    p4=OUTPUT/"16_washout.svg"
    for path, rr in ((p1,records),(p2,summary_rows)):
        with path.open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    p3.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    p4.write_text(washout_svg(records),encoding="utf-8")
    print("Stage 16 Washout: frozen signed_rank_full Acute-vs-DTP LOCO predictions REPRODUCED")
    for line in LINES:
        vals=[r for r in summary_rows if r["held_out_line"]==line]
        print(f"{line} (long={WASHOUT_PROTOCOL[line][2]}d): " +
              "; ".join(f"{r['group']} median_margin={r['median_margin']:+.3f} DTP_call={r['fraction_predicted_dtp']:.3f}" for r in vals))
    print("Saved outputs/acute_challenge/16_washout_{summary.json,predictions.csv,group_summary.csv} and 16_washout.svg")
    print("EXPLORATORY ONLY: uncalibrated margins, distinct harvested samples, not a biological recovery measurement")
    return report


# ----------------------- Source-only 3-class specificity challenge ------------------
THREE_CLASS_METHODS = (
    "rank_full_lr100_3class",
    "rank_selected_lr100_3class",
    "signed_rank_ovr_full_3class",
    "raw_log2tpm_lr500_3class",
)
THREE_STATES = ("Control", "Acute", "DTP")


def three_class_signatures(train_full_selected, train_local_selected, y, np):
    """Learn three class-vs-rest gene directions on training samples ONLY.

    Multiclass extension of a signed-rank score; NOT an existing, official
    singscore implementation and NOT the same frozen binary Stage-16 model.
    """
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("All three training classes required")
    directions=[]
    for klass in range(3):
        delta=train_local_selected[y==klass].mean(0) - train_local_selected[y!=klass].mean(0)
        up=delta>0
        down=delta<0
        if not up.any() or not down.any():
            raise ValueError(f"Training signature for class {klass} lacks direction balance")
        directions.append((up,down))
    def score(mat):
        return np.column_stack([mat[:,up].mean(1)-mat[:,down].mean(1) for up,down in directions])
    return directions, score(train_full_selected), score


def three_class_predict(Xtrain, ytrain, Xtest, np, rankdata, f_classif,
                        StandardScaler, LogisticRegression, SelectKBest,
                        VarianceThreshold):
    """One fixed 3-class LOCO fit, source-train-only feature selection.

    Test matrix is never used for selection/scaling/fitting. No treatment of
    washout as a fourth or surrogate class. All source gene dimensions fixed.
    """
    from sklearn.pipeline import make_pipeline
    if (Xtrain.ndim!=2 or Xtest.ndim!=2 or Xtrain.shape[1]!=Xtest.shape[1] or
        len(ytrain)!=len(Xtrain) or Counter(ytrain)!={0:9,1:9,2:9}):
        raise ValueError("Expected 27 source training observations (9 per class)")
    full_tr=rankdata(Xtrain,axis=1,method="average")/Xtrain.shape[1]
    variances=full_tr.var(axis=0)
    valid=np.flatnonzero(variances>1e-12)
    if len(valid)<600:
        raise ValueError("Too few varying source genes")
    scores,_=f_classif(full_tr[:,valid],ytrain)
    scores=np.where(np.isfinite(scores),scores,-np.inf)
    order=np.lexsort((valid,-scores))
    selected=valid[order[:100]]
    if not np.all(np.isfinite(scores[order[:100]])):
        raise ValueError("Invalid source-only multiclass gene selection")
    full_te=rankdata(Xtest,axis=1,method="average")/Xtest.shape[1]
    ftr=full_tr[:,selected]
    fte=full_te[:,selected]
    ltr=rankdata(Xtrain[:,selected],axis=1,method="average")/100
    lte=rankdata(Xtest[:,selected],axis=1,method="average")/100
    _,sig_tr,sig_apply=three_class_signatures(ftr,ltr,ytrain,np)
    sig_te=sig_apply(fte)
    recipe={
       "rank_full_lr100_3class":(ftr,fte),
       "rank_selected_lr100_3class":(ltr,lte),
       "signed_rank_ovr_full_3class":(sig_tr,sig_te),
    }
    predictions={}
    for name,(xtr,xte) in recipe.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(
            C=.01,solver="lbfgs",max_iter=3000,random_state=42))
        model.fit(xtr,ytrain)
        pp=model.predict(xte)
        margin=model.decision_function(xte)
        predictions[name]=(pp,margin)
    rawpipe=make_pipeline(VarianceThreshold(),SelectKBest(f_classif,k=500),
         StandardScaler(),LogisticRegression(C=.01,solver="lbfgs",max_iter=3000,random_state=42))
    rawpipe.fit(Xtrain,ytrain)
    predictions["raw_log2tpm_lr500_3class"]=(rawpipe.predict(Xtest),rawpipe.decision_function(Xtest))
    for name,(pr,margin) in predictions.items():
        if pr.shape!=(len(Xtest),) or margin.shape!=(len(Xtest),3):
            raise ValueError("Invalid 3-class output shape from " + name)
    return predictions,selected


def three_class_statistics(rows):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    stats=[]
    for method in THREE_CLASS_METHODS:
        mr=[r for r in rows if r["method"]==method and r["true_state"] in THREE_STATES]
        if len(mr)!=36 or len({r["gsm"] for r in mr})!=36:
            raise ValueError("Missing held-out true-class rows for "+method)
        for group in list(LINES)+["ALL"]:
            sub=mr if group=="ALL" else [r for r in mr if r["held_out_line"]==group]
            y=np.asarray([THREE_STATES.index(r["true_state"]) for r in sub])
            yp=np.asarray([THREE_STATES.index(r["predicted_state"]) for r in sub])
            expected=12 if group=="ALL" else 3
            if Counter(y)!=Counter({0:expected,1:expected,2:expected}):
                raise ValueError("Wrong 3-class test balance in "+group)
            cm=confusion_matrix(y,yp,labels=[0,1,2])
            stats.append({"method":method,"held_out_line":group,"n":len(sub),
                "balanced_accuracy":float(balanced_accuracy_score(y,yp)),
                "macro_f1":float(f1_score(y,yp,average="macro",zero_division=0)),
                "correct":int(np.sum(y==yp)),
                "recall_control":float(cm[0,0]/expected),
                "recall_acute":float(cm[1,1]/expected),
                "recall_dtp":float(cm[2,2]/expected),
                "confusion_matrix":cm.tolist()})
    return stats


def three_class_washout_summary(rows):
    result=[]
    for method in THREE_CLASS_METHODS:
        for line in LINES:
            for group in ("ShortWashout","LongWashout"):
                sub=[r for r in rows if r["method"]==method and r["held_out_line"]==line and r["true_state"]==group]
                if len(sub)!=3:
                    raise ValueError("Missing washout group in three-class report")
                counts=Counter(r["predicted_state"] for r in sub)
                result.append({"method":method,"held_out_line":line,"washout_group":group,
                   "days_after_drug_removal":1 if group=="ShortWashout" else WASHOUT_PROTOCOL[line][2],
                   "n":3,"called_control":counts["Control"],"called_acute":counts["Acute"],
                   "called_dtp":counts["DTP"]})
    return result


def three_class_svg(stats, wash):
    """Reproducible vector figure: 3-class confusion matrices + washout composition."""
    # Explicit counts, not simulated UMAP trajectories or calibrated probabilities.
    import html
    width,height=1120,870
    colors=("#2C79B8","#E79F36","#9B55B5")
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Three-class held-out classification and washout calls">',
       f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
       '<text x="40" y="42" font-family="Arial,sans-serif" font-size="22" font-weight="bold" fill="#16253A">Control / Acute / DTP — held-out cell-line challenge</text>',
       '<text x="40" y="65" font-family="Arial,sans-serif" font-size="13" fill="#576A7C">36 held-out labeled samples; washout groups were never used for training</text>']
    for k,m in enumerate(THREE_CLASS_METHODS):
        x=42+(k%2)*548
        y=108+(k//2)*260
        d=next(r for r in stats if r["method"]==m and r["held_out_line"]=="ALL")
        mat=d["confusion_matrix"]
        name=m.replace("_3class","").replace("_"," ")
        parts.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="15" font-weight="bold" fill="#16253A">{html.escape(name)}</text>')
        parts.append(f'<text x="{x}" y="{y+19}" font-family="Arial,sans-serif" font-size="12" fill="#576A7C">bACC {d["balanced_accuracy"]:.3f}  |  Macro F1 {d["macro_f1"]:.3f}  |  {d["correct"]}/36 correct</text>')
        for i in range(3):
            for j in range(3):
                xx=x+130+j*62; yy=y+42+i*47; v=mat[i][j]
                alpha=0.13+0.65*v/12
                parts.append(f'<rect x="{xx}" y="{yy}" width="58" height="43" rx="4" fill="#3378A6" fill-opacity="{alpha:.3f}"/>')
                parts.append(f'<text x="{xx+29}" y="{yy+27}" text-anchor="middle" font-family="Arial,sans-serif" font-size="17" fill="#112639">{v}</text>')
        for j,n in enumerate(THREE_STATES):
            parts.append(f'<text x="{x+159+j*62}" y="{y+202}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#475569">{n}</text>')
            parts.append(f'<text x="{x+118}" y="{y+69+j*47}" text-anchor="end" font-family="Arial,sans-serif" font-size="11" fill="#475569">{n}</text>')
        parts.append(f'<text x="{x+313}" y="{y+19}" font-family="Arial,sans-serif" font-size="10" fill="#64748B">rows: actual; columns: predicted</text>')
    p_y=638
    parts.append(f'<text x="40" y="{p_y}" font-family="Arial,sans-serif" font-size="17" font-weight="bold" fill="#16253A">Washout: predicted-state composition (multiclass signed-rank extension)</text>')
    parts.append(f'<text x="40" y="{p_y+20}" font-family="Arial,sans-serif" font-size="11" fill="#64748B">Each bar = 3 distinct samples; long washout durations differ among cell lines</text>')
    for j,name in enumerate(THREE_STATES):
        xx=720+j*112
        parts.append(f'<rect x="{xx}" y="{p_y+2}" width="12" height="12" rx="2" fill="{colors[j]}"/><text x="{xx+18}" y="{p_y+12}" font-family="Arial,sans-serif" font-size="11" fill="#425466">{name}</text>')
    for li,line in enumerate(LINES):
        for widx,group in enumerate(("ShortWashout","LongWashout")):
            r=next(row for row in wash if row["method"]=="signed_rank_ovr_full_3class" and row["held_out_line"]==line and row["washout_group"]==group)
            yy=675+li*39
            xx=285+widx*370
            xcur=xx
            for j,n in enumerate(("called_control","called_acute","called_dtp")):
                nval=r[n]; ww=55*nval
                if ww>0:
                    parts.append(f'<rect x="{xcur}" y="{yy}" width="{ww}" height="22" fill="{colors[j]}" rx="1"/>')
                    parts.append(f'<text x="{xcur+ww/2}" y="{yy+16}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="#FFFFFF">{nval}</text>')
                xcur+=ww
            if li==0:
                txt="24h post-withdrawal" if widx==0 else "Line-specific long washout"
                parts.append(f'<text x="{xx+80}" y="{yy-7}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#64748B">{txt}</text>')
            if widx==1:
                parts.append(f'<text x="{xx+178}" y="{yy+16}" font-family="Arial,sans-serif" font-size="11" fill="#526579">{r["days_after_drug_removal"]} days</text>')
        parts.append(f'<text x="205" y="{675+li*39+16}" text-anchor="end" font-family="Arial,sans-serif" font-size="13" fill="#304152">{line}</text>')
    parts.append('<text x="40" y="855" font-family="Arial,sans-serif" font-size="11" fill="#637284">Exploratory within-study result. DTP-like classification is not evidence of post-drug persistence; treatment duration is confounded.</text>')
    parts.append('</svg>')
    return "\n".join(parts)+"\n"


def run_three_class_washout():
    """3-class source-only LOCO; washout is a challenge set, NOT a training class.

    Classifier training differs from the frozen binary Stage-16 model. This
    separate experiment NEVER overwrites prior Stage-16 report artifacts.
    """
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    s7=load_module(SCRIPTS/"07_external_stress_test.py","m07_threeclass")
    previous_path=OUTPUT/"16_washout_summary.json"
    if not previous_path.is_file():
        raise FileNotFoundError("First run --exploratory-only --washout to freeze original contrast")
    prev=json.loads(previous_path.read_text(encoding="utf-8"))
    protocol=json.loads((META/"evaluation_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("source_study")!="GSE193258" or protocol["flags"].get("biology_approved_for_training") is not False:
        raise ValueError("Wrong provisional source protocol")
    genes,ids,source=s7.load_matrix(s7.safe_local_path(protocol["source_matrix"],"processed"),
                                    protocol["source_matrix_sha256"],np)
    if ids!=[r["gsm"] for r in protocol["source_gsm_to_label"]] or source.shape!=(24,19712):
        raise ValueError("Source matrix changed")
    meta=csv_rows(META/"geo_samples.csv")
    mapping=csv_rows(META/"expression_alignment.csv")
    wash_records,known=verified_washout_columns(protocol,meta,mapping)
    acute_records,acute,raw_sha=read_verified_raw(RAW,META/"expression_manifest.json",known,
                                                   genes,source,ids,np)
    wash=read_verified_washout(RAW,META/"expression_manifest.json",wash_records,
                               genes,source,ids,known,np)
    if (prev.get("source_matrix_sha256")!=protocol["source_matrix_sha256"] or
        prev.get("raw_sha256")!=raw_sha):
        raise ValueError("Previously inspected washout inputs changed")
    frozen=protocol["source_gsm_to_label"]
    all_x=np.vstack((source,acute))
    all_ids=ids+[gsm for gsm,_ in acute_records]
    all_group=np.array([r["cell_line"] for r in frozen]+[r["cell_line"] for _,r in acute_records])
    all_state=[r["state"] for r in frozen]+["Acute"]*12
    y=np.array([THREE_STATES.index(s) for s in all_state],dtype=int)
    if all_x.shape!=(36,19712) or len(set(all_ids))!=36 or Counter(all_state)!=Counter({s:12 for s in THREE_STATES}):
        raise ValueError("Wrong Control/Acute/DTP sample construction")
    records=[]; selected_hash={}
    for line in LINES:
        tr=np.flatnonzero(all_group!=line); ts=np.flatnonzero(all_group==line)
        if len(tr)!=27 or len(ts)!=9 or Counter(y[tr])!={0:9,1:9,2:9} or Counter(y[ts])!={0:3,1:3,2:3}:
            raise ValueError("Invalid three-class leave-one-cell-line-out split")
        widx=[i for i,(_,r) in enumerate(wash_records) if r["cell_line"]==line]
        if len(widx)!=6 or Counter(wash_records[i][1]["state"] for i in widx)!={"ShortWashout":3,"LongWashout":3}:
            raise ValueError("Invalid washout challenge group")
        xt=np.vstack((all_x[ts],wash[widx]))
        fit,selected=three_class_predict(all_x[tr],y[tr],xt,np,rankdata,f_classif,
                                        StandardScaler,LogisticRegression,SelectKBest,VarianceThreshold)
        selected_hash[line]=hashlib.sha256(np.asarray(selected,dtype=np.int64).tobytes()).hexdigest()
        details=[(all_ids[i],all_state[i],"") for i in ts]+[(wash_records[i][0],wash_records[i][1]["state"],
                 wash_records[i][1]["days_after_drug_removal"]) for i in widx]
        if len(details)!=15:
            raise ValueError("Unexpected total source/challenge observations")
        for name,(pred,scores) in fit.items():
            if len(pred)!=15:
                raise ValueError("Invalid three-class prediction length")
            for j,(gsm,state,days) in enumerate(details):
                vals=np.asarray(scores[j],dtype=float)
                if not np.all(np.isfinite(vals)):
                    raise ValueError("Nonfinite classifier scores")
                gap=float(np.max(vals)-np.partition(vals,-2)[-2])
                records.append({"method":name,"held_out_line":line,"gsm":gsm,"true_state":state,
                   "predicted_state":THREE_STATES[int(pred[j])],
                   "is_labeled_three_class_holdout":str(state in THREE_STATES).lower(),
                   "days_after_drug_removal":days if state in ("ShortWashout","LongWashout") else "",
                   "decision_score_control":f"{vals[0]:.12g}",
                   "decision_score_acute":f"{vals[1]:.12g}",
                   "decision_score_dtp":f"{vals[2]:.12g}",
                   "top_two_score_gap":f"{gap:.12g}"})
    if len(records)!=60*len(THREE_CLASS_METHODS) or len({(r["method"],r["gsm"]) for r in records})!=len(records):
        raise ValueError("Three-class predictions incomplete/duplicated")
    stats=three_class_statistics(records)
    washes=three_class_washout_summary(records)
    report={"analysis":"source_only_three_class_LOCO_and_washout_challenge",
      "status":"EXPLORATORY_NOT_INDEPENDENT_OR_BIOLOGICALLY_SPECIFIC",
      "source_study":"GSE193258", "trained_states":list(THREE_STATES),
      "trained_n_per_fold":27,"held_out_labeled_n_per_fold":9,"washout_test_n_per_fold":6,
      "source_matrix_sha256":protocol["source_matrix_sha256"],"raw_sha256":raw_sha,
      "prior_washout_report_sha256":sha256(previous_path),
      "threeclass_training_is_new_not_frozen_binary_weights":True,
      "train_only_gene_selection":True,"models":list(THREE_CLASS_METHODS),
      "fixed_k_rank":100,"fixed_k_raw":500,"fixed_C":.01,
      "selected_genes_sha256_by_fold":selected_hash,
      "three_class_holdout_metrics":stats,"washout_predictions_by_line":washes,
      "washout_samples_used_for_training":False,"independent_validation":False,
      "limitations":["Same source study and historically inspected folds: all conclusions exploratory.",
        "Acute=24h and DTP=21d treatment exposure: duration confounded with class.",
        "Washout samples have no gold-standard three-class label; predicted label is similarity, not biological recovery.",
        "Long washout timings differ by cell line (3,10,4,7 days).",
        "Distinct harvested replicates, not trajectories of the same cells.",
        "Decision scores and top-two score gaps are uncalibrated, not probabilities.",
        "Multiclass one-vs-rest signed-rank signature is a developmental extension, not the original frozen binary Stage-16 model or official singscore."]}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    def save(path,rows):
        with (OUTPUT/path).open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save("16_three_class_predictions.csv",records)
    # Nested 3x3 matrices are intentionally retained in the JSON, not in the flat CSV.
    save("16_three_class_metrics.csv",[{k:v for k,v in r.items() if k!="confusion_matrix"} for r in stats])
    save("16_three_class_washout_counts.csv",washes)
    (OUTPUT/"16_three_class_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUTPUT/"16_three_class_washout.svg").write_text(three_class_svg(stats,washes),encoding="utf-8")
    print("Three-class LOCO: Control / Acute / DTP, 27 training, 9 labeled test per fold")
    for name in THREE_CLASS_METHODS:
        a=next(r for r in stats if r["method"]==name and r["held_out_line"]=="ALL")
        print(f"{name}: heldout bACC={a['balanced_accuracy']:.3f}; macroF1={a['macro_f1']:.3f}; "
              f"Control={a['recall_control']:.3f} Acute={a['recall_acute']:.3f} "
              f"DTP={a['recall_dtp']:.3f}; correct={a['correct']}/36")
    for line in LINES:
        items=[w for w in washes if w["method"]=="signed_rank_ovr_full_3class" and w["held_out_line"]==line]
        print(f"{line} washout predictions (signed-rank multiclass): " + "; ".join(
          f"{r['washout_group']}={r['called_control']} Control/{r['called_acute']} Acute/{r['called_dtp']} DTP" for r in items))
    print("Saved outputs/acute_challenge/16_three_class_{summary.json,predictions.csv,metrics.csv,washout_counts.csv} and 16_three_class_washout.svg")
    print("EXPLORATORY: new 3-class source fits; treatment duration confounded; washout labels not ground truth")
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--exploratory-only",action="store_true")
    p.add_argument("--acute-vs-dtp",action="store_true",help="Explicit source-only Acute-vs-DTP LOCO; writes separate report")
    p.add_argument("--washout",action="store_true",help="Frozen Acute-vs-DTP signed-rank model on held-out Washout samples")
    p.add_argument("--three-class-washout",action="store_true",help="Train 3-class models using ONLY other source cell lines; test Control/Acute/DTP and Washout")
    args=p.parse_args()
    if args.self_test:
        self_test()
    elif sum((args.acute_vs_dtp, args.washout, args.three_class_washout))>1:
        p.error("Choose only one challenge mode")
    elif args.three_class_washout and not args.exploratory_only:
        p.error("--three-class-washout requires --exploratory-only")
    elif args.three_class_washout:
        run_three_class_washout()
    elif args.acute_vs_dtp and args.washout:
        p.error("Choose --washout OR --acute-vs-dtp, not both")
    elif args.washout and not args.exploratory_only:
        p.error("--washout requires --exploratory-only")
    elif args.washout:
        run_washout()
    elif args.acute_vs_dtp and not args.exploratory_only:
        p.error("--acute-vs-dtp requires --exploratory-only")
    elif args.acute_vs_dtp:
        run_acute_vs_dtp()
    elif args.exploratory_only:
        run()
    else:
        p.error("Real cohort needs --exploratory-only")

if __name__=="__main__":
    main()
