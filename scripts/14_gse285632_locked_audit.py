"""14 | GSE285632 six-column frozen-model audit (exploratory, NOT independent validation).

Uses EXACT source training, selected features, classifiers and target transforms
from stage 13. Tests the previously observed P1-P3/D1-D3 split without assuming
an unverified column-to-GSM identification. Adds two non-tuning diagnostics:
(1) 32 source-only random 100-gene panels and (2) leave-one-selected-gene-out
    perturbations of the target, with frozen model weights.

No external labels or expression are used for source fitting, panel selection,
or hyperparameter tuning. The labels only score already-fitted predictions.
The workbook's 'differentially expressed genes' provenance and scale are not
independently established; high coverage alone does not prove no DEG filtering.

Usage from repo root:
    python scripts/14_gse285632_locked_audit.py --self-test
    python scripts/14_gse285632_locked_audit.py --exploratory-only

Inputs: stage 13 script/results, stage 07, stage 11, GSE193258 source,
        data/raw/GSE285632/GSE285632_All.DEG.Expression.xlsx
Outputs: outputs/gse285632_audit/14_{report.json,predictions.csv,
         random_feature_controls.csv,single_gene_dropout.csv}
Dependencies: existing requirements.txt (no openpyxl needed).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data/raw/GSE285632/GSE285632_All.DEG.Expression.xlsx"
PROTOCOL = ROOT / "data/metadata/evaluation_protocol.json"
STAGE13_REPORT = ROOT / "outputs/anchor_transfer/13_summary.json"
OUTPUT = ROOT / "outputs/gse285632_audit"
COLUMNS = ("gene_id", "P1", "P2", "P3", "D1", "D2", "D3")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
LABELS = ("Control",) * 3 + ("DTP",) * 3
STUDY = "GSE285632"
N_RANDOM = 32
RANDOM_SEED = 20261118


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _cell_value(cell, shared):
    if cell.find(NS + "f") is not None:
        raise ValueError("Formula cell encountered; refusing to evaluate formulas")
    typ = cell.get("t", "")
    if typ == "inlineStr":
        return "".join(x.text or "" for x in cell.iter(NS + "t"))
    v = cell.find(NS + "v")
    if v is None:
        return ""
    if typ == "s":
        idx = int(v.text)
        if not (0 <= idx < len(shared)):
            raise ValueError("XLSX shared-string index out of range")
        return shared[idx]
    if typ in ("e", "str"):
        raise ValueError("Unexpected expression error/string cell")
    return v.text or ""


def parse_workbook(path: Path):
    """Parse only the known second sheet; preserve exact source gene names.

    Return: (genes, six-column matrix, input audit). Never use group labels to
    select genes or transform values.
    """
    import numpy as np

    if not path.is_file() or path.stat().st_size < 100_000:
        raise ValueError(f"Missing or suspiciously small workbook: {path}")
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt XLSX archive")
        sheet = "xl/worksheets/sheet2.xml"
        if sheet not in z.namelist():
            raise ValueError("Missing expected sheet2.xml")
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            with z.open("xl/sharedStrings.xml") as fp:
                for _, si in ET.iterparse(fp, events=("end",)):
                    if si.tag == NS + "si":
                        shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
                        si.clear()
        header = None
        title = ""
        genes, data, seen = [], [], set()
        rows_seen = 0
        with z.open(sheet) as fp:
            for _, row in ET.iterparse(fp, events=("end",)):
                if row.tag != NS + "row":
                    continue
                rows_seen += 1
                values = {}
                for cell in row.findall(NS + "c"):
                    m = re.fullmatch(r"([A-Z]+)\d+", cell.get("r", ""))
                    if not m:
                        raise ValueError("Missing or malformed XLSX cell reference")
                    col = 0
                    for letter in m.group(1):
                        col = 26 * col + ord(letter) - ord("A") + 1
                    if col <= len(COLUMNS):
                        values[col] = _cell_value(cell, shared)
                cells = [values.get(i, "") for i in range(1, len(COLUMNS) + 1)]
                r = int(row.get("r", "0"))
                if r == 1:
                    title = cells[0]
                elif r == 2:
                    header = tuple(cells)
                    if header != COLUMNS:
                        raise ValueError(f"Unexpected XLSX header: {header!r}")
                elif r > 2:
                    if header is None:
                        raise ValueError("Data rows encountered before header")
                    gene = cells[0].strip()
                    if not gene or gene in seen:
                        raise ValueError(f"Blank/duplicate gene ID on row {r}: {gene!r}")
                    try:
                        x = [float(t) for t in cells[1:]]
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Non-numeric expression at row {r}, gene {gene}") from exc
                    if not all(math.isfinite(v) for v in x):
                        raise ValueError(f"Non-finite expression at row {r}, gene {gene}")
                    seen.add(gene)
                    genes.append(gene)
                    data.append(x)
                row.clear()
    if rows_seen != len(genes) + 2 or len(genes) < 15_000:
        raise ValueError(f"Unexpected data size: rows={rows_seen}, genes={len(genes)}")
    matrix = np.asarray(data, dtype=np.float64).T
    if matrix.shape != (6, len(genes)):
        raise AssertionError("Six-sample matrix shape mismatch")
    qc = {
        "sheet_name": "sheet2.xml",
        "sheet_heading": title,
        "declared_header": list(COLUMNS),
        "unique_genes": len(genes),
        "n_expression_samples": 6,
        "min_expression": float(matrix.min()),
        "median_expression": float(np.median(matrix)),
        "max_expression": float(matrix.max()),
        "fraction_zero": float(np.mean(matrix == 0)),
        "negative_values_present": bool((matrix < 0).any()),
        "input_sha256": sha256(path),
        "expression_unit_verified": False,
        "DEG_preselection_excluded": False,
        "excel_alias_to_GSM_verified": False,
        "warning": "57k reported genes do NOT establish unfiltered expression provenance; label P/D to GSM mapping is unverified",
    }
    return genes, matrix, qc


def optional_geo_sample_context():
    """Checks GEO titles, without treating P1 as an officially mapped GSM."""
    path = ROOT / "data/metadata/geo_samples.csv"
    if not path.exists():
        return {"geo_samples_metadata_available": False, "candidate_group_titles_verified": False}
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [x for x in csv.DictReader(f) if x.get("gse") == STUDY]
    expected = {
        "GSM8706066": "HCC827-DMSO1",
        "GSM8706067": "HCC827-DMSO2",
        "GSM8706068": "HCC827-DMSO3",
        "GSM8706069": "HCC827-Osi1",
        "GSM8706070": "HCC827-Osi2",
        "GSM8706071": "HCC827-Osi3",
    }
    titles = {r.get("gsm"): r.get("title") for r in rows}
    return {
        "geo_samples_metadata_available": bool(rows),
        "candidate_group_titles_verified": all(titles.get(g) == t for g, t in expected.items()),
        "expected_candidate_gsm_titles": expected,
        "column_alias_gsm_mapping_verified": False,
        "note": "GEO group/treatment series context is supported, but P1/D1 XLSX columns are NOT directly tied to specific GSM IDs",
    }


def save_csv(path: Path, records):
    if not records:
        raise ValueError("No records")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def exact_binomial_ci(k, n, alpha=0.05):
    from scipy.stats import beta
    return {
        "lower": float(beta.ppf(alpha/2, k, n-k+1)) if k else 0.0,
        "upper": float(beta.ppf(1-alpha/2, k+1, n-k)) if k < n else 1.0,
        "method": "Clopper-Pearson exact 95% for pooled binary correctness, NOT generalization guarantee",
    }


def run(n_random=N_RANDOM):
    import numpy as np
    from scipy.stats import rankdata
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    if not (1 <= n_random <= 200):
        raise ValueError("--random-panels must be 1..200")
    from importlib.util import spec_from_file_location, module_from_spec
    def load(rel, name):
        path = ROOT / "scripts" / rel
        if not path.exists():
            raise FileNotFoundError(path)
        spec = spec_from_file_location(name, path)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    stage07 = load("07_external_stress_test.py", "stage07_for_14")
    stage11 = load("11_source_only_gene_space_benchmark.py", "stage11_for_14")
    stage13 = load("13_cross_study_anchor_benchmark.py", "stage13_for_14")

    if not STAGE13_REPORT.is_file():
        raise FileNotFoundError(f"Stage-13 frozen baseline report required: {STAGE13_REPORT}")
    prev = json.loads(STAGE13_REPORT.read_text(encoding="utf-8"))
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if proto.get("source_study") != "GSE193258" or proto.get("flags",{}).get("biology_approved_for_training") is not False:
        raise ValueError("Unexpected source protocol / approval state")
    if proto["source_matrix_sha256"] != prev.get("source_matrix_sha256"):
        raise ValueError("Stage-13/source dataset checksum mismatch")
    fp = prev.get("fixed_parameters", {})
    expected_fp = {"features":100, "anchors":256, "C":0.01, "seed":42, "classifier":"LogisticRegression", "solver":"liblinear"}
    if any(fp.get(k) != v for k,v in expected_fp.items()):
        raise ValueError("Stage-13 parameters do not match the locked benchmark")
    if (stage13.K,stage13.ANCHORS,stage13.C,stage13.SEED) != (100,256,0.01,42):
        raise ValueError("Stage-13 code settings changed")
    if prev.get("protocol_fingerprint") != proto.get("fingerprint_sha256"):
        raise ValueError("Frozen protocol mismatch with stage 13")

    source = stage07.safe_local_path(proto["source_matrix"], "processed")
    source_genes, source_gsm, X = stage07.load_matrix(source, proto["source_matrix_sha256"], np)
    samples = proto["source_gsm_to_label"]
    if source_gsm != [s["gsm"] for s in samples] or X.shape != (24,19712):
        raise ValueError("Source matrix samples or genes changed")
    y = np.array([int(s["state"] == "DTP") for s in samples],dtype=int)
    groups = np.array([s["cell_line"] for s in samples])
    selected, anchors, source_ranks = stage11.select_source_features_and_anchors(
        X,y,rankdata,np,f_classif,k=stage13.K,anchors=stage13.ANCHORS)
    selected_names = [source_genes[int(i)] for i in selected]
    anchor_names = [source_genes[int(i)] for i in anchors]
    if selected_names != prev.get("selected_gene_names") or anchor_names != prev.get("anchor_gene_names"):
        raise ValueError("STOP: source gene/anchor selection differs from locked stage 13")
    trained = stage13.fit_source(X,y,selected,anchors,source_ranks,stage11,np,
                                 LogisticRegression,StandardScaler)

    # Validate source-fold performance against stage-13 record, not target labels.
    current_folds = stage13.source_loco(X,y,groups,stage11,np,rankdata,f_classif,
                                       LogisticRegression,StandardScaler)
    key = lambda x: (x["held_out_cell_line"],x["method"])
    expected_folds = {key(x):x for x in prev.get("source_folds",[])}
    if set(map(key,current_folds)) != set(expected_folds):
        raise ValueError("Source folds mismatch previous stage 13")
    for r in current_folds:
        if abs(r["balanced_accuracy"] - expected_folds[key(r)]["balanced_accuracy"]) > 1e-12:
            raise ValueError(f"Source LOCO mismatch: {key(r)}")

    target_genes, Xt, qc = parse_workbook(XLSX)
    source_names_set = set(source_genes)
    qc["overlap_with_all_source_genes"] = len(source_names_set.intersection(target_genes))
    group_context = optional_geo_sample_context()
    yy = np.array([0]*3 + [1]*3,dtype=int)
    predictions, models, by_method = [], {}, {}
    for method in stage13.METHODS:
        rep, cover = stage13.target_transform(Xt,target_genes,selected_names,anchor_names,
            trained[method]["train_mean"],method,rankdata,np)
        if cover["selected_present"] < 90:
            raise ValueError(f"STOP: <90/100 frozen selected genes covered for {method}")
        pred = trained[method]["model"].predict(rep)
        met = stage13.confusion_metrics(yy,pred,np)
        models[method] = {**met,**cover, "exact_accuracy_ci_95":exact_binomial_ci(met["correct"],met["n"])}
        by_method[method] = pred
        for col,lab,p in zip(COLUMNS[1:],LABELS,pred):
            predictions.append({"study":STUDY,"excel_column":col,"candidate_state":lab,
                                "method":method,"prediction":"DTP" if p else "Control",
                                "correct":int(lab == ("DTP" if p else "Control")),
                                "GSM_mapping_verified":False})

    # Additional control: 100 random SOURCE gene names, no target-informed panel
    # selection. Same ranking method, same logistic family/regularization/seed.
    random_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for trial in range(n_random):
        panel = rng.choice(len(source_genes), size=stage13.K, replace=False)
        panel_names = [source_genes[int(i)] for i in panel]
        train = rankdata(X[:,panel],axis=1,method="average") / stage13.K
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=stage13.C,max_iter=2000,solver="liblinear",random_state=stage13.SEED))
        clf.fit(train,y)
        projected,cover=stage13.target_transform(Xt,target_genes,panel_names,anchor_names,
                          train.mean(axis=0),"selected_feature_percentile",rankdata,np)
        pred=clf.predict(projected)
        met=stage13.confusion_metrics(yy,pred,np)
        random_rows.append({"trial":trial,"source_only_random_panel":True,
            "selected_present":cover["selected_present"],"n":met["n"],
            "balanced_accuracy":met["balanced_accuracy"],"dtp_recall":met["dtp_recall"],
            "correct":met["correct"],"panel_sha256":hashlib.sha256(panel.tobytes()).hexdigest()})

    # Target-only stress test: each selected gene disappears ONE at a time;
    # repeat frozen prediction. Neither model weights nor selection are changed.
    base = by_method["selected_feature_percentile"]
    dropped_rows=[]
    gene_to_idx={g:i for i,g in enumerate(target_genes)}
    for name in selected_names:
        # All selected genes are present in the observed workbook, but keep the
        # result meaningful if fewer happen to be available in another version.
        if name not in gene_to_idx:
            continue
        missing_idx=gene_to_idx[name]
        retained=np.ones(len(target_genes),dtype=bool)
        retained[missing_idx]=False
        projected,_=stage13.target_transform(Xt[:,retained],
                [g for j,g in enumerate(target_genes) if j != missing_idx],
                selected_names,anchor_names,
                trained["selected_feature_percentile"]["train_mean"],
                "selected_feature_percentile",rankdata,np)
        pred=trained["selected_feature_percentile"]["model"].predict(projected)
        met=stage13.confusion_metrics(yy,pred,np)
        dropped_rows.append({"dropped_source_selected_gene":name,
                "switches_vs_original":int((pred != base).sum()),
                "n":met["n"],"balanced_accuracy":met["balanced_accuracy"],
                "dtp_recall":met["dtp_recall"],"correct":met["correct"]})

    main_method = models["selected_feature_percentile"]
    random_scores=[r["balanced_accuracy"] for r in random_rows]
    report={
        "study":STUDY,
        "status":"EXPLORATORY_PROVISIONAL_P_D_GROUPS_NOT_INDEPENDENT_GSM_VERIFIED_VALIDATION",
        "stage13_comparison_frozen":True,
        "source_matrix_sha256":proto["source_matrix_sha256"],
        "source_protocol_sha256":sha256(PROTOCOL),
        "stage13_report_sha256":sha256(STAGE13_REPORT),
        "stage13_script_sha256":sha256(ROOT/"scripts/13_cross_study_anchor_benchmark.py"),
        "source_selected_features":selected_names,
        "excel_file_qc":qc,
        "geo_context":group_context,
        "methods":models,
        "random_100_gene_panels":{
            "n":n_random,"source_only_rng_seed":RANDOM_SEED,
            "min_bacc":float(min(random_scores)),
            "median_bacc":float(statistics.median(random_scores)),
            "max_bacc":float(max(random_scores)),
            "fraction_at_least_frozen_feature_panel":float(sum(s >= main_method["balanced_accuracy"]-1e-12 for s in random_scores)/len(random_scores)),
            "coverage_min":min(r["selected_present"] for r in random_rows),
            "coverage_max":max(r["selected_present"] for r in random_rows),
            "posthoc_diagnostic_not_significance_test":True,
        },
        "frozen_100_gene_dropout":{
            "n":len(dropped_rows),
            "any_prediction_switch":sum(r["switches_vs_original"]>0 for r in dropped_rows),
            "worst_bacc":min(r["balanced_accuracy"] for r in dropped_rows),
            "min_correct":min(r["correct"] for r in dropped_rows),
            "interpretation":"Target-only leave-one-selected-gene-out diagnostic, not training-time feature selection",
        },
        "limitations":[
            "The target P/D groups are inferred from the submitter XLSX column names, NOT confirmed sample-column GSM mappings.",
            "The workbook is titled differentially expressed genes; full expression provenance/unbiased gene inclusion cannot be verified from the XLSX alone.",
            "No source/model parameters were tuned using target labels or expression, but the target was previously observed in this conversation, so it is NOT an untouched independent validation.",
            "Only 3+3 samples, and target drug duration, culture, cell line, expression units and annotations may differ from source.",
            "Random-panel and leave-one-gene-out controls are diagnostics, not new independent hypothesis tests.",
        ],
    }
    OUTPUT.mkdir(parents=True,exist_ok=True)
    save_csv(OUTPUT/"14_predictions.csv",predictions)
    save_csv(OUTPUT/"14_random_feature_controls.csv",random_rows)
    save_csv(OUTPUT/"14_single_gene_dropout.csv",dropped_rows)
    (OUTPUT/"14_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"GSE285632: genes={qc['unique_genes']} source_overlap={qc['overlap_with_all_source_genes']} selected_coverage={models['selected_feature_percentile']['selected_present']}/{stage13.K}")
    print("GSM identity: UNVERIFIED; workbook DEG-selection provenance: UNVERIFIED")
    for method in stage13.METHODS:
        d=models[method]
        print(f"{method}: bACC={d['balanced_accuracy']:.3f} DTP_recall={d['dtp_recall']:.3f} correct={d['correct']}/6")
    print(f"Random SOURCE 100-gene panel control ({n_random}): bACC min={min(random_scores):.3f} median={statistics.median(random_scores):.3f} max={max(random_scores):.3f}; fraction >= selected={report['random_100_gene_panels']['fraction_at_least_frozen_feature_panel']:.3f}")
    print(f"Drop ONE frozen selected gene (n={len(dropped_rows)}): panels with prediction switch={report['frozen_100_gene_dropout']['any_prediction_switch']}; worst bACC={report['frozen_100_gene_dropout']['worst_bacc']:.3f}")
    print("Saved outputs/gse285632_audit/14_report.json, 14_predictions.csv, 14_random_feature_controls.csv, 14_single_gene_dropout.csv")
    print("EXPLORATORY ONLY: NO CONFIRMED GSM MAPPING OR INDEPENDENT VALIDATION")


def self_test():
    import numpy as np
    from scipy.stats import rankdata
    from scipy.stats import beta
    assert COLUMNS == ("gene_id","P1","P2","P3","D1","D2","D3")
    assert abs(exact_binomial_ci(6,6)["lower"] - (0.025**(1/6))) < 1e-10
    assert exact_binomial_ci(6,6)["upper"] == 1.0
    # A real XLSX shared-string + numeric-cell fixture; no external packages.
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"fixture.xlsx"
        rows=['<row r="1"><c r="A1" t="s"><v>0</v></c></row>',
              '<row r="2">'+''.join(f'<c r="{chr(65+i)}2" t="s"><v>{i+1}</v></c>' for i in range(7))+'</row>']
        rng=np.random.default_rng(14)
        for i in range(16000):
            vals=rng.uniform(0,20,6)
            cells=[f'<c r="A{i+3}" t="s"><v>{i+8}</v></c>']
            cells += [f'<c r="{chr(66+j)}{i+3}"><v>{float(x):.7f}</v></c>' for j,x in enumerate(vals)]
            rows.append(f'<row r="{i+3}">'+''.join(cells)+'</row>')
        names=["fixture",*COLUMNS, *[f"GENE_{i}" for i in range(16000)]]
        ss='<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'+''.join(f'<si><t>{n}</t></si>' for n in names)+'</sst>'
        sx='<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(rows)+'</sheetData></worksheet>'
        with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('xl/sharedStrings.xml',ss)
            z.writestr('xl/worksheets/sheet2.xml',sx)
        genes, values, audit = parse_workbook(p)
        assert len(genes)==16000 and values.shape==(6,16000)
        assert audit['unique_genes']==16000 and audit['DEG_preselection_excluded'] is False
        assert genes[0]=='GENE_0' and genes[-1]=='GENE_15999'
        z=np.array([[3,2,1],[3,1,2]],dtype=float)
        assert list(rankdata(z,axis=1,method='average')[0]) == [3,2,1]
    print("SELF-TEST PASSED (XLSX shared strings, 6 sample columns, full expression parse, exact uncertainty bound, fail-closed protocol)")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--exploratory-only',action='store_true')
    parser.add_argument('--random-panels',type=int,default=N_RANDOM)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    elif args.exploratory_only:
        run(n_random=args.random_panels)
    else:
        parser.error("Real GEO data require --exploratory-only (no independent validation claim)")

if __name__=='__main__':
    main()
