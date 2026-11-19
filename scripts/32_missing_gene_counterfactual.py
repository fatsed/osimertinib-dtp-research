#!/usr/bin/env python3
"""Stage32: controlled source-PC9 missing-gene counterfactual for Stage31.

Goal: quantify whether Stage31's 7-8 missing selected gene IDs alone could
plausibly explain external score shifts.  EXACT frozen Stage31 source classifiers;
no retraining, threshold adjustment, target-label-based transformations or
feature re-selection.  Simulate Stage31's *same rank-space missingness transform*
on source-study PC9 expression, either with the precise target-coverage mask or
200 fixed-seed same-size random masks.  Show target margins only for comparison.

All analyses retrospective/exploratory.  Target cohort was previously inspected.
Source PC9 is also historically inspected; no independent validation claims.
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
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs" / "cross_study_contrast"
METHODS = ("acute_only_rank_lr100", "dual_contrast_rank_lr100")
STATES = ("Control", "Acute", "DTP")
N_RANDOM_MASKS = 200
MASK_SEED = 314159


def loadmod(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Import failure for {path}")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen Stage31 file is absent: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def encode_csv(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("No rows to write")
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    return buf.getvalue()


def immutable_outputs(files: dict[str, str]):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        p = OUT / name
        if p.exists() and p.read_bytes() != data.encode("utf-8"):
            raise FileExistsError(f"Will not overwrite a previous result: {p}")
    for name, data in files.items():
        p = OUT / name
        if not p.exists():
            p.write_bytes(data.encode("utf-8"))


def mask_rank_matrix(values: np.ndarray, missing_indices: set[int],
                     source_train_mean: np.ndarray) -> np.ndarray:
    """Exactly match Stage31 rank observed/n + source-training means for absent.

    ``values`` is expression in frozen 100-panel column order, from source PC9.
    Missingness does not use source-PC9 labels or target expression.
    """
    x = np.asarray(values, dtype=float)
    means = np.asarray(source_train_mean, dtype=float)
    if x.ndim != 2 or x.shape[1] != 100 or means.shape != (100,):
        raise ValueError("Expected (samples,100) panel and 100 source mean ranks")
    if not np.isfinite(x).all() or not np.isfinite(means).all():
        raise ValueError("Nonfinite feature values")
    if any(type(j) not in (int, np.int64) or not 0 <= j < 100 for j in missing_indices):
        raise ValueError("Invalid missing gene indices")
    observed = [j for j in range(100) if j not in missing_indices]
    if len(observed) < 90:
        raise ValueError("Coverage below Stage31 predeclared floor")
    z = np.repeat(means[None, :], len(x), axis=0)
    z[:, observed] = rankdata(x[:, observed], axis=1, method="average") / len(observed)
    return z


def masks_from_31(panels: dict[str, list[str]], coverage: list[dict],
                  predictions: list[dict]) -> dict[str, set[int]]:
    masks = {}
    for method in METHODS:
        panel = panels[method]
        if len(panel) != 100 or len(set(panel)) != 100:
            raise ValueError(f"Incorrect frozen panel for {method}")
        rows = [r for r in coverage if r["method"] == method]
        pred = [r for r in predictions if r["method"] == method]
        if len(rows) != 9 or len(pred) != 9 or len(set(r["gsm"] for r in rows)) != 9:
            raise ValueError(f"Incorrect Stage31 sample counts for {method}")
        missing_lists = [tuple(x for x in r["missing_gene_ids"].split(";") if x) for r in rows]
        if len(set(missing_lists)) != 1:
            raise ValueError("Target gene-missingness varies across samples; requires separate review")
        missing = set(missing_lists[0])
        if len(missing) != len(missing_lists[0]) or not missing <= set(panel):
            raise ValueError("Repeated or unknown Stage31 missing genes")
        expect = 8 if method == METHODS[0] else 7
        if len(missing) != expect or any(int(r["available"]) != 100-expect or
                                         int(r["missing"]) != expect for r in rows):
            raise ValueError("Stage31 frozen gene coverage has changed")
        pred_by_gsm = {r["gsm"]:r for r in pred}
        if set(pred_by_gsm) != {r["gsm"] for r in rows}:
            raise ValueError("Prediction and coverage sample identity mismatch")
        if any(int(pred_by_gsm[r["gsm"]]["panel_genes_available"]) != int(r["available"]) or
               int(pred_by_gsm[r["gsm"]]["panel_genes_imputed_source_mean"]) != int(r["missing"])
               for r in rows):
            raise ValueError("Stage31 prediction/coverage columns disagree")
        if Counter(r["true_state"] for r in pred) != {"Control":3,"Acute":3,"DTP":3}:
            raise ValueError("Unexpected target labels; descriptive comparison only")
        masks[method] = {j for j, name in enumerate(panel) if name in missing}
    return masks


def stats(states: list[str], margins: np.ndarray) -> dict:
    y = np.asarray(states)
    s = np.asarray(margins, dtype=float)
    if len(y) != 9 or s.shape != (9,) or Counter(y.tolist()) != {x:3 for x in STATES}:
        raise ValueError("Wrong PC9 sample identities or class balance")
    pred = s > 0
    return {"correct":int((pred == (y=="DTP")).sum()),
            "dtp_recall_count":int(pred[y=="DTP"].sum()),
            "control_false_dtp":int(pred[y=="Control"].sum()),
            "acute_false_dtp":int(pred[y=="Acute"].sum()),
            "median_by_group":{st:float(np.median(s[y==st])) for st in STATES}}


def validate_31_frozen(summary: dict, predictions: list[dict]) -> None:
    if (summary.get("source") != "GSE193258" or summary.get("target") != "GSE255958" or
        summary.get("source_training_lines") != ["H1975","HCC2935","HCC827"] or
        summary.get("source_train_n") != 27 or summary.get("target_PC9_n") != 9 or
        summary.get("threshold") != 0.0 or summary.get("C") != .01 or
        summary.get("source_pc9_predictions_exactly_reproduced") is not True or
        summary.get("methods") != list(METHODS)):
        raise ValueError("Stage31 frozen protocol mismatch")
    frozen_sha = summary.get("input_sha256",{})
    expected = {"stage28_predictions":ROOT/"outputs/dtp_specificity/28_predictions.csv",
                "stage28_selected_genes":ROOT/"outputs/dtp_specificity/28_selected_genes.csv",
                "stage27_selected_genes":ROOT/"outputs/dtp_specificity/27_selected_genes.csv"}
    for k,p in expected.items():
        if frozen_sha.get(k) != sha(p):
            raise ValueError(f"Stage31 source snapshot changed: {k}")
    if len(predictions) != 18:
        raise ValueError("Stage31 prediction count changed")
    for m in METHODS:
        rows=[r for r in predictions if r["method"]==m]
        if len(rows)!=9 or len({r["gsm"] for r in rows})!=9:
            raise ValueError("Stage31 duplicate GSM")
        s=np.asarray([float(r["source_only_margin"]) for r in rows])
        labels=[r["true_state"] for r in rows]
        calc=stats(labels,s)
        saved=summary["metrics"][m]
        if calc["correct"]!=saved["correct"] or calc["dtp_recall_count"]!=saved["dtp_correct"] or \
           calc["control_false_dtp"]!=saved["control_false_dtp"] or \
           calc["acute_false_dtp"]!=saved["acute_false_dtp"]:
            raise ValueError(f"Stage31 printed metrics disagree with predictions: {m}")
        for r in rows:
            expected_pred=int(float(r["source_only_margin"])>0)
            if int(r["predicted_dtp"])!=expected_pred or int(r["correct"])!=int(expected_pred==(r["true_state"]=="DTP")):
                raise ValueError(f"Stage31 per-sample prediction changed: {m}/{r['gsm']}")


def run():
    stage31 = loadmod(SCRIPTS/"31_gse255958_frozen_contrast_transfer.py", "stage31_for32")
    stage21 = loadmod(SCRIPTS/"21_group_consistent_rank_selection.py", "stage21_for32")
    summary_path = OUT/"31_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}; complete Stage31 first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    coverage = read_csv(OUT/"31_gene_coverage.csv")
    ext_predictions = read_csv(OUT/"31_predictions.csv")
    validate_31_frozen(summary, ext_predictions)
    panels = stage31.panels_from_frozen()
    masks = masks_from_31(panels, coverage, ext_predictions)

    models, frozen_source_rows, replay_provenance = stage31.source_model_replay(panels)
    genes, gsms, X, y, groups, current_prov = stage21.load_data()
    if replay_provenance != current_prov or X.shape != (36,19712):
        raise ValueError("Stage31/source dataset provenance changed")
    y=np.asarray(y,dtype=int);groups=np.asarray(groups)
    pc9_idx=np.flatnonzero(groups=="PC9")
    if len(pc9_idx)!=9 or Counter(y[pc9_idx].tolist())!={0:3,1:3,2:3}:
        raise ValueError("Missing source PC9 held-out rows")
    states=[STATES[int(y[i])] for i in pc9_idx]
    records=[];random_diagnostics={};report={}
    for method in METHODS:
        m=models[method]
        panel=m["panel"]
        ix={gene:j for j,gene in enumerate(genes)}
        if len(ix)!=len(genes) or any(g not in ix for g in panel):
            raise ValueError("Frozen selected genes no longer in source data")
        raw=X[pc9_idx][:,[ix[g] for g in panel]]
        full_features=mask_rank_matrix(raw,set(),m["train_feature_mean"])
        masked_features=mask_rank_matrix(raw,masks[method],m["train_feature_mean"])
        full=np.asarray(m["classifier"].decision_function(full_features)).reshape(-1)
        masked=np.asarray(m["classifier"].decision_function(masked_features)).reshape(-1)
        expected={r["source_gsm"]:r for r in frozen_source_rows if r["method"]==method}
        for idx,score in zip(pc9_idx,full):
            gsm=gsms[int(idx)]
            if gsm not in expected or not np.isclose(score,expected[gsm]["source_margin"],rtol=1e-7,atol=1e-8):
                raise ValueError(f"Stage31 exact PC9 source replay FAILED {method}/{gsm}")
        original=stats(states,full);simulation=stats(states,masked)
        external_rows=[r for r in ext_predictions if r["method"]==method]
        external=stats([r["true_state"] for r in external_rows],
                       np.asarray([float(r["source_only_margin"]) for r in external_rows]))
        rng=np.random.default_rng(MASK_SEED)  # deterministic; no external labels or scores
        nmiss=len(masks[method])
        rand_correct=[];rand_dtp=[];rand_median_dtp=[]
        for _ in range(N_RANDOM_MASKS):
            random_mask=set(int(i) for i in rng.choice(100,size=nmiss,replace=False))
            random_margin=np.asarray(m["classifier"].decision_function(
                mask_rank_matrix(raw,random_mask,m["train_feature_mean"]))).reshape(-1)
            st=stats(states,random_margin)
            rand_correct.append(st["correct"])
            rand_dtp.append(st["dtp_recall_count"])
            rand_median_dtp.append(st["median_by_group"]["DTP"]-original["median_by_group"]["DTP"])
        random_diagnostics[method]={"n_fixed_seed_masks":N_RANDOM_MASKS,
            "random_correct_min":int(min(rand_correct)),"random_correct_max":int(max(rand_correct)),
            "random_correct_median":float(np.median(rand_correct)),
            "random_dtp_recall_count_min":int(min(rand_dtp)),
            "random_dtp_recall_count_max":int(max(rand_dtp)),
            "random_dtp_recall_count_median":float(np.median(rand_dtp)),
            "random_median_DTP_shift_5th":float(np.quantile(rand_median_dtp,.05)),
            "random_median_DTP_shift_95th":float(np.quantile(rand_median_dtp,.95))}
        for idx,base,mask in zip(pc9_idx,full,masked):
            records.append({"method":method,"gsm":gsms[int(idx)],
                "true_state":STATES[int(y[idx])],
                "full_100_gene_margin":f"{float(base):.12g}",
                "source_pc9_with_external_missing_genes_margin":f"{float(mask):.12g}",
                "margin_delta_from_missingness":f"{float(mask-base):.12g}",
                "full_correct":int((base>0)==(y[idx]==2)),
                "masked_correct":int((mask>0)==(y[idx]==2))})
        report[method]={"n_masked_genes":nmiss,
            "masked_gene_ids":[panel[i] for i in sorted(masks[method])],
            "source_pc9_full":original,"source_pc9_masked":simulation,
            "external_pc9_stage31":external,
            "median_DTP_margin_shift_caused_by_mask_on_SOURCE":float(simulation["median_by_group"]["DTP"]-original["median_by_group"]["DTP"]),
            "median_DTP_margin_shift_source_full_to_external":float(external["median_by_group"]["DTP"]-original["median_by_group"]["DTP"]),
            "median_DTP_margin_shift_source_masked_to_external":float(external["median_by_group"]["DTP"]-simulation["median_by_group"]["DTP"]),
            "random_mask_source_only_diagnostic":random_diagnostics[method]}

    output={"status":"EXPLORATORY_COUNTERFACTUAL_NO_NEW_EXTERNAL_VALIDATION",
            "purpose":"Mechanistic missing-gene attribution on source PC9 using Stage31's exact fixed-missingness transform",
            "number_of_random_masks":N_RANDOM_MASKS,"mask_random_seed":MASK_SEED,
            "source_pc9_n":9,"target_pc9_n":9,"target_used_for_training_or_model_selection":False,
            "external_labels_used_only_for_descriptive_stage31_recap":True,
            "frozen_models_and_threshold_unchanged":True,
            "source_provenance":current_prov,
            "input_sha256":{"stage31_summary":sha(summary_path),
                "stage31_predictions":sha(OUT/"31_predictions.csv"),
                "stage31_gene_coverage":sha(OUT/"31_gene_coverage.csv")},
            "methods":report,
            "limitations":[
                "The source PC9 line and external PC9 study were inspected before developing these methods.",
                "The source missingness counterfactual applies target annotation coverage, not target gene expression.",
                "Random masks are a descriptive sensitivity reference, not a significance test.",
                "Source-to-target score shifts mix missingness, lab conditions, treatment duration, sequencing and biological effects.",
                "AUC=1 on nine previously inspected external samples does not imply validation or justify optimizing a threshold.",
                "No new algorithm novelty, clinical utility, DTP state-specific causality or independent validation established."]}
    md=["# Stage 32 — What caused the Stage31 transfer drop?", "",
        "**Exploratory source-PC9 missing-gene counterfactual, NOT external validation.**", "",
        "Source classifier, k=100, training=27 non-PC9 samples, threshold=0 all frozen. ",
        "Missing gene identities come ONLY from Stage31's target coverage annotations.",
        "The SAME missingness procedure as Stage31 is applied to source PC9 expression.", "",
        "| Method | Genes removed | Source PC9 full correct | Source PC9 masked correct | External correct (Stage31) | Source DTP recall full→masked |",
        "|---|---:|---:|---:|---:|---|"]
    for method in METHODS:
        r=report[method]; a=r["source_pc9_full"];b=r["source_pc9_masked"];ext=r["external_pc9_stage31"]
        md.append(f"| {method} | {r['n_masked_genes']} | {a['correct']}/9 | {b['correct']}/9 | {ext['correct']}/9 | {a['dtp_recall_count']}/3 → {b['dtp_recall_count']}/3 |")
    md += ["", "## Median DTP margin shifts", "",
        "| Method | Source-full to source-masked (missingness alone) | Source-full to external | Source-masked to external |",
        "|---|---:|---:|---:|"]
    for method in METHODS:
        r=report[method]
        md.append(f"| {method} | {r['median_DTP_margin_shift_caused_by_mask_on_SOURCE']:+.4f} | {r['median_DTP_margin_shift_source_full_to_external']:+.4f} | {r['median_DTP_margin_shift_source_masked_to_external']:+.4f} |")
    md += ["", "## Descriptive same-size random masking control", "",
        f"For each method, {N_RANDOM_MASKS} fixed-seed random masks remove the same number of source genes.",
        "Random masks do not use target labels, target expression, or target margins.", "",
        "| Method | Source correct median [min,max] | DTP recall median [min,max] | 5th–95th percentile DTP median-margin change |",
        "|---|---|---|---|"]
    for method in METHODS:
        x=random_diagnostics[method]
        md.append(f"| {method} | {x['random_correct_median']:.1f} [{x['random_correct_min']},{x['random_correct_max']}] | "
                  f"{x['random_dtp_recall_count_median']:.1f} [{x['random_dtp_recall_count_min']},{x['random_dtp_recall_count_max']}] | "
                  f"[{x['random_median_DTP_shift_5th']:+.4f},{x['random_median_DTP_shift_95th']:+.4f}] |")
    md += ["", "## Interpretation rules (specified before source-PC9 run)", "",
        "- If the exact target-coverage mask moves source PC9 margins substantially toward external values, missingness is a plausible contributor, NOT proof of causality.",
        "- If masked source PC9 remains well separated while the external study does not cross threshold 0, cross-study differences beyond panel coverage remain plausible.",
        "- No threshold may be moved, model chosen, or target-label calibration done using these nine external rows.",
        "- AUC=1 with low recall at the fixed threshold indicates ordering and operating-point behavior differ; small n=9 cannot establish robust rank separation.",
        "- GSE255958 has already been inspected. This is only a retrospective mechanistic diagnosis.", ""]
    immutable_outputs({"32_report.md":"\n".join(md),
                       "32_summary.json":json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
                       "32_source_pc9_mask_simulation.csv":encode_csv(records)})
    print("Stage32 complete: simulated identical missing-gene coverage on SOURCE PC9; frozen classifier and threshold unchanged")
    for method in METHODS:
        r=report[method]; a=r['source_pc9_full'];b=r['source_pc9_masked'];c=r['external_pc9_stage31']
        print(f"{method}: source full={a['correct']}/9 source masked={b['correct']}/9 external Stage31={c['correct']}/9; "
              f"DTP recall source full={a['dtp_recall_count']}/3 masked={b['dtp_recall_count']}/3; "
              f"DTP shift mask={r['median_DTP_margin_shift_caused_by_mask_on_SOURCE']:+.4f} external={r['median_DTP_margin_shift_source_full_to_external']:+.4f}")
    print("Saved outputs/cross_study_contrast/32_{report.md,summary.json,source_pc9_mask_simulation.csv}")
    print("EXPLORATORY: no new external validation, no calibration or algorithm selection")


def self_test():
    v=np.arange(1.,901.).reshape(9,100)
    mean=np.full(100,.505)
    z=mask_rank_matrix(v,set(),mean)
    assert z.shape==(9,100) and np.array_equal(z,rankdata(v,axis=1)/100)
    mm={1,20,40,60,80,90,98,99}
    masked=mask_rank_matrix(v,mm,mean)
    assert np.all(masked[:,list(mm)]==.505)
    observed=[j for j in range(100) if j not in mm]
    assert np.array_equal(masked[:,observed],rankdata(v[:,observed],axis=1)/92)
    assert np.allclose(mask_rank_matrix(np.log1p(v),mm,mean),masked)
    try:mask_rank_matrix(v,set(range(11)),mean)
    except ValueError:pass
    else:raise AssertionError("Should reject more than ten missing genes")
    labels=list(STATES)*3
    scores=np.array([-1.,-.4,.5]*3)
    metrics=stats(labels,scores)
    assert metrics['correct']==9 and metrics['dtp_recall_count']==3
    assert N_RANDOM_MASKS==200 and MASK_SEED==314159 and METHODS[1]=='dual_contrast_rank_lr100'
    print("SELF-TEST PASSED: same Stage31 missing-rank transform, monotonic rank invariance, coverage stop, source-only score metrics")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--exploratory-only',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test()
    elif args.exploratory_only:run()
    else:ap.error("A real counterfactual run requires --exploratory-only")


if __name__=='__main__':
    main()
