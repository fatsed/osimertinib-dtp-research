"""12 — Consolidate stages 06–11 into a provenance-checked EXPLORATORY report.

Usage (from repository root):
    python scripts/12_consolidate_results.py --self-test
    python scripts/12_consolidate_results.py

Only standard-library dependencies. Does not train models, adjust parameters,
read raw expression, or call external services. Does NOT certify biological labels
or claim external independent validation. Generated files are gitignored in
outputs/project_report/. Do not use a report if integrity checks fail.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "project_report"
FILES = {
    "06": "outputs/baselines/baseline_summary.json",
    "07": "outputs/external_stress/transfer_summary.json",
    "08": "outputs/robustness/08_robustness_summary.json",
    "09": "outputs/ablation/09_ablation_summary.json",
    "10": "outputs/rank_domain/10_rank_domain_summary.json",
    "11": "outputs/gene_space/11_summary.json",
}
STUDIES = ("GSE153183", "GSE303413", "GSE341996")
AA = "source_shared__target_shared"
AB = "source_shared__target_full"
BA = "source_full__target_shared"
BB = "source_full__target_full"
A09 = "A_shared_genes_shared_ranks"
B09 = "B_shared_genes_full_ranks"
C09 = "C_source_only_selection_full_ranks"
D09 = "D_source_selected_shared_rank_refit"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_inputs(root: Path):
    paths = {stage: root / name for stage, name in FILES.items()}
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing reports; run existing stages first:\n  " + "\n  ".join(missing))
    reports = {stage: json.loads(path.read_text(encoding="utf-8")) for stage, path in paths.items()}
    return reports, paths


def close(a, b, what: str, tol=1e-7):
    if not math.isclose(float(a), float(b), abs_tol=tol, rel_tol=0):
        raise ValueError(f"Cross-stage inconsistency ({what}): {a} vs {b}")


def validate(j: dict, paths: dict, root: Path):
    a, b, c, d, e, f = [j[x] for x in FILES]
    hashes = {
        "06": a["source_sha256"],
        "07": b["source_matrix_sha256"],
        "08": c["source_matrix_sha256"],
        "09": d["source_matrix_sha256"],
        "10": e["source_matrix_sha256"],
        "11": f["source_matrix_sha256"],
    }
    if len(set(hashes.values())) != 1:
        raise ValueError("STOP: stage reports refer to DIFFERENT source matrices: " + json.dumps(hashes))
    if a["protocol_fingerprint"] != f["source_protocol_fingerprint"]:
        raise ValueError("STOP: stages 06 and 11 use different evaluation protocols")
    if a["samples"] != b["source_n"] or a["samples"] != c["n_source"] or a["samples"] != d["source_n"] or a["samples"] != e["source_samples"] or a["samples"] != f["n_samples"]:
        raise ValueError("STOP: source sample counts disagree")
    if a["features_source_only"] != d["source_genes"] or a["features_source_only"] != f["source_genes"]:
        raise ValueError("STOP: source gene counts disagree")
    common = [b["shared_exact_gene_names"], c["n_exact_common_genes"], d["exact_gene_name_intersection_across_three_targets"], e["common_gene_count"]]
    if len(set(common)) != 1:
        raise ValueError("STOP: common gene universes disagree: " + repr(common))
    if b.get("gene_universe_uses_target_headers_only") is not True or e.get("target_gene_headers_used_for_common_universe") is not True:
        raise ValueError("STOP: header-use provenance changed; inspect manually")
    if not (d.get("models_fit_on_source_only") and e.get("models_frozen_between_target_rank_conditions") and f.get("training_only_feature_selection")):
        raise ValueError("STOP: expected source-only/frozen-model conditions not satisfied")
    for filename, expected in (("scripts/07_external_stress_test.py", c.get("stage07_sha256")),
                               ("scripts/07_external_stress_test.py", d.get("stage07_code_sha256")),
                               (str(paths["09"].relative_to(root)), e.get("stage09_report_sha256"))):
        if expected:
            p = root / filename
            if not p.is_file() or sha256(p) != expected:
                raise ValueError(f"STOP: upstream file missing or hash changed: {filename}")
    macro = sum(float(x["balanced_accuracy"]) for x in b["external_metrics"])/3
    close(macro, c["original_fixed_model_macro_bacc"], "07 vs 08 macro bACC")
    close(macro, d["study_macro_balanced_accuracy"][A09], "07 vs 09 method A")
    metrics10 = {(x["study"], x["condition"]): x for x in e["metrics"]}
    metrics09 = {(x["study"], x["condition"]): x for x in d["per_study_metrics"]}
    metrics07 = {x["study"]: x for x in b["external_metrics"]}
    if set(metrics07) != set(STUDIES) or set(e["per_study"]) != set(STUDIES):
        raise ValueError("STOP: expected three external studies missing or changed")
    for s in STUDIES:
        for condition, stage9 in ((AA,A09), (AB,B09), (BB,B09)):
            if condition == BB:
                rhs = metrics09[(s,B09)]
            else:
                rhs = metrics09[(s,stage9)]
            close(metrics10[(s,condition)]["balanced_accuracy"],rhs["balanced_accuracy"],f"10 vs 09 {s} {condition}")
        close(metrics10[(s,AA)]["balanced_accuracy"], metrics07[s]["balanced_accuracy"], f"10 vs 07 {s}")
        if len({metrics10[(s,AA)]["n"], metrics10[(s,AB)]["n"], metrics07[s]["n"]}) != 1:
            raise ValueError(f"STOP: target sample counts disagree for {s}")
    if a["outer_folds"] != 4 or f["folds"] != 4:
        raise ValueError("STOP: unexpected source fold count")
    if f["gate_all_passed"] is not False:
        raise ValueError("STOP: stage 11 success gate changed; review conclusions before reporting")
    return metrics07, metrics09, metrics10, macro


def prepare_rows(j, m7, m9, m10):
    rows = []
    def append(stage, scope, model, metric, val, n="", note=""):
        rows.append({"stage":stage,"scope":scope,"method":model,"metric":metric,"value":f"{float(val):.6f}","n":n,"note":note})
    for model, info in j["06"]["models"].items():
        append("06", "source_LOCO", model, "balanced_accuracy", info["mean_outer_balanced_accuracy"],24,"same-study; exploratory")
        append("06", "source_LOCO", model, "dtp_recall", info["pooled_holdout"]["dtp_recall"],24,"")
    for study in STUDIES:
        append("07",study,"target_shared_ranks","balanced_accuracy",m7[study]["balanced_accuracy"],m7[study]["n"],"posthoc inspected targets")
        for cond in (AA,AB,BA,BB):
            info=m10[(study,cond)]
            append("10",study,cond,"balanced_accuracy",info["balanced_accuracy"],info["n"],"model fixed for AA vs AB")
        append("10",study,"source_shared_model","prediction_switch_count",j["10"]["per_study"][study]["fixed_shared_model_switches_when_target_rank_changes"],m7[study]["n"],"AA vs AB")
    for method, val in j["09"]["study_macro_balanced_accuracy"].items():
        append("09","three_study_macro",method,"balanced_accuracy",val,"","posthoc ablation")
    for method, info in j["08"]["negative_controls"].items():
        n=info["macro_3study"]
        append("08","three_study_macro",method,"null_mean_bacc",n["null_mean"],j["08"]["permutations"],"exploratory permutations")
        append("08","three_study_macro",method,"exploratory_tail",n["exploratory_plus_one_tail_fraction"],j["08"]["permutations"],"not confirmatory p-value")
    for s in j["11"]["stress_summaries"]:
        append("11",s["situation"],s["representation"],"balanced_accuracy",s["mean_balanced_accuracy"],s["n_fold_mask_trials"],"simulated masking; within-study")
        append("11",s["situation"],s["representation"],"prediction_switch_fraction",s["mean_prediction_switch_fraction"],s["n_fold_mask_trials"],"")
    return rows


def plot_svg(j):
    # No external visualization dependencies. These are descriptive exploratory metrics.
    per=j["10"]["per_study"]
    width,height=900,350
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Balanced accuracy under frozen-model rank-space changes">',
       '<rect width="900" height="350" fill="#ffffff"/>',
       '<text x="28" y="32" font-family="Arial,sans-serif" font-size="20" font-weight="bold" fill="#15253b">Frozen-model rank-space sensitivity</text>',
       '<text x="28" y="52" font-family="Arial,sans-serif" font-size="13" fill="#4c6175">Post-hoc exploratory; source model fixed; each study evaluated separately</text>',
       '<rect x="632" y="25" width="13" height="13" fill="#146a85"/><text x="652" y="36" font-family="Arial,sans-serif" font-size="12">Shared-gene ranks (AA)</text>',
       '<rect x="632" y="48" width="13" height="13" fill="#d78c35"/><text x="652" y="59" font-family="Arial,sans-serif" font-size="12">Full-gene ranks (AB)</text>']
    for i, study in enumerate(STUDIES):
        y=91+i*83
        a = next(x for x in j["10"]["metrics"] if x["study"]==study and x["condition"]==AA)["balanced_accuracy"]
        b = next(x for x in j["10"]["metrics"] if x["study"]==study and x["condition"]==AB)["balanced_accuracy"]
        n=per[study]["n_samples"]
        sw=per[study]["fixed_shared_model_switches_when_target_rank_changes"]
        p.append(f'<text x="28" y="{y+14}" font-family="Arial,sans-serif" font-size="15" fill="#15253b">{escape(study)} (n={n})</text>')
        for jdx,(value,color) in enumerate(((a,"#146a85"),(b,"#d78c35"))):
            yy=y+23+jdx*22
            p.append(f'<rect x="225" y="{yy}" width="{max(0,430*float(value)):.2f}" height="15" fill="{color}" rx="3"/>')
            p.append(f'<text x="{235+430*float(value):.0f}" y="{yy+12}" font-family="Arial,sans-serif" font-size="12" fill="#15253b">{100*float(value):.1f}%</text>')
        p.append(f'<text x="724" y="{y+49}" font-family="Arial,sans-serif" font-size="12" fill="#4c6175">{sw}/{n} switched</text>')
    p.append('<text x="28" y="335" font-family="Arial,sans-serif" font-size="12" fill="#627184">Changing target rank reference is a distribution shift, not a new independent validation.</text>')
    p.append('</svg>')
    return "\n".join(p)+"\n"


def build_report(j, paths, root):
    m7, m9, m10, macro = validate(j, paths, root)
    sw = sum(j["10"]["per_study"][s]["fixed_shared_model_switches_when_target_rank_changes"] for s in STUDIES)
    tot = sum(m7[s]["n"] for s in STUDIES)
    f=j["11"]
    neg=j["08"]["negative_controls"]
    lines=[
        "# Project results — exploratory computational audit",
        "",
        "> **Not independent validation.** These studies/folds were previously inspected. Labels remain provisional; cross-study gene annotations and expression scales are not fully harmonized. Control/DTP is confounded by treatment duration, dose and culture conditions. No drug-response prediction or causal DTP claim.",
        "",
        "## Question and result",
        "",
        "**Question:** How stable is a source-trained transcriptomic classifier when the gene-reference space used to compute sample-wise ranks changes?",
        "",
        f"Source cohort: GSE193258, {j['06']['samples']} Control/DTP samples, {j['06']['features_source_only']:,} genes, four held-out cell-line folds.",
        f"Cross-study target candidates: {tot} total across 3 previously inspected studies; {j['07']['shared_exact_gene_names']:,} exact-match gene names in the target-informed shared universe.",
        f"Frozen-model target rank change: **{sw}/{tot} predictions changed**, with identical source-trained classifier and selected features (AA vs AB).",
        "",
        "## Baselines (within source study)","",
        "| Model | Mean outer LOCO bACC | Pooled DTP recall |", "|---|---:|---:|",
    ]
    for name,info in j["06"]["models"].items():
        lines.append(f"| {name} | {100*info['mean_outer_balanced_accuracy']:.1f}% | {100*info['pooled_holdout']['dtp_recall']:.1f}% |")
    lines += ["", "## Target-study fixed-model rank-reference audit (stage 10)","",
              "AA = source ranked within shared genes; target ranked within shared genes. AB = identical source-trained model, but target ranks computed within its full reported gene universe.", "",
              "| Study | n | AA bACC | AB bACC | Prediction switches |", "|---|---:|---:|---:|---:|"]
    for study in STUDIES:
        aa=m10[(study,AA)]["balanced_accuracy"]
        ab=m10[(study,AB)]["balanced_accuracy"]
        ss=j["10"]["per_study"][study]["fixed_shared_model_switches_when_target_rank_changes"]
        lines.append(f"| {study} | {m7[study]['n']} | {100*aa:.1f}% | {100*ab:.1f}% | {ss} |")
    lines += ["", "![Frozen model rank sensitivity](12_rank_space_sensitivity.svg)", "",
              "## Negative controls (stage 08)","",
              "| Control | Null mean (study-macro bACC) | Exploratory tail fraction |", "|---|---:|---:|"]
    for method,info in neg.items():
        z=info["macro_3study"]
        lines.append(f"| {method} | {100*z['null_mean']:.1f}% | {z['exploratory_plus_one_tail_fraction']:.3f} |")
    lines += ["",f"Only {j['08']['permutations']} permutations: tail values are diagnostic, NOT confirmatory p-values.","",
              "## Within-study masking feasibility (stage 11)", "",
              "| Scenario | Baseline bACC | Source-anchor bACC | Baseline switch rate | Anchor switch rate |", "|---|---:|---:|---:|---:|"]
    stress={(x['situation'],x['representation']):x for x in f['stress_summaries']}
    for scenario in ("background_30pct","background_50pct","all_10pct","all_25pct"):
        x=stress[(scenario,"all_gene_percentile")]
        y=stress[(scenario,"source_anchor_ecdf")]
        lines.append(f"| {scenario} | {100*x['mean_balanced_accuracy']:.1f}% | {100*y['mean_balanced_accuracy']:.1f}% | {100*x['mean_prediction_switch_fraction']:.1f}% | {100*y['mean_prediction_switch_fraction']:.1f}% |")
    lines += ["",f"**Stage-11 predeclared exploratory success gate passed:** {f['gate_all_passed']}. No demonstrated improvement for the proposed anchor panel on this source-only missing-gene simulation.","",
              "## Interpretation and scope", "",
              "1. Control/DTP source separation is easy in this setup; a perfect score does not establish biological specificity.",
              "2. Target rank-reference choice alone changes predictions while model weights are fixed. This is a descriptive robustness failure under a representation shift.",
              "3. The within-source artificial gene-deletion experiment was non-discriminating: both candidates remained perfect. Do not claim anchor superiority.",
              "4. External target studies were inspected during development; test-conditioned common gene identifiers were used. A held-out independent study is still required for a confirmatory generalization claim.",
              "5. Source label biology, exact gene annotation, treatment-dose/duration confounding and per-study measurement scales require review. The GSE303413 recorded dose discrepancy remains unresolved.",
              "6. Statistical uncertainty is high: studies have 4, 18 and 6 evaluated samples respectively; no inference should be made from these results alone.",
              "", "## Reproducibility", "",
              "The CSV contains the machine-readable aggregate metrics; the JSON manifest contains SHA-256 hashes of each underlying report. All values were **read, not re-trained** by stage 12.",
              "", "**Status: exploratory computational case study; no independently validated new algorithm.**", ""]
    rows=prepare_rows(j,m7,m9,m10)
    manifest={"status":"EXPLORATORY_NOT_INDEPENDENT_VALIDATION", "stage_reports":{k:{"path":str(v.relative_to(root)).replace('\\','/'),"sha256":sha256(v)} for k,v in paths.items()},"source_matrix_sha256":j['06']['source_sha256'], "source_protocol_fingerprint":j['06']['protocol_fingerprint'],"source_samples":j['06']['samples'], "shared_exact_gene_names":j['07']['shared_exact_gene_names'], "target_n":tot,"target_rank_change_prediction_switches":sw,"source_anchor_superiority_demonstrated":False,"biological_approval":False,"gene_annotation_harmonization":False,"independent_validation":False,"models_trained_by_stage12":False}
    return "\n".join(lines),rows,manifest


def write_reports(root: Path):
    j,paths=read_inputs(root)
    markdown,rows,manifest=build_report(j,paths,root)
    out=root / "outputs" / "project_report"
    out.mkdir(parents=True,exist_ok=True)
    (out/"12_research_summary.md").write_text(markdown,encoding="utf-8")
    with (out/"12_consolidated_metrics.csv").open("w",encoding="utf-8",newline="") as f:
        dw=csv.DictWriter(f,fieldnames=["stage","scope","method","metric","value","n","note"])
        dw.writeheader(); dw.writerows(rows)
    (out/"12_report_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
    (out/"12_rank_space_sensitivity.svg").write_text(plot_svg(j),encoding="utf-8")
    print(f"Stage 12: integrity checks PASSED; source n={manifest['source_samples']}, target n={manifest['target_n']}")
    print(f"Frozen-model prediction switches: {manifest['target_rank_change_prediction_switches']}/{manifest['target_n']}")
    print("Stage 11: no anchor-panel superiority; stage 07 targets already inspected; biology NOT approved")
    print("Saved outputs/project_report/12_research_summary.md, 12_consolidated_metrics.csv, 12_rank_space_sensitivity.svg, 12_report_manifest.json")
    print("EXPLORATORY — NO INDEPENDENT EXTERNAL VALIDATION")


def self_test():
    assert sha256(Path(__file__))
    assert AA != AB and len(STUDIES)==3
    try:
        close(.5, .6, "expected test failure")
        raise AssertionError("Failed to detect disagreement")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"missing"
        try:
            read_inputs(p)
            raise AssertionError("Missing input check did not fail")
        except FileNotFoundError:
            pass
    print("SELF-TEST PASSED (integrity helpers, fail-closed missing reports)")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--root",type=Path,default=ROOT,help="Project root (for verification only)")
    args=parser.parse_args()
    if args.self_test: self_test()
    else: write_reports(args.root.resolve())


if __name__=="__main__":
    main()
