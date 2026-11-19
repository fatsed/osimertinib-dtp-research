#!/usr/bin/env python3
"""Stage-47 auxiliary diagnostic: sensitivity to held-out Control choice.

This is NOT Stage 48, model development, or external validation.

Hold the *training-only* anchored_tuned threshold fixed. For each held-out
cell line and model, substitute each of its three known Controls as the target
reference and evaluate on the SAME six held-out Acute/DTP samples. Never
choose a Control based on these test outcomes. The analysis is post-hoc and
explicitly exploratory: target labels are used for reporting only.

Run from repository root:
  python scripts/stage47_reference_sensitivity.py --self-test
  python scripts/stage47_reference_sensitivity.py --run

Does not edit prior stages or their artifacts; writes one CSV with immutable
file policy into outputs/stage47_control_anchored/.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("outputs/stage47_control_anchored")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def compute(rows: list[dict[str, str]], thresholds: list[dict[str, str]]) -> list[dict]:
    required = {"fold", "variant", "split", "sample_id", "cell_line", "group", "score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Scores CSV missing fields: {sorted(required - set(rows[0] if rows else {}))}")
    allowed_variants = {"base", "protected"}
    fold_names = sorted(set(r["fold"] for r in rows))
    if len(fold_names) != 4 or {r["variant"] for r in rows} != allowed_variants:
        raise ValueError("Expected exactly four held-out lines and base/protected variants")
    if len(rows) != 288 or {r["split"] for r in rows} != {"test", "train_oof"}:
        raise ValueError("Expected complete Stage-47 288-row test/train_oof score table")
    duplicates = [(r["fold"], r["variant"], r["split"], r["sample_id"]) for r in rows]
    if len(duplicates) != len(set(duplicates)):
        raise ValueError("Duplicate sample rows in score table")
    thresholds_by_key = {}
    for r in thresholds:
        if r.get("method") == "anchored_tuned":
            k = (r["fold"], r["variant"])
            if k in thresholds_by_key:
                raise ValueError(f"Duplicate anchored threshold for {k}")
            thresholds_by_key[k] = float(r["threshold"])
    expected_keys = {(f, v) for f in fold_names for v in sorted(allowed_variants)}
    if set(thresholds_by_key) != expected_keys or not all(math.isfinite(x) for x in thresholds_by_key.values()):
        raise ValueError("Missing or invalid anchored_tuned thresholds")

    output = []
    for fold in fold_names:
        for variant in ("base", "protected"):
            test = [r for r in rows if r["fold"] == fold and r["variant"] == variant and r["split"] == "test"]
            train = [r for r in rows if r["fold"] == fold and r["variant"] == variant and r["split"] == "train_oof"]
            if len(test) != 9 or len(train) != 27 or {r["cell_line"] for r in test} != {fold}:
                raise ValueError(f"Incorrect outer fold split {fold}/{variant}")
            if fold in {r["cell_line"] for r in train} or len({r["cell_line"] for r in train}) != 3:
                raise ValueError(f"Outer test data present in training scores {fold}/{variant}")
            if Counter(r["group"] for r in test) != {"Control":3, "Acute":3, "DTP":3}:
                raise ValueError(f"Unexpected source test labels {fold}/{variant}")
            controls = sorted((r for r in test if r["group"] == "Control"), key=lambda r:r["sample_id"])
            fixed_six = [r for r in test if r["group"] in {"Acute", "DTP"}]
            threshold = thresholds_by_key[(fold, variant)]
            for anchor in controls:
                ref_score = float(anchor["score"])
                scores = [float(r["score"]) for r in fixed_six]
                if not all(math.isfinite(s) for s in scores + [ref_score]):
                    raise ValueError("Nonfinite scores")
                pred = [score - ref_score > threshold for score in scores]
                tp = sum(v and r["group"] == "DTP" for v,r in zip(pred,fixed_six))
                acute_fp = sum(v and r["group"] == "Acute" for v,r in zip(pred,fixed_six))
                output.append({
                    "fold":fold, "variant":variant,
                    "reference_id":anchor["sample_id"],
                    "is_predeclared_reference":int(anchor is controls[0]),
                    "reference_score":format(ref_score, ".15g"),
                    "frozen_train_oof_threshold":format(threshold, ".15g"),
                    "effective_raw_threshold":format(ref_score+threshold, ".15g"),
                    "dtp_tp_of_3":int(tp), "dtp_fn_of_3":int(3-tp),
                    "acute_fp_of_3":int(acute_fp), "acute_tn_of_3":int(3-acute_fp),
                    "six_sample_balanced_accuracy":format(0.5*(tp/3+(3-acute_fp)/3), ".12g"),
                })
    if len(output) != 24:
        raise AssertionError("Expected 24 (4 folds x 2 variants x 3 Controls) diagnostic rows")
    # Both models must be evaluated on identical biological reference identities.
    for fold in fold_names:
        ids = [{r["reference_id"] for r in output if r["fold"]==fold and r["variant"]==v} for v in ("base", "protected")]
        if ids[0] != ids[1]:
            raise AssertionError(f"Different controls across variants for {fold}")
    return output


def write_once(path: Path, rows: list[dict]) -> None:
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    contents = f.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != contents:
            raise FileExistsError(f"Existing diagnostic differs; refusing overwrite: {path}")
        print(f"Existing identical file retained: {path.resolve()}")
    else:
        path.write_bytes(contents)
        print(f"Saved: {path.resolve()}")


def self_test() -> None:
    # End-to-end toy cohort with the same four-fold, three-state structure.
    lines = ("L1","L2","L3","L4")
    rows = []
    for fold in lines:
        for variant in ("base", "protected"):
            for line in lines:
                for state, value in (("Control", -0.5), ("Acute", 0.1), ("DTP", 0.8)):
                    for i in range(3):
                        rows.append({"fold":fold, "variant":variant,
                                     "split":"test" if line==fold else "train_oof",
                                     "sample_id":f"{line}_{state}_{i}", "cell_line":line,
                                     "group":state, "score":str(value+0.1*i)})
    thresholds = [{"fold":f,"variant":v,"method":"anchored_tuned","threshold":"0.9"} for f in lines for v in ("base","protected")]
    out = compute(rows, thresholds)
    assert len(out)==24
    assert all(r["dtp_tp_of_3"]==3 for r in out)
    assert all(r["acute_fp_of_3"]==0 for r in out)
    assert sum(r["is_predeclared_reference"] for r in out)==8
    # Changing test labels must NOT affect the frozen threshold used.
    assert all(float(r["frozen_train_oof_threshold"]) == 0.9 for r in out)
    print("SELF-TEST PASSED: fixed thresholds, 24 references, 6-sample paired target tests (synthetic only).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--input", type=Path, default=BASE / "47_source_scores.csv")
    p.add_argument("--thresholds", type=Path, default=BASE / "thresholds_train_only.csv")
    p.add_argument("--output", type=Path, default=BASE / "47_reference_sensitivity.csv")
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    if not a.run:
        p.error("Use --run to perform the exploratory sensitivity analysis or --self-test")
    out = compute(read_csv(a.input), read_csv(a.thresholds))
    write_once(a.output, out)
    print("STAGE 47 / POST-HOC DIAGNOSTIC ONLY: frozen threshold, identical 3 Acute + 3 DTP test samples.")
    print("No anchor was chosen based on test outcomes; every available Control was enumerated.")
    print("Positive class is source GSE193258 DTP; this is NOT independent validation.")
    print("\nProtected model: DTP detected and Acute false positives across three Controls:")
    for fold in sorted(set(r["fold"] for r in out)):
        sub=[r for r in out if r["fold"]==fold and r["variant"]=="protected"]
        print(f"  {fold:8s}  DTP TP {min(r['dtp_tp_of_3'] for r in sub)}-{max(r['dtp_tp_of_3'] for r in sub)}/3; "
              f"Acute FP {min(r['acute_fp_of_3'] for r in sub)}-{max(r['acute_fp_of_3'] for r in sub)}/3; "
              f"predeclared ref: {next(r['reference_id'] for r in sub if r['is_predeclared_reference'])}")


if __name__ == "__main__":
    main()
