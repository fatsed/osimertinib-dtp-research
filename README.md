# Osimertinib DTP Research — Exploratory Cross-Study Analysis

Reproducible research code for exploratory **observed-state classification** of osimertinib-treated lung cancer cell-line transcriptomic samples. The project compares expression/rank-based representations and examines sensitivity to cell-line holdout, external-study transfer, missing genes, and differences in experimental design.

> **Research status (November 2026):** exploratory, supervisor-review stage. Neither biological state identity nor independent external validation has been established. The results **do not** establish a clinically useful response predictor, a causal/persistence-specific biomarker, or a new state-of-the-art algorithm.

## Data and experimental scope

- Development source: [GSE193258](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193258), four cell lines (H1975, HCC2935, HCC827, PC9). The primary binary task uses 24 Control/DTP samples; the three-class development task uses 36 Control/Acute/DTP samples.
- Exploratory three-class cross-study stress test: [GSE255958](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255958), nine PC9 samples (three each Control, Acute, DTP). Three *Resistant* samples are **excluded**, not relabeled DTP.
- Additional previously inspected exploratory datasets are assessed by stages 07–15; consult their audit reports for sample eligibility, expression scales, and label caveats.
- Source Acute (24 h) and DTP (21 d) labels are confounded with treatment duration. Other studies differ in dose, culture conditions, gene annotations, and expression-processing provenance.

## Project layout

```text
scripts/            Sequential data audit, modeling, stress tests, and evidence checks (01–20)
data/raw/           Downloaded source material (local; Git-ignored)
data/processed/     Derived matrices (local; Git-ignored)
data/metadata/      GEO metadata, alignment, manifests, evaluation protocol (local; Git-ignored)
outputs/            Results, figures and audit reports (local; Git-ignored)
requirements.txt   Pinned Python package versions
```

The public repository is primarily a **code snapshot**, not a complete data or results archive. The metadata, matrices and historical output files required by later stages must be generated locally from the audited original sources. A fresh clone cannot directly execute stages 16–20 without these prerequisites.

## Environment

Tested locally with Python 3.12 (64-bit) and the pinned requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The scripts should be invoked from the repository root. The data acquisition/QC workflow begins at `scripts/01_geo_audit.py` and proceeds through stages 02–05 before modeling. Review each script's usage, source-data requirements and guard conditions; do not infer that a file present on disk has approved biological labels.

## Audit and evidence commands (when upstream local inputs exist)

```powershell
python scripts/18_protocol_integrity_audit.py --self-test
python scripts/18_protocol_integrity_audit.py --full-hashes --run-upstream-self-tests

python scripts/19_confounding_feasibility_audit.py --self-test
python scripts/19_confounding_feasibility_audit.py

python scripts/20_final_evidence_pack.py --self-test
python scripts/20_final_evidence_pack.py
```

Key generated local reports:

- `outputs/protocol_audit/18_audit.md` / `18_audit.json`
- `outputs/confounding_feasibility/19_design_audit.md` / `19_design_audit.json`
- `outputs/final_evidence/20_paper_ready_tables.md` / `20_evidence_manifest.json`

Stage 20 recalculates descriptive metrics from stored per-sample predictions and checks identities against prior reports. **It does not retrain models, optimize thresholds, test significance or perform new independent validation.**

## Summary of the current three-class exploration

The source leave-one-cell-line-out evaluation contains 36 pooled held-out predictions across four distinct folds; each fold fits on 27 samples. The raw-expression logistic-regression comparator had **35/36 correct** in the pooled source holdouts. In the previously inspected GSE255958 nine-sample comparison it had **8/9 correct** when trained on all 36 source samples, and **9/9 correct** when source PC9 was excluded (27 training samples). For the selected-rank classifier the analogous external scores were **9/9** and **7/9**.

These are small, **post-hoc exploratory** scores. Do not use them to select a winner and then present the same external dataset as untouched validation. Full details and per-class recalls are in `20_paper_ready_tables.md` after local generation.

## Open limitations and required review

1. Biological labels, expression mappings, and biological replicate interpretation still require independent source-supported review.
2. Acute and DTP differ in treatment duration; their contrast cannot isolate intrinsic DTP specificity.
3. External transfer is historically inspected and is not a confirmatory validation set.
4. Cross-study gene-symbol compatibility and normalization/processing differences are not fully resolved; source and external expression units must not be pooled naïvely.
5. Six GSE303413 treatment-dose annotation discrepancies remain unresolved.
6. Stage 18 passes mechanical integrity checks, not a proof that every leakage path has been ruled out.
7. Small test sizes make precision and generalization claims unwarranted.

The appropriate manuscript framing is **exploratory observed-state classification and cross-study stress testing**. Confirmatory or clinical claims require independently adjudicated samples and a genuinely untouched test set.

## Reproducibility and responsible publication

Retain source accessions, download-file checksums, preprocessing manifests, code commit SHA and prediction-level output hashes with the final study record. Avoid committing local raw datasets, derived data, environment folders or unreviewed output files. When writing about performance, distinguish source LOCO from cross-study tests, binary from three-class tasks, and descriptive scores from independently validated results.
