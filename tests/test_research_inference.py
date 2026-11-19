"""Independent, synthetic tests for the frozen Stage40 research-only prediction CLI.

No GEO data, original training predictions, exported model payload, or network
access is required. These tests verify SOFTWARE BEHAVIOR, not scientific validity.

From repo root:
    python -m unittest discover -s tests -p 'test_*.py' -v
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "40_frozen_research_model_package.py"
MODEL_NAMES = (
    "acute_only_rank_lr100__stage28_baseline",
    "acute_only_rank_lr100__contrast_protected_shrinkage_gamma1",
    "dual_contrast_rank_lr100__stage28_baseline",
    "dual_contrast_rank_lr100__contrast_protected_shrinkage_gamma1",
)
GENES = [f"SYNTH_GENE_{i:03d}" for i in range(100)]


def load_stage40():
    if not SCRIPT.is_file():
        raise FileNotFoundError(f"Missing required repository file: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("stage40_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import Stage40")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_model(index: int) -> dict:
    """Deliberately fake parameters, not any fitted research model."""
    return {
        "panel_genes": GENES[:],
        "rank_linear_weights": [float((j % 9 - 4) * (index + 1) / 50) for j in range(100)],
        "rank_linear_intercept": float(-0.18 + index / 20),
        "source_mean_panel_ranks": [0.5] * 100,
        "n_selected_genes": 100,
        "minimum_gene_coverage": 90,
        "threshold": 0.0,
    }


def synthetic_package() -> dict:
    return {
        "version": 1,
        "status": "RESEARCH_ONLY_NOT_INDEPENDENTLY_VALIDATED",
        "n_train": 27,
        "model_parameters": {"threshold": 0.0},
        "models": {name: synthetic_model(i) for i, name in enumerate(MODEL_NAMES)},
    }


def write_matrix(path: Path, genes: list[str], *, sample2: bool = True) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        w = csv.writer(file, delimiter="\t")
        w.writerow(["gene", "SIMULATED_SAMPLE_1"] + (["SIMULATED_SAMPLE_2"] if sample2 else []))
        for gene in genes:
            i = int(gene.rsplit("_", 1)[-1])
            row = [gene, float(i % 17 + 1)]
            if sample2:
                row.append(float((i * 11) % 23 + 1))
            w.writerow(row)


def run_cli(package: Path, matrix: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--predict", str(matrix),
         "--model-path", str(package), "--output", str(output)],
        cwd=REPO, text=True, capture_output=True, check=False,
    )


def read_scores(path: Path) -> dict[tuple[str, str], tuple[str, int, int]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        data = list(csv.DictReader(file))
    return {
        (r["model"], r["sample_id"]):
        (r["DTP_like_margin"], int(r["DTP_like_call"]), int(r["genes_available"]))
        for r in data
    }


class Stage40IndependentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_stage40()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package_path = self.root / "synthetic_models.json"
        self.package_path.write_text(json.dumps(synthetic_package()), encoding="utf-8")
        self.matrix_path = self.root / "input.tsv"
        write_matrix(self.matrix_path, GENES)
        self.out = self.root / "scores.csv"

    def test_full_cli_predicts_four_models_times_two_samples(self):
        result = run_cli(self.package_path, self.matrix_path, self.out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("8 predictions", result.stdout)
        rows = read_scores(self.out)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(coverage == 100 for _, _, coverage in rows.values()))
        self.assertTrue(all(math.isfinite(float(score)) for score, _, _ in rows.values()))

    def test_order_of_gene_rows_does_not_change_scores(self):
        first = run_cli(self.package_path, self.matrix_path, self.out)
        self.assertEqual(first.returncode, 0, first.stderr)
        shuffled = self.root / "shuffled.tsv"
        write_matrix(shuffled, list(reversed(GENES)))
        other = self.root / "shuffled_scores.csv"
        second = run_cli(self.package_path, shuffled, other)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(read_scores(self.out), read_scores(other))

    def test_missing_ten_genes_allowed_and_means_are_used(self):
        write_matrix(self.matrix_path, GENES[10:])
        result = run_cli(self.package_path, self.matrix_path, self.out)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(v[2] == 90 for v in read_scores(self.out).values()))
        model = synthetic_model(0)
        z, count = self.mod.panel_rank_for_sample(
            model, {g: float(i + 1) for i, g in enumerate(GENES[10:])}
        )
        self.assertEqual(count, 90)
        np.testing.assert_array_equal(z[:10], [0.5] * 10)

    def test_missing_eleven_genes_fails_closed(self):
        write_matrix(self.matrix_path, GENES[11:] + [f"EXTRA_GENE_{i:03d}" for i in range(11)])
        result = run_cli(self.package_path, self.matrix_path, self.out)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("minimum 90", result.stderr)
        self.assertFalse(self.out.exists())

    def test_rejects_duplicate_gene_and_duplicate_sample(self):
        self.matrix_path.write_text(
            "gene\tSAMPLE_1\nSYNTH_GENE_000\t1\nSYNTH_GENE_000\t2\n",
            encoding="utf-8",
        )
        self.assertRaisesRegex(ValueError, "Duplicate/empty gene", self.mod.read_wide_matrix, self.matrix_path)
        self.matrix_path.write_text("gene\tSAMPLE_1\tSAMPLE_1\nSYNTH_GENE_000\t1\t2\n", encoding="utf-8")
        self.assertRaisesRegex(ValueError, "Duplicate/empty sample", self.mod.read_wide_matrix, self.matrix_path)

    def test_rejects_bad_expression_values(self):
        for value in ("-1", "NaN", "inf", "not_a_number"):
            with self.subTest(value=value):
                p = self.root / "bad.tsv"
                p.write_text(f"gene\tS1\nSYNTH_GENE_000\t{value}\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.mod.read_wide_matrix(p)

    def test_rejects_corrupted_package_shapes(self):
        pkg = synthetic_package()
        pkg["models"][MODEL_NAMES[0]]["rank_linear_weights"].pop()
        self.assertRaisesRegex(ValueError, "Corrupt", self.mod.validate_package, pkg)
        pkg = synthetic_package()
        pkg["models"][MODEL_NAMES[0]]["panel_genes"][1] = pkg["models"][MODEL_NAMES[0]]["panel_genes"][0]
        self.assertRaisesRegex(ValueError, "Corrupt", self.mod.validate_package, pkg)

    def test_immutable_existing_output_not_overwritten(self):
        first = run_cli(self.package_path, self.matrix_path, self.out)
        self.assertEqual(first.returncode, 0, first.stderr)
        initial = self.out.read_bytes()
        second = run_cli(self.package_path, self.matrix_path, self.out)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self.out.read_bytes(), initial)
        edited = self.root / "edited.tsv"
        write_matrix(edited, GENES)
        # Use csv.reader/writer instead of editing raw lines. On Windows,
        # mixing preserved CRLF with Path.write_text newline conversion can
        # produce CRCRLF and fail parsing before the overwrite guard is hit.
        with edited.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f, delimiter="\t"))
        # Change an expression value, not merely gene-row order.
        rows[1][1] = "100000.0"
        with edited.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerows(rows)
        changed = run_cli(self.package_path, edited, self.out)
        # A changed sample matrix must not silently rewrite an existing output.
        self.assertNotEqual(changed.returncode, 0)
        self.assertIn("not overwriting", changed.stderr)
        self.assertEqual(self.out.read_bytes(), initial)

    def test_research_only_status_is_explicit(self):
        result = run_cli(self.package_path, self.matrix_path, self.out)
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.out.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertTrue(all(r["status"] == "RESEARCH_ONLY_UNVALIDATED" for r in rows))
        self.assertNotIn("probability", ",".join(rows[0].keys()).lower())


if __name__ == "__main__":
    unittest.main()
