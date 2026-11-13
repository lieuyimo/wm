"""PR68 CPU-only checkpoint admission tests; NO synthetic weight promotion."""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import (
    ARCHIVE_BYTES, ARCHIVE_SHA256, ARMS, SEEDS,
    AdmissionError, ORIGINAL_MODEL_HEAD, ORIGINAL_VAL_SHA256,
    read_manifest, unpack_verified_archive, load_original_models,
)

MANIFEST = Path("examples/evidence/pr60-gpu-training-20261112/checkpoint_manifest.csv")


class PR61CheckpointAdmissionTests(unittest.TestCase):
    def test_original_manifest_six_distinct_controlled_runs(self):
        rows = read_manifest(MANIFEST)
        self.assertEqual(set(rows), {(arm, seed) for arm in ARMS for seed in SEEDS})
        self.assertTrue(all(int(v["params"]) == 52953 for v in rows.values()))
        self.assertEqual(len(ARCHIVE_SHA256), 64)
        self.assertEqual(len(ORIGINAL_VAL_SHA256), 64)
        self.assertEqual(len(ORIGINAL_MODEL_HEAD), 40)
        self.assertEqual(ARCHIVE_BYTES, 6661520)

    def test_archive_absent_hard_failure_not_random_weight_fallback(self):
        with TemporaryDirectory() as root:
            missing = Path(root) / "original-pr61-A100.tar.gz"
            with self.assertRaisesRegex(FileNotFoundError, "BLOCKED_ORIGINAL_PR61_ARCHIVE_MISSING"):
                unpack_verified_archive(missing, MANIFEST)
            with self.assertRaises(FileNotFoundError):
                load_original_models(missing, MANIFEST)

    def test_fake_archive_and_fake_validation_are_both_rejected(self):
        with TemporaryDirectory() as root:
            wrong = Path(root) / "pr60-gpu-checkpoints-20261112.tar.gz"
            wrong.write_bytes(b"not-PR61-original-weights")
            with self.assertRaisesRegex(AdmissionError, "archive SHA256"):
                unpack_verified_archive(wrong, MANIFEST)
            wrong_validation = Path(root) / "PR59-validation.json"
            wrong_validation.write_text('{"split":"validation","scenes":[]}')
            with self.assertRaisesRegex(AdmissionError, "frozen PR59 validation"):
                load_original_models(wrong, MANIFEST, wrong_validation)

    def test_manifest_corruption_rejected_before_weight_loading(self):
        with TemporaryDirectory() as root:
            corrupted = Path(root) / "manifest.csv"
            with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
                keys = list(rows[0])
            rows[0]["seed"] = str(rows[1]["seed"])
            with corrupted.open("w",newline="",encoding="utf-8") as f:
                writer = csv.DictWriter(f,fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(AdmissionError, "duplicate"):
                read_manifest(corrupted)

    def test_real_checkpoints_are_not_embedded_in_this_repository(self):
        # Important semantic regression test: no distribution or inference
        # metrics can be emitted without the verified external archive.
        # This check is about the dedicated handoff module, not the entire
        # repo where unrelated PETS/PR9 weights legitimately exist.
        from experiments.world_model_integration.recovery_pr68 import pr68_checkpoint
        self.assertNotIn("torch.nn.init", Path(pr68_checkpoint.__file__).read_text())
        self.assertIn("weights_only=True", Path(pr68_checkpoint.__file__).read_text())


if __name__ == "__main__":
    unittest.main()
