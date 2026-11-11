#!/usr/bin/env python3
"""Offline archival tests using tiny F32 rows, never real audit files."""

import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

import archive_rows as archive


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.job = self.root / "new-job"
        self.job.mkdir()
        self.rows = []
        for directory, filename in (("oracle", "case.f32"), ("engine", "case.actual.f32")):
            path = self.job / directory / filename
            path.parent.mkdir()
            path.write_bytes(struct.pack("<ffff", 1, 2, 3, 4) * 32)
            self.rows.append({"path": path.relative_to(self.job).as_posix(),
                              "size_bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.prepared = self.job / "prepared.json"
        self.prepared.write_text(json.dumps({"suite_id": archive.SUITE_ID}))
        self.review_path = self.job / "independent-review.json"
        self.review = {
            "valid_evidence": True, "passed": True, "suite_id": archive.SUITE_ID,
            "pilot": False, "cases_checked": 1, "rows_checked": 128,
            "job_directory": str(self.job),
            "prepared_sha256": hashlib.sha256(self.prepared.read_bytes()).hexdigest(),
            "raw_files": self.rows,
        }
        self.save_review()
        zstd = shutil.which("zstd")
        self.zstd = Path(zstd) if zstd else None

    def save_review(self):
        self.review_path.write_text(json.dumps(self.review))

    def run_archive(self):
        if self.zstd is None:
            self.skipTest("zstd is not installed")
        return archive.archive(self.root, self.job, self.zstd)

    def test_roundtrip_and_receipts(self):
        original_review = self.review_path.read_bytes()
        receipt, removal = self.run_archive()
        self.assertTrue(removal["passed"])
        self.assertEqual(self.review_path.read_bytes(), original_review)
        self.assertEqual(len(receipt["files"]), 2)
        self.assertEqual(removal["archive_sha256"],
                         hashlib.sha256((self.job / "archive.json").read_bytes()).hexdigest())
        for row, record in zip(self.rows, receipt["files"]):
            self.assertFalse((self.job / row["path"]).exists())
            self.assertEqual(archive.decompress_digest(
                self.zstd, self.job / record["compressed_path"], row["size_bytes"]),
                row["sha256"])

    def test_valid_numeric_failure_is_retained(self):
        self.review["passed"] = False
        self.save_review()
        receipt, _ = self.run_archive()
        self.assertIs(receipt["independent_review"]["passed"], False)

    def test_pilot_scope_is_retained(self):
        self.review.update(pilot=True, cases_checked=1, rows_checked=128)
        self.save_review()
        receipt, _ = self.run_archive()
        self.assertTrue(receipt["independent_review"]["pilot"])
        self.assertEqual(receipt["independent_review"]["cases_checked"], 1)
        self.assertEqual(receipt["independent_review"]["rows_checked"], 128)

    def test_exact_pilot_suite_id_is_allowed_only_as_pilot(self):
        self.prepared.write_text(json.dumps({"suite_id": archive.PILOT_SUITE_ID}))
        self.review.update(
            pilot=True, suite_id=archive.PILOT_SUITE_ID,
            prepared_sha256=hashlib.sha256(self.prepared.read_bytes()).hexdigest(),
        )
        self.save_review()
        receipt, _ = self.run_archive()
        self.assertEqual(receipt["suite_id"], archive.PILOT_SUITE_ID)
        self.assertTrue(receipt["independent_review"]["pilot"])

    def test_pilot_suite_cannot_claim_full_coverage(self):
        self.prepared.write_text(json.dumps({"suite_id": archive.PILOT_SUITE_ID}))
        self.review.update(
            pilot=False, suite_id=archive.PILOT_SUITE_ID,
            prepared_sha256=hashlib.sha256(self.prepared.read_bytes()).hexdigest(),
        )
        self.save_review()
        with self.assertRaisesRegex(ValueError, "authorized new text-coverage"):
            archive.validate_job(self.root, self.job)

    def test_invalid_review_cannot_archive(self):
        self.review["valid_evidence"] = False
        self.save_review()
        with self.assertRaisesRegex(ValueError, "not valid"):
            archive.validate_job(self.root, self.job)

    def test_truthy_string_is_not_valid(self):
        self.review["valid_evidence"] = "true"
        self.save_review()
        with self.assertRaisesRegex(ValueError, "not valid"):
            archive.validate_job(self.root, self.job)

    def test_other_job_review_is_rejected(self):
        self.review["job_directory"] = str(self.root / "other")
        self.save_review()
        with self.assertRaisesRegex(ValueError, "different job"):
            archive.validate_job(self.root, self.job)

    def test_old_suite_is_rejected(self):
        self.review["suite_id"] = "older-audit"
        self.save_review()
        with self.assertRaisesRegex(ValueError, "suite differ"):
            archive.validate_job(self.root, self.job)

    def test_old_prepared_hash_is_rejected(self):
        self.prepared.write_text(json.dumps({"suite_id": archive.SUITE_ID, "changed": True}))
        with self.assertRaisesRegex(ValueError, "bind prepared"):
            archive.validate_job(self.root, self.job)

    def test_raw_changes_are_rejected(self):
        (self.job / self.rows[0]["path"]).write_bytes(b"x" * self.rows[0]["size_bytes"])
        with self.assertRaisesRegex(ValueError, "raw file differs"):
            archive.validate_job(self.root, self.job)

    def test_unreviewed_raw_file_is_rejected(self):
        (self.job / "oracle" / "extra.f32").write_bytes(b"\x00" * 4)
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            archive.validate_job(self.root, self.job)

    def test_duplicate_raw_record_is_rejected(self):
        self.review["raw_files"].append(dict(self.rows[0]))
        self.save_review()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            archive.validate_job(self.root, self.job)

    def test_outside_root_is_rejected(self):
        child = self.root / "other-root"
        child.mkdir()
        with self.assertRaisesRegex(ValueError, "inside new-results"):
            archive.validate_job(child, self.job)

    def test_traversal_and_wrong_extension_are_rejected(self):
        for value in ("../old/raw.f32", "oracle/../case.f32", "oracle/model.gguf", "/tmp/raw.f32"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    archive.relative_raw_path(self.job, value)

    def test_raw_symlink_is_rejected(self):
        raw = self.job / self.rows[0]["path"]
        outside = self.root / "outside.f32"
        outside.write_bytes(raw.read_bytes())
        raw.unlink()
        raw.symlink_to(outside)
        with self.assertRaises(ValueError):
            archive.validate_job(self.root, self.job)
        self.assertTrue(outside.exists())

    def test_decompression_mismatch_keeps_raw(self):
        with patch.object(archive, "decompress_digest", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "decompressed SHA"):
                self.run_archive()
        self.assertTrue(all((self.job / row["path"]).exists() for row in self.rows))
        self.assertFalse((self.job / "archive.json").exists())

    def test_existing_compressed_file_is_not_overwritten(self):
        path = self.job / (self.rows[0]["path"] + ".zst")
        path.write_bytes(b"old archive")
        with self.assertRaisesRegex(ValueError, "archive already exists"):
            self.run_archive()
        self.assertEqual(path.read_bytes(), b"old archive")
        self.assertTrue(all((self.job / row["path"]).exists() for row in self.rows))

    def test_receipt_exists_before_any_unlink(self):
        original = Path.unlink
        observed = []

        def check_unlink(path, *args, **kwargs):
            if path.suffix == ".f32":
                self.assertTrue((self.job / "archive.json").is_file())
                observed.append(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "unlink", check_unlink):
            self.run_archive()
        self.assertEqual(len(observed), 2)


if __name__ == "__main__":
    unittest.main()
