"""TV1 regression tests. Generated pixels are test fixtures, never dataset data."""

import csv
import hashlib
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from freshlens_ai.data.dataset_audit import audit_dataset, inspect_image, relative_path
from freshlens_ai.data.dataset_v2 import (
    INVENTORY_COLUMNS, assign_new_splits, baseline_snapshot, build_dataset,
    load_inventory, verify_candidate_lock,
)
from freshlens_ai.data.image_io import rgb_from_bytes
from freshlens_ai.utils.file_io import file_sha256, write_csv


def record(name, group=None, split="", digest=None, fruit="apple", status="fresh"):
    return dict(path=name, fruit=fruit, status=status, source="test_fixture",
                group_id=group or name, split=split,
                sha256=digest or hashlib.sha256(name.encode()).hexdigest())


class DatasetV2Tests(unittest.TestCase):
    def test_metadata_only_does_not_claim_zero_image_errors(self):
        report, quality = audit_dataset([record("a.png", split="train")])
        self.assertEqual(report["raw_data_status"], "RAW DATA NOT AVAILABLE")
        self.assertIsNone(report["quality"]["decode_errors"])
        self.assertIsNone(report["quality"]["blur"])
        self.assertEqual(quality[0]["quality_status"], "NOT_AVAILABLE")
        self.assertFalse(report["all_image_files_verified"])

    def test_duplicate_and_leakage_detected_without_files(self):
        a = record("a.png", "physical", "train")
        b = record("b.png", "physical", "test", a["sha256"])
        report, _ = audit_dataset([a, b])
        self.assertEqual(report["declared_identity"]["cross_split_sha256"]["keys"], 1)
        self.assertEqual(report["declared_identity"]["cross_split_group_id"]["keys"], 1)
        self.assertFalse(report["metadata_valid"])

    def test_case_insensitive_path_duplicate(self):
        report, _ = audit_dataset([record("A.png", split="train"), record("a.png", split="train")])
        self.assertEqual(report["declared_identity"]["duplicate_path"]["keys"], 1)

    def test_transitive_hash_group_links_preserve_baseline(self):
        a = record("baseline.png", "old", "test")
        b = record("view1.png", "new", digest=a["sha256"])
        c = record("view2.png", "new")
        supported, _ = assign_new_splits([a], [b, c], [])
        self.assertEqual(supported[0], a)
        self.assertEqual({r["split"] for r in supported}, {"test"})

    def test_conflicting_frozen_splits_rejected(self):
        a = record("a.png", "one", "train")
        b = record("b.png", "two", "test")
        c = record("c.png", "one", digest=b["sha256"])
        with self.assertRaisesRegex(ValueError, "frozen splits"):
            assign_new_splits([a, b], [c], [])

    def test_order_independent_split_and_no_duplicate_deletion(self):
        rows = [record(f"{i}.png") for i in range(15)]
        rows.append(record("copy.png", digest=rows[0]["sha256"]))
        first, _ = assign_new_splits([], rows, [])
        random.Random(7).shuffle(rows)
        second, _ = assign_new_splits([], rows, [])
        self.assertEqual({r["path"]:r["split"] for r in first}, {r["path"]:r["split"] for r in second})
        self.assertEqual(len(first), 16)
        report, _ = audit_dataset(first)
        self.assertEqual(report["declared_identity"]["cross_split_sha256"]["keys"], 0)

    def test_unknown_is_separate_and_global_group_assignment(self):
        a = record("supported.png", "scene", "val")
        b = dict(path="unknown.png", category="multiple", subcategory="fruit_and_object",
                 source="test_fixture", group_id="scene", sha256=hashlib.sha256(b"unknown").hexdigest(), split="")
        supported, other = assign_new_splits([a], [], [b])
        self.assertEqual(supported, [a])
        self.assertEqual(other[0]["split"], "val")
        self.assertNotIn("fruit", other[0])

    def test_same_bytes_conflicting_label_rejected(self):
        a = record("a.png", split="train")
        b = record("b.png", digest=a["sha256"], status="rotten")
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            assign_new_splits([a], [b], [])

    def test_unsafe_paths_rejected(self):
        for path in ("../a.jpg", "C:/a.jpg", "/a.jpg", "a.jpg:stream", "a\x00.jpg"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative_path(path)

    def test_real_byte_quality_and_decode_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (40, 30), (0, 0, 0)).save(root / "dark.png")
            row = record("dark.png", split="train", digest=file_sha256(root / "dark.png"))
            q = inspect_image(root, row)
            self.assertEqual(q["sha256_status"], "MATCH")
            self.assertEqual(q["quality_status"], "LOW_QUALITY")
            self.assertEqual((q["width"], q["height"]), (40, 30))
            self.assertTrue(all(q[k] for k in ("low_resolution", "low_brightness", "blur")))
            (root / "broken.png").write_bytes(b"not an image")
            broken = record("broken.png", split="train", digest=file_sha256(root / "broken.png"))
            self.assertEqual(inspect_image(root, broken)["decode_status"], "DECODE_ERROR")
            self.assertEqual(inspect_image(root, record("missing.png"))["decode_status"], "MISSING_FILE")

    def test_hash_mismatch_is_not_hidden_by_successful_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (110, 110), "white").save(root / "a.png")
            report, _ = audit_dataset([record("a.png", split="train")], root=root)
            self.assertEqual(report["quality"]["sha256_mismatches"], 1)
            self.assertFalse(report["all_image_files_verified"])

    def test_exif_alpha_grayscale_use_production_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgba = root / "alpha.png"
            Image.new("RGBA", (4, 3), (255, 0, 0, 0)).save(rgba)
            image = rgb_from_bytes(rgba.read_bytes())
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            exif = Image.Exif(); exif[274] = 6
            Image.new("L", (40, 20), 100).save(root / "gray.jpg", exif=exif)
            q = inspect_image(root, record("gray.jpg", digest=file_sha256(root / "gray.jpg")))
            self.assertEqual((q["width"], q["height"]), (20, 40))
            self.assertEqual(q["original_mode"], "L")
            self.assertEqual(q["exif_orientation"], 6)

    def test_camera_requires_evidence_and_single_group_per_specimen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a.png", "b.png"):
                Image.new("RGB", (4, 3), "white").save(root / name)
            entries = []
            for name, group in (("a.png", "group1"), ("b.png", "group2")):
                row = dict.fromkeys(INVENTORY_COLUMNS, "")
                row.update(path=name, data_kind="supported", fruit="apple", status="fresh",
                           source="camera", group_id=group, capture_device="phone",
                           specimen_id="physical_1", specimen_evidence="capture_log_1", session_id="session_1",
                           lighting="yellow", background="desk", viewpoint="side", distance="far")
                entries.append(row)
            write_csv(root / "inventory.csv", entries, INVENTORY_COLUMNS)
            with self.assertRaisesRegex(ValueError, "different group_id"):
                load_inventory(root / "inventory.csv", root)
            entries[0]["specimen_evidence"] = ""
            write_csv(root / "inventory.csv", entries[:1], INVENTORY_COLUMNS)
            with self.assertRaisesRegex(ValueError, "specimen evidence"):
                load_inventory(root / "inventory.csv", root)

    def test_baseline_snapshot_build_is_read_only_and_locked(self):
        baseline = Path(__file__).resolve().parents[1] / "data/cnn_dataset_v3"
        rows, hashes, _ = baseline_snapshot(baseline)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            report = build_dataset(baseline, output)
            self.assertEqual(report["supported_records"], len(rows))
            self.assertFalse(report["ready_for_pilot_training"])
            self.assertFalse(report["production_loader_compatible"])
            verify_candidate_lock(output)
            with self.assertRaisesRegex(ValueError, "NEW output"):
                build_dataset(baseline, output)
            with (output / "manifest.csv").open("ab") as handle:
                handle.write(b"tampering")
            with self.assertRaisesRegex(ValueError, "changed after build"):
                verify_candidate_lock(output)
        self.assertEqual(baseline_snapshot(baseline)[1], hashes)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_dataset([], thresholds={"min_side":96, "min_brightness":float("nan"), "min_blur_score":50})

    def test_audit_detects_specimen_leakage_even_when_group_ids_differ(self):
        a = dict(record("a.png", split="train"), specimen_id="physical1", specimen_evidence="log1")
        b = dict(record("b.png", split="val"), specimen_id="physical1", specimen_evidence="log1")
        report, _ = audit_dataset([a, b])
        self.assertFalse(report["metadata_valid"])
        self.assertIn("physical1", report["camera"]["grouping_violations"]["specimens_across_splits"])


if __name__ == "__main__":
    unittest.main()
