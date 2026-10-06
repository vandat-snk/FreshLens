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

    def test_prefix_mapping_reads_original_and_preserves_manifest_path(self):
        from freshlens_ai.data.dataset_audit import mapped_image_path
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "apple/fresh/a.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (120, 100), "white").save(path)
            row = record("raw/apple/fresh/a.png", split="train", digest=file_sha256(path))
            original_row = dict(row)
            self.assertEqual(inspect_image(root, row)["decode_status"], "MISSING_FILE")
            report, quality = audit_dataset([row], root=root, strip_prefix="raw", fingerprints=True)
            self.assertTrue(report["all_image_files_verified"])
            self.assertEqual(quality[0]["resolved_relative_path"], "apple/fresh/a.png")
            self.assertEqual(quality[0]["path"], row["path"])
            self.assertEqual(row, original_row)
            self.assertEqual(mapped_image_path("raw\\apple\\fresh\\a.png", "raw"), "apple/fresh/a.png")

    def test_sha_catalog_alias_resolves_without_changing_baseline(self):
        from freshlens_ai.data.dataset_audit import load_path_mapping, SUPPORTED_COLUMNS
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "apple/fresh"
            folder.mkdir(parents=True)
            Image.new("RGB", (120, 100), "white").save(folder / "actual.png")
            row = record("raw/apple/fresh/missing.png", split="train", digest=file_sha256(folder / "actual.png"))
            entry = dict(raw_path="data/raw/cnn_v3/apple/fresh/actual.png", manifest_path=row["path"],
                         **{k:row[k] for k in SUPPORTED_COLUMNS[1:]})
            catalog = root / "existing_catalog.csv"
            columns = ("raw_path", "manifest_path", *SUPPORTED_COLUMNS[1:])
            write_csv(catalog, [entry], columns)
            mapping = load_path_mapping(catalog, [row], "raw")
            report, quality = audit_dataset([row], root=root, strip_prefix="raw", path_overrides=mapping)
            self.assertTrue(report["all_image_files_verified"])
            self.assertEqual(quality[0]["resolved_relative_path"], "apple/fresh/actual.png")
            self.assertEqual(row["path"], "raw/apple/fresh/missing.png")
            entry["sha256"] = "f" * 64
            write_csv(catalog, [entry], columns)
            with self.assertRaisesRegex(ValueError, "metadata disagrees"):
                load_path_mapping(catalog, [row], "raw")

    def test_prefix_matches_components_and_rejects_unsafe_mapping(self):
        from freshlens_ai.data.dataset_audit import mapped_image_path
        for path, prefix in (("raw2/a.png", "raw"), ("raw", "raw"), ("raw/a.png", "../raw"),
                             ("raw/a.png", "/raw"), ("../raw/a.png", "raw")):
            with self.subTest(path=path, prefix=prefix), self.assertRaises(ValueError):
                mapped_image_path(path, prefix)
        self.assertEqual(mapped_image_path("a.png"), "a.png")

    def test_mapped_sha_mismatch_and_decode_error_remain_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (10, 20), "white").save(root / "a.png")
            report, _ = audit_dataset([record("raw/a.png", split="train")], root=root, strip_prefix="raw")
            self.assertEqual(report["quality"]["sha256_mismatches"], 1)
            self.assertFalse(report["all_image_files_verified"])
            (root / "broken.png").write_bytes(b"broken")
            row = record("raw/broken.png", split="train", digest=file_sha256(root / "broken.png"))
            report, _ = audit_dataset([row], root=root, strip_prefix="raw")
            self.assertEqual(report["quality"]["decode_errors"], 1)
            self.assertEqual(report["quality"]["missing_files"], 0)

    def test_quality_summary_uses_actual_flags_and_measurements(self):
        from freshlens_ai.data.dataset_audit import quality_summary
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (40, 30), "black").save(root / "a.png")
            row = record("a.png", split="train", digest=file_sha256(root / "a.png"))
            report, quality = audit_dataset([row], root=root)
            summary, flat = quality_summary([row], quality)
            a = summary["class_summaries"]["apple::fresh"]
            self.assertEqual(a["low_quality"], 1)
            self.assertEqual(a["low_resolution"], 1)
            self.assertEqual(a["measurements"]["brightness"]["mean"], 0)
            self.assertEqual(a["resolution_by_min_side"]["under_96"], 1)
            self.assertEqual(a["formats"], {"PNG": 1})
            self.assertEqual(sum(x["images"] for x in flat), 1)

    def test_provenance_marker_without_available_parent_stays_unknown(self):
        from freshlens_ai.data.dataset_provenance import classify_evidence
        self.assertEqual(classify_evidence("OK", known_transform=True)[0], "unknown")
        self.assertEqual(classify_evidence("OK", named_parent=True)[0], "unknown")
        self.assertEqual(classify_evidence("OK", known_transform=True, named_parent=True)[0], "probable_augmentation")

    def test_provenance_exact_pixels_corrupt_and_explicit_new_evidence(self):
        from freshlens_ai.data.dataset_provenance import classify_evidence
        self.assertEqual(classify_evidence("OK", byte_reference="a.png")[0], "exact_duplicate")
        self.assertEqual(classify_evidence("OK", pixel_reference="a.png")[0], "exact_duplicate")
        self.assertEqual(classify_evidence("DECODE_ERROR", byte_reference="a.png")[0], "corrupt")
        self.assertEqual(classify_evidence("OK", verified_new_provenance=True)[0], "probable_new_image")

    def test_provenance_catalog_paths_cannot_escape_raw_root(self):
        from freshlens_ai.data.dataset_provenance import raw_relative_path
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(raw_relative_path(temporary, "apple/fresh/a.png"), "apple/fresh/a.png")
            for path in ("../a.png", "C:/a.png", "data/raw/another/a.png"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    raw_relative_path(temporary, path)

    def test_provenance_end_to_end_never_admits_unverified_samples(self):
        from unittest.mock import patch
        from freshlens_ai.data.dataset_audit import QUALITY_COLUMNS, SUPPORTED_COLUMNS
        from freshlens_ai.data.dataset_provenance import analyze_provenance
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, originals = root / "raw", root / "originals"
            raw_folder, original_folder = raw / "apple/fresh", originals / "apple/fresh"
            raw_folder.mkdir(parents=True); original_folder.mkdir(parents=True)
            name = "Screen Shot 2020-01-01 at 12.00.00 AM.png"
            image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (100, 120, 3), dtype=np.uint8))
            image.save(raw_folder / name); image.save(original_folder / name)
            baseline = record("raw/apple/fresh/" + name, split="train", digest=file_sha256(original_folder / name))
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(raw_folder / ("horizontal_flip_" + name))
            Image.new("RGB", (101, 121), "green").save(raw_folder / "mystery.png")
            (raw_folder / "broken.jpg").write_bytes(b"broken")
            q = inspect_image(originals, baseline, strip_prefix="raw", fingerprints=True)
            write_csv(root / "quality.csv", [q], QUALITY_COLUMNS)
            catalog_columns = ("raw_path", "manifest_path", *SUPPORTED_COLUMNS[1:])
            write_csv(root / "matched.csv", [dict(raw_path="apple/fresh/" + name, manifest_path=baseline["path"], **{k:baseline[k] for k in SUPPORTED_COLUMNS[1:]})], catalog_columns)
            write_csv(root / "duplicates.csv", [], catalog_columns)
            candidates = [dict(raw_path="apple/fresh/" + filename, sha256=file_sha256(raw_folder / filename), extension=Path(filename).suffix)
                          for filename in ("horizontal_flip_" + name, "mystery.png", "broken.jpg")]
            write_csv(root / "unmatched.csv", candidates, ("raw_path", "sha256", "extension"))
            with patch("freshlens_ai.data.dataset_provenance.baseline_snapshot", return_value=([baseline], {}, {})):
                report, rows = analyze_provenance(root, originals, raw, root / "quality.csv", root / "matched.csv",
                                                  root / "duplicates.csv", root / "unmatched.csv", progress=False)
            self.assertEqual(report["classification_counts"]["probable_augmentation"], 1)
            self.assertEqual(report["classification_counts"]["unknown"], 1)
            self.assertEqual(report["classification_counts"]["corrupt"], 1)
            self.assertTrue(report["partition_verified"])
            self.assertEqual(report["admitted_new_samples"], 0)
            self.assertTrue(all(not r["eligible_for_candidate"] for r in rows))

    def test_candidate_loader_maps_paths_in_memory_and_checks_readiness(self):
        from freshlens_ai.constants import CLASSES
        from freshlens_ai.data.dataset_v2 import load_candidate_dataset
        baseline = Path(__file__).resolve().parents[1] / "data/cnn_dataset_v3"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            build_dataset(baseline, output, strip_prefix="raw")
            with self.assertRaisesRegex(ValueError, "not ready"):
                load_candidate_dataset(output)
            rows, identity = load_candidate_dataset(output, require_ready=False)
            self.assertEqual(rows[0]["path"], rows[0]["manifest_path"].removeprefix("raw/"))
            self.assertEqual(rows[0]["target"], CLASSES.index(rows[0]["fruit"] + "::" + rows[0]["status"]))
            self.assertEqual(identity["image_path_mapping"]["strip_prefix"], "raw")
            with (output / "manifest.csv").open(encoding="utf-8-sig") as handle:
                self.assertTrue(next(csv.DictReader(handle))["path"].startswith("raw/"))

    def test_required_image_build_refuses_unverified_candidate_without_output(self):
        baseline = Path(__file__).resolve().parents[1] / "data/cnn_dataset_v3"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            with self.assertRaisesRegex(ValueError, "Verified image build"):
                build_dataset(baseline, output, strip_prefix="raw", require_images=True)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
