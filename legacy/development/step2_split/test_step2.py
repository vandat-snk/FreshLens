"""Regression tests for data separation, image identity and read-only inputs."""

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import STEP2_SPLIT as step


def record(name, fruit="apple", status="fresh", split="train", group=None, digest=None):
    path = f"raw/{fruit}/{status}/{name}"
    return dict(path=path, fruit=fruit, status=status, split=split, source="local",
                group_id=group or Path(name).stem,
                sha256=digest or step.sha_bytes(path.encode()))


def fixture_facts(rows, infos):
    # Synthetic identities for group-logic tests only. End-to-end tests below
    # use real file bytes and real decoded-pixel/perceptual hashes.
    return {i: {"pixel_sha256": step.sha_bytes((row["path"] + "pixels").encode()),
                "phash63": i, "width": 32, "height": 32}
            for i, (row, info) in enumerate(zip(rows, infos))
            if info["kind"] == "candidate_original"}


def prepare(rows):
    infos = [step.filename_info(row) for row in rows]
    facts = fixture_facts(rows, infos)
    return infos, facts, step.prepare_split(rows, infos, facts)


class GroupTests(unittest.TestCase):
    def test_old_leak_is_found_and_only_unaugmented_name_is_kept(self):
        base = "Screen Shot 2018-06-08 at 4.59.36 PM.png"
        rows = [record(base), record("rotated_by_15_" + base, split="val"),
                record("vertical_flip_" + base, split="test")]
        infos, facts, out = prepare(rows)
        audit = step.audit_filenames(rows, infos)
        self.assertEqual(audit["screen_families_across_old_splits"], 1)
        self.assertEqual(audit["rows_in_these_families"], 3)
        self.assertEqual([row["path"] for row in out["manifest"]], [rows[0]["path"]])
        self.assertEqual(len(set(out["group_ids"].values())), 1)

    def test_no_original_is_invented_for_augmented_only_family(self):
        rows = [record("rotated_by_30_Screen Shot missing original.png")]
        infos, facts, out = prepare(rows)
        self.assertEqual(out["manifest"], [])
        self.assertEqual(step.audit_filenames(rows, infos)["screen_families_without_named_original"], 1)

    def test_unknown_parent_and_unrecognized_files_are_excluded(self):
        rows = [record("Banana__Healthy_augmented_14.jpg", "banana"), record("unmapped_123.jpg")]
        infos, facts, out = prepare(rows)
        self.assertEqual(out["manifest"], [])
        self.assertEqual(set(out["reasons"].values()), {
            "offline_augmentation_parent_unknown", "unrecognized_filename_needs_source_mapping"})

    def test_case_extension_and_transitive_declared_group_links_stay_together(self):
        rows = [record("FreshApple (1).png", group="session_a"),
                record("freshApple (1).jpg", group="session_b"),
                record("freshApple (2).png", group="session_b")]
        infos, facts, out = prepare(rows)
        self.assertEqual(len({row["group_id"] for row in out["manifest"]}), 1)
        self.assertEqual(len({row["split"] for row in out["manifest"]}), 1)

    def test_byte_duplicates_with_conflicting_labels_stop_the_run(self):
        rows = [record("freshApple (1).png"), record("rottenApple (2).png", status="rotten")]
        rows[1]["sha256"] = rows[0]["sha256"]
        with self.assertRaisesRegex(step.DataError, "conflicting labels"):
            prepare(rows)

    def test_pixel_duplicates_with_conflicting_labels_stop_the_run(self):
        rows = [record("freshApple (1).png"), record("rottenApple (2).png", status="rotten")]
        infos = [step.filename_info(row) for row in rows]
        facts = fixture_facts(rows, infos)
        facts[1]["pixel_sha256"] = facts[0]["pixel_sha256"]
        with self.assertRaisesRegex(step.DataError, "conflicting labels"):
            step.prepare_split(rows, infos, facts)

    def test_exact_pixel_duplicates_keep_one_sample_and_keep_group_links(self):
        rows = [record(f"freshApple ({i}).png") for i in range(4)]
        infos = [step.filename_info(row) for row in rows]
        facts = fixture_facts(rows, infos)
        facts[1]["pixel_sha256"] = facts[0]["pixel_sha256"]
        out = step.prepare_split(rows, infos, facts)
        self.assertEqual(len(out["manifest"]), 3)
        self.assertEqual(out["group_ids"][0], out["group_ids"][1])
        self.assertEqual(out["duplicate_of"][1], rows[0]["path"])

    def test_stratification_has_all_eight_labels_in_each_split(self):
        rows = [record(f"{status}{fruit.title()} ({n}).png", fruit, status)
                for fruit in step.FRUITS for status in step.STATUSES for n in range(10)]
        infos, facts, out = prepare(rows)
        for name in step.SPLITS:
            self.assertEqual(len({step.label(row) for row in out["manifest"] if row["split"] == name}), 8)
        self.assertEqual(out["overlaps"], {"group": 0, "filename": 0, "bytes": 0, "pixels": 0})

    def test_order_and_old_split_changes_do_not_change_new_allocation(self):
        rows = [record(f"freshApple ({n}).png") for n in range(20)]
        first = prepare(rows)[2]["manifest"]
        random.Random(11).shuffle(rows)
        rows = [dict(row, split="test") for row in rows]
        second = prepare(rows)[2]["manifest"]
        self.assertEqual(first, second)

    def test_missing_pixel_checks_cannot_produce_a_manifest(self):
        rows = [record("freshApple (1).png")]
        infos = [step.filename_info(rows[0])]
        with self.assertRaisesRegex(step.DataError, "Every candidate"):
            step.prepare_split(rows, infos, {})

    def test_filename_label_disagreement_stops(self):
        with self.assertRaisesRegex(step.DataError, "disagree"):
            step.filename_info(record("rottenApple (1).png"))

    def test_incomplete_perceptual_audit_blocks_readiness(self):
        rows = [record(f"{status}{fruit.title()} ({n}).png", fruit, status)
                for fruit in step.FRUITS for status in step.STATUSES for n in range(3)]
        infos, facts, out = prepare(rows)
        report = step.build_report(rows, infos, facts, out, [], False, 42, "input")
        self.assertTrue(report["review_required"])
        self.assertFalse(report["ready_for_pilot_training"])


class PerceptualTests(unittest.TestCase):
    def test_tree_queries_match_exhaustive_search(self):
        rng = random.Random(23)
        values = set(rng.getrandbits(63) for _ in range(100))
        values.update(range(20))
        tree = step.HashTree()
        for value in values:
            tree.add(value)
        for query in [0, 3, 511, *list(values)[:10]]:
            expected = {(value, (value ^ query).bit_count()) for value in values if (value ^ query).bit_count() <= 4}
            self.assertEqual(set(tree.query(query, 4)), expected)

    def test_cross_group_candidates_are_not_automatically_merged(self):
        rows = [record(f"freshApple ({n}).png") for n in range(10)]
        infos, facts, out = prepare(rows)
        for value in facts.values():
            value["phash63"] = 42
        pairs, complete = step.find_candidates(rows, facts, out)
        self.assertTrue(complete)
        self.assertEqual(len(pairs), 45)
        self.assertTrue(any(pair["cross_split"] for pair in pairs))
        self.assertEqual(len(set(out["group_ids"].values())), 10)
        report = step.build_report(rows, infos, facts, out, pairs, complete, 42, "input")
        self.assertTrue(report["review_required"])

    def test_candidate_limit_reports_incomplete_audit(self):
        rows = [record(f"freshApple ({n}).png") for n in range(10)]
        infos, facts, out = prepare(rows)
        for value in facts.values():
            value["phash63"] = 0
        pairs, complete = step.find_candidates(rows, facts, out, max_pairs=4)
        self.assertFalse(complete)
        self.assertEqual(len(pairs), 4)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_manifest(self, rows, name="input.csv"):
        path = self.root / name
        step.write_csv(path, rows, step.COLUMNS)
        return path

    def save_image(self, name, fruit="apple", status="fresh", seed=3, image=None, **save_options):
        row = record(name, fruit, status)
        path = self.root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if image is None:
            pixels = np.random.default_rng(seed).integers(0, 256, (80, 80, 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
        image.save(path, **save_options)
        row["sha256"] = step.sha_bytes(path.read_bytes())
        return row

    def test_windows_relative_paths_are_supported(self):
        row = record("freshApple (1).png")
        row["path"] = row["path"].replace("/", "\\")
        loaded = step.read_manifest(self.write_manifest([row]))
        self.assertEqual(loaded[0]["path"], "raw/apple/fresh/freshApple (1).png")

    def test_path_traversal_and_absolute_paths_are_rejected(self):
        for unsafe in ("../private.png", "E:\\private.png", "/private.png", "raw/../../secret.png"):
            with self.subTest(unsafe=unsafe):
                row = dict(record("freshApple (1).png"), path=unsafe)
                with self.assertRaisesRegex(step.DataError, "Unsafe"):
                    step.read_manifest(self.write_manifest([row]))

    def test_case_insensitive_duplicate_paths_are_rejected(self):
        rows = [record("freshApple (1).png"), record("FreshAPPLE (1).PNG")]
        with self.assertRaisesRegex(step.DataError, "Repeated Windows path"):
            step.read_manifest(self.write_manifest(rows))

    def test_sha_mismatch_writes_no_output(self):
        row = self.save_image("freshApple (1).png")
        manifest = self.write_manifest([row])
        Image.new("RGB", (5, 5), "red").save(self.root / row["path"])
        output = self.root / "result"
        with self.assertRaisesRegex(step.DataError, "SHA-256 mismatch"):
            step.run(manifest, self.root, output)
        self.assertFalse(output.exists())

    def test_corrupt_image_writes_no_output(self):
        row = self.save_image("freshApple (1).png")
        (self.root / row["path"]).write_bytes(b"not an image")
        row["sha256"] = step.sha_bytes(b"not an image")
        with self.assertRaisesRegex(step.DataError, "Cannot decode"):
            step.run(self.write_manifest([row]), self.root, self.root / "result")
        self.assertFalse((self.root / "result").exists())

    def test_identical_pixels_with_different_file_bytes_are_deduplicated(self):
        row_a = self.save_image("freshApple (1).png", compress_level=0)
        row_b = self.save_image("freshApple (2).png", compress_level=9)
        self.assertNotEqual(row_a["sha256"], row_b["sha256"])
        rows = [row_a, row_b]
        infos = [step.filename_info(row) for row in rows]
        facts = step.inspect_images(rows, infos, self.root)
        self.assertEqual(facts[0]["pixel_sha256"], facts[1]["pixel_sha256"])
        self.assertEqual(len(step.prepare_split(rows, infos, facts)["manifest"]), 1)

    def test_phash_flags_reencoded_photo_without_claiming_pixel_identity(self):
        pixels = np.random.default_rng(12).integers(0, 256, (24, 24, 3), dtype=np.uint8)
        image = Image.fromarray(pixels).resize((200, 200), Image.Resampling.BICUBIC)
        rows = [self.save_image("freshApple (1).png", image=image),
                self.save_image("freshApple (2).jpg", image=image, quality=90)]
        infos = [step.filename_info(row) for row in rows]
        facts = step.inspect_images(rows, infos, self.root)
        self.assertNotEqual(facts[0]["pixel_sha256"], facts[1]["pixel_sha256"])
        out = step.prepare_split(rows, infos, facts)
        pairs, complete = step.find_candidates(rows, facts, out)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["cross_split"])

    def test_existing_output_is_preserved(self):
        output = self.root / "result"
        output.mkdir()
        guard = output / "keep.txt"
        guard.write_text("do not overwrite")
        with self.assertRaisesRegex(step.DataError, "already exists"):
            step.run(self.root / "unused.csv", self.root, output)
        self.assertEqual(guard.read_text(), "do not overwrite")

    def test_end_to_end_input_bytes_preserved_and_complete_outputs(self):
        rows = []
        for fruit in step.FRUITS:
            for status in step.STATUSES:
                for number in range(3):
                    rows.append(self.save_image(f"{status}{fruit.title()} ({number}).png", fruit, status, seed=len(rows)))
        # This filename-marked augmentation is not decoded or added to the new split.
        rows.append(self.save_image("Banana__Healthy_augmented_1.jpg", "banana"))
        manifest = self.write_manifest(rows)
        original_manifest = manifest.read_bytes()
        output = self.root / "result"
        report = step.run(manifest, self.root, output)
        self.assertEqual(report["output_records"], 24)
        self.assertEqual(report["excluded_records"], 1)
        self.assertEqual(report["local_image_files_checked"], 24)
        self.assertEqual(report["missing_label_split_buckets"], [])
        self.assertTrue(report["ready_for_pilot_training"])
        self.assertFalse(report["evaluation_independence_confirmed"])
        self.assertEqual(manifest.read_bytes(), original_manifest)
        for row in rows:
            self.assertEqual(step.sha_bytes((self.root / row["path"]).read_bytes()), row["sha256"])
        for name in ("manifest.csv", "lineage.csv", "excluded.csv", "near_duplicates.csv", "image_fingerprints.csv", "split_report.json"):
            self.assertTrue((output / name).is_file(), name)
        saved_report = json.loads((output / "split_report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_report["output_manifest_sha256"], step.sha_bytes((output / "manifest.csv").read_bytes()))
        shuffled = list(reversed(rows))
        step.run(self.write_manifest(shuffled, "shuffled.csv"), self.root, self.root / "result2")
        self.assertEqual((output / "manifest.csv").read_bytes(), (self.root / "result2/manifest.csv").read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
