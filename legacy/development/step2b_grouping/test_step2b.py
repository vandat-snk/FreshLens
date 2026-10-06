"""Regression tests: preserve labels, coassign groups, verify provenance."""

import contextlib
import csv
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import STEP2B_GROUP as step
import step2_core as core


def row(number, fruit="apple", status="fresh", split="train", group=None, extension="png"):
    path = f"raw/{fruit}/{status}/{status}{fruit.title()} ({number}).{extension}"
    return {"path": path, "fruit": fruit, "status": status, "source": "test_fixture",
            "group_id": group or f"old_{fruit}_{status}_{number}",
            "split": split, "sha256": core.sha_bytes(path.encode())}


def pair(a, b, number=1, distance=0):
    return {"pair_id": str(number), "path_a": a["path"], "path_b": b["path"],
            "label_a": core.label(a), "label_b": core.label(b),
            "group_a": a["group_id"], "group_b": b["group_id"],
            "split_a": a["split"], "split_b": b["split"],
            "hamming_distance": str(distance),
            "cross_split": str(a["split"] != b["split"]),
            "different_labels": str(core.label(a) != core.label(b)), "decision": "unreviewed"}


class GroupingTests(unittest.TestCase):
    def test_transitive_links_preserve_existing_group_members(self):
        rows = [row(1), row(2, split="val"), row(3, split="test"), row(4, group="old_apple_fresh_3", split="test")]
        pairs = [pair(rows[0], rows[1]), pair(rows[1], rows[2], 2)]
        output, _ = step.coassign(rows, pairs)
        self.assertEqual(len({r["group_id"] for r in output}), 1)
        self.assertEqual(len({r["split"] for r in output}), 1)
        self.assertEqual(len(output), 4)

    def test_similar_but_different_labels_are_retained_in_one_training_group(self):
        rows = [row(243, "tomato", "fresh", "val"), row(380, "tomato", "rotten", "test"), row(568, "tomato", "rotten")]
        pairs = [pair(rows[0], rows[1], 1, 2), pair(rows[0], rows[2], 2, 4)]
        output, mixed = step.coassign(rows, pairs)
        self.assertEqual(len(output), 3)
        self.assertEqual(len(mixed), 1)
        self.assertEqual({r["split"] for r in output}, {"train"})
        self.assertEqual({r["path"]: core.label(r) for r in output}, {r["path"]: core.label(r) for r in rows})

    def test_each_label_remains_in_all_three_splits_when_groups_suffice(self):
        rows = [row(n, fruit, status) for fruit in core.FRUITS for status in core.STATUSES for n in range(10)]
        pairs = [pair(rows[0], rows[1])]
        output, _ = step.coassign(rows, pairs)
        self.assertEqual(len(output), len(rows))
        for name in core.SPLITS:
            self.assertEqual(len({core.label(r) for r in output if r["split"] == name}), 8)
        before = {r["path"]: {k: v for k, v in r.items() if k not in ("group_id", "split")} for r in rows}
        after = {r["path"]: {k: v for k, v in r.items() if k not in ("group_id", "split")} for r in output}
        self.assertEqual(before, after)

    def test_row_and_pair_order_do_not_affect_split(self):
        rows = [row(n) for n in range(15)]
        pairs = [pair(rows[0], rows[1]), pair(rows[1], rows[2], 2)]
        expected = step.coassign(rows, pairs)[0]
        random.Random(99).shuffle(rows)
        self.assertEqual(step.coassign(rows, list(reversed(pairs)))[0], expected)

    def test_pair_metadata_mismatch_rejected(self):
        rows = [row(1), row(2, split="test")]
        bad = pair(*rows)
        bad["group_b"] = "another_group"
        with self.assertRaisesRegex(core.DataError, "metadata"):
            step.validate_pairs(rows, [bad])

    def test_repeated_pair_rejected_even_with_different_pair_id(self):
        rows = [row(1), row(2)]
        with self.assertRaisesRegex(core.DataError, "Repeated pair"):
            step.validate_pairs(rows, [pair(*rows), pair(*rows, number=2)])

    def test_false_cross_split_flag_rejected(self):
        rows = [row(1), row(2, split="test")]
        bad = dict(pair(*rows), cross_split="False")
        with self.assertRaisesRegex(core.DataError, "cross_split"):
            step.validate_pairs(rows, [bad])

    def test_unknown_image_in_pair_rejected(self):
        rows = [row(1), row(2)]
        bad = pair(rows[0], row(3))
        with self.assertRaisesRegex(core.DataError, "outside the manifest"):
            step.validate_pairs(rows, [bad])

    def test_unreviewed_pairs_never_marked_as_visually_inspected(self):
        rows = [row(1), row(2), row(3)]
        pairs = [pair(rows[0], rows[1]), pair(rows[1], rows[2], 2)]
        notes = {"pairs": [{"path_a": rows[0]["path"], "path_b": rows[1]["path"],
                            "sha256_a": rows[0]["sha256"], "sha256_b": rows[1]["sha256"],
                            "observation": "test observation"}]}
        self.assertEqual(step.visual_observations(rows, pairs, notes), {"1": "test observation"})
        notes["pairs"][0]["sha256_b"] = "old_image_version"
        self.assertEqual(step.visual_observations(rows, pairs, notes), {})


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.root.mkdir()

    def quiet(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def fixture(self):
        rows = []
        for fruit in core.FRUITS:
            for status in core.STATUSES:
                for n in range(4):
                    pixels = np.random.default_rng(100 + len(rows)).integers(0, 256, (24, 24, 3), dtype=np.uint8)
                    image = Image.fromarray(pixels).resize((160, 160), Image.Resampling.BICUBIC)
                    r = row(n, fruit, status)
                    p = self.root / r["path"]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    image.save(p)
                    r["sha256"] = core.sha_bytes(p.read_bytes())
                    rows.append(r)
        # One re-encoded image has identical visual source but different pixels.
        with Image.open(self.root / rows[0]["path"]) as original:
            old_path = self.root / rows[1]["path"]
            rows[1] = row(1, extension="jpg")
            new_path = self.root / rows[1]["path"]
            original.save(new_path, quality=90)
            old_path.unlink()
            rows[1]["sha256"] = core.sha_bytes(new_path.read_bytes())
        raw_manifest = self.base / "input.csv"
        core.write_csv(raw_manifest, rows, core.COLUMNS)
        directory = self.base / "v2"
        self.quiet(core.run, raw_manifest, self.root, directory)
        return directory

    def test_full_pipeline_keeps_inputs_and_labels_and_publishes_lock(self):
        source = self.fixture()
        input_bytes = {name: (source / name).read_bytes() for name in step.INPUT_FILES}
        before = core.read_manifest(source / "manifest.csv")
        output = self.base / "v3"
        report = self.quiet(step.run, source, self.root, output)
        self.assertTrue(report["ready_for_pilot_training"])
        self.assertFalse(report["physical_specimen_independence_confirmed"])
        self.assertEqual(report["pair_handling"]["total_candidate_pairs"], 1)
        self.assertEqual(report["pair_handling"]["cross_split_pairs_after"], 0)
        self.assertEqual(report["pair_handling"]["visually_inspected_contact_sheet_pairs"], 0)
        self.assertEqual(report["pair_handling"]["unreviewed_pairs_grouped_as_precaution"], 1)
        self.assertEqual(report["output_records"], 32)
        after = core.read_manifest(output / "manifest.csv")
        self.assertEqual({r["path"]: core.label(r) for r in before}, {r["path"]: core.label(r) for r in after})
        for name, data in input_bytes.items():
            self.assertEqual((source / name).read_bytes(), data)
        for r in before:
            self.assertEqual(core.sha_bytes((self.root / r["path"]).read_bytes()), r["sha256"])
        lock = json.loads((output / "dataset_lock.json").read_text())
        self.assertEqual(lock["manifest_sha256"], core.sha_bytes((output / "manifest.csv").read_bytes()))
        self.assertEqual(lock["report_sha256"], core.sha_bytes((output / "final_split_report.json").read_bytes()))

    def test_changed_manifest_is_rejected_before_pixel_scan(self):
        source = self.fixture()
        with (source / "manifest.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(core.DataError, "does not match split_report"):
            step.load_inputs(source)

    def test_incomplete_previous_audit_is_rejected(self):
        source = self.fixture()
        path = source / "split_report.json"
        report = json.loads(path.read_text())
        report["near_duplicate_review"]["complete"] = False
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(core.DataError, "incomplete"):
            step.load_inputs(source)

    def test_changed_image_stops_without_publishing(self):
        source = self.fixture()
        rows = core.read_manifest(source / "manifest.csv")
        (self.root / rows[0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(core.DataError, "SHA-256 mismatch"):
            self.quiet(step.run, source, self.root, self.base / "v3")
        self.assertFalse((self.base / "v3").exists())

    def test_omitted_candidates_are_found_by_local_recomputation(self):
        source = self.fixture()
        rows = core.read_manifest(source / "manifest.csv")
        with self.assertRaisesRegex(core.DataError, "differs from near_duplicates"):
            self.quiet(step.verify_pixels_and_candidates, rows, [], self.root)

    def test_exact_content_label_conflict_never_relabelled_automatically(self):
        rows = [row(1), row(2, status="rotten")]
        for r in rows:
            p = self.root / r["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), "red").save(p)
            r["sha256"] = core.sha_bytes(p.read_bytes())
        with self.assertRaisesRegex(core.DataError, "conflicting labels"):
            self.quiet(step.verify_pixels_and_candidates, rows, [pair(*rows)], self.root)
        self.assertEqual([r["status"] for r in rows], ["fresh", "rotten"])

    def test_existing_output_is_not_overwritten(self):
        output = self.base / "v3"
        output.mkdir()
        (output / "keep.txt").write_text("original output")
        with self.assertRaisesRegex(core.DataError, "already exists"):
            step.run(self.base / "unused", self.root, output)
        self.assertEqual((output / "keep.txt").read_text(), "original output")


if __name__ == "__main__":
    unittest.main(verbosity=2)
