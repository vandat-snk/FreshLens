"""Regression checks for step 1. No MongoDB, Cloudinary or network access."""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from scripts import db_manager as db
from scripts import export_manifest_from_db as exporter
from scripts.upload_dataset_to_db import scan_images, upload_dataset
from src.check_dataset import check_dataset
from src.data_integrity import assign_missing_splits, metadata_errors, metadata_summary
from src.dataset import load_manifest, save_manifest, scan_raw_dataset, sha256_file


def row(name: str, split: str = "", **changes) -> dict:
    record = dict(path=f"raw/apple/fresh/{name}.png", fruit="apple", status="fresh", source="phone", group_id=name, sha256="", split=split)
    record.update(changes)
    return record


class Step1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="freshlens_step1_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = io.StringIO()
        self.quiet = contextlib.redirect_stdout(self.output)
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def image_record(self, name, split="", color=(130, 90, 40)):
        record = row(name, split)
        path = self.root / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 20), color).save(path)
        record["sha256"] = sha256_file(path)
        return record

    def export(self, records, output=None, **kwargs):
        with patch.object(exporter, "test_connection", return_value=True), patch.object(exporter, "get_images_collection") as collection, patch.object(exporter, "get_all_records", return_value=copy.deepcopy(records)), patch.object(exporter, "update_splits") as update:
            result = exporter.export_manifest(output or self.root / "manifest.csv", dest_root=self.root, **kwargs)
            collection.assert_called_once_with(db.DEFAULT_URI, db.DEFAULT_DB, ensure_indexes=False)
            if not kwargs.get("write_splits"):
                update.assert_not_called()
            return result

    def test_mixed_existing_new_groups_keep_existing_splits(self):
        records = [row("a", "test"), row("b", "val"), row("c")]
        result = assign_missing_splits(records)
        self.assertEqual([r["split"] for r in result], ["test", "val", "train"])
        self.assertEqual(records[-1]["split"], "")

    def test_new_view_inherits_existing_group(self):
        result = assign_missing_splits([row("a", "val"), row("new_view", group_id="a")])
        self.assertEqual([r["split"] for r in result], ["val", "val"])

    def test_initial_split_deterministic_for_database_order(self):
        records = [row(str(i), sha256=f"{i:064x}") for i in range(12)]
        a = {r["path"]: r["split"] for r in assign_missing_splits(records)}
        b = {r["path"]: r["split"] for r in assign_missing_splits(list(reversed(records)))}
        self.assertEqual(a, b)
        self.assertEqual(set(a.values()), {"train", "val", "test"})

    def test_duplicate_content_groups_inherit_fixed_split(self):
        result = assign_missing_splits([row("a", "test", sha256="same"), row("b", sha256="same")])
        self.assertEqual([r["split"] for r in result], ["test", "test"])

    def test_group_leakage_rejected(self):
        with self.assertRaises(ValueError):
            assign_missing_splits([row("a", "train"), row("b", "test", group_id="a")])

    def test_duplicate_content_across_existing_splits_rejected(self):
        with self.assertRaises(ValueError):
            assign_missing_splits([row("a", "train", sha256="same"), row("b", "test", sha256="same")])

    def test_transitive_duplicate_group_conflict_rejected(self):
        records = [row("a", "train", sha256="x"), row("b", sha256="x"), row("b2", group_id="b", sha256="y"), row("c", "test", sha256="y")]
        with self.assertRaises(ValueError):
            assign_missing_splits(records)

    def test_invalid_and_missing_split_rejected_by_metadata_check(self):
        self.assertTrue(metadata_errors([row("a", "validation")]))
        self.assertTrue(metadata_errors([row("a")]))
        self.assertTrue(metadata_errors([]))

    def test_missing_group_is_not_silently_derived_from_name(self):
        with self.assertRaises(ValueError):
            assign_missing_splits([row("a", group_id="")])

    def test_upload_and_local_scanner_share_group_metadata(self):
        records = [self.image_record("a"), self.image_record("b", color=(20, 160, 50))]
        with (self.root / "groups.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "group_id", "source"])
            writer.writeheader()
            for record in records:
                writer.writerow(dict(relative_path=record["path"], group_id="one_actual_fruit", source="phone_A"))
        local = scan_raw_dataset(self.root / "raw", self.root)
        uploaded = scan_images(self.root / "raw", self.root)
        self.assertEqual([(r["group_id"], r["source"], r["sha256"]) for r in local], [(r["group_id"], r["source"], r["sha256"]) for r in uploaded])
        self.assertEqual({r["group_id"] for r in uploaded}, {"one_actual_fruit"})

    def test_reupload_preserves_curated_metadata_split_and_cloud_url(self):
        collection = Mock()
        collection.find_one.return_value = dict(row("curated", "test", sha256="hash"), url="https://example.test/existing.png")
        collection.update_one.return_value = SimpleNamespace(upserted_id=None)
        db.upsert_image(collection, row("fallback", sha256="hash", url="", source="local"))
        update = collection.update_one.call_args.args[1]
        for key in ("split", "url", "group_id", "source"):
            self.assertNotIn(key, update["$set"])
        self.assertEqual(update["$setOnInsert"], {"split": ""})

    def test_reupload_conflicting_labels_rejected(self):
        collection = Mock()
        collection.find_one.return_value = row("a", "train", sha256="hash")
        with self.assertRaises(ValueError):
            db.upsert_image(collection, row("a", sha256="hash", status="rotten"))
        collection.update_one.assert_not_called()

    def test_default_collection_access_does_not_create_indexes(self):
        collection = Mock()
        with patch.object(db, "get_db", return_value={db.COLLECTION_IMAGES: collection}):
            self.assertIs(db.get_images_collection(), collection)
        collection.create_index.assert_not_called()

    def test_connection_log_hides_uri_and_closes_probe_client(self):
        client = Mock()
        with patch.object(db, "get_client", return_value=client):
            self.assertTrue(db.test_connection("mongodb+srv://person:supersecret@example.invalid"))
        client.close.assert_called_once()
        self.assertNotIn("supersecret", self.output.getvalue())
        self.assertNotIn("person", self.output.getvalue())

    def test_upload_dry_run_does_not_connect(self):
        self.image_record("a")
        with patch("scripts.upload_dataset_to_db.test_connection") as connect, patch("scripts.upload_dataset_to_db.get_images_collection") as collection:
            upload_dataset(self.root, dry_run=True)
            connect.assert_not_called()
            collection.assert_not_called()

    def test_export_includes_new_records_and_is_read_only_by_default(self):
        records = [self.image_record("old", "test"), self.image_record("new", color=(70, 140, 80))]
        result = load_manifest(self.export(records))
        self.assertEqual([r["split"] for r in result], ["test", "train"])
        self.assertEqual(len(result), 2)

    def test_export_reuses_previous_local_snapshot(self):
        records = [self.image_record(str(i), color=(20 + i * 10, 90, 40)) for i in range(8)]
        first = {r["path"]: r["split"] for r in load_manifest(self.export(records))}
        records.append(self.image_record("new", color=(199, 98, 47)))
        second = {r["path"]: r["split"] for r in load_manifest(self.export(list(reversed(records))))}
        self.assertTrue(all(second[path] == split for path, split in first.items()))
        self.assertEqual(second[records[-1]["path"]], "train")

    def test_failed_download_preserves_existing_manifest(self):
        previous = self.image_record("old", "train")
        manifest = save_manifest([previous], self.root / "manifest.csv")
        original = manifest.read_bytes()
        missing = row("missing", url="https://example.test/missing.png")
        with patch.object(exporter, "_download_image", return_value=False), patch.object(exporter, "update_splits") as update:
            with self.assertRaises(RuntimeError):
                self.export([previous, missing], download=True, write_splits=True)
            update.assert_not_called()
        self.assertEqual(manifest.read_bytes(), original)

    def test_missing_local_file_does_not_create_manifest(self):
        with self.assertRaises(ValueError):
            self.export([row("missing")])
        self.assertFalse((self.root / "manifest.csv").exists())

    def test_bad_download_does_not_leave_partial_file(self):
        destination = self.root / "bad.png"
        with patch.object(exporter.urllib.request, "urlopen", return_value=io.BytesIO(b"not an image")):
            self.assertFalse(exporter._download_image("https://example.test/bad", destination))
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_valid_download_uses_timeout_and_atomic_file(self):
        buf = io.BytesIO()
        Image.new("RGB", (12, 12), "red").save(buf, format="PNG")
        destination = self.root / "good.png"
        with patch.object(exporter.urllib.request, "urlopen", return_value=io.BytesIO(buf.getvalue())) as request:
            self.assertTrue(exporter._download_image("https://example.test/good", destination))
            request.assert_called_once_with("https://example.test/good", timeout=30)
        self.assertTrue(exporter._valid_image(destination))
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_same_basename_urls_do_not_collide(self):
        records = [row("a", url="https://one.test/image.png"), row("b", url="https://two.test/image.png")]
        def fake_download(url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10), "red" if "one" in url else "blue").save(destination)
            return True
        with patch.object(exporter, "_download_image", side_effect=fake_download):
            exporter.download_images(records, self.root)
        self.assertNotEqual(records[0]["path"], records[1]["path"])

    def test_checker_recomputes_hashes_and_blocks_content_leakage(self):
        a = self.image_record("a", "train")
        b = self.image_record("b", "test")
        a["sha256"], b["sha256"] = "stale_a", "stale_b"
        manifest = save_manifest([a, b], self.root / "manifest.csv")
        report = check_dataset(self.root, manifest)
        reasons = {error["reason"] for error in report["errors"]}
        self.assertFalse(report["is_valid"])
        self.assertIn("sha256_mismatch", reasons)
        self.assertIn("duplicate_content_across_splits", reasons)

    def test_checker_empty_manifest_invalid(self):
        manifest = save_manifest([], self.root / "manifest.csv")
        self.assertFalse(check_dataset(self.root, manifest)["is_valid"])

    def test_summary_omits_credentials_urls_and_paths(self):
        record = row("a", url="https://user:secret@example.test/file", path="E:/private/photo.png")
        report = metadata_summary([record])
        text = json.dumps(report)
        self.assertNotIn("secret", text)
        self.assertNotIn("private", text)
        self.assertEqual(report["with_cloud_url"], 1)

    def test_stats_command_writes_only_local_summary(self):
        report_path = self.root / "summary.json"
        args = ["export_manifest_from_db.py", "--stats", "--report", str(report_path)]
        with patch("sys.argv", args), patch.object(exporter, "test_connection", return_value=True), patch.object(exporter, "get_images_collection") as collection, patch.object(exporter, "get_all_records", return_value=[row("a")]), patch.object(exporter, "update_splits") as update, patch.object(exporter, "download_images") as download:
            exporter.main()
            collection.assert_called_once_with(db.DEFAULT_URI, db.DEFAULT_DB, ensure_indexes=False)
            update.assert_not_called()
            download.assert_not_called()
        self.assertEqual(json.loads(report_path.read_text())["total_images"], 1)

    def test_train_stops_before_fitting_when_manifest_invalid(self):
        from src import train
        with patch.object(train, "ensure_runtime_dirs"), patch.object(train, "REPORT_ROOT", self.root), patch.object(train, "check_dataset", return_value={"is_valid": False, "errors": [{"reason": "invalid_split"}]}), patch.object(train, "fit_classifier") as fit:
            with self.assertRaises(ValueError):
                train.run_training(self.root, self.root / "manifest.csv", self.root / "model.joblib")
            fit.assert_not_called()
        self.assertFalse((self.root / "model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
