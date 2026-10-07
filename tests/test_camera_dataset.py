"""Camera supplementary inventory tests; fixtures live only in temporary directories."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from freshlens_ai.constants import CLASSES
from freshlens_ai.data.camera_dataset import (
    METADATA_COLUMNS, MANIFEST_COLUMNS, scan_camera, write_camera_inventory,
)
from freshlens_ai.data.dataset_audit import inspect_image
from freshlens_ai.utils.file_io import file_sha256, write_csv


class CameraDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "camera"
        self.root.mkdir()
        (self.base / "baseline").mkdir()
        (self.base / "baseline/dataset_lock.json").write_text("{}")
        self.baseline = []
        self.p1 = patch("freshlens_ai.data.camera_dataset.baseline_index", side_effect=lambda _:(self.baseline,{}))
        self.p2 = patch("freshlens_ai.data.camera_dataset.verify_candidate_lock", return_value={})
        self.p1.start(); self.p2.start()
        self.addCleanup(self.p1.stop); self.addCleanup(self.p2.stop)

    def photo(self, path="apple/fresh/a.png"):
        p = self.root / path
        p.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.random.default_rng(2).integers(0,256,(100,120,3),dtype=np.uint8)).save(p)
        return p

    def metadata(self, rows):
        p = self.base / "metadata.csv"
        write_csv(p,rows,METADATA_COLUMNS)
        return p

    def scan(self, metadata=None):
        return scan_camera(self.root,self.base/"baseline",metadata)

    def test_empty_and_all_eight_combinations_and_output_schema(self):
        # A directory rather than fabricated sample rows represents every label.
        for label in CLASSES:
            (self.root / label.replace("::","/")).mkdir(parents=True)
        output = self.base / "reports"
        # Only the lock-file hash helper needs a real harmless file in this fixture.
        baseline = self.base/"baseline"; baseline.mkdir(exist_ok=True)
        (baseline/"dataset_lock.json").write_text("{}")
        report = write_camera_inventory(self.root,baseline,output)
        self.assertEqual(report["records"],0)
        self.assertEqual(set(report["class_counts"]),set(CLASSES))
        with (output/"manifest.csv").open(encoding="utf-8-sig") as f:
            reader=csv.DictReader(f)
            self.assertEqual(tuple(reader.fieldnames),MANIFEST_COLUMNS)
            self.assertEqual(list(reader),[])
        write_camera_inventory(self.root,baseline,output)
        (output/"unrelated.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError,"unrelated"):
            write_camera_inventory(self.root,baseline,output)

    def test_nonempty_inventory_rerun_keeps_inventory_kind(self):
        self.photo()
        baseline=self.base/"baseline"
        output=self.base/"reports"
        first=write_camera_inventory(self.root,baseline,output)
        second=write_camera_inventory(self.root,baseline,output)
        self.assertEqual(first["inventory_kind"],"supported")
        self.assertEqual(second["inventory_kind"],"supported")
        self.assertEqual(first["manifest_sha256"],second["manifest_sha256"])

    def test_sha_quality_and_no_automatic_split_or_source_guess(self):
        p=self.photo()
        rows,report=self.scan()
        r=rows[0]
        self.assertEqual(r["sha256"],file_sha256(p))
        self.assertEqual((r["width"],r["height"]),(120,100))
        self.assertEqual(r["source"],"UNKNOWN")
        self.assertEqual(r["split"],"UNASSIGNED")
        self.assertFalse(r["eligible_for_training"])
        self.assertEqual(r["group_evidence_status"],"UNVERIFIED")
        self.assertEqual(r["glare"],"UNKNOWN")

    def test_sha_duplicate_flags_all_camera_members(self):
        p=self.photo()
        other=self.root/"apple/rotten/b.png"
        other.parent.mkdir(parents=True)
        other.write_bytes(p.read_bytes())
        rows,report=self.scan()
        self.assertEqual(report["duplicate_camera_records"],2)
        self.assertTrue(all("SHA256" in r["duplicate_type"] for r in rows))
        self.assertEqual(rows[0]["group_id"],rows[1]["group_id"])
        self.assertEqual({r["status"] for r in rows},{"fresh","rotten"})

    def test_baseline_sha_pixel_and_phash_reuse(self):
        p=self.photo()
        q=inspect_image(self.root,dict(path="apple/fresh/a.png",sha256=file_sha256(p)),fingerprints=True)
        self.baseline.append(dict(path="raw/apple/fresh/baseline.png",sha256=q["actual_sha256"],
                                  pixel_sha256=q["pixel_sha256"],phash63=q["phash63"]))
        rows,report=self.scan()
        self.assertEqual(report["duplicate_baseline_records"],1)
        self.assertIn("baseline:raw/apple/fresh/baseline.png",rows[0]["duplicate_of"])
        self.assertIn("DECODED_PIXEL",rows[0]["duplicate_type"])
        self.assertEqual(report["perceptual_baseline_candidates"],1)

    def test_missing_and_corrupt_remain_in_manifest(self):
        broken=self.root/"tomato/rotten/broken.jpg"
        broken.parent.mkdir(parents=True); broken.write_bytes(b"not an image")
        meta=self.metadata([dict(path="orange/fresh/missing.png",source="camera")])
        rows,report=self.scan(meta)
        bypath={r["path"]:r for r in rows}
        self.assertEqual(bypath["orange/fresh/missing.png"]["decode_status"],"MISSING_FILE")
        self.assertIn("MISSING_FILE",bypath["orange/fresh/missing.png"]["quality_flags"])
        self.assertEqual(bypath["tomato/rotten/broken.jpg"]["decode_status"],"DECODE_ERROR")
        self.assertTrue(broken.exists())

    def test_manual_hard_cases_and_web_provenance_never_relabel(self):
        self.photo("banana/rotten/a.png")
        meta=self.metadata([dict(path="banana/rotten/a.png",source="web",lighting="uneven",
             distance="close",angle="tilted",background="complex",multiple_objects="true",
             severity="small_defect",source_url="https://example.invalid/source",license="UNKNOWN")])
        rows,_=self.scan(meta)
        r=rows[0]
        self.assertEqual((r["fruit"],r["status"]),("banana","rotten"))
        self.assertEqual((r["lighting"],r["severity"]),("uneven","small_defect"))
        self.assertEqual(r["source"],"web")
        self.assertEqual(r["noise"],"UNKNOWN")

    def test_high_brightness_is_candidate_not_glare_label(self):
        p=self.photo()
        Image.new("RGB",(120,100),"white").save(p)
        rows,_=self.scan()
        self.assertIn("HIGH_BRIGHTNESS_CANDIDATE",rows[0]["quality_flags"])
        self.assertEqual(rows[0]["glare"],"UNKNOWN")

    def test_expected_hash_mismatch_and_invalid_metadata_paths(self):
        self.photo()
        rows,_=self.scan(self.metadata([dict(path="apple/fresh/a.png",sha256="f"*64)]))
        self.assertIn("SHA256_MISMATCH",rows[0]["quality_flags"])
        with self.assertRaises(ValueError):
            self.scan(self.metadata([dict(path="../escape.png")]))
        with self.assertRaisesRegex(ValueError,"Repeated"):
            self.scan(self.metadata([dict(path="apple/fresh/a.png"),dict(path="apple/fresh/a.png")]))

    def test_refuse_output_in_v5_and_original_image_roots(self):
        for output in (self.root,self.root/"output",Path(__file__).resolve().parents[1]/"data/cnn_dataset_v5"):
            with self.assertRaisesRegex(ValueError,"outside"):
                write_camera_inventory(self.root,self.base/"baseline",output)

    def test_live_v5_lock_and_bytes_unchanged_by_empty_scan(self):
        self.p1.stop(); self.p2.stop()
        v5=Path(__file__).resolve().parents[1]/"data/cnn_dataset_v5"
        before={p.name:file_sha256(p) for p in v5.iterdir() if p.is_file()}
        rows,report=scan_camera(self.root,v5)
        self.assertEqual(rows,[])
        self.assertEqual(report["records"],0)
        self.assertEqual(before,{p.name:file_sha256(p) for p in v5.iterdir() if p.is_file()})


if __name__ == "__main__":
    unittest.main()
