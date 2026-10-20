"""Admission, isolated candidate splits, OTHER separation and CLI integration."""
import json
import tempfile
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import unittest

import numpy as np
from PIL import Image

from freshlens_ai.data.camera_dataset import METADATA_COLUMNS, scan_camera, inventory_leakage
from freshlens_ai.data.dataset_audit import QUALITY_COLUMNS, SUPPORTED_COLUMNS, inspect_image
from freshlens_ai.data.reviewed_candidate import (
    build_reviewed_candidate, load_reviewed_candidate, make_reviewed_dataset, audit_reviewed_candidate,
)
from freshlens_ai.utils.file_io import write_csv, atomic_json, file_sha256
from scripts.dataset_v2 import main


class ReviewedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base=Path(self.temporary.name)
        self.baseline=self.base/"v5"; self.baseline.mkdir()
        self.images=self.base/"originals"; self.images.mkdir()
        baseline=[]; quality=[]
        for index,split in enumerate(("train","test")):
            relative=f"apple/fresh/base{index}.png"
            p=self.images/relative; p.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.random.default_rng(index).integers(0,256,(100,120,3),dtype=np.uint8)).save(p)
            row=dict(path="raw/"+relative,fruit="apple",status="fresh",source="local",
                     group_id="baseline"+str(index),sha256=file_sha256(p),split=split)
            baseline.append(row)
            quality.append(inspect_image(self.images,row,strip_prefix="raw",fingerprints=True))
        write_csv(self.baseline/"manifest.csv",baseline,SUPPORTED_COLUMNS)
        write_csv(self.baseline/"other_manifest.csv",[],("path","category","subcategory","source","group_id","sha256","split"))
        write_csv(self.baseline/"image_quality.csv",quality,QUALITY_COLUMNS)
        atomic_json(self.baseline/"dataset_report.json",{})
        atomic_json(self.baseline/"dataset_lock.json",dict(schema_version="freshlens_dataset_v2_lock_1",
            image_path_mapping={"strip_prefix":"raw"},supported_records=2,other_records=0,
            files={n:file_sha256(self.baseline/n) for n in ("manifest.csv","other_manifest.csv","image_quality.csv","dataset_report.json")}))
        self.camera=self.base/"camera"; self.camera.mkdir()
        self.other=self.base/"other"
        self.meta=self.base/"metadata.csv"
        self.other_meta=self.base/"other.csv"
        write_csv(self.meta,[],METADATA_COLUMNS); write_csv(self.other_meta,[],METADATA_COLUMNS)
        self.config=self.base/"sources.json"
        atomic_json(self.config,[dict(id="camera",kind="supported",root=str(self.camera),metadata=str(self.meta)),
                                dict(id="other",kind="other",root=str(self.other),metadata=str(self.other_meta))])
        self.before={p.name:file_sha256(p) for p in self.baseline.iterdir()}
        self.output=self.base/"candidate"

    def photo(self,path="banana/rotten/new.png",other=False):
        root=self.other if other else self.camera
        p=root/path; p.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.random.default_rng(18).integers(0,256,(105,130,3),dtype=np.uint8)).save(p)
        return p

    def review(self,path="banana/rotten/new.png",**kwargs):
        row=dict(path=path,source="web",source_name="fixture",source_url="https://example.invalid/test",
                 license="CC0",group_id="specimen_18",specimen_evidence="fixture log",
                 label_evidence="fixture visual review",review_status="APPROVED",reviewer="test",
                 review_evidence="fixture review log",eligible_for_training="true",requested_split="train")
        row.update(kwargs)
        return row

    def build(self,**kwargs):
        return build_reviewed_candidate(self.baseline,self.images,self.config,self.output,**kwargs)

    def test_unreviewed_excluded_and_dry_run_has_no_output(self):
        self.photo()
        report=self.build(dry_run=True)
        self.assertEqual(report["excluded_unreviewed"]["camera"],1)
        self.assertEqual(report["records"],2)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.before,{p.name:file_sha256(p) for p in self.baseline.iterdir()})

    def test_reviewed_build_audit_loader_and_frozen_baseline(self):
        self.photo()
        write_csv(self.meta,[self.review()],METADATA_COLUMNS)
        report=self.build()
        self.assertEqual(report["added_supported_records"],1)
        self.assertEqual(report["split_counts"],{"train":2,"val":0,"test":1})
        supported,other=load_reviewed_candidate(self.output)
        self.assertEqual(len(other),0)
        original=[r for r in supported if r["root_id"]=="baseline"]
        self.assertEqual([r["split"] for r in original],["train","test"])
        audit,_=audit_reviewed_candidate(self.output,self.images,self.config)
        self.assertTrue(audit["all_image_files_verified"])
        self.assertEqual(self.before,{p.name:file_sha256(p) for p in self.baseline.iterdir()})
        with self.assertRaisesRegex(ValueError,"NEW"):
            self.build()
        (self.output/"manifest.csv").write_text("tampered")
        with self.assertRaisesRegex(ValueError,"modified"):
            load_reviewed_candidate(self.output)

    def test_unknown_group_or_unreviewed_or_external_license_blocks(self):
        self.photo()
        for overrides in (dict(group_id="UNKNOWN"),dict(review_status="UNKNOWN"),dict(license="UNKNOWN")):
            write_csv(self.meta,[self.review(**overrides)],METADATA_COLUMNS)
            with self.assertRaises(ValueError):
                self.build()
            self.assertFalse(self.output.exists())

    def test_no_new_test_and_baseline_test_overlap_rejected(self):
        p=self.photo("apple/fresh/new.png")
        write_csv(self.meta,[self.review("apple/fresh/new.png",requested_split="test")],METADATA_COLUMNS)
        with self.assertRaisesRegex(ValueError,"cannot enter test"):
            self.build()
        p.write_bytes((self.images/"apple/fresh/base1.png").read_bytes())
        write_csv(self.meta,[self.review("apple/fresh/new.png",requested_split="UNASSIGNED")],METADATA_COLUMNS)
        with self.assertRaisesRegex(ValueError,"overlaps baseline test"):
            self.build()

    def test_group_and_sha_leakage_requested_splits(self):
        p=self.photo()
        second=self.camera/"banana/rotten/second.png"
        second.write_bytes(p.read_bytes())
        write_csv(self.meta,[self.review(requested_split="train"),
            self.review("banana/rotten/second.png",group_id="othergroup",requested_split="val")],METADATA_COLUMNS)
        rows,report=scan_camera(self.camera,self.baseline,self.meta)
        self.assertEqual(report["leakage"]["cross_split_sha"],1)
        with self.assertRaisesRegex(ValueError,"crosses frozen splits"):
            self.build()
        # Same group, distinct bytes also cannot be split.
        Image.new("RGB",(130,105),"blue").save(second)
        write_csv(self.meta,[self.review(),self.review("banana/rotten/second.png",requested_split="val")],METADATA_COLUMNS)
        with self.assertRaisesRegex(ValueError,"crosses frozen splits"):
            self.build()

    def test_decoded_pixel_leakage_different_encoding_is_linked(self):
        p=self.photo()
        second=self.camera/"banana/rotten/second.png"
        with Image.open(p) as img: img.save(second,compress_level=0)
        self.assertNotEqual(file_sha256(p),file_sha256(second))
        write_csv(self.meta,[self.review(),self.review("banana/rotten/second.png",group_id="other",requested_split="val")],METADATA_COLUMNS)
        with self.assertRaisesRegex(ValueError,"crosses frozen splits"):
            self.build()

    def test_other_absent_and_never_maps_to_supported_or_training(self):
        rows,report=scan_camera(self.other,self.baseline,self.other_meta,kind="other")
        self.assertEqual(rows,[])
        self.photo("fruit/mango/new.png",other=True)
        row=self.review("fruit/mango/new.png",eligible_for_training="true",eligible_for_evaluation="true",group_id="mango")
        write_csv(self.other_meta,[row],METADATA_COLUMNS)
        rows,report=scan_camera(self.other,self.baseline,self.other_meta,kind="other")
        self.assertEqual(rows[0]["label_type"],"OTHER")
        self.assertEqual(rows[0]["fruit"],"UNKNOWN")
        self.assertFalse(rows[0]["eligible_for_training"])
        report=self.build()
        supported,other=load_reviewed_candidate(self.output)
        self.assertEqual(len(supported),2)
        self.assertEqual(other[0]["category"],"other_fruit")
        self.assertEqual(other[0]["subcategory"],"mango")
        roots={"baseline":self.images,"other":self.other,"camera":self.camera}
        self.assertEqual(len(make_reviewed_dataset(self.output,roots,"train")),1)

    def test_invalid_sha_corrupt_reviewed_cannot_build(self):
        p=self.photo()
        write_csv(self.meta,[self.review(sha256="f"*64)],METADATA_COLUMNS)
        with self.assertRaisesRegex(ValueError,"SHA/decode"):
            self.build()
        p.write_bytes(b"broken")
        write_csv(self.meta,[self.review()],METADATA_COLUMNS)
        with self.assertRaisesRegex(ValueError,"SHA/decode"):
            self.build()

    def test_augmentation_only_train_reproducible_and_original_unchanged(self):
        self.photo()
        write_csv(self.meta,[self.review(requested_split="val")],METADATA_COLUMNS)
        self.build()
        roots={"baseline":self.images,"camera":self.camera}
        import torch
        val=make_reviewed_dataset(self.output,roots,"val")
        self.assertTrue(torch.equal(val[0][0],val[0][0]))
        test=make_reviewed_dataset(self.output,roots,"test")
        self.assertTrue(torch.equal(test[0][0],test[0][0]))
        with self.assertRaisesRegex(ValueError,"augmentation"):
            make_reviewed_dataset(self.output,roots,"test",training=True)
        train=make_reviewed_dataset(self.output,roots,"train",training=True)
        torch.manual_seed(5); a=train[0][0]
        torch.manual_seed(5); b=train[0][0]
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(self.before,{p.name:file_sha256(p) for p in self.baseline.iterdir()})

    def test_camera_provenance_and_shared_specimen_evidence(self):
        self.photo()
        metadata=self.review(source="camera",capture_device="phone",capture_context="room",
                             specimen_id="physical",session_id="session",angle="side")
        write_csv(self.meta,[metadata],METADATA_COLUMNS)
        self.build()
        report,_=audit_reviewed_candidate(self.output,self.images,self.config)
        self.assertTrue(report["metadata_valid"])

    def test_unknown_group_and_invalid_class_and_explicit_hard_tags(self):
        self.photo()
        rows,_=scan_camera(self.camera,self.baseline)
        self.assertEqual(rows[0]["group_id"],"UNKNOWN")
        write_csv(self.meta,[self.review(lighting="low_light",noise="true",glare="true",
            occlusion="hand_held",severity="small_defect",requested_split="val")],METADATA_COLUMNS)
        rows,_=scan_camera(self.camera,self.baseline,self.meta)
        self.assertEqual((rows[0]["fruit"],rows[0]["status"]),("banana","rotten"))
        self.assertEqual(set(rows[0]["hard_case_tags"].split(";")),{"low_light","noisy","glare","hand_held","small_defect"})
        self.photo("grape/fresh/invalid.png")
        with self.assertRaisesRegex(ValueError,"eight"):
            scan_camera(self.camera,self.baseline)

    def test_baseline_image_bytes_changed_refuse_before_output(self):
        (self.images/"apple/fresh/base0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"Baseline image changed"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_perceptual_candidate_requires_review_before_admission(self):
        from freshlens_ai.data.reviewed_candidate import admission
        self.photo()
        write_csv(self.meta,[self.review()],METADATA_COLUMNS)
        rows,_=scan_camera(self.camera,self.baseline,self.meta)
        row=rows[0]
        row["perceptual_candidates"]=json.dumps([{"reference":"baseline:raw/apple/fresh/near.png","distance":2}])
        with self.assertRaisesRegex(ValueError,"Perceptual"):
            admission(row)
        row["perceptual_review_status"]="APPROVED"
        row["perceptual_review_evidence"]="fixture: checked independent specimen"
        admission(row)

    def test_cli_help_camera_other_aliases_and_dry_build(self):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as top:
                main(["--help"])
            self.assertEqual(top.exception.code,0)
            for command in ("audit","camera","other","provenance","duplicates","leakage","quality","build"):
                with self.assertRaises(SystemExit) as context:
                    main([command,"--help"])
                self.assertEqual(context.exception.code,0)
            output=self.base/"inventory"
            self.assertEqual(main(["camera","--root",str(self.camera),"--baseline",str(self.baseline),"--output",str(output)]),0)
            with self.assertRaises(SystemExit) as top:
                main(["--help"])
            self.assertEqual(top.exception.code,0)
            for command in ("audit","duplicates","leakage","quality"):
                self.assertEqual(main([command,"--data",str(output),"--root",str(self.camera),"--baseline",str(self.baseline)]),0)
            self.assertEqual(main(["build","--baseline",str(self.baseline),"--root",str(self.images),
                                  "--sources-json",str(self.config),"--dry-run"]),0)


if __name__=="__main__":
    unittest.main()
