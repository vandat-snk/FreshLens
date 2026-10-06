"""Single TV1 CLI for manifest building, duplicate/leakage and quality audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freshlens_ai.data.dataset_audit import (
    DEFAULT_THRESHOLDS, OTHER_COLUMNS, QUALITY_COLUMNS, SUPPORTED_COLUMNS,
    audit_dataset, read_csv,
)
from freshlens_ai.data.dataset_v2 import baseline_snapshot, build_dataset, verify_candidate_lock
from freshlens_ai.utils.file_io import atomic_json, write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit metadata and optionally real image bytes")
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--root", type=Path, help="Root containing manifest-relative image paths")
    audit.add_argument("--report", type=Path)
    audit.add_argument("--quality-csv", type=Path)
    build = commands.add_parser("build", help="Create a NEW candidate directory, never overwrite")
    build.add_argument("--baseline", type=Path, default=Path("data/cnn_dataset_v3"))
    build.add_argument("--root", type=Path)
    build.add_argument("--inventory", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=42)
    for command in (audit, build):
        command.add_argument("--min-side", type=int, default=DEFAULT_THRESHOLDS["min_side"])
        command.add_argument("--min-brightness", type=float, default=DEFAULT_THRESHOLDS["min_brightness"])
        command.add_argument("--min-blur-score", type=float, default=DEFAULT_THRESHOLDS["min_blur_score"])
    args = parser.parse_args(argv)
    thresholds = {"min_side": args.min_side, "min_brightness": args.min_brightness, "min_blur_score": args.min_blur_score}
    try:
        if args.root is not None and not args.root.is_dir():
            raise ValueError("Image root does not exist")
        if args.command == "build":
            report = build_dataset(args.baseline, args.output, args.root, args.inventory, args.seed, thresholds)
        else:
            protected = args.data.resolve()
            outputs = [p for p in (args.report, args.quality_csv) if p is not None]
            if any(p.resolve().is_relative_to(protected) for p in outputs):
                raise ValueError("Audit outputs must be outside the audited dataset directory")
            if len({p.resolve() for p in outputs}) != len(outputs):
                raise ValueError("Audit report and quality CSV must have different paths")
            candidate = (args.data / "dataset_report.json").is_file()
            if candidate:
                verify_candidate_lock(args.data)
                rows = read_csv(args.data / "manifest.csv", SUPPORTED_COLUMNS)
                other = read_csv(args.data / "other_manifest.csv", OTHER_COLUMNS)
                identity = None
            else:
                rows, baseline_fingerprints, identity = baseline_snapshot(args.data)
                other = []
            report, quality = audit_dataset(rows, other, args.root, thresholds)
            report["baseline_lock_identity"] = identity
            if not candidate:
                report["baseline_fingerprints"] = baseline_fingerprints
            if args.report:
                atomic_json(args.report, report)
            if args.quality_csv:
                write_csv(args.quality_csv, quality, QUALITY_COLUMNS)
        summary = {key: report[key] for key in ("records", "supported_records", "other_records", "split_counts", "raw_data_status", "metadata_valid", "all_image_files_verified", "final_test_status")}
        print(json.dumps(summary, indent=2))
        if not report["metadata_valid"] or report["quality"]["sha256_mismatches"] or report["quality"]["decode_errors"]:
            return 1
        # Exit 2 deliberately means incomplete image verification, not a clean data gate.
        return 0 if report["all_image_files_verified"] else 2
    except (ValueError, OSError, RuntimeError) as exc:
        print("[ERROR] " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
