"""Single TV1 CLI for manifest building, provenance, leakage and quality audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freshlens_ai.data.dataset_audit import (
    DEFAULT_THRESHOLDS, OTHER_COLUMNS, QUALITY_COLUMNS, SUPPORTED_COLUMNS,
    audit_dataset, leakage_summary, load_path_mapping, mapped_image_path, quality_summary, read_csv,
)
from freshlens_ai.data.dataset_v2 import baseline_snapshot, build_dataset, verify_candidate_lock
from freshlens_ai.utils.file_io import atomic_json, write_csv


def check_output_paths(outputs, protected):
    outputs = [p for p in outputs if p is not None]
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError("Output files must have different paths")
    if any(p.resolve().is_relative_to(d.resolve()) for p in outputs for d in protected):
        raise ValueError("Report outputs must be outside protected input/dataset directories")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit metadata and real image bytes")
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--root", type=Path, help="Root containing mapped image paths")
    audit.add_argument("--report", type=Path)
    audit.add_argument("--quality-csv", type=Path)
    audit.add_argument("--quality-summary", type=Path)
    audit.add_argument("--quality-summary-csv", type=Path)
    audit.add_argument("--leakage-report", type=Path)
    audit.add_argument("--fingerprints", action="store_true", help="Compute decoded-pixel SHA and legacy pHash for provenance review")
    build = commands.add_parser("build", help="Create a NEW candidate directory, never overwrite")
    build.add_argument("--baseline", type=Path, default=Path("data/cnn_dataset_v3"))
    build.add_argument("--root", type=Path)
    build.add_argument("--inventory", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--require-images", action="store_true", help="Refuse metadata-only/partial candidates")
    provenance = commands.add_parser("provenance", help="Read-only review of raw SHA partition and unmatched files")
    provenance.add_argument("--baseline", type=Path, default=Path("data/cnn_dataset_v3"))
    provenance.add_argument("--originals-root", type=Path, required=True)
    provenance.add_argument("--raw-root", type=Path, required=True)
    provenance.add_argument("--original-quality-csv", type=Path, required=True)
    provenance.add_argument("--unmatched-csv", type=Path, required=True)
    provenance.add_argument("--matched-csv", type=Path, required=True)
    provenance.add_argument("--duplicates-csv", type=Path, required=True)
    provenance.add_argument("--report", type=Path, required=True)
    provenance.add_argument("--csv", type=Path, required=True)
    provenance.add_argument("--strip-prefix", default="raw")
    provenance.add_argument("--workers", type=int, default=4)
    for command in (audit, build):
        command.add_argument("--path-map-csv", type=Path, help="Existing SHA-match catalog for renamed originals")
        command.add_argument("--strip-prefix", default="", help="Remove an explicit path-component prefix only for image lookup")
        command.add_argument("--min-side", type=int, default=DEFAULT_THRESHOLDS["min_side"])
        command.add_argument("--min-brightness", type=float, default=DEFAULT_THRESHOLDS["min_brightness"])
        command.add_argument("--min-blur-score", type=float, default=DEFAULT_THRESHOLDS["min_blur_score"])
    args = parser.parse_args(argv)
    try:
        if args.command == "provenance":
            from freshlens_ai.data.dataset_provenance import analyze_provenance
            check_output_paths([args.report, args.csv], [args.baseline, args.originals_root, args.raw_root,
                                                       args.unmatched_csv.parent, args.original_quality_csv])
            report, rows = analyze_provenance(
                args.baseline, args.originals_root, args.raw_root, args.original_quality_csv,
                args.matched_csv, args.duplicates_csv, args.unmatched_csv, args.strip_prefix, workers=args.workers,
            )
            atomic_json(args.report, report)
            write_csv(args.csv, rows, list(rows[0]) if rows else ["raw_path", "classification"])
            print(json.dumps({k:report[k] for k in ("raw_image_files", "partition_counts", "classification_counts", "partition_verified")}, indent=2))
            return 0
        thresholds = {"min_side": args.min_side, "min_brightness": args.min_brightness, "min_blur_score": args.min_blur_score}
        if args.root is not None and not args.root.is_dir():
            raise ValueError("Image root does not exist")
        if args.command == "build":
            report = build_dataset(args.baseline, args.output, args.root, args.inventory, args.seed, thresholds,
                                   args.strip_prefix, args.require_images, args.path_map_csv)
        else:
            outputs = [args.report, args.quality_csv, args.quality_summary, args.quality_summary_csv, args.leakage_report]
            protected = [args.data] + ([args.path_map_csv] if args.path_map_csv else [])
            check_output_paths(outputs, protected)
            candidate = (args.data / "dataset_report.json").is_file()
            if candidate:
                verify_candidate_lock(args.data)
                rows = read_csv(args.data / "manifest.csv", SUPPORTED_COLUMNS)
                other = read_csv(args.data / "other_manifest.csv", OTHER_COLUMNS)
                identity = None
            else:
                rows, baseline_fingerprints, identity = baseline_snapshot(args.data)
                other = []
            overrides = load_path_mapping(args.path_map_csv, rows, args.strip_prefix)
            # Preserve --root . behavior while preventing any output from replacing an input image.
            if args.root is not None:
                image_paths = {(args.root / overrides.get(r["path"], mapped_image_path(r["path"], args.strip_prefix))).resolve() for r in rows + other}
                if any(p.resolve() in image_paths for p in outputs if p is not None):
                    raise ValueError("Report output must not overwrite an input image")
            report, quality = audit_dataset(rows, other, args.root, thresholds, args.strip_prefix, args.fingerprints, progress=True, path_overrides=overrides)
            report["baseline_lock_identity"] = identity
            if not candidate:
                report["baseline_fingerprints"] = baseline_fingerprints
            if args.report:
                atomic_json(args.report, report)
            if args.quality_csv:
                write_csv(args.quality_csv, quality, QUALITY_COLUMNS)
            if args.quality_summary or args.quality_summary_csv:
                summary, flat = quality_summary(rows, quality)
                summary["candidate_thresholds"] = thresholds
                if args.quality_summary:
                    atomic_json(args.quality_summary, summary)
                if args.quality_summary_csv:
                    write_csv(args.quality_summary_csv, flat, list(flat[0]))
            if args.leakage_report:
                atomic_json(args.leakage_report, leakage_summary(report))
        summary = {key: report[key] for key in ("records", "supported_records", "other_records", "split_counts", "raw_data_status", "metadata_valid", "all_image_files_verified", "final_test_status")}
        summary["quality"] = {k:report["quality"][k] for k in ("missing_files", "sha256_mismatches", "decode_errors", "low_quality_count")}
        print(json.dumps(summary, indent=2))
        if not report["metadata_valid"] or report["quality"]["sha256_mismatches"] or report["quality"]["decode_errors"] or report["quality"]["read_errors"]:
            return 1
        return 0 if report["all_image_files_verified"] else 2
    except (ValueError, OSError, RuntimeError) as exc:
        print("[ERROR] " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
