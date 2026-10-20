# External V2 calibration recheck

Executed on 2026-10-20 using the committed production checkpoint and gate,
CPU inference, and config/quality.json with quality rejection enabled.
103 real existing dataset photos: 81 supported, 22 unknown. This set had already
been used to calibrate the gate; this run is NOT an independent final test.

See run/metrics.json, predictions.csv, confusion_matrix.png, manifest.csv and
quality_config.json. Images themselves remain outside Git; predictions.csv names
error-gallery files generated locally, but those photos are intentionally ignored.
The manifest retains relative paths and actual SHA-256 for a repeatable rerun.

The policy accepts 53/81 known; 51/81 are both accepted and jointly correct.
The raw CNN is jointly correct on 75/81. All 22 calibration unknowns are rejected.
25 known images fail experimental quality thresholds, with no human quality
labels supplied, so a quality false-rejection rate cannot be established.
This evidence does not establish the target camera KPI.
