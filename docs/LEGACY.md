# FreshLens legacy and development-history folders

The repository intentionally retains older code so the project history remains auditable and the V2 refactor can be checked against the previous implementation.

## Production code

For normal use, start from:

```text
freshlens_ai/
TRAIN_CNN_V2.py
EVALUATE_CNN_V2.py
PREDICT_CNN_V2.py
BUILD_OPENSET_GATE_V2.py
APP_CNN_V2.py
```

## Legacy / reference classification

| Path | Role now |
| --- | --- |
| `legacy/svm/` | Older SVM / handcrafted-feature baseline. Not production CNN V2. |
| `legacy/development/step1_data/` | Historical data-preparation / database integration stage. |
| `legacy/development/step2_split/` | Historical split-preparation stage. |
| `legacy/development/step2b_grouping/` | Historical grouping / near-duplicate handling stage. |
| `legacy/development/step3_cnn/` | Pre-refactor CNN implementation used as an equivalence reference. |
| `legacy/development/step4_external_test/` | External-test tooling/history. |
| `legacy/development/step4b_external_v2/` | External V2 tooling/history. |
| `legacy/development/step5_app/` | Former Streamlit/open-set application implementation. |
| `legacy/development/step5_fix/` | Corrected pre-refactor production open-set implementation used as a behavior reference. |

## Why these folders are still present

Several `tests/refactor/TEST_*_REFACTOR.py` checks intentionally compare the modular V2 implementation with legacy behavior. Deleting the old files immediately would remove those reference implementations and weaken the refactor audit trail.

Therefore:

- do not use legacy folders as the default runtime;
- do not add new production features there;
- keep them read-only except when repairing an equivalence test;
- do not delete them until the team explicitly decides that equivalence tests no longer need them.

## Archive status

Historical implementations have been moved under `legacy/`. The active
production runtime lives in `freshlens_ai/` and the root V2 entry points.
The archive is retained only for reproducibility, regression checks, and
refactor-equivalence tests.
