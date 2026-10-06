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
| `src/` | Older SVM / handcrafted-feature baseline. Not production CNN V2. |
| `FreshLens_Buoc1_DuLieu/` | Historical data-preparation / database integration stage. |
| `FreshLens_Buoc2_ChiaTap/` | Historical split-preparation stage. |
| `FreshLens_Buoc2B_GomNhom/` | Historical grouping / near-duplicate handling stage. |
| `FreshLens_Buoc3_CNN/` | Pre-refactor CNN implementation used as an equivalence reference. |
| `FreshLens_Buoc4_ExternalTest/` | External-test tooling/history. |
| `FreshLens_Buoc4B_MoRongExternalTest/` | External V2 tooling/history. |
| `FreshLens_Buoc5_AppThucTe/` | Former Streamlit/open-set application implementation. |
| `FreshLens_Buoc5_FIX/` | Corrected pre-refactor production open-set implementation used as a behavior reference. |

## Why these folders are still present

Several `TEST_*_REFACTOR.py` checks intentionally compare the modular V2 implementation with legacy behavior. Deleting the old files immediately would remove those reference implementations and weaken the refactor audit trail.

Therefore:

- do not use legacy folders as the default runtime;
- do not add new production features there;
- keep them read-only except when repairing an equivalence test;
- do not delete them until the team explicitly decides that equivalence tests no longer need them.

## Cleanup rule

A later archival step may move historical code under a dedicated archive layout, but that should be a separate commit after all path-sensitive tests have been updated and rerun.
