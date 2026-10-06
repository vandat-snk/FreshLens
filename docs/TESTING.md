# FreshLens Stage integration suite - Final integration check

Run from the project root:

```cmd
.\.venv\Scripts\python.exe -m tests.integration.TEST_FULL_PIPELINE_V2 ^
  --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" ^
  --data "data\cnn_dataset_v3"
```

The script:

1. runs every equivalence/smoke test from Stage B through G4;
2. audits the new modular package and V2 entry points for legacy imports;
3. loads the production `best.pt`;
4. loads and validates the production open-set gate;
5. runs one CPU prediction and open-set inference end-to-end.

It does **not** retrain the CNN and does **not** rebuild/overwrite the production gate.

After it passes, manually run:

```cmd
.\.venv\Scripts\python.exe -m streamlit run APP_CNN_V2.py
```

Verify:
- upload a supported fruit;
- switch to a different image before analyzing and confirm the old result is not shown;
- analyze the new image;
- upload an unsupported image;
- test camera capture.

Only after both automated and manual checks pass should `Dat` be merged into `main`.
