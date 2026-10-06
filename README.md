# FreshLens CNN V2

FreshLens is an image-classification project for recognizing the **fruit type** and the visible **fresh / rotten condition** of one main fruit in a photo.

The current production pipeline is the modular **CNN V2** implementation based on **EfficientNet-B0 + PyTorch**. The older SVM and step-by-step development folders are retained only for reference, audit, and behavior-equivalence testing.

## 1. Supported scope

Production V2 supports 4 fruit types and 2 visible conditions:

| Fruit | Conditions |
| --- | --- |
| apple | fresh, rotten |
| banana | fresh, rotten |
| orange | fresh, rotten |
| tomato | fresh, rotten |

This gives 8 joint classes:

```text
apple::fresh
apple::rotten
banana::fresh
banana::rotten
orange::fresh
orange::rotten
tomato::fresh
tomato::rotten
```

FreshLens analyzes visible image cues only. It is **not** a food-safety, laboratory, or edibility certification system.

## 2. Production architecture

```text
Image upload / camera
        |
        v
EXIF correction + RGB conversion
        |
        v
White letterbox to 224 x 224
        |
        v
Tensor + ImageNet normalization
        |
        v
EfficientNet-B0
        |
        v
8-class softmax
        |
        v
Fruit-first decoder
        |
        +----------------------+
        |                      |
        v                      v
Fruit prediction        Condition within fruit
        |
        v
Open-set support gate
        |
    +---+---+
    |       |
    v       v
supported  unsupported
```

The deployed decoder is intentionally **fruit-first**:

1. reshape the 8 joint probabilities into `4 fruits x 2 conditions`;
2. sum the two condition probabilities for each fruit;
3. select the fruit with the highest marginal probability;
4. select `fresh` or `rotten` only inside that selected fruit.

Do not replace this rule with raw 8-class `argmax` unless a new experiment is explicitly designed and validated.

## 3. Repository structure

```text
FreshLens/
|
|-- freshlens_ai/                  # Production modular package
|   |-- data/                      # Dataset loading, transforms, validation
|   |-- models/                    # EfficientNet-B0, checkpoint handling
|   |-- training/                  # Training engine
|   |-- evaluation/                # Metrics and evaluation
|   |-- inference/                 # Prediction, open-set gate, app state
|   |-- utils/                     # Shared helpers
|   |-- constants.py
|   `-- errors.py
|
|-- TRAIN_CNN_V2.py                # Production training entry point
|-- EVALUATE_CNN_V2.py             # Production evaluation entry point
|-- PREDICT_CNN_V2.py              # Single-image prediction CLI
|-- BUILD_OPENSET_GATE_V2.py       # Open-set gate builder
|-- APP_CNN_V2.py                  # Streamlit production app
|-- RUN_APP_V2.cmd
|-- run_app.bat                    # Convenience launcher -> APP_CNN_V2.py
|-- run_train.bat                  # Convenience launcher -> TRAIN_CNN_V2.py
|
|-- models/
|   `-- cnn_efficientnet_b0/
|       |-- best.pt
|       |-- open_set_gate.npz
|       `-- open_set_gate.json
|
|-- data/                           # Local dataset artifacts; ignored where appropriate
|-- reports/                        # Generated reports; local/ignored
|-- docs/
|-- scripts/
|
|-- tests/
|   |-- integration/
|   |   `-- TEST_FULL_PIPELINE_V2.py
|   `-- refactor/
|       `-- TEST_*_REFACTOR.py
|
|-- requirements.txt
|-- requirements-torch-cu128.txt
|-- requirements-db.txt
|-- requirements-legacy.txt
`-- pyproject.toml
```

Historical implementations are archived under `legacy/`. See [`docs/LEGACY.md`](docs/LEGACY.md) before using them.

## 4. Environment

Minimum project metadata requires Python 3.10+. The current validated Windows development environment uses:

```text
Python       3.13.7
PyTorch      2.10.0+cu128
torchvision  0.25.0+cu128
CUDA build   12.8
```

An NVIDIA GPU is optional for inference. CUDA is strongly recommended for training.

### Create a virtual environment

From the repository root:

```cmd
py -m venv .venv
```

### Install shared dependencies

```cmd
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Install the validated CUDA 12.8 PyTorch build

```cmd
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cu128.txt
```

For a different CUDA version or CPU-only installation, install the matching PyTorch build instead of blindly using the CUDA 12.8 file.

Optional MongoDB / Cloudinary dependencies:

```cmd
.\.venv\Scripts\python.exe -m pip install -r requirements-db.txt
```

Legacy SVM dependencies are separated into:

```cmd
.\.venv\Scripts\python.exe -m pip install -r requirements-legacy.txt
```

They are not required by the production CNN V2 runtime.

## 5. Run the application

The easiest Windows launcher is:

```cmd
run_app.bat
```

Equivalent direct command:

```cmd
.\.venv\Scripts\python.exe -m streamlit run APP_CNN_V2.py
```

The app supports:

- image upload;
- browser camera capture;
- fruit prediction;
- fresh / rotten prediction;
- open-set support decision;
- protection against showing a stale result after the selected image changes.

Production artifacts expected by the app:

```text
models/cnn_efficientnet_b0/best.pt
models/cnn_efficientnet_b0/open_set_gate.npz
models/cnn_efficientnet_b0/open_set_gate.json
```

Do not rebuild or replace the production open-set gate casually. The gate must match the checkpoint metadata.

## 6. Predict one image from the command line

```cmd
.\.venv\Scripts\python.exe PREDICT_CNN_V2.py --image "D:\path\to\fruit.jpg"
```

Choose CUDA explicitly when desired:

```cmd
.\.venv\Scripts\python.exe PREDICT_CNN_V2.py --image "D:\path\to\fruit.jpg" --device cuda
```

The command prints a JSON result containing fruit, condition, fruit probability, conditional condition probability, and joint-class scores.

## 7. Training

Inspect all training options first:

```cmd
run_train.bat --help
```

or:

```cmd
.\.venv\Scripts\python.exe TRAIN_CNN_V2.py --help
```

A training run requires the original image root and the locked dataset metadata path. Example:

```cmd
run_train.bat ^
  --root "E:\path\to\original_dataset" ^
  --data "data\cnn_dataset_v3" ^
  --output "models\cnn_efficientnet_b0_candidate" ^
  --device cuda
```

Important training characteristics of the validated pipeline:

- EfficientNet-B0 with ImageNet pretrained weights by default;
- two stages: classifier warm-up, then fine-tuning;
- AdamW;
- label smoothing;
- mixed precision on CUDA;
- gradient accumulation and clipping;
- cosine learning-rate schedule;
- early stopping;
- deterministic seed control;
- best checkpoint selected by joint macro-F1 with validation loss as tie-breaker;
- resumable training state.

Use `--from-scratch` only for controlled testing / ablation.

## 8. Evaluation

Inspect the evaluator arguments:

```cmd
.\.venv\Scripts\python.exe EVALUATE_CNN_V2.py --help
```

The production model should be evaluated with the locked dataset metadata and the intended checkpoint. Keep train, validation, test, external calibration, and any future untouched test data conceptually separate.

Previously observed results for the current production checkpoint include approximately:

```text
Best validation joint accuracy : 99.16%
Best validation macro-F1       : 99.13%
Internal test joint accuracy   : 98.37%

External V2 known images:
fruit accuracy                 : 97.53%
condition accuracy             : 93.83%
joint accuracy                 : 92.59%
macro-F1                       : 92.45%
```

The External V2 set was later used for open-set calibration, so it must no longer be described as a future untouched final test set.

## 9. Open-set gate

The current production gate is metadata version:

```text
1.1-external-calibration
```

It uses six features derived from classifier confidence and feature-space prototypes:

```text
fruit_score
fruit_margin
joint_max
prototype_similarity
prototype_gap
fruit_certainty
```

Current production gate artifacts use 4 prototypes per supported fruit.

To inspect the builder options:

```cmd
.\.venv\Scripts\python.exe BUILD_OPENSET_GATE_V2.py --help
```

Rebuilding the gate changes the calibrated artifact. Do it only as a deliberate experiment with a documented calibration dataset.

## 10. Tests

The modular refactor has dedicated equivalence tests for data, model construction, training, evaluation, orchestration, inference, open-set behavior, predictor CLI, and Streamlit app state.

Run individual tests, for example:

```cmd
.\.venv\Scripts\python.exe -m tests.refactor.TEST_INFERENCE_REFACTOR
.\.venv\Scripts\python.exe -m tests.refactor.TEST_APP_REFACTOR
```

Run the complete integration suite:

```cmd
.\.venv\Scripts\python.exe -m tests.integration.TEST_FULL_PIPELINE_V2 ^
  --root "E:\path\to\original_dataset" ^
  --data "data\cnn_dataset_v3"
```

The full suite additionally audits modular import boundaries and performs a production checkpoint + open-set gate smoke test.

Passing the software tests verifies implementation consistency. It does **not** by itself prove generalization to every real-world camera/domain condition.

## 11. Dataset notes

The locked CNN dataset currently contains 5,869 records:

```text
train       3,519
validation  1,187
test        1,163
```

The split is fixed with seed 42. Dataset images themselves are not intended to be committed to Git.

A known limitation is that physical-specimen independence is not formally confirmed for the current locked dataset. This should be stated when reporting experimental validity.

## 12. Production vs legacy

Use these for normal CNN V2 work:

```text
freshlens_ai/
TRAIN_CNN_V2.py
EVALUATE_CNN_V2.py
PREDICT_CNN_V2.py
BUILD_OPENSET_GATE_V2.py
APP_CNN_V2.py
run_app.bat
run_train.bat
```

Do **not** use `legacy/svm/` as the production CNN implementation. It contains the archived SVM / handcrafted-feature baseline.

The `legacy/development/` tree preserves historical data-preparation stages, external-test tooling, and the pre-refactor CNN/open-set implementations. Refactor-equivalence tests intentionally compare V2 behavior against selected archived files, so the archive should remain read-only unless those tests are intentionally redesigned.

## 13. Security and repository hygiene

Do not commit:

```text
.env
.venv/
__pycache__/
.cache/
dataset_cache/
logs/
artifacts/
backups/
local dataset images
scripts/test_cloudinary.py
```

`.env.example` is intentionally kept as a safe template.

## 14. Current limitations

- Supported fruit scope is limited to apple, banana, orange, and tomato.
- The system assumes one main fruit per image.
- Fresh / rotten refers to visible appearance, not food safety.
- Open-set rejection is a calibrated heuristic and cannot guarantee rejection of every unknown image.
- Domain shift remains a major risk: phone/camera/background/lighting conditions can reduce accuracy compared with internal data.
- The current external V2 set is calibration data after gate construction, not an untouched final benchmark.

## 15. Recommended next development stages

After the repository cleanup is complete, future work should be done as new controlled experiments rather than silently changing the production baseline:

1. collect stronger real-camera and hard-case data;
2. add a true `other / unknown` strategy if the project scope requires it;
3. preserve group-aware dataset splitting;
4. compare candidate models against the frozen V2 baseline;
5. reserve a new untouched final test set;
6. add explainability / quality checks only after validating that they improve the intended behavior.

---

**Production baseline:** EfficientNet-B0 CNN V2 + fruit-first decoding + calibrated open-set support gate.
