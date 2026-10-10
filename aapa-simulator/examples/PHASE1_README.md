# AAPA Phase 1 — repaired reproducibility workflow

This is a **replacement for `examples/run_classification.py`**, based on the supplied source bundle. Keep your original file and Git diff. This is a modified reproduction pipeline, not an untouched run of the author's code.

## What changed

1. Requires all 14 daily invocation CSVs and preserves the correct day position of each function. Missing function-days become zero-filled instead of shifting subsequent days.
2. Filters HTTP triggers explicitly. Duplicate rows for a function/day are summed. Missing numeric values become zero; negative counts raise errors.
3. Applies >=1000 invocations and >=10% active-minute filtering over the complete 14-day trace (these eligibility rules must be checked against the paper).
4. Generates 60-minute windows every 10 minutes **within each day**. This differs from the original cross-day windowing at midnight; the change avoids windows crossing split boundaries. Record this deviation in the report.
5. Uses zero-based chronological days: 0–8 train, 9–10 validation, 11–13 test.
6. Saves each function's features and weak labels to resumable `.pkl` checkpoints, using atomic file replacement.
7. Checks label coverage and numerical feature validity before training.
8. Saves metrics JSON, figures, the original `lightgbm_model.pkl`, and a `lightgbm_classifier.pkl` compatibility copy. **This does not fix the Phase 2 simulator APIs.**

## Install

From the repository root in PowerShell:

```powershell
Copy-Item examples/run_classification.py examples/run_classification.original.py
Copy-Item .\run_classification.py examples/run_classification.py
python -m pip install -r requirements.txt
python -m pip install xgboost
```

The second command assumes you downloaded this file into the repository root. Alternatively copy the downloaded file directly to `examples/run_classification.py`.

## Run

```powershell
# Fast end-to-end verification; separate output and checkpoints
python examples/run_classification.py --smoke-test

# Medium verification (recommended before full experiment)
python examples/run_classification.py --max-functions 100 --checkpoint-dir data/processed/phase1_100 --output-dir models/phase1_100

# Full run — high RAM/disk demand; only after successful smoke and medium tests
python examples/run_classification.py

# Resume after interruption, with same parameters
python examples/run_classification.py --resume
```

**Important:** A full 14-day trace generates 139 windows/day/function, or **1,946 windows per function**. At 18,988 functions this is ~36.95 million windows before filtering. Loading the full labeled feature table into RAM may exhaust a 16GB machine. Per-function checkpoints prevent re-extraction but **do not yet make full training out-of-core**. Prefer a carefully documented subset until an out-of-core training approach is implemented. Processing raw CSVs for all 14 days is still required in the current loader, even for smoke mode.

## Caveats

- The original `TimeSeriesFeatureExtractor` may generate non-finite values on constant windows; the replacement sanitizes these before fitting, but feature computation still takes time.
- Labels come from heuristic weak supervision, **not independently annotated ground truth**. Classification accuracy is agreement with these heuristic labels and should not be represented as independent workload truth.
- The classifier's feature vector must be generated identically in Phase 2; that path is not yet verified.
- No full Azure dataset or trained model was available here, so end-to-end training has **not** been executed or validated. Only syntax compilation of the replacement script was performed.
- The source `src/models/classifier.py` imports XGBoost unconditionally. Installing `xgboost` avoids an import error even though this script trains LightGBM only.
- The source `save_models()` uses Python pickle, whereas Phase 2 uses joblib to load. Usually compatible for basic pickle, but Phase 2 still needs its own interface repairs.
