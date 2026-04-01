# ML Engine Instructions

This engine parses resumes, extracts 165 ATS-focused features, and scores them using an embedded Random Forest model.

## 1. Zero-Setup Testing (The Easy Method)

If you are at the project root (`resume-intelligence-platform`) or inside `ml-engine`, just run the shell script. It auto-discovers the Python environment, clears stale cache, and tests the pipeline.

```bash
# Easiest way, from any folder:
bash ml-service/ml-engine/run_test.sh

# Or, if you are already inside ml-service/ml-engine/:
bash run_test.sh
```

## 2. Using Pytest (Without `.sh`)

You can run the tests directly using `pytest` via the virtual environment.

```bash
cd ml-service/ml-engine

# Run all tests (pipeline, training, prediction)
../../.venv/bin/python -m pytest tests/ -v

# Run just the full parsed resume pipeline tests
../../.venv/bin/python -m pytest tests/test_pipeline.py -v
```

## 3. Training the ML Model

The machine learning layer relies on a trained `RandomForestModel`. If you ever want to re-train the model (e.g., after tweaking `feature_extractor.py`), run the train pipeline directly.

```bash
cd ml-service/ml-engine

# Generates 2000 synthetic resumes, trains RF model, evaluates, and saves to artifacts/
../../.venv/bin/python -m ml_engine.ml.training.train_pipeline
```
*Note: This creates the file `ml_engine/ml/artifacts/resume_rf_model.joblib`, which the `Predictor` loads.*

## 4. How to call the Engine from Python (e.g. FastAPI / Backend)

When wiring this up to your web backend, simply use the `ResumePipeline` and `ResumePredictor` like this:

```python
from pathlib import Path
from ml_engine.pipeline.resume_pipeline import ResumePipeline
from ml_engine.ml.inference.predictor import ResumePredictor

# 1. Parse PDF / DOCX to extract raw features & scores
pipeline = ResumePipeline()
resume = pipeline.parse("path/to/resume.pdf")

# 2. Extract the features dictionary (all 165 features)
features = resume.to_dict().get("features", {})

# 3. Load ML model and predict
model_path = Path("ml_engine/ml/artifacts/resume_rf_model.joblib")
predictor = ResumePredictor(model_path)
prediction = predictor.predict(features)

print("Prediction  :", prediction["label_name"])  # e.g., "Average Candidate"
print("Confidence  :", prediction["confidence"])  # e.g., 0.81
print("Readiness % :", prediction["readiness"])   # e.g., 65.3
```
