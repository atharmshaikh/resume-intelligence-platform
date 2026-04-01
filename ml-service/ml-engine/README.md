# 🤖 Resume Intelligence Platform - ML Engine

An industry-grade, configuration-driven machine learning engine for automated resume screening and candidate ranking. This module transforms raw resumes into actionable hiring decisions using a robust 166-feature extraction pipeline and a versioned Random Forest model.

## 🚀 Key Features
- **Industry-Grade Architecture**: Modular, package-based Python structure (Core, Data, Engine, Inference, Pipelines).
- **3-Tier Decision Routing**: Intelligent classification into **Shortlisted**, **Manual Review**, or **Rejected** based on configurable thresholds.
- **Hard-Rejection Guardrails**: Automated rejection for candidates missing mandatory contact info or domain-specific skills.
- **166-Feature Vectorization**: Deep analysis across 11 categories (G01-G11) including ATS quality, section detection, and readiness scores.
- **Privacy-First Data Flow**: Optimized intermediate data storage (no raw text persistence) and automated 24-hour file shredding for raw uploads.
- **Hardware-Agnostic Performance**: Iterator-based batch processing designed to run smoothly on anything from an i3 3rd Gen to high-end workstations.

## 📁 Standardized Directory Structure
The engine uses a tiered storage system to maintain a clean workspace:
- `data/uploads/`: Raw `.pdf` / `.docx` resumes (Temporary, auto-deleted after 24h).
- `data/processed/`: Optimized Feature JSONs (ML-Ready, no raw text).
- `data/results/`: Final Decision JSONs (Decision, Score, Rankings).
- `ml_engine/ml/artifacts/`: Versioned model binaries and metadata.

## 🛠️ Quick Start

### 1. Environment Setup
The engine includes a cross-platform setup script that automatically manages your virtual environment and dependencies.

**On Linux/macOS:**
```bash
./run_batch.sh
```

**On Windows:**
```cmd
run_batch.bat
```

### 2. Basic Configuration
Adjust thresholds and business rules in `ml_engine/ml/configs/training_config.yaml`:
```yaml
inference_rules:
  tier_thresholds:
    shortlisted: 90
    manual_review: 75
  limits:
    max_shortlisted: 10
    max_manual_review: 20
```

## 🧠 Machine Learning Overview
The engine uses a **Model Registry** pattern, allowing for plug-and-play architecture. While it defaults to a stabilized **Random Forest** model, it is pre-configured to support **XGBoost** and other architectures. Every trained model is saved with a unique version ID and a JSON metadata sidecar for full traceability in production.

---
Developed as part of the **Resume Intelligence Platform** suite. 
**Offline-Only | Privacy-First | Industry-Grade**