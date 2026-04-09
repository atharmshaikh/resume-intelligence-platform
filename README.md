# 🌟 Resume Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

An industry-grade **Resume Intelligence Platform** designed to solve the "Recruiter Fatigue" problem. This system automates the transformation of unstructured resumes into high-quality hiring decisions using a hardened **18-signal "Extreme Depth" feature vector** optimized for true applicant ranking.

It bridges the gap between raw document parsing (Layout-Aware) and intelligent candidate ranking, all while maintaining a 100% offline, privacy-first data lifecycle.

---

### 🏛️ System Architecture & Flow

```mermaid
graph TD
    %% Styling
    classDef primary fill:#6750A4,stroke:#fff,stroke-width:2px,color:#fff;
    classDef secondary fill:#958DA5,stroke:#fff,stroke-width:1px,color:#fff;
    classDef highlight fill:#F7931E,stroke:#fff,stroke-width:2px,color:#fff;

    subgraph "1. Data Ingestion"
        A[<b>Upload Resumes</b><br/>PDF / DOCX]:::primary
        B[<b>Resume Engine</b><br/>Parsing & Cleaning]:::secondary
    end

    subgraph "2. Feature Engineering"
        C[<b>166-Feature Vector</b><br/>G01-G11 Metrics]:::secondary
        D[<b>Optimized Storage</b><br/>Lean JSON Layer]:::highlight
    end

    subgraph "3. Advanced ML Inference"
        E[<b>Hard Rule Filter</b><br/>Contact/Skills Guard]:::primary
        F[<b>ML Ranker</b><br/>Logistic Regression v1.1]:::secondary
    end

    subgraph "4. Intelligent Routing"
        G[<b>Shortlisted</b><br/>Score > 90]:::primary
        H[<b>Manual Review</b><br/>Score 75-90]:::secondary
        I[<b>Rejected</b><br/>Score < 75]:::highlight
    end

    A --> B --> C --> D --> E --> F
    F --> G & H & I
```

---

### 🔥 Key Innovations

#### 🛠️ **"Extreme Depth" Architectural Hardening**
The platform is built on a modular, package-based architecture modeled after top-tier MLOps patterns. 
*   **Core**: Unified metadata and versioning (v1.1.0).
*   **Engine**: Plug-and-play model registry (LR/RF/XGB).
*   **Data**: JSON-First Lifecycle—eliminating legacy CSV dependency.
*   **Parser**: Layout-aware extraction using PyMuPDF (fitz) for complex multi-column resumes.

#### 🛡️ **Zero-Trust CI/CD & Type-Safety (v1.1.1)**
The platform now enforces a strict **Zero-Trust ML Pipeline** to ensure supply-chain security and runtime stability.
*   **100% Type-Safety**: Achieved a 0-error Pyright status across the entire ML engine, eliminating technical debt and runtime logic errors.
*   **Isolated CI Build**: GitHub Actions environment is configured with `permissions: contents: read` and restricted binary execution.
*   **Dependency Hardening**: All 3rd-party ML libraries (`sklearn`, `pandas`, `numpy`) are handled via defensive, lazy-loading imports with static type markers to prevent environment-specific failures.
*   **Safe-Trigger Logic**: CI/CD runs are skipped for non-runtime changes (e.g., `.md` docs) to optimize build cycles.

---

### 💻 Tech Stack & Tools

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Machine Learning** | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![Joblib](https://img.shields.io/badge/Joblib-4B8BBE?style=flat-square) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![YAML](https://img.shields.io/badge/YAML-CB171E?style=flat-square&logo=yaml&logoColor=white) |
| **Parsing** | ![PyMuPDF](https://img.shields.io/badge/PyMuPDF--Fitz-000000?style=flat-square) ![Docx](https://img.shields.io/badge/Python--Docx-4B8BBE?style=flat-square) |
| **Static Analysis** | ![Pyright](https://img.shields.io/badge/Pyright-Type--Safe-blue?style=flat-square) |
| **CI/CD** | ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Zero--Trust-2088FF?style=flat-square&logo=github-actions&logoColor=white) |

---

### 🚀 Getting Started

#### 📦 **Installation**
The platform features automated environment management. You just need to run the setup script for your OS.

**Linux/macOS (Inference & Training):**
```bash
cd ml-service/ml-engine/
./run_batch.sh       # Run Batch Inference
./train_pipeline.sh  # Run End-to-End Training
```

**Windows (Inference & Training):**
```cmd
cd ml-service\ml-engine\
run_batch.bat        # Run Batch Inference
train_pipeline.bat   # Run End-to-End Training
```

#### 📂 **Input/Output Workflow**
1.  Drop resumes into `/data/uploads/` (Project Root).
2.  Run the batch or training script above.
3.  Check `/data/processed/` for extraction JSONs and `/data/results/` for final decisions.

---

### 🤝 References & Credits
*   **Vectorization**: Based on the 165+ feature engineering ATS schema for resume processing.
*   **Architecture**: Specialized for Low-Resource Environments (i3-sim) to High-End RTX Workstations.
*   **Inspiration**: Built for the **Resume Intelligence Platform** suite.

---

<p align="center">
  <img src="https://img.shields.io/badge/Developed_With-Passion-6750A4?style=for-the-badge" alt="Passion">
  <img src="https://img.shields.io/badge/Version-1.1.1--Zero--Trust-blue?style=for-the-badge" alt="Version">
</p>
