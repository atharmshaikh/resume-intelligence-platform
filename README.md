# 🌟 Resume Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

### 🚀 Overview
An industry-grade **Resume Intelligence Platform** designed to solve the "Recruiter Fatigue" problem. This system automates the transformation of unstructured resumes into high-quality hiring decisions using a 166-feature vectorization pipeline.

It bridges the gap between raw document parsing and intelligent candidate ranking, all while maintaining a 100% offline, privacy-first data lifecycle.

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
        F[<b>ML Ranker</b><br/>Random Forest V1.0]:::secondary
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

#### 🛠️ **Industry-Grade Layout Refactoring**
The platform is built on a modular, package-based architecture modeled after top-tier MLOps patterns. 
*   **Core**: Unified metadata and versioning.
*   **Engine**: Plug-and-play model registry.
*   **Data**: Resource-optimized ingestion layers.

#### 🚦 **3-Tier Routing & Hard Rejection**
*   **Tier 1 (Shortlisted)**: Direct candidate ranking for high-potential applicants.
*   **Tier 2 (Manual Review)**: Intelligent queue management with capacity-limited pruning to prevent burnout.
*   **Tier 3 (Rejected)**: Immediate feedback with explainable rejection reasons.
*   **Guardrails**: Automated rejection for missing Name, Email, or Phone to ensure database integrity.

#### 🔐 **Resource Safety & Automated Shredding**
*   **Lean Persistence**: Intermediate data excludes heavy raw text to save 90% storage space.
*   **Auto-Cleanup**: Automated 24-hour file shredding for raw uploads, ensuring strict privacy compliance.

---

### 💻 Tech Stack & Tools

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Machine Learning** | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![Joblib](https://img.shields.io/badge/Joblib-4B8BBE?style=flat-square) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![YAML](https://img.shields.io/badge/YAML-CB171E?style=flat-square&logo=yaml&logoColor=white) |
| **Parsing** | ![PDFMiner](https://img.shields.io/badge/PDFMiner.six-000000?style=flat-square) ![Docx](https://img.shields.io/badge/Python--Docx-4B8BBE?style=flat-square) |
| **Inference Framework** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |

---

### 🚀 Getting Started

#### 📦 **Installation**
The platform features automated environment management. You just need to run the setup script for your OS.

**Linux/macOS:**
```bash
cd ml-service/ml-engine/
./run_batch.sh
```

**Windows:**
```cmd
cd ml-service\ml-engine\
run_batch.bat
```

#### 📂 **Input/Output Workflow**
1.  Drop resumes into `ml_engine/data/uploads/`.
2.  Run the batch script above.
3.  Check `ml_engine/data/results/` for final hiring decisions.

---

### 🤝 References & Credits
*   **Vectorization**: Based on the 165+ feature engineering ATS schema for resume processing.
*   **Architecture**: Specialized for Low-Resource Environments (i3-sim) to High-End RTX Workstations.
*   **Inspiration**: Built for the **Resume Intelligence Platform** suite.

---

<p align="center">
  <img src="https://img.shields.io/badge/Developed_With-Passion-6750A4?style=for-the-badge" alt="Passion">
  <img src="https://img.shields.io/badge/Version-1.0.1--Stable-blue?style=for-the-badge" alt="Version">
</p>
