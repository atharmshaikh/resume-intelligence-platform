"""
Streamlit Demo
Resume Intelligence Platform

Demonstrates ML candidate ranking.
"""

from pathlib import Path
import streamlit as st

from ml_engine.ml.inference.predictor import ResumePredictor
from ml_engine.ml.feature_store.feature_schema import FEATURE_SCHEMA


# --------------------------------------------------
# Load model
# --------------------------------------------------

MODEL_PATH = Path("ml_engine/ml/artifacts/resume_model.pkl")

predictor = ResumePredictor(MODEL_PATH)


# --------------------------------------------------
# UI
# --------------------------------------------------

st.set_page_config(
    page_title="Resume Intelligence Platform",
    layout="centered"
)

st.title("Resume Intelligence Platform")
st.subheader("AI Candidate Ranking Demo")

st.write(
    "Adjust resume attributes and predict candidate strength."
)


# --------------------------------------------------
# Feature Inputs
# --------------------------------------------------

features = FEATURE_SCHEMA.default_row()

st.header("Resume Features")

features["skills_count"] = st.slider(
    "Skills Count", 0, 20, 8
)

features["projects_count"] = st.slider(
    "Projects Count", 0, 10, 2
)

features["resume_word_count"] = st.slider(
    "Resume Word Count", 100, 1000, 400
)

features["experience_years_estimate"] = st.slider(
    "Experience Years", 0.0, 10.0, 1.0
)

features["programming_languages_count"] = st.slider(
    "Programming Languages", 0, 10, 3
)

features["framework_count"] = st.slider(
    "Frameworks", 0, 10, 2
)

features["database_count"] = st.slider(
    "Databases", 0, 5, 1
)

features["tool_count"] = st.slider(
    "Developer Tools", 0, 10, 3
)

features["projects_count"] = st.slider(
    "Projects Built", 0, 10, 2
)

features["achievement_count"] = st.slider(
    "Achievements", 0, 5, 1
)


# --------------------------------------------------
# Predict Button
# --------------------------------------------------

if st.button("Predict Candidate Quality"):

    try:

        prediction = predictor.predict(features)

        st.divider()

        st.header("Prediction Result")

        if prediction == 2:
            st.success("Strong Candidate")

        elif prediction == 1:
            st.warning("Average Candidate")

        else:
            st.error("Weak Candidate")

        st.subheader("Feature Snapshot")
        st.json(features)

    except Exception as e:

        st.error(str(e))