"""
tests/test_dataset_generation.py
================================
Verification script for synthetic dataset generation.
"""

from ml_engine.ml.data.generator import SyntheticDatasetGenerator

def main():
    generator = SyntheticDatasetGenerator()
    # Generate 100 samples for quick verification
    generator.generate(100)
    print("✅ Successfully generated 100 samples to datasets/resume_dataset.csv")

if __name__ == "__main__":
    main()