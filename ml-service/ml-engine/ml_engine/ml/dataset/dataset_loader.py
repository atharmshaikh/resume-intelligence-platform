"""
Dataset Loader

Loads CSV datasets for training and evaluation.
"""

import pandas as pd


class DatasetLoaderError(Exception):
    pass


class DatasetLoader:

    def load(self, dataset_path: str):

        try:

            df = pd.read_csv(dataset_path)

        except Exception as e:
            raise DatasetLoaderError(f"Failed to load dataset: {e}")

        if "label" not in df.columns:
            raise DatasetLoaderError("Dataset must contain 'label' column")

        X = df.drop(columns=["label"])
        y = df["label"]

        return X, y


DATASET_LOADER = DatasetLoader()