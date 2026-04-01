"""
Dataset Writer

Handles safe CSV writing.
"""

import csv
from pathlib import Path
from typing import List, Dict


class DatasetWriterError(Exception):
    pass


class DatasetWriter:

    def write_csv(self, rows: List[Dict], output_path: str):

        if not rows:
            raise DatasetWriterError("Dataset rows are empty")

        path = Path(output_path)

        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(rows[0].keys())

        with path.open("w", newline="") as f:

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()

            for row in rows:
                writer.writerow(row)


DATASET_WRITER = DatasetWriter()