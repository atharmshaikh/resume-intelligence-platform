"""
Integration test for ResumePipeline.

This script runs the complete resume processing pipeline
on a sample resume file and prints the structured output.

The goal is to verify that all pipeline stages operate
correctly together.
"""

import json
import sys
from pathlib import Path

from ml_engine.pipeline import ResumePipeline


def find_resume_file(resume_dir: Path) -> Path:
    """
    Locate the first supported resume file in the sample directory.
    """

    if not resume_dir.exists():
        raise FileNotFoundError(
            f"Sample resume directory not found: {resume_dir}"
        )

    if not resume_dir.is_dir():
        raise NotADirectoryError(
            f"Expected directory but found file: {resume_dir}"
        )

    supported_ext = {".pdf", ".docx"}

    for file in sorted(resume_dir.iterdir()):
        if file.is_file() and file.suffix.lower() in supported_ext:
            return file

    raise FileNotFoundError(
        "No supported resume file (.pdf or .docx) found in sample_resumes folder"
    )


def main() -> None:
    """
    Execute pipeline test.
    """

    try:

        pipeline = ResumePipeline()

        base_dir = Path(__file__).resolve().parents[1]

        resume_dir = base_dir / "sample_resumes"

        resume_file = find_resume_file(resume_dir)

        print("\n-------------------------------------")
        print(f"Processing Resume: {resume_file.name}")
        print("-------------------------------------\n")

        result = pipeline.parse(resume_file)

        print("Pipeline completed successfully.\n")

        print("Structured Resume Output:\n")

        print(
            json.dumps(
                result.to_dict(),
                indent=2,
                ensure_ascii=False
            )
        )

        print("\n-------------------------------------")
        print("Test finished successfully")
        print("-------------------------------------\n")

    except Exception as exc:

        print("\nPipeline test failed.\n")
        print(f"Error: {exc}\n")

        sys.exit(1)


if __name__ == "__main__":
    main()