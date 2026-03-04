"""
Basic test runner for resume pipeline.
"""

from ml_engine.pipeline.resume_pipeline import ResumePipeline


def main():

    pipeline = ResumePipeline()

    file_path = "sample_resumes/test_resume.pdf"

    result = pipeline.parse(file_path)

    print("\nParsed Resume\n")

    print(result.to_dict())


if __name__ == "__main__":
    main()