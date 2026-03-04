from pathlib import Path

from ml_engine.pipeline.resume_pipeline import ResumePipeline


def main():

    pipeline = ResumePipeline()

    # Resolve path relative to project
    base_dir = Path(__file__).resolve().parents[1]
    resume_file = base_dir / "sample_resumes" / "test_resume.pdf"

    result = pipeline.parse(resume_file)

    print("\nParsed Resume\n")
    print(result.to_dict())


if __name__ == "__main__":
    main()