from pathlib import Path
from ml_engine.pipeline.resume_pipeline import ResumePipeline


def find_resume_file(resume_dir: Path):

    for file in resume_dir.iterdir():

        if file.suffix.lower() in [".pdf", ".docx"]:
            return file

    raise FileNotFoundError("No resume file found in sample_resumes folder")


def main():

    pipeline = ResumePipeline()

    base_dir = Path(__file__).resolve().parents[1]

    resume_dir = base_dir / "sample_resumes"

    resume_file = find_resume_file(resume_dir)

    print(f"\nProcessing Resume: {resume_file.name}\n")

    result = pipeline.parse(resume_file)

    print("\nParsed Resume\n")
    print(result.to_dict())


if __name__ == "__main__":
    main()