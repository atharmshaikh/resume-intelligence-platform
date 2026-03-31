from ml_engine.ml.dataset.synthetic_dataset import SyntheticDatasetGenerator


def main():

    generator = SyntheticDatasetGenerator()

    generator.generate(500)


if __name__ == "__main__":
    main()