"""Generate validation-selected prediction and E7 figures."""

from pathlib import Path

from src.evaluation.explainability import generate_e7_artifacts
from src.evaluation.prediction_artifacts import generate_prediction_artifacts


def main() -> None:
    results = Path("results/tables/corrected_paper_experiments.csv")
    output = Path("results/figures")
    generate_prediction_artifacts(results, output)
    generate_e7_artifacts(results, output)


if __name__ == "__main__":
    main()
