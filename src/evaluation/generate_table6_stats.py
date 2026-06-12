"""Generate paired significance tests from fold-and-seed-level results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.training.result_store import read_current_results


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values."""
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_maximum = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running_maximum = max(running_maximum, value)
        adjusted[index] = running_maximum
    return adjusted.tolist()


def paired_comparisons(
    dataframe: pd.DataFrame,
    reference: str,
    *,
    experiment_column: str = "experiment",
    metric: str = "recording_macro_f1",
) -> pd.DataFrame:
    """Compare every experiment with a reference on identical fold/seed cells."""
    required = {experiment_column, "fold_id", "seed", metric}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required result columns: {sorted(missing)}")
    completed = dataframe.copy()
    if "status" in completed:
        completed = completed[completed["status"] == "COMPLETED"]
    reference_rows = completed[
        completed[experiment_column].astype(str) == reference
    ][["fold_id", "seed", metric]].rename(columns={metric: "reference_value"})
    if reference_rows.empty:
        raise ValueError(f"No completed reference rows found for {reference}")

    rows = []
    for experiment, group in completed.groupby(experiment_column):
        if str(experiment) == reference:
            continue
        paired = reference_rows.merge(
            group[["fold_id", "seed", metric]].rename(
                columns={metric: "comparison_value"}
            ),
            on=["fold_id", "seed"],
            how="inner",
            validate="one_to_one",
        ).dropna()
        if len(paired) < 2:
            continue
        differences = paired["reference_value"] - paired["comparison_value"]
        if np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(
                paired["reference_value"],
                paired["comparison_value"],
                alternative="two-sided",
                zero_method="wilcox",
            )
        rows.append(
            {
                "reference": reference,
                "comparison": experiment,
                "metric": metric,
                "n_pairs": len(paired),
                "reference_mean": paired["reference_value"].mean(),
                "comparison_mean": paired["comparison_value"].mean(),
                "mean_paired_difference": differences.mean(),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["p_value_holm"] = holm_adjust(output["p_value"].tolist())
        output["significant_0p05"] = output["p_value_holm"] < 0.05
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/tables/corrected_paper_experiments.csv",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--experiment-column", default="model")
    parser.add_argument("--metric", default="recording_macro_f1")
    parser.add_argument(
        "--output", default="results/tables/significance_results.csv"
    )
    args = parser.parse_args()

    results = paired_comparisons(
        read_current_results(Path(args.input)),
        args.reference,
        experiment_column=args.experiment_column,
        metric=args.metric,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Wrote {len(results)} paired comparisons to {output_path}")


if __name__ == "__main__":
    main()
