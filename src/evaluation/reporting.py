"""Generate corrected paper tables, summaries, figures, and archives."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from src.training.result_store import read_current_results


SUMMARY_METRICS = (
    "recording_macro_f1",
    "recording_balanced_acc",
    "recording_early_fault_recall",
    "recording_auroc",
    "recording_auprc",
    "recording_mcc",
)


def summarize_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize corrected results without pooling away fold structure."""
    completed = dataframe[dataframe["status"] == "COMPLETED"].copy()
    groups = [
        "paper_experiment",
        "protocol",
        "dataset",
        "model",
        "configuration",
        "label_budget",
    ]
    rows = []
    for keys, group in completed.groupby(groups, dropna=False):
        row = dict(zip(groups, keys))
        row["n_folds"] = group["fold_id"].nunique()
        row["n_seeds"] = group["seed"].nunique()
        row["n_runs"] = len(group)
        for metric in SUMMARY_METRICS:
            if metric not in group:
                continue
            numeric = group.assign(
                _value=pd.to_numeric(group[metric], errors="coerce")
            ).dropna(subset=["_value"])
            fold_means = numeric.groupby("fold_id")["_value"].mean()
            seed_stds = (
                numeric.groupby("fold_id")["_value"].std(ddof=1).dropna()
            )
            row[f"{metric}_mean"] = numeric["_value"].mean()
            row[f"{metric}_pooled_std"] = numeric["_value"].std(ddof=1)
            row[f"{metric}_mean_of_fold_means"] = fold_means.mean()
            row[f"{metric}_between_fold_std"] = fold_means.std(ddof=1)
            row[f"{metric}_mean_seed_std"] = seed_stds.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def write_experiment_tables(
    results_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    dataframe = read_current_results(results_path)
    summary = summarize_results(dataframe)
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings = {
        "main_results.csv": ("E1",),
        "ablation_results.csv": ("E2", "E4"),
        "generalization_results.csv": ("E3", "E5"),
        "limited_label_results.csv": ("E6",),
    }
    written = {}
    all_path = output_dir / "corrected_results_summary.csv"
    summary.to_csv(all_path, index=False, na_rep="N/A")
    written["summary"] = all_path
    for filename, experiments in mappings.items():
        path = output_dir / filename
        summary[
            summary["paper_experiment"].isin(experiments)
        ].to_csv(path, index=False, na_rep="N/A")
        written[filename] = path
    return written


def generate_summary_figures(
    results_path: Path,
    output_dir: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return []
    summary = summarize_results(read_current_results(results_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    plots = (
        ("E1", "model", "fig_e1_main_comparison.pdf"),
        ("E2", "configuration", "fig_e2_modality_ablation.pdf"),
        ("E4", "configuration", "fig_e4_curriculum_ablation.pdf"),
        ("E6", "label_budget", "fig_e6_limited_labels.pdf"),
    )
    metric = "recording_macro_f1_mean_of_fold_means"
    for experiment, x_column, filename in plots:
        subset = summary[summary["paper_experiment"] == experiment]
        if subset.empty or metric not in subset:
            continue
        figure, axis = plt.subplots(figsize=(9, 5))
        if experiment == "E6":
            sns.lineplot(
                data=subset,
                x=x_column,
                y=metric,
                hue="model",
                marker="o",
                ax=axis,
            )
        else:
            sns.barplot(data=subset, x=x_column, y=metric, ax=axis)
        axis.set_ylabel("Recording-level Macro F1")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        figures.append(path)
    return figures


def package_corrected_results(
    results_path: Path,
    tables_dir: Path,
    figures_dir: Path,
    archive_path: Path,
) -> Path:
    write_experiment_tables(results_path, tables_dir)
    generate_summary_figures(results_path, figures_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(results_path, arcname=f"tables/{results_path.name}")
        for root in (tables_dir, figures_dir):
            for path in root.glob("*"):
                if path.is_file() and path != archive_path:
                    archive.write(path, arcname=f"{root.name}/{path.name}")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default="results/tables/corrected_paper_experiments.csv"
    )
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument(
        "--archive", default="results/corrected_paper_results.zip"
    )
    args = parser.parse_args()
    archive = package_corrected_results(
        Path(args.results),
        Path(args.tables_dir),
        Path(args.figures_dir),
        Path(args.archive),
    )
    print(archive)


if __name__ == "__main__":
    main()
