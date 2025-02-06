"""
Create plots and determine relevant biomarkers.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold

sns.set_style("whitegrid", {"grid.color": ".92", "axes.edgecolor": "0.92"})


class Arguments(BaseModel):
    """
    Command-line arguments.

    Args:
        dataset: xlsx-file containing protein intensities
        metadata: xlsx-file containing the metadata and labels
    """

    dataset: str
    metadata: str


def parse_args() -> Arguments:
    """
    Parse command-line arguments as an instance of `Arguments`.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Create plots and determine relevant biomarkers.",
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="xlsx-file containing protein intensities.",
    )
    parser.add_argument(
        "metadata",
        type=str,
        help="xlsx-file containing the metadata and labels.",
    )

    args = parser.parse_args()

    return Arguments(**vars(args))


def parse_data(dataset_filename: str, metadata_filename: str) -> pd.DataFrame:
    """
    Parse the datasets into a single DataFrame.

    Args:
        dataset_filename: the location of the xlsx file containing the dataset
        metadata_filename: the location of the xlsx file containing the metadata

    Returns:
        Parsed DataFrame
    """
    metadata = pd.read_excel(metadata_filename)
    # Create male column
    metadata["male"] = np.where(metadata["gender"] == "m", 1, 0)
    # Rename age column
    metadata = metadata.rename(columns={"age at CSF collection": "age"})
    # Create label column
    metadata["label"] = np.where(
        metadata["primary biochemical AD classification"] == "biochemical AD", 1, 0
    )

    dataset = pd.read_excel(dataset_filename, skiprows=1)
    # Rename genes column
    dataset = dataset.rename(columns={"Unnamed: 1": "Protein accessions"})

    # Remove the [n] prefixes.
    dataset.columns = dataset.columns.str.replace(r"^\[\d+\] ", "", regex=True)

    # Melt dataset to align Gene names with corresponding sample values
    dataset_melted = dataset.melt(
        id_vars=["Protein accessions"], var_name="sample name", value_name="value"
    )
    # Merge with metadata
    merged_df = pd.merge(metadata, dataset_melted, on="sample name", how="left")
    # Pivot the melted df back to get a column for each Gene name, with values from corresponding sample columns
    df = merged_df.pivot_table(
        index=["sample name", "male", "age", "label"],
        columns="Protein accessions",
        values="value",
        aggfunc="first",
    )

    # Reset index to get original columns back
    df.reset_index(inplace=True)

    return df


def make_kde_plots(data: pd.DataFrame, subset: list[str]) -> None:
    """
    Make KDE plots comparing healthy and diseased states for each entry.

    Args:
        data: the dataset
        subset: the feature subset to plot
    """
    # Set Filtered to some small value
    df = data.replace("Filtered", 1)
    for key in subset:
        short_name = key.split(";")[0]
        # log transform
        x = np.log(df[key].to_numpy())
        y = df["label"].to_numpy()
        sns.kdeplot(x[(y == 0)], label="Healthy")
        sns.kdeplot(x[(y == 1)], label="Diseased")
        plt.title(f"Density Plot of {short_name} Intensities by Disease Status")
        plt.xlabel("Log-intensity")
        plt.xlim(left=max(4, x.min()))
        plt.legend()
        plt.savefig(f"{short_name}.png", dpi=600, bbox_inches="tight")
        plt.clf()


def get_feature_importance(data: pd.DataFrame) -> list[str]:
    """
    Train a random forrest model to gain insight into feature importance.

    Args:
        data: the dataset

    Returns:
        The 10 most important features based on the feature importance analysis.
    """
    df = data.replace("Filtered", 1)
    features = df.columns.tolist()[4:] + ["age", "male"]
    X = df[features].to_numpy()
    # log transform
    X[:, :-1] = np.log(X[:, :-1])
    y = df["label"].to_numpy()
    # Do light cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(
        max_features=None, bootstrap=True, criterion="entropy"
    )
    param_grid = {"n_estimators": [100, 300, 1000]}
    grid = GridSearchCV(model, param_grid, cv=cv, verbose=1, refit=True)
    grid.fit(X, y)
    print("Best parameters:", grid.best_params_)
    print("Best score:", grid.best_score_)

    # Get feature importances
    feature_importances = grid.best_estimator_.feature_importances_
    idx = np.argsort(feature_importances)[::-1]

    return [features[i] for i in idx[:10]]


def main(args: Arguments) -> None:
    """
    Create plots and determine relevant biomarkers.

    Args:
        args: the command-line arguments
    """
    df = parse_data(dataset_filename=args.dataset, metadata_filename=args.metadata)
    subset = get_feature_importance(data=df)
    make_kde_plots(data=df, subset=subset)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
