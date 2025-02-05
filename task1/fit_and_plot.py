"""
Create plots and determine relevant biomarkers.
"""

import pandas as pd
import numpy as np
from pydantic import BaseModel
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

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
    """

    metadata = pd.read_excel(metadata_filename)
    # Create male column
    metadata["male"] = np.where(metadata["gender"] == "m", 1, 0)
    # Rename age column
    metadata = metadata.rename(columns={"age at CSF collection": "age"})
    # Create label column
    metadata["label"] = np.where(metadata["primary biochemical AD classification"] == "biochemical AD", 1, 0)

    dataset = pd.read_excel(dataset_filename, skiprows=1)
    # Rename genes column
    dataset = dataset.rename(columns={"Unnamed: 1": "Protein accessions"})

    # Remove the [n] prefixes.
    dataset.columns = dataset.columns.str.replace(r'^\[\d+\] ', '', regex=True)

    # Melt dataset to align Gene names with corresponding sample values
    dataset_melted = dataset.melt(id_vars=['Protein accessions'], var_name='sample name', value_name='value')
    # Merge with metadata
    merged_df = pd.merge(metadata, dataset_melted, on='sample name', how='left')
    # Pivot the melted df back to get a column for each Gene name, with values from corresponding sample columns
    df = merged_df.pivot_table(index=["sample name", "male", "age", "label"], columns='Protein accessions', values='value', aggfunc='first')

    # Reset index to get original columns back
    df.reset_index(inplace=True)

    # Replace Filtered with small values
    df = df.replace("Filtered", 1)

    sns.boxplot(data=df, x="label", y="A0A024QZX5;A0A087X1N8;P35237")
    plt.show()


    return df

def plot_corr():
    # Healthy vs diseased?
    df = df.replace("Filtered", np.nan)
    proteins = df.columns.tolist()[4:]

    corr = df[proteins[:10]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title('Correlation Matrix of Protein Intensities')
    plt.show()

def plot_kde():
    import seaborn as sns
    import matplotlib.pyplot as plt
    key = "A0A024QZX5;A0A087X1N8;P35237"
    x = np.log(df[key].to_numpy())
    y = df["label"].to_numpy()
    mask = x > 0
    sns.kdeplot(x[(y==0)], label="Healthy")
    sns.kdeplot(x[(y==1)], label="Diseased")
    plt.title(f'Density Plot of {key} Intensities by Disease Status')
    plt.xlim(left=max(2, x.min()))
    plt.legend()
    plt.show()

def get_feature_importance():
    from sklearn.ensemble import RandomForestClassifier

    # Train a Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    
    # Get feature importances
    feature_importances = rf_model.feature_importances_




def main(args: Arguments) -> None:
    """
    Create plots and determine relevant biomarkers.

    Args:
        args: the command-line arguments
    """
    parse_data(dataset_filename=args.dataset, metadata_filename=args.metadata)

if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
