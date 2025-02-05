"""
Find the optimal combination of drugs below the toxicity limit.
"""

import argparse

import cvxpy
import numpy as np
import pandas as pd
from jaxtyping import Integer
from numpy.typing import NDArray
from pydantic import BaseModel


class Arguments(BaseModel):
    """
    Command-line arguments.

    Args:
        protein_data: csv-file containing protein intensities in healthy and diseased states
        drug_effects: csv-file containing the effect the drugs have on some proteins.
        allow_partial_doses: allow non-integer doses in the optimization.
    """

    protein_data: str
    drug_effects: str
    allow_partial_doses: bool


def parse_args() -> Arguments:
    """
    Parse command-line arguments as an instance of `Arguments`.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Find the optimal combination of drugs below the toxicity limit.",
    )

    parser.add_argument(
        "protein_data",
        type=str,
        help="csv-file containing protein intensities in healthy and diseased state.",
    )
    parser.add_argument(
        "drug_effects",
        type=str,
        help="csv-file containing the effect the drugs have on some proteins.",
    )
    parser.add_argument(
        "--allow_partial_doses",
        action="store_true",
        help="allow non-integer doses in the optimization.",
    )

    args = parser.parse_args()

    return Arguments(**vars(args))


def parse_data(
    args: Arguments,
) -> tuple[
    Integer[NDArray[np.int_], "drugs proteins"], Integer[NDArray[np.int_], " proteins"]
]:
    """
    Read and format the data files.

    Args:
        args: command-line arguments.

    Returns:
        drug and disease effects
    """
    disease_effects, protein_lookup = parse_disease_effects(filename=args.protein_data)
    drug_effects = parse_drug_effects(
        filename=args.drug_effects, protein_lookup=protein_lookup
    )

    return drug_effects, disease_effects


def parse_disease_effects(
    filename: str,
) -> tuple[Integer[NDArray[np.int_], " proteins"], dict[str, int]]:
    """
    Create the disease effect vector from the input csv file.

    Args:
        filename: the location of the csv file.

    Returns:
        The disease effect vector together with a mapping of protein name to index.
    """
    df = pd.read_csv(filename)

    disease_effects = (df["Diseased"] - df["Healthy"]).to_numpy()
    protein_lookup = {key: i for i, key in enumerate(df["Protein"])}

    return disease_effects, protein_lookup


def parse_drug_effects(
    filename: str, protein_lookup: dict[str, int]
) -> Integer[NDArray[np.int_], "drugs proteins"]:
    """
    Create the drug effect matrix from the input csv file.

    Args:
        filename: the location of the csv file.
        protein_lookup: protein to index lookup

    Returns:
        The parsed matrix
    """
    df = pd.read_csv(filename)

    drug_lookup = {f"Drug{i + 1}": i for i in range(10)}

    # Convert names to indices
    df["Drug_idx"] = df["Drug"].map(drug_lookup)
    df["Protein_idx"] = df["Protein"].map(protein_lookup)

    drug_effects = np.zeros((len(drug_lookup), len(protein_lookup)), dtype=int)
    drug_effects[df["Drug_idx"], df["Protein_idx"]] = df["Effect"]

    return drug_effects


def main(args: Arguments) -> None:
    """
    Find the optimal combination of drugs over a range of toxicity-limits.

    Args:
        args: The command-line arguments
    """
    drug_effects, disease_effects = parse_data(args)
    toxicity = np.arange(10) + 1

    solve(
        drug_effects=drug_effects,
        disease_effects=disease_effects,
        toxicity=toxicity,
        toxicity_limit=10,
        allow_partial_doses=args.allow_partial_doses,
    )


def solve(
    drug_effects: Integer[NDArray[np.int_], "drugs proteins"],
    disease_effects: Integer[NDArray[np.int_], " proteins"],
    toxicity: Integer[NDArray[np.int_], " drugs"],
    toxicity_limit: int,
    allow_partial_doses: bool = False,
) -> None:
    """
    Solve the constrained mixed-integer problem.

    Args:
        drug_effects: the effect each drug has on proteins.
        disease_effects: the difference between disease and healthy.
        toxicity: the toxicity per dose of drug.
        toxicity_limit: the upper bound on toxicity
        allow_partial_doses: allow non-integer doses.
    """
    n_drugs, n_proteins = drug_effects.shape
    doses = cvxpy.Variable(n_drugs, integer=not allow_partial_doses)

    objective = cvxpy.Minimize(
        cvxpy.sum_squares(doses @ drug_effects + disease_effects)  # type: ignore[no-untyped-call]
    )

    # Positive constraint specified in the variable definition
    constraints = [toxicity @ doses <= toxicity_limit, doses >= 0]
    problem = cvxpy.Problem(objective, constraints)
    result = problem.solve(solver=cvxpy.ECOS if allow_partial_doses else cvxpy.SCIP)  # type: ignore[no-untyped-call]

    print("Optimal objective value:", result)
    print(
        "Optimal doses:",
        np.asarray(doses.value, dtype=float if allow_partial_doses else int),
    )


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
