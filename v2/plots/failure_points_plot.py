"""
Generates scatter plots to visualize solver failure points.

This script queries a benchmark database to identify interior-point method (IPM)
iterations where the solver's residual norm exceeded the PCG tolerance. It then
creates a grid of scatter plots, one for each solver, showing these failure points
as a function of the problem size (number of edges).
"""

import sqlite3
import polars as pl
import matplotlib.pyplot as plt
import argparse
import re  # For regex matching patterns
import numpy as np  # For unique colors


def plot_failure_points(db_path, problem_pattern=None, solver_pattern=None):
    """
    Queries the database and plots IPM iterations where residual norm exceeds tolerance.

    Args:
        db_path (str): Path to the SQLite database file.
        problem_pattern (str, optional): A regex pattern to filter problem names.
            Defaults to None.
        solver_pattern (str, optional): A regex pattern to filter solver names.
            Defaults to None.
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        p.name AS problem_name,
        p.num_edges,
        r.solver_name,
        sh.ipm_iter,
        sh.residual_norm,
        c.pcg_tol
    FROM
        solver_history sh
    JOIN
        runs r ON sh.run_id = r.id
    JOIN
        problems p ON r.problem_id = p.id
    JOIN
        configs c ON r.config_id = c.id
    WHERE
        r.status = 'Trm_Optimal'
    ORDER BY
        p.name, r.solver_name, sh.ipm_iter
    """
    df = pl.read_database(query, conn)
    conn.close()

    if df.is_empty():
        print("No data found matching the criteria.")
        return

    # Filter by patterns if provided
    if problem_pattern:
        df = df.filter(
            pl.col("problem_name").str.contains(problem_pattern, strict=False)
        )
    if solver_pattern:
        df = df.filter(pl.col("solver_name").str.contains(solver_pattern, strict=False))

    if df.is_empty():
        print(
            f"No data found after applying filters (problem_pattern='{problem_pattern}', solver_pattern='{solver_pattern}')."
        )
        return

    # Identify failures: residual_norm > pcg_tol
    df_failures = df.filter(pl.col("residual_norm") > pl.col("pcg_tol"))

    if df_failures.is_empty():
        print("No failures found after applying filters.")
        return

    unique_solvers = df_failures["solver_name"].unique().sort().to_list()
    unique_problems = df_failures["problem_name"].unique().sort().to_list()

    # Generate a color map for problems
    colors = plt.get_cmap("tab20", len(unique_problems))
    problem_color_map = {
        problem: colors(i) for i, problem in enumerate(unique_problems)
    }

    # Determine grid size for subplots (one per solver)
    num_solvers = len(unique_solvers)
    if num_solvers == 0:
        print("No solvers with failures to plot.")
        return

    cols = min(num_solvers, 2)  # Max 2 columns for solvers
    rows = (num_solvers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    for i, solver in enumerate(unique_solvers):
        ax = axes[i]
        solver_df = df_failures.filter(pl.col("solver_name") == solver)

        for problem in unique_problems:
            problem_solver_df = solver_df.filter(pl.col("problem_name") == problem)
            if not problem_solver_df.is_empty():
                ax.scatter(
                    problem_solver_df["num_edges"],
                    problem_solver_df["ipm_iter"],
                    color=problem_color_map[problem],
                    alpha=0.7,
                    s=20,
                )  # s is marker size

        ax.set_xscale("log")
        ax.set_xlabel("Number of Edges (log scale)", fontsize=9)
        ax.set_ylabel("IPM Iteration", fontsize=9)
        ax.set_title(f"Solver: {solver}", fontsize=10)
        ax.grid(True, which="both", ls="--")
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=6)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("IPM Iterations at which Residual Norm > pcg_tol", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot IPM iterations where residual norm exceeds pcg_tol, grouped by solver."
    )
    parser.add_argument("db_file", help="Path to the SQLite database file.")
    parser.add_argument(
        "--problem-pattern",
        help="Regex pattern to filter problem names (e.g., 'grid_wide_08*').",
    )
    parser.add_argument(
        "--solver-pattern",
        help="Regex pattern to filter solver names (e.g., 'tulip_*').",
    )
    args = parser.parse_args()

    plot_failure_points(args.db_file, args.problem_pattern, args.solver_pattern)
