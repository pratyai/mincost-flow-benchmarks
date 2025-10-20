"""
Generates a line plot to visualize solver convergence failures over IPM iterations.

This script queries a benchmark database to count, for each IPM iteration, how many
problems are "failing" for each solver. A failure is defined as the residual norm
exceeding the PCG tolerance. The results are plotted as a line graph showing the
number of failing problems versus the IPM iteration number, with a separate line
for each solver.
"""

import sqlite3
import polars as pl
import matplotlib.pyplot as plt
import argparse
import re  # For regex matching patterns


def plot_convergence_failures(db_path, problem_pattern=None, solver_pattern=None):
    """
    Queries the database and plots the number of failing problems per IPM iteration.

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

    # Calculate the "failure" condition: residual_norm > pcg_tol
    df_failures = df.with_columns(
        (pl.col("residual_norm") > pl.col("pcg_tol")).alias("is_failing")
    )

    # Then, for each solver and iteration, count unique problems that are failing
    failure_counts = (
        df_failures.group_by(["solver_name", "ipm_iter"])
        .agg(
            pl.col("problem_name")
            .filter(pl.col("is_failing"))
            .n_unique()
            .alias("num_failing_problems")
        )
        .sort(["solver_name", "ipm_iter"])
    )

    plt.figure(figsize=(10, 6))

    for solver_name, group in failure_counts.group_by("solver_name"):
        plt.plot(
            group["ipm_iter"],
            group["num_failing_problems"],
            label=solver_name[0],
            marker="o",
            linestyle="-",
        )

    plt.xlabel("IPM Iteration")
    plt.ylabel(f"Number of Problems with Residual Norm > pcg_tol")
    plt.title(f"Convergence Failures (Residual Norm > pcg_tol) Across Problems")
    plt.legend(title="Solvers")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot convergence failures (residual norm above pcg_tol) from a database."
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
    # Removed --tolerance argument as it's now dynamic from pcg_tol
    args = parser.parse_args()

    plot_convergence_failures(args.db_file, args.problem_pattern, args.solver_pattern)
