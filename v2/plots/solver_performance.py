"""
Generates scatter plots to visualize solver performance.

This script queries a benchmark database for completed runs and generates two
scatter plots on a log-log scale:
1. Solver time vs. the number of edges in the problem.
2. Number of IPM iterations vs. the number of edges in the problem.

Each solver is represented by a different color in the plots.
"""

import sqlite3
import polars as pl
import matplotlib.pyplot as plt
import argparse


def plot_solver_performance(db_path):
    """
    Queries the database and plots solver performance metrics.

    Args:
        db_path (str): Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        p.num_edges,
        r.time_s,
        r.iters,
        r.solver_name
    FROM
        runs r
    JOIN
        problems p ON r.problem_id = p.id
    WHERE
        r.status = 'Trm_Optimal'
    """
    df = pl.read_database(query, conn)
    conn.close()

    print("DataFrame head:\n", df.head())  # Debugging line

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))  # 2 rows, 1 column

    # Plot 1: Solver Performance (Time vs. Number of Edges)
    for solver_key, group in df.group_by("solver_name"):
        solver_name = solver_key[0]
        axes[0].scatter(
            group["num_edges"], group["time_s"], label=solver_name, alpha=0.7
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of Edges (log scale)")
    axes[0].set_ylabel("Time (s) (log scale)")
    axes[0].set_title("Solver Performance: Time vs. Number of Edges")
    axes[0].legend()
    axes[0].grid(True, which="both", ls="--")

    # Plot 2: Iteration Count (Iterations vs. Number of Edges)
    for solver_key, group in df.group_by("solver_name"):
        solver_name = solver_key[0]
        axes[1].scatter(
            group["num_edges"], group["iters"], label=solver_name, alpha=0.7
        )

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Number of Edges (log scale)")
    axes[1].set_ylabel("Iterations (log scale)")
    axes[1].set_title("Solver Performance: Iterations vs. Number of Edges")
    axes[1].legend()
    axes[1].grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot solver performance from a database."
    )
    parser.add_argument("db_file", help="Path to the SQLite database file.")
    args = parser.parse_args()

    plot_solver_performance(args.db_file)
