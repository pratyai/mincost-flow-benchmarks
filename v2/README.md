# MinCostFlowBenchmarksV2

## Project Overview

This project provides a comprehensive benchmarking suite for Minimum Cost Flow (MCF) solvers implemented in Julia. It offers a flexible framework to:

*   Run various MCF problem instances against different solver configurations.
*   Utilize multiple floating-point precisions (`Float64`, `Float128`, `MultiFloat`) configured per solver.
*   Store detailed benchmark results in an SQLite database.
*   Visualize solver performance and convergence characteristics using Python plotting scripts.
*   Generate custom grid-based MCF problem instances.
*   Simplify benchmark execution through an optional Python-based Graphical User Interface (GUI).

## Installation

To get started with the MinCostFlowBenchmarksV2 project, follow these steps:

1.  **Install Julia:** If you don't have Julia installed, download it from [julialang.org](https://julialang.org/downloads/).
2.  **Instantiate the Julia project:** Open a Julia REPL in the project root directory and run the following commands to install the necessary Julia dependencies, including `Quadmath` for `Float128` support and `MultiFloats` for extended precision:

    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

3.  **Install Python dependencies (for GUI and plotting):**
    If you plan to use the GUI or the plotting scripts, ensure Python is installed ([python.org](https://www.python.org/downloads/)) and install `PyQt5`, `polars`, and `matplotlib`:

    ```bash
    pip install PyQt5 polars matplotlib
    ```
    *(Note: `polars` is used for efficient data handling, `matplotlib` for plotting.)*

## Usage

The benchmark suite can be run via a command-line interface (CLI) using the main Julia script or through a user-friendly Python GUI.

### Command-Line Interface (CLI)

The primary entry point for running benchmarks is `src/main.jl`.

```bash
julia --project=. src/main.jl -i <input_spec_path...> -c <config_path...> [-o <output_db_path>] [-s <solution_dir>]
```

**Arguments:**

*   `-i <path...>`: **(Required)** Path(s) to one or more input specification files. These are CSV files listing the problem instances to be run.
    *   *Example:* `data/specs/warmup.inspec`
*   `-c <path...>`: **(Required)** Path(s) to one or more solver configuration files (TOML format). These files define the solver to use and its specific parameters.
    *   *Example:* `configs/cholmod-f64.toml`
*   `-o <path>`: (Optional) Path to the output SQLite database file. If omitted, the Julia script will automatically create a database named `spec_name.db` for each input spec file in the current directory.
*   `-s <path>`: (Optional) Path to a directory where solution flow vectors will be stored (in JLD2 format) for each problem run.

**Example Commands:**

*   **Run the warmup spec with a single solver configuration:**

    ```bash
    julia --project=. src/main.jl -i data/specs/warmup.inspec -c configs/cholmod-f64.toml
    ```

*   **Run the warmup spec with multiple solver configurations and precisions:**

    ```bash
    julia --project=. src/main.jl -i data/specs/warmup.inspec -c configs/cholmod-f64.toml configs/approxchol-f128.toml
    ```

### Graphical User Interface (GUI)

A Python-based GUI simplifies the process of configuring and launching benchmark runs.

**To run the GUI:**

```bash
python src/gui.py
```

**GUI Usage:**

*   **Input Spec Files (-i):** Use the "➕" button to add one or more input spec files.
*   **Config Files (-c):** Use the "➕" button to add one or more solver configuration files.
*   **Output Database Path (-o):** Specify the path for the output SQLite database. If left empty, the Julia script will automatically name the database `spec_name.db` for each input spec file.
*   **Solution Directory (-s):** Specify a directory to save solution flow vectors.
*   Click "Run Benchmarks" to execute the Julia script. The GUI will close, and the benchmark output will be streamed to your console.

## Solver Configuration

Solver configurations are defined in TOML files in the `configs/` directory. Each file specifies a solver, its parameters, and the desired floating-point precision.

### Precision Control

The `precision` key in the configuration file determines the floating-point type used by the solver. Supported values are `"Float64"`, `"Float128"`, and `"Float64x2"`.

### Example: `configs/cholmod-f128.toml`

This configuration uses the `tulip_cholmod` solver with `Float128` precision. For high-precision types, this solver automatically falls back to a generic LDL factorization.

```toml
solver = "tulip_cholmod"
precision = "Float128"

[parameters]
IPM_PRegMin = 1e-6
IPM_DRegMin = 1e-6

[cholmod_parameters]
NestedDissection = true
```

### Example: `configs/approxchol-f64.toml`

This configuration uses the `tulip_approxchol` solver with standard `Float64` precision.

```toml
solver = "tulip_approxchol"
precision = "Float64"

[parameters]
IPM_PRegMin = 1e-4
IPM_DRegMin = 1e-8
IPM_IterationsLimit = 200

[kustom_parameters]
pcg_maxits = 100
pcg_tol = 5e-8

[kustom_parameters.ApproxCholParams]
type = "deg"
stag_test = 0
split = 2
merge = 2
```

## Output Database

Benchmark results are systematically stored in a SQLite database. This database is structured to facilitate easy querying and analysis of solver performance.

**Tables:**

*   `problems`: Contains metadata for each problem instance, including name, file path, size (vertices, edges), and optionally the true optimal value and time taken by an external Lemon solver.
*   `configs`: Stores the unique solver configurations (as TOML text) used in the runs, along with flattened key parameters for easier filtering, including the `precision`.
*   `runs`: Records the main results for each solver-problem pair, including status, solution time, iteration count, and links to `problems` and `configs`.
*   `solver_history`: Detailed iteration-level data for solvers, such as IPM iteration number, residual norms, and PCG iterations.

You can use any SQLite client (e.g., `sqlite3` command-line tool, DB Browser for SQLite) to browse and analyze the results. The Python plotting scripts in `plots/` are designed to visualize this data.

## Plotting Results

The `plots/` directory contains Python scripts to visualize the benchmark results stored in the SQLite database.

*   `plots/solver_performance.py`: Plots solver time and iteration count against problem size (number of edges).
*   `plots/failure_points_plot.py`: Identifies and plots IPM iterations where solvers experienced convergence issues (residual norm > PCG tolerance).
*   `plots/solver_history_plot.py`: Shows the number of problems failing convergence criteria over IPM iterations for different solvers.

**Example Usage:**

```bash
python plots/solver_performance.py <path_to_output.db>
python plots/failure_points_plot.py <path_to_output.db> --problem-pattern "grid_wide_08*"
```

## Development

### Code Formatting (Julia)

To ensure consistent code style for Julia files, `JuliaFormatter.jl` is used.

1.  **Add JuliaFormatter to the project (if not already added):**

    ```bash
    julia --project=. -e 'using Pkg; Pkg.add("JuliaFormatter")'
    ```

2.  **Run the formatter:**

    ```bash
    julia --project=. -e 'using JuliaFormatter; format(".")'
    ```

### Generating Problem Instances

The benchmark suite includes scripts to generate various grid-based MCF problem instances. These are useful for creating custom benchmark sets.

*   **`scripts/generate_grid_problems.jl`**: Generates grid problems with random capacities and costs (referred to as "widegrid" problems).

    ```bash
    julia --project=. scripts/generate_grid_problems.jl
    ```

    This script creates `.min.gz` and `.min` files in `data/problems/widegrid/` and updates `data/specs/widegrid.inspec`.

*   **`scripts/generate_uniform_grid_problems.jl`**: Generates grid problems with uniform capacities and costs (all set to 1, referred to as "unifwidegrid" problems).

    ```bash
    julia --project=. scripts/generate_uniform_grid_problems.jl
    ```

    This script creates `.min.gz` and `.min` files in `data/problems/unifwidegrid/` and updates `data/specs/unifwidegrid.inspec`.