# MinCostFlowBenchmarksV2

This project provides a benchmarking suite for minimum cost flow solvers in Julia.

## Installation

1.  **Install Julia:** If you don't have Julia installed, download it from [julialang.org](https://julialang.org/downloads/).
2.  **Instantiate the project:** Open a Julia REPL in the project root directory and run the following commands to install the dependencies:

    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

## Code Formatting

To format the Julia code in this project, you can use `JuliaFormatter.jl`.

1.  **Add JuliaFormatter to the project (if not already added):**

    ```bash
    julia --project=. -e 'using Pkg; Pkg.add("JuliaFormatter")'
    ```

2.  **Run the formatter:**

    ```bash
    julia --project=. -e 'using JuliaFormatter; format(".")'
    ```

## Usage

The main entry point for running benchmarks is `src/main.jl`. It takes the following command-line arguments:

*   `-i <path>`: **(Required)** Path to the input spec file. The spec file is a CSV file that lists the problem instances to run. See `data/specs/warmup.inspec` for an example.
*   `--configs <path...>`: **(Required)** Paths to one or more solver configuration files. See the "Solver Configuration" section below for details.
*   `-o <path>`: (Optional) Path to store the output SQLite database file. Defaults to `benchmarks.db`. If the file already exists, new results will be appended.
*   `-s <path>`: (Optional) Path to a directory where solution flow vectors will be stored (in JLD2 format).

## Solver Configuration

Solver configurations are defined in TOML files located in the `configs/` directory. Each configuration file specifies the solver to use and its parameters.

### Example: `configs/cholmod.toml`

```toml
solver = "tulip_cholmod"

[parameters]
IPM_PRegMin = 1e-6
IPM_DRegMin = 1e-6

[cholmod_parameters]
NestedDissection = true
```

### Example: `configs/approxchol.toml`

```toml
solver = "tulip_approxchol"

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

The benchmark results are stored in a SQLite database file. The database contains three tables:

*   `configs`: This table stores the unique solver configurations used for the runs, including flattened parameters like `solver_name`, `ipm_preg_min`, `pcg_maxits`, etc. for easier querying.
*   `runs`: This table stores the main results for each benchmark run, with foreign keys to the `configs` and `problems` tables.
*   `problems`: This table stores details about each problem instance, including its name, input file path, and size.
*   `solver_history`: This table stores the detailed history of the linear solver's residual norm and PCG iteration count for each iteration of the interior-point method, linked to the `runs` table.

You can use any SQLite client to browse and analyze the results.

### Example Commands

*   **Run the warmup spec with a single solver configuration:**

    ```bash
    julia --project=. src/main.jl -i data/specs/warmup.inspec --configs configs/cholmod.toml
    ```

*   **Run the warmup spec with multiple solver configurations:**

    ```bash
    julia --project=. src/main.jl -i data/specs/warmup.inspec --configs configs/cholmod.toml configs/approxchol.toml
    ```

---

## Generating Problem Instances

To generate the grid problem instances used for benchmarking, there are two scripts:

*   **`scripts/generate_grid_problems.jl`**: Generates grid problems with random capacities and costs (named "widegrid").

    ```bash
    julia --project=. scripts/generate_grid_problems.jl
    ```

    This script will create `.min.gz` and `.min` files in the `data/problems/widegrid/` directory and update the `data/specs/widegrid.inspec` file with the generated problems.

*   **`scripts/generate_uniform_grid_problems.jl`**: Generates grid problems with uniform capacities and costs (all 1, named "unifwidegrid").

    ```bash
    julia --project=. scripts/generate_uniform_grid_problems.jl
    ```

    This script will create `.min.gz` and `.min` files in the `data/problems/unifwidegrid/` directory and update the `data/specs/unifwidegrid.inspec` file with the generated problems.
