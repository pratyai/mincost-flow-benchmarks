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

## Usage

The main entry point for running benchmarks is `src/main.jl`. It takes the following command-line arguments:

*   `-i <path>`: **(Required)** Path to the input spec file. The spec file is a CSV file that lists the problem instances to run. See `data/specs/warmup.inspec` for an example.
*   `--solver <name...>`: **(Required)** The name(s) of the solver(s) to use. You can specify one or more solvers. Available solvers are:
    *   `tulip_basic`
    *   `tulip_approxchol`
    *   `tulip_cholmod`
*   `-o <path>`: (Optional) Path to store the output spec file (in CSV format). If not provided, the results will be printed to standard output. If the file already exists, new results will be appended.
*   `-s <path>`: (Optional) Path to a directory where solution flow vectors will be stored (in JLD2 format).

### Example Commands

Here are a few examples of how to run the benchmarks:

*   **Run the warmup spec with the `tulip_basic` solver and print results to the console:**

    ```bash
    julia --project=. src/main.jl -i data/specs/warmup.inspec --solver tulip_basic
    ```

*   **Run the warmup spec with the `tulip_approxchol` solver and save the results to a CSV file:**

    ```bash
    julia --project=. src/main.jl -i data/specs/warmup.inspec --solver tulip_approxchol -o results.csv
    ```

*   **Run the warmup spec with multiple solvers and save the results to a CSV file:**

    ```bash
    julia --project=. src/main.jl -i data/specs/warmup.inspec --solver tulip_basic tulip_cholmod -o results.csv
    ```

*   **Run the warmup spec, save results, and also save the solution vectors:**

    ```bash
    julia --project=. src/main.jl -i data/specs/warmup.inspec --solver tulip_basic -o results.csv -s solutions
    ```