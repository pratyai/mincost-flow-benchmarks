"""
    main.jl

This script serves as the main entry point for running the MinCostFlowBenchmarksV2.jl
benchmark suite. It parses command-line arguments, sets up the SQLite database schema,
executes various solvers on specified problem instances, and records the results.

The script supports:
- Specifying input problem files (`.inspec`).
- Defining solver configurations via TOML files.
- Storing benchmark results in an SQLite database.
- Optionally saving solution flow vectors.
- Automatically calculating true optimal values using an external DIMACS solver.
"""
using ArgParse
using CSV
using DataFrames
using JLD2
using SQLite
using TOML
using Debugger
using GZip

using Dimacs

# Solver modules
include("solvers/common.jl")
include("solvers/TulipApproxChol.jl")
include("solvers/TulipCHOLMOD.jl")

const SOLVERS = Dict("tulip_approxchol" => TulipApproxChol, "tulip_cholmod" => TulipCHOLMOD)

"""
    parse_cmdargs() -> Dict

Parses command-line arguments for the benchmark runner.

# Arguments
- `-i, --input-spec-files`: Path(s) to input spec file(s) (required).
- `-o, --output-db`: Path to store the output database. If absent, a database
  named after the spec file will be created for each spec.
- `-s, --solution-dir`: Directory to store solution flow-vectors. If absent,
  solutions will not be stored.
- `--configs`: Path(s) to solver configuration files (required).

# Returns
- A `Dict` containing the parsed command-line arguments.
"""
function parse_cmdargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        "-i"
        help = "input spec file(s)"
        arg_type = String
        nargs = '+' # Allow multiple input spec files
        required = true
        "-o"
        help = "path to store the output database (if absent, will use spec_name.db for each spec)"
        arg_type = Union{Nothing,String}
        default = nothing
        "-s"
        help = "output solution flow-vectors directory (if absent, will not store solutions)"
        arg_type = Union{Nothing,String}
        required = false
        default = nothing
        "-c"
        help = "paths to solver config files"
        arg_type = String
        nargs = '+'
        required = true
    end
    return parse_args(s)
end

"""
    setup_database_schema(db::SQLite.DB)

Sets up the necessary tables in the SQLite database for storing benchmark results.
Tables include `problems`, `configs`, `runs`, and `solver_history`.

# Arguments
- `db::SQLite.DB`: An opened SQLite database connection.
"""
function setup_database_schema(db::SQLite.DB)
    SQLite.execute(
        db,
        """
  CREATE TABLE IF NOT EXISTS problems (
      id INTEGER PRIMARY KEY,
      name TEXT UNIQUE,
      input_file TEXT,
      bytes INTEGER,
      num_vertices INTEGER,
      num_edges INTEGER,
      true_optimal REAL,
      lemon_time_s REAL
  )
""",
    )
    SQLite.execute(
        db,
        """
  CREATE TABLE IF NOT EXISTS configs (
      id INTEGER PRIMARY KEY,
      config_text TEXT UNIQUE,
      solver_name TEXT,
      ipm_preg_min REAL,
      ipm_dreg_min REAL,
      ipm_iterations_limit INTEGER,
      pcg_maxits INTEGER,
      pcg_tol REAL,
      approxchol_type TEXT,
      approxchol_stag_test INTEGER,
      approxchol_split INTEGER,
      approxchol_merge INTEGER,
      cholmod_nested_dissection BOOLEAN
  )
""",
    )
    SQLite.execute(
        db,
        """
  CREATE TABLE IF NOT EXISTS runs (
      id INTEGER PRIMARY KEY,
      name TEXT,
      status TEXT,
      solver_name TEXT,
      config_id INTEGER,
      problem_id INTEGER,
      time_s REAL,
      iters INTEGER,
      solution_file TEXT,
      fact_s REAL,
      solv_s REAL,
      sddm_calls INTEGER,
      optimal_value REAL,
      FOREIGN KEY (config_id) REFERENCES configs(id),
      FOREIGN KEY (problem_id) REFERENCES problems(id)
  )
""",
    )
    SQLite.execute(
        db,
        """
  CREATE TABLE IF NOT EXISTS solver_history (
      id INTEGER PRIMARY KEY,
      run_id INTEGER,
      ipm_iter INTEGER,
      solve_in_iter INTEGER,
      residual_norm REAL,
      pcg_iterations INTEGER,
      FOREIGN KEY (run_id) REFERENCES runs(id)
  )
""",
    )
end

"""
    get_true_optimal_value(dimacs_solver_path::String, indimacs_file::String) -> Tuple{Union{Float64,Missing}, Union{Float64,Missing}}

Executes an external DIMACS solver to obtain the true optimal cost and solution time
for a given problem instance. Handles gzipped input files by decompressing them
to a temporary file.

# Arguments
- `dimacs_solver_path::String`: The absolute path to the DIMACS solver executable.
- `indimacs_file::String`: The path to the DIMACS problem file (`.min` or `.min.gz`).

# Returns
- A `Tuple` containing:
  - `true_optimal::Union{Float64,Missing}`: The true optimal cost, or `missing` if parsing fails.
  - `lemon_time_s::Union{Float64,Missing}`: The time taken by the DIMACS solver, or `missing` if parsing fails.
"""
function get_true_optimal_value(dimacs_solver_path::String, indimacs_file::String)
    local true_optimal::Union{Float64,Missing} = missing
    local lemon_time_s::Union{Float64,Missing} = missing
    local file_to_solve = indimacs_file
    local is_temp_file = false

    if endswith(indimacs_file, ".gz")
        temp_file = mktemp()[1] # Create a temporary file
        GZip.open(indimacs_file) do gz_file
            open(temp_file, "w") do out_file
                write(out_file, read(gz_file))
            end
        end
        file_to_solve = temp_file
        is_temp_file = true
    end

    try
        # Execute dimacs-solver
        local start_time = time()
        local stdout_pipe = Pipe()
        local stderr_pipe = Pipe()
        local process = run(
            pipeline(
                Cmd([dimacs_solver_path, file_to_solve]),
                stdout = stdout_pipe,
                stderr = stderr_pipe,
            ),
            wait = false,
        )

        # Close the write ends of the pipes in the parent process
        close(stdout_pipe.in)
        close(stderr_pipe.in)

        local stdout_output = read(stdout_pipe, String)
        local stderr_output = read(stderr_pipe, String)
        wait(process)
        close(stdout_pipe)
        close(stderr_pipe)
        local end_time = time()
        lemon_time_s = end_time - start_time

        match_obj_optimal = match(r"Min flow cost: ([-+]?\d*\.?\d+)", stderr_output)
        if match_obj_optimal !== nothing
            true_optimal = parse(Float64, match_obj_optimal.captures[1])
        else
            println("DEBUG: stderr_output from dimacs-solver:\n" * stderr_output)
            error(
                "Could not parse optimal cost from dimacs-solver output for ",
                file_to_solve,
            )
        end

        match_obj_time = match(
            r"Run NetworkSimplex: u: [^,]+, s: [^,]+, cu: [^,]+, cs: [^,]+, real: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)s",
            stderr_output,
        )
        if match_obj_time !== nothing
            lemon_time_s = parse(Float64, match_obj_time.captures[1])
        else
            println("DEBUG: stderr_output from dimacs-solver:\n" * stderr_output)
            error("Could not parse real time from dimacs-solver output for ", file_to_solve)
        end
    catch e
        println("Error running dimacs-solver for ", indimacs_file, ": ", e)
    finally
        if is_temp_file
            rm(file_to_solve) # Delete the temporary file
        end
    end
    return (true_optimal, lemon_time_s)
end

"""
    process_single_spec(input_spec_file::String, output_db_path::String, solution_dir::Union{Nothing,String}, config_files::Vector{String})

Processes a single input spec file, running benchmarks for each problem and solver
configuration, and storing the results in the specified SQLite database.

# Arguments
- `input_spec_file::String`: Path to the input spec file (CSV format).
- `output_db_path::String`: Path to the SQLite database where results will be stored.
- `solution_dir::Union{Nothing,String}`: Optional directory to save solution flow-vectors.
- `config_files::Vector{String}`: A list of paths to solver configuration files.
"""
function process_single_spec(
    input_spec_file::String,
    output_db_path::String,
    solution_dir::Union{Nothing,String},
    config_files::Vector{String},
)
    local probspec = CSV.read(input_spec_file, DataFrame)

    # Setup database
    db = SQLite.DB(output_db_path)
    # Populate problems table
    problem_ids = Dict{String,Int}()
    dimacs_solver_path = "/Users/pmz/Downloads/lemon-1.3.1/build/tools/dimacs-solver"

    for r in eachrow(probspec)
        local indimacs::String = joinpath(dirname(Base.@__DIR__), r[:input_file])
        local netw = Dimacs.ReadDimacs(indimacs)
        local num_vertices = netw.G.n
        local num_edges = netw.G.m

        local problem_id = missing
        local existing_true_optimal::Union{Float64,Missing} = missing
        local existing_lemon_time_s::Union{Float64,Missing} = missing

        query = DBInterface.execute(
            db,
            "SELECT id, true_optimal, lemon_time_s FROM problems WHERE name = ?",
            (r[:name],),
        )
        for row in query
            problem_id = row.id
            existing_true_optimal = row.true_optimal
            existing_lemon_time_s = row.lemon_time_s
        end

        local true_optimal::Union{Float64,Missing} = missing
        local lemon_time_s::Union{Float64,Missing} = missing

        if ismissing(problem_id)
            DBInterface.execute(
                db,
                "INSERT INTO problems (name, input_file, bytes, num_vertices, num_edges, true_optimal, lemon_time_s) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    r[:name],
                    r[:input_file],
                    r[:bytes],
                    num_vertices,
                    num_edges,
                    missing,
                    missing,
                ),
            )
            problem_id = SQLite.last_insert_rowid(db)
        end

        # Always try to get true optimal value and lemon time, and update if missing
        if ismissing(existing_true_optimal) || ismissing(existing_lemon_time_s)
            (true_optimal, lemon_time_s) =
                get_true_optimal_value(dimacs_solver_path, indimacs)
            DBInterface.execute(
                db,
                "UPDATE problems SET true_optimal = ?, lemon_time_s = ? WHERE id = ?",
                (true_optimal, lemon_time_s, problem_id),
            )
        else
            true_optimal = existing_true_optimal
            lemon_time_s = existing_lemon_time_s
            println(
                "  True optimal value and lemon solver time for `",
                r[:name],
                "` already exists; skipping recalculation.",
            )
        end
        problem_ids[r[:name]] = problem_id
    end

    # Store parsed configs for later use
    parsed_configs = []
    for config_file in config_files
        config_text = read(config_file, String)
        config = TOML.parse(config_text)
        solver_name = config["solver"]
        if !haskey(SOLVERS, solver_name)
            println(
                "Error: Solver `",
                solver_name,
                "` from config file `",
                config_file,
                "` not found.",
            )
            continue
        end
        solver_module = SOLVERS[solver_name]

        # Get or create config_id
        # Extract common parameters
        ipm_preg_min =
            get(config, "parameters", Dict()) |> (p -> get(p, "IPM_PRegMin", missing))
        ipm_dreg_min =
            get(config, "parameters", Dict()) |> (p -> get(p, "IPM_DRegMin", missing))
        ipm_iterations_limit =
            get(config, "parameters", Dict()) |>
            (p -> get(p, "IPM_IterationsLimit", missing))

        # Extract approxchol specific parameters
        pcg_maxits =
            get(config, "kustom_parameters", Dict()) |>
            (kp -> get(kp, "pcg_maxits", missing))
        pcg_tol =
            get(config, "kustom_parameters", Dict()) |> (kp -> get(kp, "pcg_tol", missing))

        approxchol_params_dict =
            get(config, "kustom_parameters", Dict()) |>
            (kp -> get(kp, "ApproxCholParams", Dict()))
        approxchol_type = get(approxchol_params_dict, "type", missing)
        approxchol_stag_test = get(approxchol_params_dict, "stag_test", missing)
        approxchol_split = get(approxchol_params_dict, "split", missing)
        approxchol_merge = get(approxchol_params_dict, "merge", missing)

        # Extract cholmod specific parameters
        cholmod_params = get(config, "cholmod_parameters", Dict())
        cholmod_nested_dissection = get(cholmod_params, "NestedDissection", missing)

        query = DBInterface.execute(
            db,
            "SELECT id FROM configs WHERE config_text = ?",
            (config_text,),
        )
        config_id = missing
        for row in query
            config_id = row.id
        end

        if ismissing(config_id)
            DBInterface.execute(
                db,
                """
    INSERT INTO configs (
        config_text, solver_name, ipm_preg_min, ipm_dreg_min, ipm_iterations_limit,
        pcg_maxits, pcg_tol, approxchol_type, approxchol_stag_test,
        approxchol_split, approxchol_merge, cholmod_nested_dissection
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
                (
                    config_text,
                    solver_name,
                    ipm_preg_min,
                    ipm_dreg_min,
                    ipm_iterations_limit,
                    pcg_maxits,
                    pcg_tol,
                    approxchol_type,
                    approxchol_stag_test,
                    approxchol_split,
                    approxchol_merge,
                    cholmod_nested_dissection,
                ),
            )
            config_id = SQLite.last_insert_rowid(db)
        end
        push!(
            parsed_configs,
            (
                config_file = config_file,
                config = config,
                solver_module = solver_module,
                config_id = config_id,
                solver_name = solver_name,
            ),
        )
    end

    for r in eachrow(probspec)
        println("Processing problem: ", r[:name])
        local indimacs::String = joinpath(dirname(Base.@__DIR__), r[:input_file])
        local netw = Dimacs.ReadDimacs(indimacs)
        local current_problem_id = problem_ids[r[:name]]

        for parsed_config in parsed_configs
            config_file = parsed_config.config_file
            config = parsed_config.config
            solver_module = parsed_config.solver_module
            config_id = parsed_config.config_id
            solver_name = parsed_config.solver_name

            println("  with config: ", config_file)

            # Check if result already exists
            query = DBInterface.execute(
                db,
                "SELECT id FROM runs WHERE name = ? AND config_id = ? AND problem_id = ?",
                (r[:name], config_id, current_problem_id),
            )
            if !isempty(query)
                println(
                    "  record already exists for `",
                    r[:name],
                    "` with config from `",
                    config_file,
                    "`; skipping it",
                )
                continue
            end

            local results = solver_module.solve(netw, config)

            # Save solution if asked for.
            local sol_file::Union{String,Missing} = missing
            if !isnothing(solution_dir)
                mkpath(solution_dir)
                sol_file = joinpath(
                    solution_dir,
                    r[:name] * "_" * splitext(basename(config_file))[1] * ".jld2",
                )
                jldsave(sol_file, true; x = results.solution)
            end

            # Insert main results
            fact_s = haskey(results, :fact_s) ? results.fact_s : missing
            solv_s = haskey(results, :solv_s) ? results.solv_s : missing
            sddm_calls = haskey(results, :sddm_calls) ? results.sddm_calls : missing



            # Get optimal value
            local optimal_value =
                haskey(results, :objective_value) ? results.objective_value : missing

            DBInterface.execute(
                db,
                """
  INSERT INTO runs (name, status, solver_name, config_id, problem_id, time_s, iters, solution_file, fact_s, solv_s, sddm_calls, optimal_value)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) 
""",
                (
                    r[:name],
                    String(Symbol(results.status)),
                    solver_name,
                    config_id,
                    current_problem_id,
                    results.seconds,
                    results.iters,
                    sol_file,
                    fact_s,
                    solv_s,
                    sddm_calls,
                    optimal_value,
                ),
            )

            local run_id = SQLite.last_insert_rowid(db)

            # Insert history
            if haskey(results, :residual_history) && !ismissing(results.residual_history)
                pcg_history =
                    haskey(results, :pcg_iterations_history) ?
                    results.pcg_iterations_history : missing
                for (idx, (ipm_iter, solve_in_iter, residual)) in
                    enumerate(results.residual_history)
                    pcg_iters =
                        ismissing(pcg_history) || isempty(pcg_history) ? missing :
                        pcg_history[idx]
                    DBInterface.execute(
                        db,
                        "INSERT INTO solver_history (run_id, ipm_iter, solve_in_iter, residual_norm, pcg_iterations) VALUES (?, ?, ?, ?, ?)",
                        (run_id, ipm_iter, solve_in_iter, residual, pcg_iters),
                    )
                end
            end
        end # Closes 'for parsed_config in parsed_configs'
    end # Closes 'for r in eachrow(probspec)'
end # Closes 'function process_single_spec()'

"""
    run_benchmarks(args::Dict)

Orchestrates the execution of benchmarks based on parsed command-line arguments.
It iterates through input spec files, sets up the database, and calls `process_single_spec`
for each spec.

# Arguments
- `args::Dict`: A dictionary containing parsed command-line arguments.
"""
function run_benchmarks(args::Dict)

    local config_files = args["configs"]

    local input_specs = args["i"]
    local output_db_arg = args["o"]
    local solution_dir = args["s"]

    if isnothing(output_db_arg)
        # No output DB specified, create one per input spec
        for input_spec_file in input_specs
            spec_name = splitext(basename(input_spec_file))[1]
            output_db_name = "$(spec_name).db"
            println("Running benchmarks for $(input_spec_file) into $(output_db_name)")
            db = SQLite.DB(output_db_name)
            setup_database_schema(db)
            process_single_spec(input_spec_file, output_db_name, solution_dir, config_files)
        end
    else
        # Output DB specified, dump all into it
        db = SQLite.DB(output_db_arg)
        setup_database_schema(db)
        for input_spec_file in input_specs
            println("Running benchmarks for $(input_spec_file) into $(output_db_arg)")
            process_single_spec(input_spec_file, output_db_arg, solution_dir, config_files)
        end
    end
end

function main()
    local args = parse_cmdargs()
    @show args

    # Run benchmarks with the parsed arguments
    run_benchmarks(args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
