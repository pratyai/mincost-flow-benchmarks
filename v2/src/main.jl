#=
    main.jl

This script serves as the main entry point for running the MinCostFlowBenchmarksV2.jl
benchmark suite. It parses command-line arguments, sets up the SQLite database schema,
executes various solvers on specified problem instances, and records the results.

The script supports:
- Specifying input problem files (`.inspec`).
- Defining solver configurations via TOML files, including floating-point precision.
- Storing benchmark results in an SQLite database.
- Optionally saving solution flow vectors.
- Automatically calculating true optimal values using an external DIMACS solver.
=#
using ArgParse
using CSV
using DataFrames
using JLD2
using SQLite
using TOML
using Debugger
using GZip
using Base.Threads # For multi-threading
using Quadmath
using MultiFloats
using REPL
using REPL.TerminalMenus

using Dimacs

# Solver modules
include("solvers/common.jl")
include("solvers/TulipApproxChol.jl")
include("solvers/TulipCHOLMOD.jl")
include("solvers/ECOSSolver.jl")
include("solvers/MosekSolver.jl")
include("solvers/GurobiSolver.jl")
include("solvers/MehrotraSolver.jl")
include("solvers/MehrotraApproxChol.jl")
include("solvers/RandomizedSparseSolver.jl")

const SOLVERS = Dict(
    "tulip_approxchol" => TulipApproxChol,
    "tulip_cholmod" => TulipCHOLMOD,
    "ecos" => ECOSSolver,
    "mosek" => MosekSolver,
    "gurobi" => GurobiSolver,
    "mehrotra" => MehrotraSolver,
    "mehrotra_approxchol" => MehrotraApproxChol,
    "randomized_sparse" => RandomizedSparseSolver,
)
const SUPPORTED_FLOAT_TYPES =
    Dict("Float64" => Float64, "Float128" => Float128, "Float64x2" => Float64x2)

# Custom MultiSelectMenu that uses Space for toggle and Enter for Done
import REPL.TerminalMenus: AbstractMenu, keypress, header, options, pick, cancel, writeline, numoptions, config

mutable struct SpaceMultiSelectMenu <: AbstractMenu
    options::Vector{String}
    pagesize::Int
    pageoffset::Int
    selected::Set{Int}
    cursor::Int
    config::TerminalMenus.Config
end

function SpaceMultiSelectMenu(options; pagesize=10, config=TerminalMenus.Config())
    SpaceMultiSelectMenu(options, pagesize, 0, Set{Int}(), 1, config)
end

TerminalMenus.options(m::SpaceMultiSelectMenu) = m.options
TerminalMenus.numoptions(m::SpaceMultiSelectMenu) = length(m.options)
TerminalMenus.header(m::SpaceMultiSelectMenu) = "[press: Space=toggle, Enter=done, q=abort]"
TerminalMenus.config(m::SpaceMultiSelectMenu) = m.config

function TerminalMenus.writeline(buf::IO, m::SpaceMultiSelectMenu, idx::Int, iscursor::Bool)
    if iscursor
        m.cursor = idx
    end
    print(buf, iscursor ? "> " : "  ")
    print(buf, idx in m.selected ? "[X] " : "[ ] ")
    print(buf, m.options[idx])
end

function TerminalMenus.keypress(m::SpaceMultiSelectMenu, key::UInt32)
    if key == UInt32(' ')
        if m.cursor in m.selected
            delete!(m.selected, m.cursor)
        else
            push!(m.selected, m.cursor)
        end
    elseif key == UInt32('\r')
        return true
    elseif key == UInt32('q')
        empty!(m.selected)
        return true
    end
    return false
end

TerminalMenus.pick(m::SpaceMultiSelectMenu, cursor::Int) = true
TerminalMenus.cancel(m::SpaceMultiSelectMenu) = empty!(m.selected)


"""
    expand_and_select_configs(paths::Vector{String}) -> Vector{String}

Expands directory paths to include all .toml files within them, and then presents
an interactive menu for the user to select which configuration files to use.

- Files explicitly passed in `paths` are selected by default.
- Files found within passed directories are not selected by default.
"""
function expand_and_select_configs(paths::Vector{String})
    if isempty(paths)
        return String[]
    end

    # If no directory is passed, skip UI and run with provided files
    if !any(isdir, paths)
        return paths
    end

    selection_map = Dict{String, Bool}() # path -> is_selected
    ordered_files = String[]
    
    for p in paths
        if isdir(p)
            # It's a directory: list all .toml files, default unselected
            try
                files = readdir(p, join=true)
                for f in files
                    if endswith(f, ".toml")
                        if !haskey(selection_map, f)
                            push!(ordered_files, f)
                            selection_map[f] = false
                        end
                        # If already in map (e.g. from previous specific file arg), leave it as is
                    end
                end
            catch e
                @warn "Could not read directory $p: $e"
            end
        elseif isfile(p)
            # It's a file: default selected
            if !haskey(selection_map, p)
                push!(ordered_files, p)
                selection_map[p] = true
            else
                selection_map[p] = true # Ensure it's selected if it was previously unselected
            end
        else
            @warn "Config path not found: $p"
        end
    end
    
    if isempty(ordered_files)
        println("No configuration files found.")
        return String[]
    end

    # Interactive selection using custom SpaceMultiSelectMenu
    menu = SpaceMultiSelectMenu(ordered_files; pagesize=20, config=TerminalMenus.Config(scroll_wrap=true))
    for (i, f) in enumerate(ordered_files)
        if selection_map[f]
            push!(menu.selected, i)
        end
    end

    request("Select configuration files:", menu)
    
    # Use menu.selected directly
    indices = collect(menu.selected)
    if isempty(indices)
        return String[]
    end
    
    return ordered_files[indices]
end

"""
    expand_and_select_inspecs(paths::Vector{String}) -> Vector{String}

Processes input spec paths. If any path is a directory, it lists all .inspec files
within all provided paths (explicit files + directory contents) and asks the user
to select EXTREMELY ONE file to run.

- If no directory is provided (only explicit files), it returns them as-is (no menu).
- If a directory is provided, a RadioMenu is shown to enforce single selection.
"""
function expand_and_select_inspecs(paths::Vector{String})
    # Check if any path is a directory
    has_dir = any(isdir, paths)
    
    if !has_dir
        return paths # No interaction needed, just run what was given
    end

    # Gather all candidates
    candidates = String[]
    for p in paths
        if isdir(p)
             try
                files = readdir(p, join=true)
                for f in files
                    if endswith(f, ".inspec")
                        push!(candidates, f)
                    end
                end
            catch e
                @warn "Could not read directory $p: $e"
            end
        elseif isfile(p)
            push!(candidates, p)
        end
    end
    
    unique!(candidates)

    if isempty(candidates)
         println("No .inspec files found.")
         return String[]
    end
    
    # Sort for niceness
    sort!(candidates)
    
    # RadioMenu for single selection
    println("\nDirectory detected in input specs. Please select ONE input spec file:")
    menu = RadioMenu(candidates; pagesize=20, scroll_wrap=true)
    choice = request("Select input spec file:", menu)
    
    if choice == -1
        println("Selection cancelled.")
        return String[]
    end
    
    return [candidates[choice]]
end

"""
    parse_cmdargs() -> Dict

Parses command-line arguments for the benchmark runner.

# Arguments
- `-i`: Path(s) to input spec file(s) (required).
- `-o`: Path to store the output database. If absent, a database
  named after the spec file will be created for each spec.
- `-s`: Directory to store solution flow-vectors. If absent,
  solutions will not be stored.
- `-c`: Path(s) to solver configuration files (required).

# Returns
- A `Dict` containing the parsed command-line arguments.
"""
function parse_cmdargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        "-i"
        help = "input spec file(s) (default: data/specs/)"
        arg_type = String
        action = :append_arg
        default = String[]
        required = false
        dest_name = "input_spec_files"
        "-o"
        help = "path to store the output database (if absent, will use spec_name.db for each spec)"
        arg_type = Union{Nothing,String}
        default = nothing
        dest_name = "output_db"
        "-s"
        help = "output solution flow-vectors directory (if absent, will not store solutions)"
        arg_type = Union{Nothing,String}
        required = false
        default = nothing
        dest_name = "solution_dir"
        "-c"
        help = "paths to solver config files (default: configs/)"
        arg_type = String
        action = :append_arg
        default = String[]
        required = false
        dest_name = "config_files"
        "--run-true-optimal"
        help = "run external DIMACS solver to calculate true optimal values"
        action = :store_true
        dest_name = "run_true_optimal"
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
    # Enable WAL mode and set busy timeout for better concurrency
    SQLite.execute(db, "PRAGMA busy_timeout = 30000;")
    SQLite.execute(db, "PRAGMA journal_mode = WAL;")

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
      precision TEXT,
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
      relative_residual_norm REAL,
      absolute_residual_norm REAL,
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
    process_single_spec(input_spec_file::String, output_db_path::String, solution_dir::Union{Nothing,String}, config_files::Vector{String}, run_true_optimal::Bool)

Processes a single input spec file, running benchmarks for each problem and solver
configuration, and storing the results in the specified SQLite database.

# Arguments
- `input_spec_file::String`: Path to the input spec file (CSV format).
- `output_db_path::String`: Path to the SQLite database where results will be stored.
- `solution_dir::Union{Nothing,String}`: Optional directory to save solution flow-vectors.
- `config_files::Vector{String}`: A list of paths to solver configuration files.
- `run_true_optimal::Bool`: Whether to run the external DIMACS solver.
"""
function process_single_spec(
    input_spec_file::String,
    output_db_path::String,
    solution_dir::Union{Nothing,String},
    config_files::Vector{String},
    run_true_optimal::Bool,
)
    local probspec = CSV.read(input_spec_file, DataFrame)

    db = SQLite.DB(output_db_path)
    problem_ids = Dict{String,Int}()
    dimacs_solver_path = "/Users/pmz/Downloads/lemon-1.3.1/build/tools/dimacs-solver"

    # --- Pass 1: Populate problems table and get problem_ids ---
    # Create a channel to send problems to the background solver thread
    problem_channel = Channel(Inf)

    local lemon_solver_task = nothing # Initialize to nothing
    local problems_to_solve_in_bg = false

    for r in eachrow(probspec)
        local indimacs::String = joinpath(dirname(Base.@__DIR__), r[:input_file])

        local problem_id = missing
        local existing_true_optimal::Union{Float64,Missing} = missing
        local existing_lemon_time_s::Union{Float64,Missing} = missing
        local existing_num_vertices::Union{Int,Missing} = missing
        local existing_num_edges::Union{Int,Missing} = missing

        query = DBInterface.execute(
            db,
            "SELECT id, true_optimal, lemon_time_s, num_vertices, num_edges FROM problems WHERE name = ?",
            (r[:name],),
        )
        for row in query
            problem_id = row.id
            existing_true_optimal = row.true_optimal
            existing_lemon_time_s = row.lemon_time_s
            existing_num_vertices = row.num_vertices
            existing_num_edges = row.num_edges
        end

        local num_vertices::Union{Int,Missing} = existing_num_vertices
        local num_edges::Union{Int,Missing} = existing_num_edges

        if ismissing(problem_id)
            # If problem is new, parse Dimacs to get num_vertices and num_edges
            local netw = Dimacs.ReadDimacs(indimacs)
            num_vertices = netw.G.n
            num_edges = netw.G.m

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
        elseif ismissing(existing_num_vertices) || ismissing(existing_num_edges)
            # If problem exists but num_vertices/num_edges are missing, parse Dimacs to fill them
            local netw = Dimacs.ReadDimacs(indimacs)
            num_vertices = netw.G.n
            num_edges = netw.G.m
            DBInterface.execute(
                db,
                "UPDATE problems SET num_vertices = ?, num_edges = ? WHERE id = ?",
                (num_vertices, num_edges, problem_id),
            )
        end

        if run_true_optimal && (ismissing(existing_true_optimal) || ismissing(existing_lemon_time_s))
            # Put the problem into the channel for the background solver
            put!(problem_channel, (r[:name], problem_id, indimacs))
            problems_to_solve_in_bg = true
        end
        problem_ids[r[:name]] = problem_id
    end

    # Close the channel to signal that no more problems will be sent
    close(problem_channel)

    if problems_to_solve_in_bg
        lemon_solver_task = Threads.@spawn begin
            for (problem_name, problem_id, indimacs) in problem_channel
                try
                    (true_optimal, lemon_time_s) =
                        get_true_optimal_value(dimacs_solver_path, indimacs)
                    DBInterface.execute(
                        db,
                        "UPDATE problems SET true_optimal = ?, lemon_time_s = ? WHERE id = ?",
                        (true_optimal, lemon_time_s, problem_id),
                    )
                catch e
                    println("Error in background lemon solver for ", problem_name, ": ", e)
                    # Optionally, update the problem status in DB to indicate failure
                end
            end
        end
    end

    # --- Pass 2: Parse configs and get config_ids ---
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
        float_type_str = get(config, "precision", "Float64")
        if !haskey(SUPPORTED_FLOAT_TYPES, float_type_str)
            error("Unsupported float type: $float_type_str")
        end
        float_type = SUPPORTED_FLOAT_TYPES[float_type_str]

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
        config_text, solver_name, precision, ipm_preg_min, ipm_dreg_min, ipm_iterations_limit,
        pcg_maxits, pcg_tol, approxchol_type, approxchol_stag_test,
        approxchol_split, approxchol_merge, cholmod_nested_dissection
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
                (
                    config_text,
                    solver_name,
                    float_type_str,
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
                float_type = float_type,
            ),
        )
    end

    is_warmup = occursin("warmup", basename(input_spec_file))

    # --- Pass 3: Run benchmarks, skipping problems if all runs exist ---
    for r in eachrow(probspec)
        local current_problem_id = problem_ids[r[:name]]
        local indimacs::String = joinpath(dirname(Base.@__DIR__), r[:input_file])

        # Check if all runs for this problem already exist
        should_skip_problem = !is_warmup
        if should_skip_problem
            for parsed_config in parsed_configs
                query = DBInterface.execute(
                    db,
                    "SELECT id FROM runs WHERE name = ? AND config_id = ? AND problem_id = ?",
                    (r[:name], parsed_config.config_id, current_problem_id),
                )
                if isempty(query)
                    should_skip_problem = false
                    break # Found at least one missing run, so don't skip this problem
                end
            end
        end

        if should_skip_problem
            println("Skipping problem `", r[:name], "` as all runs already exist.")
            continue # Skip to the next problem
        end

        println("Processing problem: ", r[:name])
        local netw = Dimacs.ReadDimacs(indimacs) # Parse only if not skipped

        for parsed_config in parsed_configs
            config_file = parsed_config.config_file
            config = parsed_config.config
            solver_module = parsed_config.solver_module
            config_id = parsed_config.config_id
            solver_name = parsed_config.solver_name
            float_type = parsed_config.float_type

            println("  with config: ", config_file)

            # Check if result already exists (this check is still needed for individual runs)
            if !is_warmup
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
            else
                # If warmup, clear previous results for this run to avoid duplicates and ensure fresh data
                DBInterface.execute(db, 
                    "DELETE FROM solver_history WHERE run_id IN (SELECT id FROM runs WHERE name = ? AND config_id = ? AND problem_id = ?)",
                    (r[:name], config_id, current_problem_id)
                )
                DBInterface.execute(db,
                    "DELETE FROM runs WHERE name = ? AND config_id = ? AND problem_id = ?",
                    (r[:name], config_id, current_problem_id)
                )
            end

            local results = solver_module.solve(netw, config, float_type)

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
                for (
                    idx,
                    (ipm_iter, solve_in_iter, relative_residual, absolute_residual),
                ) in enumerate(results.residual_history)
                    pcg_iters =
                        ismissing(pcg_history) || isempty(pcg_history) ? missing :
                        pcg_history[idx]
                    DBInterface.execute(
                        db,
                        "INSERT INTO solver_history (run_id, ipm_iter, solve_in_iter, relative_residual_norm, absolute_residual_norm, pcg_iterations) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            run_id,
                            ipm_iter,
                            solve_in_iter,
                            relative_residual,
                            absolute_residual,
                            pcg_iters,
                        ),
                    )
                end
            end
        end # Closes 'for parsed_config in parsed_configs'
    end # Closes 'for r in eachrow(probspec)'
end

"""
    run_benchmarks(args::Dict)

Orchestrates the execution of benchmarks based on parsed command-line arguments.
It iterates through input spec files, sets up the database, and calls `process_single_spec`
for each spec.

# Arguments
- `args::Dict`: A dictionary containing parsed command-line arguments.
"""
function run_benchmarks(args::Dict)

    local config_files = args["config_files"]

    local input_specs = args["input_spec_files"]
    local output_db_arg = args["output_db"]
    local solution_dir = args["solution_dir"]
    local run_true_optimal = args["run_true_optimal"]

    local all_lemon_tasks = [] # Collect all lemon solver tasks

    if isnothing(output_db_arg)
        # No output DB specified, create one per input spec
        for input_spec_file in input_specs
            spec_name = splitext(basename(input_spec_file))[1]
            output_db_name = "$(spec_name).db"
            println("Running benchmarks for $(input_spec_file) into $(output_db_name)")
            db = SQLite.DB(output_db_name)
            setup_database_schema(db)
            lemon_task = process_single_spec(
                input_spec_file,
                output_db_name,
                solution_dir,
                config_files,
                run_true_optimal,
            )
            if lemon_task !== nothing
                push!(all_lemon_tasks, lemon_task)
            end
        end
    else
        # Output DB specified, dump all into it
        db = SQLite.DB(output_db_arg)
        setup_database_schema(db)
        for input_spec_file in input_specs
            println("Running benchmarks for $(input_spec_file) into $(output_db_arg)")
            lemon_task = process_single_spec(
                input_spec_file,
                output_db_arg,
                solution_dir,
                config_files,
                run_true_optimal,
            )
            if lemon_task !== nothing
                push!(all_lemon_tasks, lemon_task)
            end
        end
    end

    # Wait for all background lemon solver tasks to complete
    for task in all_lemon_tasks
        wait(task)
    end
end

function main()
    local args = parse_cmdargs()
    
    # Set default directories if none provided
    root_dir = dirname(Base.@__DIR__)
    if isempty(args["input_spec_files"])
        push!(args["input_spec_files"], joinpath(root_dir, "data", "specs"))
    end
    if isempty(args["config_files"])
        push!(args["config_files"], joinpath(root_dir, "configs"))
    end
    
    # Interactive input spec selection
    if haskey(args, "input_spec_files")
        args["input_spec_files"] = expand_and_select_inspecs(args["input_spec_files"])
    end

    if isempty(args["input_spec_files"])
        println("No input spec files selected. Exiting.")
        return
    end
    
    # Interactive config selection
    if haskey(args, "config_files")
        args["config_files"] = expand_and_select_configs(args["config_files"])
    end

    if isempty(args["config_files"])
        println("No configuration files selected. Exiting.")
        return
    end

    @show args

    # Print reproduction command
    repro_cmd = "julia --project=. $(relpath(PROGRAM_FILE))"
    for spec in args["input_spec_files"]
        repro_cmd *= " -i \"$(spec)\""
    end
    for config in args["config_files"]
        repro_cmd *= " -c \"$(config)\""
    end
    if !isnothing(args["output_db"])
        repro_cmd *= " -o \"$(args["output_db"])\""
    end
    if !isnothing(args["solution_dir"])
        repro_cmd *= " -s \"$(args["solution_dir"])\""
    end
    println("\nTo reproduce this run, use:")
    println(repro_cmd, "\n")

    # Run benchmarks with the parsed arguments
    run_benchmarks(args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
