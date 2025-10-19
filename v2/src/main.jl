using ArgParse
using CSV
using DataFrames
using JLD2
using SQLite
using TOML
using Debugger

using Dimacs

# Solver modules
include("solvers/common.jl")
include("solvers/TulipApproxChol.jl")
include("solvers/TulipCHOLMOD.jl")

const SOLVERS = Dict("tulip_approxchol" => TulipApproxChol, "tulip_cholmod" => TulipCHOLMOD)

function parse_cmdargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        "-i"
        help = "input spec file"
        arg_type = String
        required = true
        "-o"
        help = "path to store the output database (if absent, will use benchmarks.db)"
        arg_type = String
        default = "benchmarks.db"
        "-s"
        help = "output solution flow-vectors directory (if absent, will not store solutions)"
        arg_type = Union{Nothing,String}
        required = false
        default = nothing
        "--configs"
        help = "paths to solver config files"
        arg_type = String
        nargs = '+'
        required = true
    end
    return parse_args(s)
end

function main()
    local args = parse_cmdargs()
    @show args

    local config_files = args["configs"]

    local input_spec = args["i"]
    if !isnothing(input_spec)
        input_spec = strip(input_spec)
    end
    local output_db = args["o"]
    if !isnothing(output_db)
        output_db = strip(output_db)
    end
    local solution_dir = args["s"]
    if !isnothing(solution_dir)
        solution_dir = strip(solution_dir)
    end

    local probspec = CSV.read(input_spec, DataFrame)

    # Setup database
    db = SQLite.DB(output_db)
    SQLite.execute(
        db,
        """
  CREATE TABLE IF NOT EXISTS problems (
      id INTEGER PRIMARY KEY,
      name TEXT UNIQUE,
      input_file TEXT,
      bytes INTEGER
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

    # Populate problems table
    problem_ids = Dict{String,Int}()
    for r in eachrow(probspec)
        query =
            DBInterface.execute(db, "SELECT id FROM problems WHERE name = ?", (r[:name],))
        problem_id = missing
        for row in query
            problem_id = row.id
        end

        if ismissing(problem_id)
            DBInterface.execute(
                db,
                "INSERT INTO problems (name, input_file, bytes) VALUES (?, ?, ?)",
                (r[:name], r[:input_file], r[:bytes]),
            )
            problem_id = SQLite.last_insert_rowid(db)
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

            DBInterface.execute(
                db,
                """
  INSERT INTO runs (name, status, solver_name, config_id, problem_id, time_s, iters, solution_file, fact_s, solv_s, sddm_calls)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) 
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
end # Closes 'function main()'

main()
