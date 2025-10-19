using ArgParse
using CSV
using DataFrames
using JLD2
using SQLite
using TOML

using Dimacs

# Solver modules
include("solvers/common.jl")
include("solvers/TulipApproxChol.jl")
include("solvers/TulipCHOLMOD.jl")

const SOLVERS = Dict(
    "tulip_approxchol" => TulipApproxChol,
    "tulip_cholmod" => TulipCHOLMOD,
)

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
  SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS configs (
        id INTEGER PRIMARY KEY,
        config_text TEXT UNIQUE
    )
  """)
  SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY,
        name TEXT,
        status TEXT,
        solver_name TEXT,
        config_id INTEGER,
        time_s REAL,
        iters INTEGER,
        solution_file TEXT,
        fact_s REAL,
        solv_s REAL,
        sddm_calls INTEGER,
        FOREIGN KEY (config_id) REFERENCES configs(id)
    )
  """)
  SQLite.execute(db, """
    CREATE TABLE IF NOT EXISTS solver_history (
        id INTEGER PRIMARY KEY,
        run_id INTEGER,
        ipm_iter INTEGER,
        solve_in_iter INTEGER,
        residual_norm REAL,
        FOREIGN KEY (run_id) REFERENCES runs(id)
    )
  """)

  for config_file in config_files
    config_text = read(config_file, String)
    config = TOML.parse(config_text)
    solver_name = config["solver"]
    if !haskey(SOLVERS, solver_name)
        println("Error: Solver `", solver_name, "` from config file `", config_file, "` not found.")
        continue
    end
    solver = SOLVERS[solver_name]

    # Get or create config_id
    query = DBInterface.execute(db, "SELECT id FROM configs WHERE config_text = ?", (config_text,))
    config_id = missing
    for row in query
        config_id = row.id
    end

    if ismissing(config_id)
        DBInterface.execute(db, "INSERT INTO configs (config_text) VALUES (?)", (config_text,))
        config_id = SQLite.last_insert_rowid(db)
    end

    for r in eachrow(probspec)
      println("processing: ", r[:name], " with config: ", config_file)

      # Check if result already exists
      query = DBInterface.execute(db, "SELECT id FROM runs WHERE name = ? AND config_id = ?", (r[:name], config_id))
      if !isempty(query)
        println("record already exists for `", r[:name], "` with config from `", config_file, "`; skipping it")
        continue
      end

      local indimacs::String = joinpath(dirname(Base.@__DIR__), r[:input_file])
      local netw = Dimacs.ReadDimacs(indimacs)
      
      local results = solver.solve(netw, config)

      # Save solution if asked for.
      local sol_file::Union{String, Missing} = missing
      if !isnothing(solution_dir)
        mkpath(solution_dir)
        sol_file = joinpath(solution_dir, r[:name] * "_" * splitext(basename(config_file))[1] * ".jld2")
        jldsave(sol_file, true; x = results.solution)
      end

      # Insert main results
      fact_s = haskey(results, :fact_s) ? results.fact_s : missing
      solv_s = haskey(results, :solv_s) ? results.solv_s : missing
      sddm_calls = haskey(results, :sddm_calls) ? results.sddm_calls : missing

      DBInterface.execute(db, """
        INSERT INTO runs (name, status, solver_name, config_id, time_s, iters, solution_file, fact_s, solv_s, sddm_calls)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """, (r[:name], String(Symbol(results.status)), solver_name, config_id, results.seconds, results.iters, sol_file, fact_s, solv_s, sddm_calls))

      local run_id = SQLite.last_insert_rowid(db)

      # Insert history
      if haskey(results, :residual_history) && !ismissing(results.residual_history)
        for (ipm_iter, solve_in_iter, residual) in results.residual_history
          DBInterface.execute(db, "INSERT INTO solver_history (run_id, ipm_iter, solve_in_iter, residual_norm) VALUES (?, ?, ?, ?)", (run_id, ipm_iter, solve_in_iter, residual))
        end
      end
    end
  end
end

main()
