using ArgParse
using CSV
using DataFrames
using JLD2



using Dimacs

# Solver modules
include("solvers/common.jl")
include("solvers/TulipBasic.jl")
include("solvers/TulipApproxChol.jl")
include("solvers/TulipCHOLMOD.jl")

const SOLVERS = Dict(
    "tulip_basic" => TulipBasic,
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
    help = "path to store the output spec (if absent, will print on stdout)"
    arg_type = Union{Nothing,String}
    required = false
    default = nothing
    "-s"
    help = "output solution flow-vectors directory (if absent, will not store solutions)"
    arg_type = Union{Nothing,String}
    required = false
    default = nothing
    "--solver"
    help = "solver to use"
    arg_type = String
    nargs = '+'
    required = true
  end
  return parse_args(s)
end

function main()
  local args = parse_cmdargs()
  @show args

  local solver_names = args["solver"]
  for solver_name in solver_names
    if !haskey(SOLVERS, solver_name)
      println("Error: Solver `", solver_name, "` not found.")
      return
    end
  end

  local input_spec = args["i"]
  if !isnothing(input_spec)
    input_spec = strip(input_spec)
  end
  local output_spec = args["o"]
  if !isnothing(output_spec)
    output_spec = strip(output_spec)
  end
  local solution_dir = args["s"]
  if !isnothing(solution_dir)
    solution_dir = strip(solution_dir)
  end

  local probspec = CSV.read(input_spec, DataFrame)
  local out = DataFrame(
    name = String[],
    status = String[],
    solver_name = String[],
    solver_config = String[],
    time_s = Float64[],
    iters = Int[],
    solution_file = Union{String, Missing}[],
    fact_s = Union{Float64, Missing}[],
    solv_s = Union{Float64, Missing}[],
    sddm_calls = Union{Int, Missing}[],
  )

  if !isnothing(output_spec) && isfile(output_spec)
    try
      existing_out = CSV.read(output_spec, DataFrame)
      out = vcat(out, existing_out, cols=:union)
    catch e
      println(
        "output spec `",
        output_spec,
        "` already exits but cannot be read as a table: ",
        e,
      )
    end
  end

  for solver_name in solver_names
    local solver = SOLVERS[solver_name]
    local solver_config = solver.get_config_string()

    for r in eachrow(probspec)
      println("processing: ", r[:name], " with solver: ", solver_name)

      # Check if result already exists
      if !isempty(out) && any(row -> row.name == r[:name] && row.solver_name == solver_name && row.solver_config == solver_config, eachrow(out))
        println("record already exists for `", r[:name], "`, solver `", solver_name, "`, and config `", solver_config, "`; skipping it")
        continue
      end

      local indimacs::String = joinpath(dirname(Base.@__DIR__), r[:input_file])
      local netw = Dimacs.ReadDimacs(indimacs)
      
      local results = solver.solve(netw)

      # Save solution if asked for.
      local sol_file::Union{String, Missing} = missing
      if !isnothing(solution_dir)
        mkpath(solution_dir)
        sol_file = joinpath(solution_dir, r[:name] * "_" * solver_name * ".jld2")
        jldsave(sol_file, true; x = results.solution)
      end

      local row_data = Dict(
        :name => r[:name],
        :status => String(Symbol(results.status)),
        :solver_name => solver_name,
        :solver_config => solver_config,
        :time_s => results.seconds,
        :iters => results.iters,
        :solution_file => sol_file,
        :fact_s => missing,
        :solv_s => missing,
        :sddm_calls => missing,
      )
      if solver_name == "tulip_approxchol"
        row_data[:fact_s] = results.fact_s
        row_data[:solv_s] = results.solv_s
        row_data[:sddm_calls] = results.sddm_calls
      end

      push!(out, row_data, cols=:union)

      if !isnothing(output_spec)
        mkpath(dirname(output_spec))
        CSV.write(output_spec, out)
      end
    end
  end

  if isnothing(output_spec)
    @show out
  end
end

main()
