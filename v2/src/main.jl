using ArgParse
using CSV
using DataFrames
using JLD2



using Dimacs

# Solver modules
include("solvers/TulipBasic.jl")
include("solvers/TulipApproxChol.jl")

const SOLVERS = Dict(
    "tulip_basic" => TulipBasic,
    "tulip_approxchol" => TulipApproxChol,
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
    required = true
  end
  return parse_args(s)
end

function main()
  local args = parse_cmdargs()
  @show args

  local solver_name = args["solver"]
  if !haskey(SOLVERS, solver_name)
    println("Error: Solver `", solver_name, "` not found.")
    return
  end
  local solver = SOLVERS[solver_name]

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
    time_s = Float64[],
    iters = Int[],
    solution_file = String[],
  )

  if !isnothing(output_spec) && isfile(output_spec)
    try
      out = CSV.read(output_spec, DataFrame)
    catch e
      println(
        "output spec `",
        output_spec,
        "` already exits but cannot be read as a table: ",
        e,
      )
    end
  end

  for r in eachrow(probspec)
    println("processing: ", r[:name], " => ", r[:input_file])
    if r[:name] in out[:, :name]
      println("record already exists for `", r[:name], "`; skipping it")
      continue
    end

    local indimacs::String = joinpath(dirname(Base.@__DIR__), r[:input_file])
    local netw = Dimacs.ReadDimacs(indimacs)
    
    local results = solver.solve(netw)

    # Save solution if asked for.
    local sol_file = ""
    if !isnothing(solution_dir)
      mkpath(solution_dir)
      sol_file = joinpath(solution_dir, r[:name] * ".jld2")
      jldsave(sol_file, true; x = results.solution)
    end

    push!(
      out,
      Dict(
        :name => r[:name],
        :status => String(Symbol(results.status)),
        :time_s => results.seconds,
        :iters => results.iters,
        :solution_file => sol_file,
      );
      promote = true
    )
    if !isnothing(output_spec)
      mkpath(dirname(output_spec))
      CSV.write(output_spec, out)
    end
  end

  if !isnothing(output_spec)
    mkpath(dirname(output_spec))
    CSV.write(output_spec, out)
  else
    @show out
  end
end

main()