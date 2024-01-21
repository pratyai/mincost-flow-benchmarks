using FromFile
@from "Dimacs.jl" import Dimacs
@from "Integration.jl" import Integration

using ArgParse
using StatProfilerHTML

using CSV
using DataFrames
using JLD2

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
    "-t"
    help = "warmup spec file"
    arg_type = Union{Nothing,String}
    required = false
    default = nothing
  end
  return parse_args(s)
end

function main()
  local args = parse_cmdargs()
  @show args

  local warmup_spec = args["t"]
  if !isnothing(warmup_spec)
    warmup_spec = strip(warmup_spec)
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


  if !isnothing(warmup_spec)
    local warmspec = CSV.read(warmup_spec, DataFrame)
    for r in eachrow(warmspec)
      println("warmup: ", r[:name], " => ", r[:input_file])

      local indimacs::String = r[:input_file]
      local netw = Dimacs.ReadDimacs(indimacs)
      local lp = Integration.construct_tulip_model(netw, Float64)
      local lp, status, iters, seconds = Integration.solve_tulip_model(lp)

      # Throw away everything because this is just an warmup.
    end
  end

  local probspec = CSV.read(input_spec, DataFrame)
  local out = DataFrame(
    name = String[],
    status = String[],
    time_s = Float64[],
    iters = Int[],
    solution_file = String[],
  )
  for r in eachrow(probspec)
    println("processing: ", r[:name], " => ", r[:input_file])

    local indimacs::String = r[:input_file]
    local netw = Dimacs.ReadDimacs(indimacs)
    local lp = Integration.construct_tulip_model(netw, Float64)
    local lp, status, iters, seconds = Integration.solve_tulip_model(lp)

    # Save solution if asked for.
    local sol_file = ""
    if !isnothing(solution_dir)
      mkpath(solution_dir)
      sol_file = joinpath(solution_dir, r[:name] * ".jld2")
      jldsave(sol_file, true; x = lp.solution.x)
    end

    push!(
      out,
      Dict(
        :name => r[:name],
        :status => String(Symbol(status)),
        :time_s => seconds,
        :iters => iters,
        :solution_file => sol_file,
      ),
    )
  end

  if !isnothing(output_spec)
    mkpath(dirname(output_spec))
    CSV.write(output_spec, out)
  else
    @show out
  end
end
main()
