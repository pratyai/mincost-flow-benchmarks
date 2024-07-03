using FromFile
using Dimacs
@from "Integration.jl" import Integration

using ArgParse
using StatProfilerHTML

using CSV
using DataFrames
using JLD2
using MultiFloats
using TimerOutputs
using Statistics

UseFloatType = Float64

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
      local lp = Integration.construct_tulip_model(netw, UseFloatType)
      local lp, status, iters, seconds = Integration.solve_tulip_model(lp)

      # Throw away everything because this is just an warmup.
    end
  end

  local probspec = CSV.read(input_spec, DataFrame)
  local out = DataFrame(
    name = String[],
    status = String[],
    time_s = Float64[],
    fact_s = Float64[],
    solv_s = Float64[],
    iters = Int[],
    sddm_calls = Int[],
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

    local indimacs::String = r[:input_file]
    local netw = Dimacs.ReadDimacs(indimacs)

    local ntrials = 1
    if netw.G.m < 1000
      ntrials = 11
    elseif netw.G.m < 10000
      ntrials = 3
    end
    local trials = []
    for ntri = 1:ntrials
      GC.gc()
      local lp = Integration.construct_tulip_model(netw, UseFloatType)
      local lp, status, iters, seconds = Integration.solve_tulip_model(lp)
      push!(trials, (lp.solver.timer, status, iters, seconds))
    end
    local medt = median([t[3] for t in trials])
    local medidx = findfirst(t -> t[3] == medt, trials)
    local to, status, iters, seconds = trials[medidx]

    local fact_ns = TimerOutputs.time(to["Main loop"]["Step"]["Factorization"])
    local solv_ns = TimerOutputs.time(to["Main loop"]["Step"]["Newton"]["KKT"])
    local sddm_calls = TimerOutputs.ncalls(to["Main loop"]["Step"]["Newton"]["KKT"])

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
        :fact_s => fact_ns * 1e-9,
        :solv_s => solv_ns * 1e-9,
        :iters => iters,
        :sddm_calls => sddm_calls,
        :solution_file => sol_file,
      );
      promote = true,
    )
    if !isnothing(output_spec)
      mkpath(dirname(output_spec))
      CSV.write(output_spec, sort(out, [:name]))
    else
      @show out
    end
  end

  if !isnothing(output_spec)
    mkpath(dirname(output_spec))
    CSV.write(output_spec, sort(out, [:name]))
  else
    @show out
  end
end
main()
