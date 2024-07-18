using FromFile

using Dimacs
using ArgParse
using CSV
using DataFrames
using Downloads
using MatrixMarket
using SparseArrays

function parse_cmdargs()
  s = ArgParseSettings()
  @add_arg_table s begin
    "-o"
    help = "path to store the input spec (if empty, will print on stdout)"
    arg_type = Union{Nothing,String}
    required = false
    default = nothing
    "-p"
    help = "path to find / store the downloaded problems (if empty, will not download anything; otherwise, will download only when the problem is missing)"
    arg_type = Union{Nothing,String}
    required = false
    default = nothing
    "problems"
    help = "list of the paths (or just names of `-p` is provided) of the problems we want to cover in the input spec."
    arg_type = String
    nargs = '*'
    required = false
  end
  return parse_args(s)
end

function main()
  local args = parse_cmdargs()
  @show args

  local probs = String[strip(f) for f in args["problems"]]
  local probdir = args["p"]
  if !isnothing(probdir)
    probdir = strip(probdir)
  end
  local specfile = args["o"]
  if !isnothing(specfile)
    specfile = strip(specfile)
  end

  local out = DataFrame(
    name = String[],
    input_file = String[],
    bytes = Int[],
    vertices = Int[],
    arcs = Int[],
  )
  allowmissing!(out)
  if !isnothing(specfile) && isfile(specfile)
    # use if the specfile already exists to avoid downloading unnecessary data.
    local t = CSV.read(specfile, DataFrame)
    append!(out, t; cols = :subset)
  end
  for p in probs
    # clear/insert the rows which are explicitly requested for
    local actp = chopsuffix(p, ".mm")
    if actp in out[:, :name]
      out[out.name.==actp, :input_file] .= missing
    else
      push!(out, Dict(:name => actp); cols = :subset)
    end
  end
  @assert nrow(out) > 0

  for pr in eachrow(out)
    local p = pr[:name]
    if !ismissing(pr[:input_file]) && isfile(pr[:input_file])
      continue
    end
    local probname = p
    local probpath = nothing

    # first handle the corner cases
    if !isnothing(probdir)
      local actp = nothing
      # try to find a few other options
      for ap in [p, p * ".mcf"]
        if isfile(joinpath(probdir, ap))
          actp = joinpath(probdir, ap)
          break
        end
      end
      if isnothing(actp)
        continue
      else
        probpath = actp
      end
    else
      if !isfile(p)
        # the file is missing and we are not supposed to download it
        continue
      else
        probpath = p
      end
    end

    @assert !isnothing(probpath)
    @assert probname in out[:, :name]
    println("processing ", probname, " : ", probpath)
    local netw = Dimacs.ReadDimacs(probpath)
    out[out.name.==probname, :input_file] .= probpath
    out[out.name.==probname, :bytes] .= lstat(probpath).size
    out[out.name.==probname, :vertices] .= netw.G.n
    out[out.name.==probname, :arcs] .= netw.G.m

    if !isnothing(specfile)
      mkpath(dirname(specfile))
      CSV.write(specfile, out)
    else
      @show out
    end
  end

  if !isnothing(specfile)
    mkpath(dirname(specfile))
    CSV.write(specfile, out)
  else
    @show out
  end
end
main()
