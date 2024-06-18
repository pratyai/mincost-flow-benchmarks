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
      for ap in [p, p * ".min.gz"]
        if isfile(joinpath(probdir, ap))
          actp = joinpath(probdir, ap)
          break
        end
      end
      if isnothing(actp)
        # no auto-downloading with this one
        local mmpath = joinpath(probdir, p * ".mm")
        @assert isfile(mmpath)
        local A::SparseMatrixCSC = MatrixMarket.mmread(mmpath)
        # but then we have to produce a min-cost flow on it
        local mingz_path = joinpath(probdir, probname * ".min.gz")
        local n = size(A, 1)
        # generate edges in both directions
        local us::Vector{Int}, vs::Vector{Int} = [], []
        local rows, vals = rowvals(A), nonzeros(A)
        for u = 1:n
          for k in nzrange(A, u)
            local v = rows[k]
            local w = vals[k]
            if u != v && !iszero(w)
              push!(us, u)
              push!(vs, v)
            end
          end
        end
        local G = Dimacs.FromEdgeList(n, hcat(us, vs))
        # try to match the lemon dataset
        local caps::Vector{Int}, costs::Vector{Int} = rand(Int, G.m), rand(Int, G.m)
        caps = (abs.(caps) .% 1000) .+ 1
        costs = (abs.(costs) .% 10000) .+ 1
        local demands::Vector{Int} = zeros(Int, n)
        local netD = G.n * 1000
        local tops = sum(caps)
        local floAssignments = abs.(rand(Int, G.m)) .% (min.(caps, tops) .+ 1)
        local demAssignments = G.IncidenceMatrix * floAssignments
        local att = 0
        while sum(abs.(demAssignments)) > netD && tops > 0
          floAssignments .= abs.(rand(Int, G.m)) .% (min.(caps, tops) .+ 1)
          demAssignments .= G.IncidenceMatrix * floAssignments
          att += 1
          tops = Int(floor(tops / 2))
        end
        @show att, tops, maximum(abs.(demAssignments)), maximum(floAssignments)
        demands .= demAssignments
        local netw = Dimacs.McfpNet(G = G, Cost = costs, Cap = caps, Demand = demands)
        @assert all(netw.G.IncidenceMatrix * floAssignments .== netw.Demand)
        @assert all(floAssignments .>= 0)
        @assert all(floAssignments .<= netw.Cap)

        netw.Cap .= 1
        netw.Cost .= 1
        netw.Demand .= 0
        local degs = sum(netw.G.AdjacencyMatrix; dims = 1)
        local mdegs = maximum(degs)
        local times = 0
        for u = 1:netw.G.n
          if degs[u] == 2
            continue
          end
          if degs[u] == mdegs
            if times == 0
              netw.Demand[u] = mdegs / 2
            else
              netw.Demand[u] = -mdegs / 2
            end
            times += 1
          end
        end
        @assert times == 2

        Dimacs.WriteDimacs(mingz_path, netw)
        probpath = mingz_path
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
  end

  if !isnothing(specfile)
    mkpath(dirname(specfile))
    CSV.write(specfile, out)
  else
    @show out
  end
end
main()
