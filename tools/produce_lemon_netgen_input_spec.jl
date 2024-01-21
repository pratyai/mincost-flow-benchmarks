using FromFile
using ArgParse
using CSV
using DataFrames
using Downloads

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
    nargs = '+'
    required = true
  end
  return parse_args(s)
end

const NETGEN_URL = "http://lime.cs.elte.hu/~kpeter/data/mcf/netgen/"

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

  local probnames = String[]
  local probpaths = String[]
  local probbytes = Int[]

  for p in probs
    local probname = split(basename(p), ".")[1]
    local probpath = nothing

    # first handle the corner cases
    if !isnothing(probdir)
      # create dir if nonexistent
      mkpath(probdir)
      local actp = nothing
      # try to find a few other options
      for ap in [p, p * ".min", p * ".min.gz"]
        if isfile(joinpath(probdir, ap))
          actp = joinpath(probdir, ap)
          break
        end
      end
      if isnothing(actp)
        # download as a last resort
        local dfrm = joinpath(NETGEN_URL, probname * ".min.gz")
        probpath = joinpath(probdir, probname * ".min.gz")
        println("downloading: ", dfrm, " to ", probpath)
        local tmpat = Downloads.download(dfrm)
        mv(tmpat, probpath)
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
    push!(probnames, probname)
    push!(probpaths, probpath)
    push!(probbytes, lstat(probpath).size)
  end

  local out = DataFrame(name = probnames, input_file = probpaths, bytes = probbytes)
  if !isnothing(specfile)
    mkpath(dirname(specfile))
    CSV.write(specfile, out)
  else
    @show out
  end
end
main()
