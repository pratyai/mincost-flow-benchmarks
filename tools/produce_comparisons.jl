using FromFile
@from "Dimacs.jl" import Dimacs

using ArgParse
using CSV
using DataFrames
using CairoMakie

function parse_cmdargs()
  s = ArgParseSettings()
  @add_arg_table s begin
    "-o"
    help = "directory to store the produced artefacts"
    arg_type = String
    required = true
    "-i"
    help = "the input spec (entries in the output spec outside the input spec will be ignored)"
    arg_type = String
    required = true
    "outspecs"
    help = "list of <name>=<outspec path>"
    arg_type = String
    nargs = '+'
    required = true
  end
  return parse_args(s)
end

function main()
  local args = parse_cmdargs()
  @show args

  CairoMakie.activate!(; px_per_unit = 10.0)

  local probs = [[strip(x) for x in split(f, "=")] for f in args["outspecs"]]
  local inspec = strip(args["i"])
  local outdir = strip(args["o"])

  local spec = CSV.read(inspec, DataFrame)
  local ospecs = Dict(name => CSV.read(ospec, DataFrame) for (name, ospec) in probs)

  local f = Figure()
  for (axid, metric) in enumerate([:time_s, :iters])
    local ax = Axis(f[axid, 1], xlabel = "#arcs", ylabel = string(metric))
    for (i, (name, ospec)) in enumerate(ospecs)
      local succ = st -> st in ["Trm_Optimal", "LOCALLY_SOLVED"]
      local notsucc = st -> !succ(st)
      local good = innerjoin(spec, ospec[succ.(ospec.status), [:name, metric]], on = :name)
      local bad =
        innerjoin(spec, ospec[notsucc.(ospec.status), [:name, metric]], on = :name)
      scatter!(
        ax,
        good[:, :arcs],
        good[:, metric],
        label = name,
        marker = :circle,
        markersize = 5,
        alpha = 0.5,
        color = Cycled(i),
      )
      scatter!(
        ax,
        bad[:, :arcs],
        bad[:, metric],
        marker = :circle,
        markersize = 5,
        alpha = 0.5,
        color = Cycled(i),
        strokecolor = :red,
        strokewidth = 0.5,
      )
    end
    axislegend(ax; position = :lt, nbanks = 4, backgroundcolor = :transparent)
  end
  mkpath(outdir)
  save(joinpath(outdir, split(basename(inspec), ".")[1] * ".svg"), f)
  display(f)
end
main()
