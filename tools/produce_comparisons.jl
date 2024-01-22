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

  local probs = [[strip(x) for x in split(f, "=")] for f in args["outspecs"]]
  local inspec = strip(args["i"])
  local outdir = strip(args["o"])

  local spec = CSV.read(inspec, DataFrame)
  local ospecs = Dict(name => CSV.read(ospec, DataFrame) for (name, ospec) in probs)

  for metric in [:time_s]
    local f = Figure()
    local ax = Axis(f[1, 1], xlabel = "#arcs", ylabel = string(metric), yscale = log10)
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
        color = Cycled(i),
      )
      scatter!(
        ax,
        bad[:, :arcs],
        bad[:, metric],
        marker = :xcross,
        markersize = 8,
        color = Cycled(i),
      )
    end
    axislegend()
    display(f)
  end
end
main()
