using FromFile

using ArgParse
using CSV
using DataFrames
using CairoMakie
using ColorSchemes
using AlgebraOfGraphics
using Statistics
using DataFramesMeta
using DataStructures
using Statistics

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
  local fontsize_theme = Theme(fontsize = 10)
  update_theme!(fontsize_theme)
  # local color_theme = Theme(palette = (color = ColorSchemes.tol_medcontrast,))
  # update_theme!(color_theme)

  local probs = [[strip(x) for x in split(f, "=")] for f in args["outspecs"]]
  local inspec = strip(args["i"])
  local outdir = strip(args["o"])

  local spec = CSV.read(inspec, DataFrame)
  local ospecs = Dict(name => CSV.read(ospec, DataFrame) for (name, ospec) in probs)

  local f = Figure()
  for (axid, metric) in enumerate([:time_s, :iters])
    local ax = Axis(f[axid, 1], xlabel = "#arcs", ylabel = string(metric))
    local legentries = []
    for (i, (name, ospec)) in enumerate(ospecs)
      local succ = st -> st in ["Trm_Optimal", "LOCALLY_SOLVED"]
      local notsucc = st -> !succ(st)
      local good = innerjoin(spec, ospec[succ.(ospec.status), [:name, metric]], on = :name)
      local bad =
        innerjoin(spec, ospec[notsucc.(ospec.status), [:name, metric]], on = :name)
      local sc1 = scatter!(
        ax,
        good[:, :arcs],
        good[:, metric],
        marker = :circle,
        markersize = 5,
        color = Cycled(i),
      )
      local sc2 = scatter!(
        ax,
        bad[:, :arcs],
        bad[:, metric],
        marker = :xcross,
        markersize = 5,
        color = Cycled(i),
        strokecolor = :red,
        strokewidth = 0.8,
      )
      local meangood = groupby(good, :arcs)
      meangood = combine(meangood, metric => mean => metric)
      local ln1 = lines!(
        ax,
        meangood[:, :arcs],
        meangood[:, metric],
        color = Cycled(i),
        linestyle = :dash,
        linewidth = 1,
      )
      push!(legentries, (name, [sc1]))
    end
    f[axid, 2] = Legend(
      f,
      [en[2] for en in legentries],
      [en[1] for en in legentries],
      framevisible = false,
    )
  end
  mkpath(outdir)
  save(joinpath(outdir, split(basename(inspec), ".")[1] * ".svg"), f)
  display(f)
end
main()
