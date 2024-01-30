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

  set_aog_theme!()
  CairoMakie.activate!(type = "svg"; px_per_unit = 10.0)
  local fontsize_theme = Theme(fontsize = 6)
  update_theme!(fontsize_theme)

  local probs = [[strip(x) for x in split(f, "=")] for f in args["outspecs"]]
  local inspec = strip(args["i"])
  local outdir = strip(args["o"])

  local spec = CSV.read(inspec, DataFrame)
  local ospecs = Dict(
    name => insertcols!(CSV.read(ospec, DataFrame), :solver => name) for
    (name, ospec) in probs
  )
  local allospecs = leftjoin(vcat(values(ospecs)...), spec; on = :name)
  @transform! allospecs begin
    :time_s_per_arc_iter = :time_s ./ (:arcs .* :iters)
    :ok_run = [st in ["Trm_Optimal", "LOCALLY_SOLVED"] for st in :status]
  end

  local f = Figure()
  local bar =
    data(allospecs) *
    (
      mapping(
        :name,
        :time_s_per_arc_iter;
        color = :solver,
        dodge = :solver,
        row = :arcs => (t -> "1"),
      ) +
      mapping(:name, :iters; color = :solver, dodge = :solver, row = :arcs => (t -> "2"))
    ) *
    visual(BarPlot)
  local plt = draw!(
    f,
    bar;
    axis = (; xlabel = "", xticklabelrotation = pi / 6),
    facet = (; linkyaxes = :none),
  )
  legend!(
    f[end+1, :],
    plt,
    orientation = :horizontal,
    titleposition = :left,
    patchsize = (10, 10),
  )
  display(f)
  mkpath(outdir)
  save(joinpath(outdir, split(basename(inspec), ".")[1] * ".svg"), f)
end
main()
