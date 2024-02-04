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
    action = :append_arg
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
  CairoMakie.activate!(; px_per_unit = 10.0)
  local fontsize_theme = Theme(fontsize = 10)
  update_theme!(fontsize_theme)

  local probs = [[strip(x) for x in split(f, "=")] for f in args["outspecs"]]
  local inspecs = [strip(x) for x in args["i"]]
  local outdir = strip(args["o"])

  local spec = reduce(vcat, [CSV.read(sp, DataFrame) for sp in inspecs])
  local ospecs = Dict{String,DataFrame}()
  for (name, ospec) in probs
    local t = insertcols!(CSV.read(ospec, DataFrame), :solver => name)
    if name in keys(ospecs)
      ospecs[name] = vcat(ospecs[name], t)
    else
      ospecs[name] = t
    end
  end
  local allospecs = leftjoin(vcat(values(ospecs)...), spec; on = :name)
  @transform! allospecs begin
    :time_s_per_arc_iter = :time_s ./ (:arcs .* :iters)
    :ok_run = [st in ["Trm_Optimal", "LOCALLY_SOLVED"] for st in :status]
  end
  local allerrors = @by allospecs [:solver, :arcs] begin
    :time_s_mean = mean(:time_s)
    :time_s_per_arc_iter_mean = mean(:time_s_per_arc_iter)
    :iters_mean = mean(:iters)
    :time_s_std = std(:time_s)
    :time_s_per_arc_iter_std = std(:time_s_per_arc_iter)
    :iters_std = std(:iters)
  end

  local f = Figure(size = (600, 600))
  local pl =
    (
      data(allerrors) *
      (
        mapping(
          :arcs,
          :time_s_mean => "time_s",
          color = :solver,
          dodge = :solver,
          row = :arcs => (t -> "1"),
        ) +
        mapping(
          :arcs,
          :time_s_per_arc_iter_mean => "time_s_per_arc_iter",
          color = :solver,
          dodge = :solver,
          row = :arcs => (t -> "2"),
        ) +
        mapping(
          :arcs,
          :iters_mean => "iters",
          color = :solver,
          dodge = :solver,
          row = :arcs => (t -> "3"),
        )
      ) *
      visual(Lines; linewidth = 1)
    ) + (
      data(allospecs) *
      (
        mapping(:arcs, :time_s => "time_s"; color = :solver, row = :arcs => (t -> "1")) +
        mapping(
          :arcs,
          :time_s_per_arc_iter => "time_s_per_arc_iter";
          color = :solver,
          row = :arcs => (t -> "2"),
        ) +
        mapping(:arcs, :iters => "iters"; color = :solver, row = :arcs => (t -> "3"))
      ) *
      visual(Scatter, markersize = 2)
    )
  local plt = draw!(f, pl)
  legend!(
    f[end+1, :],
    plt,
    orientation = :horizontal,
    titleposition = :left,
    patchsize = (10, 10),
  )
  mkpath(outdir)
  local figname = [split(basename(sp), ".")[1] for sp in inspecs]
  if length(figname) == 1
    figname = figname[1]
  else
    lcp = function (xs)
      local p = ""
      while true
        if any(length(p) .>= length.(xs))
          return p
        end
        local at = length(p) + 1
        if length(unique([x[at] for x in xs])) != 1
          return p
        end
        p *= string(xs[1][at])
      end
    end
    figname = lcp(figname) * "all"
  end
  save(joinpath(outdir, figname * ".svg"), f)
  display(f)
end
main()
