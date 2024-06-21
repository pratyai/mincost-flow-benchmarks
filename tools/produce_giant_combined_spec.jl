using CSV
using DataFrames
using DataFramesMeta

function labels_from_problem_name(name::String)
  local labels = String[]
  if startswith(name, "netgen_8_")
    push!(labels, "netgen_8")
  elseif startswith(name, "netgen_sr_")
    push!(labels, "netgen_sr")
  elseif startswith(name, "netgen_deg_")
    push!(labels, "netgen_deg")
  elseif startswith(name, "gridgen_8_")
    push!(labels, "gridgen_8")
  elseif startswith(name, "gridgen_sr_")
    push!(labels, "gridgen_sr")
  elseif startswith(name, "gridgen_deg_")
    push!(labels, "gridgen_deg")
  elseif startswith(name, "goto_8_")
    push!(labels, "goto_8")
  elseif startswith(name, "goto_sr_")
    push!(labels, "goto_sr")
  elseif startswith(name, "grid_long_")
    push!(labels, "grid_long")
  elseif startswith(name, "grid_square_")
    push!(labels, "grid_square")
  elseif startswith(name, "grid_wide_")
    push!(labels, "grid_wide")
  elseif startswith(name, "spielman")
    push!(labels, "spielman")
  elseif startswith(name, "road_path")
    push!(labels, "road_path")
  elseif startswith(name, "road_flow")
    push!(labels, "road_flow")
  end
  return labels
end

function main()
  local ins = DataFrame()
  local outs = DataFrame()
  for line in eachline(stdin)
    local l = strip(line)
    @assert endswith(l, ".inspec") || endswith(l, ".outspec")
    if endswith(l, ".inspec")
      local t = CSV.read(l, DataFrame)
      append!(ins, t; cols = :union)
    else
      local t = CSV.read(l, DataFrame)
      @transform! t @astable begin
        :solver = basename(dirname(l))
      end
      append!(outs, t; cols = :union)
    end
  end

  local t = innerjoin(ins, outs, on = :name)
  t = t[!, Not(:input_file)]
  t = t[!, Not(:bytes)]
  t = t[!, Not(:status)]
  t = t[!, Not(:solution_file)]

  @transform! t begin
    :labels = join.(labels_from_problem_name.(:name), ",")
    :time_s_per_arc_per_iter = :time_s ./ (:arcs .* :iters)
    :fact_s_per_arc_per_iter = :fact_s ./ (:arcs .* :iters)
    :solv_s_per_arc_per_iter = :solv_s ./ (:arcs .* :iters)
    :sddm_calls_per_iter = :sddm_calls ./ :iters
    :solv_s_per_arc_per_sddm_call = :solv_s ./ (:arcs .* :sddm_calls)
  end

  CSV.write(stdout, t)
end
main()
