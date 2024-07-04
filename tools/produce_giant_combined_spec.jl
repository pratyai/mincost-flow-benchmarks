using CSV
using DataFrames
using DataFramesMeta
using Scanf

function baseline_from_file_name(name::AbstractString)
  return !occursin("alt", name)
end

function floatbits_from_file_name(name::AbstractString)
  if occursin("f64", name)
    return 64
  elseif occursin("f128", name)
    return 128
  end
  return 64
end

function epcg_from_file_name(name::AbstractString)
  if occursin("pcg=", name)
    local pos = findfirst("pcg=", name)
    local r, pcg = @scanf(name[pos.stop+1:end], "%lf", Float64)
    @assert r == 1
    return pcg
  end
  return missing
end

function rhos_from_file_name(name::AbstractString)
  if occursin("rhos=", name)
    local pos = findfirst("rhos=", name)
    local r, rhop, rhod = @scanf(name[pos.stop+1:end], "%lf,%lf", Float64, Float64)
    @assert r == 2
    return rhop, rhod
  end
  return missing, missing
end

function probclass_from_problem_name(name::AbstractString)
  if startswith(name, "netgen_8_")
    return "netgen_8"
  elseif startswith(name, "netgen_sr_")
    return "netgen_sr"
  elseif startswith(name, "netgen_deg_")
    return "netgen_deg"
  elseif startswith(name, "gridgen_8_")
    return "gridgen_8"
  elseif startswith(name, "gridgen_sr_")
    return "gridgen_sr"
  elseif startswith(name, "gridgen_deg_")
    return "gridgen_deg"
  elseif startswith(name, "goto_8_")
    return "goto_8"
  elseif startswith(name, "goto_sr_")
    return "goto_sr"
  elseif startswith(name, "grid_long_")
    return "grid_long"
  elseif startswith(name, "grid_square_")
    return "grid_square"
  elseif startswith(name, "grid_wide_")
    return "grid_wide"
  elseif startswith(name, "spielman")
    return "spielman"
  elseif startswith(name, "road_path")
    return "road_path"
  elseif startswith(name, "road_flow")
    return "road_flow"
  elseif startswith(name, "vision_rnd")
    return "vision_rnd"
  elseif startswith(name, "vision_prop")
    return "vision_prop"
  elseif startswith(name, "vision_inv")
    return "vision_inv"
  end
  return missing
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
      local fbits = floatbits_from_file_name(l)
      local epcg = epcg_from_file_name(l)
      local rhop, rhod = rhos_from_file_name(l)
      local baseline = baseline_from_file_name(l)
      @transform! t @astable begin
        :solver = basename(dirname(l))
        :floatbits = fbits
        :epcg = epcg
        :rhop = rhop
        :rhod = rhod
        :baseline = baseline
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
    :probclass = probclass_from_problem_name.(:name)
    :time_s_per_arc_per_iter = :time_s ./ (:arcs .* :iters)
    :fact_s_per_arc_per_iter = :fact_s ./ (:arcs .* :iters)
    :solv_s_per_arc_per_iter = :solv_s ./ (:arcs .* :iters)
    :sddm_calls_per_iter = :sddm_calls ./ :iters
    :solv_s_per_arc_per_sddm_call = :solv_s ./ (:arcs .* :sddm_calls)
  end
  sort!(t, [:solver, :name])
  select!(t, :solver, :name, Not([:solver, :name]))

  CSV.write(stdout, t)
end
main()
