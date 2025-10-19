"""
This module defines a basic Tulip solver. It uses the default KKT solver.
"""
module TulipBasic

using Tulip
using Dimacs
using SparseArrays
using LinearAlgebra
using Random

include("common.jl")
using .SolverCommon

const CONFIG = Dict(
    "IPM_PRegMin" => 1e-6,
    "IPM_DRegMin" => 1e-6,
)

function construct_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv}) where {Tv<:Number}
  lp = SolverCommon.create_tulip_model(netw, Tv)

  Tulip.set_parameter(lp, "OutputLevel", 0)  # disable output
  Tulip.set_parameter(lp, "Presolve_Level", 0)  # disable presolve
  for (k, v) in CONFIG
    Tulip.set_parameter(lp, k, Tv(v))
  end
  return lp
end

"""
    solve(netw::Dimacs.McfpNet)

Solve a minimum cost flow problem using the TulipBasic solver.

# Arguments
- `netw::Dimacs.McfpNet`: The minimum cost flow problem to solve.

# Returns
- A named tuple with the solver status, number of iterations, solution time, and solution vector.
"""
function solve(netw::Dimacs.McfpNet)
  lp = construct_tulip_model(netw, Float64)
  Tulip.optimize!(lp)

  status = Tulip.get_attribute(lp, Tulip.Status())
  iters = Tulip.get_attribute(lp, Tulip.BarrierIterations())
  seconds = Tulip.get_attribute(lp, Tulip.SolutionTime())
  solution = lp.solution.x

  return (; status, iters, seconds, solution)
end

"""
    get_config_string() -> String

Get a string representation of the solver's configuration.
"""
function get_config_string()
  return "KKT_Backend=TlpCholmod, " * join(["$k=$v" for (k, v) in CONFIG], ", ")
end

end # module TulipBasic