module Integration

using FromFile
@from "Dimacs.jl" import Dimacs

using SparseArrays
using LinearAlgebra
using MultiFloats
using DoubleFloats
using Tulip
using Random

function construct_tulip_model(netw::Dimacs.McfpNet, _::Type{Tv}) where {Tv<:Number}
  Random.seed!(1)
  local A = sparse(Int8.(netw.G.IncidenceMatrix))
  local b = Tv.(netw.Demand)
  local u = Tv.(netw.Cap)
  local c = Tv.(netw.Cost)
  local n, m = size(A)

  local lp = Tulip.Model{Tv}()
  local pb = lp.pbdata
  Tulip.load_problem!(
    pb,
    "mcfp",  # some arbitrary name
    true,  # true := minimize
    c,  # objective vector := cost
    zero(Tv),  # no constant term in the cost
    A,  # equality constraint matrix
    b,  # equality constraints := up = down = b
    b,
    zero(u),  # box constraints := lower bound = 0, upper bound = u
    u,
    repeat([""], n),  # leave empty strings as variable and constraint names
    repeat([""], m),
  )

  Tulip.set_parameter(lp, "OutputLevel", 1)  # enable output

  return lp
end

function solve_tulip_model(lp::Tulip.Model{Tv}) where {Tv<:Number}
  Tulip.optimize!(lp)

  local status = Tulip.get_attribute(lp, Tulip.Status())
  local iters = Tulip.get_attribute(lp, Tulip.BarrierIterations())
  local seconds = Tulip.get_attribute(lp, Tulip.SolutionTime())

  return lp, status, iters, seconds
end

end
