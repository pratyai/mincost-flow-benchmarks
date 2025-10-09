module TulipBasic

using Tulip
using Dimacs
using SparseArrays
using LinearAlgebra
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

  Tulip.set_parameter(lp, "OutputLevel", 0)  # disable output
  Tulip.set_parameter(lp, "Presolve_Level", 0)  # disable presolve
  Tulip.set_parameter(lp, "IPM_PRegMin", Tv(1e-6))
  Tulip.set_parameter(lp, "IPM_DRegMin", Tv(1e-6))
  return lp
end

function solve(netw::Dimacs.McfpNet)
  lp = construct_tulip_model(netw, Float64)
  Tulip.optimize!(lp)

  status = Tulip.get_attribute(lp, Tulip.Status())
  iters = Tulip.get_attribute(lp, Tulip.BarrierIterations())
  seconds = Tulip.get_attribute(lp, Tulip.SolutionTime())
  solution = lp.solution.x

  return (; status, iters, seconds, solution)
end

end # module TulipBasic
