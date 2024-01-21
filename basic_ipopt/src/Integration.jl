module Integration

using FromFile
@from "Dimacs.jl" import Dimacs

using SparseArrays
using LinearAlgebra
using MultiFloats
using DoubleFloats
using Random
using JuMP
using Ipopt

function construct_ipopt_model(netw::Dimacs.McfpNet, _::Type{Tv}) where {Tv<:Number}
  Random.seed!(1)
  local A = sparse(Int8.(netw.G.IncidenceMatrix))
  local b = Tv.(netw.Demand)
  local u = Tv.(netw.Cap)
  local c = Tv.(netw.Cost)
  local n, m = size(A)

  local lp = JuMP.Model(Ipopt.Optimizer)
  @variable(lp, x[1:m])
  @constraint(lp, A * x == b)
  @constraint(lp, x >= 0)
  @constraint(lp, x <= u)
  @objective(lp, Min, c' * x)

  return lp
end

function solve_ipopt_model(lp::JuMP.Model) where {Tv<:Number}
  JuMP.optimize!(lp)

  local status = JuMP.termination_status(lp)
  local iters = JuMP.barrier_iterations(lp)
  local seconds = JuMP.solve_time(lp)

  return lp, status, iters, seconds
end

end
