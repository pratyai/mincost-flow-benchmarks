"""
This module defines a Tulip solver that uses a custom CHOLMOD-based KKT solver.
This provides explicit control over the KKT system solution and is an example of how to integrate a custom KKT solver with Tulip.
"""
module TulipCHOLMOD

using Tulip
using Dimacs
using SparseArrays
using LinearAlgebra
using Random
using SuiteSparse

"""
Custom KKT solver using CHOLMOD, adapted from Tulip.jl's internal implementation.
This module defines a custom KKT backend that uses CHOLMOD to solve the normal equations system.
"""
module CholmodKKT

using Tulip
using SparseArrays
using LinearAlgebra
using SuiteSparse

using SparseArrays: AbstractSparseMatrix
using Tulip.KKT: AbstractKKTBackend, AbstractKKTSolver, K1

"""
Backend for the custom CHOLMOD KKT solver.
"""
Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend
  nested_dissection::Bool = false
end

"""
Solver for the custom CHOLMOD KKT solver.
"""
Base.@kwdef mutable struct Solver{Tv<:Number,Ti<:Integer} <: AbstractKKTSolver{Tv}
  # Problem data
  m::Ti
  n::Ti
  A::AbstractSparseMatrix{Tv,Ti}

  # Workspace
  θ::Vector{Tv} # Diagonal scaling
  regP::Vector{Tv} # Primal regularization
  regD::Vector{Tv} # Dual regularization
  K::SparseMatrixCSC{Tv,Ti} # KKT matrix
  ξ::Vector{Tv} # RHS of KKT system
  
  # CHOLMOD Factorization
  chol_factor::SuiteSparse.CHOLMOD.Factor{Tv}

  # Solution quality
  ipm_iter::Int
  solve_in_iter::Int
  residual_history::Vector{Tuple{Int, Int, Tv}}
end

Tulip.KKT.backend(::Solver) = "CustomCHOLMOD"
Tulip.KKT.linear_system(::Solver) = "Normal equations (K1)"

"""
    Tulip.KKT.setup(A, ::K1, bk::Backend)

Set up the KKT solver.
"""
function Tulip.KKT.setup(
  A::AbstractSparseMatrix{Tv,Ti},
  ::K1,
  bk::Backend,
) where {Tv<:Number,Ti<:Integer}
  local m, n = size(A)

  local θ = ones(Tv, n)
  local regP = ones(Tv, n)
  local regD = ones(Tv, m)
  local ξ = zeros(Tv, m)
  local K = sparse(A * A') + spdiagm(0 => regD)

  local K_cholmod = SuiteSparse.CHOLMOD.Sparse(Symmetric(K))
  local F = SuiteSparse.CHOLMOD.symbolic(K_cholmod; nested_dissection=bk.nested_dissection)
  local chol_factor = SuiteSparse.CHOLMOD.cholesky!(F, K_cholmod)

  return Solver{Tv,Ti}(m, n, A, θ, regP, regD, K, ξ, chol_factor, 0, 0, [])
end

"""
    Tulip.KKT.update!(kkt, θ, regP, regD)

Update the KKT system with new scaling and regularization terms.
"""
function Tulip.KKT.update!(
  kkt::Solver{Tv,Ti},
  θ::AbstractVector{Tv},
  regP::AbstractVector{Tv},
  regD::AbstractVector{Tv},
) where {Tv<:Number,Ti<:Integer}
  local m, n = kkt.m, kkt.n

  kkt.ipm_iter += 1
  kkt.solve_in_iter = 0

  copyto!(kkt.θ, θ)
  copyto!(kkt.regP, regP)
  copyto!(kkt.regD, regD)

  # Form normal equations matrix
  local D = spdiagm(one(Tv) ./ (kkt.θ .+ kkt.regP))
  kkt.K = (kkt.A * D * kkt.A') + spdiagm(0 => kkt.regD)

  cholesky!(kkt.chol_factor, Symmetric(kkt.K), check=false)
  issuccess(kkt.chol_factor) || throw(PosDefException(0))

  return nothing
end

"""
    Tulip.KKT.solve!(dx, dy, kkt, ξp, ξd)

Solve the KKT system.
"""
function Tulip.KKT.solve!(
  dx::AbstractVector{Tv},
  dy::AbstractVector{Tv},
  kkt::Solver{Tv,Ti},
  ξp::AbstractVector{Tv},
  ξd::AbstractVector{Tv},
) where {Tv<:Number,Ti<:Integer}
  kkt.solve_in_iter += 1

  local d = one(Tv) ./ (kkt.θ .+ kkt.regP)
  copyto!(kkt.ξ, ξp)
  mul!(kkt.ξ, kkt.A, d .* ξd, true, true)

  # Solve normal equations
  dy .= kkt.chol_factor \ kkt.ξ

  # Compute residual norm
  local residual_norm = norm(kkt.K * dy - kkt.ξ)
  push!(kkt.residual_history, (kkt.ipm_iter, kkt.solve_in_iter, residual_norm))

  # Recover dx
  copyto!(dx, ξd)
  mul!(dx, kkt.A', dy, 1.0, -1.0)
  dx .*= d

  return nothing
end

end # module CholmodKKT

include("common.jl")
using .SolverCommon

function construct_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv}, config::Dict) where {Tv<:Number}
  lp = SolverCommon.create_tulip_model(netw, Tv)

  Tulip.set_parameter(lp, "OutputLevel", 0)
  Tulip.set_parameter(lp, "Presolve_Level", 0)
  Tulip.set_parameter(lp, "KKT_System", Tulip.KKT.K1())
  
  cholmod_params = get(config, "cholmod_parameters", Dict())
  nested_dissection = get(cholmod_params, "NestedDissection", false)
  Tulip.set_parameter(lp, "KKT_Backend", CholmodKKT.Backend{Tv}(nested_dissection=nested_dissection))

  params = get(config, "parameters", Dict())
  for (k, v) in params
    Tulip.set_parameter(lp, k, Tv(v))
  end
  return lp
end

"""
    solve(netw::Dimacs.McfpNet, config::Dict)

Solve a minimum cost flow problem using the TulipCHOLMOD solver with a given configuration.

# Arguments
- `netw::Dimacs.McfpNet`: The minimum cost flow problem to solve.
- `config::Dict`: A dictionary with the solver configuration.

# Returns
- A named tuple with the solver status, number of iterations, solution time, and solution vector.
"""
function solve(netw::Dimacs.McfpNet, config::Dict)
  lp = construct_tulip_model(netw, Float64, config)
  Tulip.optimize!(lp)

  status = Tulip.get_attribute(lp, Tulip.Status())
  iters = Tulip.get_attribute(lp, Tulip.BarrierIterations())
  seconds = Tulip.get_attribute(lp, Tulip.SolutionTime())
  solution = lp.solution.x
  residual_history = lp.solver.kkt.residual_history

  return (; status, iters, seconds, solution, residual_history)
end

end # module TulipCHOLMOD