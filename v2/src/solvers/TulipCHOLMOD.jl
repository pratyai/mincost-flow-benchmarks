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

  local chol_factor = cholesky(Symmetric(K))

  return Solver{Tv,Ti}(m, n, A, θ, regP, regD, K, ξ, chol_factor)
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
  local d = one(Tv) ./ (kkt.θ .+ kkt.regP)
  copyto!(kkt.ξ, ξp)
  mul!(kkt.ξ, kkt.A, d .* ξd, true, true)

  # Solve normal equations
  dy .= kkt.chol_factor \ kkt.ξ

  # Recover dx
  copyto!(dx, ξd)
  mul!(dx, kkt.A', dy, 1.0, -1.0)
  dx .*= d

  return nothing
end

end # module CholmodKKT

include("common.jl")
using .SolverCommon

const CONFIG = Dict(
    "IPM_PRegMin" => 1e-6,
    "IPM_DRegMin" => 1e-6,
)

function construct_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv}) where {Tv<:Number}
  lp = SolverCommon.create_tulip_model(netw, Tv)

  Tulip.set_parameter(lp, "OutputLevel", 0)
  Tulip.set_parameter(lp, "Presolve_Level", 0)
  Tulip.set_parameter(lp, "KKT_System", Tulip.KKT.K1())
  Tulip.set_parameter(lp, "KKT_Backend", CholmodKKT.Backend{Tv}())

  for (k, v) in CONFIG
    Tulip.set_parameter(lp, k, Tv(v))
  end
  return lp
end

"""
    solve(netw::Dimacs.McfpNet)

Solve a minimum cost flow problem using the TulipCHOLMOD solver.

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
  return "KKT_Backend=CustomCHOLMOD, " * join(["$k=$v" for (k, v) in CONFIG], ", ")
end

end # module TulipCHOLMOD