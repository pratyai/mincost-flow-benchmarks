"""
Module TulipCHOLMOD

This module defines a Tulip solver that utilizes a custom KKT solver based on CHOLMOD.
It provides explicit control over the KKT system solution, serving as an example
of integrating a custom KKT solver with Tulip.jl for Minimum Cost Flow Problems (MCFP).
"""
module TulipCHOLMOD

using Tulip
using Dimacs
using SparseArrays
using LinearAlgebra
using Random
using SuiteSparse

"""
Module CholmodKKT

This submodule implements a custom KKT solver backend using CHOLMOD for solving
the normal equations system within Tulip.jl's interior-point method. It's adapted
from Tulip.jl's internal implementation to allow for specific CHOLMOD configurations.
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

# Fields
- `nested_dissection::Bool`: If `true`, enables nested dissection ordering for CHOLMOD,
  which can improve factorization performance for certain sparse matrix structures.
"""
Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend
  nested_dissection::Bool = false
end

"""
Solver for the custom CHOLMOD KKT solver.

This mutable struct holds the problem data, workspace variables, CHOLMOD factorization
object, and solution quality metrics for the KKT system.

# Fields
- `m::Ti`: Number of constraints.
- `n::Ti`: Number of variables.
- `A::AbstractSparseMatrix{Tv,Ti}`: The constraint matrix.
- `θ::Vector{Tv}`: Diagonal scaling vector.
- `regP::Vector{Tv}`: Primal regularization vector.
- `regD::Vector{Tv}`: Dual regularization vector.
- `K::SparseMatrixCSC{Tv,Ti}`: The KKT matrix.
- `ξ::Vector{Tv}`: Right-hand side vector of the KKT system.
- `chol_factor::SuiteSparse.CHOLMOD.Factor{Tv}`: The CHOLMOD factorization object.
- `ipm_iter::Int`: Current Interior Point Method iteration count.
- `solve_in_iter::Int`: Counter for solves within the current IPM iteration.
- `residual_history::Vector{Tuple{Int, Int, Tv}}`: History of residual norms.
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

Sets up the KKT solver for the CHOLMOD backend. This involves initializing
the necessary data structures and performing a symbolic factorization using CHOLMOD.

# Arguments
- `A::AbstractSparseMatrix{Tv,Ti}`: The constraint matrix of the optimization problem.
- `::K1`: Indicates the K1 KKT system formulation.
- `bk::Backend`: The backend configuration for the CHOLMOD solver.

# Returns
- A `Solver` instance initialized for the CHOLMOD KKT system.
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

Updates the KKT system with new scaling and regularization terms. This function
reconstructs the KKT matrix and performs a numerical factorization using CHOLMOD.

# Arguments
- `kkt::Solver{Tv,Ti}`: The KKT solver instance to update.
- `θ::AbstractVector{Tv}`: New diagonal scaling terms.
- `regP::AbstractVector{Tv}`: New primal regularization terms.
- `regD::AbstractVector{Tv}`: New dual regularization terms.
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

Solves the KKT system for `dx` and `dy` using the CHOLMOD factorization.

# Arguments
- `dx::AbstractVector{Tv}`: Output vector for primal step.
- `dy::AbstractVector{Tv}`: Output vector for dual step.
- `kkt::Solver{Tv,Ti}`: The KKT solver instance.
- `ξp::AbstractVector{Tv}`: Primal right-hand side vector.
- `ξd::AbstractVector{Tv}`: Dual right-hand side vector.
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

"""
    construct_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv}, config::Dict)

Constructs a Tulip.jl model configured to use the CHOLMOD KKT solver.
This function sets up the problem and applies solver-specific parameters from the configuration.

# Arguments
- `netw::Dimacs.McfpNet`: The DIMACS MCFP network data.
- `Tv::Type`: The numeric type to use for the model (e.g., `Float64`).
- `config::Dict`: A dictionary containing solver-specific configurations,
  including `cholmod_parameters` for `nested_dissection`.

# Returns
- A `Tulip.Model{Tv}` instance ready for optimization with the CHOLMOD backend.
"""
function construct_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv}, config::Dict) where {Tv<:Number}
  lp = SolverCommon.create_tulip_model(netw, Tv)

  Tulip.set_parameter(lp, "OutputLevel", 0)
  Tulip.set_parameter(lp, "Presolve_Level", 0)
  Tulip.set_parameter(lp, "KKT_System", Tulip.KKT.K1())
  
  cholmod_params = get(config, "cholmod_parameters", Dict())
  nested_dissection = get(cholmod_params, "NestedDissection", false)
  Tulip.set_parameter(lp, "KKT_Backend", CholmodKKT.Backend{Tv}(nested_dissection=nested_dissection))

  params = get(config, "parameters", Dict())
  # Explicitly set parameters, ensuring correct types
  Tulip.set_parameter(lp, "IPM_PRegMin", Tv(get(params, "IPM_PRegMin", 1e-6)))
  Tulip.set_parameter(lp, "IPM_DRegMin", Tv(get(params, "IPM_DRegMin", 1e-6)))
  Tulip.set_parameter(lp, "IPM_IterationsLimit", Int(get(params, "IPM_IterationsLimit", 200)))
  return lp
end

"""
    solve(netw::Dimacs.McfpNet, config::Dict)

Solves a Minimum Cost Flow Problem (MCFP) using the TulipCHOLMOD solver.
This function constructs the Tulip model, optimizes it, and returns detailed
solver statistics.

# Arguments
- `netw::Dimacs.McfpNet`: The minimum cost flow problem to solve.
- `config::Dict`: A dictionary with the solver configuration, including parameters
  for CHOLMOD and general Tulip settings.

# Returns
- A named tuple containing:
  - `status`: The termination status of the solver.
  - `iters`: The number of Interior Point Method iterations.
  - `seconds`: The total solution time in seconds.
  - `solution`: The optimal solution vector `x`.
  - `residual_history`: A history of residual norms during the optimization.
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