"""
Module TulipApproxChol

This module defines a specialized Tulip solver that integrates a custom KKT solver
utilizing an approximate Cholesky factorization from the Laplacians.jl library.
It is designed for solving Minimum Cost Flow Problems (MCFP) with improved
performance characteristics for certain graph structures.
"""
module TulipApproxChol

using Tulip
using Dimacs
using SparseArrays
using LinearAlgebra
using Laplacians
using Random
using TimerOutputs

include("common.jl")
using .SolverCommon

"""
Module Kustom

This submodule defines a custom KKT (Karush-Kuhn-Tucker) backend for Tulip.jl
that leverages an approximate Cholesky factorization provided by Laplacians.jl.
It customizes the solution of the KKT system within the interior-point method.
"""
module Kustom

using Tulip
using Laplacians
using SparseArrays
using LinearAlgebra
using ..SolverCommon

using SparseArrays: AbstractSparseMatrix
using Tulip.KKT: AbstractKKTBackend, AbstractKKTSolver, K1
using Laplacians: ApproxCholParams, approxchol_sddm

"""
Backend for the custom approximate Cholesky KKT solver.

# Fields
- `params::ApproxCholParams`: Parameters for the approximate Cholesky factorization,
  controlling aspects like ordering strategy, edge splitting, and merging.
- `pcgtol::Tv`: Tolerance for the Preconditioned Conjugate Gradient (PCG) solver
  used within the approximate Cholesky factorization.
"""
Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend
  params::ApproxCholParams = ApproxCholParams(:deg, 0, 2, 2)
  pcgtol::Tv = 5e-8
end

"""
Solver for the custom approximate Cholesky KKT solver.

This mutable struct holds the problem data, Laplacians-related parameters,
workspace variables, and solution quality metrics for the KKT system.

# Fields
- `m::Ti`: Number of constraints.
- `n::Ti`: Number of variables.
- `A::AbstractSparseMatrix{Tv,Ti}`: The constraint matrix.
- `params::ApproxCholParams`: Parameters for the approximate Cholesky factorization.
- `pcgtol::Tv`: Tolerance for the PCG solver.
- `θ::Vector{Tv}`: Diagonal scaling vector.
- `regP::Vector{Tv}`: Primal regularization vector.
- `regD::Vector{Tv}`: Dual regularization vector.
- `K::SparseMatrixCSC{Tv,Ti}`: The KKT matrix.
- `ξ::Vector{Tv}`: Right-hand side vector of the KKT system.
- `sddm_solve::Function`: Function to solve the SDDM system, yielding `dy`.
- `ipm_iter::Int`: Current Interior Point Method iteration count.
- `solve_in_iter::Int`: Counter for solves within the current IPM iteration.
- `residual_history::Vector{Tuple{Int, Int, Tv}}`: History of residual norms.
"""
Base.@kwdef mutable struct Solver{Tv<:Number,Ti<:Integer} <: AbstractKKTSolver{Tv}
  # Problem data
  m::Ti
  n::Ti
  A::AbstractSparseMatrix{Tv,Ti}
  # Laplacians related
  params::ApproxCholParams  # Parameter to use when constructing the SDDM solver
  pcgtol::Tv  # Parameter to use when constructing the SDDM solver

  # Workspace
  θ::Vector{Tv} # Diagonal scaling
  regP::Vector{Tv} # Primal regularization
  regD::Vector{Tv} # Dual regularization
  K::SparseMatrixCSC{Tv,Ti} # KKT matrix
  ξ::Vector{Tv} # RHS of KKT system
  # Laplacians related
  sddm_solve::Function  # Solver for the SDDM system that yields `dy`

  # Solution quality
  ipm_iter::Int
  solve_in_iter::Int
  residual_history::Vector{Tuple{Int, Int, Tv}}
end

Tulip.KKT.backend(::Solver) = "Mcfp Tulip K1 Sddm"
Tulip.KKT.linear_system(::Solver) = "Normal equations (K1)"

"""
    Tulip.KKT.setup(A, ::K1, bk::Backend)

Sets up the KKT solver for the approximate Cholesky backend. This involves initializing
the necessary data structures and pre-factorizing the KKT matrix symbolically.

# Arguments
- `A::AbstractSparseMatrix{Tv,Ti}`: The constraint matrix of the optimization problem.
- `::K1`: Indicates the K1 KKT system formulation.
- `bk::Backend`: The backend configuration for the approximate Cholesky solver.

# Returns
- A `Solver` instance initialized for the approximate Cholesky KKT system.
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

  local sddm_solve =
    approxchol_sddm(sparse(Symmetric(K)); params = bk.params, tol = bk.pcgtol)

  return Solver{Tv,Ti}(m, n, A, bk.params, bk.pcgtol, θ, regP, regD, K, ξ, sddm_solve, 0, 0, [])
end

"""
    Tulip.KKT.update!(kkt, θ, regP, regD)

Updates the KKT system with new scaling and regularization terms. This function
reconstructs the KKT matrix and re-initializes the SDDM solver with the updated terms.

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

  # Sanity checks
  length(θ) == n ||
    throw(DimensionMismatch("length(θ)=$(length(θ)) but KKT solver has n=$n."))
  length(regP) == n ||
    throw(DimensionMismatch("length(regP)=$(length(regP)) but KKT solver has n=$n"))
  length(regD) == m ||
    throw(DimensionMismatch("length(regD)=$(length(regD)) but KKT solver has m=$m"))

  copyto!(kkt.θ, θ)
  copyto!(kkt.regP, regP)
  copyto!(kkt.regD, regD)

  # Form normal equations matrix
  local D = spdiagm(one(Tv) ./ (kkt.θ .+ kkt.regP))
  kkt.K = (kkt.A * D * kkt.A') + spdiagm(0 => kkt.regD)

  kkt.sddm_solve =
    approxchol_sddm(sparse(Symmetric(kkt.K)); params = kkt.params, tol = kkt.pcgtol)

  return nothing
end

"""
    Tulip.KKT.solve!(dx, dy, kkt, ξp, ξd)

Solves the KKT system for `dx` and `dy` using the approximate Cholesky factorization.

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
  dy .= kkt.sddm_solve(kkt.ξ; maxits = 100)

  # Compute residual norm
  local residual_norm = norm(kkt.K * dy - kkt.ξ)
  push!(kkt.residual_history, (kkt.ipm_iter, kkt.solve_in_iter, residual_norm))

  # Recover dx
  copyto!(dx, ξd)
  mul!(dx, kkt.A', dy, 1.0, -1.0)
  dx .*= d

  return nothing
end

function Tulip.update_solver_status!(
  hsd::Tulip.HSD{T},
  ϵp::T,
  ϵd::T,
  ϵg::T,
  ϵi::T,
) where {T}
"""
    Tulip.update_solver_status!(hsd, ϵp, ϵd, ϵg, ϵi)

Extends Tulip's default solver status update function to use a simplified
convergence criteria for the approximate Cholesky solver. This function
dispatches to `SolverCommon._approxchol_update_solver_status!`.

# Arguments
- `hsd::Tulip.HSD{T}`: The Homogeneous Self-Dual (HSD) solver object.
- `ϵp::T`: Primal feasibility tolerance.
- `ϵd::T`: Dual feasibility tolerance.
- `ϵg::T`: Duality gap tolerance.
- `ϵi::T`: Infeasibility tolerance.
"""
  SolverCommon._approxchol_update_solver_status!(hsd, ϵp, ϵd, ϵg, ϵi)
end

end # module Kustom

"""
    construct_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv}, config::Dict)

Constructs a Tulip.jl model configured to use the approximate Cholesky KKT solver.
This function sets up the problem and applies solver-specific parameters from the configuration.

# Arguments
- `netw::Dimacs.McfpNet`: The DIMACS MCFP network data.
- `Tv::Type`: The numeric type to use for the model (e.g., `Float64`).
- `config::Dict`: A dictionary containing solver-specific configurations,
  including `kustom_parameters` for `ApproxCholParams` and `pcgtol`.

# Returns
- A `Tulip.Model{Tv}` instance ready for optimization with the approximate Cholesky backend.
"""
function construct_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv}, config::Dict) where {Tv<:Number}
  lp = SolverCommon.create_tulip_model(netw, Tv)

  Tulip.set_parameter(lp, "OutputLevel", 1)  # enable output
  Tulip.set_parameter(lp, "Presolve_Level", 0)  # disable presolve
  Tulip.set_parameter(lp, "KKT_System", Tulip.KKT.K1())

  kustom_params = get(config, "kustom_parameters", Dict())
  pcgtol = get(kustom_params, "pcgtol", 5e-8)
  
  approxchol_params_dict = get(kustom_params, "ApproxCholParams", Dict())
  approxchol_type = Symbol(get(approxchol_params_dict, "type", "deg"))
  approxchol_stag_test = get(approxchol_params_dict, "stag_test", 0)
  approxchol_split = get(approxchol_params_dict, "split", 2)
  approxchol_merge = get(approxchol_params_dict, "merge", 2)
  approxchol_params = ApproxCholParams(approxchol_type, approxchol_stag_test, approxchol_split, approxchol_merge)

  Tulip.set_parameter(lp, "KKT_Backend", Kustom.Backend{Tv}(; params=approxchol_params, pcgtol=Tv(pcgtol)))

  params = get(config, "parameters", Dict())
  for (k, v) in params
    Tulip.set_parameter(lp, k, v)
  end

  se = Tv(sqrt(eps(Float64)))
  Tulip.set_parameter(lp, "IPM_TolerancePFeas", se)
  Tulip.set_parameter(lp, "IPM_ToleranceDFeas", se)
  Tulip.set_parameter(lp, "IPM_ToleranceRGap", se)
  Tulip.set_parameter(lp, "IPM_ToleranceIFeas", se)
  return lp
end

"""
    solve(netw::Dimacs.McfpNet, config::Dict)

Solves a Minimum Cost Flow Problem (MCFP) using the TulipApproxChol solver.
This function constructs the Tulip model, optimizes it, and returns detailed
solver statistics.

# Arguments
- `netw::Dimacs.McfpNet`: The minimum cost flow problem to solve.
- `config::Dict`: A dictionary with the solver configuration, including parameters
  for the approximate Cholesky factorization and general Tulip settings.

# Returns
- A named tuple containing:
  - `status`: The termination status of the solver.
  - `iters`: The number of Interior Point Method iterations.
  - `seconds`: The total solution time in seconds.
  - `solution`: The optimal solution vector `x`.
  - `fact_s`: Factorization time in seconds.
  - `solv_s`: KKT system solve time in seconds.
  - `sddm_calls`: Number of SDDM solver calls.
  - `residual_history`: A history of residual norms during the optimization.
"""
function solve(netw::Dimacs.McfpNet, config::Dict)
  lp = construct_tulip_model(netw, Float64, config)
  Tulip.optimize!(lp)

  status = Tulip.get_attribute(lp, Tulip.Status())
  iters = Tulip.get_attribute(lp, Tulip.BarrierIterations())
  seconds = Tulip.get_attribute(lp, Tulip.SolutionTime())
  solution = lp.solution.x

  # Extract additional metrics from the solver timer
  to = lp.solver.timer
  fact_ns = TimerOutputs.time(to["Main loop"]["Step"]["Factorization"])
  solv_ns = TimerOutputs.time(to["Main loop"]["Step"]["Newton"]["KKT"])
  sddm_calls = TimerOutputs.ncalls(to["Main loop"]["Step"]["Newton"]["KKT"])

  residual_history = lp.solver.kkt.residual_history

  return (; status, iters, seconds, solution, fact_s = fact_ns * 1e-9, solv_s = solv_ns * 1e-9, sddm_calls, residual_history)
end

end # module TulipApproxChol
