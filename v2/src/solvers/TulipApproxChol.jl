"""
This module defines a Tulip solver that uses a custom KKT solver with an approximate Cholesky factorization.
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

# Kustom.jl content adapted here
"""
This module defines a custom KKT backend that uses an approximate Cholesky factorization from Laplacians.jl.
"""
module Kustom

using Tulip
using Laplacians
using SparseArrays
using LinearAlgebra

using SparseArrays: AbstractSparseMatrix
using Tulip.KKT: AbstractKKTBackend, AbstractKKTSolver, K1
using Laplacians: ApproxCholParams, approxchol_sddm

"""
Backend for the custom approximate Cholesky KKT solver.
"""
Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend
  params::ApproxCholParams = ApproxCholParams(:deg, 0, 2, 2)
  pcgtol::Tv = 5e-8
end

"""
Solver for the custom approximate Cholesky KKT solver.
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

  local sddm_solve =
    approxchol_sddm(sparse(Symmetric(K)); params = bk.params, tol = bk.pcgtol)

  return Solver{Tv,Ti}(m, n, A, bk.params, bk.pcgtol, θ, regP, regD, K, ξ, sddm_solve, 0, 0, [])
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

"""
    update_solver_status!(hsd, ϵp, ϵd, ϵg, ϵi)

Custom implementation of the solver status update function.
This is likely a copy of an older version of Tulip's function or a custom version with different convergence criteria.
"""
function Tulip.update_solver_status!(
  hsd::Tulip.HSD{T},
  ϵp::T,
  ϵd::T,
  ϵg::T,
  ϵi::T,
) where {T}
  hsd.solver_status = Tulip.Trm_Unknown

  pt, res = hsd.pt, hsd.res
  dat = hsd.dat

  ρp = max(
    res.rp_nrm / (pt.τ * (one(T) + norm(dat.b, Inf))),
    res.rl_nrm / (pt.τ * (one(T) + norm(dat.l .* dat.lflag, Inf))),
    res.ru_nrm / (pt.τ * (one(T) + norm(dat.u .* dat.uflag, Inf))),
  )
  ρd = res.rd_nrm / (pt.τ * (one(T) + norm(dat.c, Inf)))
  ρg = abs(hsd.primal_objective - hsd.dual_objective) / (one(T) + abs(hsd.dual_objective))

  if ρp <= ϵp
    hsd.primal_status = Tulip.Sln_FeasiblePoint
  else
    hsd.primal_status = Tulip.Sln_Unknown
  end

  if ρd <= ϵd
    hsd.dual_status = Tulip.Sln_FeasiblePoint
  else
    hsd.dual_status = Tulip.Sln_Unknown
  end

  if ρp <= ϵp && ρd <= ϵd && ρg <= ϵg
    hsd.primal_status = Tulip.Sln_Optimal
    hsd.dual_status = Tulip.Sln_Optimal
    hsd.solver_status = Tulip.Trm_Optimal
    return nothing
  end

  return nothing
end

end # module Kustom

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

Solve a minimum cost flow problem using the TulipApproxChol solver.

# Arguments
- `netw::Dimacs.McfpNet`: The minimum cost flow problem to solve.
- `config::Dict`: A dictionary with the solver configuration.

# Returns
- A named tuple with the solver status, number of iterations, solution time, solution vector, and additional solver-specific statistics.
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
