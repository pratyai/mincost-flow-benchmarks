__precompile__(false)  # Disable precompilation (we're messily overwriting some functions from foreign package)

"""
Module to mostly copy-paste a few things from Tulip's code,
with some overriding to make it easy to integrate our custom Multigrid solver.
"""
module Kustom

using Tulip
using SparseArrays
using LinearAlgebra
using AlgebraicMultigrid

using SparseArrays: AbstractSparseMatrix
using Tulip.KKT: AbstractKKTBackend, AbstractKKTSolver, K1

struct Backend{Tv<:Number} <: AbstractKKTBackend end

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
  # Laplacians related
  prec::AlgebraicMultigrid.MultiLevel
  amg_solve::Function  # Solver for the SDDM system that yields `dy`
end

Tulip.KKT.backend(::Solver) = "Mcfp Tulip K1 AlgebraicMultigrid"
Tulip.KKT.linear_system(::Solver) = "Normal equations (K1)"

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

  local prec = ruge_stuben(sparse(Symmetric(K)))
  local amg_solve = function (b)
    return AlgebraicMultigrid._solve(prec, b)
  end

  return Solver{Tv,Ti}(m, n, A, θ, regP, regD, K, ξ, prec, amg_solve)
end

function Tulip.KKT.update!(
  kkt::Solver{Tv,Ti},
  θ::AbstractVector{Tv},
  regP::AbstractVector{Tv},
  regD::AbstractVector{Tv},
) where {Tv<:Number,Ti<:Integer}
  local m, n = kkt.m, kkt.n

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

  kkt.prec = ruge_stuben(sparse(Symmetric(kkt.K)))
  kkt.amg_solve = function (b)
    return AlgebraicMultigrid._solve(kkt.prec, b)
  end

  return nothing
end

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
  dy .= kkt.amg_solve(kkt.ξ)

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

  local gap_bound = (
    res.rg +
    0.5 *
    (norm(res.rp_nrm, Inf) + norm(res.rl_nrm, Inf) + norm(res.ru_nrm, Inf)) *
    norm(dat.c, 1) +
    0.5 * norm(res.rd_nrm, Inf) * norm(dat.b, 1)
  )
  # @show gap_bound, pt.κ, pt.τ
  # @show count(res.rl .< 0), count(res.ru .< 0)

  # Check for feasibility
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

  # Check for optimal solution
  if ρp <= ϵp && ρd <= ϵd && ρg <= ϵg && (gap_bound < 0 || gap_bound >= pt.τ)
    @show gap_bound, pt.τ
  end
  if ρp <= ϵp && ρd <= ϵd && ρg <= ϵg# && gap_bound < pt.τ
    hsd.primal_status = Tulip.Sln_Optimal
    hsd.dual_status = Tulip.Sln_Optimal
    hsd.solver_status = Tulip.Trm_Optimal
    return nothing
  end

  #= DISABLE INFEASIBILITY CHECK BECAUSE WE KNOW THE PROBLEM IS FEASIBLE
  # Check for infeasibility certificates
  if max(
      norm(dat.A * pt.x, Inf),
      norm((pt.x .- pt.xl) .* dat.lflag, Inf),
      norm((pt.x .+ pt.xu) .* dat.uflag, Inf)
  ) * (norm(dat.c, Inf) / max(1, norm(dat.b, Inf))) < - ϵi * dot(dat.c, pt.x)
      # Dual infeasible, i.e., primal unbounded
      hsd.primal_status = Tulip.Sln_InfeasibilityCertificate
      hsd.solver_status = Tulip.Trm_DualInfeasible
      return nothing
  end

  δ = dat.A' * pt.y .+ (pt.zl .* dat.lflag) .- (pt.zu .* dat.uflag)
  if norm(δ, Inf) * max(
      norm(dat.l .* dat.lflag, Inf),
      norm(dat.u .* dat.uflag, Inf),
      norm(dat.b, Inf)
  ) / (max(one(T), norm(dat.c, Inf)))  < (dot(dat.b, pt.y) + dot(dat.l .* dat.lflag, pt.zl)- dot(dat.u .* dat.uflag, pt.zu)) * ϵi
      # Primal infeasible
      hsd.dual_status = Tulip.Sln_InfeasibilityCertificate
      hsd.solver_status = Tulip.Trm_PrimalInfeasible
      return nothing
  end
  =#

  return nothing
end

end
