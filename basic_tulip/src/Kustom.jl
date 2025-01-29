__precompile__(false)  # Disable precompilation (we're messily overwriting some functions from foreign package)

"""
Module to mostly copy-paste a few things from Tulip's code,
with some overriding to make it easy to integrate our custom SDDM solver.
"""
module Kustom

using Tulip
using SparseArrays
using LinearAlgebra
using JLD2
using Printf

using SparseArrays: AbstractSparseMatrix
using Tulip.KKT: AbstractKKTBackend, AbstractKKTSolver, K1
using Tulip.KKT.TlpCholmod: CholmodSolver
using Tulip.KKT.TlpCholmod

Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend
  pcgtol::Tv = 5e-8
  # pcgtol::Tv = 1e-11
  # pcgtol::Tv = 1e-16
  badmat::Function = function () end
end

Base.@kwdef mutable struct Solver{Tv<:Number,Ti<:Integer} <: AbstractKKTSolver{Tv}
  cholmod::CholmodSolver{Tv, K1}

  pcgtol::Tv
  at_iter::Int = 0
  at_solve::Int = 0
  at_total_solve::Int = 0
  badmat::Function = function () end
end

Tulip.KKT.backend(::Solver) = "Mcfp Tulip K1 Sddm"
Tulip.KKT.linear_system(::Solver) = "Normal equations (K1)"

function Tulip.KKT.setup(
  A::AbstractSparseMatrix{Tv,Ti},
  k1::K1,
  bk::Backend,
) where {Tv<:Number,Ti<:Integer}
  local m, n = size(A)
  local θ = ones(Tv, n)
  local regP = ones(Tv, n)
  local regD = ones(Tv, m)
  local ξ = zeros(Tv, m)
  local K = sparse(A * A') + spdiagm(0 => regD)
  local F = cholesky(Symmetric(K))

  local cholmod = CholmodSolver{Tv,K1}(m, n, A, θ, regP, regD, K, F, ξ)
  return Solver{Tv,Ti}(
    cholmod,
    bk.pcgtol,
    0,
    0,
    0,
    bk.badmat,
  )
end

function Tulip.KKT.update!(
  kkt::Solver{Tv,Ti},
  θ::AbstractVector{Tv},
  regP::AbstractVector{Tv},
  regD::AbstractVector{Tv},
) where {Tv<:Number,Ti<:Integer}
  TlpCholmod.update!(kkt.cholmod, θ, regP, regD)

  kkt.at_iter += 1
  kkt.at_solve = 1

  return nothing
end

function Tulip.KKT.solve!(
  dx::AbstractVector{Tv},
  dy::AbstractVector{Tv},
  kkt::Solver{Tv,Ti},
  ξp::AbstractVector{Tv},
  ξd::AbstractVector{Tv},
) where {Tv<:Number,Ti<:Integer}
  local m, n = kkt.cholmod.m, kkt.cholmod.n
  local D = inv(Diagonal(kkt.cholmod.θ .+ kkt.cholmod.regP))
  copyto!(kkt.cholmod.ξ, ξp)
  mul!(kkt.cholmod.ξ, kkt.cholmod.A, D * ξd, true, true)

  # Solve normal equations
  # CHOLMOD doesn't have in-place solve, so this line will allocate
  dy .= kkt.cholmod.F \ kkt.cholmod.ξ
  rel_r = if norm(kkt.cholmod.ξ) != 0
    r = kkt.cholmod.K * dy - kkt.cholmod.ξ
    norm(r) / norm(kkt.cholmod.ξ)
  else
    Tv(0)
  end
  @show rel_r, kkt.pcgtol
  #=
  fo = open("rhsnorm.csv", "a")
  if kkt.at_iter == 1 && kkt.at_solve == 1
    @printf fo "it,sol,rhsl1,rhsl2,rhslinf,dyl1,dyl2,dylinf,relerr\n"
  end
  @printf fo "%d,%d,%e,%e,%e,%e,%e,%e,%e\n" kkt.at_iter kkt.at_solve norm(kkt.ξ, 1) norm(kkt.ξ, 2) norm(kkt.ξ, Inf) norm(dy, 1) norm(dy, 2) norm(dy, Inf) rel_r
  close(fo)
  =#

  if rel_r > kkt.pcgtol
    T = sparse(hcat(kkt.cholmod.K, kkt.cholmod.ξ))
    slug = @sprintf "it=%d_sol=%d_tol_=%e_maxits=%d" kkt.at_iter kkt.at_solve kkt.pcgtol 100
    kkt.badmat(slug, T)
  end
  kkt.at_solve += 1
  kkt.at_total_solve += 1

  # Recover dx
  copyto!(dx, ξd)
  mul!(dx, kkt.cholmod.A', dy, 1.0, -1.0)
  lmul!(D, dx)

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

  #=
  local gap_bound = (
    res.rg +
    0.5 *
    (norm(res.rp_nrm, Inf) + norm(res.rl_nrm, Inf) + norm(res.ru_nrm, Inf)) *
    norm(dat.c, 1) +
    0.5 * norm(res.rd_nrm, Inf) * norm(dat.b, 1)
  )=#
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
  #=
  if ρp <= ϵp && ρd <= ϵd && ρg <= ϵg && (gap_bound < 0 || gap_bound >= pt.τ)
    # @show gap_bound, pt.τ
  end
  =#
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
