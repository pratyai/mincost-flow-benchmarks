module TulipApproxChol

using Tulip
using Dimacs
using SparseArrays
using LinearAlgebra
using Laplacians
using Random
using TimerOutputs

# Kustom.jl content adapted here
module Kustom

using Tulip
using Laplacians
using SparseArrays
using LinearAlgebra

using SparseArrays: AbstractSparseMatrix
using Tulip.KKT: AbstractKKTBackend, AbstractKKTSolver, K1
using Laplacians: ApproxCholParams, approxchol_sddm

Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend
  params::ApproxCholParams = ApproxCholParams(:deg, 0, 2, 2)
  pcgtol::Tv = 5e-8
end

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
end

Tulip.KKT.backend(::Solver) = "Mcfp Tulip K1 Sddm"
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

  local sddm_solve =
    approxchol_sddm(sparse(Symmetric(K)); params = bk.params, tol = bk.pcgtol)

  return Solver{Tv,Ti}(m, n, A, bk.params, bk.pcgtol, θ, regP, regD, K, ξ, sddm_solve)
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

  kkt.sddm_solve =
    approxchol_sddm(sparse(Symmetric(kkt.K)); params = kkt.params, tol = kkt.pcgtol)

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
  dy .= kkt.sddm_solve(kkt.ξ; maxits = 100)

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
  Tulip.set_parameter(lp, "Presolve_Level", 0)  # disable presolve
  Tulip.set_parameter(lp, "KKT_System", Tulip.KKT.K1())
  Tulip.set_parameter(lp, "KKT_Backend", Kustom.Backend{Tv}())

  # Tulip.set_parameter(lp, "IPM_PRegMin", Tv(1e-1))
  Tulip.set_parameter(lp, "IPM_PRegMin", Tv(1e-4))
  # Tulip.set_parameter(lp, "IPM_PRegMin", Tv(1e-6))
  # Tulip.set_parameter(lp, "IPM_PRegMin", Tv(1e-8))

  # Tulip.set_parameter(lp, "IPM_DRegMin", Tv(1e-1))
  # Tulip.set_parameter(lp, "IPM_DRegMin", Tv(1e-4))
  # Tulip.set_parameter(lp, "IPM_DRegMin", Tv(1e-6))
  Tulip.set_parameter(lp, "IPM_DRegMin", Tv(1e-8))
  # Tulip.set_parameter(lp, "IPM_DRegMin", Tv(1e-10))
  # Tulip.set_parameter(lp, "IPM_DRegMin", Tv(1e-12))

  Tulip.set_parameter(lp, "IPM_IterationsLimit", 200)

  se = Tv(sqrt(eps(Float64)))
  Tulip.set_parameter(lp, "IPM_TolerancePFeas", se)
  Tulip.set_parameter(lp, "IPM_ToleranceDFeas", se)
  Tulip.set_parameter(lp, "IPM_ToleranceRGap", se)
  Tulip.set_parameter(lp, "IPM_ToleranceIFeas", se)
  return lp
end

function solve(netw::Dimacs.McfpNet)
  lp = construct_tulip_model(netw, Float64)
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

  return (; status, iters, seconds, solution, fact_s = fact_ns * 1e-9, solv_s = solv_ns * 1e-9, sddm_calls)
end

end # module TulipApproxChol
