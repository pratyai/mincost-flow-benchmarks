"""
Module SolverCommon

This module provides common utilities and helper functions for integrating different
solvers with Tulip.jl, particularly for Minimum Cost Flow Problems (MCFP).
It includes functionalities for creating Tulip models from DIMACS network data
and custom solver status update logic.
"""
module SolverCommon

using Tulip
using Dimacs
using SparseArrays
using Random
using LinearAlgebra

"""
    create_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv})

Creates a Tulip.jl optimization model from a DIMACS Minimum Cost Flow Problem (MCFP) network.
This function sets up the problem data (objective, constraints, bounds) for Tulip.jl.

# Arguments
- `netw::Dimacs.McfpNet`: The DIMACS MCFP network data.
- `Tv::Type`: The numeric type to use for the model (e.g., `Float64`).

# Returns
- A `Tulip.Model{Tv}` instance configured with the MCFP problem.
"""
function create_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv}) where {Tv<:Number}
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
    return lp
end

function _approxchol_update_solver_status!(
  hsd::Tulip.HSD{T},
  ϵp::T,
  ϵd::T,
  ϵg::T,
  ϵi::T,
) where {T}
"""
    _approxchol_update_solver_status!(hsd, ϵp, ϵd, ϵg, ϵi)

Custom and simplified implementation of the solver status update logic for Tulip.jl.
This function checks for optimality and feasibility based on primal, dual, and gap residuals,
but it omits the checks for infeasibility certificates present in the full Tulip implementation.

# Arguments
- `hsd::Tulip.HSD{T}`: The Homogeneous Self-Dual (HSD) solver object.
- `ϵp::T`: Primal feasibility tolerance.
- `ϵd::T`: Dual feasibility tolerance.
- `ϵg::T`: Duality gap tolerance.
- `ϵi::T`: Infeasibility tolerance (not used in this simplified version).
"""
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

end # module SolverCommon
