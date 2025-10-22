"""
Module TulipCHOLMOD

This module defines a meta-solver that uses a custom KKT solver based on CHOLMOD
for standard floating-point types (`Float32`, `Float64`) and automatically falls back
to a pure Julia Cholesky factorization from `CliqueTrees.jl` for higher-precision types.
"""
module TulipCHOLMOD

using Tulip
using Dimacs
using SparseArrays
using LinearAlgebra
using Random
using SuiteSparse
using CliqueTrees

"""
Module CholmodKKT

This submodule implements a custom KKT solver backend using CHOLMOD for solving
the normal equations system within Tulip.jl's interior-point method. It is adapted
from Tulip.jl's internal implementation to allow for specific CHOLMOD configurations.
This backend is only used for `Float32` and `Float64` due to CHOLMOD's limitations.
"""
module CholmodKKT

using Tulip
using SparseArrays
using LinearAlgebra
using SuiteSparse

using SparseArrays: AbstractSparseMatrix
using Tulip.KKT: AbstractKKTBackend, AbstractKKTSolver, K1

Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend
    nested_dissection::Bool = false
end

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
    residual_history::Vector{Tuple{Int,Int,Tv,Tv}}
end

Tulip.KKT.backend(::Solver) = "CustomCHOLMOD"
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

    local K_cholmod = SuiteSparse.CHOLMOD.Sparse(Symmetric(K))
    local F =
        SuiteSparse.CHOLMOD.symbolic(K_cholmod; nested_dissection = bk.nested_dissection)
    local chol_factor = SuiteSparse.CHOLMOD.cholesky!(F, K_cholmod)

    return Solver{Tv,Ti}(
        m,
        n,
        A,
        θ,
        regP,
        regD,
        K,
        ξ,
        chol_factor,
        0,
        0,
        Tuple{Int,Int,Tv,Tv}[],
    )
end

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

    cholesky!(kkt.chol_factor, Symmetric(kkt.K), check = false)
    issuccess(kkt.chol_factor) || throw(PosDefException(0))

    return nothing
end

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

    # Compute relative residual norm
    local absolute_residual_norm = norm(kkt.K * dy - kkt.ξ)
    local rhs_norm = norm(kkt.ξ)
    local relative_residual_norm =
        (rhs_norm == 0) ? absolute_residual_norm : absolute_residual_norm / rhs_norm
    push!(
        kkt.residual_history,
        (kkt.ipm_iter, kkt.solve_in_iter, relative_residual_norm, absolute_residual_norm),
    )

    # Recover dx
    copyto!(dx, ξd)
    mul!(dx, kkt.A', dy, 1.0, -1.0)
    dx .*= d

    return nothing
end

end # module CholmodKKT

"""
Module CliqueTreeKKT

This submodule implements a custom KKT solver backend using CliqueTrees.jl.
This pure Julia implementation is used as a fallback for high-precision float types
not supported by CHOLMOD.
"""
module CliqueTreeKKT

using Tulip
using SparseArrays
using LinearAlgebra
using CliqueTrees

using SparseArrays: AbstractSparseMatrix
using Tulip.KKT: AbstractKKTBackend, AbstractKKTSolver, K1

Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend end

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

    # Cholesky Factorization
    chol_factor::CliqueTrees.CholFact{Tv,Ti}

    # Solution quality
    ipm_iter::Int
    solve_in_iter::Int
    residual_history::Vector{Tuple{Int,Int,Tv,Tv}}
end

Tulip.KKT.backend(::Solver) = "CliqueTreeKKT"
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

    local chol_factor = CliqueTrees.cholesky(Symmetric(K))

    return Solver{Tv,Ti}(
        m,
        n,
        A,
        θ,
        regP,
        regD,
        K,
        ξ,
        chol_factor,
        0,
        0,
        Tuple{Int,Int,Tv,Tv}[],
    )
end

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

    # Re-factorize the matrix. CliqueTrees.jl doesn't have an in-place update.
    kkt.chol_factor = CliqueTrees.cholesky(Symmetric(kkt.K))

    return nothing
end

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

    # Compute relative residual norm
    local absolute_residual_norm = norm(kkt.K * dy - kkt.ξ)
    local rhs_norm = norm(kkt.ξ)
    local relative_residual_norm =
        (rhs_norm == 0) ? absolute_residual_norm : absolute_residual_norm / rhs_norm
    push!(
        kkt.residual_history,
        (kkt.ipm_iter, kkt.solve_in_iter, relative_residual_norm, absolute_residual_norm),
    )

    # Recover dx
    copyto!(dx, ξd)
    mul!(dx, kkt.A', dy, 1.0, -1.0)
    dx .*= d

    return nothing
end

end # module CliqueTreeKKT


include("common.jl")
using .SolverCommon


"""
    construct_tulip_model(netw::Dimacs.McfpNet, ::Type{Tv}, config::Dict)

Constructs a Tulip.jl model, automatically selecting the appropriate KKT backend
based on the floating-point precision `Tv`.

For `Float32` and `Float64`, it uses the high-performance `CholmodKKT` backend.
For other types (e.g., `Float128`, `MultiFloat`), it falls back to the pure Julia
`CliqueTreeKKT` backend.

# Arguments
- `netw::Dimacs.McfpNet`: The DIMACS MCFP network data.
- `Tv::Type`: The numeric type to use for the model.
- `config::Dict`: A dictionary containing solver-specific configurations.

# Returns
- A `Tulip.Model{Tv}` instance ready for optimization.
"""
function construct_tulip_model(
    netw::Dimacs.McfpNet,
    ::Type{Tv},
    config::Dict,
) where {Tv<:Number}
    lp = SolverCommon.create_tulip_model(netw, Tv)

    Tulip.set_parameter(lp, "OutputLevel", 0)
    Tulip.set_parameter(lp, "Presolve_Level", 0)
    Tulip.set_parameter(lp, "KKT_System", Tulip.KKT.K1())

    if Tv in [Float32, Float64]
        cholmod_params = get(config, "cholmod_parameters", Dict())
        nested_dissection = get(cholmod_params, "NestedDissection", false)
        Tulip.set_parameter(
            lp,
            "KKT_Backend",
            CholmodKKT.Backend{Tv}(nested_dissection = nested_dissection),
        )
    else
        Tulip.set_parameter(lp, "KKT_Backend", CliqueTreeKKT.Backend{Tv}())
    end

    params = get(config, "parameters", Dict())
    # Explicitly set parameters, ensuring correct types
    Tulip.set_parameter(lp, "IPM_PRegMin", Tv(get(params, "IPM_PRegMin", 1e-6)))
    Tulip.set_parameter(lp, "IPM_DRegMin", Tv(get(params, "IPM_DRegMin", 1e-6)))
    Tulip.set_parameter(
        lp,
        "IPM_IterationsLimit",
        Int(get(params, "IPM_IterationsLimit", 200)),
    )
    return lp
end

"""
    solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number})

Solves a Minimum Cost Flow Problem (MCFP) using the TulipCHOLMOD solver.
This function constructs the Tulip model, optimizes it, and returns detailed
solver statistics.

# Arguments
- `netw::Dimacs.McfpNet`: The minimum cost flow problem to solve.
- `config::Dict`: A dictionary with the solver configuration.
- `float_type::Type{<:Number}`: The floating-point type to use for the solver.

# Returns
- A named tuple containing:
  - `status`: The termination status of the solver.
  - `iters`: The number of Interior Point Method iterations.
  - `seconds`: The total solution time in seconds.
  - `solution`: The optimal solution vector `x`.
  - `objective_value`: The objective value of the solution.
  - `residual_history`: A history of residual norms during the optimization.
"""
function solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number} = Float64)
    lp = construct_tulip_model(netw, float_type, config)
    Tulip.optimize!(lp)

    status = Tulip.get_attribute(lp, Tulip.Status())
    iters = Tulip.get_attribute(lp, Tulip.BarrierIterations())
    seconds = Tulip.get_attribute(lp, Tulip.SolutionTime())
    solution = lp.solution.x
    objective_value = Tulip.get_attribute(lp, Tulip.ObjectiveValue())
    residual_history = lp.solver.kkt.residual_history

    return (; status, iters, seconds, solution, objective_value, residual_history)
end

end # module TulipCHOLMOD
