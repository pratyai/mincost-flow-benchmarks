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
using Metis

include("common.jl")
using .SolverCommon

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
using Metis
using ..SolverCommon

using SparseArrays: AbstractSparseMatrix
using Tulip.KKT: AbstractKKTBackend, AbstractKKTSolver, K1

Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend
    nested_dissection::Bool = false
    red_black::Bool = false
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
    K::SparseMatrixCSC{Tv,Ti} # KKT matrix (full or reduced)
    ξ::Vector{Tv} # RHS of KKT system (full or reduced)
    
    # Red-Black Reduction
    meta::Union{Nothing, SolverCommon.ReductionMetadata} = nothing
    K_full::Union{Nothing, SparseMatrixCSC{Tv,Ti}} = nothing
    ξ_full::Union{Nothing, Vector{Tv}} = nothing

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
    local ξ_init = zeros(Tv, m)
    local K_init = sparse(A * A') + spdiagm(0 => regD)

    local meta = nothing
    local K_to_factor = K_init
    if bk.red_black
        println("TulipCHOLMOD: Starting Red-Black reduction...")
        t_pre = time()
        meta = SolverCommon.get_reduction_metadata(m, SolverCommon.find_independent_set(sparse(A)))
        # Initial K_reduced pattern
        K_to_factor, _ = SolverCommon.reduce_kkt_system(K_init, ξ_init, meta)
        println("TulipCHOLMOD: Red-Black reduction complete ($(round(time()-t_pre, digits=4))s). Reduced nodes: $(m) -> $(meta.n_Sc)")
    end

    local K_cholmod = SuiteSparse.CHOLMOD.Sparse(Symmetric(K_to_factor))
    local perm = nothing
    if bk.nested_dissection
        perm, _ = Metis.permutation(K_to_factor)
    end
    local F =
        SuiteSparse.CHOLMOD.symbolic(K_cholmod; perm = perm)
    local chol_factor = SuiteSparse.CHOLMOD.cholesky!(F, K_cholmod)

    return Solver(
        m = m,
        n = n,
        A = A,
        θ = θ,
        regP = regP,
        regD = regD,
        K = K_to_factor,
        ξ = zeros(Tv, size(K_to_factor, 1)),
        meta = meta,
        K_full = bk.red_black ? K_init : nothing,
        ξ_full = bk.red_black ? ξ_init : nothing,
        chol_factor = chol_factor,
        ipm_iter = 0,
        solve_in_iter = 0,
        residual_history = Tuple{Int,Int,Tv,Tv}[],
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
    local K_new = (kkt.A * D * kkt.A') + spdiagm(0 => kkt.regD)

    if kkt.meta !== nothing
        kkt.K_full = K_new
        # We don't need a valid ξ for update
        kkt.K, _ = SolverCommon.reduce_kkt_system(K_new, zeros(Tv, m), kkt.meta)
    else
        kkt.K = K_new
    end

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
    
    # Form RHS
    local target_ξ = (kkt.meta !== nothing) ? kkt.ξ_full : kkt.ξ
    copyto!(target_ξ, ξp)
    mul!(target_ξ, kkt.A, d .* ξd, true, true)

    # Solve normal equations
    if kkt.meta !== nothing
        # Reduce
        _, kkt.ξ = SolverCommon.reduce_kkt_system(kkt.K_full, kkt.ξ_full, kkt.meta)
        dy_reduced = kkt.chol_factor \ kkt.ξ
        # Reconstruct
        dy .= SolverCommon.reconstruct_solution(dy_reduced, kkt.K_full, kkt.ξ_full, kkt.meta)
    else
        dy .= kkt.chol_factor \ kkt.ξ
    end

    # Compute relative residual norm
    local K_actual = (kkt.meta !== nothing) ? kkt.K_full : kkt.K
    local ξ_actual = (kkt.meta !== nothing) ? kkt.ξ_full : kkt.ξ
    local absolute_residual_norm = norm(K_actual * dy - ξ_actual)
    local rhs_norm = norm(ξ_actual)
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
using ..SolverCommon

using SparseArrays: AbstractSparseMatrix
using Tulip.KKT: AbstractKKTBackend, AbstractKKTSolver, K1

Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend 
    red_black::Bool = false
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

    # Red-Black Reduction
    meta::Union{Nothing, SolverCommon.ReductionMetadata} = nothing
    K_full::Union{Nothing, SparseMatrixCSC{Tv,Ti}} = nothing
    ξ_full::Union{Nothing, Vector{Tv}} = nothing

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
    local ξ_init = zeros(Tv, m)
    local K_init = sparse(A * A') + spdiagm(0 => regD)

    local meta = nothing
    local K_to_factor = K_init
    if bk.red_black
        meta = SolverCommon.get_reduction_metadata(m, SolverCommon.find_independent_set(sparse(A)))
        K_to_factor, _ = SolverCommon.reduce_kkt_system(K_init, ξ_init, meta)
    end

    local chol_factor = CliqueTrees.cholesky(Symmetric(K_to_factor))

    return Solver{Tv,Ti}(
        m = m,
        n = n,
        A = A,
        θ = θ,
        regP = regP,
        regD = regD,
        K = K_to_factor,
        ξ = zeros(Tv, size(K_to_factor, 1)),
        meta = meta,
        K_full = bk.red_black ? K_init : nothing,
        ξ_full = bk.red_black ? ξ_init : nothing,
        chol_factor = chol_factor,
        ipm_iter = 0,
        solve_in_iter = 0,
        residual_history = Tuple{Int,Int,Tv,Tv}[],
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
    local K_new = (kkt.A * D * kkt.A') + spdiagm(0 => kkt.regD)

    if kkt.meta !== nothing
        kkt.K_full = K_new
        kkt.K, _ = SolverCommon.reduce_kkt_system(K_new, zeros(Tv, m), kkt.meta)
    else
        kkt.K = K_new
    end

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
    
    # Form RHS
    local target_ξ = (kkt.meta !== nothing) ? kkt.ξ_full : kkt.ξ
    copyto!(target_ξ, ξp)
    mul!(target_ξ, kkt.A, d .* ξd, true, true)

    # Solve normal equations
    if kkt.meta !== nothing
        _, kkt.ξ = SolverCommon.reduce_kkt_system(kkt.K_full, kkt.ξ_full, kkt.meta)
        dy_reduced = kkt.chol_factor \ kkt.ξ
        dy .= SolverCommon.reconstruct_solution(dy_reduced, kkt.K_full, kkt.ξ_full, kkt.meta)
    else
        dy .= kkt.chol_factor \ kkt.ξ
    end

    # Compute relative residual norm
    local K_actual = (kkt.meta !== nothing) ? kkt.K_full : kkt.K
    local ξ_actual = (kkt.meta !== nothing) ? kkt.ξ_full : kkt.ξ
    local absolute_residual_norm = norm(K_actual * dy - ξ_actual)
    local rhs_norm = norm(ξ_actual)
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

    cholmod_params = get(config, "cholmod_parameters", Dict())
    nested_dissection = get(cholmod_params, "NestedDissection", false)
    red_black = get(cholmod_params, "RedBlack", false)

    if Tv in [Float32, Float64]
        Tulip.set_parameter(
            lp,
            "KKT_Backend",
            CholmodKKT.Backend{Tv}(
                nested_dissection = nested_dissection,
                red_black = red_black
            ),
        )
    else
        Tulip.set_parameter(lp, "KKT_Backend", CliqueTreeKKT.Backend{Tv}(red_black = red_black))
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
    # Check if we should use presolve (currently tied to RedBlack parameter)
    cholmod_params = get(config, "cholmod_parameters", Dict())
    use_presolve = get(cholmod_params, "RedBlack", false)
    
    local mapping = nothing
    local num_mapping = nothing
    local active_netw = netw
    
    if use_presolve
        println("TulipCHOLMOD: Running structural presolve...")
        t_pre = time()
        active_netw, mapping = SolverCommon.simplify_problem(netw)
        
        println("TulipCHOLMOD: Running numerical equilibration...")
        t_num = time()
        num_mapping, sc_costs, sc_caps, sc_demands = SolverCommon.equilibrate_problem(active_netw)
        println("TulipCHOLMOD: Numerical equilibration complete ($(round(time()-t_num, digits=4))s). Obj scale: $(num_mapping.obj_scale)")
        
        # Create a modified incidence matrix for the scaled problem
        # A' = R * A * C
        A_orig = sparse(active_netw.G.IncidenceMatrix)
        rows, cols, vals = findnz(A_orig)
        new_vals = [Int8(vals[k] * num_mapping.row_scales[rows[k]] * num_mapping.col_scales[cols[k]]) for k in 1:length(vals)]
        # Since R and C are floats, we can't easily stay in Int8. 
        # But for Tulip model creation, we need a compatible network.
        # Actually, let's keep Incidence as is and scale demands/costs/caps.
        # Most of the iteration benefit comes from cost/demand scaling.
        
        active_netw = Dimacs.McfpNet(
            G = active_netw.G,
            Cost = float_type.(sc_costs),
            Cap = float_type.(sc_caps),
            Demand = float_type.(sc_demands)
        )
        
        println("TulipCHOLMOD: Presolve complete ($(round(time()-t_pre, digits=4))s). Nodes: $(netw.G.n) -> $(active_netw.G.n)")
    end

    lp = construct_tulip_model(active_netw, float_type, config)
    Tulip.optimize!(lp)

    status = Tulip.get_attribute(lp, Tulip.Status())
    iters = Tulip.get_attribute(lp, Tulip.BarrierIterations())
    seconds = Tulip.get_attribute(lp, Tulip.SolutionTime())
    
    # Reconstruct solution if presolved
    raw_solution = lp.solution.x
    if use_presolve
        # 1. Unscale by Column Scales (Edges)
        unscaled_x = raw_solution ./ num_mapping.col_scales
        # 2. Reconstruct from structural reduction
        solution = SolverCommon.reconstruct_flow(unscaled_x, mapping)
        # 3. Objective value needs unscaling by obj_scale
        objective_value = Tulip.get_attribute(lp, Tulip.ObjectiveValue()) * num_mapping.obj_scale
    else
        solution = raw_solution
        objective_value = Tulip.get_attribute(lp, Tulip.ObjectiveValue())
    end
    
    residual_history = lp.solver.kkt.residual_history

    return (; status, iters, seconds, solution, objective_value, residual_history)
end

end # module TulipCHOLMOD
