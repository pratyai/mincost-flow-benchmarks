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
- `pcg_tol::Tv`: Tolerance for the Preconditioned Conjugate Gradient (PCG) solver
  used within the approximate Cholesky factorization.
- `red_black::Bool`: Whether to use Independent Set (Red-Black) KKT reduction.
"""
Base.@kwdef struct Backend{Tv<:Number} <: AbstractKKTBackend
    params::ApproxCholParams = ApproxCholParams(:deg, 0, 2, 2)
    pcg_maxits::Int = 100
    pcg_tol::Tv = 5e-8
    red_black::Bool = false
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
- `pcg_tol::Tv`: Tolerance for the PCG solver.
- `θ::Vector{Tv}`: Diagonal scaling vector.
- `regP::Vector{Tv}`: Primal regularization vector.
- `regD::Vector{Tv}`: Dual regularization vector.
- `K::SparseMatrixCSC{Tv,Ti}`: The KKT matrix (reduced if red_black enabled).
- `ξ::Vector{Tv}`: Right-hand side vector of the KKT system (reduced if red_black enabled).
- `sddm_solve::Function`: Function to solve the SDDM system, yielding `dy`.
- `ipm_iter::Int`: Current Interior Point Method iteration count.
- `solve_in_iter::Int`: Counter for solves within the current IPM iteration.
- `residual_history::Vector{Tuple{Int, Int, Tv, Tv}}`: History of residual norms.
- `pcg_iterations_history::Vector{Int}`: History of PCG iterations for each solve.
"""
Base.@kwdef mutable struct Solver{Tv<:Number,Ti<:Integer} <: AbstractKKTSolver{Tv}
    # Problem data
    m::Ti
    n::Ti
    A::AbstractSparseMatrix{Tv,Ti}
    # Laplacians related
    params::ApproxCholParams  # Parameter to use when constructing the SDDM solver
    pcg_maxits::Int  # Maximum PCG iterations
    pcg_tol::Tv  # PCG tolerance

    # Workspace
    θ::Vector{Tv} # Diagonal scaling
    regP::Vector{Tv} # Primal regularization
    regD::Vector{Tv} # Dual regularization
    K::SparseMatrixCSC{Tv,Ti} # KKT matrix
    ξ::Vector{Tv} # RHS of KKT system
    # Laplacians related
    sddm_solve::Function  # Solver for the SDDM system that yields `dy`

    # Red-Black Reduction
    meta::Union{Nothing, SolverCommon.ReductionMetadata} = nothing
    K_full::Union{Nothing, SparseMatrixCSC{Tv,Ti}} = nothing
    ξ_full::Union{Nothing, Vector{Tv}} = nothing

    # Solution quality
    ipm_iter::Int
    solve_in_iter::Int
    residual_history::Vector{Tuple{Int,Int,Tv,Tv}}
    pcg_iterations_history::Vector{Int}
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
    local ξ_init = zeros(Tv, m)
    local K_init = sparse(A * A') + spdiagm(0 => regD)

    local meta = nothing
    local K_to_factor = K_init
    if bk.red_black
        println("TulipApproxChol: Starting Red-Black reduction...")
        t_pre = time()
        meta = SolverCommon.get_reduction_metadata(m, SolverCommon.find_independent_set(sparse(A)))
        K_to_factor, _ = SolverCommon.reduce_kkt_system(K_init, ξ_init, meta)
        println("TulipApproxChol: Red-Black reduction complete ($(round(time()-t_pre, digits=4))s). Reduced nodes: $(m) -> $(meta.n_Sc)")
    end

    local sddm_solve = approxchol_sddm(
        sparse(Symmetric(K_to_factor));
        params = bk.params,
        maxits = bk.pcg_maxits,
        tol = bk.pcg_tol,
    )

    return Solver{Tv,Ti}(;
        m = m,
        n = n,
        A = A,
        params = bk.params,
        pcg_maxits = bk.pcg_maxits,
        pcg_tol = bk.pcg_tol,
        θ = θ,
        regP = regP,
        regD = regD,
        K = K_to_factor,
        ξ = zeros(Tv, size(K_to_factor, 1)),
        sddm_solve = sddm_solve,
        meta = meta,
        K_full = bk.red_black ? K_init : nothing,
        ξ_full = bk.red_black ? ξ_init : nothing,
        ipm_iter = 0,
        solve_in_iter = 0,
        residual_history = Tuple{Int,Int,Tv,Tv}[],
        pcg_iterations_history = [],
    )
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
    local K_new = (kkt.A * D * kkt.A') + spdiagm(0 => kkt.regD)

    if kkt.meta !== nothing
        kkt.K_full = K_new
        kkt.K, _ = SolverCommon.reduce_kkt_system(K_new, zeros(Tv, m), kkt.meta)
    else
        kkt.K = K_new
    end

    kkt.sddm_solve = approxchol_sddm(
        sparse(Symmetric(kkt.K));
        params = kkt.params,
        maxits = kkt.pcg_maxits,
        tol = kkt.pcg_tol,
    )

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
    
    # Form RHS
    local target_ξ = (kkt.meta !== nothing) ? kkt.ξ_full : kkt.ξ
    copyto!(target_ξ, ξp)
    mul!(target_ξ, kkt.A, d .* ξd, true, true)

    # Solve normal equations
    local current_pcg_its = [0] # Initialize with a single element for pcg to set
    if kkt.meta !== nothing
        # Reduce
        _, kkt.ξ = SolverCommon.reduce_kkt_system(kkt.K_full, kkt.ξ_full, kkt.meta)
        dy_reduced = kkt.sddm_solve(kkt.ξ; maxits = kkt.pcg_maxits, pcgIts = current_pcg_its)
        # Reconstruct
        dy .= SolverCommon.reconstruct_solution(dy_reduced, kkt.K_full, kkt.ξ_full, kkt.meta)
    else
        dy .= kkt.sddm_solve(kkt.ξ; maxits = kkt.pcg_maxits, pcgIts = current_pcg_its)
    end
    push!(kkt.pcg_iterations_history, current_pcg_its[1]) # Append the single iteration count

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
  including `kustom_parameters` for `ApproxCholParams`, `pcg_maxits`, and `pcg_tol`.

# Returns
- A `Tulip.Model{Tv}` instance ready for optimization with the approximate Cholesky backend.
"""
function construct_tulip_model(
    netw::Dimacs.McfpNet,
    ::Type{Tv},
    config::Dict,
) where {Tv<:Number}
    lp = SolverCommon.create_tulip_model(netw, Tv)

    Tulip.set_parameter(lp, "OutputLevel", 1)  # enable output
    Tulip.set_parameter(lp, "Presolve_Level", 0)  # disable presolve
    Tulip.set_parameter(lp, "KKT_System", Tulip.KKT.K1())

    kustom_params = get(config, "kustom_parameters", Dict())
    pcg_maxits = get(kustom_params, "pcg_maxits", 100)
    pcg_tol = get(kustom_params, "pcg_tol", 5e-8)
    red_black = get(kustom_params, "RedBlack", false)

    approxchol_params_dict = get(kustom_params, "ApproxCholParams", Dict())
    approxchol_type = Symbol(get(approxchol_params_dict, "type", "deg"))
    approxchol_stag_test = get(approxchol_params_dict, "stag_test", 0)
    approxchol_split = get(approxchol_params_dict, "split", 2)
    approxchol_merge = get(approxchol_params_dict, "merge", 2)
    approxchol_params = ApproxCholParams(
        approxchol_type,
        approxchol_stag_test,
        approxchol_split,
        approxchol_merge,
    )

    Tulip.set_parameter(
        lp,
        "KKT_Backend",
        Kustom.Backend{Tv}(;
            params = approxchol_params,
            pcg_maxits = pcg_maxits,
            pcg_tol = Tv(pcg_tol),
            red_black = red_black,
        ),
    )



    params = get(config, "parameters", Dict())
    # Explicitly set parameters, ensuring correct types
    Tulip.set_parameter(lp, "IPM_PRegMin", Tv(get(params, "IPM_PRegMin", 1e-4)))
    Tulip.set_parameter(lp, "IPM_DRegMin", Tv(get(params, "IPM_DRegMin", 1e-8)))
    Tulip.set_parameter(
        lp,
        "IPM_IterationsLimit",
        Int(get(params, "IPM_IterationsLimit", 200)),
    )

    return lp
end

"""
    solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number})

Solves a Minimum Cost Flow Problem (MCFP) using the TulipApproxChol solver.
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
  - `fact_s`: Factorization time in seconds.
  - `solv_s`: KKT system solve time in seconds.
  - `sddm_calls`: Number of SDDM solver calls.
  - `residual_history`: A history of residual norms during the optimization.
  - `pcg_iterations_history`: A history of PCG iterations for each solve.
"""
function solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number} = Float64)
    # Check if we should use presolve (currently tied to RedBlack parameter)
    kustom_params = get(config, "kustom_parameters", Dict())
    use_presolve = get(kustom_params, "RedBlack", false)
    
    local mapping = nothing
    local num_mapping = nothing
    local active_netw = netw
    
    if use_presolve
        println("TulipApproxChol: Running structural presolve...")
        t_pre = time()
        active_netw, mapping = SolverCommon.simplify_problem(netw)
        
        println("TulipApproxChol: Running numerical equilibration...")
        t_num = time()
        num_mapping, sc_costs, sc_caps, sc_demands = SolverCommon.equilibrate_problem(active_netw)
        println("TulipApproxChol: Numerical equilibration complete ($(round(time()-t_num, digits=4))s). Obj scale: $(num_mapping.obj_scale)")
        
        active_netw = Dimacs.McfpNet(
            G = active_netw.G,
            Cost = float_type.(sc_costs),
            Cap = float_type.(sc_caps),
            Demand = float_type.(sc_demands)
        )
        
        println("TulipApproxChol: Presolve complete ($(round(time()-t_pre, digits=4))s). Nodes: $(netw.G.n) -> $(active_netw.G.n)")
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

    # Extract additional metrics from the solver timer
    to = lp.solver.timer
    fact_ns = TimerOutputs.time(to["Main loop"]["Step"]["Factorization"])
    solv_ns = TimerOutputs.time(to["Main loop"]["Step"]["Newton"]["KKT"])
    sddm_calls = TimerOutputs.ncalls(to["Main loop"]["Step"]["Newton"]["KKT"])

    residual_history = lp.solver.kkt.residual_history
    pcg_iterations_history = lp.solver.kkt.pcg_iterations_history

    return (;
        status,
        iters,
        seconds,
        solution,
        objective_value,
        fact_s = fact_ns * 1e-9,
        solv_s = solv_ns * 1e-9,
        sddm_calls,
        residual_history,
        pcg_iterations_history,
    )
end

end # module TulipApproxChol
