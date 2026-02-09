module RandomizedSparseSolver

using SparseArrays
using LinearAlgebra
using SuiteSparse
using Metis
using Dimacs
using Printf
using Random

include("common.jl")
using .SolverCommon

# ==============================================================================
# Helper Types & Structures
# ==============================================================================

"""
    Checkpoint{T}

Snapshot of the solver's primal-dual state. Used for "True Rollback" to recover
from speculative fixing errors without expensive re-stabilization.
"""
struct Checkpoint{T}
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}
    w::Vector{T}
    active_mask::Vector{Bool}
    fixed_vals::Vector{T}
    bar_mask_low::Vector{Bool}
    bar_mask_high::Vector{Bool}
    mu::T
end

"""
    SolverState{T<:Number}

Holds the mutable state of the solver during execution.

# Fields
- `A_full`: The full rank constraint matrix (m x n-1).
- `x`, `s`: Primal variables (flow, capacity slack).
- `y`: Dual variables (node potentials).
- `z`, `w`: Dual slacks (reduced cost components).
- `active_mask`: Boolean mask of edges currently in the "active" IPM system.
- `fixed_vals`: Values (0 or capacity) for edges fixed by the sparse logic.
- `fix_history`: Stack of fixed edge indices for rollback.
- `fix_countdown`: Number of iterations to suspend fixing after a rollback.
- `tau_scale`: Adaptive scaling factor for the fixing threshold.
- `checkpoint`: Last known good state for rollback.
- `bar_mask_low`, `bar_mask_high`: Masks for barrier suspension (True = Active).
- `active_degrees`: Vector storing the count of active edges incident to each node.
"""
mutable struct SolverState{T<:Number}
    # Problem Data
    n_nodes::Int
    n_edges::Int
    A_full::SparseMatrixCSC{T, Int}
    b::Vector{T}
    c::Vector{T}
    u::Vector{T}
    c_orig::Vector{T} # For final reporting
    scale_c::T
    scale_b::T

    # Variables
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}
    w::Vector{T}

    # Solver Status
    active_mask::Vector{Bool}
    fixed_vals::Vector{T}
    fix_history::Vector{Vector{Int}}
    fix_countdown::Int
    tau_scale::T
    active_degrees::Vector{Int} # Combinatorial tracking
    
    # Barrier Suspension Status
    bar_mask_low::Vector{Bool}
    bar_mask_high::Vector{Bool}
    
    # State Checkpoint for True Rollback
    checkpoint::Union{Checkpoint{T}, Nothing}
    
    # Linear Algebra
    F_full::SuiteSparse.CHOLMOD.Factor{T}
    
    # Iteration Stats
    iters::Int
    t_start::Float64
end

# ==============================================================================
# Initialization
# ==============================================================================

function setup_solver_state(netw::Dimacs.McfpNet, float_type::Type{<:Number})
    # 1. Setup Data & Full-Rank Matrix
    n_nodes_full = netw.G.n
    n_edges = netw.G.m
    A_orig_full = sparse(netw.G.IncidenceMatrix)
    
    # Remove last node to ensure full rank
    # Also cast to float_type here to avoid type instability later
    A_full = SparseMatrixCSC{float_type, Int}(A_orig_full[1:n_nodes_full-1, :])
    b_full_red = float_type.(netw.Demand)[1:n_nodes_full-1]
    c_orig = float_type.(netw.Cost)
    u_orig = float_type.(netw.Cap)
    n_nodes = n_nodes_full - 1

    # Capacity Capping (avoid Inf in numerical ops)
    u_full = [isinf(val) || val > 1e10 ? 1e10 : val for val in u_orig]

    # Data Scaling (Crucial for stability)
    scale_c = max(1.0, norm(c_orig, Inf))
    scale_b = max(1.0, norm(b_full_red, Inf))
    c = c_orig ./ scale_c
    b = b_full_red ./ scale_b
    u = u_full ./ scale_b

    # Initial Pruning (Zero capacity edges are permanently fixed to 0)
    active_mask = ones(Bool, n_edges)
    fixed_vals = zeros(float_type, n_edges)
    for i in 1:n_edges
        if u[i] < 1e-12
            active_mask[i] = false
            fixed_vals[i] = 0.0
        end
    end
    
    # Initialize Active Degrees
    # We must scan A_full to count active edges per node.
    active_degrees = zeros(Int, n_nodes)
    for j in 1:n_edges
        if active_mask[j]
            p = A_full.colptr[j]
            q = A_full.colptr[j+1]-1
            for k in p:q
                row = A_full.rowval[k]
                active_degrees[row] += 1
            end
        end
    end

    # Initial Point: Start centered in the bounds (Analytic Center of box)
    x = u ./ 2.0
    s = u .- x
    y = zeros(float_type, n_nodes)
    z = ones(float_type, n_edges)
    w = ones(float_type, n_edges)
    
    # Barriers are all active initially
    bar_mask_low = fill(true, n_edges)
    bar_mask_high = fill(true, n_edges)

    # Symbolic Factorization (Pre-computed on the full graph topology)
    # We use A*A' + I regularization, so the sparsity pattern is static.
    # This allows us to reuse the symbolic analysis even as we drop edges.
    K_pattern = A_full * A_full' + spdiagm(0 => ones(n_nodes))
    perm_full, _ = Metis.permutation(K_pattern)
    F_full = SuiteSparse.CHOLMOD.symbolic(SuiteSparse.CHOLMOD.Sparse(Symmetric(K_pattern)); perm = perm_full)

    return SolverState(
        n_nodes, n_edges, A_full, b, c, u, c_orig, scale_c, scale_b,
        x, s, y, z, w,
        active_mask, fixed_vals, Vector{Vector{Int}}(), 0, 1.0, active_degrees,
        bar_mask_low, bar_mask_high,
        nothing, # No checkpoint initially
        F_full, 0, time()
    )
end

# ==============================================================================
# Core Math Helpers
# ==============================================================================

# Standard IPM Ratio Test
function get_step(v, dv, tau)
    alpha = 1.0
    for i in 1:length(v)
        if dv[i] < -1e-15
            alpha = min(alpha, -tau * v[i] / dv[i])
        end
    end
    return alpha
end

"""
    get_rc(edge_idx, c_val, A_mat, y_vec)

Efficiently computes reduced cost `rc = c - A'y` for a single edge.
Directly accesses the CSC column to avoid calculating the full vector.
"""
function get_rc(edge_idx, c_val, A_mat, y_vec)
    val = c_val
    start_ptr = A_mat.colptr[edge_idx]
    end_ptr = A_mat.colptr[edge_idx+1] - 1
    for k in start_ptr:end_ptr
        row_idx = A_mat.rowval[k]
        coeff = A_mat.nzval[k]
        val -= coeff * y_vec[row_idx]
    end
    return val
end

"""
    sync_fixed_duals!(state::SolverState, mu)

Updates the dual slacks (z, w) for fixed (inactive) edges to satisfy KKT conditions.
For a fixed variable x_i = 0, we must have w_i = 0, z_i = rc (if rc > 0).
For a fixed variable x_i = u, we must have z_i = 0, w_i = -rc (if rc < 0).
This ensures that even 'inactive' edges contribute correctly to the dual gap.
"""
function sync_fixed_duals!(state::SolverState, mu)
    eps_mu = max(1e-14, mu * 0.01)
    z = state.z; w = state.w; y = state.y
    for i in 1:length(z)
        if !state.active_mask[i]
            rc = get_rc(i, state.c[i], state.A_full, y)
            
            # Use fixed value to determine which bound is active
            if state.fixed_vals[i] < 0.5 * state.u[i] 
                # Fixed at Lower Bound (0) => z absorbs rc
                z[i] = max(eps_mu, rc)
                w[i] = max(eps_mu, z[i] - rc)
            else
                # Fixed at Upper Bound (u) => w absorbs -rc
                w[i] = max(eps_mu, -rc)
                z[i] = max(eps_mu, w[i] + rc)
            end
        end
    end
end

"""
    unfix_edge!(idx, state)

Re-activates an edge that was previously fixed.
Updates combinatorial degree tracking.
CRITICAL: We must nudge the primal value 'x' slightly off the bound.
Standard IPM requires strictly interior points (x > 0, s > 0).
Starting exactly at a bound (0 or u) causes division by zero in theta calculation.
"""
function unfix_edge!(idx, state::SolverState)
    state.active_mask[idx] = true
    # Reactivate barriers
    state.bar_mask_low[idx] = true
    state.bar_mask_high[idx] = true
    
    # Update Active Degrees
    p = state.A_full.colptr[idx]
    q = state.A_full.colptr[idx+1]-1
    for k in p:q
        node_idx = state.A_full.rowval[k]
        state.active_degrees[node_idx] += 1
    end
    
    # Nudge off the bound to satisfy strict interiority
    if state.fixed_vals[idx] == 0.0
        state.x[idx] = max(1e-8, 1e-4 * state.u[idx])
    else
        state.x[idx] = min(state.u[idx] - 1e-8, (1.0 - 1e-4) * state.u[idx])
    end
    state.s[idx] = state.u[idx] - state.x[idx]
end

# ==============================================================================
# Checkpointing Logic
# ==============================================================================

"""
    save_checkpoint!(state, mu_curr)

Saves a deep copy of the current primal-dual state.
"""
function save_checkpoint!(state::SolverState, mu_curr)
    state.checkpoint = Checkpoint(
        copy(state.x), copy(state.s), copy(state.y), copy(state.z), copy(state.w),
        copy(state.active_mask), copy(state.fixed_vals), 
        copy(state.bar_mask_low), copy(state.bar_mask_high),
        mu_curr
    )
end

"""
    restore_checkpoint!(state)

Restores the solver state from the last checkpoint.
Returns true if successful, false if no checkpoint exists.
This allows us to 'time travel' back to a valid state before a bad speculative fix.
"""
function restore_checkpoint!(state::SolverState)
    if state.checkpoint === nothing
        return false
    end
    cp = state.checkpoint
    state.x .= cp.x
    state.s .= cp.s
    state.y .= cp.y
    state.z .= cp.z
    state.w .= cp.w
    state.active_mask .= cp.active_mask
    state.fixed_vals .= cp.fixed_vals
    
    # Restore barrier masks to conservative (all active) on rollback
    # This is safer than restoring potentially aggressive suspensions
    fill!(state.bar_mask_low, true)
    fill!(state.bar_mask_high, true)
    
    # Re-calculate active degrees from scratch based on restored mask
    # This is robust against drift
    fill!(state.active_degrees, 0)
    for j in 1:state.n_edges
        if state.active_mask[j]
            p = state.A_full.colptr[j]
            q = state.A_full.colptr[j+1]-1
            for k in p:q
                row = state.A_full.rowval[k]
                state.active_degrees[row] += 1
            end
        end
    end
    
    return true
end

# ==============================================================================
# IPM Logic (With Barrier Suspension)
# ==============================================================================

function perform_ipm_step!(state::SolverState, mode::Symbol)
    # 1. Prepare Subsystem Indices
    idx_a = findall(state.active_mask)
    n_a = length(idx_a)
    idx_f = findall(.!state.active_mask)

    # Shift RHS to account for flow on fixed edges: Ax = b becomes A_a x_a = b - A_f x_f
    b_shift = zeros(state.n_nodes)
    if !isempty(idx_f)
        b_shift = state.A_full[:, idx_f] * state.fixed_vals[idx_f]
    end
    b_eff = state.b - b_shift

    # Views for active set
    A_s = state.A_full[:, state.active_mask]
    c_s = state.c[state.active_mask]; u_s = state.u[state.active_mask]
    x_s = state.x[state.active_mask]; s_s = state.s[state.active_mask]
    z_s = state.z[state.active_mask]; w_s = state.w[state.active_mask]
    
    # Active set masks for barrier suspension
    mask_low_s = state.bar_mask_low[state.active_mask]
    mask_high_s = state.bar_mask_high[state.active_mask]

    # 2. Factorize Normal Equations
    # K = A * Theta * A'
    # Theta = (X^-1 Z + S^-1 W)^-1 -- standard IPM scaling
    # If a barrier is suspended, the corresponding term (X^-1 Z or S^-1 W) is zero.
    
    eps_T = eps(eltype(state.x))
    # Add epsilon to denominator to prevent division by zero when barriers are suspended
    # We do NOT clamp theta arbitrarily. We trust the logic to fix variables before theta explodes.
    # However, numerical limits exist.
    theta = (x_s .* s_s) ./ (z_s .* s_s .+ w_s .* x_s .+ eps_T)
    
    # Adaptive Regularization:
    # Scale regularization by machine precision and the problem's current stiffness.
    # This makes the solver scale-invariant.
    # We measure stiffness by the mean of theta (trace of D).
    mean_theta = isempty(theta) ? 1.0 : sum(theta) / length(theta)
    K_reg = max(1e-12, eps_T * mean_theta)
    
    K = A_s * spdiagm(0 => theta) * A_s' + spdiagm(0 => fill(K_reg, state.n_nodes))
    
    # Reuse symbolic factorization (pattern is subset of full graph)
    # Adaptive Shift Strategy:
    # Instead of a hardcoded list, we start with 0 and increase by powers of 10 if singular.
    chol = nothing
    shift = 0.0
    for retry in 1:6
        try
            chol = SuiteSparse.CHOLMOD.cholesky!(state.F_full, SuiteSparse.CHOLMOD.Sparse(Symmetric(K)); shift=shift, check=true)
            break
        catch e
            if !isa(e, SuiteSparse.CHOLMOD.CHOLMODException) && !isa(e, LinearAlgebra.PosDefException)
                rethrow(e)
            end
            # Increase shift adaptively
            shift = (shift == 0.0) ? 1e-10 : shift * 10.0
        end
    end
    if chol === nothing
        error("Matrix factorization failed even with large shift.")
    end

    # 3. Residuals & Predictor Step
    rp = b_eff - A_s * x_s
    rc_s = c_s - A_s' * state.y
    rd_s = rc_s - z_s + w_s
    rs = u_s - x_s - s_s

    # Augmented RHS for the linear system
    term_aff = rd_s .- (-x_s .* z_s) ./ x_s .+ (-s_s .* w_s .- w_s .* rs) ./ s_s
    
    dy_aff = chol \ (rp + A_s * (theta .* term_aff))
    dx_aff = theta .* (A_s' * dy_aff .- term_aff)
    ds_aff = rs .- dx_aff
    dz_aff = (-x_s .* z_s .- z_s .* dx_aff) ./ x_s
    dw_aff = (-s_s .* w_s .- w_s .* ds_aff) ./ s_s
    
    # Force zero steps for suspended barriers
    dz_aff[.!mask_low_s] .= 0.0
    dw_aff[.!mask_high_s] .= 0.0

    # 4. Mehrotra's Heuristic for Mu & Sigma
    if n_a > 0
        mu_curr = (dot(x_s, z_s) + dot(s_s, w_s)) / (2 * n_a)
        a_p_aff = min(get_step(x_s, dx_aff, 1.0), get_step(s_s, ds_aff, 1.0))
        a_d_aff = min(get_step(z_s, dz_aff, 1.0), get_step(w_s, dw_aff, 1.0))
        mu_aff = (dot(x_s + a_p_aff*dx_aff, z_s + a_d_aff*dz_aff) + dot(s_s + a_p_aff*ds_aff, w_s + a_d_aff*dw_aff)) / (2*n_a)
        sigma = (mode == :AC) ? 1.0 : clamp((mu_aff/max(mu_curr, 1e-16))^3, 0.0, 1.0)
    else
        # Handle empty active set case gracefully
        mu_curr = 0.0
        sigma = 0.0
        a_p_aff = 1.0; a_d_aff = 1.0
    end

    # 5. Corrector Step
    r_xz = sigma * mu_curr .- x_s .* z_s .- dx_aff .* ((-x_s .* z_s .- z_s .* dx_aff) ./ x_s)
    r_sw = sigma * mu_curr .- s_s .* w_s .- ds_aff .* dw_aff
    
    # Complementarity target is 0 for suspended barriers
    r_xz[.!mask_low_s] .= 0.0
    r_sw[.!mask_high_s] .= 0.0
    
    term_corr = rd_s .- r_xz ./ x_s .+ (r_sw .- w_s .* rs) ./ s_s
    dy = chol \ (rp + A_s * (theta .* term_corr))
    dx = theta .* (A_s' * dy .- term_corr)
    ds = rs .- dx
    dz = (r_xz .- z_s .* dx) ./ x_s
    dw = (r_sw .- w_s .* ds) ./ s_s
    
    dz[.!mask_low_s] .= 0.0
    dw[.!mask_high_s] .= 0.0

    # 6. Update Variables
    tau_step = max(0.99, 1.0 - mu_curr)
    a_p = min(get_step(x_s, dx, tau_step), get_step(s_s, ds, tau_step))
    a_d = min(get_step(z_s, dz, tau_step), get_step(w_s, dw, tau_step))

    x_s .= max.(1e-15, x_s + a_p .* dx); s_s .= max.(1e-15, s_s + a_p .* ds)
    state.y .+= a_d .* dy
    z_s .= max.(1e-15, z_s + a_d .* dz); w_s .= max.(1e-15, w_s + a_d .* dw)

    # Sync back to global state
    state.x[state.active_mask] .= x_s; state.s[state.active_mask] .= s_s
    state.z[state.active_mask] .= z_s; state.w[state.active_mask] .= w_s
    
    mu_global = (dot(state.x, state.z) + dot(state.s, state.w)) / (2 * state.n_edges)
    sync_fixed_duals!(state, mu_global)
    
    return n_a, mu_global
end

# ==============================================================================
# Sparse Logic
# ==============================================================================

function perform_ac_screening!(state::SolverState, config::Dict, verbose::Bool)
    # AC Screening acts as a "cold start" filter. 
    # We rely on the duality gap to set the initial threshold.
    mu_global = (dot(state.x, state.z) + dot(state.s, state.w)) / (2 * state.n_edges)
    tau_curr = 100.0 * mu_global 
    
    fixed_count = 0
    # Global Scan
    for i in 1:state.n_edges
        if state.active_mask[i]
            rc = get_rc(i, state.c[i], state.A_full, state.y)
            
            # Combinatorial Safety Check even during screening
            p = state.A_full.colptr[i]; q = state.A_full.colptr[i+1]-1
            safe = true
            for k in p:q
                node = state.A_full.rowval[k]
                if state.active_degrees[node] <= 1
                    safe = false; break
                end
            end
            if !safe; continue; end

            # Fix if rc is significant relative to current gap
            if rc > tau_curr && state.x[i] < mu_global
                # Fix to Lower Bound
                state.active_mask[i] = false
                state.fixed_vals[i] = 0.0
                state.x[i] = 0.0; state.s[i] = state.u[i]
                for k in p:q; state.active_degrees[state.A_full.rowval[k]] -= 1; end
                fixed_count += 1
            elseif rc < -tau_curr && state.s[i] < mu_global
                # Fix to Upper Bound
                state.active_mask[i] = false
                state.fixed_vals[i] = state.u[i]
                state.x[i] = state.u[i]; state.s[i] = 0.0
                for k in p:q; state.active_degrees[state.A_full.rowval[k]] -= 1; end
                fixed_count += 1
            end
        end
    end
    
    if verbose; println(">> AC Screening: Fixed $fixed_count edges."); end
    # Save checkpoint after screening as a clean baseline for future rollbacks
    save_checkpoint!(state, mu_global)
end

function handle_sparse_logic!(state::SolverState, config::Dict, 
                              gap_ext, pobj, pres, dres, mu_global, verbose::Bool)
    
    params = get(config, "parameters", Dict())
    rollback_window   = get(params, "rollback_window", 3)
    sample_size       = get(params, "sample_size", min(state.n_edges, 5000))
    max_fix_per_round = get(params, "max_fix_per_round", 1000)
    dual_tol          = get(params, "dual_tol", 1e-6)

    # A. Verify & Rollback
    # Check if we broke dual feasibility significantly
    idx_f = findall(.!state.active_mask)
    # Check if dual residual exploded relative to gap
    check_now = (dres > 10.0 * mu_global) || (state.iters % 5 == 0)

    if check_now && !isempty(state.fix_history)
        bad_fixes = false
        for i in idx_f
            rc = get_rc(i, state.c[i], state.A_full, state.y)
            # Check consistency based on where we froze it
            if state.fixed_vals[i] < 0.5 * state.u[i]
                # Frozen Low: requires rc > -tol
                if rc < -dual_tol
                    bad_fixes = true; break
                end
            else
                # Frozen High: requires rc < tol
                if rc > dual_tol
                    bad_fixes = true; break
                end
            end
        end

        if bad_fixes
            state.tau_scale *= 2.0 # Adaptive: become more conservative
            
            # Execute TRUE ROLLBACK Strategy:
            if restore_checkpoint!(state)
                if verbose; println("   !! Dual Violation. Restored Checkpoint. Suspending fixes for 10 iters."); end
                state.fix_countdown = 10
                empty!(state.fix_history) 
                return 
            else
                # Fallback
                if verbose; println("   !! Dual Violation. Fallback Unfix."); end
                for _ in 1:min(length(state.fix_history), rollback_window)
                    batch = pop!(state.fix_history)
                    for idx in batch
                        unfix_edge!(idx, state)
                    end
                end
            end
        end
    end

    # B. Barrier Suspension (Using AC property)
    # Dynamically suspend barriers for constraints the solver is "moving away from"
    # Use relative threshold based on capacity.
    safe_margin = 0.1
    for i in 1:state.n_edges
        if state.active_mask[i]
            # Lower Barrier: Suspend if x is far from 0
            if state.bar_mask_low[i] && state.x[i] > safe_margin * max(1.0, state.u[i])
                state.bar_mask_low[i] = false
                state.z[i] = 0.0 
            elseif !state.bar_mask_low[i] && state.x[i] < 0.05 * max(1.0, state.u[i])
                state.bar_mask_low[i] = true 
            end
            
            # Upper Barrier: Suspend if s is far from 0
            if state.bar_mask_high[i] && state.s[i] > safe_margin * max(1.0, state.u[i])
                state.bar_mask_high[i] = false
                state.w[i] = 0.0
            elseif !state.bar_mask_high[i] && state.s[i] < 0.05 * max(1.0, state.u[i])
                state.bar_mask_high[i] = true
            end
        end
    end

    # C. Speculative Fixing (Combinatorial Feasibility Aware)
    if state.fix_countdown > 0
        state.fix_countdown -= 1
        return
    end

    # Only fix when we are in the "active" phase of optimization (mu < 1e-2)
    if mu_global < 1e-2
        active_indices = findall(state.active_mask)
        n_sample = min(length(active_indices), sample_size)
        candidates = rand(active_indices, n_sample)
        
        fixes_this_round = Int[]
        # Adaptive Threshold: fixing threshold scales with the duality gap.
        # If rc >> mu, the edge is costly relative to current precision.
        tau_curr = 10.0 * mu_global * state.tau_scale
        
        limit_logN = max(1, floor(Int, log(state.n_nodes)))
        max_to_fix = min(max_fix_per_round, limit_logN)

        for i in candidates
            if length(fixes_this_round) >= max_to_fix; break; end
            
            # Combinatorial Gate: Degree Check
            # STRICT RULE: Do not isolate nodes.
            # We check if removing edge i would reduce any endpoint degree to <= 1.
            p = state.A_full.colptr[i]; q = state.A_full.colptr[i+1]-1
            is_combinatorially_safe = true
            
            for k in p:q
                node_idx = state.A_full.rowval[k]
                if state.active_degrees[node_idx] <= 1
                    is_combinatorially_safe = false
                    break
                end
            end
            
            if !is_combinatorially_safe
                continue
            end

            # Pricing
            rc = get_rc(i, state.c[i], state.A_full, state.y)
            
            # Fix if rc is significant AND primal is essentially at bound (relative to mu)
            # Threshold: x < mu. If x is smaller than the duality gap, it's effectively zero.
            if rc > tau_curr && state.x[i] < mu_global
                state.active_mask[i] = false
                state.fixed_vals[i] = 0.0
                state.x[i] = 0.0; state.s[i] = state.u[i]
                for k in p:q; state.active_degrees[state.A_full.rowval[k]] -= 1; end
                push!(fixes_this_round, i)
            elseif rc < -tau_curr && state.s[i] < mu_global
                state.active_mask[i] = false
                state.fixed_vals[i] = state.u[i]
                state.x[i] = state.u[i]; state.s[i] = 0.0
                for k in p:q; state.active_degrees[state.A_full.rowval[k]] -= 1; end
                push!(fixes_this_round, i)
            end
        end
        if !isempty(fixes_this_round)
            # Success! Decay tau_scale
            state.tau_scale = max(1.0, state.tau_scale * 0.95)
            
            if isempty(state.fix_history)
                save_checkpoint!(state, mu_global)
            end
            push!(state.fix_history, fixes_this_round)
        end
    end
end

# ==============================================================================
# Main Entry Point
# ==============================================================================

"""
    solve(netw, config, float_type)

Orchestrates the Randomized Sparse Solver.
"""
function solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number} = Float64)
    state = setup_solver_state(netw, float_type)
    
    params = get(config, "parameters", Dict())
    max_iters = get(params, "max_iters", 200)
    verbose   = get(params, "verbose", true)
    
    mode = :AC
    
    if verbose
        @printf("%3s  %3s  %8s  %10s  %10s  %10s  %10s  %8s  %8s\n", "Ph", "It", "Active", "pobj", "dobj", "gap_ext", "mu", "pres", "dres")
    end

    while state.iters < max_iters
        state.iters += 1
        
        try
            n_active, mu_global = perform_ipm_step!(state, mode)
            
            # Metrics Calculation
            rp_vec = state.b - state.A_full * state.x
            rd_vec = state.c - state.A_full' * state.y - state.z + state.w
            
            pres = norm(rp_vec, Inf) / (1 + norm(state.b, Inf))
            dres = norm(rd_vec, Inf) / (1 + norm(state.c, Inf))
            
            pobj = dot(state.c_orig, state.x .* state.scale_b)
            dobj = (dot(state.b, state.y) - dot(state.u, state.w)) * state.scale_c * state.scale_b
            gap_compl = (dot(state.x, state.z) + dot(state.s, state.w)) * state.scale_c * state.scale_b
            
            term_inf_p = dot(state.y, rp_vec) * state.scale_c * state.scale_b
            term_inf_d = dot(state.x, rd_vec) * state.scale_c * state.scale_b 
            gap_extended = gap_compl + abs(term_inf_p) + abs(term_inf_d)

            if verbose
                @printf("%2s  %3d  %8d  %10.4e  %10.4e  %10.4e  %10.2e  %8.2e  %8.2e\n", 
                        string(mode), state.iters, n_active, pobj, dobj, gap_extended, mu_global, pres, dres)
            end

            if mode == :AC && pres < 1e-3 && dres < 1e-3
                mode = :Sparse
                if verbose; println(">> Switching to Sparse Mode"); end
                perform_ac_screening!(state, config, verbose)
            end

            if mode == :Sparse
                # Pass plain types, no rp_vec (removed from signature)
                handle_sparse_logic!(state, config, gap_extended, pobj, pres, dres, mu_global, verbose)
            end

            if (gap_extended < 0.5) || (pres < 1e-7 && (gap_compl / (1 + abs(pobj))) < 1e-7)
                if verbose; println(">> Converged. Gap Extended: $(gap_extended)"); end
                break
            end
            
            if any(isnan, state.x)
                error("NaN detected in iterates")
            end

        catch e
            if isa(e, SuiteSparse.CHOLMOD.CHOLMODException)
                println("CHOLMOD Error. Aborting.")
                return (; status=:Error, iters=state.iters, seconds=time()-state.t_start, solution=float_type.(state.x .* state.scale_b), objective_value=NaN, residual_history=[])
            else
                rethrow(e)
            end
        end
    end

    return (; status=:Optimal, iters=state.iters, seconds=time()-state.t_start, solution=float_type.(state.x .* state.scale_b), objective_value=dot(state.c_orig, state.x .* state.scale_b), residual_history=[])
end

end # module