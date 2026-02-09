module MehrotraSolver

using SparseArrays
using LinearAlgebra
using SuiteSparse
using Metis
using Dimacs
using Printf

include("common.jl")
using .SolverCommon

"""
    solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number} = Float64)

Mehrotra Predictor-Corrector IPM for MCFP with robust initialization and centering start.
"""
function solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number} = Float64)
    # 1. Setup Data & Internal Scaling
    n_nodes_full = netw.G.n
    n_edges = netw.G.m
    A_full = sparse(netw.G.IncidenceMatrix)
    
    c_orig = Float64.(netw.Cost)
    b_orig = Float64.(netw.Demand)
    u_orig = Float64.(netw.Cap)
    
    scale_c = max(1.0, norm(c_orig, Inf))
    scale_b = max(1.0, norm(b_orig, Inf))
    
    c = c_orig ./ scale_c
    b = b_orig ./ scale_b
    u = u_orig ./ scale_b
    
    # Drop last row for full rank
    A = A_full[1:n_nodes_full-1, :]
    b_red = b[1:n_nodes_full-1]
    n_nodes = n_nodes_full - 1

    # 2. Initialization (Wright/Mehrotra Standard)
    K_init = A * A' + spdiagm(0 => fill(1e-6, n_nodes))
    chol_init = SuiteSparse.CHOLMOD.cholesky(Symmetric(K_init))
    
    # Primal LS: min ||x|| s.t. Ax = b
    x_tilde = A' * (chol_init \ b_red)
    # Dual LS: min ||y|| s.t. A'y + z - w = c
    y = chol_init \ (A * c)
    zw_tilde = c - A' * y
    
    # Ensure x and s are positive and stay within bounds [0, u]
    # Standard MPC start for bounds: max(delta, x_tilde)
    x = [min(max(0.1, x_tilde[i]), u[i] - 0.1) for i in 1:n_edges]
    # For edges with small capacity, be even more conservative
    for i in 1:n_edges
        if u[i] < 0.2
            x[i] = 0.5 * u[i]
        end
    end
    s = u .- x
    
    # Duals: ensure z, w are positive
    z = [max(0.1, zw_tilde[i]) for i in 1:n_edges]
    w = [max(0.1, z[i] - zw_tilde[i]) for i in 1:n_edges]
    
    # Further refinement: balance mu
    mu = (dot(x, z) + dot(s, w)) / (2 * n_edges)
    delta_x = 0.5 * mu / sum(z); x .+= delta_x; s .+= delta_x # Naive but helps centering
    delta_z = 0.5 * mu / sum(x); z .+= delta_z; w .+= delta_z

    # 3. Symbolic Factorization
    K_pattern = sparse(A * A') + spdiagm(0 => ones(n_nodes))
    K_cholmod = SuiteSparse.CHOLMOD.Sparse(Symmetric(K_pattern))
    perm, _ = Metis.permutation(K_pattern)
    F = SuiteSparse.CHOLMOD.symbolic(K_cholmod; perm = perm)

    # 4. Parameters
    max_iter = get(get(config, "parameters", Dict()), "max_iter", 100)
    tol = get(get(config, "parameters", Dict()), "tol", 1e-8)
    verbose = get(get(config, "parameters", Dict()), "verbose", true)

    # 5. IPM Loop
    iters = 0
    status = :MaxIter
    t_start = time()
    residual_history = Tuple{Int,Int,Float64,Float64}[]
    
    # Adaptive Centering Control
    centering_phase = true
    centering_tol = 1e-3
    max_centering_iters = 10

    function get_step(v, dv, tau)
        alpha = 1.0
        for i in 1:length(v)
            if dv[i] < -1e-15
                alpha = min(alpha, -tau * v[i] / dv[i])
            end
        end
        return alpha
    end

    function solve_kkt(chol, K, rhs)
        sol = chol \ rhs
        res = rhs - K * sol
        sol .+= chol \ res
        return sol
    end

    if verbose
        @printf("%3s  %10s  %10s  %10s  %10s  %10s\n", "It", "obj", "pres", "dres", "mu", "gap")
    end

    while iters < max_iter
        iters += 1

        # Calculate current residuals
        rp = b_red - A * x
        rd = c - A' * y - z + w
        rs = u - x - s
        
        # Complementarity and objective metrics
        mu = (dot(x, z) + dot(s, w)) / (2 * n_edges)
        obj_p = dot(c, x)
        obj_d = dot(b_red, y) - dot(u, w)
        gap = abs(obj_p - obj_d) / (1 + abs(obj_p))
        
        # Normalized residuals
        res_p = norm(rp, Inf) / (1 + norm(b_red, Inf))
        res_d = norm(rd, Inf) / (1 + norm(c, Inf))
        
        if verbose
            @printf("%3d  %10.4e  %10.2e  %10.2e  %10.2e  %10.2e\n", iters, obj_p*scale_c*scale_b, res_p, res_d, mu, gap)
        end
        push!(residual_history, (iters, 1, res_p, res_d))

        # Check termination (only if not in centering phase)
        if !centering_phase && res_p < tol && res_d < tol && gap < tol
            status = :Optimal
            break
        end

        # Weights
        theta_inv = z ./ x .+ w ./ s
        d_vec = 1.0 ./ clamp.(theta_inv, 1e-16, 1e16)
        K = A * spdiagm(0 => d_vec) * A' + spdiagm(0 => fill(1e-11, n_nodes))

        try
            chol_factor = SuiteSparse.CHOLMOD.cholesky!(F, SuiteSparse.CHOLMOD.Sparse(Symmetric(K)))

            # --- Predictor Step ---
            r_xz_aff = -x .* z
            r_sw_aff = -s .* w
            term_aff = rd .- r_xz_aff ./ x .+ (r_sw_aff .- w .* rs) ./ s
            rhs_aff = rp + A * (d_vec .* term_aff)
            
            dy_aff = solve_kkt(chol_factor, K, rhs_aff)
            dx_aff = d_vec .* (A' * dy_aff .- term_aff)
            ds_aff = rs .- dx_aff
            dz_aff = (r_xz_aff .- z .* dx_aff) ./ x
            dw_aff = (r_sw_aff .- w .* ds_aff) ./ s

            # Calculate sigma
            if centering_phase
                sigma = 1.0
                if (res_p < centering_tol && res_d < centering_tol) || iters >= max_centering_iters
                    centering_phase = false
                    if verbose
                        println("MehrotraSolver: Centering complete ($iters iters). Starting optimization...")
                    end
                end
            else
                a_p_aff = min(get_step(x, dx_aff, 1.0), get_step(s, ds_aff, 1.0))
                a_d_aff = min(get_step(z, dz_aff, 1.0), get_step(w, dw_aff, 1.0))
                mu_aff = (dot(x + a_p_aff*dx_aff, z + a_d_aff*dz_aff) + 
                          dot(s + a_p_aff*ds_aff, w + a_d_aff*dw_aff)) / (2 * n_edges)
                sigma = clamp((mu_aff / mu)^3, 0.0, 1.0)
            end

            # --- Combined Step ---
            r_xz_corr = sigma * mu .- x .* z .- dx_aff .* dz_aff
            r_sw_corr = sigma * mu .- s .* w .- ds_aff .* dw_aff
            term_corr = rd .- r_xz_corr ./ x .+ (r_sw_corr .- w .* rs) ./ s
            rhs_corr = rp + A * (d_vec .* term_corr)
            
            dy = solve_kkt(chol_factor, K, rhs_corr)
            dx = d_vec .* (A' * dy .- term_corr)
            ds = rs .- dx
            dz = (r_xz_corr .- z .* dx) ./ x
            dw = (r_sw_corr .- w .* ds) ./ s

            tau = max(0.99, 1.0 - mu)
            a_p = min(get_step(x, dx, tau), get_step(s, ds, tau))
            a_d = min(get_step(z, dz, tau), get_step(w, dw, tau))

            x .+= a_p .* dx
            s .+= a_p .* ds
            y .+= a_d .* dy
            z .+= a_d .* dz
            w .+= a_d .* dw
            
            if any(isnan, x) || any(isnan, y) || any(isnan, z) || any(isnan, w)
                error("MehrotraSolver: NaN detected at iteration $iters")
            end
            
        catch e
            rethrow(e)
        end
    end

    seconds = time() - t_start
    objective_value = dot(c_orig, x .* scale_b)
    solution = float_type.(x .* scale_b)

    return (; status, iters, seconds, solution, objective_value, residual_history)
end

end # module
