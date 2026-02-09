module GurobiSolver

using Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Dimacs
using SparseArrays
using LinearAlgebra

"""
    solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number})

Solves the Minimum Cost Flow Problem using Gurobi via MathOptInterface.
"""
function solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number} = Float64)
    if float_type != Float64
        @warn "Gurobi only supports Float64. Using Float64."
    end

    n_edges = netw.G.m
    n_nodes = netw.G.n
    
    c = Float64.(netw.Cost)
    A = sparse(netw.G.IncidenceMatrix)
    rows, cols, vals = findnz(A)
    b = Float64.(netw.Demand)
    u = Float64.(netw.Cap)

    # Setup Optimizer
    # Use MOI.instantiate to get default bridges
    optimizer = MOI.instantiate(Gurobi.Optimizer; with_cache_type = Float64)
    
    # Config
    params = get(config, "parameters", Dict())
    verbose = get(params, "verbose", get(config, "verbose", false))
    MOI.set(optimizer, MOI.Silent(), !verbose)
    
    # Pass other parameters
    for (key, val) in params
        if key != "verbose"
            MOI.set(optimizer, MOI.RawOptimizerAttribute(key), val)
        end
    end

    # Variables
    x = MOI.add_variables(optimizer, n_edges)

    # Constraints
    
    # 1. Bounds: 0 <= x[i] <= u[i]
    # Use Scalar constraints for robustness
    MOI.add_constraints(optimizer, x, [MOI.Interval(0.0, u[i]) for i in 1:n_edges])

    # 2. Flow conservation: Ax = b
    # Group terms by row index.
    row_terms = [MOI.ScalarAffineTerm{Float64}[] for _ in 1:n_nodes]
    for k in 1:length(vals)
        r = rows[k]
        c_idx = cols[k]
        val = vals[k]
        push!(row_terms[r], MOI.ScalarAffineTerm(Float64(val), x[c_idx]))
    end
    
    # Create functions and sets
    eq_funcs = [MOI.ScalarAffineFunction(row_terms[i], 0.0) for i in 1:n_nodes]
    eq_sets = [MOI.EqualTo(b[i]) for i in 1:n_nodes]
    
    MOI.add_constraints(optimizer, eq_funcs, eq_sets)

    # Objective
    scale_factor = max(1.0, norm(c, Inf))
    c_scaled = c ./ scale_factor
    obj_terms = [MOI.ScalarAffineTerm(c_scaled[i], x[i]) for i in 1:n_edges]
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(obj_terms, 0.0))
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # Optimize
    t_start = time()
    MOI.optimize!(optimizer)
    wall_seconds = time() - t_start
    
    seconds = wall_seconds
    try
        seconds = MOI.get(optimizer, MOI.SolveTimeSec())
    catch
    end

    # Results
    term_status = MOI.get(optimizer, MOI.TerminationStatus())
    status = map_status(term_status)
    
    solution = zeros(Float64, n_edges)
    objective_value = 0.0
    
    if status in [:Optimal, :MaxIter, :TimeLimit, :NumericalError]
        try
            solution .= MOI.get(optimizer, MOI.VariablePrimal(), x)
        catch
        end
        try
            val = MOI.get(optimizer, MOI.ObjectiveValue())
            objective_value = val * scale_factor
        catch
        end
    end

    iters = 0
    try
        iters = MOI.get(optimizer, MOI.BarrierIterations())
    catch
        # Gurobi might use simplex. 
        # MOI.SimplexIterations() ?
        try
             iters = MOI.get(optimizer, MOI.SimplexIterations())
        catch
        end
    end

    return (; status, iters, seconds, solution, objective_value, residual_history=Tuple{Int,Int,Float64,Float64}[])
end

function map_status(st)
    if st == MOI.OPTIMAL
        return :Optimal
    elseif st == MOI.INFEASIBLE
        return :PrimalInfeasible
    elseif st == MOI.DUAL_INFEASIBLE
        return :DualInfeasible
    elseif st == MOI.ITERATION_LIMIT
        return :MaxIter
    elseif st == MOI.TIME_LIMIT
        return :TimeLimit
    elseif st == MOI.NUMERICAL_ERROR
        return :NumericalError
    else
        return Symbol(st)
    end
end

end
