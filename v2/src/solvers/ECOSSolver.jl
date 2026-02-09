module ECOSSolver

using ECOS
using MathOptInterface
const MOI = MathOptInterface
using Dimacs
using SparseArrays
using LinearAlgebra

"""
    solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number})

Solves the Minimum Cost Flow Problem using ECOS via MathOptInterface.
Note: ECOS only supports Float64.
"""
function solve(netw::Dimacs.McfpNet, config::Dict, float_type::Type{<:Number} = Float64)
    if float_type != Float64
        @warn "ECOS only supports Float64. Using Float64."
    end

    # 1. Extract Data
    n_edges = netw.G.m # number of variables
    n_nodes = netw.G.n # number of constraints
    
    c = Float64.(netw.Cost)
    # Convert IncidenceMatrix to SparseMatrixCSC{Float64, Int} for ECOS
    # common.jl used Int8, but MOI expects Float64 coeffs usually.
    A_full = sparse(netw.G.IncidenceMatrix) 
    
    # ECOS requires the equality constraint matrix A to have full row rank.
    # However, Python ECOS solves it fine without dropping rows.
    # We will try using the full matrix first, as dropping might be causing dual scaling issues.
    A = A_full
    
    rows, cols, vals = findnz(A)
    
    b_full = Float64.(netw.Demand)
    b = b_full
    
    u = Float64.(netw.Cap)

    # 2. Setup Optimizer
    # ECOS.Optimizer requires caching to support incremental modification (add_variables)
    optimizer = MOI.instantiate(ECOS.Optimizer; with_cache_type = Float64)
    
    # Check for verbose flag in config (default to false/silent)
    params = get(config, "parameters", Dict())
    verbose = get(params, "verbose", get(config, "verbose", false))
    MOI.set(optimizer, MOI.Silent(), !verbose)
    
    # Pass other parameters to ECOS
    for (key, val) in params
        if key != "verbose"
            MOI.set(optimizer, MOI.RawOptimizerAttribute(key), val)
        end
    end

    # 3. Add Variables
    x = MOI.add_variables(optimizer, n_edges)

    # 4. Add Constraints

    # Bounds: 0 <= x[i] <= u[i]
    # We use a single VectorAffineFunction in Nonnegatives cone:
    # [ x ] >= [ 0 ]
    # [ u - x ] >= [ 0 ]
    
    bound_terms = Vector{MOI.VectorAffineTerm{Float64}}(undef, 2 * n_edges)
    for i in 1:n_edges
        # x_i >= 0
        bound_terms[i] = MOI.VectorAffineTerm(i, MOI.ScalarAffineTerm(1.0, x[i]))
        # u_i - x_i >= 0  =>  -x_i + u_i >= 0
        bound_terms[n_edges + i] = MOI.VectorAffineTerm(n_edges + i, MOI.ScalarAffineTerm(-1.0, x[i]))
    end
    
    bound_constants = zeros(Float64, 2 * n_edges)
    for i in 1:n_edges
        bound_constants[n_edges + i] = u[i]
    end
    
    MOI.add_constraint(optimizer, MOI.VectorAffineFunction(bound_terms, bound_constants), MOI.Nonnegatives(2 * n_edges))

    # Flow Conservation: A * x = b
    # A * x - b = 0
    # Vector constraint: A*x + (-b) \in Zeros(m)
    # Note: We are using the reduced A and b (size n_nodes-1)
    
    terms = Vector{MOI.VectorAffineTerm{Float64}}(undef, length(vals))
    for k in 1:length(vals)
        terms[k] = MOI.VectorAffineTerm(
            Int64(rows[k]), 
            MOI.ScalarAffineTerm(Float64(vals[k]), x[cols[k]])
        )
    end
    
    # Constant vector should be -b because: sum(aij * xj) + const_i = 0 => sum = -const_i
    # We want sum = bi => const_i = -bi
    func = MOI.VectorAffineFunction(terms, -b)
    MOI.add_constraint(optimizer, func, MOI.Zeros(length(b)))

    # 5. Objective
    # min c'x
    # Scale objective to improve numerical stability
    scale_factor = max(1.0, norm(c, Inf))
    c_scaled = c ./ scale_factor
    
    obj_terms = [MOI.ScalarAffineTerm(c_scaled[i], x[i]) for i in 1:n_edges]
    obj_func = MOI.ScalarAffineFunction(obj_terms, 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{typeof(obj_func)}(), obj_func)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # 6. Optimize
    t_start = time()
    MOI.optimize!(optimizer)
    wall_seconds = time() - t_start
    
    seconds = wall_seconds
    try
        seconds = MOI.get(optimizer, MOI.SolveTimeSec())
    catch
        # fallback to wall clock
    end

    # 7. Results
    term_status = MOI.get(optimizer, MOI.TerminationStatus())
    status = map_status(term_status)
    
    if status == :NumericalError || status == :Unknown
        try
            raw = MOI.get(optimizer, MOI.RawStatusString())
            println("ECOS Solver Error. Status: $status. Raw: $raw")
        catch e
             println("ECOS Solver Error. Status: $status. Could not get raw status: $e")
        end
    end
    
    solution = zeros(Float64, n_edges)
    objective_value = 0.0
    
    if status == :Optimal || status == :MaxIter || status == :TimeLimit || status == :NumericalError
        # Try to get solution even if not optimal, ECOS might provide something
        try
            solution .= MOI.get(optimizer, MOI.VariablePrimal(), x)
        catch
            # ignore
        end
        
        try
            val = MOI.get(optimizer, MOI.ObjectiveValue())
            objective_value = val * scale_factor
        catch
        end
    end

    iters = 0 
    # ECOS specific: get iter count? 
    # MOI.get(optimizer, MOI.BarrierIterations()) ?
    try
        iters = MOI.get(optimizer, MOI.BarrierIterations())
    catch
    end

    return (; 
        status, 
        iters, 
        seconds, 
        solution, 
        objective_value, 
        residual_history=Tuple{Int,Int,Float64,Float64}[]
    )
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

end # module