module GridGraphs

using Dimacs
using SparseArrays
using Random
using DataStructures # For Queue

function _calculate_max_flow(
    s_node::Int,
    t_node::Int,
    total_nodes::Int,
    dimacs_directed_edges::Vector{Tuple{Int,Int}},
    arc_capacities::Vector{Int},
)
    # Build residual graph representation for Edmonds-Karp
    arc_to_idx = Dict{Tuple{Int,Int},Int}()
    for (i, (u, v)) in enumerate(dimacs_directed_edges)
        arc_to_idx[(u, v)] = i
    end

    max_flow_adj = [Vector{Tuple{Int,Int,Int}}() for _ = 1:total_nodes]
    for (i, (u, v)) in enumerate(dimacs_directed_edges)
        reverse_arc_idx = get(arc_to_idx, (v, u), 0)
        push!(max_flow_adj[u], (v, i, reverse_arc_idx))
    end

    residual_cap = copy(arc_capacities)
    parent_edge_idx = zeros(Int, total_nodes)
    parent_node = zeros(Int, total_nodes)

    max_flow_value = 0

    function bfs_find_path(s, t)
        fill!(parent_node, 0)
        fill!(parent_edge_idx, 0)
        q = Queue{Int}()
        enqueue!(q, s)
        parent_node[s] = -1

        while !isempty(q)
            u = dequeue!(q)

            for (v, cap_idx, rev_cap_idx) in max_flow_adj[u]
                if parent_node[v] == 0 && residual_cap[cap_idx] > 0
                    parent_node[v] = u
                    parent_edge_idx[v] = cap_idx
                    enqueue!(q, v)
                    if v == t
                        return true
                    end
                end
            end
        end
        return false
    end

    while bfs_find_path(s_node, t_node)
        path_flow = typemax(Int)

        v = t_node
        while v != s_node
            u = parent_node[v]
            cap_idx = parent_edge_idx[v]
            path_flow = min(path_flow, residual_cap[cap_idx])
            v = u
        end

        v = t_node
        while v != s_node
            u = parent_node[v]
            cap_idx = parent_edge_idx[v]
            residual_cap[cap_idx] -= path_flow

            rev_cap_idx = -1
            for (neighbor, c_idx, r_idx) in max_flow_adj[u]
                if neighbor == v && c_idx == cap_idx
                    rev_cap_idx = r_idx
                    break
                end
            end
            residual_cap[rev_cap_idx] += path_flow
            v = u
        end
        max_flow_value += path_flow
    end
    return max_flow_value
end



"""
    generate_grid_graph_mcfp(h::Int, w::Int, max_cap::Int, max_cost::Int, rng::AbstractRNG)

Generates a directed grid graph for max-flow min-cost problems, aligning with the
structure and properties of the `ggraph1.f` FORTRAN generator.

Nodes consist of an `h` x `w` grid, plus a single source node `s` and a single sink node `t`.
Arcs are generated from `s` to the first column, from the last column to `t`,
and between adjacent grid nodes (rightwards and downwards).
Capacities, costs, and demands are assigned according to the `ggraph1.f` specification.

This function now includes an Edmonds-Karp max-flow algorithm to calculate the
exact maximum flow, which is then used to set the supply at `s` and demand at `t`,
ensuring the generated problem is feasible, as per `ggraph1.f`.

# Arguments
- `h::Int`: The number of rows in the grid.
- `w::Int`: The number of columns in the grid.
- `max_cap::Int`: Maximum capacity for grid arcs.
- `max_cost::Int`: Maximum cost for grid arcs.
- `rng::AbstractRNG`: A random number generator for reproducibility.

# Returns
- A `Dimacs.McfpNet` object representing the generated grid graph MCFP instance.
"""
function generate_grid_graph_mcfp(
    h::Int,
    w::Int,
    max_cap::Int,
    max_cost::Int,
    rng::AbstractRNG,
)
    num_grid_nodes = h * w
    s_node = num_grid_nodes + 1
    t_node = num_grid_nodes + 2
    total_nodes = num_grid_nodes + 2

    # Map 1-based (row, col) to a 1-based linear node index for grid nodes
    node_idx(r, c) = (r - 1) * w + c

    dimacs_directed_edges = Vector{Tuple{Int,Int}}()
    arc_capacities = Vector{Int}()
    arc_costs = Vector{Int}()

    # Arrays to store accumulated capacities and costs for source/sink arcs
    # as per ggraph1.f's logic
    caps = zeros(Int, h) # For s -> (r,1) arcs
    costs = zeros(Int, h) # For s -> (r,1) arcs
    capt = zeros(Int, h) # For (r,w) -> t arcs
    costt = zeros(Int, h) # For (r,w) -> t arcs

    # --- Grid Arcs (Rightwards and Downwards) ---
    for r = 1:h
        for c = 1:w
            u_idx = node_idx(r, c)

            # Arc to the right: (r, c) -> (r, c+1)
            if c < w
                v_idx = node_idx(r, c + 1)
                cap = rand(rng, 1:max_cap)
                cost = rand(rng, 1:max_cost)
                push!(dimacs_directed_edges, (u_idx, v_idx))
                push!(arc_capacities, cap)
                push!(arc_costs, cost)
                # Add reverse arc with 0 initial capacity and cost
                push!(dimacs_directed_edges, (v_idx, u_idx))
                push!(arc_capacities, 0)
                push!(arc_costs, 0)

                # Accumulate for source/sink arcs if connected to first/last column
                if c == 1 # Arc from (r,1) to (r,2) contributes to caps[r]
                    caps[r] += cap
                    costs[r] = rand(rng, 1:max_cost) # ggraph1.f assigns random cost here
                end
                if c == w - 1 # Arc from (r,w-1) to (r,w) contributes to capt[r]
                    capt[r] += cap
                    costt[r] = rand(rng, 1:max_cost) # ggraph1.f assigns random cost here
                end
            end

            # Arc downwards: (r, c) -> (r+1, c)
            if r < h
                v_idx = node_idx(r + 1, c)
                cap = rand(rng, 1:max_cap)
                cost = rand(rng, 1:max_cost)
                push!(dimacs_directed_edges, (u_idx, v_idx))
                push!(arc_capacities, cap)
                push!(arc_costs, cost)
                # Add reverse arc with 0 initial capacity and cost
                push!(dimacs_directed_edges, (v_idx, u_idx))
                push!(arc_capacities, 0)
                push!(arc_costs, 0)

                # Accumulate for source/sink arcs if connected to first/last column
                if c == 1 # Arc from (r,1) to (r+1,1) contributes to caps[r]
                    caps[r] += cap
                    # costs[r] is already set by horizontal arc or will be set by next horizontal arc
                end
                if c == w # Arc from (r,w) to (r+1,w) contributes to capt[r+1]
                    capt[r+1] += cap
                    costt[r+1] = rand(rng, 1:max_cost) # ggraph1.f assigns random cost here
                end
            end
        end
    end

    # --- Source Arcs (s_node to first column grid nodes) ---
    # Capacities and costs are derived from accumulated grid arc properties in ggraph1.f
    for r = 1:h
        grid_node = node_idx(r, 1)
        push!(dimacs_directed_edges, (s_node, grid_node))
        push!(arc_capacities, caps[r])
        push!(arc_costs, costs[r])
        # Add reverse arc with 0 initial capacity and cost
        push!(dimacs_directed_edges, (grid_node, s_node))
        push!(arc_capacities, 0)
        push!(arc_costs, 0)
    end

    # --- Sink Arcs (last column grid nodes to t_node) ---
    # Capacities and costs are derived from accumulated grid arc properties in ggraph1.f
    for r = 1:h
        grid_node = node_idx(r, w)
        push!(dimacs_directed_edges, (grid_node, t_node))
        push!(arc_capacities, capt[r])
        push!(arc_costs, costt[r])
        # Add reverse arc with 0 initial capacity and cost
        push!(dimacs_directed_edges, (t_node, grid_node))
        push!(arc_capacities, 0)
        push!(arc_costs, 0)
    end

    max_flow_value = _calculate_max_flow(
        s_node,
        t_node,
        total_nodes,
        dimacs_directed_edges,
        arc_capacities,
    )

    # --- Demands ---
    # Supply at s_node, Demand at t_node, 0 elsewhere.
    # Set supply/demand to the calculated max_flow_value
    node_demands = zeros(Int, total_nodes)
    node_demands[s_node] = -max_flow_value # Internal demand for source should be negative to become positive after Dimacs.jl inversion
    node_demands[t_node] = max_flow_value  # Internal demand for sink should be positive to become negative after Dimacs.jl inversion

    # Create Dimacs.Graph object
    E_matrix = Matrix{Int}(undef, length(dimacs_directed_edges), 2)
    for (i, (u, v)) in enumerate(dimacs_directed_edges)
        E_matrix[i, 1] = u
        E_matrix[i, 2] = v
    end
    G = Dimacs.FromEdgeList(total_nodes, E_matrix)

    return Dimacs.McfpNet(
        G = G,
        Demand = node_demands,
        Cap = arc_capacities,
        Cost = arc_costs,
    )
end

"""
    generate_grid_graph_mcfp_uniform(h::Int, w::Int)

Generates a directed grid graph for max-flow min-cost problems with uniform
capacities and costs (all set to 1).

Nodes consist of an `h` x `w` grid, plus a single source node `s` and a single sink node `t`.
Arcs are generated from `s` to the first column, from the last column to `t`,
and between adjacent grid nodes (rightwards and downwards).
Capacities, costs, and demands are assigned uniformly (value 1).

This function includes an Edmonds-Karp max-flow algorithm to calculate the
exact maximum flow, which is then used to set the supply at `s` and demand at `t`,
ensuring the generated problem is feasible.

# Arguments
- `h::Int`: The number of rows in the grid.
- `w::Int`: The number of columns in the grid.

# Returns
- A `Dimacs.McfpNet` object representing the generated grid graph MCFP instance.
"""
function generate_grid_graph_mcfp_uniform(
    h::Int,
    w::Int,
)
    num_grid_nodes = h * w
    s_node = num_grid_nodes + 1
    t_node = num_grid_nodes + 2
    total_nodes = num_grid_nodes + 2

    # Map 1-based (row, col) to a 1-based linear node index for grid nodes
    node_idx(r, c) = (r - 1) * w + c

    dimacs_directed_edges = Vector{Tuple{Int,Int}}()
    arc_capacities = Vector{Int}()
    arc_costs = Vector{Int}()

    # Arrays to store accumulated capacities and costs for source/sink arcs
    caps = zeros(Int, h) # For s -> (r,1) arcs
    costs = zeros(Int, h) # For s -> (r,1) arcs
    capt = zeros(Int, h) # For (r,w) -> t arcs
    costt = zeros(Int, h) # For (r,w) -> t arcs

    # --- Grid Arcs (Rightwards and Downwards) ---
    for r = 1:h
        for c = 1:w
            u_idx = node_idx(r, c)

            # Arc to the right: (r, c) -> (r, c+1)
            if c < w
                v_idx = node_idx(r, c + 1)
                cap = 1 # Fixed capacity
                cost = 1 # Fixed cost
                push!(dimacs_directed_edges, (u_idx, v_idx))
                push!(arc_capacities, cap)
                push!(arc_costs, cost)
                # Add reverse arc with 0 initial capacity and cost
                push!(dimacs_directed_edges, (v_idx, u_idx))
                push!(arc_capacities, 0)
                push!(arc_costs, 0)

                # Accumulate for source/sink arcs
                if c == 1
                    caps[r] += cap
                    costs[r] = 1 # Fixed cost
                end
                if c == w - 1
                    capt[r] += cap
                    costt[r] = 1 # Fixed cost
                end
            end

            # Arc downwards: (r, c) -> (r+1, c)
            if r < h
                v_idx = node_idx(r + 1, c)
                cap = 1 # Fixed capacity
                cost = 1 # Fixed cost
                push!(dimacs_directed_edges, (u_idx, v_idx))
                push!(arc_capacities, cap)
                push!(arc_costs, cost)
                # Add reverse arc with 0 initial capacity and cost
                push!(dimacs_directed_edges, (v_idx, u_idx))
                push!(arc_capacities, 0)
                push!(arc_costs, 0)

                # Accumulate for source/sink arcs
                if c == 1
                    caps[r] += cap
                end
                if c == w
                    capt[r+1] += cap
                    costt[r+1] = 1 # Fixed cost
                end
            end
        end
    end

    # --- Source Arcs (s_node to first column grid nodes) ---
    for r = 1:h
        grid_node = node_idx(r, 1)
        push!(dimacs_directed_edges, (s_node, grid_node))
        push!(arc_capacities, caps[r])
        push!(arc_costs, costs[r])
        # Add reverse arc with 0 initial capacity and cost
        push!(dimacs_directed_edges, (grid_node, s_node))
        push!(arc_capacities, 0)
        push!(arc_costs, 0)
    end

    # --- Sink Arcs (last column grid nodes to t_node) ---
    for r = 1:h
        grid_node = node_idx(r, w)
        push!(dimacs_directed_edges, (grid_node, t_node))
        push!(arc_capacities, capt[r])
        push!(arc_costs, costt[r])
        # Add reverse arc with 0 initial capacity and cost
        push!(dimacs_directed_edges, (t_node, grid_node))
        push!(arc_capacities, 0)
        push!(arc_costs, 0)
    end

    max_flow_value = _calculate_max_flow(
        s_node,
        t_node,
        total_nodes,
        dimacs_directed_edges,
        arc_capacities,
    )

    # --- Demands ---
    node_demands = zeros(Int, total_nodes)
    node_demands[s_node] = -max_flow_value
    node_demands[t_node] = max_flow_value

    # Create Dimacs.Graph object
    E_matrix = Matrix{Int}(undef, length(dimacs_directed_edges), 2)
    for (i, (u, v)) in enumerate(dimacs_directed_edges)
        E_matrix[i, 1] = u
        E_matrix[i, 2] = v
    end
    G = Dimacs.FromEdgeList(total_nodes, E_matrix)

    return Dimacs.McfpNet(
        G = G,
        Demand = node_demands,
        Cap = arc_capacities,
        Cost = arc_costs,
    )
end

end # module GridGraphs
