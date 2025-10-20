"""
    Dimacs

A module for parsing and writing DIMACS-formatted Minimum Cost Flow Problem (MCFP) files.

This module provides data structures to represent MCFP networks and functions to read
them from `.min` or `.min.gz` files, and to write them back to a gzipped `.min.gz` file.
"""
module Dimacs

using SparseArrays
using Scanf
using Printf
using GZip

# Problem description reader utility

"""
    Graph

A struct representing the topology of a directed graph.

# Fields
- `n::Int`: The number of nodes in the graph.
- `m::Int`: The number of edges in the graph.
- `EdgeList::Matrix{Int}`: An `m x 2` matrix where each row `[u, v]` represents a directed edge from node `u` to node `v`.
- `IncidenceMatrix::SparseMatrixCSC{Int8,Int}`: A sparse `n x m` node-arc incidence matrix.
- `AdjacencyMatrix::SparseMatrixCSC{Int,Int}`: A sparse `n x n` adjacency matrix.
"""
@kwdef struct Graph
    n::Int
    m::Int
    EdgeList::Matrix{Int}
    IncidenceMatrix::SparseMatrixCSC{Int8,Int}
    AdjacencyMatrix::SparseMatrixCSC{Int,Int}
end

"""
    McfpNet{Tv<:Number}

A struct representing a Minimum Cost Flow Problem (MCFP) network.

# Fields
- `G::Graph`: The underlying graph structure.
- `Cost::Vector{Tv}`: A vector of costs for each edge in `G.EdgeList`.
- `Cap::Vector{Tv}`: A vector of capacities for each edge in `G.EdgeList`.
- `Demand::Vector{Tv}`: A vector of demands for each node in `G`. Positive values are demands, negative values are supplies.
"""
@kwdef struct McfpNet{Tv<:Number}
    G::Graph
    Cost::Vector{Tv}
    Cap::Vector{Tv}
    Demand::Vector{Tv}
end

"""
    FromEdgeList(n::Int, E::Matrix{Int}) -> Graph

Constructs a `Graph` object from a given number of nodes and an edge list.

# Arguments
- `n::Int`: The total number of nodes.
- `E::Matrix{Int}`: An `m x 2` matrix representing the `m` directed edges.

# Returns
- A `Graph` object.
"""
function FromEdgeList(n::Int, E::Matrix{Int})
    local m = size(E, 1)
    @assert size(E) == (m, 2)
    Inc = MakeIncidenceMatrix(n, E)
    Adj = MakeAdjacencyMatrix(n, E, ones(m))
    return Graph(n = n, m = m, EdgeList = E, IncidenceMatrix = Inc, AdjacencyMatrix = Adj)
end

"""
    MakeIncidenceMatrix(n::Int, E::Matrix{Int}) -> SparseMatrixCSC{Int8,Int}

Creates a sparse node-arc incidence matrix from an edge list.

# Arguments
- `n::Int`: The number of nodes.
- `E::Matrix{Int}`: An `m x 2` edge list matrix.

# Returns
- An `n x m` sparse incidence matrix.
"""
function MakeIncidenceMatrix(n::Int, E::Matrix{Int})
    local m = size(E, 1)
    @assert size(E) == (m, 2)
    local I = vcat(E[1:m, 1], E[1:m, 2])
    local J = vcat(1:m, 1:m)
    local V = vcat(-ones(Int8, m), ones(Int8, m))
    local A = sparse(I, J, V, n, m)
    return A
end

"""
    MakeAdjacencyMatrix(n::Int, E::Matrix{Int}, w::Vector{Tv}) where {Tv<:Number} -> SparseMatrixCSC{Int,Int}

Creates a sparse adjacency matrix from an edge list and corresponding weights.

# Arguments
- `n::Int`: The number of nodes.
- `E::Matrix{Int}`: An `m x 2` edge list matrix.
- `w::Vector{Tv}`: A vector of weights for each edge.

# Returns
- An `n x n` sparse adjacency matrix.
"""
function MakeAdjacencyMatrix(n::Int, E::Matrix{Int}, w::Vector{Tv}) where {Tv<:Number}
    local m = size(E, 1)
    @assert size(E) == (m, 2)
    @assert size(w) == (m,)
    local I = E[1:m, 1]
    local J = E[1:m, 2]
    local V = w
    local Adj = sparse(I, J, V)
    return Adj
end

"""
    ReadDimacs(path::String) -> McfpNet

Reads a DIMACS-formatted MCFP file (`.min` or `.min.gz`) and constructs an `McfpNet` object.

The parser handles problem lines (`p`), node descriptor lines (`n`), and arc descriptor lines (`a`).
It assumes the convention where supplies are positive and demands are negative in the file,
but inverts this to match the internal convention (demand is positive).

# Arguments
- `path::String`: The path to the DIMACS file.

# Returns
- An `McfpNet` object representing the problem.
"""
function ReadDimacs(path::String)
    local n, m, E = nothing, nothing, nothing
    local C, U, B = nothing, nothing, nothing

    local nxtarc = 1
    local f = endswith(path, ".gz") ? GZip.open(path) : Base.open(path)
    local content = read(f, String)
    f = IOBuffer(content)
    while !eof(f)
        local r, c = @scanf(f, "%s", String)
        if c == "c"
            c = readline(f)
        elseif c == "p"
            r, dir, n, m = @scanf(f, "%s %d %d", String, Int, Int)
            E, C, U = zeros(Int, m, 2), zeros(Int, m), zeros(Int, m)
            B = zeros(Int, n)
        elseif c == "n"
            r, v, b = @scanf(f, "%d %d", Int, Int)
            # we adopted the opposite convention :(
            B[v] = -b
        elseif c == "a"
            r, i, j, l, u, c = @scanf(f, "%d %d %d %d %d", Int, Int, Int, Int, Int)
            E[nxtarc, :] = [i j]
            C[nxtarc], U[nxtarc] = c, u
            nxtarc += 1
            if nxtarc % 1000000 == 1
                @printf("%dM arcs read\n", fld(nxtarc, 1000000))
            end
        end
    end
    local netw = McfpNet(G = FromEdgeList(n, E), Cost = C, Cap = U, Demand = B)

    @assert sum(netw.Demand) == 0
    @assert !any(E[:, 1] .== E[:, 2])
    return netw
end

"""
    WriteDimacs(path::String, G::McfpNet)

Writes an `McfpNet` object to a gzipped DIMACS-formatted file.

# Arguments
- `path::String`: The path to the output `.min.gz` file.
- `G::McfpNet`: The MCFP network to write.
"""
function WriteDimacs(path::String, G::McfpNet)
    GZip.open(path, "w") do f
        @printf(f, "p min %d %d\n", G.G.n, G.G.m)
        for i = 1:G.G.n
            @printf(f, "n %d %d\n", i, -G.Demand[i])
        end
        for i = 1:G.G.m
            u, v = G.G.EdgeList[i, :]
            @printf(f, "a %d %d 0 %d %d\n", u, v, G.Cap[i], G.Cost[i])
        end
    end
end

end
