module Dimacs

using SparseArrays
using Scanf
using Printf

# Problem description reader utility

@kwdef struct Graph
  n::Int
  m::Int
  EdgeList::Matrix{Int}
  IncidenceMatrix::SparseMatrixCSC{Int8,Int}
  AdjacencyMatrix::SparseMatrixCSC{Int8,Int}
end

@kwdef struct McfpNet{Tv<:Number}
  G::Graph
  Cost::Vector{Tv}
  Cap::Vector{Tv}
  Demand::Vector{Tv}
end

function FromEdgeList(n::Int, E::Matrix{Int})
  local m = size(E, 1)
  @assert size(E) == (m, 2)
  return Graph(
    n = n,
    m = m,
    EdgeList = E,
    IncidenceMatrix = MakeIncidenceMatrix(n, E),
    AdjacencyMatrix = MakeAdjacencyMatrix(n, E, ones(m)),
  )
end

function MakeIncidenceMatrix(n::Int, E::Matrix{Int})
  local m = size(E, 1)
  @assert size(E) == (m, 2)
  local I = vcat(E[1:m, 1], E[1:m, 2])
  local J = vcat(1:m, 1:m)
  local V = vcat(-ones(Int8, m), ones(Int8, m))
  local A = sparse(I, J, V, n, m)
  return A
end

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

function ReadDimacs(path::String)
  local n, m, E = nothing, nothing, nothing
  local C, U, B = nothing, nothing, nothing

  local nxtarc = 1
  local f = open(path)
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

end
