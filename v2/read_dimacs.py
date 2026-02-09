import gzip
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from dataclasses import dataclass
from typing import List, Union

@dataclass
class Graph:
    n: int
    m: int
    edge_list: np.ndarray  # m x 2 matrix
    incidence_matrix: csc_matrix  # n x m sparse matrix (int8)
    adjacency_matrix: csc_matrix  # n x n sparse matrix (int)

@dataclass
class McfpNet:
    G: Graph
    cost: np.ndarray  # vector of size m
    cap: np.ndarray   # vector of size m
    demand: np.ndarray  # vector of size n

def make_incidence_matrix(n: int, E: np.ndarray) -> csc_matrix:
    m = E.shape[0]
    row_indices = np.concatenate([E[:, 0], E[:, 1]])
    col_indices = np.concatenate([np.arange(m), np.arange(m)])
    values = np.concatenate([-np.ones(m, dtype=np.int8), np.ones(m, dtype=np.int8)])
    A = coo_matrix((values, (row_indices, col_indices)), shape=(n, m)).tocsc()
    return A

def make_adjacency_matrix(n: int, E: np.ndarray, w: np.ndarray) -> csc_matrix:
    m = E.shape[0]
    rows = E[:, 0]
    cols = E[:, 1]
    return coo_matrix((w, (rows, cols)), shape=(n, n)).tocsc()

def from_edge_list(n: int, E: np.ndarray) -> Graph:
    m = E.shape[0]
    inc = make_incidence_matrix(n, E)
    adj = make_adjacency_matrix(n, E, np.ones(m, dtype=int))
    return Graph(n, m, E, inc, adj)

def read_dimacs(path: str) -> McfpNet:
    open_func = gzip.open if path.endswith(".gz") else open
    n, m = 0, 0
    E, C, U, B = None, None, None, None
    nxtarc = 0
    with open_func(path, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'): continue
            parts = line.split()
            type_char = parts[0]
            if type_char == 'p':
                n, m = int(parts[2]), int(parts[3])
                E, C, U, B = np.zeros((m, 2), dtype=int), np.zeros(m), np.zeros(m), np.zeros(n)
            elif type_char == 'n':
                v, val = int(parts[1]) - 1, float(parts[2])
                B[v] = -val
            elif type_char == 'a':
                u, v, cap, cost = int(parts[1]) - 1, int(parts[2]) - 1, float(parts[4]), float(parts[5])
                E[nxtarc] = [u, v]
                U[nxtarc], C[nxtarc] = cap, cost
                nxtarc += 1
    graph = from_edge_list(n, E)
    net = McfpNet(graph, C, U, B)
    return net
