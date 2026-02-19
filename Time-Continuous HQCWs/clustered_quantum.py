# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:34:35 2025

Quantum Random Walk Embedding with Word2Vec (Optimized)
"""

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from qutip import mcsolve, Qobj, basis
import pandas as pd
import time

# ---------------------------------------------------------------------------
# Helper Function: Create Hamiltonian and Collapse Operators
# ---------------------------------------------------------------------------
def prepare_quantum_operators(adj_matrix, alpha):
    num_nodes = adj_matrix.shape[0]
    hamiltonian = Qobj(adj_matrix)

    out_degrees = adj_matrix.sum(axis=1)
    transition_matrix = adj_matrix / out_degrees[:, None]

    collapse_operators = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                rate = np.sqrt(transition_matrix[i, j])
                L_ij = rate * basis(num_nodes, j) * basis(num_nodes, i).dag()
                collapse_operators.append(L_ij)

    return hamiltonian, collapse_operators

# ---------------------------------------------------------------------------
# Quantum Walk for One Node
# ---------------------------------------------------------------------------
def quantum_jumps_single_node(hamiltonian, collapse_operators, num_nodes, evolution_time, num_traj, alpha, node):
    def find_state_after_jump(matrix):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != 0:
                    return j

    options = {
        'store_states': False,
        'progress_bar': False
    }
    list_states_after_jumps = []

    tlist = np.linspace(0, evolution_time, int(evolution_time * 10))

    for _ in range(num_traj):
        resultmc = mcsolve(
            H=(1 - alpha) * hamiltonian,
            state=basis(num_nodes, node),
            tlist=tlist,
            c_ops=[np.sqrt(alpha) * L for L in collapse_operators],
            ntraj=1,
            options=options,
        )

        states_after_jumps = []
        for coll in resultmc.col_which[0]:
            states_after_jumps.append(find_state_after_jump(collapse_operators[coll]))

        list_states_after_jumps.append(states_after_jumps)

    return list_states_after_jumps

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

# Load the adjacency matrix from the file
adj_matrix = np.loadtxt("adjacency_matrix.txt", dtype=int)
G = nx.from_numpy_array(adj_matrix)

print(f"Number of nodes in the graph: {G.number_of_nodes()}")

edges = nx.to_pandas_edgelist(G).rename(columns={'source': 'subject', 'target': 'object'})
nodes = pd.DataFrame({"name": list(G.nodes)})
nodes["type"] = 0  # Optional dummy column

# Parameters
alpha = 0.7
total_time = 14.5
aprox_n_jumps = alpha * total_time
num_traj = 10
num_nodes = adj_matrix.shape[0]

for i in (3):
    
    start_time = time.time()
    
    alpha = 0.1 * i
    print(f"Alpha = {alpha:.1f}")
    print(f"Evolution time: {aprox_n_jumps / alpha:.2f} seconds")
        
    # Precompute Hamiltonian and collapse operators
    hamiltonian, collapse_operators = prepare_quantum_operators(adj_matrix, alpha)
    
    # Generate quantum walks
    Qwalks = []
    for node in G.nodes():
        print(f"[Node {node}]")
        
        jumps = quantum_jumps_single_node(hamiltonian, collapse_operators, num_nodes, aprox_n_jumps / alpha, num_traj, alpha, node)
        if jumps:
            Qwalks.extend([[str(n) for n in walk] for walk in jumps])
            
    total_jumps = sum(len(walk) for walk in Qwalks)
    total_walks = len(Qwalks)
    ave_jumps_per_walk = total_jumps / total_walks if total_walks > 0 else 0
    print(f"\nAverage jumps per walk: {ave_jumps_per_walk:.2f}")
        
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")

