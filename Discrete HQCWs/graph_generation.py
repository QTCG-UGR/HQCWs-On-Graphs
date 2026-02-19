import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math

import graph_utils as gutils

# Grape
from grape import Graph
from grape import GraphVisualizer

sizes_of_graphs = [100, 200, 500, 1000, 3000]

for size in sizes_of_graphs:
    p = np.log(size) / size    
    edges, nodes = gutils.generate_erdos_renyi_graph(
        n=size,
        p=p,
        num_node_types=1
    )
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    nodes_df = pd.DataFrame({"id": list(range(0, size))})
    edges_df.to_csv(f"generated_graphs/ER_n{size}_edges.csv", index=False)
    nodes_df.to_csv(f"generated_graphs/ER_n{size}_nodes.csv", index=False)
    

for size in sizes_of_graphs:
    PL_edges, PL_nodes = gutils.generate_power_law_nx(n=size, alpha=2.1, k_min=5, k_max=800, num_node_types=1)

    edges_df = pd.DataFrame(PL_edges, columns=["source", "target"])
    nodes_df = pd.DataFrame({"id": list(range(0, size))})
    edges_df.to_csv(f"generated_graphs/PL_n{size}_edges.csv", index=False)
    nodes_df.to_csv(f"generated_graphs/PL_n{size}_nodes.csv", index=False)

for size in sizes_of_graphs:
    ba_edges, ba_nodes = gutils.generate_babarasi_albert(n=size, m=5, num_node_types=1)

    edges_df = pd.DataFrame(ba_edges, columns=["source", "target"])
    nodes_df = pd.DataFrame({"id": list(range(0, size))})
    edges_df.to_csv(f"generated_graphs/BA_n{size}_edges.csv", index=False)
    nodes_df.to_csv(f"generated_graphs/BA_n{size}_nodes.csv", index=False)


for size in sizes_of_graphs:
    rows = int(math.floor(math.sqrt(size)))
    cols = int(math.ceil(size / rows))
    grid_edges, grid_nodes = gutils.generate_grid(rows=rows, columns=cols)
    print(size - rows*cols)

    edges_df = pd.DataFrame(grid_edges, columns=["source", "target"])
    nodes_df = pd.DataFrame({"id": list(range(0, rows*cols))})
    edges_df.to_csv(f"generated_graphs/grid_n{size}_edges.csv", index=False)
    nodes_df.to_csv(f"generated_graphs/grid_n{size}_nodes.csv", index=False)


'''for size in sizes_of_graphs:
    cycle_edges, cycle_nodes = gutils.generate_cycle(n=size, n_types=1)

    # Save the edges and nodes as CSV files on a folder named "generated_graphs"
    edges_df = pd.DataFrame(cycle_edges, columns=["source", "target"])
    nodes_df = pd.DataFrame({"id": list(range(0, size))})
    edges_df.to_csv(f"generated_graphs/cycle_n{size}_edges.csv", index=False)
    nodes_df.to_csv(f"generated_graphs/cycle_n{size}_nodes.csv", index=False)'''


for size in sizes_of_graphs:
    n1 = int(math.floor(math.sqrt(size)))
    n2 = int(math.ceil(size / n1))
    bipartite_edges, bipartite_nodes = gutils.generate_complete_bipartite(n1=n1, n2=n2)
    print(size - n1*n2)

    edges_df = pd.DataFrame(bipartite_edges, columns=["source", "target"])
    nodes_df = pd.DataFrame({"id": list(range(0, size))})
    edges_df.to_csv(f"generated_graphs/bipartite_n{size}_edges.csv", index=False)
    nodes_df.to_csv(f"generated_graphs/bipartite_n{size}_nodes.csv", index=False)


'''for size in sizes_of_graphs:
    Kn_edges, Kn_nodes = gutils.generate_complete_graph(n=size, k=1)

    # Save the edges and nodes as CSV files on a folder named "generated_graphs"
    edges_df = pd.DataFrame(Kn_edges, columns=["source", "target"])
    nodes_df = pd.DataFrame({"id": list(range(0, size))})
    edges_df.to_csv(f"generated_graphs/Kn_n{size}_edges.csv", index=False)
    nodes_df.to_csv(f"generated_graphs/Kn_n{size}_nodes.csv", index=False)'''


for size in sizes_of_graphs:

    comm_edges, comm_nodes = gutils.generate_community_graph(
        k=5,
        min_size=size//5,
        max_size=size//5,
        intra_prob=0.7,
        inter_prob=0.05,
        seed=42
    )


    edges_df = pd.DataFrame(comm_edges, columns=["source", "target"])

    nodes_df = pd.DataFrame({"id": list(range(len(comm_nodes)))})

    edges_df.to_csv(f"generated_graphs/comm_n{size}_edges.csv", index=False)
    nodes_df.to_csv(f"generated_graphs/comm_n{size}_nodes.csv", index=False)


'''for size in sizes_of_graphs:

    def best_tree_depth(size, degree, max_depth=15):
        """
        For a given desired size, return the depth that generates
        a regular tree whose node count is closest to the target.
        """
        best_d = None
        best_N = None
        best_diff = float("inf")

        for d in range(max_depth + 1):
            N = (degree**(d+1) - 1) // (degree - 1)
            diff = abs(N - size)

            if diff < best_diff:
                best_diff = diff
                best_d = d
                best_N = N

        return best_d, best_N

    degree = 2

    depth, actual_nodes = best_tree_depth(size, degree)

    print(f"Requested size: {size}, using depth {depth} → {actual_nodes} nodes")

    tree_edges, tree_nodes = gutils.generate_regular_tree(tree_depth=depth, degree=degree)

    # Save the edges and nodes as CSV files on a folder named "generated_graphs"
    edges_df = pd.DataFrame(tree_edges, columns=["source", "target"])
    nodes_df = pd.DataFrame({"id": list(range(0, actual_nodes))})
    edges_df.to_csv(f"generated_graphs/tree_n{size}_edges.csv", index=False)
    nodes_df.to_csv(f"generated_graphs/tree_n{size}_nodes.csv", index=False)'''