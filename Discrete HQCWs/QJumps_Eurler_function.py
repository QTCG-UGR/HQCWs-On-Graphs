# -*- coding: utf-8 -*-
import numpy as np
import time
from gensim.models import Word2Vec
import os


# ---------------------------------------------------------------------------
# Read Config Parameters
# ---------------------------------------------------------------------------
def read_config(path="config.txt"):
    params = {}
    with open(path, 'r') as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=")
                params[key.strip()] = float(val.strip())
    return int(params["num_traj"]), int(params["desired_jumps"]), int(params["window"]), int(params["embedding_dimension"])


# ---------------------------------------------------------------------------
# Load Graph from edges.csv and nodes.csv
# ---------------------------------------------------------------------------
def load_graph_from_csv(nodes_path, edges_path, edges_delimiter="\t"):
    import csv
    import numpy as np
    import networkx as nx

    nodes_set = set()

    with open(nodes_path, "r", newline="") as f:
        reader = csv.reader(f)  
        next(reader, None)     
        for row in reader:
            if not row:
                continue
            node = row[0].strip()
            if node == "":
                continue
            if node.isdigit():
                nodes_set.add(int(node))
            else:
                nodes_set.add(node)

    edges = []
    with open(edges_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=edges_delimiter)
        next(reader, None) 
        for row in reader:
            if len(row) < 2:
                continue
            u, v = row[0].strip(), row[1].strip()
            if u == "" or v == "":
                continue

            if u.isdigit() and v.isdigit():
                u, v = int(u), int(v)
            edges.append((u, v))
            nodes_set.add(u)
            nodes_set.add(v)

    if nodes_set and all(isinstance(x, int) for x in nodes_set):
        node_list = sorted(nodes_set)
    else:
        node_list = sorted(nodes_set, key=str)

    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    n = len(node_list)

    adj_matrix = np.zeros((n, n), dtype=float)
    for u, v in edges:
        i, j = node_to_idx[u], node_to_idx[v]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1

    G = nx.from_numpy_array(adj_matrix)
    return G, adj_matrix, node_to_idx


# ---------------------------------------------------------------------------
# Prepare Hamiltonian H
# ---------------------------------------------------------------------------
def prepare_quantum_operators(adj_matrix):
    """
    From adjacency matrix (adj), produce:
      - H: Hamiltonian 
      - collapse_indices: list of (i,j) with adj[i,j]>0
      - rates[(i,j)] = sqrt(P[i,j]) with P = row-normalized adj
    """
    num_nodes = adj_matrix.shape[0]
    H = adj_matrix.astype(np.complex128)

    out_deg = adj_matrix.sum(axis=1)
    P = np.divide(adj_matrix, out_deg[:, None], where=out_deg[:, None] != 0)

    collapse_indices = [
        (i, j)
        for i in range(num_nodes)
        for j in range(num_nodes)
        if adj_matrix[i, j] > 0
    ]
    rates = {(i, j): np.sqrt(P[i, j]) for (i, j) in collapse_indices}

    return H, collapse_indices, rates

# ---------------------------------------------------------------------------
# Euler Quantum Jump Walks
# ---------------------------------------------------------------------------
def quantum_trajectories_euler(
        H, collapse_indices, rates,
        num_nodes, gamma,
        num_traj, jumps_per_traj, start_node):
    
    Coherent_steps_before_jump = max(0, gamma)

    dt = 1

    V = np.eye(num_nodes, dtype=np.complex128) - 1j * H * dt

    all_trajs = []
    for _ in range(num_traj):
        psi = np.zeros(num_nodes, dtype=np.complex128)
        psi[start_node] = 1
        jumps = []
        jumps.append(start_node)

        for _ in range(jumps_per_traj):
            # Coherent Euler block
            for _ in range(Coherent_steps_before_jump):
                psi = V.dot(psi)
                psi /= np.linalg.norm(psi)

            # Compute jump probabilities
            probs = np.array([
                (abs(psi[i]) ** 2) * (rates[(i, j)] ** 2)
                for (i, j) in collapse_indices
            ])
            total_p = probs.sum()
            if total_p <= 0:
                break
            probs /= total_p

            _, j = collapse_indices[np.random.choice(len(collapse_indices), p=probs)]

            psi = np.zeros_like(psi)
            psi[j] = 1
            jumps.append(j)

        #print(jumps)
        all_trajs.append([str(j) for j in jumps])
        
    return all_trajs

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def QEuler_simulation(topology, aprox_size, hold_id, nodes_file, edges_file, num_traj=3, jumps_per_traj=10, gamma_array=[1,2]):

    print(f"\n=== Running simulation on {nodes_file} / {edges_file} ===")

    print(nodes_file) 

    # Load graph
    G, adj_matrix, node_to_idx = load_graph_from_csv(
    f"generated_graphs/{nodes_file}",
    f"holdouts_trial/{edges_file}"
    )

    node_list = sorted([int(n) for n in node_to_idx.keys()])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(base_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    trajectories_dir = os.path.join(base_dir, "trajectories")
    os.makedirs(trajectories_dir, exist_ok=True)

    print("Adjacency sum:", np.sum(adj_matrix))
    if np.sum(adj_matrix) == 0:
        raise ValueError("Graph has no edges. Check edges.csv formatting.")

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Prepare operators
    H, collapse_indices, rates = prepare_quantum_operators(adj_matrix)

    graph_size = adj_matrix.shape[0]

    # Loop over gamma values
    for gamma in gamma_array:
        start_time = time.time()
        print(f"\n--- gamma = {gamma}, simulating trajectories...")

        all_sentences = []
        for start_node in node_list:
            trajs = quantum_trajectories_euler(
                H,
                collapse_indices,
                rates,
                num_nodes=graph_size,
                gamma=gamma,
                num_traj=num_traj,
                jumps_per_traj=jumps_per_traj,
                start_node=start_node,
            )
            all_sentences.extend(trajs)

        elapsed = time.time() - start_time
        print(f"Done in {elapsed:.2f}s")

        traj_file = os.path.join(trajectories_dir, f"trajs_{topology}_n{aprox_size}_holdout_{hold_id}_Q_g={gamma}.txt")
        with open(traj_file, "w") as f:
            for traj in all_sentences:
                f.write(" ".join(traj) + "\n")

def train_embeddings_from_trajectories(topology, hold_id, graph_size, method, gamma, emb_dimension=80, window=5):
    """
    Train Word2Vec embeddings from precomputed trajectories.
    """
    if method == "Q":
        traj_file = os.path.join("trajectories", f"trajs_{topology}_n{graph_size}_holdout_{hold_id}_{method}_g={gamma}.txt")
        embedding_filename = f"emb_{topology}_n{graph_size}_holdout_{hold_id}_{method}_g={gamma}_dim={emb_dimension}.txt"
    elif method == "C1":
        traj_file = os.path.join("trajectories", f"trajs_{topology}_n{graph_size}_holdout_{hold_id}_{method}.txt")
        embedding_filename = f"emb_{topology}_n{graph_size}_holdout_{hold_id}_{method}_dim={emb_dimension}.txt"
    elif method == "C2":
        traj_file = os.path.join("trajectories", f"trajs_{topology}_n{graph_size}_holdout_{hold_id}_{method}.txt")
        embedding_filename = f"emb_{topology}_n{graph_size}_holdout_{hold_id}_{method}_dim={emb_dimension}.txt"

    all_sentences = []
    with open(traj_file, "r") as f:
        for line in f:
            traj = line.strip().split()
            all_sentences.append(traj)

    model = Word2Vec(
        sentences=all_sentences,
        vector_size=emb_dimension,
        window=window,
        sg=1,
        negative=10,
        min_count=1,
        workers=4,
        seed=42,
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(base_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    print(f"Saving embeddings to {embedding_filename}...")

    # Build output filename
    
    embedding_path = os.path.join(embeddings_dir, embedding_filename)

    with open(embedding_path, "w") as f:
        for node in model.wv.index_to_key:
            embedding = model.wv[node]
            emb_str = " ".join(map(str, embedding))
            f.write(f"{node} {emb_str}\n")