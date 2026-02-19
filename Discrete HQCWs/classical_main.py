'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import csv
import os


# ---------------------------------------------------------------------------
# Parse CLI Arguments
# ---------------------------------------------------------------------------
def parse_args():
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='edges.txt',
                        help='Input graph path')

	parser.add_argument('--output', nargs='?', default='2ndRW_walks.txt',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=12,
	                    help='Length of walk per source.')

	parser.add_argument('--num-walks', type=int, default=2,
	                    help='Number of walks per source.')

	parser.add_argument('--window-size', type=int, default=5,
                    	help='Context size for optimization.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=4,
	                    help='Number of parallel workers.')

	parser.add_argument('--p', type=float, default=3.0,
	                    help='Return hyperparameter.')

	parser.add_argument('--q', type=float, default=0.5,
	                    help='Inout hyperparameter.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Graph is weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	args = parser.parse_args()

	return args

# ---------------------------------------------------------------------------
# Read the Graph
# ---------------------------------------------------------------------------
import csv
import networkx as nx

def load_graph_from_csv(nodes_file, edges_file):
    G = nx.Graph()

    # Load nodes: one column, header "id"
    with open(nodes_file, "r", newline="") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if line:
                G.add_node(int(line))

    # Load edges: tab separated, header "subject object"
    with open(edges_file, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            u = row[0].strip()
            v = row[1].strip()
            if not u.isdigit() or not v.isdigit():
                continue
            G.add_edge(int(u), int(v), weight=1.0)

    return G


# ---------------------------------------------------------------------------
def write_walks(walks, path):
    with open(path, 'w', encoding='utf-8') as f:
        for walk in walks:
            f.write(" ".join(map(str, walk)) + "\n")
    print(f"✅ {len(walks)} walks written to {path}")

# ---------------------------------------------------------------------------


def classical_simulations(topology, aprox_size, hold_id, nodes_file, edges_file, num_traj, desired_jumps, p, q):

    args = parse_args()
    nx_G = load_graph_from_csv(f"generated_graphs/{nodes_file}", f"holdouts_trial/{edges_file}")

    args.num_walks = num_traj
    args.walk_length = desired_jumps

    G = node2vec.Graph(nx_G, False, p, q)
    G.preprocess_transition_probs()

    walks = G.simulate_walks(args.num_walks, args.walk_length)

    trajectories_dir = "trajectories"
    os.makedirs(trajectories_dir, exist_ok=True)

    if (p == 1.0) and (q == 1.0):
        filename = f"trajs_{topology}_n{aprox_size}_holdout_{hold_id}_C1.txt"
    else:
        filename = f"trajs_{topology}_n{aprox_size}_holdout_{hold_id}_C2.txt"

    full_path = os.path.join(trajectories_dir, filename)

    write_walks(walks, full_path)

