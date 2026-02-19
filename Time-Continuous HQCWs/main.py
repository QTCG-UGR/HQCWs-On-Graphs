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

# ---------------------------------------------------------------------------
# Read Parameters from Config File
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
# Parse CLI Arguments (some will be overwritten by config.txt)
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

	# Overwrite values from config.txt
	num_traj, desired_jumps, window, emb_dims = read_config()
	args.num_walks = num_traj
	args.walk_length = desired_jumps
	args.window_size = window
	args.dimensions = emb_dims

	print("📥 Loaded parameters from config.txt:")
	print(f" - num_walks = {args.num_walks}")
	print(f" - walk_length = {args.walk_length}")
	print(f" - window_size = {args.window_size}")
	print(f" - Embedding dimensions = {args.dimensions}\n")

	return args

# ---------------------------------------------------------------------------
# Read the Graph
# ---------------------------------------------------------------------------
def read_graph():
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

# ---------------------------------------------------------------------------
# Learn Embeddings
# ---------------------------------------------------------------------------
def write_walks(walks, path):
    with open(path, 'w', encoding='utf-8') as f:
        for walk in walks:
            # each walk is a list of ints; convert to str
            f.write(" ".join(map(str, walk)) + "\n")
    print(f"✅ {len(walks)} walks written to {path}")

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main(args):
	nx_G = read_graph()
	G = node2vec.Graph(nx_G, False, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	args.output = f"2ndRW_walks_{args.num_walks}_wl_{args.walk_length}.txt"
	write_walks(walks, args.output)

if __name__ == "__main__":
	args = parse_args()
	main(args)
