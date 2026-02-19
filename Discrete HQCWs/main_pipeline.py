import os
import numpy as np
import pandas as pd
import sklearn
from sklearn import tree

import grape
from helper_lib import predict


from holdout_generation import generate_holdouts
from QJumps_Eurler_function import QEuler_simulation, train_embeddings_from_trajectories
from classical_main import classical_simulations


# ---------------------------------------------------
# OTHER FUNCTIONS
# ---------------------------------------------------

def load_node_embeddings_from_txt(txt_path, embedding_name):

    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            node = parts[0]
            vec = [float(x) for x in parts[1:]]
            rows.append((node, vec))

    nodes = [r[0] for r in rows]
    mat = np.array([r[1] for r in rows])

    df = pd.DataFrame(mat, index=nodes)
    df.index.name = "node"

    embedding = grape.EmbeddingResult(
        embedding_method_name=embedding_name,
        node_embeddings=[df],
    )

    return embedding


def embedding_txt_path(embeddings_dir, topology, size, hold_id, method, emb_dim, gamma=None):

    if method == "Q":
        return os.path.join(
            embeddings_dir,
            f"emb_{topology}_n{size}_holdout_{hold_id}_Q_g={gamma}_dim={emb_dim}.txt"
        )

    if method == "C1" or method == "C2":
        return os.path.join(
            embeddings_dir,
            f"emb_{topology}_n{size}_holdout_{hold_id}_{method}_dim={emb_dim}.txt"
        )

    raise ValueError("Unknown method")


def export_examples_to_parquet(examples, output_path, filename, parquet_compression="brotli"):

    os.makedirs(output_path, exist_ok=True)
    df = pd.DataFrame(examples)
    out_file = os.path.join(output_path, filename)
    df.to_parquet(out_file, compression=parquet_compression)

    return out_file


def build_edge_embedding_parquets_for_holdout(
    holdout,
    node_embeddings,
    out_dir,
    holdout_id,
    node_embeddings_concatenation_method="Concatenate",
):

    keys = [
        "positive_train_graph",
        "negative_train_graph",
        "positive_test_graph",
        "negative_test_graph",
    ]

    for key in keys:

        edge_embeddings = predict.graph_to_edge_embeddings(
            embedding=node_embeddings,
            graph=holdout[key],
            node_embeddings_concatenation_method=node_embeddings_concatenation_method,
            treat_edges_as_bidirectional=True,
        )

        export_examples_to_parquet(
            examples=edge_embeddings,
            output_path=out_dir,
            filename=f"{key}_edge_embeddings_{holdout_id}.parquet",
        )


def run_edge_prediction_auc_from_parquets(holdout_dir, holdout_id, random_state=42):

    model = tree.DecisionTreeClassifier(
        random_state=random_state,
        max_depth=100,
        max_features="sqrt",
    )

    pos_train = pd.read_parquet(
        os.path.join(holdout_dir, f"positive_train_graph_edge_embeddings_{holdout_id}.parquet")
    )
    neg_train = pd.read_parquet(
        os.path.join(holdout_dir, f"negative_train_graph_edge_embeddings_{holdout_id}.parquet")
    )

    X_train = np.concatenate([pos_train.values, neg_train.values])
    y_train = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
    X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=random_state)

    model.fit(X_train, y_train)

    train_auc = sklearn.metrics.roc_auc_score(
        y_train, model.predict_proba(X_train)[:, 1]
    )

    pos_test = pd.read_parquet(
        os.path.join(holdout_dir, f"positive_test_graph_edge_embeddings_{holdout_id}.parquet")
    )
    neg_test = pd.read_parquet(
        os.path.join(holdout_dir, f"negative_test_graph_edge_embeddings_{holdout_id}.parquet")
    )

    X_test = np.concatenate([pos_test.values, neg_test.values])
    y_test = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))])

    test_auc = sklearn.metrics.roc_auc_score(
        y_test, model.predict_proba(X_test)[:, 1]
    )

    return train_auc, test_auc


def edge_prediction_each_graph(topology, graph_size, gamma_array, emb_dimensions, number_of_holdouts,
                               num_traj, jumps_per_traj, holdouts_map):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(base_dir, "embeddings")
    edge_features_root = os.path.join(base_dir, "edge_features")

    results_rows = []

    holdouts = holdouts_map[(topology, graph_size)]

    for hold_id in range(number_of_holdouts):
        holdout = holdouts[hold_id]

        for emb_dim in emb_dimensions:

            # HQCW
            for gamma in gamma_array:

                emb_path = embedding_txt_path(
                    embeddings_dir, topology, graph_size, hold_id,
                    method="Q", emb_dim=emb_dim, gamma=gamma
                )

                node_emb = load_node_embeddings_from_txt(
                    emb_path, f"HQCW_g{gamma}_d{emb_dim}"
                )

                run_dir = os.path.join(
                    edge_features_root,
                    f"{topology}_n{graph_size}_h{hold_id}_HQCW_g{gamma}_d{emb_dim}"
                )

                build_edge_embedding_parquets_for_holdout(
                    holdout, node_emb, run_dir, holdout_id=hold_id
                )

                train_auc, test_auc = run_edge_prediction_auc_from_parquets(
                    run_dir, holdout_id=hold_id
                )

                results_rows.append({
                    "topology": topology,
                    "n_nodes": graph_size,
                    "method": "HQCW",
                    "method_parameter": gamma,
                    "num_traj": num_traj,
                    "jumps_per_traj": jumps_per_traj,
                    "emb_dimension": emb_dim,
                    "holdout_id": hold_id,
                    "training_result": train_auc,
                    "testing_result": test_auc,
                })

            # CRW
            emb_path = embedding_txt_path(
                embeddings_dir, topology, graph_size, hold_id,
                method="C1", emb_dim=emb_dim, gamma=None
            )

            node_emb = load_node_embeddings_from_txt(
                emb_path, f"CRW_d{emb_dim}"
            )

            run_dir = os.path.join(
                edge_features_root,
                f"{topology}_n{graph_size}_h{hold_id}_CRW_d{emb_dim}"
            )

            build_edge_embedding_parquets_for_holdout(
                holdout, node_emb, run_dir, holdout_id=hold_id
            )

            train_auc, test_auc = run_edge_prediction_auc_from_parquets(
                run_dir, holdout_id=hold_id
            )

            results_rows.append({
                "topology": topology,
                "n_nodes": graph_size,
                "method": "CRW",
                "method_parameter": None,
                "num_traj": num_traj,
                "jumps_per_traj": jumps_per_traj,
                "emb_dimension": emb_dim,
                "holdout_id": hold_id,
                "training_result": train_auc,
                "testing_result": test_auc,
            })

            # 2O CRW
            emb_path = embedding_txt_path(
                embeddings_dir, topology, graph_size, hold_id,
                method="C2", emb_dim=emb_dim, gamma=None
            )

            node_emb = load_node_embeddings_from_txt(
                emb_path, f"2O-CRW_d{emb_dim}"
            )

            run_dir = os.path.join(
                edge_features_root,
                f"{topology}_n{graph_size}_h{hold_id}_2O-CRW_d{emb_dim}"
            )

            build_edge_embedding_parquets_for_holdout(
                holdout, node_emb, run_dir, holdout_id=hold_id
            )

            train_auc, test_auc = run_edge_prediction_auc_from_parquets(
                run_dir, holdout_id=hold_id
            )

            results_rows.append({
                "topology": topology,
                "n_nodes": graph_size,
                "method": "2O-CRW",
                "method_parameter": None,
                "num_traj": num_traj,
                "jumps_per_traj": jumps_per_traj,
                "emb_dimension": emb_dim,
                "holdout_id": hold_id,
                "training_result": train_auc,
                "testing_result": test_auc,
            })

    return pd.DataFrame(results_rows)


if __name__ == "__main__":

    topologies = ["comm"] #, "PL", "ER", "BA", "PL", "ER", "BA", "grid"
    methods = ["HQCW", "CRW", "2O-CRW"]
    graph_sizes = [100, 200, 500, 1000]

    num_traj = 3
    jumps_per_traj = 10
    gamma_array = [1, 2, 3, 4]
    emb_dimensions = [16, 32, 64, 128]
    number_of_holdouts = 5

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ---------------------------------------------------
    # 1) HOLDOUTS
    # ---------------------------------------------------
    holdouts_map = {}

    for topology in topologies:
        for size in graph_sizes:
            holdouts = generate_holdouts(topology, size, number_of_holdouts)
            holdouts_map[(topology, size)] = holdouts

    # ---------------------------------------------------
    # 2) TRAJECTORIES
    # ---------------------------------------------------
    for topology in topologies:
        for size in graph_sizes:
            for hold_id in range(number_of_holdouts):

                nodes_file = f"{topology}_n{size}_nodes.csv"
                edges_file = f"{topology}_n{size}_holdout_{hold_id}.csv"

                QEuler_simulation(
                    topology, size, hold_id,
                    nodes_file, edges_file,
                    num_traj=num_traj,
                    jumps_per_traj=jumps_per_traj,
                    gamma_array=gamma_array
                )

                classical_simulations(
                    topology, size, hold_id,
                    nodes_file, edges_file,
                    num_traj=num_traj,
                    desired_jumps=jumps_per_traj,
                    p=1.0, q=1.0
                )

                classical_simulations(
                    topology, size, hold_id,
                    nodes_file, edges_file,
                    num_traj=num_traj,
                    desired_jumps=jumps_per_traj,
                    p=1.5, q=0.5
                )

    # ---------------------------------------------------
    # 3) NODE EMBEDDINGS
    # ---------------------------------------------------
    for topology in topologies:
        for size in graph_sizes:
            for hold_id in range(number_of_holdouts):
                for emb_dimension in emb_dimensions:

                    for gamma in gamma_array:
                        train_embeddings_from_trajectories(
                            topology=topology,
                            hold_id=hold_id,
                            graph_size=size,
                            method="Q",
                            gamma=gamma,
                            emb_dimension=emb_dimension
                        )

                    train_embeddings_from_trajectories(
                        topology=topology,
                        hold_id=hold_id,
                        graph_size=size,
                        method="C1",
                        gamma=None,
                        emb_dimension=emb_dimension
                    )

                    train_embeddings_from_trajectories(
                        topology=topology,
                        hold_id=hold_id,
                        graph_size=size,
                        method="C2",
                        gamma=None,
                        emb_dimension=emb_dimension
                    )

    # ---------------------------------------------------
    # 4) EDGE PREDICTION AND FINAL RESULTS DATAFRAME
    # ---------------------------------------------------
    all_results = []

    for topology in topologies:
        for size in graph_sizes:

            df_one = edge_prediction_each_graph(
                topology=topology,
                graph_size=size,
                gamma_array=gamma_array,
                emb_dimensions=emb_dimensions,
                number_of_holdouts=number_of_holdouts,
                num_traj=num_traj,
                jumps_per_traj=jumps_per_traj,
                holdouts_map=holdouts_map
            )

            all_results.append(df_one)

    results_df = pd.concat(all_results, ignore_index=True)

    out_csv = os.path.join(base_dir, "edge_prediction_results.csv")

    if os.path.exists(out_csv):
        old_df = pd.read_csv(out_csv)
        combined = pd.concat([old_df, results_df], ignore_index=True)

        key_cols = [
            "topology",
            "n_nodes",
            "method",
            "method_parameter",
            "emb_dimension",
            "holdout_id",
            "num_traj",
            "jumps_per_traj",
        ]

        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        results_df = combined

    # Save CSV
    results_df.to_csv(out_csv, index=False)
    print("Saved CSV file:", out_csv)

    # Save Excel
    excel_path = os.path.join(base_dir, "edge_prediction_results.xlsx")
    results_df.to_excel(excel_path, index=False)
    print("Saved Excel file:", excel_path)

