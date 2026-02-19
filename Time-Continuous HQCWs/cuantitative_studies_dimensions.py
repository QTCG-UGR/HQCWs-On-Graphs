import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import pandas as pd
import time

# -------------------------
# Utility functions
# -------------------------
def purity_score(y_true, y_pred):
    cont = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cont, axis=0)) / np.sum(cont)

def hungarian_accuracy(y_true, y_pred):
    cont = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cont)
    return cont[row_ind, col_ind].sum() / cont.sum()

# -------------------------
# One clustering evaluation run
# -------------------------
def eval_one_run(embedding, y_true, n_clusters=4, random_state=None, scale=True):
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(embedding)
    else:
        X = embedding.copy()

    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
    y_pred = kmeans.fit_predict(X)

    results = {}
    results['ARI'] = adjusted_rand_score(y_true, y_pred)
    results['NMI'] = normalized_mutual_info_score(y_true, y_pred)
    results['Vmeas'] = v_measure_score(y_true, y_pred)
    results['FMI'] = fowlkes_mallows_score(y_true, y_pred)
    results['Purity'] = purity_score(y_true, y_pred)
    results['HungarianAcc'] = hungarian_accuracy(y_true, y_pred)
    results['ConfMat'] = confusion_matrix(y_true, y_pred)
    results['Report'] = classification_report(y_true, y_pred, output_dict=True)
    return results

# -------------------------
# Multiple runs for stability
# -------------------------
def evaluate_embedding_multiple_runs(embedding, y_true, n_clusters=4, n_runs=30, seed=0):
    metrics = {'ARI':[], 'NMI':[], 'Vmeas':[], 'FMI':[], 'Purity':[], 'HungarianAcc':[]}
    confmats = []
    for r in range(n_runs):
        res = eval_one_run(embedding, y_true, n_clusters=n_clusters, random_state=seed+r)
        for k in metrics:
            metrics[k].append(res[k])
        confmats.append(res['ConfMat'])
    summary = {k: (np.mean(v), np.std(v)) for k,v in metrics.items()}
    return summary, confmats

# -------------------------
# k-NN supervised evaluation
# -------------------------
def knn_supervised_score(embedding, y_true, k=5, cv=5, scale=True):
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(embedding)
    else:
        X = embedding.copy()
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, X, y_true, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

# -------------------------
# Permutation test for ARI
# -------------------------
def permutation_test_ari(embedding, y_true, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    base_ari = eval_one_run(embedding, y_true)['ARI']
    perm_aris = []
    for _ in range(n_perm):
        y_shuf = rng.permuted(y_true)
        perm_aris.append(eval_one_run(embedding, y_shuf)['ARI'])
    perm_aris = np.array(perm_aris)
    p_value = (np.sum(perm_aris >= base_ari) + 1) / (n_perm + 1)
    return base_ari, p_value, perm_aris

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    # Load ground truth labels
    labels = np.loadtxt("node_colors.txt", dtype=int)
    y_true = labels[:, 1]

    # Embedding configurations
    embedding_sizes = [16, 32, 64, 128]
    # Names of the embedding files
    embedding_files = {
        "2nd_CRW": "cuant_2ndRW_emb_{}.txt",      
        "8_HQRW": "cuant_Qnode_emb_alpha_8_{}.txt"
        
    }

    summary_rows = []

    for emb_name, file_pattern in embedding_files.items():
        for size in embedding_sizes:
            filename = file_pattern.format(size)
            print(f"\n--- Evaluating {emb_name} (size {size}) ---")
            
            # Load embedding
            try:
                emb = np.loadtxt(filename)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                continue

            # Clustering evaluation
            print("- Clustering evaluation...")
            start_time = time.time()
            summ, _ = evaluate_embedding_multiple_runs(emb, y_true, n_clusters=4, n_runs=30)
            elapsed = time.time() - start_time
            print(f"Clustering evaluation took {elapsed:.3f} seconds")

            # k-NN supervised evaluation
            print("- k-NN supervised evaluation...")
            start_time = time.time()
            knn_mean, knn_std = knn_supervised_score(emb, y_true, k=5, cv=5)
            elapsed = time.time() - start_time
            print(f"k-NN evaluation took {elapsed:.3f} seconds")

            # ARI permutation test
            print("- Permutation test for ARI...")
            start_time = time.time()
            ari, pval, _ = permutation_test_ari(emb, y_true, n_perm=200)
            elapsed = time.time() - start_time
            print(f"Permutation test for ARI took {elapsed:.3f} seconds")

            # Collect results
            row = {"Embedding": f"{emb_name}_{size}"}
            for k, v in summ.items():
                row[k] = f"{v[0]:.3f} ± {v[1]:.3f}"
            row["kNN"] = f"{knn_mean:.3f} ± {knn_std:.3f}"
            row["ARI (obs)"] = f"{ari:.3f}"
            row["ARI p-val"] = f"{pval:.4f}"
            summary_rows.append(row)

    # Make results table
    df = pd.DataFrame(summary_rows)
    print("\n=== Summary of Results ===")
    print(df.to_string(index=False))
