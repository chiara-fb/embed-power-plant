import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, jaccard_score
from sklearn.cluster import AgglomerativeClustering
from itertools import permutations
from tqdm import tqdm


def determine_n_clusters(
    dist_mat: pd.DataFrame,
    max_clusters: int = 10,
) -> tuple:
    """Given a DataFrame of embeddings, determine the number of clusters that maximizes
    the silhouette score.
    Parameters:
        dist_mat[pd.DataFrame]: DataFrame of distance matrices (n_units,n_units).
        max_clusters[int]: Maximum number of clusters to consider.
        seed[int]: Random seed for KMeans.
    Returns:
        best_cluster[int]: No. of clusters maximizing the silhouette score.
        best_score[float]: Max silhouette score."""

    scores = []
    for n_clusters in range(2, max_clusters + 1):
        model = AgglomerativeClustering(n_clusters=n_clusters, 
                                        metric="precomputed",
                                        linkage="complete",
                                        )
        clusters = model.fit_predict(dist_mat)
        score = silhouette_score(dist_mat, clusters, metric="precomputed")
        scores.append(score)
    scores = np.array(scores)

    best_cluster, best_score = np.argmax(scores) + 2, scores.max()
    return best_cluster, best_score, scores


def cluster_vec_to_sets(cluster_vec: pd.Series) -> dict:
    """Convert a cluster vector to a dictionary of sets,
    where the keys are the cluster labels and the values are the set of units in that cluster.
    Parameters:
        cluster_vec[np.array]: 1D array of cluster labels.
    Returns:
        cluster_sets[dict]: Dictionary of sets of units in each cluster."""

    cluster_sets = {
        i: set(np.where(cluster_vec == i)[0]) for i in np.unique(cluster_vec)
    }
    return cluster_sets


def find_best_permutation(vec_0: pd.Series, vec_1: pd.Series) -> tuple:
    """Given two dictionaries of sets, find the permutation of the keys of the first dictionary
    that maximizes the overlap with the keys of the second dictionary.
    Parameters:
        vec_0[np.array or pd.Series]: 1D array of cluster labels.
        vec_1[np.array or pd.Series]: 1D array of cluster labels.
    Returns:
        (vec_0, vec_1): original vectors where vec_1 has indices permuted according to best_perm.
        best_overlap[float]: Overlap between the two vectors after best_perm.
        best_perm[np.array]: Permutation of the indices that maximizes overlap."""

    swap = False
    assert len(vec_0) == len(vec_1), "Cluster vectors must be of equal length."

    if max(vec_0) < max(vec_1):
        swap = True
        vec_0, vec_1 = vec_1, vec_0

    perm_vec, perm_max = vec_1, max(vec_1)
    best_perm, best_overlap = None, 0
    pbar = tqdm(total=np.math.factorial(perm_max + 1))

    for perm in permutations(range(perm_max + 1)):
        new_vec = vec_1.copy()
        map_func = np.vectorize(lambda x: perm[x])
        new_vec = map_func(new_vec)
        overlap = np.mean(new_vec == vec_0)
        if overlap > best_overlap:
            best_overlap = overlap
            best_perm = perm
            perm_vec = new_vec
        pbar.update(1)

    if type(vec_1) == pd.Series:
        perm_vec = pd.Series(perm_vec, index=vec_1.index)

    return (vec_0, perm_vec) if not swap else (perm_vec, vec_0), best_overlap, best_perm


def compute_jaccard_indices(
    vec_0: pd.Series, vec_1: pd.Series
) -> np.array:  # symmetric metric
    """Given two vectors of clusters, compute the Jaccard index between them.
    Parameters:
        vec_0[np.array or pd.Series]: 1D array of cluster labels.
        vec_1[np.array or pd.Series]: 1D array of cluster labels.
    Returns:
        jaccard_indices[np.array]: Jaccard index for each cluster label."""

    assert len(vec_0) == len(vec_1), "Cluster vectors must be of equal length."
    jaccard_indices = jaccard_score(
        vec_0, vec_1, average=None, labels=range(max(max(vec_0), max(vec_1)) + 1)
    )
    return jaccard_indices
