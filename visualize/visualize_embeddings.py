from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
import pandas as pd
import seaborn as sns
from clusters.compare_clusters import determine_n_clusters
plt.rcParams.update({'font.size': 12})

def heatmap_embeddings(emb_param:pd.DataFrame, **fig_kwargs)-> tuple:
    """Plot the heatmap of the similarity matrix. Figure 2b in paper."""
    fig, ax = plt.subplots(1, **fig_kwargs)
    fig.suptitle("Embedding matrix")
    sns.heatmap(emb_param, ax=ax, cmap="Reds", cbar=True, **fig_kwargs)
    ax.set_xticks(range(len(emb_param.columns)), 
                  labels=range(len(emb_param.columns)), 
                  rotation=0)
    return fig, ax


def similarity_embeddings(
    emb_param0: pd.Series, emb_param1: pd.Series, **fig_kwargs
) -> tuple:
    
    for emb_param in [emb_param0, emb_param1]:
        assert (
            emb_param.apply(lambda x: np.isclose((x**2).sum(), 1), axis=0).all(),
            "Embeddings should be l2-normalized",
        )
    fig, ax = plt.subplots(1, figsize=(30, 30))
    fig.suptitle("Cosine similarity between embeddings")
    similarity = emb_param0.T.dot(emb_param1)

    sns.heatmap(
        similarity, ax=ax, cmap="Reds", vmin=-1, vmax=1, cbar=True, **fig_kwargs
    )
    ax.set_xticks(range(len(emb_param0)), rotation=0)
    return fig, ax


def plot_pca_embeddings(
    emb_param: pd.Series, n_components: int = None, **fig_kwargs
) -> tuple:
    assert (
        emb_param.apply(lambda x: np.isclose((x**2).sum(), 1), axis=0).all(),
        "Embeddings should be l2-normalized",
    )
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(40, 10), tight_layout=True, **fig_kwargs
    )
    fig.suptitle("PCA reduction")
    # if n_components is not set all components are kept
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(emb_param.T)
    explained_var_ratio = pca.explained_variance_ratio_
    ax0.grid(True)
    ax0.bar(range(len(explained_var_ratio)), explained_var_ratio)
    ax0.set_title("Explained variance ratio")
    x, y = pca_embeddings[:, 0], pca_embeddings[:, 1]
    ax1.scatter(x, y)
    ax1.grid(True)

    for i, name in enumerate(emb_param.columns):
        ax1.annotate(name, (x[i], y[i]))
    return fig, pca_embeddings



def plot_clusters(emb_param:pd.DataFrame, clusters:np.array, cfg:SimpleNamespace, reduction:str="PCA", annotate=False,**fig_kwargs)->plt.Figure:
    """Plot the clusters in 2D using PCA or TSNE reduction. Figure 2d in paper."""
    if reduction == "PCA":
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(emb_param.T)
        x, y = pca_embeddings[:, 0], pca_embeddings[:, 1]
    elif reduction == "TSNE":
        tsne = TSNE(n_components=2, metric="precomputed", init="random",random_state=42)
        tsne_embeddings = tsne.fit_transform(emb_param)
        x, y = tsne_embeddings[:, 0], tsne_embeddings[:, 1]
    else:
        raise ValueError("reduction should be either PCA or TSNE")
    print(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(**fig_kwargs)
    for cl in np.unique(clusters):
        ax.scatter(x[clusters == cl], 
                   y[clusters == cl], 
                   c=cfg.colors[cl], 
                   marker=cfg.markers[cl],
                   s=100,
                   label=f"Cluster {cl+1}")
    #ax.scatter(x, y, c=cluster_colors, marker='D', s=30)
    if annotate:
        for i, name in enumerate(emb_param.columns):
            ax.annotate(i, (x[i], y[i]), color="k", fontsize=12)
        #ax.annotate(i, (x[i]+np.random.randn(), y[i]+np.random.randn()), color="k", fontsize=14)

    ax.set_xlabel(f"Feature 1 ({reduction})")
    ax.set_ylabel(f"Feature 2 ({reduction})")
    ax.grid(True)
    
    return fig, ax




def plot_silhouette_score(dist_mat:pd.DataFrame, max_clusters:int)-> tuple:
    """Plot silhouette score for different number of clusters (Figure 2c in paper)."""
    fig, ax = plt.subplots()
    ax.grid(True)
    best_cluster, best_score, scores  = determine_n_clusters(dist_mat, max_clusters)
    ax.plot(range(2, max_clusters+1), scores)
    ax.axvline(x=best_cluster, color='r')
    ax.plot(best_cluster, best_score, c="r", marker="D")
    ax.set_ylim(0, 1)
    ax.annotate(f"Best score: {best_score:.2f}\nNo. of clusters: {best_cluster}", (best_cluster+.2, best_score),color='r')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette Score')
    
    return fig, ax