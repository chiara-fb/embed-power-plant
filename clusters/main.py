import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from sklearn.cluster import AgglomerativeClustering
# for the pictures
from visualize.visualize_embeddings import *
from visualize.visualize_data import class_frequency
# for the clusters
from clusters.retrieve_embeddings import model_file_to_embeddings, from_features_to_cosine_dist
from clusters.compare_clusters import find_best_permutation, compute_jaccard_indices
from clusters.cluster_statistics import cluster_statistics
from pathlib import Path
import json
import pandas as pd
from itertools import combinations
from types import SimpleNamespace
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt


def load_embs_dist_cfg(path:str, model_name:str) -> tuple:
    """Load configuration and model files for a given path 
    and model_name. Returns embeddings and distance matrix.
    """

    cfg_name = Path(path) / (model_name + '.json')
    model_name = Path(path) / (model_name + '.pth')

    with open(cfg_name, 'r') as f:
        cfg_var = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    emb_param = model_file_to_embeddings(model_name, cfg_var, normalized=True)

    if 'encodings' in cfg_var.__dict__:
        dist_mat = from_features_to_cosine_dist(cfg_var, emb_param) 
    else:
        dist_mat = 1 - emb_param.T.dot(emb_param)

    return emb_param, dist_mat, cfg_var


def save_all_figures(emb_param:pd.DataFrame, 
                     dist_mat:pd.DataFrame, 
                     cluster:pd.Series, 
                     cfg_var:SimpleNamespace, 
                     model_var:str,
                     color_dict:dict) -> None:
    """Save all figures for a given model variation"""
    cluster_colors = [color_dict[i] for i in cluster]
    fig0, _ = heatmap_embeddings(emb_param)
    fig1, ax1 = plt.subplots()
    sns.heatmap(dist_mat.values, cmap='Reds', ax=ax1, vmin=0, vmax=2)
    ax1.set_title('Cosine distance')
    fig2, _ = plot_pca_embeddings(emb_param)         
    fig4, _ = plot_silhouette_score(dist_mat, cfg_var.num_units -2)
    fig5 = plot_clusters(dist_mat,cluster_colors, reduction="TSNE")
    fig6, _, _ = class_frequency(pd.read_parquet(cfg_var.data_file), cfg_var)
    fig0.savefig(f'pictures/embeddings{model_var}.svg')
    fig1.savefig(f'pictures/dist_mat{model_var}.svg')
    fig2.savefig(f'pictures/pca{model_var}.svg')
    fig4.savefig(f'pictures/silhouette_score{model_var}.svg')
    fig5.savefig(f'pictures/{len(set(cluster))}clusters_{model_var}.svg') 
    fig6.savefig(f'pictures/class_frequency{model_var}.svg')


def main(path, 
         models, 
         n_clusters=None, 
         stats_file=None,
         save_fig=False,
         to_excel=False,
         ):

    #Step 1: Load the configuration file
    EMBEDDINGS, DIST_MAT, CLUSTERS, CFG = {}, {}, {}, {}
    n = n_clusters if n_clusters else 4
    agg_cluster = AgglomerativeClustering(n_clusters=n, 
                                    metric='precomputed', 
                                    linkage='complete')
    make_clusters = lambda d, cfg: pd.Series(agg_cluster.fit_predict(d), 
                                        name='clusters', index=cfg.units)
    
    #Step 2: load the embeddings
    for model_name in models:
        print(f"model_var: {model_name}")
        emb_param, dist_mat, cfg_var = load_embs_dist_cfg(path, model_name)
        dist_mat = dist_mat.clip(0, 2)
        cluster = make_clusters(dist_mat, cfg_var)
        color_dict = {i: to_rgba(c) for i, c in enumerate(cfg_var.colors)}

        EMBEDDINGS[model_name] = emb_param
        DIST_MAT[model_name] = dist_mat
        CLUSTERS[model_name] = cluster
        CFG[model_name] = cfg_var
        
        if save_fig:
            save_all_figures(emb_param, dist_mat, cluster, cfg_var, model_name, color_dict)
        
    
        if to_excel:
            with pd.ExcelWriter(f'{model_name}.xlsx', engine='openpyxl') as writer:
                #Step 3: Save the embeddings and distance matrix
                pd.DataFrame(EMBEDDINGS[model_name]).to_excel(writer, sheet_name=f'embeddings')
                pd.DataFrame(DIST_MAT[model_name]).to_excel(writer, sheet_name=f'dist_mat')
                #Step 4: Save cluster statistics
                if stats_file: 
                    avg_stats, mad_stats = cluster_statistics(CLUSTERS[model_name], 
                                                            stats_file, 
                                                            cfg_var)
                    avg_stats.to_excel(writer, sheet_name=f'avg_stats')
                    mad_stats.to_excel(writer, sheet_name=f'mad_stats')                
                
                clusters = pd.DataFrame(CLUSTERS[model_name])
                clusters.to_excel(writer, sheet_name='clusters')

    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("model_variations", type=str, nargs='+')
    parser.add_argument("--n_clusters", type=int, default=None, required=False)
    parser.add_argument("--stats_file", type=str, default=None, required=False)
    parser.add_argument("--save_fig", action="store_true", required=False)
    parser.add_argument("--to_excel", action="store_true", required=False)

    args = parser.parse_args()
    main(args.path, 
         args.model_variations, 
         n_clusters=args.n_clusters, 
         stats_file=args.stats_file,
         save_fig=args.save_fig, 
         to_excel=args.to_excel, 
         )