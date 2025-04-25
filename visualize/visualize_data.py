import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from types import SimpleNamespace

plt.rcParams.update({'font.size': 12})

def generate_color_gradient(start_color: str, end_color: str, steps: int) -> list:
    """
    Generate a list of colors forming a gradient between start_color and end_color.

    Parameters:
        start_color[str]: Starting color in any matplotlib-compatible format (e.g., '#999999').
        end_color[str]: Ending color in any matplotlib-compatible format (e.g., '#000000').
        steps[int]: Number of colors in the gradient.

    Returns:
        List[str]: List of colors in the gradient.
    """
    start_rgb = mcolors.to_rgb(start_color)
    end_rgb = mcolors.to_rgb(end_color)

    # Interpolate RGB values
    gradient = [
        mcolors.to_hex(
            (
                np.array(start_rgb) * (1 - i / (steps - 1))
                + np.array(end_rgb) * (i / (steps - 1))
            )
        )
        for i in range(steps)
    ]
    return gradient




## Imbalance classes ###
def class_frequency(df: pd.DataFrame, cfg: SimpleNamespace, **fig_kwargs) -> tuple:
    fig, ax = plt.subplots(figsize=(20, 15))
    #fig, ax = plt.subplots(**fig_kwargs)
    discrete_data = df["ActualGenerationOutput"][cfg.units]
    discrete_data = MinMaxScaler().fit_transform(discrete_data)
    discrete_data = (discrete_data * cfg.num_classes).astype(int)
    discrete_data = discrete_data.clip(0, cfg.num_classes - 1)
    freq_matrix = np.stack(
        [(discrete_data == cl).mean(axis=0) for cl in range(cfg.num_classes)]
    )
    sns.heatmap(freq_matrix, ax=ax, vmin=0, vmax=1, cmap="Reds", cbar=True)
    # ax.set_yticklabels(range(num_classes-1, -1, -1))
    #ax.set_xticks(range(cfg.num_units), labels=cfg.units, rotation=45)
    ax.set_xticks(range(cfg.num_units), labels=range(cfg.num_units), fontsize=12, rotation=0)
    fig.suptitle("Class frequency")
    return fig, ax, freq_matrix

def cluster_by_ownertype(stats:pd.DataFrame, cfg:SimpleNamespace) -> tuple[plt.Figure, plt.Axes]:
    """Cluster the stats by owner type, given that stats contains the columns "Owner short" and
    the "clusters" column. Called by cluster_statistics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_by_owner = stats.groupby(["clusters","Owner type"]).size().unstack().fillna(0)
    cluster_by_owner = cluster_by_owner.div(cluster_by_owner.sum(axis=1), axis=0)
    cluster_by_owner = cluster_by_owner.rename(columns={'Big': 'Major utility', 
                                                        'SW': 'Municipal utility', 
                                                        'Other': 'Other utility'})
    cluster_by_owner = cluster_by_owner.sort_index(axis=1)
    owner_map = mcolors.ListedColormap(cfg.colors[:len(cluster_by_owner.columns)])
    cluster_by_owner.plot(kind='barh', stacked=True, ax=ax, colormap=owner_map)
    ax.set_title("Ownership type by cluster")
    ax.set_xticklabels([f"{int(x*100)}%" for x in ax.get_xticks()])
    ax.set_yticklabels([f"{int(x)+1}" for x in ax.get_yticks()])
    ax.grid(axis='x')
    ax.legend(title="Owner type", loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4)
    ax.invert_yaxis()

    return fig, ax