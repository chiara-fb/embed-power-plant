import pandas as pd
import json
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

plt.rcParams['font.size'] = 16
plt.style.use('ggplot')



def plot_cluster_heating(stats:pd.DataFrame, cfg:SimpleNamespace) -> tuple[plt.Figure, plt.Axes]:
    """Scatterplot of seasonal differences in generation and correlation with heat demand for 
    each unit in the cluster (Figure 4)."""
    
    stats['Cluster'] = stats['Cluster'].map(lambda x: f"Cluster {x+1}")
    cluster_palette = {f"Cluster {i+1}":cfg.colors[i] for i in range(4)}
    fig, axes = plt.subplots(2,2, figsize=(20, 15), tight_layout=True, sharex=True, sharey=True)
    fig.text(0.5, 0.04, 'Difference between cold and warm season (%/MWh)', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Correlation with residential heat demand', va='center', rotation='vertical', fontsize=20)

    for i in range(4):
        ax = axes[i//2, i%2]
        ax.axhline(0, color='black', linestyle='--')
        ax.axvline(0, color='black', linestyle='--')
        j = i+cfg.years[0]
        sns.scatterplot(x=f"Winter-Summer_{j}", y=f"MFH_{j}", 
                        data=stats[stats['Cluster'] != 'Cluster 4'].sort_values('Cluster'), 
                        hue='Cluster', palette=cluster_palette, ax=ax, style='CHP', s=100)
        ax.set_xlabel('')
        ax.set_xticklabels([f"{tick:.0%}" for tick in ax.get_xticks()])
        ax.set_ylabel('')
        ax.set_title(j)
    
    return fig, axes



def plot_freq(freq:pd.DataFrame, 
              cfg:SimpleNamespace, 
              yaxis:int) -> tuple[plt.Figure, plt.Axes]:
    """Scatterplot of differences in unit states (Figure not in paper)."""
    
    assert yaxis in [1,2], 'yaxis must be 1 or 2'
    fig, ax = plt.subplots(figsize=(10, 6))
    ax_histx = ax.inset_axes([0, 1.0, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1, 0, 0.25, 1], sharey=ax)
    ax_histx.axis('off')
    ax_histy.axis('off')
    ax.grid(True)
    for cl in range(1,5):
        ax.scatter(freq[f'Cluster {cl} - 0'], 
                   freq[f'Cluster {cl} - {yaxis}'], 
                   color=cfg.colors[cl-1], 
                   marker=cfg.markers[cl-1], 
                   label=f'Cluster {cl}', s=100)
        sns.kdeplot(x=freq[f'Cluster {cl} - 0'], 
                    color=cfg.colors[cl-1], 
                    ax=ax_histx)
        sns.kdeplot(y=freq[f'Cluster {cl} - {yaxis}'], 
                    color=cfg.colors[cl-1], 
                    ax=ax_histy)
        ax.legend(fontsize=12)

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], labels=['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_xlabel('Off')
    ax.set_ylabel('Full-load' if yaxis == 2 else 'Mid-load')
    return fig, ax


def hide_spines(ax:plt.Axes) -> plt.Axes:
    ax.spines['top'].set_visible(False)  # Hide top spine
    ax.spines['right'].set_visible(False)  # Hide right spine
    ax.spines['left'].set_visible(False)  # Hide left spine
    ax.yaxis.set_visible(False)  # Hide y-axis
    ax.set_facecolor(color="#F5F5F5")
    return ax

def hex_to_rgba(hex:str) -> tuple[float, float, float, float]:
    """
    Convert a hexadecimal color to an RGBA tuple.
    
    Parameters:
        hex (str): A hexadecimal color string, e.g., #990099
        
    Returns:
        tuple: A tuple (R, G, B, A) where each value is in the range [0, 1].
    """
    hex = hex.lstrip('#')  
    rgb = tuple(int(hex[i:i+2], 16) / 255 for i in (0, 2, 4)) + (1,)
    return rgb


def plot_cluster_kde(df:pd.DataFrame,
                     cfg:SimpleNamespace,
                     col:str,
                     name:str,
                     measure_unit:str="",
                     kde_args:dict=None) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the kernel density estimate (KDE) of a given column in the DataFrame, colored by clusters.
    (Figure not in paper)."""

    fig, ax = plt.subplots(figsize=(10, 6))
    rgba_colors = [hex_to_rgba(col) for col in cfg.colors]
    ax = hide_spines(ax)
    ax = sns.kdeplot(data=df, 
                    x=col, 
                    hue="clusters", 
                    linewidth=2,
                    common_norm=False,
                    ax = ax, 
                    palette=rgba_colors,
                    legend=True,
                    **(kde_args if kde_args is not None else {}),
                    )
    handles_dict = {}

    for line in ax.get_lines():
        x, y = line.get_data()
        color = line.get_color()
        cluster = rgba_colors.index(color)
        marker = cfg.markers[cluster]
        label = "Cluster "+str(cluster+1)
        ax.plot(x[::10], y[::10], color=color, marker=marker, linestyle="none", markersize=8)
        handle = Line2D([0], [0], color=color, marker=marker, label=label, linestyle="none", markersize=8)
        handles_dict[cluster] = handle
    
    handles_list = [val for _,val in sorted(handles_dict.items(), key=lambda x: x[0])]
    ax.legend(handles=handles_list, loc='upper right')
    fig.suptitle(name)
    ax.set_xlabel(measure_unit)

    return fig, ax

def plot_cluster_bar(df:pd.DataFrame, 
                     col:str, 
                     name:str, 
                     cfg:SimpleNamespace) -> tuple[plt.Figure, plt.Axes]:
    """Bar plot of the mean of a given column in the DataFrame, grouped by clusters.
    (Figure not in paper)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    hide_spines(ax)
    df.groupby("clusters")[col].mean().plot(kind='barh', stacked=True, ax=ax, color=cfg.colors)
    ax.set_title(name)
    ax.invert_yaxis()
    ax.set_ylabel("")
    ax.set_xticks([0,0.2,0.4,0.6,0.8,1], labels=["0%","20%","40%","60%","80%","100%"])
    return fig, ax

