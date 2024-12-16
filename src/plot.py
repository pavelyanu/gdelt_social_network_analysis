from typing import Dict

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_gdelt_network(G: nx.Graph) -> None:
    """
    Plots the GDELT network.
    """
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=20, font_size=8)
    plt.show()
    

def plot_top_centrality_nodes(df: pd.DataFrame, centrality_measure: str, top_n: int = 10) -> None:
    """
    Plot a bar chart of the top N nodes for a given centrality measure.
    df: DataFrame with centrality scores (index are nodes).
    """
    if centrality_measure not in df.columns:
        raise ValueError(f"{centrality_measure} not found in DataFrame. Available columns: {df.columns}")
    top_nodes = df[centrality_measure].dropna().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    top_nodes.plot.barh()
    plt.title(f"Top {top_n} by {centrality_measure}")
    plt.gca().invert_yaxis()  # highest at top
    plt.xlabel(centrality_measure)
    plt.show()


def plot_communities(G: nx.Graph, communities: pd.Series) -> None:
    """
    Color nodes by community and display the network.
    """
    pos = nx.spring_layout(G, seed=42)  # fixed layout for consistency

    unique_comms = communities.unique()
    color_map = plt.cm.get_cmap('hsv', len(unique_comms))
    node_colors = [color_map(c) for c in communities]

    plt.figure(figsize=(10,10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    # If labels needed:
    nx.draw_networkx_labels(G, pos, font_size=6)
    plt.title("Communities")
    plt.axis('off')
    plt.show()


def plot_network_summary_over_time(summary_df: pd.DataFrame) -> None:
    """
    Given a summary DataFrame with columns like year, density, etc.,
    plot how these metrics evolve over time.
    """
    if 'year' not in summary_df.columns:
        raise ValueError("summary_df must have a 'year' column")

    metrics = [col for col in summary_df.columns if col not in ['year']]
    summary_df = summary_df.set_index('year')

    summary_df[metrics].plot(subplots=True, layout=(len(metrics),1), figsize=(10, 6), sharex=True)
    plt.tight_layout()
    plt.show()


def plot_network_with_year_slider(networks: Dict[int, nx.Graph]):
    years = sorted(networks.keys())
    initial_year = years[0]
    G = networks[initial_year]

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)

    pos = nx.spring_layout(G, seed=42)

    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50)
    edges = nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    labels = nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    ax.set_title(f"Network for Year {initial_year}")
    ax.axis('off')

    metric_ax = fig.add_axes([0.85, 0.4, 0.13, 0.5])
    metric_ax.axis('off')

    def update_metrics(G_current, year):
        metric_ax.clear()
        metric_ax.axis('off')
        num_nodes = G_current.number_of_nodes()
        num_edges = G_current.number_of_edges()
        density = nx.density(G_current)
        metric_ax.text(0, 0.8, f"Year: {year}", fontsize=10)
        metric_ax.text(0, 0.6, f"Nodes: {num_nodes}", fontsize=10)
        metric_ax.text(0, 0.4, f"Edges: {num_edges}", fontsize=10)
        metric_ax.text(0, 0.2, f"Density: {density:.4f}", fontsize=10)

    update_metrics(G, initial_year)

    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    year_slider = Slider(slider_ax, 'Year', years[0], years[-1], valinit=initial_year, valstep=1)

    def update(val):
        year = int(year_slider.val)
        if year in networks:
            G_new = networks[year]
            ax.clear()
            ax.axis('off')
            pos_new = nx.spring_layout(G_new, seed=42)
            nx.draw_networkx_nodes(G_new, pos_new, ax=ax, node_size=50)
            nx.draw_networkx_edges(G_new, pos_new, ax=ax, alpha=0.3)
            nx.draw_networkx_labels(G_new, pos_new, font_size=6, ax=ax)
            ax.set_title(f"Network for Year {year}")
            update_metrics(G_new, year)
            fig.canvas.draw_idle()

    year_slider.on_changed(update)

    plt.show()