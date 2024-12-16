import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from networkx.algorithms import community
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_gdelt_network(gdelt_sub, weight_col='weighted_sum_goldstein'):
    """
    Given a subset of the GDELT DataFrame for a specific year,
    create a directed NetworkX graph with edges weighted by the chosen metric.
    """
    G = nx.DiGraph()
    for _, row in gdelt_sub.iterrows():
        src = row['Actor1Country']
        tgt = row['Actor2Country']
        w = row[weight_col]
        G.add_edge(src, tgt, weight=w)
    return G

def create_migration_network(migration_sub, weight_col='flow'):
    """
    Given a subset of the Migration DataFrame for a specific year,
    create a directed graph with edges weighted by migration flows.
    """
    M = nx.DiGraph()
    for _, row in migration_sub.iterrows():
        src = row['iso_or']  # or 'origin' if you want consistent naming
        tgt = row['iso_des']
        w = row[weight_col] if not pd.isnull(row[weight_col]) else 0
        M.add_edge(src, tgt, weight=w)
    return M

def extract_common_edges_and_weights(G, M):
    """
    Extract edge weights for edges present in both graphs G and M.
    Returns two numpy arrays: gdelt_weights, migration_weights
    aligned by the same set of edges.
    """
    # Consider only the intersection of edges
    common_edges = set(G.edges()) & set(M.edges())
    gdelt_weights = []
    migration_weights = []
    for e in common_edges:
        gdelt_weights.append(G[e[0]][e[1]]['weight'])
        migration_weights.append(M[e[0]][e[1]]['weight'])
    return np.array(gdelt_weights), np.array(migration_weights)

def compute_correlation(gdelt_weights, migration_weights):
    """
    Compute a correlation measure (e.g. Pearson's r) between two arrays.
    """
    if len(gdelt_weights) == 0:
        return np.nan
    corr, _ = pearsonr(gdelt_weights, migration_weights)
    return corr

def shuffle_gdelt_weights(G):
    """
    Shuffle the edge weights of G while preserving its topology.
    This creates a new graph with the same edges but permuted weights.
    """
    edges = list(G.edges())
    weights = [G[u][v]['weight'] for u,v in edges]
    np.random.shuffle(weights)
    G_shuffled = G.copy()
    for (u,v), w in zip(edges, weights):
        G_shuffled[u][v]['weight'] = w
    return G_shuffled

def shuffle_once(G, M):
    # This helper function performs one shuffle iteration and returns a null correlation
    G_shuffled = shuffle_gdelt_weights(G)
    gdelt_w_shuffled, migration_w_shuffled = extract_common_edges_and_weights(G_shuffled, M)
    null_corr = compute_correlation(gdelt_w_shuffled, migration_w_shuffled)
    return null_corr

def run_shuffle_test(gdelt_df: pd.DataFrame, migration_df: pd.DataFrame, years: list[int], weight_col_gdelt: str = 'weighted_sum_goldstein', weight_col_migration: str = 'flow', n_jobs: int = -1):
    n_permutations = 1000

    results = []

    for year in years:
        # Subset data for this year
        gdelt_sub = gdelt_df[gdelt_df['Year'] == year]
        migration_sub = migration_df[migration_df['year'] == year + 1]

        # Create networks
        G = create_gdelt_network(gdelt_sub, weight_col=weight_col_gdelt)
        M = create_migration_network(migration_sub, weight_col=weight_col_migration)

        # Extract weights for common edges
        gdelt_w, migration_w = extract_common_edges_and_weights(G, M)

        # If you are testing influence on next year's migration (t->t+1),
        # you might need to shift the migration network by one year.
        # For demonstration, we assume same-year correlation,
        # but you can adjust accordingly.
        observed_corr = compute_correlation(gdelt_w, migration_w)

        # -----------------------------
        # Step 3: Shuffle Test
        # -----------------------------
        # null_distributions = []
        # for i in range(n_permutations):
        #     G_shuffled = shuffle_gdelt_weights(G)
        #     gdelt_w_shuffled, migration_w_shuffled = extract_common_edges_and_weights(G_shuffled, M)
        #     null_corr = compute_correlation(gdelt_w_shuffled, migration_w_shuffled)
        #     if not np.isnan(null_corr):
        #         null_distributions.append(null_corr)

        # null_distributions = np.array(null_distributions)

        # Run parallel permutations
        null_distributions = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(shuffle_once)(G, M) 
            for _ in range(n_permutations)
        )
        
        # Filter out NaNs
        null_distributions = np.array([x for x in null_distributions if not np.isnan(x)])


        # Compare observed correlation to the null distribution
        # For a two-sided test, compute the proportion of null values more extreme than observed
        p_value = (np.sum(np.abs(null_distributions) >= np.abs(observed_corr)) / len(null_distributions)) if len(null_distributions) > 0 else np.nan

        results.append({
            'year': year,
            'observed_corr': observed_corr,
            'p_value': p_value,
            'null_mean': np.mean(null_distributions) if len(null_distributions) > 0 else np.nan,
            'null_std': np.std(null_distributions) if len(null_distributions) > 0 else np.nan
        })

    results_df = pd.DataFrame(results)
    return results_df

def build_full_migration_network(df: pd.DataFrame):
    years = df['year'].unique()

    graphs_by_year = {}
    for y in years:
        df_y = df[df['year'] == y]
        G_y = nx.DiGraph()
        # Add nodes and edges
        origins = df_y['iso_or'].unique()
        destinations = df_y['iso_des'].unique()
        countries = set(origins).union(destinations)
        G_y.add_nodes_from(countries)
        for _, row in df_y.iterrows():
            if pd.notnull(row['stock']):
                G_y.add_edge(row['iso_or'], row['iso_des'], weight=row['stock'])
        graphs_by_year[y] = G_y

    return graphs_by_year

def get_full_info(graphs_by_year: dict):
    info_all = {}

    for y, G_y in graphs_by_year.items():
        info = get_G_info(G_y)
        info_all[y] = info

    return info_all

def plot_bar(top_list: list[(str, float)]):
    top_list.sort(key = lambda x: x[1], reverse=False)
    country_names = [c for c, _ in top_list]
    meas = [m for _, m in top_list]
    plt.barh(country_names, meas)

def get_animation(info_all: dict, measure = "betweenness"):
    frames = [int(x) for x in info_all.keys() if int(x) > 1990]
    
    fig, ax = plt.subplots(figsize = (12,6))

    def animate(frame):
        ax.clear()
        vals = get_top_values(info_all[frame][measure], 10)
        vals.sort(key = lambda x: x[1], reverse=False)
        country_names = [c for c, _ in vals]
        meas = [m for _, m in vals]
        ax.barh(country_names, meas)
        ax.set_title(f"Top 10 {measure} - Frame: {frame}")
        ax.set_xlabel(measure)
        ax.set_ylabel("Country")

    an = animation.FuncAnimation(fig, animate, frames, interval=500)
    return an



def build_year_migration_network(df: pd.DataFrame, year: int):
    df_year = df[df['year'] == year]
    G = nx.DiGraph()
    origins = df_year['iso_or'].unique()
    destinations = df_year['iso_des'].unique()
    all_countries = set(origins).union(set(destinations))
    G.add_nodes_from(all_countries)

    for _, row in df_year.iterrows():
        if not pd.isnull(row['stock']):
            G.add_edge(row['iso_or'], row['iso_des'], weight=row['stock'])

    return G 

def get_G_info(G: nx.DiGraph):
    info = {}

    in_degree = G.in_degree(weight='weight')
    out_degree = G.out_degree(weight='weight')
    info["in_degrees"] = {node: val for (node, val) in in_degree}
    info["out_degrees"] = {node: val for (node, val) in out_degree}
    info['node_num'] = len(G.nodes)
    info['edge_num'] = len(G.edges)

    max_edges = len(G.nodes) * (len(G.nodes) - 1)
    actual_edges = len(G.edges)
    info['density'] = actual_edges / max_edges if max_edges > 0 else 0

    info["betweenness"] = nx.betweenness_centrality(G, normalized=True)

    info['clustering_coefficients'] = nx.clustering(G)

    reciprocal_pairs = sum(1 for u, v in G.edges if G.has_edge(v, u))
    assert reciprocal_pairs % 2 == 0
    reciprocal_pairs /= 2
    info['reciprocity'] = reciprocal_pairs / (len(G.edges) - reciprocal_pairs) if len(G.edges) > 0 else 0

    if nx.is_strongly_connected(G):
        info['diameter'] = nx.diameter(G)
    else:
        info['diameter'] = float('inf')

    weak_components = nx.weakly_connected_components(G)
    info['weak_component_num'] = sum(1 for _ in weak_components)
    strong_components = nx.strongly_connected_components(G)
    info['strong_component_num'] = sum(1 for _ in strong_components)

    return info

def report_G_info(info: dict):
    for key, val in info.items():
        if type(val) != dict:
            print(f'{key}: {val}')


def get_communities(G):
    return community.greedy_modularity_communities(G, weight='weight')

def visualise_comm(G):
    comm = get_communities(G)
    node_to_comm = {}
    for i, c in enumerate(comm):
        for node in c:
            node_to_comm[node] = i

    colors = [node_to_comm[node] for node in G.nodes()]

    # Create a layout for the graph
    pos = nx.spring_layout(G, seed=42)  # seed for reproducibility

    # Draw the nodes, with the color determined by their community
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.tab20, node_size=100)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.axis('off')
    plt.title("Communities detected by Greedy Modularity")
    plt.show()


def get_top_values(dict: dict, k: int):
    return sorted(dict.items(), key=lambda x: x[1], reverse=True)[:k]

def visualise_graph(g: nx.DiGraph, year: int = 2011):
    plt.figure()
    pos = nx.spring_layout(g, k=0.5, seed=42)
    nx.draw(g, pos, node_size=300, alpha=0.7, with_labels=True)
    plt.title(f"Migration in {year}")
    plt.show()

def create_subgraph(g: nx.DiGraph, nodes_to_include: list):
    return g.subgraph(nodes_to_include).copy()

from typing import Union, List, Dict, Iterable

from tqdm.auto import tqdm

from src.data import GDELT

"""
GDELT HEAD:
year,actor1country,actor2country,weighted_sum_avgtone,weighted_sum_goldstein,sum_nummentions
1979,ABW,NLD,5,3.4,4
1979,AFG,AFG,5.5960831546356324,0.50163652024117145,1161
1979,AFG,ARE,6.79611650485437,1.9,8
1979,AFG,BEL,4.96613995485327,2.5,18
1979,AFG,BGR,6.0363729323650475,3.0799999999999992,100
1979,AFG,CAN,2.4390243902439,-10,18
1979,AFG,CHN,5.8165367352912618,-1.2832000000000003,125
1979,AFG,CUB,5.1874295187414967,3.4255813953488374,43
1979,AFG,CZE,9.1038284625995161,2.7956521739130431,46
"""

def gdelt_network_vanilla(
        gdelt: GDELT,
        years: Union[int, Iterable[int]] = 2020
    ) -> Union[nx.Graph, Dict[int, nx.Graph]]:

    years = [years] if isinstance(years, int) else years
    networks: Dict[int, nx.Graph] = {}
    for year in years:
        if year not in gdelt.df['year'].unique():
            raise ValueError(f"Year {year} not in GDELT data.")
        G = nx.Graph()
        tqdm.pandas(desc=f"Creating network for year {year}")
        df = gdelt.df[gdelt.df['year'] == year]
        df.progress_apply(
            lambda row: G.add_edge(
                row['actor1country_name'],
                row['actor2country_name'],
                weight=row['sum_nummentions']
            ),
            axis=1
        )
        networks[year] = G
    return networks if len(years) > 1 else networks[years[0]]
