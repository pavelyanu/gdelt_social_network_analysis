from typing import Dict

import networkx as nx
import pandas as pd
from networkx.algorithms import community

def compute_basic_centralities(G: nx.Graph) -> pd.DataFrame:
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigen = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigen = {n: None for n in G.nodes()}

    df = pd.DataFrame({
        'degree_centrality': degree_centrality,
        'betweenness': betweenness,
        'closeness': closeness,
        'eigenvector': eigen
    })
    return df


def compute_communities(G: nx.Graph) -> pd.DataFrame:
    """
    Compute communities and return a DataFrame with columns: ['iso', 'country_name', 'community'].
    """
    comms = community.greedy_modularity_communities(G, weight='weight')
    records = []
    for i, c in enumerate(comms):
        for node in c:
            iso = G.nodes[node]['iso'] if 'iso' in G.nodes[node] else None
            records.append({
                'country_name': node,
                'iso': iso,
                'community': i
            })
    df = pd.DataFrame(records)
    return df

def summarize_network(G: nx.Graph) -> pd.DataFrame:
    data = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
    }
    if not G.is_directed():
        data['avg_clustering'] = nx.average_clustering(G, weight='weight')

    return pd.DataFrame([data])


def temporal_analysis(networks: Dict[int, nx.Graph]) -> pd.DataFrame:
    results = []
    for year, G in networks.items():
        row = {
            'year': year,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G)
        }
        # Add more if needed
        results.append(row)
    return pd.DataFrame(results)
