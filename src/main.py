from argparse import ArgumentParser, Namespace

import networkx as nx

from src.analysis import *
from src.data import *
from src.network import *
from src.plot import *

parser = ArgumentParser()

parser.add_argument('--migration_path', type=str, default="data/migration_bilateral.csv", help='Path to migration file. Absolute or relative path.')
parser.add_argument('--refugee_path', type=str, default="data/refugee_bilateral.csv", help='Path to refugee file. Absolute or relative path.')
parser.add_argument('--gdelt_path', default="data/gdelt_social.csv", type=str, help='Path to GDELT file. Absolute or relative path.')

DATA_PATHS = {
    'migration': "data/migration_bilateral.csv",
    'refugee': "data/refugee_bilateral.csv",
    'gdelt': "data/gdelt_social.csv"
}

CENTRALITIES = [
    'degree_centrality',
    'betweenness',
    'closeness',
    'eigenvector'
]

DATA_PATHS = {dataset: "../" + path for dataset, path in DATA_PATHS.items()}

def main(args: Namespace):
    migration = get_migration(args.migration_path)
    refugee = get_refugee(args.refugee_path)
    gdelt = get_gdelt(args.gdelt_path)
    gdelt.clean_data()

    networks = gdelt_network_vanilla(gdelt, years=range(2000, 2025))
    plot_yearly_centralities(networks)
    plot_yearly_communities(networks)

def print_netwrork_summary(gdelt):
    network = gdelt_network_vanilla(gdelt, years=2024)
    summary = summarize_network(network)
    print(summary)

def plot_temporal_analysis(networks: Dict[int, nx.Graph]):
    df = temporal_analysis(networks)
    plot_network_summary_over_time(df)

def plot_yearly_centralities(networks: Dict[int, nx.Graph]):
    centralities_per_year = compute_yearly_centralities(networks)
    for centrality in CENTRALITIES:
        plot_top_k_centrality_by_year(centralities_per_year, centrality, top_k=10)

def compute_yearly_centralities(networks: Dict[int, nx.Graph]):
    centralities_per_year = {}
    for year, G in networks.items():
        centralities_per_year[year] = compute_basic_centralities(G)
    return centralities_per_year

def plot_yearly_communities(networks: Dict[int, nx.Graph]):
    communities_per_year = compute_yearly_communities(networks)
    return plot_communities_on_map(communities_per_year)

def compute_yearly_communities(networks: Dict[int, nx.Graph]):
    communities_per_year = {}
    for year, G in networks.items():
        communities_per_year[year] = compute_communities(G)
    return communities_per_year

def plot_network(gdelt: GDELT):
    centroids = load_country_centroids()
    df = gdelt.df
    return plot_network_on_map_with_slider(df, centroids)

if __name__ == '__main__':
    args: Namespace = parser.parse_args()
    main(args)