from argparse import ArgumentParser, Namespace

from src.analysis import *
from src.data import *
from src.network import *
from src.plot import *

parser = ArgumentParser()

parser.add_argument('--migration_path', type=str, default="data/migration_bilateral.csv", help='Path to migration file. Absolute or relative path.')
parser.add_argument('--refugee_path', type=str, default="data/refugee_bilateral.csv", help='Path to refugee file. Absolute or relative path.')
parser.add_argument('--gdelt_path', default="data/gdelt_social.csv", type=str, help='Path to GDELT file. Absolute or relative path.')

def main(args: Namespace):
    migration = get_migration(args.migration_path)
    refugee = get_refugee(args.refugee_path)
    gdelt = get_gdelt(args.gdelt_path)
    gdelt.clean_data()

    G_2020 = gdelt_network_vanilla(gdelt, 2020)

    centralities = compute_basic_centralities(G_2020)
    communities = compute_communities(G_2020)
    summary = summarize_network(G_2020)
    print(summary)

    plot_top_centrality_nodes(centralities, 'degree_centrality', top_n=10)
    plot_top_centrality_nodes(centralities, 'betweenness', top_n=10)
    plot_top_centrality_nodes(centralities, 'closeness', top_n=10)
    plot_top_centrality_nodes(centralities, 'eigenvector', top_n=10)
    plot_communities(G_2020, communities)

    years = range(1990, 2001)
    all_networks = gdelt_network_vanilla(gdelt, years)
    time_summary = temporal_analysis(all_networks)
    plot_network_summary_over_time(time_summary)

    plot_network_with_year_slider(all_networks)

if __name__ == '__main__':
    args: Namespace = parser.parse_args()
    main(args)