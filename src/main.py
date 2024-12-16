from argparse import ArgumentParser, Namespace

from src.data import *
from src.network import gdelt_network_vanilla
from src.plot import plot_gdelt_network

parser = ArgumentParser()

parser.add_argument('--migration_path', type=str, default="data/migration_bilateral.csv", help='Path to migration file. Absolute or relative path.')
parser.add_argument('--refugee_path', type=str, default="data/refugee_bilateral.csv", help='Path to refugee file. Absolute or relative path.')
parser.add_argument('--gdelt_path', default="data/gdelt_social.csv", type=str, help='Path to GDELT file. Absolute or relative path.')

def main(args: Namespace):
    migration = get_migration(args.migration_path)
    refugee = get_refugee(args.refugee_path)
    gdelt = get_gdelt(args.gdelt_path)

    G = gdelt_network_vanilla(gdelt)
    plot_gdelt_network(G)



if __name__ == '__main__':
    args: Namespace = parser.parse_args()
    main(args)