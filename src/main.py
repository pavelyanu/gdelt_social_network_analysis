from argparse import ArgumentParser, Namespace

from src.data import *

parser = ArgumentParser()

parser.add_argument('--migration_path', type=str, default="data/migration_bilateral.csv", help='Path to migration file. Absolute or relative path.')
parser.add_argument('--refugee_path', type=str, default="data/refugee_bilateral.csv", help='Path to refugee file. Absolute or relative path.')
parser.add_argument('--gdelt_path', default="data/gdelt_social.csv", type=str, help='Path to GDELT file. Absolute or relative path.')

def main(args: Namespace):
    migration = get_migration(args.migration_path)
    refugee = get_refugee(args.refugee_path)



if __name__ == '__main__':
    args: Namespace = parser.parse_args()
    main(args)