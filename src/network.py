import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from joblib import Parallel, delayed

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