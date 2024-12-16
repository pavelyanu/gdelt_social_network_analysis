from typing import Union, List, Dict, Iterable

import networkx as nx
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

        def add_edge(row):
            if not G.has_node(row['actor1country_name']):
                G.add_node(row['actor1country_name'], iso=row['actor1country'])
            if not G.has_node(row['actor2country_name']):
                G.add_node(row['actor2country_name'], iso=row['actor2country'])

            G.add_edge(
                row['actor1country_name'],
                row['actor2country_name'],
                weight=row['sum_nummentions']
            )

        df.progress_apply(add_edge, axis=1)
        networks[year] = G
    return networks if len(years) > 1 else networks[years[0]]

