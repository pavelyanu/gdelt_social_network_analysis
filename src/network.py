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

def gdelt_network_vanilla(gdelt: GDELT) -> nx.Graph:
    G = nx.Graph()
    tqdm.pandas(desc="Creating network")
    gdelt.df.progress_apply(
        lambda row: G.add_edge(
            row['actor1country'],
            row['actor2country'],
            weight=row['sum_nummentions']
        ),
        axis=1
    )
    return G
