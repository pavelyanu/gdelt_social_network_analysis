import networkx as nx
import matplotlib.pyplot as plt

def plot_gdelt_network(G: nx.Graph) -> None:
    """
    Plots the GDELT network.
    """
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=20, font_size=8)
    plt.show()