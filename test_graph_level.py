import networkx as nx
from gcn import GCN
import numpy as np


G = nx.karate_club_graph()
x = np.random.normal(size=(len(G.nodes), 2))
y = 1

gcn = GCN()
gcn.fit_graphs([
    (G, x, y)
])
