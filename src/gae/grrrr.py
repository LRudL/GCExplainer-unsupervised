import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


len = 10

sources = (np.arange(0, len),)
targets = np.concatenate((np.arange(1, len), np.array([0])))

edges = np.vstack((sources, targets)).transpose()
labels = np.zeros(len)

G = nx.Graph()
G.add_edges_from(edges)
nx.draw(G)
plt.show()
print("drawn")
