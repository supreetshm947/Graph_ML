import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import graph_utils as utils

file_path = "../musae-facebook-pagepage-network/musae_facebook_edges.csv"

G = utils.read_graph(file_path)

degrees = [G.degree(node) for node in G.nodes()]
plt.hist(degrees, bins=100)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')
plt.show()