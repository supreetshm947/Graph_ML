import networkx as nx
import graph_utils as utils

file_path = "../musae-facebook-pagepage-network/musae_facebook_edges.csv"

G = utils.read_graph(file_path)

clustering_coefficient = nx.average_clustering(G)

print(clustering_coefficient)
