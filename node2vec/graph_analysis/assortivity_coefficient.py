import networkx as nx
import graph_utils as utils

file_path = "../musae-facebook-pagepage-network/musae_facebook_edges.csv"

G = utils.read_graph(file_path)

assortivity_coefficient = nx.degree_assortativity_coefficient(G)

print(assortivity_coefficient)