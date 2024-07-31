import pandas as pd
import networkx as nx

def read_graph(file_path):
    df = pd.read_csv(file_path, header=0)
    edge_list = [(edge[0], edge[1]) for edge in df.values.tolist()]
    G = nx.Graph(edge_list)
    return G