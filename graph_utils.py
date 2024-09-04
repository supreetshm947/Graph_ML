import pandas as pd
import networkx as nx
from torch_geometric.data import Data
import json
import torch


def read_graph(file_path):
    df = pd.read_csv(file_path, header=0)
    edge_list = [(edge[0], edge[1]) for edge in df.values.tolist()]
    G = nx.Graph(edge_list)
    return G


def read_graph_from_jsonl(jsonl_path, device, max_nodes=0):
    datasets = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data_dict = json.loads(line)
            node_feat = data_dict.get("node_feat")
            # padding the node features
            if len(node_feat) < max_nodes:
                node_feat = node_feat + [0] * (max_nodes - len(node_feat))
            node_feat = torch.tensor(node_feat, dtype=torch.float32).to(device)
            edge_index = torch.tensor(data_dict.get("edge_index"), dtype=int).to(device)
            edge_attr = torch.tensor(data_dict.get("edge_attr"), dtype=int).to(device)
            y = torch.tensor(data_dict['y'], dtype=torch.float).to(device)
            data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_attr, y=y)

            datasets.append(data)

    return datasets


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
