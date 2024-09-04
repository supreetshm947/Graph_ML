from torch import nn
import torch.nn.functional as f
from torch_geometric.utils import scatter


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.Linear = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels


    def forward(self, x, edge_index):
        x = self.Linear(x)
        in_node, out_node = edge_index
        num_nodes = x.size(0)
        out = scatter(x[in_node], out_node, dim=0, dim_size=num_nodes, reduce="mean")
        return f.relu(out)


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.layer1 = GCNLayer(in_channels, hidden_channels)
        self.layer2 = GCNLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        # batch contains index value which corresponds to which graph x belongs to
        x = scatter(x, batch, dim=0, reduce="mean")
        return x
