from torch import nn
import torch.nn.functional as f
from torch_geometric.utils import scatter
from torch_geometric.nn.norm import BatchNorm


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super(GCNLayer, self).__init__()
        self.Linear = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.BatchNorm = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, edge_index):
        x = self.Linear(x)
        in_node, out_node = edge_index
        num_nodes = x.size(0)
        out = scatter(x[in_node], out_node, dim=0, dim_size=num_nodes, reduce="mean")
        out = self.BatchNorm(out)
        out = f.relu(out)
        return self.dropout(out)


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.layer1 = GCNLayer(in_channels, hidden_channels)
        self.layer2 = GCNLayer(hidden_channels, hidden_channels)
        self.layer3 = GCNLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        # batch contains index value which corresponds to which graph x belongs to
        x = scatter(x, batch, dim=0, reduce="mean")
        return x
