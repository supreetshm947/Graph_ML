from torch import nn
import torch
from torch_scatter import scatter_softmax, scatter_sum, scatter
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, heads = 1, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.heads = heads
        self.out_channels = out_channels// self.heads

        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_lin = nn.Linear(edge_dim, out_channels, bias=False)
        self.attn = nn.Parameter(torch.Tensor(heads, 3 * self.out_channels))
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

        nn.init.xavier_uniform_(self.attn.data, gain=1.414)

    def forward(self, x, edge_index, edge_attr):
        """
            Args:
               x: Node feature matrix (N, in_features)
               edge_attr: Edge Attributes (E*edge_dim)
               edge_index: Graph connectivity matrix (2, E)
            Returns:
               Updated node feature matrix (N, out_features)
        """
        x = self.W(x)  # (N, out_features)
        x = x.view(-1, self.heads, self.out_channels)   # (N, H, out_features/H)

        edge_attr = self.edge_lin(edge_attr).view(-1, self.heads, self.out_channels)
        src, dst = edge_index

        x_src = x[src]  # (E, out_features)
        x_dst = x[dst]  # (E, out_features)

        edge_features = torch.cat([x_src, x_dst, edge_attr], dim=-1)  # (E, 3*out_features)

        attn_input = self.attn[:, :3 * self.out_channels]

        e = self.leakyrelu((edge_features * self.attn.unsqueeze(0)).sum(dim=-1))   # (E, H)

        alpha = scatter_softmax(e, dst, dim=0) # (E, H)

        x_aggr = x_src * alpha.unsqueeze(-1) # (E, H, F_out/H)
        out = scatter_sum(x_aggr, dst, dim=0, dim_size=x.size(0))    # (N, H, F_out/H)

        out = out.view(-1, self.heads * self.out_channels)  # (N, F_out)

        return self.dropout(out)


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels , heads=1, num_layer=2, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_channels, hidden_channels, edge_dim,  heads=heads, dropout=dropout, alpha=alpha))
        for i in range(num_layer - 2):
            self.layers.append(
                GATLayer(hidden_channels, hidden_channels, edge_dim, heads=heads, dropout=dropout, alpha=alpha))

        self.layers.append(GATLayer(hidden_channels, out_channels, edge_dim, heads=1, dropout=dropout, alpha=alpha))


    def forward(self, x, edge_index, edge_attrib, batch):
            for layer in self.layers:
                x = layer(x, edge_index, edge_attrib)
                x = F.elu(
                    x) # Exponential Linear Unit (ELU) activation to add non-linearity and stabilizes training and prevent vanishing gradients.
            x = scatter(x, batch, dim=0, reduce="mean")
            return x
