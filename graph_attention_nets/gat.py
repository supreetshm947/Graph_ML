from torch import nn
import torch
from torch_scatter import scatter_softmax


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()

        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, 2 * out_channels))
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

        nn.init.xavier_uniform(self.attn.data, gain=1.414)

        def forward(self, x, edge_index):
            """
                Args:
                   x: Node feature matrix (N, in_features)
                   edge_index: Graph connectivity matrix (2, E)
                Returns:
                   Updated node feature matrix (N, out_features)
            """
            x = self.W(x) # (N, out_features)

            src, dst = edge_index

            x_src = x[src]  # (E, out_features)
            x_dst = x[dst]  # (E, out_features)

            edge_features = torch.cat([x_src, x_dst], dim=1)    # (E, 2*out_features)

            e = self.leakyrelu(torch.matmul(edge_features, self.attn.T).squeeze())

            alpha = scatter_softmax(e, dst, dim=0)
            




