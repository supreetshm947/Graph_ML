from torch_geometric.utils import scatter
import torch

x = torch.tensor([[1.0],  # Node 0 feature
                  [2.0],  # Node 1 feature
                  [3.0]])


# Edge index (source nodes, target nodes)
edge_index = torch.tensor([[0, 0],  # in_node (source)
                           [1, 2]]) # out_node (target)

# Extract in_node and out_node from edge_index
in_node = edge_index[0]  # [0, 1]
out_node = edge_index[1]  # [1, 2]

# Select source node features
x_in = x[in_node]  # x[0], x[1] -> [[1.0], [2.0]]
print("Source node features (x_in):\n", x_in)

# Perform scatter aggregation
# Aggregate source node features into target nodes (using mean)
num_nodes = x.size(0)  # Total number of nodes
out = scatter(x_in, out_node, dim=0, dim_size=num_nodes, reduce='sum')

print("\nAggregated node features after scatter:\n", out)