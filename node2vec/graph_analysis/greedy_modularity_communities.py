from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx
import graph_utils as utils
import matplotlib.pyplot as plt

file_path = "../musae-facebook-pagepage-network/musae_facebook_edges.csv"

G = nx.karate_club_graph()

def plot_graph(G, communities, title):
    pos = nx.spring_layout(G)
    colors = [plt.cm.tab10(i) for i in range(len(communities))]
    for community, color in zip(communities, colors):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=[color])
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.show()

# Initial state: each node is its own community
initial_communities = [{node} for node in G.nodes()]
plot_graph(G, initial_communities, "Initial State")

# First merge
first_merge = greedy_modularity_communities(G)
plot_graph(G, first_merge, "First Merge")

# Final communities
plot_graph(G, first_merge, "Final Communities")