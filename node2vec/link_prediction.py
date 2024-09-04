import networkx as nx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import graph_utils as utils
from sklearn.model_selection import train_test_split
import numpy as np


def create_edge_embeddings(edges, node_embeddings, node_to_index):
    edge_embeddings = []
    for edge in edges:
        # performing hadamard product
        node1_index = node_to_index.get(edge[0])
        node2_index = node_to_index.get(edge[1])
        edge_embedding = np.multiply(node_embeddings[node1_index], node_embeddings[node2_index])
        edge_embeddings.append(edge_embedding)
    return edge_embeddings


def generate_samples(G, node_embeddings):
    positive_edges = list(G.edges())
    postive_labels = [1] * len(positive_edges)
    non_edges = list(nx.non_edges(G))
    np.random.shuffle(non_edges)
    non_edges = non_edges[:min(len(positive_edges), len(non_edges))]
    negative_labels = [0] * len(non_edges)
    edges = positive_edges + non_edges
    labels = postive_labels + negative_labels

    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}

    return create_edge_embeddings(edges, node_embeddings, node_to_index), labels


def main():
    file_path = "../datasets/cora/cora_cites.csv"

    G = utils.read_graph(file_path)

    node_embeddings = np.load("node_embeddings_simple_64.npy")

    edge_embeddings, labels = generate_samples(G, node_embeddings)

    X_train, X_test, y_train, y_test = train_test_split(edge_embeddings, labels, test_size=0.05, random_state=4113175)

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Support Vector Machine': SVC(),
        'k-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        # 'Gradient Boosting': GradientBoostingClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier()
    }

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name} Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    main()
