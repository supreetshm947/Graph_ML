from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import graph_utils as utils

file_path = "../datasets/cora/cora_cites.csv"

G = utils.read_graph(file_path)

node_embeddings = np.load("node_embeddings_simple_64.npy")

# Normalizing embeddings
scaler = StandardScaler()
node_embeddings = scaler.fit_transform(node_embeddings)

content = pd.read_csv("../datasets/cora/cora.content", sep="\t", skiprows=0, header=None)
content = content.set_index(content.columns[0]).loc[list(G.nodes())].reset_index()

labels = content[content.columns[-1]]

le = LabelEncoder()
labels = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(node_embeddings, labels, test_size=0.05, random_state=4113175)

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