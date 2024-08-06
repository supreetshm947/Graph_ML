import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import copy
import graph_utils as utils

class NodeClassificationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NodeClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out_fc1 = self.fc1(x)
        out_act = self.relu(out_fc1)
        out_dropout = self.dropout(out_act)
        out = self.fc2(out_dropout)

        return out


def eval_classifier(classifier, X_test, y_test):
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Accuracy: {accuracy * 100:.2f}%')


def train_classifier(data, val_data, input_size, hidden_size, num_classes, batch_size=64, lr=0.001, epochs=5, patience=5):
    classifier = NodeClassifier(input_size, hidden_size, num_classes).to(device=device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataset = NodeClassificationDataset(data)
    val_dataset = NodeClassificationDataset(val_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        for X, y in tqdm(train_dataloader):
            X = torch.tensor(X, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            classifier.zero_grad()
            output = classifier(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}"):
                X = torch.tensor(X, dtype=torch.float32).to(device)
                y = torch.tensor(y, dtype=torch.long).to(device)
                output = classifier(X)
                loss = loss_fn(output, y)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        val_loss /= len(val_dataloader)
        val_accuracy = correct / total

        print(
            f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = copy.deepcopy(classifier.state_dict())  # Use deepcopy to save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                print(best_val_loss)
                classifier.load_state_dict(best_model)  # Load the best model
                break

    return classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "../datasets/cora/cora_cites.csv"

G = utils.read_graph(file_path)

node_embeddings = np.load("node_embeddings_64.npy")

# Normalizing embeddings
scaler = StandardScaler()
node_embeddings = scaler.fit_transform(node_embeddings)

content = pd.read_csv("../datasets/cora/cora.content", sep="\t", skiprows=0, header=None)
content = content.set_index(content.columns[0]).loc[list(G.nodes())].reset_index()

labels = content[content.columns[-1]]

le = LabelEncoder()
labels = le.fit_transform(labels)

input_size = node_embeddings.shape[1]
hidden_size = 128
num_classes = len(np.unique(labels))
learning_rate = 0.001
num_epochs = 20

X_train, X_test, y_train, y_test = train_test_split(node_embeddings, labels, test_size=0.2, random_state=11)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

classifier = train_classifier(data=list(zip(X_train, y_train)), val_data=list(zip(X_test, y_test)), input_size=input_size, hidden_size=hidden_size,
                              num_classes=num_classes, batch_size=64, lr=learning_rate, epochs=20, patience=10)


classifier.eval()
with torch.no_grad():
    outputs = classifier(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Final Validation Accuracy: {accuracy * 100:.2f}%')