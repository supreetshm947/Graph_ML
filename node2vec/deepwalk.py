import graph_utils as utils
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Hyper-params
num_walks = 10
walk_length = 20
embed_size = 128
window_size = 5
batch_size = 512
hidden_size = 64


class SkipGramDataset(Dataset):
    def __init__(self, G):
        walks = generate_random_walks(G, num_walks=num_walks, walk_length=walk_length)
        data = generate_skipgram_data(walks, window_size=window_size)
        vocab = {node: i for i, node in enumerate(G.nodes())}
        data = [(vocab[target], vocab[context]) for target, context in data]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_random_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                neighbors = list(G.neighbors(node))
                if neighbors:
                    walk.append(random.choice(neighbors))
                else:
                    break
            walks.append(walk)
    return walks


def generate_skipgram_data(walks, window_size):
    data = []
    for walk in walks:
        for i, node in enumerate(walk):
            for j in range(max(0, i - window_size), min(i + window_size, len(walk))):
                if i != j:
                    data.append((node, walk[j]))
    return data


class Node2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Node2Vec, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.hidden_layer = torch.nn.Linear(embed_size, hidden_size)
        self.batch_norm = torch.nn.BatchNorm1d(hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, vocab_size)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, target):
        embed_out = self.embedding(target)
        hidden_out = self.hidden_layer(embed_out)
        hidden_norm = self.batch_norm(hidden_out)
        hidden_drop = self.dropout(hidden_norm)  # Applying dropout
        output = self.output_layer(hidden_drop)
        return output

        return output


class Node2Vec_simple(torch.nn.Module):
    def __init__(self, embed_size):
        super(Node2Vec_simple, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)

    def forward(self, target):
        output = self.embedding(target)

        return output


def train_deepwalk(graph, vocab_size, embed_size, hidden_size, batch_size=64, lr=0.001, epochs=5):
    model = Node2Vec(vocab_size, embed_size, hidden_size).to(device=device)
    # model = Node2Vec_simple(embed_size).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,
                                                           factor=0.5)  # Learning rate scheduler

    dataset = SkipGramDataset(graph)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for target, context in tqdm(dataloader):
            target = torch.LongTensor(target).to(device)
            context = torch.LongTensor(context).to(device)
            model.zero_grad()
            output = model(target)
            # score = torch.matmul(output, model.embedding.weight.t())
            loss = loss_fn(output, context)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        # Step the scheduler
        scheduler.step(avg_loss)
        node_embeddings = model.embedding.weight.data.cpu().numpy()
        np.save(f'fb_node_embeddings_simple_{embed_size}.npy', node_embeddings)
    return model


file_path = "../datasets/musae-facebook-pagepage-network/musae_facebook_edges.csv"

G = utils.read_graph(file_path)
vocab_size = len(G.nodes())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = train_deepwalk(G, vocab_size, embed_size, hidden_size, batch_size, epochs=50)
