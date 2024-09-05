from graph_neural_nets.gnn import GNN
from graph_neural_nets.graph_dataset import GraphDataset
from graph_utils import read_graph_from_jsonl, get_device
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm

BATCH_SIZE = 128
LEARNING_RATE = 0.01
NUM_EPOCHS = 10

HIDDEN_PARAM = 128
IN_CHANNEL = 1
# out channels will be 1 for regression
OUT_CHANNEL = 1


def prepare_data(path, device):
    data = read_graph_from_jsonl(path, device)
    dataset = GraphDataset(data)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        out = out.squeeze(-1)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        out = out.squeeze(-1)
        loss = criterion(out, batch.y)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = get_device()
    model = GNN(IN_CHANNEL, HIDDEN_PARAM, OUT_CHANNEL).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    train_loader = prepare_data("../datasets/zinc/train.jsonl", device)
    val_loader = prepare_data("../datasets/zinc/val.jsonl", device)
    test_loader = prepare_data("../datasets/zinc/test.jsonl", device)

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, '
              f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    test_loss = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')


if __name__ == "__main__":
    main()
