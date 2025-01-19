from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.nn.attention import TemporalGAT
import torch.nn as nn
import torch.optim as optim

# Definir el modelo
class TGAT(nn.Module):
    def __init__(self, node_features, hidden_channels):
        super(TGAT, self).__init__()
        self.temporal_gat = TemporalGAT(
            in_channels=node_features,
            out_channels=hidden_channels,
            heads=4
        )
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.temporal_gat(x, edge_index, edge_weight)
        return self.linear(h)

def train_model(model, temporal_signal):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    # Entrenamiento
    model.train()
    for epoch in range(100):
        loss_epoch = 0
        for snapshot in temporal_signal:
            x, edge_index, edge_weight, y = snapshot.x, snapshot.edge_index, snapshot.edge_attr, snapshot.y
            optimizer.zero_grad()
            out = model(x, edge_index, edge_weight)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {loss_epoch:.4f}")
