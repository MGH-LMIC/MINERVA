import torch.nn
import torch
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import HANConv
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class HAN_encoder(torch.nn.Module):
    def __init__(self, data, in_channels, hidden_channels=128, heads=8):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.4, metadata=data.metadata())

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        return out


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_features=1):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_features)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['microbes'][row], z_dict['diseases'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z

class InnerProductDecoder(torch.nn.Module):
    def forward(self, x_dict, edge_label_index):
        x_src = x_dict['microbes'][edge_label_index[0]]
        x_dst = x_dict['diseases'][edge_label_index[1]]
        return (x_src * x_dst).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data, out_features=1, inner_product_dec=False, han_encoder=False):
        super().__init__()
        if han_encoder:
            self.encoder = HAN_encoder(data, in_channels=-1, hidden_channels=hidden_channels, heads=8)
        else:
            self.encoder = GNNEncoder(hidden_channels, hidden_channels)
            self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')

        if inner_product_dec:
            self.decoder = InnerProductDecoder()
        else:
            self.decoder = EdgeDecoder(hidden_channels, out_features=out_features)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)





