import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # 输入层
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))

        # 输出层
        self.convs.append(GCNConv(hidden_dim, num_classes))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 输出层
        x = self.convs[-1](x, edge_index)
        return x
