import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool


class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, model_type='gcn', num_layers=2):
        super(GNNModel, self).__init__()

        self.model_type = model_type
        self.num_layers = num_layers

        # 根据模型类型选择不同的卷积层
        if model_type == 'gcn':
            conv_layer = GCNConv
        elif model_type == 'gat':
            conv_layer = GATConv
        elif model_type == 'sage':
            conv_layer = SAGEConv
        else:
            raise ValueError(f"不支持的模型类型: {model_type}，可选类型: 'gcn', 'gat', 'sage'")

        # 构建卷积层
        self.convs = torch.nn.ModuleList()
        self.convs.append(conv_layer(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))

        # 输出层
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

        #  dropout层用于防止过拟合
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        """前向传播"""
        # 逐层进行图卷积
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            if i < self.num_layers - 1:  # 最后一层不使用dropout
                x = self.dropout(x)

        # 最终分类
        x = self.fc(x)
        return x
