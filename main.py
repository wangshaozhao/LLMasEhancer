import torch
import torch.nn.functional as F
from config import get_cfg_defaults
from data_utils.load import load_data
from GNNs.GCN.model import GCN


def train(model, data, optimizer, cfg):
    """训练GNN模型"""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    """测试模型性能"""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())
        accs.append(acc)

    return accs


def main():
    # 加载配置
    cfg = get_cfg_defaults()
    print(f"使用配置: {cfg}")

    # 加载数据
    print("加载数据集...")
    dataset = load_data(cfg)
    data = dataset["data"].to(cfg.DEVICE)
    texts = dataset["texts"]
    num_classes = dataset["num_classes"]
    num_features = dataset["num_features"]

    print(f"数据集信息: 节点数={data.num_nodes}, 边数={data.num_edges}, 类别数={num_classes}")
    print(f"{texts}")

    # 初始化模型
    model = GCN(
        num_features=num_features,
        hidden_dim=cfg.GNN.HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=cfg.GNN.NUM_LAYERS,
        dropout=cfg.GNN.DROPOUT
    ).to(cfg.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.GNN.LEARNING_RATE,
        weight_decay=cfg.GNN.WEIGHT_DECAY
    )

    # 训练模型
    print("开始训练...")
    best_val_acc = 0
    for epoch in range(1, cfg.GNN.EPOCHS + 1):
        loss = train(model, data, optimizer, cfg)
        train_acc, val_acc, test_acc = test(model, data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                  f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    print(f"最佳验证集准确率: {best_val_acc:.4f}")
    print(f"对应测试集准确率: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()
