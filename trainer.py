import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Trainer:
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 记录训练过程
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train_epoch(self, data):
        """训练一个epoch"""
        self.model.train()
        self.optimizer.zero_grad()

        # 移动数据到设备
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.to(self.device)
        mask = data.train_mask.to(self.device)

        # 前向传播
        out = self.model(x, edge_index)
        loss = F.cross_entropy(out[mask], y[mask])

        # 反向传播和优化
        loss.backward()
        self.optimizer.step()

        # 计算准确率
        pred = out[mask].argmax(dim=1)
        acc = int((pred == y[mask]).sum()) / int(mask.sum())

        return loss.item(), acc

    def evaluate(self, data, mask):
        """在指定数据集上评估模型"""
        self.model.eval()

        # 移动数据到设备
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            out = self.model(x, edge_index)
            loss = F.cross_entropy(out[mask], y[mask]).item()
            pred = out[mask].argmax(dim=1)
            acc = int((pred == y[mask]).sum()) / int(mask.sum())

        return loss, acc, pred.cpu().numpy(), y[mask].cpu().numpy()

    def train(self, data, epochs=100, patience=10):
        """训练模型，使用早停策略防止过拟合"""
        best_val_acc = 0.0
        best_epoch = 0
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(data)

            # 验证
            val_loss, val_acc, _, _ = self.evaluate(data, data.val_mask)

            # 记录指标
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # 打印信息
            if epoch % 10 == 0:
                print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve_epochs = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"早停在第 {epoch} 轮，最佳验证准确率在第 {best_epoch} 轮: {best_val_acc:.4f}")
                    break

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        return best_val_acc

    def test(self, data):
        """在测试集上评估模型"""
        test_loss, test_acc, pred, true = self.evaluate(data, data.test_mask)

        # 打印详细分类报告
        print("\n分类报告:")
        print(classification_report(true, pred, zero_division=1))

        return test_acc, pred, true

    def plot_metrics(self):
        """绘制训练过程中的损失和准确率曲线"""
        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练与验证损失')
        plt.legend()

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='训练准确率')
        plt.plot(self.val_accs, label='验证准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.title('训练与验证准确率')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        print("训练指标图表已保存为 training_metrics.png")
        plt.show()
