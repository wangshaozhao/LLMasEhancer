import os
import torch
from data_processor import DataProcessor
from model import GNNModel
from trainer import Trainer


def main():
    # 设置随机种子，保证结果可复现
    torch.manual_seed(42)

    # 1. 数据处理
    print("===== 开始数据处理 =====")
    processor = DataProcessor()
    processor.load_data()

    # 根据实际CSV列名调整参数
    graph_data = processor.preprocess_data(
        text_cols=['title', 'abstract'],  # 文本特征列
        categorical_cols=['category'],  # 分类特征列
        label_col='label',  # 标签列
        id_col='node_id',  # 节点ID列
        arxiv_id_col='arxiv_id'  # arxiv唯一标识符列
    )

    # 分割数据
    graph_data = processor.split_data(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )

    # 2. 创建模型
    print("\n===== 初始化模型 =====")
    input_dim = graph_data.x.shape[1]  # 输入特征维度
    output_dim = len(processor.label_encoder.classes_)  # 输出类别数

    # 可以尝试不同的模型类型：'gcn', 'gat', 'sage'
    model = GNNModel(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=output_dim,
        model_type='gcn',
        num_layers=2
    )

    print(f"模型参数 - 输入维度: {input_dim}, 隐藏维度: 128, 输出维度: {output_dim}, 模型类型: gcn")

    # 3. 训练模型
    print("\n===== 开始训练模型 =====")
    trainer = Trainer(model)
    best_val_acc = trainer.train(graph_data, epochs=200, patience=20)
    print(f"最佳验证集准确率: {best_val_acc:.4f}")

    # 4. 测试模型
    print("\n===== 测试模型性能 =====")
    test_acc, _, _ = trainer.test(graph_data)
    print(f"最终测试集准确率: {test_acc:.4f}")



if __name__ == "__main__":
    main()
