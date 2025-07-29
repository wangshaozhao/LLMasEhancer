import os
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load(cfg):
    """加载Cora数据集的图结构和文本数据"""
    # 创建数据集目录
    dataset_dir = os.path.join(cfg.DATA_DIR, "cora")
    os.makedirs(dataset_dir, exist_ok=True)

    # 加载图结构数据（自动下载）
    dataset = Planetoid(
        root=dataset_dir,
        name="Cora",
        transform=NormalizeFeatures()
    )
    data = dataset[0]

    # 加载文本数据（需要手动准备）
    texts = load_text_data(cfg)

    return {
        "data": data,
        "texts": texts,
        "num_classes": dataset.num_classes,
        "num_features": dataset.num_features
    }


def load_text_data(cfg):
    """加载Cora的文本数据（标题和摘要）"""
    text_dir = os.path.join(cfg.DATA_DIR, "cora_orig", "mccallum", "cora", "extractions")
    papers_path = os.path.join(cfg.DATA_DIR, "cora_orig", "mccallum", "cora", "papers")

    # 如果文本数据不存在，提示用户下载
    if not os.path.exists(text_dir) or not os.path.exists(papers_path):
        print("=" * 50)
        print("Cora文本数据未找到，请按照以下步骤准备：")
        print("1. 下载Cora文本数据：https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz")
        print(f"2. 解压到 {os.path.join(cfg.DATA_DIR, 'cora_orig')} 目录")
        print("3. 确保目录结构如下：")
        print(f"   {os.path.join(cfg.DATA_DIR, 'cora_orig', 'mccallum', 'cora', 'extractions')}")
        print(f"   {os.path.join(cfg.DATA_DIR, 'cora_orig', 'mccallum', 'cora', 'papers')}")
        print("=" * 50)
        # 返回空文本数据，仅用于演示
        return [""] * 2708  # Cora数据集有2708个节点

    # 解析论文ID与文件名的映射
    pid_to_filename = {}
    with open(papers_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pid, filename = parts
                pid_to_filename[pid] = filename

    # 读取文本内容
    texts = []
    for i in range(2708):  # Cora固定节点数
        pid = f"paper:{i}"
        if pid in pid_to_filename:
            filename = pid_to_filename[pid]
            file_path = os.path.join(text_dir, filename)
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                    texts.append(text)
            except:
                texts.append("")
        else:
            texts.append("")

    return texts
