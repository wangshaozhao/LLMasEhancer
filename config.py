import os
from yacs.config import CfgNode as CN
import torch

# 基础配置
_C = CN()
_C.DATASET = "cora"  # 数据集名称: cora, pubmed, arxiv, products, wisconsin
_C.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_C.SEED = 42
_C.DATA_DIR = "./dataset"  # 数据集存储根目录
_C.OUTPUT_DIR = "./outputs"  # 输出结果目录

# GNN配置
_C.GNN = CN()
_C.GNN.MODEL = "gcn"  # gcn, sage, mlp
_C.GNN.HIDDEN_DIM = 128
_C.GNN.NUM_LAYERS = 2
_C.GNN.DROPOUT = 0.5
_C.GNN.LEARNING_RATE = 0.01
_C.GNN.WEIGHT_DECAY = 5e-4
_C.GNN.EPOCHS = 200

# LLM配置
_C.LLM = CN()
_C.LLM.MODEL_NAME = "bert-base-uncased"  # 基础语言模型
_C.LLM.LLM_MODEL = "vicuna-7b"  # 大型语言模型
_C.LLM.BATCH_SIZE = 32
_C.LLM.LEARNING_RATE = 2e-5

# 创建输出目录
os.makedirs(_C.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(_C.DATA_DIR), exist_ok=True)

def get_cfg_defaults():
    return _C.clone()
