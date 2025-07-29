from config import get_cfg_defaults
import importlib


def load_data(cfg):
    """根据数据集名称动态加载对应的数据加载函数"""
    dataset_name = cfg.DATASET.lower()

    # 映射数据集到对应的加载模块
    dataset_module_map = {
        "cora": "load_cora",
        "pubmed": "load_pubmed",
        "arxiv": "load_arxiv",
        "products": "load_products",
        "wisconsin": "load_webkb",
        "cornell": "load_webkb"
    }

    if dataset_name not in dataset_module_map:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 动态导入对应的加载模块
    module = importlib.import_module(f"data_utils.{dataset_module_map[dataset_name]}")

    # 调用加载函数
    return module.load(cfg)
