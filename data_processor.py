import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import string

# 下载停用词（首次运行需要）
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english') + list(string.punctuation))


class DataProcessor:
    def __init__(self):
        self.data_path = os.path.join('dataset', 'arxiv_2023_orig', 'paper_info.csv')
        self.raw_data = None
        self.graph_data = None
        self.label_encoder = LabelEncoder()
        self.category_encoder = OneHotEncoder(sparse_output=False)
        self.w2v_model = None

    def load_data(self):
        """加载CSV数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件未找到: {self.data_path}")

        self.raw_data = pd.read_csv(self.data_path)
        print(f"成功加载数据，共 {len(self.raw_data)} 条记录")
        return self.raw_data

    def preprocess_text(self, text):
        """预处理文本：分词、去除停用词和标点"""
        if pd.isna(text):
            return []

        # 分词并转为小写
        tokens = simple_preprocess(text, deacc=True)

        # 去除停用词和短词
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        return tokens

    def train_word2vec(self, texts, vector_size=100, window=5, min_count=5, epochs=10):
        """训练Word2Vec模型"""
        print(f"训练Word2Vec模型，文本数量: {len(texts)}")

        # 预处理所有文本
        processed_texts = [self.preprocess_text(text) for text in texts]

        # 过滤空文本
        processed_texts = [text for text in processed_texts if len(text) > 0]

        # 训练Word2Vec模型
        self.w2v_model = Word2Vec(
            sentences=processed_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs
        )

        print(f"Word2Vec模型训练完成，词汇表大小: {len(self.w2v_model.wv)}")
        return self.w2v_model

    def text_to_vector(self, text):
        """将文本转换为Word2Vec向量（词向量的平均值）"""
        if self.w2v_model is None:
            raise ValueError("Word2Vec模型尚未训练，请先调用train_word2vec方法")

        tokens = self.preprocess_text(text)
        if not tokens:
            return np.zeros(self.w2v_model.vector_size)

        # 过滤不在词汇表中的词
        vectors = [self.w2v_model.wv[token] for token in tokens if token in self.w2v_model.wv]

        if not vectors:
            return np.zeros(self.w2v_model.vector_size)

        # 返回词向量的平均值
        return np.mean(vectors, axis=0)

    def build_graph(self, df):
        """构建图结构，这里使用简单的基于索引的连接，实际应用中可根据需要修改"""
        # 假设每个节点与下一个节点相连（简单的链式图，仅作示例）
        edges = []
        for i in range(len(df) - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # 无向图

        # 转换为PyTorch Geometric需要的格式
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def preprocess_data(self, text_cols=['title', 'abstract'], cat_cols=['category'],
                        label_col='label', id_col='node_id', w2v_params=None):
        """
        预处理数据：文本编码、特征整合、图构建
        """
        if self.raw_data is None:
            self.load_data()

        df = self.raw_data.copy()

        # 设置默认的Word2Vec参数
        if w2v_params is None:
            w2v_params = {
                'vector_size': 100,
                'window': 5,
                'min_count': 5,
                'epochs': 10
            }

        # 1. 处理文本特征 - 使用Word2Vec
        # 合并多个文本列
        df['combined_text'] = df[text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)

        # 训练Word2Vec模型
        self.train_word2vec(df['combined_text'], **w2v_params)

        # 将文本转换为向量
        text_vectors = df['combined_text'].apply(self.text_to_vector)
        text_features = np.array(text_vectors.tolist())
        print(f"文本特征形状: {text_features.shape}")

        # 2. 处理分类特征（如category）
        cat_features = self.category_encoder.fit_transform(df[cat_cols])
        print(f"分类特征形状: {cat_features.shape}")

        # 3. 合并所有特征
        all_features = np.hstack([text_features, cat_features])
        print(f"所有特征合并后形状: {all_features.shape}")

        # 4. 处理标签
        labels = self.label_encoder.fit_transform(df[label_col])
        print(f"标签类别数量: {len(self.label_encoder.classes_)}")

        # 5. 构建图
        edge_index = self.build_graph(df)
        print(f"图边数量: {edge_index.shape[1] // 2}")  # 除以2因为无向图每条边存储两次

        # 6. 创建PyTorch Geometric数据对象
        self.graph_data = Data(
            x=torch.tensor(all_features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long),
            node_id=torch.tensor(df[id_col].values, dtype=torch.long)
        )

        return self.graph_data

    def split_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """分割数据集为训练集、验证集和测试集"""
        if self.graph_data is None:
            raise ValueError("请先调用preprocess_data方法预处理数据")

        # 检查比例是否合法
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

        num_nodes = self.graph_data.num_nodes
        indices = np.arange(num_nodes)

        # 先分割训练集和临时集
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=1 - train_ratio,
            random_state=random_state
        )

        # 再分割验证集和测试集
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_state
        )

        # 创建掩码
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        # 添加到图数据中
        self.graph_data.train_mask = train_mask
        self.graph_data.val_mask = val_mask
        self.graph_data.test_mask = test_mask

        print(f"数据集分割完成 - 训练集: {sum(train_mask)}, 验证集: {sum(val_mask)}, 测试集: {sum(test_mask)}")
        return self.graph_data
