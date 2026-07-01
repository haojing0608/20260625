import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import lil_matrix
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

# ==================== 1. 特征重要性（PCA） ====================
class FeatureImportancePCA:

    def __init__(self, n_components=None, method="pca"):
        self.n_components = n_components
        self.method = method.lower()
        self.scaler = StandardScaler()
        if self.method == "pca":
            self.pca = PCA(n_components=self.n_components)
        self.X_pca = None
        self.importance_ = None

    def fit(self, X):
        if X.shape[0] > 600000:
            np.random.seed(42)
            X_fit = X[np.random.choice(X.shape[0], 50000, replace=False)]
        else:
            X_fit = X
        if self.method == "pca":
            self.pca.fit(X_fit)
            self.X_pca = self.pca.transform(X)
            self.importance_ = np.sum((self.pca.components_ ** 2) * self.pca.explained_variance_ratio_[:, np.newaxis],axis=0)
        elif self.method == "variance":
            self.importance_ = np.var(X_fit, axis=0)
            self.X_pca = X
        elif self.method == "mad":
            # Mean Absolute Deviation
            mean = np.mean(X_fit, axis=0)
            self.importance_ = np.mean(np.abs(X_fit - mean), axis=0)
            self.X_pca = X
        else:
            raise ValueError(f"Unknown importance method: {self.method}")
        return self

    def get_importance_df(self, feature_names=None):
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(len(self.importance_))]
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.importance_,
            'Norm': self.importance_ / self.importance_.sum()
        })
        return df.sort_values('Importance', ascending=False)

class SampleHierarchy:
    def __init__(self, n_clusters=100, random_state=42, selection_strategy="random"):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.selection_strategy = selection_strategy
        self.scaler = StandardScaler()
        self.clusters = None  # 保存每个簇的样本
        self.cluster_reps = None  # 每个簇的代表点
        self.all_indices = None  # 所有样本的索引
        self.selected_mask = None  # 记录每个样本是否被选过

    def fit(self, X):
        """
        只做一次聚类，分成 n_clusters 个簇
        """

        X = np.asarray(X, dtype=np.float32)
        # X_scaled = self.scaler.fit_transform(X)  # 归一化
        self.all_indices = np.arange(X.shape[0])
        self.selected_mask = np.zeros(X.shape[0], dtype=bool)  # 初始都没选
        self.X = X
        # self.X_scaled = X_scaled
        print(f"聚类: {X.shape[0]} 样本 -> {self.n_clusters} 个簇")

        # 聚类
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=1000,
            max_iter=30,
            n_init=1
        )
        labels = kmeans.fit_predict(X)
        self.cluster_centers = kmeans.cluster_centers_
        # 保存每个簇的样本
        self.clusters = []
        self.cluster_labels = labels  # 保存每个样本的簇标签

        for k in range(self.n_clusters):
            mask = np.where(labels == k)[0]
            self.clusters.append(mask)

        print(f"  簇大小: 最小={min(len(c) for c in self.clusters)}, "
              f"最大={max(len(c) for c in self.clusters)}, "
              f"平均={np.mean([len(c) for c in self.clusters]):.1f}")
        return self


    def select_samples_from_clusters(self, ratio_per_cluster=0.05):
        new_samples = []
        for k, cluster in enumerate(self.clusters):
            unselected_in_cluster = [idx for idx in cluster if not self.selected_mask[idx]]
            if len(unselected_in_cluster) == 0:
                continue

            cluster_center = self.cluster_centers[k]
            X_unselected = self.X[unselected_in_cluster]
            distances = np.linalg.norm(X_unselected - cluster_center, axis=1)
            n_to_select = max(int(len(unselected_in_cluster) * ratio_per_cluster),1)
            n_to_select = min(n_to_select,len(unselected_in_cluster))
            if self.selection_strategy == "random":
                selected = np.random.choice(unselected_in_cluster, n_to_select, replace=False)
            elif self.selection_strategy == "closest":
                sorted_indices = np.argsort(distances)
                selected = [unselected_in_cluster[i] for i in sorted_indices[:n_to_select]]

            elif self.selection_strategy == "farthest":
                sorted_indices = np.argsort(-distances)
                selected = [unselected_in_cluster[i] for i in sorted_indices[:n_to_select]]

            else:
                raise ValueError("Unknown selection strategy.")
            new_samples.extend(selected)
            for idx in selected:
                self.selected_mask[idx] = True
        return np.array(new_samples)

    def summary(self):
        if self.clusters is None:
            # print("还没有聚类，请先运行 fit()")
            return
        # 簇大小统计
        sizes = [len(c) for c in self.clusters]
        #  print(f"簇大小: 最小={min(sizes)}, 最大={max(sizes)}, 平均={np.mean(sizes):.1f}")

    def get_selected_ratio(self):
        return np.sum(self.selected_mask) / len(self.selected_mask)

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 从txt文件加载数据（逗号分隔）
    import numpy as np
    import time
    # 记录开始时间
    start_time = time.time()

    # 读取数据
    filename = 'MNIST.txt'  # 请替换为实际文件名
    X = np.loadtxt(filename, delimiter=',')

    # print(f"数据集形状: {X.shape}")
    # print(f"样本数: {X.shape[0]}")
    # print(f"特征数: {X.shape[1]}")

    # print("\n特征重要性计算")
    # 计算特征重要性
    fi_start = time.time()
    fi = FeatureImportancePCA(n_components=10, features_method="pca").fit(X)
    df = fi.get_importance_df([f"F{i}" for i in range(X.shape[1])])
    # print(df.head(50))
    # print(f"特征重要性计算时间: {time.time() - fi_start:.2f} 秒")

    # print("\n样本层次结构构建")
    # 层次聚类
    sh_start = time.time()
    sh = SampleHierarchy(n_clusters=10, selection_strategy="random")

    sh.fit(X)
    sh.summary()
    # print(f"层次聚类时间: {time.time() - sh_start:.2f} 秒")

    # 获取聚类标签
    labels = sh.get_labels(len(X))
    # print(f"\n最终簇数: {len(np.unique(labels))}")
  # 总运行时间
    print(f"\n总运行时间: {time.time() - start_time:.2f} 秒")