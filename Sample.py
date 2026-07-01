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
        self.clusters = None  
        self.cluster_reps = None  
        self.all_indices = None 
        self.selected_mask = None  

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        # X_scaled = self.scaler.fit_transform(X)  
        self.all_indices = np.arange(X.shape[0])
        self.selected_mask = np.zeros(X.shape[0], dtype=bool)  
        self.X = X
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=1000,
            max_iter=30,
            n_init=1
        )
        labels = kmeans.fit_predict(X)
        self.cluster_centers = kmeans.cluster_centers_
        self.clusters = []
        self.cluster_labels = labels  
        for k in range(self.n_clusters):
            mask = np.where(labels == k)[0]
            self.clusters.append(mask)
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
            return
        sizes = [len(c) for c in self.clusters]

    def get_selected_ratio(self):
        return np.sum(self.selected_mask) / len(self.selected_mask)

if __name__ == "__main__":
    import numpy as np
    import time
    start_time = time.time()
    filename = 'MNIST.txt' 
    X = np.loadtxt(filename, delimiter=',')
    fi_start = time.time()
    fi = FeatureImportancePCA(n_components=10, features_method="pca").fit(X)
    df = fi.get_importance_df([f"F{i}" for i in range(X.shape[1])])
    sh_start = time.time()
    sh = SampleHierarchy(n_clusters=10, selection_strategy="random")
    sh.fit(X)
    sh.summary()
    labels = sh.get_labels(len(X))
    print(f"\n总运行时间: {time.time() - start_time:.2f} 秒")
