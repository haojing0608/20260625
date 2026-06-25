# clustering.py
import os
import torch
from cdc import CDC 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU 1

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)

print("=" * 50)
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"使用GPU: {torch.cuda.get_device_name()}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 50)

import time
import datetime
import os
import warnings
import numpy as np
from sklearn.cluster import MiniBatchKMeans, DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering, KMeans, \
    Birch, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment
from idec import IDEC

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from scipy.sparse import issparse

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

def clustering_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-w)
    return w[row_ind, col_ind].sum() / y_pred.size

def clean_miniboone_data(X):
    X_clean = X.copy()
    for i in range(X.shape[1]):
        feature = X[:, i]
        abnormal_mask = (feature < -900) | (feature > 1e6)
        if np.sum(abnormal_mask) > 0:
            normal_values = feature[~abnormal_mask]
            if len(normal_values) > 0:
                X_clean[abnormal_mask, i] = np.mean(normal_values)
    return X_clean

def get_dataset(dataset_name):
    import pandas as pd
    import numpy as np
    if dataset_name == 'CIFAR-10 (ResNet)':
        import scipy.io
        data_file = './cifar10_resnet_features.mat'
        mat = scipy.io.loadmat(data_file)
        return mat['data'].astype(np.float32), mat['labels'].flatten().astype(np.int64)
    elif dataset_name == 'STAR_1M':
        data = np.loadtxt('/home/haojing/data/spectra/star_1M_normalized.csv',
                          delimiter=',', dtype=np.float32)
        X = data[:, :-1]
        y = data[:, -1].astype(np.int64)
        return X, y
    elif dataset_name == 'REUTERS':
        from sklearn.datasets import load_svmlight_file
        import os
        train_file = './rcv1_topics_train.svm'
        X, y = load_svmlight_file(train_file, multilabel=True)
        X = X.toarray().astype(np.float32)
        if y.ndim > 1 or isinstance(y, np.ndarray) == False:
            y = y.toarray()[:, 0].astype(np.int64)
        return X, y

    elif dataset_name == 'MNIST':
        filename = 'MNIST_full.txt'
        data = np.loadtxt(filename, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    elif dataset_name == 'MiniBooNE':
        import pandas as pd
        filename = 'MiniBooNE.txt'
        with open(filename, 'r') as f:
            first_line = f.readline().strip().split()
            n_signal, n_background = int(first_line[0]), int(first_line[1])
            total_samples = n_signal + n_background
        df = pd.read_csv(filename, skiprows=1, header=None, sep='\s+')
        X = df.values
        y = np.array([1] * n_signal + [0] * n_background)
        X = clean_miniboone_data(X)
        return X, y
    elif dataset_name == 'forest':
        import pandas as pd
        df = pd.read_csv('forest.csv')
        last_col = df.columns[-1]
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        return X, y

    elif dataset_name == 'kdd_cup99_10_percent':
        import pandas as pd
        df = pd.read_csv('kdd_cup99_10_percent.csv')
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        nan_mask = np.isnan(X).any(axis=1)
        inf_mask = np.isinf(X).any(axis=1)
        bad_mask = nan_mask | inf_mask
        if bad_mask.any():
            X = X[~bad_mask]
            y = y[~bad_mask]
        from scipy import stats
        z_scores = np.abs(stats.zscore(X))
        outlier_mask = (z_scores > 5).any(axis=1)
        if outlier_mask.any():
            X = X[~outlier_mask]
            y = y[~outlier_mask]
        print(f"处理后样本数: {X.shape[0]}")
        return X, y
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

def run_method(method='kmeans', dataset_name="20NEWS"):
    LOG_FILE = "baseline_results.log"
    X, y = get_dataset(dataset_name)
    if y.min() == 1:
        y = y - 1
    print(f"标签已转换: {np.unique(y)}")

    # ==========归一化 ==========
    from scipy.sparse import issparse
    if dataset_name == 'REUTERS' and issparse(X):
        X_scaled = X
    elif dataset_name == 'STAR_1M':
        X_scaled = X
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        stds = np.std(X_scaled, axis=0)
        non_const_cols = np.where(stds > 1e-8)[0]
        if len(non_const_cols) < X_scaled.shape[1]:
            print(f"  移除 {X_scaled.shape[1] - len(non_const_cols)} 个常数列")
            X_scaled = X_scaled[:, non_const_cols]
            X_scaled = np.clip(X_scaled, -10, 10)
    # X_scaled = X
    true_k = len(np.unique(y))
    print(f"\n数据形状: {X_scaled.shape}, 真实簇数: {true_k}")

    print("\n" + "=" * 60)
    print(f"运行 {method.upper()}...")
    print("=" * 60)
    start_time = time.time()
    n_noise = 0

    if method == 'MiniBatchKMeans':
        km = MiniBatchKMeans(n_clusters=true_k, batch_size=50000)
        labels = km.fit_predict(X)
        n_clusters = true_k
    elif method == 'kmeans':
        km = KMeans(n_clusters=true_k, random_state=42, n_init=1)
        labels = km.fit_predict(X_scaled)
        n_clusters = true_k
    elif method == 'gmm':
        k = max(min(true_k, X_scaled.shape[0] // 10), 2)
        gmm = GaussianMixture(n_components=k, covariance_type='full', max_iter=100, n_init=3, reg_covar=1e-6)
        labels = gmm.fit_predict(X_scaled)
        n_clusters = len(np.unique(labels))
    elif method == 'hierarchical':
        agg = AgglomerativeClustering(n_clusters=true_k, linkage='ward')
        labels = agg.fit_predict(X_scaled)
        n_clusters = true_k
    elif method == 'meanshift':
        X_for_meanshift = X_scaled
        if X_scaled.shape[1] > 15:
            X_for_meanshift = pca.fit_transform(X_scaled)
        bandwidth = estimate_bandwidth(X_for_meanshift, quantile=0.2, n_samples=10000)
        if bandwidth > 8.0:
            bandwidth = 4.5
        elif bandwidth < 0.1:
            bandwidth = 2.0
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
        labels = ms.fit_predict(X_for_meanshift)
        n_clusters = len(np.unique(labels))
    elif method == 'cdc':
        k = min(true_k, X_scaled.shape[0] - 1) if X_scaled.shape[0] > true_k else true_k
        k = max(k, 2)
        cdc_model = CDC(
            input_dim=X_scaled.shape[1],
            n_clusters=k,
            latent_dim=128,
            hidden_dims=[1024, 512, 256, 128],
            # hidden_dims=[128, 64, 32],
            alpha=0.1,
            beta=0.5,
            device='cuda'
        )
        cdc_model.pretrain(
            X_scaled,
            epochs=50,
            batch_size=min(256, len(X_scaled)),
            noise_factor=0.2
        )

        cdc_model.train_epoch(
            X_scaled,
            epochs=50,
            batch_size=min(256, len(X_scaled))
        )

        labels, centers, z = cdc_model.predict(X_scaled)
        n_clusters = len(np.unique(labels))
        n_noise = 0

    elif method == 'spectral':
        k = min(true_k, X_scaled.shape[0] - 1) if X_scaled.shape[0] > true_k else true_k
        k = max(k, 2)
        spectral = SpectralClustering(
            n_clusters=k,
            affinity='rbf',
            gamma=1.0 / X_scaled.shape[1],
            n_neighbors=10,
            random_state=None,
            assign_labels='kmeans',
            n_init=10)

        labels = spectral.fit_predict(X_scaled)
        centers = []
        for i in range(k):
            mask = labels == i
            if np.sum(mask) > 0:
                centers.append(X_scaled[mask].mean(axis=0))
            else:
                centers.append(X_scaled[np.random.randint(0, len(X_scaled))])
        centers = np.array(centers)
        n_clusters = len(np.unique(labels))


    elif method == 'dbscan':
        from sklearn.decomposition import PCA
        n_components = min(50, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X_scaled)

        data_range = np.ptp(X_reduced, axis=0).mean()
        eps = data_range * 0.5
        min_samples = 100

        print(f"DBSCAN参数: eps={eps:.4f}, min_samples={min_samples}")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_reduced)

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(labels == -1)
        print(f"  结果: 簇数={n_clusters}, 噪声={n_noise}")
   
    elif method == 'idec':
        k = min(true_k, X_scaled.shape[0] - 1) if X_scaled.shape[0] > true_k else true_k
        k = max(k, 2)
        idec_model = IDEC(
            n_clusters=k,
            input_dim=X_scaled.shape[1],
            hidden_dims=[500, 500, 1000],
            # hidden_dims = [128, 64],
            latent_dim=100,
            alpha=0.5,
            device='cuda'
        )
        labels = idec_model.fit(
            X_scaled,
            epochs=100,
            batch_size=min(256, len(X_scaled)),
            verbose=True
        )

        n_clusters = len(np.unique(labels))
        n_noise = 0

    else:
        print(f"未知方法: {method}")
        return

    elapsed = time.time() - start_time
    valid_mask = labels != -1
    y_valid = y[valid_mask]
    labels_valid = labels[valid_mask]

    acc = clustering_accuracy(y_valid, labels_valid)
    nmi = normalized_mutual_info_score(y_valid, labels_valid)
    if len(y_valid) > 200000:  
        y_list = [int(x) for x in y_valid]
        labels_list = [int(x) for x in labels_valid]
        ari = adjusted_rand_score(y_list, labels_list)
    else:  
        ari = adjusted_rand_score(y_valid, labels_valid)
    print(f"\n结果: 准确率: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f} | 耗时: {elapsed:.2f} 秒")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"=== 基准实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"数据集: {dataset_name} | 样本数: {X_scaled.shape[0]} | 特征数: {X_scaled.shape[1]}\n")
        f.write(f"[参数设置]\n  - 基准聚类方法: {method}\n")
        f.write(f"[评估指标]\n  - 簇数: {n_clusters} (真实: {true_k})\n")
        f.write(f"  - NMI: {nmi:.4f} | ACC: {acc:.4f} | ARI: {ari:.4f}\n")
        f.write(f"  - 总耗时: {elapsed:.2f} 秒\n" + "=" * 60 + "\n\n")


if __name__ == "__main__":
    # 可选方法: 'kmeans', 'dbscan', 'meanshift', 'hierarchical', 'gmm', 'spectral', 'birch', 'MiniBatchKMeans''idec''cdc'
    # 可选数据集: 'CIFAR-10 (ResNet)', '20NEWS', 'REUTERS', 'STL-10','MNIST', 'MiniBooNE''kdd_cup99_10_percent''forest''STAR_1M'
    run_method(method='meanshift', dataset_name='MiniBooNE')
