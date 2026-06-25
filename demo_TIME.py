# -*- coding: utf-8 -*-
# ===== GPU配置 - 放在文件最开头 =====
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("=" * 50)
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"使用GPU: {torch.cuda.get_device_name()}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 50)
# ===== GPU配置结束 =====
import numpy as np
import time
import datetime
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from Sample import FeatureImportancePCA, SampleHierarchy
from RL_TIME import RLClustering

# 屏蔽 sklearn 的 UserWarning 和 FutureWarning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


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


def clustering_accuracy(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-w)
    return w[row_ind, col_ind].sum() / y_pred.size


def get_dataset(dataset_name):
    """统一数据加载工厂"""
    if dataset_name == 'CIFAR-10 (ResNet)':
        import numpy as np
        import scipy.io
        data_file = './cifar10_resnet_features.mat'
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"找不到 {data_file}，请先运行 extract_features.py！")
        mat = scipy.io.loadmat(data_file)
        return mat['data'].astype(np.float32), mat['labels'].flatten().astype(np.int64)
    elif dataset_name == 'CIFAR-10 (Raw)':
        from tensorflow.keras.datasets import cifar10
        import numpy as np
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0).flatten()
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        return X, y
    elif dataset_name == 'STAR_1M':
        df = pd.read_csv('/home/haojing/data/spectra/star_1M_normalized.csv', header=None)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        return X, y
    elif dataset_name == '20NEWS':
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.datasets import load_files
        import numpy as np
        print("正在加载 20NEWS 文本数据集...")
        train_data = load_files('./20news-bydate-train', encoding='latin1')
        test_data = load_files('./20news-bydate-test', encoding='latin1')
        X_text = list(train_data.data) + list(test_data.data)
        y = np.concatenate([train_data.target, test_data.target])
        print(f"总样本数: {len(X_text)}")
        print(f"类别数: {len(train_data.target_names)}")
        print("正在提取 TF-IDF 特征（保留所有词）...")
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        X = vectorizer.fit_transform(X_text).toarray().astype(np.float32)
        print(f"20NEWS 数据集加载完成!")
        print(f"样本数: {X.shape[0]}")
        print(f"特征数: {X.shape[1]}")
        return X, y
    elif dataset_name == 'MNIST':
        import numpy as np
        filename = 'MNIST_full.txt'
        data = np.loadtxt(filename, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    elif dataset_name == 'REUTERS':
        import numpy as np
        from sklearn.datasets import load_svmlight_file
        train_file = './rcv1_topics_train.svm'
        X, y_list = load_svmlight_file(train_file, multilabel=True)
        X = X.toarray().astype(np.float32)
        all_labels = set()
        for labels in y_list:
            all_labels.update(labels)
        sorted_labels = sorted(all_labels)
        n_classes = len(all_labels)
        label_map = {label: idx for idx, label in enumerate(sorted_labels)}
        y = np.zeros(len(y_list), dtype=np.int64)
        for i, labels in enumerate(y_list):
            if len(labels) > 0:
                y[i] = label_map[labels[0]]
            else:
                y[i] = 0
        print(f"数据集加载完成！")
        print(f"样本数: {X.shape[0]}")
        print(f"特征数: {X.shape[1]}")
        return X, y
    elif dataset_name == 'STL-10':
        import torchvision
        import torchvision.transforms as transforms
        print("正在下载/加载 STL-10 图像数据集 (原始像素)...")
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
        testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
        X_list, y_list = [], []
        for ds in [trainset, testset]:
            for img, label in ds:
                X_list.append(img.numpy().flatten())
                y_list.append(label)
        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list).astype(np.int64)
        return X, y
    elif dataset_name == 'MiniBooNE':
        import pandas as pd
        import numpy as np
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
        import numpy as np
        print("正在加载 forest 数据集...")
        df = pd.read_csv('forest.csv')
        print(f"原始数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()[:5]}...")
        last_col = df.columns[-1]
        print(f"最后一列: {last_col}")
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        print(f"forest 数据集加载完成!")
        print(f"样本数: {X.shape[0]}")
        print(f"特征数: {X.shape[1]}")
        print(f"类别数: {len(np.unique(y))}")
        return X, y
    elif dataset_name == 'kdd_cup99_10_percent':
        import pandas as pd
        import numpy as np
        print("正在加载 kdd_cup99_10_percent 数据集...")
        df = pd.read_csv('kdd_cup99_10_percent.csv')
        print(f"原始数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()[:5]}...")
        last_col = df.columns[-1]
        print(f"最后一列: {last_col}")
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        print(f"forest 数据集加载完成!")
        print(f"样本数: {X.shape[0]}")
        print(f"特征数: {X.shape[1]}")
        print(f"类别数: {len(np.unique(y))}")
        return X, y
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")



if __name__ == "__main__":
    start_time = time.time()
    # ======================== 实验参数配置 ========================
    # 可选: 'CIFAR-10 (ResNet)128', 'MiniBooNE(16)' 'MNIST 64' 'CIFAR-10 (Raw)'
    # 'kdd_cup99_10_percent''forest''STAR_1M'
    #  'meanshift ' 'hierarchical'  'birch' 'kmeans' 'MiniBatchKMeans' 'idec' 'cdc'

    RL_METHOD = 'meanshift'
    DATASET_NAME = 'MNIST'
    PCA_COMPONENTS = 15
    SH_CLUSTERS = 100
    MAX_EPISODES = 1
    BATCH_SIZE = 8
    TARGET_NMI = 0.0455
    LOG_FILE = "experiment_results.log"
    # ==============================================================
    # 1. 加载数据
    X, y = get_dataset(DATASET_NAME)

    if DATASET_NAME in ['20NEWS', 'REUTERS']:
        from sklearn.preprocessing import Normalizer
        X_scaled = Normalizer(norm='l2').fit_transform(X)
        print("已对文本数据应用 L2 归一化！")
    elif DATASET_NAME == 'STAR_1M':
        X_scaled = X
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # 2. 特征重要性计算
    target_reached_time = -1.0
    algo_init_start = time.time()
    n_comp = min(PCA_COMPONENTS, X_scaled.shape[1] - 1)
    fi = FeatureImportancePCA(n_components=n_comp).fit(X_scaled)
    df = fi.get_importance_df([f"F{i}" for i in range(X_scaled.shape[1])])

    # 3. 层次聚类初始化
    sh_start = time.time()
    n_sh_clusters = min(SH_CLUSTERS, len(X_scaled) // 10)
    sh = SampleHierarchy(n_clusters=n_sh_clusters, random_state=42).fit(X_scaled)


    # 4. 运行RL聚类
    rl = RLClustering(
        X=X_scaled, y_true=y, feature_importance=df,
        sample_hierarchy=sh, method=RL_METHOD
    )
    init_algo_time = time.time() - algo_init_start

    def check_progress(episode, step, rl_instance, rl_algo_time):
        global target_reached_time
        if target_reached_time > 0 or rl_instance.current_centers is None or len(rl_instance.current_centers) < 2:
            return
        Z_all = X_scaled[:, rl_instance.selected_features]
        if RL_METHOD == 'idec' and hasattr(rl_instance, 'idec'):
            with torch.no_grad():
                Z_all = rl_instance.idec.autoencoder.encode(torch.FloatTensor(Z_all).to(rl_instance.idec.device)).cpu().numpy()
        elif RL_METHOD == 'cdc' and hasattr(rl_instance, 'cdc_model'):
            with torch.no_grad():
                Z_all = rl_instance.cdc_model.encode(torch.FloatTensor(Z_all).to(rl_instance.cdc_model.device)).cpu().numpy()
        from sklearn.metrics.pairwise import pairwise_distances
        y_pred_fast = np.argmin(pairwise_distances(Z_all, rl_instance.current_centers), axis=1)
        current_nmi = normalized_mutual_info_score(y, y_pred_fast)
        if current_nmi >= TARGET_NMI:
            target_reached_time = init_algo_time + rl_algo_time
            print(f"达成目标! NMI >= {TARGET_NMI}。纯算法耗时: {target_reached_time:.2f} 秒")
    history = rl.train(max_episodes=MAX_EPISODES, batch_size=BATCH_SIZE, callback=check_progress)

    # 5. 最终结果评估
    final_samples = rl.selected_samples
    final_features = rl.selected_features
    final_labels = rl.current_labels
    final_centers = rl.current_centers

    print(f"\n最终选择的样本数: {len(final_samples) if final_samples is not None else 0}")
    print(f"最终选择的特征数: {len(final_features) if final_features is not None else 0}")

    all_labels = -np.ones(len(X_scaled), dtype=int)
    all_labels[final_samples] = final_labels
    remaining_samples = np.where(all_labels == -1)[0]

    acc, nmi_score, ari = 0.0, 0.0, 0.0

    if len(remaining_samples) > 0 and rl.current_centers is not None:
        from sklearn.metrics import pairwise_distances
        from sklearn.cluster import KMeans, AgglomerativeClustering
        true_k = len(np.unique(y))

        if RL_METHOD == 'idec':
            if hasattr(rl, 'best_idec') and rl.best_idec is not None:
                idec = rl.best_idec
                print(f"\n使用最优IDEC模型处理剩余样本 (最佳NMI={rl.global_best_nmi:.4f})")
            else:
                idec = rl.idec
            X_for_idec = X_scaled[remaining_samples][:, rl.selected_features]
            X_remain_tensor = torch.FloatTensor(X_for_idec).to(idec.device)
            Z_remain = idec.autoencoder.encode(X_remain_tensor).detach().cpu().numpy()
            dists = pairwise_distances(Z_remain, rl.current_centers)
            X_remain = X_for_idec
        elif RL_METHOD == 'cdc':
            if hasattr(rl, 'best_cdc') and rl.best_cdc is not None:
                idec = rl.best_cdc
                print(f"\n使用最优IDEC模型处理剩余样本 (最佳NMI={rl.global_best_nmi:.4f})")
            else:
                idec = rl.cdc_model
            X_for_cdc = X_scaled[remaining_samples][:, rl.selected_features]
            X_remain_tensor = torch.FloatTensor(X_for_cdc).to(cdc.device)
            Z_remain = cdc.encode(X_remain_tensor).detach().cpu().numpy()
            dists = pairwise_distances(Z_remain, rl.current_centers)
            X_remain = X_for_cdc
        else:
            X_remain = X_scaled[remaining_samples][:, final_features]
            dists = pairwise_distances(X_remain, final_centers)
        min_dists = np.min(dists, axis=1)
        nearest = np.argmin(dists, axis=1)

        if len(final_centers) > 1:
            center_dists = pairwise_distances(final_centers)
            np.fill_diagonal(center_dists, np.inf)
            threshold = np.percentile(center_dists[center_dists < np.inf], 75)
            print(f"簇间距离中位数阈值: {threshold:.4f}")
        else:
            threshold = np.percentile(min_dists, 75)
            print(f"单簇阈值（50分位数）: {threshold:.4f}")

        new_cluster_centers = []
        predicted_labels = np.zeros(len(X_remain), dtype=int)
        far_mask = min_dists > threshold
        close_mask = ~far_mask
        predicted_labels[close_mask] = nearest[close_mask]
        print(f"近距离样本: {np.sum(close_mask)} 个")
        print(f"远距离样本: {np.sum(far_mask)} 个")

        if np.sum(far_mask) > 0:
            X_far = X_remain[far_mask]
            n_new = max(1, min(5, len(X_far) // 6000))
            if RL_METHOD == 'idec':
                with torch.no_grad():
                    X_far_tensor = torch.FloatTensor(X_far).to(idec.device)
                    Z_far = idec.autoencoder.encode(X_far_tensor).cpu().numpy()
            elif RL_METHOD == 'cdc':
                with torch.no_grad():
                    X_far_tensor = torch.FloatTensor(X_far).to(rl.cdc.device)
                    Z_far = cdc.encode(X_far_tensor).cpu().numpy()
            else:
                Z_far = X_far
            kmeans_new = KMeans(n_clusters=n_new, random_state=42, n_init=5)
            far_labels = kmeans_new.fit_predict(Z_far)
            base_id = len(final_centers)
            for i, label in enumerate(far_labels):
                predicted_labels[far_mask][i] = base_id + label
            for label in range(n_new):
                mask = far_labels == label
                if np.sum(mask) > 0:
                    center = Z_far[mask].mean(axis=0)
                    new_cluster_centers.append(center)
            print(f"  成功创建 {len(new_cluster_centers)} 个新簇")

        all_centers = np.vstack([final_centers, new_cluster_centers]) if new_cluster_centers else final_centers
        n_total_clusters = len(all_centers)
        true_k = len(np.unique(y))
        print(f"\n当前总簇数: {n_total_clusters}, 真实簇数: {true_k}")

        y_pred_full = np.zeros(len(X_scaled), dtype=int)

        if n_total_clusters > true_k:
            print(f"簇数过多 ({n_total_clusters} > {true_k})，开始贪心合并最近的簇...")
            from sklearn.metrics.pairwise import pairwise_distances
            centers = all_centers.copy()
            n_merge = n_total_clusters - true_k
            print(f"  需要合并 {n_merge} 次，每次合并最近的两个簇")
            for step in range(n_merge):
                dists = pairwise_distances(centers)
                np.fill_diagonal(dists, np.inf)
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                print(f"  步骤{step + 1}: 合并簇{i}和簇{j} (距离={dists[i, j]:.4f})")
                centers[i] = (centers[i] + centers[j]) / 2
                centers = np.delete(centers, j, axis=0)
            print("  用合并后的中心重新分配样本...")
            if RL_METHOD == 'idec':
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled[:, final_features]).to(idec.device)
                    Z = idec.autoencoder.encode(X_tensor).cpu().numpy()
            elif RL_METHOD == 'cdc':
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled[:, final_features]).to(cdc.device)
                    Z = cdc.encode(X_tensor).cpu().numpy()
            else:
                Z = X_scaled[:, final_features]
            dists_to_new_centers = pairwise_distances(Z, centers)
            y_pred_full = np.argmin(dists_to_new_centers, axis=1)
            print(f"合并后簇数: {len(np.unique(y_pred_full))}")
        else:
            y_pred_full = np.zeros(len(X_scaled), dtype=int)
            for i, idx in enumerate(final_samples):
                y_pred_full[idx] = final_labels[i]
            for i, idx in enumerate(remaining_samples):
                y_pred_full[idx] = predicted_labels[i]

        valid_mask = y_pred_full != -1
        y_true_valid = y[valid_mask]
        y_pred_valid = y_pred_full[valid_mask]
        acc = clustering_accuracy(y_true_valid, y_pred_valid)
        nmi_score = normalized_mutual_info_score(y_true_valid, y_pred_valid)
        ari = adjusted_rand_score(y_true_valid, y_pred_valid)
        print(f"有效样本数: {len(y_true_valid)} / {len(y)}")
        print(f"离群点数: {np.sum(y_pred_full == -1)}")
        print(f"\n最终结果:")
        print(f"  Corrected NMI: {nmi_score:.4f}")
        print(f"  准确率 (Accuracy): {acc:.4f}")
        print(f"  调整兰德指数 (ARI): {ari:.4f}")
        print(f"  最终簇数: {len(np.unique(y_pred_full))}")
    if target_reached_time > 0:
        print(f"达到目标 NMI {TARGET_NMI} 的确切算法时间: {target_reached_time:.2f} 秒")
    else:
        print(f"训练结束，未能达到目标 NMI {TARGET_NMI}")
    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time:.2f} 秒")

    # 6. 记录日志
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"=== 实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"数据集: {DATASET_NAME} | 样本数: {X_scaled.shape[0]} | 特征数: {X_scaled.shape[1]}\n")
        f.write(f"[参数设置]\n  - PCA 降维组件数: {n_comp}\n  - 预聚类簇数: {n_sh_clusters}\n")
        f.write(f"  - 方法: {RL_METHOD} | Episodes: {MAX_EPISODES} | Batch: {BATCH_SIZE}\n")
        f.write(f"[最终状态]\n  - 选定样本数: {len(final_samples)} | 选定特征数: {len(final_features)}\n")
        f.write(f"[评估指标]\n  - NMI: {nmi_score:.4f} | ACC: {acc:.4f} | ARI: {ari:.4f}\n")
        f.write(f"  - 总耗时: {total_time:.2f} 秒\n" + "=" * 60 + "\n\n")