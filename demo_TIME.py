import os
import urllib.request
import gzip
import shutil
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import torch
torch.set_num_threads(2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def download_file(url, dest, desc=None):
    if os.path.exists(dest):
        return
    if desc:
        print(f"正在下载 {desc}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"已保存到 {dest}")
    except Exception as e:
        print(f"下载失败: {e}")
        raise

def ensure_dataset(dataset_name):
    if dataset_name == 'MNIST':
        if not os.path.exists('MNIST_full.txt'):
            print("MNIST 数据集不存在，正在自动下载")
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
            X = mnist.data.astype(np.float32)
            y = mnist.target.astype(np.int64).reshape(-1, 1)
            np.savetxt('MNIST_full.txt', np.hstack([X, y]), delimiter=',', fmt='%.6f')
            print("MNIST 下载完成")

    elif dataset_name == 'MiniBooNE':
        if not os.path.exists('MiniBooNE.txt'):
            print("MiniBooNE 数据集不存在，正在自动下载")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt"
            download_file(url, 'MiniBooNE_PID.txt', 'MiniBooNE')
            os.rename('MiniBooNE_PID.txt', 'MiniBooNE.txt')
            print("MiniBooNE 下载完成")

    elif dataset_name == 'forest':
        if not os.path.exists('forest.csv'):
            print("forest数据集不存在，正在自动下载")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
            download_file(url, 'covtype.data.gz', 'Covertype')
            with gzip.open('covtype.data.gz', 'rb') as f_in:
                with open('forest.csv', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove('covtype.data.gz')
            print("forest下载完成")

    elif dataset_name == 'kdd_cup99_10_percent':
        if not os.path.exists('kdd_cup99_10_percent.csv'):
            print("KDD数据集不存在，正在自动下载")
            url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
            download_file(url, 'kddcup.data_10_percent.gz', 'KDD Cup 99 10%')
            with gzip.open('kddcup.data_10_percent.gz', 'rb') as f_in:
                with open('kdd_cup99_10_percent.csv', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove('kddcup.data_10_percent.gz')
            print("KDD Cup 99 下载完成")

    elif dataset_name == 'CIFAR-10':
        if not os.path.exists('./cifar10_resnet_features.mat'):
            print("cifar10数据集不存在，正在自动下载")
            extract_cifar10_resnet_features()
            print("cifar10 下载完成")

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

def extract_cifar10_resnet_features():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import scipy.io
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("正在加载 CIFAR-10 图像数据 (若未下载会自动下载)...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    full_dataset = torch.utils.data.ConcatDataset([trainset, testset])
    dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=256, shuffle=False, num_workers=0)
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            features = model(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.append(targets.numpy())
    X_deep = np.vstack(features_list)
    y_deep = np.concatenate(labels_list)
    output_file = 'cifar10_resnet_features.mat'
    scipy.io.savemat(output_file, {'data': X_deep, 'labels': y_deep})
    print(f"已保存至: {os.path.abspath(output_file)} (shape={X_deep.shape})")

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
    ensure_dataset(dataset_name)
    if dataset_name == 'CIFAR-10':
        import numpy as np
        import scipy.io
        data_file = './cifar10_resnet_features.mat'
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"找不到 {data_file}，请先运行 extract_features.py！")
        mat = scipy.io.loadmat(data_file)
        return mat['data'].astype(np.float32), mat['labels'].flatten().astype(np.int64)
    elif dataset_name == 'STAR_1M':
        import h5py
        import numpy as np
        h5_path = './star_1M_normalized.h5'
        f = h5py.File(h5_path, 'r')
        X = f['X'][:].astype(np.float32)
        y = f['y'][:].astype(np.int64)
        f.close()
        return X, y
    elif dataset_name == 'STAR_cdc':
        import h5py
        import numpy as np
        h5_path = './star_01_normalized.h5'
        f = h5py.File(h5_path, 'r')
        X = f['X'][:].astype(np.float32)
        y = f['y'][:].astype(np.int64)
        f.close()
        return X, y
    elif dataset_name == 'MNIST':
        import numpy as np
        filename = 'MNIST_full.txt'
        data = np.loadtxt(filename, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    elif dataset_name == 'MiniBooNE':
        import pandas as pd
        import numpy as np
        filename = 'MiniBooNE.txt'
        with open(filename, 'r') as f:
            first_line = f.readline().strip().split()
            n_signal, n_background = int(first_line[0]), int(first_line[1])
        df = pd.read_csv(filename, skiprows=1, header=None, sep='\s+')
        X = df.values
        y = np.array([1] * n_signal + [0] * n_background)
        X = clean_miniboone_data(X)
        return X, y
    elif dataset_name == 'forest':
        import pandas as pd
        import numpy as np
        df = pd.read_csv('forest.csv')
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        return X, y
    elif dataset_name == 'kdd_cup99_10_percent':
        import pandas as pd
        import numpy as np
        df = pd.read_csv('kdd_cup99_10_percent.csv')
        last_col = df.columns[-1]
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        return X, y
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")


if __name__ == "__main__":
    start_time = time.time()
    # ========== 命令行参数 + 配置加载 ==========
    import sys, json
    METHOD = sys.argv[1] if len(sys.argv) > 1 else 'MiniBatchKMeans'
    DATASET = sys.argv[2] if len(sys.argv) > 2 else 'MNIST'
    print(f"运行: method={METHOD}, dataset={DATASET}")
    with open('configs/params_registry.json', 'r', encoding='utf-8') as f:
        registry = json.load(f)
    method_key = METHOD.lower()
    if method_key == 'kmeans':
        method_key = 'minibatchkmeans'
    def _merge_config(registry, method, dataset):
        cfg = registry.get('_default', {}).copy()
        method_defaults = registry.get('_method_default', {}).get(method, {})
        cfg.update(method_defaults)
        exact = registry.get(method, {}).get(dataset, registry.get(method, {}).get('default', {}))
        cfg.update(exact)
        return cfg
    cfg = _merge_config(registry, method_key, DATASET)
    if not cfg:
        print(f"警告: 未找到 {METHOD}/{DATASET} 的配置，使用默认值")
        cfg = {}
    PCA_COMPONENTS = cfg.get('pca_components', 64)
    SH_CLUSTERS = cfg.get('sh_clusters', 100)
    MAX_EPISODES = cfg.get('max_episodes', 1)
    BATCH_SIZE = cfg.get('batch_size', 8)
    TARGET_NMI = cfg.get('target_nmi', 0.1696)
    selection_strategy = cfg.get('selection_strategy', 'random')
    feature_method = cfg.get('feature_method', 'pca')
    RL_METHOD = METHOD
    DATASET_NAME = DATASET
    LOG_FILE = "experiment_results.log"

    # ==============================================================
    # 1. 加载数据
    X, y = get_dataset(DATASET_NAME)

    if DATASET_NAME in ['STAR_1M', 'STAR_cdc']:
        X_scaled = X
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # 2. 特征重要性计算
    target_reached_time = -1.0
    algo_init_start = time.time()
    n_comp = min(PCA_COMPONENTS, X_scaled.shape[1] - 1)
    fi = FeatureImportancePCA(n_components=n_comp, method=feature_method).fit(X_scaled)
    df = fi.get_importance_df([f"F{i}" for i in range(X_scaled.shape[1])])

    # 3. 层次聚类初始化
    sh_start = time.time()
    n_sh_clusters = min(SH_CLUSTERS, len(X_scaled) // 10)
    sh = SampleHierarchy(n_clusters=n_sh_clusters, random_state=42, selection_strategy=selection_strategy).fit(X_scaled)

    # 4. 运行RL聚类
    rl = RLClustering(X=X_scaled, y_true=y, feature_importance=df,sample_hierarchy=sh, method=RL_METHOD, dataset_name=DATASET_NAME)
    init_algo_time = time.time() - algo_init_start

    def check_progress(episode, step, rl_instance, rl_algo_time):
        global target_reached_time
        if target_reached_time > 0 or rl_instance.current_centers is None or len(rl_instance.current_centers) < 2:
            return
        Z_all = X_scaled[:, rl_instance.selected_features]
        if RL_METHOD == 'idec' and hasattr(rl_instance, 'idec'):
            with torch.no_grad():
                Z_all = rl_instance.idec.autoencoder.encode(
                    torch.FloatTensor(Z_all).to(rl_instance.idec.device)).cpu().numpy()
        elif RL_METHOD == 'cdc' and hasattr(rl_instance, 'cdc_model'):
            with torch.no_grad():
                Z_all = rl_instance.cdc_model.encode(
                    torch.FloatTensor(Z_all).to(rl_instance.cdc_model.device)).cpu().numpy()
        if np.isnan(Z_all).any():
            Z_all = np.nan_to_num(Z_all)
        if rl_instance.current_centers is not None and np.isnan(rl_instance.current_centers).any():
            rl_instance.current_centers = np.nan_to_num(rl_instance.current_centers)
        from sklearn.metrics.pairwise import pairwise_distances
        y_pred_fast = np.argmin(pairwise_distances(Z_all, rl_instance.current_centers), axis=1)
        current_nmi = normalized_mutual_info_score(y, y_pred_fast)
        if current_nmi >= TARGET_NMI:
            target_reached_time = init_algo_time + rl_algo_time
            print(f"达成目标! NMI >= {TARGET_NMI}。耗时: {target_reached_time:.2f} 秒")
    history = rl.train(max_episodes=MAX_EPISODES, batch_size=BATCH_SIZE, callback=check_progress)

    # 5. 最终结果评估
    final_samples = rl.selected_samples
    final_features = rl.selected_features
    final_labels = rl.current_labels
    final_centers = rl.current_centers
    all_labels = -np.ones(len(X_scaled), dtype=int)
    all_labels[final_samples] = final_labels
    remaining_samples = np.where(all_labels == -1)[0]

    acc, nmi_score, ari = 0.0, 0.0, 0.0
    if rl.current_centers is not None and len(rl.current_centers) > 0:
        from sklearn.metrics import pairwise_distances

        Z_all = X_scaled[:, rl.selected_features]

        if RL_METHOD == 'idec' and hasattr(rl, 'idec') and rl.idec is not None:
            with torch.no_grad():
                Z_all = rl.idec.autoencoder.encode(
                    torch.FloatTensor(Z_all).to(rl.idec.device)
                ).cpu().numpy()
        elif RL_METHOD == 'cdc' and hasattr(rl, 'cdc_model') and rl.cdc_model is not None:
            with torch.no_grad():
                Z_all = rl.cdc_model.encode(
                    torch.FloatTensor(Z_all).to(rl.cdc_model.device)
                ).cpu().numpy()

        if np.isnan(Z_all).any():
            Z_all = np.nan_to_num(Z_all)

        centers = rl.current_centers.copy()
        if np.isnan(centers).any():
            centers = np.nan_to_num(centers)
        y_pred_full = np.argmin(pairwise_distances(Z_all, centers), axis=1)

        nmi_score = normalized_mutual_info_score(y, y_pred_full)
        acc = clustering_accuracy(y, y_pred_full)
        ari = adjusted_rand_score(y, y_pred_full)

    if len(remaining_samples) > 0 and rl.current_centers is not None:
        from sklearn.metrics import pairwise_distances
        from sklearn.cluster import KMeans, AgglomerativeClustering

        true_k = len(np.unique(y))

        # ==================== 剩余样本特征提取 ====================
        if RL_METHOD == 'idec':
            if hasattr(rl, 'best_idec') and rl.best_idec is not None:
                idec = rl.best_idec
                print(f"\n使用最优IDEC模型处理剩余样本 (最佳NMI={rl.global_best_nmi:.4f})")
                active_features = rl.best_features
                final_centers = rl.global_best_centers
            else:
                idec = rl.idec
                active_features = rl.selected_features
                final_centers = rl.current_centers
            X_for_idec = X_scaled[remaining_samples][:, active_features]
            X_remain_tensor = torch.FloatTensor(X_for_idec).to(idec.device)
            Z_remain = idec.autoencoder.encode(X_remain_tensor).detach().cpu().numpy()
            dists = pairwise_distances(Z_remain, final_centers)
            X_remain = X_for_idec
        elif RL_METHOD == 'cdc':
            if hasattr(rl, 'best_cdc') and rl.best_cdc is not None:
                cdc = rl.best_cdc
                active_features = rl.best_features
                final_centers = rl.global_best_centers
                print(f"\n使用最优CDC模型处理剩余样本 (最佳NMI={rl.global_best_nmi:.4f})")
            else:
                cdc = rl.cdc_model
                active_features = rl.selected_features
                final_centers = rl.current_centers
            X_for_cdc = X_scaled[remaining_samples][:, active_features]
            X_remain_tensor = torch.FloatTensor(X_for_cdc).to(cdc.device)
            Z_remain = cdc.encode(X_remain_tensor).detach().cpu().numpy()
            dists = pairwise_distances(Z_remain, final_centers)
            X_remain = X_for_cdc
        else:
            X_remain = X_scaled[remaining_samples][:, final_features]
            dists = pairwise_distances(X_remain, final_centers)

        min_dists = np.min(dists, axis=1)
        nearest = np.argmin(dists, axis=1)

        if len(final_centers) > 1:
            center_dists = pairwise_distances(final_centers)
            np.fill_diagonal(center_dists, np.inf)
            percent = cfg.get('percent', 75)
            threshold = np.percentile(center_dists[center_dists < np.inf], percent)
        else:
            percent = cfg.get('percent', 75)
            threshold = np.percentile(min_dists,percent)

        new_cluster_centers = []
        predicted_labels = np.zeros(len(X_remain), dtype=int)
        far_mask = min_dists > threshold
        close_mask = ~far_mask
        predicted_labels[close_mask] = nearest[close_mask]

        # ==========
        if DATASET_NAME == 'MNIST' or 'CIFAR-10' in DATASET_NAME:
            from sklearn.neighbors import KNeighborsClassifier
            if RL_METHOD == 'idec':
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled[:, active_features]).to(idec.device)
                    Z_all = idec.autoencoder.encode(X_tensor).cpu().numpy()
            elif RL_METHOD == 'cdc':
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled[:, active_features]).to(cdc.device)
                    Z_all = cdc.encode(X_tensor).cpu().numpy()
            else:
                Z_all = X_scaled[:, final_features]
            Z_train = Z_all[final_samples]
            Z_remain = Z_all[remaining_samples]
            knn = KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1)
            knn.fit(Z_train, final_labels)
            if np.sum(close_mask) > 0:
                predicted_labels[close_mask] = knn.predict(Z_remain[close_mask])
            if np.sum(far_mask) > 0:
                predicted_labels[far_mask] = knn.predict(Z_remain[far_mask])
            far_mask = np.zeros(len(X_remain), dtype=bool)

        if np.sum(far_mask) > 0:
            X_far = X_remain[far_mask]
            n_new = max(1, min(5, len(X_far) // 6000))
            if RL_METHOD == 'idec':
                with torch.no_grad():
                    X_far_tensor = torch.FloatTensor(X_far).to(idec.device)
                    Z_far = idec.autoencoder.encode(X_far_tensor).cpu().numpy()
            elif RL_METHOD == 'cdc':
                with torch.no_grad():
                    X_far_tensor = torch.FloatTensor(X_far).to(cdc.device)
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
        all_centers = np.vstack([final_centers, new_cluster_centers]) if new_cluster_centers else final_centers
        n_total_clusters = len(all_centers)
        true_k = len(np.unique(y))


        y_pred_full = np.zeros(len(X_scaled), dtype=int)
        if n_total_clusters > true_k:
            from sklearn.metrics.pairwise import pairwise_distances
            centers = all_centers.copy()
            n_merge = n_total_clusters - true_k
            for step in range(n_merge):
                dists = pairwise_distances(centers)
                np.fill_diagonal(dists, np.inf)
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                centers[i] = (centers[i] + centers[j]) / 2
                centers = np.delete(centers, j, axis=0)
            if RL_METHOD == 'idec':
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled[:, active_features]).to(idec.device)
                    Z = idec.autoencoder.encode(X_tensor).cpu().numpy()
            elif RL_METHOD == 'cdc':
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled[:, active_features]).to(cdc.device)
                    Z = cdc.encode(X_tensor).cpu().numpy()
            else:
                Z = X_scaled[:, final_features]
            dists_to_new_centers = pairwise_distances(Z, centers)
            y_pred_full = np.argmin(dists_to_new_centers, axis=1)
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
        print(f"\n最终结果:")
        print(f"  NMI: {nmi_score:.4f}")
        print(f"  Acc: {acc:.4f}")
        print(f"  ARI: {ari:.4f}")
    if target_reached_time > 0:
        print(f"达到目标 NMI {TARGET_NMI} 的确切算法时间: {target_reached_time:.2f} 秒")
    else:
        print(f"训练结束，未能达到目标 NMI {TARGET_NMI}")
    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time:.2f} 秒")
