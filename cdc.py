# cdc.py - CDC完整实现 (ICLR 2025) - 修复版
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torch.optim import Adam


class FeatureExtractor(nn.Module):
    """特征提取器 f(Θ) - DAE预训练"""

    def __init__(self, input_dim, hidden_dims=[500, 500, 2000], latent_dim=128):
        super().__init__()
        self.input_dim = input_dim  # 保存输入维度

        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器（用于预训练）
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return z

    def reconstruct(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class CalibrationHead(nn.Module):
    """校准头 g(θ_cal) - 降低ECE"""

    def __init__(self, latent_dim, n_clusters):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_clusters)

    def forward(self, z):
        return F.softmax(self.fc(z), dim=1)


class ClusteringHead(nn.Module):
    """聚类头 g(θ_clu) - 生成伪标签"""

    def __init__(self, latent_dim, n_clusters):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_clusters)

    def forward(self, z):
        return F.softmax(self.fc(z), dim=1)


class CDC(nn.Module):
    """
    CDC: Calibrated Deep Clustering (ICLR 2025)
    双头网络：校准头 + 聚类头，解决过自信问题
    """

    def __init__(self, input_dim, n_clusters, latent_dim=128,
                 hidden_dims=[500, 500, 2000], alpha=0.1, beta=0.5, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha  # 校准损失权重
        self.beta = beta  # 负熵损失权重
        self.device = device

        # 双头网络
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dims, latent_dim).to(device)
        self.calibration_head = CalibrationHead(latent_dim, n_clusters).to(device)
        self.clustering_head = ClusteringHead(latent_dim, n_clusters).to(device)

        # 优化器
        self.opt_extractor = Adam(self.feature_extractor.parameters(), lr=1e-3)
        self.opt_cal = Adam(self.calibration_head.parameters(), lr=1e-3)
        self.opt_clu = Adam(self.clustering_head.parameters(), lr=1e-3)

        self.cluster_centers = None

    def encode(self, x):
        """提取特征"""
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x).to(self.device)
        # 确保维度匹配
        if x.shape[1] != self.input_dim:
            # 维度不匹配，需要重新初始化或截断/填充
            raise ValueError(f"输入维度{x.shape[1]}与模型维度{self.input_dim}不匹配")
        return self.feature_extractor(x)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x).to(self.device)
        z = self.encode(x)
        p_cal = self.calibration_head(z)
        p_clu = self.clustering_head(z)
        return z, p_cal, p_clu

    def pretrain(self, X, epochs=50, batch_size=256, noise_factor=0.2):
        """DAE预训练"""
        print(f"CDC DAE预训练 {epochs} epochs...")
        self.train()

        optimizer = Adam(list(self.feature_extractor.encoder.parameters()) +
                         list(self.feature_extractor.decoder.parameters()), lr=1e-3)

        n_samples = len(X)
        X_tensor = torch.FloatTensor(X).to(self.device)

        for epoch in range(epochs):
            # 添加噪声
            X_noise = X_tensor + noise_factor * torch.randn_like(X_tensor)
            X_noise = torch.clamp(X_noise, 0, 1)

            indices = torch.randperm(n_samples)
            total_loss = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                x_batch = X_noise[batch_idx]
                x_target = X_tensor[batch_idx]

                x_hat, z = self.feature_extractor.reconstruct(x_batch)
                loss = F.mse_loss(x_hat, x_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: Recon Loss={total_loss / n_samples:.4f}")

        print("CDC预训练完成")

    def compute_calibration_loss(self, Z, p_cal, p_clu, K=5):
        """
        校准损失（核心创新）
        K: 每个簇的迷你簇数量
        """
        n_samples = len(Z)

        # K-means将特征分为K*n_clusters个迷你簇
        Z_np = Z.detach().cpu().numpy()
        # 修复：移除重复的n_init参数
        kmeans_mini = KMeans(n_clusters=min(K * self.n_clusters, n_samples // 2),
                             n_init=10, random_state=42)
        mini_labels = kmeans_mini.fit_predict(Z_np)

        # 计算每个迷你簇的平均预测（目标分布）
        loss_cal = 0
        count = 0
        for k in range(kmeans_mini.n_clusters):
            mask = (mini_labels == k)
            if mask.sum() == 0:
                continue

            # 该迷你簇的样本
            p_clu_k = p_clu[mask]  # (n_k, n_clusters)

            # 平均预测作为目标（停止梯度）
            q_hat_k = p_clu_k.mean(dim=0).detach()  # (n_clusters,)

            # 校准头预测
            p_cal_k = p_cal[mask]  # (n_k, n_clusters)

            # KL散度：校准头学习匹配聚类头的平均预测
            kl = F.kl_div(p_cal_k.log(), q_hat_k.expand(len(p_cal_k), -1), reduction='batchmean')
            loss_cal += kl
            count += 1

        loss_cal = loss_cal / max(count, 1)

        # 负熵损失（防止塌陷，鼓励均匀分布）
        entropy = -(p_cal * torch.log(p_cal + 1e-10)).sum(dim=1).mean()
        loss_en = -entropy  # 最大化熵 = 最小化负熵

        return loss_cal + self.beta * loss_en

    def select_pseudo_labels(self, p_cal, p_clu, top_B=0.9):
        """
        动态伪标签选择（核心创新）
        每类选择校准置信度最高的样本
        """
        n_samples = len(p_cal)
        pseudo_labels = torch.zeros(n_samples, dtype=torch.long, device=self.device) - 1

        # 对每个簇，按校准置信度选择
        for c in range(self.n_clusters):
            # 该类在聚类头的预测置信度
            conf_c = p_clu[:, c]

            # 找出预测为c的样本（top置信度）
            sorted_conf, sorted_idx = torch.sort(conf_c, descending=True)

            # 动态数量：选择置信度累积达到top_B的样本
            if sorted_conf.sum() > 0:
                cumsum_conf = torch.cumsum(sorted_conf, dim=0)
                threshold = top_B * cumsum_conf[-1]
                n_select = (cumsum_conf <= threshold).sum().item()
                n_select = max(n_select, 1)  # 至少选一个
            else:
                n_select = 1

            selected = sorted_idx[:n_select]
            pseudo_labels[selected] = c

        return pseudo_labels

    def train_epoch(self, X, epochs=10, batch_size=256):
        """CDC训练"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        n_samples = len(X)

        for epoch in range(epochs):
            self.train()
            indices = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                x_batch = X_tensor[batch_idx]

                # ========== 步骤1：更新校准头 ==========
                z, p_cal, p_clu = self(x_batch)

                loss_calibration = self.compute_calibration_loss(z, p_cal, p_clu)

                self.opt_cal.zero_grad()
                self.opt_extractor.zero_grad()
                loss_calibration.backward(retain_graph=True)
                self.opt_cal.step()
                self.opt_extractor.step()

                # ========== 步骤2：更新聚类头 ==========
                # 重新前向（CDC原文推荐重新计算）
                z_new, p_cal_new, p_clu_new = self(x_batch)

                # 动态选择伪标签
                pseudo_labels = self.select_pseudo_labels(p_cal_new, p_clu_new)

                # 只对有伪标签的样本计算损失
                labeled_mask = (pseudo_labels != -1)
                if labeled_mask.sum() > 0:
                    # 强增强（简化版：加噪声）
                    z_aug = z_new + 0.1 * torch.randn_like(z_new)
                    p_clu_aug = self.clustering_head(z_aug)

                    loss_clustering = F.cross_entropy(
                        p_clu_aug[labeled_mask],
                        pseudo_labels[labeled_mask]
                    )

                    self.opt_clu.zero_grad()
                    self.opt_extractor.zero_grad()
                    loss_clustering.backward()
                    self.opt_clu.step()
                    self.opt_extractor.step()

            # 每5轮评估
            if (epoch + 1) % 5 == 0:
                acc, nmi = self.evaluate(X)
                print(f"  CDC Epoch {epoch + 1}: ACC={acc:.4f}, NMI={nmi:.4f}")

    def predict(self, X):
        """预测聚类标签"""
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(X):
                X = torch.FloatTensor(X).to(self.device)
            z, p_cal, p_clu = self(X)

            # 用校准头的输出（ECE更低）
            labels = torch.argmax(p_cal, dim=1).cpu().numpy()

            # 计算中心
            z_np = z.cpu().numpy()
            centers = np.array([z_np[labels == k].mean(axis=0)
                                for k in range(self.n_clusters)
                                if np.sum(labels == k) > 0])

            # 如果某些簇为空，用随机点填充
            if len(centers) < self.n_clusters:
                existing_k = len(centers)
                for k in range(existing_k, self.n_clusters):
                    # 复制已有中心或随机初始化
                    if existing_k > 0:
                        centers = np.vstack([centers, centers[k % existing_k]])
                    else:
                        centers = np.vstack([centers, np.random.randn(self.latent_dim)])

        return labels, centers, z_np

    def evaluate(self, X, y_true=None):
        """评估"""
        labels, centers, z = self.predict(X)

        if y_true is not None:
            from sklearn.metrics import normalized_mutual_info_score
            from scipy.optimize import linear_sum_assignment

            # 匈牙利匹配计算ACC
            y_true = y_true.astype(np.int64)
            D = max(labels.max(), y_true.max()) + 1
            w = np.zeros((D, D), dtype=np.int64)
            for i in range(len(labels)):
                w[labels[i], y_true[i]] += 1
            row_ind, col_ind = linear_sum_assignment(-w)
            acc = w[row_ind, col_ind].sum() / len(labels)

            nmi = normalized_mutual_info_score(y_true, labels)
            return acc, nmi

        return 0, 0